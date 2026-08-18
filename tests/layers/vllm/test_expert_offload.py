# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only unit tests for the host-backed MoE expert offload bank.

Covers the MXFP4/FP4 scale contract added for DeepSeek-V4-0731:
register -> initial slot allocation -> _load_one eviction refresh ->
slot_weights 4-tuple, with fp32 block scales (0731 dims [N, 8, 1, 4096] /
[N, 4, 1, 4096]) following the packed weights through every step. Also
covers the hash_indices_table registration guard and E1 (env unset -> no
bank, no behavior change).

TPU numerical falsification (full-bank fp4 apply vs slot-fed fp4 apply,
max|diff| <= 6.1e-05) is TPU-bound and intentionally NOT attempted here.
"""

import os

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm import expert_offload

# DeepSeek-V4-0731 scale dims (block 512, hidden 4096, moe_intermediate 2048)
# as produced by process_moe_weights for the MXFP4 requant path.
W13_SCALE_BLOCKS = 8  # hidden_size // block_size
W2_SCALE_BLOCKS = 4   # moe_intermediate_size // block_size
HIDDEN = 4096

N_EXPERTS = 16
S_SLOTS = 4


@pytest.fixture(autouse=True)
def _offload_clean_env(monkeypatch):
    """Start every test with offload disabled and an empty registry.

    Tests opt in by setting MOE_EXPERT_OFFLOAD / MOE_EXPERT_OFFLOAD_SLOTS.
    tpu_inference.envs resolves lazily via os.getenv (enable_envs_cache is
    never invoked outside service init), so monkeypatch toggles are visible.
    """
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD", raising=False)
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD_SLOTS", raising=False)
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD_LAYERS", raising=False)
    expert_offload.clear_all()
    yield
    expert_offload.clear_all()


@pytest.fixture
def cpu_mesh():
    """1-device CPU mesh; axis name 'model' matches ShardingAxisName2D."""
    return Mesh(np.array(jax.devices()), axis_names=("model",))


def _shardings(cpu_mesh):
    """Weight (2-D) + scale (4-D) shardings exactly as the real gates pass."""
    w13_sharding = NamedSharding(
        cpu_mesh, PartitionSpec(None, None, ShardingAxisName.MLP_TENSOR))
    w2_sharding = NamedSharding(
        cpu_mesh, PartitionSpec(None, ShardingAxisName.MLP_TENSOR, None))
    w13_scale_sharding = NamedSharding(
        cpu_mesh,
        PartitionSpec(None, None, None, ShardingAxisName.MLP_TENSOR))
    w2_scale_sharding = NamedSharding(
        cpu_mesh,
        PartitionSpec(None, ShardingAxisName.MLP_TENSOR, None, None))
    return (w13_sharding, w2_sharding, w13_scale_sharding, w2_scale_sharding)


def _make_bank_data(n_experts=N_EXPERTS):
    """Synthetic bank payloads; scales use the REAL 0731 dims.

    Every tensor is filled with a per-expert constant marker so eviction
    can prove weight AND scale swapped together by content, not just shape.
    """
    w13_host = np.zeros((n_experts, 64, 64), dtype=np.float32)
    w2_host = np.zeros((n_experts, 64, 64), dtype=np.float32)
    w13_scale_host = np.zeros((n_experts, W13_SCALE_BLOCKS, 1, HIDDEN),
                              dtype=np.float32)
    w2_scale_host = np.zeros((n_experts, W2_SCALE_BLOCKS, 1, HIDDEN),
                             dtype=np.float32)
    for e in range(n_experts):
        w13_host[e] = 1000.0 + e
        w2_host[e] = 2000.0 + e
        w13_scale_host[e] = 3000.0 + e
        w2_scale_host[e] = 4000.0 + e
    return (w13_host, w2_host, w13_scale_host, w2_scale_host)


class _FakeLayer:
    """Minimal stand-in for the RoutedExperts layer used at the gates."""

    def __init__(self, hash_indices_table=None):
        self.layer_name = "model.layers.3.ffn.experts"
        self.hash_indices_table = hash_indices_table


def _enable_offload(monkeypatch, slots=S_SLOTS, layers=""):
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD", "1")
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD_SLOTS", str(slots))
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD_LAYERS", layers)


def test_register_bank_with_scales(cpu_mesh, monkeypatch):
    """Scales follow weights into the bank and the initial S device slots."""
    _enable_offload(monkeypatch)
    (w13, w2, w13s, w2s) = _make_bank_data()
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)

    bank = expert_offload.register_bank(
        "model.layers.3.ffn.experts", w13, w2, w13_sh, w2_sh,
        w13_scale_host=w13s, w2_scale_host=w2s,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer())

    assert bank is not None
    assert expert_offload.get_bank(
        "model.layers.3.ffn.experts") is bank
    # Slot scale shapes carry the real 0731 dims, restricted to S slots.
    assert bank.slot_w13_scale.shape == (S_SLOTS, W13_SCALE_BLOCKS, 1, HIDDEN)
    assert bank.slot_w2_scale.shape == (S_SLOTS, W2_SCALE_BLOCKS, 1, HIDDEN)
    # slot_weights is the 4-tuple accessor.
    (sw13, sw2, sw13s, sw2s) = bank.slot_weights()
    assert sw13.shape == (S_SLOTS, 64, 64)
    assert sw2.shape == (S_SLOTS, 64, 64)
    assert sw13s.shape == bank.slot_w13_scale.shape
    assert sw2s.shape == bank.slot_w2_scale.shape
    # Initial residency: experts 0..S-1, slot 0 pinned to expert 0.
    assert bank.slot_to_expert == list(range(S_SLOTS))
    assert np.allclose(np.asarray(jax.device_get(sw13s))[0],
                       w13s[0])


def test_load_one_refreshes_scale_with_weight(cpu_mesh, monkeypatch):
    """_load_one swaps BOTH the weight and the scale for the new expert."""
    _enable_offload(monkeypatch)
    (w13, w2, w13s, w2s) = _make_bank_data()
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)

    bank = expert_offload.register_bank(
        "model.layers.3.ffn.experts", w13, w2, w13_sh, w2_sh,
        w13_scale_host=w13s, w2_scale_host=w2s,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer())
    # Evict expert 1 (slot 1, non-padding) in favor of expert 7.
    bank._load_one(slot=1, expert_id=7)

    assert bank.slot_to_expert[1] == 7
    # Host slot rows refreshed: weight AND scale both carry expert 7's marker.
    assert bank.slot13_host[1][0, 0] == 1000.0 + 7
    assert bank.slot2_host[1][0, 0] == 2000.0 + 7
    assert bank.slot13_scale_host[1][0, 0, 0] == 3000.0 + 7
    assert bank.slot2_scale_host[1][0, 0, 0] == 4000.0 + 7
    # Device views refreshed with the same shapes.
    assert np.allclose(np.asarray(jax.device_get(bank.slot_w13_scale))[1],
                       w13s[7])
    assert np.allclose(np.asarray(jax.device_get(bank.slot_w2_scale))[1],
                       w2s[7])
    assert bank.slot_w13_scale.shape == (S_SLOTS, W13_SCALE_BLOCKS, 1, HIDDEN)
    # The padding slot (0) is untouched.
    assert bank.slot_to_expert[0] == 0
    assert bank.slot13_scale_host[0][0, 0, 0] == 3000.0


def test_eviction_swaps_weight_and_scale_atomically(cpu_mesh, monkeypatch):
    """ensure_resident eviction swaps weight+scale pairs for every victim."""
    _enable_offload(monkeypatch)
    (w13, w2, w13s, w2s) = _make_bank_data()
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)

    bank = expert_offload.register_bank(
        "model.layers.3.ffn.experts", w13, w2, w13_sh, w2_sh,
        w13_scale_host=w13s, w2_scale_host=w2s,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer())
    # Request experts 5 and 6 (not resident): S=4 -> two victims evicted.
    bank.ensure_resident(np.array([5, 6]))

    for slot, expert in enumerate(bank.slot_to_expert):
        assert bank.slot13_host[slot][0, 0] == 1000.0 + expert
        assert bank.slot2_host[slot][0, 0] == 2000.0 + expert
        assert bank.slot13_scale_host[slot][0, 0, 0] == 3000.0 + expert
        assert bank.slot2_scale_host[slot][0, 0, 0] == 4000.0 + expert
    assert bank.expert_to_slot[5] == bank.slot_to_expert.index(5)
    assert bank.expert_to_slot[6] == bank.slot_to_expert.index(6)
    # Slot 0 is still pinned to expert 0.
    assert bank.slot_to_expert[0] == 0


def test_hash_indices_table_guard_refuses_registration(cpu_mesh,
                                                       monkeypatch):
    """Hash-routed layers are refused at the shared register_bank choke point."""
    _enable_offload(monkeypatch)
    (w13, w2, w13s, w2s) = _make_bank_data()
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)
    hash_layer = _FakeLayer(hash_indices_table=np.zeros((128, 6)))

    bank = expert_offload.register_bank(
        "model.layers.1.ffn.experts", w13, w2, w13_sh, w2_sh,
        w13_scale_host=w13s, w2_scale_host=w2s,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=hash_layer)

    assert bank is None
    assert expert_offload.get_bank("model.layers.1.ffn.experts") is None
    # A non-hash layer at the same key registers fine (guard is per-layer).
    bank2 = expert_offload.register_bank(
        "model.layers.1.ffn.experts", w13, w2, w13_sh, w2_sh,
        w13_scale_host=w13s, w2_scale_host=w2s,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer())
    assert bank2 is not None


def test_e1_env_off_no_bank_no_behavior_change(cpu_mesh, monkeypatch):
    """E1: env unset -> register_bank returns None, registry stays empty."""
    (w13, w2, w13s, w2s) = _make_bank_data()
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)

    assert expert_offload.offload_enabled() is False
    assert expert_offload.layer_enabled("model.layers.3.ffn.experts") is False

    bank = expert_offload.register_bank(
        "model.layers.3.ffn.experts", w13, w2, w13_sh, w2_sh,
        w13_scale_host=w13s, w2_scale_host=w2s,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer())

    assert bank is None
    assert expert_offload.get_bank("model.layers.3.ffn.experts") is None
    assert expert_offload._BANKS == {}


def test_unquantized_bank_scale_none_feed_preserved(cpu_mesh, monkeypatch):
    """Unquantized banks carry None scales; the moe.py feed stays identical."""
    _enable_offload(monkeypatch)
    (w13, w2, w13s, w2s) = _make_bank_data()
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)

    # Unquantized registration: no scale args (the unquantized.py call site).
    bank = expert_offload.register_bank(
        "model.layers.3.ffn.experts", w13, w2, w13_sh, w2_sh,
        layer=_FakeLayer())

    assert bank.slot_w13_scale is None
    assert bank.slot_w2_scale is None
    (sw13, sw2, sw13s, sw2s) = bank.slot_weights()
    assert sw13s is None and sw2s is None
    # The interface feed (moe.py vllm_moe_apply bank branch) passes the bank
    # scales straight into FusedMoEWeights; for an unquantized bank that is
    # None, byte-identical to the pre-patch hardcoded None.
    weights = FusedMoEWeights(
        w13_weight=bank.slot_w13,
        w13_weight_scale=bank.slot_w13_scale,
        w13_bias=None,
        w2_weight=bank.slot_w2,
        w2_weight_scale=bank.slot_w2_scale,
        w2_bias=None,
    )
    assert weights.w13_weight_scale is None
    assert weights.w2_weight_scale is None


def test_quantized_feed_carries_scales(cpu_mesh, monkeypatch):
    """The moe.py feed carries slot scales for quantized banks (P-INC guard)."""
    _enable_offload(monkeypatch)
    (w13, w2, w13s, w2s) = _make_bank_data()
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)

    bank = expert_offload.register_bank(
        "model.layers.3.ffn.experts", w13, w2, w13_sh, w2_sh,
        w13_scale_host=w13s, w2_scale_host=w2s,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer())
    weights = FusedMoEWeights(
        w13_weight=bank.slot_w13,
        w13_weight_scale=bank.slot_w13_scale,
        w13_bias=None,
        w2_weight=bank.slot_w2,
        w2_weight_scale=bank.slot_w2_scale,
        w2_bias=None,
    )
    assert weights.w13_weight_scale.shape == (
        S_SLOTS, W13_SCALE_BLOCKS, 1, HIDDEN)
    assert weights.w2_weight_scale.shape == (
        S_SLOTS, W2_SCALE_BLOCKS, 1, HIDDEN)


def test_scale_args_must_be_a_pair(cpu_mesh, monkeypatch):
    """Partial scale registration fails fast (w13 without w2 is a contract bug)."""
    _enable_offload(monkeypatch)
    (w13, w2, w13s, w2s) = _make_bank_data()
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)

    with pytest.raises(ValueError, match="together"):
        expert_offload.register_bank(
            "model.layers.3.ffn.experts", w13, w2, w13_sh, w2_sh,
            w13_scale_host=w13s, w2_scale_host=None,
            dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
            layer=_FakeLayer())
    # Scales provided but shardings missing -> contract error, not silent.
    with pytest.raises(ValueError, match="shardings"):
        expert_offload.register_bank(
            "model.layers.3.ffn.experts", w13, w2, w13_sh, w2_sh,
            w13_scale_host=w13s, w2_scale_host=w2s,
            layer=_FakeLayer())