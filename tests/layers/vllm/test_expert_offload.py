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

# DeepSeek-V4-0731 scale dims (block 256, hidden 4096, moe_intermediate 2048)
# as produced by process_moe_weights for the MXFP4 requant path.
W13_SCALE_BLOCKS = 16  # hidden_size // block_size
W2_SCALE_BLOCKS = 8    # moe_intermediate_size // block_size
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
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD_HOST_MEMORY_GUARD", raising=False)
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD_HOST_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD_CPU_WORKING_SET_GIB", raising=False)
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD_STORE", raising=False)
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD_STORE_DIR", raising=False)
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD_PUSH_MODE", raising=False)
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD_HOT_CACHE_GIB", raising=False)
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD_RAW_JIT", raising=False)
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


def test_memory_guard_rejects_before_cgroup_exhaustion(monkeypatch):
    """The admission guard rejects a peak that would consume the reserve."""
    gib = 1 << 30
    assert expert_offload._memory_guard_message(100 * gib, 48 * gib,
                                                24 * gib) is None
    message = expert_offload._memory_guard_message(71 * gib, 48 * gib,
                                                   24 * gib)
    assert message is not None
    assert "available=71.00 GiB" in message
    assert "safety reserve=24.00 GiB" in message


def test_memory_guard_uses_cgroup_snapshot(monkeypatch):
    """A cgroup limit is honored even when host MemAvailable is larger."""
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD_HOST_MEMORY_GUARD", "1")
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD_HOST_MEMORY_RESERVE_GIB", "24")
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD_CPU_WORKING_SET_GIB", "48")
    monkeypatch.setattr(
        expert_offload, "_host_memory_snapshot", lambda: {
            "cgroup_current": 259 * (1 << 30),
            "cgroup_limit": 330 * (1 << 30),
            "cgroup_available": 71 * (1 << 30),
            "proc_available": 381 * (1 << 30),
            "available": 71 * (1 << 30),
            "cgroup_committed": 259 * (1 << 30),
            "cgroup_reclaimable_file": 0,
        })

    with pytest.raises(RuntimeError, match="host memory guard refused"):
        expert_offload.check_host_memory_budget("model.layers.21.ffn.experts",
                                                source_bytes=1 << 30)


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

# ---------------------------------------------------------------------------
# Design D store-first hybrid — CPU witnesses W0-W3
# ---------------------------------------------------------------------------

FLOAT4 = np.dtype("float4_e2m1fn")


def _make_fp4_data(n_experts=N_EXPERTS):
    """Synthetic GMM-processed FP4 payloads with per-expert code markers.

    w13/w2 are float4_e2m1fn (itemsize 1; only the low nibble is meaningful),
    filled with random valid FP4 codes so a corrupted or swapped record can
    never match by content. Scales are 4-D fp32 with the same per-expert
    marker discipline as _make_bank_data.
    """
    rng = np.random.default_rng(7 + n_experts)
    w13 = rng.integers(0, 16, size=(n_experts, 16, 1, 64),
                       dtype=np.uint8).view(FLOAT4)
    w2 = rng.integers(0, 16, size=(n_experts, 8, 1, 64),
                      dtype=np.uint8).view(FLOAT4)
    w13s = rng.random((n_experts, W13_SCALE_BLOCKS, 1, 8),
                      dtype=np.float32) + (1.0 + np.arange(
                          n_experts, dtype=np.float32)).reshape(-1, 1, 1, 1)
    w2s = rng.random((n_experts, W2_SCALE_BLOCKS, 1, 8),
                     dtype=np.float32) + (10.0 + np.arange(
                         n_experts, dtype=np.float32)).reshape(-1, 1, 1, 1)
    return w13, w2, w13s, w2s


def _write_test_store(tmp_path, layer_id=3, n_experts=N_EXPERTS):
    """Write a synthetic FP4 store for layer ``layer_id``; return its path."""
    w13, w2, w13s, w2s = _make_fp4_data(n_experts)
    path = tmp_path / f"layer_{layer_id:03d}.rec"
    record_bytes = expert_offload.write_expert_store(
        path, layer_id, w13, w2, w13s, w2s)
    return path, record_bytes, (w13, w2, w13s, w2s)


def _enable_store_env(monkeypatch, tmp_path, slots=S_SLOTS,
                      push_mode="scatter"):
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD", "1")
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD_SLOTS", str(slots))
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD_STORE", "1")
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("MOE_EXPERT_OFFLOAD_PUSH_MODE", push_mode)


def _store_row(bank, slot, which="w13"):
    """Device slot row as a uint8/float32 numpy array (for content checks)."""
    if which == "w13":
        arr = np.asarray(jax.device_get(bank.slot_w13))[slot]
    elif which == "w2":
        arr = np.asarray(jax.device_get(bank.slot_w2))[slot]
    elif which == "w13s":
        return np.asarray(jax.device_get(bank.slot_w13_scale))[slot]
    else:
        return np.asarray(jax.device_get(bank.slot_w2_scale))[slot]
    return arr.view(np.uint8)


def _expected_unpack(bank, expert_id, which="w13"):
    """The store record row unpacked to kernel form (the W1 byte contract).

    Weight rows are returned as uint8 FP4 codes (the same canonical view as
    _store_row) so array_equal compares bytes, not mixed dtypes.
    """
    w13_row, w2_row, s13_row, s2_row = bank.store.read_record(expert_id)
    if which == "w13":
        return bank._unpack_weight_rows(w13_row, bank._w13_packed).view(np.uint8)
    if which == "w2":
        return bank._unpack_weight_rows(w2_row, bank._w2_packed).view(np.uint8)
    if which == "w13s":
        return s13_row
    return s2_row


def test_w0_store_roundtrip_fp4_byte_identity(tmp_path):
    """W0: write -> open -> read every expert; unpacked rows == originals."""
    path, record_bytes, (w13, w2, w13s, w2s) = _write_test_store(tmp_path)
    store = expert_offload.open_expert_store(path, 3)
    try:
        assert store.n_experts == N_EXPERTS
        # Exact byte contract: record = w13(16*1*32) + w2(8*1*32)
        # + s13(16*1*8*4) + s2(8*1*8*4) = 512 + 256 + 512 + 256 = 1536 B.
        assert store.record_bytes == record_bytes == 1536
        assert path.stat().st_size == (
            expert_offload._STORE_HEADER_BYTES + record_bytes * N_EXPERTS)
        for e in range(N_EXPERTS):
            w13_row, w2_row, s13_row, s2_row = store.read_record(e)
            assert np.array_equal(
                w13_row.view(np.uint8),
                ((w13[e].view(np.uint8)[..., 0::2] & 0x0F)
                 | ((w13[e].view(np.uint8)[..., 1::2] & 0x0F) << 4)))
    finally:
        store.close()


def test_w0_store_roundtrip_content(tmp_path):
    """W0: unpacked store rows are content-identical to the source arrays."""
    path, _, (w13, w2, w13s, w2s) = _write_test_store(tmp_path)
    store = expert_offload.open_expert_store(path, 3)
    try:
        for e in range(N_EXPERTS):
            w13_row, w2_row, s13_row, s2_row = store.read_record(e)
            unpacked13 = w13_row.view(np.uint8)
            assert unpacked13.shape == (16, 1, 32)
            full = np.empty((16, 1, 64), dtype=np.uint8)
            full[..., 0::2] = unpacked13 & 0x0F
            full[..., 1::2] = (unpacked13 >> 4) & 0x0F
            assert np.array_equal(full, w13[e].view(np.uint8))
            assert np.array_equal(_unpack2(w2_row), w2[e].view(np.uint8))
            assert np.array_equal(s13_row, w13s[e])
            assert np.array_equal(s2_row, w2s[e])
    finally:
        store.close()


def _unpack2(packed_row):
    full = np.empty(list(packed_row.shape[:-1]) + [packed_row.shape[-1] * 2],
                    dtype=np.uint8)
    full[..., 0::2] = packed_row & 0x0F
    full[..., 1::2] = (packed_row >> 4) & 0x0F
    return full


def test_w0_store_open_time_corruption_detected(tmp_path):
    """W0: a flipped data byte fails open verification (loud, pre-bank)."""
    path, _, _ = _write_test_store(tmp_path)
    size = path.stat().st_size
    blob = bytearray(path.read_bytes())
    # Flip one byte inside record 5's weight area.
    blob[expert_offload._STORE_HEADER_BYTES + 5 * 1536 + 10] ^= 0x01
    path.write_bytes(bytes(blob))
    with pytest.raises(expert_offload.ExpertStoreError, match="sha256"):
        expert_offload.open_expert_store(path, 3)
    assert path.stat().st_size == size


def test_w0_store_header_checks(tmp_path):
    """W0: bad magic, wrong layer id, and truncation all fail loudly."""
    path, _, _ = _write_test_store(tmp_path)
    with pytest.raises(expert_offload.ExpertStoreError, match="layer id"):
        expert_offload.open_expert_store(path, 4)
    blob = bytearray(path.read_bytes())
    blob[0] = 0x00
    bad = tmp_path / "bad.rec"
    bad.write_bytes(bytes(blob))
    with pytest.raises(expert_offload.ExpertStoreError, match="magic"):
        expert_offload.open_expert_store(bad, 3)
    trunc = tmp_path / "trunc.rec"
    trunc.write_bytes(bytes(blob)[:100])
    with pytest.raises(expert_offload.ExpertStoreError, match="truncated"):
        expert_offload.open_expert_store(trunc, 3)
    missing = tmp_path / "missing.rec"
    with pytest.raises(expert_offload.ExpertStoreError, match="missing"):
        expert_offload.open_expert_store(missing, 3)


def test_w0_store_scale_pairing(tmp_path):
    """W0: one scale without the other is a write-time contract error."""
    w13, w2, w13s, w2s = _make_fp4_data()
    path = tmp_path / "layer_003.rec"
    with pytest.raises(ValueError, match="together"):
        expert_offload.write_expert_store(path, 3, w13, w2, w13s, None)


def test_w2_store_bank_scatter(tmp_path, cpu_mesh, monkeypatch):
    """W2: scatter-mode store bank — no host mirror, slots from store."""
    _enable_store_env(monkeypatch, tmp_path, push_mode="scatter")
    path, _, _ = _write_test_store(tmp_path)
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)

    bank = expert_offload.register_bank(
        "model.layers.3.ffn.experts",
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        w13_sh, w2_sh,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer(), store_path=str(path))

    assert bank is not None
    assert bank.push_mode == "scatter"
    # Scatter mode keeps NO host mirror (Design D: anon = O(hot cache)).
    assert bank.slot13_host is None and bank.slot2_host is None
    assert bank.slot13_scale_host is None and bank.slot2_scale_host is None
    # Initial residency from store records 0..S-1, content-verified.
    for slot in range(S_SLOTS):
        assert bank.slot_to_expert[slot] == slot
        assert np.array_equal(_store_row(bank, slot, "w13"),
                              _expected_unpack(bank, slot, "w13"))
        assert np.array_equal(_store_row(bank, slot, "w2"),
                              _expected_unpack(bank, slot, "w2"))
        assert np.allclose(_store_row(bank, slot, "w13s"),
                           _expected_unpack(bank, slot, "w13s"))
    # Eviction: request experts 5 and 6 -> two victims, padding pinned.
    bank.ensure_resident(np.array([5, 6]))
    for slot, expert in enumerate(bank.slot_to_expert):
        assert np.array_equal(_store_row(bank, slot, "w13"),
                              _expected_unpack(bank, expert, "w13"))
        assert np.array_equal(_store_row(bank, slot, "w2"),
                              _expected_unpack(bank, expert, "w2"))
        assert np.allclose(_store_row(bank, slot, "w13s"),
                           _expected_unpack(bank, expert, "w13s"))
        assert np.allclose(_store_row(bank, slot, "w2s"),
                           _expected_unpack(bank, expert, "w2s"))
    assert bank.slot_to_expert[0] == 0
    assert 5 in bank.expert_to_slot and 6 in bank.expert_to_slot


def test_w2_store_bank_full_push(tmp_path, cpu_mesh, monkeypatch):
    """W2: full-push store bank keeps a packed S-slot mirror, refreshes it."""
    _enable_store_env(monkeypatch, tmp_path, push_mode="full")
    path, _, _ = _write_test_store(tmp_path)
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)

    bank = expert_offload.register_bank(
        "model.layers.3.ffn.experts",
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        w13_sh, w2_sh,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer(), store_path=str(path))

    assert bank is not None
    assert bank.push_mode == "full"
    # Packed (storage-form) mirror: uint8, last axis halved vs float4 form.
    assert bank.slot13_host is not None
    assert bank.slot13_host.dtype == np.uint8
    assert bank.slot13_host.shape == (S_SLOTS, 16, 1, 32)
    assert bank.slot13_scale_host.shape == (
        S_SLOTS, W13_SCALE_BLOCKS, 1, 8)
    # Evict slot 1 (expert 1) in favor of expert 7.
    bank._load_one(slot=1, expert_id=7)
    assert bank.slot_to_expert[1] == 7
    assert np.array_equal(bank.slot13_host[1],
                          bank.store.read_record(7)[0])
    assert np.array_equal(
        _store_row(bank, 1, "w13"), _expected_unpack(bank, 7, "w13"))
    assert np.allclose(_store_row(bank, 1, "w13s"),
                       _expected_unpack(bank, 7, "w13s"))


def test_w3_store_bank_route_gating_consistency(tmp_path, cpu_mesh,
                                                 monkeypatch):
    """W3: routing stress — gating maps resident experts without loss.

    Each round concentrates routing on a bounded pool (including the pinned
    padding expert 0) so the batch union fits S slots, then checks: every
    selected token's gating value equals its original logit at the mapped
    slot, unselected slots are -inf, and EVERY resident slot's device
    content matches its store record (eviction + re-residency across rounds).
    """
    _enable_store_env(monkeypatch, tmp_path, push_mode="scatter")
    path, _, _ = _write_test_store(tmp_path)
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)
    bank = expert_offload.register_bank(
        "model.layers.3.ffn.experts",
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        w13_sh, w2_sh,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer(), store_path=str(path))

    rng = np.random.default_rng(42)
    # Round 3 re-requests round 1's pool after round 2 evicted it.
    rounds = [[0, 4, 9, 15], [0, 2, 6, 12], [0, 4, 9, 15]]
    for pool in rounds:
        T, TOP_K = 8, 2
        logits = np.full((T, N_EXPERTS), -10.0, dtype=np.float32)
        for t in range(T):
            for e in rng.choice(pool, size=TOP_K, replace=False):
                logits[t, int(e)] = float(rng.normal(2.0, 1.0))
        gating = bank.route(logits, TOP_K)
        assert gating.shape == (T, S_SLOTS)
        for t in range(T):
            top_ids = np.argpartition(logits[t], -TOP_K)[-TOP_K:]
            seen = set()
            for e in top_ids:
                e = int(e)
                s = bank.expert_to_slot[e]
                assert np.isclose(gating[t, s], logits[t, e])
                seen.add(s)
            for s in range(S_SLOTS):
                if s not in seen:
                    assert np.isneginf(gating[t, s])
        # Invariant: every resident slot carries exactly its mapped
        # expert's store record (weights AND scales).
        for slot, expert in enumerate(bank.slot_to_expert):
            assert np.array_equal(_store_row(bank, slot, "w13"),
                                  _expected_unpack(bank, expert, "w13"))
            assert np.array_equal(_store_row(bank, slot, "w2"),
                                  _expected_unpack(bank, expert, "w2"))
            assert np.allclose(_store_row(bank, slot, "w13s"),
                               _expected_unpack(bank, expert, "w13s"))
            assert np.allclose(_store_row(bank, slot, "w2s"),
                               _expected_unpack(bank, expert, "w2s"))


def test_store_register_rejects_hash_routed_and_env_off(tmp_path, cpu_mesh,
                                                        monkeypatch):
    """Store path honors the shared guards: hash-routed refuse, env off."""
    _enable_store_env(monkeypatch, tmp_path)
    path, _, _ = _write_test_store(tmp_path)
    (w13_sh, w2_sh, w13s_sh, w2s_sh) = _shardings(cpu_mesh)
    bank = expert_offload.register_bank(
        "model.layers.1.ffn.experts",
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        w13_sh, w2_sh,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer(hash_indices_table=np.zeros((128, 6))),
        store_path=str(path))
    assert bank is None
    # env off -> None even with a valid store path
    monkeypatch.delenv("MOE_EXPERT_OFFLOAD")
    bank2 = expert_offload.register_bank(
        "model.layers.3.ffn.experts",
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        w13_sh, w2_sh,
        dev_w13_scale_sharding=w13s_sh, dev_w2_scale_sharding=w2s_sh,
        layer=_FakeLayer(), store_path=str(path))
    assert bank2 is None
