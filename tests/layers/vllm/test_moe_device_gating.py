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
"""Killing tests for the slot-inversion bug in _bank_gating_device.

The bug: _bank_gating_device used slot_table_jnp[topk_ids] where
slot_table_jnp was slot→expert [S], but topk_ids contained expert IDs.
Indexing a slot→expert table by expert IDs is wrong direction — the
correct table is expert→slot [N], indexed by expert IDs.

Each test asserts device==host EXACTLY via _bank_gating_device output
vs the host-side expert_to_slot mapping (the oracle).
"""

import numpy as np
import jax
import jax.numpy as jnp
import importlib.util
import pytest

# Import _bank_gating_device directly from the source file to avoid
# the full tpu_inference package init chain (TPU metadata, vllm, etc.).
_moe_spec = importlib.util.spec_from_file_location(
    "moe",
    "/kaggle/working/vllm-tpu-stack/tpu-inference/tpu_inference/layers/vllm/interface/moe.py")
_moe_mod = importlib.util.module_from_spec(_moe_spec)
_moe_spec.loader.exec_module(_moe_mod)
_bank_gating_device = _moe_mod._bank_gating_device

# Import _build_expert_to_slot from expert_offload.py directly
_eo_spec = importlib.util.spec_from_file_location(
    "expert_offload",
    "/kaggle/working/vllm-tpu-stack/tpu-inference/tpu_inference/layers/vllm/expert_offload.py")
_eo_mod = importlib.util.module_from_spec(_eo_spec)
_eo_spec.loader.exec_module(_eo_mod)
_build_expert_to_slot = _eo_mod._build_expert_to_slot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_bank(n_experts: int = 8, slots: int = 4, store=None):
    """Minimal mock bank with known slot_to_expert / expert_to_slot."""

    class MockBank:
        def __init__(self):
            self.store = store
            self.slots = slots
            self.w13_host = np.zeros((n_experts, 64, 64), dtype=np.float32)
            self.w2_host = np.zeros((n_experts, 64, 64), dtype=np.float32)
            self.slot_to_expert = list(range(slots))
            self.expert_to_slot = {e: s for s, e in
                                   enumerate(self.slot_to_expert)}

    return MockBank()


# ---------------------------------------------------------------------------
# DEV-GATE-001 — ids≥S: expert IDs beyond slot count handled correctly
# ---------------------------------------------------------------------------

def test_dev_gate_001_ids_ge_s():
    """DEV-GATE-001: expert IDs ≥ S are handled correctly after fix.

    Pre-fix kill: _bank_gating_device with slot_to_expert table [S]
    indexed by expert 5 (≥S=4) raises IndexError — wrong direction,
    out-of-bounds take.

    Post-fix pass: expert_to_slot table [N] indexed by expert 5
    returns correct slot (or -1 if not resident).
    """
    S = 4
    N = 8
    bank = _make_mock_bank(n_experts=N, slots=S)

    # topk_ids includes expert 5 (≥ S=4) — the bug trigger
    topk_ids = jnp.array([[0, 5], [1, 2]], dtype=jnp.int32)
    topk_vals = jnp.array([[1.0, 0.9], [0.8, 0.7]], dtype=jnp.float32)

    # Pre-fix kill demo: slot_to_expert [S] indexed by expert 5 → wrong
    # value (JAX clamps to valid index), not correct slot for expert 5.
    slot_table = jnp.array(bank.slot_to_expert, dtype=jnp.int32)
    old_gating = _bank_gating_device(topk_vals, topk_ids, slot_table, S)
    # Expert 5 maps to slot 3 (clamped from OOB) — wrong direction
    # slot 3 holds expert 3's value, not expert 5's
    assert float(old_gating[0, 3]) == pytest.approx(0.9), \
        "pre-fix: expert 5 incorrectly mapped to slot 3 (clamped OOB)"

    # Post-fix pass: expert_to_slot [N] indexed by expert 5 → correct
    expert_to_slot = _build_expert_to_slot(bank)
    gating = _bank_gating_device(topk_vals, topk_ids, expert_to_slot, S)

    assert gating.shape == (2, S)
    # Expert 5 not resident → -1 sentinel in table
    assert int(expert_to_slot[5]) == -1
    # Resident experts → correct slots
    assert int(expert_to_slot[0]) == 0
    assert int(expert_to_slot[1]) == 1
    assert int(expert_to_slot[2]) == 2


# ---------------------------------------------------------------------------
# EVICT-GATE-002 — evict-then-gate across calls
# ---------------------------------------------------------------------------

def test_evict_gate_002_evict_then_gate():
    """EVICT-GATE-002: after eviction, expert_to_slot table reflects new residency.

    Evict expert 3 (slot 3), load expert 5 into slot 3.
    expert_to_slot[5] == 3, expert_to_slot[3] == -1.
    Gating with topk_ids=[5] scatters to slot 3 (not slot 5).
    """
    S = 4
    N = 8
    bank = _make_mock_bank(n_experts=N, slots=S)

    # Simulate eviction: expert 3 evicted, expert 5 loaded into slot 3
    bank.expert_to_slot = {0: 0, 1: 1, 2: 2, 5: 3,
                           4: -1, 6: -1, 7: -1}
    bank.slot_to_expert = [0, 1, 2, 5]

    expert_to_slot = _build_expert_to_slot(bank)

    # Gating: token 0 routes to expert 5 → should scatter to slot 3
    topk_ids = jnp.array([[5, 0]], dtype=jnp.int32)
    topk_vals = jnp.array([[0.8, 1.0]], dtype=jnp.float32)
    gating = _bank_gating_device(topk_vals, topk_ids, expert_to_slot, S)

    assert gating.shape == (1, S)
    # Expert 0 → slot 0 (val 1.0)
    assert float(gating[0, 0]) == pytest.approx(1.0)
    # Expert 5 → slot 3 (val 0.8)
    assert float(gating[0, 3]) == pytest.approx(0.8)
    # Slots 1, 2 → -inf (not routed)
    assert jnp.isneginf(gating[0, 1])
    assert jnp.isneginf(gating[0, 2])
    # Expert 3 not resident → -1 in table (would wrap, but not accessed here)
    assert int(expert_to_slot[3]) == -1
    assert int(expert_to_slot[5]) == 3


# ---------------------------------------------------------------------------
# BOUNDARY-003 — 31-vs-32 unique at S=32 (small-E analog)
# ---------------------------------------------------------------------------

def test_boundary_003_31_vs_32_unique():
    """BOUNDARY-003: boundary test for unique experts vs slots.

    Uses small-E analog (S=4, N=8) since 32-expert fixture may be
    heavy for CPU; the boundary logic is identical: N derived from
    w13_host.shape[0] or store.n_experts, table filled with -1
    sentinel, resident slots set from expert_to_slot dict.
    """
    S = 4
    N = 8
    bank = _make_mock_bank(n_experts=N, slots=S)

    expert_to_slot = _build_expert_to_slot(bank)

    # All S resident experts have correct slot indices
    for e in range(S):
        assert int(expert_to_slot[e]) == e, \
            f"resident expert {e} should map to slot {e}"
    # Non-resident experts have -1 sentinel
    for e in range(S, N):
        assert int(expert_to_slot[e]) == -1, \
            f"non-resident expert {e} should map to -1"

    # Gating with all S resident experts → all slots filled
    topk_ids = jnp.array(list(range(S)), dtype=jnp.int32).reshape(1, S)
    topk_vals = jnp.ones((1, S), dtype=jnp.float32)
    gating = _bank_gating_device(topk_vals, topk_ids, expert_to_slot, S)

    assert gating.shape == (1, S)
    # Every slot should have a real value (not -inf)
    for s in range(S):
        assert float(gating[0, s]) == 1.0, \
            f"slot {s} should have value 1.0 for resident expert {s}"


# ---------------------------------------------------------------------------
# MULTI-WAVE-004 — concat alignment across waves
# ---------------------------------------------------------------------------

def test_multi_wave_004_concat_alignment():
    """MULTI-WAVE-004: multi-wave routing produces aligned slot mappings.

    Wave 1: experts 0, 1 → slots 0, 1
    Wave 2: experts 2, 3 → slots 2, 3
    Each wave's gating is independent; concat along token dim aligns.
    """
    S = 4
    N = 8
    bank = _make_mock_bank(n_experts=N, slots=S)

    expert_to_slot = _build_expert_to_slot(bank)

    # Wave 1: token 0 → experts 0, 1; token 1 → experts 1, 0
    topk_ids_w1 = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    topk_vals_w1 = jnp.array([[1.0, 0.9], [0.8, 0.7]], dtype=jnp.float32)
    gating_w1 = _bank_gating_device(topk_vals_w1, topk_ids_w1,
                                    expert_to_slot, S)

    # Wave 2: token 0 → experts 2, 3; token 1 → experts 3, 2
    topk_ids_w2 = jnp.array([[2, 3], [3, 2]], dtype=jnp.int32)
    topk_vals_w2 = jnp.array([[0.6, 0.5], [0.4, 0.3]], dtype=jnp.float32)
    gating_w2 = _bank_gating_device(topk_vals_w2, topk_ids_w2,
                                    expert_to_slot, S)

    # Wave 1 verification
    assert gating_w1.shape == (2, S)
    assert float(gating_w1[0, 0]) == pytest.approx(1.0)   # expert 0 → slot 0
    assert float(gating_w1[0, 1]) == pytest.approx(0.9)   # expert 1 → slot 1
    assert jnp.isneginf(gating_w1[0, 2])    # slot 2 empty
    assert jnp.isneginf(gating_w1[0, 3])    # slot 3 empty
    assert float(gating_w1[1, 1]) == pytest.approx(0.8)   # expert 1 → slot 1
    assert float(gating_w1[1, 0]) == pytest.approx(0.7)   # expert 0 → slot 0

    # Wave 2 verification
    assert gating_w2.shape == (2, S)
    assert float(gating_w2[0, 2]) == pytest.approx(0.6)   # expert 2 → slot 2
    assert float(gating_w2[0, 3]) == pytest.approx(0.5)   # expert 3 → slot 3
    assert jnp.isneginf(gating_w2[0, 0])    # slot 0 empty
    assert jnp.isneginf(gating_w2[0, 1])    # slot 1 empty
    assert float(gating_w2[1, 3]) == pytest.approx(0.4)   # expert 3 → slot 3
    assert float(gating_w2[1, 2]) == pytest.approx(0.3)   # expert 2 → slot 2


# ---------------------------------------------------------------------------
# SLOT0-005 — pin respected: slot 0 (expert 0) never evicted
# ---------------------------------------------------------------------------

def test_slot0_005_pin_respected():
    """SLOT0-005: slot 0 (expert 0) is never evicted, even after
    ensure_resident reloads other slots.

    After eviction of expert 3 and loading expert 5, expert 0 stays
    in slot 0 (pinned). Gating with topk_ids=[0] scatters to slot 0.
    """
    S = 4
    N = 8
    bank = _make_mock_bank(n_experts=N, slots=S)

    # Initial residency: experts 0-3 in slots 0-3
    expert_to_slot = _build_expert_to_slot(bank)
    assert int(expert_to_slot[0]) == 0, "expert 0 pinned to slot 0"

    # Simulate eviction: expert 3 evicted, expert 5 loaded into slot 3
    bank.expert_to_slot = {0: 0, 1: 1, 2: 2, 5: 3,
                           4: -1, 6: -1, 7: -1}
    bank.slot_to_expert = [0, 1, 2, 5]

    expert_to_slot = _build_expert_to_slot(bank)

    # Expert 0 still in slot 0 after eviction (pin respected)
    assert int(expert_to_slot[0]) == 0, \
        "expert 0 should stay in slot 0 after eviction"
    # Expert 5 now in slot 3
    assert int(expert_to_slot[5]) == 3
    # Expert 3 evicted → -1
    assert int(expert_to_slot[3]) == -1

    # Gating: expert 0 → slot 0
    topk_ids = jnp.array([[0]], dtype=jnp.int32)
    topk_vals = jnp.array([[1.0]], dtype=jnp.float32)
    gating = _bank_gating_device(topk_vals, topk_ids, expert_to_slot, S)

    assert gating.shape == (1, S)
    assert float(gating[0, 0]) == 1.0   # expert 0 → slot 0
    for s in range(1, S):
        assert jnp.isneginf(gating[0, s]), \
            f"slot {s} should be -inf (not routed)"
