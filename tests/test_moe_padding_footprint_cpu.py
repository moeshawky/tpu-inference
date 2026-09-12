"""CPU-only check for MoE padding footprint accounting.

Loads _footprint_unique_ids from interface/moe.py via AST without
importing jax/torch, then checks the S-1 fallback decision on
adversarial padded inputs.
"""

import ast
from pathlib import Path

import numpy as np

MOE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tpu_inference"
    / "layers"
    / "vllm"
    / "interface"
    / "moe.py"
)
# When placed at tests/test_moe_padding_footprint_cpu.py, parents[1] is the
# repo root only if tests/ is one level deep; resolve robustly instead.
if not MOE_PATH.exists():
    # Fall back: walk up until tpu_inference/layers/vllm/interface/moe.py found.
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tpu_inference" / "layers" / "vllm" / "interface" / "moe.py"
        if cand.exists():
            MOE_PATH = cand
            break


def _load_footprint_fn():
    src = MOE_PATH.read_text()
    tree = ast.parse(src)
    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_footprint_unique_ids":
            target = node
            break
    assert target is not None, "_footprint_unique_ids not found in moe.py"
    seg = ast.get_source_segment(src, target)
    assert seg is not None, "could not extract _footprint_unique_ids source"
    ns = {"np": np}
    exec(compile(ast.parse(seg), str(MOE_PATH), "exec"), ns)
    return ns["_footprint_unique_ids"], src


def test_padded_no_fallback():
    fn, _ = _load_footprint_fn()
    T, K, valid, S = 16, 8, 3, 32
    topk = np.zeros((T, K), dtype=np.int64)
    # Valid rows: small footprint {1, 2, 3, 5}.
    topk[0] = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    topk[1] = np.array([2, 3, 2, 3, 2, 3, 2, 3])
    topk[2] = np.array([1, 5, 1, 5, 1, 5, 1, 5])
    # Padding rows: adversarial cycle over 0..31 to inflate raw unique.
    seq = np.arange((T - valid) * K) % 32
    topk[valid:] = seq.reshape(T - valid, K)
    raw_unique = np.unique(topk)
    masked_unique = fn(topk, valid)
    assert len(raw_unique) > S - 1, f"setup weak: raw {len(raw_unique)} <= 31"
    assert len(masked_unique) <= S - 1, (
        f"padding inflated footprint: {len(masked_unique)} > 31: {masked_unique}")
    assert set(masked_unique.tolist()) == {0, 1, 2, 3, 5}, masked_unique.tolist()
    print(f"PASS padded-no-fallback raw={len(raw_unique)} masked={len(masked_unique)}")


def test_valid_overflow_triggers():
    fn, _ = _load_footprint_fn()
    T, K, valid, S = 16, 16, 3, 32
    topk = np.zeros((T, K), dtype=np.int64)
    # Valid rows alone cover 0..31 (48 slots hold 32 distinct).
    topk[0] = np.arange(0, 16, dtype=np.int64)
    topk[1] = np.arange(16, 32, dtype=np.int64)
    topk[2] = np.arange(0, 16, dtype=np.int64)
    topk[valid:] = 7  # padding content irrelevant: masked to 0 downstream
    masked_unique = fn(topk, valid)
    assert len(masked_unique) > S - 1, (
        f"valid overflow missed: {len(masked_unique)} <= 31")
    print(f"PASS valid-overflow-triggers masked={len(masked_unique)}")


def test_none_preserves_raw():
    fn, _ = _load_footprint_fn()
    rng = np.arange(16 * 8, dtype=np.int64).reshape(16, 8) % 32
    assert np.array_equal(fn(rng, None), np.unique(rng))
    print("PASS none-preserves-raw")


def test_source_masks_before_unique():
    _, src = _load_footprint_fn()
    assert "_footprint_unique_ids(topk_ids_np" in src, "call site must use helper"
    assert "masked[n_valid:, :] = 0" in src, "helper must clamp rows >= n_valid to 0"
    assert "extra_kwargs.get(\"num_valid_tokens\"" in src, "must read num_valid_tokens"
    assert "bank.route(logits_np" in src, "legacy fallback path must remain"
    print("PASS source-masks-before-unique")


def test_ensure_resident_receives_masked_ids():
    """Negative control: ensure_resident at :278 must receive
    masked IDs via _footprint_unique_ids, NOT bare topk_ids_np.
    FAILS on unpatched ac2f2bc4 (arg is bare topk_ids_np) —
    that bug sent 36 raw padded IDs to ensure_resident while
    the S-1 check at :253 saw masked ≤31, causing
    expert_offload.py:633 RuntimeError."""
    src = MOE_PATH.read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "ensure_resident"):
            found = True
            first_arg = node.args[0]
            assert isinstance(first_arg, ast.Call), (
                f"ensure_resident arg is bare {ast.dump(first_arg)[:80]} "
                f"— not _footprint_unique_ids call")
            # _footprint_unique_ids is a module-level function:
            # func is ast.Name, not ast.Attribute
            assert isinstance(first_arg.func, ast.Name), (
                f"ensure_resident arg.func is {ast.dump(first_arg.func)[:80]} "
                f"— not a Name (expected _footprint_unique_ids)")
            assert first_arg.func.id == "_footprint_unique_ids", (
                f"ensure_resident arg is {first_arg.func.id!r} "
                f"— expected _footprint_unique_ids")
    assert found, "ensure_resident call not found in moe.py"
    print("PASS ensure-resident-receives-masked-ids")


def test_masked_unique_fits_slots():
    """Positive: T=16/K=8/valid=3/S=32, padding covers 0..31.
    Masked unique ≤ S-1=31 so ensure_resident receives ≤31
    unique experts — no RuntimeError at expert_offload.py:633."""
    fn, _ = _load_footprint_fn()
    T, K, valid, S = 16, 8, 3, 32
    topk = np.zeros((T, K), dtype=np.int64)
    topk[0] = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    topk[1] = np.array([2, 3, 2, 3, 2, 3, 2, 3])
    topk[2] = np.array([1, 5, 1, 5, 1, 5, 1, 5])
    seq = np.arange((T - valid) * K) % 32
    topk[valid:] = seq.reshape(T - valid, K)
    masked = fn(topk, valid)
    assert len(masked) <= S - 1, (
        f"masked unique {len(masked)} > S-1={S-1}: {masked}")
    assert 0 in masked, "expert 0 (padding slot) must be in masked set"
    print("PASS masked-unique-fits-slots")


if __name__ == "__main__":
    test_padded_no_fallback()
    test_valid_overflow_triggers()
    test_none_preserves_raw()
    test_source_masks_before_unique()
    test_ensure_resident_receives_masked_ids()
    test_masked_unique_fits_slots()
    print("ALL CPU CHECKS PASSED")
