# reconcile(weight-loading) — 06 invariant record

Files: `tpu_inference/layers/vllm/quantization/unquantized.py`, `tests/layers/vllm/test_unquantized.py`
Resolved conflict: content (dry-run `b037a1736` vs base `d6c3a7ad:unquantized.py:104`).

Invariant: unquantized TPU load = host-canonical bf16 bit-cast (`_host_numpy_from_torch`) → direct
`device_put` (no bf16→f32 upcast, no 4 GiB VFIO RESOURCE_EXHAUSTED). Opportunistic CPU-storage
release tolerates non-resizable (safetensors-backed) storage.

Decision: PORT LOCAL — upstream `d6c3a7ad` has plain `t2j` and zero `_host_numpy_from_torch` (grep 0).
Upstream does not implement the same concern.

Evidence (HEAD): `_host_numpy_from_torch` at `unquantized.py:69`; call at `:130`; `_release_cpu_storage`
count = 8; 4 unit tests preserved in `test_unquantized.py` (67-line delta).

Verify: `python -m py_compile` OK; `rg '^<<<<<<<'` = 0.
Full entry: task/evidence/06-conflicts/resolution-log.md §T1/T2.
