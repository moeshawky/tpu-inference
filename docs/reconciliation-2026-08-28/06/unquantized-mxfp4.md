# reconcile(unquantized-mxfp4) — 06 invariant record

Files: `tpu_inference/layers/vllm/quantization/mxfp4.py`
Resolved conflict: content (dry-run `3dd3765da`/`9ef180e02`).

Invariant: MXFP4 requant block size = 256 for TP=8 divisibility (w2 contraction 2048/512 is not
shardable across MLP_TENSOR=8). No upstream equivalent (`d6c3a7ad` MXFP4_REQUANTIZED_BLOCK_SIZE grep 0).

Decision: PORT LOCAL.

Evidence (HEAD): `MXFP4_REQUANTIZED_BLOCK_SIZE` at `mxfp4.py:50` and `:223`. Delta `--stat: 150 +-`.

Verify: `python -m py_compile` OK; `rg '^<<<<<<<'` = 0.
Full entry: task/evidence/06-conflicts/resolution-log.md §T5.
