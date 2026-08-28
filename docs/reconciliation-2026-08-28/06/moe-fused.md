# reconcile(moe-fused) — 06 invariant record

Files: `tpu_inference/layers/vllm/interface/moe.py`, `tpu_inference/layers/vllm/custom_ops/fused_moe.py`
Resolved conflict: content (dry-run, fork interception block).

Invariant: an aborted precompile trace must not poison the next forward (shared-expert output slot
cleared at entry; TPU write/read pair synchronous within one forward); when
`expert_offload.get_bank(layer.layer_name)` is None the interception is inert.

Decision: HYBRID — upstream structure kept, fork interception injected.

Evidence (HEAD): `moe.py:38` import, `:172` `bank = expert_offload.get_bank(...)`; shared-expert slot
clear at `fused_moe.py` forward entry (target-delta Target 4).

Verify: `python -m py_compile` OK; `rg '^<<<<<<<'` = 0.
Full entry: task/evidence/06-conflicts/resolution-log.md §T4.
