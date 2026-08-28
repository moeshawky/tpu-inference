# reconcile(expert-offload) — 06 invariant record

Files: `tpu_inference/layers/vllm/expert_offload.py`, `tpu_inference/envs.py`,
`tests/layers/vllm/test_expert_offload.py`
Resolved conflict: modify/delete (dry-run `42e22b998`/`3e9d33a38`/`4c959be92`) → KEPT.

Invariant: host-backed expert residency for sparse MoE (S device slots + LRU host refresh + [T,S]
gating remap); DSV4EPRS v1 store is a breaking boundary — any on-disk store version-checked by magic.
13 `MOE_EXPERT_OFFLOAD*` env vars drive the bank.

Decision: PORT LOCAL (keep). NOT the "upstream removed abstraction" STOP case — upstream `d6c3a7ad`
has zero `expert_offload` references (measured grep 0). This is a fork-only feature carried forward.

Evidence (HEAD): `expert_offload.py` 1336 lines present; `interface/moe.py:38` imports it, `:172`
`bank = expert_offload.get_bank(...)`; `envs.py:86-111` 13 vars; `test_expert_offload.py` 687 lines.

Verify: `python -m py_compile` OK; `rg '^<<<<<<<'` = 0.
Full entry: task/evidence/06-conflicts/resolution-log.md §T3/T6/T7.
