# Session Handoff — 2026-09-13 TPU Flash B04

## Serving Truth
- **vLLM**: a4d1a25bf2bec0c30940dbcb1fba9f9a4ddf2ecb
- **Canonical i-j-port**: 697cdf043890861d23f580dcebe492ca250ae589

## Candidate Lineage (full SHAs)
- 697cdf043890861d23f580dcebe492ca250ae589 → fix-replicated-kv-mesh-conditional-i-j-port (i-j-port origin)
- ac2f2bc4571a169337d7b23bc004f432e3ebb532 → fix-moe-padding-footprint-i-j-port (count padding rows as expert 0)
- 410afe51e236e87bb1dd9c8b35e985f2a138ff2c → fix-moe-padding-footprint-ii-i-j-port (pass masked IDs to ensure_resident)
- 619c8da4217cb8790e82cd04bb6601a4a2ca11c9 → B01 wave reconcile onto 410afe5
- 636cfed35a1db5e4cf90102e08bcb090fd945135 → B02 _compute_waves domain fix
- 638008038467dc2e7ab7b4c6674c76504c9bfacc → B03 GMM-aligned execution waves
- 5c6ab17e94b34d98eed45181735366ebfc9d2166 → first B04 attempt
- 237430b317eb9885fdbdbe19868b22f141e4ea58 → corrected B04 (diag-only)

## Run09 Conclusion
- Legacy: 0
- Wave-overs entered wave path: 40
- Need-slots: 0
- GMM assert: 0
- No traceback, no completion
- Stall after final MoE host orchestration
- True boundary UNKNOWN pending B04

## Next Action
- Run vLLM a4d1a25 + 237430b3 same Run09 geometry
- Read B04 markers for first missing READY/RETURN

## B03 VALIDITY-DOMAIN BUG CONFIRMED
- wave_valid rebased but footprint uses global _num_valid_int
- _compute_waves sees raw physical rows
- NOT fixed in B04
- Fix only after localization

## Backlog
- 375b13b5 runner
- 4420cae Mamba prefix-cache
- be74283d FP8 host staging
- stock vLLM Qwen4Exp does not replace TPU port — record, do not integrate

## Runtime Evidence Filenames
- run09_clean_report.md
- Run08 report
- B04 state

