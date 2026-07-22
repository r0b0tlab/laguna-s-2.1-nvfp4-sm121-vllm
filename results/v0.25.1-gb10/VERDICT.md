# Release verdict

**PASS** — Laguna S 2.1 NVFP4 is qualified on one NVIDIA GB10 / SM121 using stable vLLM 0.25.1, native FlashInfer CUTLASS NVFP4, FP8 KV, and DFlash K=7.

Immutable image: `ghcr.io/r0b0tlab/vllm-laguna-s-2.1-nvfp4-sm121@sha256:8b0e3d07dad370853bca77441ff7c8619c41a0cb80d59a150bb0faf17ee15ef3`

- Target revision: `216d1f13878dd4e715bc7412848d0f330e95bba6`
- Draft revision: `723794750422b3efbf3a7b3af76dffb4ba035943`
- vLLM commit: `752a3a504485790a2e8491cacbb35c137339ad34`
- GSM8K 0-shot flexible exact match: `94.768764%` (1250/1319, zero request errors)
- Context retrieval: PASS at beginning/middle/end through 261,885 reported prompt tokens
- Non-root full inference: PASS as UID/GID 65532
- Nsight native FP4 gate: PASS

## Full 8,620-case battery

- Completion: PASS — 8,620/8,620 official logical cases
- BFCL v4 `multi_turn_base`: 137/200 (`68.50%`)
- Generated-answer lane: 6,838/7,715 (`88.63%`)
- IFEval prompt strict / loose: `82.44%` / `86.69%`
- IFEval instruction strict / loose: `87.53%` / `90.89%`
- HumanEval pass@1: 154/164 (`93.90%`), networkless non-root sandbox PASS
- Generated empty outputs / length stops: 0 / 0
- Public aggregate evidence: [report](full-battery/REPORT.md), [scorecard](full-battery/scorecard.json), and [methodology](full-battery/METHODOLOGY.md)

## Publication verification

- Public GitHub repository and release: PASS
- Public immutable GHCR manifest: PASS
- Anonymous immutable GHCR pull: PASS
- Anonymous clean clone, 48 tests, release verifier, and SparkRun 0.2.40 recipe validation: PASS
