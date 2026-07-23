# Poolside checkpoint 0761412 quick validation

Status: PASS

This update validates Poolside's `poolside/Laguna-S-2.1-NVFP4@07614121b31898586430f189d27a25a0be310843` checkpoint, published as `nvfp4: spinquantless norot weights, 256K`. It reuses the repository's qualified vLLM 0.25.1, native SM121, FP8 KV, and DFlash K=7 profile. No DFlash K sweep was repeated.

## Native and API sanity

- Runtime audit: PASS
- Active NVFP4 MoE backend: `FLASHINFER_CUTLASS`
- Active Marlin, emulation, or fallback markers: 0
- DFlash K=7: active
- Speculative counters: 9121 drafted / 1508 accepted
- KV cache: FP8
- SM capability: 12.1

- long_generation: PASS (512 completion tokens)
- math: PASS (13 completion tokens)
- multi_tool: PASS (77 completion tokens)
- preserved_reasoning: PASS (3 completion tokens)
- reasoning: PASS (128 completion tokens)
- tool: PASS (35 completion tokens)

## llama-benchy 0.4.0

Method: 2,048-token prompt, exactly 128 generated tokens, concurrency 1, three measured runs per depth, API latency mode, cache-busting requests, thinking disabled. The coherence gate passed.

| Prior context tokens | Prefill tok/s | Decode tok/s | Peak tok/s | E2E TTFT ms |
|---:|---:|---:|---:|---:|
| 0 | 2,383.85 | 22.23 | 38.67 | 863.36 |
| 4,096 | 3,394.94 | 20.67 | 27.00 | 1,814.22 |
| 8,192 | 3,309.69 | 20.55 | 26.67 | 3,098.55 |
| 16,384 | 3,313.48 | 19.88 | 27.33 | 5,568.57 |

API latency: 3.76 ms.

## Evidence boundary

This is a short checkpoint-update validation, not a new full model qualification. The existing 8,620-case BFCL, generated-answer, IFEval, and HumanEval battery remains historical evidence for target revision `216d1f13878dd4e715bc7412848d0f330e95bba6`. Results from the two revisions are not merged or relabeled.

Files:

- `sanity-summary.json`: sanitized canary and native-path verdict
- `llama-benchy.json`: aggregate llama-benchy output with host paths removed
- `provenance.json`: checkpoint, runtime, image-parent, and method identity
- `SHA256SUMS`: artifact checksums
