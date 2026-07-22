# Full 8,620-case benchmark methodology

This result bundle evaluates the immutable production image:

`ghcr.io/r0b0tlab/vllm-laguna-s-2.1-nvfp4-sm121@sha256:8b0e3d07dad370853bca77441ff7c8619c41a0cb80d59a150bb0faf17ee15ef3`

The endpoint served exact target revision `216d1f13878dd4e715bc7412848d0f330e95bba6` with exact DFlash draft revision `723794750422b3efbf3a7b3af76dffb4ba035943`, DFlash K=7, FP8 KV cache, and native FlashInfer CUTLASS NVFP4 on one NVIDIA GB10 / SM121.

## Official logical cases

| Lane | Cases |
|---|---:|
| BFCL v4 `multi_turn_base` | 200 |
| Generated-answer quality | 7,715 |
| IFEval | 541 |
| HumanEval | 164 |
| **Total** | **8,620** |

The generated-answer lane contains GSM8K, ARC-Challenge, PIQA, WinoGrande XL, TruthfulQA MC1, and eight frozen MMLU subjects. Dataset cardinalities, Hugging Face fingerprints, and canonical row-content SHA-256 values were checked before execution.

## Request contract

Non-BFCL requests used Chat Completions with temperature 0, top-p 1, seed 0, runtime-default reasoning, and a fixed 32,768-token completion envelope. Wrong answers, extraction failures, empty visible outputs, and length stops were retained and scored as model outcomes. Only transport, endpoint-identity, persistence, scorer, or sandbox failures could stop the campaign.

BFCL used `bfcl-eval==2025.12.17`, official `multi_turn_base`, temperature 0.001, one generation thread, and a 32,768-token envelope. All 200 canonical IDs were generated and evaluated in one package process against the already-running endpoint.

## Concurrency admission and calibration

A 15-schema probe ran at c1, c2, c4, c8, and c16. The admitted concurrency was the highest contiguous candidate whose extracted/scored semantics exactly matched c1. c2 passed; c4 and c8 differed on WinoGrande, so c2 was frozen even though c16 happened to match the one-case probe.

Two fresh 105-case calibration repeats then ran at c2. Their 210 logical cases were excluded from official totals and were never reused in the full run. Six temperature-zero semantic disagreements were retained as diagnostics rather than cherry-picked.

## Official scorers and sandbox

IFEval used `lm_eval==0.4.12` and pinned scorer implementation SHA-256 `1ab8f14808c826f93f2364883487ed63cf4267980bf4761fda8053899c013632`. Both prompt- and instruction-level strict/loose metrics are reported.

HumanEval used one completion per problem and executed candidates in digest-pinned Python containers with networking disabled, a read-only root filesystem, all capabilities dropped, `no-new-privileges`, non-root UID/GID 1001, bounded CPU/memory/PIDs, and a 16-second timeout. Sandbox allow/deny preflight passed before official execution. No HumanEval infrastructure timeouts occurred.

## Evidence handling

Every official logical case produced an atomic, resumable private row bound to the immutable run manifest. Public files contain only sanitized aggregate evidence. Raw prompts, responses, host paths, credentials, and model weights are not published.

There is no opaque composite score. BFCL, generated-answer tasks, IFEval, and HumanEval are reported separately. Previously published performance, context, power, and Nsight evidence was reused only after immutable image/profile equivalence was verified against the pinned pre-battery result commit.
