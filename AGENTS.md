# AGENTS.md

## Purpose

This repository builds, audits, qualifies, and publishes a reproducible vLLM runtime for the exact checkpoint `poolside/Laguna-S-2.1-NVFP4` on one NVIDIA GB10 / SM121 system.

The target model is never replaced by a similarly named Laguna checkpoint. `poolside/Laguna-S-2.1-DFlash-NVFP4` is an optional quantization-matched speculative drafter only.

## Release identity

- Target: `poolside/Laguna-S-2.1-NVFP4`
- Target revision: `216d1f13878dd4e715bc7412848d0f330e95bba6`
- Draft: `poolside/Laguna-S-2.1-DFlash-NVFP4`
- Draft revision: `723794750422b3efbf3a7b3af76dffb4ba035943`
- Platform: NVIDIA GB10 / SM121, aarch64
- vLLM: newest official non-prerelease release resolved immediately before build; current lock is `v0.25.1` at `752a3a504485790a2e8491cacbb35c137339ad34`
- Current Torch/CUDA: `2.11.0+cu130` / CUDA 13.0
- Current FlashInfer: `0.6.15.dev20260712` python/cubin/jit-cache trio
- KV cache: FP8 only
- Context contract: 262,144 tokens
- Parsers: `poolside_v1` reasoning and tool parsers

`docker/dependency-manifest.json` and `docker/runtime-manifest.production.json` are the machine-readable source of truth.

## Hardware isolation

All builds, downloads, model loads, JIT work, serving, and benchmarks run only on the admitted free node. `scripts/admit_node.sh` must pass before each heavy phase. Never stop or alter unrelated containers or processes. Never use global Docker, BuildKit, or Hugging Face cache pruning.

## Build constraints

1. Run `scripts/resolve_vllm_release.py` immediately before build. GitHub Releases, the tags API, and PyPI must agree.
2. Build from the exact selected vLLM tag and full commit. Untagged `main` is edge evidence, not a production identity.
3. Use the selected release's Torch/CUDA dependency contract. For v0.25.1, keep Torch cu130 and CUDA/nvcc 13.0.
4. Keep build and runtime JIT fan-out at `MAX_JOBS=4`, `NVCC_THREADS=2`, and `FLASHINFER_NVCC_THREADS=2`.
5. Preserve the FlashInfer cache across launches. A cold Laguna MoE compile can consume most unified memory.
6. Do not add Marlin, emulation, weight-only repacks, metadata shortcuts, or NVFP4 KV.
7. Run long builds in tmux with explicit return-code and PASS/FAIL stamps.
8. Keep expensive native compilation separate from fallible identity assertions.

## Runtime contract

The entrypoint always runs `scripts/audit_runtime.py` before executing any child. It preserves argv boundaries and child exit codes. The `audit` subcommand runs only the audit.

Production must use native compressed-tensors NVFP4 routed experts, FlashInfer attention, FP8 KV, CUDA graphs, and no active Marlin/emulation/fallback. AR is the initial baseline. DFlash is promoted only after a matched K sweep proves a correctness-preserving speedup.

Default sampling is temperature 0.7, top-p 0.95, with checkpoint top-k 20. Do not add `min_p` under speculative decoding. Do not force `flashinfer_b12x` on v0.25.1.

## Verification

Before committing or publishing:

```bash
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py tests/*.py
python3 -m unittest discover -s tests -v
python3 scripts/public_safety_scan.py .
```

Before a runtime or benchmark claim:

1. `docker run --rm --gpus all <image> audit`
2. Verify exact image/model/runtime identity.
3. Verify `/health`, `/v1/models`, semantic output, reasoning, single/multi-tool calls, preserved reasoning, and long generation.
4. Verify positive native FlashInfer/CUTLASS NVFP4 markers and zero active Marlin/emulation/fallback markers.
5. Qualify matched AR before DFlash and require exact drafted/accepted counter deltas for DFlash.
6. Record prompt/decode throughput, TTFT, ITL, acceptance, power, temperature, utilization, host memory, swap, and all errors.
7. Use GSM8K 0-shot with flexible extraction and Chat Completions for release-safety quality.

Synthetic unit tests are scaffold evidence only, never GPU/runtime evidence.

## Publication rules

- Do not redistribute model weights.
- Public artefacts contain no credentials, private paths, LAN addresses, hostnames, raw diagnostics, or unpublished traces.
- Publish immutable source and image identities only after clean-clone, anonymous-pull, non-root, and SparkRun checks pass.
- Credit Poolside, vLLM, FlashInfer, PyTorch, and NVIDIA appropriately.
- Do not open upstream issues or pull requests unless explicitly requested.
