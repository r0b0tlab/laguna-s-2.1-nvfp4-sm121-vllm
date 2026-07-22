# Laguna S 2.1 NVFP4 on NVIDIA GB10

A production-oriented, fail-closed vLLM implementation for the exact Poolside checkpoint [`poolside/Laguna-S-2.1-NVFP4`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4) on NVIDIA GB10 / SM121.

This repository follows r0b0tlab's proven release-contract structure: immutable dependency/model identity, source-built runtime, audit-before-exec entrypoint, native-path gates, matched AR/speculative qualification, machine-readable evidence, and SparkRun packaging.

## Status

Implementation candidate. No performance, quality, context, registry, or release claim is made until `VERDICT.md` and the checksummed public result bundle exist.

## Exact model contract

| Component | Identity |
|---|---|
| Target | `poolside/Laguna-S-2.1-NVFP4@216d1f13878dd4e715bc7412848d0f330e95bba6` |
| Optional draft | `poolside/Laguna-S-2.1-DFlash-NVFP4@723794750422b3efbf3a7b3af76dffb4ba035943` |
| KV cache | FP8 |
| Configured context | 262,144 tokens |
| Reasoning/tool parsers | `poolside_v1` |

The target is not interchangeable with Laguna BF16, FP8, INT4, GGUF, XS, or M checkpoints. The DFlash repository is only a speculative accelerator and never replaces the target model.

## Runtime selection

`scripts/resolve_vllm_release.py` resolves the newest official non-draft, non-prerelease vLLM release by requiring GitHub Releases, the GitHub tag, and non-yanked PyPI metadata to agree. The exact tag and full commit are then frozen in `docker/dependency-manifest.json`.

Current lock (2026-07-21):

- vLLM `v0.25.1` / `752a3a504485790a2e8491cacbb35c137339ad34`
- PyTorch `2.11.0+cu130`
- CUDA 13.0
- FlashInfer `0.6.15.dev20260712` python/cubin/jit-cache trio

A newer stable release available before build becomes the primary candidate. Untagged `main` is not used for production by default.

## Build

On an admitted GB10 node:

```bash
python3 scripts/resolve_vllm_release.py --write docker/dependency-manifest.json
bash scripts/admit_node.sh

tmux new-session -d -s laguna-s21-build \
  "bash scripts/build_release.sh"
```

The durable build writes its log, return code, image inspection, and PASS/FAIL stamps under ignored private evidence storage. Build and runtime JIT are capped at `MAX_JOBS=4`, `NVCC_THREADS=2`, and `FLASHINFER_NVCC_THREADS=2`.

## Run AR baseline

```bash
docker run --rm --gpus all --ipc=host --network host \
  -v /path/to/Laguna-S-2.1-NVFP4:/models/Laguna-S-2.1-NVFP4:ro \
  -v /path/to/flashinfer-cache:/var/cache/flashinfer \
  laguna-s21-sm121-vllm:v0.25.1-candidate
```

The default remains AR until the DFlash depth sweep is complete.

## Run DFlash candidate

```bash
docker run --rm --gpus all --ipc=host --network host \
  -e DFLASH_TOKENS=7 \
  -v /path/to/Laguna-S-2.1-NVFP4:/models/Laguna-S-2.1-NVFP4:ro \
  -v /path/to/Laguna-S-2.1-DFlash-NVFP4:/models/Laguna-S-2.1-DFlash-NVFP4:ro \
  -v /path/to/flashinfer-cache:/var/cache/flashinfer \
  laguna-s21-sm121-vllm:v0.25.1-candidate
```

Matched K={3,5,7,11,15} qualification selected K=7 for the production profile. DFlash requests must expose positive drafted/accepted counter deltas.

## Safety boundaries

- Native compressed-tensors NVFP4 expert path only; no active Marlin/emulation/fallback.
- FP8 KV only. NVFP4 KV is rejected by the entrypoint.
- Do not force `flashinfer_b12x` on vLLM 0.25.1.
- No `min_p` under speculative decoding.
- No model weights are included in this repository or image.
- The software performs no telemetry, callbacks, or tracking.

## Verification

```bash
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py tests/*.py
python3 -m unittest discover -s tests -v
python3 scripts/public_safety_scan.py .
```

Live qualification tools:

```bash
# Matched AR or DFlash performance, latency, metrics, and telemetry
python3 scripts/benchmark.py --model poolside/Laguna-S-2.1-NVFP4 \
  --container-name laguna-s21-ar --speculation disabled --output results/raw/ar.json

# Pinned 1,319-item GSM8K 0-shot run (requires pyarrow in the client environment)
python3 scripts/eval_gsm8k.py --output-dir results/raw/gsm8k

# Calibrated 64K, 128K, and 262K retrieval probes
python3 scripts/context_probe.py --output results/raw/context.json

# Parse a captured Nsight Systems report and reject fallback kernels
python3 scripts/profile_nvfp4.py profile.nsys-rep --output results/raw/nsight-kernels.json
```

`eval_gsm8k.py` pins dataset revision `740312add88f781978c0658806c59bc2815b9866` and verifies the test parquet SHA-256 before scoring. Raw samples and host-private evidence remain ignored; only sanitized, checksummed summaries are published.

A healthy API alone is insufficient. Release gates include exact identity, native-kernel markers, semantic/reasoning/tool canaries, long generation, AR/DFlash comparison, latency/throughput/power telemetry, 256K-context qualification, GSM8K 0-shot flexible extraction, immutable image verification, non-root SparkRun, anonymous pull, and clean clone.

Detailed references:

- [`docs/architecture.md`](docs/architecture.md) — checkpoint, native SM121 path, DFlash, and container boundaries
- [`docs/methodology.md`](docs/methodology.md) — identity, native, performance, context, quality, and publication gates
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — fail-closed recovery for build, JIT, model load, DFlash, and packaging issues

## License and credit

Repository code is MIT licensed. vLLM and FlashInfer are Apache-2.0 projects. Model use is governed by Poolside's OpenMDW-1.1 model license and acceptable-use terms. This repository redistributes no model weights.
