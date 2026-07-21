# Troubleshooting

## Cold FlashInfer JIT appears stalled

The first native Laguna MoE launch can spend substantial time compiling and may use most unified memory. Keep `MAX_JOBS=4`, `NVCC_THREADS=2`, and `FLASHINFER_NVCC_THREADS=2`. Inspect live logs and memory; do not restart merely because output is quiet. Preserve and reuse the cache mounted at `/var/cache/flashinfer`.

## Build exits during native compilation

Check the explicit build return code and the first real compiler error. Do not infer success from an image tag. Verify disk and available memory, then retry the exact source commit so BuildKit can reuse successful layers. Do not increase compile fan-out and do not prune global Docker caches.

## Dependency check reports FlashInfer 0.6.13 mismatch

vLLM 0.25.1 metadata pins FlashInfer 0.6.13, while Poolside's GB10 recipe requires the exact `0.6.15.dev20260712` python/cubin/jit-cache trio. `check_dependencies.py` accepts only those two exact metadata mismatch lines and the separately verified NVIDIA SBSA wheel-tag warning. Any additional warning fails closed.

## Model load selects a fallback

Stop qualification. Confirm the exact checkpoint revision and quantization metadata, compute capability 12.1, native FlashInfer trio, and current server logs. Do not substitute Marlin, a BF16 checkpoint, a repack, or dequantize-then-GEMM. Capture authoritative evidence before changing flags.

## DFlash has no positive counters

The run is invalid as a DFlash result. Confirm the draft path/revision, `method=dflash`, positive `num_speculative_tokens`, and that vLLM exports drafted/accepted counters. Do not report a speculative speedup from elapsed time alone.

## Long-context request is rejected

Use `context_probe.py` rather than estimating words or bytes. It calibrates with `/tokenize` and reserves generation headroom. Verify the server was actually launched with 262,144 configured tokens and that reported prompt usage is within the requested window.

## Non-root launch cannot write the JIT cache

Mount a writable persistent directory at `/var/cache/flashinfer`. The image creates that directory with sticky world-writable semantics specifically for arbitrary runtime UIDs. Do not redirect a non-root runtime to `/root`.

## Disk usage is high

Use project-scoped inspection first. Never run global Docker, BuildKit, Hugging Face, or package-cache pruning on a shared node. Remove only recorded project-owned artefacts after verifying exact names, labels, and paths.
