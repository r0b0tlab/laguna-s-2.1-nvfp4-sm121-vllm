# Qualification methodology

## Identity gate

Every result is tied to exact target/draft revisions, vLLM tag and commit, PyTorch/CUDA versions, the FlashInfer trio, repository commit, local image ID, and public image digest. The latest stable vLLM resolver requires GitHub Release, tag, and PyPI agreement immediately before build.

## Admission and isolation

Heavy work runs only after `scripts/admit_node.sh` verifies the expected GB10 host, compute capability 12.1, minimum disk headroom, no unrelated containers, and no competing build/model/benchmark processes. Downloads, source build, JIT, server load, and profilers do not overlap.

## Correctness and native path

Qualification requires:

- runtime audit and real FP4 GEMM micro-test;
- `/health` and exact `/v1/models` identity;
- semantic, reasoning, single-tool, multi-tool, preserved-reasoning, and long-generation probes;
- positive FlashInfer attention and native NVFP4 kernel evidence;
- zero active Marlin, emulation, fallback, corruption, NaN, or OOM markers;
- Nsight Systems CUDA-kernel evidence.

## Performance

AR is measured first. DFlash K={3,5,7,11,15} uses identical prompts, sampling, context, batch limits, and warmed JIT cache. Narrow sweeps identify candidates; only the selected profile receives the full concurrency ramp 1, 2, 4, 8, 16, 32 with three measured repeats after warmup.

The harness reports client output throughput separately from server prompt/decode throughput. TTFT is request start to first non-empty SSE output. ITL is measured between subsequent non-empty SSE outputs. Power, temperature, GPU utilization, host available memory, swap, all request errors, and exact speculative counters are recorded.

## Context and quality

Configured context is 262,144 tokens. The context probe uses `/tokenize` to calibrate prompts with output headroom and retrieves a unique needle near the beginning, middle, and end at 64K, 128K, and 262K targets.

Quality is the exact 1,319-example `openai/gsm8k` test split at revision `740312add88f781978c0658806c59bc2815b9866`. The parquet SHA-256 is verified. The protocol is 0-shot Chat Completions with thinking disabled, flexible numeric extraction, and zero request errors.

## Publication

Only sanitized summaries, reports, recipes, and manifests are public. Raw prompts, samples, logs, traces, hostnames, private paths, LAN addresses, and credentials remain excluded. Release requires clean-clone tests, immutable GHCR digest, anonymous pull, non-root execution, real SparkRun validation, and public-safety scanning.
