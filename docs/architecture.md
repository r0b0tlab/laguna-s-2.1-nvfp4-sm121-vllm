# Architecture

## Target checkpoint

`poolside/Laguna-S-2.1-NVFP4@07614121b31898586430f189d27a25a0be310843` is Poolside's updated "spinquantless norot weights, 256K" checkpoint. It is a 48-layer sparse MoE model with hidden size 3,072, 48 attention heads, eight KV heads, 256 routed experts, and top-10 routing. The checkpoint quantizes routed expert gate/up/down projections using compressed-tensors `nvfp4-pack-quantized`; non-targeted parameters retain their published dtypes.

The model alternates full and 512-token sliding-window attention and uses distinct RoPE policies. Full-attention layers use YaRN for the 262,144-token envelope. The runtime must preserve the model's per-layer attention behavior; applying the sliding window globally is incorrect.

## Native SM121 path

The production contract is:

1. vLLM's native `LagunaForCausalLM` implementation.
2. Compressed-tensors native NVFP4 routed-expert dispatch.
3. FlashInfer/CUTLASS NVFP4 kernels on compute capability 12.1.
4. FlashInfer attention.
5. FP8 KV cache.
6. CUDA graphs after cold JIT.

Marlin, weight-only fallback, dequantize-then-GEMM emulation, and NVFP4 KV are outside the release contract. The runtime audit executes a real FP4 quantize/GEMM micro-test before model load; server logs and Nsight evidence prove the loaded model's active path.

## DFlash

`poolside/Laguna-S-2.1-DFlash-NVFP4@723794750422b3efbf3a7b3af76dffb4ba035943` is a separate drafter. It shares target embeddings and LM head through vLLM's DFlash proposer and consumes selected target hidden states. It never replaces the target checkpoint.

AR is the correctness and performance baseline. DFlash K values 3, 5, 7, 11, and 15 are tested on identical prompts. Promotion requires positive exact drafted/accepted deltas, matched semantic/tool/context behavior, and an end-to-end win after TTFT and JIT effects.

## Container boundary

The source-built vLLM environment is copied into a minimal Ubuntu 24.04 runtime with CUDA 13.0 development components retained for FlashInfer JIT. The audit-before-exec entrypoint validates package, CUDA, GPU, model, parser, native-GEMM, dependency, and cache contracts. The FlashInfer cache lives at `/var/cache/flashinfer`, which supports an arbitrary non-root runtime UID when mounted persistently.
