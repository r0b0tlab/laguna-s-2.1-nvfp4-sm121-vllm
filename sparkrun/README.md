# SparkRun

This recipe is the non-root packaging surface for Laguna S 2.1 NVFP4 on one GB10.

During local qualification it points to the exact local candidate tag. Phase 4 replaces that value with the immutable public GHCR digest and reruns the real non-root launch contract before release.

```bash
sparkrun registry add https://github.com/r0b0tlab/laguna-s-2.1-nvfp4-sm121-vllm
sparkrun recipe validate @r0b0tlab/laguna-s-2.1-nvfp4-vllm-r0b0tlab
sparkrun run @r0b0tlab/laguna-s-2.1-nvfp4-vllm-r0b0tlab --solo
```

Local validation:

```bash
sparkrun recipe validate sparkrun/recipes/laguna-s-2.1-nvfp4-vllm-r0b0tlab.yaml
```

The recipe intentionally defaults to AR until the matched DFlash depth sweep selects and qualifies a production K. It keeps FP8 KV, 262,144-token configured context, Poolside parsers, and capped FlashInfer JIT fan-out.
