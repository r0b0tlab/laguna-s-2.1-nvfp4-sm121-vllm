#!/usr/bin/env python3
"""Fail-closed pre-load audit for the Laguna S 2.1 SM121 runtime."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

MANIFEST_PATH = Path("/opt/r0b0tlab/runtime-manifest.json")


def fp4_microtest() -> dict[str, Any]:
    import torch
    from vllm import _custom_ops as ops

    torch.manual_seed(423)
    dtype = torch.bfloat16
    m, n, k = 128, 128, 128
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    b = torch.randn((n, k), dtype=dtype, device="cuda")
    fp4_max = 6.0
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    a_global = ((fp8_max * fp4_max) / torch.amax(a.flatten())).float()
    b_global = ((fp8_max * fp4_max) / torch.amax(b.flatten())).float()
    alpha = 1.0 / (a_global * b_global)
    aq, a_scale = ops.scaled_fp4_quant(a, a_global)
    bq, b_scale = ops.scaled_fp4_quant(b, b_global)
    out = ops.cutlass_scaled_fp4_mm(aq, bq, a_scale, b_scale, alpha, dtype)
    torch.cuda.synchronize()
    reference = torch.matmul(a, b.t())
    denom = reference.float().abs().mean().item()
    nmae = (out.float() - reference.float()).abs().mean().item() / max(denom, 1e-6)
    return {
        "shape": [m, n, k],
        "finite": bool(torch.isfinite(out).all().item()),
        "nonzero": bool(torch.count_nonzero(out).item()),
        "normalized_mean_absolute_error": nmae,
        "ok": bool(torch.isfinite(out).all().item()) and bool(torch.count_nonzero(out).item()) and nmae <= 0.25,
    }


def main() -> int:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
        add("runtime_manifest", True, str(MANIFEST_PATH))
    except Exception as exc:
        manifest = {}
        add("runtime_manifest", False, repr(exc))

    expected_packages = {
        "vllm": manifest.get("vllm_package_version"),
        **(manifest.get("flashinfer_package_versions") or {}),
        "cuda-tile": manifest.get("cuda_tile"),
        "nccl4py": manifest.get("nccl4py"),
    }
    add("profile", manifest.get("profile") in {"production-fp8-ar-candidate", "production-fp8-dflash", "production-fp8-dflash-k7"}, manifest.get("profile"))
    add("target_model", manifest.get("model_id") == "poolside/Laguna-S-2.1-NVFP4", manifest.get("model_id"))
    add("target_revision", manifest.get("model_revision") == "216d1f13878dd4e715bc7412848d0f330e95bba6", manifest.get("model_revision"))
    add("draft_model", manifest.get("draft_model_id") == "poolside/Laguna-S-2.1-DFlash-NVFP4", manifest.get("draft_model_id"))
    add("draft_revision", manifest.get("draft_model_revision") == "723794750422b3efbf3a7b3af76dffb4ba035943", manifest.get("draft_model_revision"))
    add("fp8_kv_only", manifest.get("default_kv_cache_dtype") == "fp8" and manifest.get("nvfp4_kv_enabled") is False)

    try:
        import torch
        import vllm

        for package, expected in expected_packages.items():
            try:
                actual = metadata.version(package)
                add(f"package_{package}", isinstance(expected, str) and actual == expected, actual)
            except Exception as exc:
                add(f"package_{package}", False, repr(exc))
        add("torch_version", torch.__version__.startswith("2.11.0+cu130"), torch.__version__)
        add("torch_cuda", torch.version.cuda == "13.0", torch.version.cuda)
        capability = torch.cuda.get_device_capability()
        add("cuda_capability", capability == (12, 1), capability)
        add("vllm_module_version", getattr(vllm, "__version__", None) == expected_packages["vllm"], getattr(vllm, "__version__", None))
    except Exception as exc:
        add("core_imports", False, repr(exc))

    for module, class_name in (
        ("vllm.model_executor.models.laguna", "LagunaForCausalLM"),
        ("vllm.model_executor.models.laguna_dflash", "DFlashLagunaForCausalLM"),
        ("vllm.reasoning.poolside_v1_reasoning_parser", "PoolsideV1ReasoningParser"),
        ("vllm.tool_parsers.poolside_v1_tool_parser", "PoolsideV1ToolParser"),
    ):
        try:
            imported = __import__(module, fromlist=[class_name])
            add(f"import_{class_name}", hasattr(imported, class_name), module)
        except Exception as exc:
            add(f"import_{class_name}", False, repr(exc))

    for module in ("vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch"):
        try:
            __import__(module)
            add(f"import_{module}", True)
        except Exception as exc:
            add(f"import_{module}", False, repr(exc))

    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            text = subprocess.check_output([nvcc, "--version"], text=True, timeout=20)
            add("nvcc_13_0", "release 13.0" in text, text.splitlines()[-1])
        except Exception as exc:
            add("nvcc_13_0", False, repr(exc))
    else:
        add("nvcc_13_0", False, "not on PATH")

    dependency_checker = shutil.which("check_dependencies.py")
    try:
        result = subprocess.run([dependency_checker] if dependency_checker else ["/missing/check_dependencies.py"], text=True, capture_output=True, timeout=120, check=False)
        add("dependency_check", result.returncode == 0, (result.stdout or result.stderr).strip())
    except Exception as exc:
        add("dependency_check", False, repr(exc))

    try:
        help_result = subprocess.run(["vllm", "serve", "--help=all"], text=True, capture_output=True, timeout=180, check=False)
        help_text = help_result.stdout + help_result.stderr
        required = ("--speculative-config", "--reasoning-parser", "--tool-call-parser", "--override-generation-config", "--max-model-len", "--max-num-seqs", "--max-num-batched-tokens")
        add("vllm_cli_contract", help_result.returncode == 0 and all(flag in help_text for flag in required), {flag: flag in help_text for flag in required})
    except Exception as exc:
        add("vllm_cli_contract", False, repr(exc))

    add("max_jobs", os.getenv("MAX_JOBS") == "4", os.getenv("MAX_JOBS"))
    add("nvcc_threads", os.getenv("NVCC_THREADS") == "2", os.getenv("NVCC_THREADS"))
    add("flashinfer_nvcc_threads", os.getenv("FLASHINFER_NVCC_THREADS") == "2", os.getenv("FLASHINFER_NVCC_THREADS"))
    add("cute_dsl_arch", os.getenv("CUTE_DSL_ARCH") == "sm_121a", os.getenv("CUTE_DSL_ARCH"))
    cache = Path(os.getenv("FLASHINFER_CACHE_DIR", str(Path.home() / ".cache/flashinfer")))
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / ".r0b0tlab-write-test"
        probe.write_text("ok")
        probe.unlink()
        add("flashinfer_cache_writable", True, str(cache))
    except Exception as exc:
        add("flashinfer_cache_writable", False, repr(exc))

    model_path = Path(os.getenv("MODEL_PATH", "/models/Laguna-S-2.1-NVFP4"))
    if model_path.is_dir():
        try:
            config = json.loads((model_path / "config.json").read_text())
            quant = config.get("quantization_config") or {}
            add("mounted_model_architecture", config.get("architectures") == ["LagunaForCausalLM"], config.get("architectures"))
            add("mounted_model_quantization", quant.get("format") == "nvfp4-pack-quantized" and quant.get("quant_method") == "compressed-tensors", quant)
        except Exception as exc:
            add("mounted_model_config", False, repr(exc))

    try:
        micro = fp4_microtest()
        add("native_fp4_microtest", micro["ok"], micro)
    except Exception as exc:
        add("native_fp4_microtest", False, repr(exc))

    failed = [item for item in checks if not item["ok"]]
    report = {"schema_version": 1, "status": "PASS" if not failed else "FAIL", "manifest": manifest, "checks": checks}
    print("R0B0TLAB_RUNTIME_AUDIT=" + json.dumps(report, sort_keys=True, default=str))
    for item in checks:
        print(f"{'PASS' if item['ok'] else 'FAIL'} {item['name']}: {item['detail']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
