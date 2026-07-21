#!/usr/bin/env python3
"""Fail-closed verifier for the Laguna S 2.1 public release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

EXPECTED = {
    "model_id": "poolside/Laguna-S-2.1-NVFP4",
    "model_revision": "216d1f13878dd4e715bc7412848d0f330e95bba6",
    "draft_model_id": "poolside/Laguna-S-2.1-DFlash-NVFP4",
    "draft_model_revision": "723794750422b3efbf3a7b3af76dffb4ba035943",
    "vllm": "0.25.1",
    "vllm_commit": "752a3a504485790a2e8491cacbb35c137339ad34",
    "torch": "2.11.0+cu130",
    "cuda": "13.0",
    "flashinfer": "0.6.15.dev20260712",
}
EXPECTED_CONCURRENCY = [1, 2, 4, 8, 16, 32]
EXPECTED_K = [3, 5, 7, 11, 15]
EXPECTED_IMAGE = "ghcr.io/r0b0tlab/vllm-laguna-s-2.1-nvfp4-sm121"
FORBIDDEN_PUBLIC_TEXT = ("/home/", "192.168.", "gn100-2eea", "r0b0tdgx", "r0b0t-dgx", "GITHUB_TOKEN", "HF_TOKEN")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path, manifest: Path, errors: list[str]) -> None:
    if not manifest.is_file():
        errors.append(f"missing manifest: {manifest}")
        return
    seen: set[str] = set()
    for line_number, raw in enumerate(manifest.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        if not match:
            errors.append(f"manifest line {line_number} is malformed")
            continue
        expected_hash, relative = match.groups()
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            errors.append(f"manifest path escapes result root: {relative}")
            continue
        if relative in seen:
            errors.append(f"duplicate manifest path: {relative}")
            continue
        seen.add(relative)
        if not candidate.is_file():
            errors.append(f"manifest file is missing: {relative}")
        elif sha256(candidate) != expected_hash:
            errors.append(f"manifest hash mismatch: {relative}")
    if not seen:
        errors.append("manifest is empty")


def verify(summary_path: Path, recipe_path: Path, readme_path: Path, html_path: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        summary = load_json(summary_path)
    except Exception as exc:
        return [f"cannot read summary: {exc}"]
    if summary.get("schema_version") != 1 or summary.get("status") != "PASS":
        errors.append("summary schema/status must be 1/PASS")

    release = summary.get("release") or {}
    for key, expected in EXPECTED.items():
        if release.get(key) != expected:
            errors.append(f"release.{key} must equal {expected!r}")
    if release.get("ghcr_image") != EXPECTED_IMAGE:
        errors.append("release.ghcr_image is unexpected")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(release.get("local_image_id", ""))):
        errors.append("release.local_image_id must be immutable")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(release.get("ghcr_digest", ""))):
        errors.append("release.ghcr_digest must be immutable")

    profile = summary.get("profile") or {}
    if profile.get("kv_cache_dtype") != "fp8":
        errors.append("production KV cache must be fp8")
    spec = profile.get("speculation") or {}
    if spec.get("method") != "dflash" or spec.get("num_speculative_tokens") not in EXPECTED_K:
        errors.append("production speculation must be a qualified DFlash K")
    if profile.get("configured_context_tokens") != 262144:
        errors.append("configured context must be exactly 262144")
    if profile.get("max_num_seqs") != 32:
        errors.append("max_num_seqs must be 32")

    sweep = summary.get("dflash_depth_sweep") or {}
    if sweep.get("tested_k") != EXPECTED_K:
        errors.append(f"DFlash tested_k must equal {EXPECTED_K}")
    if sweep.get("selected_k") != spec.get("num_speculative_tokens"):
        errors.append("selected DFlash K disagrees with production profile")
    if sweep.get("selection_rule_pass") is not True:
        errors.append("DFlash selection rule did not pass")

    native = summary.get("native_gate") or {}
    if native.get("runtime_audit") != "PASS":
        errors.append("runtime audit did not pass")
    for marker in ("marlin_markers", "emulation_markers", "fallback_markers"):
        if native.get(marker) != 0:
            errors.append(f"native_gate.{marker} must be zero")
    for marker in ("flashinfer_cutlass_nvfp4", "flashinfer_attention", "nsight_kernel_table"):
        if not native.get(marker):
            errors.append(f"native_gate.{marker} is absent")

    canaries = summary.get("canaries") or {}
    for name in ("models", "semantic", "reasoning", "tool_call", "multi_tool", "preserved_reasoning", "long_generation", "context_262k"):
        if canaries.get(name) is not True:
            errors.append(f"canary {name} did not pass")

    performance = summary.get("performance") or {}
    if performance.get("errors") not in ([], None):
        errors.append("performance evidence contains errors")
    rows = performance.get("rows") or []
    if [row.get("concurrency") for row in rows if isinstance(row, dict)] != EXPECTED_CONCURRENCY:
        errors.append(f"performance concurrency levels must be {EXPECTED_CONCURRENCY}")
    required = ("output_tokens_per_second", "prompt_tokens_per_second", "ttft_p50_seconds", "ttft_p90_seconds", "ttft_p99_seconds", "itl_p50_seconds", "itl_p90_seconds", "itl_p99_seconds", "power_mean_watts", "dflash_drafted", "dflash_accepted", "dflash_accepted_length")
    for row in rows:
        if not isinstance(row, dict):
            errors.append("performance row is not an object")
            continue
        if row.get("requests_ok") != row.get("requests_total") or not row.get("requests_total"):
            errors.append(f"concurrency {row.get('concurrency')} has failed or missing requests")
        for metric in required:
            if not isinstance(row.get(metric), (int, float)):
                errors.append(f"concurrency {row.get('concurrency')} lacks numeric {metric}")
        drafted, accepted = row.get("dflash_drafted"), row.get("dflash_accepted")
        if isinstance(drafted, (int, float)) and isinstance(accepted, (int, float)) and not (0 < accepted <= drafted):
            errors.append(f"concurrency {row.get('concurrency')} has invalid DFlash counters")

    quality = summary.get("quality") or {}
    expected_quality = {"task": "gsm8k", "dataset": "openai/gsm8k", "num_fewshot": 0, "sample_count": 1319, "endpoint": "chat-completions", "enable_thinking": False, "request_errors": 0}
    for key, expected in expected_quality.items():
        if quality.get(key) != expected:
            errors.append(f"quality.{key} must equal {expected!r}")
    score = quality.get("flexible_extract_exact_match")
    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
        errors.append("quality score is missing or outside [0,1]")

    for path in (summary_path, recipe_path, readme_path, html_path):
        if not path.is_file():
            errors.append(f"missing public artifact: {path}")
            continue
        text = path.read_text(errors="replace")
        for forbidden in FORBIDDEN_PUBLIC_TEXT:
            if forbidden.lower() in text.lower():
                errors.append(f"forbidden public text {forbidden!r} in {path}")
    digest = release.get("ghcr_digest", "")
    immutable = f"{EXPECTED_IMAGE}@{digest}"
    recipe_text = recipe_path.read_text(errors="replace") if recipe_path.is_file() else ""
    readme_text = readme_path.read_text(errors="replace") if readme_path.is_file() else ""
    html_text = html_path.read_text(errors="replace") if html_path.is_file() else ""
    if immutable not in recipe_text:
        errors.append("recipe does not use the immutable GHCR digest")
    for label, text in (("README", readme_text), ("HTML", html_text)):
        if immutable not in text:
            errors.append(f"{label} omits immutable GHCR reference")
        for value in (EXPECTED["model_revision"], EXPECTED["draft_model_revision"], EXPECTED["vllm_commit"]):
            if value not in text:
                errors.append(f"{label} omits release identity {value}")

    verify_manifest(summary_path.parent, manifest_path, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path("results/v0.25.1-gb10")
    parser.add_argument("--summary", type=Path, default=root / "summary.json")
    parser.add_argument("--recipe", type=Path, default=Path("sparkrun/recipes/laguna-s-2.1-nvfp4-vllm-r0b0tlab.yaml"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--html", type=Path, default=root / "benchmark.html")
    parser.add_argument("--manifest", type=Path, default=root / "MANIFEST.sha256")
    args = parser.parse_args()
    errors = verify(args.summary, args.recipe, args.readme, args.html, args.manifest)
    if errors:
        for error in errors:
            print(f"RELEASE_VERIFY_FAIL: {error}")
        return 1
    print("RELEASE_VERIFY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
