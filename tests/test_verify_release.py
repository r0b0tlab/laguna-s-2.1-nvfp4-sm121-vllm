#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_release", ROOT / "scripts" / "verify_release.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DIGEST = "sha256:" + "a" * 64
IMAGE_ID = "sha256:" + "b" * 64
IMAGE = "ghcr.io/r0b0tlab/vllm-laguna-s-2.1-nvfp4-sm121"
MODEL_REV = "216d1f13878dd4e715bc7412848d0f330e95bba6"
DRAFT_REV = "723794750422b3efbf3a7b3af76dffb4ba035943"
VLLM_COMMIT = "752a3a504485790a2e8491cacbb35c137339ad34"
IMMUTABLE = f"{IMAGE}@{DIGEST}"


def valid_summary() -> dict:
    rows = []
    for concurrency in (1, 2, 4, 8, 16, 32):
        rows.append({"concurrency":concurrency,"requests_ok":concurrency*3,"requests_total":concurrency*3,"output_tokens_per_second":10.0*concurrency,"prompt_tokens_per_second":50.0*concurrency,"ttft_p50_seconds":0.1,"ttft_p90_seconds":0.2,"ttft_p99_seconds":0.3,"itl_p50_seconds":0.05,"itl_p90_seconds":0.06,"itl_p99_seconds":0.07,"power_mean_watts":40.0,"dflash_drafted":100.0,"dflash_accepted":60.0,"dflash_accepted_length":3.0})
    return {
        "schema_version":1,"status":"PASS",
        "release":{"model_id":"poolside/Laguna-S-2.1-NVFP4","model_revision":MODEL_REV,"draft_model_id":"poolside/Laguna-S-2.1-DFlash-NVFP4","draft_model_revision":DRAFT_REV,"vllm":"0.25.1","vllm_commit":VLLM_COMMIT,"torch":"2.11.0+cu130","cuda":"13.0","flashinfer":"0.6.15.dev20260712","local_image_id":IMAGE_ID,"ghcr_image":IMAGE,"ghcr_digest":DIGEST},
        "profile":{"kv_cache_dtype":"fp8","speculation":{"method":"dflash","num_speculative_tokens":15},"configured_context_tokens":262144,"max_num_seqs":32},
        "dflash_depth_sweep":{"tested_k":[3,5,7,11,15],"selected_k":15,"selection_rule_pass":True},
        "native_gate":{"runtime_audit":"PASS","marlin_markers":0,"emulation_markers":0,"fallback_markers":0,"flashinfer_cutlass_nvfp4":True,"flashinfer_attention":True,"nsight_kernel_table":[{"kernel":"nvfp4","time_percent":90.0}]},
        "canaries":{"models":True,"semantic":True,"reasoning":True,"tool_call":True,"multi_tool":True,"preserved_reasoning":True,"long_generation":True,"context_262k":True},
        "performance":{"errors":[],"rows":rows},
        "quality":{"task":"gsm8k","dataset":"openai/gsm8k","num_fewshot":0,"sample_count":1319,"endpoint":"chat-completions","enable_thinking":False,"request_errors":0,"flexible_extract_exact_match":0.85},
    }


class ReleaseVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.summary = self.root / "summary.json"
        self.recipe = self.root / "recipe.yaml"
        self.readme = self.root / "README.md"
        self.html = self.root / "benchmark.html"
        self.manifest = self.root / "MANIFEST.sha256"
        self.data = valid_summary()
        self.write_fixture()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_fixture(self) -> None:
        self.summary.write_text(json.dumps(self.data, indent=2) + "\n")
        public = f"{IMMUTABLE}\n{MODEL_REV}\n{DRAFT_REV}\n{VLLM_COMMIT}\n"
        self.recipe.write_text(f"container: {IMMUTABLE}\n")
        self.readme.write_text(public)
        self.html.write_text("<html><body>" + public + "</body></html>")
        paths = (self.summary, self.recipe, self.readme, self.html)
        self.manifest.write_text("\n".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in paths) + "\n")

    def errors(self) -> list[str]:
        return MODULE.verify(self.summary, self.recipe, self.readme, self.html, self.manifest)

    def test_valid_bundle_passes(self) -> None:
        self.assertEqual(self.errors(), [])

    def test_active_marlin_is_rejected(self) -> None:
        self.data["native_gate"]["marlin_markers"] = 1
        self.write_fixture()
        self.assertTrue(any("marlin_markers" in item for item in self.errors()))

    def test_failed_request_is_rejected(self) -> None:
        self.data["performance"]["rows"][0]["requests_ok"] = 2
        self.write_fixture()
        self.assertTrue(any("failed or missing requests" in item for item in self.errors()))

    def test_incomplete_dflash_sweep_is_rejected(self) -> None:
        self.data["dflash_depth_sweep"]["tested_k"] = [7, 15]
        self.write_fixture()
        self.assertTrue(any("tested_k" in item for item in self.errors()))

    def test_mutable_recipe_is_rejected(self) -> None:
        self.recipe.write_text(f"container: {IMAGE}:latest\n")
        paths = (self.summary, self.recipe, self.readme, self.html)
        self.manifest.write_text("\n".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in paths) + "\n")
        self.assertTrue(any("immutable GHCR" in item for item in self.errors()))

    def test_incomplete_quality_is_rejected(self) -> None:
        self.data["quality"]["sample_count"] = 1318
        self.write_fixture()
        self.assertTrue(any("quality.sample_count" in item for item in self.errors()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
