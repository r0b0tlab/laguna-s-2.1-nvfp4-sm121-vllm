#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTests(unittest.TestCase):
    def test_checkpoint_0761412_quick_validation_bundle(self) -> None:
        quick = ROOT / "results/v0.25.1-gb10/checkpoint-0761412"
        sanity = json.loads((quick / "sanity-summary.json").read_text())
        bench = json.loads((quick / "llama-benchy.json").read_text())
        provenance = json.loads((quick / "provenance.json").read_text())
        self.assertEqual(sanity["status"], "PASS")
        self.assertEqual(sanity["target"]["revision"], "07614121b31898586430f189d27a25a0be310843")
        self.assertTrue(sanity["native"]["flashinfer_cutlass_nvfp4"])
        self.assertTrue(all(item["ok"] for item in sanity["smoke"].values()))
        self.assertEqual(len(bench["benchmarks"]), 4)
        self.assertEqual([row["context_size"] for row in bench["benchmarks"]], [0, 4096, 8192, 16384])
        self.assertEqual(provenance["scope"], "revision-scoped quick validation; not a replacement for the historical 8,620-case battery")
        for line in (quick / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256((quick / name).read_bytes()).hexdigest(), digest)

    def test_full_battery_bundle(self) -> None:
        results = ROOT / "results/v0.25.1-gb10"
        full = results / "full-battery"
        summary = json.loads((results / "summary.json").read_text())
        battery = summary["full_battery"]
        scorecard_path = full / "scorecard.json"
        scorecard = json.loads(scorecard_path.read_text())

        self.assertEqual((battery["status"], battery["suite"], battery["logical_cases"]), ("PASS", "r0b0bench-core-v1-rc2", 8620))
        self.assertEqual(battery["official_bfcl_v4_multi_turn_base"], {"accuracy": 0.685, "correct": 137, "n": 200})
        self.assertEqual(battery["generated_answer"]["correct"], 6838)
        self.assertEqual(battery["humaneval"]["passed"], 154)
        self.assertEqual(scorecard["status"], "PASS")
        self.assertEqual(scorecard["logical_cases"], 8620)
        self.assertEqual(scorecard["official_bfcl_v4_multi_turn_base"]["total_count"], 200)
        self.assertEqual(scorecard["humaneval"]["n"], 164)
        self.assertEqual(hashlib.sha256(scorecard_path.read_bytes()).hexdigest(), battery["scorecard"]["sha256"])

        readme = (ROOT / "README.md").read_text()
        verdict = (results / "VERDICT.md").read_text()
        report = (full / "REPORT.md").read_text()
        methodology = (full / "METHODOLOGY.md").read_text()
        manifest = (results / "MANIFEST.sha256").read_text()
        for text in (readme, verdict, report):
            self.assertIn("8,620", text)
        self.assertIn("BFCL v4", methodology)
        for name in ("full-battery/scorecard.json", "full-battery/REPORT.md", "full-battery/METHODOLOGY.md", "full-battery/SHA256SUMS"):
            self.assertIn(name, manifest)

    def test_exact_model_and_runtime_manifests(self) -> None:
        dependency = json.loads((ROOT / "docker/dependency-manifest.json").read_text())
        runtime = json.loads((ROOT / "docker/runtime-manifest.production.json").read_text())
        self.assertEqual(dependency["vllm"]["version"], "0.25.1")
        self.assertEqual(dependency["vllm"]["commit"], "752a3a504485790a2e8491cacbb35c137339ad34")
        self.assertEqual(runtime["model_id"], "poolside/Laguna-S-2.1-NVFP4")
        self.assertEqual(runtime["model_revision"], "07614121b31898586430f189d27a25a0be310843")
        self.assertEqual(runtime["draft_model_id"], "poolside/Laguna-S-2.1-DFlash-NVFP4")
        self.assertEqual(runtime["draft_model_revision"], "723794750422b3efbf3a7b3af76dffb4ba035943")
        self.assertEqual(runtime["default_kv_cache_dtype"], "fp8")
        self.assertFalse(runtime["nvfp4_kv_enabled"])
        self.assertEqual(runtime["configured_context_tokens"], 262144)

    def test_dockerfile_fetches_exact_tag_and_builds_from_source(self) -> None:
        text = (ROOT / "docker/Dockerfile.production").read_text()
        self.assertIn("ARG VLLM_TAG=v0.25.1", text)
        self.assertIn("ARG VLLM_COMMIT=752a3a504485790a2e8491cacbb35c137339ad34", text)
        self.assertEqual(text.count("ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90"), 2)
        self.assertEqual(text.count("6ea7d2737648936820e85677177957a0f6521b840d98eb0bbae0a4f003fa7249"), 2)
        self.assertIn('refs/tags/${VLLM_TAG}:refs/tags/${VLLM_TAG}', text)
        self.assertIn("describe --tags --exact-match HEAD", text)
        self.assertIn("pip install --no-build-isolation --no-deps .", text)
        self.assertIn("flashinfer-jit-cache==${FLASHINFER_VERSION}", text)
        self.assertIn('"cuda-tile==1.5.0" "nccl4py==0.3.1"', text)
        self.assertIn("MAX_JOBS=4", text)
        self.assertIn("CUTE_DSL_ARCH=sm_121a", text)
        install = text.index("pip install --no-build-isolation --no-deps .")
        dependency_check = text.index("/usr/local/bin/check_dependencies.py --prebuild")
        static_audit = text.index("BUILDER_STATIC_ELF_AUDIT_PASS")
        self.assertLess(dependency_check, install)
        self.assertLess(install, static_audit)
        self.assertNotIn("git apply", text)
        dockerignore = (ROOT / ".dockerignore").read_text()
        self.assertTrue(dockerignore.startswith("*\n"))
        for required in ("!docker/Dockerfile.production", "!docker/runtime-manifest.production.json", "!scripts/entrypoint.sh", "!scripts/audit_runtime.py", "!scripts/check_dependencies.py"):
            self.assertIn(required, dockerignore)

    def test_latest_release_resolver_is_required(self) -> None:
        resolver = (ROOT / "scripts/resolve_vllm_release.py").read_text()
        build = (ROOT / "scripts/build_release.sh").read_text()
        for endpoint in ("releases/latest", "pypi.org/pypi/vllm/json", "commits/main"):
            self.assertIn(endpoint, resolver)
        self.assertIn("latest release is draft or prerelease", resolver)
        self.assertIn("GitHub/PyPI version mismatch", resolver)
        self.assertIn("resolve_vllm_release.py", build)
        self.assertIn("dependency manifest", build)

    def test_no_unrelated_model_residue(self) -> None:
        forbidden = ("Qwen", "qwen", "nvidia/Qwen", "modelopt", "W4A4", "MTP_TOKENS")
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(errors="ignore")
            if any(word in text for word in forbidden):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_static_verification_commands_pass(self) -> None:
        subprocess.run(["bash", "-n", str(ROOT / "scripts/entrypoint.sh")], check=True)
        subprocess.run(["bash", "-n", str(ROOT / "scripts/admit_node.sh")], check=True)
        subprocess.run(["bash", "-n", str(ROOT / "scripts/build_release.sh")], check=True)
        subprocess.run([sys.executable, "-m", "py_compile", *map(str, (ROOT / "scripts").glob("*.py"))], check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
