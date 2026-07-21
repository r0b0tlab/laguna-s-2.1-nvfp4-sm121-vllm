#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTests(unittest.TestCase):
    def test_exact_model_and_runtime_manifests(self) -> None:
        dependency = json.loads((ROOT / "docker/dependency-manifest.json").read_text())
        runtime = json.loads((ROOT / "docker/runtime-manifest.production.json").read_text())
        self.assertEqual(dependency["vllm"]["version"], "0.25.1")
        self.assertEqual(dependency["vllm"]["commit"], "752a3a504485790a2e8491cacbb35c137339ad34")
        self.assertEqual(runtime["model_id"], "poolside/Laguna-S-2.1-NVFP4")
        self.assertEqual(runtime["model_revision"], "216d1f13878dd4e715bc7412848d0f330e95bba6")
        self.assertEqual(runtime["draft_model_id"], "poolside/Laguna-S-2.1-DFlash-NVFP4")
        self.assertEqual(runtime["draft_model_revision"], "723794750422b3efbf3a7b3af76dffb4ba035943")
        self.assertEqual(runtime["default_kv_cache_dtype"], "fp8")
        self.assertFalse(runtime["nvfp4_kv_enabled"])
        self.assertEqual(runtime["configured_context_tokens"], 262144)

    def test_dockerfile_fetches_exact_tag_and_builds_from_source(self) -> None:
        text = (ROOT / "docker/Dockerfile.production").read_text()
        self.assertIn("ARG VLLM_TAG=v0.25.1", text)
        self.assertIn("ARG VLLM_COMMIT=752a3a504485790a2e8491cacbb35c137339ad34", text)
        self.assertIn('refs/tags/${VLLM_TAG}:refs/tags/${VLLM_TAG}', text)
        self.assertIn("describe --tags --exact-match HEAD", text)
        self.assertIn("pip install --no-build-isolation --no-deps .", text)
        self.assertIn("flashinfer-jit-cache==${FLASHINFER_VERSION}", text)
        self.assertIn("MAX_JOBS=4", text)
        self.assertIn("CUTE_DSL_ARCH=sm_121a", text)
        install = text.index("pip install --no-build-isolation --no-deps .")
        dependency_check = text.index("/usr/local/bin/check_dependencies.py --prebuild")
        static_audit = text.index("BUILDER_STATIC_ELF_AUDIT_PASS")
        self.assertLess(dependency_check, install)
        self.assertLess(install, static_audit)
        self.assertNotIn("git apply", text)

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
