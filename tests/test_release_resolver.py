#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import resolve_vllm_release as resolver  # noqa: E402


def fixtures(*, github_version: str = "0.25.1", pypi_version: str = "0.25.1", prerelease: bool = False):
    tag = "v" + github_version
    latest = {"tag_name": tag, "draft": False, "prerelease": prerelease, "published_at": "2026-07-14T08:51:20Z"}
    tags = [{"name": tag, "commit": {"sha": "752a3a504485790a2e8491cacbb35c137339ad34"}}]
    pypi = {"info": {"version": pypi_version, "yanked": False}, "urls": []}
    main = {"sha": "d" * 40}
    return [(latest, "1" * 64), (tags, "2" * 64), (pypi, "3" * 64), (main, "4" * 64)]


class ResolverTests(unittest.TestCase):
    def test_matching_stable_release_passes(self) -> None:
        with patch.object(resolver, "fetch_json", side_effect=fixtures()):
            data = resolver.resolve()
        self.assertEqual(data["vllm"]["version"], "0.25.1")
        self.assertEqual(data["vllm"]["tag"], "v0.25.1")
        self.assertEqual(data["vllm"]["commit"], "752a3a504485790a2e8491cacbb35c137339ad34")
        self.assertEqual(data["vllm"]["main_head_observed"], "d" * 40)

    def test_github_pypi_mismatch_fails(self) -> None:
        with patch.object(resolver, "fetch_json", side_effect=fixtures(pypi_version="0.25.0")):
            with self.assertRaisesRegex(RuntimeError, "GitHub/PyPI version mismatch"):
                resolver.resolve()

    def test_prerelease_fails(self) -> None:
        with patch.object(resolver, "fetch_json", side_effect=fixtures(prerelease=True)):
            with self.assertRaisesRegex(RuntimeError, "draft or prerelease"):
                resolver.resolve()

    def test_write_preserves_compatibility_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "dependency.json"
            path.write_text(json.dumps({"schema_version":1,"torch":{"version":"2.11.0+cu130"},"flashinfer":{"version":"0.6.15.dev20260712"},"models":{"target":{"id":"exact"}},"vllm":{}}))
            argv = ["resolve_vllm_release.py", "--write", str(path)]
            with patch.object(resolver, "fetch_json", side_effect=fixtures()), patch.object(sys, "argv", argv):
                self.assertEqual(resolver.main(), 0)
            merged = json.loads(path.read_text())
            self.assertEqual(merged["torch"]["version"], "2.11.0+cu130")
            self.assertEqual(merged["models"]["target"]["id"], "exact")
            self.assertEqual(merged["vllm"]["version"], "0.25.1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
