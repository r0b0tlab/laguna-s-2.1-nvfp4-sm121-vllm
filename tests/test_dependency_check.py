#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_dependencies import KNOWN_SBSA_TAG, KNOWN_SBSA_WARNING, EXPECTED, evaluate  # noqa: E402

FLASH_WARNINGS = "\n".join((
    "vllm 0.25.1 has requirement flashinfer-python==0.6.13, but you have flashinfer-python 0.6.15.dev20260712.",
    "vllm 0.25.1 has requirement flashinfer-cubin==0.6.13, but you have flashinfer-cubin 0.6.15.dev20260712.",
))


class DependencyCheckTests(unittest.TestCase):
    def installed(self) -> dict[str, str]:
        return dict(EXPECTED)

    def test_exact_overlay_warnings_pass(self) -> None:
        result = evaluate(pip_returncode=1, pip_output=FLASH_WARNINGS, machine="aarch64", installed=self.installed())
        self.assertTrue(result.ok, result.reason)

    def test_exact_overlay_plus_verified_sbsa_passes(self) -> None:
        installed = self.installed()
        installed["nvidia-cusparselt-cu13"] = "0.8.0"
        result = evaluate(pip_returncode=1, pip_output=FLASH_WARNINGS + "\n" + KNOWN_SBSA_WARNING, machine="aarch64", installed=installed, sbsa_wheel_text=KNOWN_SBSA_TAG, sbsa_elf_header="Machine: AArch64")
        self.assertTrue(result.ok, result.reason)

    def test_unexpected_warning_fails(self) -> None:
        result = evaluate(pip_returncode=1, pip_output=FLASH_WARNINGS + "\nother conflict", machine="aarch64", installed=self.installed())
        self.assertFalse(result.ok)

    def test_wrong_version_fails(self) -> None:
        installed = self.installed()
        installed["flashinfer-python"] = "0.6.13"
        result = evaluate(pip_returncode=1, pip_output=FLASH_WARNINGS, machine="aarch64", installed=installed)
        self.assertFalse(result.ok)

    def test_wrong_architecture_fails(self) -> None:
        result = evaluate(pip_returncode=1, pip_output=FLASH_WARNINGS, machine="x86_64", installed=self.installed())
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
