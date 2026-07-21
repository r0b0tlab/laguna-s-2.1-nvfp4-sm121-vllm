#!/usr/bin/env python3
"""Strict dependency check with narrowly attested aarch64 exceptions."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Mapping, NamedTuple

KNOWN_SBSA_WARNING = "nvidia-cusparselt-cu13 0.8.0 is not supported on this platform"
KNOWN_SBSA_VERSION = "0.8.0"
KNOWN_SBSA_TAG = "Tag: py3-none-manylinux2014_sbsa"
EXPECTED = {
    "vllm": "0.25.1",
    "flashinfer-python": "0.6.15.dev20260712",
    "flashinfer-cubin": "0.6.15.dev20260712",
    "flashinfer-jit-cache": "0.6.15.dev20260712",
}
OVERLAY_RE = re.compile(
    r"^vllm 0\.25\.1 has requirement (flashinfer-(?:python|cubin))==0\.6\.13, but you have \1 0\.6\.15\.dev20260712\.$"
)


class Evaluation(NamedTuple):
    ok: bool
    reason: str


def evaluate(*, pip_returncode: int, pip_output: str, machine: str, installed: Mapping[str, str], sbsa_wheel_text: str = "", sbsa_elf_header: str = "", prebuild: bool = False) -> Evaluation:
    required = {key: value for key, value in EXPECTED.items() if not (prebuild and key == "vllm")}
    for package, expected in required.items():
        if installed.get(package) != expected:
            return Evaluation(False, f"{package} version mismatch: {installed.get(package)!r} != {expected!r}")
    lines = [line.strip() for line in pip_output.splitlines() if line.strip()]
    if pip_returncode == 0 and not lines:
        return Evaluation(True, "pip check passed without exceptions")
    if machine != "aarch64":
        return Evaluation(False, f"exceptions require aarch64, got {machine!r}")
    unknown = [line for line in lines if line != KNOWN_SBSA_WARNING and not (not prebuild and OVERLAY_RE.fullmatch(line))]
    if unknown:
        return Evaluation(False, f"unexpected pip check output: {unknown!r}")
    if not prebuild:
        overlay_packages = {OVERLAY_RE.fullmatch(line).group(1) for line in lines if OVERLAY_RE.fullmatch(line)}
        if overlay_packages != {"flashinfer-python", "flashinfer-cubin"}:
            return Evaluation(False, f"expected exact FlashInfer overlay warnings, got {sorted(overlay_packages)!r}")
    if KNOWN_SBSA_WARNING in lines:
        if installed.get("nvidia-cusparselt-cu13") != KNOWN_SBSA_VERSION:
            return Evaluation(False, "SBSA exception version mismatch")
        if KNOWN_SBSA_TAG not in sbsa_wheel_text:
            return Evaluation(False, "SBSA wheel tag is absent")
        if "Machine:" not in sbsa_elf_header or "AArch64" not in sbsa_elf_header:
            return Evaluation(False, "cuSPARSELt library is not an AArch64 ELF")
    if prebuild:
        return Evaluation(True, "prebuild dependencies passed with optional verified SBSA warning")
    return Evaluation(True, "accepted exact vLLM 0.25.1/FlashInfer nightly overlay and optional verified SBSA warning")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prebuild", action="store_true")
    args = parser.parse_args()
    result = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True, timeout=120, check=False)
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    installed: dict[str, str] = {}
    for package in (*EXPECTED, "nvidia-cusparselt-cu13"):
        try:
            installed[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    wheel_text = ""
    elf_header = ""
    if KNOWN_SBSA_WARNING in output:
        try:
            dist = metadata.distribution("nvidia-cusparselt-cu13")
            wheel_text = (Path(dist._path) / "WHEEL").read_text()  # type: ignore[attr-defined]
            library = Path(dist.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0"))
            elf_header = subprocess.check_output(["readelf", "-h", str(library)], text=True, timeout=30)
        except Exception as exc:
            print(f"DEPENDENCY_CHECK_FAIL: SBSA verification failed: {exc!r}")
            return 1
    evaluation = evaluate(pip_returncode=result.returncode, pip_output=output, machine=platform.machine(), installed=installed, sbsa_wheel_text=wheel_text, sbsa_elf_header=elf_header, prebuild=args.prebuild)
    if output:
        print("PIP_CHECK_OUTPUT=" + output.replace("\n", " | "))
    print(("DEPENDENCY_CHECK_PASS: " if evaluation.ok else "DEPENDENCY_CHECK_FAIL: ") + evaluation.reason)
    return 0 if evaluation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
