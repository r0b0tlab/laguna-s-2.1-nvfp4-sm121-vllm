#!/usr/bin/env python3
"""Resolve and freeze the newest reproducible stable vLLM release."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import urllib.request
from typing import Any

LATEST_URL = "https://api.github.com/repos/vllm-project/vllm/releases/latest"
TAGS_URL = "https://api.github.com/repos/vllm-project/vllm/tags?per_page=100"
PYPI_URL = "https://pypi.org/pypi/vllm/json"
MAIN_URL = "https://api.github.com/repos/vllm-project/vllm/commits/main"


def fetch_json(url: str) -> tuple[Any, str]:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "r0b0tlab-release-resolver"})
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def resolve() -> dict[str, Any]:
    latest, latest_hash = fetch_json(LATEST_URL)
    tags, tags_hash = fetch_json(TAGS_URL)
    pypi, pypi_hash = fetch_json(PYPI_URL)
    main, main_hash = fetch_json(MAIN_URL)

    if latest.get("draft") is not False or latest.get("prerelease") is not False:
        raise RuntimeError("GitHub latest release is draft or prerelease")
    tag = latest.get("tag_name")
    if not isinstance(tag, str) or not re.fullmatch(r"v\d+\.\d+\.\d+(?:\.post\d+)?", tag):
        raise RuntimeError(f"unexpected stable tag: {tag!r}")
    version = tag.removeprefix("v")
    if pypi.get("info", {}).get("version") != version:
        raise RuntimeError(f"GitHub/PyPI version mismatch: {tag!r} vs {pypi.get('info', {}).get('version')!r}")
    if pypi.get("info", {}).get("yanked") is True:
        raise RuntimeError(f"PyPI release {version} is yanked")
    matches = [item for item in tags if item.get("name") == tag]
    if len(matches) != 1:
        raise RuntimeError(f"expected one GitHub tag record for {tag}, got {len(matches)}")
    commit = matches[0].get("commit", {}).get("sha")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"invalid tag commit: {commit!r}")

    files = pypi.get("urls") or []
    hashes = {item.get("filename"): item.get("digests", {}).get("sha256") for item in files}
    return {
        "schema_version": 1,
        "resolved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "resolution_policy": "newest GitHub non-draft non-prerelease release whose tag agrees with non-yanked PyPI info.version",
        "sources": {
            "github_latest": {"url": LATEST_URL, "sha256": latest_hash},
            "github_tags": {"url": TAGS_URL, "sha256": tags_hash},
            "pypi": {"url": PYPI_URL, "sha256": pypi_hash},
            "github_main": {"url": MAIN_URL, "sha256": main_hash},
        },
        "vllm": {
            "version": version,
            "tag": tag,
            "commit": commit,
            "source_build": True,
            "pypi_yanked": False,
            "published_at": latest.get("published_at"),
            "pypi_files": hashes,
            "main_head_observed": main.get("sha"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path)
    parser.add_argument("--expect-version")
    parser.add_argument("--expect-commit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = resolve()
    release = data["vllm"]
    if args.expect_version and release["version"] != args.expect_version:
        raise SystemExit(f"resolved version {release['version']} != expected {args.expect_version}")
    if args.expect_commit and release["commit"] != args.expect_commit:
        raise SystemExit(f"resolved commit {release['commit']} != expected {args.expect_commit}")
    rendered = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        if args.write.is_file():
            current = json.loads(args.write.read_text())
            if isinstance(current, dict) and any(key in current for key in ("torch", "flashinfer", "models")):
                current["schema_version"] = data["schema_version"]
                current["resolved_at"] = data["resolved_at"]
                current["resolution_policy"] = data["resolution_policy"]
                current["sources"] = data["sources"]
                current["vllm"] = data["vllm"]
                data = current
                rendered = json.dumps(data, indent=2, sort_keys=True) + "\n"
        args.write.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
