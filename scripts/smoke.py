#!/usr/bin/env python3
"""Fail-closed OpenAI API smoke probes for Laguna S 2.1."""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


def request_json(base_url: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 300) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="poolside/Laguna-S-2.1-NVFP4")
    args = parser.parse_args()
    failures: list[str] = []
    models = request_json(args.base_url, "/v1/models")
    ids = [item.get("id") for item in models.get("data", [])]
    if args.model not in ids:
        failures.append(f"model catalog mismatch: {ids!r}")
    probes = [
        ("math", {"messages":[{"role":"user","content":"What is 17 * 23? Reply with only the number."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":False}}, lambda d: "391" in (d["choices"][0]["message"].get("content") or "")),
        ("reasoning", {"messages":[{"role":"user","content":"Which is larger, 91 or 19? Explain briefly."}],"temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":True}}, lambda d: bool(d["choices"][0]["message"].get("reasoning") or d["choices"][0]["message"].get("reasoning_content"))),
        ("tool", {"messages":[{"role":"user","content":"Use the add tool to add 7 and 8."}],"tools":[{"type":"function","function":{"name":"add","description":"Add two integers","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}}],"tool_choice":{"type":"function","function":{"name":"add"}},"temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":False}}, lambda d: bool(d["choices"][0]["message"].get("tool_calls"))),
    ]
    evidence = {"model_catalog": models, "probes": {}}
    for name, body, validator in probes:
        body["model"] = args.model
        result = request_json(args.base_url, "/v1/chat/completions", body)
        ok = False
        try:
            ok = bool(validator(result))
        except Exception:
            ok = False
        evidence["probes"][name] = {"ok": ok, "result": result}
        if not ok:
            failures.append(f"{name} probe failed")
    evidence["status"] = "PASS" if not failures else "FAIL"
    evidence["failures"] = failures
    print(json.dumps(evidence, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
