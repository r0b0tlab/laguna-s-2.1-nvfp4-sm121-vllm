#!/usr/bin/env python3
"""Calibrated long-context retrieval probes through vLLM's OpenAI API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request
from typing import Any

FILLER = "Neutral padding record: alpha beta gamma delta epsilon zeta eta theta. "


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    request = urllib.request.Request(base_url.rstrip("/") + path, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} did not return an object")
    return value


def token_count(base_url: str, model: str, content: str, timeout: int) -> int:
    value = post_json(base_url, "/tokenize", {"model":model,"messages":[{"role":"user","content":content}],"add_generation_prompt":True,"chat_template_kwargs":{"enable_thinking":False}}, timeout)
    count = value.get("count")
    if not isinstance(count, int) or count <= 0:
        tokens = value.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"tokenize response lacks positive count: {value!r}")
        count = len(tokens)
    return count


def make_prompt(repeats: int, position: float, needle: str) -> str:
    index = min(repeats, max(0, int(repeats * position)))
    blocks = [FILLER] * repeats
    blocks.insert(index, f"IMPORTANT RECORD: the retrieval code is {needle}. Remember it exactly. ")
    return "".join(blocks) + "\nQuestion: What is the exact retrieval code? Reply with only the code."


def calibrate_prompt(base_url: str, model: str, target_prompt_tokens: int, position: float, needle: str, timeout: int) -> tuple[str, int, int]:
    low, high = 1, max(2, target_prompt_tokens // 2)
    while token_count(base_url, model, make_prompt(high, position, needle), timeout) < target_prompt_tokens:
        low, high = high, high * 2
    best: tuple[str, int, int] | None = None
    while low <= high:
        mid = (low + high) // 2
        prompt = make_prompt(mid, position, needle)
        count = token_count(base_url, model, prompt, timeout)
        if count <= target_prompt_tokens:
            best = (prompt, count, mid)
            low = mid + 1
        else:
            high = mid - 1
    if best is None:
        raise RuntimeError("could not calibrate prompt below target")
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="poolside/Laguna-S-2.1-NVFP4")
    parser.add_argument("--targets", nargs="+", type=int, default=[65536,131072,262144])
    parser.add_argument("--positions", nargs="+", type=float, default=[0.05,0.5,0.95])
    parser.add_argument("--headroom", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=Path("context-probe.json"))
    args = parser.parse_args()
    rows = []
    for target in args.targets:
        if target <= args.headroom:
            raise SystemExit("target must exceed headroom")
        for position in args.positions:
            needle = f"LAGUNA-{target}-{int(position*100):02d}-R0B0TLAB"
            prompt, prompt_count, repeats = calibrate_prompt(args.base_url,args.model,target-args.headroom,position,needle,args.timeout)
            payload = {"model":args.model,"messages":[{"role":"user","content":prompt}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":False}}
            started = time.monotonic()
            response = post_json(args.base_url,"/v1/chat/completions",payload,args.timeout)
            elapsed = time.monotonic() - started
            message = response.get("choices",[{}])[0].get("message",{})
            content = message.get("content") or ""
            usage = response.get("usage") or {}
            actual_prompt = usage.get("prompt_tokens")
            ok = needle in content and isinstance(actual_prompt,int) and target-args.headroom-64 <= actual_prompt <= target
            rows.append({"target_tokens":target,"needle_position":position,"calibrated_prompt_tokens":prompt_count,"reported_prompt_tokens":actual_prompt,"filler_repeats":repeats,"elapsed_seconds":elapsed,"response":content[:200],"needle":needle,"ok":ok})
    result = {"schema_version":1,"status":"PASS" if all(row["ok"] for row in rows) else "FAIL","model":args.model,"headroom":args.headroom,"rows":rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
