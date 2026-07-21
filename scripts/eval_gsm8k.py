#!/usr/bin/env python3
"""Pinned GSM8K 0-shot flexible-extract evaluation via Chat Completions."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import urllib.request
from typing import Any

DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
PARQUET_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
PARQUET_URL = f"https://huggingface.co/datasets/openai/gsm8k/resolve/{DATASET_REVISION}/main/test-00000-of-00001.parquet"
NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")
FINAL_RE = re.compile(r"####\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.replace(",", "").strip().rstrip(".")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", "+0", ""} else normalized


def extract_answer(text: str) -> str | None:
    finals = FINAL_RE.findall(text)
    if finals:
        return normalize_number(finals[-1])
    numbers = NUMBER_RE.findall(text)
    return normalize_number(numbers[-1]) if numbers else None


def download_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or file_sha256(path) != PARQUET_SHA256:
        request = urllib.request.Request(PARQUET_URL, headers={"User-Agent":"r0b0tlab-gsm8k-eval"})
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    actual = file_sha256(path)
    if actual != PARQUET_SHA256:
        raise RuntimeError(f"dataset hash mismatch: {actual} != {PARQUET_SHA256}")


def post_chat(base_url: str, model: str, question: str, timeout: int, max_tokens: int) -> dict[str, Any]:
    prompt = question.strip() + '\nSolve the problem. End your response with "#### <number>".'
    payload = {"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0,"max_tokens":max_tokens,"chat_template_kwargs":{"enable_thinking":False}}
    request = urllib.request.Request(base_url.rstrip("/")+"/v1/chat/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url",default="http://127.0.0.1:8000")
    parser.add_argument("--model",default="poolside/Laguna-S-2.1-NVFP4")
    parser.add_argument("--cache",type=Path,default=Path(".cache/gsm8k-test.parquet"))
    parser.add_argument("--output-dir",type=Path,default=Path("results/raw/gsm8k"))
    parser.add_argument("--limit",type=int)
    parser.add_argument("--concurrency",type=int,default=8)
    parser.add_argument("--timeout",type=int,default=600)
    parser.add_argument("--max-tokens",type=int,default=512)
    args = parser.parse_args()
    download_dataset(args.cache)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SystemExit("pyarrow is required: install it in the evaluation client environment") from exc
    table = parquet.read_table(args.cache, columns=["question","answer"])
    rows = table.to_pylist()
    if args.limit is not None:
        rows = rows[:args.limit]

    def worker(item: tuple[int,dict[str,Any]]) -> dict[str,Any]:
        index, row = item
        expected = extract_answer(str(row["answer"]))
        try:
            response = post_chat(args.base_url,args.model,str(row["question"]),args.timeout,args.max_tokens)
            message = response["choices"][0]["message"]
            content = message.get("content") or ""
            predicted = extract_answer(content)
            error = None
            usage = response.get("usage")
        except Exception as exc:
            content,predicted,usage,error = "",None,None,f"{type(exc).__name__}: {exc}"
        return {"index":index,"expected":expected,"predicted":predicted,"correct":predicted is not None and predicted==expected,"error":error,"response":content[:2000],"usage":usage}

    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        samples = list(executor.map(worker, enumerate(rows)))
    samples.sort(key=lambda row: row["index"])
    errors = sum(1 for row in samples if row["error"])
    correct = sum(1 for row in samples if row["correct"])
    total = len(samples)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    samples_path = args.output_dir/"samples.jsonl"
    samples_path.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in samples))
    summary = {"schema_version":1,"status":"PASS" if errors==0 and total==(args.limit or 1319) else "FAIL","task":"gsm8k","dataset":"openai/gsm8k","dataset_revision":DATASET_REVISION,"dataset_parquet_sha256":PARQUET_SHA256,"num_fewshot":0,"sample_count":total,"endpoint":"chat-completions","enable_thinking":False,"request_errors":errors,"correct":correct,"flexible_extract_exact_match":correct/total if total else 0.0,"samples_sha256":file_sha256(samples_path)}
    (args.output_dir/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
