#!/usr/bin/env python3
"""Extract and gate native NVFP4 kernels from an Nsight Systems report."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

POSITIVE = ("nvfp4", "fp4", "cutlass", "sm100", "sm120", "sm121")
FORBIDDEN = ("marlin", "dequantize_then", "fallback")


def parse_cuda_kernel_csv(text: str) -> list[dict[str,object]]:
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("Processing") and not line.startswith("Generating")]
    header_index = next((i for i,line in enumerate(lines) if "Name" in line and ("Time" in line or "Total" in line)), None)
    if header_index is None:
        raise ValueError("Nsight CUDA kernel CSV header is absent")
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    rows = []
    for row in reader:
        name = str(row.get("Name") or row.get("Kernel Name") or "")
        if not name:
            continue
        rows.append({"name":name,"time_percent":row.get("Time (%)"),"total_time":row.get("Total Time (ns)") or row.get("Total Time")})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report",type=Path)
    parser.add_argument("--output",type=Path,default=Path("nsight-kernels.json"))
    args = parser.parse_args()
    completed = subprocess.run(["nsys","stats","--report","cuda_gpu_kern_sum","--format","csv","--force-export=true",str(args.report)],capture_output=True,text=True,timeout=600,check=False)
    if completed.returncode != 0:
        raise SystemExit(f"nsys stats failed: {completed.stderr.strip()}")
    rows = parse_cuda_kernel_csv(completed.stdout)
    lowered = [row["name"].lower() for row in rows]
    positive = [row for row,name in zip(rows,lowered) if any(token in name for token in POSITIVE)]
    forbidden = [row for row,name in zip(rows,lowered) if any(token in name for token in FORBIDDEN)]
    result = {"schema_version":1,"status":"PASS" if positive and not forbidden else "FAIL","kernel_count":len(rows),"native_nvfp4_kernels":positive,"forbidden_kernels":forbidden,"all_kernels":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
