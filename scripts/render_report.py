#!/usr/bin/env python3
"""Render a self-contained public benchmark report from summary.json."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def percent(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def render(summary: dict) -> str:
    release = summary.get("release") or {}
    profile = summary.get("profile") or {}
    rows = (summary.get("performance") or {}).get("rows") or []
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(key,'')))}</td>" for key in ("concurrency","output_tokens_per_second","prompt_tokens_per_second","ttft_p50_seconds","itl_p50_seconds","power_mean_watts","dflash_accepted_length")) + "</tr>"
        for row in rows
    )
    battery = summary.get("full_battery") or {}
    bfcl = battery.get("official_bfcl_v4_multi_turn_base") or {}
    generated = battery.get("generated_answer") or {}
    ifeval = battery.get("ifeval") or {}
    humaneval = battery.get("humaneval") or {}
    battery_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(result)}</td></tr>"
        for name, result in (
            ("BFCL v4 multi_turn_base", f"{bfcl.get('correct','')}/{bfcl.get('n','')} ({percent(bfcl.get('accuracy'))})"),
            ("Generated-answer quality", f"{generated.get('correct','')}/{generated.get('n','')} ({percent(generated.get('accuracy'))})"),
            ("IFEval prompt strict / loose", f"{percent(ifeval.get('prompt_strict'))} / {percent(ifeval.get('prompt_loose'))}"),
            ("IFEval instruction strict / loose", f"{percent(ifeval.get('instruction_strict'))} / {percent(ifeval.get('instruction_loose'))}"),
            ("HumanEval pass@1", f"{humaneval.get('passed','')}/{humaneval.get('n','')} ({percent(humaneval.get('pass_at_1'))})"),
        )
    )
    identity = html.escape(json.dumps(release,indent=2,sort_keys=True))
    raw = html.escape(json.dumps(summary,indent=2,sort_keys=True))
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Laguna S 2.1 NVFP4 — GB10 report</title><style>body{{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#16202a}}h1,h2{{line-height:1.2}}.ok{{color:#086b36;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #ccd5df;padding:.45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}pre{{background:#f5f7fa;padding:1rem;overflow:auto}}code{{font-family:ui-monospace,monospace}}</style></head><body><h1>Laguna S 2.1 NVFP4 on NVIDIA GB10</h1><p class='ok'>Release status: {html.escape(str(summary.get('status','UNKNOWN')))}</p><p>Exact target <code>{html.escape(str(release.get('model_id','')))}@{html.escape(str(release.get('model_revision','')))}</code>. Production profile uses FP8 KV and DFlash K={html.escape(str((profile.get('speculation') or {{}}).get('num_speculative_tokens','')))}.</p><h2>Performance</h2><table><thead><tr><th>Concurrency</th><th>Output tok/s</th><th>Prompt tok/s</th><th>TTFT p50 s</th><th>ITL p50 s</th><th>Power W</th><th>Accepted length</th></tr></thead><tbody>{table_rows}</tbody></table><h2>Full 8,620-case benchmark battery</h2><p class='ok'>Battery status: {html.escape(str(battery.get('status','UNKNOWN')))}</p><table><thead><tr><th>Pillar</th><th>Result</th></tr></thead><tbody>{battery_rows}</tbody></table><p><a href='full-battery/REPORT.md'>Human-readable scorecard</a> · <a href='full-battery/scorecard.json'>Machine-readable scorecard</a> · <a href='full-battery/METHODOLOGY.md'>Methodology</a></p><h2>Release identity</h2><pre>{identity}</pre><h2>Machine-readable evidence</h2><pre>{raw}</pre></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary",type=Path)
    parser.add_argument("--output",type=Path,default=Path("benchmark.html"))
    args = parser.parse_args()
    summary=json.loads(args.summary.read_text())
    output=render(summary)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
