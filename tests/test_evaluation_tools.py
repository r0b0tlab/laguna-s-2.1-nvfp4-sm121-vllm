#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_gsm8k = load("eval_gsm8k")
context_probe = load("context_probe")
profile_nvfp4 = load("profile_nvfp4")
render_report = load("render_report")


class EvaluationToolTests(unittest.TestCase):
    def test_flexible_number_extraction(self) -> None:
        self.assertEqual(eval_gsm8k.extract_answer("work\n#### 1,234.00"), "1234")
        self.assertEqual(eval_gsm8k.extract_answer("The answer is -7.50."), "-7.5")
        self.assertIsNone(eval_gsm8k.extract_answer("no numeric answer"))
        self.assertEqual(eval_gsm8k.DATASET_REVISION, "740312add88f781978c0658806c59bc2815b9866")
        self.assertEqual(len(eval_gsm8k.PARQUET_SHA256), 64)

    def test_context_prompt_contains_exact_needle(self) -> None:
        prompt = context_probe.make_prompt(100, 0.5, "EXACT-CODE")
        self.assertEqual(prompt.count("EXACT-CODE"), 1)
        self.assertIn("Reply with only the code", prompt)

    def test_context_calibration_stays_below_target(self) -> None:
        def fake_count(base_url, model, content, timeout):
            return len(content) // 10
        with patch.object(context_probe, "token_count", side_effect=fake_count):
            prompt, count, repeats = context_probe.calibrate_prompt("http://unused", "model", 500, 0.5, "CODE", 1)
        self.assertLessEqual(count, 500)
        self.assertGreater(repeats, 0)
        self.assertIn("CODE", prompt)

    def test_nsight_csv_parser(self) -> None:
        text = "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n90.0,1000,1,1000,1000,1000,1000,0,cutlass_nvfp4_sm121_kernel\n"
        rows = profile_nvfp4.parse_cuda_kernel_csv(text)
        self.assertEqual(rows[0]["name"], "cutlass_nvfp4_sm121_kernel")

    def test_report_renderer_is_self_contained(self) -> None:
        summary = {"status":"PASS","release":{"model_id":"poolside/Laguna-S-2.1-NVFP4","model_revision":"abc"},"profile":{"speculation":{"num_speculative_tokens":15}},"performance":{"rows":[{"concurrency":1,"output_tokens_per_second":10}]}}
        rendered = render_report.render(summary)
        self.assertIn("Laguna S 2.1", rendered)
        self.assertIn("poolside/Laguna-S-2.1-NVFP4", rendered)
        self.assertIn("Release status: PASS", rendered)
        self.assertNotIn("<script", rendered.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
