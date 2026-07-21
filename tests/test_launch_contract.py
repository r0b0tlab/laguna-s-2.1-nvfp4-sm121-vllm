#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts/entrypoint.sh"
DOCKERFILE = ROOT / "docker/Dockerfile.production"
RECIPE = ROOT / "sparkrun/recipes/laguna-s-2.1-nvfp4-vllm-r0b0tlab.yaml"


def staged_entrypoint(tmp: Path, audit_exit: int = 0) -> tuple[Path, Path]:
    marker = tmp / "audit-ran"
    fake = tmp / "audit_runtime.py"
    fake.write_text("#!/usr/bin/env bash\n" + f"printf ran > {marker!s}\n" + f"exit {audit_exit}\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    staged = tmp / "entrypoint.sh"
    staged.write_text(ENTRYPOINT.read_text().replace("AUDIT_BIN=/usr/local/bin/audit_runtime.py", f"AUDIT_BIN={fake}"))
    staged.chmod(staged.stat().st_mode | stat.S_IEXEC)
    return staged, marker


class LaunchContractTests(unittest.TestCase):
    def test_shell_syntax_and_manifest_permissions(self) -> None:
        subprocess.run(["bash", "-n", str(ENTRYPOINT)], check=True)
        self.assertIn("chmod 0644 /opt/r0b0tlab/runtime-manifest.json", DOCKERFILE.read_text())

    def test_entrypoint_preserves_argv_and_child_exit(self) -> None:
        expected = ["alpha beta", "gamma", "delta epsilon"]
        with tempfile.TemporaryDirectory() as raw:
            staged, marker = staged_entrypoint(Path(raw))
            result = subprocess.run([staged, sys.executable, "-c", "import json,sys; print(json.dumps(sys.argv[1:]))", *expected], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), expected)
            self.assertTrue(marker.exists())
            self.assertNotIn("eval ", ENTRYPOINT.read_text())
        with tempfile.TemporaryDirectory() as raw:
            staged, _ = staged_entrypoint(Path(raw))
            result = subprocess.run([staged, "/bin/bash", "-c", "exit 37"], check=False)
            self.assertEqual(result.returncode, 37)

    def test_failed_audit_blocks_child(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            staged, marker = staged_entrypoint(tmp, audit_exit=23)
            child = tmp / "child-ran"
            result = subprocess.run([staged, "/bin/bash", "-c", f"printf ran > {child}"], check=False)
            self.assertEqual(result.returncode, 23)
            self.assertTrue(marker.exists())
            self.assertFalse(child.exists())

    def test_rejects_nvfp4_kv_and_b12x_before_audit(self) -> None:
        for argv, env_update, expected_code in (
            (["/bin/true", "--kv-cache-dtype", "nvfp4"], {}, 64),
            (["/bin/true", "--kv-cache-dtype=nvfp4"], {}, 64),
            (["/bin/true"], {"KV_CACHE_DTYPE": "nvfp4"}, 64),
            (["bash", "-c", "vllm serve /m --linear-backend flashinfer_b12x"], {}, 65),
        ):
            with self.subTest(argv=argv), tempfile.TemporaryDirectory() as raw:
                staged, marker = staged_entrypoint(Path(raw))
                env = os.environ.copy()
                env.update(env_update)
                result = subprocess.run([staged, *argv], env=env, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, expected_code)
                self.assertFalse(marker.exists())

    def test_zero_arg_defaults_are_laguna_native(self) -> None:
        text = ENTRYPOINT.read_text()
        for expected in (
            'MODEL_PATH="${MODEL_PATH:-/models/Laguna-S-2.1-NVFP4}"',
            'SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-poolside/Laguna-S-2.1-NVFP4}"',
            'MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"',
            'MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"',
            'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"',
            'DFLASH_TOKENS="${DFLASH_TOKENS:-0}"',
            'DRAFT_MODEL_REVISION="${DRAFT_MODEL_REVISION:-723794750422b3efbf3a7b3af76dffb4ba035943}"',
            "--reasoning-parser poolside_v1",
            "--tool-call-parser poolside_v1",
            '"method":"dflash"',
            '"revision"',
        ):
            self.assertIn(expected, text)
        self.assertNotIn("--enable-prefix-caching", text)
        self.assertIn("--override-generation-config", text)
        self.assertIn('"top_k":20', text)

    def test_recipe_is_exact_candidate_contract(self) -> None:
        text = RECIPE.read_text()
        self.assertIn("model: poolside/Laguna-S-2.1-NVFP4", text)
        self.assertIn("model_revision: 216d1f13878dd4e715bc7412848d0f330e95bba6", text)
        self.assertIn("draft_revision: 723794750422b3efbf3a7b3af76dffb4ba035943", text)
        self.assertIn("--kv-cache-dtype fp8", text)
        self.assertIn("max_model_len: 262144", text)
        self.assertIn("--reasoning-parser poolside_v1", text)
        self.assertIn("--tool-call-parser poolside_v1", text)
        self.assertNotIn("--speculative-config", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
