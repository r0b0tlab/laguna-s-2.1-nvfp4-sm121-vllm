#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/admit_node.sh"


class AdmissionContractTests(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_wrong_host_fails_before_mutation(self) -> None:
        result = subprocess.run([SCRIPT], env={"PATH":"/usr/bin:/bin","EXPECTED_HOST":"definitely-not-this-host","HOME":"/tmp"}, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 40)
        self.assertIn("expected_host", result.stderr.replace(" ", "_").lower())

    def test_required_gates_are_present(self) -> None:
        text = SCRIPT.read_text()
        for required in ("gn100-2eea", "MIN_FREE_GIB:-180", "compute_cap", "12.1", "docker ps", "MemAvailable", "competing_processes", "ADMISSION_PASS"):
            self.assertIn(required, text)
        self.assertNotIn("docker stop", text)
        self.assertNotIn("docker rm", text)
        self.assertNotIn("system prune", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
