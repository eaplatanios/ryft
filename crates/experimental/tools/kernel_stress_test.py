"""Deterministic process and replay-ledger tests; no fabricated native qualification."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import kernel_stress


class KernelStressTest(unittest.TestCase):
    """Validate bounded orchestration independently of GPU availability."""

    def test_run_case(self):
        """Observe successful child completion and preserve its output."""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            self.assertEqual(
                kernel_stress.run_case([sys.executable, "-c", "print('completed')"], os.environ.copy(), 5, log),
                (0, False),
            )
            self.assertEqual(log.read_text(), "completed\n")

    def test_run_case_timeout(self):
        """A blocked child is killed and reaped within the declared timeout."""
        with tempfile.TemporaryDirectory() as directory:
            result = kernel_stress.run_case(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                os.environ.copy(), 0.05, Path(directory) / "run.log",
            )
            self.assertEqual(result, (-9, True))

    def test_stop(self):
        """A process exiting during the timeout race is still reaped without hiding other errors."""
        process = mock.Mock(pid=123)
        with mock.patch.object(kernel_stress.os, "killpg", side_effect=ProcessLookupError):
            kernel_stress.stop(process)
        process.wait.assert_called_once_with()

    def test_campaign(self):
        """Replay records precede subprocess launch and completed iterations retain unique directories."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "fixture"
            binary.write_text(
                '#!/bin/sh\necho "compiler stress passed: seed=$RYFT_KERNEL_STRESS_SEED '
                'iteration=$RYFT_KERNEL_STRESS_ITERATION backend=$RYFT_KERNEL_STRESS_BACKEND"\n'
            )
            binary.chmod(0o700)
            self.assertEqual(kernel_stress.campaign(binary, root / "output", "both", 7, 2, 3, 5, 10), 0)
            records = [json.loads(line) for line in (root / "output/campaign.jsonl").read_text().splitlines()]
            self.assertEqual([record["status"] for record in records], ["started", "passed", "started", "passed"])
            replay = json.loads((root / "output/seed-7-iteration-3/replay.json").read_text())
            self.assertEqual((replay["seed"], replay["iteration"], replay["backend"]), (7, 3, "both"))
            with self.assertRaises(FileExistsError):
                kernel_stress.campaign(binary, root / "output", "both", 7, 1, 3, 5, 10)

    def test_campaign_failure(self):
        """A failed iteration stops the campaign with its evidence intact."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "fixture"
            binary.write_text("#!/bin/sh\nexit 3\n")
            binary.chmod(0o700)
            self.assertEqual(kernel_stress.campaign(binary, root / "output", "mosaic", 1, 2, 0, 5, 10), 1)
            records = [json.loads(line) for line in (root / "output/campaign.jsonl").read_text().splitlines()]
            self.assertEqual([record["status"] for record in records], ["started", "failed"])
            self.assertEqual(records[-1]["returncode"], 3)

    def test_campaign_missing_completion(self):
        """Zero executed tests cannot masquerade as successful native compiler qualification."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "fixture"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o700)
            self.assertEqual(kernel_stress.campaign(binary, root / "output", "both", 7, 1, 0, 5, 10), 1)

    def test_campaign_limits(self):
        """Invalid controls fail before creating or launching a campaign."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with self.assertRaisesRegex(ValueError, "seed, iteration, or count is out of bounds"):
                kernel_stress.campaign(Path(sys.executable), output, "both", -1, 1, 0, 5, 10)
            with self.assertRaisesRegex(ValueError, "time limits must be finite"):
                kernel_stress.campaign(Path(sys.executable), output, "both", 7, 1, 0, float("inf"), 10)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
