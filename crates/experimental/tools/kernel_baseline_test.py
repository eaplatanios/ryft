"""Owner tests for strict baseline identity, stability and reviewed regression budgets."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import kernel_baseline as baseline


def report(samples=None):
    """Create independently specified measurements for a fixed workload."""
    return {"schema": 1, "identity": {"semantic_digest": "a" * 64, "configuration_digest": "b" * 64,
            "execution_digest": "c" * 64, "environment": "isolated GPU", "methodology": "process median ns"},
            "provenance": [{"source_sha256": "d" * 64, "binary_sha256": "e" * 64}],
            "metrics": {"execution": {"unit": "ns", "samples": samples or [99, 100, 100, 101, 100]}}}


def budget():
    """Provide explicit example policy; these numbers are not release thresholds."""
    return {"schema": 1, "baseline_sha256": "f" * 64, "reason": "test policy only", "metrics": {
        "execution": {"direction": "lower", "minimum_samples": 5, "maximum_relative_spread": 0.1,
                      "maximum_ratio": 1.1, "absolute_allowance": 2, "reason": "test boundary"}}}


class BaselineTests(unittest.TestCase):
    def test_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            data = json.dumps(report()).encode()
            path.write_bytes(data)
            self.assertEqual(baseline.load(path), (report(), hashlib.sha256(data).hexdigest()))
            path.write_text('{"schema":1,"schema":1}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                baseline.load(path)
            path.write_text('{"value":NaN}')
            with self.assertRaisesRegex(ValueError, "nonfinite JSON number"):
                baseline.load(path)
            path.write_bytes(b" " * (baseline.MAX_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "exceeds 8 MiB"):
                baseline.load(path)

    def test_validate_report(self):
        baseline.validate_report(report())
        for bad in (True, -1, 2**64, 1.5):
            value = report([bad])
            with self.assertRaisesRegex(ValueError, "sample must be an integer"):
                baseline.validate_report(value)
        value = report()
        value["identity"]["unexpected"] = "field"
        with self.assertRaisesRegex(ValueError, "identity requires exactly"):
            baseline.validate_report(value)

    def test_merge(self):
        first, second = report([1]), report([2])
        second["provenance"][0]["source_sha256"] = "0" * 64
        merged = baseline.merge([first, second])
        self.assertEqual(merged["metrics"]["execution"]["samples"], [1, 2])
        self.assertEqual(merged["provenance"], first["provenance"] + second["provenance"])
        self.assertEqual(first["metrics"]["execution"]["samples"], [1])
        second["identity"]["environment"] = "other GPU"
        with self.assertRaisesRegex(ValueError, "identities differ"):
            baseline.merge([first, second])

    def test_compare(self):
        result = baseline.compare(report(), report([112] * 5), budget(), "f" * 64)
        self.assertEqual(result["metrics"]["execution"]["maximum_candidate_median"], "112")
        self.assertTrue(result["passed"])
        result = baseline.compare(report(), report([113] * 5), budget(), "f" * 64)
        self.assertEqual(result["metrics"]["execution"]["status"], "regression")
        self.assertFalse(result["passed"])

    def test_compare_stability(self):
        for samples in ([1, 100, 100, 100, 100], [100] * 4, [0, 0, 0, 0, 1]):
            result = baseline.compare(report(), report(samples), budget(), "f" * 64)
            self.assertEqual(result["metrics"]["execution"]["status"], "unstable")
        self.assertTrue(baseline.compare(report([0] * 5), report([0] * 5), budget(), "f" * 64)["passed"])
        policy = budget()
        policy["metrics"]["execution"]["maximum_relative_spread"] = 2
        result = baseline.compare(report([100, 100, 110, 110, 110, 110]), report([123] * 5), policy, "f" * 64)
        self.assertEqual(result["metrics"]["execution"]["baseline_median"], 110)
        self.assertTrue(result["passed"])

    def test_compare_identity_and_policy(self):
        with self.assertRaisesRegex(ValueError, "baseline SHA256 differs"):
            baseline.compare(report(), report(), budget(), "0" * 64)
        for field, value in (("minimum_samples", 1), ("maximum_ratio", True),
                             ("absolute_allowance", -1), ("reason", " "), ("direction", "higher")):
            policy = budget()
            policy["metrics"]["execution"][field] = value
            with self.assertRaises(ValueError):
                baseline.compare(report(), report(), policy, "f" * 64)
        candidate = report()
        candidate["metrics"]["execution"]["unit"] = "bytes"
        with self.assertRaisesRegex(ValueError, "units differ"):
            baseline.compare(report(), candidate, budget(), "f" * 64)

    def test_compare_bytes_and_decimal_boundary(self):
        original, candidate, policy = report([10]), report([11]), budget()
        original["metrics"]["execution"]["unit"] = "bytes"
        candidate["metrics"]["execution"]["unit"] = "bytes"
        policy["metrics"]["execution"].update(minimum_samples=1, absolute_allowance=0)
        self.assertTrue(baseline.compare(original, candidate, policy, "f" * 64)["passed"])
        policy["metrics"]["execution"]["maximum_ratio"] = Decimal("1.099999999999999999999999")
        self.assertFalse(baseline.compare(original, candidate, policy, "f" * 64)["passed"])

    def test_fraction_resource_bounds(self):
        for value in (Decimal("1e999999999"), Decimal("1e-999999999"),
                      Decimal("0." + "1" * 39)):
            with self.assertRaisesRegex(ValueError, "at most 38 digits and an exponent"):
                baseline.fraction(value, "ratio", 1)
        with self.assertRaisesRegex(ValueError, "must not exceed 1000000"):
            baseline.fraction(Decimal("1000001"), "ratio", 1)
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            baseline.fraction(Decimal("Infinity"), "ratio", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.json"
            path.write_text("[]")
            with patch.object(baseline.json, "loads", side_effect=RecursionError):
                with self.assertRaisesRegex(ValueError, "JSON nesting exceeds decoder limit"):
                    baseline.load(path)
            path.write_text('{"ratio":1e999999999}')
            value, _ = baseline.load(path)
            with self.assertRaisesRegex(ValueError, "at most 38 digits and an exponent"):
                baseline.fraction(value["ratio"], "ratio", 1)

    def test_main(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("baseline.json", "candidate.json", "budget.json")]
            paths[0].write_text(json.dumps(report()))
            paths[1].write_text(json.dumps(report([113] * 5)))
            policy = budget()
            policy["baseline_sha256"] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
            paths[2].write_text(json.dumps(policy))
            command = [sys.executable, str(Path(baseline.__file__)), "compare", *map(str, paths)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(result.stdout)["passed"])
            result = subprocess.run(command[:2] + ["merge", str(paths[0]), str(paths[1])],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(len(json.loads(result.stdout)["metrics"]["execution"]["samples"]), 10)


if __name__ == "__main__":
    unittest.main()
