"""Unit tests for the benchmark comparison workflow helpers."""

from __future__ import annotations

import unittest

from ryft.jax.benchmark_parity import build_report_entries, render_report, rust_benchmark_command_prefix


def build_record(case_id: str, surface: str, equation_count: int, constant_count: int) -> dict[str, object]:
    """Builds one minimal comparison record for report-generation tests."""

    return {
        "case_id": case_id,
        "category": "scalar",
        "surface": surface,
        "raw_ir": f"{case_id}:{surface}",
        "summary": {
            "equation_count": equation_count,
            "constant_count": constant_count,
            "nested_region_count": 0,
            "max_dependency_depth": 0,
            "op_histogram": {"add": equation_count},
        },
    }


class BenchmarkParityTest(unittest.TestCase):
    """Covers the lightweight report-building helpers."""

    def test_rust_benchmark_command_prefix_uses_the_required_feature_set(self) -> None:
        self.assertEqual(
            rust_benchmark_command_prefix(),
            [
                "cargo",
                "run",
                "-p",
                "ryft-core",
                "--bin",
                "ir_benchmark",
                "--features",
                "benchmarking ndarray xla",
                "--",
            ],
        )

    def test_render_report_describes_matching_histograms(self) -> None:
        entries = build_report_entries(
            [build_record("scalar_case", "jit", equation_count=2, constant_count=1)],
            [build_record("scalar_case", "jit", equation_count=2, constant_count=1)],
        )

        report = render_report(entries)

        self.assertEqual(entries[0]["delta"]["equation_count"], 0)
        self.assertIn("Ryft and JAX have matching MLIR operation counts", report)
        self.assertIn("## scalar_case / jit", report)


if __name__ == "__main__":
    unittest.main()
