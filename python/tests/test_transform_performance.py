"""Unit tests for the runtime transform performance comparison helpers."""

from __future__ import annotations

import unittest

from ryft.jax.transform_performance import (
    RuntimeBenchmarkComparison,
    RuntimeBenchmarkRecord,
    format_comparison_table,
    runtime_case_by_id,
    selected_runtime_cases,
)


class TransformPerformanceTest(unittest.TestCase):
    """Covers the lightweight pieces of the Ryft/JAX transform benchmark harness."""

    def test_runtime_case_registry_contains_scalar_direct_jvp(self) -> None:
        case = runtime_case_by_id("scalar_jvp_direct")

        self.assertEqual(case.category, "scalar")
        self.assertEqual(case.transform, "jvp")

    def test_selected_runtime_cases_preserves_requested_order(self) -> None:
        cases = selected_runtime_cases(["array_hessian_scalar", "scalar_jvp_direct"])

        self.assertEqual([case.case_id for case in cases], ["array_hessian_scalar", "scalar_jvp_direct"])

    def test_comparison_table_marks_ratio_failures(self) -> None:
        ryft = RuntimeBenchmarkRecord(
            case_id="case",
            category="scalar",
            transform="jvp",
            warmup=1,
            iterations=1,
            min_ns=20,
            median_ns=20,
            mean_ns=20,
            max_ns=20,
            checksum=0,
        )
        jax = RuntimeBenchmarkRecord(
            case_id="case",
            category="scalar",
            transform="jvp",
            warmup=1,
            iterations=1,
            min_ns=10,
            median_ns=10,
            mean_ns=10,
            max_ns=10,
            checksum=0,
        )
        table = format_comparison_table(
            [RuntimeBenchmarkComparison("case", "scalar", "jvp", ryft, jax)],
            max_ratio=1.0,
        )

        self.assertIn("2.000", table)
        self.assertIn("fail", table)


if __name__ == "__main__":
    unittest.main()
