"""Snapshot-backed tests for the benchmark IR corpus."""

from __future__ import annotations

import subprocess
import unittest
from typing import Any

from ryft.jax.benchmark_parity import collect_jax_records, run_ryft_benchmark, rust_benchmark_command_prefix
from ryft.jax.benchmark_snapshots import (
    assert_records_match_snapshot_cases,
    benchmark_snapshot_case_by_id,
    benchmark_snapshot_cases,
    repo_root,
)


def summarize_process_error(error: subprocess.CalledProcessError) -> str:
    """Builds a short readable summary for one failed subprocess."""

    stderr_lines = (error.stderr or "").strip().splitlines()
    if stderr_lines:
        return "\n".join(stderr_lines[:12])
    return str(error)


class BenchmarkSnapshotCommandTest(unittest.TestCase):
    """Covers small command-level helpers used by the snapshot workflow."""

    def test_rust_benchmark_command_prefix_uses_the_ryft_xla_binary(self) -> None:
        self.assertEqual(
            rust_benchmark_command_prefix(),
            [
                "cargo",
                "run",
                "-p",
                "ryft-xla",
                "--bin",
                "ir_benchmark",
                "--features",
                "benchmarking ndarray",
                "--",
            ],
        )


class BenchmarkSnapshotTestBase(unittest.TestCase):
    """Shared snapshot assertion helpers for one live benchmark producer."""

    maxDiff = None
    record_side: str
    records_by_key: dict[tuple[str, str], dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        case_ids = [case.case_id for case in benchmark_snapshot_cases()]

        try:
            if cls.record_side == "jax":
                records = collect_jax_records(repo_root(), case_ids)
            else:
                records = run_ryft_benchmark(repo_root(), case_ids)
        except subprocess.CalledProcessError as error:
            raise unittest.SkipTest(
                f"unable to collect live {cls.record_side} benchmark IR:\n{summarize_process_error(error)}"
            ) from error

        cls.records_by_key = {(record["case_id"], record["surface"]): record for record in records}

    def assert_case_matches_snapshot(self, case_id: str) -> None:
        """Asserts that one benchmark case matches its committed snapshot."""

        case = benchmark_snapshot_case_by_id(case_id)
        record_key = (case.case_id, case.surface)
        self.assertIn(record_key, self.records_by_key)
        assert_records_match_snapshot_cases(
            [self.records_by_key[record_key]],
            self.record_side,  # type: ignore[arg-type]
            (case,),
        )


class JaxBenchmarkSnapshotTest(BenchmarkSnapshotTestBase):
    """Verifies live JAX benchmark IR against the committed snapshots."""

    record_side = "jax"

    def test_grad_around_shard_map(self) -> None:
        self.assert_case_matches_snapshot("grad_around_shard_map")

    def test_matrix_matmul_jit(self) -> None:
        self.assert_case_matches_snapshot("matrix_matmul_jit")

    def test_matrix_matmul_vjp_pullback(self) -> None:
        self.assert_case_matches_snapshot("matrix_matmul_vjp_pullback")

    def test_matrix_three_matmul_sine_hessian_style(self) -> None:
        self.assert_case_matches_snapshot("matrix_three_matmul_sine_hessian_style")

    def test_nested_shard_map(self) -> None:
        self.assert_case_matches_snapshot("nested_shard_map")

    def test_scalar_bilinear_sin_jit(self) -> None:
        self.assert_case_matches_snapshot("scalar_bilinear_sin_jit")

    def test_scalar_bilinear_sin_jvp(self) -> None:
        self.assert_case_matches_snapshot("scalar_bilinear_sin_jvp")

    def test_scalar_bilinear_sin_vjp_pullback(self) -> None:
        self.assert_case_matches_snapshot("scalar_bilinear_sin_vjp_pullback")

    def test_scalar_quartic_plus_sin_grad(self) -> None:
        self.assert_case_matches_snapshot("scalar_quartic_plus_sin_grad")

    def test_scalar_quartic_plus_sin_hessian_style(self) -> None:
        self.assert_case_matches_snapshot("scalar_quartic_plus_sin_hessian_style")

    def test_scalar_quartic_plus_sin_linearize_pushforward(self) -> None:
        self.assert_case_matches_snapshot("scalar_quartic_plus_sin_linearize_pushforward")

    def test_scalar_quartic_plus_sin_value_and_grad(self) -> None:
        self.assert_case_matches_snapshot("scalar_quartic_plus_sin_value_and_grad")

    def test_shard_map_basic(self) -> None:
        self.assert_case_matches_snapshot("shard_map_basic")

    def test_shard_map_grad_inside(self) -> None:
        self.assert_case_matches_snapshot("shard_map_grad_inside")

    def test_shard_map_matmul(self) -> None:
        self.assert_case_matches_snapshot("shard_map_matmul")


class RyftBenchmarkSnapshotTest(BenchmarkSnapshotTestBase):
    """Verifies live Ryft benchmark IR against the committed snapshots."""

    record_side = "ryft"

    def test_grad_around_shard_map(self) -> None:
        self.assert_case_matches_snapshot("grad_around_shard_map")

    def test_matrix_matmul_jit(self) -> None:
        self.assert_case_matches_snapshot("matrix_matmul_jit")

    def test_matrix_matmul_vjp_pullback(self) -> None:
        self.assert_case_matches_snapshot("matrix_matmul_vjp_pullback")

    def test_matrix_three_matmul_sine_hessian_style(self) -> None:
        self.assert_case_matches_snapshot("matrix_three_matmul_sine_hessian_style")

    def test_nested_shard_map(self) -> None:
        self.assert_case_matches_snapshot("nested_shard_map")

    def test_scalar_bilinear_sin_jit(self) -> None:
        self.assert_case_matches_snapshot("scalar_bilinear_sin_jit")

    def test_scalar_bilinear_sin_jvp(self) -> None:
        self.assert_case_matches_snapshot("scalar_bilinear_sin_jvp")

    def test_scalar_bilinear_sin_vjp_pullback(self) -> None:
        self.assert_case_matches_snapshot("scalar_bilinear_sin_vjp_pullback")

    def test_scalar_quartic_plus_sin_grad(self) -> None:
        self.assert_case_matches_snapshot("scalar_quartic_plus_sin_grad")

    def test_scalar_quartic_plus_sin_hessian_style(self) -> None:
        self.assert_case_matches_snapshot("scalar_quartic_plus_sin_hessian_style")

    def test_scalar_quartic_plus_sin_linearize_pushforward(self) -> None:
        self.assert_case_matches_snapshot("scalar_quartic_plus_sin_linearize_pushforward")

    def test_scalar_quartic_plus_sin_value_and_grad(self) -> None:
        self.assert_case_matches_snapshot("scalar_quartic_plus_sin_value_and_grad")

    def test_shard_map_basic(self) -> None:
        self.assert_case_matches_snapshot("shard_map_basic")

    def test_shard_map_grad_inside(self) -> None:
        self.assert_case_matches_snapshot("shard_map_grad_inside")

    def test_shard_map_matmul(self) -> None:
        self.assert_case_matches_snapshot("shard_map_matmul")


if __name__ == "__main__":
    unittest.main()
