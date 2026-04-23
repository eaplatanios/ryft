"""Snapshot helpers for the Python IR benchmark verification workflow."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


SnapshotSide = Literal["jax", "ryft"]


@dataclass(frozen=True)
class BenchmarkSnapshotCase:
    """Describes one committed IR benchmark snapshot pair."""

    case_id: str
    category: str
    surface: str

    def snapshot_directory(self, snapshot_root: Path | None = None) -> Path:
        """Returns the directory that contains this case's snapshot files."""

        return (snapshot_root or benchmark_snapshot_root()) / self.case_id / self.surface

    def snapshot_path(self, side: SnapshotSide, snapshot_root: Path | None = None) -> Path:
        """Returns the snapshot file path for one benchmark side."""

        return self.snapshot_directory(snapshot_root) / f"{side}.mlir"


BENCHMARK_SNAPSHOT_CASES = (
    BenchmarkSnapshotCase("grad_around_shard_map", "xla", "program"),
    BenchmarkSnapshotCase("matrix_matmul_jit", "matrix", "jit"),
    BenchmarkSnapshotCase("matrix_matmul_vjp_pullback", "matrix", "vjp_pullback"),
    BenchmarkSnapshotCase("matrix_three_matmul_sine_hessian_style", "matrix", "hessian_style"),
    BenchmarkSnapshotCase("nested_shard_map", "xla", "program"),
    BenchmarkSnapshotCase("scalar_bilinear_sin_jit", "scalar", "jit"),
    BenchmarkSnapshotCase("scalar_bilinear_sin_jvp", "scalar", "jvp_pushforward"),
    BenchmarkSnapshotCase("scalar_bilinear_sin_vjp_pullback", "scalar", "vjp_pullback"),
    BenchmarkSnapshotCase("scalar_quartic_plus_sin_grad", "scalar", "grad"),
    BenchmarkSnapshotCase("scalar_quartic_plus_sin_hessian_style", "scalar", "hessian_style"),
    BenchmarkSnapshotCase(
        "scalar_quartic_plus_sin_linearize_pushforward",
        "scalar",
        "linearize_pushforward",
    ),
    BenchmarkSnapshotCase("scalar_quartic_plus_sin_value_and_grad", "scalar", "value_and_grad"),
    BenchmarkSnapshotCase("shard_map_basic", "xla", "program"),
    BenchmarkSnapshotCase("shard_map_grad_inside", "xla", "program"),
    BenchmarkSnapshotCase("shard_map_matmul", "xla", "program"),
)

BENCHMARK_SNAPSHOT_CASES_BY_ID = {case.case_id: case for case in BENCHMARK_SNAPSHOT_CASES}


def python_root() -> Path:
    """Returns the root of the repository's `python` directory."""

    return Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    """Returns the repository root."""

    return python_root().parent


def benchmark_snapshot_root() -> Path:
    """Returns the root directory for committed benchmark snapshot files."""

    return python_root() / "tests" / "snapshots" / "ir_benchmark"


def benchmark_snapshot_cases() -> tuple[BenchmarkSnapshotCase, ...]:
    """Returns the full committed benchmark snapshot case list."""

    return BENCHMARK_SNAPSHOT_CASES


def benchmark_snapshot_case_by_id(case_id: str) -> BenchmarkSnapshotCase:
    """Returns one committed benchmark snapshot case by its case ID."""

    try:
        return BENCHMARK_SNAPSHOT_CASES_BY_ID[case_id]
    except KeyError as error:
        available_case_ids = ", ".join(case.case_id for case in BENCHMARK_SNAPSHOT_CASES)
        raise ValueError(
            f"unknown benchmark snapshot case '{case_id}'; available cases: {available_case_ids}"
        ) from error


def load_snapshot_text(
    case: BenchmarkSnapshotCase,
    side: SnapshotSide,
    snapshot_root: Path | None = None,
) -> str:
    """Loads one committed snapshot text payload."""

    return case.snapshot_path(side, snapshot_root).read_text(encoding="utf-8")


def normalize_text_for_snapshot(text: str) -> str:
    """Normalizes one text payload for snapshot storage and comparison."""

    normalized_text = text.replace("\r\n", "\n")
    if normalized_text and not normalized_text.endswith("\n"):
        normalized_text += "\n"
    return normalized_text


def write_snapshot_text(
    case: BenchmarkSnapshotCase,
    side: SnapshotSide,
    text: str,
    snapshot_root: Path | None = None,
) -> None:
    """Writes one committed snapshot text payload."""

    snapshot_path = case.snapshot_path(side, snapshot_root)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(normalize_text_for_snapshot(text), encoding="utf-8")


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    """Returns the lookup key for one benchmark record."""

    return record["case_id"], record["surface"]


def unified_text_diff(expected_text: str, actual_text: str, expected_name: str, actual_name: str) -> str:
    """Builds one unified diff for a snapshot mismatch."""

    return "".join(
        difflib.unified_diff(
            expected_text.splitlines(keepends=True),
            actual_text.splitlines(keepends=True),
            fromfile=expected_name,
            tofile=actual_name,
        )
    )


def assert_record_matches_snapshot(
    case: BenchmarkSnapshotCase,
    side: SnapshotSide,
    record: dict[str, Any],
    snapshot_root: Path | None = None,
) -> None:
    """Asserts that one benchmark record matches its committed snapshot."""

    if record["case_id"] != case.case_id:
        raise AssertionError(f"expected case_id '{case.case_id}' but got '{record['case_id']}'")
    if record["category"] != case.category:
        raise AssertionError(f"expected category '{case.category}' but got '{record['category']}'")
    if record["surface"] != case.surface:
        raise AssertionError(f"expected surface '{case.surface}' but got '{record['surface']}'")

    expected_text = load_snapshot_text(case, side, snapshot_root)
    actual_text = normalize_text_for_snapshot(record["raw_ir"])
    if actual_text != expected_text:
        snapshot_path = case.snapshot_path(side, snapshot_root)
        diff = unified_text_diff(
            expected_text,
            actual_text,
            str(snapshot_path),
            f"{case.case_id}/{case.surface}/{side} actual",
        )
        raise AssertionError(
            f"snapshot mismatch for {case.case_id} / {case.surface} / {side}\n{diff}"
        )


def assert_records_match_snapshot_cases(
    records: list[dict[str, Any]],
    side: SnapshotSide,
    cases: tuple[BenchmarkSnapshotCase, ...] | None = None,
    snapshot_root: Path | None = None,
) -> None:
    """Asserts that a benchmark record collection matches the committed snapshot corpus."""

    selected_cases = cases or benchmark_snapshot_cases()
    expected_keys = {(case.case_id, case.surface) for case in selected_cases}
    records_by_key = {record_key(record): record for record in records}
    actual_keys = set(records_by_key)

    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if missing_keys or unexpected_keys:
        problems: list[str] = []
        if missing_keys:
            problems.append(f"missing snapshot records: {missing_keys}")
        if unexpected_keys:
            problems.append(f"unexpected snapshot records: {unexpected_keys}")
        raise AssertionError("; ".join(problems))

    for case in selected_cases:
        assert_record_matches_snapshot(
            case,
            side,
            records_by_key[(case.case_id, case.surface)],
            snapshot_root,
        )
__all__ = [
    "BENCHMARK_SNAPSHOT_CASES",
    "BenchmarkSnapshotCase",
    "assert_record_matches_snapshot",
    "assert_records_match_snapshot_cases",
    "benchmark_snapshot_case_by_id",
    "benchmark_snapshot_cases",
    "benchmark_snapshot_root",
    "load_snapshot_text",
    "normalize_text_for_snapshot",
    "repo_root",
]
