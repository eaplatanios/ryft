"""Collects live benchmark IR and verifies it against committed Python snapshots."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ryft.jax.benchmark_cases import emit_single_jax_case
from ryft.jax.benchmark_snapshots import (
    assert_records_match_snapshot_cases,
    benchmark_snapshot_case_by_id,
    benchmark_snapshot_cases,
    python_root,
    repo_root,
)


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for live benchmark snapshot verification."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        default=[],
        help="Restrict verification to the named benchmark case. Repeat this flag to verify multiple cases.",
    )
    parser.add_argument(
        "--side",
        choices=("both", "jax", "ryft"),
        default="both",
        help="Restrict verification to one benchmark producer.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available benchmark snapshot cases and exit.",
    )
    parser.add_argument("--emit-jax-case", help=argparse.SUPPRESS)
    parser.add_argument("--emit-jax-dump-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    """Runs one subprocess and returns its stdout on success."""

    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout


def python_subprocess_environment(root: Path) -> dict[str, str]:
    """Builds the Python subprocess environment for local package execution."""

    environment = os.environ.copy()
    python_path_entries = [str(root / "python" / "src")]
    existing_python_path = environment.get("PYTHONPATH", "").strip()
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    return environment


def rust_benchmark_command_prefix() -> list[str]:
    """Returns the Rust benchmark command prefix used for live Ryft verification."""

    return [
        "cargo",
        "run",
        "-p",
        "ryft-xla",
        "--bin",
        "ir_benchmark",
        "--features",
        "benchmarking ndarray",
        "--",
    ]


def selected_snapshot_cases(case_ids: list[str]) -> tuple[Any, ...]:
    """Returns the selected committed snapshot cases in the requested order."""

    if not case_ids:
        return benchmark_snapshot_cases()

    selected_cases = []
    missing_case_ids = []
    for case_id in case_ids:
        try:
            selected_cases.append(benchmark_snapshot_case_by_id(case_id))
        except ValueError:
            missing_case_ids.append(case_id)

    if missing_case_ids:
        available_case_ids = ", ".join(case.case_id for case in benchmark_snapshot_cases())
        raise ValueError(
            "unknown benchmark case(s): "
            + ", ".join(missing_case_ids)
            + "; available cases: "
            + available_case_ids
        )
    return tuple(selected_cases)


def run_ryft_benchmark(root: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    """Runs the Rust-side benchmark emitter and returns its JSON records."""

    command = rust_benchmark_command_prefix()
    for case_id in case_ids:
        command.extend(["--case", case_id])
    records = json.loads(run_command(command, root))
    records.sort(key=lambda record: (record["case_id"], record["surface"]))
    return records


def collect_jax_records(root: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    """Collects JAX records in isolated subprocesses so XLA dump flags stay case-local."""

    temporary_root = python_root() / ".tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    dump_root = Path(
        tempfile.mkdtemp(
            dir=temporary_root,
            prefix="ir_benchmark_jax_",
        )
    )
    project_root = python_root()
    environment = python_subprocess_environment(root)
    try:
        for case_id in case_ids:
            case_dump_dir = dump_root / case_id
            command = [
                "uv",
                "run",
                "python",
                "-m",
                "ryft.jax.benchmark_parity",
                "--emit-jax-case",
                case_id,
                "--emit-jax-dump-dir",
                str(case_dump_dir),
            ]
            records.extend(json.loads(run_command(command, project_root, environment)))
    finally:
        shutil.rmtree(dump_root, ignore_errors=True)

    records.sort(key=lambda record: (record["case_id"], record["surface"]))
    return records


def verify_jax_snapshots(root: Path, case_ids: list[str]) -> int:
    """Verifies the selected JAX benchmark outputs against committed snapshots."""

    cases = selected_snapshot_cases(case_ids)
    records = collect_jax_records(root, [case.case_id for case in cases])
    assert_records_match_snapshot_cases(records, "jax", cases)
    print(f"validated {len(cases)} JAX benchmark snapshot(s)")
    return 0


def verify_ryft_snapshots(root: Path, case_ids: list[str]) -> int:
    """Verifies the selected Ryft benchmark outputs against committed snapshots."""

    cases = selected_snapshot_cases(case_ids)
    records = run_ryft_benchmark(root, [case.case_id for case in cases])
    assert_records_match_snapshot_cases(records, "ryft", cases)
    print(f"validated {len(cases)} Ryft benchmark snapshot(s)")
    return 0


def main() -> int:
    """Runs the requested live benchmark snapshot verification workflow."""

    arguments = parse_args()
    root = repo_root()

    if arguments.emit_jax_case is not None:
        if arguments.emit_jax_dump_dir is None:
            raise SystemExit("--emit-jax-case requires --emit-jax-dump-dir")
        print(json.dumps(emit_single_jax_case(arguments.emit_jax_case, arguments.emit_jax_dump_dir), indent=2))
        return 0

    cases = selected_snapshot_cases(arguments.case_ids)
    if arguments.list:
        for case in cases:
            print(f"{case.case_id}: {case.category} / {case.surface}")
        return 0

    try:
        if arguments.side in {"both", "jax"}:
            verify_jax_snapshots(root, [case.case_id for case in cases])
        if arguments.side in {"both", "ryft"}:
            verify_ryft_snapshots(root, [case.case_id for case in cases])
    except (AssertionError, ValueError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            stderr_text = (error.stderr or "").strip()
            if stderr_text:
                print(stderr_text, file=sys.stderr)
            else:
                print(str(error), file=sys.stderr)
        else:
            print(str(error), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
