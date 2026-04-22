from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ryft.jax.benchmark_cases import emit_single_jax_case
from ryft.jax.ir_analysis import normalize_mlir_records


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the benchmark comparison workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", default=[], help="Exact benchmark case ID to run"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Explicit output directory for artifacts"
    )
    parser.add_argument(
        "--run-name",
        help="Stable artifact directory name under .artifacts/ir_benchmark",
    )
    parser.add_argument(
        "--skip-jax",
        action="store_true",
        help="Emit only Ryft artifacts and skip the JAX comparison",
    )
    parser.add_argument("--emit-jax-case", help=argparse.SUPPRESS)
    parser.add_argument("--emit-jax-dump-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def repo_root() -> Path:
    """Returns the repository root inferred from the package module path."""

    return Path(__file__).resolve().parents[4]


def run_command(command: list[str], cwd: Path) -> str:
    """Runs one subprocess and returns its stdout on success."""

    completed = subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout


def rust_benchmark_command_prefix() -> list[str]:
    """Returns the shared Rust benchmark command prefix with the required feature set."""

    return [
        "cargo",
        "run",
        "-p",
        "ryft-core",
        "--bin",
        "ir_benchmark",
        "--features",
        "benchmarking ndarray xla",
        "--",
    ]


def load_rust_case_ids(root: Path) -> list[str]:
    """Loads the Rust-side benchmark case registry from the benchmark binary."""

    output = run_command([*rust_benchmark_command_prefix(), "--list"], root)
    return json.loads(output)


def run_rust_benchmark(root: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    """Runs the Rust-side benchmark emitter and returns its JSON records."""

    command = rust_benchmark_command_prefix()
    for case_id in case_ids:
        command.extend(["--case", case_id])
    return json.loads(run_command(command, root))


def collect_jax_records(
    root: Path, output_dir: Path, case_ids: list[str]
) -> list[dict[str, Any]]:
    """Collects JAX records by running one isolated subprocess per benchmark case."""

    records: list[dict[str, Any]] = []
    for case_id in case_ids:
        case_dump_dir = output_dir / "jax_dumps" / case_id
        command = [
            sys.executable,
            "-m",
            "ryft.jax.benchmark_parity",
            "--emit-jax-case",
            case_id,
            "--emit-jax-dump-dir",
            str(case_dump_dir),
        ]
        records.extend(json.loads(run_command(command, root)))
    records.sort(key=lambda record: (record["case_id"], record["surface"]))
    return records


def write_json(path: Path, payload: Any) -> None:
    """Writes one JSON payload to disk using stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_output_dir(root: Path, arguments: argparse.Namespace) -> Path:
    """Builds the output directory for one benchmark run."""

    if arguments.output_dir is not None:
        return arguments.output_dir

    run_name = arguments.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / ".artifacts" / "ir_benchmark" / run_name


def build_report_entries(
    ryft_records: list[dict[str, Any]], jax_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Builds machine-readable comparison entries keyed by case ID and surface."""

    ryft_index = {
        (record["case_id"], record["surface"]): record for record in ryft_records
    }
    jax_index = {
        (record["case_id"], record["surface"]): record for record in jax_records
    }
    keys = sorted(set(ryft_index) | set(jax_index))

    entries: list[dict[str, Any]] = []
    for key in keys:
        ryft_record = ryft_index.get(key)
        jax_record = jax_index.get(key)
        entry = {
            "case_id": key[0],
            "surface": key[1],
            "ryft": ryft_record,
            "jax": jax_record,
        }
        if ryft_record is not None and jax_record is not None:
            entry["delta"] = {
                "equation_count": ryft_record["summary"]["equation_count"]
                - jax_record["summary"]["equation_count"],
                "constant_count": ryft_record["summary"]["constant_count"]
                - jax_record["summary"]["constant_count"],
                "nested_region_count": ryft_record["summary"]["nested_region_count"]
                - jax_record["summary"]["nested_region_count"],
                "max_dependency_depth": ryft_record["summary"]["max_dependency_depth"]
                - jax_record["summary"]["max_dependency_depth"],
            }
        entries.append(entry)
    return entries


def interpretation(entry: dict[str, Any]) -> str:
    """Builds a short interpretation line for one comparison entry."""

    ryft_record = entry["ryft"]
    jax_record = entry["jax"]
    if ryft_record is None:
        return "Only the JAX artifact is available for this surface."
    if jax_record is None:
        return "Only the Ryft artifact is available for this surface."

    ryft_summary = ryft_record["summary"]
    jax_summary = jax_record["summary"]
    if (
        ryft_summary["equation_count"] == jax_summary["equation_count"]
        and ryft_summary["op_histogram"] == jax_summary["op_histogram"]
    ):
        return "Ryft and JAX have matching MLIR operation counts and normalized histograms."
    if jax_summary["equation_count"] == 0:
        return "JAX emitted no counted MLIR operations for this surface while Ryft did."
    ratio = ryft_summary["equation_count"] / jax_summary["equation_count"]
    return f"Ryft emits {ratio:.2f}x as many counted MLIR operations as JAX for this surface."


def render_report(entries: list[dict[str, Any]]) -> str:
    """Renders the Markdown comparison report."""

    lines = [
        "# Ryft MLIR vs JAX Shardy MLIR",
        "",
        "| Case | Surface | Ryft eqns | JAX eqns | Ryft consts | JAX consts |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in entries:
        ryft_summary = entry["ryft"]["summary"] if entry["ryft"] is not None else None
        jax_summary = entry["jax"]["summary"] if entry["jax"] is not None else None
        lines.append(
            "| {case_id} | {surface} | {ryft_eqns} | {jax_eqns} | {ryft_consts} | {jax_consts} |".format(
                case_id=entry["case_id"],
                surface=entry["surface"],
                ryft_eqns=ryft_summary["equation_count"]
                if ryft_summary is not None
                else "-",
                jax_eqns=jax_summary["equation_count"]
                if jax_summary is not None
                else "-",
                ryft_consts=ryft_summary["constant_count"]
                if ryft_summary is not None
                else "-",
                jax_consts=jax_summary["constant_count"]
                if jax_summary is not None
                else "-",
            )
        )

    for entry in entries:
        lines.extend(
            [
                "",
                f"## {entry['case_id']} / {entry['surface']}",
                "",
                interpretation(entry),
                "",
            ]
        )

        if entry["ryft"] is not None and entry["jax"] is not None:
            lines.append("Metric deltas:")
            lines.append(
                "- equation_count: {0}".format(
                    entry["ryft"]["summary"]["equation_count"]
                    - entry["jax"]["summary"]["equation_count"]
                )
            )
            lines.append(
                "- constant_count: {0}".format(
                    entry["ryft"]["summary"]["constant_count"]
                    - entry["jax"]["summary"]["constant_count"]
                )
            )
            lines.append(
                "- nested_region_count: {0}".format(
                    entry["ryft"]["summary"]["nested_region_count"]
                    - entry["jax"]["summary"]["nested_region_count"]
                )
            )
            lines.append(
                "- max_dependency_depth: {0}".format(
                    entry["ryft"]["summary"]["max_dependency_depth"]
                    - entry["jax"]["summary"]["max_dependency_depth"]
                )
            )
            lines.append("")

        if entry["ryft"] is not None:
            lines.extend(["### Ryft", "", "```", entry["ryft"]["raw_ir"], "```", ""])
        if entry["jax"] is not None:
            lines.extend(["### JAX", "", "```", entry["jax"]["raw_ir"], "```", ""])

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    """Runs the full benchmark comparison workflow."""

    arguments = parse_args()
    root = repo_root()

    if arguments.emit_jax_case is not None:
        if arguments.emit_jax_dump_dir is None:
            raise SystemExit("--emit-jax-case requires --emit-jax-dump-dir")
        print(
            json.dumps(
                emit_single_jax_case(
                    arguments.emit_jax_case, arguments.emit_jax_dump_dir
                ),
                indent=2,
            )
        )
        return 0

    output_dir = build_output_dir(root, arguments)

    rust_case_ids = load_rust_case_ids(root)
    selected_case_ids = arguments.case or rust_case_ids
    unknown_case_ids = sorted(set(selected_case_ids) - set(rust_case_ids))
    if unknown_case_ids:
        raise SystemExit("unknown benchmark case IDs: " + ", ".join(unknown_case_ids))

    ryft_records = normalize_mlir_records(run_rust_benchmark(root, selected_case_ids))
    write_json(output_dir / "ryft.json", ryft_records)

    if arguments.skip_jax:
        report = "# Ryft MLIR vs JAX Shardy MLIR\n\nJAX was skipped for this run.\n"
        (output_dir / "report.md").write_text(report, encoding="utf-8")
        return 0

    jax_records = normalize_mlir_records(
        collect_jax_records(root, output_dir, selected_case_ids)
    )
    write_json(output_dir / "jax.json", jax_records)

    entries = build_report_entries(ryft_records, jax_records)
    write_json(output_dir / "comparison.json", entries)
    (output_dir / "report.md").write_text(render_report(entries), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
