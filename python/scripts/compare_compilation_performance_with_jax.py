"""Runs the matched Ryft and JAX compilation benchmark harnesses."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPOSITORY_ROOT / "python"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--cache-dir", type=Path, help="parent directory for isolated Ryft and JAX caches")
    parser.add_argument("--output", type=Path, help="write the combined JSON report to this path")
    parser.add_argument("--smoke", action="store_true", help="run small inputs and enable invariant assertions")
    return parser.parse_args()


def run_json(command: list[str], working_directory: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=working_directory, check=True, capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"benchmark did not emit valid JSON: {command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from error


def main() -> int:
    arguments = parse_arguments()
    common = ["--iterations", str(arguments.iterations), "--size", str(arguments.size)]
    if arguments.smoke:
        common.append("--smoke")

    ryft_arguments = common.copy()
    jax_arguments = common.copy()
    if arguments.cache_dir is not None:
        ryft_cache = arguments.cache_dir / "ryft"
        jax_cache = arguments.cache_dir / "jax"
        ryft_arguments.extend(["--cache-dir", str(ryft_cache)])
        jax_arguments.extend(["--cache-dir", str(jax_cache)])

    ryft = run_json(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "ryft-xla",
            "--features",
            "performance-benchmarking",
            "--bin",
            "compilation_benchmark",
            "--",
            *ryft_arguments,
        ],
        REPOSITORY_ROOT,
    )
    jax = run_json(
        ["uv", "run", "python", "scripts/benchmark_jax_compilation.py", *jax_arguments],
        PYTHON_ROOT,
    )

    if ryft["schema"] != jax["schema"]:
        raise RuntimeError(f"schema mismatch: Ryft={ryft['schema']!r}, JAX={jax['schema']!r}")
    if ryft["workload"] != jax["workload"]:
        raise RuntimeError(f"workload mismatch:\nRyft: {ryft['workload']}\nJAX: {jax['workload']}")

    report = {
        "schema": "ryft-jax-compilation-comparison-v1",
        "workload": ryft["workload"],
        "ryft": ryft,
        "jax": jax,
        "interpretation": {
            "cold_lifecycle": "trace, lower, and backend compile are timed through each framework's staged API",
            "warm_dispatch": "includes framework dispatch and asynchronous enqueue, but excludes synchronization",
            "enqueue_only": "uses an already compiled callable and synchronizes only after all timed samples",
            "synchronized": "includes callable invocation and explicit device readiness",
            "parity_claim": False,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
