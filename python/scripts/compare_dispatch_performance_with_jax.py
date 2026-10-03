"""Compares the per-call dispatch latency of small eager and jitted operations between Ryft and JAX.

Every case applies one tiny operation (an `f32[8]` add, or a `cond` over such adds) on a single CPU device, so the
measured time is dispatch overhead rather than computation. Each case runs 100 warm-up calls and then several timed
rounds without per-call synchronization, matching the `dispatch_benchmark` binary of `ryft-xla`, and reports the median
and minimum per-call mean across rounds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000, help="timed calls per round")
    parser.add_argument("--rounds", type=int, default=5, help="timed rounds per case")
    parser.add_argument("--output", type=Path, help="write the combined JSON report to this path")
    return parser.parse_args()


def measure(call: Callable[[], Any], iterations: int, rounds: int) -> dict[str, Any]:
    for _ in range(min(100, iterations)):
        call()
    jax.block_until_ready(call())
    means = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            call()
        means.append((time.perf_counter_ns() - start) / iterations)
    return {"round_means_ns": means, "minimum_ns": min(means), "median_ns": statistics.median(means)}


def measure_jax(iterations: int, rounds: int) -> dict[str, Any]:
    with jax.default_device(jax.devices("cpu")[0]):
        value = jnp.arange(8, dtype=jnp.float32)
        predicate = jnp.asarray(True)
        jax.block_until_ready(value)
        jitted_add = jax.jit(lambda x: x + x)
        return {
            "eager_add_operator": measure(lambda: value + value, iterations, rounds),
            "eager_add_lax": measure(lambda: lax.add(value, value), iterations, rounds),
            # JAX re-traces and recompiles an eager `cond` whose branches are fresh lambdas on every call (tens of
            # milliseconds each), so this case uses far fewer calls per round to keep the comparison tractable.
            "eager_condition": measure(
                lambda: lax.cond(predicate, lambda x: x + x, lambda x: x * x, value),
                max(1, iterations // 1000),
                rounds,
            ),
            "jit_add": measure(lambda: jitted_add(value), iterations, rounds),
        }


def measure_ryft(iterations: int, rounds: int) -> dict[str, Any]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "-p",
        "ryft-xla",
        "--features",
        "performance-benchmarking",
        "--bin",
        "dispatch_benchmark",
        "--",
        "--iterations",
        str(iterations),
        "--rounds",
        str(rounds),
    ]
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True)
    try:
        return json.loads(result.stdout)["cases"]
    except json.JSONDecodeError as error:
        raise RuntimeError(f"`dispatch_benchmark` did not emit valid JSON:\n{result.stdout}\n{result.stderr}") from error


def main() -> int:
    arguments = parse_arguments()
    jax_cases = measure_jax(arguments.iterations, arguments.rounds)
    ryft_cases = measure_ryft(arguments.iterations, arguments.rounds)
    comparisons = [
        ("eager add (`x + x`)", "eager_add_operator", "eager_add"),
        ("eager add (`lax.add`)", "eager_add_lax", "eager_add"),
        ("eager condition (branches traced per call)", "eager_condition", "eager_condition"),
        ("warm `jit` add", "jit_add", "jit_add"),
        ("raw PJRT execute (Ryft only)", None, "pjrt_add"),
    ]
    print(f"{'case':46s} {'JAX (µs)':>10s} {'Ryft (µs)':>10s}")
    for label, jax_case, ryft_case in comparisons:
        jax_time = f"{jax_cases[jax_case]['median_ns'] / 1000:10.2f}" if jax_case else f"{'-':>10s}"
        print(f"{label:46s} {jax_time} {ryft_cases[ryft_case]['median_ns'] / 1000:10.2f}")
    report = {
        "schema": "ryft-jax-dispatch-comparison-v1",
        "jax_version": jax.__version__,
        "iterations": arguments.iterations,
        "rounds": arguments.rounds,
        "jax": jax_cases,
        "ryft": ryft_cases,
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
