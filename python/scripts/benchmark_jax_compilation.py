"""Matched JAX compilation and asynchronous-execution benchmark.

The paired Ryft benchmark is the `ryft-xla` `compilation_benchmark` binary. Both
harnesses use `sin(x * x + x)` over one float32 vector and emit the same JSON
schema. Timings are deliberately split at explicit lifecycle and readiness
boundaries; enqueue-only samples never synchronize inside the timed region.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp


SCHEMA = "ryft-jax-compilation-benchmark-v1"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50, help="timed warm iterations")
    parser.add_argument("--size", type=int, default=1 << 20, help="number of float32 elements")
    parser.add_argument("--cache-dir", type=Path, help="persistent compilation cache directory")
    parser.add_argument("--smoke", action="store_true", help="use small defaults and assert trace invariants")
    arguments = parser.parse_args()
    if arguments.smoke:
        arguments.iterations = min(arguments.iterations, 3)
        arguments.size = min(arguments.size, 1024)
    if arguments.iterations <= 0 or arguments.size <= 0:
        parser.error("--iterations and --size must both be greater than zero")
    return arguments


def command_output(arguments: list[str]) -> str | None:
    try:
        return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def measure(function: Callable[[], Any]) -> tuple[Any, int]:
    start = time.perf_counter_ns()
    output = function()
    return output, time.perf_counter_ns() - start


def distribution(samples: list[int]) -> dict[str, int]:
    ordered = sorted(samples)

    def percentile(percent: int) -> int:
        index = ((len(ordered) - 1) * percent + 99) // 100
        return ordered[index]

    return {
        "samples": len(ordered),
        "minimum_ns": ordered[0],
        "p50_ns": percentile(50),
        "p95_ns": percentile(95),
        "p99_ns": percentile(99),
        "maximum_ns": ordered[-1],
    }


def cache_files(directory: Path) -> list[str]:
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file())


def main() -> int:
    arguments = parse_arguments()
    if arguments.cache_dir is not None:
        arguments.cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(arguments.cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    trace_count = 0

    def workload(value: jax.Array) -> jax.Array:
        nonlocal trace_count
        trace_count += 1
        return jnp.sin(value * value + value)

    value = (jnp.arange(arguments.size, dtype=jnp.float32) % jnp.float32(1024.0)) / jnp.float32(1024.0)
    value.block_until_ready()
    jitted = jax.jit(workload)

    traced, cold_trace_ns = measure(lambda: jitted.trace(value))
    lowered, cold_lower_ns = measure(traced.lower)
    compiled, cold_compile_ns = measure(lowered.compile)
    compiled(value).block_until_ready()
    jitted(value).block_until_ready()
    trace_count_after_cold = trace_count

    enqueue_samples: list[int] = []
    pending_outputs: list[jax.Array] = []
    for _ in range(arguments.iterations):
        output, duration = measure(lambda: compiled(value))
        pending_outputs.append(output)
        enqueue_samples.append(duration)
    for output in pending_outputs:
        output.block_until_ready()

    synchronized_samples: list[int] = []
    for _ in range(arguments.iterations):
        _, duration = measure(lambda: compiled(value).block_until_ready())
        synchronized_samples.append(duration)

    warm_dispatch_samples: list[int] = []
    warm_outputs: list[jax.Array] = []
    for _ in range(arguments.iterations):
        output, duration = measure(lambda: jitted(value))
        warm_outputs.append(output)
        warm_dispatch_samples.append(duration)
    for output in warm_outputs:
        output.block_until_ready()
    trace_count_after_warm = trace_count

    if arguments.smoke:
        if trace_count_after_cold != 1:
            raise AssertionError(f"cold lifecycle traced {trace_count_after_cold} times instead of once")
        if trace_count_after_warm != trace_count_after_cold:
            raise AssertionError("warm JAX dispatch retraced the Python workload")

    persistent: dict[str, Any] = {"requested": False}
    if arguments.cache_dir is not None:
        files_before = cache_files(arguments.cache_dir)
        jax.clear_caches()
        restored_jitted = jax.jit(lambda input_value: jnp.sin(input_value * input_value + input_value))
        restored_traced = restored_jitted.trace(value)
        restored_lowered = restored_traced.lower()
        _, restore_duration_ns = measure(restored_lowered.compile)
        files_after = cache_files(arguments.cache_dir)
        persistent = {
            "requested": True,
            "duration_ns": restore_duration_ns,
            "cache_files_before": len(files_before),
            "cache_files_after": len(files_after),
            "restored": bool(files_before),
            "note": (
                "JAX does not expose a stable public per-call persistent-hit counter; restored=true means a cache "
                "entry existed before the second compile, not that the internal lookup is independently proven"
            ),
        }

    devices = jax.devices()
    report = {
        "schema": SCHEMA,
        "framework": "jax",
        "workload": {
            "name": "elementwise_polynomial_sine",
            "expression": "sin(x * x + x)",
            "dtype": "float32",
            "shape": [arguments.size],
            "device_count": 1,
        },
        "configuration": {
            "iterations": arguments.iterations,
            "smoke": arguments.smoke,
            "cache_directory": str(arguments.cache_dir) if arguments.cache_dir else None,
            "synchronization": "jax.Array.block_until_ready",
        },
        "metadata": {
            "jax_version": jax.__version__,
            "jaxlib_version": getattr(jax.lib, "__version__", None),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "python_version": platform.python_version(),
            "operating_system": platform.platform(),
            "backend": jax.default_backend(),
            "device_id": devices[0].id,
            "device_kind": devices[0].device_kind,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
        },
        "lifecycle": {
            "cold_trace_ns": cold_trace_ns,
            "cold_lower_ns": cold_lower_ns,
            "cold_compile_ns": cold_compile_ns,
            "persistent_restore": persistent,
        },
        "execution": {
            "warm_dispatch_call": distribution(warm_dispatch_samples),
            "enqueue_only": distribution(enqueue_samples),
            "synchronized": distribution(synchronized_samples),
        },
        "counters": {
            "python_traces_after_cold": trace_count_after_cold,
            "python_traces_after_warm": trace_count_after_warm,
            "backend_compilation_count": None,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
