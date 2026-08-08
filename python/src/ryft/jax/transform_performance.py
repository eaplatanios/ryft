"""Runtime transform benchmark comparison between Ryft and JAX."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ITERATIONS = 1_000
DEFAULT_WARMUP = 50
DEFAULT_MAX_RATIO = 1.0


@dataclass(frozen=True)
class RuntimeBenchmarkCase:
    """Describes one runtime transform benchmark case."""

    case_id: str
    category: str
    transform: str
    build_jax_body: Callable[[Any, Any, Any], Callable[[], Any]]


@dataclass(frozen=True)
class RuntimeBenchmarkRecord:
    """One measured runtime benchmark record."""

    case_id: str
    category: str
    transform: str
    warmup: int
    iterations: int
    min_ns: int
    median_ns: int
    mean_ns: int
    max_ns: int
    checksum: int

    @classmethod
    def from_json_record(cls, record: dict[str, Any]) -> RuntimeBenchmarkRecord:
        """Builds a runtime benchmark record from the Rust JSON emitter payload."""

        return cls(
            case_id=record["case_id"],
            category=record["category"],
            transform=record["transform"],
            warmup=record["warmup"],
            iterations=record["iterations"],
            min_ns=record["min_ns"],
            median_ns=record["median_ns"],
            mean_ns=record["mean_ns"],
            max_ns=record["max_ns"],
            checksum=record["checksum"],
        )

    def to_json_record(self) -> dict[str, Any]:
        """Converts this record to a JSON-serializable dictionary."""

        return {
            "case_id": self.case_id,
            "category": self.category,
            "transform": self.transform,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "min_ns": self.min_ns,
            "median_ns": self.median_ns,
            "mean_ns": self.mean_ns,
            "max_ns": self.max_ns,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class RuntimeBenchmarkComparison:
    """One Ryft/JAX runtime comparison row."""

    case_id: str
    category: str
    transform: str
    ryft: RuntimeBenchmarkRecord
    jax: RuntimeBenchmarkRecord

    @property
    def median_ratio(self) -> float:
        """Returns the Ryft median runtime divided by the JAX median runtime."""

        return self.ryft.median_ns / max(self.jax.median_ns, 1)

    def to_json_record(self) -> dict[str, Any]:
        """Converts this comparison to a JSON-serializable dictionary."""

        return {
            "case_id": self.case_id,
            "category": self.category,
            "transform": self.transform,
            "median_ratio": self.median_ratio,
            "ryft": self.ryft.to_json_record(),
            "jax": self.jax.to_json_record(),
        }


def rotate_left(value: int, shift: int) -> int:
    """Rotates one unsigned 64-bit integer left by `shift` bits."""

    shift %= 64
    value &= (1 << 64) - 1
    return ((value << shift) | (value >> (64 - shift))) & ((1 << 64) - 1)


def block_until_ready(value: Any) -> Any:
    """Blocks on a JAX value tree and returns the original value."""

    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return value
    if isinstance(value, dict):
        for child in value.values():
            block_until_ready(child)
        return value
    if isinstance(value, (tuple, list)):
        for child in value:
            block_until_ready(child)
    return value


def checksum_value(np: Any, value: Any) -> int:
    """Returns a deterministic checksum for one JAX output tree."""

    if isinstance(value, dict):
        checksum = 0
        for key in sorted(value):
            checksum = rotate_left(checksum, 1) ^ checksum_value(np, value[key])
        return checksum
    if isinstance(value, (tuple, list)):
        checksum = 0
        for child in value:
            checksum = rotate_left(checksum, 1) ^ checksum_value(np, child)
        return checksum

    array = np.ascontiguousarray(np.asarray(value))
    payload = repr(array.shape).encode("utf-8") + array.dtype.str.encode("utf-8") + array.tobytes()
    return zlib.crc32(payload) & 0xFFFFFFFF


def measure_jax_case(
    case: RuntimeBenchmarkCase,
    jax: Any,
    jnp: Any,
    np: Any,
    iterations: int,
    warmup: int,
) -> RuntimeBenchmarkRecord:
    """Measures one JAX benchmark case."""

    body = case.build_jax_body(jax, jnp, np)
    for _ in range(warmup):
        checksum_value(np, block_until_ready(body()))

    durations: list[int] = []
    checksum = 0
    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        output = block_until_ready(body())
        durations.append(time.perf_counter_ns() - start_ns)
        checksum = rotate_left(checksum, 1) ^ checksum_value(np, output)

    durations.sort()
    return RuntimeBenchmarkRecord(
        case_id=case.case_id,
        category=case.category,
        transform=case.transform,
        warmup=warmup,
        iterations=iterations,
        min_ns=durations[0] if durations else 0,
        median_ns=durations[iterations // 2] if durations else 0,
        mean_ns=round(statistics.fmean(durations)) if durations else 0,
        max_ns=durations[-1] if durations else 0,
        checksum=checksum,
    )


def bilinear_sin_tuple(jnp: Any, inputs: tuple[Any, Any]) -> Any:
    """Returns `x * y + sin(x)` for a tuple input."""

    left, right = inputs
    return left * right + jnp.sin(left)


def quartic_plus_sin(jnp: Any, value: Any) -> Any:
    """Returns `x**4 + sin(x)`."""

    return value * value * value * value + jnp.sin(value)


def build_runtime_cases() -> tuple[RuntimeBenchmarkCase, ...]:
    """Builds the stable runtime transform case registry."""

    def scalar_inputs(jnp: Any) -> tuple[Any, Any, Any, Any]:
        return (
            jnp.array(2.0, dtype=jnp.float64),
            jnp.array(3.0, dtype=jnp.float64),
            jnp.array(1.0, dtype=jnp.float64),
            jnp.array(-1.0, dtype=jnp.float64),
        )

    def scalar_jvp_direct(jax: Any, jnp: Any, np: Any) -> Callable[[], Any]:
        del np
        left, right, tangent_left, tangent_right = scalar_inputs(jnp)

        def body() -> Any:
            return jax.jvp(
                lambda inputs: bilinear_sin_tuple(jnp, inputs),
                ((left, right),),
                ((tangent_left, tangent_right),),
            )

        return body

    def scalar_linearize_build(jax: Any, jnp: Any, np: Any) -> Callable[[], Any]:
        del np
        value, _, _, _ = scalar_inputs(jnp)

        def body() -> Any:
            output, pushforward = jax.linearize(lambda x: quartic_plus_sin(jnp, x), value)
            return output, pushforward is not None

        return body

    def scalar_pushforward_apply(jax: Any, jnp: Any, np: Any) -> Callable[[], Any]:
        del np
        value, _, tangent, _ = scalar_inputs(jnp)
        _, pushforward = jax.linearize(lambda x: quartic_plus_sin(jnp, x), value)

        def body() -> Any:
            return pushforward(tangent)

        return body

    def scalar_vjp_build(jax: Any, jnp: Any, np: Any) -> Callable[[], Any]:
        del np
        left, right, _, _ = scalar_inputs(jnp)

        def body() -> Any:
            output, pullback = jax.vjp(lambda inputs: bilinear_sin_tuple(jnp, inputs), (left, right))
            return output, pullback is not None

        return body

    def scalar_pullback_apply(jax: Any, jnp: Any, np: Any) -> Callable[[], Any]:
        del np
        left, right, tangent, _ = scalar_inputs(jnp)
        _, pullback = jax.vjp(lambda inputs: bilinear_sin_tuple(jnp, inputs), (left, right))

        def body() -> Any:
            return pullback(tangent)

        return body

    def scalar_value_and_grad(jax: Any, jnp: Any, np: Any) -> Callable[[], Any]:
        del np
        value, _, _, _ = scalar_inputs(jnp)
        value_and_grad = jax.value_and_grad(lambda x: quartic_plus_sin(jnp, x))

        def body() -> Any:
            return value_and_grad(value)

        return body

    def matrix_jvp_matmul(jax: Any, jnp: Any, np: Any) -> Callable[[], Any]:
        del np
        left = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
        right = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float64)
        tangent_left = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
        tangent_right = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float64)

        def body() -> Any:
            return jax.jvp(
                lambda inputs: inputs[0] @ inputs[1],
                ((left, right),),
                ((tangent_left, tangent_right),),
            )

        return body

    def array_jacfwd_vector(jax: Any, jnp: Any, np: Any) -> Callable[[], Any]:
        del np
        value = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
        jacfwd = jax.jacfwd(lambda x: x * x + jnp.sin(x))

        def body() -> Any:
            return jacfwd(value)

        return body

    def array_jacrev_vector(jax: Any, jnp: Any, np: Any) -> Callable[[], Any]:
        del np
        value = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
        jacrev = jax.jacrev(lambda x: x * x + jnp.zeros_like(x))

        def body() -> Any:
            return jacrev(value)

        return body

    def array_hessian_scalar(jax: Any, jnp: Any, np: Any) -> Callable[[], Any]:
        del np
        value = jnp.array(2.0, dtype=jnp.float64)
        hessian = jax.hessian(lambda x: x * x + jnp.sin(x))

        def body() -> Any:
            return hessian(value)

        return body

    return (
        RuntimeBenchmarkCase("scalar_jvp_direct", "scalar", "jvp", scalar_jvp_direct),
        RuntimeBenchmarkCase("scalar_linearize_build", "scalar", "linearize", scalar_linearize_build),
        RuntimeBenchmarkCase(
            "scalar_pushforward_apply",
            "scalar",
            "pushforward_apply",
            scalar_pushforward_apply,
        ),
        RuntimeBenchmarkCase("scalar_vjp_build", "scalar", "vjp", scalar_vjp_build),
        RuntimeBenchmarkCase("scalar_pullback_apply", "scalar", "pullback_apply", scalar_pullback_apply),
        RuntimeBenchmarkCase("scalar_value_and_grad", "scalar", "value_and_grad", scalar_value_and_grad),
        RuntimeBenchmarkCase("matrix_jvp_matmul", "matrix", "jvp", matrix_jvp_matmul),
        RuntimeBenchmarkCase("array_jacfwd_vector", "array", "jacfwd", array_jacfwd_vector),
        RuntimeBenchmarkCase("array_jacrev_vector", "array", "jacrev", array_jacrev_vector),
        RuntimeBenchmarkCase("array_hessian_scalar", "array", "hessian", array_hessian_scalar),
    )


def runtime_case_by_id(case_id: str) -> RuntimeBenchmarkCase:
    """Returns one runtime benchmark case by ID."""

    for case in build_runtime_cases():
        if case.case_id == case_id:
            return case
    available_case_ids = ", ".join(case.case_id for case in build_runtime_cases())
    raise ValueError(f"unknown runtime benchmark case '{case_id}'; available cases: {available_case_ids}")


def selected_runtime_cases(case_ids: Sequence[str]) -> tuple[RuntimeBenchmarkCase, ...]:
    """Returns the selected runtime benchmark cases in requested order."""

    if not case_ids:
        return build_runtime_cases()
    return tuple(runtime_case_by_id(case_id) for case_id in case_ids)


def repo_root() -> Path:
    """Returns the repository root."""

    return Path(__file__).resolve().parents[4]


def import_jax() -> tuple[Any, Any, Any]:
    """Imports JAX and returns the modules needed by the runtime benchmark cases."""

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
    except (
        ImportError
    ) as error:  # pragma: no cover - exercised only when JAX is missing locally.
        raise SystemExit(
            "jax is not installed locally; install JAX or rerun with --skip-jax"
        ) from error

    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    return jax, jnp, np


def rust_transform_benchmark_command(case_ids: Sequence[str], iterations: int, warmup: int) -> list[str]:
    """Builds the Rust transform benchmark command."""

    command = [
        "cargo",
        "run",
        "--release",
        "-p",
        "ryft-xla",
        "--bin",
        "transform_benchmark",
        "--features",
        "performance-benchmarking ndarray",
        "--",
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
    ]
    for case_id in case_ids:
        command.extend(["--case", case_id])
    return command


def run_command(command: Sequence[str], cwd: Path) -> str:
    """Runs one subprocess and returns stdout."""

    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout


def run_ryft_records(case_ids: Sequence[str], iterations: int, warmup: int) -> dict[str, RuntimeBenchmarkRecord]:
    """Runs the Rust benchmark emitter and returns records keyed by case ID."""

    records = json.loads(run_command(rust_transform_benchmark_command(case_ids, iterations, warmup), repo_root()))
    return {
        record["case_id"]: RuntimeBenchmarkRecord.from_json_record(record)
        for record in records
    }


def run_jax_records(
    cases: Sequence[RuntimeBenchmarkCase],
    iterations: int,
    warmup: int,
) -> dict[str, RuntimeBenchmarkRecord]:
    """Runs JAX benchmark cases and returns records keyed by case ID."""

    jax, jnp, np = import_jax()
    return {
        case.case_id: measure_jax_case(case, jax, jnp, np, iterations, warmup)
        for case in cases
    }


def compare_runtime_records(
    cases: Sequence[RuntimeBenchmarkCase],
    ryft_records: dict[str, RuntimeBenchmarkRecord],
    jax_records: dict[str, RuntimeBenchmarkRecord],
) -> list[RuntimeBenchmarkComparison]:
    """Builds ordered Ryft/JAX comparison rows."""

    comparisons = []
    for case in cases:
        comparisons.append(
            RuntimeBenchmarkComparison(
                case_id=case.case_id,
                category=case.category,
                transform=case.transform,
                ryft=ryft_records[case.case_id],
                jax=jax_records[case.case_id],
            )
        )
    return comparisons


def format_duration(duration_ns: int) -> str:
    """Formats a nanosecond duration for benchmark tables."""

    if duration_ns >= 1_000_000:
        return f"{duration_ns / 1_000_000:.3f} ms"
    if duration_ns >= 1_000:
        return f"{duration_ns / 1_000:.3f} us"
    return f"{duration_ns} ns"


def format_comparison_table(comparisons: Sequence[RuntimeBenchmarkComparison], max_ratio: float) -> str:
    """Formats comparison rows as a compact text table."""

    rows = [
        ("case", "transform", "ryft median", "jax median", "ratio", "status"),
        ("-" * 30, "-" * 17, "-" * 12, "-" * 11, "-" * 7, "-" * 6),
    ]
    for comparison in comparisons:
        ratio = comparison.median_ratio
        rows.append(
            (
                comparison.case_id,
                comparison.transform,
                format_duration(comparison.ryft.median_ns),
                format_duration(comparison.jax.median_ns),
                f"{ratio:.3f}",
                "pass" if ratio <= max_ratio else "fail",
            )
        )

    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(widths[column]) for column, value in enumerate(row))
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    """Parses runtime transform benchmark comparison arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        default=[],
        help="Restrict benchmarking to one case. Repeat this flag to run multiple cases.",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Measured iterations per case.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations per case.")
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=DEFAULT_MAX_RATIO,
        help="Maximum allowed Ryft/JAX median runtime ratio.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON comparison records instead of a table.")
    parser.add_argument("--list", action="store_true", help="List available runtime benchmark cases and exit.")
    return parser.parse_args()


def main() -> int:
    """Runs the Ryft/JAX runtime transform benchmark comparison."""

    arguments = parse_args()
    cases = selected_runtime_cases(arguments.case_ids)

    if arguments.list:
        for case in cases:
            print(f"{case.case_id}: {case.category} / {case.transform}")
        return 0

    case_ids = [case.case_id for case in cases]
    try:
        ryft_records = run_ryft_records(case_ids, arguments.iterations, arguments.warmup)
        jax_records = run_jax_records(cases, arguments.iterations, arguments.warmup)
        comparisons = compare_runtime_records(cases, ryft_records, jax_records)
    except (ValueError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            stderr = (error.stderr or "").strip()
            stdout = (error.stdout or "").strip()
            if stdout:
                print(stdout, file=sys.stderr)
            print(stderr or str(error), file=sys.stderr)
        else:
            print(str(error), file=sys.stderr)
        return 1

    if arguments.json:
        print(json.dumps([comparison.to_json_record() for comparison in comparisons], indent=2))
    else:
        print(format_comparison_table(comparisons, arguments.max_ratio))

    failures = [comparison for comparison in comparisons if comparison.median_ratio > arguments.max_ratio]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
