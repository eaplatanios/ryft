from __future__ import annotations

from dataclasses import dataclass
import difflib
from pathlib import Path
import re
import textwrap

import jax

jax.config.update("jax_use_shardy_partitioner", True)

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ryft.jax.extraction import ProgramCase, lower_to_stablehlo

ROOT = Path(__file__).resolve().parents[4]
RESHAPE_RS = ROOT / "crates/ryft-xla/src/experimental/operations/reshape.rs"
FUNCTION_MLIR_PATTERN = re.compile(
    r"fn (?P<name>test_[^(]+)\(\) \{.*?assert_eq!\(\s*traced\.to_mlir_module\(\"main\"\)\.unwrap\(\),\s*"
    r"indoc! \{r#\"\n(?P<mlir>.*?)\n\s*\"#\}\s*\);",
    re.DOTALL,
)


@dataclass(frozen=True)
class ComparisonResult:
    """Captures the comparison outcome for one reshape lowering example."""

    case_name: str
    exact_match: bool
    normalized_match: bool
    jax_mlir: str
    rust_mlir: str


def load_expected_mlir() -> dict[str, str]:
    """Loads the current Rust-side reshape MLIR expectations from `ryft-xla`."""

    source = RESHAPE_RS.read_text()
    expected_mlir = {
        match.group("name"): textwrap.dedent(match.group("mlir")).strip() + "\n"
        for match in FUNCTION_MLIR_PATTERN.finditer(source)
    }
    required_cases = {
        "test_trace_reshape_with_sharding_constraint_renders_stablehlo_and_shardy",
        "test_shard_map_reshape_renders_singleton_axis_sharding_propagation",
        "test_shard_map_reshape_renders_replicated_merge_sharding_propagation",
        "test_shard_map_reshape_renders_replicated_split_sharding_propagation",
    }
    missing_case_names = sorted(required_cases - expected_mlir.keys())
    if missing_case_names:
        missing_cases = ", ".join(missing_case_names)
        raise RuntimeError(f"failed to find Rust-side MLIR expectations for: {missing_cases}")
    return {case_name: expected_mlir[case_name] for case_name in sorted(required_cases)}


def normalize_mlir(mlir: str) -> str:
    """Normalizes wrapper-only JAX lowering differences before textual comparison."""

    normalized_mlir = textwrap.dedent(mlir).strip() + "\n"
    normalized_mlir = re.sub(
        r"^module @\S+ attributes \{[^\n]*\} \{$",
        "module {",
        normalized_mlir,
        flags=re.MULTILINE,
    )
    normalized_mlir = normalized_mlir.replace("func.func public @main", "func.func @main")
    normalized_mlir = normalized_mlir.replace('{jax.result_info = "result", sdy.sharding = ', "{sdy.sharding = ")
    normalized_mlir = normalized_mlir.replace(', jax.result_info = "result"', "")
    normalized_mlir = normalized_mlir.replace(' {jax.result_info = "result"}', "")
    normalized_mlir = re.sub(r"-> \(tensor<([^>]+)>\s*\)", r"-> tensor<\1>", normalized_mlir)
    return normalized_mlir


def create_mesh(axis_size: int) -> Mesh:
    """Creates the logical mesh used by the Rust reshape tests."""

    devices = np.array(jax.devices())
    if devices.size < axis_size:
        raise RuntimeError(
            "expected at least 4 JAX devices; rerun with "
            "`XLA_FLAGS=--xla_force_host_platform_device_count=4`"
        )
    return Mesh(devices[:axis_size], ("x",))


def build_reshape_cases(mesh: Mesh) -> list[ProgramCase]:
    """Builds the reshape programs that should match the Rust-side MLIR assertions."""

    sharding = NamedSharding(mesh, P("x"))
    return [
        ProgramCase(
            name="test_trace_reshape_with_sharding_constraint_renders_stablehlo_and_shardy",
            description="Plain reshape with a Shardy sharding constraint.",
            function=lambda x: jnp.reshape(jax.lax.with_sharding_constraint(x, sharding), (1, 8, 1)),
            abstract_args=(jax.ShapeDtypeStruct((8,), jnp.float32),),
        ),
        ProgramCase(
            name="test_shard_map_reshape_renders_singleton_axis_sharding_propagation",
            description="`shard_map` reshape that adds singleton axes.",
            function=jax.shard_map(
                lambda x: jnp.reshape(x, (1, 2, 1)),
                mesh=mesh,
                in_specs=P("x"),
                out_specs=P(None, "x", None),
                axis_names={"x"},
            ),
            abstract_args=(jax.ShapeDtypeStruct((8,), jnp.float32),),
            jit_kwargs={
                "in_shardings": NamedSharding(mesh, P("x")),
                "out_shardings": NamedSharding(mesh, P(None, "x", None)),
            },
        ),
        ProgramCase(
            name="test_shard_map_reshape_renders_replicated_merge_sharding_propagation",
            description="`shard_map` reshape that merges replicated axes.",
            function=jax.shard_map(
                lambda x: jnp.reshape(x, (2, 6)),
                mesh=mesh,
                in_specs=P("x", None, None),
                out_specs=P("x", None),
                axis_names={"x"},
            ),
            abstract_args=(jax.ShapeDtypeStruct((8, 2, 3), jnp.float32),),
            jit_kwargs={
                "in_shardings": NamedSharding(mesh, P("x", None, None)),
                "out_shardings": NamedSharding(mesh, P("x", None)),
            },
        ),
        ProgramCase(
            name="test_shard_map_reshape_renders_replicated_split_sharding_propagation",
            description="`shard_map` reshape that splits a replicated axis.",
            function=jax.shard_map(
                lambda x: jnp.reshape(x, (2, 2, 3)),
                mesh=mesh,
                in_specs=P("x", None),
                out_specs=P("x", None, None),
                axis_names={"x"},
            ),
            abstract_args=(jax.ShapeDtypeStruct((8, 6), jnp.float32),),
            jit_kwargs={
                "in_shardings": NamedSharding(mesh, P("x", None)),
                "out_shardings": NamedSharding(mesh, P("x", None, None)),
            },
        ),
    ]


def compare_case(case: ProgramCase, rust_mlir: str) -> ComparisonResult:
    """Builds the comparison summary for one reshape case."""

    jax_mlir = lower_to_stablehlo(case.function, *case.lowering_args(), jit_kwargs=case.jit_kwargs)
    return ComparisonResult(
        case_name=case.name,
        exact_match=jax_mlir == rust_mlir,
        normalized_match=normalize_mlir(jax_mlir) == normalize_mlir(rust_mlir),
        jax_mlir=jax_mlir,
        rust_mlir=rust_mlir,
    )


def print_result(result: ComparisonResult) -> None:
    """Prints a human-readable comparison summary for one reshape case."""

    print(result.case_name)
    print(f"  exact_match: {result.exact_match}")
    print(f"  normalized_match: {result.normalized_match}")
    if result.normalized_match:
        print("  note: matches after removing JAX-only wrapper metadata")
        return

    normalized_jax_mlir = normalize_mlir(result.jax_mlir).splitlines(keepends=True)
    normalized_rust_mlir = normalize_mlir(result.rust_mlir).splitlines(keepends=True)
    print("  normalized_diff:")
    print(
        "".join(
            difflib.unified_diff(
                normalized_rust_mlir,
                normalized_jax_mlir,
                fromfile="rust",
                tofile="jax",
            )
        ),
        end="",
    )


def main() -> int:
    """Runs all reshape MLIR comparisons and returns a process status code."""

    expected_mlir = load_expected_mlir()
    mesh = create_mesh(4)
    results = [compare_case(case, expected_mlir[case.name]) for case in build_reshape_cases(mesh)]

    for result in results:
        print_result(result)

    return 0 if all(result.normalized_match for result in results) else 1
