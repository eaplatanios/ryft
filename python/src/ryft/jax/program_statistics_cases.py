"""JAX-side workloads matching the Ryft program statistics case registry.

Each case builds a closed jaxpr for the JAX equivalent of one Ryft workload and collects the same structural
statistics from it. Nothing here compiles, lowers to MLIR, or writes compiler dumps: the comparison operates on
program structure only.

This module never imports JAX at load time. `build_jax_case_record` configures `XLA_FLAGS` and only then imports
JAX, which is what makes the one-child-process-per-case isolation in `ryft.jax.program_statistics` effective: a
module-level `import jax` anywhere on the import path would silently defeat it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from ryft.jax.program_statistics import (
    ProgramStatistics,
    ProgramStatisticsRecord,
    collect_program_statistics,
)


@dataclass(frozen=True)
class JaxModules:
    """Bundles the JAX modules and sharding types used by the case builders."""

    jax: Any
    jax_numpy: Any
    numpy: Any
    core: Any
    axis_type: Any
    mesh_type: Any
    partition_spec_type: Any


@dataclass(frozen=True)
class ProgramStatisticsCase:
    """One matched Ryft/JAX program statistics case.

    `comparable` is opt-in: it is enabled only for cases whose normalized operation vocabulary and depth metric
    measure corresponding structure on both sides. Enforcement is never enabled by default.
    """

    case_id: str
    category: str
    surface: str
    comparable: bool
    build: Callable[[], ProgramStatistics]


def configure_jax_device_environment() -> None:
    """Configures `XLA_FLAGS` for the four-device host mesh used by the shard-map cases.

    This must run before JAX is imported in the current process, so it is called at the top of the single public
    entry point that builds a case.
    """

    xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    required_flag = "--xla_force_host_platform_device_count=4"
    if required_flag not in xla_flags:
        os.environ["XLA_FLAGS"] = " ".join(flag for flag in [xla_flags, required_flag] if flag)


def import_jax() -> JaxModules:
    """Imports JAX and returns the modules needed by the case builders."""

    try:
        import jax
        import jax.numpy as jax_numpy
        import numpy
        from jax import core as jax_core
        from jax.sharding import AxisType, Mesh, PartitionSpec
    except ImportError as error:  # pragma: no cover - exercised only when JAX is missing locally.
        raise SystemExit("jax is not installed locally; install the python project dependencies") from error

    jax.config.update("jax_enable_x64", True)
    return JaxModules(
        jax=jax,
        jax_numpy=jax_numpy,
        numpy=numpy,
        core=jax_core,
        axis_type=AxisType,
        mesh_type=Mesh,
        partition_spec_type=PartitionSpec,
    )


def resolve_shard_map_function(jax: Any) -> Callable[..., Any]:
    """Returns the best available `shard_map` implementation for the local JAX version."""

    try:
        return jax.shard_map
    except AttributeError:
        from jax.experimental.shard_map import shard_map as shard_map_function  # type: ignore[import-not-found]

        return shard_map_function


def build_scalar_bilinear_sin_jit() -> ProgramStatistics:
    """Builds the plain traced statistics of `f(x, y) = x * y + sin(x)`."""

    modules = import_jax()
    two = modules.numpy.array(2.0, dtype=modules.numpy.float64)
    three = modules.numpy.array(3.0, dtype=modules.numpy.float64)

    def bilinear_sin(left: Any, right: Any) -> Any:
        return left * right + modules.jax_numpy.sin(left)

    return collect_program_statistics(modules.jax.make_jaxpr(bilinear_sin)(two, three), modules.core)


def build_scalar_bilinear_sin_vjp_pullback() -> ProgramStatistics:
    """Builds the statistics of the reverse-mode pullback of `f(x, y) = x * y + sin(x)`."""

    modules = import_jax()
    two = modules.numpy.array(2.0, dtype=modules.numpy.float64)
    three = modules.numpy.array(3.0, dtype=modules.numpy.float64)
    cotangent = modules.numpy.array(1.0, dtype=modules.numpy.float64)

    def bilinear_sin(left: Any, right: Any) -> Any:
        return left * right + modules.jax_numpy.sin(left)

    _primal_output, pullback = modules.jax.vjp(bilinear_sin, two, three)
    return collect_program_statistics(modules.jax.make_jaxpr(pullback)(cotangent), modules.core)


def quartic_plus_sin(modules: JaxModules, x: Any) -> Any:
    """Evaluates `f(x) = x * x * x * x + sin(x)`, the scalar higher-order case family's workload."""

    return x * x * x * x + modules.jax_numpy.sin(x)


def build_scalar_quartic_plus_sin_grad() -> ProgramStatistics:
    """Builds the statistics of the reverse-mode gradient of `f(x) = x⁴ + sin(x)`."""

    modules = import_jax()
    two = modules.numpy.array(2.0, dtype=modules.numpy.float64)
    gradient = modules.jax.grad(lambda x: quartic_plus_sin(modules, x))
    return collect_program_statistics(modules.jax.make_jaxpr(gradient)(two), modules.core)


def build_scalar_quartic_plus_sin_value_and_gradient() -> ProgramStatistics:
    """Builds the statistics of the value-and-gradient of `f(x) = x⁴ + sin(x)`."""

    modules = import_jax()
    two = modules.numpy.array(2.0, dtype=modules.numpy.float64)
    value_and_gradient = modules.jax.value_and_grad(lambda x: quartic_plus_sin(modules, x))
    return collect_program_statistics(modules.jax.make_jaxpr(value_and_gradient)(two), modules.core)


def build_scalar_quartic_plus_sin_linearize_pushforward() -> ProgramStatistics:
    """Builds the statistics of the linearized pushforward of `f(x) = x⁴ + sin(x)`."""

    modules = import_jax()
    two = modules.numpy.array(2.0, dtype=modules.numpy.float64)
    tangent = modules.numpy.array(1.0, dtype=modules.numpy.float64)
    _primal_output, pushforward = modules.jax.linearize(lambda x: quartic_plus_sin(modules, x), two)
    return collect_program_statistics(modules.jax.make_jaxpr(pushforward)(tangent), modules.core)


def single_axis_manual_mesh(modules: JaxModules) -> Any:
    """Returns the four-device single-axis manual mesh shared by the flat shard-map cases."""

    devices = modules.numpy.array(modules.jax.devices()[:4], dtype=object)
    return modules.mesh_type(devices.reshape((4,)), ("x",), axis_types=(modules.axis_type.Manual,))


def build_shard_map_basic() -> ProgramStatistics:
    """Builds the statistics of a sharded elementwise sine over a single manual mesh axis."""

    modules = import_jax()
    shard_map_function = resolve_shard_map_function(modules.jax)
    mesh = single_axis_manual_mesh(modules)
    vector_input = modules.numpy.arange(1.0, 9.0, dtype=modules.numpy.float32)
    sharded = shard_map_function(
        lambda x: modules.jax_numpy.sin(x),
        mesh=mesh,
        in_specs=modules.partition_spec_type("x"),
        out_specs=modules.partition_spec_type("x"),
    )
    return collect_program_statistics(modules.jax.make_jaxpr(sharded)(vector_input), modules.core)


def build_shard_map_matmul() -> ProgramStatistics:
    """Builds the statistics of a sharded matrix multiplication over a single manual mesh axis."""

    modules = import_jax()
    shard_map_function = resolve_shard_map_function(modules.jax)
    mesh = single_axis_manual_mesh(modules)
    left = modules.numpy.arange(1.0, 33.0, dtype=modules.numpy.float32).reshape((8, 4))
    right = modules.numpy.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=modules.numpy.float32)
    sharded = shard_map_function(
        lambda lhs, rhs: lhs @ rhs,
        mesh=mesh,
        in_specs=(modules.partition_spec_type("x", None), modules.partition_spec_type(None, None)),
        out_specs=modules.partition_spec_type("x", None),
    )
    return collect_program_statistics(modules.jax.make_jaxpr(sharded)(left, right), modules.core)


def build_nested_shard_map() -> ProgramStatistics:
    """Builds the statistics of a shard map nested inside another shard map over a two-axis mesh."""

    modules = import_jax()
    shard_map_function = resolve_shard_map_function(modules.jax)
    devices = modules.numpy.array(modules.jax.devices()[:4], dtype=object)
    nested_mesh = modules.mesh_type(
        devices.reshape((2, 2)),
        ("x", "y"),
        axis_types=(modules.axis_type.Manual, modules.axis_type.Manual),
    )
    vector_input = modules.numpy.arange(1.0, 9.0, dtype=modules.numpy.float32)
    inner = shard_map_function(
        lambda x: x + x,
        mesh=nested_mesh,
        in_specs=modules.partition_spec_type("y"),
        out_specs=modules.partition_spec_type("y"),
        axis_names=frozenset({"y"}),
        check_vma=False,
    )
    outer = shard_map_function(
        lambda x: inner(x) + x,
        mesh=nested_mesh,
        in_specs=modules.partition_spec_type("x"),
        out_specs=modules.partition_spec_type("x"),
        axis_names=frozenset({"x"}),
        check_vma=False,
    )
    return collect_program_statistics(modules.jax.make_jaxpr(outer)(vector_input), modules.core)


PROGRAM_STATISTICS_CASES = (
    ProgramStatisticsCase(
        case_id="scalar_bilinear_sin_jit",
        category="scalar",
        surface="jit",
        # Verified congruent: both sides report entry counts 2/1/3, normalized histogram
        # {add=1, mul=1, sin=1}, and depth 2, with no unmapped operation names on either side.
        comparable=True,
        build=build_scalar_bilinear_sin_jit,
    ),
    ProgramStatisticsCase(
        case_id="scalar_bilinear_sin_vjp_pullback",
        category="scalar",
        surface="vjp_pullback",
        comparable=False,
        build=build_scalar_bilinear_sin_vjp_pullback,
    ),
    ProgramStatisticsCase(
        case_id="scalar_quartic_plus_sin_grad",
        category="scalar",
        surface="grad",
        comparable=False,
        build=build_scalar_quartic_plus_sin_grad,
    ),
    ProgramStatisticsCase(
        case_id="scalar_quartic_plus_sin_value_and_gradient",
        category="scalar",
        surface="value_and_gradient",
        comparable=False,
        build=build_scalar_quartic_plus_sin_value_and_gradient,
    ),
    ProgramStatisticsCase(
        case_id="scalar_quartic_plus_sin_linearize_pushforward",
        category="scalar",
        surface="linearize_pushforward",
        # Verified congruent: both sides report entry counts 1/1/11, normalized histogram
        # {add=4, mul=7}, and depth 7, with no unmapped operation names on either side.
        comparable=True,
        build=build_scalar_quartic_plus_sin_linearize_pushforward,
    ),
    ProgramStatisticsCase(
        case_id="shard_map_basic",
        category="xla",
        surface="program",
        comparable=False,
        build=build_shard_map_basic,
    ),
    ProgramStatisticsCase(
        case_id="shard_map_matmul",
        category="xla",
        surface="program",
        comparable=False,
        build=build_shard_map_matmul,
    ),
    ProgramStatisticsCase(
        case_id="nested_shard_map",
        category="xla",
        surface="program",
        comparable=False,
        build=build_nested_shard_map,
    ),
)


def program_statistics_cases() -> tuple[ProgramStatisticsCase, ...]:
    """Returns the full case registry, in the same order as the Rust binary's `--list` output."""

    return PROGRAM_STATISTICS_CASES


def program_statistics_case_by_id(case_id: str) -> ProgramStatisticsCase:
    """Returns one registry case by its case ID.

    # Parameters

      - `case_id`: Stable case identifier.
    """

    for case in PROGRAM_STATISTICS_CASES:
        if case.case_id == case_id:
            return case
    available_case_ids = ", ".join(case.case_id for case in PROGRAM_STATISTICS_CASES)
    raise ValueError(f"unknown program statistics case '{case_id}'; available cases: {available_case_ids}")


def build_jax_case_record(case_id: str) -> ProgramStatisticsRecord:
    """Builds the JAX-side statistics record for one case inside a freshly started process.

    # Parameters

      - `case_id`: Stable case identifier.
    """

    configure_jax_device_environment()
    case = program_statistics_case_by_id(case_id)
    return ProgramStatisticsRecord(
        case_id=case.case_id,
        category=case.category,
        surface=case.surface,
        statistics=case.build(),
    )


__all__ = [
    "PROGRAM_STATISTICS_CASES",
    "JaxModules",
    "ProgramStatisticsCase",
    "build_jax_case_record",
    "configure_jax_device_environment",
    "import_jax",
    "program_statistics_case_by_id",
    "program_statistics_cases",
    "resolve_shard_map_function",
]
