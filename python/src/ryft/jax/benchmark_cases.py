from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable

from ryft.jax.ir_analysis import make_mlir_record, strip_mlir_location_markers

type BenchmarkEmitter = Callable[[], list[dict[str, Any]]]


def import_jax() -> tuple[Any, Any, Any, tuple[Any, Any, Any]]:
    """Imports JAX and returns the modules needed by the benchmark emitters."""

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, PartitionSpec
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

    return jax, jnp, np, (AxisType, Mesh, PartitionSpec)


def resolve_shard_map_function(jax: Any) -> Callable[..., Any]:
    """Returns the best available `shard_map` implementation for the local JAX version."""

    try:
        return jax.shard_map
    except AttributeError:
        from jax.experimental.shard_map import shard_map as shard_map_fn  # type: ignore

        return shard_map_fn


def configure_jax_dump_environment(dump_dir: Path) -> None:
    """Configures XLA flags so JAX emits Shardy compiler dumps into `dump_dir`."""

    dump_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dump_dir, ignore_errors=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    required_flags = [
        "--xla_force_host_platform_device_count=4",
        f"--xla_dump_to={dump_dir}",
        "--xla_dump_hlo_pass_re=shardy",
    ]
    combined_flags = " ".join([flag for flag in [xla_flags, *required_flags] if flag])
    os.environ["XLA_FLAGS"] = combined_flags


def select_jax_dumped_module(dump_dir: Path) -> str:
    """Selects the primary dumped JAX MLIR module for one benchmark case."""

    candidate_paths = sorted(dump_dir.rglob("module.mlir"))
    if not candidate_paths:
        raise SystemExit(
            f"jax compiler dumps under '{dump_dir}' did not contain a module.mlir artifact"
        )

    def candidate_score(path: Path) -> tuple[int, int, int, str]:
        module_text = path.read_text(encoding="utf-8")
        return (
            1
            if "sdy.manual_computation" in module_text or "sdy.mesh" in module_text
            else 0,
            module_text.count("func.func"),
            len(module_text),
            str(path),
        )

    best_path = max(candidate_paths, key=candidate_score)
    return best_path.read_text(encoding="utf-8")


def lower_to_jax_mlir(
    jax: Any, function: Callable[..., Any], dump_dir: Path, *arguments: Any
) -> str:
    """Compiles one callable and returns the dumped JAX MLIR module."""

    jax.jit(function).lower(*arguments).compile()
    return strip_mlir_location_markers(select_jax_dumped_module(dump_dir))


def emit_single_record(
    jax: Any,
    dump_dir: Path,
    case_id: str,
    category: str,
    surface: str,
    function: Callable[..., Any],
    *arguments: Any,
) -> list[dict[str, Any]]:
    """Builds one single-record benchmark emission from a callable and its inputs."""

    return [
        make_mlir_record(
            case_id,
            category,
            surface,
            lower_to_jax_mlir(jax, function, dump_dir, *arguments),
        )
    ]


def build_scalar_case_emitters(
    jax: Any, jnp: Any, np: Any, dump_dir: Path
) -> dict[str, BenchmarkEmitter]:
    """Builds the scalar JAX-side benchmark case registry."""

    scalar_two = np.array(2.0, dtype=np.float64)
    scalar_three = np.array(3.0, dtype=np.float64)
    scalar_tangent = np.array(1.0, dtype=np.float64)
    scalar_negative_tangent = np.array(-1.0, dtype=np.float64)

    def emit(
        case_id: str, surface: str, function: Callable[..., Any], *arguments: Any
    ) -> list[dict[str, Any]]:
        return emit_single_record(
            jax, dump_dir, case_id, "scalar", surface, function, *arguments
        )

    def bilinear_sin(left: Any, right: Any) -> Any:
        return left * right + jnp.sin(left)

    def quartic_plus_sin(x: Any) -> Any:
        return x * x * x * x + jnp.sin(x)

    def square_plus_sin(x: Any) -> Any:
        return x * x + jnp.sin(x)

    def scalar_hessian_style_second_derivative(x: Any) -> Any:
        return jax.jvp(jax.grad(quartic_plus_sin), (x,), (jnp.ones_like(x),))[1]

    def emit_scalar_bilinear_sin_jit() -> list[dict[str, Any]]:
        return emit(
            "scalar_bilinear_sin_jit", "jit", bilinear_sin, scalar_two, scalar_three
        )

    def emit_scalar_bilinear_sin_jvp() -> list[dict[str, Any]]:
        _primal_output, pushforward = jax.linearize(
            bilinear_sin, scalar_two, scalar_three
        )
        return emit(
            "scalar_bilinear_sin_jvp",
            "jvp_pushforward",
            pushforward,
            scalar_tangent,
            scalar_negative_tangent,
        )

    def emit_scalar_bilinear_sin_vjp_pullback() -> list[dict[str, Any]]:
        _primal_output, pullback = jax.vjp(bilinear_sin, scalar_two, scalar_three)
        return emit(
            "scalar_bilinear_sin_vjp_pullback", "vjp_pullback", pullback, scalar_tangent
        )

    def emit_scalar_quartic_plus_sin_grad() -> list[dict[str, Any]]:
        return emit(
            "scalar_quartic_plus_sin_grad",
            "grad",
            jax.grad(quartic_plus_sin),
            scalar_two,
        )

    def emit_scalar_quartic_plus_sin_value_and_grad() -> list[dict[str, Any]]:
        return emit(
            "scalar_quartic_plus_sin_value_and_grad",
            "value_and_grad",
            jax.value_and_grad(quartic_plus_sin),
            scalar_two,
        )

    def emit_scalar_quartic_plus_sin_linearize_pushforward() -> list[dict[str, Any]]:
        _primal_output, pushforward = jax.linearize(quartic_plus_sin, scalar_two)
        return emit(
            "scalar_quartic_plus_sin_linearize_pushforward",
            "linearize_pushforward",
            pushforward,
            scalar_tangent,
        )

    def emit_scalar_quartic_plus_sin_hessian_style() -> list[dict[str, Any]]:
        return emit(
            "scalar_quartic_plus_sin_hessian_style",
            "hessian_style",
            scalar_hessian_style_second_derivative,
            scalar_two,
        )

    return {
        "scalar_bilinear_sin_jit": emit_scalar_bilinear_sin_jit,
        "scalar_bilinear_sin_jvp": emit_scalar_bilinear_sin_jvp,
        "scalar_bilinear_sin_vjp_pullback": emit_scalar_bilinear_sin_vjp_pullback,
        "scalar_quartic_plus_sin_grad": emit_scalar_quartic_plus_sin_grad,
        "scalar_quartic_plus_sin_value_and_grad": emit_scalar_quartic_plus_sin_value_and_grad,
        "scalar_quartic_plus_sin_linearize_pushforward": emit_scalar_quartic_plus_sin_linearize_pushforward,
        "scalar_quartic_plus_sin_hessian_style": emit_scalar_quartic_plus_sin_hessian_style,
    }


def build_matrix_case_emitters(
    jax: Any, jnp: Any, np: Any, dump_dir: Path
) -> dict[str, BenchmarkEmitter]:
    """Builds the matrix JAX-side benchmark case registry."""

    matrix_left = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    matrix_right = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    matrix_cotangent = np.ones((2, 2), dtype=np.float64)
    hessian_x = np.array([[0.7]], dtype=np.float64)
    hessian_a = np.array([[2.0]], dtype=np.float64)
    hessian_b = np.array([[-1.5]], dtype=np.float64)
    hessian_c = np.array([[4.0]], dtype=np.float64)

    def emit(
        case_id: str, surface: str, function: Callable[..., Any], *arguments: Any
    ) -> list[dict[str, Any]]:
        return emit_single_record(
            jax, dump_dir, case_id, "matrix", surface, function, *arguments
        )

    def matmul_pair(left: Any, right: Any) -> Any:
        return left @ right

    def three_matmul_sine(x: Any, a: Any, b: Any, c: Any) -> Any:
        return jnp.sin(x @ a) @ b @ c

    def matrix_first_gradient(x: Any, a: Any, b: Any, c: Any) -> Any:
        outputs, pullback = jax.vjp(three_matmul_sine, x, a, b, c)
        x_bar, _, _, _ = pullback(jnp.ones_like(outputs))
        return x_bar

    def matrix_hessian_style_second_derivative(x: Any, a: Any, b: Any, c: Any) -> Any:
        return jax.jvp(
            matrix_first_gradient,
            (x, a, b, c),
            (jnp.ones_like(x), jnp.zeros_like(a), jnp.zeros_like(b), jnp.zeros_like(c)),
        )[1]

    def emit_matrix_matmul_jit() -> list[dict[str, Any]]:
        return emit("matrix_matmul_jit", "jit", matmul_pair, matrix_left, matrix_right)

    def emit_matrix_matmul_vjp_pullback() -> list[dict[str, Any]]:
        _primal_output, pullback = jax.vjp(matmul_pair, matrix_left, matrix_right)
        return emit(
            "matrix_matmul_vjp_pullback", "vjp_pullback", pullback, matrix_cotangent
        )

    def emit_matrix_three_matmul_sine_hessian_style() -> list[dict[str, Any]]:
        return emit(
            "matrix_three_matmul_sine_hessian_style",
            "hessian_style",
            matrix_hessian_style_second_derivative,
            hessian_x,
            hessian_a,
            hessian_b,
            hessian_c,
        )

    return {
        "matrix_matmul_jit": emit_matrix_matmul_jit,
        "matrix_matmul_vjp_pullback": emit_matrix_matmul_vjp_pullback,
        "matrix_three_matmul_sine_hessian_style": emit_matrix_three_matmul_sine_hessian_style,
    }


def build_shard_map_case_emitters(
    jax: Any,
    jnp: Any,
    np: Any,
    sharding_types: tuple[Any, Any, Any],
    dump_dir: Path,
) -> dict[str, BenchmarkEmitter]:
    """Builds the shard-map-heavy JAX-side benchmark case registry."""

    axis_type, mesh_type, partition_spec_type = sharding_types
    shard_map_fn = resolve_shard_map_function(jax)

    vector_input = np.arange(1.0, 9.0, dtype=np.float32)
    shard_map_left = np.arange(1.0, 33.0, dtype=np.float32).reshape((8, 4))
    shard_map_right = np.array(
        [[1.0, 2.0], [0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=np.float32
    )

    devices = np.array(jax.devices()[:4], dtype=object)
    mesh = mesh_type(devices.reshape((4,)), ("x",), axis_types=(axis_type.Manual,))
    nested_mesh = mesh_type(
        devices.reshape((2, 2)),
        ("x", "y"),
        axis_types=(axis_type.Manual, axis_type.Manual),
    )

    def emit(
        case_id: str, function: Callable[..., Any], *arguments: Any
    ) -> list[dict[str, Any]]:
        return emit_single_record(
            jax, dump_dir, case_id, "xla", "program", function, *arguments
        )

    def emit_shard_map_basic() -> list[dict[str, Any]]:
        sharded = shard_map_fn(
            lambda x: jnp.sin(x),
            mesh=mesh,
            in_specs=partition_spec_type("x"),
            out_specs=partition_spec_type("x"),
        )
        return emit("shard_map_basic", sharded, vector_input)

    def emit_shard_map_matmul() -> list[dict[str, Any]]:
        sharded = shard_map_fn(
            lambda lhs, rhs: lhs @ rhs,
            mesh=mesh,
            in_specs=(partition_spec_type("x", None), partition_spec_type(None, None)),
            out_specs=partition_spec_type("x", None),
        )
        return emit("shard_map_matmul", sharded, shard_map_left, shard_map_right)

    def emit_shard_map_grad_inside() -> list[dict[str, Any]]:
        sharded = shard_map_fn(
            lambda x: jax.grad(lambda y: jnp.sin(y).sum())(x),
            mesh=mesh,
            in_specs=partition_spec_type("x"),
            out_specs=partition_spec_type("x"),
        )
        return emit("shard_map_grad_inside", sharded, vector_input)

    def emit_grad_around_shard_map() -> list[dict[str, Any]]:
        sharded = shard_map_fn(
            lambda x: jnp.sin(x),
            mesh=mesh,
            in_specs=partition_spec_type("x"),
            out_specs=partition_spec_type("x"),
            check_vma=False,
        )

        def grad_like(x: Any) -> Any:
            outputs, pullback = jax.vjp(sharded, x)
            return pullback(jnp.ones_like(outputs))[0]

        return emit("grad_around_shard_map", grad_like, vector_input)

    def emit_nested_shard_map() -> list[dict[str, Any]]:
        inner = shard_map_fn(
            lambda x: x + x,
            mesh=nested_mesh,
            in_specs=partition_spec_type("y"),
            out_specs=partition_spec_type("y"),
            axis_names=frozenset({"y"}),
            check_vma=False,
        )
        outer = shard_map_fn(
            lambda x: inner(x) + x,
            mesh=nested_mesh,
            in_specs=partition_spec_type("x"),
            out_specs=partition_spec_type("x"),
            axis_names=frozenset({"x"}),
            check_vma=False,
        )
        return emit("nested_shard_map", outer, vector_input)

    return {
        "shard_map_basic": emit_shard_map_basic,
        "shard_map_matmul": emit_shard_map_matmul,
        "shard_map_grad_inside": emit_shard_map_grad_inside,
        "grad_around_shard_map": emit_grad_around_shard_map,
        "nested_shard_map": emit_nested_shard_map,
    }


def build_jax_case_emitters(
    jax: Any,
    jnp: Any,
    np: Any,
    sharding_types: tuple[Any, Any, Any],
    dump_dir: Path,
) -> dict[str, BenchmarkEmitter]:
    """Builds the full JAX-side benchmark case registry."""

    emitters: dict[str, BenchmarkEmitter] = {}
    emitters.update(build_scalar_case_emitters(jax, jnp, np, dump_dir))
    emitters.update(build_matrix_case_emitters(jax, jnp, np, dump_dir))
    emitters.update(
        build_shard_map_case_emitters(jax, jnp, np, sharding_types, dump_dir)
    )
    return emitters


def emit_single_jax_case(case_id: str, dump_dir: Path) -> list[dict[str, Any]]:
    """Emits the JAX-side benchmark records for one case inside a fresh process."""

    configure_jax_dump_environment(dump_dir)
    jax, jnp, np, sharding_types = import_jax()
    emitters = build_jax_case_emitters(jax, jnp, np, sharding_types, dump_dir)
    try:
        records = emitters[case_id]()
    except KeyError as error:
        raise SystemExit(
            f"jax emitter is missing benchmark case '{case_id}'"
        ) from error
    records.sort(key=lambda record: (record["case_id"], record["surface"]))
    return records


__all__ = [
    "BenchmarkEmitter",
    "build_jax_case_emitters",
    "configure_jax_dump_environment",
    "emit_single_jax_case",
    "import_jax",
    "lower_to_jax_mlir",
    "resolve_shard_map_function",
    "select_jax_dumped_module",
]
