"""Python helpers for inspecting and comparing JAX programs within `ryft`."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BenchmarkSnapshotCase": ("ryft.jax.benchmark_snapshots", "BenchmarkSnapshotCase"),
    "JaxprInspection": ("ryft.jax.extraction", "JaxprInspection"),
    "ProgramCase": ("ryft.jax.extraction", "ProgramCase"),
    "ProgramInspection": ("ryft.jax.extraction", "ProgramInspection"),
    "benchmark_snapshot_cases": ("ryft.jax.benchmark_snapshots", "benchmark_snapshot_cases"),
    "extract_jaxpr": ("ryft.jax.extraction", "extract_jaxpr"),
    "inspect_program": ("ryft.jax.extraction", "inspect_program"),
    "lower_to_stablehlo": ("ryft.jax.extraction", "lower_to_stablehlo"),
    "normalize_mlir_records": ("ryft.jax.ir_analysis", "normalize_mlir_records"),
    "program_cases_by_name": ("ryft.jax.examples", "program_cases_by_name"),
    "render_program_inspection": ("ryft.jax.extraction", "render_program_inspection"),
    "strip_mlir_location_markers": ("ryft.jax.ir_analysis", "strip_mlir_location_markers"),
    "summarize_jaxpr": ("ryft.jax.ir_analysis", "summarize_jaxpr"),
    "summarize_mlir": ("ryft.jax.ir_analysis", "summarize_mlir"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily resolves exported attributes so JAX-heavy helpers load on demand."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from error

    module = import_module(module_name)
    return getattr(module, attribute_name)
