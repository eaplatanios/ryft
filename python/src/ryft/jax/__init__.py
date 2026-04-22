"""Python helpers for inspecting and comparing JAX programs within `ryft`."""

from ryft.jax.examples import program_cases_by_name
from ryft.jax.extraction import (
    JaxprInspection,
    ProgramCase,
    ProgramInspection,
    extract_jaxpr,
    inspect_program,
    lower_to_stablehlo,
    render_program_inspection,
)
from ryft.jax.ir_analysis import normalize_mlir_records, strip_mlir_location_markers, summarize_jaxpr, summarize_mlir

__all__ = [
    "JaxprInspection",
    "ProgramCase",
    "ProgramInspection",
    "extract_jaxpr",
    "inspect_program",
    "lower_to_stablehlo",
    "normalize_mlir_records",
    "program_cases_by_name",
    "render_program_inspection",
    "strip_mlir_location_markers",
    "summarize_jaxpr",
    "summarize_mlir",
]
