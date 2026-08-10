"""Python helpers for inspecting and comparing JAX programs within `ryft`."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CaseComparison": ("ryft.jax.differential_testing", "CaseComparison"),
    "DifferentialObservation": ("ryft.jax.differential_testing", "DifferentialObservation"),
    "PROGRAM_STATISTICS_CASES": ("ryft.jax.program_statistics_cases", "PROGRAM_STATISTICS_CASES"),
    "AttachedRegionStatistics": ("ryft.jax.program_statistics", "AttachedRegionStatistics"),
    "JaxprInspection": ("ryft.jax.extraction", "JaxprInspection"),
    "PreservedDumpProgramCase": ("ryft.jax.preserved_dump_cases", "PreservedDumpProgramCase"),
    "ProgramCase": ("ryft.jax.extraction", "ProgramCase"),
    "ProgramInspection": ("ryft.jax.extraction", "ProgramInspection"),
    "ProgramStatistics": ("ryft.jax.program_statistics", "ProgramStatistics"),
    "ProgramStatisticsCase": ("ryft.jax.program_statistics_cases", "ProgramStatisticsCase"),
    "ProgramStatisticsRecord": ("ryft.jax.program_statistics", "ProgramStatisticsRecord"),
    "RegionStatistics": ("ryft.jax.program_statistics", "RegionStatistics"),
    "StableHloCollective": ("ryft.jax.differential_testing", "StableHloCollective"),
    "StagingObservation": ("ryft.jax.differential_testing", "StagingObservation"),
    "collect_program_statistics": ("ryft.jax.program_statistics", "collect_program_statistics"),
    "compare_differential_case": ("ryft.jax.differential_testing", "compare_case"),
    "differential_cases": ("ryft.jax.differential_testing", "differential_cases"),
    "extract_jaxpr": ("ryft.jax.extraction", "extract_jaxpr"),
    "inspect_program": ("ryft.jax.extraction", "inspect_program"),
    "lower_to_stablehlo": ("ryft.jax.extraction", "lower_to_stablehlo"),
    "parse_program_statistics_record": ("ryft.jax.program_statistics", "parse_program_statistics_record"),
    "parse_differential_observation": ("ryft.jax.differential_testing", "parse_observation"),
    "preserved_dump_program_case_by_id": (
        "ryft.jax.preserved_dump_cases",
        "preserved_dump_program_case_by_id",
    ),
    "preserved_dump_program_cases": ("ryft.jax.preserved_dump_cases", "preserved_dump_program_cases"),
    "program_cases_by_name": ("ryft.jax.examples", "program_cases_by_name"),
    "program_statistics_case_by_id": ("ryft.jax.program_statistics_cases", "program_statistics_case_by_id"),
    "project_collective_stablehlo": ("ryft.jax.differential_testing", "project_collective_stablehlo"),
    "render_program_inspection": ("ryft.jax.extraction", "render_program_inspection"),
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
