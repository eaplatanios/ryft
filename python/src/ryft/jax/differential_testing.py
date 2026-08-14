"""Executable behavioral and StableHLO differential testing between Ryft and pinned JAX.

The Rust side is emitted by `cargo run -p ryft-xla --features differential-testing --bin differential_testing`.
The JAX side runs in a fresh Python process so that four host devices are configured before JAX is imported. Exact
parity cases compare named values and declared semantic StableHLO contracts. The bounded data-dependent case instead
encodes an explicit capability relation: eager values agree, Ryft stages the bounded result, and pinned JAX rejects
staging because the result extent depends on traced data.

This module intentionally does not import JAX at module load time.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence


if TYPE_CHECKING:
    from ryft.jax.differential_testing_cases import DifferentialCase


SCHEMA = "ryft-jax-differential-v1"
PINNED_JAX_VERSION = "0.10.0"
BUILD_HINT = "cargo build -p ryft-xla --features differential-testing --bin differential_testing"
SUBPROCESS_TIMEOUT_SECONDS = 1800


@dataclass(frozen=True)
class StagingObservation:
    """One framework's staging result for a differential case."""

    status: str
    output_type: str | None = None
    category: str | None = None


@dataclass(frozen=True)
class DifferentialObservation:
    """One framework's observations for one shared differential case."""

    schema: str
    case_id: str
    observations: dict[str, tuple[tuple[float, ...], ...]]
    staging: StagingObservation | None = None
    stablehlo: str | None = None


@dataclass(frozen=True)
class StableHloCollective:
    """Semantic StableHLO fields compared for one collective operation."""

    operation: str
    groups: tuple[tuple[int, ...], ...]
    axis_attributes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class CaseComparison:
    """Comparison result for one Ryft/JAX case pair."""

    case_id: str
    differences: tuple[str, ...]

    def passed(self) -> bool:
        """Returns whether the case produced no differences."""

        return not self.differences


def repo_root() -> Path:
    """Returns the repository root containing the `python` and `crates` directories."""

    return Path(__file__).resolve().parents[4]


def _required_mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Returns `value` as a mapping or raises a path-specific schema error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"field '{path}' must be an object")
    return value


def _required_string(value: Any, path: str) -> str:
    """Returns `value` as a string or raises a path-specific schema error."""

    if not isinstance(value, str):
        raise ValueError(f"field '{path}' must be a string")
    return value


def _parse_observations(value: Any) -> dict[str, tuple[tuple[float, ...], ...]]:
    """Parses named floating-point observations from one record payload."""

    observations = _required_mapping(value, "observations")
    parsed: dict[str, tuple[tuple[float, ...], ...]] = {}
    for name, executions in observations.items():
        name = _required_string(name, "observations key")
        if not isinstance(executions, Sequence) or isinstance(executions, (str, bytes)):
            raise ValueError(f"field 'observations.{name}' must be an array")
        parsed_executions = []
        for execution_index, execution in enumerate(executions):
            if not isinstance(execution, Sequence) or isinstance(execution, (str, bytes)):
                raise ValueError(f"field 'observations.{name}[{execution_index}]' must be an array")
            values = []
            for value_index, element in enumerate(execution):
                if isinstance(element, bool) or not isinstance(element, (int, float)):
                    raise ValueError(
                        f"field 'observations.{name}[{execution_index}][{value_index}]' must be numeric"
                    )
                values.append(float(element))
            parsed_executions.append(tuple(values))
        parsed[name] = tuple(parsed_executions)
    return parsed


def _parse_staging(value: Any) -> StagingObservation:
    """Parses one staging observation and validates its status-specific fields."""

    fields = _required_mapping(value, "staging")
    status = _required_string(fields.get("status"), "staging.status")
    output_type = fields.get("output_type")
    category = fields.get("category")
    if status == "supported":
        return StagingObservation(status=status, output_type=_required_string(output_type, "staging.output_type"))
    if status == "rejected":
        return StagingObservation(status=status, category=_required_string(category, "staging.category"))
    raise ValueError(f"field 'staging.status' has unsupported value '{status}'")


def parse_observation(value: Any) -> DifferentialObservation:
    """Parses and validates one versioned differential observation record."""

    fields = _required_mapping(value, "record")
    schema = _required_string(fields.get("schema"), "schema")
    if schema != SCHEMA:
        raise ValueError(f"unsupported differential observation schema '{schema}'")
    stablehlo = fields.get("stablehlo")
    if stablehlo is not None:
        stablehlo = _required_string(stablehlo, "stablehlo")
    staging = fields.get("staging")
    return DifferentialObservation(
        schema=schema,
        case_id=_required_string(fields.get("case_id"), "case_id"),
        observations=_parse_observations(fields.get("observations")),
        staging=None if staging is None else _parse_staging(staging),
        stablehlo=stablehlo,
    )


def observation_payload(observation: DifferentialObservation) -> dict[str, Any]:
    """Returns the JSON-serializable payload for one observation."""

    payload = asdict(observation)
    if observation.staging is None:
        payload.pop("staging")
    else:
        payload["staging"] = {name: value for name, value in payload["staging"].items() if value is not None}
    if observation.stablehlo is None:
        payload.pop("stablehlo")
    return payload


_COLLECTIVE_PATTERN = re.compile(
    r'"stablehlo\.(all_gather|reduce_scatter|all_to_all|collective_permute)"[^\n]*'
)
_DENSE_GROUPS_PATTERN = re.compile(r"(?:replica_groups|source_target_pairs) = dense<(\[\[.*?\]\])>")
_AXIS_PATTERNS = {
    "all_gather_dim": re.compile(r"all_gather_dim = (\d+) : i64"),
    "scatter_dimension": re.compile(r"scatter_dimension = (\d+) : i64"),
    "concat_dimension": re.compile(r"concat_dimension = (\d+) : i64"),
    "split_count": re.compile(r"split_count = (\d+) : i64"),
    "split_dimension": re.compile(r"split_dimension = (\d+) : i64"),
}


def project_collective_stablehlo(module: str) -> tuple[StableHloCollective, ...]:
    """Projects collective semantics from a StableHLO module.

    The projection intentionally ignores SSA names, channel handles, tensor spellings, wrapper functions, and Shardy
    metadata. It retains the collective family, ordered replica/source-target groups, and operation-defining axis
    attributes. Those are the cross-framework contracts Phase 7 needs to compare.

    # Parameters

      - `module`: Textual StableHLO module emitted by Ryft or JAX.
    """

    collectives = []
    for match in _COLLECTIVE_PATTERN.finditer(module):
        operation = match.group(1)
        operation_text = match.group(0)
        groups_match = _DENSE_GROUPS_PATTERN.search(operation_text)
        if groups_match is None:
            raise ValueError(f"stablehlo.{operation} is missing replica/source-target groups")
        groups_payload = json.loads(groups_match.group(1))
        groups = tuple(tuple(int(member) for member in group) for group in groups_payload)
        axis_attributes = tuple(
            (name, int(axis_match.group(1)))
            for name, pattern in _AXIS_PATTERNS.items()
            if (axis_match := pattern.search(operation_text)) is not None
        )
        collectives.append(
            StableHloCollective(operation=operation, groups=groups, axis_attributes=axis_attributes)
        )
    return tuple(collectives)


def differential_cases() -> tuple[DifferentialCase, ...]:
    """Returns the shared case registry without importing JAX eagerly."""

    from ryft.jax.differential_testing_cases import DIFFERENTIAL_CASES

    return DIFFERENTIAL_CASES


def _compare_projection(
    side: str,
    module: str,
    collectives: tuple[StableHloCollective, ...],
) -> str | None:
    """Compares one framework's projected collectives against the registry expectation.

    Anchoring both frameworks to a declared expectation is what keeps the parity gate honest: two modules that no
    longer contain any recognizable collective project to the same empty tuple and would otherwise agree vacuously.

    # Parameters

      - `side`: Framework name used in the returned difference.
      - `module`: Textual StableHLO module emitted by that framework.
      - `collectives`: Expected projection declared by the registry entry.
    """

    projection = project_collective_stablehlo(module)
    missing = [
        operation
        for operation in dict.fromkeys(collective.operation for collective in collectives)
        if all(projected.operation != operation for projected in projection)
    ]
    if missing:
        return f"StableHLO collectives: {side} module is missing expected collective families {', '.join(missing)}"
    if projection != collectives:
        return f"StableHLO collectives: {side} {projection!r} != expected {collectives!r}"
    return None


def compare_case(
    relationship: str,
    collectives: tuple[StableHloCollective, ...],
    ryft: DifferentialObservation,
    jax: DifferentialObservation,
    stablehlo_patterns: tuple[str, ...] = (),
) -> CaseComparison:
    """Compares one Ryft/JAX record pair according to its declared capability relationship.

    # Parameters

      - `relationship`: Either `parity` or `ryft_exceeds_jax`.
      - `collectives`: Collective projection the registry entry expects from both frameworks.
      - `ryft`: Ryft-side observation.
      - `jax`: JAX-side observation.
      - `stablehlo_patterns`: Semantic operation or attribute spellings that both StableHLO modules must contain.
    """

    differences = []
    if ryft.case_id != jax.case_id:
        differences.append(f"case ID: ryft '{ryft.case_id}' != jax '{jax.case_id}'")
    if ryft.observations != jax.observations:
        differences.append(f"observations: ryft {ryft.observations!r} != jax {jax.observations!r}")
    if relationship == "parity":
        if ryft.staging != jax.staging:
            differences.append(f"staging: ryft {ryft.staging!r} != jax {jax.staging!r}")
        if not collectives and not stablehlo_patterns:
            differences.append("StableHLO: exact-parity case must declare a semantic contract")
        if ryft.stablehlo is None or jax.stablehlo is None:
            differences.append("StableHLO: exact-parity case requires modules from both frameworks")
        else:
            if collectives:
                differences.extend(
                    difference
                    for difference in (
                        _compare_projection("ryft", ryft.stablehlo, collectives),
                        _compare_projection("jax", jax.stablehlo, collectives),
                    )
                    if difference is not None
                )
            for side, module in (("Ryft", ryft.stablehlo), ("JAX", jax.stablehlo)):
                missing = tuple(pattern for pattern in stablehlo_patterns if pattern not in module)
                if missing:
                    differences.append(f"StableHLO: {side} module is missing semantic patterns {missing!r}")
    elif relationship == "ryft_exceeds_jax":
        if ryft.staging != StagingObservation(status="supported", output_type="f32[count]"):
            differences.append(f"Ryft staging: expected bounded symbolic support but got {ryft.staging!r}")
        if jax.staging != StagingObservation(status="rejected", category="concretization"):
            differences.append(f"JAX staging: expected concretization rejection but got {jax.staging!r}")
        if collectives or ryft.stablehlo is not None or jax.stablehlo is not None:
            differences.append("StableHLO: the staging-rejected capability case must not claim a module comparison")
    else:
        raise ValueError(f"unknown differential relationship '{relationship}'")
    return CaseComparison(case_id=ryft.case_id, differences=tuple(differences))


def _selected_cases(case_ids: Sequence[str]) -> tuple[DifferentialCase, ...]:
    """Returns registry entries selected by exact case ID, preserving registry order."""

    cases = differential_cases()
    if not case_ids:
        return cases
    unknown = [case_id for case_id in case_ids if all(case.case_id != case_id for case in cases)]
    if unknown:
        raise ValueError(f"unknown differential-testing case '{unknown[0]}'")
    selected = set(case_ids)
    return tuple(case for case in cases if case.case_id in selected)


def collect_ryft_observations(root: Path, case_ids: Sequence[str]) -> tuple[DifferentialObservation, ...]:
    """Runs the Rust emitter and parses its selected records."""

    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "ryft-xla",
        "--features",
        "differential-testing",
        "--bin",
        "differential_testing",
        "--",
    ]
    for case_id in case_ids:
        command.extend(("--case", case_id))
    try:
        result = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Ryft differential emitter failed:\n{error.stderr.strip()}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Ryft differential emitter timed out after {SUBPROCESS_TIMEOUT_SECONDS} seconds. A cold compile of the "
            f"emitter can exceed that budget; pre-build it with '{BUILD_HINT}' and rerun."
        ) from None
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise ValueError("Ryft differential emitter must return a JSON array")
    return tuple(parse_observation(record) for record in payload)


def collect_jax_observations(root: Path, case_ids: Sequence[str]) -> tuple[DifferentialObservation, ...]:
    """Runs the JAX emitters in a fresh process and parses their selected records."""

    command = [sys.executable, "-m", "ryft.jax.differential_testing", "--emit-jax"]
    for case_id in case_ids:
        command.extend(("--case", case_id))
    try:
        result = subprocess.run(
            command,
            cwd=root / "python",
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"JAX differential emitter failed:\n{error.stderr.strip()}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"JAX differential emitter timed out after {SUBPROCESS_TIMEOUT_SECONDS} seconds"
        ) from None
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise ValueError("JAX differential emitter must return a JSON array")
    return tuple(parse_observation(record) for record in payload)


def _record_map(records: Sequence[DifferentialObservation], side: str) -> dict[str, DifferentialObservation]:
    """Indexes records by case ID while rejecting duplicates."""

    indexed = {}
    for record in records:
        if record.case_id in indexed:
            raise ValueError(f"{side} emitted duplicate case '{record.case_id}'")
        indexed[record.case_id] = record
    return indexed


def run_comparison(root: Path, case_ids: Sequence[str]) -> tuple[CaseComparison, ...]:
    """Runs both frameworks and returns comparisons for the selected registry entries."""

    cases = _selected_cases(case_ids)
    selected_ids = tuple(case.case_id for case in cases)
    ryft = _record_map(collect_ryft_observations(root, selected_ids), "Ryft")
    jax = _record_map(collect_jax_observations(root, selected_ids), "JAX")
    if set(ryft) != set(selected_ids):
        raise ValueError(f"Ryft case set {sorted(ryft)} does not match selected cases {sorted(selected_ids)}")
    if set(jax) != set(selected_ids):
        raise ValueError(f"JAX case set {sorted(jax)} does not match selected cases {sorted(selected_ids)}")
    return tuple(
        compare_case(
            case.relationship,
            case.collectives,
            ryft[case.case_id],
            jax[case.case_id],
            case.stablehlo_patterns,
        )
        for case in cases
    )


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the comparison and private JAX-emission modes."""

    parser = argparse.ArgumentParser(description="Differentially test Ryft behavior and StableHLO against pinned JAX.")
    parser.add_argument("--case", action="append", default=[], help="select one case ID; may be repeated")
    parser.add_argument("--list", action="store_true", help="list case IDs without executing them")
    parser.add_argument("--emit-jax", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Runs the CLI and returns its process exit code."""

    parsed = parse_arguments(arguments)
    try:
        cases = _selected_cases(parsed.case)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if parsed.list:
        if parsed.emit_jax:
            raise ValueError("--list cannot be combined with --emit-jax")
        for case in cases:
            print(case.case_id)
        return 0
    if parsed.emit_jax:
        from ryft.jax.differential_testing_cases import build_jax_observations

        records = build_jax_observations(tuple(case.case_id for case in cases))
        print(json.dumps([observation_payload(record) for record in records], indent=2))
        return 0
    comparisons = run_comparison(repo_root(), tuple(case.case_id for case in cases))
    for comparison in comparisons:
        if comparison.passed():
            print(f"PASS {comparison.case_id}")
        else:
            print(f"FAIL {comparison.case_id}")
            for difference in comparison.differences:
                print(f"  {difference}")
    return 0 if all(comparison.passed() for comparison in comparisons) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CaseComparison",
    "DifferentialObservation",
    "PINNED_JAX_VERSION",
    "SCHEMA",
    "StableHloCollective",
    "StagingObservation",
    "collect_jax_observations",
    "collect_ryft_observations",
    "compare_case",
    "differential_cases",
    "main",
    "observation_payload",
    "parse_arguments",
    "parse_observation",
    "project_collective_stablehlo",
    "repo_root",
    "run_comparison",
]
