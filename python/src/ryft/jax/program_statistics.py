"""Structural program-statistics comparison between Ryft and JAX.

This module mirrors the Rust `ryft_core::programs::statistics` schema in Python, collects the same statistics from
closed JAX jaxprs, and drives a structural comparison between the two sides. The Rust side is produced by
`cargo run -p ryft-xla --features program-statistics --bin program_statistics`; the JAX side is produced by this
module itself, one fresh child process per case so that `XLA_FLAGS` is configured before JAX is imported.

The comparison is a structural diff of program shape. It is not a performance measurement, and several statistics are
deliberately reported without ever being diffed as failures: `constant_count` counts different things on the two
sides (Ryft counts constant atoms, JAX counts `constvars` plus literals), region-slot vocabularies differ across the
two IRs, and every JAX attachment edge is a `computation` region because JAX has no dormant rule-region concept.

This module never imports JAX at module load time; every JAX import happens inside a function so that the
subprocess-isolated `XLA_FLAGS` configuration in `ryft.jax.program_statistics_cases` takes effect.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - imported lazily at runtime to avoid a module import cycle.
    from ryft.jax.program_statistics_cases import ProgramStatisticsCase


@dataclass(frozen=True)
class AttachedRegionStatistics:
    """One attachment edge from an instruction slot to a region in the program's region arena."""

    instruction_index: int
    operation: str
    region_slot: str
    region_role: str
    region_index: int

    def label(self) -> str:
        """Returns the fused `operation.region_slot` display form of this attachment edge."""

        return f"{self.operation}.{self.region_slot}"


@dataclass(frozen=True)
class RegionStatistics:
    """Region-local structural statistics for one region of a program."""

    input_count: int
    output_count: int
    instruction_count: int
    constant_count: int
    operation_counts: dict[str, int]
    maximum_output_dependency_depth: int
    attached_regions: tuple[AttachedRegionStatistics, ...]


@dataclass(frozen=True)
class ProgramStatistics:
    """Structural statistics for one whole program, one entry per arena region.

    Regions are stored in ascending arena order, so descendant regions always precede their parents and the final
    entry is always the program's entry region. A region that is attached several times appears exactly once here
    while appearing once per attachment edge in the `attached_regions` lists of its parents.
    """

    regions: tuple[RegionStatistics, ...]

    def region_count(self) -> int:
        """Returns the number of distinct regions in the arena."""

        return len(self.regions)

    def entry_region_index(self) -> int:
        """Returns the arena index of the entry region, which is always the final arena entry."""

        if not self.regions:
            raise ValueError("program statistics must contain at least the entry region")
        return len(self.regions) - 1

    def entry(self) -> RegionStatistics:
        """Returns the entry region's statistics."""

        return self.regions[self.entry_region_index()]

    def total_instruction_count(self) -> int:
        """Returns the instruction count summed over all arena regions, counting a shared region once."""

        return sum(region.instruction_count for region in self.regions)

    def total_constant_count(self) -> int:
        """Returns the constant count summed over all arena regions, counting a shared region once."""

        return sum(region.constant_count for region in self.regions)

    def total_operation_counts(self) -> dict[str, int]:
        """Returns the operation histogram merged over all arena regions, counting a shared region once."""

        totals: Counter[str] = Counter()
        for region in self.regions:
            totals.update(region.operation_counts)
        return dict(sorted(totals.items()))


@dataclass(frozen=True)
class ProgramStatisticsRecord:
    """One emitted statistics record: case metadata plus the case program's structural statistics."""

    case_id: str
    category: str
    surface: str
    statistics: ProgramStatistics


def _child_path(path: str, name: str) -> str:
    """Returns the JSON path of one named child field.

    # Parameters

      - `path`: JSON path of the parent object, empty at the record root.
      - `name`: Field name of the child.
    """

    return f"{path}.{name}" if path else name


def _indexed_path(path: str, index: int) -> str:
    """Returns the JSON path of one sequence element.

    # Parameters

      - `path`: JSON path of the sequence.
      - `index`: Element index.
    """

    return f"{path}[{index}]"


def _required_field(payload: Mapping[str, Any], name: str, path: str) -> Any:
    """Returns one required field value, raising `ValueError` with its full JSON path when it is missing."""

    if name not in payload:
        raise ValueError(f"missing required field '{_child_path(path, name)}'")
    return payload[name]


def _required_mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Returns one required JSON object value, raising `ValueError` with its full JSON path on a type mismatch."""

    if not isinstance(value, Mapping):
        raise ValueError(f"field '{path}' must be an object but got {type(value).__name__}")
    return value


def _required_sequence(value: Any, path: str) -> Sequence[Any]:
    """Returns one required JSON array value, raising `ValueError` with its full JSON path on a type mismatch."""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"field '{path}' must be an array but got {type(value).__name__}")
    return value


def _required_string(value: Any, path: str) -> str:
    """Returns one required JSON string value, raising `ValueError` with its full JSON path on a type mismatch."""

    if not isinstance(value, str):
        raise ValueError(f"field '{path}' must be a string but got {type(value).__name__}")
    return value


def _required_integer(value: Any, path: str) -> int:
    """Returns one required JSON integer value, raising `ValueError` with its full JSON path on a type mismatch."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"field '{path}' must be an integer but got {type(value).__name__}")
    return value


def _parse_attached_region(payload: Any, path: str) -> AttachedRegionStatistics:
    """Parses one attachment edge, ignoring unknown fields."""

    fields = _required_mapping(payload, path)
    return AttachedRegionStatistics(
        instruction_index=_required_integer(
            _required_field(fields, "instruction_index", path), _child_path(path, "instruction_index")
        ),
        operation=_required_string(_required_field(fields, "operation", path), _child_path(path, "operation")),
        region_slot=_required_string(_required_field(fields, "region_slot", path), _child_path(path, "region_slot")),
        region_role=_required_string(_required_field(fields, "region_role", path), _child_path(path, "region_role")),
        region_index=_required_integer(
            _required_field(fields, "region_index", path), _child_path(path, "region_index")
        ),
    )


def _parse_operation_counts(payload: Any, path: str) -> dict[str, int]:
    """Parses one operation histogram keyed by raw operation names."""

    counts = _required_mapping(payload, path)
    return {
        _required_string(name, path): _required_integer(count, _child_path(path, str(name)))
        for name, count in counts.items()
    }


def _parse_region(payload: Any, path: str) -> RegionStatistics:
    """Parses one region's statistics, ignoring unknown fields."""

    fields = _required_mapping(payload, path)
    attached_path = _child_path(path, "attached_regions")
    attached_payloads = _required_sequence(_required_field(fields, "attached_regions", path), attached_path)
    return RegionStatistics(
        input_count=_required_integer(_required_field(fields, "input_count", path), _child_path(path, "input_count")),
        output_count=_required_integer(
            _required_field(fields, "output_count", path), _child_path(path, "output_count")
        ),
        instruction_count=_required_integer(
            _required_field(fields, "instruction_count", path), _child_path(path, "instruction_count")
        ),
        constant_count=_required_integer(
            _required_field(fields, "constant_count", path), _child_path(path, "constant_count")
        ),
        operation_counts=_parse_operation_counts(
            _required_field(fields, "operation_counts", path), _child_path(path, "operation_counts")
        ),
        maximum_output_dependency_depth=_required_integer(
            _required_field(fields, "maximum_output_dependency_depth", path),
            _child_path(path, "maximum_output_dependency_depth"),
        ),
        attached_regions=tuple(
            _parse_attached_region(attached_payload, _indexed_path(attached_path, index))
            for index, attached_payload in enumerate(attached_payloads)
        ),
    )


def _parse_statistics(payload: Any, path: str) -> ProgramStatistics:
    """Parses one program's statistics, ignoring unknown fields."""

    fields = _required_mapping(payload, path)
    regions_path = _child_path(path, "regions")
    region_payloads = _required_sequence(_required_field(fields, "regions", path), regions_path)
    return ProgramStatistics(
        regions=tuple(
            _parse_region(region_payload, _indexed_path(regions_path, index))
            for index, region_payload in enumerate(region_payloads)
        )
    )


def parse_program_statistics_record(payload: Mapping[str, Any]) -> ProgramStatisticsRecord:
    """Parses one complete serialized statistics record.

    Required fields are strict: a missing field or a wrongly typed field raises a `ValueError` naming the full JSON
    field path, such as `statistics.regions[2].instruction_count`. Unknown fields are silently ignored at every
    level so that adding a statistic on the Rust side does not break this tooling. The parser has no case context,
    so callers that know which case a payload came from wrap the raised error with the case ID.

    # Parameters

      - `payload`: One decoded JSON record object.
    """

    fields = _required_mapping(payload, "")
    return ProgramStatisticsRecord(
        case_id=_required_string(_required_field(fields, "case_id", ""), "case_id"),
        category=_required_string(_required_field(fields, "category", ""), "category"),
        surface=_required_string(_required_field(fields, "surface", ""), "surface"),
        statistics=_parse_statistics(_required_field(fields, "statistics", ""), "statistics"),
    )


def record_payload(record: ProgramStatisticsRecord) -> dict[str, Any]:
    """Returns the serializable JSON payload of one statistics record."""

    return asdict(record)


def normalize_operation_name(name: str) -> str:
    """Normalizes one raw operation name onto the shared display-only comparison vocabulary.

    Names outside the shared vocabulary are returned with an `unknown:` prefix so an unmapped name is visible in a
    comparison report rather than silently colliding with a mapped one.

    # Parameters

      - `name`: Raw operation or primitive name as stored in an operation histogram.
    """

    if name in {"add", "add_any"}:
        return "add"
    if name == "mul":
        return "mul"
    if name == "neg":
        return "neg"
    if name == "sin":
        return "sin"
    if name == "cos":
        return "cos"
    if name in {"dot", "matmul", "dot_general", "left_matmul", "right_matmul"}:
        return "matmul"
    if name in {"matrix_transpose", "linear_matrix_transpose", "transpose"}:
        return "transpose"
    if name == "scale":
        return "scale"
    if name in {"const", "constant"}:
        return "const"
    if name in {"shard_map", "linear_shard_map"}:
        return "shard_map"
    return f"unknown:{name}"


def normalize_operation_counts(operation_counts: Mapping[str, int]) -> dict[str, int]:
    """Normalizes one raw operation histogram, aggregating aliases that share a canonical name.

    Two raw names that normalize to the same canonical name have their counts summed, so a histogram containing
    both `add` and `add_any` produces a single `add` entry holding their sum.

    # Parameters

      - `operation_counts`: Histogram keyed by raw operation names.
    """

    normalized: Counter[str] = Counter()
    for name, count in operation_counts.items():
        normalized[normalize_operation_name(name)] += count
    return dict(sorted(normalized.items()))


def is_literal(jax_core: Any, variable: Any) -> bool:
    """Returns whether one JAX jaxpr variable is a literal.

    # Parameters

      - `jax_core`: The `jax.core` module.
      - `variable`: One jaxpr variable or literal.
    """

    literal_type = getattr(jax_core, "Literal", None)
    if literal_type is not None and isinstance(variable, literal_type):
        return True
    return type(variable).__name__ == "Literal" and hasattr(variable, "val")


def unwrap_jaxpr_payload(payload: Any) -> tuple[Any, tuple[Any, ...]]:
    """Returns a `(jaxpr, consts)` pair from a JAX region payload.

    A `ClosedJaxpr` wrapper may be a distinct object around a shared inner jaxpr, so arena deduplication keys on the
    identity of the returned jaxpr rather than the identity of the payload.

    # Parameters

      - `payload`: One `ClosedJaxpr`, `Jaxpr`, or jaxpr-carrying equation parameter.
    """

    if hasattr(payload, "jaxpr") and hasattr(payload, "consts"):
        return payload.jaxpr, tuple(payload.consts)
    if hasattr(payload, "jaxpr"):
        return payload.jaxpr, ()
    return payload, ()


def collect_nested_region_payloads(value: Any, label: str) -> list[tuple[str, Any]]:
    """Collects nested JAX region payloads recursively from one equation parameter.

    Mappings are visited in a total ordering over the string form of their keys and sequences are visited in index
    order, so the resulting region indices are stable across runs and across mixed key types.

    # Parameters

      - `value`: One equation parameter value.
      - `label`: Slot label accumulated so far, starting from the parameter name.
    """

    if value is None:
        return []

    if hasattr(value, "jaxpr") or (hasattr(value, "eqns") and hasattr(value, "invars") and hasattr(value, "outvars")):
        return [(label, value)]

    if isinstance(value, Mapping):
        nested_regions: list[tuple[str, Any]] = []
        for key, nested_value in sorted(value.items(), key=lambda item: str(item[0])):
            nested_regions.extend(collect_nested_region_payloads(nested_value, f"{label}.{key}"))
        return nested_regions

    if isinstance(value, (list, tuple)):
        nested_regions = []
        for index, nested_value in enumerate(value):
            nested_regions.extend(collect_nested_region_payloads(nested_value, f"{label}[{index}]"))
        return nested_regions

    return []


def _collect_region(
    payload: Any,
    jax_core: Any,
    regions: list[RegionStatistics],
    regions_by_identity: dict[int, tuple[int, Any]],
) -> int:
    """Collects one region and its descendants into the arena, returning the region's arena index.

    Descendants are collected before their parent, so the arena mirrors the Rust ordering: descendants precede
    parents and the entry region is the final entry. A jaxpr that is reached more than once yields one arena entry
    and one attachment edge per reference.

    # Parameters

      - `payload`: One `ClosedJaxpr`, `Jaxpr`, or jaxpr-carrying equation parameter.
      - `jax_core`: The `jax.core` module.
      - `regions`: Arena being built, in ascending index order.
      - `regions_by_identity`: Map from unwrapped-jaxpr object identity to its `(arena index, jaxpr)` pair. The
        jaxpr is retained so that its identity cannot be recycled while the arena is being built.
    """

    jaxpr, _consts = unwrap_jaxpr_payload(payload)
    existing = regions_by_identity.get(id(jaxpr))
    if existing is not None:
        return existing[0]

    input_variables = tuple(getattr(jaxpr, "invars", ()))
    constant_variables = tuple(getattr(jaxpr, "constvars", ()))
    output_variables = tuple(getattr(jaxpr, "outvars", ()))
    equations = tuple(getattr(jaxpr, "eqns", ()))

    operation_counts: Counter[str] = Counter()
    attached_regions: list[AttachedRegionStatistics] = []
    depth_by_variable: dict[int, int] = {id(variable): 0 for variable in input_variables}
    depth_by_variable.update({id(variable): 0 for variable in constant_variables})
    literals_by_identity: dict[int, Any] = {}

    for instruction_index, equation in enumerate(equations):
        operation_counts[equation.primitive.name] += 1

        input_depth = 0
        for input_variable in equation.invars:
            if is_literal(jax_core, input_variable):
                literals_by_identity[id(input_variable)] = input_variable
            else:
                input_depth = max(input_depth, depth_by_variable.get(id(input_variable), 0))

        output_depth = input_depth + 1
        for output_variable in equation.outvars:
            depth_by_variable[id(output_variable)] = output_depth

        for parameter_name, parameter_value in sorted(equation.params.items(), key=lambda item: str(item[0])):
            for region_slot, nested_payload in collect_nested_region_payloads(parameter_value, str(parameter_name)):
                region_index = _collect_region(nested_payload, jax_core, regions, regions_by_identity)
                attached_regions.append(
                    AttachedRegionStatistics(
                        instruction_index=instruction_index,
                        operation=equation.primitive.name,
                        region_slot=region_slot,
                        region_role="computation",
                        region_index=region_index,
                    )
                )

    maximum_output_dependency_depth = 0
    for output_variable in output_variables:
        if is_literal(jax_core, output_variable):
            literals_by_identity[id(output_variable)] = output_variable
            output_depth = 0
        else:
            output_depth = depth_by_variable.get(id(output_variable), 0)
        maximum_output_dependency_depth = max(maximum_output_dependency_depth, output_depth)

    regions.append(
        RegionStatistics(
            input_count=len(input_variables),
            output_count=len(output_variables),
            instruction_count=len(equations),
            constant_count=len(constant_variables) + len(literals_by_identity),
            operation_counts=dict(sorted(operation_counts.items())),
            maximum_output_dependency_depth=maximum_output_dependency_depth,
            attached_regions=tuple(attached_regions),
        )
    )
    region_index = len(regions) - 1
    regions_by_identity[id(jaxpr)] = (region_index, jaxpr)
    return region_index


def collect_program_statistics(payload: Any, jax_core: Any) -> ProgramStatistics:
    """Collects Ryft-shaped structural statistics from one closed jaxpr.

    Operation histograms hold raw primitive names; normalization is applied only at comparison and display time.
    Depth is region-local: it is the longest chain of equations feeding a region output, attached regions
    contribute nothing to it, and a region output that is an input variable or a literal has depth zero.

    # Parameters

      - `payload`: One `ClosedJaxpr` or `Jaxpr`.
      - `jax_core`: The `jax.core` module, used to recognize literals.
    """

    regions: list[RegionStatistics] = []
    _collect_region(payload, jax_core, regions, {})
    return ProgramStatistics(regions=tuple(regions))


def python_root() -> Path:
    """Returns the root of the repository's `python` directory."""

    return Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    """Returns the repository root."""

    return python_root().parent


def program_statistics_cases() -> tuple[ProgramStatisticsCase, ...]:
    """Returns the Python-side case registry, imported lazily to keep this module free of JAX at load time."""

    from ryft.jax.program_statistics_cases import program_statistics_cases as cases

    return cases()


def selected_cases(case_ids: Sequence[str]) -> tuple[ProgramStatisticsCase, ...]:
    """Returns the selected registry cases in the requested order, or all cases when no ID is requested.

    # Parameters

      - `case_ids`: Requested case IDs, possibly empty.
    """

    cases = program_statistics_cases()
    if not case_ids:
        return cases

    cases_by_id = {case.case_id: case for case in cases}
    unknown_case_ids = [case_id for case_id in case_ids if case_id not in cases_by_id]
    if unknown_case_ids:
        available_case_ids = ", ".join(case.case_id for case in cases)
        raise ValueError(
            "unknown program statistics case(s): "
            + ", ".join(unknown_case_ids)
            + "; available cases: "
            + available_case_ids
        )
    return tuple(cases_by_id[case_id] for case_id in case_ids)


def rust_statistics_command_prefix() -> list[str]:
    """Returns the cargo command prefix that runs the Rust program statistics binary."""

    return [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "ryft-xla",
        "--features",
        "program-statistics",
        "--bin",
        "program_statistics",
        "--",
    ]


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    """Runs one subprocess and returns its stdout, raising `subprocess.CalledProcessError` on failure."""

    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, env=env)
    return completed.stdout


def python_subprocess_environment(root: Path) -> dict[str, str]:
    """Builds the Python subprocess environment for local package execution."""

    environment = os.environ.copy()
    python_path_entries = [str(root / "python" / "src")]
    existing_python_path = environment.get("PYTHONPATH", "").strip()
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    return environment


def _parse_records(payloads: Sequence[Any], side: str) -> list[ProgramStatisticsRecord]:
    """Parses one side's decoded records, wrapping parser errors with the offending record's case ID.

    # Parameters

      - `payloads`: Decoded JSON record objects.
      - `side`: Producer name used in error messages, either `ryft` or `jax`.
    """

    records: list[ProgramStatisticsRecord] = []
    for index, payload in enumerate(payloads):
        case_id = payload.get("case_id") if isinstance(payload, Mapping) else None
        context = f"case '{case_id}'" if isinstance(case_id, str) else f"record at index {index}"
        try:
            records.append(parse_program_statistics_record(payload))
        except ValueError as error:
            raise ValueError(f"invalid {side} statistics record for {context}: {error}") from error
    return records


def collect_ryft_records(root: Path, case_ids: Sequence[str]) -> list[ProgramStatisticsRecord]:
    """Runs the Rust statistics binary and returns its parsed records.

    # Parameters

      - `root`: Repository root, used as the cargo working directory.
      - `case_ids`: Case IDs to emit.
    """

    command = rust_statistics_command_prefix()
    for case_id in case_ids:
        command.extend(["--case", case_id])
    payloads = json.loads(run_command(command, root))
    return _parse_records(payloads, "ryft")


def collect_rust_case_ids(root: Path) -> list[str]:
    """Returns the Rust-side registry case IDs, in registry order."""

    listing = run_command(rust_statistics_command_prefix() + ["--list"], root)
    return [line.strip() for line in listing.splitlines() if line.strip()]


def collect_jax_records(root: Path, case_ids: Sequence[str]) -> list[ProgramStatisticsRecord]:
    """Collects the JAX-side records, one fresh child process per case.

    A fresh process per case is what lets each case configure `XLA_FLAGS` before JAX is imported, which the
    shard-map cases need for their four-device mesh.

    # Parameters

      - `root`: Repository root, used to build the child process import path.
      - `case_ids`: Case IDs to emit.
    """

    environment = python_subprocess_environment(root)
    project_root = python_root()
    records: list[ProgramStatisticsRecord] = []
    for case_id in case_ids:
        command = ["uv", "run", "python", "-m", "ryft.jax.program_statistics", "--emit-jax-case", case_id]
        payload = json.loads(run_command(command, project_root, environment))
        records.extend(_parse_records([payload], "jax"))
    return records


def validate_record_set(
    records: Sequence[ProgramStatisticsRecord],
    cases: Sequence[ProgramStatisticsCase],
    side: str,
) -> dict[str, ProgramStatisticsRecord]:
    """Validates one side's record set against the selected registry cases and returns it keyed by case ID.

    Exactly one record must be present per selected case: duplicate, missing, and extra records are hard errors,
    and each record's `category` and `surface` must equal its registry entry's values.

    # Parameters

      - `records`: Records emitted by one producer.
      - `cases`: Selected registry cases.
      - `side`: Producer name used in error messages, either `ryft` or `jax`.
    """

    records_by_case_id: dict[str, ProgramStatisticsRecord] = {}
    duplicate_case_ids: list[str] = []
    for record in records:
        if record.case_id in records_by_case_id:
            duplicate_case_ids.append(record.case_id)
        records_by_case_id[record.case_id] = record
    if duplicate_case_ids:
        raise ValueError(f"{side} emitted duplicate records for case(s): {', '.join(sorted(set(duplicate_case_ids)))}")

    selected_case_ids = [case.case_id for case in cases]
    missing_case_ids = [case_id for case_id in selected_case_ids if case_id not in records_by_case_id]
    if missing_case_ids:
        raise ValueError(f"{side} emitted no record for case(s): {', '.join(missing_case_ids)}")

    extra_case_ids = sorted(set(records_by_case_id) - set(selected_case_ids))
    if extra_case_ids:
        raise ValueError(f"{side} emitted unexpected record(s) for case(s): {', '.join(extra_case_ids)}")

    for case in cases:
        record = records_by_case_id[case.case_id]
        if record.category != case.category:
            raise ValueError(
                f"{side} record for case '{case.case_id}' has category '{record.category}' "
                f"but the registry declares '{case.category}'"
            )
        if record.surface != case.surface:
            raise ValueError(
                f"{side} record for case '{case.case_id}' has surface '{record.surface}' "
                f"but the registry declares '{case.surface}'"
            )
    return records_by_case_id


def assert_registries_agree(rust_case_ids: Sequence[str], cases: Sequence[ProgramStatisticsCase]) -> None:
    """Asserts that the Rust and Python registries hold exactly the same case IDs in the same order.

    # Parameters

      - `rust_case_ids`: Case IDs printed by the Rust binary's `--list`.
      - `cases`: The full Python-side registry.
    """

    python_case_ids = [case.case_id for case in cases]
    if list(rust_case_ids) != python_case_ids:
        raise ValueError(
            "the Rust and Python program statistics registries disagree; "
            f"rust: {list(rust_case_ids)}; python: {python_case_ids}"
        )


@dataclass(frozen=True)
class CaseComparison:
    """One case's structural comparison between the Ryft and JAX sides."""

    case: ProgramStatisticsCase
    ryft: ProgramStatisticsRecord
    jax: ProgramStatisticsRecord
    differences: tuple[str, ...]

    def failed(self) -> bool:
        """Returns whether this case is enforced and its enforced metrics disagree."""

        return self.case.comparable and bool(self.differences)


def compare_case(
    case: ProgramStatisticsCase,
    ryft_record: ProgramStatisticsRecord,
    jax_record: ProgramStatisticsRecord,
) -> CaseComparison:
    """Compares one case's entry-region statistics across the two sides.

    The compared metrics are the entry region's input count, output count, instruction count, normalized operation
    histogram, and maximum output dependency depth. Constant counts and attachment-edge vocabularies are reported
    elsewhere but never compared, because the two IRs define them differently.

    # Parameters

      - `case`: Registry entry for the case.
      - `ryft_record`: Record emitted by the Rust binary.
      - `jax_record`: Record emitted by the JAX side.
    """

    ryft_entry = ryft_record.statistics.entry()
    jax_entry = jax_record.statistics.entry()
    differences: list[str] = []
    for metric in ("input_count", "output_count", "instruction_count", "maximum_output_dependency_depth"):
        ryft_value = getattr(ryft_entry, metric)
        jax_value = getattr(jax_entry, metric)
        if ryft_value != jax_value:
            differences.append(f"{metric}: ryft {ryft_value} != jax {jax_value}")

    ryft_operation_counts = normalize_operation_counts(ryft_entry.operation_counts)
    jax_operation_counts = normalize_operation_counts(jax_entry.operation_counts)
    if ryft_operation_counts != jax_operation_counts:
        differences.append(
            f"operation_counts: ryft {format_operation_counts(ryft_operation_counts)} "
            f"!= jax {format_operation_counts(jax_operation_counts)}"
        )
    return CaseComparison(case=case, ryft=ryft_record, jax=jax_record, differences=tuple(differences))


def format_operation_counts(operation_counts: Mapping[str, int]) -> str:
    """Formats one operation histogram as a compact single-line display string."""

    if not operation_counts:
        return "{}"
    return "{" + ", ".join(f"{name}={count}" for name, count in sorted(operation_counts.items())) + "}"


def format_attachments(region: RegionStatistics) -> str:
    """Formats one region's attachment edges as a compact single-line display string."""

    if not region.attached_regions:
        return "none"
    return ", ".join(
        f"#{edge.instruction_index} {edge.label()} -> region {edge.region_index} ({edge.region_role})"
        for edge in region.attached_regions
    )


def format_case_report(comparison: CaseComparison) -> str:
    """Formats one case's structural comparison as a human-readable block."""

    case = comparison.case
    ryft_statistics = comparison.ryft.statistics
    jax_statistics = comparison.jax.statistics
    ryft_entry = ryft_statistics.entry()
    jax_entry = jax_statistics.entry()

    rows: list[str | tuple[str, str, str]] = []

    def add_row(label: str, ryft_value: str, jax_value: str) -> None:
        rows.append((label, ryft_value, jax_value))

    add_row("metric", "ryft", "jax")
    add_row("region_count", str(ryft_statistics.region_count()), str(jax_statistics.region_count()))
    add_row("input_count", str(ryft_entry.input_count), str(jax_entry.input_count))
    add_row("output_count", str(ryft_entry.output_count), str(jax_entry.output_count))
    add_row("instruction_count", str(ryft_entry.instruction_count), str(jax_entry.instruction_count))
    add_row(
        "maximum_output_dependency_depth",
        str(ryft_entry.maximum_output_dependency_depth),
        str(jax_entry.maximum_output_dependency_depth),
    )
    add_row(
        "operation_counts (raw)",
        format_operation_counts(ryft_entry.operation_counts),
        format_operation_counts(jax_entry.operation_counts),
    )
    add_row(
        "operation_counts (normalized)",
        format_operation_counts(normalize_operation_counts(ryft_entry.operation_counts)),
        format_operation_counts(normalize_operation_counts(jax_entry.operation_counts)),
    )
    rows.append("reported only, never compared:")
    add_row("  constant_count", str(ryft_entry.constant_count), str(jax_entry.constant_count))
    rows.append(f"  attachments (ryft): {format_attachments(ryft_entry)}")
    rows.append(f"  attachments (jax):  {format_attachments(jax_entry)}")

    label_width = max(len(row[0]) for row in rows if isinstance(row, tuple)) + 2
    value_width = max(len(row[1]) for row in rows if isinstance(row, tuple)) + 2
    lines = [f"=== {case.case_id} ({case.category} / {case.surface}) ==="]
    for row in rows:
        if isinstance(row, tuple):
            label, ryft_value, jax_value = row
            lines.append(f"  {label:<{label_width}}{ryft_value:<{value_width}}{jax_value}")
        else:
            lines.append(f"  {row}")

    if comparison.differences:
        status = "enforced mismatch" if case.comparable else "informational difference"
        lines.append(f"  {status}:")
        lines.extend(f"    {difference}" for difference in comparison.differences)
    elif case.comparable:
        lines.append("  enforced metrics agree")
    else:
        lines.append("  informational: enforced comparison is disabled for this case")
    return "\n".join(lines)


def compare_program_statistics(root: Path, cases: Sequence[ProgramStatisticsCase]) -> int:
    """Runs both producers for the selected cases, prints the structural comparison, and returns an exit code.

    # Parameters

      - `root`: Repository root.
      - `cases`: Selected registry cases.
    """

    assert_registries_agree(collect_rust_case_ids(root), program_statistics_cases())

    case_ids = [case.case_id for case in cases]
    ryft_records = validate_record_set(collect_ryft_records(root, case_ids), cases, "ryft")
    jax_records = validate_record_set(collect_jax_records(root, case_ids), cases, "jax")

    comparisons = [compare_case(case, ryft_records[case.case_id], jax_records[case.case_id]) for case in cases]
    for comparison in comparisons:
        print(format_case_report(comparison))
        print()

    failed_case_ids = [comparison.case.case_id for comparison in comparisons if comparison.failed()]
    if failed_case_ids:
        print(f"enforced structural parity failed for case(s): {', '.join(failed_case_ids)}", file=sys.stderr)
        return 1

    enforced_count = sum(1 for comparison in comparisons if comparison.case.comparable)
    informational_count = len(comparisons) - enforced_count
    print(f"compared {len(comparisons)} case(s); {enforced_count} enforced, {informational_count} informational")
    return 0


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the command-line arguments of the statistics comparison driver."""

    parser = argparse.ArgumentParser(description="Compare Ryft and JAX structural program statistics.")
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        default=[],
        help="Restrict the comparison to the named case. Repeat this flag to compare multiple cases.",
    )
    parser.add_argument("--list", action="store_true", help="List the available case IDs and exit.")
    parser.add_argument("--emit-jax-case", help=argparse.SUPPRESS)
    return parser.parse_args(list(argv) if argv is not None else None)


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Validates the parsed arguments, rejecting the hidden emission flag alongside the public selection flags."""

    if arguments.emit_jax_case is None:
        return
    if arguments.list:
        raise ValueError("--emit-jax-case cannot be combined with --list")
    if arguments.case_ids:
        raise ValueError("--emit-jax-case cannot be combined with --case")


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the structural program statistics comparison workflow.

    # Parameters

      - `argv`: Command-line arguments, defaulting to `sys.argv[1:]`.
    """

    arguments = parse_arguments(argv)
    try:
        validate_arguments(arguments)

        if arguments.emit_jax_case is not None:
            from ryft.jax.program_statistics_cases import build_jax_case_record

            print(json.dumps(record_payload(build_jax_case_record(arguments.emit_jax_case)), indent=2))
            return 0

        cases = selected_cases(arguments.case_ids)
        if arguments.list:
            for case in cases:
                print(case.case_id)
            return 0

        return compare_program_statistics(repo_root(), cases)
    except subprocess.CalledProcessError as error:
        stderr_text = (error.stderr or "").strip()
        print(stderr_text or str(error), file=sys.stderr)
        return 1
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1


__all__ = [
    "AttachedRegionStatistics",
    "CaseComparison",
    "ProgramStatistics",
    "ProgramStatisticsRecord",
    "RegionStatistics",
    "collect_jax_records",
    "collect_program_statistics",
    "collect_ryft_records",
    "compare_case",
    "compare_program_statistics",
    "main",
    "normalize_operation_counts",
    "normalize_operation_name",
    "parse_program_statistics_record",
    "program_statistics_cases",
    "repo_root",
    "rust_statistics_command_prefix",
    "selected_cases",
    "validate_record_set",
]


if __name__ == "__main__":
    raise SystemExit(main())
