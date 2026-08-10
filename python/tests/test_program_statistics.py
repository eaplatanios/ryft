"""Unit and live tests for the Ryft/JAX structural program statistics workflow."""

from __future__ import annotations

import contextlib
import copy
import io
import unittest
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import core as jax_core

from ryft.jax.program_statistics import (
    AttachedRegionStatistics,
    ProgramStatistics,
    ProgramStatisticsRecord,
    RegionStatistics,
    collect_jax_records,
    collect_program_statistics,
    collect_rust_case_ids,
    compare_case,
    main,
    normalize_operation_counts,
    parse_arguments,
    parse_program_statistics_record,
    program_statistics_cases,
    repo_root,
    selected_cases,
    validate_arguments,
    validate_record_set,
)
from ryft.jax.program_statistics_cases import ProgramStatisticsCase, program_statistics_case_by_id


EXPECTED_CASE_IDS = (
    "scalar_bilinear_sin_jit",
    "scalar_bilinear_sin_vjp_pullback",
    "scalar_quartic_plus_sin_grad",
    "scalar_quartic_plus_sin_value_and_gradient",
    "scalar_quartic_plus_sin_linearize_pushforward",
    "shard_map_basic",
    "shard_map_matmul",
    "nested_shard_map",
)

SAMPLE_RECORD_PAYLOAD: dict[str, Any] = {
    "case_id": "shard_map_basic",
    "category": "xla",
    "surface": "program",
    "statistics": {
        "regions": [
            {
                "input_count": 1,
                "output_count": 1,
                "constant_count": 0,
                "instruction_count": 1,
                "operation_counts": {"sin": 1},
                "maximum_output_dependency_depth": 1,
                "attached_regions": [],
            },
            {
                "input_count": 1,
                "output_count": 1,
                "constant_count": 0,
                "instruction_count": 1,
                "operation_counts": {"shard_map": 1},
                "maximum_output_dependency_depth": 1,
                "attached_regions": [
                    {
                        "instruction_index": 0,
                        "operation": "shard_map",
                        "region_slot": "body",
                        "region_role": "computation",
                        "region_index": 0,
                    }
                ],
            },
        ]
    },
}


@dataclass
class StandInPrimitive:
    """Stand-in for a JAX primitive, exposing only the attribute the collector reads."""

    name: str


@dataclass
class StandInVariable:
    """Stand-in for a jaxpr variable; only its object identity matters to the collector."""

    label: str


@dataclass
class StandInEquation:
    """Stand-in for a jaxpr equation, exposing the attribute surface the collector reads."""

    primitive: StandInPrimitive
    invars: list[StandInVariable]
    outvars: list[StandInVariable]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StandInJaxpr:
    """Stand-in for a jaxpr, exposing the attribute surface the collector reads."""

    invars: list[StandInVariable]
    outvars: list[StandInVariable]
    eqns: list[StandInEquation]
    constvars: list[StandInVariable] = field(default_factory=list)


@dataclass
class StandInClosedJaxpr:
    """Stand-in for a closed jaxpr wrapper around a possibly shared inner jaxpr."""

    jaxpr: StandInJaxpr
    consts: tuple[Any, ...] = ()


def empty_statistics() -> ProgramStatistics:
    """Returns a minimal single-region statistics value for registry and validation fixtures."""

    return ProgramStatistics(
        regions=(
            RegionStatistics(
                input_count=1,
                output_count=1,
                constant_count=0,
                instruction_count=1,
                operation_counts={"sin": 1},
                maximum_output_dependency_depth=1,
                attached_regions=(),
            ),
        )
    )


def make_case(case_id: str, category: str = "scalar", surface: str = "jit") -> ProgramStatisticsCase:
    """Builds one registry case fixture."""

    return ProgramStatisticsCase(
        case_id=case_id,
        category=category,
        surface=surface,
        comparable=False,
        build=empty_statistics,
    )


def make_record(case_id: str, category: str = "scalar", surface: str = "jit") -> ProgramStatisticsRecord:
    """Builds one statistics record fixture."""

    return ProgramStatisticsRecord(
        case_id=case_id,
        category=category,
        surface=surface,
        statistics=empty_statistics(),
    )


class JaxprCollectorTest(unittest.TestCase):
    """Covers the JAX-side structural statistics collector."""

    maxDiff = None

    def test_collector_counts_operations_constants_and_depth(self) -> None:
        statistics = collect_program_statistics(jax.make_jaxpr(lambda x: x * x + 1.0)(2.0), jax_core)

        self.assertEqual(statistics.region_count(), 1)
        self.assertEqual(statistics.entry_region_index(), 0)
        self.assertEqual(
            statistics.entry(),
            RegionStatistics(
                input_count=1,
                output_count=1,
                constant_count=1,
                instruction_count=2,
                operation_counts={"add": 1, "mul": 1},
                maximum_output_dependency_depth=2,
                attached_regions=(),
            ),
        )

    def test_collector_treats_literal_outputs_as_constants_at_depth_zero(self) -> None:
        statistics = collect_program_statistics(jax.make_jaxpr(lambda x: 1.0)(2.0), jax_core)

        entry = statistics.entry()
        self.assertEqual(entry.instruction_count, 0)
        self.assertEqual(entry.constant_count, 1)
        self.assertEqual(entry.maximum_output_dependency_depth, 0)

    def test_collector_orders_nested_regions_with_descendants_before_the_entry(self) -> None:
        closed_jaxpr = jax.make_jaxpr(lambda x: jax.lax.cond(x > 0.0, jnp.sin, jnp.cos, x))(1.0)

        statistics = collect_program_statistics(closed_jaxpr, jax_core)

        self.assertEqual(statistics.region_count(), 3)
        self.assertEqual(statistics.entry_region_index(), 2)
        self.assertEqual(statistics.regions[0].operation_counts, {"cos": 1})
        self.assertEqual(statistics.regions[1].operation_counts, {"sin": 1})
        entry = statistics.entry()
        self.assertEqual(
            entry.attached_regions,
            (
                AttachedRegionStatistics(
                    instruction_index=2,
                    operation="cond",
                    region_slot="branches[0]",
                    region_role="computation",
                    region_index=0,
                ),
                AttachedRegionStatistics(
                    instruction_index=2,
                    operation="cond",
                    region_slot="branches[1]",
                    region_role="computation",
                    region_index=1,
                ),
            ),
        )
        self.assertEqual(entry.attached_regions[0].label(), "cond.branches[0]")

    def test_collector_deduplicates_shared_regions_by_unwrapped_jaxpr_identity(self) -> None:
        shared_input = StandInVariable("shared_input")
        shared_output = StandInVariable("shared_output")
        shared_jaxpr = StandInJaxpr(
            invars=[shared_input],
            outvars=[shared_output],
            eqns=[
                StandInEquation(
                    primitive=StandInPrimitive("sin"),
                    invars=[shared_input],
                    outvars=[shared_output],
                )
            ],
        )
        entry_input = StandInVariable("entry_input")
        first_output = StandInVariable("first_output")
        second_output = StandInVariable("second_output")
        entry_jaxpr = StandInJaxpr(
            invars=[entry_input],
            outvars=[second_output],
            eqns=[
                StandInEquation(
                    primitive=StandInPrimitive("call"),
                    invars=[entry_input],
                    outvars=[first_output],
                    # Two distinct closed-jaxpr wrappers around one shared inner jaxpr.
                    params={"jaxpr": StandInClosedJaxpr(shared_jaxpr)},
                ),
                StandInEquation(
                    primitive=StandInPrimitive("call"),
                    invars=[first_output],
                    outvars=[second_output],
                    params={"jaxpr": StandInClosedJaxpr(shared_jaxpr)},
                ),
            ],
        )

        statistics = collect_program_statistics(entry_jaxpr, jax_core)

        self.assertEqual(statistics.region_count(), 2)
        self.assertEqual(statistics.regions[0].operation_counts, {"sin": 1})
        entry = statistics.entry()
        self.assertEqual([edge.region_index for edge in entry.attached_regions], [0, 0])
        self.assertEqual([edge.instruction_index for edge in entry.attached_regions], [0, 1])
        self.assertEqual(entry.maximum_output_dependency_depth, 2)
        self.assertEqual(statistics.total_instruction_count(), 3)
        self.assertEqual(statistics.total_operation_counts(), {"call": 2, "sin": 1})

    def test_collector_visits_mapping_parameters_in_a_total_key_ordering(self) -> None:
        def make_leaf(operation: str) -> StandInJaxpr:
            leaf_input = StandInVariable(f"{operation}_input")
            leaf_output = StandInVariable(f"{operation}_output")
            return StandInJaxpr(
                invars=[leaf_input],
                outvars=[leaf_output],
                eqns=[
                    StandInEquation(
                        primitive=StandInPrimitive(operation),
                        invars=[leaf_input],
                        outvars=[leaf_output],
                    )
                ],
            )

        entry_input = StandInVariable("entry_input")
        entry_output = StandInVariable("entry_output")
        entry_jaxpr = StandInJaxpr(
            invars=[entry_input],
            outvars=[entry_output],
            eqns=[
                StandInEquation(
                    primitive=StandInPrimitive("switch"),
                    invars=[entry_input],
                    outvars=[entry_output],
                    # Mixed key types must not raise and must sort by their string form.
                    params={"branches": {2: make_leaf("cos"), "alpha": make_leaf("sin")}},
                )
            ],
        )

        statistics = collect_program_statistics(entry_jaxpr, jax_core)

        self.assertEqual(
            [edge.region_slot for edge in statistics.entry().attached_regions],
            ["branches.2", "branches.alpha"],
        )
        self.assertEqual(statistics.regions[0].operation_counts, {"cos": 1})
        self.assertEqual(statistics.regions[1].operation_counts, {"sin": 1})


class OperationNormalizationTest(unittest.TestCase):
    """Covers the display-only operation-name normalization vocabulary."""

    def test_normalization_aggregates_alias_counts(self) -> None:
        normalized = normalize_operation_counts({"add": 2, "add_any": 3, "sin": 1})

        self.assertEqual(normalized, {"add": 5, "sin": 1})

    def test_normalization_marks_names_outside_the_shared_vocabulary(self) -> None:
        self.assertEqual(normalize_operation_counts({"pvary": 1}), {"unknown:pvary": 1})

    def test_comparable_case_rejects_matching_unmapped_operation_names(self) -> None:
        case = ProgramStatisticsCase(
            case_id="unmapped",
            category="scalar",
            surface="jit",
            comparable=True,
            build=empty_statistics,
        )
        statistics = ProgramStatistics(
            regions=(
                RegionStatistics(
                    input_count=1,
                    output_count=1,
                    constant_count=0,
                    instruction_count=1,
                    operation_counts={"unmapped_operation": 1},
                    maximum_output_dependency_depth=1,
                    attached_regions=(),
                ),
            )
        )
        record = ProgramStatisticsRecord(
            case_id=case.case_id,
            category=case.category,
            surface=case.surface,
            statistics=statistics,
        )

        comparison = compare_case(case, record, record)

        self.assertTrue(comparison.failed())
        self.assertEqual(len(comparison.differences), 1)
        self.assertIn("unknown:unmapped_operation", comparison.differences[0])


class RecordParserTest(unittest.TestCase):
    """Covers the strict-but-forward-compatible statistics record parser."""

    maxDiff = None

    def test_parser_round_trips_one_full_record(self) -> None:
        record = parse_program_statistics_record(copy.deepcopy(SAMPLE_RECORD_PAYLOAD))

        self.assertEqual(record.case_id, "shard_map_basic")
        self.assertEqual(record.category, "xla")
        self.assertEqual(record.surface, "program")
        self.assertEqual(record.statistics.region_count(), 2)
        self.assertEqual(record.statistics.entry().operation_counts, {"shard_map": 1})
        self.assertEqual(
            record.statistics.entry().attached_regions,
            (
                AttachedRegionStatistics(
                    instruction_index=0,
                    operation="shard_map",
                    region_slot="body",
                    region_role="computation",
                    region_index=0,
                ),
            ),
        )

    def test_parser_reports_the_full_path_of_a_missing_nested_field(self) -> None:
        payload = copy.deepcopy(SAMPLE_RECORD_PAYLOAD)
        del payload["statistics"]["regions"][0]["instruction_count"]

        with self.assertRaises(ValueError) as raised:
            parse_program_statistics_record(payload)

        self.assertIn("statistics.regions[0].instruction_count", str(raised.exception))

    def test_parser_reports_the_full_path_of_a_wrongly_typed_nested_field(self) -> None:
        payload = copy.deepcopy(SAMPLE_RECORD_PAYLOAD)
        payload["statistics"]["regions"][1]["attached_regions"][0]["region_index"] = "zero"

        with self.assertRaises(ValueError) as raised:
            parse_program_statistics_record(payload)

        self.assertIn("statistics.regions[1].attached_regions[0].region_index", str(raised.exception))

    def test_parser_validates_the_record_level_fields(self) -> None:
        missing_case_id = copy.deepcopy(SAMPLE_RECORD_PAYLOAD)
        del missing_case_id["case_id"]
        with self.assertRaises(ValueError) as missing_raised:
            parse_program_statistics_record(missing_case_id)
        self.assertIn("case_id", str(missing_raised.exception))

        wrongly_typed_surface = copy.deepcopy(SAMPLE_RECORD_PAYLOAD)
        wrongly_typed_surface["surface"] = 7
        with self.assertRaises(ValueError) as typed_raised:
            parse_program_statistics_record(wrongly_typed_surface)
        self.assertIn("surface", str(typed_raised.exception))

        missing_statistics = copy.deepcopy(SAMPLE_RECORD_PAYLOAD)
        del missing_statistics["statistics"]
        with self.assertRaises(ValueError) as statistics_raised:
            parse_program_statistics_record(missing_statistics)
        self.assertIn("statistics", str(statistics_raised.exception))

    def test_parser_ignores_unknown_record_and_region_fields(self) -> None:
        payload = copy.deepcopy(SAMPLE_RECORD_PAYLOAD)
        payload["future_record_field"] = {"anything": True}
        payload["statistics"]["future_statistics_field"] = 1
        payload["statistics"]["regions"][0]["future_region_field"] = [1, 2, 3]
        payload["statistics"]["regions"][1]["attached_regions"][0]["future_edge_field"] = "value"

        record = parse_program_statistics_record(payload)

        self.assertEqual(record, parse_program_statistics_record(copy.deepcopy(SAMPLE_RECORD_PAYLOAD)))


class CaseRegistryTest(unittest.TestCase):
    """Covers the Python-side case registry."""

    def test_registry_holds_the_expected_case_ids_in_order(self) -> None:
        self.assertEqual(tuple(case.case_id for case in program_statistics_cases()), EXPECTED_CASE_IDS)

    def test_registry_case_ids_are_unique(self) -> None:
        case_ids = [case.case_id for case in program_statistics_cases()]

        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_registry_lookup_reports_unknown_case_ids(self) -> None:
        with self.assertRaises(ValueError) as raised:
            program_statistics_case_by_id("missing_case")

        self.assertIn("missing_case", str(raised.exception))


class RecordSetValidationTest(unittest.TestCase):
    """Covers record-set validation against the selected registry cases."""

    def test_validation_accepts_exactly_one_record_per_case(self) -> None:
        cases = (make_case("first"), make_case("second", surface="grad"))
        records = [make_record("second", surface="grad"), make_record("first")]

        records_by_case_id = validate_record_set(records, cases, "ryft")

        self.assertEqual(sorted(records_by_case_id), ["first", "second"])

    def test_validation_rejects_duplicate_records(self) -> None:
        cases = (make_case("first"),)

        with self.assertRaises(ValueError) as raised:
            validate_record_set([make_record("first"), make_record("first")], cases, "ryft")

        self.assertIn("duplicate", str(raised.exception))
        self.assertIn("first", str(raised.exception))

    def test_validation_rejects_missing_records(self) -> None:
        cases = (make_case("first"), make_case("second"))

        with self.assertRaises(ValueError) as raised:
            validate_record_set([make_record("first")], cases, "jax")

        self.assertIn("second", str(raised.exception))

    def test_validation_rejects_extra_records(self) -> None:
        cases = (make_case("first"),)

        with self.assertRaises(ValueError) as raised:
            validate_record_set([make_record("first"), make_record("unselected")], cases, "jax")

        self.assertIn("unselected", str(raised.exception))

    def test_validation_rejects_a_category_or_surface_disagreement(self) -> None:
        cases = (make_case("first", category="scalar", surface="jit"),)

        with self.assertRaises(ValueError) as category_raised:
            validate_record_set([make_record("first", category="xla")], cases, "ryft")
        self.assertIn("category", str(category_raised.exception))

        with self.assertRaises(ValueError) as surface_raised:
            validate_record_set([make_record("first", surface="grad")], cases, "ryft")
        self.assertIn("surface", str(surface_raised.exception))


class CommandLineTest(unittest.TestCase):
    """Covers argument parsing and case selection without running any subprocess."""

    def test_list_prints_exactly_the_registry_case_ids(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main(["--list"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(tuple(output.getvalue().split()), EXPECTED_CASE_IDS)

    def test_repeated_case_selection_preserves_the_requested_order(self) -> None:
        cases = selected_cases(["nested_shard_map", "scalar_bilinear_sin_jit"])

        self.assertEqual([case.case_id for case in cases], ["nested_shard_map", "scalar_bilinear_sin_jit"])

    def test_duplicate_case_selection_is_a_hard_error_naming_the_case(self) -> None:
        with self.assertRaises(ValueError) as raised:
            selected_cases(["scalar_bilinear_sin_jit", "scalar_bilinear_sin_jit"])

        self.assertIn("requested more than once", str(raised.exception))
        self.assertIn("scalar_bilinear_sin_jit", str(raised.exception))

    def test_unknown_case_selection_is_a_hard_error_naming_the_case(self) -> None:
        with self.assertRaises(ValueError) as raised:
            selected_cases(["missing_case"])

        self.assertIn("missing_case", str(raised.exception))

    def test_unknown_case_selection_exits_non_zero(self) -> None:
        error_output = io.StringIO()

        with contextlib.redirect_stderr(error_output):
            exit_code = main(["--case", "missing_case"])

        self.assertEqual(exit_code, 1)
        self.assertIn("missing_case", error_output.getvalue())

    def test_emit_jax_case_accepts_exactly_one_case_id(self) -> None:
        self.assertEqual(parse_arguments(["--emit-jax-case", "shard_map_basic"]).emit_jax_case, "shard_map_basic")
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parse_arguments(["--emit-jax-case"])

    def test_emit_jax_case_is_rejected_alongside_the_public_selection_flags(self) -> None:
        with self.assertRaises(ValueError):
            validate_arguments(parse_arguments(["--emit-jax-case", "shard_map_basic", "--list"]))
        with self.assertRaises(ValueError):
            validate_arguments(parse_arguments(["--emit-jax-case", "shard_map_basic", "--case", "shard_map_basic"]))


class LiveProgramStatisticsTest(unittest.TestCase):
    """Covers the live workflow; a failing subprocess is a test failure, never a skip."""

    maxDiff = None

    def test_rust_registry_matches_the_python_registry(self) -> None:
        self.assertEqual(tuple(collect_rust_case_ids(repo_root())), EXPECTED_CASE_IDS)

    def test_hidden_jax_emission_produces_one_matching_record(self) -> None:
        records = collect_jax_records(repo_root(), ["scalar_bilinear_sin_jit"])

        self.assertEqual(len(records), 1)
        case = program_statistics_case_by_id("scalar_bilinear_sin_jit")
        self.assertEqual(records[0].case_id, case.case_id)
        self.assertEqual(records[0].category, case.category)
        self.assertEqual(records[0].surface, case.surface)
        self.assertGreaterEqual(records[0].statistics.region_count(), 1)

    def test_one_scalar_case_compares_end_to_end(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main(["--case", "scalar_bilinear_sin_jit"])

        self.assertEqual(exit_code, 0, output.getvalue())
        self.assertIn("scalar_bilinear_sin_jit", output.getvalue())


if __name__ == "__main__":
    unittest.main()
