"""Unit and live coverage for the Ryft/JAX differential-testing harness."""

from __future__ import annotations

import unittest

from ryft.jax.differential_testing import (
    SCHEMA,
    CaseComparison,
    DifferentialObservation,
    StableHloCollective,
    StagingObservation,
    compare_case,
    differential_cases,
    observation_payload,
    parse_observation,
    project_collective_stablehlo,
    repo_root,
    run_comparison,
)


GROUPED_STABLEHLO = "\n".join(
    (
        '%0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, '
        "replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>}>",
        '%1 = "stablehlo.reduce_scatter"(%arg0) <{'
        "replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>, scatter_dimension = 0 : i64}>",
        '%2 = "stablehlo.all_to_all"(%arg0) <{concat_dimension = 0 : i64, '
        "replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>, "
        "split_count = 2 : i64, split_dimension = 0 : i64}>",
    )
)


GROUPED_COLLECTIVES = (
    StableHloCollective("all_gather", ((0, 2), (3, 1)), (("all_gather_dim", 0),)),
    StableHloCollective("reduce_scatter", ((0, 2), (3, 1)), (("scatter_dimension", 0),)),
    StableHloCollective(
        "all_to_all",
        ((0, 2), (3, 1)),
        (("concat_dimension", 0), ("split_count", 2), ("split_dimension", 0)),
    ),
)

MISSING_ALL_FAMILIES = "all_gather, reduce_scatter, all_to_all"


def grouped_observation() -> DifferentialObservation:
    """Returns one minimal exact-parity grouped-collective observation."""

    return DifferentialObservation(
        schema=SCHEMA,
        case_id="grouped_shape_changing_collectives",
        observations={"all_gather": ((0.0, 1.0), (2.0, 3.0))},
        stablehlo=GROUPED_STABLEHLO,
    )


class DifferentialTestingTest(unittest.TestCase):
    """Covers schema validation, StableHLO projection, and relationship-aware comparison."""

    maxDiff = None

    def test_registry(self) -> None:
        self.assertEqual(
            [(case.case_id, case.relationship, bool(case.collectives)) for case in differential_cases()],
            [
                ("grouped_shape_changing_collectives", "parity", True),
                ("pshuffle", "parity", True),
                ("pswapaxes", "parity", True),
                ("data_dependent_prefix_take", "ryft_exceeds_jax", False),
            ],
        )

    def test_observation_schema(self) -> None:
        observation = parse_observation(
            {
                "schema": SCHEMA,
                "case_id": "data_dependent_prefix_take",
                "observations": {"two_matches": [[10, 20]], "zero_matches": [[]]},
                "staging": {"status": "rejected", "category": "concretization"},
            }
        )
        self.assertEqual(
            observation,
            DifferentialObservation(
                schema=SCHEMA,
                case_id="data_dependent_prefix_take",
                observations={"two_matches": ((10.0, 20.0),), "zero_matches": ((),)},
                staging=StagingObservation(status="rejected", category="concretization"),
            ),
        )
        self.assertEqual(
            observation_payload(observation)["staging"],
            {"status": "rejected", "category": "concretization"},
        )
        with self.assertRaisesRegex(ValueError, "unsupported differential observation schema 'old'"):
            parse_observation({"schema": "old"})
        with self.assertRaisesRegex(ValueError, "staging.status.*unsupported value 'unknown'"):
            parse_observation(
                {
                    "schema": SCHEMA,
                    "case_id": "case",
                    "observations": {},
                    "staging": {"status": "unknown"},
                }
            )

    def test_collective_stablehlo_projection(self) -> None:
        self.assertEqual(project_collective_stablehlo(GROUPED_STABLEHLO), GROUPED_COLLECTIVES)
        with self.assertRaisesRegex(ValueError, "stablehlo.all_gather is missing replica/source-target groups"):
            project_collective_stablehlo('"stablehlo.all_gather"(%0) <{all_gather_dim = 0 : i64}>')

    def test_exact_parity_comparison(self) -> None:
        observation = grouped_observation()
        self.assertTrue(compare_case("parity", GROUPED_COLLECTIVES, observation, observation).passed())

        changed = DifferentialObservation(
            schema=SCHEMA,
            case_id=observation.case_id,
            observations={"all_gather": ((0.0, 2.0),)},
            stablehlo=observation.stablehlo,
        )
        comparison = compare_case("parity", GROUPED_COLLECTIVES, observation, changed)
        self.assertEqual(len(comparison.differences), 1)
        self.assertTrue(comparison.differences[0].startswith("observations:"))

    def test_exact_parity_comparison_rejects_differing_groups(self) -> None:
        observation = grouped_observation()
        regrouped = DifferentialObservation(
            schema=SCHEMA,
            case_id=observation.case_id,
            observations=observation.observations,
            stablehlo=GROUPED_STABLEHLO.replace("[[0, 2], [3, 1]]", "[[0, 1], [2, 3]]"),
        )
        comparison = compare_case("parity", GROUPED_COLLECTIVES, observation, regrouped)
        self.assertEqual(len(comparison.differences), 1)
        self.assertTrue(comparison.differences[0].startswith("StableHLO collectives: jax "))
        self.assertIn("!= expected", comparison.differences[0])

    def test_exact_parity_comparison_rejects_empty_projection(self) -> None:
        # A printer or lowering regression that stops emitting recognizable collectives empties both projections
        # symmetrically. The declared expectation is what turns that silent agreement into a named failure.
        empty = DifferentialObservation(
            schema=SCHEMA,
            case_id="grouped_shape_changing_collectives",
            observations={"all_gather": ((0.0, 1.0), (2.0, 3.0))},
            stablehlo="func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {\n  return %arg0 : tensor<4xf32>\n}",
        )
        self.assertEqual(
            compare_case("parity", GROUPED_COLLECTIVES, empty, empty).differences,
            (
                f"StableHLO collectives: ryft module is missing expected collective families {MISSING_ALL_FAMILIES}",
                f"StableHLO collectives: jax module is missing expected collective families {MISSING_ALL_FAMILIES}",
            ),
        )

    def test_ryft_exceeds_jax_comparison(self) -> None:
        observations = {"two_matches": ((10.0, 20.0),), "zero_matches": ((),)}
        ryft = DifferentialObservation(
            schema=SCHEMA,
            case_id="data_dependent_prefix_take",
            observations=observations,
            staging=StagingObservation(status="supported", output_type="f32[count]"),
        )
        jax = DifferentialObservation(
            schema=SCHEMA,
            case_id="data_dependent_prefix_take",
            observations=observations,
            staging=StagingObservation(status="rejected", category="concretization"),
        )
        self.assertTrue(compare_case("ryft_exceeds_jax", (), ryft, jax).passed())

        jax_supported = DifferentialObservation(
            schema=SCHEMA,
            case_id=jax.case_id,
            observations=observations,
            staging=StagingObservation(status="supported", output_type="unexpected"),
        )
        self.assertEqual(
            compare_case("ryft_exceeds_jax", (), ryft, jax_supported).differences,
            ("JAX staging: expected concretization rejection but got StagingObservation(status='supported', "
             "output_type='unexpected', category=None)",),
        )

    def test_live_data_dependent_comparison(self) -> None:
        comparisons = run_comparison(repo_root(), ("data_dependent_prefix_take",))

        self.assertEqual(comparisons, (CaseComparison(case_id="data_dependent_prefix_take", differences=()),))


if __name__ == "__main__":
    unittest.main()
