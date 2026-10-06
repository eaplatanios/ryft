"""Unit and live coverage for the Ryft/JAX differential-testing harness."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ryft.jax import differential_testing
from ryft.jax.differential_testing import (
    SCHEMA,
    CaseComparison,
    DifferentialObservation,
    StableHloCollective,
    StagingObservation,
    compare_case,
    differential_cases,
    main,
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
        cases = differential_cases()
        self.assertEqual(
            [
                (case.case_id, case.relationship, bool(case.collectives), bool(case.stablehlo_patterns))
                for case in cases[:7]
            ],
            [
                ("grouped_shape_changing_collectives", "parity", True, False),
                ("pshuffle", "parity", True, False),
                ("pswapaxes", "parity", True, False),
                ("data_dependent_prefix_take", "ryft_exceeds_jax", False, False),
                ("scaled_dot_and_matmul", "parity", False, True),
                ("dot_product_attention", "parity", False, True),
                ("negative_dynamic_slice", "parity", False, True),
            ],
        )
        self.assertEqual(len({case.case_id for case in cases}), len(cases))
        self.assertTrue(cases[7:])
        self.assertTrue(all(
            case.reference is not None and case.suite in ("collectives", *differential_testing.CUDA_SUITES)
            for case in cases[7:]
        ))

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
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "observations.output.*must be finite"):
                parse_observation(
                    {"schema": SCHEMA, "case_id": "case", "observations": {"output": [[value]]}}
                )

    def test_collective_stablehlo_projection(self) -> None:
        self.assertEqual(project_collective_stablehlo(GROUPED_STABLEHLO), GROUPED_COLLECTIVES)
        with self.assertRaisesRegex(ValueError, "stablehlo.all_gather is missing replica/source-target groups"):
            project_collective_stablehlo('"stablehlo.all_gather"(%0) <{all_gather_dim = 0 : i64}>')

    def test_collective_stablehlo_projection_reductions_and_singleton_groups(self) -> None:
        self.assertEqual(
            project_collective_stablehlo(
                '"stablehlo.all_reduce"(%arg0) <{replica_groups = dense<0> : tensor<1x1xi64>}> ({\n'
                '  stablehlo.add %left, %right : tensor<f32>\n})'
            ),
            (StableHloCollective("all_reduce", ((0,),), ()),),
        )

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

    def test_exact_parity_comparison_rejects_shared_numerical_error(self) -> None:
        observation = grouped_observation()
        reference = {"all_gather": ((9.0, 1.0), (2.0, 3.0))}
        comparison = compare_case("parity", GROUPED_COLLECTIVES, observation, observation, reference=reference)
        self.assertEqual(
            comparison.differences,
            (
                f"reference observations: ryft {observation.observations!r} != expected {reference!r}",
                f"reference observations: jax {observation.observations!r} != expected {reference!r}",
            ),
        )
        self.assertTrue(
            compare_case(
                "parity", GROUPED_COLLECTIVES, observation, observation, reference=observation.observations,
            ).passed()
        )

    def test_exact_parity_comparison_reference_checks_participant_count_and_output_names(self) -> None:
        observation = grouped_observation()
        for reference in (
            {"all_gather": ((0.0, 1.0),)},
            {"all_gather": ((0.0,), (2.0,))},
            {"different_output": observation.observations["all_gather"]},
        ):
            comparison = compare_case("parity", GROUPED_COLLECTIVES, observation, observation, reference=reference)
            self.assertEqual(len(comparison.differences), 2)
            self.assertTrue(all(
                difference.startswith("reference observations:") for difference in comparison.differences
            ))

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

    def test_exact_parity_comparison_checks_semantic_patterns(self) -> None:
        observation = DifferentialObservation(
            schema=SCHEMA,
            case_id="semantic",
            observations={"output": ((1.0,),)},
            stablehlo='"stablehlo.composite"() {name = "xla.scaled_dot"}',
        )
        self.assertTrue(compare_case("parity", (), observation, observation, ("xla.scaled_dot",)).passed())
        self.assertEqual(
            compare_case("parity", (), observation, observation, ("dimension_numbers",)).differences,
            (
                "StableHLO: Ryft module is missing semantic patterns ('dimension_numbers',)",
                "StableHLO: JAX module is missing semantic patterns ('dimension_numbers',)",
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

    def test_main_collective_suite_selection(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(("--suite", "collectives", "--list")), 0)
        expected = [case.case_id for case in differential_cases() if case.suite == "collectives"]
        self.assertEqual(output.getvalue().splitlines(), expected)
        self.assertNotIn("data_dependent_prefix_take", expected)
        self.assertIn("grouped_shape_changing_collectives", expected)

        errors = StringIO()
        with redirect_stderr(errors):
            self.assertEqual(main(("--suite", "collectives", "--case", "scaled_dot_and_matmul", "--list")), 2)
        self.assertEqual(errors.getvalue(), "case 'scaled_dot_and_matmul' is not in suite 'collectives'\n")

    def test_main_cuda_suite_selection(self) -> None:
        cuda_ids = [case.case_id for case in differential_cases() if case.suite == "cuda-collectives"]
        self.assertTrue(cuda_ids)
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(("--suite", "cuda-collectives", "--list")), 0)
        self.assertEqual(output.getvalue().splitlines(), cuda_ids)
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(("--list",)), 0)
        self.assertTrue(set(cuda_ids).isdisjoint(output.getvalue().splitlines()))
        errors = StringIO()
        with redirect_stderr(errors):
            self.assertEqual(main(("--case", cuda_ids[0], "--case", "pshuffle", "--list")), 2)
        self.assertEqual(errors.getvalue(), "CPU and CUDA cases must run in separate invocations\n")

    def test_collect_ryft_observations_cuda_feature(self) -> None:
        case_id = next(case.case_id for case in differential_cases() if case.suite == "cuda-collectives")
        with patch("ryft.jax.differential_testing._run_emitter", return_value="[]") as emitter:
            self.assertEqual(differential_testing.collect_ryft_observations(repo_root(), (case_id,)), ())
        self.assertEqual(emitter.call_args.args[0], [
            "cargo", "run", "--quiet", "-p", "ryft-xla", "--features", "differential-testing,cuda-13",
            "--bin", "differential_testing", "--", "--case", case_id,
        ])

    def test_main_subprocess_configuration(self) -> None:
        with patch("ryft.jax.differential_testing.run_comparison", return_value=()) as comparison:
            self.assertEqual(
                main(("--case", "pshuffle", "--timeout", "60", "--ryft-binary", "/tmp/ryft-emitter")), 0,
            )
        comparison.assert_called_once_with(
            repo_root(), ("pshuffle",), timeout=60, binary=Path("/tmp/ryft-emitter"), log_directory=None,
        )

        errors = StringIO()
        with redirect_stderr(errors):
            self.assertEqual(main(("--timeout", "0", "--list")), 2)
        self.assertEqual(errors.getvalue(), "--timeout must be positive\n")

    def test_main_subprocess_failure(self) -> None:
        errors = StringIO()
        with (
            patch("ryft.jax.differential_testing.run_comparison", side_effect=RuntimeError("execution timed out")),
            redirect_stderr(errors),
        ):
            self.assertEqual(main(("--case", "pshuffle")), 1)
        self.assertEqual(errors.getvalue(), "execution timed out\n")

    @unittest.skipUnless(os.name == "posix", "process-group cancellation requires POSIX")
    def test_emitter_timeout_terminates_child_processes(self) -> None:
        # The child inherits the output pipe and blocks. After cancellation, communicate() can finish only when
        # the child has also exited; killing just the launcher would leave this test blocked on that pipe.
        program = (
            "import subprocess, sys, threading; "
            "subprocess.Popen([sys.executable, '-c', 'import threading; threading.Event().wait()']); "
            "threading.Event().wait()"
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            differential_testing._run_emitter((sys.executable, "-c", program), repo_root(), timeout=1)


if __name__ == "__main__":
    unittest.main()
