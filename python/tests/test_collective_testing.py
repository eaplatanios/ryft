"""Independent host semantics and simulated-device integration for the collective matrix."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from dataclasses import replace

from ryft.jax.collective_testing import (
    collective_descriptors,
    collective_reference,
    participant_groups,
    validate_collective_observation,
    validate_collective_shapes,
)
from ryft.jax.differential_testing import (
    SCHEMA, DifferentialObservation, parse_observation, project_collective_stablehlo,
)
from ryft.jax.differential_testing_cases import DIFFERENTIAL_CASES


class CollectiveTestingTest(unittest.TestCase):
    """Pins host-reference behavior independently from either backend's implementations."""

    def test_collective_descriptors(self) -> None:
        descriptors = collective_descriptors()
        self.assertEqual(len(descriptors), 51)
        self.assertEqual(len({descriptor["id"] for descriptor in descriptors}), len(descriptors))
        self.assertEqual({descriptor["participants"] for descriptor in descriptors}, {1, 2, 4, 8})
        self.assertEqual(
            {descriptor.get("transform", "primal") for descriptor in descriptors}, {"primal", "batch", "jvp", "vjp"}
        )

    def test_participant_groups(self) -> None:
        descriptor = dict(participants=4, mesh_shape=[2, 2], axis_name="x")
        self.assertEqual(participant_groups(descriptor), ((0, 2), (1, 3)))
        self.assertEqual(participant_groups({**descriptor, "axis_name": "y"}), ((0, 1), (2, 3)))
        self.assertEqual(participant_groups(dict(participants=4, groups=[[0, 2], [3, 1]])), ((0, 2), (3, 1)))

    def test_collective_reference_gather_order(self) -> None:
        descriptor = dict(operation="all_gather", participants=4, local_shape=[2], groups=[[0, 2], [3, 1]], tiled=True)
        self.assertEqual(collective_reference(descriptor), {
            "primal": ((1., 2., 5., 6.), (7., 8., 3., 4.), (1., 2., 5., 6.), (7., 8., 3., 4.))
        })

    def test_collective_reference_scatter_order(self) -> None:
        descriptor = dict(
            operation="sum_scatter", participants=4, local_shape=[2], groups=[[0, 2], [3, 1]], tiled=False,
        )
        self.assertEqual(collective_reference(descriptor), {"primal": ((6.,), (12.,), (8.,), (10.,))})

    def test_collective_reference_all_to_all(self) -> None:
        descriptor = dict(
            operation="all_to_all", participants=2, local_shape=[2, 2],
            split_axis=0, concatenation_axis=1, tiled=True,
        )
        self.assertEqual(collective_reference(descriptor), {"primal": ((1., 2., 5., 6.), (3., 4., 7., 8.))})

    def test_collective_reference_reduction(self) -> None:
        descriptor = dict(operation="reduce", participants=2, local_shape=[2], reduction="sum")
        self.assertEqual(collective_reference(descriptor), {"primal": ((4., 6.), (4., 6.))})
        self.assertEqual(collective_reference({**descriptor, "reduction": "mean"}), {"primal": ((2., 3.), (2., 3.))})
        self.assertEqual(collective_reference({**descriptor, "reduction": "product"}), {"primal": ((3., 8.), (3., 8.))})

    def test_collective_reference_permute_missing_destination(self) -> None:
        descriptor = dict(operation="permute", participants=4, local_shape=[1], pairs=[[0, 2], [3, 1]])
        self.assertEqual(collective_reference(descriptor), {"primal": ((0.,), (4.,), (1.,), (0.,))})

    def test_collective_reference_axis_index(self) -> None:
        descriptor = dict(operation="axis_index", participants=4, local_shape=[2], mesh_shape=[2, 2], axis_name="x")
        self.assertEqual(collective_reference(descriptor), {"primal": ((0., 0.), (0., 0.), (1., 1.), (1., 1.))})

    def test_collective_reference_batch(self) -> None:
        descriptor = dict(operation="all_gather", participants=2, local_shape=[2, 1], tiled=True, transform="batch")
        self.assertEqual(collective_reference(descriptor), {"primal": ((1., 3., 2., 4.), (1., 3., 2., 4.))})

    def test_collective_reference_jvp_vjp(self) -> None:
        descriptor = dict(operation="permute", participants=4, local_shape=[1], pairs=[[0, 2], [3, 1]])
        self.assertEqual(collective_reference({**descriptor, "transform": "jvp"}), {
            "primal": ((0.,), (4.,), (1.,), (0.,)), "tangent": ((0.,), (4.,), (1.,), (0.,))
        })
        self.assertEqual(collective_reference({**descriptor, "transform": "vjp"}), {
            "primal": ((0.,), (4.,), (1.,), (0.,)), "cotangent": ((3.,), (0.,), (0.,), (2.,))
        })

    def test_validate_collective_shapes(self) -> None:
        import numpy

        descriptor = dict(id="shape_test", operation="all_gather", participants=2, local_shape=[2], tiled=False)
        validate_collective_shapes(descriptor, {"primal": numpy.zeros((2, 2, 2), dtype=numpy.float32)})
        with self.assertRaisesRegex(
            ValueError,
            "collective 'shape_test' primal shape \\(2, 4\\) differs from reference \\(2, 2, 2\\)",
        ):
            validate_collective_shapes(descriptor, {"primal": numpy.zeros((2, 4), dtype=numpy.float32)})

    def test_validate_collective_shapes_dtype(self) -> None:
        import numpy

        descriptor = dict(id="dtype_test", operation="all_gather", participants=2, local_shape=[2], tiled=False)
        with self.assertRaisesRegex(
            ValueError,
            "collective 'dtype_test' primal dtype float64 differs from reference float32",
        ):
            validate_collective_shapes(descriptor, {"primal": numpy.zeros((2, 2, 2), dtype=numpy.float64)})

    def test_validate_collective_observation(self) -> None:
        descriptor = dict(operation="permute", participants=4, local_shape=[1], pairs=[[0, 2], [3, 1]], transform="vjp")
        observation = DifferentialObservation(SCHEMA, "test", collective_reference(descriptor))
        self.assertEqual(validate_collective_observation(descriptor, observation), ())
        incorrect = replace(observation, observations={**observation.observations, "cotangent": ((0.,),) * 4})
        self.assertEqual(validate_collective_observation(descriptor, incorrect), (
            "adjoint identity: output inner product 11.0 differs from input inner product 0.0",
        ))
        missing = replace(observation, observations={"primal": observation.observations["primal"]})
        self.assertEqual(validate_collective_observation(descriptor, missing), (
            "adjoint identity requires the expected participant-local cotangent shapes",
        ))

    def test_build_jax_observations_collective_matrix(self) -> None:
        cases = {case.case_id: case for case in DIFFERENTIAL_CASES if case.suite == "collectives"}
        completed = subprocess.run(
            [sys.executable, "-m", "ryft.jax.differential_testing", "--emit-jax", "--suite", "collectives"],
            check=True, capture_output=True, text=True, timeout=300,
        )
        observations = tuple(parse_observation(record) for record in json.loads(completed.stdout))
        self.assertEqual(len(observations), 54)
        for observation in observations:
            with self.subTest(case_id=observation.case_id):
                case = cases[observation.case_id]
                self.assertEqual(observation.observations, case.reference())
                if case.collectives:
                    self.assertEqual(project_collective_stablehlo(observation.stablehlo), case.collectives)
                if case.validate is not None:
                    self.assertEqual(case.validate(observation), ())


if __name__ == "__main__":
    unittest.main()
