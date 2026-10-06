"""Independent host semantics for optional native CUDA ragged execution workloads."""

from __future__ import annotations

import unittest
from dataclasses import replace

from ryft.jax.differential_testing import SCHEMA, DifferentialObservation
from ryft.jax.differential_testing_cases import DIFFERENTIAL_CASES
from ryft.jax.ragged_collective_testing import (
    ragged_descriptors,
    ragged_inputs,
    ragged_reference,
    validate_ragged_observation,
)


class RaggedCollectiveTestingTest(unittest.TestCase):
    """Pins seeded transfer and adjoint semantics without pretending to execute CUDA on CPU."""

    def test_ragged_descriptors(self) -> None:
        descriptors = ragged_descriptors()
        self.assertEqual(len(descriptors), 8)
        self.assertEqual(len({descriptor["id"] for descriptor in descriptors}), 8)
        self.assertEqual({descriptor["participants"] for descriptor in descriptors}, {1})
        self.assertEqual({descriptor["metadata_type"] for descriptor in descriptors}, {"i32", "u64"})
        self.assertEqual({descriptor["transform"] for descriptor in descriptors}, {"primal", "jvp", "vjp", "batch"})
        registered = {case.case_id for case in DIFFERENTIAL_CASES if case.suite == "cuda-collectives"}
        self.assertEqual(registered, {descriptor["id"] for descriptor in descriptors})

    def test_ragged_inputs(self) -> None:
        import numpy

        for descriptor in ragged_descriptors():
            with self.subTest(case_id=descriptor["id"]):
                operand, seed, *metadata = ragged_inputs(descriptor)
                self.assertEqual(operand.shape, (1, *descriptor["local_shape"]))
                self.assertEqual(seed.shape, (1, *descriptor["output_shape"]))
                self.assertEqual(operand.dtype, numpy.dtype(numpy.float32))
                self.assertEqual(seed.dtype, numpy.dtype(numpy.float32))
                expected = numpy.int32 if descriptor["metadata_type"] == "i32" else numpy.uint64
                self.assertEqual({value.dtype for value in metadata}, {numpy.dtype(expected)})
        batched = ragged_inputs(ragged_descriptors()[-1])
        self.assertEqual(batched[2].shape, (1, 2, 3))

    def test_ragged_reference(self) -> None:
        descriptors = {descriptor["id"]: descriptor for descriptor in ragged_descriptors()}
        expected = {
            "cuda_ragged_seed_holes_i32": {"primal": ((100., 1., 2., 103., 3.),)},
            "cuda_ragged_repeated_reads_i32": {"primal": ((100., 1., 2., 103., 1.),)},
            "cuda_ragged_empty_u64": {"primal": ((100., 101., 102., 103., 104.),)},
            "cuda_ragged_trailing_rows_u64": {
                "primal": ((100., 101., 1., 2., 3., 4., 106., 107., 5., 6.),),
            },
            "cuda_ragged_jvp_repeated_reads_i32": {
                "primal": ((100., 1., 2., 103., 1.),), "tangent": ((1., 1., 2., 1., 1.),),
            },
            "cuda_ragged_vjp_i32": {
                "primal": ((100., 1., 2., 103., 3.),),
                "operand_cotangent": ((2., 3., 5., 0.),),
                "seed_cotangent": ((1., 0., 0., 4., 0.),),
            },
            "cuda_ragged_vjp_trailing_rows_u64": {
                "primal": ((100., 101., 1., 2., 3., 4., 106., 107., 5., 6.),),
                "operand_cotangent": ((3., 4., 5., 6., 9., 10., 0., 0.),),
                "seed_cotangent": ((1., 2., 0., 0., 0., 0., 7., 8., 0., 0.),),
            },
            "cuda_ragged_batch_i32": {
                "primal": ((100., 101., 1., 2., 3., 4., 106., 107., 5., 6.,
                            11., 12., 112., 113., 114., 115., 15., 16., 118., 119.),),
            },
        }
        for case_id, observations in expected.items():
            with self.subTest(case_id=case_id):
                self.assertEqual(ragged_reference(descriptors[case_id]), observations)

    def test_ragged_reference_invalid_metadata(self) -> None:
        descriptor = ragged_descriptors()[0]
        with self.assertRaisesRegex(ValueError, "invalid ragged transfer metadata"):
            ragged_reference({**descriptor, "send_sizes": [-1, 1, 0], "receive_sizes": [-1, 1, 0]})
        with self.assertRaisesRegex(ValueError, "ragged transfer exceeds its data extent"):
            ragged_reference({**descriptor, "input_offsets": [3, 2, 4]})
        with self.assertRaisesRegex(ValueError, "ragged received regions overlap"):
            ragged_reference({**descriptor, "output_offsets": [1, 2, 5]})


    def test_validate_ragged_observation(self) -> None:
        for descriptor in ragged_descriptors():
            if descriptor["transform"] == "vjp":
                observation = DifferentialObservation(SCHEMA, descriptor["id"], ragged_reference(descriptor))
                self.assertEqual(validate_ragged_observation(descriptor, observation), ())
        descriptor = next(
            descriptor for descriptor in ragged_descriptors() if descriptor["id"] == "cuda_ragged_vjp_i32"
        )
        observation = DifferentialObservation(SCHEMA, descriptor["id"], ragged_reference(descriptor))
        incorrect = replace(observation, observations={**observation.observations, "operand_cotangent": ((0.,) * 4,)})
        self.assertEqual(validate_ragged_observation(descriptor, incorrect), (
            "ragged adjoint identity: output inner product 535.0 differs from input inner product 512.0",
        ))
        missing = replace(observation, observations={"primal": observation.observations["primal"]})
        self.assertEqual(validate_ragged_observation(descriptor, missing), (
            "ragged adjoint identity requires all expected participant-local output shapes",
        ))



if __name__ == "__main__":
    unittest.main()
