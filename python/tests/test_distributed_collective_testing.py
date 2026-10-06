"""Host references for explicitly selected two-process CUDA correctness experiments."""

from __future__ import annotations

import unittest

from ryft.jax.differential_testing_cases import DIFFERENTIAL_CASES
from ryft.jax.distributed_collective_testing import (
    distributed_contract,
    distributed_descriptors,
    distributed_inputs,
    distributed_reference,
)


class DistributedCollectiveTestingTest(unittest.TestCase):
    """Pins cross-rank data movement independently from either backend's communication implementation."""

    def test_distributed_descriptors(self) -> None:
        descriptors = distributed_descriptors()
        self.assertEqual(len(descriptors), 7)
        self.assertEqual({descriptor["participants"] for descriptor in descriptors}, {2})
        self.assertEqual(len({descriptor["id"] for descriptor in descriptors}), 7)
        self.assertEqual(
            {descriptor["operation"] for descriptor in descriptors},
            {"sum", "all_gather", "all_to_all", "ragged_all_to_all"},
        )
        registered = {case.case_id for case in DIFFERENTIAL_CASES if case.suite == "cuda-distributed-collectives"}
        self.assertEqual(registered, {descriptor["id"] for descriptor in descriptors})

    def test_distributed_inputs(self) -> None:
        import numpy

        for descriptor in distributed_descriptors():
            for rank in (0, 1):
                with self.subTest(case_id=descriptor["id"], rank=rank):
                    inputs = distributed_inputs(descriptor, rank)
                    self.assertEqual(inputs[0].shape, (1, *descriptor["local_shape"]))
                    self.assertEqual(inputs[0].dtype, numpy.dtype(numpy.float32))
                    count = numpy.prod(descriptor["local_shape"])
                    self.assertEqual(float(inputs[0].flat[0]), rank * count + 1)
                    if descriptor["operation"] == "ragged_all_to_all":
                        self.assertEqual(inputs[1].shape, (1, *descriptor["output_shape"]))
                        count = numpy.prod(descriptor["output_shape"])
                        self.assertEqual(float(inputs[1].flat[0]), rank * count + 100)
                        expected_type = numpy.int32 if descriptor["metadata_type"] == "i32" else numpy.uint64
                        self.assertEqual({value.dtype for value in inputs[2:]}, {numpy.dtype(expected_type)})
                        self.assertEqual({value.shape for value in inputs[2:]}, {(1, 2)})
        with self.assertRaisesRegex(ValueError, "distributed CUDA collective rank must be zero or one"):
            distributed_inputs(distributed_descriptors()[0], 2)

    def test_distributed_reference(self) -> None:
        expected = {
            "cuda_distributed_sum": ((6., 8., 10., 12.), (6., 8., 10., 12.)),
            "cuda_distributed_all_gather": ((1., 2., 3., 4., 5., 6., 7., 8.),) * 2,
            "cuda_distributed_all_to_all": (
                (1., 2., 9., 10., 3., 4., 11., 12.),
                (5., 6., 13., 14., 7., 8., 15., 16.),
            ),
            "cuda_distributed_ragged_i32": ((100., 1., 102., 5., 6.), (2., 3., 107., 7., 109.)),
            "cuda_distributed_ragged_u64_rows": (
                (100., 101., 1., 2., 104., 105., 9., 10., 11., 12.),
                (3., 4., 5., 6., 114., 115., 13., 14., 118., 119.),
            ),
            "cuda_distributed_ragged_zero_transfers": (
                (100., 101., 102., 5., 104.), (105., 2., 3., 108., 109.),
            ),
            "cuda_distributed_ragged_repeated_reads": ((1., 101., 6., 7., 104.), (1., 2., 107., 6., 109.)),
        }
        for descriptor in distributed_descriptors():
            with self.subTest(case_id=descriptor["id"]):
                self.assertEqual(distributed_reference(descriptor), {"primal": expected[descriptor["id"]]})

    def test_distributed_reference_invalid_metadata(self) -> None:
        descriptor = next(
            descriptor for descriptor in distributed_descriptors() if descriptor["id"] == "cuda_distributed_ragged_i32"
        )
        with self.assertRaisesRegex(ValueError, "distributed ragged send and receive metadata are inconsistent"):
            distributed_reference({**descriptor, "receive_sizes": [[1, 1], [2, 1]]})
        with self.assertRaisesRegex(ValueError, "distributed ragged transfer exceeds its data extent"):
            distributed_reference({**descriptor, "input_offsets": [[0, 3], [0, 2]]})
        with self.assertRaisesRegex(ValueError, "distributed ragged received regions overlap"):
            distributed_reference({**descriptor, "output_offsets": [[1, 0], [1, 3]]})

    def test_distributed_contract(self) -> None:
        descriptors = distributed_descriptors()
        for descriptor in descriptors[:3]:
            contract = distributed_contract(descriptor)
            self.assertEqual(len(contract), 1)
            self.assertEqual(contract[0].groups, ((0, 1),))
        self.assertEqual(distributed_contract(descriptors[0])[0].operation, "all_reduce")
        self.assertEqual(distributed_contract(descriptors[1])[0].axis_attributes, (("all_gather_dim", 0),))
        self.assertEqual(distributed_contract(descriptors[2])[0].axis_attributes, (
            ("concat_dimension", 1), ("split_count", 2), ("split_dimension", 0),
        ))
        self.assertEqual(distributed_contract(descriptors[3]), ())


if __name__ == "__main__":
    unittest.main()
