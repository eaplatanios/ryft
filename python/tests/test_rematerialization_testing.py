"""Unit tests for the rematerialization differential cases, which run the pinned JAX in process but never Cargo."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jax_numpy
import numpy

from ryft.jax.rematerialization_testing import (
    REMATERIALIZATION_CASES,
    STABLEHLO_PATTERNS,
    build_rematerialization_jax,
    jax_residuals,
    rematerialization_reference,
    residual_observations,
)


CASES_BY_ID = {case.case_id: case for case in REMATERIALIZATION_CASES}


class RematerializationTestingTest(unittest.TestCase):
    """Covers residual encoding, the analytic references, and the JAX builders against those references."""

    maxDiff = None

    def test_residual_observations(self) -> None:
        self.assertEqual(
            residual_observations([(6, 32), (2, 16), (4, 32), (2, 32)]),
            {
                "residual_count": ((4.0,),),
                "residual_element_counts": ((2.0, 2.0, 4.0, 6.0),),
                "residual_bit_widths": ((16.0, 32.0, 32.0, 32.0),),
            },
        )
        self.assertEqual(
            residual_observations([]),
            {"residual_count": ((0.0,),), "residual_element_counts": ((),), "residual_bit_widths": ((),)},
        )

    def test_jax_residuals(self) -> None:
        saved = [
            (SimpleNamespace(shape=(2, 3), dtype=numpy.dtype(numpy.float32)), "from the argument x"),
            (SimpleNamespace(shape=(), dtype=numpy.dtype(numpy.float16)), "named 'square'"),
        ]
        self.assertEqual(jax_residuals(saved), ((6, 32), (1, 16)))

    def test_rematerialization_reference(self) -> None:
        # The policy body computes `b²` for `b = Σⱼ yᵢⱼ³` and `y = x · w = [[6, -2], [0, 2]]`, so `b = [208, 8]`.
        self.assertEqual(
            rematerialization_reference(CASES_BY_ID["rematerialization_dots_saveable"]),
            {
                "output": ((43264.0, 64.0),),
                "jvp": ((134784.0, 192.0),),
                "x_gradient": ((39936.0, 89856.0, -39936.0, 192.0, 0.0, -192.0),),
                "w_gradient": ((44928.0, 4992.0, 89856.0, 9792.0, -44928.0, -5376.0),),
                "residual_count": ((4.0,),),
                "residual_element_counts": ((2.0, 4.0, 6.0, 6.0),),
                "residual_bit_widths": ((32.0, 32.0, 32.0, 32.0),),
            },
        )

    def test_rematerialization_reference_condition_branches(self) -> None:
        # `Σ x > 0` selects `(y · y)²`, and `-x` selects `y³` for `y = -x · w`.
        reference = rematerialization_reference(CASES_BY_ID["rematerialization_dots_saveable_condition"])
        self.assertEqual(reference["output"], ((1296.0, 256.0, 0.0, 16.0),))
        self.assertEqual(reference["false_branch_output"], ((-216.0, 8.0, 0.0, -8.0),))
        self.assertEqual(reference["false_branch_x_gradient"], ((120.0, 216.0, -120.0, -12.0, 0.0, 12.0),))

    def test_rematerialization_reference_scan(self) -> None:
        reference = rematerialization_reference(CASES_BY_ID["rematerialization_dots_saveable_scan"])
        self.assertEqual(reference["output"], ((16.0, 400.0, 16.0, 64.0),))
        self.assertEqual(reference["layers_gradient"], ((-1920.0, -768.0, 768.0, 0.0, 192.0, -576.0, 96.0, -96.0),))

    def test_rematerialization_reference_scans_with_nested_regions(self) -> None:
        # The nested scans compute the values of the scan case. The condition in the scan adds the sum of the squares
        # of each layer (`3` for both), doubled at `-x` on the first iteration, whose carry sums to a negative value.
        # JAX saves the stacked dot products of both workloads rather than the carries or the outputs of the condition.
        reference = rematerialization_reference(CASES_BY_ID["rematerialization_dots_saveable_nested_scan"])
        self.assertEqual(reference["output"], ((16.0, 400.0, 16.0, 64.0),))
        self.assertEqual(reference["residual_element_counts"], ((4.0, 6.0, 6.0, 8.0, 8.0),))
        reference = rematerialization_reference(CASES_BY_ID["rematerialization_dots_saveable_condition_in_scan"])
        self.assertEqual(reference["output"], ((1524.0, 52.0, 12.0, 52.0),))
        self.assertEqual(reference["false_branch_output"], ((1767.0, 103.0, 39.0, 103.0),))
        self.assertEqual(reference["layers_gradient"], ((0.0,) * 8,))
        self.assertEqual(reference["residual_element_counts"], ((2.0, 4.0, 6.0, 6.0),))

    def test_rematerialization_reference_known_regions(self) -> None:
        # The known condition scales `x · w` by `3 wᵀ w` at `x` and by `6 wᵀ w` at `-x`, and the known scan by
        # `3 (Σₖ layerₖ + Σₖ layerₖ · layerₖ)`. JAX saves the compact values of both regions rather than their outputs.
        reference = rematerialization_reference(CASES_BY_ID["rematerialization_dots_saveable_known_condition"])
        self.assertEqual(reference["output"], ((108.0, 12.0, 0.0, 12.0),))
        self.assertEqual(reference["false_branch_output"], ((-216.0, -24.0, 0.0, -24.0),))
        self.assertEqual(reference["residual_element_counts"], ((4.0, 6.0, 6.0),))
        reference = rematerialization_reference(CASES_BY_ID["rematerialization_dots_saveable_known_scan"])
        self.assertEqual(reference["output"], ((54.0, -12.0, 0.0, 18.0),))
        self.assertEqual(reference["layers_gradient"], ((0.0,) * 8,))
        self.assertEqual(reference["residual_element_counts"], ((6.0, 6.0, 8.0, 8.0),))

    def test_rematerialization_reference_composites(self) -> None:
        # A differentiated `custom_jvp` call is replaced by its rule, whose dot product `x · w` is saved, while the
        # policy classifies an all-known call itself, which it does not save, so only the inputs are saved and the
        # call is replayed whole from `w`.
        reference = rematerialization_reference(CASES_BY_ID["rematerialization_dots_saveable_composite"])
        self.assertEqual(reference["output"], ((1296.0, 16.0, 0.0, 16.0),))
        self.assertEqual(reference["residual_element_counts"], ((4.0, 6.0, 6.0),))
        reference = rematerialization_reference(CASES_BY_ID["rematerialization_dots_saveable_known_composite"])
        self.assertEqual(reference["output"], ((108.0, 12.0, 0.0, 12.0),))
        self.assertEqual(reference["residual_element_counts"], ((6.0, 6.0),))

    def test_rematerialization_reference_offloaded_scans(self) -> None:
        # The known scan adds both scalar norms (`3` each) to its initial carry `sum(w) = 2`. The carried scan starts
        # from `x` and squares each carry before adding a norm. Only `x` carries gradients in either case.
        reference = rematerialization_reference(CASES_BY_ID["rematerialization_offloaded_known_scan"])
        self.assertEqual(reference["output"], ((40.0,),))
        self.assertEqual(reference["jvp"], ((8.0,),))
        self.assertEqual(reference["x_gradient"], ((8.0,) * 6,))
        self.assertEqual(reference["residual_element_counts"], ((2.0, 6.0),))
        reference = rematerialization_reference(CASES_BY_ID["rematerialization_offloaded_carried_scan"])
        self.assertEqual(reference["output"], ((227.0,),))
        self.assertEqual(reference["jvp"], ((164.0,),))
        self.assertEqual(reference["x_gradient"], ((35.0, 164.0, 35.0, 12.0, 35.0, 164.0),))
        self.assertEqual(reference["w_gradient"], ((0.0,) * 6,))
        self.assertEqual(reference["layers_gradient"], ((0.0,) * 8,))
        self.assertEqual(reference["residual_element_counts"], ((2.0, 6.0, 8.0),))

    def test_build_rematerialization_jax(self) -> None:
        # Every JAX observation, including the saved residuals of JAX 0.12.0.1, matches its analytic reference, and
        # every staged gradient places its recomputation behind an optimization barrier.
        for case in REMATERIALIZATION_CASES:
            with self.subTest(case=case.case_id):
                observation = build_rematerialization_jax(case, jax, jax_numpy, numpy)
                self.assertEqual(observation.case_id, case.case_id)
                self.assertEqual(observation.observations, rematerialization_reference(case))
                self.assertIsNotNone(observation.stablehlo)
                for pattern in STABLEHLO_PATTERNS:
                    self.assertIn(pattern, observation.stablehlo)


if __name__ == "__main__":
    unittest.main()
