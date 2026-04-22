"""Unit tests for the `ryft` Python JAX inspection helpers."""

from __future__ import annotations

import unittest

import jax
import numpy as np

from ryft.jax.examples import program_cases_by_name
from ryft.jax.extraction import ProgramCase, extract_jaxpr, inspect_program, lower_to_stablehlo
from ryft.jax.reshape_parity import load_expected_mlir, normalize_mlir


class JaxToolsTest(unittest.TestCase):
    """Covers the reusable helpers that back the Python JAX inspection scripts."""

    def test_extract_jaxpr_renders_scalar_addition(self) -> None:
        inspection = extract_jaxpr(lambda x: x + 1.0, 2.0)

        self.assertIn("add", inspection.text)
        self.assertEqual(inspection.consts, ())

    def test_lower_to_stablehlo_renders_scalar_addition(self) -> None:
        stablehlo = lower_to_stablehlo(
            lambda x: x + 1.0,
            jax.ShapeDtypeStruct(shape=(), dtype=np.dtype(np.float32)),
        )

        self.assertIn("stablehlo.add", stablehlo)

    def test_inspect_program_requires_example_args_for_jaxpr(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not define example arguments"):
            inspect_program(
                ProgramCase(
                    name="lowering_only",
                    description="Only has lowering inputs.",
                    function=lambda x: x,
                    abstract_args=(jax.ShapeDtypeStruct(shape=(), dtype=np.dtype(np.float32)),),
                )
            )

    def test_program_cases_include_migrated_legacy_experiments(self) -> None:
        case_names = program_cases_by_name().keys()

        self.assertIn("right_mul_4_2_transpose", case_names)
        self.assertIn("sin_plus_cos_times_sin_differential", case_names)

    def test_normalize_mlir_removes_wrapper_only_differences(self) -> None:
        jax_mlir = """
        module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
          func.func public @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {jax.result_info = "result"}) {
            return %arg0 : tensor<8xf32>
          }
        }
        """
        rust_mlir = """
        module {
          func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
            return %arg0 : tensor<8xf32>
          }
        }
        """

        self.assertEqual(normalize_mlir(jax_mlir), normalize_mlir(rust_mlir))

    def test_load_expected_mlir_reads_current_reshape_expectations(self) -> None:
        expected_mlir = load_expected_mlir()

        self.assertIn("test_trace_reshape_with_sharding_constraint_renders_stablehlo_and_shardy", expected_mlir)
        self.assertIn(
            "test_shard_map_reshape_renders_replicated_split_sharding_propagation",
            expected_mlir,
        )


if __name__ == "__main__":
    unittest.main()
