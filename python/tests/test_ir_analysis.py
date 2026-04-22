"""Unit tests for the reusable IR analysis helpers."""

from __future__ import annotations

import unittest

import jax
from jax import core as jax_core

from ryft.jax.ir_analysis import strip_mlir_location_markers, summarize_jaxpr, summarize_mlir


class IrAnalysisTest(unittest.TestCase):
    """Covers the reusable MLIR and JAXPR analysis helpers."""

    def test_strip_mlir_location_markers_removes_inline_and_trailing_locations(self) -> None:
        rendered = "\n".join(
            [
                '#loc1 = loc("x")',
                '#loc2 = loc("shard_map")',
                "module @jit__lambda attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {",
                '  sdy.mesh @mesh = <["x"=4]> loc(#loc1)',
                '  func.func public @main(%arg0: tensor<8xf32> loc("x"))'
                ' -> (tensor<8xf32> {jax.result_info = "result"}) {',
                '    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] '
                'out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"}'
                ' (%arg1: tensor<2xf32> loc("shard_map:"(#loc2))) {',
                "      %1 = stablehlo.sine %arg1 : tensor<2xf32> loc(#loc2)",
                '      sdy.return %1 : tensor<2xf32> loc("jit(<lambda>)/shard_map")',
                "    } : (tensor<8xf32>) -> tensor<8xf32> loc(#loc2)",
                "    return %0 : tensor<8xf32> loc(#loc1)",
                "  } loc(#loc1)",
                "} loc(#loc1)",
            ]
        )

        stripped = strip_mlir_location_markers(rendered)

        self.assertEqual(
            stripped,
            "\n".join(
                [
                    "module @jit__lambda attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {",
                    '  sdy.mesh @mesh = <["x"=4]>',
                    '  func.func public @main(%arg0: tensor<8xf32>)'
                    ' -> (tensor<8xf32> {jax.result_info = "result"}) {',
                    '    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] '
                    'out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {',
                    "      %1 = stablehlo.sine %arg1 : tensor<2xf32>",
                    "      sdy.return %1 : tensor<2xf32>",
                    "    } : (tensor<8xf32>) -> tensor<8xf32>",
                    "    return %0 : tensor<8xf32>",
                    "  }",
                    "}",
                    "",
                ]
            ),
        )

    def test_summarize_mlir_counts_normalized_operations(self) -> None:
        summary = summarize_mlir(
            """
            module {
              sdy.mesh @mesh = <["x"=4]>
              func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                %0 = stablehlo.constant dense<0.0> : tensor<f32>
                %1 = sdy.manual_computation(%arg0) (%arg1: tensor<8xf32>) {
                  %2 = stablehlo.sine %arg1 : tensor<8xf32>
                  sdy.return %2 : tensor<8xf32>
                } : (tensor<8xf32>) -> tensor<8xf32>
                %3 = stablehlo.multiply %1, %arg0 : tensor<8xf32>
                return %3 : tensor<8xf32>
              }
            }
            """
        )

        self.assertEqual(summary["equation_count"], 4)
        self.assertEqual(summary["constant_count"], 1)
        self.assertEqual(summary["nested_region_count"], 1)
        self.assertEqual(summary["op_histogram"], {"const": 1, "mul": 1, "shard_map": 1, "sin": 1})

    def test_summarize_jaxpr_tracks_histogram_constants_and_depth(self) -> None:
        closed_jaxpr = jax.make_jaxpr(lambda x: x * x + 1.0)(2.0)

        summary = summarize_jaxpr(closed_jaxpr, jax_core)

        self.assertEqual(summary["input_leaf_count"], 1)
        self.assertEqual(summary["output_leaf_count"], 1)
        self.assertEqual(summary["equation_count"], 2)
        self.assertEqual(summary["constant_count"], 1)
        self.assertEqual(summary["op_histogram"], {"add": 1, "mul": 1})
        self.assertEqual(summary["nested_region_count"], 0)
        self.assertEqual(summary["max_dependency_depth"], 2)


if __name__ == "__main__":
    unittest.main()
