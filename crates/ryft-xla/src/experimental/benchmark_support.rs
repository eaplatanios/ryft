//! Internal IR benchmark support for traced XLA and `shard_map`.
//!
//! This module owns the Rust-side benchmark cases that exercise the XLA tracing path and the
//! higher-order `shard_map` operation.

use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
use ryft_core::tracing_v2::{
    FloatExt, MatrixOps, One, PrimitiveOp, Program,
    benchmarking::{
        BenchmarkCase, BenchmarkError, IrBenchmarkRecord, IrBenchmarkSummary, IrNestedRegionSummary, nested_region,
        record, summarize_graph,
    },
    grad, vmap,
};

use crate::experimental::operations::{LinearShardMapEvalMode, ShardMapOp};
use ryft_core::types::{ArrayType, DataType, Shape, Size};

use crate::experimental::lowering::to_mlir_module_for_graph;
use crate::experimental::shard_map::{
    FlatTracedShardMap, ShardMapTensor, ShardMapTracer, TracedXlaProgram, shard_map, trace,
};

/// Returns the XLA-focused IR benchmark cases.
pub fn cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase { case_id: "shard_map_basic", emit: emit_shard_map_basic },
        BenchmarkCase { case_id: "shard_map_matmul", emit: emit_shard_map_matmul },
        BenchmarkCase { case_id: "shard_map_grad_inside", emit: emit_shard_map_grad_inside },
        BenchmarkCase { case_id: "grad_around_shard_map", emit: emit_grad_around_shard_map },
        BenchmarkCase { case_id: "nested_shard_map", emit: emit_nested_shard_map },
        BenchmarkCase { case_id: "shard_map_grad_vmap_composition", emit: emit_shard_map_grad_vmap_composition },
    ]
}

/// Returns the canonical single-axis manual mesh used by the benchmark cases.
fn benchmark_mesh() -> LogicalMesh {
    LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
}

/// Returns the outer mesh used by the nested shard-map benchmark.
fn nested_outer_mesh() -> LogicalMesh {
    LogicalMesh::new(vec![
        MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
        MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
    ])
    .unwrap()
}

/// Returns the inner mesh used by the nested shard-map benchmark.
fn nested_inner_mesh() -> LogicalMesh {
    LogicalMesh::new(vec![
        MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
        MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
    ])
    .unwrap()
}

/// Returns a one-dimensional sharding.
fn sharded_1d_sharding(mesh: &LogicalMesh) -> Sharding {
    Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()
}

/// Returns a two-dimensional row-sharded sharding.
fn row_sharded_sharding(mesh: &LogicalMesh) -> Sharding {
    Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap()
}

/// Returns a two-dimensional replicated sharding.
fn replicated_2d_sharding(mesh: &LogicalMesh) -> Sharding {
    Sharding::replicated(mesh.clone(), 2)
}

/// Returns a rank-1 benchmark array type.
///
/// # Parameters
///
///   - `size`: Static vector length.
fn vector_type(size: usize) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(size)]), None, None)
        .expect("benchmark vector types are constructed without sharding")
}

/// Returns a rank-2 benchmark array type.
///
/// # Parameters
///
///   - `rows`: Matrix row count.
///   - `cols`: Matrix column count.
fn matrix_type(rows: usize, cols: usize) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(rows), Size::Static(cols)]), None, None)
        .expect("benchmark matrix types are constructed without sharding")
}

/// Summarizes one erased nested shard-map body.
///
/// # Parameters
///
///   - `label`: Stable nested-region label.
///   - `body`: Nested shard-map body to summarize.
fn summarize_nested_body(
    label: &'static str,
    body: &FlatTracedShardMap,
) -> Result<IrNestedRegionSummary, BenchmarkError> {
    let program = body.compiled.program().simplify()?;
    Ok(nested_region(label, summarize_xla_program(&program)?))
}

/// Summarizes one traced XLA program, including nested shard-map bodies.
///
/// # Parameters
///
///   - `program`: Program to summarize.
fn summarize_xla_program<Input: Parameterized<ShardMapTensor>, Output: Parameterized<ShardMapTensor>>(
    program: &Program<ArrayType, ShardMapTensor, Input, Output>,
) -> Result<IrBenchmarkSummary, BenchmarkError> {
    fn summarize_linear_eval_mode(
        label: &'static str,
        eval_mode: &LinearShardMapEvalMode,
    ) -> Result<Vec<IrNestedRegionSummary>, BenchmarkError> {
        match eval_mode {
            LinearShardMapEvalMode::Body(body) => Ok(vec![summarize_nested_body(label, body)?]),
            LinearShardMapEvalMode::FactorizedTranspose(factorized) => Ok(vec![
                summarize_nested_body("linear_shard_map.residual_body", &factorized.residual_body)?,
                summarize_nested_body("linear_shard_map.apply_body", &factorized.apply_body)?,
            ]),
        }
    }

    summarize_graph(program.graph(), |op| {
        if let PrimitiveOp::Custom(custom_op) = op {
            if let Some(shard_map_op) = custom_op.extensions().get::<ShardMapOp<ShardMapTensor>>() {
                let mut nested_regions = vec![summarize_nested_body("shard_map.body", shard_map_op.body())?];
                if let Some(eval_mode) = shard_map_op.eval_mode() {
                    nested_regions.extend(summarize_linear_eval_mode("linear_shard_map.eval_body", eval_mode)?);
                }
                if let Some(transpose_mode) = shard_map_op.transpose_mode() {
                    nested_regions
                        .extend(summarize_linear_eval_mode("linear_shard_map.transpose_body", transpose_mode)?);
                }
                return Ok(nested_regions);
            }
        }

        Ok(Vec::new())
    })
}

/// Builds the program and MLIR records for one traced XLA program.
///
/// # Parameters
///
///   - `case_id`: Stable benchmark case identifier.
///   - `traced`: Traced XLA handle to render.
fn traced_xla_records<
    Input: Parameterized<ArrayType, ParameterStructure: Clone>,
    Output: Parameterized<ArrayType, ParameterStructure: Clone>,
>(
    case_id: &'static str,
    traced: &TracedXlaProgram<Input, Output>,
) -> Result<Vec<IrBenchmarkRecord>, BenchmarkError>
where
    Input::Family: ParameterizedFamily<ShardMapTensor>,
    Output::Family: ParameterizedFamily<ShardMapTensor>,
{
    let program = traced.compiled().program().simplify()?;
    let summary = summarize_xla_program(&program)?;
    Ok(vec![record(
        case_id,
        "xla",
        "program",
        to_mlir_module_for_graph(program.graph(), traced.global_input_types(), traced.global_output_types(), "main")
            .map_err(|error| BenchmarkError::External(Box::new(error)))?,
        summary,
    )])
}

/// Emits the basic traced `shard_map` benchmark.
fn emit_shard_map_basic() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let mesh = benchmark_mesh();
    let sharding = sharded_1d_sharding(&mesh);
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let mesh = mesh.clone();
            move |x: ShardMapTracer| {
                shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                    |local_x: ShardMapTracer| local_x.sin(),
                    x,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                )
                .unwrap_or_else(|error| panic!("basic shard_map IR benchmark should trace: {error}"))
            }
        },
        vector_type(8),
    )
    .map_err(|error| BenchmarkError::External(Box::new(error)))?;
    traced_xla_records("shard_map_basic", &traced)
}

/// Emits the traced `shard_map` matrix-multiplication benchmark.
fn emit_shard_map_matmul() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let mesh = benchmark_mesh();
    let lhs_spec = row_sharded_sharding(&mesh);
    let rhs_spec = replicated_2d_sharding(&mesh);
    let out_spec = row_sharded_sharding(&mesh);
    let traced: TracedXlaProgram<(ArrayType, ArrayType), ArrayType> = trace(
        {
            let mesh = mesh.clone();
            move |inputs: (ShardMapTracer, ShardMapTracer)| {
                shard_map::<_, (ShardMapTracer, ShardMapTracer), ArrayType, ShardMapTracer>(
                    |(lhs, rhs)| lhs.matmul(rhs),
                    inputs,
                    mesh.clone(),
                    (lhs_spec.clone(), rhs_spec.clone()),
                    out_spec.clone(),
                )
                .unwrap_or_else(|error| panic!("matmul shard_map IR benchmark should trace: {error}"))
            }
        },
        (matrix_type(8, 4), matrix_type(4, 2)),
    )
    .map_err(|error| BenchmarkError::External(Box::new(error)))?;
    traced_xla_records("shard_map_matmul", &traced)
}

/// Emits the traced reverse-mode-around-`shard_map` benchmark.
fn emit_grad_around_shard_map() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let mesh = benchmark_mesh();
    let sharding = sharded_1d_sharding(&mesh);
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let mesh = mesh.clone();
            move |x: ShardMapTracer| {
                grad(
                    {
                        let mesh = mesh.clone();
                        let sharding = sharding.clone();
                        move |y: ShardMapTracer| {
                            shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                                |local_x: ShardMapTracer| local_x.sin(),
                                y,
                                mesh.clone(),
                                sharding.clone(),
                                sharding.clone(),
                            )
                            .unwrap_or_else(|error| {
                                panic!("grad-around-shard-map IR benchmark should trace the inner shard_map: {error}")
                            })
                        }
                    },
                    x,
                )
                .unwrap_or_else(|error| {
                    panic!("grad-around-shard-map IR benchmark should trace the outer gradient: {error}")
                })
            }
        },
        vector_type(8),
    )
    .map_err(|error| BenchmarkError::External(Box::new(error)))?;
    traced_xla_records("grad_around_shard_map", &traced)
}

/// Emits the traced reverse-mode-inside-`shard_map` benchmark.
fn emit_shard_map_grad_inside() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let mesh = benchmark_mesh();
    let sharding = sharded_1d_sharding(&mesh);
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let mesh = mesh.clone();
            move |x: ShardMapTracer| {
                shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                    |local_x: ShardMapTracer| {
                        grad(|y| y.sin(), local_x).unwrap_or_else(|error| {
                            panic!("shard_map grad-inside IR benchmark should trace the inner gradient: {error}")
                        })
                    },
                    x,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                )
                .unwrap_or_else(|error| {
                    panic!("shard_map grad-inside IR benchmark should trace the shard_map: {error}")
                })
            }
        },
        vector_type(8),
    )
    .map_err(|error| BenchmarkError::External(Box::new(error)))?;
    traced_xla_records("shard_map_grad_inside", &traced)
}

/// Emits the nested traced `shard_map` benchmark.
fn emit_nested_shard_map() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let outer_mesh = nested_outer_mesh();
    let inner_mesh = nested_inner_mesh();
    let outer_sharding = sharded_1d_sharding(&outer_mesh);
    let inner_sharding = Sharding::new(inner_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let outer_mesh = outer_mesh.clone();
            let inner_mesh = inner_mesh.clone();
            move |x: ShardMapTracer| {
                shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                    {
                        let inner_mesh = inner_mesh.clone();
                        move |outer_x: ShardMapTracer| {
                            let nested: ShardMapTracer = shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                                |inner_x: ShardMapTracer| inner_x.clone() + inner_x,
                                outer_x.clone(),
                                inner_mesh.clone(),
                                inner_sharding.clone(),
                                inner_sharding.clone(),
                            )
                            .unwrap_or_else(|error| {
                                panic!("nested shard_map IR benchmark should trace the inner shard_map: {error}")
                            });
                            nested + outer_x
                        }
                    },
                    x,
                    outer_mesh.clone(),
                    outer_sharding.clone(),
                    outer_sharding.clone(),
                )
                .unwrap_or_else(|error| {
                    panic!("nested shard_map IR benchmark should trace the outer shard_map: {error}")
                })
            }
        },
        vector_type(8),
    )
    .map_err(|error| BenchmarkError::External(Box::new(error)))?;
    traced_xla_records("nested_shard_map", &traced)
}

/// Emits the traced `shard_map` benchmark that composes reverse mode and batching inside the body.
fn emit_shard_map_grad_vmap_composition() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let mesh = benchmark_mesh();
    let sharding = sharded_1d_sharding(&mesh);
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let mesh = mesh.clone();
            move |x: ShardMapTracer| {
                shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                    |local_x: ShardMapTracer| {
                        let gradient = grad(|y| y.sin(), local_x.clone()).unwrap_or_else(|error| {
                            panic!("shard_map grad+vmap IR benchmark should trace the inner gradient: {error}")
                        });
                        let lanes: Vec<ShardMapTracer> = vmap(
                            |batch| batch.clone() + batch.one_like(),
                            vec![gradient.clone(), gradient],
                        )
                        .unwrap_or_else(|error| {
                            panic!("shard_map grad+vmap IR benchmark should trace the inner batch transform: {error}")
                        });
                        lanes[0].clone() + lanes[1].clone()
                    },
                    x,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                )
                .unwrap_or_else(|error| panic!("shard_map grad+vmap IR benchmark should trace the shard_map: {error}"))
            }
        },
        vector_type(8),
    )
    .map_err(|error| BenchmarkError::External(Box::new(error)))?;
    traced_xla_records("shard_map_grad_vmap_composition", &traced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_grad_around_shard_map_matches_compact_single_region_lowering() {
        let records = emit_grad_around_shard_map().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].raw_ir.matches("sdy.manual_computation").count(), 1);
        assert_eq!(records[0].summary.op_histogram.get("shard_map"), Some(&1));
        assert_eq!(records[0].summary.nested_regions[0].op_histogram.get("cos"), Some(&1));
        assert_eq!(records[0].summary.nested_regions[0].op_histogram.get("mul"), Some(&1));
    }

    #[test]
    fn test_emit_shard_map_grad_vmap_composition_uses_the_expected_broadcast_ladder() {
        let records = emit_shard_map_grad_vmap_composition().unwrap();
        assert_eq!(records.len(), 1);
        assert!(
            records[0]
                .raw_ir
                .contains("stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2xf32>")
        );
        assert!(
            records[0]
                .raw_ir
                .contains("stablehlo.broadcast_in_dim %7, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>")
        );
        assert!(records[0].raw_ir.contains("tensor<1x2xf32>"));
        assert!(records[0].raw_ir.contains("dims = [0, 1] : (tensor<1x2xf32>) -> tensor<2x2xf32>"));
    }
}
