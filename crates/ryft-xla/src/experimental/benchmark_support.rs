use ryft_core::operations::trigonometric::Sin;
use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
use ryft_core::tracing::Program;
use ryft_core::tracing_v2::benchmarking::{
    BenchmarkCase, BenchmarkError, IrBenchmarkRecord, IrBenchmarkSummary, IrNestedRegionSummary, nested_region, record,
    summarize_program,
};
use ryft_core::tracing_v2::operations::dot::DotDimensionNumbers;
use ryft_core::tracing_v2::{DifferentiableContext, Dot};

use crate::experimental::operations::LinearShardMapEvalMode;
use ryft_core::types::{ArrayType, DataType, Shape, Size};

use crate::experimental::lowering::to_mlir_module_for_program;
use crate::experimental::ops::{XlaOperation, XlaOperationExtension};
use crate::experimental::shard_map::{FlatTracedShardMap, ShardMapTracer, TracedXlaProgram, shard_map, trace};

/// Returns the XLA-focused IR benchmark cases.
pub fn cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase::new("shard_map_basic", emit_shard_map_basic),
        BenchmarkCase::new("shard_map_matmul", emit_shard_map_matmul),
        BenchmarkCase::new("shard_map_grad_inside", emit_shard_map_grad_inside),
        BenchmarkCase::new("grad_around_shard_map", emit_grad_around_shard_map),
        BenchmarkCase::new("nested_shard_map", emit_nested_shard_map),
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

/// Returns a rank-0 replicated sharding.
fn scalar_sharding(mesh: &LogicalMesh) -> Sharding {
    Sharding::replicated(mesh.clone(), 0)
}

/// Returns a rank-0 benchmark array type.
fn scalar_type() -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![]), None, None)
        .expect("benchmark scalar types are constructed without sharding")
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
    let program = body.program().simplified()?;
    Ok(nested_region(label, summarize_xla_program(&program)?))
}

/// Summarizes one traced XLA program, including nested shard-map bodies.
///
/// # Parameters
///
///   - `program`: Program to summarize.
fn summarize_xla_program<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>>(
    program: &Program<ArrayType, ArrayType, XlaOperation, Input, Output>,
) -> Result<IrBenchmarkSummary, BenchmarkError> {
    fn summarize_linear_eval_mode(
        label: &'static str,
        eval_mode: &LinearShardMapEvalMode,
    ) -> Result<Vec<IrNestedRegionSummary>, BenchmarkError> {
        match eval_mode {
            LinearShardMapEvalMode::Body(body) => Ok(vec![summarize_nested_body(label, body)?]),
            LinearShardMapEvalMode::FactorizedTranspose(factorized) => Ok(vec![
                summarize_nested_body("linear_shard_map.residual_body", factorized.residual_body())?,
                summarize_nested_body("linear_shard_map.apply_body", factorized.apply_body())?,
            ]),
        }
    }

    summarize_program(program, |op| {
        if let XlaOperation::Extension(XlaOperationExtension::ShardMap(shard_map_op)) = op {
            return Ok(vec![summarize_nested_body("shard_map.body", shard_map_op.body())?]);
        }

        if let XlaOperation::Extension(XlaOperationExtension::LinearShardMap(shard_map_op)) = op {
            let mut nested_regions = vec![summarize_nested_body("shard_map.body", shard_map_op.body())?];
            nested_regions.extend(summarize_linear_eval_mode(
                "linear_shard_map.eval_body",
                shard_map_op.linear_state().eval_mode(),
            )?);
            #[cfg(feature = "benchmarking")]
            {
                nested_regions.extend(summarize_linear_eval_mode(
                    "linear_shard_map.transpose_body",
                    shard_map_op.linear_state().transpose_mode(),
                )?);
            }
            return Ok(nested_regions);
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
    Input: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType>>,
    Output: Parameterized<ArrayType, Family: ParameterizedFamily<ArrayType>>,
>(
    case_id: &'static str,
    traced: &TracedXlaProgram<Input, Output>,
) -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let program = traced.program().simplified()?;
    let summary = summarize_xla_program(&program)?;
    Ok(vec![record(
        case_id,
        "xla",
        "program",
        to_mlir_module_for_program(
            &program,
            traced.global_input_types(),
            traced.global_output_types(),
            "main",
            None,
            None,
        )
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
                .unwrap_or_else(|error| panic!("basic shard_map IR benchmark should differentiable: {error}"))
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
                    |(lhs, rhs)| lhs.dot(rhs, &DotDimensionNumbers::matmul()),
                    inputs,
                    mesh.clone(),
                    (lhs_spec.clone(), rhs_spec.clone()),
                    out_spec.clone(),
                )
                .unwrap_or_else(|error| panic!("matmul shard_map IR benchmark should differentiable: {error}"))
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
    let sharding = scalar_sharding(&mesh);
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let mesh = mesh.clone();
            move |x: ShardMapTracer| {
                let context = x.context().clone();
                context
                    .value_and_gradient(
                        {
                            let mesh = mesh.clone();
                            let sharding = sharding.clone();
                            move |y| {
                                shard_map::<_, _, ArrayType, _>(
                                    |local_x: ShardMapTracer| local_x.sin(),
                                    y,
                                    mesh.clone(),
                                    sharding.clone(),
                                    sharding.clone(),
                                )
                                .unwrap_or_else(|error| {
                                    panic!(
                                        "grad-around-shard-map IR benchmark should trace the inner shard_map: {error}"
                                    )
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
        scalar_type(),
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
                        let context = local_x.context().clone();
                        context.value_and_gradient(|y| y.sin(), local_x).unwrap_or_else(|error| {
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

#[cfg(test)]
mod tests {
    use ryft_core::operations::arithmetic::MUL_OPERATION_NAME;

    use super::*;

    #[test]
    fn test_emit_grad_around_shard_map_records_factorized_transpose_regions() {
        let records = emit_grad_around_shard_map().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].raw_ir().matches("sdy.manual_computation").count(), 2);
        assert_eq!(records[0].summary().op_histogram().get("shard_map"), Some(&1));

        let nested_region = |label: &str| {
            records[0]
                .summary()
                .nested_regions()
                .iter()
                .find(|region| region.label() == label)
                .unwrap_or_else(|| panic!("expected nested region '{label}'"))
        };
        assert_eq!(nested_region("shard_map.body").op_histogram().get("sin"), Some(&1));
        assert_eq!(nested_region("linear_shard_map.residual_body").op_histogram().get("cos"), Some(&1));
        assert_eq!(nested_region("linear_shard_map.apply_body").op_histogram().get(MUL_OPERATION_NAME), Some(&1));
        assert_eq!(nested_region("linear_shard_map.transpose_body").op_histogram().get("cos"), Some(&1));
        assert_eq!(nested_region("linear_shard_map.transpose_body").op_histogram().get(MUL_OPERATION_NAME), Some(&1),);
    }
}
