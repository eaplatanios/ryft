use ryft_core::backends::arrays::Array as CpuArray;
use ryft_core::backends::arrays::ArrayOperation;
use ryft_core::contexts::{Context, EagerContext};
use ryft_core::differentiation::ForwardModeDifferentiate;
use ryft_core::operations::math::{Dot, DotDimensionNumbers, Sin};
use ryft_core::parameters::{Parameterized, ParameterizedFamily};
use ryft_core::programs::ProgramError;
use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
use ryft_core::tracing_v2::benchmarking::{
    BenchmarkCase, BenchmarkError, IrBenchmarkRecord, IrBenchmarkSummary, record, summarize_program,
};
use ryft_core::types::{ArrayProgramType, ArrayType, DataType, Dimension, Shape};

use crate::experimental::lowering::{to_mlir_module_for_plain_program, to_mlir_module_for_program};
use crate::experimental::ops::{XlaConstant, XlaProgram};
use crate::experimental::shard_map::{ShardMapTracer, TracedXlaProgram, shard_map, trace};

/// Returns the XLA-focused IR benchmark cases.
pub fn cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase::new("shard_map_basic", emit_shard_map_basic),
        BenchmarkCase::new("shard_map_matmul", emit_shard_map_matmul),
        BenchmarkCase::new("shard_map_grad_inside", emit_shard_map_grad_inside),
        BenchmarkCase::new("grad_around_shard_map", emit_grad_around_shard_map),
        BenchmarkCase::new("nested_shard_map", emit_nested_shard_map),
        BenchmarkCase::new(
            "scalar_quartic_plus_sin_linearize_pushforward",
            emit_scalar_quartic_plus_sin_linearize_pushforward,
        ),
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
    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(size)]))
}

/// Returns a rank-2 benchmark array type.
///
/// # Parameters
///
///   - `rows`: Matrix row count.
///   - `cols`: Matrix column count.
fn matrix_type(rows: usize, cols: usize) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(rows), Dimension::Static(cols)]))
}

/// Summarizes one traced XLA program; attached nested regions (including shard-map bodies) are covered by the
/// generic region-arena walk in [`summarize_program`].
///
/// # Parameters
///
///   - `program`: Program to summarize.
fn summarize_xla_program<Input: Parameterized<XlaConstant>, Output: Parameterized<XlaConstant>>(
    program: &XlaProgram<Input, Output>,
) -> Result<IrBenchmarkSummary, BenchmarkError> {
    summarize_program(program)
}

/// Builds the program and MLIR records for one traced XLA program.
///
/// # Parameters
///
///   - `case_id`: Stable benchmark case identifier.
///   - `traced`: Traced XLA handle to render.
fn traced_xla_records<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>>(
    case_id: &'static str,
    traced: &TracedXlaProgram<Input, Output>,
) -> Result<Vec<IrBenchmarkRecord>, BenchmarkError>
where
    <Input as Parameterized<ArrayType>>::Family:
        ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayProgramType> + ParameterizedFamily<XlaConstant>,
    <Output as Parameterized<ArrayType>>::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayProgramType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
{
    let program = traced.program().simplified()?;
    let summary = summarize_xla_program(&program)?;
    Ok(vec![record(
        case_id,
        "xla",
        "program",
        to_mlir_module_for_program(
            &program,
            &[],
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

/// Emits the directly linearized pushforward of `f(x) = x⁴ + sin(x)` through the canonical XLA MLIR lowering path.
fn emit_scalar_quartic_plus_sin_linearize_pushforward() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let context = EagerContext::<CpuArray, ArrayOperation<CpuArray>>::new();
    let (_, pushforward) =
        context.linearize(|x| Ok(x.clone() * x.clone() * x.clone() * x.clone() + x.sin()?), CpuArray::scalar(2.0))?;
    let (pushforward, residuals) = pushforward.into_parts();
    let (_, closed_pushforward) = context.interpret_and_trace(
        move |tangent| {
            let tracing_context = tangent.context().clone();
            let mut inputs = vec![tangent];
            inputs.extend(
                residuals
                    .iter()
                    .cloned()
                    .map(|residual| tracing_context.lift(residual))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let mut outputs = pushforward.interpret_in_context(&tracing_context, inputs)?;
            Ok(outputs.remove(0))
        },
        CpuArray::scalar(1.0),
    )?;
    let summary = summarize_program(&closed_pushforward)?;
    let mlir = to_mlir_module_for_plain_program(&closed_pushforward, "main")
        .map_err(|error| BenchmarkError::External(Box::new(error)))?;
    Ok(vec![record("scalar_quartic_plus_sin_linearize_pushforward", "scalar", "linearize_pushforward", mlir, summary)])
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
                    |local_x: ShardMapTracer| {
                        local_x
                            .sin()
                            .unwrap_or_else(|error| panic!("basic shard_map IR benchmark should trace sine: {error}"))
                    },
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
                    |(lhs, rhs)| lhs.dot(&rhs, &DotDimensionNumbers::matmul()),
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
    Err(ProgramError::UnsupportedOperation {
        message: "reverse-mode shard_map IR benchmarks are not implemented".to_string(),
    }
    .into())
}

/// Emits the traced reverse-mode-inside-`shard_map` benchmark.
fn emit_shard_map_grad_inside() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    emit_grad_around_shard_map()
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
    use super::*;

    #[test]
    fn test_emit_scalar_linearize_pushforward_lowers_closed_mlir() {
        let records = emit_scalar_quartic_plus_sin_linearize_pushforward().unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].raw_ir().starts_with("module {"));
        assert!(!records[0].raw_ir().contains("lambda %"));
        assert_eq!(records[0].summary().input_leaf_count(), 1);
        assert_eq!(records[0].summary().constant_count(), 4);
    }

    #[test]
    fn test_reverse_mode_shard_map_benchmarks_report_unsupported_operation() {
        for error in [emit_grad_around_shard_map().unwrap_err(), emit_shard_map_grad_inside().unwrap_err()] {
            assert_eq!(error.to_string(), "reverse-mode shard_map IR benchmarks are not implemented");
        }
    }
}
