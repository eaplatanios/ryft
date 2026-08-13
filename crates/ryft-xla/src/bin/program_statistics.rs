//! Emits backend-neutral structural [`ProgramStatistics`] records for a fixed registry of traced Ryft workloads.
//!
//! This binary is the Rust side of the Ryft/JAX structural-statistics comparison workflow (see
//! `python/scripts/compare_program_statistics_with_jax.py`). It prints deterministic JSON records containing only
//! case metadata and [`ProgramStatistics`]; no raw IR is emitted. `--list` prints one case ID per line, and repeated
//! `--case ID` arguments select individual cases. The case registry and its record and error types are private to
//! this binary and are not public library API.

use std::env;
use std::ops::{Add, Mul};

use serde::Serialize;
use thiserror::Error;

use ryft_core::{
    Array, ArrayOperation, ArrayType, Context, DataType, DifferentiationError, Dimension, Dot, DotDimensionNumbers,
    EagerContext, ForwardModeDifferentiate, LogicalMesh, MeshAxis, MeshAxisType, Program, ProgramError,
    ProgramStatistics, ReverseModeDifferentiate, Shape, Sharding, ShardingDimension, Sin,
};
use ryft_xla::experimental::{ShardMapTraceError, ShardMapTracer, TracedXlaProgram, shard_map, trace};

/// Error type returned by the program statistics emitters.
#[derive(Debug, Error)]
enum StatisticsError {
    /// Wrapper around tracing failures while building a case's program.
    #[error("{0}")]
    Trace(#[from] ProgramError),

    /// Wrapper around differentiation failures while building a case's program.
    #[error("{0}")]
    Differentiation(#[from] DifferentiationError),

    /// Wrapper around shard-map tracing failures while building a case's program.
    #[error("{0}")]
    ShardMapTrace(#[from] ShardMapTraceError),

    /// Error returned when a requested case ID is unknown.
    #[error("unknown program statistics case '{case_id}'")]
    UnknownCase {
        /// Unknown case identifier.
        case_id: String,
    },
}

/// One emitted program statistics record.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct ProgramStatisticsRecord {
    /// Stable case identifier.
    case_id: &'static str,

    /// High-level category, either `scalar` or `xla`.
    category: &'static str,

    /// Traced surface, such as `jit`, `vjp_pullback`, or `program`.
    surface: &'static str,

    /// Structural statistics of the case's exact traced program.
    statistics: ProgramStatistics,
}

/// Descriptor for one program statistics case.
#[derive(Copy, Clone)]
struct ProgramStatisticsCase {
    /// Stable case identifier.
    case_id: &'static str,

    /// High-level category, either `scalar` or `xla`.
    category: &'static str,

    /// Traced surface, such as `jit`, `vjp_pullback`, or `program`.
    surface: &'static str,

    /// Callback that traces the case's workload and returns its structural statistics.
    emit: fn() -> Result<ProgramStatistics, StatisticsError>,
}

/// Returns the stable case registry, asserting that its case IDs are unique.
fn registry() -> Vec<ProgramStatisticsCase> {
    let cases = vec![
        ProgramStatisticsCase {
            case_id: "scalar_bilinear_sin_jit",
            category: "scalar",
            surface: "jit",
            emit: emit_scalar_bilinear_sin_jit,
        },
        ProgramStatisticsCase {
            case_id: "scalar_bilinear_sin_vjp_pullback",
            category: "scalar",
            surface: "vjp_pullback",
            emit: emit_scalar_bilinear_sin_vjp_pullback,
        },
        ProgramStatisticsCase {
            case_id: "scalar_quartic_plus_sin_grad",
            category: "scalar",
            surface: "grad",
            emit: emit_scalar_quartic_plus_sin_grad,
        },
        ProgramStatisticsCase {
            case_id: "scalar_quartic_plus_sin_value_and_gradient",
            category: "scalar",
            surface: "value_and_gradient",
            emit: emit_scalar_quartic_plus_sin_value_and_gradient,
        },
        ProgramStatisticsCase {
            case_id: "scalar_quartic_plus_sin_linearize_pushforward",
            category: "scalar",
            surface: "linearize_pushforward",
            emit: emit_scalar_quartic_plus_sin_linearize_pushforward,
        },
        ProgramStatisticsCase {
            case_id: "shard_map_basic",
            category: "xla",
            surface: "program",
            emit: emit_shard_map_basic,
        },
        ProgramStatisticsCase {
            case_id: "shard_map_matmul",
            category: "xla",
            surface: "program",
            emit: emit_shard_map_matmul,
        },
        ProgramStatisticsCase {
            case_id: "nested_shard_map",
            category: "xla",
            surface: "program",
            emit: emit_nested_shard_map,
        },
    ];
    for (index, case) in cases.iter().enumerate() {
        assert!(
            cases[..index].iter().all(|previous| previous.case_id != case.case_id),
            "duplicate program statistics case ID '{}'",
            case.case_id,
        );
    }
    cases
}

/// Emits the records for the requested case IDs, in deterministic `(case_id, surface)` order. When `case_ids` is
/// empty, all registered cases are emitted.
///
/// # Parameters
///
///   - `case_ids`: Optional exact case IDs to emit.
fn collect_records(case_ids: &[String]) -> Result<Vec<ProgramStatisticsRecord>, StatisticsError> {
    let all_cases = registry();
    let selected_cases = if case_ids.is_empty() {
        all_cases
    } else {
        case_ids
            .iter()
            .map(|case_id| {
                all_cases
                    .iter()
                    .copied()
                    .find(|case| case.case_id == case_id)
                    .ok_or_else(|| StatisticsError::UnknownCase { case_id: case_id.clone() })
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut records = Vec::new();
    for case in selected_cases {
        records.push(ProgramStatisticsRecord {
            case_id: case.case_id,
            category: case.category,
            surface: case.surface,
            statistics: (case.emit)()?,
        });
    }
    records.sort_by(|left, right| left.case_id.cmp(right.case_id).then(left.surface.cmp(right.surface)));
    Ok(records)
}

/// Statistics helper used by the scalar higher-order case family.
///
/// # Parameters
///
///   - `x`: Rank-zero array input.
fn quartic_plus_sin<T: Clone + Sin + Add<Output = T> + Mul<Output = T>>(x: T) -> T {
    x.clone() * x.clone() * x.clone() * x.clone() + x.sin().unwrap()
}

/// Emits the plain JIT scalar bilinear case.
fn emit_scalar_bilinear_sin_jit() -> Result<ProgramStatistics, StatisticsError> {
    let (_, program): (Array, Program<Array, ArrayOperation<Array>, (Array, Array), Array>) =
        EagerContext::<Array, ArrayOperation<Array>>::new().interpret_and_trace(
            |inputs| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()?),
            (Array::scalar(2.0), Array::scalar(3.0)),
        )?;
    Ok(program.statistics())
}

/// Emits the staged scalar bilinear pullback case.
fn emit_scalar_bilinear_sin_vjp_pullback() -> Result<ProgramStatistics, StatisticsError> {
    let (_, pullback): (Array, _) = EagerContext::<Array, ArrayOperation<Array>>::new().vjp(
        |inputs, ()| Ok(inputs.0.clone() * inputs.1 + inputs.0.sin()?),
        (Array::scalar(2.0), Array::scalar(3.0)),
        (),
    )?;
    let (pullback, _residuals) = pullback.into_parts();
    Ok(pullback.statistics())
}

/// Emits the staged scalar reverse-mode gradient case.
fn emit_scalar_quartic_plus_sin_grad() -> Result<ProgramStatistics, StatisticsError> {
    let (_, program): (Array, Program<Array, ArrayOperation<Array>, Array, Array>) =
        EagerContext::<Array, ArrayOperation<Array>>::new().interpret_and_trace(
            |x| {
                let context = x.context().clone();
                // `interpret_and_trace` fixes its closure error to `ProgramError`, so fold the inner gradient's
                // differentiation error into a program error. A non-scalar gradient output cannot occur for this
                // scalar case function.
                context.gradient(|input, ()| quartic_plus_sin(input), x, ()).map_err(|error| match error {
                    DifferentiationError::Program(error) => error,
                    error => ProgramError::MalformedProgram(error.to_string()),
                })
            },
            Array::scalar(2.0),
        )?;
    Ok(program.statistics())
}

/// Emits the staged scalar value-and-gradient case.
fn emit_scalar_quartic_plus_sin_value_and_gradient() -> Result<ProgramStatistics, StatisticsError> {
    let (_, program): ((Array, Array), Program<Array, ArrayOperation<Array>, Array, (Array, Array)>) =
        EagerContext::<Array, ArrayOperation<Array>>::new().interpret_and_trace(
            |x| {
                let context = x.context().clone();
                // `interpret_and_trace` fixes its closure error to `ProgramError`, so fold the inner gradient's
                // differentiation error into a program error. A non-scalar gradient output cannot occur for this
                // scalar case function.
                context.value_and_gradient(|input, ()| quartic_plus_sin(input), x, ()).map_err(|error| match error {
                    DifferentiationError::Program(error) => error,
                    error => ProgramError::MalformedProgram(error.to_string()),
                })
            },
            Array::scalar(2.0),
        )?;
    Ok(program.statistics())
}

/// Emits the directly linearized pushforward of `f(x) = x⁴ + sin(x)`, closed over its lifted residuals.
fn emit_scalar_quartic_plus_sin_linearize_pushforward() -> Result<ProgramStatistics, StatisticsError> {
    let context = EagerContext::<Array, ArrayOperation<Array>>::new();
    let (_, pushforward) = context.linearize(
        |x, ()| Ok(x.clone() * x.clone() * x.clone() * x.clone() + x.sin()?),
        Array::scalar(2.0),
        (),
    )?;
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
        Array::scalar(1.0),
    )?;
    Ok(closed_pushforward.statistics())
}

/// Returns the canonical single-axis manual mesh used by the shard-map cases.
fn shard_map_mesh() -> LogicalMesh {
    LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
}

/// Returns a rank-1 case array type.
///
/// # Parameters
///
///   - `size`: Static vector length.
fn vector_type(size: usize) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(size)]))
}

/// Returns a rank-2 case array type.
///
/// # Parameters
///
///   - `rows`: Matrix row count.
///   - `cols`: Matrix column count.
fn matrix_type(rows: usize, cols: usize) -> ArrayType {
    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(rows), Dimension::Static(cols)]))
}

/// Emits the basic traced `shard_map` case.
fn emit_shard_map_basic() -> Result<ProgramStatistics, StatisticsError> {
    let mesh = shard_map_mesh();
    let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let mesh = mesh.clone();
            move |x| {
                shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                    |local_x: ShardMapTracer| {
                        local_x.sin().unwrap_or_else(|error| panic!("basic shard_map case should trace sine: {error}"))
                    },
                    x,
                    mesh.clone(),
                    sharding.clone(),
                    sharding.clone(),
                )
                .unwrap_or_else(|error| panic!("basic shard_map case should trace: {error}"))
            }
        },
        vector_type(8),
    )?;
    Ok(traced.statistics())
}

/// Emits the traced `shard_map` matrix-multiplication case.
fn emit_shard_map_matmul() -> Result<ProgramStatistics, StatisticsError> {
    let mesh = shard_map_mesh();
    let lhs_spec =
        Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap();
    let rhs_spec = Sharding::replicated(mesh.clone(), 2);
    let out_spec = lhs_spec.clone();
    let traced: TracedXlaProgram<(ArrayType, ArrayType), ArrayType> = trace(
        {
            let mesh = mesh.clone();
            move |inputs| {
                shard_map::<_, (ShardMapTracer, ShardMapTracer), ArrayType, ShardMapTracer>(
                    |(lhs, rhs): (ShardMapTracer, ShardMapTracer)| lhs.dot(&rhs, &DotDimensionNumbers::matmul()),
                    inputs,
                    mesh.clone(),
                    (lhs_spec.clone(), rhs_spec.clone()),
                    out_spec.clone(),
                )
                .unwrap_or_else(|error| panic!("matmul shard_map case should trace: {error}"))
            }
        },
        (matrix_type(8, 4), matrix_type(4, 2)),
    )?;
    Ok(traced.statistics())
}

/// Emits the nested traced `shard_map` case.
fn emit_nested_shard_map() -> Result<ProgramStatistics, StatisticsError> {
    let outer_mesh = LogicalMesh::new(vec![
        MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
        MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
    ])
    .unwrap();
    let inner_mesh = LogicalMesh::new(vec![
        MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
        MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
    ])
    .unwrap();
    let outer_sharding = Sharding::new(outer_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    let inner_sharding = Sharding::new(inner_mesh.clone(), vec![ShardingDimension::sharded(["y"])]).unwrap();
    let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
        {
            let outer_mesh = outer_mesh.clone();
            let inner_mesh = inner_mesh.clone();
            move |x| {
                shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                    {
                        let inner_mesh = inner_mesh.clone();
                        move |outer_x: ShardMapTracer| {
                            let nested = shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                                |inner_x: ShardMapTracer| inner_x.clone() + inner_x,
                                outer_x.clone(),
                                inner_mesh.clone(),
                                inner_sharding.clone(),
                                inner_sharding.clone(),
                            )
                            .unwrap_or_else(|error| {
                                panic!("nested shard_map case should trace the inner shard_map: {error}")
                            });
                            nested + outer_x
                        }
                    },
                    x,
                    outer_mesh.clone(),
                    outer_sharding.clone(),
                    outer_sharding.clone(),
                )
                .unwrap_or_else(|error| panic!("nested shard_map case should trace the outer shard_map: {error}"))
            }
        },
        vector_type(8),
    )?;
    Ok(traced.statistics())
}

/// Runs the program statistics emitter and prints JSON records to stdout.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut case_ids = Vec::new();
    let mut list_cases = false;

    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--list" => list_cases = true,
            "--case" => {
                let case_id = arguments.next().ok_or("expected a case ID after --case")?;
                case_ids.push(case_id);
            }
            other => {
                return Err(format!("unknown argument '{other}'").into());
            }
        }
    }

    if list_cases {
        for case in registry() {
            println!("{}", case.case_id);
        }
        return Ok(());
    }

    let records = collect_records(case_ids.as_slice())?;
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    /// Returns the emitted statistics of the case with the provided ID as a JSON value.
    ///
    /// # Parameters
    ///
    ///   - `case_id`: Stable case identifier.
    fn case_statistics(case_id: &str) -> serde_json::Value {
        let records = collect_records(&[case_id.to_string()]).unwrap();
        assert_eq!(records.len(), 1);
        serde_json::to_value(&records[0].statistics).unwrap()
    }

    /// Verifies that the registry contains exactly the expected case IDs, in deterministic order.
    #[test]
    fn test_registry_case_ids_are_unique_and_deterministic() {
        let case_ids = registry().into_iter().map(|case| case.case_id).collect::<Vec<_>>();
        assert_eq!(
            case_ids,
            vec![
                "scalar_bilinear_sin_jit",
                "scalar_bilinear_sin_vjp_pullback",
                "scalar_quartic_plus_sin_grad",
                "scalar_quartic_plus_sin_value_and_gradient",
                "scalar_quartic_plus_sin_linearize_pushforward",
                "shard_map_basic",
                "shard_map_matmul",
                "nested_shard_map",
            ],
        );
    }

    /// Verifies that selecting an unknown case fails with an error naming the offending ID.
    #[test]
    fn test_collect_records_rejects_unknown_case() {
        let error = collect_records(&["missing_case".to_string()]).unwrap_err();
        assert_eq!(error.to_string(), "unknown program statistics case 'missing_case'");
    }

    /// Verifies the exact expected statistics for every scalar case. These values are the primary structural
    /// regression guard for the tracing and transform pipeline; every expectation below was hand-verified against
    /// the corresponding staged program rendering before being pinned.
    #[test]
    fn test_scalar_case_statistics() {
        assert_eq!(
            case_statistics("scalar_bilinear_sin_jit"),
            json!({
                "regions": [
                    {
                        "input_count": 2,
                        "output_count": 1,
                        "constant_count": 0,
                        "instruction_count": 3,
                        "operation_counts": { "add": 1, "mul": 1, "sin": 1 },
                        "maximum_output_dependency_depth": 2,
                        "attached_regions": [],
                    },
                ],
            }),
        );
        assert_eq!(
            case_statistics("scalar_bilinear_sin_vjp_pullback"),
            json!({
                "regions": [
                    {
                        "input_count": 4,
                        "output_count": 2,
                        "constant_count": 0,
                        "instruction_count": 4,
                        "operation_counts": { "add": 1, "mul": 3 },
                        "maximum_output_dependency_depth": 2,
                        "attached_regions": [],
                    },
                ],
            }),
        );
        assert_eq!(
            case_statistics("scalar_quartic_plus_sin_grad"),
            json!({
                "regions": [
                    {
                        "input_count": 1,
                        "output_count": 1,
                        "constant_count": 0,
                        "instruction_count": 15,
                        "operation_counts": { "add": 4, "cos": 1, "mul": 9, "one": 1 },
                        "maximum_output_dependency_depth": 7,
                        "attached_regions": [],
                    },
                ],
            }),
        );
        assert_eq!(
            case_statistics("scalar_quartic_plus_sin_value_and_gradient"),
            json!({
                "regions": [
                    {
                        "input_count": 1,
                        "output_count": 2,
                        "constant_count": 0,
                        "instruction_count": 18,
                        "operation_counts": { "add": 5, "cos": 1, "mul": 10, "one": 1, "sin": 1 },
                        "maximum_output_dependency_depth": 7,
                        "attached_regions": [],
                    },
                ],
            }),
        );
        assert_eq!(
            case_statistics("scalar_quartic_plus_sin_linearize_pushforward"),
            json!({
                "regions": [
                    {
                        "input_count": 1,
                        "output_count": 1,
                        "constant_count": 4,
                        "instruction_count": 11,
                        "operation_counts": { "add": 4, "mul": 7 },
                        "maximum_output_dependency_depth": 7,
                        "attached_regions": [],
                    },
                ],
            }),
        );
    }

    /// Verifies the exact expected statistics for every shard-map case, including the nested case's shared-arena
    /// attachment edges. The `shard_map_basic` expectation is anchored against a program rendering by the
    /// `TracedXlaProgram::statistics` owner-module test in `experimental::shard_map`; the other two fixtures were
    /// pinned from the binary's emitted values after checking them for internal consistency, so they serve as
    /// change detectors rather than independently derived ground truth.
    #[test]
    fn test_shard_map_case_statistics() {
        assert_eq!(
            case_statistics("shard_map_basic"),
            json!({
                "regions": [
                    {
                        "input_count": 1,
                        "output_count": 1,
                        "constant_count": 0,
                        "instruction_count": 1,
                        "operation_counts": { "sin": 1 },
                        "maximum_output_dependency_depth": 1,
                        "attached_regions": [],
                    },
                    {
                        "input_count": 1,
                        "output_count": 1,
                        "constant_count": 0,
                        "instruction_count": 1,
                        "operation_counts": { "shard_map": 1 },
                        "maximum_output_dependency_depth": 1,
                        "attached_regions": [
                            {
                                "instruction_index": 0,
                                "operation": "shard_map",
                                "region_slot": "body",
                                "region_role": "computation",
                                "region_index": 0,
                            },
                        ],
                    },
                ],
            }),
        );
        assert_eq!(
            case_statistics("shard_map_matmul"),
            json!({
                "regions": [
                    {
                        "input_count": 2,
                        "output_count": 1,
                        "constant_count": 0,
                        "instruction_count": 1,
                        "operation_counts": { "dot": 1 },
                        "maximum_output_dependency_depth": 1,
                        "attached_regions": [],
                    },
                    {
                        "input_count": 2,
                        "output_count": 1,
                        "constant_count": 0,
                        "instruction_count": 1,
                        "operation_counts": { "shard_map": 1 },
                        "maximum_output_dependency_depth": 1,
                        "attached_regions": [
                            {
                                "instruction_index": 0,
                                "operation": "shard_map",
                                "region_slot": "body",
                                "region_role": "computation",
                                "region_index": 0,
                            },
                        ],
                    },
                ],
            }),
        );
        assert_eq!(
            case_statistics("nested_shard_map"),
            json!({
                "regions": [
                    {
                        "input_count": 1,
                        "output_count": 1,
                        "constant_count": 0,
                        "instruction_count": 1,
                        "operation_counts": { "add": 1 },
                        "maximum_output_dependency_depth": 1,
                        "attached_regions": [],
                    },
                    {
                        "input_count": 1,
                        "output_count": 1,
                        "constant_count": 0,
                        "instruction_count": 2,
                        "operation_counts": { "add": 1, "shard_map": 1 },
                        "maximum_output_dependency_depth": 2,
                        "attached_regions": [
                            {
                                "instruction_index": 0,
                                "operation": "shard_map",
                                "region_slot": "body",
                                "region_role": "computation",
                                "region_index": 0,
                            },
                        ],
                    },
                    {
                        "input_count": 1,
                        "output_count": 1,
                        "constant_count": 0,
                        "instruction_count": 1,
                        "operation_counts": { "shard_map": 1 },
                        "maximum_output_dependency_depth": 1,
                        "attached_regions": [
                            {
                                "instruction_index": 0,
                                "operation": "shard_map",
                                "region_slot": "body",
                                "region_role": "computation",
                                "region_index": 1,
                            },
                        ],
                    },
                ],
            }),
        );
    }

    /// Verifies the exact serialized record schema for one full record.
    #[test]
    fn test_record_schema() {
        let records = collect_records(&["scalar_bilinear_sin_jit".to_string()]).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(
            serde_json::to_value(&records[0]).unwrap(),
            json!({
                "case_id": "scalar_bilinear_sin_jit",
                "category": "scalar",
                "surface": "jit",
                "statistics": {
                    "regions": [
                        {
                            "input_count": 2,
                            "output_count": 1,
                            "constant_count": 0,
                            "instruction_count": 3,
                            "operation_counts": { "add": 1, "mul": 1, "sin": 1 },
                            "maximum_output_dependency_depth": 2,
                            "attached_regions": [],
                        },
                    ],
                },
            }),
        );
    }
}
