//! Diagnostic IR benchmarking support for `tracing_v2`.
//!
//! This module collects a named suite of staged Ryft programs and emits textual MLIR artifacts plus
//! normalized structural summaries that can be compared against external MLIR producers such as
//! JAX StableHLO lowering.

use std::{collections::BTreeMap, fmt::Display};

use serde::Serialize;
use thiserror::Error;

use crate::parameters::Parameterized;
use crate::types::ArrayType;

use super::{Atom, Graph, Op, TraceError, Traceable};

/// Error type returned by the IR benchmark tooling.
#[derive(Debug, Error)]
pub enum BenchmarkError {
    /// Wrapper around tracing failures while building or summarizing a benchmark case.
    #[error("{0}")]
    Trace(#[from] TraceError),

    /// Wrapper around a boxed error returned by an external benchmark case provider.
    #[error("{0}")]
    External(#[from] Box<dyn std::error::Error>),

    /// Error returned when the requested case ID is unknown.
    #[error("unknown IR benchmark case '{case_id}'")]
    UnknownCase {
        /// Unknown case identifier.
        case_id: String,
    },
}

/// Structural summary of one immediate nested region.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct IrNestedRegionSummary {
    /// Stable label describing the nested region relative to its parent op.
    pub label: String,

    /// Number of input leaves accepted by the nested region.
    pub input_leaf_count: usize,

    /// Number of output leaves produced by the nested region.
    pub output_leaf_count: usize,

    /// Number of equations in the nested region.
    pub equation_count: usize,

    /// Number of constant atoms in the nested region.
    pub constant_count: usize,

    /// Histogram of normalized operation names in the nested region.
    pub op_histogram: BTreeMap<String, usize>,

    /// Total number of nested regions reachable from this nested region.
    pub nested_region_count: usize,

    /// Maximum data-dependency depth of the nested region outputs.
    pub max_dependency_depth: usize,
}

/// Structural summary of one IR artifact.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct IrBenchmarkSummary {
    /// Number of input leaves accepted by the artifact.
    pub input_leaf_count: usize,

    /// Number of output leaves produced by the artifact.
    pub output_leaf_count: usize,

    /// Number of equations in the artifact.
    pub equation_count: usize,

    /// Number of constant atoms in the artifact.
    pub constant_count: usize,

    /// Histogram of normalized operation names.
    pub op_histogram: BTreeMap<String, usize>,

    /// Total number of nested regions reachable from this artifact.
    pub nested_region_count: usize,

    /// Immediate nested regions attached to the artifact's operations.
    pub nested_regions: Vec<IrNestedRegionSummary>,

    /// Maximum data-dependency depth of the artifact outputs.
    pub max_dependency_depth: usize,
}

/// One emitted benchmark artifact.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct IrBenchmarkRecord {
    /// Stable benchmark case identifier.
    pub case_id: String,

    /// High-level category such as `scalar`, `matrix`, or `xla`.
    pub category: String,

    /// Artifact surface such as `jit`, `vjp_pullback`, `program`, or `shard_map_mlir`.
    pub surface: String,

    /// Full raw textual IR artifact.
    pub raw_ir: String,

    /// Normalized structural summary derived from the staged IR.
    pub summary: IrBenchmarkSummary,
}

/// Descriptor for one benchmark case.
#[derive(Clone, Copy)]
pub struct BenchmarkCase {
    /// Stable case identifier.
    pub case_id: &'static str,

    /// Callback that emits one or more records for the case.
    pub emit: fn() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError>,
}

/// Returns the stable set of benchmark case IDs supported by the Rust-side emitter.
///
/// # Parameters
///
///   - `extra_cases`: Additional benchmark cases to include (e.g., from `ryft-xla`).
pub fn benchmark_case_ids(extra_cases: &[BenchmarkCase]) -> Vec<&'static str> {
    let mut cases = tracing_v2_cases();
    cases.extend_from_slice(extra_cases);
    cases.into_iter().map(|case| case.case_id).collect()
}

/// Emits the requested benchmark records.
///
/// When `case_ids` is empty, all known benchmark cases are emitted.
///
/// # Parameters
///
///   - `extra_cases`: Additional benchmark cases to include (e.g., from `ryft-xla`).
///   - `case_ids`: Optional exact case IDs to emit.
pub fn collect_ir_benchmark_records(
    extra_cases: &[BenchmarkCase],
    case_ids: &[String],
) -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
    let mut all_cases = tracing_v2_cases();
    all_cases.extend_from_slice(extra_cases);

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
                    .ok_or_else(|| BenchmarkError::UnknownCase { case_id: case_id.clone() })
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut records = Vec::new();
    for case in selected_cases {
        records.extend((case.emit)()?);
    }
    records.sort_by(|left, right| left.case_id.cmp(&right.case_id).then(left.surface.cmp(&right.surface)));
    Ok(records)
}

/// Builds one benchmark record from its parts.
///
/// # Parameters
///
///   - `case_id`: Stable benchmark case identifier.
///   - `category`: High-level category of the case.
///   - `surface`: Artifact surface for the emitted record.
///   - `raw_ir`: Full raw textual IR artifact.
///   - `summary`: Normalized structural summary for the artifact.
pub fn record(
    case_id: &'static str,
    category: &'static str,
    surface: &'static str,
    raw_ir: String,
    summary: IrBenchmarkSummary,
) -> IrBenchmarkRecord {
    IrBenchmarkRecord {
        case_id: case_id.to_string(),
        category: category.to_string(),
        surface: surface.to_string(),
        raw_ir,
        summary,
    }
}

/// Normalizes an operation name onto the shared comparison vocabulary.
///
/// # Parameters
///
///   - `name`: Operation name to normalize.
pub(crate) fn normalize_op_name(name: &str) -> String {
    match name {
        "add" | "add_any" => "add".to_string(),
        "mul" => "mul".to_string(),
        "neg" => "neg".to_string(),
        "sin" => "sin".to_string(),
        "cos" => "cos".to_string(),
        "matmul" | "dot_general" | "left_matmul" | "right_matmul" => "matmul".to_string(),
        "matrix_transpose" | "linear_matrix_transpose" | "transpose" => "transpose".to_string(),
        "scale" => "scale".to_string(),
        "const" | "constant" => "const".to_string(),
        "shard_map" | "linear_shard_map" => "shard_map".to_string(),
        other => format!("unknown:{other}"),
    }
}

/// Summarizes one staged graph and its immediate nested regions.
///
/// # Parameters
///
///   - `graph`: Graph to summarize.
///   - `nested_regions_for_op`: Callback that returns the immediate nested regions carried by one
///     staged op.
pub fn summarize_graph<V, Input, Output, O, F>(
    graph: &Graph<O, V, Input, Output>,
    nested_regions_for_op: F,
) -> Result<IrBenchmarkSummary, BenchmarkError>
where
    V: Traceable<ArrayType>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Clone + Display + Op,
    F: Fn(&O) -> Result<Vec<IrNestedRegionSummary>, BenchmarkError>,
{
    let mut op_histogram = BTreeMap::new();
    let mut nested_regions = Vec::new();
    let mut depth_by_atom = vec![0usize; graph.atom_count()];

    for (atom_id, atom) in (0..graph.atom_count()).filter_map(|atom_id| graph.atom(atom_id).map(|atom| (atom_id, atom)))
    {
        if matches!(atom, Atom::Input { .. } | Atom::Constant { .. }) {
            depth_by_atom[atom_id] = 0;
        }
    }

    for equation in graph.equations() {
        let normalized_name = normalize_op_name(equation.op.name());
        *op_histogram.entry(normalized_name).or_insert(0) += 1;

        let input_depth = equation.inputs.iter().map(|input| depth_by_atom[*input]).max().unwrap_or(0);
        for output in equation.outputs.iter().copied() {
            depth_by_atom[output] = input_depth + 1;
        }

        nested_regions.extend(nested_regions_for_op(&equation.op)?);
    }

    let nested_region_count = nested_regions.len()
        + nested_regions.iter().map(|nested_region| nested_region.nested_region_count).sum::<usize>();
    let max_dependency_depth = graph.outputs().iter().map(|output| depth_by_atom[*output]).max().unwrap_or(0);

    Ok(IrBenchmarkSummary {
        input_leaf_count: graph.input_atoms().len(),
        output_leaf_count: graph.outputs().len(),
        equation_count: graph.equations().len(),
        constant_count: (0..graph.atom_count())
            .filter_map(|atom_id| graph.atom(atom_id))
            .filter(|atom| matches!(atom, Atom::Constant { .. }))
            .count(),
        op_histogram,
        nested_region_count,
        nested_regions,
        max_dependency_depth,
    })
}

/// Converts one nested-region summary from a child graph into the public nested-region shape.
///
/// # Parameters
///
///   - `label`: Stable nested-region label.
///   - `summary`: Child graph summary.
pub fn nested_region(label: &'static str, summary: IrBenchmarkSummary) -> IrNestedRegionSummary {
    IrNestedRegionSummary {
        label: label.to_string(),
        input_leaf_count: summary.input_leaf_count,
        output_leaf_count: summary.output_leaf_count,
        equation_count: summary.equation_count,
        constant_count: summary.constant_count,
        op_histogram: summary.op_histogram,
        nested_region_count: summary.nested_region_count,
        max_dependency_depth: summary.max_dependency_depth,
    }
}

/// Returns the tracing-only benchmark cases.
pub(crate) fn tracing_v2_cases() -> Vec<BenchmarkCase> {
    super::benchmark_support::cases()
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::tracing_v2::{CompiledFunction, FloatExt, JitTracer, OneLike, jit};

    use super::*;

    /// Summarizes a small scalar graph and verifies the structural metrics.
    #[test]
    fn test_summarize_graph_counts_constants_and_depth() {
        let (_, compiled): (f64, CompiledFunction<ArrayType, f64, f64, f64>) = jit(
            |x: JitTracer<ArrayType, f64>| {
                let with_constant = x.clone() + x.one_like();
                with_constant.sin()
            },
            2.0f64,
        )
        .unwrap();

        let summary = summarize_graph(compiled.graph(), |_| Ok(Vec::new())).unwrap();
        assert_eq!(
            summary,
            IrBenchmarkSummary {
                input_leaf_count: 1,
                output_leaf_count: 1,
                equation_count: 2,
                constant_count: 1,
                op_histogram: BTreeMap::from([("add".to_string(), 1usize), ("sin".to_string(), 1usize),]),
                nested_region_count: 0,
                nested_regions: Vec::new(),
                max_dependency_depth: 2,
            }
        );
    }

    /// Verifies the stable benchmark case registry.
    #[test]
    fn test_benchmark_case_registry_contains_expected_ids() {
        let case_ids = benchmark_case_ids(&[]);
        assert!(case_ids.contains(&"scalar_bilinear_sin_jit"));
        assert!(case_ids.contains(&"scalar_bilinear_sin_jvp"));
        assert!(case_ids.contains(&"scalar_bilinear_sin_vjp_pullback"));
        assert!(case_ids.contains(&"scalar_quartic_plus_sin_grad"));
        assert!(case_ids.contains(&"scalar_quartic_plus_sin_value_and_grad"));
        assert!(case_ids.contains(&"scalar_quartic_plus_sin_linearize_pushforward"));
        assert!(case_ids.contains(&"scalar_quartic_plus_sin_hessian_style"));
        assert!(case_ids.contains(&"scalar_grad_of_vmap"));
        assert!(case_ids.contains(&"scalar_vmap_of_grad"));
        #[cfg(feature = "ndarray")]
        assert!(case_ids.contains(&"matrix_matmul_jit"));
        #[cfg(feature = "ndarray")]
        assert!(case_ids.contains(&"matrix_matmul_vjp_pullback"));
        #[cfg(feature = "ndarray")]
        assert!(case_ids.contains(&"matrix_three_matmul_sine_hessian_style"));
    }

    /// Verifies that exact case filtering emits only the requested case.
    #[test]
    fn test_collect_ir_benchmark_records_filters_by_case_id() {
        let records = collect_ir_benchmark_records(&[], &["scalar_bilinear_sin_jit".to_string()]).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].case_id, "scalar_bilinear_sin_jit");
        assert_eq!(records[0].surface, "jit");
        assert_eq!(
            records[0].raw_ir.trim_end(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %0 %1
                    %3:f64[] = sin %0
                    %4:f64[] = add %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }
}
