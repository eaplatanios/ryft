use std::collections::BTreeMap;

use serde::Serialize;
use thiserror::Error;

use crate::operations::Operation;
use crate::operations::arithmetic::{ADD_OPERATION_NAME, MUL_OPERATION_NAME};
use crate::parameters::Parameterized;
use crate::tracing::{Atom, AtomId, Program, Traceable, TracingError};
use crate::types::Type;

/// Error type returned by the IR benchmark tooling.
#[derive(Debug, Error)]
pub enum BenchmarkError {
    /// Wrapper around tracing failures while building or summarizing a benchmark case.
    #[error("{0}")]
    Trace(#[from] TracingError),

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
    label: String,

    /// Number of input leaves accepted by the nested region.
    input_leaf_count: usize,

    /// Number of output leaves produced by the nested region.
    output_leaf_count: usize,

    /// Number of instructions in the nested region.
    instruction_count: usize,

    /// Number of constant atoms in the nested region.
    constant_count: usize,

    /// Histogram of normalized operation names in the nested region.
    op_histogram: BTreeMap<String, usize>,

    /// Total number of nested regions reachable from this nested region.
    nested_region_count: usize,

    /// Maximum data-dependency depth of the nested region outputs.
    max_dependency_depth: usize,
}

impl IrNestedRegionSummary {
    /// Creates a summary of one immediate nested region.
    #[inline]
    pub fn new(
        label: impl Into<String>,
        input_leaf_count: usize,
        output_leaf_count: usize,
        instruction_count: usize,
        constant_count: usize,
        op_histogram: BTreeMap<String, usize>,
        nested_region_count: usize,
        max_dependency_depth: usize,
    ) -> Self {
        Self {
            label: label.into(),
            input_leaf_count,
            output_leaf_count,
            instruction_count,
            constant_count,
            op_histogram,
            nested_region_count,
            max_dependency_depth,
        }
    }

    /// Returns the stable label describing the nested region relative to its parent op.
    #[inline]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the number of input leaves accepted by the nested region.
    #[inline]
    pub fn input_leaf_count(&self) -> usize {
        self.input_leaf_count
    }

    /// Returns the number of output leaves produced by the nested region.
    #[inline]
    pub fn output_leaf_count(&self) -> usize {
        self.output_leaf_count
    }

    /// Returns the number of instructions in the nested region.
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.instruction_count
    }

    /// Returns the number of constant atoms in the nested region.
    #[inline]
    pub fn constant_count(&self) -> usize {
        self.constant_count
    }

    /// Returns the histogram of normalized operation names in the nested region.
    #[inline]
    pub fn op_histogram(&self) -> &BTreeMap<String, usize> {
        &self.op_histogram
    }

    /// Returns the total number of nested regions reachable from this nested region.
    #[inline]
    pub fn nested_region_count(&self) -> usize {
        self.nested_region_count
    }

    /// Returns the maximum data-dependency depth of the nested region outputs.
    #[inline]
    pub fn max_dependency_depth(&self) -> usize {
        self.max_dependency_depth
    }
}

/// Structural summary of one IR artifact.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct IrBenchmarkSummary {
    /// Number of input leaves accepted by the artifact.
    input_leaf_count: usize,

    /// Number of output leaves produced by the artifact.
    output_leaf_count: usize,

    /// Number of instructions in the artifact.
    instruction_count: usize,

    /// Number of constant atoms in the artifact.
    constant_count: usize,

    /// Histogram of normalized operation names.
    op_histogram: BTreeMap<String, usize>,

    /// Total number of nested regions reachable from this artifact.
    nested_region_count: usize,

    /// Immediate nested regions attached to the artifact's operations.
    nested_regions: Vec<IrNestedRegionSummary>,

    /// Maximum data-dependency depth of the artifact outputs.
    max_dependency_depth: usize,
}

impl IrBenchmarkSummary {
    /// Creates a structural summary of one IR artifact.
    #[inline]
    pub fn new(
        input_leaf_count: usize,
        output_leaf_count: usize,
        instruction_count: usize,
        constant_count: usize,
        op_histogram: BTreeMap<String, usize>,
        nested_region_count: usize,
        nested_regions: Vec<IrNestedRegionSummary>,
        max_dependency_depth: usize,
    ) -> Self {
        Self {
            input_leaf_count,
            output_leaf_count,
            instruction_count,
            constant_count,
            op_histogram,
            nested_region_count,
            nested_regions,
            max_dependency_depth,
        }
    }

    /// Returns the number of input leaves accepted by the artifact.
    #[inline]
    pub fn input_leaf_count(&self) -> usize {
        self.input_leaf_count
    }

    /// Returns the number of output leaves produced by the artifact.
    #[inline]
    pub fn output_leaf_count(&self) -> usize {
        self.output_leaf_count
    }

    /// Returns the number of instructions in the artifact.
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.instruction_count
    }

    /// Returns the number of constant atoms in the artifact.
    #[inline]
    pub fn constant_count(&self) -> usize {
        self.constant_count
    }

    /// Returns the histogram of normalized operation names.
    #[inline]
    pub fn op_histogram(&self) -> &BTreeMap<String, usize> {
        &self.op_histogram
    }

    /// Returns the total number of nested regions reachable from this artifact.
    #[inline]
    pub fn nested_region_count(&self) -> usize {
        self.nested_region_count
    }

    /// Returns the immediate nested regions attached to the artifact's operations.
    #[inline]
    pub fn nested_regions(&self) -> &[IrNestedRegionSummary] {
        &self.nested_regions
    }

    /// Returns the maximum data-dependency depth of the artifact outputs.
    #[inline]
    pub fn max_dependency_depth(&self) -> usize {
        self.max_dependency_depth
    }
}

/// One emitted benchmark artifact.
///
/// Each record keeps both the raw textual IR and a normalized structural summary so Rust-side
/// tracing artifacts can be compared against other MLIR producers.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct IrBenchmarkRecord {
    /// Stable benchmark case identifier.
    case_id: String,

    /// High-level category such as `scalar`, `matrix`, or `xla`.
    category: String,

    /// Artifact surface such as `jit`, `vjp_pullback`, `program`, or `shard_map_mlir`.
    surface: String,

    /// Full raw textual IR artifact.
    raw_ir: String,

    /// Normalized structural summary derived from the staged IR.
    summary: IrBenchmarkSummary,
}

impl IrBenchmarkRecord {
    /// Creates one emitted benchmark artifact.
    #[inline]
    pub fn new(
        case_id: impl Into<String>,
        category: impl Into<String>,
        surface: impl Into<String>,
        raw_ir: String,
        summary: IrBenchmarkSummary,
    ) -> Self {
        Self { case_id: case_id.into(), category: category.into(), surface: surface.into(), raw_ir, summary }
    }

    /// Returns the stable benchmark case identifier.
    #[inline]
    pub fn case_id(&self) -> &str {
        &self.case_id
    }

    /// Returns the high-level benchmark category.
    #[inline]
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Returns the artifact surface.
    #[inline]
    pub fn surface(&self) -> &str {
        &self.surface
    }

    /// Returns the full raw textual IR artifact.
    #[inline]
    pub fn raw_ir(&self) -> &str {
        &self.raw_ir
    }

    /// Returns the normalized structural summary derived from the staged IR.
    #[inline]
    pub fn summary(&self) -> &IrBenchmarkSummary {
        &self.summary
    }
}

/// Descriptor for one benchmark case.
#[derive(Copy, Clone)]
pub struct BenchmarkCase {
    /// Stable case identifier.
    case_id: &'static str,

    /// Callback that emits one or more records for the case.
    emit: fn() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError>,
}

impl BenchmarkCase {
    /// Creates a benchmark case descriptor.
    #[inline]
    pub const fn new(case_id: &'static str, emit: fn() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError>) -> Self {
        Self { case_id, emit }
    }

    /// Returns the stable case identifier.
    #[inline]
    pub fn case_id(self) -> &'static str {
        self.case_id
    }

    /// Returns the callback that emits one or more records for the case.
    #[inline]
    pub fn emit(self) -> fn() -> Result<Vec<IrBenchmarkRecord>, BenchmarkError> {
        self.emit
    }
}

/// Returns the stable set of benchmark case IDs supported by the Rust-side emitter.
///
/// # Parameters
///
///   - `extra_cases`: Additional benchmark cases to include (e.g., from `ryft-xla`).
pub fn benchmark_case_ids(extra_cases: &[BenchmarkCase]) -> Vec<&'static str> {
    let mut cases = tracing_v2_cases();
    cases.extend_from_slice(extra_cases);
    cases.into_iter().map(BenchmarkCase::case_id).collect()
}

/// Emits the requested benchmark records.
///
/// When `case_ids` is empty, all known benchmark cases are emitted. The returned records pair raw
/// textual IR with normalized summaries so callers can diff structure separately from printing.
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
                    .find(|case| case.case_id() == case_id)
                    .ok_or_else(|| BenchmarkError::UnknownCase { case_id: case_id.clone() })
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut records = Vec::new();
    for case in selected_cases {
        records.extend((case.emit())()?);
    }
    records.sort_by(|left, right| left.case_id().cmp(right.case_id()).then(left.surface().cmp(right.surface())));
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
    IrBenchmarkRecord::new(case_id, category, surface, raw_ir, summary)
}

/// Normalizes an operation name onto the shared comparison vocabulary.
///
/// # Parameters
///
///   - `name`: Operation name to normalize.
pub(crate) fn normalize_op_name(name: &str) -> String {
    match name {
        ADD_OPERATION_NAME | "add_any" => ADD_OPERATION_NAME.to_string(),
        MUL_OPERATION_NAME => MUL_OPERATION_NAME.to_string(),
        "neg" => "neg".to_string(),
        "sin" => "sin".to_string(),
        "cos" => "cos".to_string(),
        "matmul" | "dot_general" | "left_matmul" | "right_matmul" => "matmul".to_string(),
        "matrix_transpose" | "transpose" => "transpose".to_string(),
        "scale" => "scale".to_string(),
        "const" | "constant" => "const".to_string(),
        "shard_map" | "linear_shard_map" => "shard_map".to_string(),
        other => format!("unknown:{other}"),
    }
}

/// Summarizes one staged program and its immediate nested regions.
///
/// # Parameters
///
///   - `program`: Program to summarize.
///   - `nested_regions_for_op`: Callback that returns the immediate nested regions carried by one
///     staged op.
pub fn summarize_program<T, V, Input, Output, O, F>(
    program: &Program<T, V, O, Input, Output>,
    nested_regions_for_op: F,
) -> Result<IrBenchmarkSummary, BenchmarkError>
where
    T: Type,
    V: Traceable<T>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Operation<T>,
    F: Fn(&O) -> Result<Vec<IrNestedRegionSummary>, BenchmarkError>,
{
    let mut op_histogram = BTreeMap::new();
    let mut nested_regions = Vec::new();
    let mut depth_by_atom = vec![0usize; program.atoms().len()];
    let mut input_atom_flags = vec![false; program.atoms().len()];
    for input_atom in program.input_ids().iter().copied() {
        input_atom_flags[input_atom.index()] = true;
    }

    for (atom_id, atom) in
        (0..program.atoms().len()).map(|atom_index| (AtomId::new(atom_index), &program.atoms()[atom_index]))
    {
        if input_atom_flags[atom_id.index()] || matches!(atom, Atom::Constant(_)) {
            depth_by_atom[atom_id.index()] = 0;
        }
    }

    for instruction in program.instructions() {
        let normalized_name = normalize_op_name(instruction.operation().name());
        *op_histogram.entry(normalized_name).or_insert(0) += 1;

        let input_depth = instruction.inputs().iter().map(|input| depth_by_atom[input.index()]).max().unwrap_or(0);
        for output in instruction.outputs().iter().copied() {
            depth_by_atom[output.index()] = input_depth + 1;
        }

        nested_regions.extend(nested_regions_for_op(instruction.operation())?);
    }

    let nested_region_count = nested_regions.len()
        + nested_regions.iter().map(|nested_region| nested_region.nested_region_count).sum::<usize>();
    let max_dependency_depth =
        program.output_ids().iter().map(|output| depth_by_atom[output.index()]).max().unwrap_or(0);

    Ok(IrBenchmarkSummary::new(
        program.input_ids().len(),
        program.output_ids().len(),
        program.instructions().len(),
        (0..program.atoms().len())
            .map(|atom_index| &program.atoms()[atom_index])
            .filter(|atom| matches!(atom, Atom::Constant(_)))
            .count(),
        op_histogram,
        nested_region_count,
        nested_regions,
        max_dependency_depth,
    ))
}

/// Converts one nested-region summary from a child program into the public nested-region shape.
///
/// # Parameters
///
///   - `label`: Stable nested-region label.
///   - `summary`: Child program summary.
pub fn nested_region(label: &'static str, summary: IrBenchmarkSummary) -> IrNestedRegionSummary {
    IrNestedRegionSummary::new(
        label,
        summary.input_leaf_count,
        summary.output_leaf_count,
        summary.instruction_count,
        summary.constant_count,
        summary.op_histogram,
        summary.nested_region_count,
        summary.max_dependency_depth,
    )
}

/// Returns the tracing-only benchmark cases.
pub(crate) fn tracing_v2_cases() -> Vec<BenchmarkCase> {
    super::benchmark_support::cases()
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::Sin;
    use crate::tracing::domains::ScalarDomain;
    use crate::tracing::{Program, TracingContext};
    use crate::types::DataType;

    use super::*;

    /// Summarizes a small scalar program and verifies the structural metrics.
    #[test]
    fn test_summarize_program_counts_constants_and_depth() {
        let domain = ScalarDomain::<f64>::new();
        let (_, compiled): (f64, Program<DataType, f64, ScalarOperation<f64>, f64, f64>) =
            TracingContext::interpret_and_trace(
                &domain,
                |x| {
                    let with_constant = x.clone() + x.context().constant(1.0);
                    Ok(with_constant.sin())
                },
                2.0f64,
            )
            .unwrap();

        let summary = summarize_program(&compiled, |_| Ok(Vec::new())).unwrap();
        assert_eq!(
            summary,
            IrBenchmarkSummary {
                input_leaf_count: 1,
                output_leaf_count: 1,
                instruction_count: 2,
                constant_count: 1,
                op_histogram: BTreeMap::from([(ADD_OPERATION_NAME.to_string(), 1usize), ("sin".to_string(), 1usize),]),
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
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %0 %1
                    %3:f64 = sin %0
                    %4:f64 = add %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }
}
