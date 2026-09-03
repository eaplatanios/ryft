//! Preserved-reference kernel boundary for externally stateful XLA kernels.
//!
//! Ordinary XLA lowering discharges every reference before StableHLO and rejects any reference that survives
//! (`experimental::lowering`). A kernel body is the one place where references are *preserved* instead: its array
//! operands enter as reference-typed region inputs, its body reads and mutates them in place through the ordinary
//! reference operations and views, and the kernel publishes updated arrays at its outer, array-typed boundary. This
//! module owns the static validation of such a body. It runs the array view overlay
//! ([`ArrayReferenceAnalysis`](ryft_core::ArrayReferenceAnalysis)) over the body region and checks the body against
//! a [`KernelBoundaryContract`](crate::experimental::reference_kernels::KernelBoundaryContract) declaring one
//! [`KernelParameterAccess`](crate::experimental::reference_kernels::KernelParameterAccess) per reference-typed input.
//! Both analyses are kernel-owned validation infrastructure invoked here explicitly; neither is a standing lint on
//! ordinary programs.
//!
//! The admitted semantics follow the kernel-call model of the Pallas plan: read-only operands contain their entering
//! values and publish nothing; read-write operands contain their entering values and publish an updated result;
//! write-only results start uninitialized, so reading one is invalid and only definite writes are admitted. A swap
//! whose old-value result is provably dead is a plain store and is therefore admitted on a write-only operand, while a
//! swap whose old value is live is an exchange that reads the operand. The summary's
//! [`swap_lowering`](crate::experimental::reference_kernels::KernelReferenceSummary::swap_lowering) records that
//! classification for every swap in the body so that lowering never has to re-derive it. Kernel bodies
//! publish arrays only (reference outputs are rejected), capture no references (reference-typed constants are rejected
//! by the generic analysis), and never consume their operands (consumption of an entering root is rejected by the
//! generic lifetime rules). Nested control-flow regions inside the body are covered by the generic analysis, including
//! its region access policies and root-only boundaries; a swap inside a nested region is classified by that region's
//! liveness.
//!
//! Scratch bindings are deliberately unsupported: a scratch operand starts uninitialized, so admitting it requires the
//! definite-initialization analysis that lands with uninitialized allocation semantics.
//!
//! The validator is attached to a standalone body region today.
// TODO(eaplatanios): Phase 7 attaches this validator to the real kernel operation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;

use ryft_core::{
    ArrayReferenceAnalysis, ArrayReferenceAnalysisError, ArrayReferenceView, ArrayType, AtomId, InstructionId,
    Operation, ReferenceAccessMode, ReferenceRoot, RegionId, RegionRef, ValueId,
};
use thiserror::Error;

use crate::experimental::ops::{XlaConstant, XlaOperation};

/// Error produced by [`validate_kernel_body`] when a kernel body or its [`KernelBoundaryContract`] violates the
/// preserved-reference kernel boundary.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum KernelValidationError {
    /// The array view overlay or the generic reference analysis rejected the body. This covers reference-typed
    /// constants (kernel bodies capture no references) and consumption of an entering operand, both of which the
    /// generic lifetime and capture rules reject before any kernel-specific rule runs.
    #[error(transparent)]
    Analysis(#[from] ArrayReferenceAnalysisError),

    /// The contract declares a different number of parameters than the body has inputs.
    #[error("kernel boundary contract declares {actual} parameters but the kernel body has {expected} inputs")]
    ParameterCountMismatch {
        /// Number of body inputs.
        expected: usize,

        /// Number of contract parameters.
        actual: usize,
    },

    /// A reference-typed body input has no declared access.
    #[error("kernel body input {input_index} is a reference but the boundary contract declares no access for it")]
    UndeclaredReferenceParameter {
        /// Position of the undeclared input.
        input_index: usize,
    },

    /// The contract declares an access for a body input that is not a reference.
    #[error("kernel boundary contract declares {access} access for input {input_index}, which is not a reference")]
    NonReferenceParameter {
        /// Position of the non-reference input.
        input_index: usize,

        /// Declared access.
        access: KernelParameterAccess,
    },

    /// A scratch binding was declared before uninitialized allocation semantics exist.
    #[error("kernel scratch bindings are unsupported until uninitialized allocation semantics exist")]
    ScratchUnsupported,

    /// The body publishes a reference; kernel bodies publish arrays only.
    #[error("kernel body output {output_index} is a reference; kernel bodies publish arrays only")]
    ReferenceOutput {
        /// Position of the reference-typed output.
        output_index: usize,
    },

    /// An access performed on a kernel operand, directly or inside a nested region, is not admitted by the operand's
    /// declared access.
    #[error(
        "operation `{operation}` at {instruction} performs a `{mode}` access on kernel input {input_index}, which the \
         boundary contract declares {access}"
    )]
    DisallowedAccess {
        /// Position of the accessed input.
        input_index: usize,

        /// Declared access of the input.
        access: KernelParameterAccess,

        /// Mode of the offending access.
        mode: ReferenceAccessMode,

        /// Name of the accessing operation.
        operation: &'static str,

        /// Instruction performing the access.
        instruction: InstructionId,
    },
}

/// Access that a kernel declares for one reference-typed operand of its body.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelParameterAccess {
    /// The operand contains its entering value and publishes nothing; the body may only read it.
    ReadOnly,

    /// The operand starts uninitialized and publishes its final value; the body may only write it, including through
    /// swaps whose old-value result is provably dead.
    WriteOnly,

    /// The operand contains its entering value and publishes its final value; the body may perform every
    /// non-consuming access on it.
    ReadWrite,
}

impl Display for KernelParameterAccess {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadOnly => write!(formatter, "read-only"),
            Self::WriteOnly => write!(formatter, "write-only"),
            Self::ReadWrite => write!(formatter, "read-write"),
        }
    }
}

/// Declared accesses of one kernel body, with exactly one entry per body input: [`Some`] for every reference-typed
/// operand and [`None`] for every ordinary array or scalar input. [`validate_kernel_body`] checks the declaration
/// against the body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelBoundaryContract {
    /// Declared access per body input.
    parameters: Vec<Option<KernelParameterAccess>>,
}

impl KernelBoundaryContract {
    /// Creates a new [`KernelBoundaryContract`] from the declared access of each body input, in input order. The
    /// declaration is checked against a body only by [`validate_kernel_body`].
    #[inline]
    pub fn new(parameters: Vec<Option<KernelParameterAccess>>) -> Self {
        Self { parameters }
    }

    /// Creates a new [`KernelBoundaryContract`] that additionally binds program-local scratch allocations with the
    /// provided referent types. Scratch is currently rejected with [`KernelValidationError::ScratchUnsupported`]
    /// whenever `scratch` is non-empty, because a scratch operand starts uninitialized and admitting it requires the
    /// definite-initialization analysis that lands together with uninitialized allocation semantics.
    // TODO(eaplatanios): Phase 9 supplies uninitialized allocation semantics and turns this into a real scratch
    // binding.
    pub fn with_scratch(
        parameters: Vec<Option<KernelParameterAccess>>,
        scratch: Vec<ArrayType>,
    ) -> Result<Self, KernelValidationError> {
        if !scratch.is_empty() {
            return Err(KernelValidationError::ScratchUnsupported);
        }
        Ok(Self::new(parameters))
    }

    /// Returns the declared access per body input, in input order.
    #[inline]
    pub fn parameters(&self) -> &[Option<KernelParameterAccess>] {
        self.parameters.as_slice()
    }
}

/// Lowering of one `reference_swap` inside a kernel body, chosen by the liveness of its old-value result.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelSwapLowering {
    /// The old-value result is provably dead, so the swap is a plain store that never reads the operand.
    Store,

    /// The old-value result is live, so the swap reads the operand before replacing it.
    Exchange,
}

/// Validated facts about one reference-typed kernel operand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelParameterSummary {
    /// Root of the operand in the body region.
    root: ReferenceRoot,

    /// Declared access of the operand.
    access: KernelParameterAccess,

    /// Every access mode observed on the operand, directly or inside nested regions.
    modes: BTreeSet<ReferenceAccessMode>,

    /// Whether any observed access mutates the operand.
    mutated: bool,
}

impl KernelParameterSummary {
    /// Returns the root of the operand in the body region.
    #[inline]
    pub fn root(&self) -> ReferenceRoot {
        self.root
    }

    /// Returns the declared access of the operand.
    #[inline]
    pub fn access(&self) -> KernelParameterAccess {
        self.access
    }

    /// Returns every access mode observed on the operand, directly or inside nested regions, in
    /// [`ReferenceAccessMode`] declaration order.
    #[inline]
    pub fn modes(&self) -> &BTreeSet<ReferenceAccessMode> {
        &self.modes
    }

    /// Returns whether any observed access writes, swaps, or accumulates into the operand.
    #[inline]
    pub fn is_mutated(&self) -> bool {
        self.mutated
    }
}

/// Result of [`validate_kernel_body`]: the array view overlay of the body together with the per-operand summaries and
/// the lowering of every swap in the body.
#[derive(Clone, Debug)]
pub struct KernelReferenceSummary {
    /// Array view overlay of the body.
    analysis: ArrayReferenceAnalysis,

    /// Summary per body input, with [`None`] for non-reference inputs.
    parameters: Vec<Option<KernelParameterSummary>>,

    /// Lowering of every swap in the body, keyed by instruction.
    swap_lowerings: BTreeMap<InstructionId, KernelSwapLowering>,
}

impl KernelReferenceSummary {
    /// Returns the array view overlay of the body.
    #[inline]
    pub fn analysis(&self) -> &ArrayReferenceAnalysis {
        &self.analysis
    }

    /// Returns the summary of the operand at `input_index`, or [`None`] when that input is not a reference or is out
    /// of range.
    #[inline]
    pub fn parameter(&self, input_index: usize) -> Option<&KernelParameterSummary> {
        self.parameters.get(input_index).and_then(Option::as_ref)
    }

    /// Returns the summary per body input, with [`None`] for non-reference inputs.
    #[inline]
    pub fn parameters(&self) -> &[Option<KernelParameterSummary>] {
        self.parameters.as_slice()
    }

    /// Returns the [`ArrayReferenceView`] of the reference-typed `value`, or [`None`] when `value` is not a
    /// reference-typed value of the body closure. Refer to the documentation of [`ArrayReferenceAnalysis::view`] for
    /// more information.
    #[inline]
    pub fn view(&self, value: ValueId) -> Option<&ArrayReferenceView> {
        self.analysis.view(value)
    }

    /// Returns the lowering of the swap at `instruction`, or [`None`] when that instruction is not a swap of the body
    /// closure.
    #[inline]
    pub fn swap_lowering(&self, instruction: InstructionId) -> Option<KernelSwapLowering> {
        self.swap_lowerings.get(&instruction).copied()
    }
}

/// Validates the kernel body `region` against `contract` and returns its [`KernelReferenceSummary`].
///
/// The body is analyzed with [`ArrayReferenceAnalysis`] under an empty capture scope, so every reference-typed
/// constant is rejected. The contract must declare exactly one entry per body input, [`Some`] for every
/// reference-typed input and [`None`] for every other input. The body may publish no reference and may not consume an
/// operand. Every access observed on an operand, directly or inside nested regions, must be admitted by its declared
/// access: a read-only operand admits reads only; a write-only operand admits writes and swaps whose old-value result
/// is provably dead (not used by any instruction of its region and not a region output), which lower as
/// [`KernelSwapLowering::Store`]; a read-write operand admits every non-consuming access. Every swap in the body is
/// classified as [`KernelSwapLowering::Store`] or [`KernelSwapLowering::Exchange`] by the same liveness rule,
/// regardless of the operand it targets.
///
/// # Errors
///
/// Returns the first [`KernelValidationError`] in the order above; access violations are reported in program order
/// within each region.
pub fn validate_kernel_body(
    region: RegionRef<'_, XlaConstant, XlaOperation>,
    contract: &KernelBoundaryContract,
) -> Result<KernelReferenceSummary, KernelValidationError> {
    // Kernel bodies capture no references, so no constant names a capture and every reference-typed constant is
    // rejected by the generic analysis.
    let analysis = ArrayReferenceAnalysis::new(region, 0)?;
    let generic = analysis.analysis();
    let entry = region.id();
    let inputs = region.input_ids();
    if contract.parameters.len() != inputs.len() {
        return Err(KernelValidationError::ParameterCountMismatch {
            expected: inputs.len(),
            actual: contract.parameters.len(),
        });
    }

    let mut parameters = Vec::with_capacity(inputs.len());
    let mut accesses_by_root = BTreeMap::new();
    for (input_index, (input, access)) in inputs.iter().zip(contract.parameters.iter()).enumerate() {
        let root = ReferenceRoot::RegionInput { region: entry, input_index };
        let is_reference = generic.root_of(ValueId::new(entry, *input)).is_some();
        match (is_reference, *access) {
            (true, Some(access)) => {
                accesses_by_root.insert(root, (input_index, access));
                parameters.push(Some(KernelParameterSummary {
                    root,
                    access,
                    modes: generic.access_modes(root).collect(),
                    mutated: generic.is_mutated(root),
                }));
            }
            (true, None) => return Err(KernelValidationError::UndeclaredReferenceParameter { input_index }),
            (false, Some(access)) => return Err(KernelValidationError::NonReferenceParameter { input_index, access }),
            (false, None) => parameters.push(None),
        }
    }
    if let Some(output_index) = generic.output_roots().iter().position(Option::is_some) {
        return Err(KernelValidationError::ReferenceOutput { output_index });
    }

    // Direct accesses inside nested regions are recorded against the nested region's own inputs, which the bindings
    // map back to the roots they denote in the attaching region; following the bindings up to the body region yields
    // the operands an access reaches. A shared region attached under different operands reaches all of them.
    let mut bindings = BTreeMap::<ReferenceRoot, BTreeSet<ReferenceRoot>>::new();
    for binding in generic.region_input_bindings() {
        // Bound inputs are reference-typed inputs of the attached region, which the analysis resolved before recording
        // the binding.
        let nested = generic.root_of(binding.input()).unwrap();
        bindings.entry(nested).or_default().insert(binding.root());
    }

    let mut swap_lowerings = BTreeMap::new();
    for access in generic.accesses() {
        let instruction = access.instruction();
        // The generic analysis resolved every attached region of the closure, so the lookup cannot fail here.
        let containing = region.with_id(instruction.region()).unwrap();
        let operation = containing.instructions()[instruction.index()].operation();
        let mode = access.mode();
        let lowering = match operation {
            XlaOperation::ReferenceSwap(_) if mode == ReferenceAccessMode::ReadWrite => {
                let old_value = containing.instructions()[instruction.index()].outputs()[0];
                let lowering = if is_dead(containing, old_value) {
                    KernelSwapLowering::Store
                } else {
                    KernelSwapLowering::Exchange
                };
                swap_lowerings.insert(instruction, lowering);
                Some(lowering)
            }
            _ => None,
        };
        let mut operands = BTreeSet::new();
        entry_roots(access.root(), entry, &bindings, &mut operands);
        for root in operands {
            // Every root of the body region is a reference-typed input and therefore a declared operand.
            let (input_index, declared) = accesses_by_root[&root];
            let admitted = match declared {
                KernelParameterAccess::ReadOnly => mode == ReferenceAccessMode::Read,
                KernelParameterAccess::WriteOnly => {
                    mode == ReferenceAccessMode::Write || lowering == Some(KernelSwapLowering::Store)
                }
                KernelParameterAccess::ReadWrite => !mode.is_consuming(),
            };
            if !admitted {
                return Err(KernelValidationError::DisallowedAccess {
                    input_index,
                    access: declared,
                    mode,
                    operation: operation.name(),
                    instruction,
                });
            }
        }
    }

    Ok(KernelReferenceSummary { analysis, parameters, swap_lowerings })
}

/// Returns whether `atom` is provably dead in `region`: no instruction of the region uses it and the region does not
/// return it.
fn is_dead(region: RegionRef<'_, XlaConstant, XlaOperation>, atom: AtomId) -> bool {
    !region.output_ids().contains(&atom)
        && !region.instructions().iter().any(|instruction| instruction.inputs().contains(&atom))
}

/// Collects into `operands` the roots of the body region `entry` that `root` denotes, following nested region input
/// bindings upward. Allocations local to the body or to a nested region denote no operand.
fn entry_roots(
    root: ReferenceRoot,
    entry: RegionId,
    bindings: &BTreeMap<ReferenceRoot, BTreeSet<ReferenceRoot>>,
    operands: &mut BTreeSet<ReferenceRoot>,
) {
    match root {
        ReferenceRoot::RegionInput { region, .. } if region == entry => {
            operands.insert(root);
        }
        ReferenceRoot::RegionInput { .. } => {
            for caller in bindings.get(&root).into_iter().flatten() {
                entry_roots(*caller, entry, bindings, operands);
            }
        }
        ReferenceRoot::Allocation { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayIrType, ArrayReferenceViewTransform, ArraySliceAxis, CaptureReference, ConditionOperation, DataType,
        Placeholder, ReferenceAddUpdateOperation, ReferenceAnalysisError, ReferenceFreezeOperation,
        ReferenceIndexOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceSliceOperation,
        ReferenceSource, ReferenceSwapOperation, ReferenceType, ReferenceWriteOperation,
    };

    use crate::experimental::lowering::{LoweringError, lower_mlir_module_for_program};
    use crate::experimental::ops::{FlatXlaProgram, XlaProgramBuilder};

    use super::*;

    fn id(region: usize, index: usize) -> InstructionId {
        InstructionId::new(RegionId::new(region), index)
    }

    fn value(region: usize, atom: usize) -> ValueId {
        ValueId::new(RegionId::new(region), AtomId::new(atom))
    }

    fn input_root(region: usize, input_index: usize) -> ReferenceRoot {
        ReferenceRoot::RegionInput { region: RegionId::new(region), input_index }
    }

    fn array_type(dimensions: impl Into<Vec<usize>>) -> ArrayIrType {
        ArrayIrType::Array(ArrayType::new_static(DataType::F32, dimensions))
    }

    fn reference_type(dimensions: impl Into<Vec<usize>>) -> ArrayIrType {
        ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, dimensions)))
    }

    /// Builds the representative accepted body shared by the summary accessor tests: a read-only vector sliced and
    /// read, a write-only vector written through an element view, and a read-write vector read, accumulated, and
    /// swapped with a dead old value, followed by one ordinary scalar input.
    fn accepted_body() -> (FlatXlaProgram, KernelBoundaryContract) {
        let mut builder = XlaProgramBuilder::new();
        let read_only = builder.add_input(reference_type([2]));
        let write_only = builder.add_input(reference_type([2]));
        let read_write = builder.add_input(reference_type([2]));
        let scalar = builder.add_input(array_type([]));
        let prefix = builder
            .add_instruction(
                XlaOperation::ReferenceSlice(ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 1, 1)])),
                Vec::new(),
                vec![read_only],
                None,
            )
            .unwrap()[0];
        let snapshot = builder
            .add_instruction(XlaOperation::ReferenceRead(ReferenceReadOperation::new()), Vec::new(), vec![prefix], None)
            .unwrap()[0];
        let element = builder
            .add_instruction(
                XlaOperation::ReferenceIndex(ReferenceIndexOperation::new(0, 0)),
                Vec::new(),
                vec![write_only],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                XlaOperation::ReferenceWrite(ReferenceWriteOperation::new()),
                Vec::new(),
                vec![element, scalar],
                None,
            )
            .unwrap();
        let current = builder
            .add_instruction(
                XlaOperation::ReferenceRead(ReferenceReadOperation::new()),
                Vec::new(),
                vec![read_write],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(
                XlaOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                Vec::new(),
                vec![read_write, current],
                None,
            )
            .unwrap();
        builder
            .add_instruction(
                XlaOperation::ReferenceSwap(ReferenceSwapOperation::new()),
                Vec::new(),
                vec![read_write, current],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let contract = KernelBoundaryContract::new(vec![
            Some(KernelParameterAccess::ReadOnly),
            Some(KernelParameterAccess::WriteOnly),
            Some(KernelParameterAccess::ReadWrite),
            None,
        ]);
        (program, contract)
    }

    /// Builds a body with one write-only vector operand, one scalar input, and one swap of the operand whose old-value
    /// result is used as described by `use_old_value`: not at all, by a later write, or as a region output.
    fn swapping_body(use_old_value: Option<bool>) -> FlatXlaProgram {
        let mut builder = XlaProgramBuilder::new();
        let reference = builder.add_input(reference_type([]));
        let scalar = builder.add_input(array_type([]));
        let old_value = builder
            .add_instruction(
                XlaOperation::ReferenceSwap(ReferenceSwapOperation::new()),
                Vec::new(),
                vec![reference, scalar],
                None,
            )
            .unwrap()[0];
        let outputs = match use_old_value {
            None => Vec::new(),
            Some(false) => {
                builder
                    .add_instruction(
                        XlaOperation::ReferenceWrite(ReferenceWriteOperation::new()),
                        Vec::new(),
                        vec![reference, old_value],
                        None,
                    )
                    .unwrap();
                Vec::new()
            }
            Some(true) => vec![old_value],
        };
        let output_structure = vec![Placeholder; outputs.len()];
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], output_structure)
            .unwrap()
    }

    #[test]
    fn test_kernel_validation_error() {
        assert_eq!(
            KernelValidationError::from(ArrayReferenceAnalysisError::MissingView {
                operation: "view",
                instruction: id(0, 1),
            })
            .to_string(),
            "operation `view` at ^0[1] derives a reference view but exposes no view transform",
        );
        assert_eq!(
            KernelValidationError::ParameterCountMismatch { expected: 3, actual: 2 }.to_string(),
            "kernel boundary contract declares 2 parameters but the kernel body has 3 inputs",
        );
        assert_eq!(
            KernelValidationError::UndeclaredReferenceParameter { input_index: 1 }.to_string(),
            "kernel body input 1 is a reference but the boundary contract declares no access for it",
        );
        assert_eq!(
            KernelValidationError::NonReferenceParameter { input_index: 1, access: KernelParameterAccess::ReadOnly }
                .to_string(),
            "kernel boundary contract declares read-only access for input 1, which is not a reference",
        );
        assert_eq!(
            KernelValidationError::ScratchUnsupported.to_string(),
            "kernel scratch bindings are unsupported until uninitialized allocation semantics exist",
        );
        assert_eq!(
            KernelValidationError::ReferenceOutput { output_index: 0 }.to_string(),
            "kernel body output 0 is a reference; kernel bodies publish arrays only",
        );
        assert_eq!(
            KernelValidationError::DisallowedAccess {
                input_index: 1,
                access: KernelParameterAccess::WriteOnly,
                mode: ReferenceAccessMode::ReadWrite,
                operation: "reference_swap",
                instruction: id(0, 2),
            }
            .to_string(),
            "operation `reference_swap` at ^0[2] performs a `read/write` access on kernel input 1, which the boundary \
             contract declares write-only",
        );
    }

    #[test]
    fn test_kernel_parameter_access() {
        assert_eq!(KernelParameterAccess::ReadOnly.to_string(), "read-only");
        assert_eq!(KernelParameterAccess::WriteOnly.to_string(), "write-only");
        assert_eq!(KernelParameterAccess::ReadWrite.to_string(), "read-write");
    }

    #[test]
    fn test_kernel_boundary_contract_new() {
        let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly), None]);
        assert_eq!(contract.parameters(), &[Some(KernelParameterAccess::ReadOnly), None]);
        assert_eq!(KernelBoundaryContract::new(Vec::new()).parameters(), &[]);
    }

    #[test]
    fn test_kernel_boundary_contract_with_scratch() {
        let parameters = vec![Some(KernelParameterAccess::ReadWrite)];
        assert_eq!(
            KernelBoundaryContract::with_scratch(parameters.clone(), Vec::new()),
            Ok(KernelBoundaryContract::new(parameters.clone())),
        );
        assert_eq!(
            KernelBoundaryContract::with_scratch(parameters, vec![ArrayType::new_static(DataType::F32, [4])]),
            Err(KernelValidationError::ScratchUnsupported),
        );
    }

    #[test]
    fn test_kernel_boundary_contract_parameters() {
        let parameters = vec![None, Some(KernelParameterAccess::WriteOnly), Some(KernelParameterAccess::ReadWrite)];
        assert_eq!(KernelBoundaryContract::new(parameters.clone()).parameters(), parameters.as_slice());
    }

    #[test]
    fn test_kernel_parameter_summary() {
        let (program, contract) = accepted_body();
        let summary = validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        let read_write = summary.parameter(2).unwrap();
        assert_eq!(read_write.root(), input_root(0, 2));
        assert_eq!(read_write.access(), KernelParameterAccess::ReadWrite);
        assert_eq!(
            read_write.modes(),
            &BTreeSet::from([
                ReferenceAccessMode::Read,
                ReferenceAccessMode::ReadWrite,
                ReferenceAccessMode::Accumulate,
            ]),
        );
        assert!(read_write.is_mutated());
        let read_only = summary.parameter(0).unwrap();
        assert_eq!(read_only.modes(), &BTreeSet::from([ReferenceAccessMode::Read]));
        assert!(!read_only.is_mutated());
    }

    #[test]
    fn test_kernel_reference_summary_analysis() {
        let (program, contract) = accepted_body();
        let summary = validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        assert_eq!(
            summary.analysis().analysis().roots().collect::<Vec<_>>(),
            vec![input_root(0, 0), input_root(0, 1), input_root(0, 2)],
        );
        assert_eq!(
            summary.analysis().analysis().external_source(input_root(0, 1)),
            Some(ReferenceSource::Input { index: 1 }),
        );
    }

    #[test]
    fn test_kernel_reference_summary_parameter() {
        let (program, contract) = accepted_body();
        let summary = validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        assert_eq!(summary.parameter(0).map(KernelParameterSummary::access), Some(KernelParameterAccess::ReadOnly));
        assert_eq!(summary.parameter(1).map(KernelParameterSummary::access), Some(KernelParameterAccess::WriteOnly));
        assert_eq!(summary.parameter(2).map(KernelParameterSummary::access), Some(KernelParameterAccess::ReadWrite));
        assert_eq!(summary.parameter(3), None);
        assert_eq!(summary.parameter(4), None);
    }

    #[test]
    fn test_kernel_reference_summary_parameters() {
        let (program, contract) = accepted_body();
        let summary = validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        assert_eq!(
            summary.parameters(),
            &[
                Some(KernelParameterSummary {
                    root: input_root(0, 0),
                    access: KernelParameterAccess::ReadOnly,
                    modes: BTreeSet::from([ReferenceAccessMode::Read]),
                    mutated: false,
                }),
                Some(KernelParameterSummary {
                    root: input_root(0, 1),
                    access: KernelParameterAccess::WriteOnly,
                    modes: BTreeSet::from([ReferenceAccessMode::Write]),
                    mutated: true,
                }),
                Some(KernelParameterSummary {
                    root: input_root(0, 2),
                    access: KernelParameterAccess::ReadWrite,
                    modes: BTreeSet::from([
                        ReferenceAccessMode::Read,
                        ReferenceAccessMode::ReadWrite,
                        ReferenceAccessMode::Accumulate,
                    ]),
                    mutated: true,
                }),
                None,
            ],
        );
    }

    #[test]
    fn test_kernel_reference_summary_view() {
        let (program, contract) = accepted_body();
        let summary = validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        assert_eq!(summary.view(value(0, 0)), Some(&ArrayReferenceView::root()));
        assert_eq!(
            summary.view(value(0, 4)).map(ArrayReferenceView::transforms),
            Some(&[ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1)] }][..]),
        );
        assert_eq!(
            summary.view(value(0, 6)).map(ArrayReferenceView::transforms),
            Some(&[ArrayReferenceViewTransform::Index { axis: 0, index: 0 }][..]),
        );
        assert_eq!(summary.view(value(0, 3)), None);
        assert_eq!(summary.view(value(0, 5)), None);
    }

    #[test]
    fn test_kernel_reference_summary_swap_lowering() {
        let (program, contract) = accepted_body();
        let summary = validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        assert_eq!(summary.swap_lowering(id(0, 6)), Some(KernelSwapLowering::Store));
        assert_eq!(summary.swap_lowering(id(0, 5)), None);
        assert_eq!(summary.swap_lowering(id(1, 0)), None);
    }

    #[test]
    fn test_validate_kernel_body() {
        // Read-only, write-only, and read-write operands each admit exactly the accesses the accepted body performs
        // on them, the ordinary scalar input carries no summary, and no reference reaches the boundary.
        let (program, contract) = accepted_body();
        let summary = validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        assert_eq!(summary.parameters().len(), 4);
        assert_eq!(summary.parameter(0).map(KernelParameterSummary::root), Some(input_root(0, 0)));
        assert_eq!(summary.parameter(1).map(KernelParameterSummary::root), Some(input_root(0, 1)));
        assert_eq!(summary.parameter(2).map(KernelParameterSummary::root), Some(input_root(0, 2)));
        assert_eq!(summary.parameter(3), None);
        assert_eq!(summary.analysis().analysis().output_roots(), &[None]);
        assert_eq!(summary.swap_lowering(id(0, 6)), Some(KernelSwapLowering::Store));
    }

    #[test]
    fn test_validate_kernel_body_lowers_swaps_by_result_liveness() {
        // A swap whose old value is dead is a store and is admitted on a write-only operand.
        let program = swapping_body(None);
        let write_only = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::WriteOnly), None]);
        let summary = validate_kernel_body(program.entry_region_ref(), &write_only).unwrap();
        assert_eq!(summary.swap_lowering(id(0, 0)), Some(KernelSwapLowering::Store));
        assert_eq!(summary.parameter(0).unwrap().modes(), &BTreeSet::from([ReferenceAccessMode::ReadWrite]),);

        // A swap whose old value feeds a later instruction is an exchange: rejected on a write-only operand and
        // classified as an exchange on a read-write operand.
        let program = swapping_body(Some(false));
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &write_only).err(),
            Some(KernelValidationError::DisallowedAccess {
                input_index: 0,
                access: KernelParameterAccess::WriteOnly,
                mode: ReferenceAccessMode::ReadWrite,
                operation: "reference_swap",
                instruction: id(0, 0),
            }),
        );
        let read_write = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadWrite), None]);
        let summary = validate_kernel_body(program.entry_region_ref(), &read_write).unwrap();
        assert_eq!(summary.swap_lowering(id(0, 0)), Some(KernelSwapLowering::Exchange));

        // A swap whose old value is a region output is likewise an exchange.
        let program = swapping_body(Some(true));
        assert!(matches!(
            validate_kernel_body(program.entry_region_ref(), &write_only),
            Err(KernelValidationError::DisallowedAccess { input_index: 0, mode: ReferenceAccessMode::ReadWrite, .. }),
        ));
        let summary = validate_kernel_body(program.entry_region_ref(), &read_write).unwrap();
        assert_eq!(summary.swap_lowering(id(0, 0)), Some(KernelSwapLowering::Exchange));
    }

    #[test]
    fn test_validate_kernel_body_rejects_parameter_count_mismatch() {
        let (program, _) = accepted_body();
        let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly)]);
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &contract).err(),
            Some(KernelValidationError::ParameterCountMismatch { expected: 4, actual: 1 }),
        );
    }

    #[test]
    fn test_validate_kernel_body_rejects_undeclared_reference_parameters() {
        let (program, _) = accepted_body();
        let contract = KernelBoundaryContract::new(vec![
            Some(KernelParameterAccess::ReadOnly),
            None,
            Some(KernelParameterAccess::ReadWrite),
            None,
        ]);
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &contract).err(),
            Some(KernelValidationError::UndeclaredReferenceParameter { input_index: 1 }),
        );
    }

    #[test]
    fn test_validate_kernel_body_rejects_non_reference_parameters() {
        let (program, _) = accepted_body();
        let contract = KernelBoundaryContract::new(vec![
            Some(KernelParameterAccess::ReadOnly),
            Some(KernelParameterAccess::WriteOnly),
            Some(KernelParameterAccess::ReadWrite),
            Some(KernelParameterAccess::ReadOnly),
        ]);
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &contract).err(),
            Some(KernelValidationError::NonReferenceParameter {
                input_index: 3,
                access: KernelParameterAccess::ReadOnly,
            }),
        );
    }

    #[test]
    fn test_validate_kernel_body_rejects_reference_outputs() {
        // Publishing either an operand or a body-local allocation is rejected; the local allocation stays legal
        // otherwise because the kernel rule concerns the boundary, not local state.
        let mut builder = XlaProgramBuilder::new();
        let reference = builder.add_input(reference_type([]));
        let scalar = builder.add_input(array_type([]));
        let local = builder
            .add_instruction(XlaOperation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![scalar], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![scalar, local, reference],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();
        let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly), None]);
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &contract).err(),
            Some(KernelValidationError::ReferenceOutput { output_index: 1 }),
        );
    }

    #[test]
    fn test_validate_kernel_body_rejects_consumed_parameters() {
        // Consuming an operand violates the generic lifetime rule for external roots before any kernel rule applies.
        let mut builder = XlaProgramBuilder::new();
        let reference = builder.add_input(reference_type([]));
        let frozen = builder
            .add_instruction(
                XlaOperation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadWrite)]);
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &contract).err(),
            Some(KernelValidationError::Analysis(ArrayReferenceAnalysisError::Analysis(
                ReferenceAnalysisError::ConsumeExternal {
                    operation: "reference_freeze",
                    instruction: id(0, 0),
                    root: input_root(0, 0),
                    external_source: ReferenceSource::Input { index: 0 },
                },
            ))),
        );
    }

    #[test]
    fn test_validate_kernel_body_rejects_reference_constants() {
        // Kernel bodies capture no references, so a captured reference constant names no capture in the body's empty
        // capture scope.
        let mut builder = XlaProgramBuilder::new();
        let captured = builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, reference_type([]))));
        let snapshot = builder
            .add_instruction(
                XlaOperation::ReferenceRead(ReferenceReadOperation::new()),
                Vec::new(),
                vec![captured],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], Vec::new(), vec![Placeholder])
            .unwrap();
        let contract = KernelBoundaryContract::new(Vec::new());
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &contract).err(),
            Some(KernelValidationError::Analysis(ArrayReferenceAnalysisError::Analysis(
                ReferenceAnalysisError::ReferenceConstant { region: RegionId::new(0), atom: AtomId::new(0) },
            ))),
        );
    }

    #[test]
    fn test_validate_kernel_body_rejects_disallowed_accesses() {
        // Writing a read-only operand and reading a write-only operand are both rejected, naming the instruction.
        let mut builder = XlaProgramBuilder::new();
        let reference = builder.add_input(reference_type([]));
        let scalar = builder.add_input(array_type([]));
        builder
            .add_instruction(
                XlaOperation::ReferenceWrite(ReferenceWriteOperation::new()),
                Vec::new(),
                vec![reference, scalar],
                None,
            )
            .unwrap();
        let snapshot = builder
            .add_instruction(
                XlaOperation::ReferenceRead(ReferenceReadOperation::new()),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let read_only = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly), None]);
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &read_only).err(),
            Some(KernelValidationError::DisallowedAccess {
                input_index: 0,
                access: KernelParameterAccess::ReadOnly,
                mode: ReferenceAccessMode::Write,
                operation: "reference_write",
                instruction: id(0, 0),
            }),
        );
        let write_only = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::WriteOnly), None]);
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &write_only).err(),
            Some(KernelValidationError::DisallowedAccess {
                input_index: 0,
                access: KernelParameterAccess::WriteOnly,
                mode: ReferenceAccessMode::Read,
                operation: "reference_read",
                instruction: id(0, 1),
            }),
        );
        let read_write = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadWrite), None]);
        assert!(validate_kernel_body(program.entry_region_ref(), &read_write).is_ok());
    }

    #[test]
    fn test_validate_kernel_body_validates_nested_conditions() {
        // Both branches write the write-only operand through the condition's root-only boundary, and the swap inside
        // the false branch is classified by that branch's own liveness.
        let make_branch = |swap: bool| {
            let mut branch = XlaProgramBuilder::new();
            let reference = branch.add_input(reference_type([]));
            let scalar = branch.add_input(array_type([]));
            let operation = if swap {
                XlaOperation::ReferenceSwap(ReferenceSwapOperation::new())
            } else {
                XlaOperation::ReferenceWrite(ReferenceWriteOperation::new())
            };
            branch.add_instruction(operation, Vec::new(), vec![reference, scalar], None).unwrap();
            branch
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![scalar], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = XlaProgramBuilder::new();
        let true_branch = builder.import_region(make_branch(false).entry_region_ref());
        let false_branch = builder.import_region(make_branch(true).entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let reference = builder.add_input(reference_type([]));
        let scalar = builder.add_input(array_type([]));
        let forwarded = builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                vec![predicate, reference, scalar],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![forwarded], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let write_only = KernelBoundaryContract::new(vec![None, Some(KernelParameterAccess::WriteOnly), None]);
        let summary = validate_kernel_body(program.entry_region_ref(), &write_only).unwrap();
        assert_eq!(
            summary.parameter(1).unwrap().modes(),
            &BTreeSet::from([ReferenceAccessMode::Write, ReferenceAccessMode::ReadWrite]),
        );
        assert!(summary.parameter(1).unwrap().is_mutated());
        assert_eq!(summary.swap_lowering(id(1, 0)), Some(KernelSwapLowering::Store));
        assert_eq!(summary.view(value(0, 0)), Some(&ArrayReferenceView::root()));
        assert_eq!(summary.view(value(1, 0)), Some(&ArrayReferenceView::root()));

        // The nested writes are attributed to the operand, so a read-only declaration is rejected at the branch
        // instruction that performs the first write.
        let read_only = KernelBoundaryContract::new(vec![None, Some(KernelParameterAccess::ReadOnly), None]);
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &read_only).err(),
            Some(KernelValidationError::DisallowedAccess {
                input_index: 1,
                access: KernelParameterAccess::ReadOnly,
                mode: ReferenceAccessMode::Write,
                operation: "reference_write",
                instruction: id(0, 0),
            }),
        );
    }

    #[test]
    fn test_xla_lowering_rejects_preserved_reference_entries() {
        // A body that validates as a kernel is still rejected by ordinary XLA lowering, which admits references only
        // after discharge.
        let mut builder = XlaProgramBuilder::new();
        let reference = builder.add_input(reference_type([]));
        let snapshot = builder
            .add_instruction(
                XlaOperation::ReferenceRead(ReferenceReadOperation::new()),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly)]);
        assert!(validate_kernel_body(program.entry_region_ref(), &contract).is_ok());
        let scalar_type = ArrayType::scalar(DataType::F32);
        assert!(matches!(
            lower_mlir_module_for_program(
                &program,
                &[],
                &vec![scalar_type.clone()],
                &vec![scalar_type],
                "main",
                None,
                None,
                None,
            ),
            Err(LoweringError::UnresolvedReference { construct }) if construct == "program",
        ));
    }
}
