//! Backend-independent validation of preserved-reference kernel bodies.
//!
//! A kernel's array operands enter as reference-typed region inputs. Its body reads and mutates them through ordinary
//! reference operations and views, then publishes updated arrays at its outer, array-typed boundary. This module
//! validates those reference accesses using the [`ArrayReferenceAnalysis`] retained on the body region through
//! [`RegionRef::reference_view_analysis`]. A [`KernelBoundaryContract`] declares one [`KernelParameterAccess`] per
//! reference-typed input. These analyses run explicitly for kernel validation; neither is a standing lint on ordinary
//! programs.
//!
//! Read-only operands contain their entering values and publish nothing; read-write operands contain their entering
//! values and publish an updated result. Write-only results start uninitialized, so this boundary admits stores and
//! rejects reads. A swap whose old-value result is provably dead is admitted as a plain store, while a live old-value
//! result requires read-write access. [`KernelReferenceSummary::swap_lowering`] records this classification for each
//! swap so that lowering does not re-derive it. This access check does not prove that every output element has been
//! initialized; complete initialization and publication require the kernel's separate coverage analysis.
//!
//! Kernel bodies publish no references, capture no references, and never consume their operands. Generic reference
//! analysis rejects reference-typed constants and consumption of entering roots. Its region access policies and
//! root-only boundaries also cover nested control flow; a nested swap is classified by its own region's liveness.
//!
//! Scratch bindings are deliberately unsupported: a scratch operand starts uninitialized, so admitting it requires the
//! definite-initialization analysis that lands with uninitialized allocation semantics.
//!
//! The validator is attached to a standalone body region today.
// TODO(eaplatanios): Phase 7 attaches this validator to the real kernel operation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::sync::Arc;

use thiserror::Error;

use crate::arrays::{
    ArrayIrOperation, ArrayIrType, ArrayReferenceAnalysis, ArrayReferenceView, ArrayReferenceViewPath, ArrayType,
};
use crate::programs::{
    AtomId, InstructionId, Operation, ReferenceAccessMode, ReferenceRoot, ReferenceViewAnalysisError,
    ReferenceViewOperation, RegionId, RegionRef, Value, ValueId,
};

/// Error produced by [`validate_kernel_body`] when a kernel body or its [`KernelBoundaryContract`] violates the
/// preserved-reference kernel boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum KernelValidationError {
    /// The array view overlay or the generic reference analysis rejected the body. This covers reference-typed
    /// constants (kernel bodies capture no references) and consumption of an entering operand, both of which the
    /// generic lifetime and capture rules reject before any kernel-specific rule runs.
    #[error(transparent)]
    Analysis(#[from] ReferenceViewAnalysisError),

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

/// Operation-local reference semantics used by kernel boundary validation. Implementations identify an old-value
/// result only for a swap whose first reference operand is read and replaced; the result index must exist and denote
/// exactly that previous value. Ordinary read-write operations and unrecognized extensions return `None`.
pub trait KernelReferenceOperation: ReferenceViewOperation<Type = ArrayIrType, View = ArrayReferenceView> {
    /// Returns the old-value output of a swap, allowing dead results to be classified as stores.
    fn swap_output_index(&self) -> Option<usize> {
        None
    }
}

impl<A: Value<Type = ArrayType>> KernelReferenceOperation for ArrayIrOperation<A> {
    fn swap_output_index(&self) -> Option<usize> {
        matches!(self, Self::ReferenceSwap(_)).then_some(0)
    }
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    /// Array view overlay of the body, shared with the region's transform cache.
    analysis: Arc<ArrayReferenceAnalysis>,

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

    /// Returns the [`ArrayReferenceViewPath`] of the reference-typed `value`, or [`None`] when `value` is not a
    /// reference-typed value of the body closure. Refer to the documentation of [`ArrayReferenceAnalysis::path`] for
    /// more information.
    #[inline]
    pub fn view(&self, value: ValueId) -> Option<&ArrayReferenceViewPath> {
        self.analysis.path(value)
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
/// The body's [`ArrayReferenceAnalysis`] is obtained through the cached [`RegionRef::reference_view_analysis`]
/// accessor under an empty capture scope, so every reference-typed constant is rejected and a body validated twice is
/// analyzed once. The contract must declare exactly one entry per body input, [`Some`] for every reference-typed input
/// and [`None`] for every other input. The body may publish no reference and may not consume an operand. Every access
/// observed on an operand, directly or inside nested regions, must be admitted by its declared
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
pub fn validate_kernel_body<V: Value<Type = ArrayIrType>, O>(
    region: RegionRef<'_, V, O>,
    contract: &KernelBoundaryContract,
) -> Result<KernelReferenceSummary, KernelValidationError>
where
    O: KernelReferenceOperation,
{
    // Kernel bodies capture no references, so no constant names a capture and every reference-typed constant is
    // rejected by the generic analysis. The overlay is the retained one on the region, keyed by the capture count.
    let analysis = region.reference_view_analysis(0)?;
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
                    modes: generic.access_modes_for(root).collect(),
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
    for access in generic.access_modes() {
        let instruction = access.instruction();
        // The generic analysis resolved every attached region of the closure, so the lookup cannot fail here.
        let containing = region.with_id(instruction.region()).unwrap();
        let operation = containing.instructions()[instruction.index()].operation();
        let mode = access.mode();
        let lowering = if mode == ReferenceAccessMode::ReadWrite
            && let Some(output_index) = operation.swap_output_index()
        {
            let old_value = containing.instructions()[instruction.index()].outputs()[output_index];
            let lowering =
                if is_dead(containing, old_value) { KernelSwapLowering::Store } else { KernelSwapLowering::Exchange };
            swap_lowerings.insert(instruction, lowering);
            Some(lowering)
        } else {
            None
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
fn is_dead<V: Value, O: Operation<Type = V::Type>>(region: RegionRef<'_, V, O>, atom: AtomId) -> bool {
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
        ReferenceRoot::Constant { .. } => unreachable!("kernel reference analysis requires lifted captures"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayReferenceViewIndex, ArraySliceAxis, DataType,
        ReferenceIndexOperation, ReferenceSliceOperation,
    };
    use crate::captures::CaptureReference;
    use crate::operations::{
        ConditionOperation, ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation,
        ReferenceReadOperation, ReferenceSwapOperation, ReferenceWriteOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder, ReferenceAnalysisError, ReferenceSource, ReferenceType};

    use super::*;

    /// Concrete core values for portable kernel validation fixtures.
    type TestValue = ArrayIrValue<Array>;

    /// Existing composite array operation family, including references and attached regions.
    type TestOperation = ArrayIrOperation<Array>;

    /// Core builder used by portable kernel validation fixtures.
    type TestBuilder = ProgramBuilder<TestValue, TestOperation>;

    /// Flat portable kernel body containing concrete core constants.
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    /// Identifies an instruction in a fixture's numbered region.
    fn id(region: usize, index: usize) -> InstructionId {
        InstructionId::new(RegionId::new(region), index)
    }

    /// Identifies a value in a fixture's numbered region.
    fn value(region: usize, atom: usize) -> ValueId {
        ValueId::new(RegionId::new(region), AtomId::new(atom))
    }

    /// Identifies an entering reference in a fixture's numbered region.
    fn input_root(region: usize, input_index: usize) -> ReferenceRoot {
        ReferenceRoot::RegionInput { region: RegionId::new(region), input_index }
    }

    /// Constructs an ordinary array type for a fixture.
    fn array_type(dimensions: impl Into<Vec<usize>>) -> ArrayIrType {
        ArrayIrType::Array(ArrayType::new_static(DataType::F32, dimensions))
    }

    /// Constructs a reference type for a fixture.
    fn reference_type(dimensions: impl Into<Vec<usize>>) -> ArrayIrType {
        ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, dimensions)))
    }

    /// Builds the representative accepted body shared by the summary accessor tests: a read-only vector sliced and
    /// read, a write-only vector written through an element view, and a read-write vector read, accumulated, and
    /// swapped with a dead old value, followed by one ordinary scalar input.
    fn accepted_body() -> (TestProgram, KernelBoundaryContract) {
        let mut builder = TestBuilder::new();
        let read_only = builder.add_input(reference_type([2]));
        let write_only = builder.add_input(reference_type([2]));
        let read_write = builder.add_input(reference_type([2]));
        let scalar = builder.add_input(array_type([]));
        let prefix = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 1, 1)]),
                Vec::new(),
                vec![read_only],
                None,
            )
            .unwrap()[0];
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![prefix], None).unwrap()[0];
        let element = builder
            .add_instruction(ReferenceIndexOperation::new(0, 0), Vec::new(), vec![write_only], None)
            .unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![element, scalar], None)
            .unwrap();
        let current =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![read_write], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![read_write, current], None)
            .unwrap();
        builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![read_write, current], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 4], vec![Placeholder])
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
    fn swapping_body(use_old_value: Option<bool>) -> TestProgram {
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type([]));
        let scalar = builder.add_input(array_type([]));
        let old_value = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, scalar], None)
            .unwrap()[0];
        let outputs = match use_old_value {
            None => Vec::new(),
            Some(false) => {
                builder
                    .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, old_value], None)
                    .unwrap();
                Vec::new()
            }
            Some(true) => vec![old_value],
        };
        let output_structure = vec![Placeholder; outputs.len()];
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 2], output_structure)
            .unwrap()
    }

    #[test]
    fn test_array_ir_operation_swap_output_index() {
        let swap = ArrayIrOperation::<Array>::from(ReferenceSwapOperation::new());
        let read = ArrayIrOperation::<Array>::from(ReferenceReadOperation::new());
        assert_eq!(swap.swap_output_index(), Some(0));
        assert_eq!(read.swap_output_index(), None);
    }

    #[test]
    fn test_kernel_validation_error() {
        assert_eq!(
            KernelValidationError::from(ReferenceViewAnalysisError::MissingView {
                operation: "view",
                instruction: id(0, 1),
                output_index: 0,
            })
            .to_string(),
            "operation `view` at ^0[1] declares a reference view at output 0 but describes no view",
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
        let contracts = HashMap::from([(contract.clone(), "read-only input")]);
        assert_eq!(
            contracts.get(&KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly), None])),
            Some(&"read-only input"),
        );
        assert_eq!(contracts.get(&KernelBoundaryContract::new(Vec::new())), None);
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
        assert_eq!(summary.view(value(0, 0)), Some(&ArrayReferenceViewPath::root()));
        assert_eq!(
            summary.view(value(0, 4)).map(|view| view.views().cloned().collect::<Vec<_>>()),
            Some(vec![ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1)] }]),
        );
        assert_eq!(
            summary.view(value(0, 6)).map(|view| view.views().cloned().collect::<Vec<_>>()),
            Some(vec![ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Static(0) }]),
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
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type([]));
        let scalar = builder.add_input(array_type([]));
        let local = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![scalar], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
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
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type([]));
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadWrite)]);
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &contract).err(),
            Some(KernelValidationError::Analysis(ReferenceViewAnalysisError::Analysis(
                ReferenceAnalysisError::ExternalReferenceConsumption {
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
        let mut builder = ProgramBuilder::<CaptureReference<ArrayIrType>, TestOperation>::new();
        let captured = builder.add_constant(CaptureReference::new(0, reference_type([])));
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![captured], None).unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<ArrayIrType>>, Vec<CaptureReference<ArrayIrType>>>(
                vec![snapshot],
                Vec::new(),
                vec![Placeholder],
            )
            .unwrap();
        let contract = KernelBoundaryContract::new(Vec::new());
        assert_eq!(
            validate_kernel_body(program.entry_region_ref(), &contract).err(),
            Some(KernelValidationError::Analysis(ReferenceViewAnalysisError::Analysis(
                ReferenceAnalysisError::InvalidReferenceCapture {
                    region: RegionId::new(0),
                    atom: AtomId::new(0),
                    capture_index: 0,
                    capture_count: 0,
                },
            ))),
        );
    }

    #[test]
    fn test_validate_kernel_body_rejects_disallowed_accesses() {
        // Writing a read-only operand and reading a write-only operand are both rejected, naming the instruction.
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type([]));
        let scalar = builder.add_input(array_type([]));
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, scalar], None)
            .unwrap();
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
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
            let mut branch = TestBuilder::new();
            let reference = branch.add_input(reference_type([]));
            let scalar = branch.add_input(array_type([]));
            let operation = if swap {
                TestOperation::ReferenceSwap(ReferenceSwapOperation::new())
            } else {
                TestOperation::ReferenceWrite(ReferenceWriteOperation::new())
            };
            branch.add_instruction(operation, Vec::new(), vec![reference, scalar], None).unwrap();
            branch
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![scalar], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = TestBuilder::new();
        let true_branch = builder.import_region(make_branch(false).entry_region_ref());
        let false_branch = builder.import_region(make_branch(true).entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let reference = builder.add_input(reference_type([]));
        let scalar = builder.add_input(array_type([]));
        let forwarded = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                vec![predicate, reference, scalar],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![forwarded], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let write_only = KernelBoundaryContract::new(vec![None, Some(KernelParameterAccess::WriteOnly), None]);
        let summary = validate_kernel_body(program.entry_region_ref(), &write_only).unwrap();
        assert_eq!(
            summary.parameter(1).unwrap().modes(),
            &BTreeSet::from([ReferenceAccessMode::Write, ReferenceAccessMode::ReadWrite]),
        );
        assert!(summary.parameter(1).unwrap().is_mutated());
        assert_eq!(summary.swap_lowering(id(1, 0)), Some(KernelSwapLowering::Store));
        assert_eq!(summary.view(value(0, 0)), Some(&ArrayReferenceViewPath::root()));
        assert_eq!(summary.view(value(1, 0)), Some(&ArrayReferenceViewPath::root()));

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
}
