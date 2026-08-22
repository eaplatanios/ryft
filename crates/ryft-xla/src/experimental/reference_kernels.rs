//! Experimental preserved-reference kernel boundary and deterministic mock lowering contract.
//!
//! Ordinary XLA compilation functionalizes references before StableHLO lowering. A kernel implementation instead
//! needs a narrowly scoped boundary at which the same logical roots, views, and access summaries remain references.
//! This module proves that boundary without adding a production kernel language or changing the ordinary XLA
//! operation family: callers validate one standalone body region explicitly, and the resulting artifact records the
//! reference operations a future Mosaic-style lowerer must preserve.
//!
//! # Boundary Flow
//!
//! ```text
//! operation = PreservedReferenceKernelOperation::new([read_only, read_write])
//! body      = (ref<input>, ref<output>) -> ()
//! kernel    = operation.validate_body(body.entry_region_ref())
//!             -> canonical roots, views, accesses, bindings, and mock lowering steps
//! outer ABI = (input_array, output_array) -> (updated_output_array)
//!             with operation-local output/operand reuse metadata only
//! ```

use std::fmt::Display;
use std::num::NonZeroUsize;

use ryft_core::{
    ArrayIrType, ArrayReferenceView, ArrayType, InstructionId, Operation, OperationFormatter, ProgramError,
    ReferenceAccessMode, ReferenceAddUpdateOperation, ReferenceAnalysis, ReferenceDischargeOperation,
    ReferenceDischargeRule, ReferenceIndexOperation, ReferenceReadOperation, ReferenceRoot, ReferenceSwapOperation,
    ReferenceType, RegionInterface, RegionRef, RegionSlot, Type, TypeError, TypeIdentityRenaming, Typed, Value,
    ValueId,
};
use thiserror::Error;

/// Preserved-reference kernel-contract failure.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum KernelReferenceError {
    /// The body boundary does not contain one reference input per declared parameter.
    #[error("kernel body input count {actual} must match parameter count {expected}")]
    InputCount {
        /// Declared parameter count.
        expected: usize,

        /// Actual body input count.
        actual: usize,
    },

    /// A body input is not an array reference.
    #[error("kernel body input {parameter_index} must be an array reference, but has type `{actual}`")]
    NonReferenceInput {
        /// Body parameter position.
        parameter_index: usize,

        /// Actual input type.
        actual: String,
    },

    /// A body exposed ordinary or reference results instead of publishing mutations through the outer array ABI.
    #[error("kernel body must have no outputs, but has {output_count}")]
    BodyOutputs {
        /// Actual body output count.
        output_count: usize,
    },

    /// The mock contract does not yet admit implicit scratch parameters.
    #[error("kernel scratch parameter {parameter_index} is not supported by the preserved-reference mock lowering")]
    ScratchUnsupported {
        /// Scratch parameter position.
        parameter_index: usize,
    },

    /// An outer array operand requested a non-global kernel address space.
    #[error("kernel operand {parameter_index} must use global address space, but requested `{address_space}`")]
    OperandAddressSpace {
        /// Operand parameter position.
        parameter_index: usize,

        /// Requested address space.
        address_space: KernelAddressSpace,
    },

    /// The mock lowering cannot prove a view-alignment requirement greater than one byte.
    #[error(
        "kernel operand {parameter_index} requires {bytes}-byte view alignment, which the mock lowering cannot prove"
    )]
    ViewAlignment {
        /// Operand parameter position.
        parameter_index: usize,

        /// Requested minimum alignment in bytes.
        bytes: usize,
    },

    /// Atomic lowering was requested even though the mock target has no atomic operations.
    #[error(
        "kernel operand {parameter_index} requests atomic accesses, which the preserved-reference mock lowering does \
         not support"
    )]
    AtomicUnsupported {
        /// Operand parameter position.
        parameter_index: usize,
    },

    /// Synchronization was requested even though the mock IR has no synchronization operations.
    #[error("kernel synchronization `{synchronization}` is not supported by the preserved-reference mock lowering")]
    SynchronizationUnsupported {
        /// Requested synchronization contract.
        synchronization: KernelSynchronization,
    },

    /// A nested region would require a recursive kernel-region lowering contract that this MVP does not define.
    #[error(
        "kernel instruction `{instruction}` attaches nested regions, which the preserved-reference mock lowering does \
         not support"
    )]
    NestedRegion {
        /// Instruction attaching nested regions.
        instruction: InstructionId,
    },

    /// A local reference allocation would require the future scratch-allocation contract.
    #[error(
        "kernel instruction `{instruction}` allocates a local reference, but uninitialized scratch is not supported"
    )]
    LocalAllocationUnsupported {
        /// Allocation instruction.
        instruction: InstructionId,
    },

    /// One boundary capability does not permit an analyzed logical access.
    #[error("kernel parameter {parameter_index} with `{access}` access cannot perform reference access `{mode}`")]
    ForbiddenAccess {
        /// Body parameter position.
        parameter_index: usize,

        /// Declared boundary access.
        access: KernelOperandAccess,

        /// Analyzed logical access.
        mode: ReferenceAccessMode,
    },

    /// A write-only operand's swap result is data-live and therefore requires reading the previous value.
    #[error("kernel write-only parameter {parameter_index} cannot use the live old-value result of `{instruction}`")]
    LiveWriteOnlySwap {
        /// Body parameter position.
        parameter_index: usize,

        /// Swap instruction whose old value is live.
        instruction: InstructionId,
    },

    /// A write-only operand attempted a partial-view update that must preserve untouched root elements.
    #[error("kernel write-only parameter {parameter_index} cannot swap a non-root view in `{instruction}`")]
    PartialWriteOnlySwap {
        /// Body parameter position.
        parameter_index: usize,

        /// Swap instruction targeting the derived view.
        instruction: InstructionId,
    },

    /// The mock lowering does not recognize an operation carrying reference state.
    #[error("kernel instruction `{instruction}` uses unsupported reference operation `{operation}`")]
    UnsupportedReferenceOperation {
        /// Unsupported instruction.
        instruction: InstructionId,

        /// Operation name.
        operation: String,
    },
}

/// Capability granted to one outer array operand inside a preserved-reference kernel body.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelOperandAccess {
    /// The body may read but not mutate the operand.
    ReadOnly,

    /// The body may replace the operand without observing its initial contents.
    WriteOnly,

    /// The body may read, replace, and perform ordered additive updates on the operand.
    ReadWrite,
}

impl Display for KernelOperandAccess {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadOnly => formatter.write_str("read-only"),
            Self::WriteOnly => formatter.write_str("write-only"),
            Self::ReadWrite => formatter.write_str("read/write"),
        }
    }
}

/// Kernel eligibility metadata for one target-local logical address-space class.
///
/// This is not a parallel reference-memory model: [`ArrayType::memory`] remains the source of physical placement for
/// outer arrays, while this enum constrains how a validated kernel boundary may expose that storage or future scratch
/// to its target lowerer. It deliberately never appears in [`ReferenceType`] or [`ArrayReferenceView`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelAddressSpace {
    /// Storage supplied by an outer array operand.
    Global,

    /// Workgroup- or block-shared scratch storage.
    Shared,

    /// Invocation-private scratch storage.
    Local,
}

impl Display for KernelAddressSpace {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Global => formatter.write_str("global"),
            Self::Shared => formatter.write_str("shared"),
            Self::Local => formatter.write_str("local"),
        }
    }
}

/// Atomicity requested by a kernel boundary independently of logical ordered accumulation semantics.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum KernelAtomicity {
    /// Preserve the source program's deterministic ordered access semantics.
    #[default]
    Ordered,

    /// Require target atomic operations.
    Atomic,
}

impl Display for KernelAtomicity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordered => formatter.write_str("ordered"),
            Self::Atomic => formatter.write_str("atomic"),
        }
    }
}

/// Synchronization required by a kernel independently of its reference access descriptors.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum KernelSynchronization {
    /// The body requires no explicit kernel synchronization.
    #[default]
    None,

    /// The body requires a target barrier.
    Barrier,
}

impl Display for KernelSynchronization {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => formatter.write_str("none"),
            Self::Barrier => formatter.write_str("barrier"),
        }
    }
}

/// Minimum byte alignment required for every view accessed through one kernel parameter.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelViewAlignment(NonZeroUsize);

impl KernelViewAlignment {
    /// One-byte alignment, which imposes no stronger condition than byte addressability.
    pub const BYTE: Self = Self(NonZeroUsize::MIN);

    /// Creates an alignment requirement from a nonzero byte count.
    #[inline]
    pub const fn new(bytes: NonZeroUsize) -> Self {
        Self(bytes)
    }

    /// Returns the required byte alignment.
    #[inline]
    pub const fn bytes(self) -> NonZeroUsize {
        self.0
    }
}

impl Display for KernelViewAlignment {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}B", self.bytes())
    }
}

/// Contract for one outer array operand exposed as a reference inside the kernel body.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelOperandContract {
    /// Permitted logical accesses.
    access: KernelOperandAccess,

    /// Kernel-owned address space.
    address_space: KernelAddressSpace,

    /// Required view alignment.
    view_alignment: KernelViewAlignment,

    /// Requested atomicity.
    atomicity: KernelAtomicity,
}

impl KernelOperandContract {
    /// Creates a global, byte-aligned, ordered operand contract with the provided access capability.
    #[inline]
    pub const fn new(access: KernelOperandAccess) -> Self {
        Self {
            access,
            address_space: KernelAddressSpace::Global,
            view_alignment: KernelViewAlignment::BYTE,
            atomicity: KernelAtomicity::Ordered,
        }
    }

    /// Returns this contract with `address_space`.
    #[inline]
    pub const fn with_address_space(mut self, address_space: KernelAddressSpace) -> Self {
        self.address_space = address_space;
        self
    }

    /// Returns this contract with `view_alignment`.
    #[inline]
    pub const fn with_view_alignment(mut self, view_alignment: KernelViewAlignment) -> Self {
        self.view_alignment = view_alignment;
        self
    }

    /// Returns this contract with `atomicity`.
    #[inline]
    pub const fn with_atomicity(mut self, atomicity: KernelAtomicity) -> Self {
        self.atomicity = atomicity;
        self
    }

    /// Returns the permitted logical accesses.
    #[inline]
    pub const fn access(self) -> KernelOperandAccess {
        self.access
    }

    /// Returns the kernel-owned address space.
    #[inline]
    pub const fn address_space(self) -> KernelAddressSpace {
        self.address_space
    }

    /// Returns the required view alignment.
    #[inline]
    pub const fn view_alignment(self) -> KernelViewAlignment {
        self.view_alignment
    }

    /// Returns the requested atomicity.
    #[inline]
    pub const fn atomicity(self) -> KernelAtomicity {
        self.atomicity
    }
}

/// Contract for one implicit scratch reference parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelScratchContract {
    /// Scratch referent type.
    r#type: ArrayType,

    /// Scratch address space.
    address_space: KernelAddressSpace,

    /// Required scratch alignment.
    alignment: KernelViewAlignment,
}

impl KernelScratchContract {
    /// Creates byte-aligned scratch in `address_space`.
    #[inline]
    pub const fn new(r#type: ArrayType, address_space: KernelAddressSpace) -> Self {
        Self { r#type, address_space, alignment: KernelViewAlignment::BYTE }
    }

    /// Returns this contract with `alignment`.
    #[inline]
    pub const fn with_alignment(mut self, alignment: KernelViewAlignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Returns the scratch referent type.
    #[inline]
    pub const fn r#type(&self) -> &ArrayType {
        &self.r#type
    }

    /// Returns the scratch address space.
    #[inline]
    pub const fn address_space(&self) -> KernelAddressSpace {
        self.address_space
    }

    /// Returns the required scratch alignment.
    #[inline]
    pub const fn alignment(&self) -> KernelViewAlignment {
        self.alignment
    }
}

/// One reference parameter of a kernel body.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelParameterContract {
    /// Reference derived from an outer array operand.
    Operand(KernelOperandContract),

    /// Implicit nonescaping scratch reference, reserved until uninitialized allocation semantics exist.
    Scratch(KernelScratchContract),
}

impl Display for KernelParameterContract {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Operand(operand) => write!(
                formatter,
                "operand({},{},{},{})",
                operand.access, operand.address_space, operand.view_alignment, operand.atomicity,
            ),
            Self::Scratch(scratch) => {
                write!(formatter, "scratch({},{},{})", scratch.r#type, scratch.address_space, scratch.alignment)
            }
        }
    }
}

/// Operation-local reuse metadata from one outer array operand to one outer kernel-call result.
///
/// This descriptor is consumed only at the preserved kernel call boundary. It never represents executable-entry
/// `tf.aliasing_output` metadata or an `XlaReferenceStateSignature`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelOutputOperandAlias {
    /// Outer array operand position.
    operand_index: usize,

    /// Outer result position.
    output_index: usize,
}

impl KernelOutputOperandAlias {
    /// Returns the aliased operand position.
    #[inline]
    pub const fn operand_index(self) -> usize {
        self.operand_index
    }

    /// Returns the aliased result position.
    #[inline]
    pub const fn output_index(self) -> usize {
        self.output_index
    }
}

/// Standalone higher-order kernel-call payload with an array outer ABI and one reference-preserving body region.
///
/// This payload is intentionally not a variant of [`crate::experimental::ops::XlaOperation`]. It defines and tests
/// the future boundary contract without making ordinary XLA lowering accept or accidentally discharge kernel bodies.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PreservedReferenceKernelOperation {
    /// Ordered body parameter contracts.
    parameters: Vec<KernelParameterContract>,

    /// Whole-kernel synchronization contract.
    synchronization: KernelSynchronization,
}

impl PreservedReferenceKernelOperation {
    /// Creates a kernel operation with no explicit synchronization.
    #[inline]
    pub fn new(parameters: Vec<KernelParameterContract>) -> Self {
        Self { parameters, synchronization: KernelSynchronization::None }
    }

    /// Returns this operation with `synchronization`.
    #[inline]
    pub const fn with_synchronization(mut self, synchronization: KernelSynchronization) -> Self {
        self.synchronization = synchronization;
        self
    }

    /// Returns the ordered body parameter contracts.
    #[inline]
    pub fn parameters(&self) -> &[KernelParameterContract] {
        self.parameters.as_slice()
    }

    /// Returns operation-local reuse aliases that publish mutated operands through the functional outer ABI.
    ///
    /// These aliases belong only to the future preserved kernel call. They never become executable-entry
    /// `tf.aliasing_output` metadata or an `XlaReferenceStateSignature`.
    pub fn output_operand_aliases(&self) -> Vec<KernelOutputOperandAlias> {
        let mut operand_index = 0;
        let mut output_index = 0;
        let mut aliases = Vec::new();
        for parameter in &self.parameters {
            let KernelParameterContract::Operand(operand) = parameter else {
                continue;
            };
            if operand.access != KernelOperandAccess::ReadOnly {
                aliases.push(KernelOutputOperandAlias { operand_index, output_index });
                output_index += 1;
            }
            operand_index += 1;
        }
        aliases
    }

    /// Validates `body` and returns the canonical analysis plus deterministic mock lowering decisions.
    pub fn validate_body<V, O>(&self, body: RegionRef<'_, V, O>) -> Result<PreservedReferenceKernel, ProgramError>
    where
        V: Value<Type = ArrayIrType>,
        O: ReferenceDischargeOperation,
    {
        if body.input_ids().len() != self.parameters.len() {
            return Err(ProgramError::custom(KernelReferenceError::InputCount {
                expected: self.parameters.len(),
                actual: body.input_ids().len(),
            }));
        }
        if !body.output_ids().is_empty() {
            return Err(ProgramError::custom(KernelReferenceError::BodyOutputs {
                output_count: body.output_ids().len(),
            }));
        }
        if self.synchronization != KernelSynchronization::None {
            return Err(ProgramError::custom(KernelReferenceError::SynchronizationUnsupported {
                synchronization: self.synchronization,
            }));
        }

        for (parameter_index, (parameter, input)) in
            self.parameters.iter().zip(body.input_ids().iter().copied()).enumerate()
        {
            let input_type = body.atoms()[input.index()].r#type();
            <&ReferenceType<ArrayType>>::try_from(input_type.as_ref()).map_err(|_| {
                ProgramError::custom(KernelReferenceError::NonReferenceInput {
                    parameter_index,
                    actual: input_type.to_string(),
                })
            })?;
            match parameter {
                KernelParameterContract::Operand(operand) => {
                    if operand.address_space != KernelAddressSpace::Global {
                        return Err(ProgramError::custom(KernelReferenceError::OperandAddressSpace {
                            parameter_index,
                            address_space: operand.address_space,
                        }));
                    }
                    if operand.view_alignment.bytes().get() != 1 {
                        return Err(ProgramError::custom(KernelReferenceError::ViewAlignment {
                            parameter_index,
                            bytes: operand.view_alignment.bytes().get(),
                        }));
                    }
                    if operand.atomicity != KernelAtomicity::Ordered {
                        return Err(ProgramError::custom(KernelReferenceError::AtomicUnsupported { parameter_index }));
                    }
                }
                KernelParameterContract::Scratch(_) => {
                    return Err(ProgramError::custom(KernelReferenceError::ScratchUnsupported { parameter_index }));
                }
            }
        }

        for (instruction_index, instruction) in body.instructions().iter().enumerate() {
            if !instruction.regions().is_empty() {
                return Err(ProgramError::custom(KernelReferenceError::NestedRegion {
                    instruction: InstructionId::new(body.id(), instruction_index),
                }));
            }
        }

        let analysis = body.analyze_references(0)?;
        let mut bindings = Vec::with_capacity(self.parameters.len());
        // Canonical analysis reports exactly one external root per reference-typed region input, in input order, so
        // the parameter position and the external-root position always coincide here.
        for (parameter_index, external) in analysis.external_roots().iter().enumerate() {
            let KernelParameterContract::Operand(operand) = self.parameters[parameter_index] else {
                unreachable!("scratch parameters are rejected before reference analysis")
            };
            bindings.push(KernelReferenceBinding {
                parameter_index,
                operand_index: parameter_index,
                root: external.root(),
                access: operand.access,
            });
        }

        let data_liveness = kernel_data_liveness(body)?;
        let mut lowering = Vec::new();
        for (instruction_index, instruction) in body.instructions().iter().enumerate() {
            let instruction_id = InstructionId::new(body.id(), instruction_index);
            let rule = instruction.operation().reference_discharge_rule();
            if rule == ReferenceDischargeRule::NewReference {
                return Err(ProgramError::custom(KernelReferenceError::LocalAllocationUnsupported {
                    instruction: instruction_id,
                }));
            }
            if matches!(
                rule,
                ReferenceDischargeRule::Freeze
                    | ReferenceDischargeRule::Condition
                    | ReferenceDischargeRule::While
                    | ReferenceDischargeRule::Scan { .. }
                    | ReferenceDischargeRule::Call,
            ) {
                return Err(ProgramError::custom(KernelReferenceError::UnsupportedReferenceOperation {
                    instruction: instruction_id,
                    operation: instruction.operation().name().to_string(),
                }));
            }
            if rule == ReferenceDischargeRule::Ordinary
                && (!instruction.operation().reference_semantics().is_empty()
                    || instruction
                        .inputs()
                        .iter()
                        .chain(instruction.outputs())
                        .any(|atom| body.atoms()[atom.index()].r#type().is_reference()))
            {
                return Err(ProgramError::MalformedProgram(format!(
                    "kernel instruction `{instruction_id}` operation `{}` reports ordinary lowering for a \
                     reference-bearing contract",
                    instruction.operation().name(),
                )));
            }
            let accesses = analysis
                .accesses()
                .iter()
                .filter(|access| access.instruction() == instruction_id)
                .collect::<Vec<_>>();
            let boundary_types = || {
                let inputs = instruction
                    .inputs()
                    .iter()
                    .map(|input| body.atoms()[input.index()].r#type().into_owned())
                    .collect::<Vec<_>>();
                let outputs = instruction
                    .outputs()
                    .iter()
                    .map(|output| body.atoms()[output.index()].r#type().into_owned())
                    .collect::<Vec<_>>();
                (inputs, outputs)
            };
            let primitive_contract = match rule {
                ReferenceDischargeRule::View => Some((
                    1,
                    1,
                    None,
                    "no reference accesses",
                    matches_reference_semantics_and_effects(
                        instruction.operation(),
                        &ReferenceIndexOperation::new(0, 0),
                    ),
                )),
                ReferenceDischargeRule::Read => {
                    let (input_types, output_types) = boundary_types();
                    Some((
                        1,
                        1,
                        Some(ReferenceAccessMode::Read),
                        "exactly one `Read` reference access on input 0",
                        matches_reference_primitive(
                            instruction.operation(),
                            &ReferenceReadOperation,
                            input_types.as_slice(),
                            output_types.as_slice(),
                        ),
                    ))
                }
                ReferenceDischargeRule::Swap => {
                    let (input_types, output_types) = boundary_types();
                    Some((
                        2,
                        1,
                        Some(ReferenceAccessMode::Write),
                        "exactly one `Write` reference access on input 0",
                        matches_reference_primitive(
                            instruction.operation(),
                            &ReferenceSwapOperation,
                            input_types.as_slice(),
                            output_types.as_slice(),
                        ),
                    ))
                }
                ReferenceDischargeRule::AddUpdate => {
                    let (input_types, output_types) = boundary_types();
                    Some((
                        2,
                        0,
                        Some(ReferenceAccessMode::Accumulate),
                        "exactly one `Accumulate` reference access on input 0",
                        matches_reference_primitive(
                            instruction.operation(),
                            &ReferenceAddUpdateOperation,
                            input_types.as_slice(),
                            output_types.as_slice(),
                        ),
                    ))
                }
                _ => None,
            };
            if let Some((expected_input_count, expected_output_count, expected_mode, expected_access, canonical)) =
                primitive_contract
            {
                let accesses_are_valid = match expected_mode {
                    None => accesses.is_empty(),
                    Some(expected_mode) => matches!(
                        accesses.as_slice(),
                        [access] if access.input_index() == 0 && access.mode() == expected_mode
                    ),
                };
                if instruction.inputs().len() != expected_input_count
                    || instruction.outputs().len() != expected_output_count
                    || !accesses_are_valid
                    || !canonical
                {
                    return Err(ProgramError::MalformedProgram(format!(
                        concat!(
                            "kernel instruction `{}` operation `{}` violates the `{}` contract: ",
                            "expected {} inputs, {} outputs, {}, and canonical semantics, types, and effects",
                        ),
                        instruction_id,
                        instruction.operation().name(),
                        rule.name(),
                        expected_input_count,
                        expected_output_count,
                        expected_access,
                    )));
                }
            }
            let access = accesses.first().copied();
            let Some(access) = access else {
                let ordinary_is_live = instruction.outputs().iter().any(|output| data_liveness[output.index()]);
                if rule == ReferenceDischargeRule::Ordinary
                    && (!body.instruction_effects(instruction_index)?.is_pure() || ordinary_is_live)
                {
                    lowering.push(MockKernelInstruction::Ordinary { instruction: instruction_id });
                }
                continue;
            };
            // Local allocations are rejected above, so every analyzed access names one of the external roots that
            // produced `bindings`.
            let binding = bindings.iter().find(|binding| binding.root == access.root()).unwrap();
            let old_value_is_live = rule == ReferenceDischargeRule::Swap
                && instruction.outputs().first().is_some_and(|output| data_liveness[output.index()]);
            let input = instruction.inputs()[access.input_index()];
            let value = ValueId::new(body.id(), input);
            let view = analysis.view(value).cloned().unwrap();
            validate_access(binding, access.mode(), instruction_id, old_value_is_live, view.is_root())?;
            match rule {
                ReferenceDischargeRule::Read => lowering.push(MockKernelInstruction::Read {
                    instruction: instruction_id,
                    output: ValueId::new(body.id(), instruction.outputs()[0]),
                    root: access.root(),
                    view,
                }),
                ReferenceDischargeRule::Swap if old_value_is_live => lowering.push(MockKernelInstruction::Exchange {
                    instruction: instruction_id,
                    replacement: ValueId::new(body.id(), instruction.inputs()[1]),
                    output: ValueId::new(body.id(), instruction.outputs()[0]),
                    root: access.root(),
                    view,
                }),
                ReferenceDischargeRule::Swap => lowering.push(MockKernelInstruction::Store {
                    instruction: instruction_id,
                    replacement: ValueId::new(body.id(), instruction.inputs()[1]),
                    root: access.root(),
                    view,
                }),
                ReferenceDischargeRule::AddUpdate => lowering.push(MockKernelInstruction::Accumulate {
                    instruction: instruction_id,
                    update: ValueId::new(body.id(), instruction.inputs()[1]),
                    root: access.root(),
                    view,
                }),
                // Earlier validation already rejected every named rule that can reach a reference access here.
                _ => {
                    return Err(ProgramError::custom(KernelReferenceError::UnsupportedReferenceOperation {
                        instruction: instruction_id,
                        operation: instruction.operation().name().to_string(),
                    }));
                }
            }
        }

        Ok(PreservedReferenceKernel { analysis, bindings, lowering })
    }

    /// Derives the exact reference body signature and validates the outer array ABI.
    fn expected_body_input_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        if region_interfaces.len() != 1 {
            return Err(TypeError::invalid(format!(
                "`{}` expects 1 region but got {}",
                self.name(),
                region_interfaces.len(),
            )));
        }
        if !region_interfaces[0].output_types().is_empty() {
            return Err(TypeError::invalid(format!(
                "`{}` body must have no outputs but got {}",
                self.name(),
                region_interfaces[0].output_types().len(),
            )));
        }
        let operand_count = self
            .parameters
            .iter()
            .filter(|parameter| matches!(parameter, KernelParameterContract::Operand(_)))
            .count();
        if input_types.len() != operand_count {
            return Err(TypeError::invalid(format!(
                "`{}` expects {operand_count} array inputs but got {}",
                self.name(),
                input_types.len(),
            )));
        }

        let mut input_index = 0;
        let mut expected = Vec::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            let referent = match parameter {
                KernelParameterContract::Operand(_) => {
                    let r#type = <&ArrayType>::try_from(&input_types[input_index])?.clone();
                    input_index += 1;
                    r#type
                }
                KernelParameterContract::Scratch(scratch) => scratch.r#type.clone(),
            };
            expected.push(ArrayIrType::Reference(ReferenceType::new(referent)));
        }
        Ok(expected)
    }
}

/// Returns whether `operation` has the canonical operation-local reference semantics and effects of `canonical`.
fn matches_reference_semantics_and_effects<O, Canonical>(operation: &O, canonical: &Canonical) -> bool
where
    O: Operation<Type = ArrayIrType>,
    Canonical: Operation<Type = ArrayIrType>,
{
    operation.reference_semantics() == canonical.reference_semantics() && operation.effects() == canonical.effects()
}

/// Returns whether `operation` also satisfies the canonical primitive's exact boundary type contract.
fn matches_reference_primitive<O, Canonical>(
    operation: &O,
    canonical: &Canonical,
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
) -> bool
where
    O: Operation<Type = ArrayIrType>,
    Canonical: Operation<Type = ArrayIrType>,
{
    matches_reference_semantics_and_effects(operation, canonical)
        && canonical.infer_output_types(input_types, &[]) == Ok(output_types.to_vec())
}

impl Display for PreservedReferenceKernelOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for PreservedReferenceKernelOperation {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        "preserved_reference_kernel"
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("body")] }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        let expected = self.expected_body_input_types(input_types, region_interfaces)?;
        Ok(vec![(region_interfaces[0].input_types() != expected).then_some(expected)])
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        let expected = self.expected_body_input_types(input_types, region_interfaces)?;
        if region_interfaces[0].input_types() != expected {
            let expected = expected.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
            let actual =
                region_interfaces[0].input_types().iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
            return Err(TypeError::invalid(format!(
                "`{}` body input types must be [{expected}] but got [{actual}]",
                self.name(),
            )));
        }
        let mut outputs = Vec::new();
        let mut input_index = 0;
        for parameter in &self.parameters {
            if let KernelParameterContract::Operand(operand) = parameter {
                if operand.access != KernelOperandAccess::ReadOnly {
                    outputs.push(input_types[input_index].clone());
                }
                input_index += 1;
            }
        }
        Ok(outputs)
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        let parameters = self
            .parameters
            .iter()
            .map(|parameter| match parameter {
                KernelParameterContract::Operand(operand) => Ok(KernelParameterContract::Operand(*operand)),
                KernelParameterContract::Scratch(scratch) => {
                    Ok(KernelParameterContract::Scratch(KernelScratchContract {
                        r#type: scratch.r#type.rename_identities(renaming)?,
                        address_space: scratch.address_space,
                        alignment: scratch.alignment,
                    }))
                }
            })
            .collect::<Result<Vec<_>, TypeError>>()?;
        Ok(Self { parameters, synchronization: self.synchronization })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let parameters = self.parameters.iter().map(ToString::to_string).collect::<Vec<_>>().join(";");
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("parameters", parameters)?;
            operation.field("synchronization", self.synchronization)
        })
    }
}

/// Binding between one kernel body reference parameter and its outer array operand.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelReferenceBinding {
    /// Body parameter position.
    parameter_index: usize,

    /// Outer array operand position.
    operand_index: usize,

    /// Canonical body root.
    root: ReferenceRoot,

    /// Declared boundary access.
    access: KernelOperandAccess,
}

impl KernelReferenceBinding {
    /// Returns the body parameter position.
    #[inline]
    pub const fn parameter_index(self) -> usize {
        self.parameter_index
    }

    /// Returns the outer array operand position.
    #[inline]
    pub const fn operand_index(self) -> usize {
        self.operand_index
    }

    /// Returns the canonical body root.
    #[inline]
    pub const fn root(self) -> ReferenceRoot {
        self.root
    }

    /// Returns the declared boundary access.
    #[inline]
    pub const fn access(self) -> KernelOperandAccess {
        self.access
    }
}

/// Deterministic reference-specific step emitted by the preserved-reference mock lowerer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MockKernelInstruction {
    /// Reference-free operation retained for the target's ordinary value lowerer.
    Ordinary {
        /// Source instruction.
        instruction: InstructionId,
    },

    /// Snapshot read from one root-relative view.
    Read {
        /// Source instruction.
        instruction: InstructionId,

        /// SSA value receiving the snapshot.
        output: ValueId,

        /// Canonical reference root.
        root: ReferenceRoot,

        /// Root-relative view.
        view: ArrayReferenceView,
    },

    /// Plain store selected because a swap's old-value result is data-dead.
    Store {
        /// Source instruction.
        instruction: InstructionId,

        /// Replacement SSA value.
        replacement: ValueId,

        /// Canonical reference root.
        root: ReferenceRoot,

        /// Root-relative view.
        view: ArrayReferenceView,
    },

    /// Exchange selected because a swap's old-value result is data-live.
    Exchange {
        /// Source instruction.
        instruction: InstructionId,

        /// Replacement SSA value.
        replacement: ValueId,

        /// SSA value receiving the old snapshot.
        output: ValueId,

        /// Canonical reference root.
        root: ReferenceRoot,

        /// Root-relative view.
        view: ArrayReferenceView,
    },

    /// Ordered additive update, deliberately carrying no atomicity promise.
    Accumulate {
        /// Source instruction.
        instruction: InstructionId,

        /// Update SSA value.
        update: ValueId,

        /// Canonical reference root.
        root: ReferenceRoot,

        /// Root-relative view.
        view: ArrayReferenceView,
    },
}

/// Validated preserved-reference body together with its deterministic mock lowering decisions.
#[derive(Debug)]
pub struct PreservedReferenceKernel {
    /// Canonical core reference analysis.
    analysis: ReferenceAnalysis,

    /// Ordered boundary bindings.
    bindings: Vec<KernelReferenceBinding>,

    /// Deterministic reference-specific lowering steps.
    lowering: Vec<MockKernelInstruction>,
}

impl PreservedReferenceKernel {
    /// Returns the canonical root, view, and access analysis reused by this eligibility proof.
    #[inline]
    pub fn analysis(&self) -> &ReferenceAnalysis {
        &self.analysis
    }

    /// Returns boundary bindings in body parameter order.
    #[inline]
    pub fn bindings(&self) -> &[KernelReferenceBinding] {
        self.bindings.as_slice()
    }

    /// Returns deterministic mock lowering steps in body instruction order.
    #[inline]
    pub fn lowering(&self) -> &[MockKernelInstruction] {
        self.lowering.as_slice()
    }
}

/// Computes data liveness while retaining effectful operations through their inputs, not through their optional
/// results. This distinction is what allows an unused `reference_swap` old-value result to become a plain store.
fn kernel_data_liveness<V, O>(body: RegionRef<'_, V, O>) -> Result<Vec<bool>, ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: Operation<Type = ArrayIrType>,
{
    let mut live = vec![false; body.atoms().len()];
    for output in body.output_ids().iter().copied() {
        live[output.index()] = true;
    }
    for (instruction_index, instruction) in body.instructions().iter().enumerate() {
        if !body.instruction_effects(instruction_index)?.is_pure() {
            for input in instruction.inputs().iter().copied() {
                live[input.index()] = true;
            }
        }
    }
    for instruction in body.instructions().iter().rev() {
        if instruction.outputs().iter().any(|output| live[output.index()]) {
            for input in instruction.inputs().iter().copied() {
                live[input.index()] = true;
            }
        }
    }
    Ok(live)
}

/// Validates one analyzed access against its boundary capability and operation-specific swap liveness.
fn validate_access(
    binding: &KernelReferenceBinding,
    mode: ReferenceAccessMode,
    instruction: InstructionId,
    old_value_is_live: bool,
    view_is_root: bool,
) -> Result<(), ProgramError> {
    let permitted = match (binding.access, mode) {
        (KernelOperandAccess::ReadOnly, ReferenceAccessMode::Read)
        | (KernelOperandAccess::WriteOnly, ReferenceAccessMode::Write)
        | (
            KernelOperandAccess::ReadWrite,
            ReferenceAccessMode::Read | ReferenceAccessMode::Write | ReferenceAccessMode::Accumulate,
        ) => true,
        _ => false,
    };
    if !permitted {
        return Err(ProgramError::custom(KernelReferenceError::ForbiddenAccess {
            parameter_index: binding.parameter_index,
            access: binding.access,
            mode,
        }));
    }
    if binding.access == KernelOperandAccess::WriteOnly && old_value_is_live {
        return Err(ProgramError::custom(KernelReferenceError::LiveWriteOnlySwap {
            parameter_index: binding.parameter_index,
            instruction,
        }));
    }
    if binding.access == KernelOperandAccess::WriteOnly && !view_is_root {
        return Err(ProgramError::custom(KernelReferenceError::PartialWriteOnlySwap {
            parameter_index: binding.parameter_index,
            instruction,
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, Array as CpuArray, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayReferenceOperation,
        ArrayReferenceViewTransform, ArraySliceAxis, ConditionOperation, DataType, Dimension, Effects,
        FreezeReferenceOperation, NewReferenceOperation, Placeholder, PrintOperation, Program, ProgramBuilder,
        ReferenceAnalysisError, ReferenceOperationSemantics, ReferenceSliceOperation, ReshapeOperation, Shape,
        SliceOperation, UpdateSliceOperation, ValueProjection,
    };

    use crate::experimental::lowering::{LoweringError, lower_mlir_module_for_program};
    use crate::experimental::ops::{FlatXlaProgram, XlaOperation, XlaProgramBuilder};

    use super::*;

    type TestValue = ArrayIrValue<CpuArray>;
    type TestOperation = ArrayIrOperation<CpuArray>;
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    #[derive(Copy, Clone, Debug)]
    struct WrongTypeReadOperation;

    impl Operation for WrongTypeReadOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            "wrong_type_read"
        }

        fn infer_output_types(
            &self,
            _input_types: &[ArrayIrType],
            _region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))])
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            ReferenceReadOperation.reference_semantics()
        }

        fn effects(&self) -> Effects {
            ReferenceReadOperation.effects()
        }

        fn rename_type_identities(
            &self,
            _renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
        ) -> Result<Self, TypeError> {
            Ok(*self)
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            let _operation = OperationFormatter::new(formatter, indentation, self.name())?;
            Ok(())
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct OrdinaryReferenceIndexOperation(ReferenceIndexOperation);

    impl Operation for OrdinaryReferenceIndexOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            self.0.name()
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            self.0.infer_output_types(input_types, region_interfaces)
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            self.0.reference_semantics()
        }

        fn effects(&self) -> Effects {
            self.0.effects()
        }

        fn rename_type_identities(
            &self,
            renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
        ) -> Result<Self, TypeError> {
            Ok(Self(self.0.rename_type_identities(renaming)?))
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            self.0.render(formatter, indentation)
        }
    }

    #[derive(Clone, Debug, ryft_macros::Operation)]
    #[ryft(crate = "ryft_core", type = ArrayIrType, constant = TestValue)]
    enum MalformedReferenceRuleOperation {
        Native(TestOperation),
        ReferenceIndex(ReferenceIndexOperation),
        OrdinaryReferenceIndex(OrdinaryReferenceIndexOperation),
        ReferenceRead(ReferenceReadOperation),
        ReferenceAddUpdate(ReferenceAddUpdateOperation),
        WrongTypeRead(WrongTypeReadOperation),
    }

    impl ArrayReferenceOperation for MalformedReferenceRuleOperation {
        fn reference_view_transform(&self) -> Option<ArrayReferenceViewTransform> {
            match self {
                Self::ReferenceIndex(operation) => Some(operation.transform()),
                Self::OrdinaryReferenceIndex(operation) => Some(operation.0.transform()),
                _ => None,
            }
        }
    }

    impl ReferenceDischargeOperation for MalformedReferenceRuleOperation {
        fn reference_discharge_rule(&self) -> ReferenceDischargeRule {
            match self {
                Self::ReferenceIndex(_) | Self::ReferenceAddUpdate(_) => ReferenceDischargeRule::Read,
                Self::OrdinaryReferenceIndex(_) => ReferenceDischargeRule::Ordinary,
                Self::ReferenceRead(_) => ReferenceDischargeRule::View,
                Self::WrongTypeRead(_) => ReferenceDischargeRule::Read,
                Self::Native(operation) => operation.reference_discharge_rule(),
            }
        }

        fn from_reference_reshape(operation: ReshapeOperation) -> Self {
            Self::Native(TestOperation::from_reference_reshape(operation))
        }

        fn from_reference_slice(operation: SliceOperation) -> Self {
            Self::Native(TestOperation::from_reference_slice(operation))
        }

        fn from_reference_update_slice(operation: UpdateSliceOperation) -> Self {
            Self::Native(TestOperation::from_reference_update_slice(operation))
        }

        fn with_added_reference_scan_carries(&self, _additional_carry_count: usize) -> Result<Self, ProgramError> {
            Ok(self.clone())
        }
    }

    impl From<AddOperation<ArrayIrType>> for MalformedReferenceRuleOperation {
        fn from(operation: AddOperation<ArrayIrType>) -> Self {
            Self::Native(operation.into())
        }
    }

    fn scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn reference_type(r#type: ArrayType) -> ArrayIrType {
        ReferenceType::new(r#type).into()
    }

    fn operand(access: KernelOperandAccess) -> KernelParameterContract {
        KernelParameterContract::Operand(KernelOperandContract::new(access))
    }

    fn read_store_body() -> TestProgram {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let source = builder.add_input(reference_type(scalar_type()));
        let destination = builder.add_input(reference_type(scalar_type()));
        let value = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![source]).unwrap()[0];
        builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![destination, value]).unwrap();
        builder.build(Vec::new(), vec![Placeholder; 2], Vec::<Placeholder>::new()).unwrap()
    }

    fn idle_body(r#type: ArrayType) -> TestProgram {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(reference_type(r#type));
        builder.build(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new()).unwrap()
    }

    fn kernel_error(error: &ProgramError) -> &KernelReferenceError {
        error.downcast_custom::<KernelReferenceError>().unwrap()
    }

    #[test]
    fn test_preserved_reference_kernel_operation_boundary_and_rendering() {
        let alignment = KernelViewAlignment::new(NonZeroUsize::new(16).unwrap());
        let operation = PreservedReferenceKernelOperation::new(vec![
            operand(KernelOperandAccess::ReadOnly),
            KernelParameterContract::Operand(
                KernelOperandContract::new(KernelOperandAccess::ReadWrite)
                    .with_view_alignment(alignment)
                    .with_atomicity(KernelAtomicity::Atomic),
            ),
            KernelParameterContract::Scratch(
                KernelScratchContract::new(scalar_type(), KernelAddressSpace::Shared).with_alignment(alignment),
            ),
        ])
        .with_synchronization(KernelSynchronization::Barrier);

        // The declared contracts are readable through the public accessor surface, which is what a future lowerer
        // consumes: builder defaults stay global, byte-aligned, and ordered unless overridden.
        let [
            KernelParameterContract::Operand(read_only),
            KernelParameterContract::Operand(read_write),
            KernelParameterContract::Scratch(scratch),
        ] = operation.parameters()
        else {
            panic!("the kernel declares two operand parameters followed by one scratch parameter")
        };
        assert_eq!(read_only.access(), KernelOperandAccess::ReadOnly);
        assert_eq!(read_only.address_space(), KernelAddressSpace::Global);
        assert_eq!(read_only.view_alignment(), KernelViewAlignment::BYTE);
        assert_eq!(read_only.view_alignment().bytes().get(), 1);
        assert_eq!(read_only.atomicity(), KernelAtomicity::Ordered);
        assert_eq!(read_write.access(), KernelOperandAccess::ReadWrite);
        assert_eq!(read_write.address_space(), KernelAddressSpace::Global);
        assert_eq!(read_write.view_alignment(), alignment);
        assert_eq!(read_write.atomicity(), KernelAtomicity::Atomic);
        assert_eq!(scratch.r#type(), &scalar_type());
        assert_eq!(scratch.address_space(), KernelAddressSpace::Shared);
        assert_eq!(scratch.alignment(), alignment);

        assert_eq!(
            operation.to_string(),
            concat!(
                "preserved_reference_kernel [\n",
                "    parameters=operand(read-only,global,1B,ordered);",
                "operand(read/write,global,16B,atomic);scratch(f32[],shared,16B),\n",
                "    synchronization=barrier,\n",
                "]",
            ),
        );

        let scalar = ArrayIrType::Array(scalar_type());
        let expected_body_inputs =
            vec![reference_type(scalar_type()), reference_type(scalar_type()), reference_type(scalar_type())];
        let interface = RegionInterface::new(expected_body_inputs.clone(), Vec::new(), Effects::PURE);
        assert_eq!(
            operation.infer_region_input_types(&[scalar.clone(), scalar.clone()], std::slice::from_ref(&interface)),
            Ok(vec![None]),
        );
        assert_eq!(
            operation.infer_output_types(&[scalar.clone(), scalar], std::slice::from_ref(&interface)),
            Ok(vec![ArrayIrType::Array(scalar_type())]),
        );
        // Only the mutable operand publishes a reuse alias, and it names the single outer result position.
        let aliases = operation.output_operand_aliases();
        assert_eq!(aliases.len(), 1);
        assert_eq!(aliases[0].operand_index(), 1);
        assert_eq!(aliases[0].output_index(), 0);

        let bad_interface = RegionInterface::new(
            vec![reference_type(scalar_type()), ArrayIrType::Array(scalar_type()), reference_type(scalar_type())],
            Vec::new(),
            Effects::PURE,
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayIrType::Array(scalar_type()), ArrayIrType::Array(scalar_type())],
                std::slice::from_ref(&bad_interface),
            ),
            Err(TypeError::invalid(
                "`preserved_reference_kernel` body input types must be [ref<f32[]>, ref<f32[]>, ref<f32[]>] but got \
                 [ref<f32[]>, f32[], ref<f32[]>]",
            )),
        );
    }

    #[test]
    fn test_preserved_reference_kernel_modes_and_swap_liveness() {
        let body = read_store_body();
        let operation = PreservedReferenceKernelOperation::new(vec![
            operand(KernelOperandAccess::ReadOnly),
            operand(KernelOperandAccess::WriteOnly),
        ]);
        let kernel = operation.validate_body(body.entry_region_ref()).unwrap();
        let entry = body.entry();
        let source_root = ReferenceRoot::RegionInput { region: entry, input_index: 0 };
        let destination_root = ReferenceRoot::RegionInput { region: entry, input_index: 1 };
        let bindings = kernel.bindings();
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].parameter_index(), 0);
        assert_eq!(bindings[0].operand_index(), 0);
        assert_eq!(bindings[0].root(), source_root);
        assert_eq!(bindings[0].access(), KernelOperandAccess::ReadOnly);
        assert_eq!(bindings[1].parameter_index(), 1);
        assert_eq!(bindings[1].operand_index(), 1);
        assert_eq!(bindings[1].root(), destination_root);
        assert_eq!(bindings[1].access(), KernelOperandAccess::WriteOnly);
        assert_eq!(
            kernel.lowering(),
            &[
                MockKernelInstruction::Read {
                    instruction: InstructionId::new(entry, 0),
                    output: ValueId::new(entry, body.instructions()[0].outputs()[0]),
                    root: source_root,
                    view: ArrayReferenceView::root(),
                },
                MockKernelInstruction::Store {
                    instruction: InstructionId::new(entry, 1),
                    replacement: ValueId::new(entry, body.instructions()[1].inputs()[1]),
                    root: destination_root,
                    view: ArrayReferenceView::root(),
                },
            ],
        );

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = builder.add_input(reference_type(scalar_type()));
        let second = builder.add_input(reference_type(scalar_type()));
        let source = builder.add_input(reference_type(scalar_type()));
        let replacement = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![source]).unwrap()[0];
        let old = builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![first, replacement]).unwrap()[0];
        builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![second, old]).unwrap();
        let live_body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 3], Vec::<Placeholder>::new())
            .unwrap();
        let live_operation = PreservedReferenceKernelOperation::new(vec![
            operand(KernelOperandAccess::ReadWrite),
            operand(KernelOperandAccess::WriteOnly),
            operand(KernelOperandAccess::ReadOnly),
        ]);
        let live_kernel = live_operation.validate_body(live_body.entry_region_ref()).unwrap();
        let entry = live_body.entry();
        assert_eq!(
            &live_kernel.lowering()[1..],
            &[
                MockKernelInstruction::Exchange {
                    instruction: InstructionId::new(entry, 1),
                    replacement: ValueId::new(entry, live_body.instructions()[1].inputs()[1]),
                    output: ValueId::new(entry, live_body.instructions()[1].outputs()[0]),
                    root: ReferenceRoot::RegionInput { region: entry, input_index: 0 },
                    view: ArrayReferenceView::root(),
                },
                MockKernelInstruction::Store {
                    instruction: InstructionId::new(entry, 2),
                    replacement: ValueId::new(entry, live_body.instructions()[2].inputs()[1]),
                    root: ReferenceRoot::RegionInput { region: entry, input_index: 1 },
                    view: ArrayReferenceView::root(),
                },
            ],
        );

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let destination = builder.add_input(reference_type(scalar_type()));
        let replacement = builder.add_constant(TestValue::Array(CpuArray::scalar(1.0_f32)));
        let increment = builder.add_constant(TestValue::Array(CpuArray::scalar(2.0_f32)));
        let old =
            builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![destination, replacement]).unwrap()[0];
        builder.add_instruction(AddOperation::new(), Vec::new(), vec![old, increment]).unwrap();
        let dead_consumer_body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let dead_consumer_kernel =
            PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::WriteOnly)])
                .validate_body(dead_consumer_body.entry_region_ref())
                .unwrap();
        assert_eq!(
            dead_consumer_kernel.lowering(),
            &[MockKernelInstruction::Store {
                instruction: InstructionId::new(dead_consumer_body.entry(), 0),
                replacement: ValueId::new(dead_consumer_body.entry(), dead_consumer_body.instructions()[0].inputs()[1]),
                root: ReferenceRoot::RegionInput { region: dead_consumer_body.entry(), input_index: 0 },
                view: ArrayReferenceView::root(),
            }],
        );

        let invalid = PreservedReferenceKernelOperation::new(vec![
            operand(KernelOperandAccess::WriteOnly),
            operand(KernelOperandAccess::WriteOnly),
            operand(KernelOperandAccess::ReadOnly),
        ])
        .validate_body(live_body.entry_region_ref())
        .unwrap_err();
        assert_eq!(
            kernel_error(&invalid),
            &KernelReferenceError::LiveWriteOnlySwap {
                parameter_index: 0,
                instruction: InstructionId::new(live_body.entry(), 1),
            },
        );

        let invalid = PreservedReferenceKernelOperation::new(vec![
            operand(KernelOperandAccess::WriteOnly),
            operand(KernelOperandAccess::WriteOnly),
        ])
        .validate_body(body.entry_region_ref())
        .unwrap_err();
        assert_eq!(
            kernel_error(&invalid),
            &KernelReferenceError::ForbiddenAccess {
                parameter_index: 0,
                access: KernelOperandAccess::WriteOnly,
                mode: ReferenceAccessMode::Read,
            },
        );

        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type(vector_type));
        let view = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1)]),
                Vec::new(),
                vec![reference],
            )
            .unwrap()[0];
        let replacement = builder.add_constant(TestValue::Array(CpuArray::vector(vec![5.0_f32, 6.0])));
        builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![view, replacement]).unwrap();
        let partial_write = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let invalid = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::WriteOnly)])
            .validate_body(partial_write.entry_region_ref())
            .unwrap_err();
        assert_eq!(
            kernel_error(&invalid),
            &KernelReferenceError::PartialWriteOnlySwap {
                parameter_index: 0,
                instruction: InstructionId::new(partial_write.entry(), 1),
            },
        );

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type(scalar_type()));
        let update = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update]).unwrap();
        let accumulate_body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let accumulate = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadWrite)])
            .validate_body(accumulate_body.entry_region_ref())
            .unwrap();
        assert_eq!(
            accumulate.lowering(),
            &[
                MockKernelInstruction::Read {
                    instruction: InstructionId::new(accumulate_body.entry(), 0),
                    output: ValueId::new(accumulate_body.entry(), accumulate_body.instructions()[0].outputs()[0]),
                    root: ReferenceRoot::RegionInput { region: accumulate_body.entry(), input_index: 0 },
                    view: ArrayReferenceView::root(),
                },
                MockKernelInstruction::Accumulate {
                    instruction: InstructionId::new(accumulate_body.entry(), 1),
                    update: ValueId::new(accumulate_body.entry(), accumulate_body.instructions()[1].inputs()[1]),
                    root: ReferenceRoot::RegionInput { region: accumulate_body.entry(), input_index: 0 },
                    view: ArrayReferenceView::root(),
                },
            ],
        );
    }

    #[test]
    fn test_preserved_reference_kernel_retains_effectful_and_live_ordinary_instructions() {
        // An effectful ordinary instruction is retained even though its result is dead, because dropping it would
        // drop the effect.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type(scalar_type()));
        let value = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        builder
            .add_instruction(ArrayOperation::Print(PrintOperation::new("kernel")), Vec::new(), vec![value])
            .unwrap();
        let effectful_body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let effectful_kernel = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .validate_body(effectful_body.entry_region_ref())
            .unwrap();
        let entry = effectful_body.entry();
        assert_eq!(
            effectful_kernel.lowering(),
            &[
                MockKernelInstruction::Read {
                    instruction: InstructionId::new(entry, 0),
                    output: ValueId::new(entry, effectful_body.instructions()[0].outputs()[0]),
                    root: ReferenceRoot::RegionInput { region: entry, input_index: 0 },
                    view: ArrayReferenceView::root(),
                },
                MockKernelInstruction::Ordinary { instruction: InstructionId::new(entry, 1) },
            ],
        );

        // A pure ordinary instruction is retained when its result is live, here because the sum feeds the store.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type(scalar_type()));
        let increment = builder.add_constant(TestValue::Array(CpuArray::scalar(2.0_f32)));
        let current = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        let updated = builder.add_instruction(AddOperation::new(), Vec::new(), vec![current, increment]).unwrap()[0];
        builder.add_instruction(ReferenceSwapOperation, Vec::new(), vec![reference, updated]).unwrap();
        let live_body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let live_kernel = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadWrite)])
            .validate_body(live_body.entry_region_ref())
            .unwrap();
        let entry = live_body.entry();
        assert_eq!(
            live_kernel.lowering(),
            &[
                MockKernelInstruction::Read {
                    instruction: InstructionId::new(entry, 0),
                    output: ValueId::new(entry, live_body.instructions()[0].outputs()[0]),
                    root: ReferenceRoot::RegionInput { region: entry, input_index: 0 },
                    view: ArrayReferenceView::root(),
                },
                MockKernelInstruction::Ordinary { instruction: InstructionId::new(entry, 1) },
                MockKernelInstruction::Store {
                    instruction: InstructionId::new(entry, 2),
                    replacement: ValueId::new(entry, live_body.instructions()[2].inputs()[1]),
                    root: ReferenceRoot::RegionInput { region: entry, input_index: 0 },
                    view: ArrayReferenceView::root(),
                },
            ],
        );
    }

    #[test]
    fn test_preserved_reference_kernel_rejects_forbidden_boundary_accesses() {
        // A read-only parameter cannot be written: the second body parameter stores into its reference.
        let body = read_store_body();
        let invalid = PreservedReferenceKernelOperation::new(vec![
            operand(KernelOperandAccess::ReadOnly),
            operand(KernelOperandAccess::ReadOnly),
        ])
        .validate_body(body.entry_region_ref())
        .unwrap_err();
        assert_eq!(
            kernel_error(&invalid),
            &KernelReferenceError::ForbiddenAccess {
                parameter_index: 1,
                access: KernelOperandAccess::ReadOnly,
                mode: ReferenceAccessMode::Write,
            },
        );

        // A read-only parameter cannot accumulate either, even though the preceding read is permitted.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type(scalar_type()));
        let update = builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![reference]).unwrap()[0];
        builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update]).unwrap();
        let read_accumulate_body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let invalid = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .validate_body(read_accumulate_body.entry_region_ref())
            .unwrap_err();
        assert_eq!(
            kernel_error(&invalid),
            &KernelReferenceError::ForbiddenAccess {
                parameter_index: 0,
                access: KernelOperandAccess::ReadOnly,
                mode: ReferenceAccessMode::Accumulate,
            },
        );

        // A write-only parameter cannot accumulate because accumulation also reads the current state.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type(scalar_type()));
        let update = builder.add_constant(TestValue::Array(CpuArray::scalar(1.0_f32)));
        builder.add_instruction(ReferenceAddUpdateOperation, Vec::new(), vec![reference, update]).unwrap();
        let accumulate_body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let invalid = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::WriteOnly)])
            .validate_body(accumulate_body.entry_region_ref())
            .unwrap_err();
        assert_eq!(
            kernel_error(&invalid),
            &KernelReferenceError::ForbiddenAccess {
                parameter_index: 0,
                access: KernelOperandAccess::WriteOnly,
                mode: ReferenceAccessMode::Accumulate,
            },
        );
    }

    #[test]
    fn test_preserved_reference_kernel_reuses_composed_views() {
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let root = builder.add_input(reference_type(array_type));
        let slice = builder
            .add_instruction(ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 2, 1)]), Vec::new(), vec![root])
            .unwrap()[0];
        let sibling = builder
            .add_instruction(ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 1)]), Vec::new(), vec![root])
            .unwrap()[0];
        let index = builder.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![slice]).unwrap()[0];
        builder.add_instruction(ReferenceReadOperation, Vec::new(), vec![index]).unwrap();
        let body = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let kernel = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .validate_body(body.entry_region_ref())
            .unwrap();
        let MockKernelInstruction::Read { root, view, .. } = &kernel.lowering()[0] else {
            panic!("the final reference read must lower as a mock read")
        };
        assert_eq!(*root, ReferenceRoot::RegionInput { region: body.entry(), input_index: 0 });
        assert_eq!(
            view.transforms(),
            &[
                ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(1, 2, 1)] },
                ArrayReferenceViewTransform::Index { axis: 0, index: 1 },
            ],
        );
        assert_eq!(kernel.analysis().view(ValueId::new(body.entry(), index)).unwrap(), view);
        assert_eq!(
            kernel.analysis().root(ValueId::new(body.entry(), slice)),
            kernel.analysis().root(ValueId::new(body.entry(), sibling)),
        );
        assert_ne!(
            kernel.analysis().view(ValueId::new(body.entry(), slice)),
            kernel.analysis().view(ValueId::new(body.entry(), sibling)),
        );
        assert_eq!(
            kernel.lowering(),
            &[MockKernelInstruction::Read {
                instruction: InstructionId::new(body.entry(), 3),
                output: ValueId::new(body.entry(), body.instructions()[3].outputs()[0]),
                root: ReferenceRoot::RegionInput { region: body.entry(), input_index: 0 },
                view: view.clone(),
            }],
        );
    }

    #[test]
    fn test_preserved_reference_kernel_rejects_unimplemented_kernel_contracts() {
        let body = idle_body(scalar_type());
        let shared = PreservedReferenceKernelOperation::new(vec![KernelParameterContract::Operand(
            KernelOperandContract::new(KernelOperandAccess::ReadOnly).with_address_space(KernelAddressSpace::Shared),
        )])
        .validate_body(body.entry_region_ref())
        .unwrap_err();
        assert_eq!(
            kernel_error(&shared),
            &KernelReferenceError::OperandAddressSpace {
                parameter_index: 0,
                address_space: KernelAddressSpace::Shared,
            },
        );

        let aligned = PreservedReferenceKernelOperation::new(vec![KernelParameterContract::Operand(
            KernelOperandContract::new(KernelOperandAccess::ReadOnly)
                .with_view_alignment(KernelViewAlignment::new(NonZeroUsize::new(8).unwrap())),
        )])
        .validate_body(body.entry_region_ref())
        .unwrap_err();
        assert_eq!(kernel_error(&aligned), &KernelReferenceError::ViewAlignment { parameter_index: 0, bytes: 8 });

        let atomic = PreservedReferenceKernelOperation::new(vec![KernelParameterContract::Operand(
            KernelOperandContract::new(KernelOperandAccess::ReadOnly).with_atomicity(KernelAtomicity::Atomic),
        )])
        .validate_body(body.entry_region_ref())
        .unwrap_err();
        assert_eq!(kernel_error(&atomic), &KernelReferenceError::AtomicUnsupported { parameter_index: 0 });

        let synchronized = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .with_synchronization(KernelSynchronization::Barrier)
            .validate_body(body.entry_region_ref())
            .unwrap_err();
        assert_eq!(
            kernel_error(&synchronized),
            &KernelReferenceError::SynchronizationUnsupported { synchronization: KernelSynchronization::Barrier },
        );

        let scratch = PreservedReferenceKernelOperation::new(vec![KernelParameterContract::Scratch(
            KernelScratchContract::new(scalar_type(), KernelAddressSpace::Local),
        )])
        .validate_body(body.entry_region_ref())
        .unwrap_err();
        assert_eq!(kernel_error(&scratch), &KernelReferenceError::ScratchUnsupported { parameter_index: 0 });

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_constant(TestValue::Array(CpuArray::scalar(1.0_f32)));
        builder.add_instruction(NewReferenceOperation, Vec::new(), vec![initial]).unwrap();
        let local = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), Vec::<Placeholder>::new(), Vec::<Placeholder>::new())
            .unwrap();
        let local_error = PreservedReferenceKernelOperation::new(Vec::new())
            .validate_body(local.entry_region_ref())
            .unwrap_err();
        assert_eq!(
            kernel_error(&local_error),
            &KernelReferenceError::LocalAllocationUnsupported { instruction: InstructionId::new(local.entry(), 0) },
        );
    }

    #[test]
    fn test_preserved_reference_kernel_rejects_malformed_boundaries_and_nested_regions() {
        let body = idle_body(scalar_type());
        let count_error = PreservedReferenceKernelOperation::new(vec![
            operand(KernelOperandAccess::ReadOnly),
            operand(KernelOperandAccess::ReadOnly),
        ])
        .validate_body(body.entry_region_ref())
        .unwrap_err();
        assert_eq!(kernel_error(&count_error), &KernelReferenceError::InputCount { expected: 2, actual: 1 });

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(ArrayIrType::Array(scalar_type()));
        let non_reference = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let type_error = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .validate_body(non_reference.entry_region_ref())
            .unwrap_err();
        assert_eq!(
            kernel_error(&type_error),
            &KernelReferenceError::NonReferenceInput { parameter_index: 0, actual: "f32[]".to_string() },
        );

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type(scalar_type()));
        let escaping = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let escape_error = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .validate_body(escaping.entry_region_ref())
            .unwrap_err();
        assert_eq!(kernel_error(&escape_error), &KernelReferenceError::BodyOutputs { output_count: 1 });

        let branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), Vec::<Placeholder>::new(), Vec::<Placeholder>::new())
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first_branch = builder.import_region(branch.entry_region_ref());
        let second_branch = builder.import_region(branch.entry_region_ref());
        builder.add_input(reference_type(scalar_type()));
        let predicate = builder.add_constant(TestValue::Array(CpuArray::scalar(true)));
        builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![first_branch, second_branch], vec![predicate])
            .unwrap();
        let nested = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let nested_error = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .validate_body(nested.entry_region_ref())
            .unwrap_err();
        assert_eq!(
            kernel_error(&nested_error),
            &KernelReferenceError::NestedRegion { instruction: InstructionId::new(nested.entry(), 0) },
        );

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type(scalar_type()));
        builder.add_instruction(FreezeReferenceOperation, Vec::new(), vec![reference]).unwrap();
        let consuming = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let consume_error = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadWrite)])
            .validate_body(consuming.entry_region_ref())
            .unwrap_err();
        assert_eq!(
            consume_error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidConsume {
                instruction: InstructionId::new(consuming.entry(), 0),
                operation: "freeze_reference".to_string(),
                input_index: 0,
                root: ReferenceRoot::RegionInput { region: consuming.entry(), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_preserved_reference_kernel_rejects_malformed_public_reference_rules() {
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceRuleOperation>::new();
        let reference = builder.add_input(reference_type(scalar_type()));
        let update = builder.add_constant(TestValue::Array(CpuArray::scalar(1.0_f32)));
        builder
            .add_instruction(
                MalformedReferenceRuleOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation),
                Vec::new(),
                vec![reference, update],
            )
            .unwrap();
        let malformed_arity = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let error = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadWrite)])
            .validate_body(malformed_arity.entry_region_ref())
            .unwrap_err();
        let ProgramError::MalformedProgram(message) = error else {
            panic!("a malformed public reference rule must produce a malformed-program error")
        };
        assert_eq!(
            message,
            format!(
                concat!(
                    "kernel instruction `{}` operation `reference_add_update` violates the `read` contract: ",
                    "expected 1 inputs, 1 outputs, exactly one `Read` reference access on input 0, and canonical ",
                    "semantics, types, and effects",
                ),
                InstructionId::new(malformed_arity.entry(), 0),
            ),
        );

        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceRuleOperation>::new();
        let reference = builder.add_input(reference_type(array_type));
        builder
            .add_instruction(
                MalformedReferenceRuleOperation::ReferenceIndex(ReferenceIndexOperation::new(0, 0)),
                Vec::new(),
                vec![reference],
            )
            .unwrap();
        let missing_access = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let error = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .validate_body(missing_access.entry_region_ref())
            .unwrap_err();
        let ProgramError::MalformedProgram(message) = error else {
            panic!("a missing claimed primitive access must produce a malformed-program error")
        };
        assert_eq!(
            message,
            format!(
                concat!(
                    "kernel instruction `{}` operation `reference_index` violates the `read` contract: expected 1 ",
                    "inputs, 1 outputs, exactly one `Read` reference access on input 0, and canonical semantics, ",
                    "types, and effects",
                ),
                InstructionId::new(missing_access.entry(), 0),
            ),
        );

        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceRuleOperation>::new();
        let reference = builder.add_input(reference_type(scalar_type()));
        builder
            .add_instruction(
                MalformedReferenceRuleOperation::ReferenceRead(ReferenceReadOperation),
                Vec::new(),
                vec![reference],
            )
            .unwrap();
        let accessing_view = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let error = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .validate_body(accessing_view.entry_region_ref())
            .unwrap_err();
        let ProgramError::MalformedProgram(message) = error else {
            panic!("an accessing view rule must produce a malformed-program error")
        };
        assert_eq!(
            message,
            format!(
                "kernel instruction `{}` operation `reference_read` violates the `view` contract: expected 1 inputs, \
                 1 outputs, no reference accesses, and canonical semantics, types, and effects",
                InstructionId::new(accessing_view.entry(), 0),
            ),
        );

        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceRuleOperation>::new();
        let reference = builder.add_input(reference_type(scalar_type()));
        builder
            .add_instruction(
                MalformedReferenceRuleOperation::WrongTypeRead(WrongTypeReadOperation),
                Vec::new(),
                vec![reference],
            )
            .unwrap();
        let wrong_type = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let error = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .validate_body(wrong_type.entry_region_ref())
            .unwrap_err();
        let ProgramError::MalformedProgram(message) = error else {
            panic!("a wrongly typed primitive must produce a malformed-program error")
        };
        assert_eq!(
            message,
            format!(
                concat!(
                    "kernel instruction `{}` operation `wrong_type_read` violates the `read` contract: expected 1 ",
                    "inputs, 1 outputs, exactly one `Read` reference access on input 0, and canonical semantics, ",
                    "types, and effects",
                ),
                InstructionId::new(wrong_type.entry(), 0),
            ),
        );

        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceRuleOperation>::new();
        let reference = builder.add_input(reference_type(array_type));
        builder
            .add_instruction(
                MalformedReferenceRuleOperation::OrdinaryReferenceIndex(OrdinaryReferenceIndexOperation(
                    ReferenceIndexOperation::new(0, 0),
                )),
                Vec::new(),
                vec![reference],
            )
            .unwrap();
        let ordinary_view = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::<Placeholder>::new())
            .unwrap();
        let error = PreservedReferenceKernelOperation::new(vec![operand(KernelOperandAccess::ReadOnly)])
            .validate_body(ordinary_view.entry_region_ref())
            .unwrap_err();
        let ProgramError::MalformedProgram(message) = error else {
            panic!("a reference-bearing ordinary rule must produce a malformed-program error")
        };
        assert_eq!(
            message,
            format!(
                "kernel instruction `{}` operation `reference_index` reports ordinary lowering for a \
                 reference-bearing contract",
                InstructionId::new(ordinary_view.entry(), 0),
            ),
        );
    }

    #[test]
    fn test_preserved_reference_kernel_matches_discharge_and_oracle() {
        let source_value = CpuArray::scalar(3.0_f32);
        let destination_value = CpuArray::scalar(9.0_f32);
        let body = read_store_body();
        let operation = PreservedReferenceKernelOperation::new(vec![
            operand(KernelOperandAccess::ReadOnly),
            operand(KernelOperandAccess::WriteOnly),
        ]);
        let kernel = operation.validate_body(body.entry_region_ref()).unwrap();

        let mut states = kernel
            .bindings()
            .iter()
            .copied()
            .zip([source_value.clone(), destination_value])
            .map(|(binding, value)| (binding.root(), value))
            .collect::<HashMap<_, _>>();
        let mut values = HashMap::new();
        for instruction in kernel.lowering() {
            match instruction {
                MockKernelInstruction::Read { output, root, view, .. } => {
                    assert!(view.is_root());
                    values.insert(*output, states[root].clone());
                }
                MockKernelInstruction::Store { replacement, root, view, .. } => {
                    assert!(view.is_root());
                    states.insert(*root, values[replacement].clone());
                }
                MockKernelInstruction::Ordinary { .. } => {}
                MockKernelInstruction::Exchange { .. } | MockKernelInstruction::Accumulate { .. } => {
                    panic!("the conformance fixture lowers only a read and dead-result store")
                }
            }
        }
        let mock = vec![states[&kernel.bindings()[1].root()].clone()];

        let discharged = body.clone().discharge_references(0).unwrap();
        assert_eq!(discharged.public_output_count(), 0);
        assert_eq!(discharged.external_states().len(), 2);
        assert_eq!(discharged.external_states()[0].final_state_output_index(), None);
        assert_eq!(discharged.external_states()[1].final_state_output_index(), Some(0));
        let discharged = discharged
            .program()
            .interpret(vec![TestValue::Array(source_value.clone()), TestValue::Array(CpuArray::scalar(9.0_f32))])
            .unwrap()
            .into_iter()
            .map(|value| <TestValue as ValueProjection<ArrayType>>::into_projected(value).unwrap())
            .collect::<Vec<_>>();
        let oracle = vec![source_value];
        assert_eq!(discharged, oracle);
        assert_eq!(mock, oracle);

        // The preserved artifact deliberately has only canonical roots, bindings, and mock steps. It cannot create
        // discharged external-state slots or executable-entry alias metadata.
        assert_eq!(kernel.bindings().len(), 2);
        assert_eq!(operation.output_operand_aliases().len(), 1);
    }

    #[test]
    fn test_ordinary_xla_rejects_the_same_preserved_reference_body() {
        let mut builder = XlaProgramBuilder::new();
        let source = builder.add_input(reference_type(scalar_type()));
        let destination = builder.add_input(reference_type(scalar_type()));
        let value = builder
            .add_instruction(XlaOperation::ReferenceRead(ReferenceReadOperation), Vec::new(), vec![source])
            .unwrap()[0];
        builder
            .add_instruction(XlaOperation::ReferenceSwap(ReferenceSwapOperation), Vec::new(), vec![destination, value])
            .unwrap();
        let body: FlatXlaProgram = builder.build(Vec::new(), vec![Placeholder; 2], Vec::<Placeholder>::new()).unwrap();
        PreservedReferenceKernelOperation::new(vec![
            operand(KernelOperandAccess::ReadOnly),
            operand(KernelOperandAccess::WriteOnly),
        ])
        .validate_body(body.entry_region_ref())
        .unwrap();

        let discharged = body.clone().discharge_references(0).unwrap();
        let lowered = lower_mlir_module_for_program(
            discharged.program(),
            &[],
            &vec![scalar_type(), scalar_type()],
            &vec![scalar_type()],
            "main",
            None,
            None,
            None,
        )
        .unwrap();
        let (module, signature, _) = lowered.into_parts();
        assert_eq!(signature.physical_input_count(), 2);
        assert_eq!(signature.output_mapping(), &[Some(0)]);
        assert_eq!(module.matches("tf.aliasing_output").count(), 0);

        assert!(matches!(
            lower_mlir_module_for_program(
                &body,
                &[],
                &vec![scalar_type(), scalar_type()],
                &Vec::<ArrayType>::new(),
                "main",
                None,
                None,
                None,
            ),
            Err(LoweringError::UnresolvedReference { construct })
                if construct == "program with unresolved references",
        ));
    }
}
