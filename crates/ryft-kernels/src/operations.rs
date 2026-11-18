//! Kernel operation families over the canonical array, dimension, and reference universe.
//!
//! A kernel body is an ordinary program whose instructions are [`KernelOperation`]s. This family embeds the portable
//! [`ArrayIrOperation`] family without copying its primitive variants or redefining their semantics, and adds the
//! operations that this crate owns (i.e., kernel calls, scratch storage, tile loads, masked accesses, window masks, and
//! asynchronous copies). Every variant uses [`ArrayIrType`] and retains the effects, reference identities, region
//! contracts, and diagnostics of its payload.
//!
//! # Extensions
//!
//! Adapters supply one typed extension family for their target-specific operations by implementing
//! [`KernelExtension`]. Its [`KernelExtensionMemory`] classification specifies when the reference accesses of an
//! extension operation complete, and its semantic key identifies the operation exactly. [`NoKernelExtension`] makes the
//! extension branch uninhabited for portable kernels, and [`KernelOperation::map_extension`] converts only the
//! extension payload when a kernel moves to another execution integration. Membership in an operation family alone is
//! not target admission, because kernel validation and compiler capability checks still determine which operations a
//! particular kernel may contain.
//!
//! # Conversions and Projections
//!
//! Canonical payloads convert into the portable variant through `From` implementations that reuse their array IR
//! conversions, and the payloads that this crate owns convert into their own variants. The family also implements the
//! payload projection, collective, and lockstep contracts of `ryft-core`, in which only the portable variant can be a
//! collective or have a lockstep rule, and an extension payload is never consulted.

use std::fmt::{Display, Formatter};

use ryft_core::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceTransform, ArrayType,
    ConditionOperation, ConstantOperation, Context, DimensionOperation, DimensionSizeOperation, DimensionType,
    DimensionValue, DimensionVariable, Domain, DynamicBroadcastOperation, InterpretableOperation, InterpretationDriver,
    Operation, OperationProjection, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue,
    PartiallyEvaluatableOperation, ProgramError, ReferenceAccessDescriptor, ReferenceAccessOperation,
    ReferenceAddUpdateOperation, ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation,
    ReferenceReadOperation, ReferenceSwapOperation, ReferenceWriteOperation, RegionInterface, TypeError, Typed,
};

use crate::calls::{KernelCallOperation, KernelDefinition};
use crate::indexing::TileLoadOperation;
use crate::memory::{
    AsyncCopyOperation, MaskedLoadOperation, MaskedStoreOperation, MaskedSwapOperation, ScratchOperation,
    WaitOperation, WindowMaskOperation,
};
use crate::validation::KernelReferenceOperation;

/// Initialization and lifetime behavior not implied by ordinary reference effects.
///
/// Access modes, aliases, and allocated referents come exclusively from canonical operation effects and types.
/// This descriptor specifies when those accesses complete, and it does not declare an independent memory model.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelExtensionMemory {
    /// All declared accesses complete synchronously, and reference views retain their canonical alias semantics.
    Synchronous,

    /// Creates one uninitialized local reference allocation with no accesses.
    Allocation {
        /// Index of the output that holds the new allocation.
        output_index: usize,
    },

    /// Reserves all declared accesses until the newly allocated completion reference is waited.
    Asynchronous {
        /// Index of the output that holds the new completion reference.
        completion_output_index: usize,
    },

    /// Observes a pending completion reference of its [`KernelExtension::completion_protocol`] without completing its
    /// accesses or publishing writes. Hardware group counts and commit ordering beyond what that protocol requires
    /// remain the responsibility of the adapter.
    Commit {
        /// Index of the input that holds the observed completion reference.
        completion_input_index: usize,
    },

    /// Consumes a pending completion reference of its [`KernelExtension::completion_protocol`] and publishes its
    /// deferred writes.
    Wait {
        /// Index of the input that holds the consumed completion reference.
        completion_input_index: usize,
    },

    /// Consumes a local allocation after all outstanding accesses complete.
    Release {
        /// Index of the input that holds the consumed allocation.
        input_index: usize,
    },
}

/// Completion protocol that the token of an asynchronous extension operation follows from the operation that
/// produced it to the wait that consumes it (see [`KernelExtension::completion_protocol`]).
///
/// Two protocols are compatible exactly when they are equal (i.e., when they have the same name and the same commit
/// requirement). The portable [`AsyncCopyOperation`] and [`WaitOperation`] follow their own protocol, which is not a
/// value of this type, so no extension protocol is compatible with it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelCompletionProtocol {
    /// Name of the protocol, which includes the owner namespace of the extension (e.g., `mosaic_gpu.tmem`).
    name: &'static str,

    /// Whether exactly one commit must precede the wait.
    requires_commit: bool,
}

impl KernelCompletionProtocol {
    /// Creates a protocol named `name` whose tokens may be committed any number of times, including none, before
    /// their wait.
    pub const fn new(name: &'static str) -> Self {
        Self { name, requires_commit: false }
    }

    /// Returns this protocol with the requirement that exactly one commit precede the wait of each token.
    pub const fn with_required_commit(self) -> Self {
        Self { requires_commit: true, ..self }
    }

    /// Returns the name of this protocol.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Returns whether exactly one commit must precede the wait of each token of this protocol.
    pub const fn requires_commit(&self) -> bool {
        self.requires_commit
    }
}

impl Default for KernelCompletionProtocol {
    /// Returns the protocol named `extension`, whose tokens may be committed any number of times before their wait.
    fn default() -> Self {
        Self::new("extension")
    }
}

/// Kernel semantics that an adapter owns, over the canonical array and reference universe.
///
/// Implementations are trusted semantic contracts, not target admission. A compiler must still validate the exact
/// architecture, instruction capabilities, and resource requirements before compiling an extension. Extensions are
/// `'static` so that kernel interpretation can attach regions of any lifetime to the extended kernel operations
/// through [`AttachedRegions`](ryft_core::AttachedRegions).
pub trait KernelExtension:
    'static + ReferenceAccessOperation<Type = ArrayIrType, Transform = ArrayReferenceTransform>
{
    /// Returns the checked initialization timing and lifetime contract for this operation.
    ///
    /// The verifier checks consistency with canonical effects, output referents, and completion token types. Nested
    /// extension regions are not admitted by this contract. Unimplemented classifications fail conservatively.
    fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
        Err(TypeError::invalid(format!("kernel extension `{}` has no initialization contract", self.name())))
    }

    /// Returns the completion protocol of the token that this operation produces, commits, or waits on.
    ///
    /// The verifier and the host interpreter consult this function only for operations whose
    /// [`Self::memory_semantics`] is [`KernelExtensionMemory::Asynchronous`], [`KernelExtensionMemory::Commit`], or
    /// [`KernelExtensionMemory::Wait`]. A token carries the protocol of the asynchronous operation that produced it,
    /// and a commit or a wait accepts only a token whose protocol equals its own. It therefore never clears the
    /// reservations or publishes the deferred writes of an operation whose protocol it does not implement. When the
    /// protocol requires a commit, exactly one commit must precede the wait. Tokens of the portable
    /// [`AsyncCopyOperation`] follow a protocol that no extension can name, so an extension commit or wait never
    /// completes a portable copy and a [`WaitOperation`] never completes an extension token.
    ///
    /// The default is [`KernelCompletionProtocol::default`] for every operation of the extension family, which admits
    /// any number of commits, including none. This keeps every pairing of an extension producer with an extension
    /// commit or wait that was admitted before protocols existed, and it rejects only the pairings with portable
    /// copies, whose deferred state an extension operation cannot complete. An extension that has more than one
    /// asynchronous protocol, or a protocol that requires a commit, must override this function for all of its
    /// asynchronous, commit, and wait operations.
    fn completion_protocol(&self) -> KernelCompletionProtocol {
        KernelCompletionProtocol::default()
    }

    /// Returns an upper bound on the number of host replay steps that any binding of this operation takes, or [`None`]
    /// when no bound is known.
    ///
    /// A replay step is an operation binding or a region entry (see
    /// [`KernelDefinition::interpret_with_limits`]). The bound covers the binding of this operation and every binding
    /// and region entry that its interpretation rule performs through its interpretation driver, including those of
    /// the operations that it binds, so it is at least 1. It also covers a binding that fails after its rule bound
    /// further operations (e.g., to produce a diagnostic), because otherwise such an order would exhaust its planned
    /// budget and report a step limit instead of its own error. It must not depend on the inputs of the binding,
    /// because [`KernelDefinition::interpret_interleavings`] plans the replay cost of every order from it before any
    /// order runs. The default is [`None`], with which the explorer rejects the operation before exploration, because
    /// an interpretation rule may bind any number of further operations.
    fn replay_cost(&self) -> Option<usize> {
        None
    }

    /// Returns an exact, versioned encoding of this operation's semantic payload.
    ///
    /// The encoding must include an unambiguous owner namespace and schema version, every instruction mode and
    /// semantic option, and exact floating-point bits. It must not depend on diagnostic formatting or addresses.
    /// Inputs, results, regions, and their types are encoded by the enclosing kernel. An implementation encodes every
    /// type identity that it stores as the index that `identity` returns for it, and never by its nominal name. The
    /// enclosing kernel numbers the identities of the whole kernel through that function in order of occurrence and
    /// records their bounds, so equal identities receive equal indices, including identities that appear nowhere
    /// else in the kernel. Unsupported payloads fail explicitly, and implementing this function does not enable
    /// source deserialization.
    fn semantic_key(&self, identity: &mut dyn FnMut(&DimensionVariable) -> usize) -> Result<Vec<u8>, TypeError> {
        let _ = identity;
        Err(TypeError::invalid(format!("kernel extension `{}` has no exact semantic identity contract", self.name())))
    }
}

/// Empty extension family for kernels containing only portable operations.
///
/// This enum has no values, so a portable kernel cannot contain an unknown or unchecked target payload.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NoKernelExtension {}

impl KernelExtension for NoKernelExtension {
    fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
        match *self {}
    }

    fn semantic_key(&self, _identity: &mut dyn FnMut(&DimensionVariable) -> usize) -> Result<Vec<u8>, TypeError> {
        match *self {}
    }
}

impl Display for NoKernelExtension {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match (*self, formatter) {}
    }
}

impl Operation for NoKernelExtension {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        match *self {}
    }

    fn infer_output_types(
        &self,
        _input_types: &[ArrayIrType],
        _region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        match *self {}
    }
}

impl<C: Domain<Type = ArrayIrType>> InterpretableOperation<C> for NoKernelExtension {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        match *self {}
    }
}

impl ReferenceAccessOperation for NoKernelExtension {
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        match *self {}
    }

    fn reference_access_descriptor(
        &self,
        _input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
        match *self {}
    }

    fn with_reference_access_transforms(
        &self,
        _input_index: usize,
        _transforms: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        match *self {}
    }
}

/// Kernel body operation with a canonical portable family and an extension family that an adapter owns.
///
/// All variants use [`ArrayIrType`] and retain their payload's effects, reference identities, region contracts, and
/// diagnostics. The extension has no implicit conversion into portable operations and no fallback interpretation.
/// Construct an extension explicitly with [`KernelOperation::Extension`]. Backend selection is not stored here.
///
/// The `Operation` contract, interpretation, payload projection, collective and lockstep contracts, guard provider, and
/// payload conversions are derived through `#[derive(Operation)]`, which treats [`Portable`](Self::Portable) as a
/// nested operation family of the same type universe (i.e., `#[ryft(family)]`), so that payload projection looks
/// through it.
///
/// Partial evaluation uses the existing conservative rule that folds or residualizes the complete operation. This
/// preserves effects and nested regions without pretending that a nested portable program has a different operation
/// universe. More specialized partitioning rules can be added when the owning kernel transform contract requires them.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Operation)]
#[ryft(crate = "ryft_core", type(ArrayIrType), constant(ArrayIrValue<Array>))]
pub enum KernelOperation<Extension: Operation<Type = ArrayIrType> = NoKernelExtension> {
    /// An operation from the existing array, dimension, and reference family.
    #[ryft(family, collective, lockstep)]
    Portable(ArrayIrOperation<Array>),

    /// A functional call whose attached region owns its kernel reference windows.
    Call(KernelCallOperation),

    /// Logically uninitialized storage owned by a qualified kernel body.
    Scratch(ScratchOperation),

    /// Reads a dynamically selected reference tile with explicit padding.
    TileLoad(TileLoadOperation),

    /// Starts a qualified asynchronous reference copy.
    AsyncCopy(AsyncCopyOperation),

    /// Completes a pending qualified asynchronous copy.
    Wait(WaitOperation),

    /// Reads only selected elements from an already valid reference view.
    MaskedLoad(MaskedLoadOperation),

    /// Writes only selected elements of an already valid reference view.
    MaskedStore(MaskedStoreOperation),

    /// Replaces selected elements and returns their old values with explicit fallback elsewhere.
    MaskedSwap(MaskedSwapOperation),

    /// Returns which elements of a reference are valid for the current grid point without accessing its storage.
    WindowMask(WindowMaskOperation),

    /// A typed exact operation whose semantics and supported targets belong to its adapter.
    Extension(Extension),
}

impl<Extension: Operation<Type = ArrayIrType>> KernelOperation<Extension> {
    /// Converts only the extension payload that the adapter owns, preserving every canonical portable operation
    /// unchanged. The conversion can reject an extension family unavailable to the destination execution integration.
    pub fn map_extension<Other: Operation<Type = ArrayIrType>>(
        self,
        map: impl FnOnce(Extension) -> Result<Other, ProgramError>,
    ) -> Result<KernelOperation<Other>, ProgramError> {
        Ok(match self {
            Self::Portable(operation) => KernelOperation::Portable(operation),
            Self::Call(operation) => KernelOperation::Call(operation),
            Self::Scratch(operation) => KernelOperation::Scratch(operation),
            Self::TileLoad(operation) => KernelOperation::TileLoad(operation),
            Self::AsyncCopy(operation) => KernelOperation::AsyncCopy(operation),
            Self::Wait(operation) => KernelOperation::Wait(operation),
            Self::MaskedLoad(operation) => KernelOperation::MaskedLoad(operation),
            Self::MaskedStore(operation) => KernelOperation::MaskedStore(operation),
            Self::MaskedSwap(operation) => KernelOperation::MaskedSwap(operation),
            Self::WindowMask(operation) => KernelOperation::WindowMask(operation),
            Self::Extension(operation) => KernelOperation::Extension(map(operation)?),
        })
    }
}

impl<C, Extension: KernelExtension> PartiallyEvaluatableOperation<C> for KernelOperation<Extension>
where
    C: Context<Type = ArrayIrType, Constant = ArrayIrValue<Array>, Operation = Self>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let regions = driver.region_programs();
        if let Self::Call(call) = self {
            // The inputs are validated against the original call signature first, because the trailing scalar prefetch
            // values are split off by position.
            let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            let output_types = call.infer_output_types(&input_types, &driver.region_interfaces())?;

            // Only concrete scalar prefetch values specialize the call. Ordinary data inputs remain residual, even when
            // known, because static transforms must never execute the mutations of a kernel. Specialization executes
            // the prefetch conversions and the mapping arithmetic over constants now, so it folds their assertions
            // exactly as the default rule folds a known operation. When the context cannot fold those effects (e.g.,
            // because an earlier ordered effect is residual), the original call keeps its prefetched operands, so that
            // a failure is reported in program order when the residual program runs.
            if !call.prefetch_types().is_empty() && context.can_fold_effects(call.mapping_program_effects()) {
                let remaining = inputs.len() - call.prefetch_types().len();
                let values = inputs[remaining..]
                    .iter()
                    .map(|input| match context.parent().resolve(input.as_known()?).into_constant()? {
                        ArrayIrValue::Array(array) => Some(array),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>();
                if let Some(values) = values {
                    // The signature check above admits a prefetched value whose type is its physical prefetch type up
                    // to a sharding, so the value specializes the call as an element of that physical type.
                    let values = values
                        .iter()
                        .zip(call.prefetch_types())
                        .map(|(value, r#type)| Array::from_logical_bytes(r#type.clone(), &value.logical_bytes()))
                        .collect::<Result<Vec<_>, _>>()?;
                    let definition = KernelDefinition::new(call.clone(), driver.region(0)?.to_program())
                        .and_then(|definition| definition.specialize_prefetch(&values))
                        .map_err(ProgramError::custom)?;

                    // The placement of a prefetched value can contribute to the placement of the results (e.g., a
                    // variation along a manual mesh axis), which the specialized call no longer receives. The original
                    // call then remains residual, so that its results keep the types inferred from all of its inputs.
                    if definition
                        .operation()
                        .output_types_for(&input_types[..remaining])
                        .is_ok_and(|specialized| specialized == output_types)
                    {
                        return context.residualize(
                            Self::Call(definition.operation().clone()),
                            vec![definition.body().clone()],
                            &inputs[..remaining],
                        );
                    }
                }
            }
            context.residualize(self.clone(), regions, inputs)
        } else {
            context.fold_or_residualize(self.clone(), regions, inputs)
        }
    }
}

impl<Extension> ReferenceAccessOperation for KernelOperation<Extension>
where
    Extension: ReferenceAccessOperation<Type = ArrayIrType, Transform = ArrayReferenceTransform>,
{
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        match self {
            Self::Portable(operation) => operation.base_input_count(),
            Self::Call(operation) => operation.base_input_count(),
            Self::Scratch(operation) => operation.base_input_count(),
            Self::TileLoad(operation) => operation.base_input_count(),
            Self::AsyncCopy(operation) => operation.base_input_count(),
            Self::Wait(operation) => operation.base_input_count(),
            Self::MaskedLoad(operation) => operation.base_input_count(),
            Self::MaskedStore(operation) => operation.base_input_count(),
            Self::MaskedSwap(operation) => operation.base_input_count(),
            Self::WindowMask(operation) => operation.base_input_count(),
            Self::Extension(operation) => operation.base_input_count(),
        }
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
        match self {
            Self::Portable(operation) => operation.reference_access_descriptor(input_index),
            Self::Call(operation) => operation.reference_access_descriptor(input_index),
            Self::Scratch(operation) => operation.reference_access_descriptor(input_index),
            Self::TileLoad(operation) => operation.reference_access_descriptor(input_index),
            Self::AsyncCopy(operation) => operation.reference_access_descriptor(input_index),
            Self::Wait(operation) => operation.reference_access_descriptor(input_index),
            Self::MaskedLoad(operation) => operation.reference_access_descriptor(input_index),
            Self::MaskedStore(operation) => operation.reference_access_descriptor(input_index),
            Self::MaskedSwap(operation) => operation.reference_access_descriptor(input_index),
            Self::WindowMask(operation) => operation.reference_access_descriptor(input_index),
            Self::Extension(operation) => operation.reference_access_descriptor(input_index),
        }
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        Ok(match self {
            Self::Portable(operation) => {
                Self::Portable(operation.with_reference_access_transforms(input_index, transforms)?)
            }
            Self::Call(operation) => Self::Call(operation.with_reference_access_transforms(input_index, transforms)?),
            Self::Scratch(operation) => {
                Self::Scratch(operation.with_reference_access_transforms(input_index, transforms)?)
            }
            Self::TileLoad(operation) => {
                Self::TileLoad(operation.with_reference_access_transforms(input_index, transforms)?)
            }
            Self::AsyncCopy(operation) => {
                Self::AsyncCopy(operation.with_reference_access_transforms(input_index, transforms)?)
            }
            Self::Wait(operation) => Self::Wait(operation.with_reference_access_transforms(input_index, transforms)?),
            Self::MaskedLoad(operation) => {
                Self::MaskedLoad(operation.with_reference_access_transforms(input_index, transforms)?)
            }
            Self::MaskedStore(operation) => {
                Self::MaskedStore(operation.with_reference_access_transforms(input_index, transforms)?)
            }
            Self::MaskedSwap(operation) => {
                Self::MaskedSwap(operation.with_reference_access_transforms(input_index, transforms)?)
            }
            Self::WindowMask(operation) => {
                Self::WindowMask(operation.with_reference_access_transforms(input_index, transforms)?)
            }
            Self::Extension(operation) => {
                Self::Extension(operation.with_reference_access_transforms(input_index, transforms)?)
            }
        })
    }
}

/// Implements the lift of one canonical payload into the portable variant of the kernel operation family.
macro_rules! kernel_operation_from {
    // Lifts the payload through its existing array IR conversion, which preserves its normalization.
    ($operation:ty) => {
        impl<Extension: Operation<Type = ArrayIrType>> From<$operation> for KernelOperation<Extension> {
            fn from(operation: $operation) -> Self {
                Self::Portable(operation.into())
            }
        }
    };
}

kernel_operation_from!(ArrayOperation<Array>);
kernel_operation_from!(ConditionOperation<ArrayIrType>);
kernel_operation_from!(DimensionOperation<DimensionValue>);
kernel_operation_from!(ConstantOperation<DimensionValue>);
kernel_operation_from!(DimensionSizeOperation);
kernel_operation_from!(DynamicBroadcastOperation);
kernel_operation_from!(ReferenceNewOperation<ArrayType, ArrayIrType>);
kernel_operation_from!(ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>);
kernel_operation_from!(ReferenceWriteOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>);
kernel_operation_from!(ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>);
kernel_operation_from!(ReferenceAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>);
kernel_operation_from!(ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>);
kernel_operation_from!(ReferenceFreezeOperation<ArrayType, ArrayIrType>);

impl<Extension> KernelReferenceOperation for KernelOperation<Extension>
where
    Extension: ReferenceAccessOperation<Type = ArrayIrType, Transform = ArrayReferenceTransform>,
{
    fn swap_output_index(&self) -> Option<usize> {
        match self {
            Self::Portable(operation) => operation.swap_output_index(),
            Self::MaskedSwap(_) => Some(0),
            // Unknown extension swaps retain their declared read-write effects until explicitly recognized.
            _ => None,
        }
    }
}

impl<Extension: Operation<Type = ArrayIrType>> OperationProjection<ArrayType> for KernelOperation<Extension> {
    type Projected = ArrayOperation<Array>;
}

impl<Extension: Operation<Type = ArrayIrType>> OperationProjection<DimensionType> for KernelOperation<Extension> {
    type Projected = DimensionOperation<DimensionValue>;
}

impl<'o, Extension: Operation<Type = ArrayIrType>> TryFrom<&'o KernelOperation<Extension>>
    for &'o ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>
{
    type Error = TypeError;

    fn try_from(operation: &'o KernelOperation<Extension>) -> Result<Self, TypeError> {
        match operation {
            KernelOperation::Portable(operation) => operation.try_into(),
            KernelOperation::Call(_)
            | KernelOperation::Scratch(_)
            | KernelOperation::TileLoad(_)
            | KernelOperation::AsyncCopy(_)
            | KernelOperation::Wait(_)
            | KernelOperation::MaskedLoad(_)
            | KernelOperation::MaskedStore(_)
            | KernelOperation::MaskedSwap(_)
            | KernelOperation::WindowMask(_) => {
                Err(TypeError::invalid("cannot project a kernel-specific operation into a reference swap"))
            }
            KernelOperation::Extension(operation) => Err(TypeError::invalid(format!(
                "cannot project extension operation `{}` into a canonical reference swap",
                operation.name(),
            ))),
        }
    }
}

/// Returns whether a commit (when `commit` is set) or a wait that follows the `completion` protocol may complete a
/// pending token that follows the `producer` protocol now, where `committed` records whether a commit has already
/// observed the token. [`None`] denotes the protocol of the portable async copy and wait, which has no commit.
pub(crate) fn admits_completion(
    producer: Option<KernelCompletionProtocol>,
    committed: bool,
    completion: Option<KernelCompletionProtocol>,
    commit: bool,
) -> bool {
    producer == completion && producer.is_none_or(|protocol| !protocol.requires_commit() || committed != commit)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Cow;
    use std::cell::Cell;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::macros::check_count;
    use ryft_core::{
        AddOperation, ArrayReference, ArrayReferenceTransformIndex, AssertOperation, AssertionError, AttachedRegions,
        Condition, DataType, Dimension, DimensionBounds, DimensionError, DimensionMulOperation, DimensionVariable,
        EagerContext, EffectClasses, Effects, EmptyRegionDriver, ErasedOperation, LogicalMesh, MeshAxis, MeshAxisType,
        NegOperation, OperationFoldOutput, OperationPayloadProjection, PartialEvaluation, PartialEvaluationOutput,
        PartialValue, PartitionedProgram, Placeholder, ProgramBuilder, ReferenceAccessMode, ReferenceEffect,
        ReferenceRead, ReferenceSwap, ReferenceType, ReferenceWrite, RegionDataFlow, RegionDriver, RegionRef,
        ScanOperation, Shape, Sharding, SubOperation, TagOperation, Tracer, TracingContext, TypeIdentityRenaming,
    };

    use crate::authoring::{MappingExpression, ParameterWindow, mapped_call};
    use crate::calls::{KernelError, KernelParameter};
    use crate::grids::Grid;
    use crate::mappings::{BlockMapping, BlockMappingError, BoundaryPolicy};
    use crate::validation::KernelParameterAccess;

    use super::*;

    /// Kernel extension family shared by the kernel unit tests, so that they instantiate one extended operation family.
    ///
    /// Each variant reproduces one test fixture. The trait implementations dispatch per variant, and variants that do
    /// not implement a contract fail with the same diagnostics as the corresponding [`KernelExtension`] defaults.
    #[derive(Clone, Debug)]
    pub(crate) enum TestExtension {
        /// Subtracts its input from itself by binding a `sub` through its interpretation driver, as higher-order rules
        /// that execute nested work on their own behalf do.
        Rebinding,

        /// Payload intentionally omitted from the diagnostic text, demonstrating conservative eligibility. Only an
        /// `exact` extension provides a versioned semantic key over its payload.
        HiddenKey { key: u32, exact: bool },

        /// Carries a dimension identity in its payload, which its exact semantic key encodes by name and bounds. It
        /// has no inputs, results, or effects.
        Dimension(DimensionVariable),

        /// Creates one uninitialized local reference allocation with the given referent.
        Allocate(ArrayType),

        /// Allocates a reference while claiming synchronous memory semantics.
        MisclassifiedAllocation,

        /// Has no accesses and completes synchronously.
        Synchronous,

        /// Synchronously copies the selection of its source reference into the selection of its destination reference.
        /// The source is input 0 and the destination input 1 when `source_first` is set, and the other way around
        /// otherwise, so the canonical effects declare the read first only in the former case. `transforms` holds the
        /// static transforms applied to each input, indexed by input position.
        CopyInto { source_first: bool, transforms: [Vec<ArrayReferenceTransform>; 2] },

        /// Asynchronous update from two sources that reads two references, writes a third, and allocates a completion.
        /// Its host interpreter writes the first source into the destination when it is issued, which no other
        /// instruction can observe before the wait, because the destination stays reserved until then. When `strict`
        /// is set, its completion follows a protocol that requires exactly one commit before the wait, and otherwise
        /// it follows the default protocol of [`KernelExtension::completion_protocol`].
        Update { strict: bool },

        /// Observes a pending completion reference of the protocol that `strict` selects.
        Commit { strict: bool },

        /// Consumes a pending completion reference of the protocol that `strict` selects.
        Wait { strict: bool },

        /// Consumes a local allocation.
        Release,
    }

    impl Operation for TestExtension {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Rebinding => "test_rebinding",
                Self::HiddenKey { .. } => "hidden_key_extension",
                Self::Dimension(_) => "test_dimension",
                Self::Allocate(_) => "test_allocate",
                Self::MisclassifiedAllocation => "test_misclassified_allocation",
                Self::Synchronous => "test_synchronous",
                Self::CopyInto { .. } => "test_copy_into",
                Self::Update { .. } => "test_async_update",
                Self::Commit { .. } => "test_commit",
                Self::Wait { .. } => "test_wait",
                Self::Release => "test_release",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            _region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(match self {
                Self::Rebinding => {
                    check_count!("input", input_types, 1, TypeError);
                    input_types.to_vec()
                }
                Self::HiddenKey { .. }
                | Self::Dimension(_)
                | Self::Synchronous
                | Self::CopyInto { .. }
                | Self::Commit { .. }
                | Self::Wait { .. }
                | Self::Release => vec![],
                Self::Allocate(referent) => vec![ArrayIrType::Reference(ReferenceType::new(referent.clone()))],
                Self::MisclassifiedAllocation => {
                    vec![ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::I32)))]
                }
                Self::Update { .. } => {
                    vec![ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)))]
                }
            })
        }

        fn effects(&self) -> Cow<'_, Effects> {
            let declarations = match self {
                Self::Rebinding | Self::HiddenKey { .. } | Self::Dimension(_) => {
                    return Cow::Borrowed(Effects::empty());
                }
                Self::Allocate(_) | Self::MisclassifiedAllocation => {
                    vec![ReferenceEffect::Allocate { output_index: 0 }]
                }
                Self::Synchronous => vec![],
                Self::CopyInto { source_first, .. } => {
                    let source = usize::from(!*source_first);
                    vec![
                        ReferenceEffect::Access { input_index: source, mode: ReferenceAccessMode::Read },
                        ReferenceEffect::Access { input_index: 1 - source, mode: ReferenceAccessMode::Write },
                    ]
                }
                Self::Update { .. } => vec![
                    ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 1, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Write },
                    ReferenceEffect::Allocate { output_index: 0 },
                ],
                Self::Commit { .. } => {
                    vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }]
                }
                Self::Wait { .. } | Self::Release => {
                    vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Consume }]
                }
            };
            Cow::Owned(Effects::new(EffectClasses::NONE, declarations).unwrap())
        }

        fn rename_type_identities(
            &self,
            renaming: &TypeIdentityRenaming<DimensionVariable>,
        ) -> Result<Self, TypeError> {
            Ok(match self {
                Self::Dimension(variable) => Self::Dimension(renaming.rename(variable)),
                _ => self.clone(),
            })
        }
    }

    impl ReferenceAccessOperation for TestExtension {
        type Transform = ArrayReferenceTransform;

        fn base_input_count(&self) -> usize {
            match self {
                Self::Rebinding
                | Self::HiddenKey { .. }
                | Self::Dimension(_)
                | Self::Allocate(_)
                | Self::MisclassifiedAllocation
                | Self::Synchronous => 0,
                Self::CopyInto { .. } => 2,
                Self::Update { .. } => 3,
                Self::Commit { .. } | Self::Wait { .. } | Self::Release => 1,
            }
        }

        fn reference_access_descriptor(
            &self,
            input_index: usize,
        ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
            match self {
                Self::Rebinding | Self::HiddenKey { .. } | Self::Dimension(_) => None,
                Self::CopyInto { transforms, .. } => {
                    transforms.get(input_index).map(|transforms| ReferenceAccessDescriptor::new(transforms, 2..2))
                }
                _ => self.effects().reference_accesses().any(|(index, _)| index == input_index).then(|| {
                    let count = self.base_input_count();
                    ReferenceAccessDescriptor::new(&[], count..count)
                }),
            }
        }

        fn with_reference_access_transforms(
            &self,
            input_index: usize,
            transforms: Vec<ArrayReferenceTransform>,
        ) -> Result<Self, ProgramError> {
            match self {
                Self::Rebinding => Err(ProgramError::UnsupportedOperation {
                    message: "test extension has no reference accesses".to_owned(),
                }),
                Self::HiddenKey { .. } => Err(ProgramError::UnsupportedOperation {
                    message: "hidden extension has no reference accesses".to_owned(),
                }),
                Self::CopyInto { source_first, transforms: current } if input_index < 2 => {
                    let mut current = current.clone();
                    current[input_index] = transforms;
                    Ok(Self::CopyInto { source_first: *source_first, transforms: current })
                }
                _ if transforms.is_empty() && self.reference_access_descriptor(input_index).is_some() => {
                    Ok(self.clone())
                }
                _ => Err(ProgramError::UnsupportedOperation {
                    message: "test extension has no reference transforms".to_owned(),
                }),
            }
        }
    }

    impl KernelExtension for TestExtension {
        fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
            Ok(match self {
                Self::Rebinding
                | Self::Dimension(_)
                | Self::MisclassifiedAllocation
                | Self::Synchronous
                | Self::CopyInto { .. } => KernelExtensionMemory::Synchronous,
                Self::HiddenKey { .. } => {
                    return Err(TypeError::invalid(format!(
                        "kernel extension `{}` has no initialization contract",
                        self.name(),
                    )));
                }
                Self::Allocate(_) => KernelExtensionMemory::Allocation { output_index: 0 },
                Self::Update { .. } => KernelExtensionMemory::Asynchronous { completion_output_index: 0 },
                Self::Commit { .. } => KernelExtensionMemory::Commit { completion_input_index: 0 },
                Self::Wait { .. } => KernelExtensionMemory::Wait { completion_input_index: 0 },
                Self::Release => KernelExtensionMemory::Release { input_index: 0 },
            })
        }

        fn completion_protocol(&self) -> KernelCompletionProtocol {
            match self {
                Self::Update { strict: true } | Self::Commit { strict: true } | Self::Wait { strict: true } => {
                    KernelCompletionProtocol::new("test.strict").with_required_commit()
                }
                _ => KernelCompletionProtocol::default(),
            }
        }

        fn replay_cost(&self) -> Option<usize> {
            match self {
                // The binding of the extension and of the `sub` that its rule binds.
                Self::Rebinding => Some(2),
                _ => None,
            }
        }

        fn semantic_key(&self, identity: &mut dyn FnMut(&DimensionVariable) -> usize) -> Result<Vec<u8>, TypeError> {
            match self {
                Self::HiddenKey { key: payload, exact: true } => {
                    let mut key = b"test.hidden_key_extension\0v1\0".to_vec();
                    key.extend_from_slice(&payload.to_le_bytes());
                    Ok(key)
                }
                Self::Dimension(variable) => {
                    let mut key = b"test.dimension\0v1\0".to_vec();
                    key.extend_from_slice(&identity(variable).to_le_bytes());
                    Ok(key)
                }
                _ => Err(TypeError::invalid(format!(
                    "kernel extension `{}` has no exact semantic identity contract",
                    self.name(),
                ))),
            }
        }
    }

    impl InterpretableOperation<EagerContext<ArrayIrValue<Array>, KernelOperation<TestExtension>>> for TestExtension {
        fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation<TestExtension>>>>(
            &self,
            context: &EagerContext<ArrayIrValue<Array>, KernelOperation<TestExtension>>,
            driver: &D,
            inputs: &[ArrayIrValue<Array>],
        ) -> Result<Vec<ArrayIrValue<Array>>, ProgramError> {
            match self {
                Self::Rebinding => {
                    check_count!("input", inputs, 1, ProgramError);
                    let operation =
                        KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Sub(SubOperation::new())));
                    driver.bind(context, operation, AttachedRegions::Empty, &[inputs[0].clone(), inputs[0].clone()])
                }
                Self::Update { .. } => {
                    check_count!("input", inputs, 3, ProgramError);
                    let (ArrayIrValue::Reference(source), ArrayIrValue::Reference(destination)) =
                        (&inputs[0], &inputs[2])
                    else {
                        return Err(ProgramError::MalformedProgram("test update requires references".to_owned()));
                    };
                    destination.write(source.read()?)?;
                    let token = ArrayReference::new(Array::new(ArrayType::scalar(DataType::Token), vec![])?);
                    Ok(vec![ArrayIrValue::Reference(token)])
                }
                Self::Commit { .. } => Ok(vec![]),
                Self::Wait { .. } => {
                    let [ArrayIrValue::Reference(token)] = inputs else {
                        return Err(ProgramError::MalformedProgram("test wait requires one reference".to_owned()));
                    };
                    token.freeze()?;
                    Ok(vec![])
                }
                _ => {
                    Err(ProgramError::MalformedProgram(format!("test extension `{}` has no interpreter", self.name())))
                }
            }
        }
    }

    #[test]
    fn test_kernel_operation() {
        let operation: KernelOperation = KernelOperation::from(ArrayOperation::Add(AddOperation::new()));
        assert_eq!(operation.name(), "add");
        assert_eq!(operation.to_string(), "add");
        assert_eq!(operation.region_slots(), &[]);
    }

    #[test]
    fn test_kernel_operation_map_extension() {
        let portable = KernelOperation::<NoKernelExtension>::from(ArrayOperation::Add(AddOperation::new()));
        let mapped = portable.map_extension::<ArrayIrOperation<Array>>(|extension| match extension {}).unwrap();
        assert!(matches!(mapped, KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Add(_)))));
        let extension =
            KernelOperation::Extension(ArrayIrOperation::<Array>::from(ArrayOperation::Add(AddOperation::new())));
        assert!(matches!(extension.clone().map_extension(Ok).unwrap(), KernelOperation::Extension(_)));
        assert_eq!(
            extension
                .map_extension::<NoKernelExtension>(|_| Err(ProgramError::UnsupportedOperation {
                    message: "destination rejects this extension family".to_owned(),
                }))
                .unwrap_err(),
            ProgramError::UnsupportedOperation { message: "destination rejects this extension family".to_owned() },
        );
    }

    #[test]
    fn test_kernel_operation_identity_forwarding() {
        #[derive(Clone)]
        struct IdentityDefiningOperation {
            identity: DimensionVariable,
        }

        impl Operation for IdentityDefiningOperation {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                "identity_defining"
            }

            fn defined_type_identities(&self) -> Cow<'_, [DimensionVariable]> {
                Cow::Borrowed(std::slice::from_ref(&self.identity))
            }

            fn infer_output_types(
                &self,
                _input_types: &[ArrayIrType],
                _region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                Ok(Vec::new())
            }
        }

        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        let extension = KernelOperation::Extension(IdentityDefiningOperation { identity: rows.clone() });
        assert_eq!(extension.defined_type_identities().as_ref(), std::slice::from_ref(&rows));
        let portable = KernelOperation::<IdentityDefiningOperation>::from(ArrayOperation::Add(AddOperation::new()));
        assert!(portable.defined_type_identities().is_empty());
    }

    #[test]
    fn test_kernel_operation_type_inference() {
        let operation = KernelOperation::<NoKernelExtension>::from(ArrayOperation::Add(AddOperation::new()));
        let r#type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        assert_eq!(operation.infer_output_types(&[r#type.clone(), r#type.clone()], &[]), Ok(vec![r#type]));
    }

    #[test]
    fn test_kernel_operation_condition_region_interfaces() {
        // Type inference and the validation of region interfaces dispatch through the kernel wrapper exactly like
        // through the bare `condition` operation that it wraps.
        let bare = ConditionOperation::<ArrayIrType>::new();
        let operation =
            KernelOperation::<NoKernelExtension>::Portable(ArrayIrOperation::Condition(ConditionOperation::new()));
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dynamic_type =
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)])));
        let predicate_type = ArrayIrType::Array(ArrayType::scalar(DataType::Boolean));
        let three_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        let four_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [4]));

        // The identity and constant branches declare different outputs, `f64[extent]` and `f64[3]`, which agree once an
        // `f64[3]` input refines `f64[extent]`.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(dynamic_type.clone());
        let identity_branch = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(dynamic_type);
        let output = builder.add_constant(ArrayIrValue::Array(Array::vector(vec![2.0f64; 3]).unwrap()));
        let constant_branch = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let interfaces = [identity_branch.interface().clone(), constant_branch.interface().clone()];
        let refined_input_types = [predicate_type.clone(), three_type.clone()];
        let mismatched_input_types = [predicate_type, four_type];
        let mismatch =
            TypeError::invalid("`condition` branch output type signature mismatch: expected [f64[4]] but got [f64[3]]");
        assert_eq!(operation.infer_output_types(&refined_input_types, &interfaces), Ok(vec![three_type.clone()]));
        assert_eq!(bare.infer_output_types(&refined_input_types, &interfaces), Ok(vec![three_type.clone()]));
        assert_eq!(operation.infer_output_types(&mismatched_input_types, &interfaces), Err(mismatch.clone()));
        assert_eq!(bare.infer_output_types(&mismatched_input_types, &interfaces), Err(mismatch));
        assert_eq!(operation.validate_region_interfaces(&interfaces), Ok(()));
        assert_eq!(bare.validate_region_interfaces(&interfaces), Ok(()));

        // Branches must declare the same input types, even when refinement by the actual inputs would make them agree.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(three_type);
        let static_identity_branch = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let interfaces = [identity_branch.interface().clone(), static_identity_branch.interface().clone()];
        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`condition` branch input type signature mismatch: expected [f64[extent]] but got [f64[3]]",
        )));
        assert_eq!(operation.validate_region_interfaces(&interfaces), expected);
        assert_eq!(bare.validate_region_interfaces(&interfaces), expected);
    }

    #[test]
    fn test_kernel_operation_fold() {
        // A kernel body must fold exactly like the portable family that it wraps, so that the same source stages the
        // same instructions in both families.
        let rows = DimensionType::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let one = DimensionType::new("one", DimensionBounds::new(1, Some(2)).unwrap());
        let multiplication = ArrayIrOperation::<Array>::from(DimensionMulOperation::new(&rows, &one).unwrap());
        let operation = KernelOperation::<NoKernelExtension>::Portable(multiplication.clone());
        let input_types = [ArrayIrType::Dimension(rows), ArrayIrType::Dimension(one)];
        assert_eq!(multiplication.fold(&input_types, &[]), Ok(Some(vec![OperationFoldOutput::Input(0)])));
        assert_eq!(operation.fold(&input_types, &[]), Ok(Some(vec![OperationFoldOutput::Input(0)])));

        // Operations without a fold rule keep reporting that they do not fold.
        let addition = KernelOperation::<NoKernelExtension>::from(ArrayOperation::Add(AddOperation::new()));
        let r#type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        assert_eq!(addition.fold(&[r#type.clone(), r#type], &[]), Ok(None));
    }

    #[test]
    fn test_kernel_operation_region_data_flow() {
        let opaque = KernelOperation::<NoKernelExtension>::from(ArrayOperation::Add(AddOperation::new()));
        let ordinary = KernelOperation::<NoKernelExtension>::from(ArrayOperation::Condition(ConditionOperation::new()));
        assert!(matches!(opaque.region_data_flow(), RegionDataFlow::Opaque));
        assert!(matches!(ordinary.region_data_flow(), RegionDataFlow::Provenance));

        // Kernel dispatch forwards the custom region data flow of a portable or extension payload itself.
        let recurrent = ArrayIrOperation::<Array>::from(ArrayOperation::Scan(ScanOperation::new(2, 3usize)));
        let portable = KernelOperation::<ArrayIrOperation<Array>>::Portable(recurrent.clone());
        let KernelOperation::Portable(payload) = &portable else { unreachable!() };
        assert!(matches!(
            (portable.region_data_flow(), payload.region_data_flow()),
            (RegionDataFlow::Custom(forwarded), RegionDataFlow::Custom(expected)) if std::ptr::eq(forwarded, expected),
        ));
        let extension = KernelOperation::<ArrayIrOperation<Array>>::Extension(recurrent);
        let KernelOperation::Extension(payload) = &extension else { unreachable!() };
        assert!(matches!(
            (extension.region_data_flow(), payload.region_data_flow()),
            (RegionDataFlow::Custom(forwarded), RegionDataFlow::Custom(expected)) if std::ptr::eq(forwarded, expected),
        ));
    }

    #[test]
    fn test_kernel_operation_defers_reference_discharge() {
        #[derive(Clone)]
        struct DeferringOperation;

        impl Operation for DeferringOperation {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                "deferring"
            }

            fn infer_output_types(
                &self,
                _input_types: &[ArrayIrType],
                _region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                Ok(Vec::new())
            }

            fn defers_reference_discharge(&self) -> bool {
                true
            }
        }

        assert!(KernelOperation::Extension(DeferringOperation).defers_reference_discharge());
        let portable = KernelOperation::<DeferringOperation>::from(ArrayOperation::Add(AddOperation::new()));
        assert!(!portable.defers_reference_discharge());
    }

    #[test]
    fn test_kernel_operation_interpretation() {
        let r#type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        let (output_types, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<_>| {
                inputs[0].context().bind(
                    ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())),
                    Vec::new(),
                    &inputs,
                )
            },
            vec![r#type.clone(), r#type.clone()],
        )
        .unwrap();
        let (canonical_output_types, canonical_program) =
            TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
                |inputs: Vec<_>| {
                    inputs[0].context().bind(
                        ArrayIrOperation::from(ArrayOperation::Add(AddOperation::new())),
                        Vec::new(),
                        &inputs,
                    )
                },
                vec![r#type.clone(), r#type.clone()],
            )
            .unwrap();
        assert_eq!(output_types, vec![r#type]);
        assert_eq!(output_types, canonical_output_types);
        assert_eq!(program.to_string(), canonical_program.to_string());
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![3.0f32, 5.0]).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![4.0f32, 7.0]).unwrap())]),
        );
    }

    #[test]
    fn test_kernel_operation_condition() {
        // `condition` serves the kernel operation family that the `#[kernel]` macro lowers Rust `if` statements to,
        // tracing each branch function exactly once and selecting its result at runtime.
        let true_calls = Cell::new(0);
        let false_calls = Cell::new(0);
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<Tracer<_>>| {
                inputs[0].condition(
                    |inputs| {
                        true_calls.set(true_calls.get() + 1);
                        inputs[0].context().bind(
                            ArrayIrOperation::<Array>::from(ArrayOperation::Add(AddOperation::new())),
                            Vec::new(),
                            &[inputs[0].clone(), inputs[0].clone()],
                        )
                    },
                    |inputs| {
                        false_calls.set(false_calls.get() + 1);
                        inputs[0].context().bind(
                            ArrayIrOperation::<Array>::from(ArrayOperation::Neg(NegOperation::new())),
                            Vec::new(),
                            &inputs,
                        )
                    },
                    vec![inputs[1].clone()],
                )
            },
            vec![
                ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)),
                ArrayIrType::Array(ArrayType::scalar(DataType::I64)),
            ],
        )
        .unwrap();
        assert_eq!((true_calls.get(), false_calls.get()), (1, 1));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:i64[] .
                let %2:i64[] = condition %0 %1 [
                    true={
                        lambda %0:i64[] .
                        let %1:i64[] = add %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:i64[] .
                        let %1:i64[] = neg %0
                        in (%1)
                    },
                ]
                in (%2)"},
        );
        for (predicate, expected) in [(true, 6i64), (false, -3i64)] {
            assert_eq!(
                program.interpret(vec![
                    ArrayIrValue::Array(Array::scalar(predicate).unwrap()),
                    ArrayIrValue::Array(Array::scalar(3i64).unwrap()),
                ]),
                Ok(vec![ArrayIrValue::Array(Array::scalar(expected).unwrap())]),
            );
        }
    }

    #[test]
    fn test_kernel_operation_partial_evaluation() {
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        mapping.add_input(DimensionType::new("offset", DimensionBounds::non_negative(Some(4)).unwrap()).into());
        let call = KernelCallOperation::new_with_prefetch(
            Grid::new(vec![]).unwrap(),
            vec![
                KernelParameter::new(
                    ArrayType::scalar(DataType::I32),
                    KernelParameterAccess::WriteOnly,
                    BlockMapping::new(
                        mapping.build(vec![], vec![Placeholder], vec![]).unwrap(),
                        vec![],
                        BoundaryPolicy::InBounds,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace_with_prefetch(call, |(references, _, values)| references[0].write(&values[0]))
                .unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let region = builder.import_region(definition.body().entry_region_ref());
        let outputs = builder
            .add_instruction(KernelOperation::Call(definition.operation().clone()), vec![region], vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let value = ArrayIrValue::Array(Array::scalar(2i32).unwrap());
        let evaluation = program.partially_evaluate(&[PartialValue::Known(value.clone())]).unwrap();
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda  .
                let %0:i32[] = kernel_call [
                    grid=[],
                    parameter_0=write-only i32[] [] InBounds,
                    mapping_0={
                        lambda  .
                        let %0:dimension<2> = const 2
                        in ()
                    },
                ] [
                    body={
                        lambda %0:ref<i32[]> .
                        let %1:i32[] = const 2
                            () = reference_write %0 %1
                        in ()
                    },
                ]
                in (%0)"},
        );
        assert_eq!(evaluation.interpret(&EagerContext::new(), &[]).unwrap(), vec![value]);
        let unknown = program
            .partially_evaluate(&[PartialValue::Unknown(ArrayType::scalar(DataType::I32).into())])
            .unwrap();
        assert_eq!(
            unknown.program().to_string(),
            indoc! {"
                lambda %0:i32[] .
                let %1:i32[] = kernel_call [
                    grid=[],
                    prefetch_0=i32[],
                    identities=[0],
                    identity_0_bounds=[0, 4),
                    parameter_0=write-only i32[] [] InBounds,
                    mapping_0={
                        lambda %0:dimension<offset ∈ [0, 4)> .
                        in ()
                    },
                ] %0 [
                    body={
                        lambda %0:ref<i32[]>, %1:i32[] .
                        let () = reference_write %0 %1
                        in ()
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_kernel_operation_partial_evaluation_mapping_failures() {
        // A read-only call with no results whose mapping computes `0 / d` for the prefetched `d`.
        let call = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, vec![4]),
                KernelParameterAccess::ReadOnly,
                ParameterWindow::ElementStarts {
                    block_shape: vec![1],
                    boundary: BoundaryPolicy::InBounds,
                    starts: vec![MappingExpression::Constant(0) / MappingExpression::Prefetch(0)],
                },
            )],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace_with_prefetch(call, |_| Ok(())).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::I32, vec![4]).into());
        let prefetch = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let region = builder.import_region(definition.body().entry_region_ref());
        builder
            .add_instruction(
                KernelOperation::Call(definition.operation().clone()),
                vec![region],
                vec![input, prefetch],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 2], vec![])
            .unwrap();
        let data = PartialValue::Unknown(ArrayType::new_static(DataType::I32, vec![4]).into());

        // An unknown prefetched value residualizes the call, which retains its assertion although it has no results.
        let unknown = program
            .partially_evaluate(&[data.clone(), PartialValue::Unknown(ArrayType::scalar(DataType::I32).into())])
            .unwrap();
        assert_eq!(
            unknown.program().to_string(),
            indoc! {"
                lambda %0:i32[4], %1:i32[] .
                let () = kernel_call [
                    grid=[],
                    prefetch_0=i32[],
                    identities=[0],
                    identity_0_bounds=[0, ∞),
                    parameter_0=read-only i32[4] [1] InBounds,
                    mapping_0={
                        lambda %0:dimension<prefetch_0 ∈ [0, ∞)> .
                        let %1:dimension<0> = const 0
                            %2:dimension<0> = dimension_div [requires_runtime_assertion=true] %1 %0
                        in (%2)
                    },
                ] %0 %1 [
                    body={
                        lambda %0:ref<i32[1]>, %1:i32[] .
                        in ()
                    },
                ]
                in ()"},
        );

        // A known zero fails specialization exactly as interpretation does.
        let zero = ArrayIrValue::Array(Array::scalar(0i32).unwrap());
        let violation = ProgramError::custom(DimensionError::RequirementViolation {
            message: "`dimension_div` requires right > 0; got left=0, right=0".to_owned(),
        });
        assert_eq!(
            program.partially_evaluate(&[data.clone(), PartialValue::Known(zero)]).unwrap_err(),
            ProgramError::custom(KernelError::Mapping(BlockMappingError::Program(violation))),
        );

        // A known valid value specializes the mapping to a constant start, so the pure call is eliminated.
        let two = ArrayIrValue::Array(Array::scalar(2i32).unwrap());
        let known = program.partially_evaluate(&[data, PartialValue::Known(two)]).unwrap();
        assert_eq!(
            known.program().to_string(),
            indoc! {"
                lambda %0:i32[4] .
                in ()"},
        );
    }

    #[test]
    fn test_kernel_operation_partial_evaluation_preserves_failure_order() {
        // An assertion precedes a read-only call whose mapping computes `0 / d` for the prefetched `d`.
        let call = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![(
                ArrayType::new_static(DataType::I32, vec![4]),
                KernelParameterAccess::ReadOnly,
                ParameterWindow::ElementStarts {
                    block_shape: vec![1],
                    boundary: BoundaryPolicy::InBounds,
                    starts: vec![MappingExpression::Constant(0) / MappingExpression::Prefetch(0)],
                },
            )],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace_with_prefetch(call, |_| Ok(())).unwrap();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
        let condition = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(ArrayType::new_static(DataType::I32, vec![4]).into());
        let prefetch = builder.add_input(ArrayType::scalar(DataType::I32).into());
        let assertion = KernelOperation::Portable(ArrayIrOperation::Assert(AssertOperation::new("first")));
        builder.add_instruction(assertion, vec![], vec![condition], None).unwrap();
        let region = builder.import_region(definition.body().entry_region_ref());
        builder
            .add_instruction(
                KernelOperation::Call(definition.operation().clone()),
                vec![region],
                vec![input, prefetch],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(vec![], vec![Placeholder; 3], vec![])
            .unwrap();
        let data = Array::vector(vec![1i32, 2, 3, 4]).unwrap();
        let zero = Array::scalar(0i32).unwrap();
        let first = AssertionError::Failed { message: "first".to_owned(), observations: vec![] };
        let error = program
            .interpret(vec![
                ArrayIrValue::Array(Array::scalar(false).unwrap()),
                ArrayIrValue::Array(data.clone()),
                ArrayIrValue::Array(zero.clone()),
            ])
            .unwrap_err();
        assert_eq!(error.downcast_custom::<AssertionError>(), Some(&first));

        // A residual assertion keeps the known zero from being specialized ahead of it, so the original call and its
        // prefetched operand remain residual and the assertion still fails first.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(ArrayType::scalar(DataType::Boolean).into()),
                PartialValue::Unknown(ArrayType::new_static(DataType::I32, vec![4]).into()),
                PartialValue::Known(ArrayIrValue::Array(zero)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:bool[], %1:i32[4], %2:i32[] .
                let () = assert [message="first", labels=[]] %0
                    () = kernel_call [
                        grid=[],
                        prefetch_0=i32[],
                        identities=[0],
                        identity_0_bounds=[0, ∞),
                        parameter_0=read-only i32[4] [1] InBounds,
                        mapping_0={
                            lambda %0:dimension<prefetch_0 ∈ [0, ∞)> .
                            let %1:dimension<0> = const 0
                                %2:dimension<0> = dimension_div [requires_runtime_assertion=true] %1 %0
                            in (%2)
                        },
                    ] %1 %2 [
                        body={
                            lambda %0:ref<i32[1]>, %1:i32[] .
                            in ()
                        },
                    ]
                in ()"#},
        );
        let condition = |value: bool| ArrayIrValue::Array(Array::scalar(value).unwrap());
        let error = evaluation
            .interpret(&EagerContext::new(), &[condition(false), ArrayIrValue::Array(data.clone())])
            .unwrap_err();
        assert_eq!(error.downcast_custom::<AssertionError>(), Some(&first));
        let violation = ProgramError::custom(DimensionError::RequirementViolation {
            message: "`dimension_div` requires right > 0; got left=0, right=0".to_owned(),
        });
        assert_eq!(
            evaluation.interpret(&EagerContext::new(), &[condition(true), ArrayIrValue::Array(data)]),
            Err(ProgramError::custom(KernelError::Mapping(BlockMappingError::Program(violation)))),
        );
    }

    #[test]
    fn test_kernel_operation_partial_evaluation_inputs() {
        /// Region driver that exposes the body of one kernel call and supports no recursive partial evaluation.
        struct CallBodyDriver<'o> {
            definition: &'o KernelDefinition,
        }

        impl RegionDriver<ArrayIrValue<Array>, KernelOperation> for CallBodyDriver<'_> {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, ArrayIrValue<Array>, KernelOperation>>
            where
                ArrayIrValue<Array>: 'r,
                KernelOperation: 'r,
            {
                std::iter::once(self.definition.body().entry_region_ref())
            }
        }

        impl PartialEvaluationDriver<EagerContext<ArrayIrValue<Array>, KernelOperation>> for CallBodyDriver<'_> {
            fn partially_evaluate_region(
                &self,
                _context: &PartialEvaluationContext<EagerContext<ArrayIrValue<Array>, KernelOperation>>,
                _index: usize,
                _inputs: Vec<PartialEvaluationValue<ArrayIrValue<Array>>>,
            ) -> Result<Vec<PartialEvaluationValue<ArrayIrValue<Array>>>, ProgramError> {
                panic!("a kernel call never partially evaluates its body")
            }

            fn partially_evaluate_program(
                &self,
                _context: &PartialEvaluationContext<EagerContext<ArrayIrValue<Array>, KernelOperation>>,
                _region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation>,
                _knowledge: &[PartialValue<ArrayIrValue<Array>>],
            ) -> Result<PartialEvaluation<EagerContext<ArrayIrValue<Array>, KernelOperation>>, ProgramError>
            {
                panic!("a kernel call never partially evaluates its body")
            }

            fn partition_program(
                &self,
                _context: &PartialEvaluationContext<EagerContext<ArrayIrValue<Array>, KernelOperation>>,
                _region: RegionRef<'_, ArrayIrValue<Array>, KernelOperation>,
                _input_known: &[bool],
            ) -> Result<PartitionedProgram<ArrayIrValue<Array>, KernelOperation>, ProgramError> {
                panic!("a kernel call never partitions its body")
            }
        }

        // A call that copies the element of its `i32[4]` data input at the prefetched start into its `i32[1]` result,
        // whose logical inputs are therefore `[i32[4], i32[]]`.
        let data_type = ArrayType::new_static(DataType::I32, vec![4]);
        let prefetch_type = ArrayType::scalar(DataType::I32);
        let call = mapped_call(
            Grid::new(vec![]).unwrap(),
            vec![
                (
                    data_type.clone(),
                    KernelParameterAccess::ReadOnly,
                    ParameterWindow::ElementStarts {
                        block_shape: vec![1],
                        boundary: BoundaryPolicy::InBounds,
                        starts: vec![MappingExpression::Prefetch(0)],
                    },
                ),
                (
                    ArrayType::new_static(DataType::I32, vec![1]),
                    KernelParameterAccess::WriteOnly,
                    ParameterWindow::WholeArray,
                ),
            ],
            vec![prefetch_type.clone()],
        )
        .unwrap();
        let definition: KernelDefinition = KernelDefinition::trace_with_prefetch(call, |(references, _, _)| {
            let value = references[0].read()?;
            references[1].write(&value)
        })
        .unwrap();
        let driver = CallBodyDriver { definition: &definition };
        let operation = KernelOperation::Call(definition.operation().clone());
        let data = ArrayIrValue::Array(Array::vector(vec![5i32, 6, 7, 8]).unwrap());
        let two = ArrayIrValue::Array(Array::scalar(2i32).unwrap());
        let evaluate = |inputs: &[Option<ArrayIrValue<Array>>], types: &[ArrayType]| {
            let context = PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, KernelOperation>::new());
            let inputs = inputs
                .iter()
                .zip(types)
                .enumerate()
                .map(|(index, (input, r#type))| match input {
                    Some(value) => PartialEvaluationValue::known_input(value.clone()),
                    None => context.unknown_input(r#type.clone().into(), index),
                })
                .collect::<Vec<_>>();
            let outputs = operation.partially_evaluate(&context, &driver, &inputs)?;
            context.into_evaluation(outputs)
        };
        let mismatch =
            ProgramError::Type(TypeError::invalid("`kernel_call` input types do not match its logical signature"));

        // Inputs that do not match the original signature are rejected before the prefetch values are split off,
        // whether the trailing input is known or not.
        let i64_two = ArrayIrValue::Array(Array::scalar(2i64).unwrap());
        let i64_type = ArrayType::scalar(DataType::I64);
        assert_eq!(evaluate(&[], &[]).map(|_| ()), Err(mismatch.clone()));
        let prefetch_types = std::slice::from_ref(&prefetch_type);
        assert_eq!(evaluate(&[Some(two.clone())], prefetch_types).map(|_| ()), Err(mismatch.clone()));
        assert_eq!(evaluate(&[None], prefetch_types).map(|_| ()), Err(mismatch.clone()));
        assert_eq!(
            evaluate(&[Some(data.clone())], std::slice::from_ref(&data_type)).map(|_| ()),
            Err(mismatch.clone()),
        );
        assert_eq!(
            evaluate(&[None, None, Some(two.clone())], &[data_type.clone(), data_type.clone(), prefetch_type.clone()],)
                .map(|_| ()),
            Err(mismatch.clone()),
        );
        assert_eq!(
            evaluate(&[None, Some(i64_two)], &[data_type.clone(), i64_type.clone()]).map(|_| ()),
            Err(mismatch.clone()),
        );
        assert_eq!(evaluate(&[None, None], &[data_type.clone(), i64_type]).map(|_| ()), Err(mismatch.clone()));
        assert_eq!(
            evaluate(&[Some(two.clone()), None], &[prefetch_type.clone(), data_type.clone()]).map(|_| ()),
            Err(mismatch),
        );

        // A known prefetch value specializes the call, whose residual takes only the data input, known or not.
        let expected = vec![ArrayIrValue::Array(Array::vector(vec![7i32]).unwrap())];
        let specialized = indoc! {"
            lambda %0:i32[4] .
            let %1:i32[1] = kernel_call [
                grid=[],
                parameter_0=read-only i32[4] [1] InBounds,
                mapping_0={
                    lambda  .
                    let %0:dimension<2> = const 2
                    in (%0)
                },
                parameter_1=write-only i32[1] [1] InBounds,
                mapping_1={
                    lambda  .
                    let %0:dimension<2> = const 2
                        %1:dimension<0> = const 0
                    in (%1)
                },
            ] %0 [
                body={
                    lambda %0:ref<i32[1]>, %1:ref<i32[1]> .
                    let %2:i32[1] = reference_read %0
                        () = reference_write %1 %2
                    in ()
                },
            ]
            in (%1)"};
        let evaluation = evaluate(&[None, Some(two.clone())], &[data_type.clone(), prefetch_type.clone()]).unwrap();
        assert_eq!(evaluation.program().to_string(), specialized);
        assert_eq!(evaluation.interpret(&EagerContext::new(), std::slice::from_ref(&data)), Ok(expected.clone()));
        let evaluation =
            evaluate(&[Some(data.clone()), Some(two.clone())], &[data_type.clone(), prefetch_type.clone()]).unwrap();
        assert_eq!(evaluation.program().to_string(), specialized);
        assert_eq!(evaluation.interpret(&EagerContext::new(), &[]), Ok(expected.clone()));

        // An unknown prefetch value leaves the original call and its complete input list residual. The known data input
        // becomes a residual program input after the unknown prefetch value.
        let evaluation = evaluate(&[Some(data.clone()), None], &[data_type.clone(), prefetch_type.clone()]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:i32[], %1:i32[4] .
                let %2:i32[1] = kernel_call [
                    grid=[],
                    prefetch_0=i32[],
                    identities=[0, 0],
                    identity_0_bounds=[0, ∞),
                    parameter_0=read-only i32[4] [1] InBounds,
                    mapping_0={
                        lambda %0:dimension<prefetch_0 ∈ [0, ∞)> .
                        in (%0)
                    },
                    parameter_1=write-only i32[1] [1] InBounds,
                    mapping_1={
                        lambda %0:dimension<prefetch_0 ∈ [0, ∞)> .
                        let %1:dimension<0> = const 0
                        in (%1)
                    },
                ] %1 %0 [
                    body={
                        lambda %0:ref<i32[1]>, %1:ref<i32[1]>, %2:i32[] .
                        let %3:i32[1] = reference_read %0
                            () = reference_write %1 %3
                        in ()
                    },
                ]
                in (%2)"},
        );
        assert_eq!(evaluation.interpret(&EagerContext::new(), std::slice::from_ref(&two)), Ok(expected.clone()));
        let evaluation = evaluate(&[None, None], &[data_type, prefetch_type]).unwrap();
        assert_eq!(evaluation.interpret(&EagerContext::new(), &[data, two]), Ok(expected));
    }

    #[test]
    fn test_kernel_operation_partial_evaluation_prefetch_placement() {
        // A call that copies its scalar prefetch into its write-only result.
        let mut mapping = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        mapping.add_input(DimensionType::new("offset", DimensionBounds::non_negative(Some(4)).unwrap()).into());
        let mapping = BlockMapping::new(
            mapping.build(vec![], vec![Placeholder], vec![]).unwrap(),
            vec![],
            BoundaryPolicy::InBounds,
        )
        .unwrap();
        let call = KernelCallOperation::new_with_prefetch(
            Grid::new(vec![]).unwrap(),
            vec![
                KernelParameter::new(ArrayType::scalar(DataType::I32), KernelParameterAccess::WriteOnly, mapping)
                    .unwrap(),
            ],
            vec![ArrayType::scalar(DataType::I32)],
        )
        .unwrap();
        let definition: KernelDefinition =
            KernelDefinition::trace_with_prefetch(call, |(references, _, values)| references[0].write(&values[0]))
                .unwrap();
        let program = |input_type: ArrayType| {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, KernelOperation>::new();
            let input = builder.add_input(ArrayIrType::Array(input_type));
            let region = builder.import_region(definition.body().entry_region_ref());
            let outputs = builder
                .add_instruction(KernelOperation::Call(definition.operation().clone()), vec![region], vec![input], None)
                .unwrap()
                .to_vec();
            let structure = vec![Placeholder];
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(outputs, structure.clone(), structure)
                .unwrap()
        };
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let bytes = Array::scalar(3i32).unwrap().logical_bytes();
        let placed = |sharding: Sharding| {
            let r#type = ArrayType::scalar(DataType::I32).with_sharding(Some(sharding)).unwrap();
            (r#type.clone(), ArrayIrValue::Array(Array::from_logical_bytes(r#type, &bytes).unwrap()))
        };

        // A merely replicated prefetch value specializes like its physical value, and the result keeps the type that
        // the original call infers from it.
        let (replicated_type, replicated) = placed(Sharding::replicated(mesh.clone(), 0));
        let replicated_program = program(replicated_type);
        let expected = replicated_program.interpret(vec![replicated.clone()]).unwrap();
        assert_eq!(expected, vec![ArrayIrValue::Array(Array::scalar(3i32).unwrap())]);
        let evaluation = replicated_program.partially_evaluate(&[PartialValue::Known(replicated)]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda  .
                let %0:i32[] = kernel_call [
                    grid=[],
                    parameter_0=write-only i32[] [] InBounds,
                    mapping_0={
                        lambda  .
                        let %0:dimension<3> = const 3
                        in ()
                    },
                ] [
                    body={
                        lambda %0:ref<i32[]> .
                        let %1:i32[] = const 3
                            () = reference_write %0 %1
                        in ()
                    },
                ]
                in (%0)"},
        );
        assert_eq!(evaluation.interpret(&EagerContext::new(), &[]), Ok(expected));

        // A prefetch value that varies along `x` makes the result vary along `x`, which the specialized call could not
        // express, so the original call remains residual with its prefetched operand.
        let (varying_type, varying) = placed(Sharding::replicated(mesh, 0).with_varying_manual_axes(["x"]).unwrap());
        let varying_program = program(varying_type);
        let expected = varying_program.interpret(vec![varying.clone()]).unwrap();
        let evaluation = varying_program.partially_evaluate(&[PartialValue::Known(varying)]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:i32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
                let %1:i32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = kernel_call [
                    grid=[],
                    prefetch_0=i32[],
                    identities=[0],
                    identity_0_bounds=[0, 4),
                    parameter_0=write-only i32[] [] InBounds,
                    mapping_0={
                        lambda %0:dimension<offset ∈ [0, 4)> .
                        in ()
                    },
                ] %0 [
                    body={
                        lambda %0:ref<i32[]>, %1:i32[] .
                        let () = reference_write %0 %1
                        in ()
                    },
                ]
                in (%1)"},
        );
        assert_eq!(evaluation.interpret(&EagerContext::new(), &[]), Ok(expected));
    }

    #[test]
    fn test_kernel_operation_reference_access_descriptor() {
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) };
        let canonical =
            ArrayIrOperation::<Array>::from(ReferenceReadOperation::new().with_transforms(vec![transform.clone()]));
        let operation: KernelOperation = KernelOperation::from(canonical.clone());
        assert_eq!(operation.reference_access_descriptor(0).unwrap().transforms(), &[transform]);
        assert_eq!(operation.reference_access_descriptor(0).unwrap().bindings(), 1..1);
        assert!(operation.reference_access_descriptor(1).is_none());
        assert_eq!(operation.effects(), canonical.effects());
        let source = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        let target = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        assert_eq!(operation.infer_output_types(&[source], &[]), Ok(vec![target]));
    }

    #[test]
    fn test_kernel_operation_project_payload() {
        // Payloads that this crate owns are exposed directly, portable payloads through the array IR family, and
        // extension payloads not at all.
        let wait = KernelOperation::<NoKernelExtension>::from(WaitOperation);
        let add = KernelOperation::<NoKernelExtension>::from(ArrayOperation::Add(AddOperation::new()));
        let extension =
            KernelOperation::Extension(ArrayIrOperation::<Array>::from(ArrayOperation::Add(AddOperation::new())));
        assert_eq!(wait.projected_payload::<WaitOperation>(), Some(&WaitOperation));
        assert!(wait.projected_payload::<AddOperation<ArrayType>>().is_none());
        assert!(add.projected_payload::<AddOperation<ArrayType>>().is_some());
        assert!(add.projected_payload::<WaitOperation>().is_none());
        assert!(extension.projected_payload::<AddOperation<ArrayType>>().is_none());
    }

    #[test]
    fn test_kernel_operation_from_payload() {
        // Payloads that this crate owns construct their own variants and portable payloads construct through the array
        // IR family, while payloads that no variant holds are returned unchanged.
        let wait = KernelOperation::<NoKernelExtension>::from_payload(ErasedOperation::new(WaitOperation));
        assert!(matches!(wait, Ok(KernelOperation::Wait(WaitOperation))));
        let add = ErasedOperation::new(AddOperation::<ArrayType>::new());
        let add = KernelOperation::<NoKernelExtension>::from_payload(add);
        assert!(matches!(add, Ok(KernelOperation::Portable(ArrayIrOperation::Array(ArrayOperation::Add(_))))));
        let payload = ErasedOperation::new(TagOperation::<ArrayIrType>::new("unsupported"));
        let payload = KernelOperation::<NoKernelExtension>::from_payload(payload).unwrap_err();
        assert_eq!(payload.type_name(), std::any::type_name::<TagOperation<ArrayIrType>>());
    }

    #[test]
    fn test_kernel_operation_reference_staging() {
        let r#type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::I32)));
        let (_, program) = TracingContext::<ArrayIrValue<Array>, KernelOperation>::trace(
            |inputs: Vec<Tracer<_>>| {
                let value = inputs[0].read()?;
                Ok(vec![inputs[0].swap(&value)?])
            },
            vec![r#type],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<i32[]> .
                let %1:i32[] = reference_read %0
                    %2:i32[] = reference_swap %0 %1
                in (%2)"},
        );
        let swap = program.instructions()[1].operation();
        let projected =
            <&ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>>::try_from(swap).unwrap();
        assert_eq!(projected.name(), "reference_swap");
        assert!(matches!(swap, KernelOperation::Portable(ArrayIrOperation::ReferenceSwap(_))));
        assert_eq!(swap.effects(), projected.effects());
    }

    #[test]
    fn test_kernel_operation_extension_rejection() {
        /// An adapter operation whose required input is deliberately absent.
        #[derive(Clone, Debug)]
        struct UnavailableExtension;

        impl Operation for UnavailableExtension {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                "unavailable_extension"
            }

            fn infer_output_types(
                &self,
                _input_types: &[ArrayIrType],
                _region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                Err(TypeError::invalid("extension requires a reference input"))
            }
        }

        impl<C: Domain<Type = ArrayIrType>> InterpretableOperation<C> for UnavailableExtension {
            fn interpret<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                _inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                Err(ProgramError::MalformedProgram("extension has no registered interpreter".to_owned()))
            }
        }

        impl ReferenceAccessOperation for UnavailableExtension {
            type Transform = ArrayReferenceTransform;

            fn base_input_count(&self) -> usize {
                0
            }

            fn reference_access_descriptor(
                &self,
                _input_index: usize,
            ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
                None
            }

            fn with_reference_access_transforms(
                &self,
                _input_index: usize,
                _transforms: Vec<ArrayReferenceTransform>,
            ) -> Result<Self, ProgramError> {
                Err(ProgramError::UnsupportedOperation { message: "extension has no reference accesses".to_owned() })
            }
        }

        impl KernelExtension for UnavailableExtension {}

        assert_eq!(
            UnavailableExtension.semantic_key(&mut |_| 0),
            Err(TypeError::invalid("kernel extension `unavailable_extension` has no exact semantic identity contract")),
        );
        assert_eq!(
            UnavailableExtension.memory_semantics(),
            Err(TypeError::invalid("kernel extension `unavailable_extension` has no initialization contract")),
        );

        let operation = KernelOperation::Extension(UnavailableExtension);
        assert_eq!(operation.to_string(), "unavailable_extension");
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Err(TypeError::invalid("extension requires a reference input")),
        );
        assert_eq!(
            operation.interpret(&EagerContext::<ArrayIrValue<Array>, KernelOperation>::new(), &EmptyRegionDriver, &[]),
            Err(ProgramError::MalformedProgram("extension has no registered interpreter".to_owned())),
        );
        assert_eq!(
            <&ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>>::try_from(&operation)
                .map(|swap| swap.name()),
            Err(TypeError::invalid(
                "cannot project extension operation `unavailable_extension` into a canonical reference swap",
            )),
        );
    }
}
