//! Kernel operation families over the canonical array, dimension, and reference universe.
//!
//! [`KernelOperation`] embeds [`ArrayIrOperation`] without copying its primitive variants or redefining their
//! semantics. Adapters supply a typed extension family; [`NoKernelExtension`] makes the portable family uninhabited
//! on that branch. Operation-family membership alone is not target admission: kernel validation and compiler
//! capability checks still determine which operations a particular kernel may contain.

use std::borrow::Cow;
use std::fmt::{Display, Formatter};

use crate::arrays::{
    Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReferenceTransform, ArrayType,
    DimensionOperation, DimensionType, DimensionValue,
};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::kernels::calls::{KernelCallOperation, KernelDefinition};
use crate::kernels::indexing::TileLoadOperation;
use crate::kernels::memory::{
    AsyncCopyOperation, MaskedLoadOperation, MaskedStoreOperation, MaskedSwapOperation, ScratchOperation, WaitOperation,
};
use crate::kernels::validation::KernelReferenceOperation;
use crate::operations::{
    ConstantOperation, DimensionSizeOperation, DynamicBroadcastOperation, ReferenceAddUpdateOperation,
    ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
    ReferenceSwapOperation, ReferenceWriteOperation,
};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    Effects, InputRegionProvenance, Operation, OperationProjection, OutputRegionProvenance, ProgramError,
    ReferenceAccessDescriptor, ReferenceAccessMode, ReferenceAccessOperation, RegionInterface, RegionSlot, Type,
    TypeError, TypeIdentityRenaming,
};

/// Initialization and lifetime behavior not implied by ordinary reference effects.
///
/// Access modes, aliases, and allocated referents come exclusively from canonical operation effects and types.
/// This descriptor specifies when those accesses complete; it does not declare an independent memory model.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelExtensionMemory {
    /// All declared accesses complete synchronously; reference views retain their canonical alias semantics.
    Synchronous,

    /// Creates one uninitialized local reference allocation with no accesses.
    Allocation { output_index: usize },

    /// Reserves all declared accesses until the newly allocated completion reference is waited.
    Asynchronous { completion_output_index: usize },

    /// Observes a pending completion reference without completing its accesses or publishing writes.
    /// Hardware group counts and commit ordering remain the adapter's responsibility.
    Commit { completion_input_index: usize },

    /// Consumes a pending completion reference and publishes its deferred writes.
    Wait { completion_input_index: usize },

    /// Consumes a local allocation after all outstanding accesses complete.
    Release { input_index: usize },
}

/// Adapter-owned kernel semantics over the canonical array and reference universe.
///
/// Implementations are trusted semantic contracts, not target admission. A compiler must still validate the exact
/// architecture, instruction capabilities, and resource requirements before compiling an extension.
pub trait KernelExtension: ReferenceAccessOperation<Type = ArrayIrType, Transform = ArrayReferenceTransform> {
    /// Returns the checked initialization timing and lifetime contract for this operation.
    ///
    /// The verifier checks consistency with canonical effects, output referents, and completion-token types. Nested
    /// extension regions are not admitted by this contract. Unimplemented classifications fail conservatively.
    fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
        Err(TypeError::invalid(format!("kernel extension `{}` has no initialization contract", self.name())))
    }

    /// Returns an exact, versioned encoding of this operation's semantic payload.
    ///
    /// The encoding must include an unambiguous owner namespace and schema version, every instruction mode and
    /// semantic option, and exact floating-point bits. It must not depend on diagnostic formatting or addresses.
    /// Operands, results, regions, and their types are encoded by the enclosing kernel. Type identities in the
    /// payload must participate in ordinary operation identity renaming before this function is called.
    /// Unsupported payloads fail explicitly; implementing this function does not enable source deserialization.
    fn semantic_key(&self) -> Result<Vec<u8>, TypeError> {
        Err(TypeError::invalid(format!("kernel extension `{}` has no exact semantic identity contract", self.name(),)))
    }
}

/// Empty extension family for kernels containing only portable operations.
///
/// This enum has no values, so a portable kernel cannot contain an unknown or unchecked target payload.
#[derive(Copy, Clone, Debug)]
pub enum NoKernelExtension {}

impl KernelExtension for NoKernelExtension {
    fn memory_semantics(&self) -> Result<KernelExtensionMemory, TypeError> {
        match *self {}
    }

    fn semantic_key(&self) -> Result<Vec<u8>, TypeError> {
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

impl<C: Domain> InterpretableOperation<C> for NoKernelExtension {
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
        _views: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        match *self {}
    }
}

/// Kernel body operation with a canonical portable family and an adapter-owned extension family.
///
/// All variants use [`ArrayIrType`] and retain their payload's effects, reference identities, region contracts, and
/// diagnostics. The extension has no implicit conversion into portable operations and no fallback interpretation.
/// Construct an extension explicitly with [`KernelOperation::Extension`]. Backend selection is not stored here.
///
/// Partial evaluation uses the existing conservative fold-or-residualize rule for the complete operation. This
/// preserves effects and nested regions without pretending that a nested portable program has a different operation
/// universe. More specialized partitioning rules can be added when the owning kernel transform contract requires them.
#[derive(Clone, Debug)]
pub enum KernelOperation<Extension: Operation<Type = ArrayIrType> = NoKernelExtension> {
    /// An operation from the existing array, dimension, and reference family.
    Portable(ArrayIrOperation<Array>),

    /// A functional call whose attached region owns its kernel reference windows.
    Call(KernelCallOperation),

    /// Logically uninitialized storage owned by a qualified kernel body.
    Scratch(ScratchOperation),

    /// Reads a dynamically selected reference tile with explicit padding.
    TileLoad(TileLoadOperation),

    /// Starts a qualified asynchronous reference copy.
    AsyncCopy(AsyncCopyOperation),

    /// Completes pending qualified asynchronous copies.
    Wait(WaitOperation),

    /// Reads only selected elements from an already valid reference view.
    MaskedLoad(MaskedLoadOperation),

    /// Writes only selected elements of an already valid reference view.
    MaskedStore(MaskedStoreOperation),

    /// Replaces selected elements and returns their old values with explicit fallback elsewhere.
    MaskedSwap(MaskedSwapOperation),

    /// A typed exact operation whose semantics and supported targets belong to its adapter.
    Extension(Extension),
}

impl<Extension: Operation<Type = ArrayIrType>> KernelOperation<Extension> {
    /// Converts only the adapter-owned payload, preserving every canonical portable operation unchanged.
    /// The conversion can reject an extension family unavailable to the destination execution integration.
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
            Self::Extension(operation) => KernelOperation::Extension(map(operation)?),
        })
    }
}

impl<Extension: Operation<Type = ArrayIrType>> Display for KernelOperation<Extension> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<Extension: Operation<Type = ArrayIrType>> Operation for KernelOperation<Extension> {
    type Type = ArrayIrType;

    fn name(&self) -> &'static str {
        match self {
            Self::Portable(operation) => operation.name(),
            Self::Call(operation) => operation.name(),
            Self::Scratch(operation) => operation.name(),
            Self::TileLoad(operation) => operation.name(),
            Self::AsyncCopy(operation) => operation.name(),
            Self::Wait(operation) => operation.name(),
            Self::MaskedLoad(operation) => operation.name(),
            Self::MaskedStore(operation) => operation.name(),
            Self::MaskedSwap(operation) => operation.name(),
            Self::Extension(operation) => operation.name(),
        }
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        match self {
            Self::Portable(operation) => operation.region_slots(),
            Self::Call(operation) => operation.region_slots(),
            Self::Scratch(operation) => operation.region_slots(),
            Self::TileLoad(operation) => operation.region_slots(),
            Self::AsyncCopy(operation) => operation.region_slots(),
            Self::Wait(operation) => operation.region_slots(),
            Self::MaskedLoad(operation) => operation.region_slots(),
            Self::MaskedStore(operation) => operation.region_slots(),
            Self::MaskedSwap(operation) => operation.region_slots(),
            Self::Extension(operation) => operation.region_slots(),
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        match self {
            Self::Portable(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::Call(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::Scratch(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::TileLoad(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::AsyncCopy(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::Wait(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::MaskedLoad(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::MaskedStore(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::MaskedSwap(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            Self::Extension(operation) => operation.infer_region_input_types(input_types, region_interfaces),
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        match self {
            Self::Portable(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Call(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Scratch(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::TileLoad(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::AsyncCopy(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Wait(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::MaskedLoad(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::MaskedStore(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::MaskedSwap(operation) => operation.infer_output_types(input_types, region_interfaces),
            Self::Extension(operation) => operation.infer_output_types(input_types, region_interfaces),
        }
    }

    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        match self {
            Self::Portable(operation) => operation.input_region_provenance(region_index, input_index),
            Self::Call(operation) => operation.input_region_provenance(region_index, input_index),
            Self::Scratch(operation) => operation.input_region_provenance(region_index, input_index),
            Self::TileLoad(operation) => operation.input_region_provenance(region_index, input_index),
            Self::AsyncCopy(operation) => operation.input_region_provenance(region_index, input_index),
            Self::Wait(operation) => operation.input_region_provenance(region_index, input_index),
            Self::MaskedLoad(operation) => operation.input_region_provenance(region_index, input_index),
            Self::MaskedStore(operation) => operation.input_region_provenance(region_index, input_index),
            Self::MaskedSwap(operation) => operation.input_region_provenance(region_index, input_index),
            Self::Extension(operation) => operation.input_region_provenance(region_index, input_index),
        }
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        match self {
            Self::Portable(operation) => operation.output_region_provenance(output_index),
            Self::Call(operation) => operation.output_region_provenance(output_index),
            Self::Scratch(operation) => operation.output_region_provenance(output_index),
            Self::TileLoad(operation) => operation.output_region_provenance(output_index),
            Self::AsyncCopy(operation) => operation.output_region_provenance(output_index),
            Self::Wait(operation) => operation.output_region_provenance(output_index),
            Self::MaskedLoad(operation) => operation.output_region_provenance(output_index),
            Self::MaskedStore(operation) => operation.output_region_provenance(output_index),
            Self::MaskedSwap(operation) => operation.output_region_provenance(output_index),
            Self::Extension(operation) => operation.output_region_provenance(output_index),
        }
    }

    fn is_zero(&self, output_index: usize) -> bool {
        match self {
            Self::Portable(operation) => operation.is_zero(output_index),
            Self::Call(operation) => operation.is_zero(output_index),
            Self::Scratch(operation) => operation.is_zero(output_index),
            Self::TileLoad(operation) => operation.is_zero(output_index),
            Self::AsyncCopy(operation) => operation.is_zero(output_index),
            Self::Wait(operation) => operation.is_zero(output_index),
            Self::MaskedLoad(operation) => operation.is_zero(output_index),
            Self::MaskedStore(operation) => operation.is_zero(output_index),
            Self::MaskedSwap(operation) => operation.is_zero(output_index),
            Self::Extension(operation) => operation.is_zero(output_index),
        }
    }

    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        match self {
            Self::Portable(operation) => operation.region_capture_input_count(region_index),
            Self::Call(operation) => operation.region_capture_input_count(region_index),
            Self::Scratch(operation) => operation.region_capture_input_count(region_index),
            Self::TileLoad(operation) => operation.region_capture_input_count(region_index),
            Self::AsyncCopy(operation) => operation.region_capture_input_count(region_index),
            Self::Wait(operation) => operation.region_capture_input_count(region_index),
            Self::MaskedLoad(operation) => operation.region_capture_input_count(region_index),
            Self::MaskedStore(operation) => operation.region_capture_input_count(region_index),
            Self::MaskedSwap(operation) => operation.region_capture_input_count(region_index),
            Self::Extension(operation) => operation.region_capture_input_count(region_index),
        }
    }

    fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
        match self {
            Self::Portable(operation) => operation.reference_output_identity_input(output_index),
            Self::Call(operation) => operation.reference_output_identity_input(output_index),
            Self::Scratch(operation) => operation.reference_output_identity_input(output_index),
            Self::TileLoad(operation) => operation.reference_output_identity_input(output_index),
            Self::AsyncCopy(operation) => operation.reference_output_identity_input(output_index),
            Self::Wait(operation) => operation.reference_output_identity_input(output_index),
            Self::MaskedLoad(operation) => operation.reference_output_identity_input(output_index),
            Self::MaskedStore(operation) => operation.reference_output_identity_input(output_index),
            Self::MaskedSwap(operation) => operation.reference_output_identity_input(output_index),
            Self::Extension(operation) => operation.reference_output_identity_input(output_index),
        }
    }

    fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
        match self {
            Self::Portable(operation) => operation.allows_reference_access_through_region_input(region_index, mode),
            Self::Call(operation) => operation.allows_reference_access_through_region_input(region_index, mode),
            Self::Scratch(operation) => operation.allows_reference_access_through_region_input(region_index, mode),
            Self::TileLoad(operation) => operation.allows_reference_access_through_region_input(region_index, mode),
            Self::AsyncCopy(operation) => operation.allows_reference_access_through_region_input(region_index, mode),
            Self::Wait(operation) => operation.allows_reference_access_through_region_input(region_index, mode),
            Self::MaskedLoad(operation) => operation.allows_reference_access_through_region_input(region_index, mode),
            Self::MaskedStore(operation) => operation.allows_reference_access_through_region_input(region_index, mode),
            Self::MaskedSwap(operation) => operation.allows_reference_access_through_region_input(region_index, mode),
            Self::Extension(operation) => operation.allows_reference_access_through_region_input(region_index, mode),
        }
    }

    fn effects(&self) -> Cow<'_, Effects> {
        match self {
            Self::Portable(operation) => operation.effects(),
            Self::Call(operation) => operation.effects(),
            Self::Scratch(operation) => operation.effects(),
            Self::TileLoad(operation) => operation.effects(),
            Self::AsyncCopy(operation) => operation.effects(),
            Self::Wait(operation) => operation.effects(),
            Self::MaskedLoad(operation) => operation.effects(),
            Self::MaskedStore(operation) => operation.effects(),
            Self::MaskedSwap(operation) => operation.effects(),
            Self::Extension(operation) => operation.effects(),
        }
    }

    fn render(&self, formatter: &mut Formatter<'_>, indentation: usize) -> std::fmt::Result {
        match self {
            Self::Portable(operation) => operation.render(formatter, indentation),
            Self::Call(operation) => operation.render(formatter, indentation),
            Self::Scratch(operation) => operation.render(formatter, indentation),
            Self::TileLoad(operation) => operation.render(formatter, indentation),
            Self::AsyncCopy(operation) => operation.render(formatter, indentation),
            Self::Wait(operation) => operation.render(formatter, indentation),
            Self::MaskedLoad(operation) => operation.render(formatter, indentation),
            Self::MaskedStore(operation) => operation.render(formatter, indentation),
            Self::MaskedSwap(operation) => operation.render(formatter, indentation),
            Self::Extension(operation) => operation.render(formatter, indentation),
        }
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Portable(operation) => Ok(Self::Portable(operation.rename_type_identities(renaming)?)),
            Self::Call(operation) => Ok(Self::Call(operation.rename_type_identities(renaming)?)),
            Self::Scratch(operation) => Ok(Self::Scratch(operation.rename_type_identities(renaming)?)),
            Self::TileLoad(operation) => Ok(Self::TileLoad(operation.rename_type_identities(renaming)?)),
            Self::AsyncCopy(operation) => Ok(Self::AsyncCopy(operation.rename_type_identities(renaming)?)),
            Self::Wait(operation) => Ok(Self::Wait(operation.rename_type_identities(renaming)?)),
            Self::MaskedLoad(operation) => Ok(Self::MaskedLoad(operation.rename_type_identities(renaming)?)),
            Self::MaskedStore(operation) => Ok(Self::MaskedStore(operation.rename_type_identities(renaming)?)),
            Self::MaskedSwap(operation) => Ok(Self::MaskedSwap(operation.rename_type_identities(renaming)?)),
            Self::Extension(operation) => Ok(Self::Extension(operation.rename_type_identities(renaming)?)),
        }
    }
}

impl<C: Domain, Extension> InterpretableOperation<C> for KernelOperation<Extension>
where
    Extension: Operation<Type = ArrayIrType> + InterpretableOperation<C>,
    ArrayIrOperation<Array>: InterpretableOperation<C>,
    KernelCallOperation: InterpretableOperation<C>,
    TileLoadOperation: InterpretableOperation<C>,
    MaskedLoadOperation: InterpretableOperation<C>,
    MaskedStoreOperation: InterpretableOperation<C>,
    MaskedSwapOperation: InterpretableOperation<C>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        match self {
            Self::Portable(operation) => operation.interpret(context, driver, inputs),
            Self::Call(operation) => operation.interpret(context, driver, inputs),
            Self::Scratch(operation) => operation.interpret(context, driver, inputs),
            Self::TileLoad(operation) => operation.interpret(context, driver, inputs),
            Self::AsyncCopy(operation) => operation.interpret(context, driver, inputs),
            Self::Wait(operation) => operation.interpret(context, driver, inputs),
            Self::MaskedLoad(operation) => operation.interpret(context, driver, inputs),
            Self::MaskedStore(operation) => operation.interpret(context, driver, inputs),
            Self::MaskedSwap(operation) => operation.interpret(context, driver, inputs),
            Self::Extension(operation) => operation.interpret(context, driver, inputs),
        }
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
        let regions = driver.regions().map(|region| region.to_program()).collect();
        if let Self::Call(call) = self {
            // Only concrete scalar-prefetch values specialize the call. Ordinary data inputs remain residual,
            // even when known: static transforms must never execute a kernel's mutations.
            if !call.prefetch_types().is_empty() {
                let remaining = inputs.len() - call.prefetch_types().len();
                let values = inputs[remaining..]
                    .iter()
                    .map(|input| match context.parent().resolve(input.as_known()?).into_constant()? {
                        ArrayIrValue::Array(array) => Some(array),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>();
                if let Some(values) = values {
                    let definition = KernelDefinition::new(call.clone(), driver.region(0)?.to_program())
                        .and_then(|definition| definition.specialize_prefetch(&values))
                        .map_err(ProgramError::custom)?;
                    return context.residualize(
                        Self::Call(definition.operation().clone()),
                        vec![definition.body().clone()],
                        &inputs[..remaining],
                    );
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
            Self::Call(operation) => operation.input_types().len(),
            Self::Scratch(_) => 0,
            Self::TileLoad(operation) => operation.base_input_count(),
            Self::AsyncCopy(operation) => operation.base_input_count(),
            Self::Wait(_) => 1,
            Self::MaskedLoad(operation) => operation.base_input_count(),
            Self::MaskedStore(operation) => operation.base_input_count(),
            Self::MaskedSwap(operation) => operation.base_input_count(),
            Self::Extension(operation) => operation.base_input_count(),
        }
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
        match self {
            Self::Portable(operation) => operation.reference_access_descriptor(input_index),
            Self::Call(_) | Self::Scratch(_) => None,
            Self::TileLoad(operation) => operation.reference_access_descriptor(input_index),
            Self::AsyncCopy(operation) => operation.reference_access_descriptor(input_index),
            Self::Wait(_) => (input_index == 0).then(|| ReferenceAccessDescriptor::new(&[], 1..1)),
            Self::MaskedLoad(operation) => operation.reference_access_descriptor(input_index),
            Self::MaskedStore(operation) => operation.reference_access_descriptor(input_index),
            Self::MaskedSwap(operation) => operation.reference_access_descriptor(input_index),
            Self::Extension(operation) => operation.reference_access_descriptor(input_index),
        }
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<ArrayReferenceTransform>,
    ) -> Result<Self, ProgramError> {
        Ok(match self {
            Self::Portable(operation) => Self::Portable(operation.with_reference_access_transforms(input_index, transforms)?),
            Self::TileLoad(operation) => Self::TileLoad(operation.with_reference_access_transforms(input_index, transforms)?),
            Self::AsyncCopy(operation) => Self::AsyncCopy(operation.with_reference_access_transforms(input_index, transforms)?),
            Self::MaskedLoad(operation) => Self::MaskedLoad(operation.with_reference_access_transforms(input_index, transforms)?),
            Self::MaskedStore(operation) => {
                Self::MaskedStore(operation.with_reference_access_transforms(input_index, transforms)?)
            }
            Self::MaskedSwap(operation) => Self::MaskedSwap(operation.with_reference_access_transforms(input_index, transforms)?),
            Self::Extension(operation) => Self::Extension(operation.with_reference_access_transforms(input_index, transforms)?),
            Self::Wait(_) if input_index == 0 && transforms.is_empty() => self.clone(),
            _ => {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{}` does not support the requested views at input {input_index}",
                    self.name(),
                )));
            }
        })
    }
}

/// Implements lifts of canonical payloads into the kernel operation family.
macro_rules! kernel_operation_from {
    // Each listed payload uses its existing array-IR conversion, preserving normalization.
    ($($operation:ty),* $(,)?) => {
        $(impl<Extension: Operation<Type = ArrayIrType>> From<$operation> for KernelOperation<Extension> {
            fn from(operation: $operation) -> Self {
                Self::Portable(operation.into())
            }
        })*
    };
}

kernel_operation_from!(
    ArrayIrOperation<Array>,
    ArrayOperation<Array>,
    DimensionOperation<DimensionValue>,
    ConstantOperation<DimensionValue>,
    DimensionSizeOperation,
    DynamicBroadcastOperation,
    ReferenceNewOperation<ArrayType, ArrayIrType>,
    ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>,
    ReferenceWriteOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>,
    ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>,
    ReferenceAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>,
    ReferenceAtomicAddUpdateOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>,
    ReferenceFreezeOperation<ArrayType, ArrayIrType>,
);

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

impl<Extension: Operation<Type = ArrayIrType>> From<MaskedLoadOperation> for KernelOperation<Extension> {
    fn from(operation: MaskedLoadOperation) -> Self {
        Self::MaskedLoad(operation)
    }
}

impl<Extension: Operation<Type = ArrayIrType>> From<MaskedStoreOperation> for KernelOperation<Extension> {
    fn from(operation: MaskedStoreOperation) -> Self {
        Self::MaskedStore(operation)
    }
}

impl<Extension: Operation<Type = ArrayIrType>> From<MaskedSwapOperation> for KernelOperation<Extension> {
    fn from(operation: MaskedSwapOperation) -> Self {
        Self::MaskedSwap(operation)
    }
}

impl<Extension: Operation<Type = ArrayIrType>> From<AsyncCopyOperation> for KernelOperation<Extension> {
    fn from(operation: AsyncCopyOperation) -> Self {
        Self::AsyncCopy(operation)
    }
}

impl<Extension: Operation<Type = ArrayIrType>> From<WaitOperation> for KernelOperation<Extension> {
    fn from(operation: WaitOperation) -> Self {
        Self::Wait(operation)
    }
}

impl<Extension: Operation<Type = ArrayIrType>> From<TileLoadOperation> for KernelOperation<Extension> {
    fn from(operation: TileLoadOperation) -> Self {
        Self::TileLoad(operation)
    }
}

impl<Extension: Operation<Type = ArrayIrType>> From<ScratchOperation> for KernelOperation<Extension> {
    fn from(operation: ScratchOperation) -> Self {
        Self::Scratch(operation)
    }
}

impl<Extension: Operation<Type = ArrayIrType>> From<KernelCallOperation> for KernelOperation<Extension> {
    fn from(operation: KernelCallOperation) -> Self {
        Self::Call(operation)
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
            | KernelOperation::MaskedSwap(_) => {
                Err(TypeError::invalid("cannot project a kernel-specific operation into a reference swap"))
            }
            KernelOperation::Extension(operation) => Err(TypeError::invalid(format!(
                "cannot project extension operation `{}` into a canonical reference swap",
                operation.name(),
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayIrValue, ArrayReferenceTransformIndex, DataType};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::{AddOperation, ReferenceRead, ReferenceSwap};
    use crate::programs::{EmptyRegionDriver, ReferenceType};
    use crate::tracing::TracingContext;

    use super::*;

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
    fn test_kernel_operation_type_inference() {
        let operation = KernelOperation::<NoKernelExtension>::from(ArrayOperation::Add(AddOperation::new()));
        let r#type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [2]));
        assert_eq!(operation.infer_output_types(&[r#type.clone(), r#type.clone()], &[]), Ok(vec![r#type]));
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
    fn test_kernel_operation_reference_access_descriptor() {
        let view = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) };
        let canonical = ArrayIrOperation::<Array>::from(ReferenceReadOperation::new().with_transforms(vec![view.clone()]));
        let operation: KernelOperation = KernelOperation::from(canonical.clone());
        assert_eq!(operation.reference_access_descriptor(0).unwrap().transforms(), &[view]);
        assert_eq!(operation.reference_access_descriptor(0).unwrap().bindings(), 1..1);
        assert!(operation.reference_access_descriptor(1).is_none());
        assert_eq!(operation.effects(), canonical.effects());
        let source = ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [2])));
        let target = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        assert_eq!(operation.infer_output_types(&[source], &[]), Ok(vec![target]));
    }

    #[test]
    fn test_kernel_operation_reference_staging() {
        let context = TracingContext::<ArrayIrValue<Array>, KernelOperation>::new();
        let r#type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::I32)));
        let reference = context.input(r#type);
        let value = reference.read().unwrap();
        reference.swap(&value).unwrap();
        let builder = context.builder().borrow();
        let swap = builder.instructions()[1].operation();
        let projected = <&ReferenceSwapOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>>::try_from(swap).unwrap();
        assert_eq!(projected.name(), "reference_swap");
        assert!(matches!(swap, KernelOperation::Portable(ArrayIrOperation::ReferenceSwap(_))));
        assert_eq!(swap.effects(), projected.effects());
    }

    #[test]
    fn test_kernel_operation_extension_rejection() {
        /// An adapter operation whose required operand is deliberately absent.
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
                Err(TypeError::invalid("extension requires a reference operand"))
            }
        }

        impl<C: Domain> InterpretableOperation<C> for UnavailableExtension {
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
                _views: Vec<ArrayReferenceTransform>,
            ) -> Result<Self, ProgramError> {
                Err(ProgramError::UnsupportedOperation { message: "extension has no reference accesses".to_owned() })
            }
        }

        impl KernelExtension for UnavailableExtension {}

        assert_eq!(
            UnavailableExtension.semantic_key(),
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
            Err(TypeError::invalid("extension requires a reference operand")),
        );
        assert_eq!(
            operation.interpret(&EagerContext::<ArrayIrValue<Array>, KernelOperation>::new(), &EmptyRegionDriver, &[],),
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
    #[test]
    fn test_kernel_operation_partially_evaluate() {
        use crate::arrays::DimensionBounds;
        use crate::kernels::calls::KernelParameter;
        use crate::kernels::grids::Grid;
        use crate::kernels::mappings::{BlockMapping, BoundaryPolicy};
        use crate::kernels::validation::KernelParameterAccess;
        use crate::operations::ReferenceWrite;
        use crate::parameters::Placeholder;
        use crate::partial::{PartialEvaluationOutput, PartialValue};
        use crate::programs::ProgramBuilder;

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
        assert_eq!(evaluation.program().instructions().len(), 1);
        assert!(evaluation.program().input_types().is_empty());
        let KernelOperation::Call(specialized) = evaluation.program().instructions()[0].operation() else {
            panic!("expected a residual kernel call");
        };
        assert!(specialized.prefetch_types().is_empty());
        assert!(specialized.parameters()[0].mapping().program().input_types().is_empty());
        assert_eq!(evaluation.interpret(&EagerContext::new(), &[]).unwrap(), vec![value]);
        let unknown = program
            .partially_evaluate(&[PartialValue::Unknown(ArrayType::scalar(DataType::I32).into())])
            .unwrap();
        let KernelOperation::Call(residual) = unknown.program().instructions()[0].operation() else {
            panic!("expected a residual kernel call");
        };
        assert_eq!(residual.prefetch_types(), definition.operation().prefetch_types());
    }
}
