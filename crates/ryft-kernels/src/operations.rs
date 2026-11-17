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
    DimensionValue, Domain, DynamicBroadcastOperation, InterpretableOperation, InterpretationDriver, Operation,
    OperationProjection, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue,
    PartiallyEvaluatableOperation, ProgramError, ReferenceAccessDescriptor, ReferenceAccessOperation,
    ReferenceAddUpdateOperation, ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation,
    ReferenceReadOperation, ReferenceSwapOperation, ReferenceWriteOperation, RegionInterface, TypeError,
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

    /// Observes a pending completion reference without completing its accesses or publishing writes.
    /// Hardware group counts and commit ordering remain the responsibility of the adapter.
    Commit {
        /// Index of the input that holds the observed completion reference.
        completion_input_index: usize,
    },

    /// Consumes a pending completion reference and publishes its deferred writes.
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

    /// Returns an exact, versioned encoding of this operation's semantic payload.
    ///
    /// The encoding must include an unambiguous owner namespace and schema version, every instruction mode and
    /// semantic option, and exact floating-point bits. It must not depend on diagnostic formatting or addresses.
    /// Inputs, results, regions, and their types are encoded by the enclosing kernel. Type identities in the
    /// payload must participate in ordinary operation identity renaming before this function is called.
    /// Unsupported payloads fail explicitly, and implementing this function does not enable source deserialization.
    fn semantic_key(&self) -> Result<Vec<u8>, TypeError> {
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
            // Only concrete scalar prefetch values specialize the call. Ordinary data inputs remain residual, even when
            // known, because static transforms must never execute the mutations of a kernel.
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

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Cow;
    use std::cell::Cell;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::macros::check_count;
    use ryft_core::{
        AddOperation, ArrayReferenceTransformIndex, AttachedRegions, Condition, DataType, Dimension, DimensionBounds,
        DimensionMulOperation, DimensionVariable, EagerContext, EffectClasses, Effects, EmptyRegionDriver,
        ErasedOperation, NegOperation, OperationFoldOutput, OperationPayloadProjection, PartialEvaluationOutput,
        PartialValue, Placeholder, ProgramBuilder, ReferenceAccessMode, ReferenceEffect, ReferenceRead, ReferenceSwap,
        ReferenceType, ReferenceWrite, RegionDataFlow, ScanOperation, Shape, SubOperation, TagOperation, Tracer,
        TracingContext,
    };

    use crate::calls::KernelParameter;
    use crate::grids::Grid;
    use crate::mappings::{BlockMapping, BoundaryPolicy};
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

        /// Creates one uninitialized local reference allocation with the given referent.
        Allocate(ArrayType),

        /// Allocates a reference while claiming synchronous memory semantics.
        MisclassifiedAllocation,

        /// Has no accesses and completes synchronously.
        Synchronous,

        /// Asynchronous update from two sources that reads two references, writes a third, and allocates a completion.
        Update,

        /// Observes a pending completion reference.
        Commit,

        /// Consumes a pending completion reference.
        Wait,

        /// Consumes a local allocation.
        Release,
    }

    impl Operation for TestExtension {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Rebinding => "test_rebinding",
                Self::HiddenKey { .. } => "hidden_key_extension",
                Self::Allocate(_) => "test_allocate",
                Self::MisclassifiedAllocation => "test_misclassified_allocation",
                Self::Synchronous => "test_synchronous",
                Self::Update => "test_async_update",
                Self::Commit => "test_commit",
                Self::Wait => "test_wait",
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
                Self::HiddenKey { .. } | Self::Synchronous | Self::Commit | Self::Wait | Self::Release => vec![],
                Self::Allocate(referent) => vec![ArrayIrType::Reference(ReferenceType::new(referent.clone()))],
                Self::MisclassifiedAllocation => {
                    vec![ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::I32)))]
                }
                Self::Update => vec![ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Token)))],
            })
        }

        fn effects(&self) -> Cow<'_, Effects> {
            let declarations = match self {
                Self::Rebinding | Self::HiddenKey { .. } => return Cow::Borrowed(Effects::empty()),
                Self::Allocate(_) | Self::MisclassifiedAllocation => {
                    vec![ReferenceEffect::Allocate { output_index: 0 }]
                }
                Self::Synchronous => vec![],
                Self::Update => vec![
                    ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 1, mode: ReferenceAccessMode::Read },
                    ReferenceEffect::Access { input_index: 2, mode: ReferenceAccessMode::Write },
                    ReferenceEffect::Allocate { output_index: 0 },
                ],
                Self::Commit => vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }],
                Self::Wait | Self::Release => {
                    vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Consume }]
                }
            };
            Cow::Owned(Effects::new(EffectClasses::NONE, declarations).unwrap())
        }
    }

    impl ReferenceAccessOperation for TestExtension {
        type Transform = ArrayReferenceTransform;

        fn base_input_count(&self) -> usize {
            match self {
                Self::Rebinding
                | Self::HiddenKey { .. }
                | Self::Allocate(_)
                | Self::MisclassifiedAllocation
                | Self::Synchronous => 0,
                Self::Update => 3,
                Self::Commit | Self::Wait | Self::Release => 1,
            }
        }

        fn reference_access_descriptor(
            &self,
            input_index: usize,
        ) -> Option<ReferenceAccessDescriptor<'_, ArrayReferenceTransform>> {
            match self {
                Self::Rebinding | Self::HiddenKey { .. } => None,
                _ => self.effects().accesses().any(|(index, _)| index == input_index).then(|| {
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
                Self::Rebinding | Self::MisclassifiedAllocation | Self::Synchronous => {
                    KernelExtensionMemory::Synchronous
                }
                Self::HiddenKey { .. } => {
                    return Err(TypeError::invalid(format!(
                        "kernel extension `{}` has no initialization contract",
                        self.name(),
                    )));
                }
                Self::Allocate(_) => KernelExtensionMemory::Allocation { output_index: 0 },
                Self::Update => KernelExtensionMemory::Asynchronous { completion_output_index: 0 },
                Self::Commit => KernelExtensionMemory::Commit { completion_input_index: 0 },
                Self::Wait => KernelExtensionMemory::Wait { completion_input_index: 0 },
                Self::Release => KernelExtensionMemory::Release { input_index: 0 },
            })
        }

        fn semantic_key(&self) -> Result<Vec<u8>, TypeError> {
            match self {
                Self::HiddenKey { key: payload, exact: true } => {
                    let mut key = b"test.hidden_key_extension\0v1\0".to_vec();
                    key.extend_from_slice(&payload.to_le_bytes());
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
