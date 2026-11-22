//! XLA backend operation family and the StableHLO lowering of its operations.
//!
//! [`Operation`] is the universe of staged operations that the XLA backend owns. It flattens the core array operation
//! payloads of [`ArrayIrOperation`] into one family, together with backend operations such as
//! [`jit_call`](JitCallOperation) and portable kernel instructions, so that the regions of higher-order instructions
//! can contain backend operations. [`Program`], [`ProgramBuilder`], and [`FlatProgram`] specialize the core program
//! types to this family. Each submodule implements [`LowerableOperation`](crate::LowerableOperation) for one group of
//! operations, and this module also provides the elementwise operand normalization and the result accuracy conversion
//! that those lowerings share.

use ryft_core::macros::check_count;
use ryft_core::{
    AbsOperation, Accuracy as CoreAccuracy, AddOperation, AndOperation, ArgMaxOperation, ArgMinOperation, Array,
    ArrayIrOperation, ArrayOperation, ArrayReferenceTransform, ArrayType, AssertOperation, Atan2Operation,
    AxisIndexOperation, BroadcastLikeOperation, BroadcastOperation, CastOperation, CeilOperation, ClampOperation,
    CompareOperation, ComplexOperation, CompositeConversion, CompositeOperation, CompositeTransposeOperation,
    ConcatenateOperation, ConditionOperation, ConjugateOperation, ConstantOperation, ConstrainShardingOperation,
    ConvertElementTypeOperation, CosOperation, CumulativeOperation, CustomCallOperation, Dimension,
    DimensionAddOperation, DimensionDivOperation, DimensionFromScalarOperation, DimensionMaxOperation,
    DimensionMinOperation, DimensionMulOperation, DimensionOperation, DimensionPowOperation, DimensionRemOperation,
    DimensionSizeOperation, DimensionSubOperation, DimensionToScalarOperation, DimensionType, DimensionValue,
    DivOperation, DotOperation, DropoutOperation, DynamicBroadcastOperation, DynamicReshapeOperation,
    DynamicSliceOperation, DynamicUpdateSliceOperation, ErfOperation, ExpM1Operation, ExpOperation, FillLikeOperation,
    FloorOperation, GatherOperation, GuardOperation, ImaginaryOperation, IotaLikeOperation, IotaOperation,
    LiftedCustomRules, LinearCallOperation, Ln1pOperation, LogAddExpOperation, LogOperation, LogisticOperation,
    MaxOperation, MinOperation, MulOperation, NegOperation, NotOperation, OneLikeOperation, OneOperation,
    Operation as CoreOperation, OperationProvider, OrOperation, PadOperation, ParallelAllGatherOperation,
    ParallelAllToAllOperation, ParallelBroadcastOperation, ParallelCastOperation, ParallelPermuteOperation,
    ParallelRaggedAllToAllOperation, ParallelReduceOperation, ParallelSumScatterOperation, ParallelVaryOperation,
    PlaceOperation, PowOperation, PrintOperation, Program as CoreProgram, ProgramBuilder as CoreProgramBuilder,
    ProgramError, RaggedDotOperation, RealOperation, ReduceOperation, ReducePrecisionOperation,
    ReferenceAccessDescriptor, ReferenceAccessOperation, ReferenceAddUpdateOperation,
    ReferenceAtomicAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
    ReferenceSwapOperation, ReferenceWriteOperation, RemOperation, RematerializeOperation, ReshapeOperation,
    ReshardOperation, RngBitGeneratorOperation, RoundOperation, RsqrtOperation, SaturatingSubOperation,
    ScaledDotOperation, ScanOperation, ScatterOperation, SelectOperation, ShardMapOperation, SignOperation,
    SinOperation, SliceOperation, SortOperation, SqrtOperation, StopGradientOperation, SubOperation, TagOperation,
    TanOperation, TanhOperation, TransferToMemoryOperation, TransposeOperation, UnavailableCustomRules,
    UpdateSliceOperation, Value, ValueProjection, WhileOperation, XorOperation, ZeroLikeOperation, ZeroOperation,
};
use ryft_kernels::KernelReferenceOperation;
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::Accuracy;
use ryft_mlir::{
    Block, Context as MlirContext, Location, Operation as MlirOperation, TensorTypeRef, Type as MlirType,
    Value as MlirValue, ValueRef,
};

use crate::lowering::types::{lower_restore_dynamic_dimensions, lower_tensor_type, physical_bound_type};
use crate::operations::constants::lower_layout_constraint;
use crate::{Constant, Error, KernelOperation, LoweringError, Type};

// TODO(eaplatanios): Review this module.

/// StableHLO lowering of arithmetic operations.
pub(crate) mod arithmetic;
/// Runtime assertions, including their StableHLO lowering to the `ryft.assert` XLA FFI custom call and the handler
/// that reports failed assertions on the host.
pub(crate) mod assertions;
/// StableHLO lowering of dot product attention operations.
pub(crate) mod attention;
/// StableHLO lowering of collective operations.
pub(crate) mod collectives;
/// StableHLO lowering of comparison operations.
pub(crate) mod comparisons;
/// StableHLO lowering of complex number operations.
pub(crate) mod complex;
/// StableHLO lowering of Ryft composites. A named composite call lowers to the backend implementation that replaces it
/// or to a call of the shared function that implements its body, any other composite call inlines only its primal
/// program, and composite transposes are rejected because transposition must replace them before lowering.
pub(crate) mod composites;
/// StableHLO lowering of constant and constructor operations (e.g., `zero`, `one`, `iota`, and literal constants).
///
/// Eager evaluation and staging allow unbounded dimensions, but executable zero, one, iota, and like constructors need
/// finite allocation bounds. Their dynamic extents must fit signed 32-bit size carriers. A bound reserves storage,
/// while the explicit dimension inputs or exemplar sizes determine the runtime shape. Explicit constructor layouts
/// support dense minor-to-major axis permutations, and byte-strided layouts and physical tiles are rejected.
pub(crate) mod constants;
/// StableHLO lowering of control flow operations.
pub(crate) mod control_flow;
/// StableHLO lowering of cumulative operations (i.e., prefix scans).
pub(crate) mod cumulative;
/// StableHLO lowering of custom calls.
pub(crate) mod custom_call;
/// Debugging support through host callbacks, including the StableHLO lowering of `print` to the `ryft.print` XLA FFI
/// custom call, its handler, and its capturable print sink.
pub mod debugging;
/// StableHLO lowering of operations specific to differentiation (e.g., `rematerialize` and `linear_call`).
pub(crate) mod differentiation;
/// StableHLO lowering of first-class dimension operations.
pub(crate) mod dimensions;
/// StableHLO lowering of generalized dot products, including grouped dots (i.e., ragged dots).
pub mod dot;
/// StableHLO lowering of exponential and logarithmic operations.
pub(crate) mod exponential;
/// StableHLO lowering of extrema operations (i.e., `min`, `max`, and `clamp`).
pub(crate) mod extrema;
/// The `jit_call` operation that the backend owns, which calls staged XLA programs, and its lowering, including the
/// deduplication of call sites.
pub mod jit_call;
/// StableHLO lowering of logical operations.
pub(crate) mod logical;
/// StableHLO lowering of array manipulation operations.
pub(crate) mod manipulation;
/// StableHLO lowering of quantized operations such as scaled dot products.
pub(crate) mod quantization;
/// StableHLO lowering of random number generation operations.
pub(crate) mod random;
/// StableHLO lowering of reduction operations.
pub(crate) mod reductions;
/// Rejection of unresolved reference operations, which reference discharge must remove before StableHLO lowering.
pub(crate) mod references;
/// StableHLO lowering of rounding operations.
pub(crate) mod rounding;
/// StableHLO and Shardy lowering of sharding operations.
pub mod sharding;
/// StableHLO lowering of sorting operations.
pub(crate) mod sorting;
/// StableHLO lowering of special functions.
pub(crate) mod special;
/// StableHLO lowering of value tagging.
pub(crate) mod tagging;
/// StableHLO lowering of trigonometric and hyperbolic operations.
pub(crate) mod trigonometric;

pub use debugging::{
    PRINT_CUSTOM_CALL_TARGET, PRINT_LABEL_ATTRIBUTE, ensure_print_handler_registered, with_captured_prints,
};
pub use dot::RaggedDotLoweringStrategy;
pub use jit_call::{JIT_CALL_OPERATION_NAME, JitCallOperation, transpose_primal_jit_call};

/// Ordinary universe of staged operations that the XLA backend owns.
///
/// This enum flattens the core array operation payloads directly into the operation family that the backend owns.
/// Higher-order instructions attach their nested computations as regions of the containing XLA program, so those
/// regions can contain operations specific to the backend such as [`jit_call`](JitCallOperation) and
/// [`shard_map`](ShardMapOperation).
///
/// The [`CoreOperation`](ryft_core::Operation) contract, the interpretation and partial evaluation dispatchers, the
/// forward mode and transposition dispatchers, the projections onto the member operation families, and the payload
/// conversions are all derived. The variant classes mirror [`ArrayIrOperation`] exactly for the payloads the two
/// families share, so both dispatchers report identical semantics for every shared member and mixed payload. Only the
/// surfaces that the backend owns stay handwritten, namely the normalizing conversions that select between a member
/// and a mixed carrier, the providers of zeros and residual zeros, the canonical view as a core operation that lowering
/// uses, and the MLIR lowering dispatch.
#[derive(Clone, Debug, ryft_macros::Operation)]
#[ryft(
    crate = "ryft_core",
    type(Type),
    constant(Payload),
    members(ArrayType, structural(DimensionType)),
    dispatch(identity, discharge, partial_evaluation, batching, differentiation, transposition)
)]
pub enum Operation<Payload = Constant>
where
    Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    /// Portable kernel instruction whose actual body remains attached until explicit XLA compiler selection.
    Kernel(KernelOperation),

    /// Mixed zero constructor whose explicit first-class dimension operands provide its dynamic result extents.
    /// This variant cannot be represented by the homogeneous array member because its signature crosses member
    /// kinds, since it consumes dimension members and produces an array member.
    #[ryft(mixed(structural), skip_from)]
    Zero(ZeroOperation<ArrayType>),

    /// Mixed one constructor whose explicit operands provide its dynamic result extents.
    #[ryft(mixed(structural), skip_from)]
    DynamicOne(OneOperation<ArrayType>),

    /// Mixed iota constructor whose explicit operands provide its dynamic result extents.
    #[ryft(mixed(structural), skip_from)]
    DynamicIota(IotaOperation<ArrayType>),

    /// Homogeneous array operation. Member zero constructors are promoted to their mixed array IR carrier when an
    /// array operation is lifted into this family.
    #[ryft(projected(ArrayType), collective, lockstep, skip_from)]
    Array(ArrayOperation<<Payload as ValueProjection<ArrayType>>::Projected>),

    /// Homogeneous first-class dimension operation.
    #[ryft(projected(DimensionType, structural))]
    Dimension(DimensionOperation<DimensionValue>),

    /// Mixed comparison of two dimensions producing Boolean array data.
    Compare(CompareOperation<Type>),

    /// Ordered assertion with optional scalar diagnostic inputs.
    Assert(AssertOperation<Type>),

    /// Runtime check of the identity relationships of an entry that the actual input types do not prove, which
    /// returns its inputs narrowed by the facts that those relationships prove.
    Guard(GuardOperation<Type>),

    /// Conversion of a value to a compatible array or first-class dimension type.
    Cast(CastOperation<Type>),

    /// Reads an array extent as a first-class dimension.
    DimensionSize(DimensionSizeOperation),

    /// Unresolved allocation of a reference to a whole array, retained until reference discharge.
    ReferenceNew(ReferenceNewOperation<ArrayType, Type>),

    /// Unresolved read through an optional folded transform path retained until reference discharge.
    ReferenceRead(ReferenceReadOperation<ArrayType, Type, ArrayReferenceTransform>),

    /// Unresolved replacement that only writes, through an optional folded transform path, retained until reference
    /// discharge.
    ReferenceWrite(ReferenceWriteOperation<ArrayType, Type, ArrayReferenceTransform>),

    /// Unresolved replacement through an optional folded transform path retained until reference discharge.
    ReferenceSwap(ReferenceSwapOperation<ArrayType, Type, ArrayReferenceTransform>),

    /// Unresolved additive update through an optional folded transform path retained until reference discharge.
    ReferenceAddUpdate(ReferenceAddUpdateOperation<ArrayType, Type, ArrayReferenceTransform>),

    /// Unresolved atomic additive update retained until sequential reference discharge or a supporting kernel lowering.
    ReferenceAtomicAddUpdate(ReferenceAtomicAddUpdateOperation<ArrayType, Type, ArrayReferenceTransform>),

    /// Unresolved consuming freeze of a reference to a whole array, retained until reference discharge.
    ReferenceFreeze(ReferenceFreezeOperation<ArrayType, Type>),

    /// Converts scalar array data into a checked first-class dimension.
    DimensionFromScalar(DimensionFromScalarOperation),

    /// Converts a first-class dimension into scalar array data.
    DimensionToScalar(DimensionToScalarOperation),

    /// Reshapes an array using explicit dimension operands.
    Reshape(DynamicReshapeOperation),

    /// Broadcasts an array using explicit dimension operands.
    Broadcast(DynamicBroadcastOperation),

    /// Concatenates arrays with an explicit result extent.
    Concatenate(ConcatenateOperation<Type>),

    /// Calls a foreign kernel with explicit dynamic result extents.
    #[ryft(mixed)]
    CustomCall(CustomCallOperation),

    /// Pads an array with explicit result extents.
    Pad(PadOperation<Type>),

    /// Slices an array using first-class start and size dimensions.
    DynamicSlice(DynamicSliceOperation<Type>),

    /// Generates random bits with explicit dynamic result extents.
    RngBitGenerator(RngBitGeneratorOperation<Type>),

    /// Gathers values with one explicit extent per result axis.
    #[ryft(mixed, collective, lockstep)]
    ParallelAllGather(ParallelAllGatherOperation),

    /// Scatters values with one explicit extent per result axis.
    #[ryft(mixed, collective, lockstep)]
    ParallelSumScatter(ParallelSumScatterOperation),

    /// Exchanges values with one explicit extent per result axis.
    #[ryft(mixed, collective, lockstep)]
    ParallelAllToAll(ParallelAllToAllOperation),

    /// Exchanges slices of variable length along the leading axis through an XLA FFI custom call.
    #[ryft(mixed, collective, lockstep)]
    ParallelRaggedAllToAll(ParallelRaggedAllToAllOperation),

    /// Condition owned by the backend, whose attached branch regions can contain XLA operations.
    #[ryft(lockstep)]
    Condition(ConditionOperation<Type>),

    /// Loop owned by the backend, whose attached condition and body regions can contain XLA operations.
    #[ryft(lockstep)]
    While(WhileOperation<Type>),

    /// Scan owned by the backend, whose attached body region can contain XLA operations.
    #[ryft(lockstep)]
    Scan(ScanOperation<Type>),

    /// Composite call owned by the backend, whose primal region can contain XLA operations, with either attached rule
    /// regions that can contain XLA operations or retained rules registered in this family, which are traced lazily.
    /// Its retained rules can communicate along mesh axes that no instruction of the call names, which the call reports
    /// as hidden axes, so this variant delegates the collective metadata of the family to it.
    #[ryft(collective, lockstep)]
    Composite(CompositeOperation<Payload, Operation<Payload>>),

    /// Reverse mode carrier of a [`Composite`](Self::Composite) call, which transposition replaces with the call's
    /// backward rule.
    #[ryft(collective)]
    CompositeTranspose(CompositeTransposeOperation<Payload, Operation<Payload>>),

    /// Call owned by differentiation to an explicitly transposable linear map with ordinary trailing residual operands,
    /// which lowers by inlining its forward region.
    #[ryft(lockstep)]
    LinearCall(LinearCallOperation<Type>),

    /// Rematerialized call owned by the backend, whose body can contain XLA operations.
    #[ryft(lockstep)]
    Rematerialize(RematerializeOperation<Type>),

    /// Composite call whose retained rules are registered in the [`ArrayOperation`] member family, promoted from its
    /// member payload. Its derivative rules are specialized by the member definition and converted into this family as
    /// [`LiftedCustomRules`]. Conversions never produce such a call with attached rules, which follow their
    /// call into the [`Composite`](Self::Composite) variant instead.
    #[ryft(collective, lockstep)]
    LiftedComposite(
        CompositeOperation<
            Payload,
            Operation<Payload>,
            LiftedCustomRules<
                <Payload as ValueProjection<ArrayType>>::Projected,
                ArrayOperation<<Payload as ValueProjection<ArrayType>>::Projected>,
            >,
        >,
    ),

    /// Reverse mode carrier of a [`LiftedComposite`](Self::LiftedComposite) call, whose backward rule is specialized by
    /// the member definition and converted into this family.
    #[ryft(collective)]
    LiftedCompositeTranspose(
        CompositeTransposeOperation<
            Payload,
            Operation<Payload>,
            LiftedCustomRules<
                <Payload as ValueProjection<ArrayType>>::Projected,
                ArrayOperation<<Payload as ValueProjection<ArrayType>>::Projected>,
            >,
        >,
    ),

    /// Call to a flat staged XLA program.
    JitCall(JitCallOperation<Type>),

    /// Core `shard_map` operation, i.e., a [`ShardMapOperation`], which lowering emits as a Shardy
    /// `sdy.manual_computation` through its dedicated path rather than through the core array IR lowering.
    ShardMap(Box<ShardMapOperation>),
}

impl<Payload> ReferenceAccessOperation for Operation<Payload>
where
    Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Transform = ArrayReferenceTransform;

    fn base_input_count(&self) -> usize {
        match self {
            Self::Kernel(operation) => operation.operation().base_input_count(),
            Self::ReferenceRead(operation) => operation.base_input_count(),
            Self::ReferenceWrite(operation) => operation.base_input_count(),
            Self::ReferenceSwap(operation) => operation.base_input_count(),
            Self::ReferenceAddUpdate(operation) => operation.base_input_count(),
            Self::ReferenceAtomicAddUpdate(operation) => operation.base_input_count(),
            Self::ReferenceFreeze(_) => 1,
            _ => 0,
        }
    }

    fn reference_access_descriptor(
        &self,
        input_index: usize,
    ) -> Option<ReferenceAccessDescriptor<'_, Self::Transform>> {
        match self {
            Self::Kernel(operation) => operation.operation().reference_access_descriptor(input_index),
            Self::ReferenceRead(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceWrite(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceSwap(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceAddUpdate(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceAtomicAddUpdate(operation) => operation.reference_access_descriptor(input_index),
            Self::ReferenceFreeze(_) if input_index == 0 => Some(ReferenceAccessDescriptor::new(&[], 1..1)),
            _ => None,
        }
    }

    fn with_reference_access_transforms(
        &self,
        input_index: usize,
        transforms: Vec<Self::Transform>,
    ) -> Result<Self, ProgramError> {
        match self {
            Self::Kernel(operation) => operation
                .operation()
                .with_reference_access_transforms(input_index, transforms)
                .map(|operation| Self::Kernel(operation.into())),
            Self::ReferenceRead(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceRead)
            }
            Self::ReferenceWrite(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceWrite)
            }
            Self::ReferenceSwap(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceSwap)
            }
            Self::ReferenceAddUpdate(operation) => {
                operation.with_reference_access_transforms(input_index, transforms).map(Self::ReferenceAddUpdate)
            }
            Self::ReferenceAtomicAddUpdate(operation) => operation
                .with_reference_access_transforms(input_index, transforms)
                .map(Self::ReferenceAtomicAddUpdate),
            _ if self.reference_access_descriptor(input_index).is_some() && transforms.is_empty() => Ok(self.clone()),
            _ => Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` cannot replace the reference transforms at input {input_index}", self.name()),
            }),
        }
    }
}

impl<Payload> KernelReferenceOperation for Operation<Payload>
where
    Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn swap_output_index(&self) -> Option<usize> {
        match self {
            Self::Kernel(operation) => operation.operation().swap_output_index(),
            _ => matches!(self, Self::ReferenceSwap(_)).then_some(0),
        }
    }
}

impl<Payload> From<ArrayOperation<Payload::Projected>> for Operation<Payload>
where
    Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: ArrayOperation<Payload::Projected>) -> Self {
        // Delegating to the array IR family's conversion keeps constructor normalization and member control flow
        // promotion identical across both families. `Condition`, `While`, and `Scan` must become array IR carriers
        // because the projected `Array` variant cannot own array IR regions.
        ArrayIrOperation::<Payload::Projected>::from(operation).into()
    }
}

impl<Payload> From<ArrayIrOperation<Payload::Projected>> for Operation<Payload>
where
    Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn from(operation: ArrayIrOperation<Payload::Projected>) -> Self {
        match operation {
            ArrayIrOperation::Zero(operation) => Self::Zero(operation),
            ArrayIrOperation::One(operation) => Self::DynamicOne(operation),
            ArrayIrOperation::Iota(operation) => Self::DynamicIota(operation),
            ArrayIrOperation::Array(operation) => Self::Array(operation),
            ArrayIrOperation::Dimension(operation) => Self::Dimension(operation),
            ArrayIrOperation::Compare(operation) => Self::Compare(operation),
            ArrayIrOperation::Assert(operation) => Self::Assert(operation),
            ArrayIrOperation::Guard(operation) => Self::Guard(operation),
            ArrayIrOperation::Cast(operation) => Self::Cast(operation),
            ArrayIrOperation::DimensionSize(operation) => Self::DimensionSize(operation),
            ArrayIrOperation::ReferenceNew(operation) => Self::ReferenceNew(operation),
            ArrayIrOperation::ReferenceRead(operation) => Self::ReferenceRead(operation),
            ArrayIrOperation::ReferenceWrite(operation) => Self::ReferenceWrite(operation),
            ArrayIrOperation::ReferenceSwap(operation) => Self::ReferenceSwap(operation),
            ArrayIrOperation::ReferenceAddUpdate(operation) => Self::ReferenceAddUpdate(operation),
            ArrayIrOperation::ReferenceAtomicAddUpdate(operation) => Self::ReferenceAtomicAddUpdate(operation),
            ArrayIrOperation::ReferenceFreeze(operation) => Self::ReferenceFreeze(operation),
            ArrayIrOperation::DimensionFromScalar(operation) => Self::DimensionFromScalar(operation),
            ArrayIrOperation::DimensionToScalar(operation) => Self::DimensionToScalar(operation),
            ArrayIrOperation::Reshape(operation) => Self::Reshape(operation),
            ArrayIrOperation::Broadcast(operation) => Self::Broadcast(operation),
            ArrayIrOperation::Concatenate(operation) => Self::Concatenate(operation),
            ArrayIrOperation::CustomCall(operation) => Self::CustomCall(operation),
            ArrayIrOperation::Pad(operation) => Self::Pad(operation),
            ArrayIrOperation::DynamicSlice(operation) => Self::DynamicSlice(operation),
            ArrayIrOperation::RngBitGenerator(operation) => Self::RngBitGenerator(operation),
            ArrayIrOperation::ParallelAllGather(operation) => Self::ParallelAllGather(operation),
            ArrayIrOperation::ParallelSumScatter(operation) => Self::ParallelSumScatter(operation),
            ArrayIrOperation::ParallelAllToAll(operation) => Self::ParallelAllToAll(operation),
            ArrayIrOperation::ParallelRaggedAllToAll(operation) => Self::ParallelRaggedAllToAll(operation),
            ArrayIrOperation::Condition(operation) => Self::Condition(operation),
            ArrayIrOperation::While(operation) => Self::While(operation),
            ArrayIrOperation::Scan(operation) => Self::Scan(operation),
            ArrayIrOperation::LinearCall(operation) => Self::LinearCall(operation),
            ArrayIrOperation::Rematerialize(operation) => Self::Rematerialize(operation),
            ArrayIrOperation::ShardMap(operation) => Self::from(operation),
            // Composite calls and carriers with attached rules follow their regions into the native variants. Converted
            // member calls keep their member definition. Calls registered in the array IR family itself cannot follow
            // their program, because constants of this family cannot represent array IR values, so their rules become
            // unavailable while their primal still executes.
            ArrayIrOperation::Composite(operation) => match operation
                .into_family(|rules| LiftedCustomRules::Unavailable(UnavailableCustomRules::from_source(&rules)))
            {
                CompositeConversion::Attached(operation) => Self::Composite(operation),
                CompositeConversion::Retained(operation) => Self::LiftedComposite(operation),
            },
            ArrayIrOperation::CompositeTranspose(operation) => match operation
                .into_family(|rules| LiftedCustomRules::Unavailable(UnavailableCustomRules::from_source(&rules)))
            {
                CompositeConversion::Attached(operation) => Self::CompositeTranspose(operation),
                CompositeConversion::Retained(operation) => Self::LiftedCompositeTranspose(operation),
            },
            ArrayIrOperation::LiftedComposite(operation) => match operation.into_family(|rules| rules) {
                CompositeConversion::Attached(operation) => Self::Composite(operation),
                CompositeConversion::Retained(operation) => Self::LiftedComposite(operation),
            },
            ArrayIrOperation::LiftedCompositeTranspose(operation) => match operation.into_family(|rules| rules) {
                CompositeConversion::Attached(operation) => Self::CompositeTranspose(operation),
                CompositeConversion::Retained(operation) => Self::LiftedCompositeTranspose(operation),
            },
        }
    }
}

// Dimension constants additionally lift directly, so that generic staging code (e.g., `ExactShape::dimensions`)
// can bound only `From<ConstantOperation<DimensionValue>>` without naming this family's dimension member.
impl<Payload> From<ConstantOperation<DimensionValue>> for Operation<Payload>
where
    Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: ConstantOperation<DimensionValue>) -> Self {
        Self::Dimension(DimensionOperation::Constant(operation))
    }
}

macro_rules! impl_array_ir_operation_conversion {
    // Generates the array IR operation conversions that the derive cannot generate directly, namely the canonical
    // constructors whose lift selects between the homogeneous member carrier and the mixed carrier with dimension
    // operands, and the homogeneous array payload forms of mixed operations whose canonical array IR payload has a
    // promoted type.
    ($operation:ty) => {
        impl<Payload> From<$operation> for Operation<Payload>
        where
            Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
        {
            #[inline]
            fn from(operation: $operation) -> Self {
                ArrayIrOperation::<Payload::Projected>::from(operation).into()
            }
        }
    };
}

// Every first-class dimension arithmetic operation lifts directly, mirroring the array IR family, so that generic
// staging code can bound conversions such as `From<DimensionMulOperation>` without naming this family's dimension
// member.
impl_array_ir_operation_conversion!(DimensionAddOperation);
impl_array_ir_operation_conversion!(DimensionSubOperation);
impl_array_ir_operation_conversion!(SaturatingSubOperation<DimensionType>);
impl_array_ir_operation_conversion!(DimensionMulOperation);
impl_array_ir_operation_conversion!(DimensionPowOperation);
impl_array_ir_operation_conversion!(DimensionDivOperation);
impl_array_ir_operation_conversion!(DimensionRemOperation);
impl_array_ir_operation_conversion!(DimensionMinOperation);
impl_array_ir_operation_conversion!(DimensionMaxOperation);

impl_array_ir_operation_conversion!(ZeroLikeOperation<Type>);
impl_array_ir_operation_conversion!(OneLikeOperation<Type>);
impl_array_ir_operation_conversion!(FillLikeOperation<Type>);
impl_array_ir_operation_conversion!(IotaLikeOperation<Type>);
impl_array_ir_operation_conversion!(ZeroOperation<ArrayType>);
impl_array_ir_operation_conversion!(OneOperation<ArrayType>);
impl_array_ir_operation_conversion!(IotaOperation<ArrayType>);
impl_array_ir_operation_conversion!(ConcatenateOperation<ArrayType>);
impl_array_ir_operation_conversion!(PadOperation<ArrayType>);

macro_rules! impl_array_operation_conversion {
    // Generates the conversions of homogeneous array operations through the canonical projected member family.
    ($operation:ty) => {
        impl<Payload> From<$operation> for Operation<Payload>
        where
            Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
        {
            #[inline]
            fn from(operation: $operation) -> Self {
                ArrayOperation::<Payload::Projected>::from(operation).into()
            }
        }
    };
}

impl_array_operation_conversion!(ZeroLikeOperation<ArrayType>);
impl_array_operation_conversion!(OneLikeOperation<ArrayType>);
impl_array_operation_conversion!(ConstantOperation<Array>);
impl_array_operation_conversion!(FillLikeOperation<ArrayType>);
impl_array_operation_conversion!(IotaLikeOperation<ArrayType>);
impl_array_operation_conversion!(AbsOperation<ArrayType>);
impl_array_operation_conversion!(NegOperation<ArrayType>);
impl_array_operation_conversion!(AddOperation<ArrayType>);
impl_array_operation_conversion!(SubOperation<ArrayType>);
impl_array_operation_conversion!(SaturatingSubOperation<ArrayType>);
impl_array_operation_conversion!(MulOperation<ArrayType>);
impl_array_operation_conversion!(DivOperation<ArrayType>);
impl_array_operation_conversion!(SinOperation<ArrayType>);
impl_array_operation_conversion!(CosOperation<ArrayType>);
impl_array_operation_conversion!(TanOperation<ArrayType>);
impl_array_operation_conversion!(Atan2Operation<ArrayType>);
impl_array_operation_conversion!(ExpOperation<ArrayType>);
impl_array_operation_conversion!(ExpM1Operation<ArrayType>);
impl_array_operation_conversion!(LogOperation<ArrayType>);
impl_array_operation_conversion!(Ln1pOperation<ArrayType>);
impl_array_operation_conversion!(LogAddExpOperation<ArrayType>);
impl_array_operation_conversion!(SqrtOperation<ArrayType>);
impl_array_operation_conversion!(RsqrtOperation<ArrayType>);
impl_array_operation_conversion!(TanhOperation<ArrayType>);
impl_array_operation_conversion!(LogisticOperation<ArrayType>);
impl_array_operation_conversion!(ErfOperation<ArrayType>);
impl_array_operation_conversion!(PowOperation<ArrayType>);
impl_array_operation_conversion!(SignOperation<ArrayType>);
impl_array_operation_conversion!(FloorOperation<ArrayType>);
impl_array_operation_conversion!(CeilOperation<ArrayType>);
impl_array_operation_conversion!(RoundOperation<ArrayType>);
impl_array_operation_conversion!(MaxOperation<ArrayType>);
impl_array_operation_conversion!(MinOperation<ArrayType>);
impl_array_operation_conversion!(ClampOperation<ArrayType>);
impl_array_operation_conversion!(RemOperation<ArrayType>);
impl_array_operation_conversion!(NotOperation<ArrayType>);
impl_array_operation_conversion!(AndOperation<ArrayType>);
impl_array_operation_conversion!(OrOperation<ArrayType>);
impl_array_operation_conversion!(XorOperation<ArrayType>);
impl_array_operation_conversion!(ComplexOperation<ArrayType>);
impl_array_operation_conversion!(ConjugateOperation<ArrayType>);
impl_array_operation_conversion!(RealOperation<ArrayType>);
impl_array_operation_conversion!(ImaginaryOperation<ArrayType>);
impl_array_operation_conversion!(DotOperation);
impl_array_operation_conversion!(RaggedDotOperation);
impl_array_operation_conversion!(ScaledDotOperation);
impl_array_operation_conversion!(ReduceOperation);
impl_array_operation_conversion!(ArgMaxOperation);
impl_array_operation_conversion!(ArgMinOperation);
impl_array_operation_conversion!(CumulativeOperation);
impl_array_operation_conversion!(SortOperation);
impl_array_operation_conversion!(ParallelReduceOperation);
impl_array_operation_conversion!(ParallelVaryOperation);
impl_array_operation_conversion!(ParallelCastOperation);
impl_array_operation_conversion!(ParallelPermuteOperation);
impl_array_operation_conversion!(ParallelBroadcastOperation);
impl_array_operation_conversion!(AxisIndexOperation);
impl_array_operation_conversion!(TransposeOperation);
impl_array_operation_conversion!(ReshapeOperation);
impl_array_operation_conversion!(BroadcastOperation);
impl_array_operation_conversion!(BroadcastLikeOperation);
impl_array_operation_conversion!(GatherOperation);
impl_array_operation_conversion!(ScatterOperation);
impl_array_operation_conversion!(SliceOperation);
impl_array_operation_conversion!(UpdateSliceOperation);
impl_array_operation_conversion!(DynamicSliceOperation);
impl_array_operation_conversion!(DynamicUpdateSliceOperation);
impl_array_operation_conversion!(SelectOperation<ArrayType>);
impl_array_operation_conversion!(ConvertElementTypeOperation<ArrayType>);
impl_array_operation_conversion!(ReducePrecisionOperation<ArrayType>);
impl_array_operation_conversion!(TransferToMemoryOperation);
impl_array_operation_conversion!(PlaceOperation);
impl_array_operation_conversion!(ReshardOperation);
impl_array_operation_conversion!(ConstrainShardingOperation);
impl_array_operation_conversion!(StopGradientOperation<ArrayType>);
impl_array_operation_conversion!(TagOperation<ArrayType>);
impl_array_operation_conversion!(DropoutOperation);
impl_array_operation_conversion!(PrintOperation<ArrayType>);

impl_array_operation_conversion!(AssertOperation<ArrayType>);
impl_array_operation_conversion!(GuardOperation<ArrayType>);
impl_array_operation_conversion!(CastOperation<ArrayType>);

impl<Payload> OperationProvider<Type, ZeroOperation<Type>> for Operation<Payload>
where
    Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Operation = Self;

    fn provide(request: ZeroOperation<Type>, input_types: &[&Type]) -> Result<Self, ProgramError> {
        Ok(ArrayIrOperation::<Payload::Projected>::provide(request, input_types)?.into())
    }
}

impl<Payload> OperationProvider<Type, OneOperation<Type>> for Operation<Payload>
where
    Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Operation = Self;

    fn provide(request: OneOperation<Type>, input_types: &[&Type]) -> Result<Self, ProgramError> {
        Ok(ArrayIrOperation::<Payload::Projected>::provide(request, input_types)?.into())
    }
}

// Generic cotangent addition requests the dense array member without adding a second Add conversion.
impl<Payload> OperationProvider<Type, AddOperation<Type>> for Operation<Payload>
where
    Payload: Value<Type = Type> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Operation = Self;

    #[inline]
    fn provide(request: AddOperation<Type>, input_types: &[&Type]) -> Result<Self, ProgramError> {
        Ok(ArrayIrOperation::<Payload::Projected>::provide(request, input_types)?.into())
    }
}

// Array IR operations whose generic capabilities bind them through any context lift into the array member exactly as
// they do in the core array IR family.
impl_array_ir_operation_conversion!(SelectOperation<Type>);
impl_array_ir_operation_conversion!(StopGradientOperation<Type>);
impl_array_ir_operation_conversion!(TagOperation<Type>);
impl_array_ir_operation_conversion!(PrintOperation<Type>);

/// Staged XLA program over an operation family `O`, which defaults to the universe of XLA operations that the backend
/// owns.
pub type Program<Input, Output, O = Operation> = CoreProgram<Constant, O, Input, Output>;

/// Program builder specialized to the universe of XLA operations that the backend owns.
pub type ProgramBuilder = CoreProgramBuilder<Constant, Operation>;

/// Flat XLA program over an operation family `O`, which defaults to the universe of XLA operations that the backend
/// owns, used for materialized regions and shared callees.
pub type FlatProgram<O = Operation> = Program<Vec<Constant>, Vec<Constant>, O>;

/// Converts and broadcasts one implicitly compatible elementwise operand to the exact StableHLO result tensor type.
pub(crate) fn normalize_elementwise_operand<'b, 'c: 'b, 't: 'c, B, L>(
    input: ValueRef<'b, 'c, 't>,
    output_type: &ArrayType,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, Error>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let input_type = input.r#type()?;
    let input_tensor_type = input_type.cast::<TensorTypeRef>().ok_or_else(|| {
        Error::Lowering(LoweringError::UnsupportedOperation {
            operation: "normalize_elementwise_operand".to_string(),
            reason: format!("elementwise operand has non-tensor MLIR type `{input_type}`"),
        })
    })?;
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    let output_element_type = output_tensor_type.element_type()?;
    let mut input = input;
    if input_tensor_type.element_type()? != output_element_type {
        let dimensions = input_tensor_type.dimensions().collect::<Vec<_>>();
        let converted_type = context.tensor_type(output_element_type, dimensions.as_slice(), None, location)?;
        let converted = block.append_operation(stable_hlo::convert(input, converted_type, location)?)?;
        input = converted.result(0).unwrap().as_ref();
    }
    if input_tensor_type.dimensions().ne(output_tensor_type.dimensions()) {
        let output_rank = output_tensor_type.rank();
        let input_rank = input_tensor_type.rank();
        let first_dimension = output_rank.checked_sub(input_rank).ok_or_else(|| {
            Error::Lowering(LoweringError::UnsupportedOperation {
                operation: "normalize_elementwise_operand".to_string(),
                reason: format!("cannot broadcast rank-{input_rank} elementwise operand to rank-{output_rank} output"),
            })
        })?;
        let dimensions = (first_dimension..output_rank).collect::<Vec<_>>();
        let broadcast = block.append_operation(stable_hlo::broadcast(
            input,
            output_tensor_type,
            dimensions.as_slice(),
            location,
        )?)?;
        input = broadcast.result(0).unwrap().as_ref();
    }
    Ok(input)
}

/// Returns the StableHLO result accuracy that requests the provided [`CoreAccuracy`] from the backend.
pub(crate) fn lower_accuracy(accuracy: CoreAccuracy) -> Accuracy {
    match accuracy {
        CoreAccuracy::Default => Accuracy::Default,
        CoreAccuracy::Highest => Accuracy::Highest,
        CoreAccuracy::Tolerance(tolerance) => Accuracy::Tolerance {
            absolute_tolerance: tolerance.absolute(),
            relative_tolerance: tolerance.relative(),
            units_of_least_precision: tolerance.units_of_least_precision(),
        },
    }
}

/// Normalizes both operands of one implicitly broadcasting binary elementwise operation to its exact result type.
pub(crate) fn normalize_binary_elementwise_operands<'b, 'c: 'b, 't: 'c, B, L>(
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<[ValueRef<'b, 'c, 't>; 2], Error>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    check_count!("input", input_values, 2, ProgramError);
    check_count!("output", output_types, 1, ProgramError);
    let output_type = &output_types[0];
    let output_tensor_type = lower_tensor_type(output_type, context, location)?;
    let multiple_dynamic_axes = output_type
        .shape()
        .dimensions()
        .iter()
        .filter(|dimension| matches!(dimension, Dimension::Dynamic(_)))
        .count()
        > 1;
    let mut inputs = [input_values[0], input_values[1]];
    for index in 0..2 {
        let input_tensor_type = input_values[index].r#type()?.cast::<TensorTypeRef>();
        let peer_tensor_type = input_values[1 - index].r#type()?.cast::<TensorTypeRef>();
        if multiple_dynamic_axes
            && input_tensor_type.is_some_and(|r#type| r#type.rank() == 0)
            && peer_tensor_type.is_some_and(|r#type| r#type.dimensions().eq(output_tensor_type.dimensions()))
        {
            // Native broadcast accepts at most one bounded dynamic result axis. For scalar arithmetic over an array
            // with several dynamic axes (e.g., `[rows, columns]`), broadcast to physical capacity and restore both
            // logical extents from the array input. The scalar itself cannot supply these sizes, and inventing the
            // maximum sizes would incorrectly make padding observable. Other broadcasting cases use the ordinary path
            // below.
            let scalar = normalize_elementwise_operand(
                input_values[index],
                &ArrayType::scalar(output_type.data_type()),
                block,
                context,
                location,
            )?;
            let physical_type = physical_bound_type(output_type)?;
            let broadcast = block.append_operation(stable_hlo::broadcast(
                scalar,
                lower_tensor_type(&physical_type, context, location)?,
                &[],
                location,
            )?)?;
            let result = lower_layout_constraint(
                broadcast.result(0).unwrap().as_ref(),
                &physical_type,
                &mut block.as_ref(),
                location.as_ref(),
            )?;
            let sources = (0..output_type.rank()).map(|axis| (input_values[1 - index], axis)).collect::<Vec<_>>();
            inputs[index] = lower_restore_dynamic_dimensions(
                result,
                output_type,
                &sources,
                &mut block.as_ref(),
                context,
                location.as_ref(),
            )?;
        } else {
            inputs[index] = normalize_elementwise_operand(input_values[index], output_type, block, context, location)?;
        }
    }
    Ok(inputs)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Abs, Accuracy, AddOperation, AndOperation, Array as CoreArray, ArrayIrBatch, ArrayIrBatchingPolicy,
        ArrayIrOperation, ArrayIrOperations, ArrayIrValue, ArrayOperation, ArrayOperations, ArrayReferenceTransform, Concatenate, ReductionKind, UnreducedKind,
        ArrayReferenceTransformIndex, ArrayType, Assert, AssertOperation, Atan2, Atan2Operation, BatchableOperation,
        BatchingContext, CaptureReference, CapturingContext, Compare, CompareOperation, ComparisonDirection,
        CompositeJvpRule, CompositeOperation, CompositeTransposeOperation, ConditionOperation, Context, Cos,
        CustomRuleDefinition, CustomRuleRegistration, DataType, DifferentiableType, Dimension, DimensionBounds,
        DimensionDivOperation, DimensionFromScalarOperation, DimensionOperation, DimensionOperations, DimensionType,
        DimensionValue, DimensionVariable, DivOperation, DomainTracingContext, DotsSavable,
        DotsWithNoBatchDimensionsSavable, DynamicBroadcastOperation, EffectClass, EffectClasses, EmptyRegionDriver,
        ErasedOperation, EverythingSavable, Exp, ExternalReferenceBinding, LiftedCustomRules, Log, MaybeZero, Memory,
        MulOperation, NothingSavable, OffloadDotsWithNoBatchDimensions, Operation as CoreOperation,
        OperationPayloadProjection, OrOperation, PartialValue, Placeholder, Pow, ProgramBuilder as CoreProgramBuilder,
        ProgramError, ReferenceAddUpdateOperation, ReferenceAtomicAddUpdateOperation, ReferenceDischargeResult,
        ReferenceDischargeTarget, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation,
        ReferenceSource, ReferenceSwapOperation, ReferenceType, ReferenceWriteOperation, Rem,
        RematerializationOptimizationBarrier, RematerializeOperation, ResidualPolicy, ResidualPolicyReference,
        ResidualZeroProvider, Rsqrt, SaveAndOffloadOnlyTheseNames, SaveFromBothPolicies, SaveOnlyTheseNames,
        ScanOperation, SelectOperation, Shape, Sign, Sin, SinOperation, Sqrt, StagingContext, TagOperation, Tanh,
        Tolerance, Tracer as CoreTracer, TracingContext, TransferToMemoryOperation, TransposedProgramCotangent, Typed,
        TypedProgram, ValueProjection, ValueResolution, WhileOperation, XorOperation, ZeroOperation, rematerialize,
    };

    use crate::lowering::tests::{
        PlainProgram, ProgramBuilder as PlainProgramBuilder, UnreducedApplication, UnreducedInput,
        lower_unreduced_application, lowered_unary_module, to_mlir_module_for_plain_program,
    };
    use crate::operations::jit_call::JitCallOperation;
    use crate::sessions::tests::domain_mesh;
    use crate::tests::{execution_client, replicated_array, values_to_bytes, vector_type};
    use crate::{Array, ArrayConstant, ArrayTracer, Constant, Session, Tracer, Type, to_mlir_module_for_program};

    use super::*;

    fn elementwise_normalization_program() -> PlainProgram {
        let mut builder = PlainProgramBuilder::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F32));
        let left = builder
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Static(1)])));
        let right = builder
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(1), Dimension::Static(4)])));
        let condition = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let boolean_vector =
            builder.add_input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)])));
        let divide = builder.add_instruction(DivOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let atan2 = builder.add_instruction(Atan2Operation::new(), Vec::new(), vec![right, left], None).unwrap()[0];
        let compare = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![left, right],
                None,
            )
            .unwrap()[0];
        let select = builder
            .add_instruction(SelectOperation::new(), Vec::new(), vec![condition, scalar, right], None)
            .unwrap()[0];
        let and = builder
            .add_instruction(AndOperation::new(), Vec::new(), vec![condition, boolean_vector], None)
            .unwrap()[0];
        let or = builder
            .add_instruction(OrOperation::new(), Vec::new(), vec![boolean_vector, condition], None)
            .unwrap()[0];
        let xor = builder
            .add_instruction(XorOperation::new(), Vec::new(), vec![condition, boolean_vector], None)
            .unwrap()[0];
        builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(
                vec![divide, atan2, compare, select, and, or, xor],
                vec![Placeholder; 5],
                vec![Placeholder; 7],
            )
            .unwrap()
    }

    /// Asserts that `stablehlo` is the module that lowering the [`elementwise_normalization_program`] produces, in
    /// which every operand is converted and broadcast to the exact result type of its operation.
    fn assert_elementwise_operands_are_normalized(stablehlo: &str) {
        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<3x1xf32>, %arg2: tensor<1x4xf64>, %arg3: tensor<i1>, %arg4: tensor<4xi1>) -> (tensor<3x4xf64>, tensor<3x4xf64>, tensor<3x4xi1>, tensor<1x4xf64>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>) {
                    %0 = stablehlo.convert %arg1 : (tensor<3x1xf32>) -> tensor<3x1xf64>
                    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<3x1xf64>) -> tensor<3x4xf64>
                    %2 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<1x4xf64>) -> tensor<3x4xf64>
                    %3 = stablehlo.divide %1, %2 : tensor<3x4xf64>
                    %4 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<1x4xf64>) -> tensor<3x4xf64>
                    %5 = stablehlo.convert %arg1 : (tensor<3x1xf32>) -> tensor<3x1xf64>
                    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<3x1xf64>) -> tensor<3x4xf64>
                    %7 = stablehlo.atan2 %4, %6 : tensor<3x4xf64>
                    %8 = stablehlo.convert %arg1 : (tensor<3x1xf32>) -> tensor<3x1xf64>
                    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<3x1xf64>) -> tensor<3x4xf64>
                    %10 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1] : (tensor<1x4xf64>) -> tensor<3x4xf64>
                    %11 = stablehlo.compare GT, %9, %10, FLOAT : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3x4xi1>
                    %12 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<i1>) -> tensor<1x4xi1>
                    %13 = stablehlo.convert %arg0 : (tensor<f32>) -> tensor<f64>
                    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f64>) -> tensor<1x4xf64>
                    %15 = stablehlo.select %12, %14, %arg2 : tensor<1x4xi1>, tensor<1x4xf64>
                    %16 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<i1>) -> tensor<4xi1>
                    %17 = stablehlo.and %16, %arg4 : tensor<4xi1>
                    %18 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<i1>) -> tensor<4xi1>
                    %19 = stablehlo.or %arg4, %18 : tensor<4xi1>
                    %20 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<i1>) -> tensor<4xi1>
                    %21 = stablehlo.xor %20, %arg4 : tensor<4xi1>
                    return %3, %7, %11, %15, %17, %19, %21 : tensor<3x4xf64>, tensor<3x4xf64>, tensor<3x4xi1>, tensor<1x4xf64>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_operation_values_satisfy_the_core_capability_bundles() {
        // Each bundle is satisfied exactly when every one of its member capabilities is, so instantiating these
        // functions is a compile-time assertion that the XLA value families expose the complete value-level surface
        // of the operation families they pair with.
        fn requires_array_operations<V: ArrayOperations>() {}
        fn requires_array_ir_operations<V: ArrayIrOperations>() {}
        fn requires_dimension_operations<V: DimensionOperations>() {}

        requires_array_operations::<Array>();
        requires_array_operations::<ArrayTracer>();
        requires_array_operations::<crate::Value>();
        requires_array_ir_operations::<crate::Value>();
        requires_dimension_operations::<crate::Dimension>();
        requires_array_operations::<Tracer>();
        requires_array_ir_operations::<Tracer>();
        requires_array_ir_operations::<CoreTracer<TracingContext<Constant, Operation>>>();
    }

    #[test]
    fn test_operation_payload_projection_and_construction() {
        // Array payloads are reached through the projected array member in both directions, including tags, while
        // native payloads construct their own variants.
        let add = ErasedOperation::new(AddOperation::<ArrayType>::new());
        let add = Operation::<Constant>::from_payload(add).unwrap();
        assert!(matches!(add, Operation::Array(ArrayOperation::Add(_))));
        assert!(add.projected_payload::<AddOperation<ArrayType>>().is_some());
        let reference = ErasedOperation::new(ReferenceNewOperation::<ArrayType, Type>::new());
        let reference = Operation::<Constant>::from_payload(reference).unwrap();
        assert!(matches!(reference, Operation::ReferenceNew(_)));
        assert!(reference.projected_payload::<ReferenceNewOperation<ArrayType, Type>>().is_some());
        assert!(reference.projected_payload::<AddOperation<ArrayType>>().is_none());
        let tag = ErasedOperation::new(TagOperation::<ArrayType>::new("residual"));
        let tag = Operation::<Constant>::from_payload(tag).unwrap();
        assert_eq!(TagOperation::<Type>::key_of(&tag), Some("residual"));
    }

    #[test]
    fn test_operation_provides_memory_transfers() {
        // The XLA family provides memory transfers through its projected array member.
        let destination = Memory::Host { pinned: true };
        let input_type = Type::Array(ArrayType::new_static(DataType::F32, [2]));
        let operation =
            Operation::<Constant>::provide(TransferToMemoryOperation::new(destination), &[&input_type]).unwrap();
        assert!(matches!(
            &operation,
            Operation::Array(ArrayOperation::TransferToMemory(operation)) if operation.destination() == destination,
        ));
        assert_eq!(operation.to_string(), "transfer_to_memory [destination=Host[Pinned]]");
        assert_eq!(
            Operation::<Constant>::provide(TransferToMemoryOperation::new(destination), &[]).unwrap_err(),
            ProgramError::InvalidInputCount { expected: 1, actual: 0 },
        );
    }

    #[test]
    fn test_dimension_assertion_proofs() {
        let dimension_type = DimensionType::new("extent", DimensionBounds::unbounded());
        let context = TracingContext::<Constant, Operation>::new();
        let dimension = context.input(dimension_type.clone().into());
        let dimension = ValueProjection::<DimensionType>::into_projected(dimension).unwrap();
        let predicate = dimension.equal(&dimension).unwrap();
        assert!(matches!(context.resolve(&predicate), ValueResolution::Constant(Constant::Boolean(true))));
        predicate.assert("identity must agree", &[]).unwrap();
        let predicate = dimension.not_equal(&dimension).unwrap();
        predicate.assert("identity must differ", &[]).unwrap();
        assert!(matches!(
            context.builder().borrow().instructions(),
            [instruction] if matches!(
                instruction.operation(),
                Operation::Assert(operation) if operation.message() == "identity must differ",
            ),
        ));

        let context = TracingContext::<Constant, Operation>::new();
        let mut builder = ProgramBuilder::new();
        let dimension = builder.add_input(dimension_type.clone().into());
        let predicate = builder
            .add_instruction(
                CompareOperation::<Type>::new(ComparisonDirection::Equal),
                vec![],
                vec![dimension, dimension],
                None,
            )
            .unwrap()[0];
        builder
            .add_instruction(AssertOperation::<Type>::new("identity must agree"), vec![], vec![predicate], None)
            .unwrap();
        let program = builder.build::<Vec<Constant>, Vec<Constant>>(vec![], vec![Placeholder], vec![]).unwrap();
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        let evaluation = program
            .partially_evaluate_in_context(&context, &[PartialValue::Unknown(dimension_type.into())])
            .unwrap();
        assert_eq!(evaluation.program().effects().classes(), EffectClasses::NONE);
        assert!(evaluation.program().instructions().is_empty());
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_operation_batch_dimension_member() {
        // The first-class dimension member family has no batching rules of its own, so the derived dispatcher binds a
        // checked dimension division once in the parent trace under the replicated dimension policy. The staged
        // division keeps its replicated inputs and its ordered assertion that the divisor is nonzero.
        let dividend_type = DimensionType::new("dividend", DimensionBounds::new(0, Some(9)).unwrap());
        let divisor_type = DimensionType::new("divisor", DimensionBounds::new(0, Some(5)).unwrap());
        let (_, program) = TracingContext::<Constant, Operation>::trace(
            |inputs: Vec<CoreTracer<TracingContext<Constant, Operation>>>| {
                let parent = inputs[0].context();
                let extent = parent.constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);
                let operation = Operation::from(DimensionOperation::Div(DimensionDivOperation::new(
                    &dividend_type,
                    &divisor_type,
                )?));
                let outputs = operation.batch(
                    &context,
                    &EmptyRegionDriver,
                    &[ArrayIrBatch::replicated(inputs[0].clone()), ArrayIrBatch::replicated(inputs[1].clone())],
                )?;
                assert!(outputs.iter().all(|output| output.batch_axis().is_replicated()));
                Ok::<_, ProgramError>(outputs.into_iter().map(ArrayIrBatch::into_value).collect::<Vec<_>>())
            },
            vec![Type::Dimension(dividend_type.clone()), Type::Dimension(divisor_type.clone())],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<dividend ∈ [0, 9)>, %1:dimension<divisor ∈ [0, 5)> .
                let %2:dimension<3> = const 3
                    %3:dimension<dividend / divisor ∈ [0, 9)> = dimension_div [requires_runtime_assertion=true] %0 %1
                in (%3)"},
        );
        assert_eq!(program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
    }

    #[test]
    fn test_operation_reference_access_descriptor() {
        let transform = ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic };
        let operation = Operation::<Constant>::ReferenceRead(
            ReferenceReadOperation::new().with_transforms(vec![transform.clone()]),
        );
        let descriptor = operation.reference_access_descriptor(0).unwrap();
        assert_eq!(descriptor.transforms(), &[transform]);
        assert_eq!(descriptor.bindings(), 1..2);
        assert_eq!(operation.reference_access_descriptor(1), None);
        let root = ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 2])).into();
        let index = ArrayType::scalar(DataType::I64).into();
        assert_eq!(
            operation.infer_output_types(&[root, index], &[]),
            Ok(vec![ArrayType::new_static(DataType::F32, [2]).into()])
        );
    }

    #[test]
    fn test_core_reference_operations_promote_to_reference_operations() {
        let write: Operation<Constant> =
            ArrayIrOperation::<ArrayConstant>::ReferenceWrite(ReferenceWriteOperation::new()).into();
        assert!(matches!(&write, Operation::ReferenceWrite(_)));

        let swap: Operation<Constant> =
            ArrayIrOperation::<ArrayConstant>::ReferenceSwap(ReferenceSwapOperation::new()).into();
        assert!(matches!(&swap, Operation::ReferenceSwap(_)));

        let add_update: Operation<Constant> =
            ArrayIrOperation::<ArrayConstant>::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()).into();
        assert!(matches!(&add_update, Operation::ReferenceAddUpdate(_)));

        let atomic_update: Operation<Constant> =
            ArrayIrOperation::<ArrayConstant>::ReferenceAtomicAddUpdate(ReferenceAtomicAddUpdateOperation::new())
                .into();
        assert!(matches!(&atomic_update, Operation::ReferenceAtomicAddUpdate(_)));

        let freeze: Operation<Constant> =
            ArrayIrOperation::<ArrayConstant>::ReferenceFreeze(ReferenceFreezeOperation::new()).into();
        assert!(matches!(&freeze, Operation::ReferenceFreeze(_)));
    }

    #[test]
    fn test_reference_discharge_rewrites_the_flat_reference_language() {
        // The backend operation family participates in reference discharge through its generated `discharge`
        // dispatcher, so a flat backend program rewrites into explicit immutable state with its external root
        // entering at its own boundary position and publishing its final state as one hidden output.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut builder = ProgramBuilder::new();
        let external = builder.add_input(Type::Reference(ReferenceType::new(scalar_type.clone())));
        let initial = builder.add_input(Type::Array(scalar_type.clone()));
        let update = builder.add_input(Type::Array(scalar_type));
        let root = builder
            .add_instruction(Operation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![initial], None)
            .unwrap()[0];
        builder
            .add_instruction(
                Operation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                Vec::new(),
                vec![root, update],
                None,
            )
            .unwrap();
        let local = builder
            .add_instruction(Operation::ReferenceFreeze(ReferenceFreezeOperation::new()), Vec::new(), vec![root], None)
            .unwrap()[0];
        let previous = builder
            .add_instruction(
                Operation::ReferenceSwap(ReferenceSwapOperation::new()),
                Vec::new(),
                vec![external, local],
                None,
            )
            .unwrap()[0];
        let source: Program<Vec<Constant>, Vec<Constant>> =
            builder.build(vec![previous], vec![Placeholder; 3], vec![Placeholder]).unwrap();

        let discharged = source.discharge_references(0).unwrap();

        // The one external root is written, so it enters as state at its own boundary position and publishes its
        // final state as the single hidden output after the public prefix, while the local root's allocation,
        // accumulation, and consumption become ordinary array SSA.
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[] .
                let %3:f32[] = add %1 %2
                in (%0, %3)"},
        );
    }

    #[test]
    fn test_core_array_ir_control_flow_promotes_to_control_flow() {
        let condition: Operation<Constant> =
            ArrayIrOperation::<ArrayConstant>::Condition(ConditionOperation::new()).into();
        assert!(matches!(condition, Operation::Condition(_)));
        assert_eq!(condition.region_slots(), ConditionOperation::<Type>::new().region_slots());

        let r#while: Operation<Constant> =
            ArrayIrOperation::<ArrayConstant>::While(WhileOperation::new().with_iteration_bound(3).unwrap()).into();
        assert!(matches!(r#while, Operation::While(operation) if operation.iteration_bound() == Some(3)));

        let scan: Operation<Constant> =
            ArrayIrOperation::<ArrayConstant>::Scan(ScanOperation::new(2, 5).with_reverse(true)).into();
        assert!(matches!(
            scan,
            Operation::Scan(operation)
                if operation.carry_count() == 2
                    && operation.iteration_count() == &Dimension::Static(5)
                    && operation.reverse()
        ));

        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let zero: Operation<Constant> = ArrayOperation::Zero(ZeroOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent)]),
        )))
        .into();
        assert!(matches!(zero, Operation::Zero(_)));

        // Member control flow must also promote to the array IR carriers when it enters through the member family
        // conversion, because the projected `Array` variant cannot own array IR regions, so landing there would make
        // the staged regions unprojectable.
        let condition: Operation<Constant> =
            ArrayOperation::<ArrayConstant>::Condition(ConditionOperation::new()).into();
        assert!(matches!(condition, Operation::Condition(_)));
        let r#while: Operation<Constant> =
            ArrayOperation::<ArrayConstant>::While(WhileOperation::new().with_iteration_bound(3).unwrap()).into();
        assert!(matches!(r#while, Operation::While(operation) if operation.iteration_bound() == Some(3)));
        let scan: Operation<Constant> =
            ArrayOperation::<ArrayConstant>::Scan(ScanOperation::new(2, 5).with_reverse(true)).into();
        assert!(matches!(
            scan,
            Operation::Scan(operation)
                if operation.carry_count() == 2
                    && operation.iteration_count() == &Dimension::Static(5)
                    && operation.reverse()
        ));
    }

    #[test]
    fn test_core_composite_and_rematerialization_promotions_preserve_metadata() {
        // These payloads are promoted by move rather than reconstructed, so their complete stored surface must survive,
        // including the rule layout and the non-differentiated operand split of composite calls, and the policy and
        // flags of rematerialized calls. The promoted carrier must also keep contributing the payload's own operation
        // name and region slots, because the attached regions are matched against those slots by name.
        for (jvp_rule, has_vjp_rule, non_differentiated_count) in [
            (CompositeJvpRule::Explicit, false, 2),
            (CompositeJvpRule::Absent, true, 3),
            (CompositeJvpRule::Derived, true, 1),
        ] {
            let composite = CompositeOperation::from_rule_regions(jvp_rule, has_vjp_rule)
                .with_non_differentiated_count(non_differentiated_count)
                .unwrap();
            let promoted: Operation<Constant> = ArrayIrOperation::<ArrayConstant>::Composite(composite.clone()).into();
            assert!(matches!(
                &promoted,
                Operation::Composite(operation)
                    if operation.rules().is_none()
                        && operation.jvp_rule() == jvp_rule
                        && operation.has_vjp_rule() == has_vjp_rule
                        && operation.non_differentiated_count() == non_differentiated_count,
            ));
            assert_eq!(promoted.name(), composite.name());
            assert_eq!(promoted.region_slots(), composite.region_slots());

            // A call with attached rules that reaches this family from the array member family follows its regions
            // into the native variant as well.
            let promoted: Operation<Constant> = ArrayOperation::<ArrayConstant>::Composite(
                CompositeOperation::from_rule_regions(jvp_rule, has_vjp_rule),
            )
            .into();
            assert!(matches!(
                &promoted,
                Operation::Composite(operation)
                    if operation.rules().is_none()
                        && operation.jvp_rule() == jvp_rule
                        && operation.has_vjp_rule() == has_vjp_rule,
            ));
        }

        // A carrier with an attached backward rule keeps its tangent interface and follows its region into the native
        // carrier variant.
        let tangent_type = Type::Array(ArrayType::scalar(DataType::F32));
        let carrier = CompositeTransposeOperation::from_backward_region(
            1,
            vec![tangent_type.clone()],
            vec![tangent_type.clone()],
        );
        let promoted: Operation<Constant> = ArrayIrOperation::<ArrayConstant>::CompositeTranspose(carrier).into();
        assert!(matches!(
            &promoted,
            Operation::CompositeTranspose(operation)
                if operation.rules().is_none()
                    && operation.leading_input_count() == 1
                    && operation.input_tangent_types() == std::slice::from_ref(&tangent_type)
                    && operation.output_tangent_types() == std::slice::from_ref(&tangent_type),
        ));

        let rematerialize = RematerializeOperation::<Type>::new(ResidualPolicyReference::new(DotsSavable))
            .with_optimization_barrier(RematerializationOptimizationBarrier::None)
            .with_differentiated(true);
        let promoted: Operation<Constant> =
            ArrayIrOperation::<ArrayConstant>::Rematerialize(rematerialize.clone()).into();
        assert!(matches!(&promoted, Operation::Rematerialize(operation) if operation == &rematerialize));
        assert_eq!(promoted.name(), rematerialize.name());
        assert_eq!(promoted.region_slots(), rematerialize.region_slots());
    }

    #[test]
    fn test_core_composite_retained_rule_promotions() {
        // Converted member calls and carriers keep their member definition, which specializes their rules for this
        // family on demand.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let member = CustomRuleRegistration::new(
            CustomRuleDefinition::<ArrayConstant, ArrayOperation<ArrayConstant>>::new("member"),
        );
        let promoted: Operation<Constant> =
            ArrayOperation::<ArrayConstant>::Composite(CompositeOperation::new(member.reference())).into();
        assert!(matches!(
            &promoted,
            Operation::LiftedComposite(operation)
                if operation.rules() == Some(&LiftedCustomRules::Member(member.reference())),
        ));
        assert_eq!(promoted.to_string(), "composite [name=\"member\"]");
        let promoted: Operation<Constant> =
            ArrayOperation::<ArrayConstant>::CompositeTranspose(CompositeTransposeOperation::new(
                member.reference(),
                0,
                vec![scalar_type.clone()],
                vec![scalar_type.clone()],
            ))
            .into();
        assert!(matches!(&promoted, Operation::LiftedCompositeTranspose(_)));

        // Calls registered in the array IR family itself cannot follow their program, because constants of this family
        // cannot represent array IR values. Their primal still executes, but their explicit rules are unavailable.
        let array_ir = CustomRuleRegistration::new(
            CustomRuleDefinition::<ArrayIrValue<ArrayConstant>, ArrayIrOperation<ArrayConstant>>::new("array_ir")
                .with_jvp(|primals, tangents| Ok((primals.to_vec(), tangents.to_vec()))),
        );
        let promoted: Operation<Constant> =
            ArrayIrOperation::<ArrayConstant>::Composite(CompositeOperation::new(array_ir.reference())).into();
        assert!(matches!(
            &promoted,
            Operation::LiftedComposite(operation)
                if matches!(operation.rules(), Some(LiftedCustomRules::Unavailable(_))),
        ));
        let mut primal = ProgramBuilder::new();
        let input = primal.add_input(scalar_type.clone().into());
        let primal = primal
            .build::<Vec<Constant>, Vec<Constant>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::new();
        let primal = builder.import_program(primal);
        let input = builder.add_input(scalar_type.into());
        let output = builder.add_instruction(promoted, vec![primal], vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap_err().to_string(),
            "`composite` `array_ir` was converted from a family whose values its current family cannot \
             represent, so its derivative rules are unavailable",
        );
    }

    #[test]
    fn test_rematerialize_recomputes_known_jit_calls_in_its_body() {
        // A rematerialized body squares its input inside a `jit_call` operation and squares the result of the call
        // outside of it. Linearizing it with the default policy, which saves nothing, saves only the input. Its pure
        // primal call, whose inputs are all known, keeps its boundary, and the residual body replays that call to
        // recompute the multiplication.
        let scalar_type = Type::from(ArrayType::scalar(DataType::F64));
        let callee = {
            let mut builder = ProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![squared], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let function = rematerialize(move |x: Tracer| -> Result<Tracer, ProgramError> {
            let context = x.context().clone();
            let called = context.bind(Operation::JitCall(JitCallOperation::new(0)), vec![callee.clone()], &[x])?;
            let mut outputs = context.bind(MulOperation::new(), Vec::new(), &[called[0].clone(), called[0].clone()])?;
            Ok(outputs.remove(0))
        })
        .with_policy(NothingSavable);
        let (_, program) = DomainTracingContext::<Session>::trace(|x: Tracer| function.call(x), scalar_type).unwrap();
        let linearization = program.into_flat_program().linearize().unwrap();
        assert_eq!(
            format!("{}\n{}", linearization.primal(), linearization.tangent()),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = jit_call %0 [
                    callee={
                        lambda %0:f64[] .
                        let %1:f64[] = mul %0 %0
                        in (%1)
                    },
                ]
                    %2:f64[] = mul %1 %1
                in (%2, %0)
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = rematerialize [differentiated=true] %0 %1 [
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = jit_call %1 [
                            callee={
                                lambda %0:f64[] .
                                let %1:f64[] = mul %0 %0
                                in (%1)
                            },
                        ]
                            %3:f64[] = jit_call %0 %1 [
                                callee={
                                    lambda %0:f64[], %1:f64[] .
                                    let %2:f64[] = mul %1 %0
                                        %3:f64[] = mul %1 %0
                                        %4:f64[] = add %2 %3
                                    in (%4)
                                },
                            ]
                            %4:f64[] = mul %2 %3
                            %5:f64[] = mul %2 %3
                            %6:f64[] = add %4 %5
                        in (%6)
                    },
                ]
                in (%2)"},
        );
    }

    #[test]
    fn test_rematerialize_rejects_captures_registered_in_its_body() {
        // This guards an unowned capture reference rather than a legitimate closed-over array, which enters the body as
        // a session constant through `ThroughSession` instead. The rematerialized body is traced through a fresh root
        // context whose capture table is discarded, so a body that registers a capture is rejected at trace time
        // rather than leaving a reference that a later scope owning captures would resolve against an unrelated value.
        // Refer to `TracingContext::trace_with_named_axes` for the full rationale.
        let scalar_type = Type::from(ArrayType::scalar(DataType::F32));
        let captured_value = Constant::Captured(CaptureReference::new(0, scalar_type.clone()));
        let function = rematerialize(move |x: Tracer| -> Result<Tracer, ProgramError> {
            let context = x.context().clone();
            let reference = context.capture(captured_value.clone())?;
            let captured = StagingContext::constant(&context, reference);
            let mut outputs = context.bind(AddOperation::new(), Vec::new(), &[x, captured])?;
            Ok(outputs.remove(0))
        });
        let root = DomainTracingContext::<Session>::new();
        let input = root.input(scalar_type);
        let result = function.call(input);
        assert!(matches!(result, Err(ProgramError::DiscardedCaptures { count: 1 })));
    }

    #[test]
    fn test_dynamic_disconnected_pullback_omits_structural_zero_cotangent() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent)]));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::new();
        builder.add_input(dynamic_type.into());
        let scalar = builder.add_input(scalar_type.into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![scalar], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        // A disconnected cotangent remains structural and spends no extent residual or executable output.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        assert!(linearization.primal().instructions().is_empty());
        let pullback = linearization.pullback().unwrap();
        assert_eq!(pullback.cotangents(), &[TransposedProgramCotangent::Zero, TransposedProgramCotangent::Output]);
        assert_eq!(
            pullback.program().to_string(),
            indoc! {"
            lambda %0:f64[] .
            in (%0)
        "}
            .trim_end()
        );
    }

    #[test]
    fn test_residual_zero_provider_materializes_dynamic_zero_from_array_source() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let tangent_type = primal_type.tangent().unwrap();
        let context = TracingContext::<Constant, Operation>::new();
        let primal = context.input(Type::Array(primal_type));

        // The shared array IR blanket implementation captures the named extent even when the source's element
        // representation differs from the dynamic tangent zero's. The XLA family uses the same identity-directed rule
        // as the core family.
        let zero = Operation::<Constant>::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(Type::Array(tangent_type.clone())),
            std::slice::from_ref(&primal),
        )
        .unwrap();
        assert_eq!(zero.r#type().as_ref(), &Type::Array(tangent_type));

        let builder = context.builder().borrow();
        let [dimension_size, zero] = builder.instructions() else {
            panic!("expected one dimension-size instruction followed by one dynamic zero instruction");
        };
        assert!(matches!(dimension_size.operation(), Operation::DimensionSize(operation) if operation.axis() == 0));
        assert_eq!(dimension_size.inputs(), &[primal.atom_id().unwrap()]);
        assert!(matches!(zero.operation(), Operation::Zero(_)));
        assert_eq!(zero.inputs(), &[dimension_size.outputs()[0]]);
    }

    #[test]
    fn test_residual_zero_provider_assembles_dynamic_zero_from_several_sources() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(7)).unwrap());
        let zero_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns.clone())]),
        );
        let context = TracingContext::<Constant, Operation>::new();
        let sources = [
            context.input(Type::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])))),
            context.input(Type::Array(ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(rows)])))),
            context.input(Type::Dimension(DimensionType::from(columns))),
        ];

        // No single source has the zero's type, so its shape is assembled per declared residual. The statically shaped
        // candidate names neither identity and is skipped without staging anything, the dynamic array supplies the row
        // extent through a `dimension_size` read, and the first-class dimension supplies the column extent as itself.
        // The shared array IR implementation emits XLA operations directly for the selected sources.
        let zero = Operation::<Constant>::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(Type::Array(zero_type.clone())),
            &sources,
        )
        .unwrap();
        assert_eq!(zero.r#type().as_ref(), &Type::Array(zero_type));

        let builder = context.builder().borrow();
        let [dimension_size, zero] = builder.instructions() else {
            panic!("expected one dimension-size instruction followed by one dynamic zero instruction");
        };
        assert!(matches!(dimension_size.operation(), Operation::DimensionSize(operation) if operation.axis() == 0));
        assert_eq!(dimension_size.inputs(), &[sources[1].atom_id().unwrap()]);
        assert!(matches!(zero.operation(), Operation::Zero(_)));
        assert_eq!(zero.inputs(), &[dimension_size.outputs()[0], sources[2].atom_id().unwrap()]);
    }

    #[test]
    fn test_gateway_region_program_imports_with_fresh_alpha_equivalent_identities() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::from(extent.clone());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let scalar_type = ArrayType::scalar(DataType::F32);

        let branch = || {
            let mut builder = ProgramBuilder::new();
            let extent = builder.add_input(extent_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let output = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };

        let integer_type = ArrayType::scalar(DataType::I32);
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let mut source_builder = ProgramBuilder::new();
        let true_region = source_builder.import_region(branch().entry_region_ref());
        let false_region = source_builder.import_region(branch().entry_region_ref());
        let integer = source_builder.add_input(integer_type.clone().into());
        let predicate = source_builder.add_input(predicate_type.clone().into());
        let scalar = source_builder.add_input(scalar_type.clone().into());
        let gateway = source_builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![integer], None)
            .unwrap()[0];
        let output = source_builder
            .add_instruction(
                Operation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, gateway, scalar],
                None,
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(source.output_types(), vec![Type::Array(output_type)]);
        assert!(source.type_identity_signature().input_identities().is_empty());
        assert_eq!(
            source.type_identity_signature().internal_identities(),
            std::slice::from_ref(extent_type.variable().unwrap())
        );

        let mut destination = ProgramBuilder::new();
        let integer = destination.add_input(integer_type.into());
        let predicate = destination.add_input(predicate_type.into());
        let scalar = destination.add_input(scalar_type.into());
        let first = destination.splice_program(&source, &[integer, predicate, scalar]).unwrap()[0];
        let second = destination.splice_program(&source, &[integer, predicate, scalar]).unwrap()[0];

        let [first_gateway, first_condition, second_gateway, second_condition] = destination.instructions() else {
            panic!("expected two imported gateway-condition pairs");
        };
        assert_eq!(first_condition.inputs(), &[predicate, first_gateway.outputs()[0], scalar]);
        assert_eq!(second_condition.inputs(), &[predicate, second_gateway.outputs()[0], scalar]);
        assert_eq!(first_condition.outputs(), &[first]);
        assert_eq!(second_condition.outputs(), &[second]);
        assert_eq!(first_condition.regions().len(), 2);
        assert_eq!(second_condition.regions().len(), 2);

        let first_type = destination.atoms()[first_gateway.outputs()[0].index()].r#type().into_owned();
        let second_type = destination.atoms()[second_gateway.outputs()[0].index()].r#type().into_owned();
        let first_dimension = <&DimensionType>::try_from(&first_type).unwrap();
        let second_dimension = <&DimensionType>::try_from(&second_type).unwrap();
        assert_ne!(first_dimension.variable(), second_dimension.variable());
        assert_eq!(first_dimension.bounds(), second_dimension.bounds());
        for (condition, dimension) in [(first_condition, first_dimension), (second_condition, second_dimension)] {
            for region in condition.regions() {
                let interface = destination.region_ref(*region).unwrap().interface();
                assert_eq!(interface.input_types()[0], Type::Dimension(dimension.clone()));
                let output = <&ArrayType>::try_from(&interface.output_types()[0]).unwrap();
                assert_eq!(output.shape().dimensions(), &[dimension.dimension().clone()],);
            }
        }

        let destination = destination
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![first, second],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let signature = destination.type_identity_signature();
        assert!(signature.input_identities().is_empty());
        assert_eq!(
            signature.internal_identities(),
            &[first_dimension.variable().unwrap().clone(), second_dimension.variable().unwrap().clone()],
        );
    }

    #[test]
    fn test_scan_reference_discharge_widens_carries_and_preserves_scan_metadata() {
        let reference_type = ReferenceType::new(vector_type());
        let mut body_builder = CoreProgramBuilder::<Constant, Operation>::new();
        body_builder.add_input(ArrayType::scalar(DataType::I64).into());
        let body_carry = body_builder.add_input(vector_type().into());
        let reference =
            body_builder.add_constant(Constant::Captured(CaptureReference::new(0, reference_type.clone().into())));
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![body_carry, value], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let mut builder = CoreProgramBuilder::<Constant, Operation>::new();
        let body = builder.import_region(body.entry_region_ref());
        builder.add_input(reference_type.into());
        let carry = builder.add_input(vector_type().into());
        let operation = ScanOperation::new(1, 3).with_reverse(true).with_unroll(3).unwrap();
        let outputs =
            builder.add_instruction(Operation::Scan(operation), vec![body], vec![carry], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let targets = program.reference_discharge_targets(1).unwrap();
        assert_eq!(targets, vec![ReferenceDischargeTarget::External(ReferenceSource::Capture { index: 0 })],);

        // The partial discharge entry point that handles captures resolves the body's capture constant even when its
        // allocation is preserved. The scan retains its reference access and gains a carry of reference type that binds
        // the rebuilt body's capture without turning it into state.
        let preserved = program.clone().partially_discharge_references_in_capture_lifted_program(1, &[]).unwrap();
        assert!(preserved.external_reference_bindings().is_empty());
        assert!(preserved.program().entry_region_ref().contains_references_in_closure());
        let Operation::Scan(scan) = preserved.program().entry_region_ref().instructions()[0].operation() else {
            panic!("expected a partially discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 2);

        // Selecting the captured allocation through the partial entry point agrees exactly with full discharge that
        // handles captures. Threading its state widens the carry list while preserving the scan's direction and unroll
        // factor.
        let selected = ReferenceDischargeResult::try_from(
            program
                .clone()
                .partially_discharge_references_in_capture_lifted_program(1, targets.as_slice())
                .unwrap(),
        )
        .unwrap();
        let discharged = program.discharge_references_in_capture_lifted_program(1).unwrap();
        assert_eq!(selected.program().to_string(), discharged.program().to_string());
        assert_eq!(selected.external_reference_bindings(), discharged.external_reference_bindings());
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Capture { index: 0 });
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        let Operation::Scan(scan) = discharged.program().entry_region_ref().instructions()[0].operation() else {
            panic!("expected a discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 2);
        assert_eq!(scan.unroll(), 3);
        assert!(scan.reverse());
    }

    #[test]
    fn test_rematerialization_policies_are_available_for_the_operation_family() {
        // The built-in rematerialization policies, including the offloading ones, classify candidates of the
        // `Type` universe of `Operation`, whose producers they recognize through the payload projection that
        // the derive generates. This is a compile-time capability check, because the assertions below fail to compile
        // if a policy is not available for the universe.
        fn assert_policy<P: ResidualPolicy<Type>>(_policy: P) {}
        assert_policy(NothingSavable);
        assert_policy(EverythingSavable);
        assert_policy(DotsSavable);
        assert_policy(DotsWithNoBatchDimensionsSavable);
        assert_policy(SaveOnlyTheseNames::new(["u"]));
        assert_policy(SaveAndOffloadOnlyTheseNames::new(["u"], ["v"], Memory::Host { pinned: true }).unwrap());
        assert_policy(OffloadDotsWithNoBatchDimensions::new(Memory::Host { pinned: true }));
        assert_policy(SaveFromBothPolicies::new(DotsSavable, SaveOnlyTheseNames::new(["u"])));
    }

    #[test]
    fn test_normalize_elementwise_operand_in_plain_programs() {
        let program = elementwise_normalization_program();
        let stablehlo = to_mlir_module_for_plain_program(&program, "main").unwrap();
        assert_elementwise_operands_are_normalized(&stablehlo);
    }

    #[test]
    fn test_normalize_elementwise_operand_in_traced_programs() {
        let plain_program = elementwise_normalization_program();
        let input_types = plain_program.input_types().into_owned();
        let output_types = plain_program.output_types().into_owned();
        let program = plain_program.into_unprojected::<Constant, Operation>().unwrap();
        let stablehlo =
            to_mlir_module_for_program(&program, &[], &input_types, &output_types, "main", None, None).unwrap();
        assert_elementwise_operands_are_normalized(&stablehlo);
    }

    #[test]
    fn test_normalize_elementwise_operand_zero_sized_operands() {
        let mut builder = PlainProgramBuilder::new();
        let scalar = builder.add_input(ArrayType::scalar(DataType::F32));
        let empty = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0)])));
        let output = builder.add_instruction(DivOperation::new(), Vec::new(), vec![scalar, empty], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(
                vec![output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        assert_eq!(
            to_mlir_module_for_plain_program(&program, "main").unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<f32>, %arg1: tensor<0xf64>) -> tensor<0xf64> {
                    %0 = stablehlo.convert %arg0 : (tensor<f32>) -> tensor<f64>
                    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<0xf64>
                    %2 = stablehlo.divide %1, %arg1 : tensor<0xf64>
                    return %2 : tensor<0xf64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_normalize_elementwise_operand_rejects_incompatible_operands() {
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let token_type = context.stable_hlo_token_type().unwrap();
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let vector_type = ArrayType::new_static(DataType::F32, [3]);
        let block = context.block(&[
            (token_type.as_ref(), location),
            (lower_tensor_type(&matrix_type, &context, location).unwrap().as_ref(), location),
        ]);
        let token = block.as_ref().argument(0).unwrap().as_ref();
        let matrix = block.as_ref().argument(1).unwrap().as_ref();
        assert!(matches!(
            normalize_elementwise_operand(token, &vector_type, &mut block.as_ref(), &context, location),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == "normalize_elementwise_operand"
                    && reason == "elementwise operand has non-tensor MLIR type `!stablehlo.token`",
        ));
        assert!(matches!(
            normalize_elementwise_operand(matrix, &vector_type, &mut block.as_ref(), &context, location),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == "normalize_elementwise_operand"
                    && reason == "cannot broadcast rank-2 elementwise operand to rank-1 output",
        ));
    }

    #[test]
    fn test_elementwise_lowering_matches_reference_backend() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let left_values = [0.5f32, 1.25, 2.0, 3.75];
        let right_values = [4.0f32, 0.25, -1.5, 2.5];
        let left = replicated_array(&session, &mesh, DataType::F32, &[4], &values_to_bytes(&left_values));
        let right = replicated_array(&session, &mesh, DataType::F32, &[4], &values_to_bytes(&right_values));
        let reference_left = CoreArray::vector(left_values.to_vec()).unwrap();
        let reference_right = CoreArray::vector(right_values.to_vec()).unwrap();

        // Each of these operations lowers to a single StableHLO operation, so agreement with the reference backend
        // pins the operation that the lowering selects. Both backends compute in `f32` and may round transcendental
        // functions differently in the last unit of precision, so results are compared within a relative tolerance.
        let cases = [
            ("abs", right.abs().unwrap(), reference_right.abs().unwrap()),
            ("sign", right.sign().unwrap(), reference_right.sign().unwrap()),
            ("sin", left.sin().unwrap(), reference_left.sin().unwrap()),
            ("cos", left.cos().unwrap(), reference_left.cos().unwrap()),
            ("tanh", left.tanh().unwrap(), reference_left.tanh().unwrap()),
            ("exp", left.exp().unwrap(), reference_left.exp().unwrap()),
            ("log", left.log().unwrap(), reference_left.log().unwrap()),
            ("sqrt", left.sqrt().unwrap(), reference_left.sqrt().unwrap()),
            ("rsqrt", left.rsqrt().unwrap(), reference_left.rsqrt().unwrap()),
            ("atan2", left.atan2(&right).unwrap(), reference_left.atan2(&reference_right).unwrap()),
            ("pow", left.pow(&right).unwrap(), reference_left.pow(&reference_right).unwrap()),
            ("rem", left.rem(&right).unwrap(), reference_left.rem(&reference_right).unwrap()),
        ];
        for (name, output, reference) in cases {
            let values = output.copy_to_host().unwrap().elements::<f32>().unwrap();
            let expected = reference.elements::<f32>().unwrap();
            assert_eq!(values.len(), expected.len(), "`{name}`");
            for (&value, &expected) in values.iter().zip(&expected) {
                assert!(
                    (value - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                    "`{name}` computed {value} but the reference backend computed {expected}",
                );
            }
        }
    }

    #[test]
    fn test_lower_accuracy() {
        // The default accuracy keeps the default StableHLO result accuracy, while other accuracies request theirs.
        assert_eq!(
            lowered_unary_module(ArrayOperation::Sin(SinOperation::new()), DataType::F32, vec![2]).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.sine %arg0 : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "#},
        );
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Tanh(TanhOperation::new().with_accuracy(Accuracy::Highest)),
                DataType::F32,
                vec![2],
            )
            .unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.tanh %arg0 {result_accuracy = #stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "#},
        );
        let tolerance = Tolerance::new(1e-6, 0.0, 2).unwrap();
        assert_eq!(
            lowered_unary_module(
                ArrayOperation::Exp(ExpOperation::new().with_accuracy(Accuracy::Tolerance(tolerance)),),
                DataType::F32,
                vec![2],
            )
            .unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
                    %0 = stablehlo.exponential %arg0 {result_accuracy = #stablehlo.result_accuracy<atol = 9.9999999999999995E-7, ulps = 2, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : tensor<2xf32>
                    return %0 : tensor<2xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lower_accuracy_matches_reference_backend() {
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let values = [0.5f32, -1.0, 2.0];
        let input = replicated_array(&session, &mesh, DataType::F32, &[values.len()], &values_to_bytes(&values));
        let reference = CoreArray::vector(values.to_vec()).unwrap();
        let expected_sine = reference.sin().unwrap().elements::<f32>().unwrap();
        let expected_tanh = reference.tanh().unwrap().elements::<f32>().unwrap();

        // XLA compiles every requested result accuracy. Each one only bounds the error of the implementation XLA
        // selects, so results are compared with the reference within that bound.
        let tolerance = Accuracy::Tolerance(Tolerance::new(1e-6, 0.0, 0).unwrap());
        for accuracy in [Accuracy::Highest, tolerance] {
            let sine = input.sin_with_accuracy(accuracy).unwrap().copy_to_host().unwrap().elements::<f32>().unwrap();
            assert_eq!(sine.len(), expected_sine.len());
            for (sine, expected) in sine.iter().zip(&expected_sine) {
                assert!((sine - expected).abs() <= 1e-6, "{accuracy:?}: {sine} vs {expected}");
            }
            let tanh = input.tanh_with_accuracy(accuracy).unwrap().copy_to_host().unwrap().elements::<f32>().unwrap();
            assert_eq!(tanh.len(), expected_tanh.len());
            for (tanh, expected) in tanh.iter().zip(&expected_tanh) {
                assert!((tanh - expected).abs() <= 1e-6, "{accuracy:?}: {tanh} vs {expected}");
            }
        }
    }

    /// Operation of [`UnreducedRow`], which [`lower_unreduced_application`] applies to values with and without
    /// reduction state.
    #[derive(Clone, Copy, Debug)]
    enum UnreducedOperation {
        Neg,
        Add,
        Sub,
        Mul,
        Div,
        Max,
        Min,
        Clamp,
        Abs,
        Exp,
        ConvertToF64,
        Reduce(ReductionKind),
        ArgMax,
        Reshape,
        Transpose,
        Slice,
        Reverse,
        Concatenate,
        Select,
        Pad,
        CumulativeSum,
        CumulativeMax,
        Compare,
        Not,
    }

    /// Inputs of [`UnreducedRow`], each over the mesh axis `x` of size two. Values "with the kind" are unreduced over
    /// `x` with the kind of the row, and the other values are replicated, with equal contributions of both devices.
    #[derive(Clone, Copy, Debug)]
    enum UnreducedOperands {
        /// One `f32` vector with the kind.
        Vector,

        /// Two `f32` vectors with the kind.
        Vectors,

        /// Three `f32` vectors with the kind.
        ThreeVectors,

        /// An `f32` vector and an `f32` scalar, both with the kind.
        VectorAndScalar,

        /// An `f32` vector and an `f64` vector, both with the kind.
        VectorAndF64Vector,

        /// An `f32` vector with the kind and an `f32` vector that is reduced over `x`.
        VectorAndReduced,

        /// An `f32` vector with the kind and a replicated `f32` vector.
        VectorAndReplicated,

        /// A replicated Boolean vector and two `f32` vectors with the kind.
        ConditionAndVectors,

        /// A Boolean vector with the kind.
        BooleanVector,
    }

    impl UnreducedOperands {
        /// Returns these inputs, unreduced with `kind`.
        fn inputs(self, kind: UnreducedKind) -> Vec<UnreducedInput> {
            let vector = || UnreducedInput::new(DataType::F32, &[4], Some(kind), [&[1f32, 5.0, -0.0, 0.5], &[3f32, -1.0, 4.0, 0.0]]);
            let other = || UnreducedInput::new(DataType::F32, &[4], Some(kind), [&[2f32, -3.0, 0.0, 7.0], &[0.5f32, 6.0, -2.0, 1.0]]);
            let replicated = [2f32, 4.0, -1.0, 0.5];
            match self {
                Self::Vector => vec![vector()],
                Self::Vectors => vec![vector(), other()],
                Self::ThreeVectors => vec![
                    vector(),
                    other(),
                    UnreducedInput::new(DataType::F32, &[4], Some(kind), [&[4f32, 0.0, 2.0, -1.0], &[1f32, 2.0, 3.0, 4.0]]),
                ],
                Self::VectorAndScalar => {
                    vec![vector(), UnreducedInput::new(DataType::F32, &[], Some(kind), [&[1.5f32], &[-2f32]])]
                }
                Self::VectorAndF64Vector => vec![
                    vector(),
                    UnreducedInput::new(DataType::F64, &[4], Some(kind), [&[2f64, -3.0, 0.0, 7.0], &[0.5f64, 6.0, -2.0, 1.0]]),
                ],
                Self::VectorAndReduced => {
                    vec![vector(), UnreducedInput::new(DataType::F32, &[4], None, [&replicated, &replicated]).reduced()]
                }
                Self::VectorAndReplicated => {
                    vec![vector(), UnreducedInput::new(DataType::F32, &[4], None, [&replicated, &replicated])]
                }
                Self::ConditionAndVectors => vec![
                    UnreducedInput::new(DataType::Boolean, &[4], None, [&[true, false, false, true]; 2]),
                    vector(),
                    other(),
                ],
                Self::BooleanVector => vec![UnreducedInput::new(
                    DataType::Boolean,
                    &[4],
                    Some(kind),
                    [&[true, false, true, false], &[false, false, true, true]],
                )],
            }
        }
    }

    /// Row of the tests of the lowering of operations over unreduced values.
    #[derive(Clone, Copy, Debug)]
    struct UnreducedRow(UnreducedOperation, UnreducedOperands);

    impl UnreducedApplication for UnreducedRow {
        fn apply<V: ArrayOperations>(&self, inputs: &[V]) -> Result<V, ProgramError> {
            match self.0 {
                UnreducedOperation::Neg => ryft_core::Neg::neg(&inputs[0]),
                UnreducedOperation::Add => ryft_core::Add::add(&inputs[0], &inputs[1]),
                UnreducedOperation::Sub => ryft_core::Sub::sub(&inputs[0], &inputs[1]),
                UnreducedOperation::Mul => ryft_core::Mul::mul(&inputs[0], &inputs[1]),
                UnreducedOperation::Div => ryft_core::Div::div(&inputs[0], &inputs[1]),
                UnreducedOperation::Max => inputs[0].max(&inputs[1]),
                UnreducedOperation::Min => inputs[0].min(&inputs[1]),
                UnreducedOperation::Clamp => inputs[0].clamp(&inputs[1], &inputs[2]),
                UnreducedOperation::Abs => inputs[0].abs(),
                UnreducedOperation::Exp => inputs[0].exp(),
                UnreducedOperation::ConvertToF64 => inputs[0].convert_element_type(DataType::F64),
                UnreducedOperation::Reduce(kind) => inputs[0].reduce(&[0], kind),
                UnreducedOperation::ArgMax => inputs[0].argmax(0),
                UnreducedOperation::Reshape => inputs[0].reshape([2, 2]),
                UnreducedOperation::Transpose => inputs[0].reshape([2, 2])?.transpose(vec![1, 0]),
                UnreducedOperation::Slice => inputs[0].slice(&[1], &[3], &[1]),
                UnreducedOperation::Reverse => inputs[0].reverse([0]),
                UnreducedOperation::Concatenate => <V as Concatenate>::concatenate([&inputs[0], &inputs[1]], 0),
                UnreducedOperation::Select => inputs[0].select(&inputs[1], &inputs[2]),
                UnreducedOperation::Pad => inputs[0].pad(&inputs[1], &[1], &[2], &[0]),
                UnreducedOperation::CumulativeSum => inputs[0].cumulative_sum(0),
                UnreducedOperation::CumulativeMax => inputs[0].cumulative_max(0),
                UnreducedOperation::Compare => inputs[0].compare(&inputs[1], ComparisonDirection::LessThan),
                UnreducedOperation::Not => inputs[0].not(),
            }
        }
    }

    #[test]
    fn test_normalize_binary_elementwise_operands_lower_unreduced_values() {
        // A scalar operand is broadcast to the shape of the vector, and an `f32` operand is converted to `f64`, before
        // the operation. Both are computed by each device on its own contribution, so they are unreduced like the
        // operands of the operation.
        let mut modules = Vec::new();
        for (operation, operands, kind) in [
            (UnreducedOperation::Max, UnreducedOperands::VectorAndScalar, UnreducedKind::Max),
            (UnreducedOperation::Min, UnreducedOperands::VectorAndScalar, UnreducedKind::Min),
            (UnreducedOperation::Add, UnreducedOperands::VectorAndScalar, UnreducedKind::Sum),
            (UnreducedOperation::Add, UnreducedOperands::VectorAndF64Vector, UnreducedKind::Sum),
        ] {
            let lowering =
                lower_unreduced_application(&UnreducedRow(operation, operands), &operands.inputs(kind)).unwrap();
            assert_eq!(lowering.outcomes, lowering.expected_outcomes, "{operation:?} of {operands:?} with {kind}");
            modules.push(lowering.module);
        }
        assert_eq!(modules.join("\n"), "");
    }

    #[test]
    fn test_operations_lower_unreduced_values() {
        // Every operation that accepts unreduced inputs, with every kind of pending reduction, at the top level of a
        // compiled function, in the body of a `scan`, and in a called function, executes on each device what the same
        // operation computes for the contribution of that device. The core type rules reject the other combinations.
        let rows = [
            UnreducedRow(UnreducedOperation::Neg, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Add, UnreducedOperands::Vectors),
            UnreducedRow(UnreducedOperation::Add, UnreducedOperands::VectorAndScalar),
            UnreducedRow(UnreducedOperation::Add, UnreducedOperands::VectorAndF64Vector),
            UnreducedRow(UnreducedOperation::Add, UnreducedOperands::VectorAndReplicated),
            UnreducedRow(UnreducedOperation::Sub, UnreducedOperands::Vectors),
            UnreducedRow(UnreducedOperation::Mul, UnreducedOperands::VectorAndReduced),
            UnreducedRow(UnreducedOperation::Mul, UnreducedOperands::Vectors),
            UnreducedRow(UnreducedOperation::Div, UnreducedOperands::VectorAndReduced),
            UnreducedRow(UnreducedOperation::Max, UnreducedOperands::Vectors),
            UnreducedRow(UnreducedOperation::Max, UnreducedOperands::VectorAndScalar),
            UnreducedRow(UnreducedOperation::Max, UnreducedOperands::VectorAndF64Vector),
            UnreducedRow(UnreducedOperation::Min, UnreducedOperands::Vectors),
            UnreducedRow(UnreducedOperation::Min, UnreducedOperands::VectorAndScalar),
            UnreducedRow(UnreducedOperation::Min, UnreducedOperands::VectorAndF64Vector),
            UnreducedRow(UnreducedOperation::Clamp, UnreducedOperands::ThreeVectors),
            UnreducedRow(UnreducedOperation::Abs, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Exp, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::ConvertToF64, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Reduce(ReductionKind::Sum), UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Reduce(ReductionKind::Mean), UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Reduce(ReductionKind::Max), UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Reduce(ReductionKind::Min), UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Reduce(ReductionKind::Product), UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::ArgMax, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Reshape, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Transpose, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Slice, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Reverse, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Concatenate, UnreducedOperands::VectorAndReplicated),
            UnreducedRow(UnreducedOperation::Select, UnreducedOperands::ConditionAndVectors),
            UnreducedRow(UnreducedOperation::Pad, UnreducedOperands::VectorAndScalar),
            UnreducedRow(UnreducedOperation::CumulativeSum, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::CumulativeMax, UnreducedOperands::Vector),
            UnreducedRow(UnreducedOperation::Compare, UnreducedOperands::Vectors),
            UnreducedRow(UnreducedOperation::Not, UnreducedOperands::BooleanVector),
        ];
        let mut outcomes = Vec::new();
        for row in rows {
            for kind in [UnreducedKind::Sum, UnreducedKind::Max, UnreducedKind::Min] {
                let outcome = match lower_unreduced_application(&row, &row.1.inputs(kind)) {
                    Ok(lowering) if lowering.outcomes == lowering.expected_outcomes => "lowered".to_string(),
                    Ok(lowering) => format!("{}\n  expected {}", lowering.outcomes, lowering.expected_outcomes),
                    Err(error) => error.lines().next().unwrap_or_default().to_string(),
                };
                outcomes.push(format!("{:?} of {:?} with {kind}: {outcome}", row.0, row.1));
            }
        }
        assert_eq!(outcomes.join("\n"), "");
    }
}
