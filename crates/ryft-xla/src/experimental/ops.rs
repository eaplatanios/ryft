use std::borrow::Cow;
use std::fmt::Display;
use std::marker::PhantomData;
use std::sync::Arc;

use ryft_core::macros::check_count;
use ryft_core::operations::attention::{DotProductAttentionBackwardOperation, DotProductAttentionOperation};
use ryft_core::operations::collectives::{
    AllGatherOperation, AllToAllOperation, ParallelPermuteOperation, ParallelSumScatterOperation,
    RaggedAllToAllOperation,
};
use ryft_core::operations::complex::{ComplexOperation, ConjugateOperation, ImaginaryOperation, RealOperation};
use ryft_core::operations::custom_call::CustomCallOperation;
use ryft_core::operations::random::RngBitGeneratorOperation;
use ryft_core::operations::sort::SortOperation;
use ryft_core::tracing_v2::rematerialization::RematerializeOperation;
use ryft_core::{
    AbsOperation, AddOperation, AndOperation, Array as ReferenceArray, ArrayBatch, ArrayBatching, ArrayIrOperation,
    ArrayIrType, ArrayOperation, ArrayReferenceViewOperation, ArrayType, Atan2Operation, AxisIndexOperation, BatchAxis,
    BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    BroadcastOperation, CalleeRegionDriver, CaptureConstant, CaptureReference, CeilOperation, CompareOperation,
    CompiledCallOperation, ConcatenateOperation, Concretizable, ConditionOperation, ConstantOperation, Context,
    ConvertElementTypeOperation, CosOperation, CumulativeLogSumExpOperation, CumulativeMaxOperation,
    CumulativeMinOperation, CumulativeProductOperation, CumulativeSumOperation, CustomJvpOperation, CustomVjpOperation,
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    Dimension, DimensionAddOperation, DimensionDivFloorOperation, DimensionFromScalarOperation, DimensionMaxOperation,
    DimensionMinOperation, DimensionMulOperation, DimensionOperation, DimensionPowOperation, DimensionRemOperation,
    DimensionRequirementOperation, DimensionSaturatingSubOperation, DimensionSizeOperation, DimensionSubOperation,
    DimensionToScalarOperation, DimensionType, DimensionValue, DivOperation, DotOperation, DynamicBroadcastOperation,
    DynamicReshapeOperation, DynamicShapeSliceOperation, DynamicSliceOperation, DynamicUpdateSliceOperation,
    EagerContext, ErfOperation, ExpOperation, FloorOperation, GatherOperation, IotaOperation, LinearCallOperation,
    Log1pOperation, LogAddExpOperation, LogOperation, LogSumExpOperation, LogisticOperation, MaxOperation, MaybeZero,
    MinOperation, MulOperation, NegOperation, NotOperation, OneLikeOperation, OneOperation, Operation,
    OperationFormatter, OrOperation, OutputRegionProvenance, PadOperation, ParallelReduceOperation, Parameter,
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation, PowOperation, PrintOperation, Program, ProgramBatchingOutputAxesPolicy,
    ProgramBuilder, ProgramError, ProjectedValue, RaggedDotOperation, ReduceOperation, ReferenceAddUpdateOperation,
    ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy, ReferenceDischargeValue,
    ReferenceDischargeableOperation, ReferenceFreezeOperation, ReferenceIndexOperation, ReferenceNewOperation,
    ReferenceReadOperation, ReferenceSliceOperation, ReferenceSwapOperation, ReferenceWriteOperation, RegionInterface,
    RegionSlot, RemOperation, ReshapeOperation, ReshardOperation, ResidualZeroProvider, RoundOperation, RsqrtOperation,
    ScaledDotOperation, ScanOperation, ScatterOperation, SelectOperation, ShardingConstraintOperation, SignOperation,
    SinOperation, SliceOperation, SqrtOperation, StagingContext, StopGradientOperation, SubOperation, TagOperation,
    TanhOperation, Tracer, TracingContext, TransferToMemoryOperation, TransposableOperation, TransposeOperation,
    TranspositionDriver, Type, TypeError, TypeIdentityRenaming, Typed, UpdateSliceOperation, Value, ValueProjection,
    WhileOperation, XorOperation, Zero, ZeroLikeOperation, ZeroOperation, ZeroOperationProvider,
    discharge_positional_region_operation,
};
use ryft_macros::Parameter;

use crate::experimental::operations::ShardMapOperation;

/// Lifetime-free reference to an array member captured by an XLA program.
pub type XlaArrayConstant = CaptureReference<ArrayType>;

/// Constant payload stored in the atom table of a staged XLA [`Program`].
///
/// Staged XLA programs keep two kinds of constants apart, and this sum is the staged counterpart of the eager
/// [`ArrayIrValue`](ryft_core::ArrayIrValue) universe:
///
///   - **Captured runtime values:** array buffers and external references stay in the surrounding
///     compiled function's capture table, while the program stores only a lifetime-free [`CaptureReference`] carrying
///     its index and structural [`ArrayIrType`]. This keeps runtime storage out of literal IR payloads. A captured
///     reference remains metadata only and must be validated and discharged before ordinary XLA lowering.
///   - **Immediate first-class dimensions:** a [`DimensionValue`] is a checked host integer, so embedding it costs
///     nothing and it lowers to a scalar `stablehlo.constant`. Unlike a capture reference it also stays usable inside
///     a nested region — most importantly a `shard_map` manual computation, which owns no capture table of its own —
///     and that is what lets shape arithmetic such as the explicit result extents of the collectives be staged
///     against a manual region's shard-local values.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum XlaConstant {
    /// Reference to a value held in the surrounding compiled function's capture table.
    Captured(CaptureReference<ArrayIrType>),

    /// Immediate checked host-side first-class dimension extent.
    Dimension(DimensionValue),
}

impl Display for XlaConstant {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Captured(value) => Display::fmt(value, formatter),
            Self::Dimension(value) => Display::fmt(value, formatter),
        }
    }
}

impl Typed for XlaConstant {
    type Type = ArrayIrType;

    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayIrType> {
        match self {
            Self::Captured(value) => value.r#type(),
            Self::Dimension(value) => Cow::Owned(ArrayIrType::Dimension(value.r#type().into_owned())),
        }
    }
}

impl Value for XlaConstant {
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self>;

    #[inline]
    fn dispatch_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    #[inline]
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayIrType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Captured(value) => Ok(Self::Captured(value.rename_type_identities(renaming)?)),
            Self::Dimension(value) => Ok(Self::Dimension(value.rename_type_identities(renaming)?)),
        }
    }
}

impl CaptureConstant for XlaConstant {
    #[inline]
    fn capture_index(&self) -> Option<usize> {
        match self {
            Self::Captured(value) => Some(value.index()),
            Self::Dimension(_) => None,
        }
    }

    #[inline]
    fn map_capture_index<F: FnOnce(usize) -> usize>(&self, map: F) -> Self {
        match self {
            Self::Captured(value) => Self::Captured(value.map_capture_index(map)),
            Self::Dimension(value) => Self::Dimension(value.clone()),
        }
    }
}

impl ValueProjection<ArrayType> for XlaConstant {
    type Projected = XlaArrayConstant;
    type ProjectedRef<'v> = ProjectedValue<ArrayType, &'v Self>;

    #[inline]
    fn from_projected(value: XlaArrayConstant) -> Self {
        Self::Captured(ValueProjection::<ArrayType>::from_projected(value))
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
    where
        ArrayType: 'v,
    {
        Ok(ProjectedValue::new(self, <&ArrayType>::try_from(self.r#type().as_ref())?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<XlaArrayConstant, TypeError> {
        match self {
            Self::Captured(value) => value.into_projected(),
            Self::Dimension(_) => Err(TypeError::invalid("expected array type but got dimension type")),
        }
    }
}

impl ValueProjection<DimensionType> for XlaConstant {
    type Projected = DimensionValue;
    type ProjectedRef<'v> = &'v DimensionValue;

    #[inline]
    fn from_projected(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v DimensionValue, TypeError>
    where
        DimensionType: 'v,
    {
        match self {
            Self::Captured(_) => Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
            Self::Dimension(value) => Ok(value),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<DimensionValue, TypeError> {
        match self {
            Self::Captured(_) => Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
            Self::Dimension(value) => Ok(value),
        }
    }
}

impl From<CaptureReference<ArrayIrType>> for XlaConstant {
    #[inline]
    fn from(value: CaptureReference<ArrayIrType>) -> Self {
        Self::Captured(value)
    }
}

impl From<DimensionValue> for XlaConstant {
    #[inline]
    fn from(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }
}

/// A captured constant is a reference into a side table rather than the concrete predicate value itself, and an
/// immediate dimension is an extent rather than Boolean array data, and so neither variant can be read back as a
/// concrete predicate. Control-flow staging must keep predicates in the IR or add a transform-specific rule instead.
impl Concretizable<bool> for XlaConstant {
    #[inline]
    fn concretize(&self) -> Result<bool, ProgramError> {
        Err(ProgramError::Concretization {
            message: format!("cannot extract a concrete boolean from the staged xla constant `{self}`"),
        })
    }
}

/// Ordinary staged-operation universe owned by the XLA backend.
///
/// This enum flattens the core array operation payloads directly into the backend-owned operation family. Higher-order
/// instructions attach their nested computations as regions of the containing XLA program, so those regions can
/// contain backend-specific operations such as [`jit_call`](JitCallOperation) and
/// [`shard_map`](ShardMapOperation).
///
/// The [`Operation`] contract, the interpretation and partial-evaluation dispatchers, the forward-mode and
/// transposition dispatchers, the member operation-family projections, and the payload conversions are all derived.
/// The variant classes mirror [`ArrayIrOperation`] exactly for the payloads the two families share, so both
/// dispatchers report identical semantics for every shared member and mixed payload. Only the backend-owned surfaces
/// stay handwritten: the normalizing conversions that select between a member and a mixed carrier, the zero and
/// residual-zero providers, the canonical core-operation view used by lowering, and the MLIR lowering dispatch.
#[derive(Clone, Debug, ryft_macros::Operation)]
#[ryft(crate = "ryft_core", type = ArrayIrType, constant = Constant)]
#[ryft(members(ArrayType, structural(DimensionType)))]
#[ryft(dispatch(discharge, differentiation, transposition))]
pub enum XlaOperation<Constant = XlaConstant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    /// Mixed zero constructor whose explicit first-class dimension operands provide its dynamic result extents.
    /// This variant cannot be represented by the homogeneous array member because its signature crosses member
    /// kinds: it consumes dimension members and produces an array member.
    #[ryft(mixed(structural), skip_from)]
    Zero(ZeroOperation<ArrayType>),

    /// Mixed one constructor with explicit dynamic-extent operands.
    #[ryft(mixed(structural), skip_from)]
    DynamicOne(OneOperation<ArrayType>),

    /// Mixed iota constructor with explicit dynamic-extent operands.
    #[ryft(mixed(structural), skip_from)]
    DynamicIota(IotaOperation<ArrayType>),

    /// Homogeneous array operation. Member zero constructors are promoted to their mixed composite carrier when an
    /// array operation is lifted into this family.
    #[ryft(projected(ArrayType), skip_from)]
    Array(ArrayOperation<Constant::Projected>),

    /// Homogeneous first-class-dimension operation.
    #[ryft(projected(DimensionType, structural))]
    Dimension(DimensionOperation<DimensionValue>),

    /// Mixed comparison of two dimensions producing Boolean array data.
    Compare(CompareOperation<ArrayIrType>),

    /// Reads an array extent as a first-class dimension.
    DimensionSize(DimensionSizeOperation),

    /// Unresolved whole-array reference allocation retained until reference discharge.
    ReferenceNew(ReferenceNewOperation<ArrayType, ArrayIrType>),

    /// Unresolved axis-removing reference view retained until reference discharge.
    ReferenceIndex(ReferenceIndexOperation),

    /// Unresolved static slice reference view retained until reference discharge.
    ReferenceSlice(ReferenceSliceOperation),

    /// Unresolved read from a root reference or derived view retained until reference discharge.
    ReferenceRead(ReferenceReadOperation<ArrayType, ArrayIrType>),

    /// Unresolved write-only replacement through a root reference or derived view retained until reference discharge.
    ReferenceWrite(ReferenceWriteOperation<ArrayType, ArrayIrType>),

    /// Unresolved replacement through a root reference or derived view retained until reference discharge.
    ReferenceSwap(ReferenceSwapOperation<ArrayType, ArrayIrType>),

    /// Unresolved additive update through a root reference or derived view retained until reference discharge.
    ReferenceAddUpdate(ReferenceAddUpdateOperation<ArrayType, ArrayIrType>),

    /// Unresolved consuming whole-array reference freeze retained until reference discharge.
    ReferenceFreeze(ReferenceFreezeOperation<ArrayType, ArrayIrType>),

    /// Converts scalar array data into a checked first-class dimension.
    DimensionFromScalar(DimensionFromScalarOperation),

    /// Converts a first-class dimension into scalar array data.
    DimensionToScalar(DimensionToScalarOperation),

    /// Reshapes an array using explicit dimension operands.
    Reshape(DynamicReshapeOperation),

    /// Broadcasts an array using explicit dimension operands.
    Broadcast(DynamicBroadcastOperation),

    /// Concatenates arrays with an explicit result extent.
    Concatenate(ConcatenateOperation<ArrayIrType>),

    /// Calls a foreign kernel with explicit dynamic result extents.
    CustomCall(CustomCallOperation<ArrayIrType>),

    /// Pads an array with explicit result extents.
    Pad(PadOperation<ArrayIrType>),

    /// Slices an array using first-class start and size dimensions.
    DynamicShapeSlice(DynamicShapeSliceOperation),

    /// Generates random bits with explicit dynamic result extents.
    RngBitGenerator(RngBitGeneratorOperation<ArrayIrType>),

    /// Gathers values with one explicit extent per result axis.
    #[ryft(mixed)]
    AllGather(AllGatherOperation),

    /// Scatters values with one explicit extent per result axis.
    #[ryft(mixed)]
    ParallelSumScatter(ParallelSumScatterOperation),

    /// Exchanges values with one explicit extent per result axis.
    #[ryft(mixed)]
    AllToAll(AllToAllOperation),

    /// Exchanges variable-length leading-axis slices through an XLA FFI custom call.
    #[ryft(mixed)]
    RaggedAllToAll(RaggedAllToAllOperation),

    /// Backend-owned condition whose attached branch regions can contain XLA operations.
    Condition(ConditionOperation<Constant>),

    /// Backend-owned loop whose attached condition and body regions can contain XLA operations.
    While(WhileOperation<ArrayIrType>),

    /// Backend-owned scan whose attached body region can contain XLA operations.
    Scan(ScanOperation<Constant>),

    /// Backend-owned custom JVP call whose attached regions can contain XLA operations.
    CustomJvp(CustomJvpOperation<ArrayIrType>),

    /// Backend-owned custom VJP call whose attached regions can contain XLA operations.
    CustomVjp(CustomVjpOperation<ArrayIrType>),

    /// Differentiation-owned call to an explicitly transposable linear map with ordinary trailing residual
    /// operands. This variant carries both carrier forms: the forward-and-transpose form lowers by inlining its
    /// forward region, while the reverse-only transpose-only form (attached transpose region only) cannot be lowered
    /// and reports the canonical reverse-only diagnostic.
    LinearCall(LinearCallOperation<ArrayIrType>),

    /// Backend-owned rematerialized call whose attached regions can contain XLA operations.
    Rematerialize(RematerializeOperation<ArrayIrType>),

    /// Call to a flat jitted XLA sub-program.
    JitCall(JitCallOperation<ArrayIrType>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<Constant>>),
}

impl<Constant> ArrayReferenceViewOperation for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn from_reference_reshape(operation: ReshapeOperation) -> Self {
        Self::Array(ArrayOperation::Reshape(operation))
    }

    fn from_reference_slice(operation: SliceOperation) -> Self {
        Self::Array(ArrayOperation::Slice(operation))
    }

    fn from_reference_update_slice(operation: UpdateSliceOperation) -> Self {
        Self::Array(ArrayOperation::UpdateSlice(operation))
    }
}

impl<Constant> From<ArrayOperation<Constant::Projected>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: ArrayOperation<Constant::Projected>) -> Self {
        // Delegating to the composite family's conversion keeps constructor normalization and member control-flow
        // promotion identical across both families: `Condition`, `While`, and `Scan` must become composite carriers
        // because the projected `Array` variant cannot own composite regions.
        ArrayIrOperation::<Constant::Projected>::from(operation).into()
    }
}

impl<Constant> From<ArrayIrOperation<Constant::Projected>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn from(operation: ArrayIrOperation<Constant::Projected>) -> Self {
        match operation {
            ArrayIrOperation::Zero(operation) => Self::Zero(operation),
            ArrayIrOperation::DynamicOne(operation) => Self::DynamicOne(operation),
            ArrayIrOperation::DynamicIota(operation) => Self::DynamicIota(operation),
            ArrayIrOperation::Array(operation) => Self::Array(operation),
            ArrayIrOperation::Dimension(operation) => Self::Dimension(operation),
            ArrayIrOperation::Compare(operation) => Self::Compare(operation),
            ArrayIrOperation::DimensionSize(operation) => Self::DimensionSize(operation),
            ArrayIrOperation::ReferenceNew(operation) => Self::ReferenceNew(operation),
            ArrayIrOperation::ReferenceIndex(operation) => Self::ReferenceIndex(operation),
            ArrayIrOperation::ReferenceSlice(operation) => Self::ReferenceSlice(operation),
            ArrayIrOperation::ReferenceRead(operation) => Self::ReferenceRead(operation),
            ArrayIrOperation::ReferenceWrite(operation) => Self::ReferenceWrite(operation),
            ArrayIrOperation::ReferenceSwap(operation) => Self::ReferenceSwap(operation),
            ArrayIrOperation::ReferenceAddUpdate(operation) => Self::ReferenceAddUpdate(operation),
            ArrayIrOperation::ReferenceFreeze(operation) => Self::ReferenceFreeze(operation),
            ArrayIrOperation::DimensionFromScalar(operation) => Self::DimensionFromScalar(operation),
            ArrayIrOperation::DimensionToScalar(operation) => Self::DimensionToScalar(operation),
            ArrayIrOperation::Reshape(operation) => Self::Reshape(operation),
            ArrayIrOperation::Broadcast(operation) => Self::Broadcast(operation),
            ArrayIrOperation::Concatenate(operation) => Self::Concatenate(operation),
            ArrayIrOperation::CustomCall(operation) => Self::CustomCall(operation),
            ArrayIrOperation::Pad(operation) => Self::Pad(operation),
            ArrayIrOperation::DynamicShapeSlice(operation) => Self::DynamicShapeSlice(operation),
            ArrayIrOperation::RngBitGenerator(operation) => Self::RngBitGenerator(operation),
            ArrayIrOperation::AllGather(operation) => Self::AllGather(operation),
            ArrayIrOperation::ParallelSumScatter(operation) => Self::ParallelSumScatter(operation),
            ArrayIrOperation::AllToAll(operation) => Self::AllToAll(operation),
            ArrayIrOperation::RaggedAllToAll(operation) => Self::RaggedAllToAll(operation),
            ArrayIrOperation::Condition(_) => Self::Condition(ConditionOperation::new()),
            ArrayIrOperation::While(operation) => Self::While(operation),
            ArrayIrOperation::Scan(operation) => {
                let captures = operation
                    .captures()
                    .iter()
                    .cloned()
                    .map(|capture| match capture {
                        ryft_core::arrays::ArrayIrValue::Array(capture) => Constant::from_projected(capture),
                        ryft_core::arrays::ArrayIrValue::Dimension(_)
                        | ryft_core::arrays::ArrayIrValue::Reference(_) => {
                            // Scan captures are validated as stacked arrays during `infer_output_types`
                            // (`validate_scan_capture`); this conversion is infallible, so a dimension or reference
                            // capture reaching it means a scan was converted before that validation ran.
                            unreachable!(
                                "scan captures must be validated as stacked arrays before converting the scan; \
                                dimension and reference captures are rejected by scan capture validation"
                            )
                        }
                    })
                    .collect();
                Self::Scan(
                    ScanOperation::<Constant>::new(operation.carry_count(), operation.length())
                        .with_reverse(operation.reverse())
                        .with_unroll(operation.unroll())
                        .unwrap()
                        .with_captures(captures),
                )
            }
            ArrayIrOperation::CustomJvp(operation) => Self::CustomJvp(operation),
            ArrayIrOperation::CustomVjp(operation) => Self::CustomVjp(operation),
            ArrayIrOperation::LinearCall(operation) => Self::LinearCall(operation),
            ArrayIrOperation::Rematerialize(operation) => Self::Rematerialize(operation),
        }
    }
}

impl<Constant> From<DimensionRequirementOperation> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: DimensionRequirementOperation) -> Self {
        Self::Dimension(DimensionOperation::Requirement(operation))
    }
}

// Dimension constants additionally lift directly, so that generic staging code (e.g., `ExactShape::dimensions`)
// can bound only `From<ConstantOperation<DimensionValue>>` without naming this family's dimension member.
impl<Constant> From<ConstantOperation<DimensionValue>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: ConstantOperation<DimensionValue>) -> Self {
        Self::Dimension(DimensionOperation::Constant(operation))
    }
}

macro_rules! impl_composite_operation_conversion {
    // Generates the composite-operation conversions the derive cannot generate directly: the canonical constructors
    // whose lift selects between the homogeneous member carrier and the mixed dimension-operand carrier, and the
    // homogeneous-array payload forms of mixed operations whose canonical composite payload is type-promoted.
    ($($operation:ty),+ $(,)?) => {
        $(
            impl<Constant> From<$operation> for XlaOperation<Constant>
            where
                Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            {
                #[inline]
                fn from(operation: $operation) -> Self {
                    ArrayIrOperation::<Constant::Projected>::from(operation).into()
                }
            }
        )+
    };
}

// Every first-class dimension arithmetic operation lifts directly, mirroring the composite family, so that generic
// staging code (e.g., the `DimensionArithmetic` capability) can bound `From<DimensionMulOperation>`-style conversions
// without naming this family's dimension member.
impl_composite_operation_conversion!(
    DimensionAddOperation,
    DimensionSubOperation,
    DimensionSaturatingSubOperation,
    DimensionMulOperation,
    DimensionPowOperation,
    DimensionDivFloorOperation,
    DimensionRemOperation,
    DimensionMinOperation,
    DimensionMaxOperation,
);

impl_composite_operation_conversion!(
    ZeroLikeOperation<ArrayIrType>,
    ZeroOperation<ArrayType>,
    OneOperation<ArrayType>,
    IotaOperation<ArrayType>,
    ConcatenateOperation<ArrayType>,
    CustomCallOperation<ArrayType>,
    PadOperation<ArrayType>,
);

macro_rules! impl_array_operation_conversion {
    // Generates homogeneous array-operation conversions through the canonical projected member family.
    ($($operation:ty),+ $(,)?) => {
        $(
            impl<Constant> From<$operation> for XlaOperation<Constant>
            where
                Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            {
                #[inline]
                fn from(operation: $operation) -> Self {
                    ArrayOperation::<Constant::Projected>::from(operation).into()
                }
            }
        )+
    };
}

impl_array_operation_conversion!(
    ZeroLikeOperation<ArrayType>,
    OneLikeOperation<ArrayType>,
    ConstantOperation<ReferenceArray>,
    AbsOperation<ArrayType>,
    NegOperation<ArrayType>,
    SubOperation<ArrayType>,
    MulOperation<ArrayType>,
    DivOperation<ArrayType>,
    SinOperation<ArrayType>,
    CosOperation<ArrayType>,
    Atan2Operation<ArrayType>,
    ExpOperation<ArrayType>,
    LogOperation<ArrayType>,
    Log1pOperation<ArrayType>,
    LogAddExpOperation<ArrayType>,
    SqrtOperation<ArrayType>,
    RsqrtOperation<ArrayType>,
    TanhOperation<ArrayType>,
    LogisticOperation<ArrayType>,
    ErfOperation<ArrayType>,
    PowOperation<ArrayType>,
    SignOperation<ArrayType>,
    FloorOperation<ArrayType>,
    CeilOperation<ArrayType>,
    RoundOperation<ArrayType>,
    MaxOperation<ArrayType>,
    MinOperation<ArrayType>,
    RemOperation<ArrayType>,
    NotOperation<ArrayType>,
    AndOperation<ArrayType>,
    OrOperation<ArrayType>,
    XorOperation<ArrayType>,
    ComplexOperation<ArrayType>,
    ConjugateOperation<ArrayType>,
    RealOperation<ArrayType>,
    ImaginaryOperation<ArrayType>,
    DotOperation,
    RaggedDotOperation,
    ScaledDotOperation,
    DotProductAttentionOperation,
    DotProductAttentionBackwardOperation,
    ReduceOperation,
    LogSumExpOperation,
    CumulativeSumOperation,
    CumulativeProductOperation,
    CumulativeMaxOperation,
    CumulativeMinOperation,
    CumulativeLogSumExpOperation,
    SortOperation,
    ParallelReduceOperation,
    ParallelPermuteOperation,
    AxisIndexOperation,
    TransposeOperation,
    ReshapeOperation,
    BroadcastOperation,
    GatherOperation,
    ScatterOperation,
    SliceOperation,
    UpdateSliceOperation,
    DynamicSliceOperation,
    DynamicUpdateSliceOperation,
    SelectOperation<ArrayType>,
    ConvertElementTypeOperation<ArrayType>,
    TransferToMemoryOperation,
    ReshardOperation,
    ShardingConstraintOperation,
    StopGradientOperation<ArrayType>,
    TagOperation<ArrayType>,
    PrintOperation<ArrayType>,
);

impl<Constant> ZeroOperationProvider<ArrayIrType> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn zero_operation(r#type: ArrayIrType) -> Result<Self, ProgramError> {
        Ok(ArrayIrOperation::<Constant::Projected>::zero_operation(r#type)?.into())
    }
}

impl<Constant> From<AddOperation<ArrayIrType>> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: AddOperation<ArrayIrType>) -> Self {
        ArrayIrOperation::<Constant::Projected>::from(operation).into()
    }
}

impl<Constant> ResidualZeroProvider<ArrayIrType> for XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn zero_residual_types(r#type: &ArrayIrType) -> Vec<ArrayIrType> {
        <ArrayIrOperation<Constant::Projected> as ResidualZeroProvider<ArrayIrType>>::zero_residual_types(r#type)
    }

    fn capture_zero_residuals<V: Value<Type = ArrayIrType>>(
        builder: &mut ProgramBuilder<V, Self>,
        source: ryft_core::AtomId,
        r#type: &ArrayIrType,
    ) -> Result<Vec<ryft_core::AtomId>, ProgramError> {
        ArrayIrOperation::<Constant::Projected>::capture_zero_residuals(builder, source, r#type)
    }

    fn capture_zero_residual_value<C: Context<Type = ArrayIrType, Operation = Self>>(
        context: &C,
        source: &C::Value,
        residual_type: &ArrayIrType,
    ) -> Result<Option<C::Value>, ProgramError> {
        ArrayIrOperation::<Constant::Projected>::capture_zero_residual_value(context, source, residual_type)
    }

    fn zero_operation_with_residuals<R: Clone>(
        r#type: ArrayIrType,
        residuals: &[R],
    ) -> Result<(Self, Vec<R>), ProgramError> {
        let (operation, operands) =
            ArrayIrOperation::<Constant::Projected>::zero_operation_with_residuals(r#type, residuals)?;
        Ok((operation.into(), operands))
    }
}

impl<Constant> XlaOperation<Constant>
where
    Constant: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    /// Returns the canonical core operation for a member or mixed primitive, or `None` for an XLA-owned
    /// higher-order operation.
    ///
    /// This conversion clones the payload, so it is reserved for methods that need the composite family's boundary
    /// projection or reconstruct an operation anyway (e.g., type inference, identity renaming, interpretation, and
    /// partial evaluation). Cheap per-instruction accessors dispatch to the borrowed payload directly instead.
    pub(crate) fn to_core_operation(&self) -> Option<ArrayIrOperation<Constant::Projected>> {
        Some(match self {
            Self::Zero(operation) => ArrayIrOperation::Zero(operation.clone()),
            Self::DynamicOne(operation) => ArrayIrOperation::DynamicOne(operation.clone()),
            Self::DynamicIota(operation) => ArrayIrOperation::DynamicIota(operation.clone()),
            Self::Array(operation) => ArrayIrOperation::Array(operation.clone()),
            Self::Dimension(operation) => ArrayIrOperation::Dimension(operation.clone()),
            Self::Compare(operation) => ArrayIrOperation::Compare(operation.clone()),
            Self::DimensionSize(operation) => ArrayIrOperation::DimensionSize(operation.clone()),
            Self::ReferenceNew(operation) => ArrayIrOperation::ReferenceNew(*operation),
            Self::ReferenceIndex(operation) => ArrayIrOperation::ReferenceIndex(*operation),
            Self::ReferenceSlice(operation) => ArrayIrOperation::ReferenceSlice(operation.clone()),
            Self::ReferenceRead(operation) => ArrayIrOperation::ReferenceRead(*operation),
            Self::ReferenceWrite(operation) => ArrayIrOperation::ReferenceWrite(*operation),
            Self::ReferenceSwap(operation) => ArrayIrOperation::ReferenceSwap(*operation),
            Self::ReferenceAddUpdate(operation) => ArrayIrOperation::ReferenceAddUpdate(*operation),
            Self::ReferenceFreeze(operation) => ArrayIrOperation::ReferenceFreeze(*operation),
            Self::DimensionFromScalar(operation) => ArrayIrOperation::DimensionFromScalar(operation.clone()),
            Self::DimensionToScalar(operation) => ArrayIrOperation::DimensionToScalar(*operation),
            Self::Reshape(operation) => ArrayIrOperation::Reshape(operation.clone()),
            Self::Broadcast(operation) => ArrayIrOperation::Broadcast(operation.clone()),
            Self::Concatenate(operation) => ArrayIrOperation::Concatenate(operation.clone()),
            Self::CustomCall(operation) => ArrayIrOperation::CustomCall(operation.clone()),
            Self::Pad(operation) => ArrayIrOperation::Pad(operation.clone()),
            Self::DynamicShapeSlice(operation) => ArrayIrOperation::DynamicShapeSlice(operation.clone()),
            Self::RngBitGenerator(operation) => ArrayIrOperation::RngBitGenerator(operation.clone()),
            Self::AllGather(operation) => ArrayIrOperation::AllGather(operation.clone()),
            Self::ParallelSumScatter(operation) => ArrayIrOperation::ParallelSumScatter(operation.clone()),
            Self::AllToAll(operation) => ArrayIrOperation::AllToAll(operation.clone()),
            Self::RaggedAllToAll(operation) => ArrayIrOperation::RaggedAllToAll(operation.clone()),
            Self::Condition(_)
            | Self::While(_)
            | Self::Scan(_)
            | Self::CustomJvp(_)
            | Self::CustomVjp(_)
            | Self::LinearCall(_)
            | Self::Rematerialize(_)
            | Self::JitCall(_)
            | Self::ShardMap(_) => return None,
        })
    }
}

/// Staged XLA program specialized to the backend-owned XLA op universe.
pub type XlaProgram<Input, Output> = Program<XlaConstant, XlaOperation, Input, Output>;

/// Program builder specialized to the backend-owned XLA op universe.
pub type XlaProgramBuilder = ProgramBuilder<XlaConstant, XlaOperation>;

/// Flat XLA program over the backend-owned operation universe, used for materialized regions and shared callees.
pub type FlatXlaProgram = XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>;

/// Canonical operation name for [`JitCallOperation`].
pub const JIT_CALL_OPERATION_NAME: &str = "jit_call";

/// Staged call to a flat jitted XLA program. The callee program is not part of this payload: it is a shared
/// callee root [`Region`](ryft_core::Region) attached to the [`Instruction`](ryft_core::Instruction) applying the
/// operation (the single `["callee"]` slot), interned by [`Arc`] identity when the call is staged through the
/// [`BindingRegionDriver`](ryft_core::BindingRegionDriver) passed to [`Context::bind`], so repeated calls staged from
/// one function handle share one callee root and remain identity-comparable for call-site deduplication at lowering.
/// The `T` parameter fixes the callee boundary's type universe, allowing the reusable homogeneous-array batching form
/// and the executable composite array IR form to remain distinct payload types with one [`Operation`] contract each.
/// The retained `capture_count` names the callee's exact leading lifted-capture input prefix; it participates in
/// operation equality and callee-deduplication identity and scopes reference-capture resolution during analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JitCallOperation<T: Type> {
    /// Number of leading callee inputs that form its lifted lexical capture prefix.
    capture_count: usize,

    /// Type universe of the callee boundary.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> Copy for JitCallOperation<T> {}

impl<T: Type> JitCallOperation<T> {
    /// Creates a staged jitted-call operation. The flat callee program is supplied as a shared region attachment to
    /// [`Context::bind`], and `capture_count` identifies its exact leading lifted-capture prefix.
    #[inline]
    pub(crate) fn new(capture_count: usize) -> Self {
        Self { capture_count, marker: PhantomData }
    }

    /// Returns the exact number of leading callee inputs that form the lifted-capture prefix.
    #[inline]
    pub(crate) fn capture_count(&self) -> usize {
        self.capture_count
    }
}

impl CompiledCallOperation<XlaConstant> for XlaOperation {
    #[inline]
    fn compiled_call(capture_count: usize) -> Self {
        Self::JitCall(JitCallOperation::new(capture_count))
    }
}

/// Bridges canonical internal physical positions to the public batching declaration.
fn batch_axis_from_position(axis: Option<usize>) -> BatchAxis {
    BatchAxis::from_optional_position(axis)
}

/// Recovers a canonical physical position returned by the core program batching pass.
fn batch_axis_position(axis: &BatchAxis) -> Option<usize> {
    axis.axis()
        .map(|axis| usize::try_from(axis.value()).expect("program batching returns canonical nonnegative axes"))
}

fn ensure_call_input_types<T: Type>(
    operation_name: &'static str,
    expected_types: &[T],
    input_types: &[T],
) -> Result<(), TypeError> {
    if expected_types.len() != input_types.len() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` expected {} input(s) but got {}",
            expected_types.len(),
            input_types.len(),
        )));
    }
    for (index, (expected, actual)) in expected_types.iter().zip(input_types).enumerate() {
        if expected != actual {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` input #{index} expected {expected} but got {actual}",
            )));
        }
    }
    Ok(())
}

impl<T: Type> Operation for JitCallOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        JIT_CALL_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("callee")] }
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        if region_interfaces.len() != 1 {
            return Err(TypeError::invalid(format!(
                "{} expects 1 attached callee region but got {}",
                JIT_CALL_OPERATION_NAME,
                region_interfaces.len()
            )));
        }
        let callee_interface = &region_interfaces[0];
        ensure_call_input_types(self.name(), callee_interface.input_types(), input_types)?;
        Ok(callee_interface.output_types().to_vec())
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<usize> {
        (region_index == 0).then_some(input_index)
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    #[inline]
    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        (region_index == 0).then_some(self.capture_count)
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let operation = OperationFormatter::new(formatter, indentation, JIT_CALL_OPERATION_NAME)?;
        if self.capture_count == 0 {
            Ok(())
        } else {
            operation.bracketed(|operation| operation.field("capture_count", self.capture_count))
        }
    }
}

// A jitted call forwards its operands onto its callee's inputs one for one and reports the callee's outputs as its
// own, which is the positionally forwarding shape the shared structured rewrite serves with no leading operands. The
// callee's lifted capture prefix is unaffected: threaded state is appended after every declared operand, so the prefix
// keeps its positions.
impl<T, C, P> ReferenceDischargeableOperation<C, P> for JitCallOperation<T>
where
    T: Type,
    JitCallOperation<T>: Operation<Type = C::Type>,
    C: Context<Operation: From<JitCallOperation<T>>>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        discharge_positional_region_operation(self, context, driver, inputs, 0)
    }
}

/// Online partial-evaluation rule for a staged jitted call — ryft's analogue of JAX's call partial-evaluation
/// rules: it splits the callee against the caller's known-ness while preserving the `jit_call` boundary on both
/// sides.
///
/// The split fires only when some known call input does *not* [`resolve`](Context::resolve) to a program constant in
/// the known-side context — i.e., a genuine tracer into a live outer trace, the mixed-online case this
/// rule exists for. All-known, all-unknown, and constant-resolved calls defer to the default fold-or-residualize
/// behavior, which preserves the original boundary (and today's eager behavior) exactly.
///
/// When the split fires, the callee is split through the shared
/// [`PartitionedProgram`](ryft_core::partial::PartitionedProgram) machinery: the known side is bound into the
/// enclosing known-side context
/// wrapped in a fresh `jit_call` over the original known call inputs, and the residual side is emitted as the
/// residual `jit_call` over the surviving unknown call inputs plus the known-side call's residual-edge outputs.
impl<V, C> PartiallyEvaluatableOperation<C> for JitCallOperation<ArrayIrType>
where
    V: PartialEq
        + Value<Type = ArrayIrType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + CaptureConstant
        + Concretizable<bool>,
    C: Context<Type = ArrayIrType, Constant = V, Operation = XlaOperation<V>>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Split only a mixed call with at least one known-but-symbolic input; everything else keeps the default
        // fold-or-residualize behavior and therefore the original boundary.
        if !context.any_known_is_symbolic(inputs) || inputs.iter().all(PartialEvaluationValue::is_known) {
            return context.fold_or_residualize(
                XlaOperation::JitCall(*self),
                driver.regions().map(|region| region.to_program()).collect(),
                inputs,
            );
        }
        // Split the callee through the shared online boundary machinery, bind the known side into the enclosing
        // known-side context wrapped in a fresh `jit_call` over the original known call inputs, emit the residual
        // side as the residual `jit_call`, and reassemble the original output order.
        let callee = driver.region(0)?;
        // Partitioning does not remap the absolute indices of capture constants retained in attached regions, so a
        // split boundary would leave them naming a compacted or absent capture prefix — failing lowering at best and
        // silently aliasing an unrelated leading operand at worst. Preserve the original call boundary for any callee
        // whose closure still holds a capture constant.
        if callee.contains_atom_in_closure(|atom| {
            atom.as_constant().is_some_and(|constant| constant.capture_index().is_some())
        }) {
            return context.fold_or_residualize(XlaOperation::JitCall(*self), vec![callee.to_program()], inputs);
        }
        let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
        let partition = callee.partition(input_known.as_slice())?;
        // A trivial partition — one whose known program contains no instructions — hoists no work (its known side
        // can only forward known inputs as residual edges), so keep the original boundary and let the default
        // materialize those knowns directly as residual feeders.
        if partition.known_program().instructions().is_empty() {
            return context.fold_or_residualize(XlaOperation::JitCall(*self), vec![callee.to_program()], inputs);
        }
        // Known inputs keep callee source order, so the known-side callee's leading inputs are exactly the known
        // members of the original lifted-capture prefix; the residual callee's inputs are residual edges and unknown
        // inputs, which never form a capture prefix. The guard above already preserved the boundary of any callee
        // retaining attached-region capture constants, so neither derived callee can hold one.
        let known_capture_count = input_known.iter().take(self.capture_count()).filter(|known| **known).count();
        context.inline_partitioned_program(
            partition,
            inputs,
            |known_program| (XlaOperation::JitCall(JitCallOperation::new(known_capture_count)), vec![known_program]),
            |residual_program| (XlaOperation::JitCall(JitCallOperation::new(0)), vec![residual_program]),
        )
    }
}

/// Batching rule for [`JitCallOperation`]: the callee region is rebatched over the mapped input axes (via
/// [`BatchingDriver::batch_program`]) and the batched call is bound through `context.parent()` with the
/// batched callee re-attached. An eager
/// client-backed parent (e.g., [`XlaDomain`](crate::XlaDomain)) compiles and executes the batched call immediately, a
/// staging parent stages it into the enclosing trace, and a differentiation parent dispatches it through its own
/// `jit_call` JVP rule — which is what serves `vmap` nested inside `gradient`/`linearize` closures.
impl<C> BatchableOperation<C, ArrayBatching> for JitCallOperation<ArrayType>
where
    C: Context<Type = ArrayType>,
    C::Operation: From<JitCallOperation<ArrayType>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching>>(
        &self,
        context: &BatchingContext<C, ArrayBatching>,
        driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching>, BatchingError> {
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        // Rebatch the callee region over the mapped input axes when any input carries the batch axis; an
        // all-replicated call binds its original callee unchanged.
        //
        // Unlike the forward-mode and transpose rules below, this rebatching is deliberately not served from the
        // callee region's retained transform cache: the batched program depends on live `BatchingContext` state (the
        // batch extent, the axis name, and the surrounding nesting) whose complete key surface is not yet explicit,
        // so caching it against the callee region and its input axes alone would risk exactly the unsound key reuse
        // that cache is designed to avoid. The decided follow-up is to specify that key material when composite
        // batching lands behind the `CompiledXlaFunction::batch` stub, which has to define it anyway: batching then
        // joins the region transform cache behind a gate measurement, through a batching-policy hook returning
        // `Option<key>` where `None` keeps a policy uncached, so soundness stays opt-in per policy. See
        // `.tasks/plan_general_transform_caching_analysis.md`.
        let (batched_callee, output_axes) = match ArrayBatch::common_batch_size(inputs)? {
            Some(_) => {
                let input_batch_axes = inputs
                    .iter()
                    .map(|input| batch_axis_from_position(input.batch_axis_position()))
                    .collect::<Vec<_>>();
                let (batched_callee, output_axes) = driver
                    .batch_program(
                        context,
                        driver.region(0)?,
                        input_batch_axes.as_slice(),
                        ProgramBatchingOutputAxesPolicy::Natural,
                    )?
                    .into_parts();
                let output_axes = output_axes.iter().map(batch_axis_position).collect::<Vec<_>>();
                (batched_callee.into_simplified()?, output_axes)
            }
            None => {
                let callee = driver.region(0)?;
                let output_axes = vec![None; callee.output_types().len()];
                (callee.to_program(), output_axes)
            }
        };
        let outputs =
            context
                .parent()
                .bind(*self, CalleeRegionDriver::new(&[Arc::new(batched_callee)]), &physical_inputs)?;
        Ok(outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output, batch_axis_from_position(axis)))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

/// Capture-free forward-mode (JVP) rule for [`JitCallOperation`], binding a primal `jit_call` and a tangent
/// `jit_call` as ordinary XLA-enum operations through the active context: a staging context stages both calls over
/// its shared builder, while an eager context (e.g. a client-backed [`XlaDomain`](crate::XlaDomain)) compiles and
/// executes them immediately, which is what powers top-level `jvp` over concrete arrays.
///
/// This realizes the identity `jvp(jit(f)) = jit(jvp f)`: rather than capturing the primal inputs as residual factors
/// and staging a linear `jit_call`, the rule keeps the compilation boundary and threads every residual as a plain
/// primal operand edge between two `jit_call`s, so no symbolic capture is ever introduced. The enclosing
/// partial-evaluation split then discovers the residual operand edges structurally, exactly as it does for the
/// condition and rematerialize rules.
///
/// The rule linearizes the callee program capture-free through
/// [`Program::linearize`](ryft_core::Program::linearize), giving a primal sub-program
/// `inputs -> [outputs..., residuals...]` and a tangent sub-program
/// `[live_input_tangents..., residuals...] -> [live_output_tangents...]` together with the residual count. Tangents
/// for zero differential spaces are omitted from both compact boundaries. It then:
///
///   1. Wraps the primal sub-program in a fresh `jit_call` and stages it over the operand primals, recovering the
///      primal outputs followed by the residual values (program variables produced by the staged primal call).
///   2. Wraps the tangent sub-program in a fresh `jit_call` and stages it over the live operand tangents followed by
///      those residual values, recovering the live output tangents.
///   3. Pairs each primal output tracer with its tangent output tracer, restoring structural zeros for omitted
///      zero-space outputs, into a [`DifferentiationDual`].
///
/// The callee program is materialized from the instruction's callee region in the context's constant universe `V`
/// (concretely [`XlaConstant`] for staged XLA programs), so the split halves ride the fresh primal and tangent calls
/// as shared callee regions. Preserving both `jit_call` boundaries keeps the callee body out of the caller's program,
/// so forward mode over a jitted call stays compiled rather than inlined.
///
/// # Parameters
///
///   - `context`: Active evaluation or staging context used to bind the differentiated calls.
///   - `driver`: Call-scoped access to the attached callee region.
///   - `inputs`: Primal and tangent values for the call operands.
impl<C, V> DifferentiableOperation<C> for JitCallOperation<ArrayIrType>
where
    C: Context<Type = ArrayIrType, Constant = V, Operation = XlaOperation<V>> + Zero<C::Value>,
    V: PartialEq
        + Value<Type = ArrayIrType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + CaptureConstant
        + Concretizable<bool>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let callee = driver.region(0)?;
        let input_types = callee.input_types();
        let output_types = callee.output_types();
        let output_count = output_types.len();
        check_count!("input", inputs, input_types.len(), ProgramError);

        // Linearize the callee capture-free, through the callee region's retained transform cache so that a callee
        // shared by several outer programs is linearized once. The primal sub-program produces
        // `[outputs..., residuals...]` and the tangent sub-program consumes `[input_tangents..., residuals...]`; the
        // residual count is the number of trailing outputs of the primal sub-program beyond the original callee
        // outputs.
        let (primal_program, tangent_program, _) = callee.linearize_shared()?;

        // Wrap the primal sub-program in a fresh `jit_call` and bind it over the operand primals, recovering the
        // primal outputs followed by the residual values. The shared sub-program handles are attached directly, so
        // repeated binds of one derived callee intern by `Arc` identity instead of copying it again.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        // The primal sub-program preserves the callee's inputs in order, so the original lifted-capture prefix
        // survives verbatim and its length carries over.
        let primal_call = XlaOperation::JitCall(JitCallOperation::new(self.capture_count()));
        let mut primal_call_outputs =
            context.bind(primal_call, CalleeRegionDriver::new(&[primal_program]), &primal_operands)?;
        if primal_call_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "jit_call primal program produced {} outputs which is fewer than its {output_count} primal \
                 output(s)",
                primal_call_outputs.len(),
            ))
            .into());
        }
        let residuals = primal_call_outputs.split_off(output_count);
        let primal_outputs = primal_call_outputs;

        // Wrap the tangent sub-program in a fresh `jit_call` and bind it over only the live operand tangents followed
        // by the residual values. Zero-space tangents have no compact callee boundary slot.
        let input_tangent_types = input_types.iter().map(DifferentiableType::tangent).collect::<Result<Vec<_>, _>>()?;
        let mut tangent_operands = inputs
            .iter()
            .zip(input_tangent_types)
            .filter(|(_, tangent_type)| !tangent_type.is_zero_space())
            .map(|(input, _)| input.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_operands.extend(residuals);
        // The tangent callee's inputs are live input tangents followed by residuals — never a lifted-capture prefix.
        // Attached-region capture constants cannot appear in it because fresh-root region traces reject bodies that
        // register captures (`ProgramError::DiscardedCaptures`), and a stale reference that slipped in anyway would
        // fail lowering loudly against the tangent callee's empty capture prefix instead of aliasing an unrelated
        // value.
        let tangent_call = XlaOperation::JitCall(JitCallOperation::new(0));
        let tangent_outputs =
            context.bind(tangent_call, CalleeRegionDriver::new(&[tangent_program]), &tangent_operands)?;
        let output_tangent_types =
            output_types.iter().map(DifferentiableType::tangent).collect::<Result<Vec<_>, _>>()?;
        let tangent_output_count = output_tangent_types.iter().filter(|r#type| !r#type.is_zero_space()).count();
        check_count!("output", tangent_outputs, tangent_output_count, ProgramError);

        let mut tangent_outputs = tangent_outputs.into_iter();
        Ok(primal_outputs
            .into_iter()
            .zip(output_tangent_types)
            .map(|(primal, tangent_type)| {
                if tangent_type.is_zero_space() {
                    DifferentiationDual::new_with_zero_tangent(primal)
                } else {
                    DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
                }
            })
            .collect::<Result<Vec<_>, _>>()?)
    }
}

/// Returns a tracer for `cotangent`, materializing a structural zero through the canonical XLA representation when a
/// higher-order transpose call requires an ordinary SSA operand. Dynamic array zeros read their extent operands from
/// the known dimension inputs of the differentiated call; they are never reconstructed from type metadata.
pub(crate) fn materialize_transpose_cotangent<
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
>(
    context: &TracingContext<V, XlaOperation<V>>,
    cotangent: &MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>,
    output_type: &ArrayIrType,
    inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
) -> Result<Tracer<TracingContext<V, XlaOperation<V>>>, ProgramError> {
    if let MaybeZero::Value(cotangent) = cotangent {
        return Ok(cotangent.clone());
    }

    let (operation, operands) = match output_type {
        ArrayIrType::Array(array_type)
            if array_type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) =>
        {
            // The generic input-free provider cannot construct a dynamically shaped array zero. Resolve each dynamic
            // axis from the known dimension inputs and pass those extents through the mixed zero operation instead.
            let operands = array_type
                .shape()
                .dimensions()
                .iter()
                .filter_map(Dimension::variable)
                .map(|variable| {
                    inputs
                        .iter()
                        .filter_map(PartialValue::as_known)
                        .find(|input| {
                            matches!(
                                input.r#type().as_ref(),
                                ArrayIrType::Dimension(r#type) if r#type.variable() == variable
                            )
                        })
                        .cloned()
                        .ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "cannot materialize dynamic transpose cotangent of type {output_type} because no known \
                                 dimension input defines `{variable}`",
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            (XlaOperation::from(ZeroOperation::new(array_type.clone())), operands)
        }
        _ => (XlaOperation::zero_operation(output_type.clone())?, Vec::new()),
    };
    let mut outputs = context.stage_operation(operation, Vec::new(), operands.as_slice())?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Partition-aware transpose rule for a *primal* tangent [`JitCallOperation`], the jitted-call counterpart of
/// [`transpose_primal_condition`](ryft_core::operations::control_flow::transpose_primal_condition),
/// [`transpose_primal_scan`](ryft_core::operations::control_flow::transpose_primal_scan), and
/// [`LinearCallOperation`]'s transpose rule. It is used when the direct reverse transposes a tangent program in the
/// primal [`XlaOperation`] family rather than re-keying it into a linear operation family.
///
/// The forward ([`JitCallOperation::jvp`]) stages the tangent `jit_call` over the operand tangents followed
/// by the primal call's residual values, wrapping a callee program whose inputs match that operand signature
/// one-to-one and whose outputs are the output tangents. Each operand is therefore independently linear (an input
/// tangent the reverse must accumulate) or known (a residual value, or a captured-constant tangent the differentiated
/// inputs do not flow through), and the linear operands need not form a leading run: a captured compiled function
/// threads its captured prefix as known leading operands, so a known operand can precede the linear input tangents.
/// This rule:
///
///   1. Reads the runtime value of every known operand from `operand_values`, in callee-input order, to feed the
///      transposed callee's known inputs.
///   2. Transposes the callee program with [`TranspositionDriver::transpose_program`] under the same per-operand
///      linearity mask, so the callee's own linear and known inputs match the operands. The
///      transposed callee maps `[outputs..., known_input_values...]` to `[linear_input_cotangents...]`, in
///      callee-input order on each side.
///   3. Re-wraps the transposed callee in a fresh [`JitCallOperation`] and stages it over
///      `[outputs..., known_input_values...]`, preserving the compilation boundary so that both forward mode
///      over a jitted call (`jvp ∘ jit`) and reverse mode over it (`transpose ∘ jit`) stay compiled rather than
///      inlined.
///
/// The returned cotangents place the transposed call's outputs at the linear-operand positions and a structural
/// [`MaybeZero::Zero`] at the known positions, which carry no cotangent. The callee transposition happens through
/// [`TranspositionDriver::transpose_program`] in the same operation family, so it is value-level and introduces
/// no recursive transposition obligation on [`XlaOperation`].
///
/// # Parameters
///
///   - `operation`: Primal tangent `jit_call` staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `driver`: Instruction-scoped access to the attached callee region and its recursive transposition machinery.
///   - `inputs`: Per-operand [`PartialValue`] knowledge, mirroring the callee's inputs one-to-one. The
///     [`Unknown`](PartialValue::Unknown) entries are the input tangents; the [`Known`](PartialValue::Known) entries
///     carry the residual and captured-constant-tangent tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the tangent call's outputs.
pub fn transpose_primal_jit_call<
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    D: TranspositionDriver<V, XlaOperation<V>>,
>(
    _operation: &JitCallOperation<ArrayIrType>,
    context: &mut TracingContext<V, XlaOperation<V>>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, ProgramError> {
    // A jitted call with no live output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return inputs
            .iter()
            .map(|input| input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from))
            .collect();
    }

    // Each operand maps to one callee input, independently linear (an input tangent) or known (a residual value or a
    // captured-constant tangent). The linear operands need not lead: a captured compiled function threads its captured
    // prefix as known leading operands. The dispatch guarantees a `Known` operand carries its pullback value, so each
    // known tracer is read directly in callee-input order.
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let callee = driver.region(0)?;
    check_count!("input", operand_linear, callee.input_types().len(), ProgramError);
    let known_values = inputs
        .iter()
        .filter(|input| input.is_known())
        .map(|input| input.as_known().expect("dispatch guarantees a known operand carries its pullback value").clone())
        .collect::<Vec<_>>();

    // Transpose the callee with respect to its linear inputs, through the region's retained transform cache so that a
    // callee shared by several outer programs is transposed once per linearity mask. The transposed callee maps
    // `[outputs..., known_input_values...]` to `[linear_input_cotangents...]`, in callee-input order.
    let transposed_callee = driver.transpose_program(callee, operand_linear.as_slice())?;

    // Stage the output cotangents, materializing a typed zero for each structurally zero cotangent, then stage a fresh
    // `jit_call` over the transposed callee on `[outputs..., known_input_values...]`. Its outputs are the
    // linear-input cotangents.
    let output_types = callee.output_types();
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(output_types.len() + known_values.len());
    for (cotangent, output_type) in outputs.iter().zip(output_types.iter()) {
        let output_cotangent_type = output_type.cotangent()?;
        operands.push(materialize_transpose_cotangent(context, cotangent, &output_cotangent_type, inputs)?);
    }
    operands.extend(known_values);
    // The transposed callee's inputs are output cotangents followed by known input values — never a lifted-capture
    // prefix. Attached-region capture constants cannot appear in it because region bodies are traced through
    // fresh-root contexts.
    let transposed_call = XlaOperation::JitCall(JitCallOperation::new(0));
    let input_cotangents =
        context.bind(transposed_call, CalleeRegionDriver::new(&[transposed_callee]), operands.as_slice())?;
    let linear_count = operand_linear.iter().filter(|&&linear| linear).count();
    check_count!("output", input_cotangents, linear_count, ProgramError);

    // Reassemble one cotangent per operand: the known operands carry structural zeros, while the linear input tangents
    // receive the transposed call's outputs in callee-input order.
    let mut input_cotangents = input_cotangents.into_iter().map(MaybeZero::Value);
    let cotangents = operand_linear
        .iter()
        .zip(inputs)
        .map(|(&linear, input)| {
            if linear {
                Ok(input_cotangents.next().unwrap())
            } else {
                input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(cotangents)
}

/// Transpose rule for a primal tangent [`JitCallOperation`], forwarding to [`transpose_primal_jit_call`]. The callee
/// transposition happens on the concretely [`XlaConstant`]-keyed [`FlatXlaProgram`], so the recursion is resolved once
/// at definition time and instantiating this implementation introduces no recursive [`TransposableOperation`]
/// obligation on [`XlaOperation`].
impl<V> TransposableOperation<V, XlaOperation<V>> for JitCallOperation<ArrayIrType>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn transpose<D: TranspositionDriver<V, XlaOperation<V>>>(
        &self,
        context: &mut TracingContext<V, XlaOperation<V>>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, DifferentiationError> {
        transpose_primal_jit_call(self, context, driver, inputs, outputs).map_err(DifferentiationError::from)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayIrOperation, ArrayIrOperations, ArrayIrType, ArrayOperation, ArrayOperations, ArrayType,
        CaptureReference, CapturingContext, ConditionOperation, Context, CustomJvpOperation, CustomVjpOperation,
        DataType, DifferentiableType, DifferentiationError, Dimension, DimensionBounds, DimensionFromScalarOperation,
        DimensionType, DimensionValue, DimensionVariable, DomainTracingContext, DynamicBroadcastOperation, Effects,
        EmptyRegionDriver, LogicalMesh, MaybeZero, MeshAxis, MeshAxisType, MulOperation, Operation,
        OutputRegionProvenance, PartialValue, Placeholder, ProgramBuilder, ProgramError, ReferenceAddUpdateOperation,
        ReferenceDischarge, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceSource,
        ReferenceStateBinding, ReferenceSwapOperation, ReferenceType, ReferenceWriteOperation, RegionDriver,
        RegionInterface, RegionRef, RematerializeOperation, ResidualZeroProvider, ScanOperation, Shape, Sharding,
        ShardingDimension, StagingContext, Tracer, TracingContext, TranspositionDriver, TypeError,
        TypeIdentityRenaming, Typed, Value, ValueProjection, WhileOperation, ZeroOperation,
    };

    use crate::Array;
    use crate::experimental::domains::{XlaDomain, XlaTracer};
    use crate::experimental::shard_map::ShardMapTracer;

    use super::{
        CaptureConstant, JIT_CALL_OPERATION_NAME, JitCallOperation, XlaArrayConstant, XlaConstant, XlaOperation,
        XlaProgram, XlaProgramBuilder, materialize_transpose_cotangent, transpose_primal_jit_call,
    };

    /// Test-only driver that exposes one source callee and returns a predetermined transpose for it.
    struct TestTranspositionDriver {
        /// Source callee exposed to the JIT-call transpose rule.
        source: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,

        /// Predetermined transposed callee returned by the recursive request.
        transposed: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,
    }

    impl RegionDriver<XlaConstant, XlaOperation> for TestTranspositionDriver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, XlaConstant, XlaOperation>>
        where
            XlaConstant: 'r,
            XlaOperation: 'r,
        {
            std::iter::once(self.source.entry_region_ref())
        }
    }

    impl TranspositionDriver<XlaConstant, XlaOperation> for TestTranspositionDriver {
        fn transpose_program(
            &self,
            _region: RegionRef<'_, XlaConstant, XlaOperation>,
            _input_linearity: &[bool],
        ) -> Result<Arc<XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>>, DifferentiationError> {
            Ok(Arc::new(self.transposed.clone()))
        }
    }

    fn vector_type() -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)]))
    }

    #[test]
    fn test_jit_call_reference_provenance_is_positional() {
        // A jitted call forwards its operands to the callee positionally, so region provenance, capture counts, and
        // output provenance are all index-preserving for the single callee region and absent for any other region.
        let operation = JitCallOperation::<ArrayIrType>::new(2);
        assert_eq!(operation.input_region_provenance(0, 3), Some(3));
        assert_eq!(operation.input_region_provenance(1, 3), None);
        assert_eq!(operation.region_capture_input_count(0), Some(2));
        assert_eq!(operation.region_capture_input_count(1), None);
        assert_eq!(
            std::fmt::from_fn(|formatter| operation.render(formatter, 0)).to_string(),
            "jit_call [capture_count=2]",
        );
        assert_eq!(
            operation.output_region_provenance(2),
            vec![OutputRegionProvenance { region_index: 0, output_index: 2 }],
        );
    }

    #[test]
    fn test_jit_call_residual_candidates_classify_through_callee_provenance() {
        use std::cell::RefCell;
        use std::rc::Rc;

        use ryft_core::tracing_v2::{
            NoStorage, PolicyFn, RematerializationCandidate, RematerializationDecision, RematerializationRejection,
            rematerialize,
        };

        // `jit_call` reports positional output-region provenance, so a residual produced by a computed callee output
        // is classified through the callee to that output's own leaf producer instead of to the opaque `jit_call`
        // carrier. The rematerialized body squares its input inside the callee and squares the call's result outside
        // it, so transposing the outer `mul` needs the call's output as a residual. This is the `jit_call` counterpart
        // of the core loop pin in `test_while_residual_candidates_classify_through_loop_provenance`.
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F64));
        let callee = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(scalar_type.clone());
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![squared], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let names = Rc::new(RefCell::new(Vec::new()));
        let recorded = names.clone();
        let policy = PolicyFn::new(move |candidate: &RematerializationCandidate<'_, ArrayIrType, XlaOperation>| {
            recorded
                .borrow_mut()
                .extend(candidate.producers().iter().map(|producer| producer.operation().name().to_string()));
            Ok::<_, RematerializationRejection>(RematerializationDecision::<NoStorage>::Recompute)
        });
        let function = rematerialize::<XlaDomain<'static>, _, _, _>(move |x: XlaTracer<'static>| {
            let context = x.context().clone();
            let called = context.bind(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee.clone()], &[x])?;
            let mut outputs = context.bind(MulOperation::new(), Vec::new(), &[called[0].clone(), called[0].clone()])?;
            Ok(outputs.remove(0))
        })
        .with_policy(policy);
        let root = DomainTracingContext::<XlaDomain<'static>>::new();
        function.call(root.input(scalar_type)).unwrap();
        assert_eq!(names.borrow().clone(), vec!["mul".to_string()]);
    }

    #[test]
    fn test_xla_operation_values_satisfy_the_core_capability_bundles() {
        // Each bundle is satisfied exactly when every one of its member capabilities is, so instantiating these
        // functions is a compile-time assertion that the XLA value families expose the complete value-level surface
        // of the operation families they pair with.
        fn requires_array_operations<V: ArrayOperations>() {}
        fn requires_array_ir_operations<V: ArrayIrOperations>() {}

        requires_array_operations::<Array<'static>>();
        requires_array_operations::<ShardMapTracer>();
        requires_array_ir_operations::<XlaTracer<'static>>();
        requires_array_ir_operations::<Tracer<TracingContext<XlaConstant, XlaOperation>>>();
    }

    #[test]
    fn test_xla_constant() {
        let array_type = vector_type();
        let capture = CaptureReference::new(2, ArrayIrType::Array(array_type.clone()));
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let extent = DimensionValue::new(extent_type.clone(), 4).unwrap();

        // Both variants are reachable through the payload conversions, report their own member type, and render
        // exactly like the payload they wrap.
        let captured = XlaConstant::from(capture.clone());
        let immediate = XlaConstant::from(extent.clone());
        assert_eq!(captured, XlaConstant::Captured(capture));
        assert_eq!(immediate, XlaConstant::Dimension(extent.clone()));
        assert_eq!(captured.r#type().into_owned(), ArrayIrType::Array(array_type.clone()));
        assert_eq!(immediate.r#type().into_owned(), ArrayIrType::Dimension(extent_type));
        assert_eq!(captured.to_string(), "capture#2:f64[4]");
        assert_eq!(immediate.to_string(), extent.to_string());

        // Only the captured variant names a capture-table slot, and so only it is renumbered by capture bookkeeping.
        // An immediate is index-free and passes through every remapping unchanged.
        assert_eq!(captured.capture_index(), Some(2));
        assert_eq!(immediate.capture_index(), None);
        assert_eq!(
            captured.map_capture_index(|index| index + 3),
            XlaConstant::Captured(CaptureReference::new(5, ArrayIrType::Array(array_type.clone()))),
        );
        assert_eq!(immediate.map_capture_index(|_| 7), immediate);

        // The array-typed captured case and immediate-dimension case project to their corresponding member universes
        // and reject the other one.
        assert_eq!(
            <XlaConstant as ValueProjection<ArrayType>>::into_projected(captured.clone()),
            Ok(XlaArrayConstant::new(2, array_type.clone())),
        );
        assert_eq!(
            <XlaConstant as ValueProjection<ArrayType>>::into_projected(immediate.clone()),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            <XlaConstant as ValueProjection<ArrayType>>::from_projected(XlaArrayConstant::new(2, array_type)),
            captured,
        );
        assert_eq!(<XlaConstant as ValueProjection<DimensionType>>::projected(&immediate), Ok(&extent));
        assert_eq!(<XlaConstant as ValueProjection<DimensionType>>::into_projected(immediate), Ok(extent),);
        assert_eq!(
            <XlaConstant as ValueProjection<DimensionType>>::projected(&captured),
            Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
        );

        // A reference-typed capture carries only its capture-table index and structural type metadata. Identity
        // renaming preserves that index, while ordinary array/dimension projections remain unavailable. (No
        // reference-typed `ValueProjection` exists: capture lifting is its intended consumer and has not landed.)
        let source = DimensionVariable::new("source", DimensionBounds::positive(Some(8)).unwrap());
        let target = DimensionVariable::new("target", DimensionBounds::positive(Some(8)).unwrap());
        let reference_type =
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())])));
        let reference = XlaConstant::Captured(CaptureReference::new(7, ArrayIrType::Reference(reference_type.clone())));
        assert_eq!(reference.capture_index(), Some(7));
        assert_eq!(
            <XlaConstant as ValueProjection<ArrayType>>::projected(&reference).map(|value| value.r#type().into_owned()),
            Err(TypeError::invalid("expected array type but got reference type")),
        );
        assert_eq!(
            <XlaConstant as ValueProjection<DimensionType>>::projected(&reference),
            Err(TypeError::invalid("expected an immediate dimension but got a captured value")),
        );
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target.clone()).unwrap();
        let renamed_reference = reference.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed_reference.capture_index(), Some(7));
        assert_eq!(
            renamed_reference.r#type().into_owned(),
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(target)]),
            ))),
        );
    }

    #[test]
    fn test_core_reference_operations_promote_to_xla_reference_operations() {
        let write: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::ReferenceWrite(ReferenceWriteOperation::new()).into();
        assert!(matches!(&write, XlaOperation::ReferenceWrite(_)));
        assert!(matches!(write.to_core_operation(), Some(ArrayIrOperation::ReferenceWrite(_))));

        let swap: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::ReferenceSwap(ReferenceSwapOperation::new()).into();
        assert!(matches!(&swap, XlaOperation::ReferenceSwap(_)));
        assert!(matches!(swap.to_core_operation(), Some(ArrayIrOperation::ReferenceSwap(_))));

        let add_update: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()).into();
        assert!(matches!(&add_update, XlaOperation::ReferenceAddUpdate(_)));
        assert!(matches!(add_update.to_core_operation(), Some(ArrayIrOperation::ReferenceAddUpdate(_)),));

        let freeze: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::ReferenceFreeze(ReferenceFreezeOperation::new()).into();
        assert!(matches!(&freeze, XlaOperation::ReferenceFreeze(_)));
        assert!(matches!(freeze.to_core_operation(), Some(ArrayIrOperation::ReferenceFreeze(_)),));
    }

    #[test]
    fn test_xla_reference_discharge_rewrites_the_flat_reference_language() {
        // The backend operation family participates in reference discharge through its generated `discharge`
        // dispatcher, so a flat backend program rewrites into explicit immutable state with its external root
        // entering at its own boundary position and publishing one hidden final-state output.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut builder = XlaProgramBuilder::new();
        let external = builder.add_input(ArrayIrType::Reference(ReferenceType::new(scalar_type.clone())));
        let initial = builder.add_input(ArrayIrType::Array(scalar_type.clone()));
        let update = builder.add_input(ArrayIrType::Array(scalar_type));
        let root = builder
            .add_instruction(XlaOperation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![initial], None)
            .unwrap()[0];
        builder
            .add_instruction(
                XlaOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new()),
                Vec::new(),
                vec![root, update],
                None,
            )
            .unwrap();
        let local = builder
            .add_instruction(
                XlaOperation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                Vec::new(),
                vec![root],
                None,
            )
            .unwrap()[0];
        let previous = builder
            .add_instruction(
                XlaOperation::ReferenceSwap(ReferenceSwapOperation::new()),
                Vec::new(),
                vec![external, local],
                None,
            )
            .unwrap()[0];
        let source: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>> =
            builder.build(vec![previous], vec![Placeholder; 3], vec![Placeholder]).unwrap();

        let discharged = source.discharge_references(0).unwrap();

        // The one external root is written, so it enters as state at its own boundary position and publishes its
        // final state as the single hidden output after the public prefix, while the local root's allocation,
        // accumulation, and consumption become ordinary array SSA.
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(
            discharged.external_states(),
            &[ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
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
    fn test_core_composite_control_flow_promotes_to_xla_control_flow() {
        let condition: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::Condition(ConditionOperation::new()).into();
        assert!(matches!(condition, XlaOperation::Condition(_)));
        assert_eq!(condition.region_slots(), ConditionOperation::<XlaConstant>::new().region_slots());

        let r#while: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::While(WhileOperation::new().with_iteration_bound(3).unwrap()).into();
        assert!(matches!(r#while, XlaOperation::While(operation) if operation.iteration_bound() == Some(3)));

        let scan: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::Scan(ScanOperation::new(2, 5).with_reverse(true)).into();
        assert!(matches!(
            scan,
            XlaOperation::Scan(operation)
                if operation.carry_count() == 2
                    && operation.length() == &Dimension::Static(5)
                    && operation.reverse()
        ));

        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let zero: XlaOperation<XlaConstant> = ArrayOperation::Zero(ZeroOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent)]),
        )))
        .into();
        assert!(matches!(zero, XlaOperation::Zero(_)));

        // Member control flow must also promote to the composite carriers when it enters through the member family
        // conversion: the projected `Array` variant cannot own composite regions, so landing there would make the
        // staged regions unprojectable.
        let condition: XlaOperation<XlaConstant> =
            ArrayOperation::<XlaArrayConstant>::Condition(ConditionOperation::new()).into();
        assert!(matches!(condition, XlaOperation::Condition(_)));
        let r#while: XlaOperation<XlaConstant> =
            ArrayOperation::<XlaArrayConstant>::While(WhileOperation::new().with_iteration_bound(3).unwrap()).into();
        assert!(matches!(r#while, XlaOperation::While(operation) if operation.iteration_bound() == Some(3)));
        let scan: XlaOperation<XlaConstant> =
            ArrayOperation::<XlaArrayConstant>::Scan(ScanOperation::new(2, 5).with_reverse(true)).into();
        assert!(matches!(
            scan,
            XlaOperation::Scan(operation)
                if operation.carry_count() == 2
                    && operation.length() == &Dimension::Static(5)
                    && operation.reverse()
        ));
    }

    #[test]
    fn test_core_custom_derivative_and_rematerialization_promotions_preserve_metadata() {
        // These three payloads are promoted by move rather than reconstructed, so their complete stored surface must
        // survive: the non-differentiated operand split of all three, and additionally the rematerialization
        // optimization-barrier hint. The promoted carrier must also keep contributing the payload's own operation
        // name and region slots, because the attached regions are matched against those slots by name.
        let custom_jvp = CustomJvpOperation::<ArrayIrType>::new().with_non_differentiated_count(2);
        let promoted: XlaOperation<XlaConstant> = ArrayIrOperation::<XlaArrayConstant>::CustomJvp(custom_jvp).into();
        assert!(matches!(&promoted, XlaOperation::CustomJvp(operation) if operation == &custom_jvp));
        assert_eq!(promoted.name(), custom_jvp.name());
        assert_eq!(promoted.region_slots(), custom_jvp.region_slots());

        let custom_vjp = CustomVjpOperation::<ArrayIrType>::new().with_non_differentiated_count(3);
        let promoted: XlaOperation<XlaConstant> = ArrayIrOperation::<XlaArrayConstant>::CustomVjp(custom_vjp).into();
        assert!(matches!(&promoted, XlaOperation::CustomVjp(operation) if operation == &custom_vjp));
        assert_eq!(promoted.name(), custom_vjp.name());
        assert_eq!(promoted.region_slots(), custom_vjp.region_slots());

        let rematerialize =
            RematerializeOperation::<ArrayIrType>::new().with_non_differentiated_count(1).with_prevent_cse(true);
        let promoted: XlaOperation<XlaConstant> =
            ArrayIrOperation::<XlaArrayConstant>::Rematerialize(rematerialize).into();
        assert!(matches!(&promoted, XlaOperation::Rematerialize(operation) if operation == &rematerialize));
        assert_eq!(promoted.name(), rematerialize.name());
        assert_eq!(promoted.region_slots(), rematerialize.region_slots());
    }

    #[test]
    fn test_rematerialize_rejects_captures_registered_in_its_body() {
        use ryft_core::tracing_v2::rematerialize;

        // The rematerialized body is traced through a fresh-root context whose capture table is discarded, so a
        // capturing body is rejected at trace time; refer to `TracingContext::trace_with_named_axes` for the full
        // silent-aliasing rationale.
        let scalar_type = ArrayIrType::from(ArrayType::scalar(DataType::F32));
        let captured_value = XlaConstant::Captured(CaptureReference::new(0, scalar_type.clone()));
        let function = rematerialize::<XlaDomain<'static>, _, _, _>(
            move |x: XlaTracer<'static>| -> Result<XlaTracer<'static>, ProgramError> {
                let context = x.context().clone();
                let reference = context.capture(captured_value.clone())?;
                let captured = StagingContext::constant(&context, reference);
                let mut outputs = context.bind(AddOperation::new(), Vec::new(), &[x, captured])?;
                Ok(outputs.remove(0))
            },
        );
        let root = DomainTracingContext::<XlaDomain<'static>>::new();
        let input = root.input(scalar_type);
        let result = function.call(input);
        assert!(matches!(result, Err(ProgramError::DiscardedCaptures { count: 1 })));
    }

    #[test]
    fn test_xla_dynamic_disconnected_pullback_uses_explicit_extent_residual() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent)]));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut builder = XlaProgramBuilder::new();
        builder.add_input(dynamic_type.into());
        let scalar = builder.add_input(scalar_type.into());
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![scalar],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The XLA operation family preserves the core transform contract: only the runtime extent crosses the
        // residual boundary, and the mixed zero consumes it when the disconnected cotangent is materialized.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert!(matches!(
            linearization.primal().instructions().last().unwrap().operation(),
            XlaOperation::DimensionSize(_)
        ));
        let pullback = linearization.pullback().unwrap();
        let zero = pullback.instructions().last().unwrap();
        assert!(matches!(zero.operation(), XlaOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[ryft_core::AtomId::new(1)]);
    }

    #[test]
    fn test_xla_residual_zero_provider_materializes_dynamic_zero_from_array_source() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let primal_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let tangent_type = primal_type.tangent().unwrap();
        let context = TracingContext::<XlaConstant, XlaOperation>::new();
        let primal = context.input(ArrayIrType::Array(primal_type));

        // The XLA family delegates the singular identity-directed capture hook to the Array-IR implementation. A
        // source with the same geometry but a different element representation therefore supplies the extent for the
        // dynamic tangent zero instead of falling through to the input-free default.
        let zero = XlaOperation::<XlaConstant>::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(ArrayIrType::Array(tangent_type.clone())),
            std::slice::from_ref(&primal),
        )
        .unwrap();
        assert_eq!(zero.r#type().as_ref(), &ArrayIrType::Array(tangent_type));

        let builder = context.builder().borrow();
        let [dimension_size, zero] = builder.instructions() else {
            panic!("expected one dimension-size instruction followed by one dynamic zero instruction");
        };
        assert!(matches!(dimension_size.operation(), XlaOperation::DimensionSize(operation) if operation.axis() == 0));
        assert_eq!(dimension_size.inputs(), &[primal.atom_id().unwrap()]);
        assert!(matches!(zero.operation(), XlaOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[dimension_size.outputs()[0]]);
    }

    #[test]
    fn test_xla_residual_zero_provider_assembles_dynamic_zero_from_several_sources() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(5)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(7)).unwrap());
        let zero_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns.clone())]),
        );
        let context = TracingContext::<XlaConstant, XlaOperation>::new();
        let sources = [
            context.input(ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])))),
            context.input(ArrayIrType::Array(ArrayType::new(
                DataType::F8E8M0FNU,
                Shape::new(vec![Dimension::Dynamic(rows)]),
            ))),
            context.input(ArrayIrType::Dimension(DimensionType::new(columns))),
        ];

        // No single source has the zero's type, so the geometry is assembled per declared residual: the statically
        // shaped candidate names neither identity and is skipped without staging anything, the dynamic array supplies
        // the row extent through a `dimension_size` read, and the first-class dimension supplies the column extent as
        // itself. Only the composite delegation makes this reachable in the XLA family; the input-free default would
        // report every residual as unsupplied.
        let zero = XlaOperation::<XlaConstant>::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(ArrayIrType::Array(zero_type.clone())),
            &sources,
        )
        .unwrap();
        assert_eq!(zero.r#type().as_ref(), &ArrayIrType::Array(zero_type));

        let builder = context.builder().borrow();
        let [dimension_size, zero] = builder.instructions() else {
            panic!("expected one dimension-size instruction followed by one dynamic zero instruction");
        };
        assert!(matches!(dimension_size.operation(), XlaOperation::DimensionSize(operation) if operation.axis() == 0));
        assert_eq!(dimension_size.inputs(), &[sources[1].atom_id().unwrap()]);
        assert!(matches!(zero.operation(), XlaOperation::Zero(_)));
        assert_eq!(zero.inputs(), &[dimension_size.outputs()[0], sources[2].atom_id().unwrap()]);
    }

    #[test]
    fn test_jit_call_supports_composite_region_boundaries() {
        let dimension_type = ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new(
            "size",
            DimensionBounds::positive(Some(9)).unwrap(),
        )));
        let array_type = ArrayIrType::Array(vector_type());
        let interface = RegionInterface::new(
            vec![dimension_type.clone()],
            vec![array_type.clone(), dimension_type.clone()],
            Effects::PURE,
        );
        let operation = JitCallOperation::new(0);

        assert_eq!(operation.name(), JIT_CALL_OPERATION_NAME);
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Err(TypeError::invalid(format!("{} expects 1 attached callee region but got 0", JIT_CALL_OPERATION_NAME,))),
        );
        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&dimension_type), std::slice::from_ref(&interface))
                .unwrap(),
            vec![array_type, dimension_type],
        );
    }

    #[test]
    fn test_jit_call_jvp_omits_zero_space_boundary_tangents() {
        let dimension_type = ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new(
            "size",
            DimensionBounds::positive(Some(9)).unwrap(),
        )));
        let array_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));

        let mut callee_builder = XlaProgramBuilder::new();
        let dimension = callee_builder.add_input(dimension_type.clone());
        let array = callee_builder.add_input(array_type.clone());
        let callee = callee_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![dimension, array],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = XlaProgramBuilder::new();
        let dimension = builder.add_input(dimension_type.clone());
        let array = builder.add_input(array_type.clone());
        let callee = builder.import_region(callee.entry_region_ref());
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee],
                vec![dimension, array],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types(), vec![dimension_type.clone(), array_type.clone(), array_type.clone()]);
        assert_eq!(jvp.output_types(), vec![dimension_type, array_type.clone(), array_type]);
    }

    #[test]
    fn test_gateway_region_program_imports_with_fresh_alpha_equivalent_identities() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap());
        let extent_type = DimensionType::new(extent.clone());
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())]));
        let scalar_type = ArrayType::scalar(DataType::F32);

        let branch = || {
            let mut builder = XlaProgramBuilder::new();
            let extent = builder.add_input(extent_type.clone().into());
            let scalar = builder.add_input(scalar_type.clone().into());
            let output = builder
                .add_instruction(DynamicBroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![output],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        let integer_type = ArrayType::scalar(DataType::I32);
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let mut source_builder = XlaProgramBuilder::new();
        let true_region = source_builder.import_region(branch().entry_region_ref());
        let false_region = source_builder.import_region(branch().entry_region_ref());
        let integer = source_builder.add_input(integer_type.clone().into());
        let predicate = source_builder.add_input(predicate_type.clone().into());
        let scalar = source_builder.add_input(scalar_type.clone().into());
        let gateway = source_builder
            .add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![integer], None)
            .unwrap()[0];
        let output = source_builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, gateway, scalar],
                None,
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(source.output_types(), vec![ArrayIrType::Array(output_type)]);
        assert!(source.type_identity_signature().input_identities().is_empty());
        assert_eq!(
            source.type_identity_signature().internal_identities(),
            std::slice::from_ref(extent_type.variable())
        );

        let mut destination = XlaProgramBuilder::new();
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
                assert_eq!(interface.input_types()[0], ArrayIrType::Dimension(dimension.clone()));
                let output = <&ArrayType>::try_from(&interface.output_types()[0]).unwrap();
                assert_eq!(output.shape().dimensions(), &[Dimension::Dynamic(dimension.variable().clone())],);
            }
        }

        let destination = destination
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![first, second],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let signature = destination.type_identity_signature();
        assert!(signature.input_identities().is_empty());
        assert_eq!(
            signature.internal_identities(),
            &[first_dimension.variable().clone(), second_dimension.variable().clone()],
        );
    }

    #[test]
    fn test_jit_call_zero_transpose_uses_cotangent_descriptors() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let expected = ArrayIrType::Array(tangent_type.cotangent().unwrap());
        let tangent_type = ArrayIrType::Array(tangent_type);
        let mut context = TracingContext::<XlaConstant, XlaOperation>::new();
        let cotangents = transpose_primal_jit_call(
            &JitCallOperation::new(0),
            &mut context,
            &EmptyRegionDriver,
            &[PartialValue::Unknown(tangent_type.clone())],
            &[MaybeZero::Zero(tangent_type.clone())],
        )
        .unwrap();
        assert!(matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &expected));

        let known = context.input(tangent_type.clone());
        let cotangents = transpose_primal_jit_call(
            &JitCallOperation::new(0),
            &mut context,
            &EmptyRegionDriver,
            &[PartialValue::Known(known)],
            &[MaybeZero::Zero(tangent_type)],
        )
        .unwrap();
        assert!(matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &expected));
    }

    #[test]
    fn test_jit_call_dynamic_zero_transpose_uses_known_extent_operand() {
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap()));
        let output_type = ArrayIrType::Array(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]),
        ));
        let context = TracingContext::<XlaConstant, XlaOperation>::new();
        let extent = context.input(extent_type.into());

        let zero = materialize_transpose_cotangent(
            &context,
            &MaybeZero::Zero(output_type.clone()),
            &output_type,
            &[PartialValue::Known(extent)],
        )
        .unwrap();

        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one dynamic zero instruction");
        };
        assert!(matches!(instruction.operation(), XlaOperation::Zero(_)));
        assert_eq!(instruction.inputs(), &[ryft_core::AtomId::new(0)]);
        assert_eq!(zero.atom_id().unwrap(), instruction.outputs()[0]);
    }

    #[test]
    fn test_jit_call_mixed_output_transpose_materializes_zero_space_values() {
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)]));
        let value_program_type = ArrayIrType::Array(value_type.clone());
        let predicate_program_type = ArrayIrType::Array(predicate_type.clone());
        let source = {
            let mut builder = XlaProgramBuilder::new();
            let value = builder.add_input(value_program_type.clone());
            let predicate =
                builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, predicate_program_type.clone())));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![value, predicate],
                    vec![Placeholder],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let transposed = {
            let mut builder = XlaProgramBuilder::new();
            let value_cotangent = builder.add_input(value_program_type.clone());
            let _predicate_cotangent = builder.add_input(ArrayIrType::Array(predicate_type.cotangent().unwrap()));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![value_cotangent],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let mut context = TracingContext::<XlaConstant, XlaOperation>::new();
        let value_cotangent = context.input(value_program_type.clone());

        let contributions = transpose_primal_jit_call(
            &JitCallOperation::new(0),
            &mut context,
            &driver,
            &[PartialValue::Unknown(value_program_type.clone())],
            &[
                MaybeZero::Value(value_cotangent),
                MaybeZero::Zero(ArrayIrType::Array(predicate_type.cotangent().unwrap())),
            ],
        )
        .unwrap();

        assert!(
            matches!(&contributions[..], [MaybeZero::Value(value)] if value.r#type().as_ref() == &value_program_type)
        );
    }

    /// Online partial evaluation of a mixed `jit_call` against a live outer trace — the second recorded consumer of
    /// parent-context-polymorphic partial evaluation. The known half of the callee (including a callee literal it
    /// consumes) is rewrapped as a known-side `jit_call` staged into the outer program over the symbolic known
    /// input; the unknown half stays behind a residual `jit_call` whose literal is rebuilt inline; and the
    /// known→unknown residual edge flows from the known-side call's outputs into the residual call's inputs.
    #[test]
    fn test_jit_call_online_partial_evaluation_splits_callee_against_a_live_outer_trace() {
        use ryft_core::{PartialEvaluationInput, PartialEvaluationOutput, StagingContext, TracingContext};

        let r#type = ArrayIrType::Array(vector_type());

        // Callee `f(a, x) = (a + a, x * x, (a + a) * x)` over a known `a` and an unknown `x`. Capture constants are
        // deliberately absent: a capture-bearing callee preserves its boundary instead of splitting, which
        // `test_jit_call_partial_evaluation_preserves_boundary_for_capture_bearing_callees` pins.
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            let shifted = builder
                .add_instruction(AddOperation::new(), Vec::new(), vec![known_input, known_input], None)
                .unwrap()[0];
            let scaled = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![runtime_input, runtime_input], None)
                .unwrap()[0];
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![shifted, runtime_input], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![shifted, scaled, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        // Enclosing program staging one call to the callee over `[a, x]`.
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let call = XlaOperation::JitCall(JitCallOperation::new(0));
        let outputs = builder
            .add_instruction(call, vec![callee_region], vec![known_input, runtime_input], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type.clone())])
            .unwrap();

        // The known half landed in the outer program as one known-side `jit_call` over the symbolic known input,
        // producing the fully known callee output plus the residual edge (the same folded value, twice).
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            let known_instruction = &outer_builder.instructions()[0];
            assert!(
                matches!(known_instruction.operation(), XlaOperation::JitCall(_)),
                "expected the outer program to contain the known-side jit_call",
            );
            let known_callee = outer_builder.region_ref(known_instruction.regions()[0]).unwrap().to_program();
            assert_eq!(known_callee.input_ids().len(), 1);
            assert_eq!(known_callee.output_ids().len(), 2);
            assert_eq!(known_callee.instructions().len(), 1);
            assert!(matches!(known_callee.instructions()[0].operation(), XlaOperation::Array(ArrayOperation::Add(_)),));
        }

        // The unknown half stayed behind one residual `jit_call` over the unknown input plus the residual edge.
        assert_eq!(evaluation.program().instructions().len(), 1);
        let residual_instruction = &evaluation.program().instructions()[0];
        assert!(
            matches!(residual_instruction.operation(), XlaOperation::JitCall(_)),
            "expected the residual program to contain the residual jit_call",
        );
        let residual_callee = evaluation.program().region_ref(residual_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(residual_callee.input_ids().len(), 2);
        assert_eq!(residual_callee.instructions().len(), 2);

        // The boundary descriptors: the unknown enclosing input feeds the residual call, the residual edge is a
        // known feeder naming the known-side call's staged output, and the outputs reassemble in original order.
        assert_eq!(evaluation.inputs().len(), 2);
        assert!(matches!(&evaluation.inputs()[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs()[1], PartialEvaluationInput::Known(value) if value.atom_id().is_ok()));
        assert_eq!(evaluation.outputs().len(), 3);
        assert!(matches!(&evaluation.outputs()[0], PartialEvaluationOutput::Known(value) if value.atom_id().is_ok()));
        assert!(matches!(&evaluation.outputs()[1], PartialEvaluationOutput::Unknown(0)));
        assert!(matches!(&evaluation.outputs()[2], PartialEvaluationOutput::Unknown(1)));
    }

    #[test]
    fn test_jit_call_partial_evaluation_preserves_boundary_for_capture_bearing_callees() {
        // Partitioning does not remap the absolute indices of retained capture constants, so a mixed call whose
        // callee still holds one must keep its original boundary instead of splitting into derived callees whose
        // compacted or absent capture prefixes those indices would misname.
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            let captured = builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, r#type.clone())));
            let shifted =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![known_input, captured], None).unwrap()[0];
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![shifted, runtime_input], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![shifted, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(r#type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)])
            .unwrap();

        // No known-side call was hoisted into the outer trace, and the residual program retains the original call
        // with its callee — capture constant included — intact.
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(evaluation.program().instructions().len(), 1);
        let residual_instruction = &evaluation.program().instructions()[0];
        assert!(matches!(residual_instruction.operation(), XlaOperation::JitCall(_)));
        let residual_callee = evaluation.program().region_ref(residual_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(residual_callee.instructions().len(), 2);
        assert!(
            residual_callee
                .atoms()
                .iter()
                .any(|atom| atom.as_constant().is_some_and(|constant| constant.capture_index().is_some())),
        );
    }

    #[test]
    fn test_jit_call_partial_evaluation_rejects_state_before_partitioning() {
        let r#type = ArrayIrType::Array(vector_type());
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![known_input], None).unwrap();
            let doubled = builder
                .add_instruction(AddOperation::new(), Vec::new(), vec![known_input, known_input], None)
                .unwrap()[0];
            let product = builder
                .add_instruction(MulOperation::new(), Vec::new(), vec![known_input, runtime_input], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![doubled, product],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let known_input = builder.add_input(r#type.clone());
        let runtime_input = builder.add_input(r#type.clone());
        let callee_region = builder.intern_callee(&Arc::new(callee), None).unwrap();
        let outputs = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee_region],
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(r#type.clone());
        assert_eq!(
            program
                .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(r#type)],)
                .map(|_| ()),
            Err(ProgramError::UnsupportedOperation {
                // The entry-level up-front closure check identifies the intrinsic state operation before
                // carrier-specific partitioning can stage any known work.
                message: "`reference_new` must be discharged before partial evaluation".to_string(),
            }),
        );
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_nested_jit_call_reference_discharge_threads_callee_state_into_the_caller() {
        // Two nested call levels each swap a distinct reference. Discharge must lift both states to the caller's
        // boundary in public-input order, mark both as mutated with their own final-state output slot, and leave no
        // reference-typed atom anywhere in the resulting program.
        let reference_type = ReferenceType::new(vector_type());
        let inner_callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let replacement = builder.add_input(vector_type().into());
            let snapshot = builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let inner_callee = builder.import_region(inner_callee.entry_region_ref());
            let first_reference = builder.add_input(reference_type.clone().into());
            let first_replacement = builder.add_input(vector_type().into());
            let second_reference = builder.add_input(reference_type.clone().into());
            let second_replacement = builder.add_input(vector_type().into());
            let first_snapshot = builder
                .add_instruction(
                    XlaOperation::JitCall(JitCallOperation::new(0)),
                    vec![inner_callee],
                    vec![first_reference, first_replacement],
                    None,
                )
                .unwrap()[0];
            let second_snapshot = builder
                .add_instruction(
                    XlaOperation::JitCall(JitCallOperation::new(0)),
                    vec![inner_callee],
                    vec![second_reference, second_replacement],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![first_snapshot, second_snapshot],
                    vec![Placeholder; 4],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let first_reference = builder.add_input(reference_type.clone().into());
        let first_replacement = builder.add_input(vector_type().into());
        let second_reference = builder.add_input(reference_type.into());
        let second_replacement = builder.add_input(vector_type().into());
        let snapshots = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(0)),
                vec![callee],
                vec![first_reference, first_replacement, second_reference, second_replacement],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(snapshots, vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();

        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.public_output_count(), 2);
        assert_eq!(discharged.program().output_count(), 4);
        assert_eq!(discharged.external_states().len(), 2);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Input { index: 0 });
        assert!(discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), Some(2));
        assert_eq!(discharged.external_states()[1].source(), ReferenceSource::Input { index: 2 });
        assert!(discharged.external_states()[1].is_mutated());
        assert_eq!(discharged.external_states()[1].final_state_output_index(), Some(3));
    }

    #[test]
    fn test_nested_jit_call_reference_discharge_resolves_callee_lexical_capture_scope() {
        // A branch region inside the callee reaches its reference through a capture constant, which must resolve
        // against the callee's own capture scope rather than the entry program's lifted prefix: the accessed root is
        // the caller's second public input, and only that mutated state gains a final-state output.
        let reference_type = ReferenceType::new(vector_type());
        let branch = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let replacement = builder.add_input(vector_type().into());
            let reference = builder.add_constant(XlaConstant::Captured(CaptureReference::new(
                0,
                ArrayIrType::Reference(reference_type.clone()),
            )));
            let snapshot = builder
                .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let branch = builder.import_region(branch.entry_region_ref());
            builder.add_input(reference_type.clone().into());
            let replacement = builder.add_input(vector_type().into());
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
            let snapshot = builder
                .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, replacement], None)
                .unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        builder.add_input(reference_type.clone().into());
        let reference = builder.add_input(reference_type.into());
        let replacement = builder.add_input(vector_type().into());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let snapshot = builder
            .add_instruction(
                XlaOperation::JitCall(JitCallOperation::new(1)),
                vec![callee],
                vec![reference, replacement, predicate],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![snapshot], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();

        let discharged = program.discharge_references_with_lifted_captures(0).unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.program().output_count(), 2);
        assert_eq!(discharged.external_states().len(), 2);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Input { index: 0 });
        assert!(!discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), None);
        assert_eq!(discharged.external_states()[1].source(), ReferenceSource::Input { index: 1 });
        assert!(discharged.external_states()[1].is_mutated());
        assert_eq!(discharged.external_states()[1].final_state_output_index(), Some(1));
    }

    #[test]
    fn test_jit_call_reference_discharge_preserves_forwarded_root_identity() {
        let reference_type = ReferenceType::new(vector_type());
        let mut callee_builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let reference = callee_builder.add_input(reference_type.clone().into());
        let callee = callee_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let forwarded = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], vec![reference], None)
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![forwarded], None).unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The callee only forwards its reference input, so discharge must keep the forwarded value bound to the same
        // root as the caller's public reference input instead of synthesizing a second state slot for it. Reading
        // through the forwarded result is not a mutation, so no final-state output is appended.
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert_eq!(discharged.program().output_count(), 1);
        assert_eq!(discharged.external_states().len(), 1);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Input { index: 0 });
        assert!(!discharged.external_states()[0].is_mutated());
        assert_eq!(discharged.external_states()[0].final_state_output_index(), None);
    }

    #[test]
    fn test_jit_call_reference_discharge_widens_its_callee_with_the_final_state() {
        // `jit_call` forwards its operands onto its callee one for one and reports the callee's outputs as its own, so
        // the shared positional rewrite threads state through it: the callee gains the entering state after its
        // declared inputs and publishes the final state after its declared outputs, while the call site keeps its own
        // operand and output contract. The callee mutates the root and returns only the previous snapshot, so the
        // final state is exactly what the call site would otherwise have lost.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut callee_builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee_reference = callee_builder.add_input(reference_type.into());
        let callee_update = callee_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let previous = callee_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![callee_reference, callee_update], None)
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let callee = builder.import_program(callee);
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let root = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let previous = builder
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee], vec![root, update], None)
            .unwrap()[0];
        let frozen = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![root], None).unwrap()[0];
        let source = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![previous, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        // The callee's local root is allocated, mutated, and consumed inside the call, so widening the boundary
        // leaves no external state behind and the rewritten call is the entry region's only instruction.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_states(), &[]);
        assert_eq!(discharged.program().output_types().len(), 2);
        assert_eq!(discharged.program().entry_region_ref().instructions().len(), 1);
    }

    #[test]
    fn test_scan_reference_discharge_widens_carries_and_preserves_scan_metadata() {
        let reference_type = ReferenceType::new(vector_type());
        let mut body_builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let body_carry = body_builder.add_input(vector_type().into());
        let reference =
            body_builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, reference_type.clone().into())));
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![body_carry, value],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let capture = XlaConstant::Captured(CaptureReference::new(
            1,
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(4)])).into(),
        ));
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        builder.add_input(reference_type.into());
        let carry = builder.add_input(vector_type().into());
        let operation = ScanOperation::<XlaConstant>::new(1, 3)
            .with_reverse(true)
            .with_unroll(3)
            .unwrap()
            .with_captures(vec![capture.clone()]);
        let outputs = builder
            .add_instruction(XlaOperation::Scan(operation), vec![body], vec![carry], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // Threading the captured reference state through the scan widens the carry list of the rebuilt `XlaOperation`
        // scan payload and leaves its capture environment intact.
        let discharged = program.discharge_references_with_lifted_captures(1).unwrap();
        assert_eq!(discharged.public_output_count(), 2);
        assert_eq!(discharged.external_states().len(), 1);
        assert_eq!(discharged.external_states()[0].source(), ReferenceSource::Capture { index: 0 });
        assert!(!discharged.external_states()[0].is_mutated());
        let XlaOperation::Scan(scan) = discharged.program().entry_region_ref().instructions()[0].operation() else {
            panic!("expected a discharged scan operation");
        };
        assert_eq!(scan.carry_count(), 2);
        assert_eq!(scan.unroll(), 3);
        assert_eq!(scan.captures(), &[capture]);
    }

    #[test]
    fn test_rematerialization_policies_are_available_for_the_xla_operation_family() {
        use ryft_core::Memory;
        use ryft_core::tracing_v2::{
            DotsSaveable, DotsWithNoBatchDimsSaveable, EverythingSaveable, NothingSaveable, OffloadDotsWithNoBatchDims,
            RematerializationPolicy, SaveAndOffloadOnlyTheseNames, SaveFromBothPolicies, SaveOnlyTheseNames,
        };

        // The built-in rematerialization policies — including the projection-bounded dot and tag policies and the
        // transfer-bounded offloading policies — are available for `XlaOperation` through the derive-generated
        // array projection and its `TransferToMemoryOperation` conversion. This is a compile-time capability
        // check: the assertions below fail to compile if any projection or conversion bound is unsatisfied.
        fn assert_policy<P: RematerializationPolicy<ArrayType, ArrayOperation<XlaArrayConstant>>>(_policy: P) {}
        assert_policy(NothingSaveable);
        assert_policy(EverythingSaveable);
        assert_policy(DotsSaveable);
        assert_policy(DotsWithNoBatchDimsSaveable);
        assert_policy(SaveOnlyTheseNames::new(["u"]));
        assert_policy(SaveAndOffloadOnlyTheseNames::new(["u"], ["v"], Memory::Host { pinned: true }));
        assert_policy(OffloadDotsWithNoBatchDims::new(Memory::Host { pinned: true }));
        assert_policy(SaveFromBothPolicies::new(DotsSaveable, SaveOnlyTheseNames::new(["u"])));
    }
}
