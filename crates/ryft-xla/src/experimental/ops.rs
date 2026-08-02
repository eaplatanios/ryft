use std::fmt::Display;
use std::rc::Rc;

use ryft_core::batching::{
    ArrayBatch, ArrayBatching, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use ryft_core::captures::CaptureReference;
use ryft_core::compilation::function::CompiledCallOperation;
use ryft_core::contexts::{Context, StagingContext};
use ryft_core::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationError, LinearCallOperation,
    TransposableOperation, TranspositionDriver, TranspositionZeroProvider,
};
use ryft_core::macros::check_count;
use ryft_core::operations::attention::{DotProductAttentionBackwardOperation, DotProductAttentionOperation};
use ryft_core::operations::compare::CompareOperation;
use ryft_core::operations::complex::{ComplexOperation, ConjugateOperation, ImaginaryOperation, RealOperation};
use ryft_core::operations::constants::{ConstantOperation, OneLikeOperation, ZeroLikeOperation};
use ryft_core::operations::constants::{IotaOperation, OneOperation, Zero, ZeroOperation, ZeroOperationProvider};
use ryft_core::operations::control_flow::SelectOperation;
use ryft_core::operations::control_flow::{ConditionOperation, ScanOperation, WhileOperation};
use ryft_core::operations::custom_call::CustomCallOperation;
use ryft_core::operations::differentiation::{CoordinateBasisOperation, StopGradientOperation};
use ryft_core::operations::dimensions::{
    DimensionFromScalarOperation, DimensionRequirementOperation, DimensionSizeOperation, DimensionToScalarOperation,
};
use ryft_core::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use ryft_core::operations::manipulation::{
    BroadcastOperation, ConcatenateOperation, DynamicShapeSliceOperation, PadOperation, ReshapeOperation,
};
use ryft_core::operations::manipulation::{
    ConvertElementTypeOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, GatherOperation,
    LegacyBroadcastOperation, LegacyReshapeOperation, ScatterOperation, SliceOperation, TransposeOperation,
    UpdateSliceOperation,
};
use ryft_core::operations::math::{
    AbsOperation, AddOperation, Atan2Operation, CeilOperation, CosOperation, DivOperation, DotOperation, ErfOperation,
    ExpOperation, FloorOperation, LogOperation, LogisticOperation, MaxOperation, MinOperation, MulOperation,
    NegOperation, PowOperation, ReduceOperation, RemOperation, RoundOperation, RsqrtOperation, ScaledDotOperation,
    SignOperation, SinOperation, SqrtOperation, SubOperation, TanhOperation,
};
use ryft_core::operations::random::RngBitGeneratorOperation;
use ryft_core::operations::sharding::{ReshardOperation, ShardingConstraintOperation};
use ryft_core::operations::sort::SortOperation;
use ryft_core::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use ryft_core::programs::effects::Effects;
use ryft_core::programs::identities::TypeIdentityRenaming;
use ryft_core::programs::operations::{Operation, OperationProjection};
use ryft_core::programs::regions::{CalleeRegionDriver, OutputRegionProvenance, RegionInterface, RegionSlot};
use ryft_core::programs::{Concretizable, MaybeZero, Program, ProgramBuilder, ProgramError, Value, ValueProjection};
use ryft_core::tracing::{Tracer, TracingContext};

use ryft_core::axes::AxisIndexOperation;
use ryft_core::backends::array_programs::ArrayProgramOperation;
use ryft_core::backends::arrays::{Array as ReferenceArray, ArrayOperation};
use ryft_core::backends::dimensions::{DimensionOperation, DimensionValue};
use ryft_core::differentiation::DifferentiationDual;
use ryft_core::operations::collectives::{AllGatherOperation, AllToAllOperation, PSumScatterOperation};
use ryft_core::operations::collectives::{CollectiveOperation, PpermuteOperation};
use ryft_core::operations::debugging::PrintOperation;
use ryft_core::operations::memory::TransferToMemoryOperation;
use ryft_core::operations::tag::TagOperation;
use ryft_core::programs::types::{Type, TypeError, Typed};
use ryft_core::tracing_v2::custom_derivatives::{CustomJvpOperation, CustomVjpOperation};
use ryft_core::tracing_v2::rematerialization::RematerializeOperation;
use ryft_core::types::{ArrayProgramType, ArrayType, Dimension, DimensionType, DimensionVariable};

use crate::experimental::operations::ShardMapOperation;

/// Lifetime-free reference to an array member captured by an XLA program.
pub type XlaArrayConstant = CaptureReference<ArrayType>;

/// Production XLA program constant.
pub type XlaConstant = CaptureReference<ArrayProgramType>;

/// Ordinary staged-operation universe owned by the XLA backend.
///
/// This enum flattens the core array operation payloads directly into the backend-owned operation family. Higher-order
/// instructions attach their nested computations as regions of the containing XLA program, so those regions can
/// contain backend-specific operations such as [`jit_call`](JitCallOperation) and
/// [`shard_map`](ShardMapOperation).
#[derive(Clone, Debug)]
pub enum XlaOperation<C = XlaConstant>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    /// Mixed zero constructor whose explicit first-class dimension operands provide its dynamic result extents.
    /// This variant cannot be represented by the homogeneous array member because its signature crosses member
    /// kinds: it consumes dimension members and produces an array member.
    Zero(ZeroOperation<ArrayType>),

    /// Mixed one constructor with explicit dynamic-extent operands.
    DynamicOne(OneOperation<ArrayType>),

    /// Mixed iota constructor with explicit dynamic-extent operands.
    DynamicIota(IotaOperation<ArrayType>),

    /// Homogeneous array operation.
    Array(ArrayOperation<C::Projected>),

    /// Homogeneous first-class-dimension operation.
    Dimension(DimensionOperation<DimensionValue>),

    /// Mixed comparison of two dimensions producing Boolean array data.
    Compare(CompareOperation),

    /// Reads an array extent as a first-class dimension.
    DimensionSize(DimensionSizeOperation),

    /// Converts scalar array data into a checked first-class dimension.
    DimensionFromScalar(DimensionFromScalarOperation),

    /// Converts a first-class dimension into scalar array data.
    DimensionToScalar(DimensionToScalarOperation),

    /// Reshapes an array using explicit dimension operands.
    Reshape(ReshapeOperation),

    /// Broadcasts an array using explicit dimension operands.
    Broadcast(BroadcastOperation),

    /// Concatenates arrays with an explicit result extent.
    Concatenate(ConcatenateOperation),

    /// Calls a foreign kernel with explicit dynamic result extents.
    CustomCall(CustomCallOperation),

    /// Pads an array with explicit result extents.
    Pad(PadOperation),

    /// Slices an array using first-class start and size dimensions.
    DynamicShapeSlice(DynamicShapeSliceOperation),

    /// Generates random bits with explicit dynamic result extents.
    RngBitGenerator(RngBitGeneratorOperation),

    /// Gathers values with one explicit extent per result axis.
    AllGather(AllGatherOperation),

    /// Scatters values with one explicit extent per result axis.
    PSumScatter(PSumScatterOperation),

    /// Exchanges values with one explicit extent per result axis.
    AllToAll(AllToAllOperation),

    /// Backend-owned condition whose attached branch regions can contain XLA operations.
    Condition(ConditionOperation<C>),

    /// Backend-owned loop whose attached condition and body regions can contain XLA operations.
    While(WhileOperation),

    /// Backend-owned scan whose attached body region can contain XLA operations.
    Scan(ScanOperation<C>),

    /// Backend-owned custom JVP call whose attached regions can contain XLA operations.
    CustomJvp(CustomJvpOperation),

    /// Backend-owned custom VJP call whose attached regions can contain XLA operations.
    CustomVjp(CustomVjpOperation),

    /// Differentiation-owned call to an explicitly transposable linear map with ordinary trailing residual
    /// operands. This variant carries both carrier forms: the forward-and-transpose form lowers by inlining its
    /// forward region, while the reverse-only transpose-only form (attached transpose region only) cannot be lowered
    /// and reports the canonical reverse-only diagnostic.
    LinearCall(LinearCallOperation<ArrayProgramType>),

    /// Backend-owned rematerialized call whose attached regions can contain XLA operations.
    Rematerialize(RematerializeOperation),

    /// Call to a flat jitted XLA sub-program.
    JitCall(JitCallOperation),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<C>>),
}

impl<C> From<ArrayOperation<C::Projected>> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: ArrayOperation<C::Projected>) -> Self {
        match operation {
            ArrayOperation::Zero(operation) => ArrayProgramOperation::<C::Projected>::from(operation).into(),
            operation => Self::Array(operation),
        }
    }
}

impl<C> From<ArrayProgramOperation<C::Projected>> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn from(operation: ArrayProgramOperation<C::Projected>) -> Self {
        match operation {
            ArrayProgramOperation::Zero(operation) => Self::Zero(operation),
            ArrayProgramOperation::DynamicOne(operation) => Self::DynamicOne(operation),
            ArrayProgramOperation::DynamicIota(operation) => Self::DynamicIota(operation),
            ArrayProgramOperation::Array(operation) => Self::Array(operation),
            ArrayProgramOperation::Dimension(operation) => Self::Dimension(operation),
            ArrayProgramOperation::Compare(operation) => Self::Compare(operation),
            ArrayProgramOperation::DimensionSize(operation) => Self::DimensionSize(operation),
            ArrayProgramOperation::DimensionFromScalar(operation) => Self::DimensionFromScalar(operation),
            ArrayProgramOperation::DimensionToScalar(operation) => Self::DimensionToScalar(operation),
            ArrayProgramOperation::Reshape(operation) => Self::Reshape(operation),
            ArrayProgramOperation::Broadcast(operation) => Self::Broadcast(operation),
            ArrayProgramOperation::Concatenate(operation) => Self::Concatenate(operation),
            ArrayProgramOperation::CustomCall(operation) => Self::CustomCall(operation),
            ArrayProgramOperation::Pad(operation) => Self::Pad(operation),
            ArrayProgramOperation::DynamicShapeSlice(operation) => Self::DynamicShapeSlice(operation),
            ArrayProgramOperation::RngBitGenerator(operation) => Self::RngBitGenerator(operation),
            ArrayProgramOperation::AllGather(operation) => Self::AllGather(operation),
            ArrayProgramOperation::PSumScatter(operation) => Self::PSumScatter(operation),
            ArrayProgramOperation::AllToAll(operation) => Self::AllToAll(operation),
            ArrayProgramOperation::Condition(_) => Self::Condition(ConditionOperation::new()),
            ArrayProgramOperation::While(operation) => Self::While(operation),
            ArrayProgramOperation::Scan(operation) => {
                let captures = operation
                    .captures()
                    .iter()
                    .cloned()
                    .map(|capture| match capture {
                        ryft_core::backends::array_programs::ArrayProgramValue::Array(capture) => {
                            C::from_projected(capture)
                        }
                        ryft_core::backends::array_programs::ArrayProgramValue::Dimension(_) => {
                            unreachable!("validated scan captures are always stacked arrays")
                        }
                    })
                    .collect();
                Self::Scan(
                    ScanOperation::<C>::new(operation.carry_count(), operation.length())
                        .with_reverse(operation.reverse())
                        .with_unroll(operation.unroll())
                        .unwrap()
                        .with_captures(captures),
                )
            }
            ArrayProgramOperation::LinearCall(operation) => Self::LinearCall(operation),
        }
    }
}

impl<C> From<DimensionRequirementOperation> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: DimensionRequirementOperation) -> Self {
        Self::Dimension(DimensionOperation::Requirement(operation))
    }
}

impl<C> From<LinearCallOperation<ArrayProgramType>> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: LinearCallOperation<ArrayProgramType>) -> Self {
        Self::LinearCall(operation)
    }
}

impl<C> From<DimensionOperation<DimensionValue>> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: DimensionOperation<DimensionValue>) -> Self {
        Self::Dimension(operation)
    }
}

impl<C> From<JitCallOperation> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: JitCallOperation) -> Self {
        Self::JitCall(operation)
    }
}

impl<C> From<ConditionOperation<C>> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: ConditionOperation<C>) -> Self {
        Self::Condition(operation)
    }
}

impl<C> From<WhileOperation> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: WhileOperation) -> Self {
        Self::While(operation)
    }
}

impl<C> From<ScanOperation<C>> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: ScanOperation<C>) -> Self {
        Self::Scan(operation)
    }
}

impl<C> From<CustomJvpOperation> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: CustomJvpOperation) -> Self {
        Self::CustomJvp(operation)
    }
}

impl<C> From<CustomVjpOperation> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: CustomVjpOperation) -> Self {
        Self::CustomVjp(operation)
    }
}

impl<C> From<RematerializeOperation> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: RematerializeOperation) -> Self {
        Self::Rematerialize(operation)
    }
}

impl<C> From<ShardMapOperation<C>> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn from(operation: ShardMapOperation<C>) -> Self {
        Self::ShardMap(Box::new(operation))
    }
}

macro_rules! impl_composite_operation_conversion {
    // Generates direct composite-operation conversions through the canonical core operation family.
    ($($operation:ty),+ $(,)?) => {
        $(
            impl<C> From<$operation> for XlaOperation<C>
            where
                C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            {
                #[inline]
                fn from(operation: $operation) -> Self {
                    ArrayProgramOperation::<C::Projected>::from(operation).into()
                }
            }
        )+
    };
}

impl_composite_operation_conversion!(
    ZeroOperation<ArrayType>,
    OneOperation<ArrayType>,
    IotaOperation<ArrayType>,
    CompareOperation,
    DimensionSizeOperation,
    DimensionFromScalarOperation,
    DimensionToScalarOperation,
    ReshapeOperation,
    BroadcastOperation,
    ConcatenateOperation,
    CustomCallOperation,
    PadOperation,
    RngBitGeneratorOperation,
    AllGatherOperation,
    PSumScatterOperation,
    AllToAllOperation,
);

macro_rules! impl_array_operation_conversion {
    // Generates homogeneous array-operation conversions through the canonical projected member family.
    ($($operation:ty),+ $(,)?) => {
        $(
            impl<C> From<$operation> for XlaOperation<C>
            where
                C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            {
                #[inline]
                fn from(operation: $operation) -> Self {
                    ArrayOperation::<C::Projected>::from(operation).into()
                }
            }
        )+
    };
}

impl_array_operation_conversion!(
    ZeroLikeOperation,
    OneLikeOperation,
    ConstantOperation<ReferenceArray>,
    CoordinateBasisOperation<ArrayType>,
    AbsOperation,
    NegOperation,
    AddOperation,
    SubOperation,
    MulOperation,
    DivOperation,
    SinOperation,
    CosOperation,
    Atan2Operation,
    ExpOperation,
    LogOperation,
    SqrtOperation,
    RsqrtOperation,
    TanhOperation,
    LogisticOperation,
    ErfOperation,
    PowOperation,
    SignOperation,
    FloorOperation,
    CeilOperation,
    RoundOperation,
    MaxOperation,
    MinOperation,
    RemOperation,
    NotOperation,
    AndOperation,
    OrOperation,
    XorOperation,
    ComplexOperation,
    ConjugateOperation,
    RealOperation,
    ImaginaryOperation,
    DotOperation,
    ScaledDotOperation,
    DotProductAttentionOperation,
    DotProductAttentionBackwardOperation,
    ReduceOperation,
    SortOperation,
    CollectiveOperation,
    PpermuteOperation,
    AxisIndexOperation,
    TransposeOperation,
    LegacyReshapeOperation,
    LegacyBroadcastOperation,
    GatherOperation,
    ScatterOperation,
    SliceOperation,
    UpdateSliceOperation,
    DynamicSliceOperation,
    DynamicUpdateSliceOperation,
    SelectOperation,
    ConvertElementTypeOperation,
    TransferToMemoryOperation,
    ReshardOperation,
    ShardingConstraintOperation,
    StopGradientOperation,
    TagOperation,
    PrintOperation,
);

impl<C> OperationProjection<ArrayType> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Projected = ArrayOperation<C::Projected>;
}

impl<C> OperationProjection<DimensionType> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Projected = DimensionOperation<DimensionValue>;
}

impl<Capture> ZeroOperationProvider<ArrayProgramType> for XlaOperation<Capture>
where
    Capture: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn zero_operation(r#type: ArrayProgramType) -> Result<Self, ProgramError> {
        Ok(ArrayProgramOperation::<Capture::Projected>::zero_operation(r#type)?.into())
    }
}

impl<Capture> TranspositionZeroProvider<ArrayProgramType> for XlaOperation<Capture>
where
    Capture: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn transposition_zero_residual_types(r#type: &ArrayProgramType) -> Vec<ArrayProgramType> {
        ArrayProgramOperation::<Capture::Projected>::transposition_zero_residual_types(r#type)
    }

    fn capture_transposition_zero_residuals<V: Value<Type = ArrayProgramType>>(
        builder: &mut ProgramBuilder<V, Self>,
        source: ryft_core::AtomId,
        r#type: &ArrayProgramType,
    ) -> Result<Vec<ryft_core::AtomId>, ProgramError> {
        ArrayProgramOperation::<Capture::Projected>::capture_transposition_zero_residuals(builder, source, r#type)
    }

    fn capture_transposition_zero_values<C: Context<Type = ArrayProgramType, Operation = Self>>(
        context: &C,
        source: &C::Value,
        r#type: &ArrayProgramType,
    ) -> Result<Vec<C::Value>, ProgramError> {
        ArrayProgramOperation::<Capture::Projected>::capture_transposition_zero_values(context, source, r#type)
    }

    fn add_transposition_zero<V: Value<Type = ArrayProgramType>>(
        builder: &mut ProgramBuilder<V, Self>,
        r#type: ArrayProgramType,
        residuals: &[ryft_core::AtomId],
    ) -> Result<ryft_core::AtomId, ProgramError> {
        ArrayProgramOperation::<Capture::Projected>::add_transposition_zero(builder, r#type, residuals)
    }
}

impl<C> XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    /// Returns the canonical core operation for a member or mixed primitive, or `None` for an XLA-owned
    /// higher-order operation.
    ///
    /// This conversion clones the payload, so it is reserved for methods that need the composite family's boundary
    /// projection or reconstruct an operation anyway (e.g., type inference, identity renaming, interpretation, and
    /// partial evaluation). Cheap per-instruction accessors dispatch to the borrowed payload directly instead.
    pub(crate) fn to_core_operation(&self) -> Option<ArrayProgramOperation<C::Projected>> {
        Some(match self {
            Self::Zero(operation) => ArrayProgramOperation::Zero(operation.clone()),
            Self::DynamicOne(operation) => ArrayProgramOperation::DynamicOne(operation.clone()),
            Self::DynamicIota(operation) => ArrayProgramOperation::DynamicIota(operation.clone()),
            Self::Array(operation) => ArrayProgramOperation::Array(operation.clone()),
            Self::Dimension(operation) => ArrayProgramOperation::Dimension(operation.clone()),
            Self::Compare(operation) => ArrayProgramOperation::Compare(operation.clone()),
            Self::DimensionSize(operation) => ArrayProgramOperation::DimensionSize(operation.clone()),
            Self::DimensionFromScalar(operation) => ArrayProgramOperation::DimensionFromScalar(operation.clone()),
            Self::DimensionToScalar(operation) => ArrayProgramOperation::DimensionToScalar(*operation),
            Self::Reshape(operation) => ArrayProgramOperation::Reshape(operation.clone()),
            Self::Broadcast(operation) => ArrayProgramOperation::Broadcast(operation.clone()),
            Self::Concatenate(operation) => ArrayProgramOperation::Concatenate(operation.clone()),
            Self::CustomCall(operation) => ArrayProgramOperation::CustomCall(operation.clone()),
            Self::Pad(operation) => ArrayProgramOperation::Pad(operation.clone()),
            Self::DynamicShapeSlice(operation) => ArrayProgramOperation::DynamicShapeSlice(operation.clone()),
            Self::RngBitGenerator(operation) => ArrayProgramOperation::RngBitGenerator(operation.clone()),
            Self::AllGather(operation) => ArrayProgramOperation::AllGather(operation.clone()),
            Self::PSumScatter(operation) => ArrayProgramOperation::PSumScatter(operation.clone()),
            Self::AllToAll(operation) => ArrayProgramOperation::AllToAll(operation.clone()),
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

impl<'operation, C> TryFrom<&'operation XlaOperation<C>> for &'operation WhileOperation
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    type Error = ProgramError;

    fn try_from(operation: &'operation XlaOperation<C>) -> Result<Self, Self::Error> {
        let XlaOperation::While(operation) = operation else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("expected a while operation but got '{}'", operation.name()),
            });
        };
        Ok(operation)
    }
}

macro_rules! dispatch_operation {
    // Delegates one borrowed `Operation` method to the active payload without materializing the canonical core
    // operation, so cheap per-instruction accessors (e.g., names, effects, and region slots) never clone payload
    // vectors on program-construction and validation hot paths. The per-variant trait selection mirrors the
    // corresponding delegation arms of `ArrayProgramOperation` exactly, so both dispatchers report identical
    // semantics for shared member and mixed payloads.
    ($operation:expr, $method:ident $(, $argument:expr)* $(,)?) => {
        match $operation {
            XlaOperation::Zero(operation) => operation.$method($($argument),*),
            XlaOperation::DynamicOne(operation) => operation.$method($($argument),*),
            XlaOperation::DynamicIota(operation) => operation.$method($($argument),*),
            XlaOperation::Array(operation) => operation.$method($($argument),*),
            XlaOperation::Dimension(operation) => operation.$method($($argument),*),
            XlaOperation::Compare(operation) => Operation::<ArrayProgramType>::$method(operation, $($argument),*),
            XlaOperation::DimensionSize(operation) => operation.$method($($argument),*),
            XlaOperation::DimensionFromScalar(operation) => operation.$method($($argument),*),
            XlaOperation::DimensionToScalar(operation) => operation.$method($($argument),*),
            XlaOperation::Reshape(operation) => operation.$method($($argument),*),
            XlaOperation::Broadcast(operation) => operation.$method($($argument),*),
            XlaOperation::Concatenate(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::CustomCall(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::Pad(operation) => Operation::<ArrayProgramType>::$method(operation, $($argument),*),
            XlaOperation::DynamicShapeSlice(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::RngBitGenerator(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::AllGather(operation) => Operation::<ArrayType>::$method(operation, $($argument),*),
            XlaOperation::PSumScatter(operation) => Operation::<ArrayType>::$method(operation, $($argument),*),
            XlaOperation::AllToAll(operation) => Operation::<ArrayType>::$method(operation, $($argument),*),
            XlaOperation::Condition(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::While(operation) => Operation::<ArrayProgramType>::$method(operation, $($argument),*),
            XlaOperation::Scan(operation) => Operation::<ArrayProgramType>::$method(operation, $($argument),*),
            XlaOperation::CustomJvp(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::CustomVjp(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::LinearCall(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::Rematerialize(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::JitCall(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::ShardMap(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
        }
    };
}

macro_rules! dispatch_higher_operation {
    // Delegates one `Operation<ArrayProgramType>` method to the active XLA-owned higher-order payload.
    ($operation:expr, $method:ident $(, $argument:expr)* $(,)?) => {
        match $operation {
            XlaOperation::Condition(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::While(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::Scan(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::CustomJvp(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::CustomVjp(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::LinearCall(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::Rematerialize(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::JitCall(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            XlaOperation::ShardMap(operation) => {
                Operation::<ArrayProgramType>::$method(operation, $($argument),*)
            }
            _ => unreachable!("member and mixed operations are handled by the canonical core operation family"),
        }
    };
}

impl<C> Display for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<C> Operation<ArrayProgramType> for XlaOperation<C>
where
    C: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn name(&self) -> &'static str {
        dispatch_operation!(self, name)
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        dispatch_operation!(self, region_slots)
    }

    fn infer_region_input_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<Option<Vec<ArrayProgramType>>>, TypeError> {
        match self.to_core_operation() {
            Some(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            None => dispatch_higher_operation!(self, infer_region_input_types, input_types, region_interfaces),
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayProgramType],
        region_interfaces: &[RegionInterface<ArrayProgramType>],
    ) -> Result<Vec<ArrayProgramType>, TypeError> {
        match self.to_core_operation() {
            Some(operation) => operation.infer_output_types(input_types, region_interfaces),
            None => dispatch_higher_operation!(self, infer_output_types, input_types, region_interfaces),
        }
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        dispatch_operation!(self, output_region_provenance, output_index)
    }

    fn is_zero(&self, output_index: usize) -> bool {
        dispatch_operation!(self, is_zero, output_index)
    }

    fn effects(&self) -> Effects {
        dispatch_operation!(self, effects)
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        if let Some(operation) = self.to_core_operation() {
            return Ok(operation.rename_type_identities(renaming)?.into());
        }
        match self {
            Self::Condition(operation) => {
                Ok(Self::Condition(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::While(operation) => {
                Ok(Self::While(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::Scan(operation) => {
                Ok(Self::Scan(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::CustomJvp(operation) => {
                Ok(Self::CustomJvp(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::CustomVjp(operation) => {
                Ok(Self::CustomVjp(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::LinearCall(operation) => {
                Ok(Self::LinearCall(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::Rematerialize(operation) => {
                Ok(Self::Rematerialize(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::JitCall(operation) => {
                Ok(Self::JitCall(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            Self::ShardMap(operation) => {
                Ok(Self::ShardMap(Operation::<ArrayProgramType>::rename_type_identities(operation, renaming)?))
            }
            _ => unreachable!("member and mixed operations are handled by the canonical core operation family"),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        dispatch_operation!(self, render, formatter, indentation)
    }
}

impl<Constant, C> PartiallyEvaluatableOperation<C> for XlaOperation<Constant>
where
    Constant: PartialEq
        + Value<Type = ArrayProgramType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + Concretizable<bool>,
    C: Context<Type = ArrayProgramType, Constant = Constant, Operation = XlaOperation<Constant>>,
    C::Value: PartialEq,
    ArrayProgramOperation<Constant::Projected>: PartiallyEvaluatableOperation<C>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        if let Some(operation) = self.to_core_operation() {
            return operation.partially_evaluate(context, driver, inputs);
        }
        match self {
            Self::Condition(operation) => operation.partially_evaluate(context, driver, inputs),
            Self::Scan(operation) => operation.partially_evaluate(context, driver, inputs),
            Self::While(operation) => operation.partially_evaluate(context, driver, inputs),
            Self::JitCall(operation) => operation.partially_evaluate(context, driver, inputs),
            Self::ShardMap(operation) => operation.partially_evaluate(context, driver, inputs),
            _ => context.fold_or_residualize(
                self.clone(),
                driver.regions().map(|region| region.to_program()).collect(),
                inputs,
            ),
        }
    }
}

impl<Constant, C> DifferentiableOperation<C> for XlaOperation<Constant>
where
    Constant: PartialEq
        + Value<Type = ArrayProgramType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + Concretizable<bool>,
    C: Context<Type = ArrayProgramType, Constant = Constant, Operation = XlaOperation<Constant>> + Zero<C::Value>,
    C::Value: Concretizable<bool>,
    ArrayProgramOperation<Constant::Projected>: DifferentiableOperation<C>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if let Some(operation) = self.to_core_operation() {
            return operation.jvp(context, driver, inputs);
        }
        match self {
            Self::Condition(operation) => operation.jvp(context, driver, inputs),
            Self::Scan(operation) => operation.jvp(context, driver, inputs),
            Self::While(operation) => operation.jvp(context, driver, inputs),
            Self::LinearCall(operation) => operation.jvp(context, driver, inputs),
            Self::JitCall(operation) => operation.jvp(context, driver, inputs),
            Self::ShardMap(operation) => operation.jvp(context, driver, inputs),
            _ => Err(ProgramError::UnsupportedOperation {
                message: format!("operation '{}' has no differentiation rule", self.name()),
            }
            .into()),
        }
    }
}

impl<V> TransposableOperation<V, XlaOperation<V>> for XlaOperation<V>
where
    V: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    ArrayProgramOperation<V::Projected>: TransposableOperation<V, XlaOperation<V>>,
{
    fn transpose<D: TranspositionDriver<V, XlaOperation<V>>>(
        &self,
        context: &mut TracingContext<V, XlaOperation<V>>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, DifferentiationError> {
        if let Some(operation) = self.to_core_operation() {
            return operation.transpose(context, driver, inputs, outputs);
        }
        match self {
            Self::Condition(operation) => operation.transpose(context, driver, inputs, outputs),
            Self::Scan(operation) => operation.transpose(context, driver, inputs, outputs),
            Self::While(operation) => operation.transpose(context, driver, inputs, outputs),
            Self::LinearCall(operation) => operation.transpose(context, driver, inputs, outputs),
            Self::JitCall(operation) => operation.transpose(context, driver, inputs, outputs),
            _ => Err(ProgramError::UnsupportedOperation {
                message: format!("operation '{}' is not transposable", self.name()),
            }
            .into()),
        }
    }
}

/// Staged XLA program specialized to the backend-owned XLA op universe.
pub type XlaProgram<Input, Output> = Program<XlaConstant, XlaOperation, Input, Output>;

/// Program builder specialized to the backend-owned XLA op universe.
pub type XlaProgramBuilder = ProgramBuilder<XlaConstant, XlaOperation>;

/// Flat XLA program over the backend-owned operation universe, used for materialized regions and shared callees.
pub type FlatXlaProgram = XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>;

/// Staged call to a flat jitted XLA program. The callee program is not part of this payload: it is a shared
/// callee root [`Region`](ryft_core::Region) attached to the [`Instruction`](ryft_core::Instruction) applying the
/// operation (the single `["callee"]` slot), interned by [`Rc`] identity when the call is staged through the
/// [`BindingRegionDriver`](ryft_core::BindingRegionDriver) passed to [`Context::bind`], so repeated calls staged from
/// one function handle share one callee root and remain identity-comparable for call-site deduplication at lowering.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JitCallOperation;

impl JitCallOperation {
    /// Creates a staged jitted-call operation. The flat callee program is supplied as a shared region attachment to
    /// [`Context::bind`].
    #[inline]
    pub(crate) fn new() -> Self {
        Self
    }
}

impl CompiledCallOperation<XlaConstant> for XlaOperation {
    #[inline]
    fn compiled_call() -> Self {
        Self::JitCall(JitCallOperation::new())
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
            "'{operation_name}' expected {} input(s) but got {}",
            expected_types.len(),
            input_types.len(),
        )));
    }
    for (index, (expected, actual)) in expected_types.iter().zip(input_types).enumerate() {
        if expected != actual {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' input #{index} expected {expected} but got {actual}",
            )));
        }
    }
    Ok(())
}

impl<T: Type> Operation<T> for JitCallOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "jit_call"
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
                "jit_call expects 1 attached callee region but got {}",
                region_interfaces.len()
            )));
        }
        let callee_interface = &region_interfaces[0];
        ensure_call_input_types(<Self as Operation<T>>::name(self), callee_interface.input_types(), input_types)?;
        Ok(callee_interface.output_types().to_vec())
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
impl<V, C> PartiallyEvaluatableOperation<C> for JitCallOperation
where
    V: PartialEq
        + Value<Type = ArrayProgramType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + Concretizable<bool>,
    C: Context<Type = ArrayProgramType, Constant = V, Operation = XlaOperation<V>>,
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
        let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
        let partition = callee.partition(input_known.as_slice())?;
        // A trivial partition — one whose known program contains no instructions — hoists no work (its known side
        // can only forward known inputs as residual edges), so keep the original boundary and let the default
        // materialize those knowns directly as residual feeders.
        if partition.known_program().instructions().is_empty() {
            return context.fold_or_residualize(XlaOperation::JitCall(*self), vec![callee.to_program()], inputs);
        }
        context.inline_partitioned_program(
            partition,
            inputs,
            |known_program| (XlaOperation::JitCall(JitCallOperation::new()), vec![known_program]),
            |residual_program| (XlaOperation::JitCall(JitCallOperation::new()), vec![residual_program]),
        )
    }
}

/// Batching rule for [`JitCallOperation`]: the callee region is rebatched over the mapped input axes (via
/// [`BatchingDriver::batch_program`]) and the batched call is bound through `context.parent()` with the
/// batched callee re-attached. An eager
/// client-backed parent (e.g., [`XlaDomain`](crate::XlaDomain)) compiles and executes the batched call immediately, a
/// staging parent stages it into the enclosing trace, and a differentiation parent dispatches it through its own
/// `jit_call` JVP rule — which is what serves `vmap` nested inside `gradient`/`linearize` closures.
impl<C> BatchableOperation<C, ArrayBatching> for JitCallOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<JitCallOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching>>(
        &self,
        context: &BatchingContext<C, ArrayBatching>,
        driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        // Rebatch the callee region over the mapped input axes when any input carries the batch axis; an
        // all-replicated call binds its original callee unchanged.
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
                .bind(*self, CalleeRegionDriver::new(&[Rc::new(batched_callee)]), &physical_inputs)?;
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output.r#type().into_owned(), output, batch_axis_from_position(axis)))
            .collect()
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
impl<C, V> DifferentiableOperation<C> for JitCallOperation
where
    C: Context<Type = ArrayProgramType, Constant = V, Operation = XlaOperation<V>> + Zero<C::Value>,
    V: PartialEq
        + Value<Type = ArrayProgramType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
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

        // Linearize the callee capture-free. The primal sub-program produces `[outputs..., residuals...]` and the
        // tangent sub-program consumes `[input_tangents..., residuals...]`; the residual count is the number of
        // trailing outputs of the primal sub-program beyond the original callee outputs.
        let (primal_program, tangent_program, _) = callee.linearize()?.into_parts();

        // Wrap the primal sub-program in a fresh `jit_call` and bind it over the operand primals, recovering the
        // primal outputs followed by the residual values.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_call = XlaOperation::JitCall(JitCallOperation::new());
        let mut primal_call_outputs =
            context.bind(primal_call, CalleeRegionDriver::new(&[Rc::new(primal_program)]), &primal_operands)?;
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
        let mut tangent_operands = inputs
            .iter()
            .zip(input_types)
            .filter(|(_, input_type)| !input_type.tangent().is_zero_space())
            .map(|(input, _)| input.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_operands.extend(residuals);
        let tangent_call = XlaOperation::JitCall(JitCallOperation::new());
        let tangent_outputs =
            context.bind(tangent_call, CalleeRegionDriver::new(&[Rc::new(tangent_program)]), &tangent_operands)?;
        let tangent_output_count = output_types.iter().filter(|r#type| !r#type.tangent().is_zero_space()).count();
        check_count!("output", tangent_outputs, tangent_output_count, ProgramError);

        let mut tangent_outputs = tangent_outputs.into_iter();
        Ok(primal_outputs
            .into_iter()
            .zip(output_types)
            .map(|(primal, output_type)| {
                if output_type.tangent().is_zero_space() {
                    Ok(DifferentiationDual::new_with_zero_tangent(primal))
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
    V: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
>(
    context: &TracingContext<V, XlaOperation<V>>,
    cotangent: &MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>,
    output_type: &ArrayProgramType,
    inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
) -> Result<Tracer<TracingContext<V, XlaOperation<V>>>, ProgramError> {
    if let MaybeZero::Value(cotangent) = cotangent {
        return Ok(cotangent.clone());
    }

    let (operation, operands) = match output_type {
        ArrayProgramType::Array(array_type)
            if array_type.shape().dimensions().iter().any(|dimension| matches!(dimension, Dimension::Dynamic(_))) =>
        {
            // The generic input-free provider cannot construct a reference-bearing array zero. Resolve each dynamic
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
                                ArrayProgramType::Dimension(r#type) if r#type.variable() == variable
                            )
                        })
                        .cloned()
                        .ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "cannot materialize dynamic transpose cotangent of type {output_type} because no known \
                                 dimension input defines '{variable}'",
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
    V: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    D: TranspositionDriver<V, XlaOperation<V>>,
>(
    _operation: &JitCallOperation,
    context: &mut TracingContext<V, XlaOperation<V>>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, ProgramError> {
    // A jitted call with no live output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect());
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

    // Transpose the callee with respect to its linear inputs. The transposed callee maps
    // `[outputs..., known_input_values...]` to `[linear_input_cotangents...]`, in callee-input order.
    let transposed_callee = driver.transpose_program(callee, operand_linear.as_slice())?;

    // Stage the output cotangents, materializing a typed zero for each structurally zero cotangent, then stage a fresh
    // `jit_call` over the transposed callee on `[outputs..., known_input_values...]`. Its outputs are the
    // linear-input cotangents.
    let output_types = callee.output_types();
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(output_types.len() + known_values.len());
    for (cotangent, output_type) in outputs.iter().zip(output_types.iter()) {
        operands.push(materialize_transpose_cotangent(context, cotangent, &output_type.cotangent(), inputs)?);
    }
    operands.extend(known_values);
    let transposed_call = XlaOperation::JitCall(JitCallOperation::new());
    let input_cotangents =
        context.bind(transposed_call, CalleeRegionDriver::new(&[Rc::new(transposed_callee)]), operands.as_slice())?;
    let linear_count = operand_linear.iter().filter(|&&linear| linear).count();
    check_count!("output", input_cotangents, linear_count, ProgramError);

    // Reassemble one cotangent per operand: the known operands carry structural zeros, while the linear input tangents
    // receive the transposed call's outputs in callee-input order.
    let mut input_cotangents = input_cotangents.into_iter().map(MaybeZero::Value);
    let cotangents = operand_linear
        .iter()
        .zip(inputs)
        .map(
            |(&linear, input)| {
                if linear { input_cotangents.next().unwrap() } else { MaybeZero::Zero(input.r#type().cotangent()) }
            },
        )
        .collect();
    Ok(cotangents)
}

/// Transpose rule for a primal tangent [`JitCallOperation`], forwarding to [`transpose_primal_jit_call`]. The callee
/// transposition happens on the concretely [`XlaConstant`]-keyed [`FlatXlaProgram`], so the recursion is resolved once
/// at definition time and instantiating this implementation introduces no recursive [`TransposableOperation`]
/// obligation on [`XlaOperation`].
impl<V> TransposableOperation<V, XlaOperation<V>> for JitCallOperation
where
    V: Value<Type = ArrayProgramType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
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
    use std::rc::Rc;

    use ryft_core::backends::array_programs::ArrayProgramOperation;
    use ryft_core::backends::arrays::ArrayOperation;
    use ryft_core::contexts::StagingContext;
    use ryft_core::differentiation::{DifferentiableType, DifferentiationError, TranspositionDriver};
    use ryft_core::operations::constants::ZeroOperation;
    use ryft_core::operations::control_flow::{ConditionOperation, ScanOperation, WhileOperation};
    use ryft_core::operations::dimensions::DimensionFromScalarOperation;
    use ryft_core::operations::manipulation::BroadcastOperation;
    use ryft_core::operations::math::{AddOperation, MulOperation};
    use ryft_core::parameters::Placeholder;
    use ryft_core::partial::PartialValue;
    use ryft_core::programs::MaybeZero;
    use ryft_core::programs::ProgramBuilder;
    use ryft_core::programs::effects::Effects;
    use ryft_core::programs::operations::Operation;
    use ryft_core::programs::regions::{EmptyRegionDriver, RegionDriver, RegionInterface, RegionRef};
    use ryft_core::programs::types::Typed;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tracing::TracingContext;
    use ryft_core::types::{
        ArrayProgramType, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape,
    };

    use super::{
        JitCallOperation, XlaArrayConstant, XlaConstant, XlaOperation, XlaProgram, XlaProgramBuilder,
        materialize_transpose_cotangent, transpose_primal_jit_call,
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
        ) -> Result<XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, DifferentiationError> {
            Ok(self.transposed.clone())
        }
    }

    fn vector_type() -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)]))
    }

    #[test]
    fn test_core_composite_control_flow_promotes_to_xla_control_flow() {
        let condition: XlaOperation<XlaConstant> =
            ArrayProgramOperation::<XlaArrayConstant>::Condition(ConditionOperation::new()).into();
        assert!(matches!(condition, XlaOperation::Condition(_)));
        assert_eq!(condition.region_slots(), ConditionOperation::<XlaConstant>::new().region_slots());

        let r#while: XlaOperation<XlaConstant> =
            ArrayProgramOperation::<XlaArrayConstant>::While(WhileOperation::new().with_iteration_bound(3).unwrap())
                .into();
        assert!(matches!(r#while, XlaOperation::While(operation) if operation.iteration_bound() == Some(3)));

        let scan: XlaOperation<XlaConstant> =
            ArrayProgramOperation::<XlaArrayConstant>::Scan(ScanOperation::new(2, 5).with_reverse(true)).into();
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
    fn test_jit_call_supports_composite_region_boundaries() {
        let dimension_type = ArrayProgramType::Dimension(DimensionType::new(DimensionVariable::new(
            "size",
            DimensionBounds::positive(Some(9)).unwrap(),
        )));
        let array_type = ArrayProgramType::Array(vector_type());
        let interface = RegionInterface::new(
            vec![dimension_type.clone()],
            vec![array_type.clone(), dimension_type.clone()],
            Effects::PURE,
        );
        let operation = JitCallOperation::new();

        assert_eq!(
            Operation::<ArrayProgramType>::infer_output_types(
                &operation,
                std::slice::from_ref(&dimension_type),
                std::slice::from_ref(&interface),
            )
            .unwrap(),
            vec![array_type, dimension_type],
        );
    }

    #[test]
    fn test_jit_call_jvp_omits_zero_space_boundary_tangents() {
        let dimension_type = ArrayProgramType::Dimension(DimensionType::new(DimensionVariable::new(
            "size",
            DimensionBounds::positive(Some(9)).unwrap(),
        )));
        let array_type = ArrayProgramType::Array(ArrayType::scalar(DataType::F64));

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
            .add_instruction(XlaOperation::JitCall(JitCallOperation::new()), vec![callee], vec![dimension, array])
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
                .add_instruction(BroadcastOperation::new(Vec::new()), Vec::new(), vec![scalar, extent])
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
            .add_instruction(DimensionFromScalarOperation::new(extent), Vec::new(), vec![integer])
            .unwrap()[0];
        let output = source_builder
            .add_instruction(
                XlaOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, gateway, scalar],
            )
            .unwrap()[0];
        let source = source_builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(source.output_types(), vec![ArrayProgramType::Array(output_type)]);

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
                assert_eq!(interface.input_types()[0], ArrayProgramType::Dimension(dimension.clone()));
                let output = <&ArrayType>::try_from(&interface.output_types()[0]).unwrap();
                assert_eq!(output.shape().dimensions(), &[Dimension::Dynamic(dimension.variable().clone())],);
            }
        }

        destination
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![first, second],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
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
        let expected = ArrayProgramType::Array(tangent_type.cotangent());
        let tangent_type = ArrayProgramType::Array(tangent_type);
        let mut context = TracingContext::<XlaConstant, XlaOperation>::new();
        let cotangents = transpose_primal_jit_call(
            &JitCallOperation::new(),
            &mut context,
            &EmptyRegionDriver,
            &[PartialValue::Unknown(tangent_type.clone())],
            &[MaybeZero::Zero(tangent_type.clone())],
        )
        .unwrap();
        assert!(matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &expected));

        let known = context.input(tangent_type.clone());
        let cotangents = transpose_primal_jit_call(
            &JitCallOperation::new(),
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
        let output_type = ArrayProgramType::Array(ArrayType::new(
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
        let value_program_type = ArrayProgramType::Array(value_type.clone());
        let predicate_program_type = ArrayProgramType::Array(predicate_type.clone());
        let source = {
            let mut builder = XlaProgramBuilder::new();
            let value = builder.add_input(value_program_type.clone());
            let predicate = builder.add_constant(XlaConstant::new(0, predicate_program_type.clone()));
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
            let _predicate_cotangent = builder.add_input(ArrayProgramType::Array(predicate_type.cotangent()));
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
            &JitCallOperation::new(),
            &mut context,
            &driver,
            &[PartialValue::Unknown(value_program_type.clone())],
            &[MaybeZero::Value(value_cotangent), MaybeZero::Zero(ArrayProgramType::Array(predicate_type.cotangent()))],
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
        use ryft_core::contexts::StagingContext;
        use ryft_core::partial::{PartialEvaluationInput, PartialEvaluationOutput};
        use ryft_core::tracing::TracingContext;

        let r#type = ArrayProgramType::Array(vector_type());

        // Callee `f(a, x) = (a + c, x * c, (a + c) * x)` over a known `a`, an unknown `x`, and a literal `c`.
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            let literal = builder.add_constant(XlaConstant::new(0, r#type.clone()));
            let shifted = builder.add_instruction(AddOperation, Vec::new(), vec![known_input, literal]).unwrap()[0];
            let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![runtime_input, literal]).unwrap()[0];
            let product = builder.add_instruction(MulOperation, Vec::new(), vec![shifted, runtime_input]).unwrap()[0];
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
        let callee_region = builder.intern_callee(&Rc::new(callee), None).unwrap();
        let call = XlaOperation::JitCall(JitCallOperation::new());
        let outputs = builder
            .add_instruction(call, vec![callee_region], vec![known_input, runtime_input])
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
            assert!(known_callee.atoms().iter().any(|atom| atom.is_constant()));
        }

        // The unknown half stayed behind one residual `jit_call` over the unknown input plus the residual edge, with
        // the literal rebuilt inline from its original payload.
        assert_eq!(evaluation.program().instructions().len(), 1);
        let residual_instruction = &evaluation.program().instructions()[0];
        assert!(
            matches!(residual_instruction.operation(), XlaOperation::JitCall(_)),
            "expected the residual program to contain the residual jit_call",
        );
        let residual_callee = evaluation.program().region_ref(residual_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(residual_callee.input_ids().len(), 2);
        assert_eq!(residual_callee.instructions().len(), 2);
        assert!(residual_callee.atoms().iter().any(|atom| atom.is_constant()));

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
    fn test_rematerialization_policies_are_available_for_the_xla_operation_family() {
        use ryft_core::tracing_v2::{
            DotsSaveable, DotsWithNoBatchDimsSaveable, EverythingSaveable, NothingSaveable, OffloadDotsWithNoBatchDims,
            RematerializationPolicy, SaveAndOffloadOnlyTheseNames, SaveFromBothPolicies, SaveOnlyTheseNames,
        };
        use ryft_core::types::Memory;

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
