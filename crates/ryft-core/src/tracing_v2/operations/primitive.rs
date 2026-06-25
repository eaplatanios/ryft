//! Reusable staged operation enums for built-in primitives.
//!
//! [`ArrayOperation`] and [`LinearArrayOperation`] contain the core operations implemented by `ryft-core`. Backends
//! that need additional operations should define their own operation enum that wraps these core enums together with
//! backend-specific variants, so transform, interpretation, and lowering rules remain statically typed and owned by
//! the backend that understands each operation.

use std::fmt::Debug;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};

use ryft_macros::{Operation, TransposableOperation};

use crate::batching::BatchingError;
use crate::contexts::{EagerContext, StagingContext};
use crate::domains::Domain;
use crate::macros::check_count;
use crate::operations::arithmetic::{
    AddOperation, DivOperation, MulOperation, NegOperation, Scale, ScaleOperation, SubOperation,
};
use crate::operations::compare::{Compare, CompareOperation};
use crate::operations::constants::{
    ConstantOperation, Fill, FillOperation, MaybeZeroOperation, OneLike, OneLikeOperation, OneOperation, Zero,
    ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::scan::read_scan_lane;
use crate::operations::control_flow::{ConditionOperation, ScanOperation, Select, SelectOperation, WhileOperation};
use crate::operations::differentiation::StopGradientOperation;
use crate::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, Concatenate, ConcatenateOperation, DynamicSlice, DynamicSliceOperation,
    DynamicUpdateSlice, DynamicUpdateSliceOperation, Gather, GatherOperation, LinearDynamicSliceOperation,
    LinearDynamicUpdateSliceOperation, LinearGatherOperation, LinearScatterAddOperation, Pad, PadOperation, Reshape,
    ReshapeOperation, Scatter, ScatterOperation, Slice, SliceOperation, Transpose, TransposeOperation, UpdateSlice,
    UpdateSliceOperation,
};
use crate::operations::sharding::{ConstrainSharding, Reshard, ReshardOperation, ShardingConstraintOperation};
use crate::operations::trigonometric::{Cos, CosOperation, Sin, SinOperation};
use crate::operations::{BooleanLike, Operation};
use crate::parameters::Parameterized;
use crate::payloads::{Captured, Input};
use crate::programs::{AtomId, ProgramError, Value};
use crate::tracing::Tracer;
use crate::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, BatchableProgramOperation, BatchingContext, ProgramBatchingContext,
    ProgramBatchingOutputAxes,
};
use crate::tracing_v2::differentiation::{
    CaptureParameterizedOperation, DifferentiationContext, JvpTracer, LinearOperationOf, LinearizableProgramOperation,
    LinearizationContextOf, NestedLinearization, TangentContext,
};
use crate::tracing_v2::operations::collective::CollectiveOperation;
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpCallOperation, CustomVjpOperation,
};
use crate::tracing_v2::operations::dot::{LeftDot, LeftDotOperation, MaybeDot, RightDot, RightDotOperation};
use crate::tracing_v2::operations::memory::{TransferToMemory, TransferToMemoryOperation};
use crate::tracing_v2::operations::recompute::RecomputeOperation;
use crate::tracing_v2::operations::reduce::ReduceOperation;
use crate::tracing_v2::operations::reshape::ReshapeOps;
use crate::tracing_v2::operations::select::LinearSelectOperation;
use crate::tracing_v2::operations::{DotDimensionNumbers, DotOperation, Reduce};
use crate::tracing_v2::rematerialization::{MaybeRematerializationName, RematerializationNameOperation};
use crate::tracing_v2::{DifferentiableOperation, ResidualizedOperation, ValueOrCapture};
use crate::types::{ArrayType, Typed};

use super::captures::MaterializeCaptureOperation;
use super::control_flow::{
    DefactorizableProgramOperation, DefactorizedOperation, batch_condition_with_interpreter,
    batch_while_with_interpreter, defactorize_operation_default,
};
use super::dot::DotOps;

/// Reusable operation enum for ordinary staged programs.
///
/// [`ArrayOperation`] is the ordinary operation enum for core tests and backend crates. Most variants are thin tags
/// around one semantic primitive defined elsewhere in [`super`].
///
/// Each variant wraps exactly the backing operation struct that owns the variant's semantics (type inference,
/// rendering, and interpretation): for example [`Zero`](Self::Zero) wraps a [`ZeroOperation`],
/// [`Scale`](Self::Scale) a [`ScaleOperation`], and [`Dot`](Self::Dot) a [`DotOperation`].
#[derive(Clone, Debug, Operation)]
#[ryft(bounds(interpretation(BooleanLike + Slice + UpdateSlice + Reshape)))]
pub enum ArrayOperation<V: Value<ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<ArrayType, V>),
    Fill(FillOperation<ArrayType, f64>),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Scale(ScaleOperation<ArrayType, V>),
    Mul(MulOperation),
    Div(DivOperation),
    Sin(SinOperation),
    Cos(CosOperation),
    StopGradient(StopGradientOperation),
    RematerializationName(RematerializationNameOperation),
    TransferToMemory(TransferToMemoryOperation),
    Dot(DotOperation),
    Transpose(TransposeOperation),
    Reshape(ReshapeOperation),
    Reshard(ReshardOperation),
    ShardingConstraint(ShardingConstraintOperation),
    Broadcast(BroadcastOperation),
    Slice(SliceOperation),
    UpdateSlice(UpdateSliceOperation),
    DynamicSlice(DynamicSliceOperation),
    DynamicUpdateSlice(DynamicUpdateSliceOperation),
    Pad(PadOperation),
    Concatenate(ConcatenateOperation),
    Gather(GatherOperation),
    Scatter(ScatterOperation),
    Reduce(ReduceOperation),
    Compare(CompareOperation),
    Not(NotOperation),
    And(AndOperation),
    Or(OrOperation),
    Xor(XorOperation),
    Collective(CollectiveOperation),
    Select(SelectOperation),
    Condition(Box<ConditionOperation<ArrayType, V, Self>>),
    While(Box<WhileOperation<ArrayType, V, Self>>),
    Scan(Box<ScanOperation<ArrayType, V, Self>>),
    CustomJvp(Box<CustomJvpOperation<ArrayType, V, Self>>),
    CustomVjp(Box<CustomVjpOperation<ArrayType, V, Self>>),
}

// Differentiation (JVP) for the `ArrayOperation` sum type: each variant delegates to its backing operation's own
// `DifferentiableOperation` rule. The per-variant `<Payload>: DifferentiableOperation<D>` bounds cover the
// non-self-referential variants; the self-referential higher-order
// `Condition`/`While`/`Scan` and `CustomJvp`/`CustomVjp` arms resolve against this impl's assumed
// `Self: DifferentiableOperation<D>`. The remaining where-clause spells the leaf closure of value and
// linear-operation capabilities those per-variant rules require.
impl<
    C: DifferentiationContext<Type = ArrayType, Value: ZeroLike + BooleanLike>
        + Domain<Operation = ArrayOperation<<C as Domain>::Constant>>,
    BodyOperation,
> DifferentiableOperation<C> for ArrayOperation<C::Constant>
where
    ZeroOperation<ArrayType>: DifferentiableOperation<C>,
    ZeroLikeOperation: DifferentiableOperation<C>,
    OneOperation<ArrayType>: DifferentiableOperation<C>,
    OneLikeOperation: DifferentiableOperation<C>,
    ConstantOperation<ArrayType, C::Constant>: DifferentiableOperation<C>,
    FillOperation<ArrayType, f64>: DifferentiableOperation<C>,
    NegOperation: DifferentiableOperation<C>,
    AddOperation: DifferentiableOperation<C>,
    SubOperation: DifferentiableOperation<C>,
    ScaleOperation<ArrayType, C::Constant>: DifferentiableOperation<C>,
    MulOperation: DifferentiableOperation<C>,
    DivOperation: DifferentiableOperation<C>,
    SinOperation: DifferentiableOperation<C>,
    CosOperation: DifferentiableOperation<C>,
    StopGradientOperation: DifferentiableOperation<C>,
    RematerializationNameOperation: DifferentiableOperation<C>,
    TransferToMemoryOperation: DifferentiableOperation<C>,
    DotOperation: DifferentiableOperation<C>,
    TransposeOperation: DifferentiableOperation<C>,
    ReshapeOperation: DifferentiableOperation<C>,
    ReshardOperation: DifferentiableOperation<C>,
    ShardingConstraintOperation: DifferentiableOperation<C>,
    BroadcastOperation: DifferentiableOperation<C>,
    SliceOperation: DifferentiableOperation<C>,
    UpdateSliceOperation: DifferentiableOperation<C>,
    DynamicSliceOperation: DifferentiableOperation<C>,
    DynamicUpdateSliceOperation: DifferentiableOperation<C>,
    PadOperation: DifferentiableOperation<C>,
    ConcatenateOperation: DifferentiableOperation<C>,
    GatherOperation: DifferentiableOperation<C>,
    ScatterOperation: DifferentiableOperation<C>,
    ReduceOperation: DifferentiableOperation<C>,
    CompareOperation: DifferentiableOperation<C>,
    NotOperation: DifferentiableOperation<C>,
    AndOperation: DifferentiableOperation<C>,
    OrOperation: DifferentiableOperation<C>,
    XorOperation: DifferentiableOperation<C>,
    CollectiveOperation: DifferentiableOperation<C>,
    SelectOperation: DifferentiableOperation<C>,
    BodyOperation: Operation<ArrayType> + From<LinearSelectOperation<ValueOrCapture<ArrayType, C::Tangent>>>,
    LinearOperationOf<C>: From<AddOperation>
        + From<ZeroLikeOperation>
        + From<NegOperation>
        + From<SubOperation>
        + From<ScaleOperation<ArrayType, ValueOrCapture<ArrayType, C::Value>, Input>>
        + From<LeftDotOperation<ValueOrCapture<ArrayType, C::Value>, Input>>
        + From<RightDotOperation<ValueOrCapture<ArrayType, C::Value>, Input>>
        + From<TransposeOperation>
        + From<ReshapeOperation>
        + From<BroadcastOperation>
        + From<ReduceOperation>
        + From<PadOperation>
        + From<SliceOperation>
        + From<UpdateSliceOperation>
        + From<ReshardOperation>
        + From<ShardingConstraintOperation>
        + ResidualizedOperation<C>
        + From<
            CustomVjpCallOperation<
                ArrayType,
                C::Constant,
                ArrayOperation<C::Constant>,
                ValueOrCapture<ArrayType, C::Value>,
            >,
        > + From<TransferToMemoryOperation>
        + From<ConcatenateOperation>
        + From<LinearSelectOperation<ValueOrCapture<ArrayType, C::Value>>>
        + From<LinearDynamicSliceOperation<ValueOrCapture<ArrayType, C::Value>>>
        + From<LinearDynamicUpdateSliceOperation<ValueOrCapture<ArrayType, C::Value>>>
        + From<LinearGatherOperation<ValueOrCapture<ArrayType, C::Value>>>
        + From<LinearScatterAddOperation<ValueOrCapture<ArrayType, C::Value>>>
        + From<
            ConditionOperation<
                ArrayType,
                C::Tangent,
                LinearOperationOf<C>,
                ValueOrCapture<ArrayType, C::Value>,
                Captured,
            >,
        > + From<MaterializeCaptureOperation<ValueOrCapture<ArrayType, C::Value>>>
        + From<RecomputeOperation<ArrayOperation<C::Constant>>>
        + From<WhileOperation<ArrayType, C::Tangent, LinearOperationOf<C>, Input>>
        + From<ScanOperation<ArrayType, C::Tangent, BodyOperation, ValueOrCapture<ArrayType, C::Value>, Input>>,
    LinearOperationOf<C>: CaptureParameterizedOperation<
            ArrayType,
            ValueOrCapture<ArrayType, C::Value>,
            WithCapture<ValueOrCapture<ArrayType, C::Value>> = LinearOperationOf<C>,
            WithCapture<ValueOrCapture<ArrayType, C::Tangent>> = BodyOperation,
        > + DefactorizableProgramOperation<C::Tangent, C::Value, ArrayOperation<C::Constant>>,
    LinearOperationOf<C>: MaybeZeroOperation<ArrayType>,
    ArrayOperation<C::Constant>: Clone + LinearizableProgramOperation<C>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, C>,
        inputs: &[JvpTracer<'jvp, C>],
    ) -> Result<Vec<JvpTracer<'jvp, C>>, ProgramError>
    where
        C: 'jvp,
    {
        match self {
            Self::Zero(operation) => operation.jvp(context, inputs),
            Self::ZeroLike(operation) => operation.jvp(context, inputs),
            Self::One(operation) => operation.jvp(context, inputs),
            Self::OneLike(operation) => operation.jvp(context, inputs),
            Self::Constant(operation) => operation.jvp(context, inputs),
            Self::Fill(operation) => operation.jvp(context, inputs),
            Self::Neg(operation) => operation.jvp(context, inputs),
            Self::Add(operation) => operation.jvp(context, inputs),
            Self::Sub(operation) => operation.jvp(context, inputs),
            Self::Scale(operation) => operation.jvp(context, inputs),
            Self::Mul(operation) => operation.jvp(context, inputs),
            Self::Div(operation) => operation.jvp(context, inputs),
            Self::Sin(operation) => operation.jvp(context, inputs),
            Self::Cos(operation) => operation.jvp(context, inputs),
            Self::StopGradient(operation) => operation.jvp(context, inputs),
            Self::RematerializationName(operation) => operation.jvp(context, inputs),
            Self::TransferToMemory(operation) => operation.jvp(context, inputs),
            Self::Dot(operation) => operation.jvp(context, inputs),
            Self::Transpose(operation) => operation.jvp(context, inputs),
            Self::Reshape(operation) => operation.jvp(context, inputs),
            Self::Reshard(operation) => operation.jvp(context, inputs),
            Self::ShardingConstraint(operation) => operation.jvp(context, inputs),
            Self::Broadcast(operation) => operation.jvp(context, inputs),
            Self::Slice(operation) => operation.jvp(context, inputs),
            Self::UpdateSlice(operation) => operation.jvp(context, inputs),
            Self::DynamicSlice(operation) => operation.jvp(context, inputs),
            Self::DynamicUpdateSlice(operation) => operation.jvp(context, inputs),
            Self::Pad(operation) => operation.jvp(context, inputs),
            Self::Concatenate(operation) => operation.jvp(context, inputs),
            Self::Gather(operation) => operation.jvp(context, inputs),
            Self::Scatter(operation) => operation.jvp(context, inputs),
            Self::Reduce(operation) => operation.jvp(context, inputs),
            Self::Compare(operation) => operation.jvp(context, inputs),
            Self::Not(operation) => operation.jvp(context, inputs),
            Self::And(operation) => operation.jvp(context, inputs),
            Self::Or(operation) => operation.jvp(context, inputs),
            Self::Xor(operation) => operation.jvp(context, inputs),
            Self::Collective(operation) => operation.jvp(context, inputs),
            Self::Select(operation) => operation.jvp(context, inputs),
            Self::Condition(operation) => operation.jvp(context, inputs),
            Self::While(operation) => operation.jvp(context, inputs),
            Self::Scan(operation) => operation.jvp(context, inputs),
            Self::CustomJvp(operation) => operation.jvp(context, inputs),
            Self::CustomVjp(operation) => operation.jvp(context, inputs),
        }
    }
}

/// Reusable operation enum for staged linear programs.
///
/// [`LinearArrayOperation`] is the linear-program sibling of [`ArrayOperation`]. It contains operations that can
/// appear in tangent and cotangent programs, including captured-factor linear maps such as [`LeftDot`](Self::LeftDot)
/// and [`RightDot`](Self::RightDot), and the linearized higher-order operations needed by rematerialization and
/// control flow.
///
/// Each variant wraps exactly the backing operation struct that owns the variant's semantics (type inference,
/// rendering, and interpretation). The [`Operation`]/[`Display`] and per-variant [`From`]/[`TryFrom`] impls are
/// defined for [`ArrayType`].
///
/// The `V` parameter is the linear program's value and constant-table type. It instantiates to concrete tangent
/// values for eager linear execution and to tracers when one transform stages another. The `C` parameter is the
/// constant type of captured primal programs such as [`CustomVjpCall`](Self::CustomVjpCall), which are written over
/// context constants rather than over the linear program's tangent constants.
#[derive(Clone, Debug, Operation, TransposableOperation)]
#[ryft(bounds(interpretation(BooleanLike + Slice + UpdateSlice + Reshape)))]
pub enum LinearArrayOperation<
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    F: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<ArrayType, V, Input>),
    Fill(FillOperation<ArrayType, f64>),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Scale(ScaleOperation<ArrayType, F, Input>),
    Mul(MulOperation),
    TransferToMemory(TransferToMemoryOperation),
    Transpose(TransposeOperation),
    LeftDot(LeftDotOperation<F, Input>),
    RightDot(RightDotOperation<F, Input>),
    Reshape(ReshapeOperation),
    Reshard(ReshardOperation),
    ShardingConstraint(ShardingConstraintOperation),
    Broadcast(BroadcastOperation),
    Slice(SliceOperation),
    UpdateSlice(UpdateSliceOperation),
    DynamicSlice(LinearDynamicSliceOperation<F>),
    DynamicUpdateSlice(LinearDynamicUpdateSliceOperation<F>),
    Gather(LinearGatherOperation<F>),
    ScatterAdd(LinearScatterAddOperation<F>),
    Pad(PadOperation),
    Concatenate(ConcatenateOperation),
    Reduce(ReduceOperation),
    Select(LinearSelectOperation<F>),
    Residual(MaterializeCaptureOperation<F>),
    Recompute(RecomputeOperation<P>),
    Condition(ConditionOperation<ArrayType, V, Self, F, Captured>),
    WhileCondition(ConditionOperation<ArrayType, V, Self, V, Input>),
    While(Box<WhileOperation<ArrayType, V, Self, Input>>),
    Scan(Box<ScanOperation<ArrayType, V, LinearArrayOperation<V, C, ValueOrCapture<ArrayType, V>, P>, F, Input>>),
    CustomVjpCall(Box<CustomVjpCallOperation<ArrayType, C, P, F>>),
}

impl<V> MaybeRematerializationName for ArrayOperation<V>
where
    V: Value<ArrayType>,
{
    #[inline]
    fn rematerialization_name(&self) -> Option<&str> {
        match self {
            Self::RematerializationName(operation) => Some(operation.tag()),
            _ => None,
        }
    }
}

impl<V> MaybeDot for ArrayOperation<V>
where
    V: Value<ArrayType>,
{
    #[inline]
    fn dot_dimensions(&self) -> Option<&DotDimensionNumbers> {
        match self {
            Self::Dot(operation) => Some(operation.dimensions()),
            _ => None,
        }
    }
}

/// Shared payload-mapping core behind [`CaptureParameterizedOperation::try_map_captures`] for
/// [`LinearArrayOperation`].
fn map_linear_array_operation_factors<V, C, F, MappedFactor, P, MapFactorFn>(
    operation: &LinearArrayOperation<V, C, F, P>,
    map_factor: &mut MapFactorFn,
) -> Result<LinearArrayOperation<V, C, MappedFactor, P>, ProgramError>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    F: Value<ArrayType>,
    MappedFactor: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
    MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
{
    {
        match operation {
            LinearArrayOperation::CustomVjpCall(call) => {
                Ok(LinearArrayOperation::CustomVjpCall(Box::new(call.map_captures(map_factor)?)))
            }
            LinearArrayOperation::Zero(zero) => Ok(LinearArrayOperation::Zero(zero.clone())),
            LinearArrayOperation::One(one) => Ok(LinearArrayOperation::One(one.clone())),
            LinearArrayOperation::Constant(constant) => Ok(LinearArrayOperation::Constant(constant.clone())),
            LinearArrayOperation::Fill(fill) => Ok(LinearArrayOperation::Fill(fill.clone())),
            LinearArrayOperation::ZeroLike(operation) => Ok(LinearArrayOperation::ZeroLike(operation.clone())),
            LinearArrayOperation::OneLike(operation) => Ok(LinearArrayOperation::OneLike(operation.clone())),
            LinearArrayOperation::Add(operation) => Ok(LinearArrayOperation::Add(operation.clone())),
            LinearArrayOperation::Sub(operation) => Ok(LinearArrayOperation::Sub(operation.clone())),
            LinearArrayOperation::Neg(operation) => Ok(LinearArrayOperation::Neg(operation.clone())),
            LinearArrayOperation::Mul(operation) => Ok(LinearArrayOperation::Mul(operation.clone())),
            LinearArrayOperation::TransferToMemory(operation) => {
                Ok(LinearArrayOperation::TransferToMemory(operation.clone()))
            }
            LinearArrayOperation::Transpose(operation) => Ok(LinearArrayOperation::Transpose(operation.clone())),
            LinearArrayOperation::Scale(operation) => {
                Ok(LinearArrayOperation::Scale(ScaleOperation::new(map_factor(operation.factor())?)))
            }
            LinearArrayOperation::LeftDot(operation) => Ok(LinearArrayOperation::LeftDot(
                LeftDotOperation::new(map_factor(operation.factor())?, operation.dimensions().clone())
                    .with_output_sharding(operation.output_sharding().cloned()),
            )),
            LinearArrayOperation::RightDot(operation) => Ok(LinearArrayOperation::RightDot(
                RightDotOperation::new(map_factor(operation.factor())?, operation.dimensions().clone())
                    .with_output_sharding(operation.output_sharding().cloned()),
            )),
            LinearArrayOperation::Reshape(operation) => Ok(LinearArrayOperation::Reshape(operation.clone())),
            LinearArrayOperation::Reshard(operation) => Ok(LinearArrayOperation::Reshard(operation.clone())),
            LinearArrayOperation::ShardingConstraint(operation) => {
                Ok(LinearArrayOperation::ShardingConstraint(operation.clone()))
            }
            LinearArrayOperation::Broadcast(operation) => Ok(LinearArrayOperation::Broadcast(operation.clone())),
            LinearArrayOperation::Slice(operation) => Ok(LinearArrayOperation::Slice(operation.clone())),
            LinearArrayOperation::UpdateSlice(operation) => Ok(LinearArrayOperation::UpdateSlice(operation.clone())),
            LinearArrayOperation::Pad(operation) => Ok(LinearArrayOperation::Pad(operation.clone())),
            LinearArrayOperation::Concatenate(operation) => Ok(LinearArrayOperation::Concatenate(operation.clone())),
            LinearArrayOperation::DynamicSlice(operation) => {
                Ok(LinearArrayOperation::DynamicSlice(LinearDynamicSliceOperation::new(
                    operation.start_indices().iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
                    operation.sizes().to_vec(),
                )))
            }
            LinearArrayOperation::DynamicUpdateSlice(operation) => {
                Ok(LinearArrayOperation::DynamicUpdateSlice(LinearDynamicUpdateSliceOperation::new(
                    operation.start_indices().iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
                )))
            }
            LinearArrayOperation::Gather(operation) => Ok(LinearArrayOperation::Gather(LinearGatherOperation::new(
                operation.operation().clone(),
                map_factor(operation.indices())?,
            ))),
            LinearArrayOperation::ScatterAdd(operation) => Ok(LinearArrayOperation::ScatterAdd(
                LinearScatterAddOperation::new(operation.operation().clone(), map_factor(operation.indices())?),
            )),
            LinearArrayOperation::Reduce(operation) => Ok(LinearArrayOperation::Reduce(operation.clone())),
            LinearArrayOperation::Select(operation) => {
                Ok(LinearArrayOperation::Select(LinearSelectOperation::new(map_factor(operation.condition())?)))
            }
            LinearArrayOperation::Residual(operation) => {
                Ok(LinearArrayOperation::Residual(MaterializeCaptureOperation::new(map_factor(operation.capture())?)))
            }
            LinearArrayOperation::Recompute(operation) => Ok(LinearArrayOperation::Recompute(operation.clone())),
            LinearArrayOperation::Condition(operation) => {
                Ok(LinearArrayOperation::Condition(ConditionOperation::new_captured(
                    map_factor(operation.predicate())?,
                    operation.true_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                    })?,
                    operation.false_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                    })?,
                )?))
            }
            // While-condition branches carry only closed constant factors after defactorization, but the traversal
            // stays total over them like the factor-form variant's.
            LinearArrayOperation::WhileCondition(operation) => {
                Ok(LinearArrayOperation::WhileCondition(ConditionOperation::new(
                    operation.true_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                    })?,
                    operation.false_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                    })?,
                )?))
            }
            LinearArrayOperation::While(while_operation) => {
                let condition = while_operation.condition().map_operations(|operation| {
                    map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                })?;
                let body = while_operation.body().map_operations(|operation| {
                    map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                })?;
                Ok(LinearArrayOperation::While(Box::new(
                    WhileOperation::new(condition, body)?.with_iteration_bound(while_operation.iteration_bound())?,
                )))
            }
            // The scan body's factor space is scan-local (references index `residual_stacks` per lane), so enclosing
            // factor passes map only the stack payloads and clone the body-internal factors unchanged.
            LinearArrayOperation::Scan(operation) => {
                // The factor-cloning function is passed as a `fn` pointer (not a closure) so the recursive
                // monomorphization below reaches a fixed point: nested scans reuse the exact same mapper
                // instantiation instead of minting a fresh closure type per level.
                let mut clone_scan_local_factor: fn(
                    &ValueOrCapture<ArrayType, V>,
                )
                    -> Result<ValueOrCapture<ArrayType, V>, ProgramError> = |factor| Ok(factor.clone());
                let body = operation.body().map_operations(|operation| {
                    map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, &mut clone_scan_local_factor)
                })?;
                let scan = ScanOperation::<ArrayType, _, _, MappedFactor, Input>::new_with_payload(
                    body,
                    operation.carry_count(),
                    operation.length(),
                )?
                .with_reverse(operation.reverse())
                .with_unroll(operation.unroll())?
                .with_captures(operation.captures().iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?);
                Ok(LinearArrayOperation::Scan(Box::new(scan)))
            }
        }
    }
}

// TODO(eaplatanios): Can we get rid of this similar to what we did for some of the scan-related functionality?
impl<V, C, F, P> CaptureParameterizedOperation<ArrayType, F> for LinearArrayOperation<V, C, F, P>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    F: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
{
    type WithCapture<MappedFactor: Value<ArrayType>> = LinearArrayOperation<V, C, MappedFactor, P>;

    fn try_map_captures<MappedFactor: Value<ArrayType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithCapture<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        map_linear_array_operation_factors::<_, _, _, _, _, _>(self, map_factor)
    }
}

// TODO(eaplatanios): Fold this into our `TransposableOperation` derive macro.
impl<V, C, R, P> DefactorizableProgramOperation<V, R, P> for LinearArrayOperation<V, C, ValueOrCapture<ArrayType, R>, P>
where
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    V::InterpretationContext: Zero<ArrayType, V>,
    C: Value<ArrayType>,
    R: Value<ArrayType>,
    P: Clone
        + Operation<ArrayType>
        + From<MulOperation>
        + From<DotOperation>
        + From<SelectOperation>
        + From<DynamicSliceOperation>
        + From<DynamicUpdateSliceOperation>,
{
    fn defactorize_operation(
        &self,
        residual_atoms: &[AtomId],
        inputs: Vec<AtomId>,
    ) -> Result<DefactorizedOperation<Self>, ProgramError> {
        defactorize_operation_default::<V, R, P, Self>(self, residual_atoms, inputs)
    }
}

/// Builds the common error for zero-input operation enum variants that must be handled by the staging path.
fn missing_zero_input_batch_rule(operation_enum: &str, kind: &str) -> ProgramError {
    BatchingError::UnsupportedOperation {
        message: format!(
            "{operation_enum}::{kind}: zero-input operations are lane-uniform by construction — stage them through the \
             active context, which handles the lane-uniform short-circuit, instead of invoking `batch` directly",
        ),
    }
    .into()
}

/// Dispatches non-control-flow [`ArrayOperation`] variants to their primitive batching rules.
///
/// Higher-order variants are intentionally returned as `None` so concrete impls can handle them with their specialized
/// recursive bounds instead of forcing the trait solver through one fully generic recursive operation impl.
fn batch_array_non_control_operation<F, V>(
    operation: &ArrayOperation<F>,
    context: &V::InterpretationContext,
    inputs: &[ArrayBatch<V>],
) -> Result<Option<Vec<ArrayBatch<V>>>, ProgramError>
where
    F: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    F::InterpretationContext: Zero<ArrayType, F>,
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + DotOps
        + ReshapeOps
        + Broadcast
        + Reduce
        + Pad
        + Concatenate
        + Slice
        + UpdateSlice
        + DynamicSlice
        + DynamicUpdateSlice
        + Gather
        + Scatter
        + Reshard
        + ConstrainSharding
        + Compare<Output = V>
        + BitAnd<Output = V>
        + BitOr<Output = V>
        + BitXor<Output = V>
        + Not<Output = V>
        + Select<Condition = V>,
    V::InterpretationContext: Scale<ArrayType, V, F>,
{
    let outputs = match operation {
        ArrayOperation::Add(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Sub(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Mul(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Div(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Neg(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Sin(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Cos(operation) => operation.batch(context, inputs)?,
        ArrayOperation::StopGradient(operation) => operation.batch(context, inputs)?,
        ArrayOperation::RematerializationName(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Select(operation) => operation.batch(context, inputs)?,
        ArrayOperation::ZeroLike(operation) => operation.batch(context, inputs)?,
        ArrayOperation::OneLike(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Scale(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Dot(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Transpose(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Reshape(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Reshard(operation) => operation.batch(context, inputs)?,
        ArrayOperation::ShardingConstraint(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Broadcast(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Slice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::UpdateSlice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::DynamicSlice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::DynamicUpdateSlice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Pad(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Concatenate(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Gather(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Scatter(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Reduce(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Compare(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Not(operation) => operation.batch(context, inputs)?,
        ArrayOperation::And(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Or(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Xor(operation) => operation.batch(context, inputs)?,
        ArrayOperation::TransferToMemory(_)
        | ArrayOperation::Collective(_)
        | ArrayOperation::Condition(_)
        | ArrayOperation::While(_)
        | ArrayOperation::Scan(_)
        | ArrayOperation::CustomJvp(_)
        | ArrayOperation::CustomVjp(_) => return Ok(None),
        ArrayOperation::Zero(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "Zero")),
        ArrayOperation::One(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "One")),
        ArrayOperation::Constant(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "Constant")),
        ArrayOperation::Fill(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "Fill")),
    };
    Ok(Some(outputs))
}

// TODO(eaplatanios): Why does this not simply forward to per-variant `BatchableOperation::batch` calls. It should.
/// Blanket active batching impl for the [`ArrayOperation`] sum type over a staged tracer context: each non-control
/// variant delegates to its backing operation's batching rule (shared with the eager impl through
/// [`batch_array_non_control_operation`]), while the lane-uniform memory transfer, the named-axis collective, and the
/// higher-order control-flow variants are handled by their specialized recursive rules.
impl<C, V> BatchableOperation<Tracer<C>, BatchingContext<C>> for ArrayOperation<V>
where
    C: StagingContext<Type = ArrayType, Constant = V, Operation = ArrayOperation<V>>,
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    V::InterpretationContext: Zero<ArrayType, V>,
    C::Operation: From<CollectiveOperation> + From<FillOperation<ArrayType, f64>>,
    Tracer<C>: Add<Output = Tracer<C>>
        + Sub<Output = Tracer<C>>
        + Neg<Output = Tracer<C>>
        + Mul<Output = Tracer<C>>
        + Div<Output = Tracer<C>>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + DotOps
        + ReshapeOps
        + Broadcast
        + Reduce
        + Pad
        + Concatenate
        + Slice
        + UpdateSlice
        + DynamicSlice
        + DynamicUpdateSlice
        + Gather
        + Scatter
        + Reshard
        + ConstrainSharding
        + Compare<Output = Tracer<C>>
        + BitAnd<Output = Tracer<C>>
        + BitOr<Output = Tracer<C>>
        + BitXor<Output = Tracer<C>>
        + Not<Output = Tracer<C>>
        + Select<Condition = Tracer<C>>
        + BooleanLike
        + Transpose,
    Vec<Tracer<C>>: Parameterized<Tracer<C>, To<Tracer<C>> = Vec<Tracer<C>>, ParameterStructure: Debug + PartialEq>,
    Self: BatchableProgramOperation<V>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        if let Some(outputs) = batch_array_non_control_operation(self, context.parent_context(), inputs)? {
            return Ok(outputs);
        }
        match self {
            // Memory placement is lane-uniform: the same transfer applies to every lane, so it is staged unchanged on
            // the physical batched value in the parent context and the lane axis is preserved.
            Self::TransferToMemory(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let tracer = inputs[0].value().transfer_to_memory(operation.destination());
                let physical_type = tracer.r#type().into_owned();
                Ok(vec![ArrayBatch::new(physical_type, tracer, inputs[0].batch_axis())?])
            }
            // The staged collective rule owns named-axis resolution: it consumes the lane axis when this context's
            // axis name matches and forwards the collective to the parent context otherwise.
            Self::Collective(operation) => operation.batch(context, inputs),
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::Scan(scan) => scan.batch(context, inputs),
            Self::CustomJvp(operation) => operation.batch(context, inputs),
            Self::CustomVjp(operation) => operation.batch(context, inputs),
            _ => unreachable!("non-control-flow ArrayOperation variants are handled above"),
        }
    }
}

/// Blanket value-level batching impl for the [`ArrayOperation`] sum type.
impl<V> BatchableOperation<V, EagerContext<ArrayType, V, ArrayOperation<V>>> for ArrayOperation<V>
where
    V::InterpretationContext: Default,
    V::InterpretationContext: Scale<ArrayType, V, V>,
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + DotOps
        + ReshapeOps
        + Broadcast
        + Reduce
        + Pad
        + Concatenate
        + Slice
        + UpdateSlice
        + DynamicSlice
        + DynamicUpdateSlice
        + Gather
        + Scatter
        + Reshard
        + ConstrainSharding
        + Compare<Output = V>
        + BitAnd<Output = V>
        + BitOr<Output = V>
        + BitXor<Output = V>
        + Not<Output = V>
        + Select<Condition = V>
        + BooleanLike,
    V::InterpretationContext: Zero<ArrayType, V>,
    EagerContext<ArrayType, V, ArrayOperation<V>>: Zero<ArrayType, V> + Fill<ArrayType, f64, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &EagerContext<ArrayType, V, ArrayOperation<V>>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let interpretation_context = V::InterpretationContext::default();
        if let Some(outputs) = batch_array_non_control_operation(self, &interpretation_context, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::TransferToMemory(_) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(inputs.to_vec())
            }
            Self::Collective(operation) => operation.batch(context, inputs),
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::Scan(scan) => scan.batch(context, inputs),
            Self::CustomJvp(operation) => operation.batch(context, inputs),
            Self::CustomVjp(operation) => operation.batch(context, inputs),
            _ => unreachable!("non-control-flow ArrayOperation variants are handled above"),
        }
    }
}

// TODO(eaplatanios): Fold this into our `BatchableOperation` derive macro.
/// Blanket active batching impl for the [`ArrayOperation`] sum type.
///
/// The `Operation = Self` projection equality and the
/// [`BatchableProgramOperation`](crate::tracing_v2::batching::BatchableProgramOperation) / lane-alignment bounds exist
/// for the custom-derivative arms: their re-wrapping batch rules batch the captured programs and stage a new
/// custom-derivative call into the parent context, which is only expressible when the staged operation type is this
/// enum itself. Both extra bounds are leaf obligations (a structural type equality and a closed-enum capability
/// whose impl carries no batching-context obligations of its own), so instantiating this impl never recurses into
/// another batching-context obligation.
/// Program-level batching for the [`ArrayOperation`] sum type, backing the re-wrapping `batch` rules of
/// [`CustomJvpOperation`] and [`CustomVjpOperation`]; see
/// [`BatchableProgramOperation`](crate::tracing_v2::batching::BatchableProgramOperation).
///
/// The where clauses here are deliberately the *leaf* closure of what `batch_program::<V, Self>` needs — the
/// blanket traced batching impl's bounds instantiated at [`ProgramBatchingContext`] — rather than the
/// `Self: BatchableOperation<..>` bound itself. Spelling out the leaves keeps instantiating this impl free of
/// batching-context obligations, which is what lets the traced batching impl require
/// `Self: BatchableProgramOperation<..>` without sending the trait solver into an unbounded
/// batching-context recursion.
impl<V> BatchableProgramOperation<V> for ArrayOperation<V>
where
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape + 'static,
    V::InterpretationContext: Zero<ArrayType, V>,
    Tracer<ProgramBatchingContext<V, Self>>: Add<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + Sub<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + Neg<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + Mul<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + Div<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + DotOps
        + ReshapeOps
        + Broadcast
        + Reduce
        + Pad
        + Concatenate
        + Slice
        + UpdateSlice
        + DynamicSlice
        + DynamicUpdateSlice
        + Gather
        + Scatter
        + Reshard
        + ConstrainSharding
        + Compare<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + BitAnd<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + BitOr<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + BitXor<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + Not<Output = Tracer<ProgramBatchingContext<V, Self>>>
        + Select<Condition = Tracer<ProgramBatchingContext<V, Self>>>
        + BooleanLike
        + Transpose,
    Vec<Tracer<ProgramBatchingContext<V, Self>>>: Parameterized<
            Tracer<ProgramBatchingContext<V, Self>>,
            To<Tracer<ProgramBatchingContext<V, Self>>> = Vec<Tracer<ProgramBatchingContext<V, Self>>>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    fn batch_program(
        program: &crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        axis_size: usize,
        input_batch_axes: &[Option<usize>],
        output_batch_axes: ProgramBatchingOutputAxes,
    ) -> Result<(crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>, Vec<Option<usize>>), ProgramError> {
        crate::tracing_v2::batching::batch_program::<V, Self>(program, axis_size, input_batch_axes, output_batch_axes)
    }
}

/// Nested symbolic linearization for the [`ArrayOperation`] sum type, backing the staged-condition JVP rule of
/// [`ConditionOperation`]; see [`LinearizableProgramOperation`](crate::tracing_v2::LinearizableProgramOperation).
///
/// The where clauses here are deliberately the *leaf* closure of what
/// [`Program::linearize`] needs — the generic JVP
/// dispatch impl's bounds instantiated at [`LinearizationContextOf`] — rather than the
/// `Self: DifferentiableOperation<LinearizationContextOf<E, Self>>` bound itself. Spelling out the leaves keeps
/// instantiating this impl free of derived-context differentiation obligations (the recursive obligation is
/// discharged once, as a definition-time body check), which is what lets the JVP dispatch impl require
/// `Self: LinearizableProgramOperation<E>` without sending the trait solver into an unbounded nested-context
/// recursion. The `WithCapture<V> = ..` equality pins the canonical linear operation as a fixed point of factor
/// reparameterization, which is what collapses `LinearizationContextOf<LinearizationContextOf<E, ..>, ..>`
/// to `LinearizationContextOf<E, ..>` and keeps the obligations finite for nested conditions.
impl<V: Value<ArrayType>, C: DifferentiationContext<Type = ArrayType, Constant = V>, BodyOperation>
    LinearizableProgramOperation<C> for ArrayOperation<V>
where
    C::Tangent: Transpose + Broadcast + Reduce + Slice + Reshard + ConstrainSharding,
    C::LinearOperation<C::Tangent, V>:
        CaptureParameterizedOperation<ArrayType, V, WithCapture<V> = C::LinearOperation<C::Tangent, V>>,
    // ZeroOperation<ArrayType>: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // ZeroLikeOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // OneOperation<ArrayType>: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // OneLikeOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // ConstantOperation<ArrayType, V>: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // FillOperation<ArrayType, f64>: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // NegOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // AddOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // SubOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // ScaleOperation<ArrayType, V>: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // MulOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // DivOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // SinOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // CosOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // StopGradientOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // RematerializationNameOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // TransferToMemoryOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // DotOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // TransposeOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // ReshapeOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // ReshardOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // ShardingConstraintOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // BroadcastOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // SliceOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // UpdateSliceOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // DynamicSliceOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // DynamicUpdateSliceOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // PadOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // ConcatenateOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // GatherOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // ScatterOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // ReduceOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // CompareOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // NotOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // AndOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // OrOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // XorOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // CollectiveOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    // SelectOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    BodyOperation: Operation<ArrayType> + From<LinearSelectOperation<ValueOrCapture<ArrayType, C::Tangent>>>,
    LinearOperationOf<LinearizationContextOf<C, Self>>: From<AddOperation>
        + From<ZeroLikeOperation>
        + From<NegOperation>
        + From<SubOperation>
        + From<ScaleOperation<ArrayType, ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>, Input>>
        + From<LeftDotOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>, Input>>
        + From<RightDotOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>, Input>>
        + From<TransposeOperation>
        + From<ReshapeOperation>
        + From<BroadcastOperation>
        + From<ReduceOperation>
        + From<PadOperation>
        + From<SliceOperation>
        + From<UpdateSliceOperation>
        + From<ReshardOperation>
        + From<ShardingConstraintOperation>
        + ResidualizedOperation<LinearizationContextOf<C, Self>>
        + From<ZeroOperation<ArrayType>>
        + From<
            CustomVjpCallOperation<
                ArrayType,
                V,
                Self,
                ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>,
            >,
        > + From<TransferToMemoryOperation>
        + From<ConcatenateOperation>
        + From<LinearSelectOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>>>
        + From<LinearDynamicSliceOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>>>
        + From<LinearDynamicUpdateSliceOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>>>
        + From<LinearGatherOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>>>
        + From<LinearScatterAddOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>>>
        + From<
            ConditionOperation<
                ArrayType,
                C::Tangent,
                LinearOperationOf<LinearizationContextOf<C, Self>>,
                ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>,
                Captured,
            >,
        > + DefactorizableProgramOperation<C::Tangent, Tracer<LinearizationContextOf<C, Self>>, Self>
        + From<MaterializeCaptureOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>>>
        + From<RecomputeOperation<Self>>
        + From<WhileOperation<ArrayType, C::Tangent, LinearOperationOf<LinearizationContextOf<C, Self>>, Input>>
        + From<
            ScanOperation<
                ArrayType,
                C::Tangent,
                BodyOperation,
                ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>,
                Input,
            >,
        >,
    LinearOperationOf<LinearizationContextOf<C, Self>>: CaptureParameterizedOperation<
            ArrayType,
            ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>,
            WithCapture<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>> = LinearOperationOf<
                LinearizationContextOf<C, Self>,
            >,
            WithCapture<ValueOrCapture<ArrayType, C::Value>> = LinearOperationOf<C>,
            WithCapture<ValueOrCapture<ArrayType, C::Tangent>> = BodyOperation,
        > + MaybeZeroOperation<ArrayType>,
{
    fn linearize_program(
        differentiable: &C,
        program: &crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
    ) -> Result<NestedLinearization<C, Self>, ProgramError> {
        program.linearize(differentiable)
    }
}

/// Dispatches non-control-flow [`LinearArrayOperation`] variants to their primitive batching rules.
fn batch_linear_non_control_operation<F, C, V>(
    operation: &LinearArrayOperation<F, C, F, ArrayOperation<C>>,
    context: &V::InterpretationContext,
    inputs: &[ArrayBatch<V>],
) -> Result<Option<Vec<ArrayBatch<V>>>, ProgramError>
where
    F: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    F::InterpretationContext: Zero<ArrayType, F>,
    C: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    C::InterpretationContext: Zero<ArrayType, C>,
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + OneLike
        + DotOps
        + ReshapeOps
        + Broadcast
        + Reduce
        + Pad
        + Concatenate
        + Slice
        + UpdateSlice
        + DynamicSlice
        + DynamicUpdateSlice
        + Gather
        + Scatter
        + Reshard
        + ConstrainSharding
        + BitAnd<Output = V>
        + Select<Condition = V>,
    V::InterpretationContext: Scale<ArrayType, V, F> + LeftDot<V, F, Captured> + RightDot<V, F, Captured>,
{
    let outputs = match operation {
        LinearArrayOperation::Add(_) => AddOperation.batch(context, inputs)?,
        LinearArrayOperation::Sub(_) => SubOperation.batch(context, inputs)?,
        LinearArrayOperation::Mul(_) => MulOperation.batch(context, inputs)?,
        LinearArrayOperation::Neg(_) => NegOperation.batch(context, inputs)?,
        LinearArrayOperation::ZeroLike(_) => ZeroLikeOperation.batch(context, inputs)?,
        LinearArrayOperation::OneLike(_) => OneLikeOperation.batch(context, inputs)?,
        LinearArrayOperation::Scale(operation) => {
            ScaleOperation::new(operation.factor().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::Transpose(operation) => {
            TransposeOperation::new(operation.permutation().to_vec()).batch(context, inputs)?
        }
        LinearArrayOperation::LeftDot(operation) => {
            LeftDotOperation::new(operation.factor().clone(), operation.dimensions().clone())
                .with_output_sharding(operation.output_sharding().cloned())
                .batch(context, inputs)?
        }
        LinearArrayOperation::RightDot(operation) => {
            RightDotOperation::new(operation.factor().clone(), operation.dimensions().clone())
                .with_output_sharding(operation.output_sharding().cloned())
                .batch(context, inputs)?
        }
        LinearArrayOperation::Reshape(operation) => {
            ReshapeOperation::new(operation.output_shape().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::Reshard(operation) => {
            ReshardOperation::new(operation.sharding().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::ShardingConstraint(operation) => {
            ShardingConstraintOperation::new(operation.sharding().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::Broadcast(operation) => {
            BroadcastOperation::new(operation.output_type().clone(), operation.output_axes().to_vec())
                .batch(context, inputs)?
        }
        LinearArrayOperation::Reduce(operation) => ReduceOperation::new(operation.axes().to_vec(), operation.kind())
            .with_output_sharding(operation.output_sharding().cloned())
            .batch(context, inputs)?,
        LinearArrayOperation::Slice(operation) => {
            SliceOperation::new(operation.start_indices().to_vec(), operation.limit_indices().to_vec())
                .with_strides(operation.strides().to_vec())?
                .batch(context, inputs)?
        }
        LinearArrayOperation::UpdateSlice(operation) => {
            UpdateSliceOperation::new(operation.start_indices().to_vec()).batch(context, inputs)?
        }
        LinearArrayOperation::Pad(operation) => PadOperation::new(
            operation.edge_padding_low().to_vec(),
            operation.edge_padding_high().to_vec(),
            operation.interior_padding().to_vec(),
        )?
        .batch(context, inputs)?,
        LinearArrayOperation::Concatenate(operation) => {
            ConcatenateOperation::new(operation.axis()).batch(context, inputs)?
        }
        LinearArrayOperation::TransferToMemory(_)
        | LinearArrayOperation::DynamicSlice(_)
        | LinearArrayOperation::DynamicUpdateSlice(_)
        | LinearArrayOperation::Gather(_)
        | LinearArrayOperation::ScatterAdd(_)
        | LinearArrayOperation::Select(_)
        | LinearArrayOperation::Residual(_)
        | LinearArrayOperation::Recompute(_)
        | LinearArrayOperation::Condition(_)
        | LinearArrayOperation::WhileCondition(_)
        | LinearArrayOperation::While(_)
        | LinearArrayOperation::Scan(_)
        | LinearArrayOperation::CustomVjpCall(_) => {
            return Ok(None);
        }
        LinearArrayOperation::Zero(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "Zero")),
        LinearArrayOperation::One(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "One")),
        LinearArrayOperation::Constant(_) => {
            return Err(missing_zero_input_batch_rule("LinearArrayOperation", "Constant"));
        }
        LinearArrayOperation::Fill(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "Fill")),
    };
    Ok(Some(outputs))
}

/// Blanket value-level batching impl for the [`LinearArrayOperation`] sum type.
impl<V> BatchableOperation<V, EagerContext<ArrayType, V, LinearArrayOperation<V, V, V, ArrayOperation<V>>>>
    for LinearArrayOperation<V, V, V, ArrayOperation<V>>
where
    ArrayOperation<V>: BatchableOperation<V, EagerContext<ArrayType, V, ArrayOperation<V>>>,
    V::InterpretationContext: Default,
    V::InterpretationContext: Scale<ArrayType, V, V> + LeftDot<V, V, Captured> + RightDot<V, V, Captured>,
    V: Value<ArrayType>
        + Add<Output = V>
        + Sub<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + OneLike
        + DotOps
        + ReshapeOps
        + Broadcast
        + Reduce
        + Pad
        + Concatenate
        + Slice
        + UpdateSlice
        + DynamicSlice
        + DynamicUpdateSlice
        + Gather
        + Scatter
        + Reshard
        + ConstrainSharding
        + BitAnd<Output = V>
        + Select<Condition = V>
        + BooleanLike,
    V::InterpretationContext: Zero<ArrayType, V>,
    EagerContext<ArrayType, V, LinearArrayOperation<V, V, V, ArrayOperation<V>>>: Zero<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &EagerContext<ArrayType, V, LinearArrayOperation<V, V, V, ArrayOperation<V>>>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let interpretation_context = V::InterpretationContext::default();
        if let Some(outputs) = batch_linear_non_control_operation(self, &interpretation_context, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::TransferToMemory(_) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(inputs.to_vec())
            }
            // The captured condition is lane-uniform: prepending it as an unbatched operand lets the elementwise
            // select batching rule broadcast it to the batched physical shape before selecting per lane.
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                SelectOperation.batch(
                    &interpretation_context,
                    &[ArrayBatch::unbatched(operation.condition().clone()), inputs[0].clone(), inputs[1].clone()],
                )
            }
            // The captured start indices are lane-uniform by construction: appending them as unbatched operands
            // lets the primal dynamic-slice batching rule lift the lane axis.
            Self::DynamicSlice(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs
                    .extend(operation.start_indices().iter().map(|index| ArrayBatch::unbatched(index.clone())));
                DynamicSliceOperation::new(operation.sizes().to_vec())
                    .batch(&interpretation_context, lifted_inputs.as_slice())
            }
            // The captured start indices are lane-uniform by construction: appending them as unbatched operands
            // lets the primal dynamic-update-slice batching rule lift the lane axis.
            Self::DynamicUpdateSlice(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs
                    .extend(operation.start_indices().iter().map(|index| ArrayBatch::unbatched(index.clone())));
                DynamicUpdateSliceOperation.batch(&interpretation_context, lifted_inputs.as_slice())
            }
            // The captured index operand is lane-uniform by construction: inserting it as the second (unbatched)
            // operand lets the primal gather batching rule lift the lane axis.
            Self::Gather(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                operation.operation().batch(
                    &interpretation_context,
                    &[inputs[0].clone(), ArrayBatch::unbatched(operation.indices().clone())],
                )
            }
            // The captured index operand is lane-uniform by construction: inserting it between the operand and update
            // tangents (unbatched) lets the primal scatter batching rule lift the lane axis.
            Self::ScatterAdd(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                operation.operation().batch(
                    &interpretation_context,
                    &[inputs[0].clone(), ArrayBatch::unbatched(operation.indices().clone()), inputs[1].clone()],
                )
            }
            // The captured factor is lane-uniform by construction: the same residual value applies to every lane.
            Self::Residual(operation) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![ArrayBatch::unbatched(operation.capture().clone())])
            }
            // Recomputed primal operations batch through the wrapped operation's own primal batching rule.
            Self::Recompute(operation) => {
                let primal_context = EagerContext::<ArrayType, V, ArrayOperation<V>>::new();
                operation.batch(&primal_context, inputs)
            }
            // The captured predicate is lane-uniform: prepending it as an unbatched input lets the condition
            // batching helper read the branch choice from input 0, exactly like an ordinary runtime predicate.
            Self::Condition(operation) => {
                let mut condition_inputs = Vec::with_capacity(inputs.len() + 1);
                condition_inputs.push(ArrayBatch::unbatched(operation.predicate().clone()));
                condition_inputs.extend(inputs.iter().cloned());
                batch_condition_with_interpreter(
                    operation.true_branch(),
                    operation.false_branch(),
                    condition_inputs.as_slice(),
                    |program, program_inputs| {
                        program.interpret_with(
                            program_inputs,
                            |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                            |instruction, instruction_inputs| {
                                instruction.operation().batch(context, instruction_inputs)
                            },
                        )
                    },
                )
            }
            // The while-condition form already reads its predicate from input 0, which is exactly the layout the
            // condition batching helper expects for an ordinary runtime predicate.
            Self::WhileCondition(operation) => batch_condition_with_interpreter(
                operation.true_branch(),
                operation.false_branch(),
                inputs,
                |program, program_inputs| {
                    program.interpret_with(
                        program_inputs,
                        |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                        |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
                    )
                },
            ),
            Self::While(operation) => operation.batch(context, inputs),
            // Each lane's body pushforward is bound against that lane's residual slices and batched through the
            // shared scan loop; the residual stacks are concrete values in the direct linear form.
            Self::Scan(operation) => {
                let body = operation.body();
                let carry_count = operation.carry_count();
                let residual_stacks = operation.captures();
                let y_slice_types = body.output_types().split_off(carry_count);
                crate::tracing_v2::operations::scan::batch_scan_with_interpreter(
                    carry_count,
                    operation.length(),
                    operation.reverse(),
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| context.zero(stacked_type),
                    |lane, lane_inputs| {
                        let lane_residuals = residual_stacks
                            .iter()
                            .map(|stack| read_scan_lane(stack, lane))
                            .collect::<Result<Vec<_>, _>>()?;
                        let lane_body = body.map_operations(|operation| {
                            operation.try_map_captures(&mut |factor| factor.instantiate(lane_residuals.as_slice()))
                        })?;
                        lane_body.interpret_with(
                            lane_inputs,
                            |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                            |instruction, instruction_inputs| {
                                instruction.operation().batch(context, instruction_inputs)
                            },
                        )
                    },
                )
            }
            Self::CustomVjpCall(call) => {
                let primal_context = EagerContext::<ArrayType, V, ArrayOperation<V>>::new();
                call.batch(&primal_context, inputs)
            }
            _ => unreachable!("non-control-flow LinearArrayOperation variants are handled above"),
        }
    }
}

/// Blanket active batching impl for the [`LinearArrayOperation`] sum type.
impl<C> BatchableOperation<Tracer<C>, BatchingContext<C>>
    for LinearArrayOperation<C::Constant, C::Constant, C::Constant, ArrayOperation<C::Constant>>
where
    ArrayOperation<C::Constant>: BatchableOperation<Tracer<C>, BatchingContext<C>>,
    C: StagingContext<Type = ArrayType>
        + Scale<ArrayType, Tracer<C>, C::Constant>
        + LeftDot<Tracer<C>, C::Constant, Captured>
        + RightDot<Tracer<C>, C::Constant, Captured>,
    C::Constant: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    <C::Constant as Value<ArrayType>>::InterpretationContext: Zero<ArrayType, C::Constant>,
    C::Operation: From<ZeroOperation<ArrayType>>,
    Tracer<C>: Add<Output = Tracer<C>>
        + Sub<Output = Tracer<C>>
        + Neg<Output = Tracer<C>>
        + Mul<Output = Tracer<C>>
        + ZeroLike
        + OneLike
        + DotOps
        + ReshapeOps
        + Broadcast
        + Reduce
        + Pad
        + Concatenate
        + Slice
        + UpdateSlice
        + DynamicSlice
        + DynamicUpdateSlice
        + Gather
        + Scatter
        + Reshard
        + ConstrainSharding
        + BitAnd<Output = Tracer<C>>
        + Select<Condition = Tracer<C>>
        + BooleanLike
        + TransferToMemory,
    Vec<Tracer<C>>: Parameterized<Tracer<C>, To<Tracer<C>> = Vec<Tracer<C>>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        if let Some(outputs) = batch_linear_non_control_operation(self, context.parent_context(), inputs)? {
            return Ok(outputs);
        }
        match self {
            // Memory placement is lane-uniform: the same transfer applies to every lane, so the transfer is
            // staged unchanged on the physical batched value (in its own parent context) and the lane axis is
            // preserved. The parent operation type is generic here, so the value-level capability stages it.
            Self::TransferToMemory(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let tracer = inputs[0].value().transfer_to_memory(operation.destination());
                let physical_type = tracer.r#type().into_owned();
                Ok(vec![ArrayBatch::new(physical_type, tracer, inputs[0].batch_axis())?])
            }
            // The captured condition is a lane-uniform parent-context constant: lift it into the parent trace and
            // let the elementwise select batching rule broadcast it to the batched physical shape.
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let condition = context.parent_context().constant(operation.condition().clone());
                SelectOperation.batch(
                    context.parent_context(),
                    &[ArrayBatch::unbatched(condition), inputs[0].clone(), inputs[1].clone()],
                )
            }
            // The captured start indices are lane-uniform parent-context constants: lift them into the parent
            // trace and let the primal dynamic-slice batching rule lift the lane axis.
            Self::DynamicSlice(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs.extend(
                    operation
                        .start_indices()
                        .iter()
                        .map(|index| ArrayBatch::unbatched(context.parent_context().constant(index.clone()))),
                );
                DynamicSliceOperation::new(operation.sizes().to_vec())
                    .batch(context.parent_context(), lifted_inputs.as_slice())
            }
            // The captured start indices are lane-uniform parent-context constants: lift them into the parent
            // trace and let the primal dynamic-update-slice batching rule lift the lane axis.
            Self::DynamicUpdateSlice(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs.extend(
                    operation
                        .start_indices()
                        .iter()
                        .map(|index| ArrayBatch::unbatched(context.parent_context().constant(index.clone()))),
                );
                DynamicUpdateSliceOperation.batch(context.parent_context(), lifted_inputs.as_slice())
            }
            // The captured factor is a lane-uniform parent-context constant: lift it into the parent trace.
            Self::Residual(operation) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![ArrayBatch::unbatched(context.parent_context().constant(operation.capture().clone()))])
            }
            // Recomputed primal operations batch through the wrapped operation's own primal batching rule.
            Self::Recompute(operation) => operation.batch(context, inputs),
            // The captured predicate is a lane-uniform parent-context constant, so the branch choice is concrete:
            // extract it from the factor and batch only the selected branch. Prepending a lifted predicate tracer
            // would defeat the lane-uniform extraction because tracers cannot be concretized.
            Self::Condition(operation) => {
                let branch =
                    if operation.predicate().boolean()? { operation.true_branch() } else { operation.false_branch() };
                context.interpret_program(branch, inputs.to_vec())
            }
            // The while-condition form already reads its predicate from input 0, which is exactly the layout the
            // condition batching helper expects for an ordinary runtime predicate (lane-uniform predicates extract
            // concretely, lane-varying ones run both branches and select per lane).
            Self::WhileCondition(operation) => batch_condition_with_interpreter::<C::Constant, Tracer<C>, _, _>(
                operation.true_branch(),
                operation.false_branch(),
                inputs,
                |program, program_inputs| context.interpret_program(program, program_inputs),
            ),
            // The fused doubled-state linear while keeps the operational masked-unrolling rule even under tracing:
            // its condition recomputes the loop predicate from captured residual injections (parent-context
            // constants), so the per-iteration predicate extraction stays concrete and the loop unrolls through the
            // batched tracers. The staged batching rule on the primal `WhileOperation` does not apply here because
            // the loop's nested operation type is this linear enum, not the staged program's operation type.
            Self::While(operation) => {
                batch_while_with_interpreter(operation.as_ref(), inputs, |program, program_inputs| {
                    context.interpret_program(program, program_inputs)
                })
            }
            // Each lane's body pushforward is bound against that lane's residual slices at the constant level
            // (the stacks are lane-uniform parent-context constants) and batched over the traced lanes through
            // the shared scan loop; stacked output accumulators are staged as typed zeros in the parent trace.
            Self::Scan(operation) => {
                let body = operation.body();
                let carry_count = operation.carry_count();
                let residual_stacks = operation.captures();
                let y_slice_types = body.output_types().split_off(carry_count);
                crate::tracing_v2::operations::scan::batch_scan_with_interpreter(
                    carry_count,
                    operation.length(),
                    operation.reverse(),
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| {
                        let mut outputs = context
                            .parent_context()
                            .stage_nullary_operation(C::Operation::from(ZeroOperation::new(stacked_type.clone())))?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(outputs.remove(0))
                    },
                    |lane, lane_inputs| {
                        let lane_residuals = residual_stacks
                            .iter()
                            .map(|stack| read_scan_lane(stack, lane))
                            .collect::<Result<Vec<_>, _>>()?;
                        let lane_body = body.map_operations(|operation| {
                            operation.try_map_captures(&mut |factor| factor.instantiate(lane_residuals.as_slice()))
                        })?;
                        context.interpret_program(&lane_body, lane_inputs)
                    },
                )
            }
            Self::CustomVjpCall(call) => {
                if !call.transposed() {
                    return Err(crate::types::TypeError {
                        message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                            value_and_grad, or jacrev) instead"
                            .to_string(),
                    }
                    .into());
                }
                let mut values = call
                    .residuals()
                    .iter()
                    .map(|residual| ArrayBatch::unbatched(context.parent_context().constant(residual.clone())))
                    .collect::<Vec<_>>();
                values.extend(inputs.iter().cloned());
                context.interpret_program(call.backward(), values)
            }
            _ => unreachable!("non-control-flow LinearArrayOperation variants are handled above"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::differentiation::{Cotangent, TransposableOperation};
    use crate::domains::AbstractDomain;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::tracing::AbstractTracingContext;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_linear_condition_transpose_supports_runtime_predicates() {
        // Linear-condition transposition is total: the captured predicate factor is a residual of the primal
        // computation rather than a linear operand, so it is carried verbatim into one staged transposed condition
        // over the transposed branch programs. Runtime (factor) predicates used to be rejected with an
        // `UnsupportedOperation` error.
        type TestLinearOperation = LinearArrayOperation<TestArray, TestArray, TestArray, ArrayOperation<TestArray>>;
        let scale_branch = |factor: f64| {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, TestLinearOperation>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let output = builder
                .add_instruction(
                    TestLinearOperation::Scale(ScaleOperation::new(TestArray::scalar(factor))),
                    vec![input],
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let operation = TestLinearOperation::Condition(
            ConditionOperation::new_captured(
                TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]),
                scale_branch(2.0),
                scale_branch(3.0),
            )
            .unwrap(),
        );

        let domain = AbstractDomain::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, TestLinearOperation>::new()));
        let cotangent_input = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let mut context = AbstractTracingContext::new(&domain, builder.clone());
        let cotangent = context.tracer(cotangent_input, None);
        let cotangents = operation
            .transpose(&mut context, &[&ArrayType::scalar(DataType::F64)], &[Cotangent::Staged(cotangent)])
            .unwrap();
        assert_eq!(cotangents.len(), 1);
        assert!(!cotangents[0].is_zero());
        let pullback_output = cotangents[0].as_staged().unwrap().atom_id().unwrap();
        assert!(matches!(builder.borrow().instructions()[0].operation(), TestLinearOperation::Condition { .. }));

        // Interpreting the pullback applies the transposed branch selected by the carried predicate (scale by 2).
        drop(cotangents);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let pullback = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![pullback_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let outputs = pullback.interpret(vec![TestArray::scalar(5.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![10.0]);
    }
}
