use std::cell::RefCell;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};
use std::rc::Rc;

use ryft_macros::Operation;

use ryft_core::batching::BatchingError;
use ryft_core::compilation::CapturedConstant;
use ryft_core::contexts::StagingContext;
use ryft_core::differentiation::{Cotangent, TransposableOperation};
use ryft_core::domains::Domain;
use ryft_core::macros::check_count;
use ryft_core::operations::arithmetic::{
    AddOperation, DivOperation, MulOperation, NegOperation, ScaleOperation, SubOperation,
};
use ryft_core::operations::compare::{Compare, CompareOperation};
use ryft_core::operations::constants::{
    ConstantOperation, FillOperation, MaybeZeroOperation, OneLike, OneLikeOperation, OneOperation, Zero, ZeroLike,
    ZeroLikeOperation, ZeroOperation,
};
use ryft_core::operations::control_flow::{ConditionOperation, ScanOperation, Select, SelectOperation, WhileOperation};
use ryft_core::operations::differentiation::StopGradientOperation;
use ryft_core::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use ryft_core::operations::manipulation::{
    Broadcast, BroadcastOperation, Concatenate, ConcatenateOperation, DynamicSlice, DynamicSliceOperation,
    DynamicUpdateSlice, DynamicUpdateSliceOperation, Gather, GatherOperation, LinearDynamicSliceOperation,
    LinearDynamicUpdateSliceOperation, LinearGatherOperation, LinearScatterAddOperation, Pad, PadOperation, Reshape,
    ReshapeOperation, Scatter, ScatterOperation, Slice, SliceOperation, Transpose, TransposeOperation, UpdateSlice,
    UpdateSliceOperation,
};
use ryft_core::operations::sharding::{ConstrainSharding, Reshard, ReshardOperation, ShardingConstraintOperation};
use ryft_core::operations::trigonometric::{Cos, CosOperation, Sin, SinOperation};
use ryft_core::operations::{BooleanLike, InterpretableOperation, Operation, OperationFormatter};
use ryft_core::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use ryft_core::payloads::{Captured, Input};
use ryft_core::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use ryft_core::tracing::{AbstractTracingContext, Tracer, TracingContext};
use ryft_core::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, BatchableProgramOperation, BatchingContext, ProgramBatchingContext,
    ProgramBatchingOutputAxes, batch_input_metadata, batch_program,
};
use ryft_core::tracing_v2::differentiation::LinearizationContextOf;
use ryft_core::tracing_v2::operations::control_flow::{
    DefactorizableProgramOperation, DefactorizedOperation, defactorize_operation_default,
};
use ryft_core::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpCallOperation, CustomVjpOperation,
};
use ryft_core::tracing_v2::operations::dot::{DotOps, LeftDotOperation, RightDotOperation};
use ryft_core::tracing_v2::operations::memory::{TransferToMemory, TransferToMemoryOperation};
use ryft_core::tracing_v2::operations::recompute::RecomputeOperation;
use ryft_core::tracing_v2::operations::reduce::{Reduce as ReduceValue, ReduceOperation};
use ryft_core::tracing_v2::operations::reshape::ReshapeOps;
use ryft_core::tracing_v2::operations::select::LinearSelectOperation;
use ryft_core::tracing_v2::rematerialization::{RematerializeCallOperation, RematerializeOperation};
use ryft_core::tracing_v2::{
    ArrayOperation, CaptureParameterizedOperation, CollectiveOperation, DifferentiableOperation,
    DifferentiationContext, DotOperation, JvpTracer, LinearArrayOperation, LinearizableProgramOperation,
    MaterializeCaptureOperation, NestedLinearization, RematerializationNameOperation, ResidualizedOperation,
    TangentContext, ValueOrCapture,
};
use ryft_core::types::{ArrayType, Size, TypeError, Typed};

use crate::experimental::domains::{XlaDomain, XlaTracer};
use crate::experimental::operations::{LinearShardMapOperation, ShardMapOperation};

/// Lifetime-free reference to a concrete XLA value captured by a compiled program.
pub type XlaConstant = CapturedConstant<ArrayType>;

/// Ordinary staged-operation universe owned by the XLA backend.
///
/// This enum flattens the core array operation payloads directly into the backend-owned operation family. Higher-order
/// operations that own nested programs use XLA-owned bodies so those programs can contain backend-specific operations
/// such as [`jit_call`](JitCallOperation) and [`shard_map`](ShardMapOperation).
#[derive(Clone, Debug, Operation)]
#[ryft(crate = "ryft_core")]
#[ryft(bounds(interpretation(BooleanLike + Slice + UpdateSlice + Reshape)))]
pub enum XlaOperation<V: Value<ArrayType> = XlaConstant> {
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
    /// Backend-owned condition whose branch bodies can contain XLA operations.
    Condition(Box<ConditionOperation<ArrayType, V, Self>>),

    /// Backend-owned loop whose condition and body programs can contain XLA operations.
    While(Box<WhileOperation<ArrayType, V, Self>>),

    /// Backend-owned scan whose body program can contain XLA operations.
    Scan(Box<ScanOperation<ArrayType, V, Self>>),

    /// Backend-owned custom JVP call whose nested programs can contain XLA operations.
    CustomJvp(Box<CustomJvpOperation<ArrayType, V, Self>>),

    /// Backend-owned custom VJP call whose nested programs can contain XLA operations.
    CustomVjp(Box<CustomVjpOperation<ArrayType, V, Self>>),

    /// Backend-owned rematerialized call whose nested programs can contain XLA operations.
    Rematerialize(Box<RematerializeOperation<ArrayType, V, Self>>),

    /// Call to a flat jitted XLA sub-program.
    JitCall(Box<JitCallOperation>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<V>>),

    /// XLA-specific `linear_shard_map` staged in an ordinary traced program.
    LinearShardMap(Box<LinearShardMapOperation<V>>),
}

fn map_core_xla_program<V>(
    program: &Program<ArrayType, V, ArrayOperation<V>, Vec<V>, Vec<V>>,
) -> Program<ArrayType, V, XlaOperation<V>, Vec<V>, Vec<V>>
where
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    V::InterpretationContext: Zero<ArrayType, V>,
{
    program.map_operations(|operation| Ok(XlaOperation::from(operation.clone()))).unwrap()
}

impl<V> From<ArrayOperation<V>> for XlaOperation<V>
where
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    V::InterpretationContext: Zero<ArrayType, V>,
{
    fn from(operation: ArrayOperation<V>) -> Self {
        match operation {
            ArrayOperation::Zero(operation) => Self::Zero(operation),
            ArrayOperation::ZeroLike(operation) => Self::ZeroLike(operation),
            ArrayOperation::One(operation) => Self::One(operation),
            ArrayOperation::OneLike(operation) => Self::OneLike(operation),
            ArrayOperation::Constant(operation) => Self::Constant(operation),
            ArrayOperation::Fill(operation) => Self::Fill(operation),
            ArrayOperation::Neg(operation) => Self::Neg(operation),
            ArrayOperation::Add(operation) => Self::Add(operation),
            ArrayOperation::Sub(operation) => Self::Sub(operation),
            ArrayOperation::Scale(operation) => Self::Scale(operation),
            ArrayOperation::Mul(operation) => Self::Mul(operation),
            ArrayOperation::Div(operation) => Self::Div(operation),
            ArrayOperation::Sin(operation) => Self::Sin(operation),
            ArrayOperation::Cos(operation) => Self::Cos(operation),
            ArrayOperation::StopGradient(operation) => Self::StopGradient(operation),
            ArrayOperation::RematerializationName(operation) => Self::RematerializationName(operation),
            ArrayOperation::TransferToMemory(operation) => Self::TransferToMemory(operation),
            ArrayOperation::Dot(operation) => Self::Dot(operation),
            ArrayOperation::Transpose(operation) => Self::Transpose(operation),
            ArrayOperation::Reshape(operation) => Self::Reshape(operation),
            ArrayOperation::Reshard(operation) => Self::Reshard(operation),
            ArrayOperation::ShardingConstraint(operation) => Self::ShardingConstraint(operation),
            ArrayOperation::Broadcast(operation) => Self::Broadcast(operation),
            ArrayOperation::Slice(operation) => Self::Slice(operation),
            ArrayOperation::UpdateSlice(operation) => Self::UpdateSlice(operation),
            ArrayOperation::DynamicSlice(operation) => Self::DynamicSlice(operation),
            ArrayOperation::DynamicUpdateSlice(operation) => Self::DynamicUpdateSlice(operation),
            ArrayOperation::Pad(operation) => Self::Pad(operation),
            ArrayOperation::Concatenate(operation) => Self::Concatenate(operation),
            ArrayOperation::Gather(operation) => Self::Gather(operation),
            ArrayOperation::Scatter(operation) => Self::Scatter(operation),
            ArrayOperation::Reduce(operation) => Self::Reduce(operation),
            ArrayOperation::Compare(operation) => Self::Compare(operation),
            ArrayOperation::Not(operation) => Self::Not(operation),
            ArrayOperation::And(operation) => Self::And(operation),
            ArrayOperation::Or(operation) => Self::Or(operation),
            ArrayOperation::Xor(operation) => Self::Xor(operation),
            ArrayOperation::Collective(operation) => Self::Collective(operation),
            ArrayOperation::Select(operation) => Self::Select(operation),
            ArrayOperation::Condition(operation) => XlaOperation::from(*operation),
            ArrayOperation::While(operation) => XlaOperation::from(*operation),
            ArrayOperation::Scan(operation) => XlaOperation::from(*operation),
            ArrayOperation::CustomJvp(operation) => XlaOperation::from(*operation),
            ArrayOperation::CustomVjp(operation) => XlaOperation::from(*operation),
            ArrayOperation::Rematerialize(operation) => XlaOperation::from(*operation),
        }
    }
}

impl<V> From<ConditionOperation<ArrayType, V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    V::InterpretationContext: Zero<ArrayType, V>,
{
    fn from(operation: ConditionOperation<ArrayType, V, ArrayOperation<V>>) -> Self {
        Self::Condition(Box::new(
            ConditionOperation::new(
                map_core_xla_program(operation.true_branch()),
                map_core_xla_program(operation.false_branch()),
            )
            .unwrap(),
        ))
    }
}

impl<V> From<WhileOperation<ArrayType, V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    V::InterpretationContext: Zero<ArrayType, V>,
{
    fn from(operation: WhileOperation<ArrayType, V, ArrayOperation<V>>) -> Self {
        Self::While(Box::new(
            WhileOperation::new(map_core_xla_program(operation.condition()), map_core_xla_program(operation.body()))
                .unwrap()
                .with_iteration_bound(operation.iteration_bound())
                .unwrap(),
        ))
    }
}

impl<V> From<ScanOperation<ArrayType, V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    V::InterpretationContext: Zero<ArrayType, V>,
{
    fn from(operation: ScanOperation<ArrayType, V, ArrayOperation<V>>) -> Self {
        Self::Scan(Box::new(
            ScanOperation::new(map_core_xla_program(operation.body()), operation.carry_count(), operation.length())
                .unwrap()
                .with_reverse(operation.reverse())
                .with_unroll(operation.unroll())
                .unwrap()
                .with_captures(operation.captures().to_vec()),
        ))
    }
}

impl<V> From<CustomJvpOperation<ArrayType, V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    V::InterpretationContext: Zero<ArrayType, V>,
{
    fn from(operation: CustomJvpOperation<ArrayType, V, ArrayOperation<V>>) -> Self {
        Self::CustomJvp(Box::new(
            CustomJvpOperation::new(
                map_core_xla_program(operation.primal()),
                map_core_xla_program(operation.jvp_program()),
            )
            .unwrap(),
        ))
    }
}

impl<V> From<CustomVjpOperation<ArrayType, V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    V::InterpretationContext: Zero<ArrayType, V>,
{
    fn from(operation: CustomVjpOperation<ArrayType, V, ArrayOperation<V>>) -> Self {
        Self::CustomVjp(Box::new(
            CustomVjpOperation::new(
                map_core_xla_program(operation.primal()),
                map_core_xla_program(operation.forward()),
                map_core_xla_program(operation.backward()),
            )
            .unwrap(),
        ))
    }
}

impl<V> From<RematerializeOperation<ArrayType, V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
    V::InterpretationContext: Zero<ArrayType, V>,
{
    fn from(operation: RematerializeOperation<ArrayType, V, ArrayOperation<V>>) -> Self {
        Self::Rematerialize(Box::new(
            RematerializeOperation::new(
                map_core_xla_program(operation.primal()),
                map_core_xla_program(operation.forward()),
                map_core_xla_program(operation.backward()),
                map_core_xla_program(operation.tangent()),
            )
            .unwrap()
            .with_prevent_cse(operation.prevent_cse()),
        ))
    }
}

impl<V> ryft_core::tracing_v2::operations::MaybeDot for XlaOperation<V>
where
    V: Value<ArrayType>,
{
    #[inline]
    fn dot_dimensions(&self) -> Option<&ryft_core::tracing_v2::DotDimensionNumbers> {
        match self {
            Self::Dot(operation) => Some(operation.dimensions()),
            _ => None,
        }
    }
}

impl<V> ryft_core::tracing_v2::rematerialization::MaybeRematerializationName for XlaOperation<V>
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

impl<'domain, 'context> InterpretableOperation<ArrayType, XlaTracer<'domain, 'context>> for XlaOperation {
    fn interpret(
        &self,
        context: &TracingContext<'domain, XlaDomain<'context>>,
        inputs: &[XlaTracer<'domain, 'context>],
    ) -> Result<Vec<XlaTracer<'domain, 'context>>, ProgramError> {
        context.stage_operation(self.clone(), inputs)
    }
}

impl<C> BatchableOperation<Tracer<C>, BatchingContext<C>> for XlaOperation
where
    C: StagingContext<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>,
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
        + ReduceValue
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
    Vec<Tracer<C>>:
        Parameterized<Tracer<C>, To<Tracer<C>> = Vec<Tracer<C>>, ParameterStructure: std::fmt::Debug + PartialEq>,
    Self: BatchableProgramOperation<XlaConstant>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        match self {
            Self::Add(operation) => operation.batch(context.parent_context(), inputs),
            Self::Sub(operation) => operation.batch(context.parent_context(), inputs),
            Self::Mul(operation) => operation.batch(context.parent_context(), inputs),
            Self::Div(operation) => operation.batch(context.parent_context(), inputs),
            Self::Neg(operation) => operation.batch(context.parent_context(), inputs),
            Self::Sin(operation) => operation.batch(context.parent_context(), inputs),
            Self::Cos(operation) => operation.batch(context.parent_context(), inputs),
            Self::StopGradient(operation) => operation.batch(context.parent_context(), inputs),
            Self::RematerializationName(operation) => operation.batch(context.parent_context(), inputs),
            Self::Select(operation) => operation.batch(context.parent_context(), inputs),
            Self::ZeroLike(operation) => operation.batch(context.parent_context(), inputs),
            Self::OneLike(operation) => operation.batch(context.parent_context(), inputs),
            Self::Scale(operation) => operation.batch(context.parent_context(), inputs),
            Self::Dot(operation) => operation.batch(context.parent_context(), inputs),
            Self::Transpose(operation) => operation.batch(context.parent_context(), inputs),
            Self::Reshape(operation) => operation.batch(context.parent_context(), inputs),
            Self::Reshard(operation) => operation.batch(context.parent_context(), inputs),
            Self::ShardingConstraint(operation) => operation.batch(context.parent_context(), inputs),
            Self::Broadcast(operation) => operation.batch(context.parent_context(), inputs),
            Self::Slice(operation) => operation.batch(context.parent_context(), inputs),
            Self::UpdateSlice(operation) => operation.batch(context.parent_context(), inputs),
            Self::DynamicSlice(operation) => operation.batch(context.parent_context(), inputs),
            Self::DynamicUpdateSlice(operation) => operation.batch(context.parent_context(), inputs),
            Self::Pad(operation) => operation.batch(context.parent_context(), inputs),
            Self::Concatenate(operation) => operation.batch(context.parent_context(), inputs),
            Self::Gather(operation) => operation.batch(context.parent_context(), inputs),
            Self::Scatter(operation) => operation.batch(context.parent_context(), inputs),
            Self::Reduce(operation) => operation.batch(context.parent_context(), inputs),
            Self::Compare(operation) => operation.batch(context.parent_context(), inputs),
            Self::Not(operation) => operation.batch(context.parent_context(), inputs),
            Self::And(operation) => operation.batch(context.parent_context(), inputs),
            Self::Or(operation) => operation.batch(context.parent_context(), inputs),
            Self::Xor(operation) => operation.batch(context.parent_context(), inputs),
            Self::TransferToMemory(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let tracer = inputs[0].value().transfer_to_memory(operation.destination());
                let physical_type = tracer.r#type().into_owned();
                Ok(vec![ArrayBatch::new(physical_type, tracer, inputs[0].batch_axis())?])
            }
            Self::Collective(operation) => operation.batch(context, inputs),
            Self::Zero(_) | Self::One(_) | Self::Constant(_) | Self::Fill(_) => {
                Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "zero-input operation '{}' must be staged through the batching context",
                        self.name()
                    ),
                }
                .into())
            }
            Self::Condition(operation) => operation.batch(context, inputs),
            Self::While(operation) => operation.batch(context, inputs),
            Self::Scan(operation) => operation.batch(context, inputs),
            Self::CustomJvp(operation) => operation.batch(context, inputs),
            Self::CustomVjp(operation) => operation.batch(context, inputs),
            Self::Rematerialize(operation) => operation.batch(context, inputs),
            Self::JitCall(operation) => operation.batch(context, inputs),
            Self::ShardMap(_) | Self::LinearShardMap(_) => Err(BatchingError::UnsupportedOperation {
                message: format!("missing batching rule for operation '{}'", self.name()),
            }
            .into()),
        }
    }
}

impl BatchableProgramOperation<XlaConstant> for XlaOperation
where
    Tracer<ProgramBatchingContext<XlaConstant, Self>>: Add<Output = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + Sub<Output = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + Neg<Output = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + Mul<Output = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + Div<Output = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + DotOps
        + ReshapeOps
        + Broadcast
        + ReduceValue
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
        + Compare<Output = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + BitAnd<Output = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + BitOr<Output = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + BitXor<Output = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + Not<Output = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + Select<Condition = Tracer<ProgramBatchingContext<XlaConstant, Self>>>
        + BooleanLike
        + Transpose,
    Vec<Tracer<ProgramBatchingContext<XlaConstant, Self>>>: Parameterized<
            Tracer<ProgramBatchingContext<XlaConstant, Self>>,
            To<Tracer<ProgramBatchingContext<XlaConstant, Self>>> = Vec<
                Tracer<ProgramBatchingContext<XlaConstant, Self>>,
            >,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
{
    fn batch_program(
        program: &Program<ArrayType, XlaConstant, Self, Vec<XlaConstant>, Vec<XlaConstant>>,
        axis_size: usize,
        input_batch_axes: &[Option<usize>],
        output_batch_axes: ProgramBatchingOutputAxes,
    ) -> Result<
        (Program<ArrayType, XlaConstant, Self, Vec<XlaConstant>, Vec<XlaConstant>>, Vec<Option<usize>>),
        ProgramError,
    > {
        batch_program::<XlaConstant, Self>(program, axis_size, input_batch_axes, output_batch_axes)
    }
}

impl<C> DifferentiableOperation<C> for XlaOperation
where
    C: StagingContext<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>
        + DifferentiationContext<
            LinearOperation<
                <C as DifferentiationContext>::Tangent,
                ValueOrCapture<ArrayType, <C as Domain>::Value>,
            > = LinearXlaOperation<
                <C as DifferentiationContext>::Tangent,
                XlaConstant,
                ValueOrCapture<ArrayType, <C as Domain>::Value>,
            >,
            LinearOperation<
                <C as DifferentiationContext>::Tangent,
                ValueOrCapture<ArrayType, Tracer<C>>,
            > = LinearXlaOperation<
                <C as DifferentiationContext>::Tangent,
                XlaConstant,
                ValueOrCapture<ArrayType, Tracer<C>>,
            >,
        >,
    ZeroOperation<ArrayType>: DifferentiableOperation<C>,
    ZeroLikeOperation: DifferentiableOperation<C>,
    OneOperation<ArrayType>: DifferentiableOperation<C>,
    OneLikeOperation: DifferentiableOperation<C>,
    ConstantOperation<ArrayType, XlaConstant>: DifferentiableOperation<C>,
    FillOperation<ArrayType, f64>: DifferentiableOperation<C>,
    NegOperation: DifferentiableOperation<C>,
    AddOperation: DifferentiableOperation<C>,
    SubOperation: DifferentiableOperation<C>,
    ScaleOperation<ArrayType, XlaConstant>: DifferentiableOperation<C>,
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
    Vec<XlaConstant>: Parameterized<
            XlaConstant,
            Family: ParameterizedFamily<C::Tangent, To = Vec<C::Tangent>>
                + ParameterizedFamily<<C as Domain>::Value, To = Vec<<C as Domain>::Value>>,
            To<XlaConstant> = Vec<XlaConstant>,
            To<C::Tangent> = Vec<C::Tangent>,
            To<<C as Domain>::Value> = Vec<<C as Domain>::Value>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
    Vec<<C as Domain>::Value>: Parameterized<
            <C as Domain>::Value,
            Family: ParameterizedFamily<C::Tangent, To = Vec<C::Tangent>>,
            To<C::Tangent> = Vec<C::Tangent>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
    Vec<Tracer<C>>: Parameterized<
            Tracer<C>,
            Family: ParameterizedFamily<C::Tangent, To = Vec<C::Tangent>>,
            To<C::Tangent> = Vec<C::Tangent>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
    C::Tangent: Slice + UpdateSlice + Reshape,
    <C::Tangent as Value<ArrayType>>::InterpretationContext: Zero<ArrayType, C::Tangent>,
    LinearXlaOperation<C::Tangent, XlaConstant, ValueOrCapture<ArrayType, Tracer<C>>>:
        CaptureParameterizedOperation<
            ArrayType,
            ValueOrCapture<ArrayType, Tracer<C>>,
            WithCapture<ValueOrCapture<ArrayType, C::Tangent>> =
                LinearXlaOperation<C::Tangent, XlaConstant, ValueOrCapture<ArrayType, C::Tangent>>,
        >,
    C::LinearOperation<C::Tangent, ValueOrCapture<C::Type, C::Value>>: From<AddOperation>
        + From<ZeroLikeOperation>
        + From<NegOperation>
        + From<SubOperation>
        + From<ScaleOperation<ArrayType, ValueOrCapture<ArrayType, <C as Domain>::Value>, Input>>
        + From<LeftDotOperation<ValueOrCapture<ArrayType, <C as Domain>::Value>, Input>>
        + From<RightDotOperation<ValueOrCapture<ArrayType, <C as Domain>::Value>, Input>>
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
        + From<CustomVjpCallOperation<ArrayType, XlaConstant, XlaOperation, ValueOrCapture<ArrayType, <C as Domain>::Value>>>
        + From<RematerializeCallOperation<ArrayType, XlaConstant, XlaOperation, ValueOrCapture<ArrayType, <C as Domain>::Value>>>
        + From<TransferToMemoryOperation>
        + From<ConcatenateOperation>
        + From<LinearSelectOperation<ValueOrCapture<ArrayType, <C as Domain>::Value>>>
        + From<LinearDynamicSliceOperation<ValueOrCapture<ArrayType, <C as Domain>::Value>>>
        + From<LinearDynamicUpdateSliceOperation<ValueOrCapture<ArrayType, <C as Domain>::Value>>>
        + From<LinearGatherOperation<ValueOrCapture<ArrayType, <C as Domain>::Value>>>
        + From<LinearScatterAddOperation<ValueOrCapture<ArrayType, <C as Domain>::Value>>>
        + From<
            ScanOperation<
                ArrayType,
                C::Tangent,
                LinearXlaOperation<C::Tangent, XlaConstant, ValueOrCapture<ArrayType, C::Tangent>>,
                ValueOrCapture<ArrayType, <C as Domain>::Value>,
                Input,
            >,
        >
        + From<
            ConditionOperation<
                ArrayType,
                C::Tangent,
                C::LinearOperation<C::Tangent, ValueOrCapture<C::Type, C::Value>>,
                ValueOrCapture<ArrayType, <C as Domain>::Value>,
                Captured,
            >,
        >
        + From<MaterializeCaptureOperation<ValueOrCapture<ArrayType, <C as Domain>::Value>>>
        + From<RecomputeOperation<XlaOperation>>
        + From<WhileOperation<ArrayType, C::Tangent, C::LinearOperation<C::Tangent, ValueOrCapture<C::Type, C::Value>>, Input>>,
    C::LinearOperation<C::Tangent, ValueOrCapture<C::Type, C::Value>>: CaptureParameterizedOperation<
            ArrayType,
            ValueOrCapture<ArrayType, <C as Domain>::Value>,
            WithCapture<ValueOrCapture<ArrayType, <C as Domain>::Value>> = C::LinearOperation<C::Tangent, ValueOrCapture<C::Type, C::Value>>,
        > + DefactorizableProgramOperation<C::Tangent, <C as Domain>::Value, XlaOperation>,
    Self: Clone + LinearizableProgramOperation<C>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, C>,
        inputs: &[JvpTracer<'jvp, C>],
    ) -> Result<Vec<JvpTracer<'jvp, C>>, ProgramError>
    where
        C: 'jvp,
        C::LinearOperation<C::Tangent, ValueOrCapture<C::Type, C::Value>>: From<ZeroOperation<ArrayType>>,
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
            Self::Condition(operation) => {
                <ConditionOperation<ArrayType, XlaConstant, Self> as DifferentiableOperation<C>>::jvp(
                    operation, context, inputs,
                )
            }
            Self::While(operation) => <WhileOperation<ArrayType, XlaConstant, Self> as DifferentiableOperation<C>>::jvp(
                operation, context, inputs,
            ),
            Self::Scan(operation) => <ScanOperation<ArrayType, XlaConstant, Self> as DifferentiableOperation<C>>::jvp(
                operation, context, inputs,
            ),
            Self::CustomJvp(operation) => {
                <CustomJvpOperation<ArrayType, XlaConstant, Self> as DifferentiableOperation<C>>::jvp(
                    operation, context, inputs,
                )
            }
            Self::CustomVjp(operation) => {
                <CustomVjpOperation<ArrayType, XlaConstant, Self> as DifferentiableOperation<C>>::jvp(
                    operation, context, inputs,
                )
            }
            Self::Rematerialize(operation) => {
                <RematerializeOperation<ArrayType, XlaConstant, Self> as DifferentiableOperation<C>>::jvp(
                    operation, context, inputs,
                )
            }
            Self::JitCall(operation) => operation.jvp(context, inputs),
            Self::ShardMap(operation) => operation.jvp_with_staging_context(context, inputs),
            Self::LinearShardMap(operation) => operation.jvp_with_staging_context(context, inputs),
        }
    }
}

impl<C> LinearizableProgramOperation<C> for XlaOperation
where
    C: DifferentiationContext<
        Type = ArrayType,
        Constant = XlaConstant,
        LinearOperation<
            <C as DifferentiationContext>::Tangent,
            ValueOrCapture<ArrayType, <C as Domain>::Value>,
        > = LinearXlaOperation<
            <C as DifferentiationContext>::Tangent,
            XlaConstant,
            ValueOrCapture<ArrayType, <C as Domain>::Value>,
        >,
    >,
    Self: Clone + Operation<ArrayType>,
    ZeroOperation<ArrayType>: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    ZeroLikeOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    OneOperation<ArrayType>: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    OneLikeOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    ConstantOperation<ArrayType, XlaConstant>: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    FillOperation<ArrayType, f64>: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    NegOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    AddOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    SubOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    ScaleOperation<ArrayType, XlaConstant>: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    MulOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    DivOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    SinOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    CosOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    StopGradientOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    RematerializationNameOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    TransferToMemoryOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    DotOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    TransposeOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    ReshapeOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    ReshardOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    ShardingConstraintOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    BroadcastOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    SliceOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    UpdateSliceOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    DynamicSliceOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    DynamicUpdateSliceOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    PadOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    ConcatenateOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    GatherOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    ScatterOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    ReduceOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    CompareOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    NotOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    AndOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    OrOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    XorOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    CollectiveOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    SelectOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    JitCallOperation: DifferentiableOperation<LinearizationContextOf<C, Self>>,
    C::Tangent: Transpose + Broadcast + ReduceValue + Slice + UpdateSlice + Reshape + Reshard + ConstrainSharding,
    <C::Tangent as Value<ArrayType>>::InterpretationContext: Zero<ArrayType, C::Tangent>,
    Vec<XlaConstant>: Parameterized<
            XlaConstant,
            Family: ParameterizedFamily<C::Tangent, To = Vec<C::Tangent>>
                + ParameterizedFamily<
                    Tracer<LinearizationContextOf<C, Self>>,
                    To = Vec<Tracer<LinearizationContextOf<C, Self>>>,
                >,
            To<XlaConstant> = Vec<XlaConstant>,
            To<C::Tangent> = Vec<C::Tangent>,
            To<Tracer<LinearizationContextOf<C, Self>>> = Vec<Tracer<LinearizationContextOf<C, Self>>>,
            ParameterStructure: std::fmt::Debug + PartialEq,
        >,
    C::LinearOperation<C::Tangent, XlaConstant>: CaptureParameterizedOperation<
            ArrayType,
            XlaConstant,
            WithCapture<XlaConstant> = C::LinearOperation<C::Tangent, XlaConstant>,
            WithCapture<Tracer<LinearizationContextOf<C, Self>>> = LinearXlaOperation<
                C::Tangent,
                XlaConstant,
                Tracer<LinearizationContextOf<C, Self>>,
            >,
            WithCapture<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>> = LinearXlaOperation<
                C::Tangent,
                XlaConstant,
                ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>,
            >,
        >,
    <LinearizationContextOf<C, Self> as DifferentiationContext>::LinearOperation<
        C::Tangent,
        ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>,
    >: ResidualizedOperation<LinearizationContextOf<C, Self>>
        + CaptureParameterizedOperation<
            ArrayType,
            ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>,
            WithCapture<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>> = <LinearizationContextOf<
                C,
                Self,
            > as DifferentiationContext>::LinearOperation<
                C::Tangent,
                ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<C, Self>>>,
            >,
            WithCapture<ValueOrCapture<ArrayType, <C as Domain>::Value>> = C::LinearOperation<C::Tangent, ValueOrCapture<C::Type, C::Value>>,
        > + MaybeZeroOperation<ArrayType>,
{
    fn linearize_program(
        differentiable: &C,
        program: &Program<ArrayType, XlaConstant, Self, Vec<XlaConstant>, Vec<XlaConstant>>,
    ) -> Result<NestedLinearization<C, Self>, ProgramError> {
        program.linearize(differentiable)
    }
}

/// Staged XLA program specialized to the backend-owned XLA op universe.
pub type XlaProgram<Input, Output> = Program<ArrayType, XlaConstant, XlaOperation, Input, Output>;

/// Program builder specialized to the backend-owned XLA op universe.
pub type XlaProgramBuilder = ProgramBuilder<ArrayType, XlaConstant, XlaOperation>;

/// Flat XLA program payload used by staged call operations.
pub type FlatXlaProgram = XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>;

/// Linear staged-op universe owned by the XLA backend.
#[derive(Clone, Debug, Operation, ryft_macros::TransposableOperation)]
#[ryft(crate = "ryft_core")]
#[ryft(bounds(interpretation(BooleanLike + Slice + UpdateSlice + Reshape)))]
pub enum LinearXlaOperation<
    V: Value<ArrayType>,
    Constant: Value<ArrayType> = XlaConstant,
    F: Value<ArrayType> = V,
    P: Clone + Operation<ArrayType> = XlaOperation,
    CaptureFactor: Value<ArrayType> = F,
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
    /// Backend-owned captured-predicate condition whose branch bodies can contain XLA linear operations.
    Condition(ConditionOperation<ArrayType, V, Self, F, Captured>),

    /// Backend-owned while-condition whose predicate is supplied by the fused loop state.
    WhileCondition(ConditionOperation<ArrayType, V, Self, V, Input>),

    /// Backend-owned while loop whose nested programs can contain XLA linear operations.
    While(Box<WhileOperation<ArrayType, V, Self, Input>>),

    /// Backend-owned scan whose body can contain XLA linear operations.
    Scan(Box<ScanOperation<ArrayType, V, LinearXlaOperation<V, Constant, ValueOrCapture<ArrayType, V>, P>, F, Input>>),

    CustomVjpCall(Box<CustomVjpCallOperation<ArrayType, Constant, P, F>>),

    RematerializeCall(Box<RematerializeCallOperation<ArrayType, Constant, P, F>>),

    /// Linearized call to a jitted XLA sub-program.
    LinearJitCall(Box<LinearJitCallOperation<CaptureFactor>>),

    /// XLA-specific linear `shard_map`.
    LinearShardMap(Box<LinearShardMapOperation<V, CaptureFactor>>),
}

impl<V, Constant, F, P, CaptureFactor> LinearXlaOperation<V, Constant, F, P, CaptureFactor>
where
    V: Value<ArrayType>,
    Constant: Value<ArrayType>,
    F: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
    CaptureFactor: Value<ArrayType>,
{
    fn to_core_linear_array_operation(&self) -> Option<LinearArrayOperation<V, Constant, F, P>> {
        Some(match self {
            Self::Zero(operation) => LinearArrayOperation::from(operation.clone()),
            Self::ZeroLike(operation) => LinearArrayOperation::from(operation.clone()),
            Self::One(operation) => LinearArrayOperation::from(operation.clone()),
            Self::OneLike(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Constant(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Fill(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Neg(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Add(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Sub(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Scale(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Mul(operation) => LinearArrayOperation::from(operation.clone()),
            Self::TransferToMemory(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Transpose(operation) => LinearArrayOperation::from(operation.clone()),
            Self::LeftDot(operation) => LinearArrayOperation::from(operation.clone()),
            Self::RightDot(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Reshape(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Reshard(operation) => LinearArrayOperation::from(operation.clone()),
            Self::ShardingConstraint(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Broadcast(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Slice(operation) => LinearArrayOperation::from(operation.clone()),
            Self::UpdateSlice(operation) => LinearArrayOperation::from(operation.clone()),
            Self::DynamicSlice(operation) => LinearArrayOperation::from(operation.clone()),
            Self::DynamicUpdateSlice(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Gather(operation) => LinearArrayOperation::from(operation.clone()),
            Self::ScatterAdd(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Pad(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Concatenate(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Reduce(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Select(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Residual(operation) => LinearArrayOperation::from(operation.clone()),
            Self::Recompute(operation) => LinearArrayOperation::from(operation.clone()),
            Self::CustomVjpCall(operation) => LinearArrayOperation::from((**operation).clone()),
            Self::RematerializeCall(operation) => LinearArrayOperation::from((**operation).clone()),
            Self::Condition(_)
            | Self::WhileCondition(_)
            | Self::While(_)
            | Self::Scan(_)
            | Self::LinearJitCall(_)
            | Self::LinearShardMap(_) => return None,
        })
    }
}

impl<V, Constant, F, P, CaptureFactor> From<LinearArrayOperation<V, Constant, F, P>>
    for LinearXlaOperation<V, Constant, F, P, CaptureFactor>
where
    V: Value<ArrayType>,
    Constant: Value<ArrayType>,
    F: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
    CaptureFactor: Value<ArrayType>,
{
    fn from(operation: LinearArrayOperation<V, Constant, F, P>) -> Self {
        match operation {
            LinearArrayOperation::Zero(operation) => Self::Zero(operation),
            LinearArrayOperation::ZeroLike(operation) => Self::ZeroLike(operation),
            LinearArrayOperation::One(operation) => Self::One(operation),
            LinearArrayOperation::OneLike(operation) => Self::OneLike(operation),
            LinearArrayOperation::Constant(operation) => Self::Constant(operation),
            LinearArrayOperation::Fill(operation) => Self::Fill(operation),
            LinearArrayOperation::Neg(operation) => Self::Neg(operation),
            LinearArrayOperation::Add(operation) => Self::Add(operation),
            LinearArrayOperation::Sub(operation) => Self::Sub(operation),
            LinearArrayOperation::Scale(operation) => Self::Scale(operation),
            LinearArrayOperation::Mul(operation) => Self::Mul(operation),
            LinearArrayOperation::TransferToMemory(operation) => Self::TransferToMemory(operation),
            LinearArrayOperation::Transpose(operation) => Self::Transpose(operation),
            LinearArrayOperation::LeftDot(operation) => Self::LeftDot(operation),
            LinearArrayOperation::RightDot(operation) => Self::RightDot(operation),
            LinearArrayOperation::Reshape(operation) => Self::Reshape(operation),
            LinearArrayOperation::Reshard(operation) => Self::Reshard(operation),
            LinearArrayOperation::ShardingConstraint(operation) => Self::ShardingConstraint(operation),
            LinearArrayOperation::Broadcast(operation) => Self::Broadcast(operation),
            LinearArrayOperation::Slice(operation) => Self::Slice(operation),
            LinearArrayOperation::UpdateSlice(operation) => Self::UpdateSlice(operation),
            LinearArrayOperation::DynamicSlice(operation) => Self::DynamicSlice(operation),
            LinearArrayOperation::DynamicUpdateSlice(operation) => Self::DynamicUpdateSlice(operation),
            LinearArrayOperation::Gather(operation) => Self::Gather(operation),
            LinearArrayOperation::ScatterAdd(operation) => Self::ScatterAdd(operation),
            LinearArrayOperation::Pad(operation) => Self::Pad(operation),
            LinearArrayOperation::Concatenate(operation) => Self::Concatenate(operation),
            LinearArrayOperation::Reduce(operation) => Self::Reduce(operation),
            LinearArrayOperation::Select(operation) => Self::Select(operation),
            LinearArrayOperation::Residual(operation) => Self::Residual(operation),
            LinearArrayOperation::Recompute(operation) => Self::Recompute(operation),
            LinearArrayOperation::Condition(operation) => Self::Condition(
                ConditionOperation::new_captured(
                    operation.predicate().clone(),
                    operation
                        .true_branch()
                        .map_operations(|operation| Ok(LinearXlaOperation::from(operation.clone())))
                        .unwrap(),
                    operation
                        .false_branch()
                        .map_operations(|operation| Ok(LinearXlaOperation::from(operation.clone())))
                        .unwrap(),
                )
                .unwrap(),
            ),
            LinearArrayOperation::WhileCondition(operation) => Self::WhileCondition(
                ConditionOperation::new(
                    operation
                        .true_branch()
                        .map_operations(|operation| Ok(LinearXlaOperation::from(operation.clone())))
                        .unwrap(),
                    operation
                        .false_branch()
                        .map_operations(|operation| Ok(LinearXlaOperation::from(operation.clone())))
                        .unwrap(),
                )
                .unwrap(),
            ),
            LinearArrayOperation::While(operation) => Self::While(Box::new(
                WhileOperation::new(
                    operation
                        .condition()
                        .map_operations(|operation| Ok(LinearXlaOperation::from(operation.clone())))
                        .unwrap(),
                    operation
                        .body()
                        .map_operations(|operation| Ok(LinearXlaOperation::from(operation.clone())))
                        .unwrap(),
                )
                .unwrap()
                .with_iteration_bound(operation.iteration_bound())
                .unwrap(),
            )),
            LinearArrayOperation::Scan(operation) => LinearXlaOperation::from(*operation),
            LinearArrayOperation::CustomVjpCall(operation) => Self::CustomVjpCall(operation),
            LinearArrayOperation::RematerializeCall(operation) => Self::RematerializeCall(operation),
        }
    }
}

impl<V, Constant, F, P, CaptureFactor>
    From<ScanOperation<ArrayType, V, LinearArrayOperation<V, Constant, ValueOrCapture<ArrayType, V>, P>, F, Input>>
    for LinearXlaOperation<V, Constant, F, P, CaptureFactor>
where
    V: Value<ArrayType>,
    Constant: Value<ArrayType>,
    F: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
    CaptureFactor: Value<ArrayType>,
{
    fn from(
        operation: ScanOperation<
            ArrayType,
            V,
            LinearArrayOperation<V, Constant, ValueOrCapture<ArrayType, V>, P>,
            F,
            Input,
        >,
    ) -> Self {
        let body = operation
            .body()
            .map_operations(|operation| Ok(LinearXlaOperation::from(operation.clone())))
            .unwrap();
        let scan = ScanOperation::<
            ArrayType,
            V,
            LinearXlaOperation<V, Constant, ValueOrCapture<ArrayType, V>, P>,
            F,
            Input,
        >::new_with_payload(body, operation.carry_count(), operation.length())
        .unwrap()
        .with_reverse(operation.reverse())
        .with_unroll(operation.unroll())
        .unwrap()
        .with_captures(operation.captures().to_vec());
        Self::Scan(Box::new(scan))
    }
}

fn clone_capture<F: Clone>(factor: &F) -> Result<F, ProgramError> {
    Ok(factor.clone())
}

fn map_linear_xla_operation_captures<V, Constant, F, MappedFactor, P, MapFactorFn>(
    operation: &LinearXlaOperation<V, Constant, F, P, F>,
    map_factor: &mut MapFactorFn,
) -> Result<LinearXlaOperation<V, Constant, MappedFactor, P, MappedFactor>, ProgramError>
where
    V: Value<ArrayType>,
    Constant: Value<ArrayType>,
    F: Value<ArrayType>,
    MappedFactor: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
    MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
{
    if let Some(operation) = operation.to_core_linear_array_operation() {
        return Ok(LinearXlaOperation::from(operation.try_map_captures(map_factor)?));
    }
    match operation {
        LinearXlaOperation::Condition(operation) => {
            Ok(LinearXlaOperation::Condition(ConditionOperation::new_captured(
                map_factor(operation.predicate())?,
                operation
                    .true_branch()
                    .map_operations(|operation| map_linear_xla_operation_captures(operation, map_factor))?,
                operation
                    .false_branch()
                    .map_operations(|operation| map_linear_xla_operation_captures(operation, map_factor))?,
            )?))
        }
        LinearXlaOperation::WhileCondition(operation) => {
            Ok(LinearXlaOperation::WhileCondition(ConditionOperation::new(
                operation
                    .true_branch()
                    .map_operations(|operation| map_linear_xla_operation_captures(operation, map_factor))?,
                operation
                    .false_branch()
                    .map_operations(|operation| map_linear_xla_operation_captures(operation, map_factor))?,
            )?))
        }
        LinearXlaOperation::While(operation) => {
            let condition = operation
                .condition()
                .map_operations(|operation| map_linear_xla_operation_captures(operation, map_factor))?;
            let body = operation
                .body()
                .map_operations(|operation| map_linear_xla_operation_captures(operation, map_factor))?;
            Ok(LinearXlaOperation::While(Box::new(
                WhileOperation::new(condition, body)?.with_iteration_bound(operation.iteration_bound())?,
            )))
        }
        LinearXlaOperation::Scan(operation) => {
            let mut clone_scan_local_capture = clone_capture::<ValueOrCapture<ArrayType, V>>
                as fn(&ValueOrCapture<ArrayType, V>) -> Result<ValueOrCapture<ArrayType, V>, ProgramError>;
            let body = operation.body().map_operations(|operation| {
                map_linear_xla_operation_captures(operation, &mut clone_scan_local_capture)
            })?;
            let scan = ScanOperation::<
                ArrayType,
                V,
                LinearXlaOperation<V, Constant, ValueOrCapture<ArrayType, V>, P>,
                MappedFactor,
                Input,
            >::new_with_payload(body, operation.carry_count(), operation.length())?
            .with_reverse(operation.reverse())
            .with_unroll(operation.unroll())?
            .with_captures(operation.captures().iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?);
            Ok(LinearXlaOperation::Scan(Box::new(scan)))
        }
        LinearXlaOperation::LinearJitCall(operation) => {
            Ok(LinearXlaOperation::LinearJitCall(Box::new(operation.map_captured_inputs(map_factor)?)))
        }
        LinearXlaOperation::LinearShardMap(operation) => {
            Ok(LinearXlaOperation::LinearShardMap(Box::new(operation.map_captured_global_primals(map_factor)?)))
        }
        _ => unreachable!("linear XLA leaf operation should convert to a core linear operation"),
    }
}

impl<V, Constant, F, P> CaptureParameterizedOperation<ArrayType, F> for LinearXlaOperation<V, Constant, F, P, F>
where
    V: Value<ArrayType>,
    Constant: Value<ArrayType>,
    F: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
{
    type WithCapture<MappedFactor: Value<ArrayType>> = LinearXlaOperation<V, Constant, MappedFactor, P, MappedFactor>;

    fn try_map_captures<MappedFactor: Value<ArrayType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithCapture<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        map_linear_xla_operation_captures(self, map_factor)
    }
}

impl<V, Constant, R, P> DefactorizableProgramOperation<V, R, P>
    for LinearXlaOperation<V, Constant, ValueOrCapture<ArrayType, R>, P, ValueOrCapture<ArrayType, R>>
where
    V: Value<ArrayType>,
    Constant: Value<ArrayType>,
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

/// Staged call to a flat jitted XLA program.
#[derive(Clone, Debug)]
pub struct JitCallOperation {
    /// Flat callee program called by this operation. Shared via [`Rc`] so repeated calls staged from the same
    /// function handle carry one program and remain identity-comparable for call-site deduplication at lowering.
    program: Rc<FlatXlaProgram>,
}

impl JitCallOperation {
    /// Creates a staged jitted-call operation for `program`.
    #[inline]
    pub(crate) fn new(program: Rc<FlatXlaProgram>) -> Self {
        Self { program }
    }

    /// Returns the flat callee program.
    #[inline]
    pub(crate) fn program(&self) -> &FlatXlaProgram {
        self.program.as_ref()
    }

    /// Returns the shared handle to the flat callee program, used for call-site deduplication at lowering.
    #[inline]
    pub(crate) fn program_rc(&self) -> &Rc<FlatXlaProgram> {
        &self.program
    }
}

/// Linearized jitted call used inside tangent and cotangent programs.
///
/// The captured primal prefix inputs are stored as factors of the linear program's factor carrier `F`
/// ([`ValueOrCapture`] references in residualized pushforwards, concrete values in instantiated direct programs),
/// so they flow through residual compaction, rebasing, and instantiation like every other captured primal factor.
#[derive(Clone, Debug)]
pub struct LinearJitCallOperation<F: Value<ArrayType>> {
    /// Program applied by this linear call. Its inputs are `captured_inputs` followed by the operation inputs.
    /// Shared via [`Rc`] so transposed clones carry one program and remain identity-comparable for call-site
    /// deduplication at lowering.
    program: Rc<FlatXlaProgram>,

    /// Program for the transposed linear call with the same captured prefix inputs.
    transpose_program: Rc<FlatXlaProgram>,

    /// Captured primal prefix inputs supplied to `program` before the linear operation inputs, stored as factors.
    captured_inputs: Vec<F>,

    /// Flat linear input types expected by this operation.
    input_types: Vec<ArrayType>,

    /// Flat output types produced by this operation.
    output_types: Vec<ArrayType>,
}

impl<F: Value<ArrayType>> LinearJitCallOperation<F> {
    /// Creates a linear jitted-call operation.
    fn new(
        program: Rc<FlatXlaProgram>,
        transpose_program: Rc<FlatXlaProgram>,
        captured_inputs: Vec<F>,
        input_types: Vec<ArrayType>,
        output_types: Vec<ArrayType>,
    ) -> Self {
        Self { program, transpose_program, captured_inputs, input_types, output_types }
    }

    /// Maps this call's captured prefix factors through `map_factor`, preserving the carried programs and types.
    fn map_captured_inputs<MappedFactor: Value<ArrayType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<LinearJitCallOperation<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        Ok(LinearJitCallOperation::new(
            self.program.clone(),
            self.transpose_program.clone(),
            self.captured_inputs.iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
            self.input_types.clone(),
            self.output_types.clone(),
        ))
    }
}

fn missing_traced_input() -> ProgramError {
    ProgramError::InvalidInputCount { expected: 1, actual: 0 }
}

fn ensure_call_input_types(
    operation_name: &'static str,
    expected_types: &[ArrayType],
    input_types: &[ArrayType],
) -> Result<(), TypeError> {
    if expected_types.len() != input_types.len() {
        return Err(TypeError {
            message: format!(
                "{operation_name} expected {} input(s) but got {}",
                expected_types.len(),
                input_types.len(),
            ),
        });
    }
    for (index, (expected, actual)) in expected_types.iter().zip(input_types).enumerate() {
        if expected != actual {
            return Err(TypeError {
                message: format!("{operation_name} input #{index} expected {expected} but got {actual}"),
            });
        }
    }
    Ok(())
}

fn build_jvp_call_program(program: &FlatXlaProgram) -> Result<FlatXlaProgram, ProgramError> {
    let input_types = program.input_types();
    let signature = input_types.iter().cloned().chain(input_types.iter().cloned()).collect::<Vec<_>>();
    let token = XlaDomain::token();
    let (_, traced): (Vec<ArrayType>, FlatXlaProgram) = TracingContext::trace(
        token,
        |inputs: Vec<XlaTracer<'static, 'static>>| -> Result<Vec<XlaTracer<'static, 'static>>, ProgramError> {
            let input_count = inputs.len() / 2;
            let primals = inputs[..input_count].to_vec();
            let tangents = inputs[input_count..].to_vec();
            let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
            let (_, pushforward) = context.linearize(
                |linearized_inputs| {
                    let linearization_context =
                        linearized_inputs.first().ok_or_else(missing_traced_input)?.context().clone();
                    linearization_context.stage_program(program, linearized_inputs)
                },
                primals,
            )?;
            pushforward.apply(&context, tangents)
        },
        signature,
    )?;
    traced.into_simplified()
}

fn build_pullback_call_program(program: &FlatXlaProgram) -> Result<FlatXlaProgram, ProgramError> {
    let input_types = program.input_types();
    let output_types = program.output_types();
    let signature = input_types.iter().cloned().chain(output_types.iter().cloned()).collect::<Vec<_>>();
    let token = XlaDomain::token();
    let (_, traced): (Vec<ArrayType>, FlatXlaProgram) = TracingContext::trace(
        token,
        |inputs: Vec<XlaTracer<'static, 'static>>| -> Result<Vec<XlaTracer<'static, 'static>>, ProgramError> {
            let input_count = input_types.len();
            let primals = inputs[..input_count].to_vec();
            let cotangents = inputs[input_count..].to_vec();
            let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
            let (_, pullback) = context.vjp(
                |linearized_inputs| {
                    let linearization_context =
                        linearized_inputs.first().ok_or_else(missing_traced_input)?.context().clone();
                    linearization_context.stage_program(program, linearized_inputs)
                },
                primals,
            )?;
            let interpretation_context = cotangents
                .iter()
                .find_map(|cotangent| cotangent.interpretation_context())
                .ok_or_else(missing_traced_input)?;
            pullback.interpret_in_context(&interpretation_context, cotangents)
        },
        signature,
    )?;
    traced.into_simplified()
}

fn build_batched_call_program(
    program: &FlatXlaProgram,
    input_axes: &[Option<usize>],
    axis_size: usize,
) -> Result<(FlatXlaProgram, Vec<Option<usize>>), ProgramError> {
    let logical_input_types = program.input_types();
    check_count!("input", input_axes, logical_input_types.len(), ProgramError);
    let physical_input_types = logical_input_types
        .iter()
        .zip(input_axes)
        .map(|(logical_type, axis)| match axis {
            Some(axis) => logical_type.with_inserted_dimension(*axis, Size::Static(axis_size)),
            None => Ok(logical_type.clone()),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
    let parent_context = TracingContext::new(XlaDomain::token(), builder.clone());
    let batching_context = BatchingContext::new(parent_context, axis_size);
    let mut input_tracers = Vec::with_capacity(physical_input_types.len());
    for ((physical_type, logical_type), axis) in physical_input_types.iter().zip(&logical_input_types).zip(input_axes) {
        let atom = builder.borrow_mut().add_input(physical_type.clone());
        batching_context.register_axis(atom, *axis);
        input_tracers.push(batching_context.tracer(atom, Some(logical_type.clone())));
    }
    let output_tracers = batching_context.stage_program(program, input_tracers)?;
    let output_atom_ids = output_tracers.iter().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
    let output_axes = output_atom_ids.iter().map(|atom| batching_context.axis_for(*atom)).collect::<Vec<_>>();
    drop(output_tracers);
    drop(batching_context);

    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let batched_program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
            output_atom_ids,
            vec![Placeholder; physical_input_types.len()],
            vec![Placeholder; output_axes.len()],
        )?
        .into_simplified()?;
    Ok((batched_program, output_axes))
}

impl Operation<ArrayType> for JitCallOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "jit_call"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ensure_call_input_types(self.name(), self.program.input_types().as_slice(), input_types)?;
        Ok(self.program.output_types())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("inputs", self.program.input_ids().len())?;
            operation.field("outputs", self.program.output_ids().len())
        })
    }
}

impl JitCallOperation {
    /// Creates the linear call operation corresponding to this ordinary call, capturing the primal inputs as
    /// `captured_inputs` factors.
    fn linear_call_operation<F: Value<ArrayType>>(
        &self,
        captured_inputs: Vec<F>,
    ) -> Result<LinearJitCallOperation<F>, ProgramError> {
        Ok(LinearJitCallOperation::new(
            Rc::new(build_jvp_call_program(self.program())?),
            Rc::new(build_pullback_call_program(self.program())?),
            captured_inputs,
            self.program.input_types(),
            self.program.output_types(),
        ))
    }

    /// Returns the call operation and output-axis metadata for batching this call.
    fn batched_call_operation<V: Typed<ArrayType>>(
        &self,
        inputs: &[ArrayBatch<V>],
    ) -> Result<(Self, Vec<Option<usize>>), ProgramError> {
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        let axis_size = input_axes.iter().any(Option::is_some).then_some(axis_size);
        match axis_size {
            Some(axis_size) => {
                let (batched_program, output_axes) =
                    build_batched_call_program(&self.program, input_axes.as_slice(), axis_size)?;
                Ok((JitCallOperation::new(Rc::new(batched_program)), output_axes))
            }
            None => Ok((self.clone(), vec![None; self.program.output_types().len()])),
        }
    }

    /// Completes the JVP rule after the caller has produced primal outputs in its host representation.
    ///
    /// The primal inputs are captured as residual factors through [`JvpTracer::factor`] — environment references
    /// under reusable (staged) linearization, closed constants under direct execution — so the staged linear call
    /// participates in residual compaction, rebasing, and instantiation. The primal and tangent carriers are kept
    /// separate so the rule also serves nested symbolic linearization contexts, whose primal values are nested
    /// tracers while tangents stay in the enclosing context's representation.
    fn jvp_from_primal_outputs<'jvp, C, PrimalValue, TangentValue>(
        &self,
        context: &mut TangentContext<'jvp, C>,
        inputs: &[JvpTracer<'jvp, C>],
        primal_outputs: Vec<PrimalValue>,
    ) -> Result<Vec<JvpTracer<'jvp, C>>, ProgramError>
    where
        PrimalValue: Value<ArrayType>,
        TangentValue: Value<ArrayType> + Slice + UpdateSlice + Reshape,
        TangentValue::InterpretationContext: Zero<ArrayType, TangentValue>,
        C: DifferentiationContext<
                Tangent = TangentValue,
                LinearOperation<TangentValue, ValueOrCapture<ArrayType, PrimalValue>> = LinearXlaOperation<
                    TangentValue,
                    XlaConstant,
                    ValueOrCapture<ArrayType, PrimalValue>,
                >,
            > + Domain<Type = ArrayType, Value = PrimalValue>
            + 'jvp,
        C::LinearOperation<C::Tangent, ValueOrCapture<C::Type, C::Value>>: From<ZeroOperation<ArrayType>>,
    {
        let captured_inputs = inputs.iter().map(|input| input.factor(context)).collect::<Vec<_>>();
        let tangent_inputs = inputs.iter().map(|input| input.tangent().clone()).collect::<Vec<_>>();
        let linear_operation = self.linear_call_operation(captured_inputs)?;
        let operation: LinearXlaOperation<TangentValue, XlaConstant, ValueOrCapture<ArrayType, PrimalValue>> =
            LinearXlaOperation::LinearJitCall(Box::new(linear_operation));
        let tangent_outputs = context.stage_operation(operation, tangent_inputs.as_slice())?;
        check_count!("output", tangent_outputs, primal_outputs.len(), ProgramError);
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
            .collect())
    }
}

impl<C> BatchableOperation<ArrayType, C> for JitCallOperation {
    fn batch(
        &self,
        _context: &C,
        inputs: &[ArrayBatch<ArrayType>],
    ) -> Result<Vec<ArrayBatch<ArrayType>>, ProgramError> {
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let (operation, output_axes) = self.batched_call_operation(inputs)?;
        let outputs = operation.infer_output_types(physical_inputs.as_slice())?;
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output.r#type().into_owned(), output, axis))
            .collect()
    }
}

impl<S, C> BatchableOperation<Tracer<S>, C> for JitCallOperation
where
    S: StagingContext<Type = ArrayType, Operation = XlaOperation>,
{
    fn batch(
        &self,
        _context: &C,
        inputs: &[ArrayBatch<Tracer<S>>],
    ) -> Result<Vec<ArrayBatch<Tracer<S>>>, ProgramError> {
        let context = inputs.first().ok_or_else(missing_traced_input)?.value().context().clone();
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let (operation, output_axes) = self.batched_call_operation(inputs)?;
        let outputs = context.stage_operation(operation, physical_inputs.as_slice())?;
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output.r#type().into_owned(), output, axis))
            .collect()
    }
}

/// Forward-mode rule for staged jitted calls against any staging differentiation context: the primal `jit_call` is
/// staged into the context's primal program through [`TangentContext::bind_primal`] and the linear call captures
/// the primal inputs as residual factors. This serves both ordinary XLA tracing contexts and nested symbolic
/// linearization contexts.
impl<C> DifferentiableOperation<C> for JitCallOperation
where
    C: StagingContext<Type = ArrayType, Constant = XlaConstant, Operation = XlaOperation>
        + DifferentiationContext<
            LinearOperation<
                <C as DifferentiationContext>::Tangent,
                ValueOrCapture<ArrayType, Tracer<C>>,
            > = LinearXlaOperation<
                <C as DifferentiationContext>::Tangent,
                XlaConstant,
                ValueOrCapture<ArrayType, Tracer<C>>,
            >,
        >,
    C::Tangent: Slice + UpdateSlice + Reshape,
    <C::Tangent as Value<ArrayType>>::InterpretationContext: Zero<ArrayType, C::Tangent>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, C>,
        inputs: &[JvpTracer<'jvp, C>],
    ) -> Result<Vec<JvpTracer<'jvp, C>>, ProgramError>
    where
        C: 'jvp,
        C::LinearOperation<C::Tangent, ValueOrCapture<C::Type, C::Value>>: From<ZeroOperation<ArrayType>>,
    {
        check_count!("input", inputs, self.program.input_types().len(), ProgramError);
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = context.bind_primal(
            XlaOperation::JitCall(Box::new(self.clone())),
            primals.as_slice(),
        )?;
        self.jvp_from_primal_outputs(context, inputs, primal_outputs)
    }
}

impl<F: Value<ArrayType>> Operation<ArrayType> for LinearJitCallOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        "linear_jit_call"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ensure_call_input_types(self.name(), self.input_types.as_slice(), input_types)?;
        Ok(self.output_types.clone())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("captured", self.captured_inputs.len())?;
            operation.field("inputs", self.input_types.len())?;
            operation.field("outputs", self.output_types.len())
        })
    }
}

impl<C> InterpretableOperation<ArrayType, Tracer<C>> for LinearJitCallOperation<Tracer<C>>
where
    C: StagingContext<Type = ArrayType, Operation = XlaOperation>,
{
    fn interpret(&self, _context: &C, inputs: &[Tracer<C>]) -> Result<Vec<Tracer<C>>, ProgramError> {
        let context = self
            .captured_inputs
            .first()
            .or_else(|| inputs.first())
            .ok_or_else(missing_traced_input)?
            .context()
            .clone();
        let full_inputs = self.captured_inputs.iter().cloned().chain(inputs.iter().cloned()).collect::<Vec<_>>();
        context.stage_operation(
            XlaOperation::JitCall(Box::new(JitCallOperation::new(self.program.clone()))),
            full_inputs.as_slice(),
        )
    }
}

impl<V, Factor, Target> TransposableOperation<ArrayType, V, Target> for LinearJitCallOperation<Factor>
where
    V: Value<ArrayType>,
    Factor: Value<ArrayType>,
    Target: Operation<ArrayType> + From<ZeroOperation<ArrayType>> + From<LinearJitCallOperation<Factor>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, Target>,
        _input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, Target>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, Target>>, ProgramError> {
        check_count!("output", output_cotangents, self.output_types.len(), ProgramError);
        let mut cotangent_inputs = Vec::with_capacity(output_cotangents.len());
        for (cotangent, output_type) in output_cotangents.iter().zip(self.output_types.iter()) {
            match cotangent {
                Cotangent::Staged(cotangent) => cotangent_inputs.push(cotangent.clone()),
                Cotangent::Zero => {
                    let zero_outputs = context.stage_operation(
                        Target::from(ZeroOperation::new(output_type.clone())),
                        &[] as &[ryft_core::tracing::AbstractTracer<'transpose, ArrayType, V, Target>],
                    )?;
                    check_count!("output", zero_outputs, 1, ProgramError);
                    cotangent_inputs.push(zero_outputs[0].clone());
                }
            }
        }
        let transposed = LinearJitCallOperation::new(
            self.transpose_program.clone(),
            self.program.clone(),
            self.captured_inputs.clone(),
            self.output_types.clone(),
            self.input_types.clone(),
        );
        let input_cotangents = context.stage_operation(Target::from(transposed), cotangent_inputs.as_slice())?;
        Ok(input_cotangents.into_iter().map(Cotangent::Staged).collect())
    }
}
