use std::ops::BitAnd;
use std::rc::Rc;

use ryft_macros::{BatchableOperation, DifferentiableOperation, Operation, TransposableOperation};

use ryft_core::batching::ArrayBatch;
use ryft_core::batching::BatchAxis;
use ryft_core::batching::BatchableOperation;
use ryft_core::batching::BatchingContext;
use ryft_core::batching::BatchingError;
use ryft_core::batching::ProgramBatchingOutputAxesPolicy;
use ryft_core::captures::CaptureReference;
use ryft_core::compilation::function::CompiledProgramOperation;
use ryft_core::contexts::{Context, StagingContext};
use ryft_core::differentiation::{DifferentiableOperation, DifferentiationError, TransposableOperation};
use ryft_core::effects::Effects;
use ryft_core::interpretation::InterpretableOperation;
use ryft_core::macros::check_count;
use ryft_core::operations::compare::CompareOperation;
use ryft_core::operations::complex::{ComplexOperation, ConjugateOperation, ImaginaryOperation, RealOperation};
use ryft_core::operations::constants::{
    ConstantOperation, FillOperation, IotaOperation, OneLikeOperation, OneOperation, Zero, ZeroLikeOperation,
    ZeroOperation,
};
use ryft_core::operations::control_flow::{
    ConditionOperation, MaybeWhile, ScanOperation, Select, SelectOperation, WhileOperation, WhileParts, WhilePredicate,
};
use ryft_core::operations::differentiation::StopGradientOperation;
use ryft_core::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use ryft_core::operations::manipulation::{
    Broadcast, BroadcastOperation, ConcatenateOperation, DynamicSliceOperation, DynamicUpdateSliceOperation,
    GatherOperation, PadOperation, Reshape, ReshapeOperation, ScatterOperation, Slice, SliceOperation, Transpose,
    TransposeOperation, UpdateSlice, UpdateSliceOperation,
};
use ryft_core::operations::math::{
    AbsOperation, AddOperation, Atan2Operation, CosOperation, DivOperation, MulOperation, NegOperation, SinOperation,
    SubOperation,
};
use ryft_core::operations::math::{ExpOperation, LogOperation, SqrtOperation};
use ryft_core::operations::sharding::{ReshardOperation, ShardingConstraintOperation};
use ryft_core::operations::{BooleanLike, Operation, OperationFormatter};
use ryft_core::partial::{
    PartialEvaluationContext, PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation,
};
use ryft_core::programs::{MaybeZero, Program, ProgramBuilder, ProgramError, Value};
use ryft_core::tracing::{Tracer, TracingContext};

use ryft_core::backends::scalars::Scalar;
use ryft_core::differentiation::DifferentiationDual;
use ryft_core::operations::debugging::PrintOperation;
use ryft_core::operations::tag::TagOperation;
use ryft_core::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpOperation, CustomVjpTangentOperation,
};
use ryft_core::tracing_v2::operations::memory::TransferToMemoryOperation;
use ryft_core::tracing_v2::operations::reduce::{Reduce as ReduceValue, ReduceOperation};
use ryft_core::tracing_v2::rematerialization::RematerializeOperation;
use ryft_core::tracing_v2::{ArrayOperation, AxisIndexOperation, CollectiveOperation, DotOperation};
use ryft_core::types::{ArrayType, TypeError, Typed};

use crate::experimental::operations::ShardMapOperation;

/// Lifetime-free reference to a concrete XLA value captured by a compiled program.
pub type XlaConstant = CaptureReference<ArrayType>;

/// Ordinary staged-operation universe owned by the XLA backend.
///
/// This enum flattens the core array operation payloads directly into the backend-owned operation family. Higher-order
/// operations that own nested programs use XLA-owned bodies so those programs can contain backend-specific operations
/// such as [`jit_call`](JitCallOperation) and [`shard_map`](ShardMapOperation).
#[derive(Clone, Debug, Operation, TransposableOperation, DifferentiableOperation, BatchableOperation)]
#[ryft(crate = "ryft_core")]
#[ryft(bounds(
    interpretation(BooleanLike + WhilePredicate + Slice + UpdateSlice + Reshape),
    partial_evaluation(PartialEq + BooleanLike),
    differentiation(PartialEq + BooleanLike),
    batching(
        BooleanLike + BitAnd<Output = V> + Select<Condition = V> + Broadcast + Transpose + ReduceValue + Slice
            + UpdateSlice + Reshape
    ),
))]
pub enum XlaOperation<V: Value<Type = ArrayType> = XlaConstant> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<V>),
    Fill(FillOperation<ArrayType, Scalar>),
    Iota(IotaOperation<ArrayType>),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Mul(MulOperation),
    Div(DivOperation),
    Sin(SinOperation),
    Cos(CosOperation),
    Atan2(Atan2Operation),
    Exp(ExpOperation),
    Log(LogOperation),
    Sqrt(SqrtOperation),
    Abs(AbsOperation),
    Complex(ComplexOperation),
    Conjugate(ConjugateOperation),
    Real(RealOperation),
    Imaginary(ImaginaryOperation),
    StopGradient(StopGradientOperation),
    Tag(TagOperation),
    Print(PrintOperation),
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
    AxisIndex(AxisIndexOperation),
    Select(SelectOperation),
    /// Backend-owned condition whose branch bodies can contain XLA operations.
    Condition(Box<ConditionOperation<V, Self>>),

    /// Backend-owned loop whose condition and body programs can contain XLA operations.
    While(Box<WhileOperation<V, Self>>),

    /// Backend-owned scan whose body program can contain XLA operations.
    Scan(Box<ScanOperation<V, Self>>),

    /// Backend-owned custom JVP call whose nested programs can contain XLA operations.
    CustomJvp(Box<CustomJvpOperation<V, Self>>),

    /// Backend-owned custom VJP call whose nested programs can contain XLA operations.
    CustomVjp(Box<CustomVjpOperation<V, Self>>),

    /// Backend-owned opaque custom-VJP tangent carrier, staged by the capture-free forward of a
    /// [`CustomVjp`](Self::CustomVjp) call. Its nested backward program can contain XLA operations.
    CustomVjpTangent(Box<CustomVjpTangentOperation<V, Self>>),

    /// Backend-owned rematerialized call whose nested programs can contain XLA operations.
    Rematerialize(Box<RematerializeOperation<V, Self>>),

    /// Call to a flat jitted XLA sub-program.
    JitCall(Box<JitCallOperation>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<V>>),
}

fn map_core_xla_program<V>(
    program: &Program<V, ArrayOperation<V>, Vec<V>, Vec<V>>,
) -> Program<V, XlaOperation<V>, Vec<V>, Vec<V>>
where
    V: Value<Type = ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
{
    program.map_operations(|operation| Ok(XlaOperation::from(operation.clone()))).unwrap()
}

impl<V> From<ArrayOperation<V>> for XlaOperation<V>
where
    V: Value<Type = ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
{
    fn from(operation: ArrayOperation<V>) -> Self {
        match operation {
            ArrayOperation::Zero(operation) => Self::Zero(operation),
            ArrayOperation::ZeroLike(operation) => Self::ZeroLike(operation),
            ArrayOperation::One(operation) => Self::One(operation),
            ArrayOperation::OneLike(operation) => Self::OneLike(operation),
            ArrayOperation::Constant(operation) => Self::Constant(operation),
            ArrayOperation::Fill(operation) => Self::Fill(operation),
            ArrayOperation::Iota(operation) => Self::Iota(operation),
            ArrayOperation::Neg(operation) => Self::Neg(operation),
            ArrayOperation::Add(operation) => Self::Add(operation),
            ArrayOperation::Sub(operation) => Self::Sub(operation),
            ArrayOperation::Mul(operation) => Self::Mul(operation),
            ArrayOperation::Div(operation) => Self::Div(operation),
            ArrayOperation::Sin(operation) => Self::Sin(operation),
            ArrayOperation::Cos(operation) => Self::Cos(operation),
            ArrayOperation::Atan2(operation) => Self::Atan2(operation),
            ArrayOperation::Exp(operation) => Self::Exp(operation),
            ArrayOperation::Log(operation) => Self::Log(operation),
            ArrayOperation::Sqrt(operation) => Self::Sqrt(operation),
            ArrayOperation::Abs(operation) => Self::Abs(operation),
            ArrayOperation::Complex(operation) => Self::Complex(operation),
            ArrayOperation::Conjugate(operation) => Self::Conjugate(operation),
            ArrayOperation::Real(operation) => Self::Real(operation),
            ArrayOperation::Imaginary(operation) => Self::Imaginary(operation),
            ArrayOperation::StopGradient(operation) => Self::StopGradient(operation),
            ArrayOperation::Tag(operation) => Self::Tag(operation),
            ArrayOperation::Print(operation) => Self::Print(operation),
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
            ArrayOperation::AxisIndex(operation) => Self::AxisIndex(operation),
            ArrayOperation::Select(operation) => Self::Select(operation),
            ArrayOperation::Condition(operation) => XlaOperation::from(*operation),
            ArrayOperation::While(operation) => XlaOperation::from(*operation),
            ArrayOperation::Scan(operation) => XlaOperation::from(*operation),
            ArrayOperation::CustomJvp(operation) => XlaOperation::from(*operation),
            ArrayOperation::CustomVjp(operation) => XlaOperation::from(*operation),
            ArrayOperation::CustomVjpTangent(operation) => XlaOperation::from(*operation),
            ArrayOperation::Rematerialize(operation) => XlaOperation::from(*operation),
        }
    }
}

impl<V> From<ConditionOperation<V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<Type = ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
{
    fn from(operation: ConditionOperation<V, ArrayOperation<V>>) -> Self {
        Self::Condition(Box::new(
            ConditionOperation::new(
                map_core_xla_program(operation.true_branch()),
                map_core_xla_program(operation.false_branch()),
            )
            .unwrap(),
        ))
    }
}

impl<V> From<WhileOperation<V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<Type = ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
{
    fn from(operation: WhileOperation<V, ArrayOperation<V>>) -> Self {
        Self::While(Box::new(
            WhileOperation::new(map_core_xla_program(operation.condition()), map_core_xla_program(operation.body()))
                .unwrap()
                .with_iteration_bound(operation.iteration_bound())
                .unwrap(),
        ))
    }
}

impl<V> From<ScanOperation<V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<Type = ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
{
    fn from(operation: ScanOperation<V, ArrayOperation<V>>) -> Self {
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

impl<V> From<CustomJvpOperation<V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<Type = ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
{
    fn from(operation: CustomJvpOperation<V, ArrayOperation<V>>) -> Self {
        Self::CustomJvp(Box::new(
            CustomJvpOperation::new(
                map_core_xla_program(operation.primal()),
                map_core_xla_program(operation.jvp_program()),
            )
            .unwrap(),
        ))
    }
}

impl<V> From<CustomVjpOperation<V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<Type = ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
{
    fn from(operation: CustomVjpOperation<V, ArrayOperation<V>>) -> Self {
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

impl<V> From<CustomVjpTangentOperation<V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<Type = ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
{
    fn from(operation: CustomVjpTangentOperation<V, ArrayOperation<V>>) -> Self {
        Self::CustomVjpTangent(Box::new(CustomVjpTangentOperation::new(
            map_core_xla_program(operation.backward()),
            operation.residual_count(),
            operation.transposed(),
        )))
    }
}

impl<V> From<RematerializeOperation<V, ArrayOperation<V>>> for XlaOperation<V>
where
    V: Value<Type = ArrayType> + BooleanLike + Slice + UpdateSlice + Reshape,
{
    fn from(operation: RematerializeOperation<V, ArrayOperation<V>>) -> Self {
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

/// Residual provenance for [`XlaOperation`]: `scan` outputs align index-wise with the body outputs, so a stacked
/// residual is produced per iteration by the body instruction defining the same-index body output; every other
/// operation is its own producer.
impl<V> ryft_core::tracing_v2::rematerialization::ResidualProvenance<V, XlaOperation<V>> for XlaOperation<V>
where
    V: Value<Type = ArrayType>,
{
    fn residual_provenance(
        &self,
        output_index: usize,
    ) -> ryft_core::tracing_v2::rematerialization::ResidualProducers<'_, V, XlaOperation<V>> {
        use ryft_core::tracing_v2::rematerialization::{NestedResidualSource, ResidualProducers};
        match self {
            Self::Scan(operation) => {
                let body = operation.body();
                ResidualProducers::Nested(vec![NestedResidualSource::new(body, output_index)])
            }
            _ => ResidualProducers::Leaf,
        }
    }
}

impl<V> MaybeWhile<V, XlaOperation<V>> for XlaOperation<V>
where
    V: Value<Type = ArrayType>,
{
    #[inline]
    fn as_while(&self) -> Option<WhileParts<'_, V, XlaOperation<V>>> {
        match self {
            Self::While(operation) => operation.as_while(),
            _ => None,
        }
    }
}

/// Staged replay of a jitted call over tracers stages the call operation whole onto the trace the context owns,
/// preserving the compilation boundary instead of inlining the callee.
impl<C> InterpretableOperation<Tracer<C>, C> for JitCallOperation
where
    C: StagingContext<Type = ArrayType, Operation: From<JitCallOperation>>,
{
    fn interpret(&self, context: &C, inputs: &[Tracer<C>]) -> Result<Vec<Tracer<C>>, ProgramError> {
        context.stage_operation(self.clone(), inputs)
    }
}

/// Staged replay of a captured-body shard map over tracers stages the operation whole onto the trace the context
/// owns, preserving the sharding boundary instead of inlining the local body.
impl<C> InterpretableOperation<Tracer<C>, C> for ShardMapOperation<CaptureReference<ArrayType>>
where
    C: StagingContext<Type = ArrayType, Operation: From<ShardMapOperation<CaptureReference<ArrayType>>>>,
{
    fn interpret(&self, context: &C, inputs: &[Tracer<C>]) -> Result<Vec<Tracer<C>>, ProgramError> {
        context.stage_operation(self.clone(), inputs)
    }
}

/// Staged XLA program specialized to the backend-owned XLA op universe.
pub type XlaProgram<Input, Output> = Program<XlaConstant, XlaOperation, Input, Output>;

/// Program builder specialized to the backend-owned XLA op universe.
pub type XlaProgramBuilder = ProgramBuilder<XlaConstant, XlaOperation>;

/// Flat XLA program payload used by staged call operations.
pub type FlatXlaProgram = XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>;

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

impl CompiledProgramOperation<XlaConstant> for XlaOperation {
    #[inline]
    fn compiled_call(program: Rc<FlatXlaProgram>) -> Self {
        Self::JitCall(Box::new(JitCallOperation::new(program)))
    }
}

/// Bridges canonical internal physical positions to the signed public batching declaration without reintroducing a
/// public `usize` conversion on [`BatchAxis`].
fn batch_axis_from_position(axis: Option<usize>) -> BatchAxis {
    axis.map(|axis| BatchAxis::new(isize::try_from(axis).expect("a physical array rank fits in isize")))
        .unwrap_or_default()
}

/// Recovers a canonical physical position returned by the core program batching pass.
fn batch_axis_position(axis: &BatchAxis) -> Option<usize> {
    axis.axis()
        .map(|axis| usize::try_from(axis).expect("program batching returns canonical nonnegative axes"))
}

fn ensure_call_input_types(
    operation_name: &'static str,
    expected_types: &[ArrayType],
    input_types: &[ArrayType],
) -> Result<(), TypeError> {
    if expected_types.len() != input_types.len() {
        return Err(TypeError {
            message: format!(
                "'{operation_name}' expected {} input(s) but got {}",
                expected_types.len(),
                input_types.len(),
            ),
        });
    }
    for (index, (expected, actual)) in expected_types.iter().zip(input_types).enumerate() {
        if expected != actual {
            return Err(TypeError {
                message: format!("'{operation_name}' input #{index} expected {expected} but got {actual}"),
            });
        }
    }
    Ok(())
}

fn build_batched_call_program(
    program: &FlatXlaProgram,
    batch_axes: &[Option<usize>],
    axis_size: usize,
) -> Result<(FlatXlaProgram, Vec<Option<usize>>), ProgramError> {
    // Delegate to the core program-batching pass, which replays `program` through a `BatchingContext` over a fresh
    // trace and packages the batched program with its natural (rule-produced) output axes.
    let input_batch_axes = batch_axes.iter().copied().map(batch_axis_from_position).collect::<Vec<_>>();
    let (batched_program, output_axes) =
        program.batched(axis_size, input_batch_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)?;
    let output_axes = output_axes.iter().map(batch_axis_position).collect();
    Ok((batched_program.into_simplified()?, output_axes))
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

    #[inline]
    fn effects(&self) -> Effects {
        self.program.effects()
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("inputs", self.program.input_ids().len())?;
            operation.field("outputs", self.program.output_ids().len())
        })
    }
}

/// Online partial-evaluation rule for a staged jitted call — ryft's analogue of JAX's call partial-evaluation
/// rules: it splits the callee against the caller's known-ness while preserving the `jit_call` boundary on both
/// sides.
///
/// The split fires only when some known call input does *not* [`resolve`](Context::resolve) to a concrete
/// constant in the known-side context — i.e., a genuine tracer into a live outer trace, the mixed-online case this
/// rule exists for. All-known, all-unknown, and concrete-known calls defer to the default fold-or-residualize
/// behavior, which preserves the original boundary (and today's eager behavior) exactly.
///
/// When the split fires, the callee is split through the shared
/// [`PartitionedProgram`](ryft_core::partial::PartitionedProgram) machinery: the known side is bound into the
/// enclosing known-side context
/// wrapped in a fresh `jit_call` over the original known call inputs, and the residual side is emitted as the
/// residual `jit_call` over the surviving unknown call inputs plus the known-side call's residual-edge outputs.
impl<V, C> PartiallyEvaluatableOperation<C> for JitCallOperation
where
    V: Value<Type = ArrayType>,
    C: Context<Type = ArrayType, Operation = XlaOperation<V>>,
{
    fn partially_evaluate(
        &self,
        context: &PartialEvaluationContext<C>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Split only a mixed call with at least one known-but-symbolic input; everything else keeps the default
        // fold-or-residualize behavior and therefore the original boundary.
        if !context.any_known_is_symbolic(inputs) || inputs.iter().all(PartialEvaluationValue::is_known) {
            return context.fold_or_residualize(XlaOperation::JitCall(Box::new(self.clone())), inputs);
        }

        // Split the callee through the shared online boundary machinery, bind the known side into the enclosing
        // known-side context wrapped in a fresh `jit_call` over the original known call inputs, emit the residual
        // side as the residual `jit_call`, and reassemble the original output order.
        let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
        let partition = self.program.partition(input_known.as_slice())?;
        // A trivial partition — one whose known program contains no instructions — hoists no work (its known side
        // can only forward known inputs as residual edges), so keep the original boundary and let the default
        // materialize those knowns directly as residual feeders.
        if partition.known_program().instructions().is_empty() {
            return context.fold_or_residualize(XlaOperation::JitCall(Box::new(self.clone())), inputs);
        }
        context.inline_partitioned_program(
            partition,
            inputs,
            |known_program| XlaOperation::JitCall(Box::new(JitCallOperation::new(Rc::new(known_program)))),
            |residual_program| XlaOperation::JitCall(Box::new(JitCallOperation::new(Rc::new(residual_program)))),
        )
    }
}

impl JitCallOperation {
    /// Returns the call operation and output-axis metadata for batching this call.
    fn batched_call_operation<V: Value<Type = ArrayType>>(
        &self,
        inputs: &[ArrayBatch<V>],
    ) -> Result<(Self, Vec<Option<usize>>), ProgramError> {
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        match ArrayBatch::common_batch_size(inputs)? {
            Some(axis_size) => {
                let (batched_program, output_axes) =
                    build_batched_call_program(&self.program, batch_axes.as_slice(), axis_size)?;
                Ok((JitCallOperation::new(Rc::new(batched_program)), output_axes))
            }
            None => Ok((self.clone(), vec![None; self.program.output_types().len()])),
        }
    }
}

/// Batching rule for [`JitCallOperation`]: the callee program is rebatched over the mapped input axes (via
/// [`JitCallOperation::batched_call_operation`]) and the batched call is bound through `context.parent()`. An eager
/// client-backed parent (e.g., [`XlaDomain`](crate::XlaDomain)) compiles and executes the batched call immediately, a
/// staging parent stages it into the enclosing trace, and a differentiation parent dispatches it through its own
/// `jit_call` JVP rule — which is what serves `vmap` nested inside `gradient`/`linearize` closures.
impl<C> BatchableOperation<C> for JitCallOperation
where
    C: Context<Type = ArrayType, Operation: From<JitCallOperation>>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let (operation, output_axes) = self.batched_call_operation(inputs)?;
        let outputs = context.parent().bind(operation, &[], &[], &physical_inputs)?;
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
/// `[input_tangents..., residuals...] -> [output_tangents...]` together with the residual count. It then:
///
///   1. Wraps the primal sub-program in a fresh `jit_call` and stages it over the operand primals, recovering the
///      primal outputs followed by the residual values (program variables produced by the staged primal call).
///   2. Wraps the tangent sub-program in a fresh `jit_call` and stages it over the operand tangents followed by those
///      residual values, recovering one output tangent per primal output.
///   3. Pairs each primal output tracer with its tangent output tracer into a [`DifferentiationDual`].
///
/// The callee program is concretely keyed on [`XlaConstant`] (it is a [`FlatXlaProgram`]) regardless of the enclosing
/// value type `V`, so the split halves are themselves [`FlatXlaProgram`]s and the rule is valid for every `V`.
/// Preserving both `jit_call` boundaries keeps the callee body out of the caller's program, so forward mode over a
/// jitted call stays compiled rather than inlined.
impl<C, V> DifferentiableOperation<C> for JitCallOperation
where
    C: Context<Type = ArrayType, Constant = V, Operation = XlaOperation<V>> + Zero<C::Value>,
    V: Value<Type = ArrayType>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let output_count = self.program().output_types().len();
        check_count!("input", inputs, self.program().input_types().len(), ProgramError);

        // Linearize the callee capture-free. The primal sub-program produces `[outputs..., residuals...]` and the
        // tangent sub-program consumes `[input_tangents..., residuals...]`; the residual count is the number of
        // trailing outputs of the primal sub-program beyond the original callee outputs.
        let (primal_program, tangent_program, _) = self.program().linearize()?.into_parts();

        // Wrap the primal sub-program in a fresh `jit_call` and bind it over the operand primals, recovering the
        // primal outputs followed by the residual values.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_call = XlaOperation::JitCall(Box::new(JitCallOperation::new(Rc::new(primal_program))));
        let mut primal_call_outputs = context.bind(primal_call, &[], &[], &primal_operands)?;
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

        // Wrap the tangent sub-program in a fresh `jit_call` and bind it over the operand tangents followed by the
        // residual values, recovering one output tangent per primal output.
        // The tangent `jit_call` takes every operand tangent as a real program input, so materialize structural
        // zeros at this sub-program boundary.
        let mut tangent_operands = inputs
            .iter()
            .map(|input| input.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_operands.extend(residuals);
        let tangent_call = XlaOperation::JitCall(Box::new(JitCallOperation::new(Rc::new(tangent_program))));
        let tangent_outputs = context.bind(tangent_call, &[], &[], &tangent_operands)?;
        check_count!("output", tangent_outputs, output_count, ProgramError);

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect())
    }
}

/// Partition-aware transpose rule for a *primal* tangent [`JitCallOperation`], the jitted-call counterpart of
/// [`transpose_primal_condition`], [`transpose_primal_scan`], and [`transpose_primal_custom_vjp`]. It is used when the
/// direct reverse transposes a tangent program in the primal [`XlaOperation`] family rather than re-keying it
/// into a linear operation family.
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
///   2. Transposes the callee program with [`Program::transpose_with_respect_to`](ryft_core::Program::transpose_with_respect_to)
///      under the same per-operand linearity mask, so the callee's own linear and known inputs match the operands. The
///      transposed callee maps `[outputs..., known_input_values...]` to `[linear_input_cotangents...]`, in
///      callee-input order on each side.
///   3. Re-wraps the transposed callee in a fresh [`JitCallOperation`] and stages it over
///      `[outputs..., known_input_values...]`, preserving the compilation boundary so that both forward mode
///      over a jitted call (`jvp ∘ jit`) and reverse mode over it (`transpose ∘ jit`) stay compiled rather than
///      inlined.
///
/// The returned cotangents place the transposed call's outputs at the linear-operand positions and a structural
/// [`MaybeZero::Zero`] at the known positions, which carry no cotangent. The callee transposition happens through
/// [`Program::transpose_with_respect_to`](ryft_core::Program::transpose_with_respect_to) in the same operation family, so it is
/// value-level and introduces no recursive transposition obligation on [`XlaOperation`].
///
/// # Parameters
///
///   - `operation`: Primal tangent `jit_call` staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-operand [`PartialValue`] knowledge, mirroring the callee's inputs one-to-one. The
///     [`Unknown`](PartialValue::Unknown) entries are the input tangents; the [`Known`](PartialValue::Known) entries
///     carry the residual and captured-constant-tangent tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the tangent call's outputs.
pub fn transpose_primal_jit_call<V: Value<Type = ArrayType>>(
    operation: &JitCallOperation,
    context: &mut TracingContext<V, XlaOperation<V>>,
    inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, ProgramError> {
    // A jitted call with no live output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect());
    }

    // Each operand maps to one callee input, independently linear (an input tangent) or known (a residual value or a
    // captured-constant tangent). The linear operands need not lead: a captured compiled function threads its captured
    // prefix as known leading operands. The dispatch guarantees a `Known` operand carries its pullback value, so each
    // known tracer is read directly in callee-input order.
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let callee = operation.program();
    check_count!("input", operand_linear, callee.input_types().len(), ProgramError);
    let known_values = inputs
        .iter()
        .filter(|input| input.is_known())
        .map(|input| input.as_known().expect("dispatch guarantees a known operand carries its pullback value").clone())
        .collect::<Vec<_>>();

    // Transpose the callee with respect to its linear inputs. The transposed callee maps
    // `[outputs..., known_input_values...]` to `[linear_input_cotangents...]`, in callee-input order.
    let with_respect_to = operand_linear
        .iter()
        .enumerate()
        .filter_map(|(index, &linear)| linear.then_some(index))
        .collect::<Vec<_>>();
    let transposed_callee = callee.transpose_with_respect_to(with_respect_to.as_slice())?;

    // Stage the output cotangents, materializing a typed zero for each structurally zero cotangent, then stage a fresh
    // `jit_call` over the transposed callee on `[outputs..., known_input_values...]`. Its outputs are the
    // linear-input cotangents.
    let output_types = callee.output_types();
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(output_types.len() + known_values.len());
    for (cotangent, output_type) in outputs.iter().zip(output_types.iter()) {
        match cotangent {
            MaybeZero::Value(cotangent) => operands.push(cotangent.clone()),
            MaybeZero::Zero(_) => {
                let mut zeros = context.stage_nullary_operation(ZeroOperation::new(output_type.clone()))?;
                check_count!("output", zeros, 1, ProgramError);
                operands.push(zeros.remove(0));
            }
        }
    }
    operands.extend(known_values);
    let transposed_call = XlaOperation::JitCall(Box::new(JitCallOperation::new(Rc::new(transposed_callee))));
    let input_cotangents = context.stage_operation(transposed_call, operands.as_slice())?;
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
                if linear { input_cotangents.next().unwrap() } else { MaybeZero::Zero(input.r#type().into_owned()) }
            },
        )
        .collect();
    Ok(cotangents)
}

/// Transpose rule for a primal tangent [`JitCallOperation`], forwarding to [`transpose_primal_jit_call`]. The callee
/// transposition happens on the concretely [`XlaConstant`]-keyed [`FlatXlaProgram`], so the recursion is resolved once
/// at definition time and instantiating this implementation introduces no recursive [`TransposableOperation`]
/// obligation on [`XlaOperation`].
impl<V: Value<Type = ArrayType>> TransposableOperation<V, XlaOperation<V>> for JitCallOperation {
    fn transpose(
        &self,
        context: &mut TracingContext<V, XlaOperation<V>>,
        inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, DifferentiationError> {
        transpose_primal_jit_call(self, context, inputs, outputs).map_err(DifferentiationError::from)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use ryft_core::operations::math::{AddOperation, MulOperation};
    use ryft_core::parameters::Placeholder;
    use ryft_core::partial::PartialValue;
    use ryft_core::programs::ProgramBuilder;
    use ryft_core::types::{ArrayType, DataType, Shape, Size};

    use super::{JitCallOperation, XlaConstant, XlaOperation};

    fn vector_type() -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]))
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

        let r#type = vector_type();

        // Callee `f(a, x) = (a + c, x * c, (a + c) * x)` over a known `a`, an unknown `x`, and a literal `c`.
        let callee = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let known_input = builder.add_input(r#type.clone());
            let runtime_input = builder.add_input(r#type.clone());
            let literal = builder.add_constant(XlaConstant::new(0, r#type.clone()));
            let shifted = builder.add_instruction(AddOperation, vec![known_input, literal]).unwrap()[0];
            let scaled = builder.add_instruction(MulOperation, vec![runtime_input, literal]).unwrap()[0];
            let product = builder.add_instruction(MulOperation, vec![shifted, runtime_input]).unwrap()[0];
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
        let call = XlaOperation::JitCall(Box::new(JitCallOperation::new(Rc::new(callee))));
        let outputs = builder.add_instruction(call, vec![known_input, runtime_input]).unwrap().to_vec();
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
            let XlaOperation::JitCall(known_call) = outer_builder.instructions()[0].operation() else {
                panic!("expected the outer program to contain the known-side jit_call");
            };
            assert_eq!(known_call.program().input_ids().len(), 1);
            assert_eq!(known_call.program().output_ids().len(), 2);
            assert_eq!(known_call.program().instructions().len(), 1);
            assert!(matches!(known_call.program().instructions()[0].operation(), XlaOperation::Add(_)));
            assert!(known_call.program().atoms().iter().any(|atom| atom.is_constant()));
        }

        // The unknown half stayed behind one residual `jit_call` over the unknown input plus the residual edge, with
        // the literal rebuilt inline from its original payload.
        assert_eq!(evaluation.program().instructions().len(), 1);
        let XlaOperation::JitCall(residual_call) = evaluation.program().instructions()[0].operation() else {
            panic!("expected the residual program to contain the residual jit_call");
        };
        assert_eq!(residual_call.program().input_ids().len(), 2);
        assert_eq!(residual_call.program().instructions().len(), 2);
        assert!(residual_call.program().atoms().iter().any(|atom| atom.is_constant()));

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
        // variant projections and its `TransferToMemoryOperation` conversion. This is a compile-time capability
        // check: the assertions below fail to compile if any projection or conversion bound is unsatisfied.
        fn assert_policy<P: RematerializationPolicy<ArrayType, XlaOperation>>(_policy: P) {}
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
