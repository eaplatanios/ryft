use std::rc::Rc;

use ryft_macros::Operation;

use ryft_core::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use ryft_core::captures::CaptureReference;
use ryft_core::compilation::function::CompiledCallOperation;
use ryft_core::contexts::{Context, StagingContext};
use ryft_core::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationError, TransposableOperation,
    TranspositionDriver,
};
use ryft_core::macros::check_count;
use ryft_core::operations::BooleanLike;
use ryft_core::operations::compare::CompareOperation;
use ryft_core::operations::complex::{ComplexOperation, ConjugateOperation, ImaginaryOperation, RealOperation};
use ryft_core::operations::constants::{
    ConstantOperation, FillOperation, IotaOperation, OneLikeOperation, OneOperation, Zero, ZeroLikeOperation,
    ZeroOperation,
};
use ryft_core::operations::control_flow::{ConditionOperation, ScanOperation, SelectOperation, WhileOperation};
use ryft_core::operations::differentiation::{CoordinateBasisOperation, StopGradientOperation};
use ryft_core::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use ryft_core::operations::manipulation::{
    BroadcastOperation, ConcatenateOperation, ConvertElementTypeOperation, DynamicSliceOperation,
    DynamicUpdateSliceOperation, GatherOperation, PadOperation, Reshape, ReshapeOperation, ScatterOperation, Slice,
    SliceOperation, TransposeOperation, UpdateSlice, UpdateSliceOperation,
};
use ryft_core::operations::math::{
    AbsOperation, AddOperation, Atan2Operation, CosOperation, DivOperation, ExpOperation, LogOperation, MulOperation,
    NegOperation, SinOperation, SqrtOperation, SubOperation,
};
use ryft_core::operations::sharding::{ReshardOperation, ShardingConstraintOperation};
use ryft_core::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation,
};
use ryft_core::programs::operations::Operation;
use ryft_core::programs::regions::{CalleeRegionDriver, RegionInterface};
use ryft_core::programs::{MaybeZero, Program, ProgramBuilder, ProgramError, Value};
use ryft_core::tracing::{Tracer, TracingContext};

use ryft_core::backends::arrays::ArrayOperation;
use ryft_core::backends::scalars::Scalar;
use ryft_core::differentiation::DifferentiationDual;
use ryft_core::operations::debugging::PrintOperation;
use ryft_core::operations::tag::TagOperation;
use ryft_core::programs::types::{TypeError, Typed};
use ryft_core::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpOperation, CustomVjpTangentOperation,
};
use ryft_core::tracing_v2::operations::memory::TransferToMemoryOperation;
use ryft_core::tracing_v2::operations::reduce::ReduceOperation;
use ryft_core::tracing_v2::rematerialization::RematerializeOperation;
use ryft_core::tracing_v2::{AxisIndexOperation, CollectiveOperation, DotOperation};
use ryft_core::types::ArrayType;

use crate::experimental::operations::ShardMapOperation;

/// Lifetime-free reference to a concrete XLA value captured by a compiled program.
pub type XlaConstant = CaptureReference<ArrayType>;

/// Ordinary staged-operation universe owned by the XLA backend.
///
/// This enum flattens the core array operation payloads directly into the backend-owned operation family. Higher-order
/// instructions attach their nested computations as regions of the containing XLA program, so those regions can
/// contain backend-specific operations such as [`jit_call`](JitCallOperation) and
/// [`shard_map`](ShardMapOperation).
#[derive(Clone, Debug, Operation)]
#[ryft(crate = "ryft_core")]
#[ryft(dispatch(batching, differentiation, transposition))]
pub enum XlaOperation<V: Value<Type = ArrayType> = XlaConstant> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<V>),
    ConvertElementType(ConvertElementTypeOperation),
    Fill(FillOperation<ArrayType, Scalar>),
    Iota(IotaOperation<ArrayType>),
    CoordinateBasis(CoordinateBasisOperation<ArrayType>),
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
    /// Backend-owned condition whose attached branch regions can contain XLA operations.
    Condition(ConditionOperation<V>),

    /// Backend-owned loop whose attached condition and body regions can contain XLA operations.
    While(WhileOperation),

    /// Backend-owned scan whose attached body region can contain XLA operations.
    Scan(ScanOperation<V>),

    /// Backend-owned custom JVP call whose attached regions can contain XLA operations.
    CustomJvp(CustomJvpOperation),

    /// Backend-owned custom VJP call whose attached regions can contain XLA operations.
    CustomVjp(CustomVjpOperation),

    /// Backend-owned opaque custom-VJP tangent carrier, staged by the capture-free forward of a
    /// [`CustomVjp`](Self::CustomVjp) call. Its attached backward region can contain XLA operations.
    CustomVjpTangent(CustomVjpTangentOperation<ArrayType>),

    /// Backend-owned rematerialized call whose attached regions can contain XLA operations.
    Rematerialize(RematerializeOperation),

    /// Call to a flat jitted XLA sub-program.
    JitCall(JitCallOperation),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<V>>),
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
            ArrayOperation::ConvertElementType(operation) => Self::ConvertElementType(operation),
            ArrayOperation::Fill(operation) => Self::Fill(operation),
            ArrayOperation::Iota(operation) => Self::Iota(operation),
            ArrayOperation::CoordinateBasis(operation) => Self::CoordinateBasis(operation),
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
            ArrayOperation::Condition(operation) => XlaOperation::from(operation),
            ArrayOperation::While(operation) => Self::While(operation),
            ArrayOperation::Scan(operation) => Self::Scan(operation),
            ArrayOperation::CustomJvp(operation) => Self::CustomJvp(operation),
            ArrayOperation::CustomVjp(operation) => Self::CustomVjp(operation),
            ArrayOperation::CustomVjpTangent(operation) => Self::CustomVjpTangent(operation),
            ArrayOperation::Rematerialize(operation) => Self::Rematerialize(operation),
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

impl Operation<ArrayType> for JitCallOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "jit_call"
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        if region_interfaces.len() != 1 {
            return Err(TypeError {
                message: format!("jit_call expects 1 attached callee region but got {}", region_interfaces.len()),
            });
        }
        let callee_interface = &region_interfaces[0];
        ensure_call_input_types(self.name(), callee_interface.input_types(), input_types)?;
        Ok(callee_interface.output_types().to_vec())
    }

    #[inline]
    fn region_names(&self) -> &'static [&'static str] {
        &["callee"]
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
    V: PartialEq + Value<Type = ArrayType> + BooleanLike,
    C: Context<Type = ArrayType, Constant = V, Operation = XlaOperation<V>>,
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
impl<C> BatchableOperation<C> for JitCallOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<JitCallOperation>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
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
                let (batched_callee, output_axes) = driver.batch_program(
                    context,
                    driver.region(0)?,
                    input_batch_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::Natural,
                )?;
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
/// `[input_tangents..., residuals...] -> [output_tangents...]` together with the residual count. It then:
///
///   1. Wraps the primal sub-program in a fresh `jit_call` and stages it over the operand primals, recovering the
///      primal outputs followed by the residual values (program variables produced by the staged primal call).
///   2. Wraps the tangent sub-program in a fresh `jit_call` and stages it over the operand tangents followed by those
///      residual values, recovering one output tangent per primal output.
///   3. Pairs each primal output tracer with its tangent output tracer into a [`DifferentiationDual`].
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
    C: Context<Type = ArrayType, Constant = V, Operation = XlaOperation<V>> + Zero<C::Value>,
    V: PartialEq + Value<Type = ArrayType> + BooleanLike,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let callee = driver.region(0)?;
        let output_count = callee.output_types().len();
        check_count!("input", inputs, callee.input_types().len(), ProgramError);

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

        // Wrap the tangent sub-program in a fresh `jit_call` and bind it over the operand tangents followed by the
        // residual values, recovering one output tangent per primal output.
        // The tangent `jit_call` takes every operand tangent as a real program input, so materialize structural
        // zeros at this sub-program boundary.
        let mut tangent_operands = inputs
            .iter()
            .map(|input| input.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_operands.extend(residuals);
        let tangent_call = XlaOperation::JitCall(JitCallOperation::new());
        let tangent_outputs =
            context.bind(tangent_call, CalleeRegionDriver::new(&[Rc::new(tangent_program)]), &tangent_operands)?;
        check_count!("output", tangent_outputs, output_count, ProgramError);

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

/// Partition-aware transpose rule for a *primal* tangent [`JitCallOperation`], the jitted-call counterpart of
/// [`transpose_primal_condition`](ryft_core::operations::control_flow::transpose_primal_condition),
/// [`transpose_primal_scan`](ryft_core::operations::control_flow::transpose_primal_scan), and
/// [`transpose_primal_custom_vjp`](ryft_core::tracing_v2::transpose_primal_custom_vjp). It is used when the
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
pub fn transpose_primal_jit_call<V: Value<Type = ArrayType>, D: TranspositionDriver<V, XlaOperation<V>>>(
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
        match cotangent {
            MaybeZero::Value(cotangent) => operands.push(cotangent.clone()),
            MaybeZero::Zero(_) => {
                let mut zeros = context.stage_nullary_operation(ZeroOperation::new(output_type.cotangent()))?;
                check_count!("output", zeros, 1, ProgramError);
                operands.push(zeros.remove(0));
            }
        }
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
impl<V: Value<Type = ArrayType>> TransposableOperation<V, XlaOperation<V>> for JitCallOperation {
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

    use ryft_core::contexts::StagingContext;
    use ryft_core::differentiation::{DifferentiableType, DifferentiationError, TranspositionDriver};
    use ryft_core::operations::math::{AddOperation, MulOperation};
    use ryft_core::parameters::Placeholder;
    use ryft_core::partial::PartialValue;
    use ryft_core::programs::MaybeZero;
    use ryft_core::programs::ProgramBuilder;
    use ryft_core::programs::regions::{EmptyRegionDriver, RegionDriver, RegionRef};
    use ryft_core::programs::types::Typed;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use ryft_core::tracing::TracingContext;
    use ryft_core::types::{ArrayType, DataType, Shape, Size};

    use super::{
        JitCallOperation, XlaConstant, XlaOperation, XlaProgram, XlaProgramBuilder, transpose_primal_jit_call,
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
        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4)]))
    }

    #[test]
    fn test_jit_call_zero_transpose_uses_cotangent_descriptors() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let expected = tangent_type.cotangent();
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
    fn test_jit_call_mixed_output_transpose_materializes_zero_space_values() {
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4)]));
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(4)]));
        let source = {
            let mut builder = XlaProgramBuilder::new();
            let value = builder.add_input(value_type.clone());
            let predicate = builder.add_constant(XlaConstant::new(0, predicate_type.clone()));
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
            let value_cotangent = builder.add_input(value_type.clone());
            let _predicate_cotangent = builder.add_input(predicate_type.cotangent());
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
        let value_cotangent = context.input(value_type.clone());

        let contributions = transpose_primal_jit_call(
            &JitCallOperation::new(),
            &mut context,
            &driver,
            &[PartialValue::Unknown(value_type.clone())],
            &[MaybeZero::Value(value_cotangent), MaybeZero::Zero(predicate_type.cotangent())],
        )
        .unwrap();

        assert!(matches!(&contributions[..], [MaybeZero::Value(value)] if value.r#type().as_ref() == &value_type));
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
        let callee_region = builder.intern_callee(&Rc::new(callee));
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
            assert!(matches!(known_callee.instructions()[0].operation(), XlaOperation::Add(_)));
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
