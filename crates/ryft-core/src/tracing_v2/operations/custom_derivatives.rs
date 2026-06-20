use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::contexts::{EagerContext, StagingContext};
use crate::differentiation::{Cotangent, Tangent, TransposableOperation};
use crate::domains::Domain;
use crate::macros::{check_count, check_types};
use crate::operations::constants::{ZeroLike, ZeroOperation};
use crate::operations::manipulation::{Broadcast, Transpose};
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::{AbstractTracingContext, DomainTracer, Tracer, TracingContext};
use crate::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, BatchingContext, ProgramBatchableOperation, ProgramBatchingOutputAxes,
    align_batch_axis, broadcast_to_batched,
};
use crate::tracing_v2::differentiation::{
    FactorParameterizedOperation, JvpTracer, LinearOperationOf, ResidualizedOperation, TangentContext,
};
use crate::tracing_v2::operations::control_flow::stage_cotangent;
use crate::tracing_v2::{
    DifferentiableOperation, DifferentiationContext, ProgramLinearizableOperation, ResidualFactor,
};
use crate::types::{ArrayType, Type, TypeError, Typed};

/// Higher-order operation pairing a primal program with a user-supplied JVP program — the direct analogue of JAX's
/// [`custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html).
///
/// The JVP program follows JAX's calling convention: it receives the primal inputs followed by one tangent per
/// primal input, and returns the primal outputs followed by one tangent per primal output. The primal program is
/// kept separate from the JVP program so that un-differentiated calls do not pay for tangent computation:
/// interpretation and backend lowering replay the lean primal program; linearization replays the JVP
/// program instead of differentiating the primal body, so the user-supplied derivative governs both forward and
/// reverse mode (reverse mode transposes the linearization of the JVP program, which therefore must be linear in
/// its tangent arguments).
///
/// Traced batching (`batch`) re-wraps the call around batched primal/JVP programs — mirroring JAX's
/// `custom_jvp_call_jaxpr` batching rule — so the custom derivative survives a `batch` applied *before*
/// differentiation. Value-level batching (used by dense Jacobian materialization, where the custom rule has already
/// been consumed by linearization) inlines the primal program through the standard per-operation batching rules.
#[derive(Clone, Debug)]
pub struct CustomJvpOperation<V, O, T = ArrayType>
where
    T: Type,
    V: Value<T>,
{
    /// Program computing the primal outputs from the primal inputs.
    primal: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Program computing `(outputs..., output_tangents...)` from `(inputs..., input_tangents...)`.
    jvp: Program<T, V, O, Vec<V>, Vec<V>>,
}

impl<T: Type, V: Value<T>, O: Operation<T>> CustomJvpOperation<V, O, T> {
    /// Creates a custom-JVP operation after validating that the JVP program's signature matches the primal
    /// program's: its inputs must be the primal inputs followed by their tangents (same types), and its outputs the
    /// primal outputs followed by their tangents.
    pub fn new(
        primal: Program<T, V, O, Vec<V>, Vec<V>>,
        jvp: Program<T, V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        let input_types = primal.input_types();
        let output_types = primal.output_types();
        let expected_jvp_input_types: Vec<T> = input_types.iter().chain(input_types.iter()).cloned().collect();
        check_types!("custom_jvp rule input", &expected_jvp_input_types, &jvp.input_types());
        let expected_jvp_output_types: Vec<T> = output_types.iter().chain(output_types.iter()).cloned().collect();
        check_types!("custom_jvp rule output", &expected_jvp_output_types, &jvp.output_types());
        Ok(Self { primal, jvp })
    }

    /// Returns the primal program.
    #[inline]
    pub fn primal(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.primal
    }

    /// Returns the user-supplied JVP program.
    #[inline]
    pub fn jvp_program(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.jvp
    }

    /// Returns the primal input types.
    #[inline]
    pub fn input_types(&self) -> Vec<T> {
        self.primal.input_types()
    }

    /// Returns the primal output types.
    #[inline]
    pub fn output_types(&self) -> Vec<T> {
        self.primal.output_types()
    }
}

impl<T: Type, V: Value<T>, O> Display for CustomJvpOperation<V, O, T>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> Operation<T> for CustomJvpOperation<V, O, T> {
    #[inline]
    fn name(&self) -> &'static str {
        "custom_jvp"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_types!("custom_jvp input", &self.input_types(), input_types);
        Ok(self.output_types())
    }
}

impl<T, V, O> InterpretableOperation<T, V> for CustomJvpOperation<V, O, T>
where
    T: Type,
    V: Value<T>,
    O: InterpretableOperation<T, V> + Operation<T>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(
        &self,
        context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        self.primal.interpret_in_context(context, inputs.to_vec())
    }
}

/// Shared implementation of the [`CustomJvpOperation`] JVP rule, generic over the linearization context's value
/// type so the operation enum dispatchers ([`ArrayOperation`](crate::tracing_v2::ArrayOperation),
/// [`ScalarOperation`](crate::operations::scalars::ScalarOperation)) can invoke it for any
/// [`DifferentiationContext`] whose constants match the captured programs.
///
/// The rule evaluates the JVP program's primal at `(x̂, 0)` (its first half yields the rule's primal outputs) and
/// seeds its pushforward with `(0, t̂)` so that only the user-defined — and therefore necessarily linear — tangent
/// map survives in the staged linear program.
pub(crate) fn custom_jvp_rule<'jvp, D, O>(
    operation: &CustomJvpOperation<<D as Domain>::Constant, O, D::Type>,
    context: &mut TangentContext<'jvp, D>,
    inputs: &[JvpTracer<'jvp, D>],
) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
where
    D: DifferentiationContext<Type: PartialEq> + Domain<Operation = O> + 'jvp,
    <D as Domain>::Value: ZeroLike,
    O: Clone + DifferentiableOperation<D> + ProgramLinearizableOperation<D>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
    Vec<<D as Domain>::Constant>: Parameterized<
            <D as Domain>::Constant,
            Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<<D as Domain>::Value>,
            To<<D as Domain>::Value> = Vec<<D as Domain>::Value>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    let input_count = operation.input_types().len();
    let output_count = operation.output_types().len();
    check_count!("input", inputs, input_count, ProgramError);
    // Evaluate the JVP program's primal at `(x̂, 0)`; its first half yields the rule's primal outputs.
    let mut jvp_primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    for input in inputs.iter() {
        jvp_primal_inputs.push(input.primal().zero_like());
    }
    // Seed the pushforward with `(0, t̂)`: zero tangents for the primal slots and the incoming tangents for the
    // tangent slots, so only the user-defined (linear) tangent map survives.
    let mut tangent_seeds = Vec::with_capacity(2 * inputs.len());
    for input in inputs.iter() {
        tangent_seeds.push(context.materialize_tangent(Tangent::Zero(input.primal().r#type().into_owned()))?);
    }
    for input in inputs.iter() {
        tangent_seeds.push(context.materialize_tangent(input.tangent().clone())?);
    }
    let (jvp_primal_outputs, pushforward) =
        context.differentiable().linearize_program(operation.jvp_program(), jvp_primal_inputs)?;
    // Re-register the pushforward's residual values in the enclosing residual environment instead of baking them
    // into closed constant factors: under nested symbolic linearization the residuals are tracers, and only
    // environment references survive the factor rebasing onto the enclosing context's value type. Direct execution
    // closes the values into constants through `TangentContext::factor`, preserving the previous behavior.
    let residual_factors =
        pushforward.residuals().iter().map(|residual| context.factor(residual.clone())).collect::<Vec<_>>();
    let pushforward_program = pushforward.program().map_operations(|operation| {
        operation.try_map_factors(&mut |factor: &ResidualFactor<D::Type, <D as Domain>::Value>| match factor {
            ResidualFactor::Reference { index, .. } => residual_factors
                .get(*index)
                .cloned()
                .ok_or(ProgramError::UnboundAtomId { id: crate::programs::AtomId::new(*index) }),
            ResidualFactor::Constant(value) => Ok(ResidualFactor::Constant(value.clone())),
        })
    })?;
    let tangent_outputs = context.stage_program(&pushforward_program, tangent_seeds)?;
    Ok(jvp_primal_outputs
        .into_iter()
        .take(output_count)
        .zip(tangent_outputs.into_iter().skip(output_count))
        .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
        .collect())
}

/// Differentiates a `custom_jvp` call by running its forward-derivative rule; see [`custom_jvp_rule`].
impl<V, O, D> DifferentiableOperation<D> for CustomJvpOperation<V, O, D::Type>
where
    V: Value<D::Type>,
    D: DifferentiationContext<Type: PartialEq, Constant = V> + Domain<Operation = O>,
    <D as Domain>::Value: ZeroLike,
    O: Clone + DifferentiableOperation<D> + ProgramLinearizableOperation<D>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
    Vec<V>: Parameterized<
            V,
            Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<<D as Domain>::Value>,
            To<<D as Domain>::Value> = Vec<<D as Domain>::Value>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        custom_jvp_rule(self, context, inputs)
    }
}

/// Value-level batching for [`CustomJvpOperation`]: inlines the primal program through the per-operation batching
/// rules. The custom derivative does not survive this inlining; see the type-level documentation.
impl<V, O> BatchableOperation<V, EagerContext<ArrayType, V, O>> for CustomJvpOperation<V, O, ArrayType>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<V, EagerContext<ArrayType, V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<ArrayType, V, O>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        batch_program_inline(context, &self.primal, inputs)
    }
}

/// Stages a re-wrapped custom-derivative call into the batching context's parent trace.
///
/// This is the shared body of the traced `batch` rules for [`CustomJvpOperation`] and [`CustomVjpOperation`],
/// mirroring JAX's `custom_jvp_call_jaxpr` / `custom_vjp_call_jaxpr` batching rules: instead of inlining the primal
/// program (which would lose the custom derivative and any rematerialization structure), the rule stages one new
/// custom-derivative call whose captured programs have been batched. When no input carries the mapped lane axis the
/// original operation is staged unchanged and the outputs stay lane-uniform. Otherwise every input is aligned to
/// carry the lane at axis `0` (lane-uniform inputs are broadcast, matching the convention that every custom-call
/// input is mapped at axis `0`) and every output carries the lane at axis `0`.
fn stage_rewrapped_custom_call<C, MakeOperationFn>(
    context: &BatchingContext<C>,
    inputs: &[ArrayBatch<Tracer<C>>],
    make_operation: MakeOperationFn,
) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError>
where
    C: StagingContext<Type = ArrayType>,
    Tracer<C>: Broadcast + Transpose,
    MakeOperationFn: FnOnce(Option<usize>) -> Result<C::Operation, ProgramError>,
{
    if inputs.iter().all(|input| input.batch_axis().is_none()) {
        let operation = make_operation(None)?;
        let parent_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let outputs = context.parent_context().stage_operation(operation, parent_inputs.as_slice())?;
        return outputs
            .into_iter()
            .map(|tracer| {
                let physical_type = tracer.r#type().into_owned();
                ArrayBatch::new(physical_type, tracer, None)
            })
            .collect();
    }
    let axis_size = context.axis_size();
    let aligned_inputs = inputs
        .iter()
        .map(|input| match input.batch_axis() {
            Some(_) => align_batch_axis(input, 0),
            None => broadcast_to_batched(input, 0, axis_size),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let operation = make_operation(Some(axis_size))?;
    let parent_inputs = aligned_inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
    let outputs = context.parent_context().stage_operation(operation, parent_inputs.as_slice())?;
    outputs
        .into_iter()
        .map(|tracer| {
            let physical_type = tracer.r#type().into_owned();
            ArrayBatch::new(physical_type, tracer, Some(0))
        })
        .collect()
}

/// Batches `program` using the custom-derivative rewrapping convention: every input and output is mapped at axis `0`.
fn batch_rewrapped_program<V: Value<ArrayType>, O: ProgramBatchableOperation<V>>(
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    axis_size: usize,
) -> Result<Program<ArrayType, V, O, Vec<V>, Vec<V>>, ProgramError> {
    let input_batch_axes = vec![Some(0); program.input_types().len()];
    let (program, _) =
        program.batched(axis_size, input_batch_axes.as_slice(), ProgramBatchingOutputAxes::AlignAllTo(0))?;
    Ok(program)
}

/// Traced batching for [`CustomJvpOperation`]: re-wraps the call around batched primal/JVP programs so the custom
/// derivative survives `batch`; see [`stage_rewrapped_custom_call`].
impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for CustomJvpOperation<C::Constant, O, ArrayType>
where
    C: StagingContext<Type = ArrayType, Operation = O>,
    C::Constant: Value<ArrayType>,
    O: Clone
        + Operation<ArrayType>
        + From<CustomJvpOperation<C::Constant, O, ArrayType>>
        + ProgramBatchableOperation<C::Constant>,
    Tracer<C>: Broadcast + Transpose,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        stage_rewrapped_custom_call(context, inputs, |batched| match batched {
            None => Ok(O::from(self.clone())),
            Some(axis_size) => Ok(O::from(CustomJvpOperation::new(
                batch_rewrapped_program(&self.primal, axis_size)?,
                batch_rewrapped_program(&self.jvp, axis_size)?,
            )?)),
        })
    }
}

/// Replays `program` over packed batch values, dispatching every instruction through its value-level batching rule.
fn batch_program_inline<V, O>(
    context: &EagerContext<ArrayType, V, O>,
    program: &Program<ArrayType, V, O, Vec<V>, Vec<V>>,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<V, EagerContext<ArrayType, V, O>>,
{
    program.interpret_with(
        inputs.to_vec(),
        |_, constant: &V| Ok(ArrayBatch::unbatched(constant.clone())),
        |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
    )
}

/// Higher-order operation pairing a primal program with user-supplied forward/backward (VJP) programs — the direct
/// analogue of JAX's [`custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html).
///
/// The forward program maps the primal inputs to the primal outputs followed by arbitrarily many residual values;
/// the backward program maps those residuals followed by one cotangent per primal output to one cotangent per primal
/// input. The primal program is kept separate from the forward program so that un-differentiated calls do not pay
/// for residual computation: interpretation and backend lowering replay the lean primal program, and the
/// forward program runs only under reverse-mode differentiation. Linearization evaluates the
/// forward program, captures its residuals as factors, and stages one opaque linear call whose transpose replays the
/// backward program — so reverse mode uses exactly the user-supplied gradient. Forward-mode differentiation
/// (interpreting the staged linear call) is rejected, matching JAX's `custom_vjp` semantics.
///
/// Traced batching (`batch`) re-wraps the call around batched primal/forward/backward (and tangent) programs —
/// mirroring JAX's `custom_vjp_call_jaxpr` batching rule — so the custom derivative (and any rematerialization
/// structure) survives a `batch` applied *before* differentiation. Value-level batching (used by dense Jacobian
/// materialization, where the custom rule has already been consumed by linearization) inlines the primal program
/// through the standard per-operation batching rules.
#[derive(Clone, Debug)]
pub struct CustomVjpOperation<V, O, T = ArrayType>
where
    T: Type,
    V: Value<T>,
{
    /// Program computing the primal outputs from the primal inputs.
    primal: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Program computing `(outputs..., residuals...)` from the primal inputs.
    forward: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Program computing one input cotangent per primal input from `(residuals..., output_cotangents...)`.
    backward: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Optional program computing one output tangent per primal output from `(residuals..., input_tangents...)`.
    /// When absent — always the case for user-authored custom VJPs, matching JAX — forward-mode differentiation
    /// through the staged call is rejected. Rematerializeing attaches a derived tangent program so that `jvp` works
    /// through rematerialized regions.
    tangent: Option<Program<T, V, O, Vec<V>, Vec<V>>>,

    /// Backend lowering hint requesting an optimization barrier around the derived backward/tangent program outputs;
    /// see [`Self::with_prevent_cse`].
    prevent_cse: bool,
}

impl<T: Type, V: Value<T>, O: Operation<T>> CustomVjpOperation<V, O, T> {
    /// Creates a custom-VJP operation after validating the forward/backward program signatures against the primal
    /// program's: `forward` must consume the primal inputs and produce the primal outputs followed by the residuals,
    /// and `backward` must consume those residuals followed by one cotangent per primal output and produce one
    /// cotangent per primal input.
    pub fn new(
        primal: Program<T, V, O, Vec<V>, Vec<V>>,
        forward: Program<T, V, O, Vec<V>, Vec<V>>,
        backward: Program<T, V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        let input_types = primal.input_types();
        let output_types = primal.output_types();
        check_types!("custom_vjp forward input", &input_types, &forward.input_types());
        let forward_output_types = forward.output_types();
        if forward_output_types.len() < output_types.len() {
            return Err(TypeError {
                message: format!(
                    "custom_vjp forward must produce at least the {} primal output(s) but produced {} value(s)",
                    output_types.len(),
                    forward_output_types.len(),
                ),
            });
        }
        check_types!("custom_vjp forward output", &output_types, &forward_output_types[..output_types.len()]);
        let residual_types = &forward_output_types[output_types.len()..];
        let expected_backward_input_types: Vec<T> = residual_types.iter().chain(output_types.iter()).cloned().collect();
        check_types!("custom_vjp backward input", &expected_backward_input_types, &backward.input_types(),);
        check_types!("custom_vjp backward output", &input_types, &backward.output_types());
        Ok(Self { primal, forward, backward, tangent: None, prevent_cse: false })
    }

    /// Sets whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier
    /// (e.g., StableHLO's `optimization_barrier`). Without a barrier, a compiler may common-subexpression-eliminate
    /// values recomputed by the backward program against the forward pass, silently restoring the memory cost the
    /// recomputation was meant to avoid. [`Rematerialize`](crate::tracing_v2::rematerialization::Rematerialize)
    /// enables this by default — mirroring `jax.checkpoint`'s `prevent_cse=True` — while user-authored custom VJPs
    /// leave it disabled because their backward programs do not recompute forward values.
    ///
    /// The hint applies where the staged call boundary survives to backend lowering (directly lowered pullback and
    /// tangent programs). When a transform inlines the backward program into an outer trace at staging time, the
    /// boundary dissolves and no barrier is emitted.
    pub fn with_prevent_cse(mut self, prevent_cse: bool) -> Self {
        self.prevent_cse = prevent_cse;
        self
    }

    /// Returns whether backends should wrap the lowered backward/tangent program outputs in an optimization
    /// barrier; see [`Self::with_prevent_cse`].
    #[inline]
    pub fn prevent_cse(&self) -> bool {
        self.prevent_cse
    }

    /// Attaches a tangent program mapping `(residuals..., input_tangents...)` to one tangent per primal output,
    /// enabling forward-mode differentiation through the staged call. This is used by
    /// [`Rematerialize`](crate::tracing_v2::rematerialization::Rematerialize), which derives the tangent program from the body;
    /// user-authored custom VJPs leave it absent and reject forward mode, matching JAX.
    pub fn with_tangent_program(mut self, tangent: Program<T, V, O, Vec<V>, Vec<V>>) -> Result<Self, TypeError> {
        let input_types = self.primal.input_types();
        let output_types = self.primal.output_types();
        let residual_types = self.forward.output_types().split_off(output_types.len());
        let expected_input_types: Vec<T> = residual_types.iter().chain(input_types.iter()).cloned().collect();
        check_types!("custom_vjp tangent input", &expected_input_types, &tangent.input_types());
        check_types!("custom_vjp tangent output", &output_types, &tangent.output_types());
        self.tangent = Some(tangent);
        Ok(self)
    }

    /// Returns the optional tangent program enabling forward-mode differentiation through the staged call.
    #[inline]
    pub fn tangent_program(&self) -> Option<&Program<T, V, O, Vec<V>, Vec<V>>> {
        self.tangent.as_ref()
    }

    /// Returns the primal program.
    #[inline]
    pub fn primal(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.primal
    }

    /// Returns the forward (residual-producing) program.
    #[inline]
    pub fn forward(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.forward
    }

    /// Returns the backward (cotangent-producing) program.
    #[inline]
    pub fn backward(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.backward
    }

    /// Returns the primal input types.
    #[inline]
    pub fn input_types(&self) -> Vec<T> {
        self.primal.input_types()
    }

    /// Returns the primal output types.
    #[inline]
    pub fn output_types(&self) -> Vec<T> {
        self.primal.output_types()
    }
}

impl<T: Type, V: Value<T>, O> Display for CustomVjpOperation<V, O, T>
where
    Self: Operation<T>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> Operation<T> for CustomVjpOperation<V, O, T> {
    #[inline]
    fn name(&self) -> &'static str {
        "custom_vjp"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_types!("custom_vjp input", &self.input_types(), input_types);
        Ok(self.output_types())
    }
}

impl<T, V, O> InterpretableOperation<T, V> for CustomVjpOperation<V, O, T>
where
    T: Type,
    V: Value<T>,
    O: InterpretableOperation<T, V> + Operation<T>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(
        &self,
        context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        self.primal.interpret_in_context(context, inputs.to_vec())
    }
}

/// Shared implementation of the [`CustomVjpOperation`] JVP rule, generic over the linearization context's value type
/// so the operation enum dispatchers can invoke it for any [`DifferentiationContext`] whose constants match the
/// captured programs.
///
/// The rule linearizes the forward program at the primal inputs — discarding the resulting pushforward, so the
/// forward body is never differentiated beyond what its primal evaluation requires — captures the trailing residual
/// outputs as factors, and stages one opaque [`CustomVjpCallOperation`] mapping the input tangents to the output
/// tangents. The staged call rejects forward-mode interpretation; its transpose replays the user's backward program.
pub(crate) fn custom_vjp_rule<'jvp, D, O>(
    operation: &CustomVjpOperation<<D as Domain>::Constant, O, D::Type>,
    context: &mut TangentContext<'jvp, D>,
    inputs: &[JvpTracer<'jvp, D>],
) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
where
    D: DifferentiationContext<Type: PartialEq> + Domain<Operation = O> + 'jvp,
    O: Clone + DifferentiableOperation<D> + ProgramLinearizableOperation<D>,
    LinearOperationOf<D>: ResidualizedOperation<D>
        + From<CustomVjpCallOperation<<D as Domain>::Constant, O, ResidualFactor<D::Type, <D as Domain>::Value>, D::Type>>,
    Vec<<D as Domain>::Constant>: Parameterized<
            <D as Domain>::Constant,
            Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<<D as Domain>::Value>,
            To<<D as Domain>::Value> = Vec<<D as Domain>::Value>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    let output_count = operation.output_types().len();
    check_count!("input", inputs, operation.input_types().len(), ProgramError);
    let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    let tangent_operands = inputs
        .iter()
        .map(|input| context.materialize_tangent(input.tangent().clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let (mut forward_values, _pushforward) =
        context.differentiable().linearize_program(&operation.forward, primal_operands)?;
    let residuals = forward_values.split_off(output_count);
    let factors = residuals.into_iter().map(|residual| context.factor(residual)).collect::<Vec<_>>();
    let call = LinearOperationOf::<D>::from(CustomVjpCallOperation::new(
        operation.backward.clone(),
        operation.tangent.clone(),
        factors,
        false,
        operation.prevent_cse,
    ));
    let tangent_outputs = context.stage_operation(call, tangent_operands.as_slice())?;
    check_count!("output", tangent_outputs, output_count, ProgramError);
    Ok(forward_values
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
        .collect())
}

/// Differentiates a `custom_vjp` call by running its forward-derivative rule; see [`custom_vjp_rule`].
impl<V, O, D> DifferentiableOperation<D> for CustomVjpOperation<V, O, D::Type>
where
    V: Value<D::Type>,
    D: DifferentiationContext<Type: PartialEq, Constant = V> + Domain<Operation = O>,
    O: Clone + DifferentiableOperation<D> + ProgramLinearizableOperation<D>,
    LinearOperationOf<D>: ResidualizedOperation<D>
        + From<CustomVjpCallOperation<V, O, ResidualFactor<D::Type, <D as Domain>::Value>, D::Type>>,
    Vec<V>: Parameterized<
            V,
            Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<<D as Domain>::Value>,
            To<<D as Domain>::Value> = Vec<<D as Domain>::Value>,
            To<D::Tangent> = Vec<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        custom_vjp_rule(self, context, inputs)
    }
}

/// Value-level batching for [`CustomVjpOperation`]: inlines the primal program; see [`CustomJvpOperation`]'s
/// batching documentation for the custom-derivative caveat.
impl<V, O> BatchableOperation<V, EagerContext<ArrayType, V, O>> for CustomVjpOperation<V, O, ArrayType>
where
    V: Value<ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<V, EagerContext<ArrayType, V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<ArrayType, V, O>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        batch_program_inline(context, &self.primal, inputs)
    }
}

/// Traced batching for [`CustomVjpOperation`]: re-wraps the call around batched primal/forward/backward (and
/// tangent, when present) programs so the custom derivative — and any rematerialization structure — survives
/// `batch`; see [`stage_rewrapped_custom_call`].
impl<C, O> BatchableOperation<Tracer<C>, BatchingContext<C>> for CustomVjpOperation<C::Constant, O, ArrayType>
where
    C: StagingContext<Type = ArrayType, Operation = O>,
    C::Constant: Value<ArrayType>,
    O: Clone
        + Operation<ArrayType>
        + From<CustomVjpOperation<C::Constant, O, ArrayType>>
        + ProgramBatchableOperation<C::Constant>,
    Tracer<C>: Broadcast + Transpose,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        stage_rewrapped_custom_call(context, inputs, |batched| match batched {
            None => Ok(O::from(self.clone())),
            Some(axis_size) => {
                let mut operation = CustomVjpOperation::new(
                    batch_rewrapped_program(&self.primal, axis_size)?,
                    batch_rewrapped_program(&self.forward, axis_size)?,
                    batch_rewrapped_program(&self.backward, axis_size)?,
                )?
                .with_prevent_cse(self.prevent_cse);
                if let Some(tangent) = &self.tangent {
                    operation = operation.with_tangent_program(batch_rewrapped_program(tangent, axis_size)?)?;
                }
                Ok(O::from(operation))
            }
        })
    }
}

/// Access to a custom-VJP residual payload as a concrete value during pullback interpretation.
///
/// Implemented by plain values (identity) and by [`ResidualFactor`] (whose `Constant` form yields its payload and
/// whose `Reference` form errors, since references are only meaningful before residual instantiation).
#[doc(hidden)]
pub trait CustomVjpResidual<T: Type, V: Value<T>>: Value<T> {
    /// Returns the concrete residual value.
    fn residual_value(&self) -> Result<V, ProgramError>;
}

impl<T: Type, V: Value<T>> CustomVjpResidual<T, V> for V {
    #[inline]
    fn residual_value(&self) -> Result<V, ProgramError> {
        Ok(self.clone())
    }
}

impl<T: Type, V: Value<T>> CustomVjpResidual<T, V> for ResidualFactor<T, V> {
    fn residual_value(&self) -> Result<V, ProgramError> {
        match self {
            ResidualFactor::Constant(value) => Ok(value.clone()),
            ResidualFactor::Reference { .. } => Err(TypeError {
                message: "custom_vjp pullback interpretation requires instantiated residuals".to_string(),
            }
            .into()),
        }
    }
}

/// Opaque linear operation staged by [`CustomVjpOperation`]'s JVP rule.
///
/// In its un-transposed form it stands for the (unknown) tangent map of the custom function and rejects
/// interpretation: `custom_vjp` functions are reverse-mode-only, matching JAX. Transposition replaces it with its
/// transposed form, whose interpretation replays the user's backward program on the captured residuals and the
/// incoming output cotangents.
#[derive(Clone, Debug)]
pub struct CustomVjpCallOperation<V, O, F, T = ArrayType>
where
    T: Type,
    V: Value<T>,
    F: Value<T>,
{
    /// The user's backward program, mapping `(residuals..., output_cotangents...)` to input cotangents.
    backward: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Optional tangent program mapping `(residuals..., input_tangents...)` to output tangents; see
    /// [`CustomVjpOperation::with_tangent_program`].
    tangent: Option<Program<T, V, O, Vec<V>, Vec<V>>>,

    /// Captured residual factors consumed by the backward program.
    residuals: Vec<F>,

    /// Whether this call has been transposed into its executable (pullback) form.
    transposed: bool,

    /// Backend lowering hint requesting an optimization barrier around the lowered program outputs; see
    /// [`CustomVjpOperation::with_prevent_cse`].
    prevent_cse: bool,
}

impl<T: Type, V: Value<T>, F: Value<T>, O> CustomVjpCallOperation<V, O, F, T> {
    /// Creates a custom-VJP call. Use `transposed = false` for the opaque pushforward form and `transposed = true`
    /// for the executable pullback form. `prevent_cse` carries the owning [`CustomVjpOperation`]'s lowering hint;
    /// see [`CustomVjpOperation::with_prevent_cse`].
    pub fn new(
        backward: Program<T, V, O, Vec<V>, Vec<V>>,
        tangent: Option<Program<T, V, O, Vec<V>, Vec<V>>>,
        residuals: Vec<F>,
        transposed: bool,
        prevent_cse: bool,
    ) -> Self {
        Self { backward, tangent, residuals, transposed, prevent_cse }
    }

    /// Returns the user's backward program.
    #[inline]
    pub fn backward(&self) -> &Program<T, V, O, Vec<V>, Vec<V>> {
        &self.backward
    }

    /// Returns the optional tangent program enabling forward-mode interpretation of the un-transposed call; see
    /// [`CustomVjpOperation::with_tangent_program`].
    #[inline]
    pub fn tangent_program(&self) -> Option<&Program<T, V, O, Vec<V>, Vec<V>>> {
        self.tangent.as_ref()
    }

    /// Returns the captured residual factors.
    #[inline]
    pub fn residuals(&self) -> &[F] {
        self.residuals.as_slice()
    }

    /// Returns whether this call is in its transposed (executable pullback) form.
    #[inline]
    pub fn transposed(&self) -> bool {
        self.transposed
    }

    /// Returns whether backends should wrap this call's lowered program outputs in an optimization barrier; see
    /// [`CustomVjpOperation::with_prevent_cse`].
    #[inline]
    pub fn prevent_cse(&self) -> bool {
        self.prevent_cse
    }

    /// Maps the residual factor payloads with `map_factor`, preserving the backward program and direction.
    pub fn map_factors<MappedFactor: Value<T>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<CustomVjpCallOperation<V, O, MappedFactor, T>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
        V: Clone,
        O: Clone,
    {
        Ok(CustomVjpCallOperation {
            backward: self.backward.clone(),
            tangent: self.tangent.clone(),
            residuals: self.residuals.iter().map(map_factor).collect::<Result<Vec<_>, _>>()?,
            transposed: self.transposed,
            prevent_cse: self.prevent_cse,
        })
    }
}

impl<T: Type, V: Value<T>, F: Value<T>, O: Operation<T>> CustomVjpCallOperation<V, O, F, T> {
    /// Returns the cotangent types flowing *into* the backward program (one per primal output).
    fn cotangent_types(&self) -> Vec<T> {
        self.backward.input_types().split_off(self.residuals.len())
    }
}

impl<T: Type, V: Value<T>, F: Value<T>, O> Display for CustomVjpCallOperation<V, O, F, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.transposed {
            formatter.write_str("custom_vjp_backward")
        } else {
            formatter.write_str("custom_vjp_tangent")
        }
    }
}

impl<T: Type, V: Value<T>, F: Value<T>, O: Operation<T>> Operation<T> for CustomVjpCallOperation<V, O, F, T> {
    #[inline]
    fn name(&self) -> &'static str {
        if self.transposed { "custom_vjp_backward" } else { "custom_vjp_tangent" }
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        if self.transposed {
            check_types!("custom_vjp backward cotangent", &self.cotangent_types(), input_types);
            Ok(self.backward.output_types())
        } else {
            // The un-transposed call maps input tangents (typed like the primal inputs, which are the backward
            // program's outputs) to output tangents (typed like the primal outputs, which are the backward
            // program's trailing inputs).
            check_types!("custom_vjp tangent", &self.backward.output_types(), input_types);
            Ok(self.cotangent_types())
        }
    }
}

impl<T, V, O, F> InterpretableOperation<T, V> for CustomVjpCallOperation<V, O, F, T>
where
    T: Type,
    V: Value<T>,
    F: CustomVjpResidual<T, V>,
    O: InterpretableOperation<T, V> + Operation<T>,
    Vec<V>: Parameterized<V, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(
        &self,
        context: &<V as Value<T>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        let program = if self.transposed {
            &self.backward
        } else if let Some(tangent) = &self.tangent {
            tangent
        } else {
            return Err(TypeError {
                message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                    value_and_grad, or jacrev) instead"
                    .to_string(),
            }
            .into());
        };
        let mut values = self.residuals.iter().map(CustomVjpResidual::residual_value).collect::<Result<Vec<_>, _>>()?;
        values.extend(inputs.iter().cloned());
        program.interpret_in_context(context, values)
    }
}

impl<T, V, O, S> CustomVjpCallOperation<V, O, Tracer<S>, T>
where
    T: Type,
    V: Value<T>,
    O: Clone + Operation<T>,
    S: StagingContext<Type = T, Constant = V, Operation = O>,
{
    /// Replays this call's captured program over tracer values, staging the replayed instructions into the tracers'
    /// context. The transposed form replays the backward program on `(residuals..., output_cotangents...)` and the
    /// un-transposed form replays the tangent program on `(residuals..., input_tangents...)`, rejecting forward mode
    /// when no tangent program is present — exactly mirroring concrete-value interpretation.
    ///
    /// This is what makes custom-VJP calls (and therefore rematerialized regions) nest inside outer traces: when an
    /// outer transform derives a program symbolically, the captured residuals are already instantiated to tracers of
    /// the outer trace, so the captured program — written over context constants and the context's own operation
    /// type — can be inlined into the trace with [`StagingContext::stage_program`], routing every replayed
    /// instruction through the active transform's rules.
    pub(crate) fn interpret_over_tracers(&self, inputs: &[Tracer<S>]) -> Result<Vec<Tracer<S>>, ProgramError> {
        let program = if self.transposed {
            &self.backward
        } else if let Some(tangent) = &self.tangent {
            tangent
        } else {
            return Err(TypeError {
                message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                    value_and_grad, or jacrev) instead"
                    .to_string(),
            }
            .into());
        };
        let context = self
            .residuals
            .first()
            .or_else(|| inputs.first())
            .map(|tracer| tracer.context().clone())
            .ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?;
        let mut values = self.residuals.to_vec();
        values.extend(inputs.iter().cloned());
        context.stage_program(program, values)
    }
}

/// Transpose rule for [`CustomVjpCallOperation`]: stages the flipped-direction form of the call on the output
/// cotangents, materializing structural zeros so the staged call receives every cotangent input. The rule is
/// generic over the cotangent value type `W`, which need not match the backward program's value type `V`: the staged
/// flipped call carries the programs and residuals along unchanged.
///
/// The un-transposed (tangent-map) call transposes into the executable pullback. The pullback transposes back into
/// the tangent-map call — a linear map's pullback is linear and its transpose is the original map — but executing
/// that map requires the derived tangent program, so the second transposition is only supported for
/// rematerialization-derived calls; user-authored custom VJPs (with no tangent program) keep rejecting it, matching
/// JAX's behavior for second-order reverse mode through `custom_vjp`.
impl<T, V, O, F, W, OLinear> TransposableOperation<T, W, OLinear> for CustomVjpCallOperation<V, O, F, T>
where
    T: Type,
    V: Value<T>,
    F: Value<T>,
    W: Value<T>,
    O: Clone + Operation<T>,
    OLinear: Operation<T> + From<ZeroOperation<T>> + From<CustomVjpCallOperation<V, O, F, T>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, T, W, OLinear>,
        _input_types: &[&T],
        output_cotangents: &[Cotangent<'transpose, T, W, OLinear>],
    ) -> Result<Vec<Cotangent<'transpose, T, W, OLinear>>, ProgramError> {
        if self.transposed && self.tangent.is_none() {
            return Err(TypeError {
                message: "transposing a custom_vjp pullback (second-order reverse mode through custom_vjp) is not \
                    supported"
                    .to_string(),
            }
            .into());
        }
        // The staged call's outputs are primal-output-typed for the tangent-map form and primal-input-typed for
        // the pullback form, so the incoming cotangents are typed accordingly.
        let cotangent_types = if self.transposed { self.backward.output_types() } else { self.cotangent_types() };
        check_count!("output", output_cotangents, cotangent_types.len(), ProgramError);
        let cotangent_tracers = output_cotangents
            .iter()
            .zip(cotangent_types.iter())
            .map(|(cotangent, r#type)| stage_cotangent(context, cotangent, r#type))
            .collect::<Vec<_>>();
        let call = OLinear::from(CustomVjpCallOperation::new(
            self.backward.clone(),
            self.tangent.clone(),
            self.residuals.to_vec(),
            !self.transposed,
            self.prevent_cse,
        ));
        let outputs = context.stage_operation(call, cotangent_tracers.as_slice())?;
        Ok(outputs.into_iter().map(Cotangent::Staged).collect())
    }
}

/// Value-level batching for the transposed [`CustomVjpCallOperation`]: replays the backward program through the
/// per-operation batching rules with the captured residuals as lane-uniform values. Used when a pullback containing
/// custom-VJP calls is interpreted with batched cotangents (e.g., by `jacrev`). The un-transposed form rejects
/// batching just as it rejects interpretation.
impl<V, O, F> BatchableOperation<V, EagerContext<ArrayType, V, O>> for CustomVjpCallOperation<V, O, F, ArrayType>
where
    V: Value<ArrayType>,
    F: CustomVjpResidual<ArrayType, V>,
    O: Operation<ArrayType> + BatchableOperation<V, EagerContext<ArrayType, V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<ArrayType, V, O>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let program = if self.transposed {
            &self.backward
        } else if let Some(tangent) = &self.tangent {
            tangent
        } else {
            return Err(TypeError {
                message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                    value_and_grad, or jacrev) instead"
                    .to_string(),
            }
            .into());
        };
        let mut values = self
            .residuals
            .iter()
            .map(|residual| Ok(ArrayBatch::unbatched(residual.residual_value()?)))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        values.extend(inputs.iter().cloned());
        program.interpret_with(
            values,
            |_, constant: &V| Ok(ArrayBatch::unbatched(constant.clone())),
            |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
        )
    }
}

/// Function with a user-supplied JVP rule — the ergonomic analogue of JAX's
/// [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html) /
/// [`defjvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.defjvp.html) decorator pair, built by
/// [`custom_jvp`].
///
/// The primal function and its JVP rule are stored as plain closures over [`Parameterized`] trees of
/// [`DomainTracer`]s — `ryft`'s analogue of JAX pytrees — so inputs and outputs can be single tracers, tuples, or
/// any other parameterized structure. Nothing is traced at construction time: each [`call`](Self::call) reads the
/// input types off its tracer arguments, traces both closures into programs specialized to those types, validates
/// the rule signature, and stages one [`CustomJvpOperation`] into the caller's staging context — mirroring how JAX
/// traces rule functions into jaxprs lazily at transform time. `primal` maps the input tree to the output tree, and
/// `jvp` maps `(inputs, input_tangents)` to `(outputs, output_tangents)`, exactly like a JAX `defjvp` rule.
///
/// The primal closure is kept separate from the JVP closure for efficiency rather than necessity: the JVP rule
/// computes both the outputs and their tangents, so deriving the primal from it would make every un-differentiated
/// call pay for tangent computation. Interpretation, batching, and backend lowering replay the lean primal program;
/// the JVP program runs only under differentiation.
///
/// There is no analogue of JAX's `nondiff_argnums` because closure capture subsumes it: static, non-differentiated
/// configuration is simply captured by the closures (all of them can see it), exactly like JAX threads
/// non-differentiated arguments through to the rule functions.
pub struct CustomJvp<'d, D: Domain, P, J, IT, OT> {
    /// Domain whose constant and operation types the captured programs are traced over.
    domain: &'d D,

    /// Closure computing the primal output tree from the primal input tree.
    primal: P,

    /// Closure computing `(outputs, output_tangents)` from `(inputs, input_tangents)`.
    jvp: J,

    /// Phantom marker pinning the input and output tracer-tree types named by the closure signatures.
    marker: PhantomData<fn() -> (IT, OT)>,
}

/// Creates a [`CustomJvp`] function from a primal closure and a JVP-rule closure over trees of `domain`'s tracers.
/// Refer to the documentation of [`CustomJvp`] for the calling convention and tracing semantics.
pub fn custom_jvp<'d, D, P, J, IT, OT>(domain: &'d D, primal: P, jvp: J) -> CustomJvp<'d, D, P, J, IT, OT>
where
    D: Domain,
    P: Fn(IT) -> Result<OT, ProgramError>,
    J: Fn(IT, IT) -> Result<(OT, OT), ProgramError>,
{
    CustomJvp { domain, primal, jvp, marker: PhantomData }
}

impl<'d, D, P, J, IT, OT> CustomJvp<'d, D, P, J, IT, OT>
where
    D: Domain<Type: PartialEq> + 'd,
    P: Fn(IT) -> Result<OT, ProgramError>,
    J: Fn(IT, IT) -> Result<(OT, OT), ProgramError>,
    D::Operation: Clone + Operation<D::Type> + From<CustomJvpOperation<D::Constant, D::Operation, D::Type>>,
    IT: Parameterized<DomainTracer<'d, D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    OT: Parameterized<DomainTracer<'d, D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    IT::To<D::Type>: Clone
        + Parameterized<D::Type, Family = IT::Family, To<DomainTracer<'d, D>> = IT, To<D::Constant> = IT::To<D::Constant>>,
    OT::To<D::Type>: Parameterized<D::Type, Family = OT::Family, To<DomainTracer<'d, D>> = OT, To<D::Constant> = OT::To<D::Constant>>,
{
    /// Stages this custom-JVP function on the provided tracer input tree and returns its output tree, tracing the
    /// stored closures into programs specialized to the input types. Differentiation of the staged call replays the
    /// JVP rule instead of differentiating the primal body, in both forward and reverse mode.
    pub fn call<C, ICT>(
        &self,
        input: ICT,
    ) -> Result<<OT::To<D::Type> as Parameterized<D::Type>>::To<Tracer<C>>, ProgramError>
    where
        C: StagingContext<Type = D::Type, Constant = D::Constant, Operation = D::Operation>,
        IT::Family: ParameterizedFamily<Tracer<C>>,
        OT::Family: ParameterizedFamily<Tracer<C>>,
        ICT: Parameterized<Tracer<C>, Family = IT::Family, To<D::Type> = IT::To<D::Type>>,
        <OT::To<D::Type> as Parameterized<D::Type>>::To<Tracer<C>>: Parameterized<
                Tracer<C>,
                Family = OT::Family,
                ParameterStructure = <OT::To<D::Type> as Parameterized<D::Type>>::ParameterStructure,
            >,
    {
        let mut input_tracers = Vec::new();
        let input_types = input
            .map_parameters(|tracer| {
                let r#type = tracer.r#type().into_owned();
                input_tracers.push(tracer);
                r#type
            })
            .map_err(ProgramError::from)?;
        let Some(first) = input_tracers.first() else {
            return Err(TypeError { message: "custom_jvp requires at least one input".to_string() }.into());
        };
        let (_, primal) = TracingContext::trace(self.domain, |xs| (self.primal)(xs), input_types.clone())?;
        let (output_types, jvp) =
            TracingContext::trace(self.domain, |(x, t)| (self.jvp)(x, t), (input_types.clone(), input_types))?;
        let operation = D::Operation::from(CustomJvpOperation::new(primal.to_flat_program(), jvp.to_flat_program())?);
        let outputs = first.context().stage_operation(operation, &input_tracers)?;
        let output_structure = output_types.0.parameter_structure();
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

/// Function with user-supplied forward/backward (VJP) rules — the ergonomic analogue of JAX's
/// [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) /
/// [`defvjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html) decorator pair, built by
/// [`custom_vjp`].
///
/// The primal function and its forward/backward rules are stored as plain closures over [`Parameterized`] trees of
/// [`DomainTracer`]s — `ryft`'s analogue of JAX pytrees — so inputs, outputs, and residuals can be single tracers,
/// tuples, or any other parameterized structure. Nothing is traced at construction time: each [`call`](Self::call)
/// reads the input types off its tracer arguments, traces the closures into programs specialized to those types,
/// validates the rule signatures, and stages one [`CustomVjpOperation`] into the caller's staging context —
/// mirroring how JAX traces rule functions into jaxprs lazily at transform time. `primal` maps the input tree to
/// the output tree, `forward` maps the input tree to `(outputs, residuals)` (the same structural split as a JAX
/// `f_fwd`), and `backward` maps `(residuals, output_cotangents)` to the input cotangent tree. As in JAX, the
/// resulting function supports reverse mode only; forward-mode differentiation of a staged call is rejected.
///
/// The primal closure is kept separate from the forward closure for efficiency rather than necessity: an
/// un-differentiated call should not pay for residual computation. Interpretation, batching, and backend lowering
/// replay the lean primal program; the residual-producing forward program runs only under reverse-mode
/// differentiation. Callers that do not care about the distinction can pass the same body for both — accepting that
/// the residual outputs are dead code outside of differentiation — which mirrors the common JAX idiom of writing
/// `f_fwd` as `return f(x), residuals`.
///
/// There is no analogue of JAX's `nondiff_argnums` because closure capture subsumes it: static, non-differentiated
/// configuration is simply captured by the closures (all of them can see it), exactly like JAX threads
/// non-differentiated arguments through to the rule functions.
pub struct CustomVjp<'d, D: Domain, P, F, B, IT, OT, RT> {
    /// Domain whose constant and operation types the captured programs are traced over.
    domain: &'d D,

    /// Closure computing the primal output tree from the primal input tree.
    primal: P,

    /// Closure computing `(outputs, residuals)` from the primal input tree.
    forward: F,

    /// Closure computing the input cotangent tree from `(residuals, output_cotangents)`.
    backward: B,

    /// Phantom marker pinning the input, output, and residual tracer-tree types named by the closure signatures.
    marker: PhantomData<fn() -> (IT, OT, RT)>,
}

/// Creates a [`CustomVjp`] function from primal, forward, and backward closures over trees of `domain`'s tracers.
/// Refer to the documentation of [`CustomVjp`] for the calling convention and tracing semantics.
pub fn custom_vjp<'d, D, P, F, B, IT, OT, RT>(
    domain: &'d D,
    primal: P,
    forward: F,
    backward: B,
) -> CustomVjp<'d, D, P, F, B, IT, OT, RT>
where
    D: Domain,
    P: Fn(IT) -> Result<OT, ProgramError>,
    F: Fn(IT) -> Result<(OT, RT), ProgramError>,
    B: Fn(RT, OT) -> Result<IT, ProgramError>,
{
    CustomVjp { domain, primal, forward, backward, marker: PhantomData }
}

impl<'d, D, P, F, B, IT, OT, RT> CustomVjp<'d, D, P, F, B, IT, OT, RT>
where
    D: Domain<Type: PartialEq> + 'd,
    P: Fn(IT) -> Result<OT, ProgramError>,
    F: Fn(IT) -> Result<(OT, RT), ProgramError>,
    B: Fn(RT, OT) -> Result<IT, ProgramError>,
    D::Operation: Clone + Operation<D::Type> + From<CustomVjpOperation<D::Constant, D::Operation, D::Type>>,
    IT: Parameterized<DomainTracer<'d, D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    OT: Parameterized<DomainTracer<'d, D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    RT: Parameterized<DomainTracer<'d, D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    IT::To<D::Type>: Clone
        + Parameterized<D::Type, Family = IT::Family, To<DomainTracer<'d, D>> = IT, To<D::Constant> = IT::To<D::Constant>>,
    OT::To<D::Type>: Clone
        + Parameterized<D::Type, Family = OT::Family, To<DomainTracer<'d, D>> = OT, To<D::Constant> = OT::To<D::Constant>>,
    RT::To<D::Type>: Parameterized<D::Type, Family = RT::Family, To<DomainTracer<'d, D>> = RT, To<D::Constant> = RT::To<D::Constant>>,
{
    /// Stages this custom-VJP function on the provided tracer input tree and returns its output tree, tracing the
    /// stored closures into programs specialized to the input types. Reverse-mode differentiation of the staged
    /// call replays the backward rule on the forward rule's residuals instead of differentiating the primal body.
    pub fn call<C, ICT>(
        &self,
        input: ICT,
    ) -> Result<<OT::To<D::Type> as Parameterized<D::Type>>::To<Tracer<C>>, ProgramError>
    where
        C: StagingContext<Type = D::Type, Constant = D::Constant, Operation = D::Operation>,
        IT::Family: ParameterizedFamily<Tracer<C>>,
        OT::Family: ParameterizedFamily<Tracer<C>>,
        ICT: Parameterized<Tracer<C>, Family = IT::Family, To<D::Type> = IT::To<D::Type>>,
        <OT::To<D::Type> as Parameterized<D::Type>>::To<Tracer<C>>: Parameterized<
                Tracer<C>,
                Family = OT::Family,
                ParameterStructure = <OT::To<D::Type> as Parameterized<D::Type>>::ParameterStructure,
            >,
    {
        let mut input_tracers = Vec::new();
        let input_types = input
            .map_parameters(|tracer| {
                let r#type = tracer.r#type().into_owned();
                input_tracers.push(tracer);
                r#type
            })
            .map_err(ProgramError::from)?;
        let Some(first) = input_tracers.first() else {
            return Err(TypeError { message: "custom_vjp requires at least one input".to_string() }.into());
        };
        let (output_types, primal) = TracingContext::trace(self.domain, |xs| (self.primal)(xs), input_types.clone())?;
        let (forward_output_types, forward) =
            TracingContext::trace(self.domain, |xs| (self.forward)(xs), input_types.clone())?;
        let (_, residual_types) = forward_output_types;
        let (_, backward) = TracingContext::trace(
            self.domain,
            |(residuals, cotangents)| (self.backward)(residuals, cotangents),
            (residual_types, output_types.clone()),
        )?;
        let operation = D::Operation::from(CustomVjpOperation::new(
            primal.to_flat_program(),
            forward.to_flat_program(),
            backward.to_flat_program(),
        )?);
        let outputs = first.context().stage_operation(operation, &input_tracers)?;
        let output_structure = output_types.parameter_structure();
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::contexts::StagingContext;
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::{Cos, CosOperation, Sin, SinOperation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::scalars::ScalarDomain;
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
    use crate::tracing_v2::test_util::assert_close;
    use crate::tracing_v2::{ArrayOperation, Batch, DifferentiationContext, value_and_grad};
    use crate::types::{DataType, Shape, Size};

    use super::*;

    /// Returns the canonical test array type with the provided dimensions.
    fn test_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(dimensions.iter().map(|dimension| Size::Static(*dimension)).collect()))
    }

    /// Builds `f(x) = sin(x)` over one input of the provided type.
    fn sin_program(
        r#type: &ArrayType,
    ) -> Program<ArrayType, TestArray, ArrayOperation<ArrayType, TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(SinOperation, vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong rule `jvp(x, dx) = (sin(x), 2 * cos(x) * dx)`, detectably different from the
    /// true derivative so tests can prove the custom rule is used.
    fn doubled_sin_jvp_program(
        r#type: &ArrayType,
    ) -> Program<ArrayType, TestArray, ArrayOperation<ArrayType, TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let dx = builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation, vec![x]).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation, vec![x]).unwrap()[0];
        let two = builder.add_constant(TestArray::scalar(2.0));
        let scaled = builder.add_instruction(MulOperation, vec![two, cosine]).unwrap()[0];
        let tangent = builder.add_instruction(MulOperation, vec![scaled, dx]).unwrap()[0];
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the forward rule `forward(x) = (sin(x), cos(x))`, with the cosine as the residual.
    fn sin_forward_program(
        r#type: &ArrayType,
    ) -> Program<ArrayType, TestArray, ArrayOperation<ArrayType, TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(r#type.clone());
        let y = builder.add_instruction(SinOperation, vec![x]).unwrap()[0];
        let residual = builder.add_instruction(CosOperation, vec![x]).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong rule `backward(residual, cotangent) = 3 * residual * cotangent`, detectably
    /// different from the true gradient so tests can prove the custom rule is used.
    fn tripled_sin_backward_program(
        r#type: &ArrayType,
    ) -> Program<ArrayType, TestArray, ArrayOperation<ArrayType, TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(r#type.clone());
        let cotangent = builder.add_input(r#type.clone());
        let three = builder.add_constant(TestArray::scalar(3.0));
        let scaled = builder.add_instruction(MulOperation, vec![three, residual]).unwrap()[0];
        let gradient = builder.add_instruction(MulOperation, vec![scaled, cotangent]).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_jvp_sin(r#type: &ArrayType) -> ArrayOperation<ArrayType, TestArray> {
        ArrayOperation::CustomJvp(Box::new(
            CustomJvpOperation::new(sin_program(r#type), doubled_sin_jvp_program(r#type)).unwrap(),
        ))
    }

    fn custom_vjp_sin(r#type: &ArrayType) -> ArrayOperation<ArrayType, TestArray> {
        ArrayOperation::CustomVjp(Box::new(
            CustomVjpOperation::new(
                sin_program(r#type),
                sin_forward_program(r#type),
                tripled_sin_backward_program(r#type),
            )
            .unwrap(),
        ))
    }

    #[test]
    fn test_custom_jvp_construction_validates_the_rule_signature() {
        let scalar = test_type(&[]);
        // The JVP program must take `(inputs..., tangents...)`; a primal-only signature is rejected.
        assert!(CustomJvpOperation::new(sin_program(&scalar), sin_program(&scalar)).is_err());
    }

    #[test]
    fn test_custom_vjp_construction_validates_the_rule_signatures() {
        let scalar = test_type(&[]);
        // The backward program must consume `(residuals..., output cotangents...)`; a single-input program whose
        // signature cannot line up with the forward residuals is rejected.
        assert!(
            CustomVjpOperation::new(sin_program(&scalar), sin_forward_program(&scalar), sin_program(&scalar)).is_err()
        );
    }

    #[test]
    fn test_custom_jvp_interprets_the_primal_program() {
        let scalar = test_type(&[]);
        let outputs =
            custom_jvp_sin(&scalar).interpret(&crate::EagerContext::new(), &[TestArray::scalar(2.0)]).unwrap();
        assert_close(outputs[0].values[0], 2.0f64.sin());
    }

    #[test]
    fn test_custom_jvp_governs_forward_mode() {
        let scalar = test_type(&[]);
        let (primal, tangent) = TestArrayDomain
            .jvp(
                |x| {
                    let operation = custom_jvp_sin(&test_type(&[]));
                    x.context().stage_operation(operation, &[&x]).unwrap().into_iter().next().unwrap()
                },
                TestArray::scalar(2.0),
                TestArray::scalar(1.0),
            )
            .unwrap();
        let _ = scalar;
        assert_close(primal.values[0], 2.0f64.sin());
        // The custom rule doubles the true derivative, proving it is in control.
        assert_close(tangent.values[0], 2.0 * 2.0f64.cos());
    }

    #[test]
    fn test_custom_jvp_governs_reverse_mode() {
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| {
                let operation = custom_jvp_sin(&test_type(&[]));
                x.context().stage_operation(operation, &[&x]).unwrap().into_iter().next().unwrap()
            },
            TestArray::scalar(3.0),
        )
        .unwrap();
        assert_close(value.values[0], 3.0f64.sin());
        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        assert_close(gradient.values[0], 2.0 * 3.0f64.cos());
    }

    #[test]
    fn test_custom_vjp_governs_reverse_mode() {
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| {
                let operation = custom_vjp_sin(&test_type(&[]));
                x.context().stage_operation(operation, &[&x]).unwrap().into_iter().next().unwrap()
            },
            TestArray::scalar(2.0),
        )
        .unwrap();
        assert_close(value.values[0], 2.0f64.sin());
        // The custom backward rule triples the true gradient, proving it is in control.
        assert_close(gradient.values[0], 3.0 * 2.0f64.cos());
    }

    #[test]
    fn test_custom_vjp_rejects_forward_mode() {
        // The staged linear call refuses interpretation in its un-transposed (pushforward) form, which is exactly
        // the operation `jvp` would need to execute; reverse mode transposes it first and replays `backward`.
        let scalar = test_type(&[]);
        let call = CustomVjpCallOperation::<TestArray, ArrayOperation<ArrayType, TestArray>, TestArray>::new(
            tripled_sin_backward_program(&scalar),
            None,
            vec![TestArray::scalar(2.0f64.cos())],
            false,
            false,
        );
        assert!(matches!(
            call.interpret(&crate::EagerContext::new(), &[TestArray::scalar(1.0)]),
            Err(ProgramError::Type(TypeError { message }))
                if message.starts_with("custom_vjp does not support forward-mode differentiation"),
        ));
    }

    #[test]
    fn test_jacrev_through_custom_vjp_uses_the_custom_backward_rule() {
        use crate::tracing_v2::jacrev;

        // jacrev interprets the pullback with lane-stacked cotangent bases, exercising the batched replay of the
        // custom backward program. The Jacobian of elementwise `sin` with the tripled rule is the diagonal matrix
        // `diag(3 * cos(x))`.
        let vector = test_type(&[2]);
        let jacobian = jacrev(
            &TestArrayDomain,
            |x| {
                let operation = custom_vjp_sin(&test_type(&[2]));
                Ok(x.context().stage_operation(operation, &[&x])?.into_iter().next().unwrap())
            },
            TestArray::new(vector, vec![0.5, 1.0]),
        )
        .unwrap();
        let (_, _, block) = jacobian.iter_blocks().next().unwrap();
        assert_close(block.values()[0], 3.0 * 0.5f64.cos());
        assert_close(block.values()[1], 0.0);
        assert_close(block.values()[2], 0.0);
        assert_close(block.values()[3], 3.0 * 1.0f64.cos());
    }

    /// Builds the scalar `f(x) = sin(x)` program.
    fn scalar_sin_program() -> Program<DataType, f64, ScalarOperation<f64>, Vec<f64>, Vec<f64>> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(SinOperation, vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong scalar rule `jvp(x, dx) = (sin(x), 2 * cos(x) * dx)`.
    fn scalar_doubled_sin_jvp_program() -> Program<DataType, f64, ScalarOperation<f64>, Vec<f64>, Vec<f64>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(DataType::F64);
        let dx = builder.add_input(DataType::F64);
        let y = builder.add_instruction(SinOperation, vec![x]).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation, vec![x]).unwrap()[0];
        let two = builder.add_constant(2.0);
        let scaled = builder.add_instruction(MulOperation, vec![two, cosine]).unwrap()[0];
        let tangent = builder.add_instruction(MulOperation, vec![scaled, dx]).unwrap()[0];
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the scalar forward rule `forward(x) = (sin(x), cos(x))`, with the cosine as the residual.
    fn scalar_sin_forward_program() -> Program<DataType, f64, ScalarOperation<f64>, Vec<f64>, Vec<f64>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(DataType::F64);
        let y = builder.add_instruction(SinOperation, vec![x]).unwrap()[0];
        let residual = builder.add_instruction(CosOperation, vec![x]).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong scalar rule `backward(residual, cotangent) = 3 * residual * cotangent`.
    fn scalar_tripled_sin_backward_program() -> Program<DataType, f64, ScalarOperation<f64>, Vec<f64>, Vec<f64>> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(DataType::F64);
        let cotangent = builder.add_input(DataType::F64);
        let three = builder.add_constant(3.0);
        let scaled = builder.add_instruction(MulOperation, vec![three, residual]).unwrap()[0];
        let gradient = builder.add_instruction(MulOperation, vec![scaled, cotangent]).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_scalar_custom_jvp_governs_forward_mode() {
        let (primal, tangent) = ScalarDomain::<f64>::new()
            .jvp(
                |x| {
                    let operation = ScalarOperation::CustomJvp(Box::new(
                        CustomJvpOperation::new(scalar_sin_program(), scalar_doubled_sin_jvp_program()).unwrap(),
                    ));
                    x.context().stage_operation(operation, &[&x]).unwrap().into_iter().next().unwrap()
                },
                2.0,
                1.0,
            )
            .unwrap();
        assert_close(primal, 2.0f64.sin());
        // The custom rule doubles the true derivative, proving it is in control.
        assert_close(tangent, 2.0 * 2.0f64.cos());
    }

    #[test]
    fn test_scalar_custom_vjp_governs_reverse_mode() {
        let (value, gradient) = value_and_grad(
            &ScalarDomain::<f64>::new(),
            |x| {
                let operation = ScalarOperation::CustomVjp(Box::new(
                    CustomVjpOperation::new(
                        scalar_sin_program(),
                        scalar_sin_forward_program(),
                        scalar_tripled_sin_backward_program(),
                    )
                    .unwrap(),
                ));
                x.context().stage_operation(operation, &[&x]).unwrap().into_iter().next().unwrap()
            },
            2.0,
        )
        .unwrap();
        assert_close(value, 2.0f64.sin());
        // The custom backward rule triples the true gradient, proving it is in control.
        assert_close(gradient, 3.0 * 2.0f64.cos());
    }

    #[test]
    fn test_custom_jvp_wrapper_traces_closures_lazily() {
        // No manual programs: the wrapper traces the closures at the call site, specialized to the input types.
        let domain = TestArrayDomain;
        let function = custom_jvp(
            &domain,
            |x: DomainTracer<'_, TestArrayDomain>| Ok(x.sin()),
            |x: DomainTracer<'_, TestArrayDomain>, dx| {
                // The deliberately wrong rule `jvp(x, dx) = (sin(x), cos(x) * dx + cos(x) * dx)` doubles the true
                // derivative (expressed through addition to avoid constant lifting), proving the rule is in control.
                let tangent = x.cos() * dx;
                Ok((x.sin(), tangent.clone() + tangent))
            },
        );
        let (primal, tangent) = TestArrayDomain
            .jvp(|x| function.call(x).unwrap(), TestArray::scalar(2.0), TestArray::scalar(1.0))
            .unwrap();
        assert_close(primal.values[0], 2.0f64.sin());
        assert_close(tangent.values[0], 2.0 * 2.0f64.cos());
        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        let (value, gradient) =
            value_and_grad(&TestArrayDomain, |x| function.call(x).unwrap(), TestArray::scalar(3.0)).unwrap();
        assert_close(value.values[0], 3.0f64.sin());
        assert_close(gradient.values[0], 2.0 * 3.0f64.cos());
    }

    #[test]
    fn test_custom_vjp_wrapper_governs_reverse_mode() {
        let domain = TestArrayDomain;
        let function = custom_vjp(
            &domain,
            |x: DomainTracer<'_, TestArrayDomain>| Ok(x.sin()),
            |x: DomainTracer<'_, TestArrayDomain>| Ok((x.sin(), x.cos())),
            |residual, cotangent| {
                // The deliberately wrong rule `backward(residual, cotangent) = 3 * residual * cotangent` triples the
                // true gradient (expressed through addition to avoid constant lifting).
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let (value, gradient) =
            value_and_grad(&TestArrayDomain, |x| function.call(x).unwrap(), TestArray::scalar(2.0)).unwrap();
        assert_close(value.values[0], 2.0f64.sin());
        assert_close(gradient.values[0], 3.0 * 2.0f64.cos());
    }

    #[test]
    fn test_custom_vjp_wrapper_supports_structured_signatures_and_captured_configuration() {
        // Tuple inputs and tuple residuals exercise the `Parameterized` (pytree) calling convention, and the
        // captured `triple` closure plays the role of a JAX `nondiff_argnums` argument: static configuration
        // visible to the rule closures without being differentiated or stored as a residual.
        let domain = TestArrayDomain;
        let repeats = 3usize;
        let function = custom_vjp(
            &domain,
            |(x, y): (DomainTracer<'_, TestArrayDomain>, DomainTracer<'_, TestArrayDomain>)| Ok(x * y),
            |(x, y): (DomainTracer<'_, TestArrayDomain>, DomainTracer<'_, TestArrayDomain>)| {
                Ok((x.clone() * y.clone(), (x, y)))
            },
            move |(x, y), cotangent| {
                // The deliberately wrong rule repeats both cotangents `repeats` times via the captured count.
                let (base_x, base_y) = (y * cotangent.clone(), x * cotangent);
                let (mut scaled_x, mut scaled_y) = (base_x.clone(), base_y.clone());
                for _ in 1..repeats {
                    scaled_x = scaled_x + base_x.clone();
                    scaled_y = scaled_y + base_y.clone();
                }
                Ok((scaled_x, scaled_y))
            },
        );
        let (value, (gradient_x, gradient_y)) = value_and_grad(
            &TestArrayDomain,
            |(x, y)| function.call((x, y)).unwrap(),
            (TestArray::scalar(2.0), TestArray::scalar(5.0)),
        )
        .unwrap();
        assert_close(value.values[0], 10.0);
        // The custom rule triples the true gradients `(y, x)`.
        assert_close(gradient_x.values[0], 3.0 * 5.0);
        assert_close(gradient_y.values[0], 3.0 * 2.0);
    }

    #[test]
    fn test_scalar_custom_vjp_wrapper_governs_reverse_mode() {
        let domain = ScalarDomain::<f64>::new();
        let function = custom_vjp(
            &domain,
            |x: DomainTracer<'_, ScalarDomain<f64>>| Ok(x.sin()),
            |x: DomainTracer<'_, ScalarDomain<f64>>| Ok((x.sin(), x.cos())),
            |residual, cotangent| {
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let (value, gradient) = value_and_grad(&domain, |x| function.call(x).unwrap(), 2.0).unwrap();
        assert_close(value, 2.0f64.sin());
        assert_close(gradient, 3.0 * 2.0f64.cos());
    }

    #[test]
    fn test_custom_jvp_wrapper_surfaces_rule_signature_mismatches() {
        // Arity mismatches are compile-time errors under the structured signatures, but shape mismatches remain
        // runtime concerns: this rule produces a scalar tangent for a vector-valued function, so the traced JVP
        // program fails the signature validation that `CustomJvpOperation::new` performs at the call site.
        let domain = TestArrayDomain;
        let function = custom_jvp(
            &domain,
            |x: DomainTracer<'_, TestArrayDomain>| Ok(x.sin()),
            |x: DomainTracer<'_, TestArrayDomain>, dx| {
                Ok((x.sin(), dx.dot(&dx, &DotDimensionNumbers::inner_product())))
            },
        );
        let error = crate::tracing::TracingContext::trace(&domain, |x| function.call(x), test_type(&[2])).unwrap_err();
        assert!(error.to_string().contains("custom_jvp rule output"));
    }

    #[test]
    fn test_custom_jvp_batches_by_rewrapping_the_call() {
        let scalar = test_type(&[]);
        let output: TestArray = TestArrayDomain
            .batch(
                |x| {
                    let operation = custom_jvp_sin(&scalar);
                    Ok(x.context().stage_operation(operation, &[&x])?.into_iter().next().unwrap())
                },
                TestArray::vector(vec![0.5, 1.0, 1.5]),
                Some(0),
                Some(0),
                None,
            )
            .unwrap();
        for (actual, input) in output.values.iter().zip([0.5f64, 1.0, 1.5]) {
            assert_close(*actual, input.sin());
        }
    }

    #[test]
    fn test_custom_jvp_survives_batching_and_governs_the_batched_gradient() {
        use crate::tracing_v2::batching::BatchContext;
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
        use crate::tracing_v2::{LinearizationTracer, value_and_grad};

        // Differentiating *through* a batch of the custom call must still use the (deliberately doubled) custom
        // rule: traced batching re-wraps the call around batched programs instead of inlining the primal, so the
        // custom derivative survives `batch` — mirroring JAX's `vmap`-of-`custom_jvp` semantics.
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| {
                let context = x.context().clone();
                let mapped: LinearizationTracer<'_, TestArrayDomain> = BatchContext::batch(
                    &context,
                    |lane| {
                        let operation = custom_jvp_sin(&test_type(&[]));
                        Ok(lane.context().stage_operation(operation, &[&lane])?.into_iter().next().unwrap())
                    },
                    x,
                    Some(0),
                    Some(0),
                    None,
                )
                .unwrap();
                mapped.reduce(&[0], ReductionKind::Sum)
            },
            TestArray::vector(vec![0.5, 1.0]),
        )
        .unwrap();
        assert_close(value.values[0], 0.5f64.sin() + 1.0f64.sin());
        assert_close(gradient.values[0], 2.0 * 0.5f64.cos());
        assert_close(gradient.values[1], 2.0 * 1.0f64.cos());
    }

    #[test]
    fn test_custom_vjp_survives_batching_and_governs_the_batched_gradient() {
        use crate::tracing_v2::batching::BatchContext;
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
        use crate::tracing_v2::{LinearizationTracer, value_and_grad};

        // The reverse-mode analogue of the test above: the (deliberately tripled) custom backward rule governs the
        // gradient through the batched call — mirroring JAX's `vmap`-of-`custom_vjp` semantics.
        let (value, gradient) = value_and_grad(
            &TestArrayDomain,
            |x| {
                let context = x.context().clone();
                let mapped: LinearizationTracer<'_, TestArrayDomain> = BatchContext::batch(
                    &context,
                    |lane| {
                        let operation = custom_vjp_sin(&test_type(&[]));
                        Ok(lane.context().stage_operation(operation, &[&lane])?.into_iter().next().unwrap())
                    },
                    x,
                    Some(0),
                    Some(0),
                    None,
                )
                .unwrap();
                mapped.reduce(&[0], ReductionKind::Sum)
            },
            TestArray::vector(vec![0.5, 1.0]),
        )
        .unwrap();
        assert_close(value.values[0], 0.5f64.sin() + 1.0f64.sin());
        assert_close(gradient.values[0], 3.0 * 0.5f64.cos());
        assert_close(gradient.values[1], 3.0 * 1.0f64.cos());
    }

    #[test]
    fn test_transposing_a_user_custom_vjp_pullback_is_still_rejected() {
        // Rematerialization-derived calls carry a tangent program and support pullback re-transposition; user
        // custom VJPs do not, so second-order reverse mode through them keeps erroring, matching JAX.
        let domain = TestArrayDomain;
        let function = custom_vjp(
            &domain,
            |x: DomainTracer<'_, TestArrayDomain>| Ok(x.sin()),
            |x: DomainTracer<'_, TestArrayDomain>| Ok((x.sin(), x.cos())),
            |residual, cotangent| Ok(residual * cotangent),
        );
        let (_, pullback) = domain.vjp(|x| function.call(x), TestArray::scalar(0.7)).unwrap();
        let error = domain.transpose_linear_program(&pullback).unwrap_err();
        assert!(
            error.to_string().contains("second-order reverse mode through custom_vjp"),
            "unexpected error: {error}",
        );
    }

    #[test]
    fn test_custom_vjp_batching_broadcasts_lane_uniform_inputs() {
        // Mapping only the first input exercises the lane-uniform broadcast in the re-wrapping batch rule: the
        // unmapped operand is broadcast into the lane (the all-inputs-mapped-at-0 convention) and the batched call
        // still computes per-lane products.
        let domain = TestArrayDomain;
        let function = custom_vjp(
            &domain,
            |(x, y): (DomainTracer<'_, TestArrayDomain>, DomainTracer<'_, TestArrayDomain>)| Ok(x * y),
            |(x, y): (DomainTracer<'_, TestArrayDomain>, DomainTracer<'_, TestArrayDomain>)| {
                Ok((x.clone() * y.clone(), (x, y)))
            },
            |(x, y), cotangent| Ok((y * cotangent.clone(), x * cotangent)),
        );
        let output: TestArray = TestArrayDomain
            .batch(
                |(x, y)| function.call((x, y)),
                (TestArray::vector(vec![2.0, 3.0, 4.0]), TestArray::scalar(5.0)),
                (Some(0), None),
                Some(0),
                None,
            )
            .unwrap();
        assert_eq!(output.values, vec![10.0, 15.0, 20.0]);
    }
}
