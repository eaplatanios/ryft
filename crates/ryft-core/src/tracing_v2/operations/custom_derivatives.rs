use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::batching::BatchingContext;
use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchableProgramOperation, BatchingError,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain, EagerContext, StagingContext};
use crate::differentiation::TransposableOperation;
use crate::differentiation::{DifferentiableOperation, DifferentiationDual, DifferentiationError};
use crate::effects::Effects;
use crate::interpretation::{InterpretableOperation, InterpretableProgramOperation};
use crate::macros::{check_count, check_types};
use crate::operations::Operation;
use crate::operations::constants::Constant as ConstantCapability;
use crate::operations::constants::{MaybeZeroOperation, Zero, ZeroOperation};
use crate::operations::manipulation::{Broadcast, BroadcastOperation, Transpose, TransposeOperation};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::payloads::Captured;
use crate::programs::{MaybeZero, Program, ProgramError, Value};
use crate::tracing::{DomainTracer, Trace, Tracer, TracingContext};
use crate::types::{ArrayType, TypeError, Typed};

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
pub struct CustomJvpOperation<V: Value, O> {
    /// Program computing the primal outputs from the primal inputs.
    primal: Program<V, O, Vec<V>, Vec<V>>,

    /// Program computing `(outputs..., output_tangents...)` from `(inputs..., input_tangents...)`.
    jvp: Program<V, O, Vec<V>, Vec<V>>,
}

impl<V: Value, O: Operation<V::Type>> CustomJvpOperation<V, O> {
    /// Creates a custom-JVP operation after validating that the JVP program's signature matches the primal
    /// program's: its inputs must be the primal inputs followed by their tangents (same types), and its outputs the
    /// primal outputs followed by their tangents.
    pub fn new(primal: Program<V, O, Vec<V>, Vec<V>>, jvp: Program<V, O, Vec<V>, Vec<V>>) -> Result<Self, TypeError> {
        let input_types = primal.input_types();
        let output_types = primal.output_types();
        let expected_jvp_input_types: Vec<V::Type> = input_types.iter().chain(input_types.iter()).cloned().collect();
        check_types!("custom_jvp rule input", &expected_jvp_input_types, &jvp.input_types());
        let expected_jvp_output_types: Vec<V::Type> = output_types.iter().chain(output_types.iter()).cloned().collect();
        check_types!("custom_jvp rule output", &expected_jvp_output_types, &jvp.output_types());
        Ok(Self { primal, jvp })
    }

    /// Returns the primal program.
    #[inline]
    pub fn primal(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.primal
    }

    /// Returns the user-supplied JVP program.
    #[inline]
    pub fn jvp_program(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.jvp
    }

    /// Returns the primal input types.
    #[inline]
    pub fn input_types(&self) -> Vec<V::Type> {
        self.primal.input_types()
    }

    /// Returns the primal output types.
    #[inline]
    pub fn output_types(&self) -> Vec<V::Type> {
        self.primal.output_types()
    }
}

impl<V: Value, O> Display for CustomJvpOperation<V, O>
where
    Self: Operation<V::Type>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value, O: Operation<V::Type>> Operation<V::Type> for CustomJvpOperation<V, O> {
    #[inline]
    fn name(&self) -> &'static str {
        "custom_jvp"
    }

    fn infer_output_types(&self, input_types: &[V::Type]) -> Result<Vec<V::Type>, TypeError> {
        check_types!("custom_jvp input", &self.input_types(), input_types);
        Ok(self.output_types())
    }

    #[inline]
    fn effects(&self) -> Effects {
        self.primal.effects().union(self.jvp.effects())
    }
}

impl<Constant, O, V, C> InterpretableOperation<V, C> for CustomJvpOperation<Constant, O>
where
    Constant: Value,
    V: Value<Type = Constant::Type>,
    O: InterpretableProgramOperation<V, C, Constant>,
{
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        O::interpret_program(context, &self.primal, inputs.to_vec())
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`] for a
/// [`CustomJvpOperation`]: a call with all-known operands folds by interpreting its primal, and otherwise
/// residualizes unchanged.
impl<V: Value, O: Clone + Operation<V::Type>, C: Context<Type = V::Type>> PartiallyEvaluatableOperation<C>
    for CustomJvpOperation<V, O>
where
    C::Operation: From<CustomJvpOperation<V, O>>,
{
}

/// Capture-free forward-mode (JVP) rule for [`CustomJvpOperation`]: splices the user-supplied JVP program
/// directly into the shared builder.
///
/// The JVP program is already JVP-shaped over the primal operation family — it maps
/// `(inputs..., input_tangents...)` to `(outputs..., output_tangents...)` — so the rule simply replays it
/// through [`Program::interpret_in_context`](crate::Program::interpret_in_context) over the dual inputs: the primal tracers followed by the
/// tangent tracers feed the JVP
/// program, and its outputs split into the primal outputs and the staged output tangents. Because the spliced program
/// is straight-line primal-enum operations referencing those tracers directly, it introduces no symbolic capture and
/// the enclosing partial-evaluation split discovers the residual operand edges structurally — so
/// the rule is a leaf needing no [`DifferentiableProgramOperation`](crate::differentiation::DifferentiableProgramOperation)
/// or [`LinearizableProgramOperation`](crate::differentiation::LinearizableProgramOperation)
/// witness, and reverse mode transposes the spliced bilinear operations exactly as it does for any other straight-line
/// tangent program.
impl<C: Context + Zero<C::Value>> DifferentiableOperation<C> for CustomJvpOperation<C::Constant, C::Operation>
where
    C::Constant: Clone,
    C::Operation: Clone,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let output_count = self.output_types().len();
        check_count!("input", inputs, self.input_types().len(), ProgramError);
        // The JVP program consumes `(primals..., input_tangents...)`, so feed the dual primals followed by the dual
        // tangents.
        let mut jvp_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        // The user's JVP program takes every input tangent as a real program input, so materialize structural
        // zeros.
        for input in inputs {
            jvp_inputs.push(input.tangent().clone().materialize(context)?);
        }
        let mut outputs = self.jvp_program().interpret_in_context(context, jvp_inputs)?;
        check_count!("output", outputs, 2 * output_count, ProgramError);
        let tangents = outputs.split_off(output_count);
        Ok(outputs
            .into_iter()
            .zip(tangents)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect())
    }
}

/// Transpose rule for [`CustomJvpOperation`]: the call is a higher-order primal boundary rather than a linear map,
/// so a tangent program never contains it on a linear operand (linearization splices the user-supplied JVP program
/// instead) and the rule reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V, O, W, OLinear> TransposableOperation<W, OLinear> for CustomJvpOperation<V, O>
where
    V: Value,
    W: Value<Type = V::Type>,
    O: Operation<V::Type>,
    OLinear: Operation<V::Type>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<W, OLinear>,
        _inputs: &[PartialValue<Tracer<TracingContext<W, OLinear>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<W, OLinear>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<W, OLinear>>>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("operation `{}` has no partition-aware transpose rule", self.name()),
        }
        .into())
    }
}

/// Value-level batching for [`CustomJvpOperation`]: inlines the primal program through the per-operation batching
/// rules. The custom derivative does not survive this inlining; see the type-level documentation.
impl<V, O> BatchableOperation<V, EagerContext<V, O>> for CustomJvpOperation<V, O>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<V, EagerContext<V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<V, O>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        // Replay the primal program over the packed batch values, dispatching every instruction through its
        // value-level batching rule. Eager constants are the flowing values themselves, so they replicate as-is
        // across the batch.
        self.primal.interpret_with(
            inputs.to_vec(),
            |_, constant: &V| Ok(ArrayBatch::replicated(constant.clone())),
            |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
        )
    }
}

/// Stages a re-wrapped custom-derivative call into the batching context's parent trace.
///
/// This is the shared body of the traced `batch` rules for [`CustomJvpOperation`] and [`CustomVjpOperation`],
/// mirroring JAX's `custom_jvp_call_jaxpr` / `custom_vjp_call_jaxpr` batching rules: instead of inlining the primal
/// program (which would lose the custom derivative and any rematerialization structure), the rule stages one new
/// custom-derivative call whose captured programs have been batched. When no input carries the mapped batch axis the
/// original operation is staged unchanged and the outputs stay replicated. Otherwise every input is aligned to
/// carry the batch axis at axis `0` (replicated inputs are broadcast, matching the convention that every
/// custom-call input is mapped at axis `0`) and every output carries the batch axis at axis `0`.
pub(crate) fn stage_rewrapped_custom_call<C, MakeOperationFn>(
    context: &BatchingContext<C>,
    inputs: &[ArrayBatch<<C as Domain>::Value>],
    make_operation: MakeOperationFn,
) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    <C as Domain>::Value: Broadcast + Transpose,
    MakeOperationFn: FnOnce(Option<usize>) -> Result<C::Operation, ProgramError>,
{
    if inputs.iter().all(|input| input.batch_axis().is_replicated()) {
        let operation = make_operation(None)?;
        let parent_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let outputs = context.parent().bind(operation, &parent_inputs)?;
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
        .map(|input| match input.batch_axis_position() {
            Some(_) => input.move_axis(0),
            None => input.broadcast(0, axis_size),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let operation = make_operation(Some(axis_size))?;
    let parent_inputs = aligned_inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
    let outputs = context.parent().bind(operation, &parent_inputs)?;
    outputs
        .into_iter()
        .map(|tracer| {
            let physical_type = tracer.r#type().into_owned();
            ArrayBatch::new(physical_type, tracer, Some(0))
        })
        .collect()
}

/// Batches `program` using the custom-derivative rewrapping convention: every input and output is mapped at axis `0`.
pub(crate) fn batch_rewrapped_program<V: Value<Type = ArrayType>, O: BatchableProgramOperation<V>>(
    program: &Program<V, O, Vec<V>, Vec<V>>,
    axis_size: usize,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
    let input_batch_axes = vec![BatchAxis::new(0); program.input_types().len()];
    let (program, _) = O::batch_program(
        program,
        axis_size,
        input_batch_axes.as_slice(),
        ProgramBatchingOutputAxesPolicy::AlignAllTo(0),
    )?;
    Ok(program)
}

/// Traced batching for [`CustomJvpOperation`]: re-wraps the call around batched primal/JVP programs so the custom
/// derivative survives `batch`; see `stage_rewrapped_custom_call`.
impl<C, O> BatchableOperation<<C as Domain>::Value, BatchingContext<C>> for CustomJvpOperation<C::Constant, O>
where
    C: Context<Type = ArrayType, Operation = O>,
    C::Constant: Value<Type = ArrayType>,
    <C as Domain>::Value: Broadcast + Transpose,
    O: Clone
        + Operation<ArrayType>
        + From<TransposeOperation>
        + From<BroadcastOperation>
        + From<CustomJvpOperation<C::Constant, O>>
        + BatchableProgramOperation<C::Constant>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        stage_rewrapped_custom_call(context, inputs, |batched| match batched {
            None => Ok(O::from(self.clone())),
            Some(axis_size) => Ok(O::from(CustomJvpOperation::new(
                batch_rewrapped_program(&self.primal, axis_size)?,
                batch_rewrapped_program(&self.jvp, axis_size)?,
            )?)),
        })
    }
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
/// Traced batching (`batch`) re-wraps the call around batched primal/forward/backward programs — mirroring JAX's
/// `custom_vjp_call_jaxpr` batching rule — so the custom derivative survives a `batch` applied *before*
/// differentiation. Value-level batching (used by dense Jacobian materialization, where the custom rule has already
/// been consumed by linearization) inlines the primal program through the standard per-operation batching rules.
#[derive(Clone, Debug)]
pub struct CustomVjpOperation<V: Value, O> {
    /// Program computing the primal outputs from the primal inputs.
    primal: Program<V, O, Vec<V>, Vec<V>>,

    /// Program computing `(outputs..., residuals...)` from the primal inputs.
    forward: Program<V, O, Vec<V>, Vec<V>>,

    /// Program computing one input cotangent per primal input from `(residuals..., output_cotangents...)`.
    backward: Program<V, O, Vec<V>, Vec<V>>,
}

impl<V: Value, O: Operation<V::Type>> CustomVjpOperation<V, O> {
    /// Creates a custom-VJP operation after validating the forward/backward program signatures against the primal
    /// program's: `forward` must consume the primal inputs and produce the primal outputs followed by the residuals,
    /// and `backward` must consume those residuals followed by one cotangent per primal output and produce one
    /// cotangent per primal input.
    pub fn new(
        primal: Program<V, O, Vec<V>, Vec<V>>,
        forward: Program<V, O, Vec<V>, Vec<V>>,
        backward: Program<V, O, Vec<V>, Vec<V>>,
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
        let expected_backward_input_types: Vec<V::Type> =
            residual_types.iter().chain(output_types.iter()).cloned().collect();
        check_types!("custom_vjp backward input", &expected_backward_input_types, &backward.input_types(),);
        check_types!("custom_vjp backward output", &input_types, &backward.output_types());
        Ok(Self { primal, forward, backward })
    }

    /// Returns the primal program.
    #[inline]
    pub fn primal(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.primal
    }

    /// Returns the forward (residual-producing) program.
    #[inline]
    pub fn forward(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.forward
    }

    /// Returns the backward (cotangent-producing) program.
    #[inline]
    pub fn backward(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.backward
    }

    /// Returns the primal input types.
    #[inline]
    pub fn input_types(&self) -> Vec<V::Type> {
        self.primal.input_types()
    }

    /// Returns the primal output types.
    #[inline]
    pub fn output_types(&self) -> Vec<V::Type> {
        self.primal.output_types()
    }
}

impl<V: Value, O> Display for CustomVjpOperation<V, O>
where
    Self: Operation<V::Type>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value, O: Operation<V::Type>> Operation<V::Type> for CustomVjpOperation<V, O> {
    #[inline]
    fn name(&self) -> &'static str {
        "custom_vjp"
    }

    fn infer_output_types(&self, input_types: &[V::Type]) -> Result<Vec<V::Type>, TypeError> {
        check_types!("custom_vjp input", &self.input_types(), input_types);
        Ok(self.output_types())
    }

    #[inline]
    fn effects(&self) -> Effects {
        self.primal.effects().union(self.forward.effects()).union(self.backward.effects())
    }
}

impl<Constant, O, V, C> InterpretableOperation<V, C> for CustomVjpOperation<Constant, O>
where
    Constant: Value,
    V: Value<Type = Constant::Type>,
    O: InterpretableProgramOperation<V, C, Constant>,
{
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        O::interpret_program(context, &self.primal, inputs.to_vec())
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`] for a
/// [`CustomVjpOperation`]: a call with all-known operands folds by interpreting its primal, and otherwise
/// residualizes unchanged.
impl<V: Value, O: Clone + Operation<V::Type>, C: Context<Type = V::Type>> PartiallyEvaluatableOperation<C>
    for CustomVjpOperation<V, O>
where
    C::Operation: From<CustomVjpOperation<V, O>>,
{
}

/// Value-level batching for [`CustomVjpOperation`]: inlines the primal program; see [`CustomJvpOperation`]'s
/// batching documentation for the custom-derivative caveat.
impl<V, O> BatchableOperation<V, EagerContext<V, O>> for CustomVjpOperation<V, O>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + BatchableOperation<V, EagerContext<V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<V, O>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        // Replay the primal program over the packed batch values, dispatching every instruction through its
        // value-level batching rule. Eager constants are the flowing values themselves, so they replicate as-is
        // across the batch.
        self.primal.interpret_with(
            inputs.to_vec(),
            |_, constant: &V| Ok(ArrayBatch::replicated(constant.clone())),
            |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
        )
    }
}

/// Traced batching for [`CustomVjpOperation`]: re-wraps the call around batched primal/forward/backward programs so
/// the custom derivative survives `batch`; see `stage_rewrapped_custom_call`.
impl<C, O> BatchableOperation<<C as Domain>::Value, BatchingContext<C>> for CustomVjpOperation<C::Constant, O>
where
    C: Context<Type = ArrayType, Operation = O>,
    C::Constant: Value<Type = ArrayType>,
    <C as Domain>::Value: Broadcast + Transpose,
    O: Clone
        + Operation<ArrayType>
        + From<TransposeOperation>
        + From<BroadcastOperation>
        + From<CustomVjpOperation<C::Constant, O>>
        + BatchableProgramOperation<C::Constant>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        stage_rewrapped_custom_call(context, inputs, |batched| match batched {
            None => Ok(O::from(self.clone())),
            Some(axis_size) => Ok(O::from(CustomVjpOperation::new(
                batch_rewrapped_program(&self.primal, axis_size)?,
                batch_rewrapped_program(&self.forward, axis_size)?,
                batch_rewrapped_program(&self.backward, axis_size)?,
            )?)),
        })
    }
}

/// Access to a custom-VJP residual payload as a concrete value during pullback interpretation.
///
/// Implemented by plain values as the identity.
#[doc(hidden)]
pub trait CustomVjpResidual<V: Value>: Value<Type = V::Type> {
    /// Returns the concrete residual value.
    fn residual_value(&self) -> Result<V, ProgramError>;
}

impl<V: Value> CustomVjpResidual<V> for V {
    #[inline]
    fn residual_value(&self) -> Result<V, ProgramError> {
        Ok(self.clone())
    }
}

/// Opaque linear operation staged by [`CustomVjpOperation`]'s JVP rule.
///
/// In its un-transposed form it stands for the (unknown) tangent map of the custom function and rejects
/// interpretation: `custom_vjp` functions are reverse-mode-only, matching JAX. Transposition replaces it with its
/// transposed form, whose interpretation replays the user's backward program on the captured residuals and the
/// incoming output cotangents.
#[derive(Clone, Debug)]
pub struct CustomVjpCallOperation<V: Value, O, F: Value<Type = V::Type>> {
    /// The user's backward program, mapping `(residuals..., output_cotangents...)` to input cotangents.
    backward: Program<V, O, Vec<V>, Vec<V>>,

    /// Captured residual factors consumed by the backward program.
    residuals: Vec<F>,

    /// Whether this call has been transposed into its executable (pullback) form.
    transposed: bool,
}

impl<V: Value, F: Value<Type = V::Type>, O> CustomVjpCallOperation<V, O, F> {
    /// Creates a custom-VJP call. Use `transposed = false` for the opaque pushforward form and `transposed = true` for
    /// the executable pullback form.
    pub fn new(backward: Program<V, O, Vec<V>, Vec<V>>, residuals: Vec<F>, transposed: bool) -> Self {
        Self { backward, residuals, transposed }
    }

    /// Returns the user's backward program.
    #[inline]
    pub fn backward(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.backward
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

    /// Maps the residual factor payloads with `map_factor`, preserving the backward program and direction.
    pub fn map_captures<MappedFactor: Value<Type = V::Type>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<CustomVjpCallOperation<V, O, MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
        O: Clone,
    {
        Ok(CustomVjpCallOperation {
            backward: self.backward.clone(),
            residuals: self.residuals.iter().map(map_factor).collect::<Result<Vec<_>, _>>()?,
            transposed: self.transposed,
        })
    }
}

impl<V: Value, F: Value<Type = V::Type>, O: Operation<V::Type>> CustomVjpCallOperation<V, O, F> {
    /// Returns the cotangent types flowing *into* the backward program (one per primal output).
    fn cotangent_types(&self) -> Vec<V::Type> {
        self.backward.input_types().split_off(self.residuals.len())
    }
}

impl<V: Value, F: Value<Type = V::Type>, O> Display for CustomVjpCallOperation<V, O, F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.transposed {
            formatter.write_str("custom_vjp_backward")
        } else {
            formatter.write_str("custom_vjp_tangent")
        }
    }
}

impl<V: Value, F: Value<Type = V::Type>, O: Operation<V::Type>> Operation<V::Type> for CustomVjpCallOperation<V, O, F> {
    #[inline]
    fn name(&self) -> &'static str {
        if self.transposed { "custom_vjp_backward" } else { "custom_vjp_tangent" }
    }

    fn infer_output_types(&self, input_types: &[V::Type]) -> Result<Vec<V::Type>, TypeError> {
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

    #[inline]
    fn effects(&self) -> Effects {
        self.backward.effects()
    }
}

/// Interprets a [`CustomVjpCallOperation`] in an active [`Context`].
///
/// Custom-VJP calls are context-mediated because executing the transposed call means replaying a captured backward
/// program. Eager contexts replay the program into concrete values, while staging contexts replay it into tracer
/// instructions by lifting constants and binding each captured instruction through the active context.
pub trait CustomVjpCall<Constant, O, F, V>: ConstantCapability<V, Constant, Captured>
where
    Constant: Value,
    O: Operation<Constant::Type>,
    F: Value<Type = Constant::Type>,
    V: Value<Type = Constant::Type>,
{
    /// Interprets `operation` over `inputs`.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Custom-VJP call whose backward program is replayed.
    ///   - `inputs`: Runtime tangent or cotangent inputs for the selected captured program.
    fn interpret_custom_vjp_call(
        &self,
        operation: &CustomVjpCallOperation<Constant, O, F>,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError>;
}

impl<Constant, O, F, V, C> CustomVjpCall<Constant, O, F, V> for C
where
    Constant: Value,
    V: Value<Type = Constant::Type>,
    F: CustomVjpResidual<V>,
    O: InterpretableOperation<V, C> + Operation<Constant::Type>,
    C: ConstantCapability<V, Constant, Captured>,
    Vec<Constant>: Parameterized<Constant, ParameterStructure: Debug + PartialEq>,
{
    fn interpret_custom_vjp_call(
        &self,
        operation: &CustomVjpCallOperation<Constant, O, F>,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        if !operation.transposed {
            return Err(TypeError {
                message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                    value_and_gradient, or jacrev) instead"
                    .to_string(),
            }
            .into());
        }
        let mut values = operation
            .residuals
            .iter()
            .map(|residual| residual.residual_value())
            .collect::<Result<Vec<_>, _>>()?;
        values.extend(inputs.iter().cloned());
        operation.backward.interpret_with(
            values,
            |_, constant| self.constant(constant.clone()),
            |instruction, inputs| instruction.operation().interpret(self, inputs),
        )
    }
}

impl<Constant, O, F, V, C> InterpretableOperation<V, C> for CustomVjpCallOperation<Constant, O, F>
where
    Constant: Value,
    V: Value<Type = Constant::Type>,
    F: Value<Type = Constant::Type>,
    O: Operation<Constant::Type>,
    C: CustomVjpCall<Constant, O, F, V>,
{
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        context.interpret_custom_vjp_call(self, inputs)
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`] for a
/// [`CustomVjpCallOperation`]. The residual operation family `O` is independent of the call's primal operation family
/// `CallOperation`, because partial evaluation never inlines a nested program here and so never builds a residual
/// program of its own.
impl<V, CallOperation, F, C> PartiallyEvaluatableOperation<C> for CustomVjpCallOperation<V, CallOperation, F>
where
    V: Value,
    CallOperation: Clone + Operation<V::Type>,
    F: Value<Type = V::Type>,
    C: Context<Type = V::Type>,
    C::Operation: From<CustomVjpCallOperation<V, CallOperation, F>>,
{
}

/// Transpose rule for [`CustomVjpCallOperation`]: stages the flipped-direction form of the call on the output
/// cotangents, materializing structural zeros so the staged call receives every cotangent input. The rule is
/// generic over the cotangent value type `W`, which need not match the backward program's value type `V`: the staged
/// flipped call carries the programs and residuals along unchanged.
///
/// The un-transposed (tangent-map) call transposes into the executable pullback. The pullback does not transpose back:
/// user-authored custom VJPs have no tangent program, matching JAX's behavior for second-order reverse mode through
/// `custom_vjp`.
impl<V, O, F, W, OLinear> TransposableOperation<W, OLinear> for CustomVjpCallOperation<V, O, F>
where
    V: Value,
    F: Value<Type = V::Type>,
    W: Value<Type = V::Type>,
    O: Clone + Operation<V::Type>,
    OLinear: Operation<V::Type> + From<ZeroOperation<V::Type>> + From<CustomVjpCallOperation<V, O, F>>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<W, OLinear>,
        _inputs: &[PartialValue<Tracer<TracingContext<W, OLinear>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<W, OLinear>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<W, OLinear>>>>, DifferentiationError> {
        if self.transposed {
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
        check_count!("output", outputs, cotangent_types.len(), ProgramError);
        let cotangent_tracers = outputs
            .iter()
            .map(|cotangent| cotangent.clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        let call = OLinear::from(CustomVjpCallOperation {
            backward: self.backward.clone(),
            residuals: self.residuals.to_vec(),
            transposed: !self.transposed,
        });
        let outputs = context.stage_operation(call, cotangent_tracers.as_slice())?;
        Ok(outputs.into_iter().map(MaybeZero::Value).collect())
    }
}

/// Value-level batching for the transposed [`CustomVjpCallOperation`]: replays the backward program through the
/// per-operation batching rules with the captured residuals as replicated values. Used when a pullback containing
/// custom-VJP calls is interpreted with batched cotangents (e.g., by `jacrev`). The un-transposed form rejects
/// batching just as it rejects interpretation.
impl<V, O, F> BatchableOperation<V, EagerContext<V, O>> for CustomVjpCallOperation<V, O, F>
where
    V: Value<Type = ArrayType>,
    F: CustomVjpResidual<V>,
    O: Operation<ArrayType> + BatchableOperation<V, EagerContext<V, O>>,
{
    fn batch(
        &self,
        context: &EagerContext<V, O>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        if !self.transposed {
            return Err(TypeError {
                message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                    value_and_gradient, or jacrev) instead"
                    .to_string(),
            }
            .into());
        }
        let mut values = self
            .residuals
            .iter()
            .map(|residual| Ok(ArrayBatch::replicated(residual.residual_value()?)))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        values.extend(inputs.iter().cloned());
        self.backward.interpret_with(
            values,
            |_, constant: &V| Ok(ArrayBatch::replicated(constant.clone())),
            |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
        )
    }
}

/// Capture-free forward-mode (JVP) rule for [`CustomVjpOperation`]: splices the user-supplied forward program
/// directly into the shared builder and stages one opaque [`CustomVjpTangentOperation`] carrier for the output
/// tangents.
///
/// Unlike [`CustomJvpOperation`], a `custom_vjp` function has no forward tangent program, so the forward cannot
/// compute the output tangents straight-line. Instead it reproduces — under the capture-free direct-transpose path —
/// the same structure the capture-based reverse rule builds: the forward program (already an ordinary primal-enum
/// program mapping `inputs -> (outputs..., residuals...)`) is replayed through
/// [`Program::interpret_in_context`](crate::Program::interpret_in_context) over the dual
/// primals, recovering the primal outputs and the residuals; then one [`CustomVjpTangentOperation`] is staged over
/// `[input_tangents..., residuals...]` with the residuals as ordinary *operands* (not capture factors). That carrier
/// is opaque: it stands for the unknown tangent map and rejects interpretation, so a forward-mode use through it
/// fails with the canonical reverse-only error, while [`transpose_primal_custom_vjp`] replays the user's `backward`
/// program to produce the input cotangents. Because the residuals flow as operand edges and the carrier is a leaf
/// primal-enum operation, the rule introduces no symbolic capture and needs no
/// [`DifferentiableProgramOperation`](crate::differentiation::DifferentiableProgramOperation) or
/// [`LinearizableProgramOperation`](crate::differentiation::LinearizableProgramOperation) witness.
impl<C: Context + Zero<C::Value>> DifferentiableOperation<C> for CustomVjpOperation<C::Constant, C::Operation>
where
    C::Constant: Clone,
    C::Operation: Clone + From<CustomVjpTangentOperation<C::Constant, C::Operation>>,
{
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let output_count = self.output_types().len();
        check_count!("input", inputs, self.input_types().len(), ProgramError);
        // Replay the forward program on the dual primals, recovering the primal outputs followed by the residuals.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut forward_outputs = self.forward().interpret_in_context(context, primal_operands)?;
        if forward_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "custom_vjp forward program produced {} outputs which is fewer than its {output_count} primal \
                 output(s)",
                forward_outputs.len(),
            ))
            .into());
        }
        let residuals = forward_outputs.split_off(output_count);
        let primal_outputs = forward_outputs;
        let residual_count = residuals.len();

        // Stage one opaque carrier over `[input_tangents..., residuals...]`, producing the output tangents. The
        // carrier rejects forward interpretation and transposes by replaying the user's backward program.
        // The opaque carrier takes every input tangent as a real operand, so materialize structural zeros.
        let mut carrier_operands = inputs
            .iter()
            .map(|input| input.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        carrier_operands.extend(residuals);
        let carrier = CustomVjpTangentOperation::new(self.backward.clone(), residual_count, false);
        let output_tangents = context.bind(carrier, &carrier_operands)?;
        check_count!("output", output_tangents, output_count, ProgramError);

        Ok(primal_outputs
            .into_iter()
            .zip(output_tangents)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect())
    }
}

/// Transpose rule for [`CustomVjpOperation`]: the call is a higher-order primal boundary rather than a linear map,
/// so a tangent program never contains it on a linear operand (linearization stages the opaque
/// [`CustomVjpTangentOperation`] carrier instead, which owns the executable transpose) and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V, O, W, OLinear> TransposableOperation<W, OLinear> for CustomVjpOperation<V, O>
where
    V: Value,
    W: Value<Type = V::Type>,
    O: Operation<V::Type>,
    OLinear: Operation<V::Type>,
{
    fn transpose(
        &self,
        _context: &mut TracingContext<W, OLinear>,
        _inputs: &[PartialValue<Tracer<TracingContext<W, OLinear>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<W, OLinear>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<W, OLinear>>>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("operation `{}` has no partition-aware transpose rule", self.name()),
        }
        .into())
    }
}

/// Opaque primal-enum carrier staged by [`CustomVjpOperation`]'s capture-free forward-mode rule.
///
/// This is the operand-form counterpart of [`CustomVjpCallOperation`]: it plays the same role under the
/// direct-transpose path (where tangent programs stay in the primal operation family `O` and carry their
/// residuals as ordinary program operands) that [`CustomVjpCallOperation`] plays under the capture-based reverse
/// path. The two differ only in how residuals reach the backward program: [`CustomVjpCallOperation`] closes them into
/// captured factors, whereas this carrier receives them as the trailing operands of the staged operation, after the
/// input tangents.
///
/// In its un-transposed form it stands for the (unknown) tangent map of the custom function and rejects
/// interpretation: `custom_vjp` functions are reverse-mode-only. Transposition (see [`transpose_primal_custom_vjp`])
/// reads the residual operands from the pullback and replays the user's backward program on them and the incoming
/// output cotangents, producing the input cotangents — so reverse mode uses exactly the user-supplied gradient.
#[derive(Clone, Debug)]
pub struct CustomVjpTangentOperation<V: Value, O> {
    /// The user's backward program, mapping `(residuals..., output_cotangents...)` to input cotangents.
    backward: Program<V, O, Vec<V>, Vec<V>>,

    /// Number of residual operands, used to split the backward program's inputs into the residual prefix and the
    /// output-cotangent suffix.
    residual_count: usize,

    /// Whether this carrier has been transposed into its executable (pullback) form.
    transposed: bool,
}

impl<V: Value, O> CustomVjpTangentOperation<V, O> {
    /// Creates a custom-VJP tangent carrier. Use `transposed = false` for the opaque pushforward form and
    /// `transposed = true` for the transposed pullback form.
    ///
    /// # Parameters
    ///
    ///   - `backward`: User backward program mapping `(residuals..., output_cotangents...)` to input cotangents.
    ///   - `residual_count`: Number of trailing residual operands carried alongside the tangents.
    ///   - `transposed`: Whether this carrier is in its transposed (pullback) form.
    pub fn new(backward: Program<V, O, Vec<V>, Vec<V>>, residual_count: usize, transposed: bool) -> Self {
        Self { backward, residual_count, transposed }
    }

    /// Returns the user's backward program.
    #[inline]
    pub fn backward(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.backward
    }

    /// Returns the number of trailing residual operands.
    #[inline]
    pub fn residual_count(&self) -> usize {
        self.residual_count
    }

    /// Returns whether this carrier is in its transposed (pullback) form.
    #[inline]
    pub fn transposed(&self) -> bool {
        self.transposed
    }
}

impl<V: Value, O: Operation<V::Type>> CustomVjpTangentOperation<V, O> {
    /// Returns the residual types carried as the trailing operands (the backward program's leading inputs).
    fn residual_types(&self) -> Vec<V::Type> {
        self.backward.input_types().into_iter().take(self.residual_count).collect()
    }

    /// Returns the cotangent types flowing *into* the backward program (one per primal output).
    fn cotangent_types(&self) -> Vec<V::Type> {
        self.backward.input_types().split_off(self.residual_count)
    }
}

impl<V: Value, O> Display for CustomVjpTangentOperation<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.transposed {
            formatter.write_str("custom_vjp_backward")
        } else {
            formatter.write_str("custom_vjp_tangent")
        }
    }
}

impl<V: Value, O: Operation<V::Type>> Operation<V::Type> for CustomVjpTangentOperation<V, O> {
    #[inline]
    fn name(&self) -> &'static str {
        if self.transposed { "custom_vjp_backward" } else { "custom_vjp_tangent" }
    }

    fn infer_output_types(&self, input_types: &[V::Type]) -> Result<Vec<V::Type>, TypeError> {
        if self.transposed {
            // The transposed (pullback) carrier maps `[output_cotangents..., residuals...]` to the input cotangents
            // (the backward program's outputs, typed like the primal inputs).
            let expected: Vec<V::Type> = self.cotangent_types().into_iter().chain(self.residual_types()).collect();
            check_types!("custom_vjp backward", &expected, input_types);
            Ok(self.backward.output_types())
        } else {
            // The un-transposed (tangent-map) carrier maps `[input_tangents..., residuals...]` to the output tangents.
            // The input tangents are typed like the primal inputs (the backward program's outputs).
            let expected: Vec<V::Type> =
                self.backward.output_types().into_iter().chain(self.residual_types()).collect();
            check_types!("custom_vjp tangent", &expected, input_types);
            Ok(self.cotangent_types())
        }
    }

    #[inline]
    fn effects(&self) -> Effects {
        self.backward.effects()
    }
}

/// Interprets a [`CustomVjpTangentOperation`] by rejecting forward-mode interpretation in both forms.
///
/// The un-transposed carrier is the opaque tangent map of a reverse-mode-only `custom_vjp`, so interpreting it is the
/// operation forward mode would need and is rejected. The transposed carrier is never interpreted as a staged
/// operation either: [`transpose_primal_custom_vjp`] replays the user's backward program directly through the
/// pullback builder rather than leaving a transposed carrier in the program, so reaching this interpret path means a
/// forward-mode use slipped through and the same reverse-only error applies.
impl<Constant, O, V, C> InterpretableOperation<V, C> for CustomVjpTangentOperation<Constant, O>
where
    Constant: Value,
    V: Value<Type = Constant::Type>,
    O: Operation<Constant::Type>,
{
    fn interpret(&self, _context: &C, _inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        Err(TypeError {
            message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                value_and_gradient, or jacrev) instead"
                .to_string(),
        }
        .into())
    }
}

/// Batching rule for [`CustomVjpTangentOperation`]: the opaque custom-VJP tangent carrier is a forward-mode tangent
/// map that never appears in a batched primal program — reverse mode consumes it during transposition rather than
/// batching it — so batching is rejected for every value and context.
impl<Constant, O, V, C> BatchableOperation<V, C> for CustomVjpTangentOperation<Constant, O>
where
    Constant: Value<Type = ArrayType>,
    O: Operation<ArrayType>,
    V: Value<Type = ArrayType>,
{
    fn batch(&self, _context: &C, _inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
        Err(BatchingError::UnsupportedOperation { message: format!("operation `{}` cannot be batched", self.name()) })
    }
}

/// Forward-mode (JVP) rule for [`CustomVjpTangentOperation`]: a `custom_vjp` function is reverse-mode only and has no
/// forward tangent program, so its tangent carrier rejects forward-mode linearization. The carrier never reaches this
/// rule on the supported path — the [`CustomVjpOperation`] JVP rule stages it on the tangent side — but the rule is
/// implemented so the enum dispatch can forward to it uniformly.
impl<C: Context> DifferentiableOperation<C> for CustomVjpTangentOperation<C::Constant, C::Operation>
where
    C::Operation: Clone,
{
    fn jvp(
        &self,
        _context: &C,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("{} has no forward-mode (jvp) rule; custom_vjp is reverse-mode only", self.name()),
        }
        .into())
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`] for a
/// [`CustomVjpTangentOperation`]; the opaque carrier never has all-known operands in practice (its tangent operands
/// are unknown by construction), so it residualizes unchanged.
impl<V: Value, O: Clone + Operation<V::Type>, C: Context<Type = V::Type>> PartiallyEvaluatableOperation<C>
    for CustomVjpTangentOperation<V, O>
where
    C::Operation: From<CustomVjpTangentOperation<V, O>>,
{
}

/// Partition-aware transpose rule for an opaque [`CustomVjpTangentOperation`], used by the direct reverse path
/// when it transposes a tangent program in the primal operation family `O` rather than re-keying it into the linear
/// family. This is the operand-form counterpart of the [`CustomVjpCallOperation`] transpose rule: the residuals are
/// ordinary *operands* (known values supplied through `operand_values`) instead of capture factors, so the rule reads
/// them from the pullback and replays the user's backward program forward into the pullback builder.
///
/// The forward stages the carrier over `[input_tangents(linear)..., residuals(known)...]` with the tangents
/// marked linear and the residuals marked known. This rule therefore:
///
///   1. Splits the operands by `operand_linear` into the leading linear run of input tangents and the trailing known
///      residuals, reading the residual values from the pullback through `operand_values`.
///   2. Stages the output cotangents (materializing structural zeros so the backward program receives every
///      cotangent input).
///   3. Replays the user's `backward` program over `[residuals..., output_cotangents...]` through
///      [`Program::interpret_in_context`](crate::Program::interpret_in_context), producing the input cotangents. The backward program is
///      *not* transposed — it already
///      is the pullback — so it is replayed forward into the active pullback builder.
///
/// The returned cotangents place those input cotangents at the linear (tangent) operand positions and a structural
/// [`MaybeZero::Zero`] at the known residual positions, which carry no cotangent. Because the backward program is
/// replayed in the same operation family `O` through the context's [`bind`](Context::bind), the rule is value-level
/// and introduces no recursive transposition obligation on `O`.
///
/// # Parameters
///
///   - `operation`: Opaque custom-VJP tangent carrier staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-operand [`PartialValue`] knowledge. The [`Unknown`](PartialValue::Unknown) entries are the input
///     tangents; the [`Known`](PartialValue::Known) entries carry the residual tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the carrier's outputs (one per primal output).
pub fn transpose_primal_custom_vjp<V, O>(
    operation: &CustomVjpTangentOperation<V, O>,
    context: &mut TracingContext<V, O>,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>
where
    V: Value,
    O: Clone + Operation<V::Type> + MaybeZeroOperation<V::Type> + From<ZeroOperation<V::Type>>,
{
    if operation.transposed {
        return Err(TypeError {
            message: "transposing a custom_vjp pullback (second-order reverse mode through custom_vjp) is not \
                supported"
                .to_string(),
        }
        .into());
    }

    // Operand layout is `[input_tangents(linear)..., residuals(known)...]`. The input tangents are exactly the linear
    // operands, and the residuals are the trailing known operands read from the pullback. The dispatch guarantees a
    // `Known` operand carries its pullback value, so each residual tracer is read directly.
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let tangent_count = operand_linear.iter().filter(|&&linear| linear).count();
    let residual_count = operation.residual_count;
    check_count!("input", operand_linear, tangent_count + residual_count, ProgramError);
    let residuals = (tangent_count..inputs.len())
        .map(|index| {
            inputs[index]
                .as_known()
                .ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "custom_vjp tangent transpose operand {index} has no known residual value"
                    ))
                })
                .cloned()
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;

    // Stage the output cotangents, then replay the user's backward program on `[residuals..., outputs...]`.
    // The backward program is the pullback already, so it is replayed forward into the pullback builder rather than
    // transposed.
    let cotangent_types = operation.cotangent_types();
    check_count!("output", outputs, cotangent_types.len(), ProgramError);
    let mut backward_inputs = residuals;
    for cotangent in outputs {
        backward_inputs.push(cotangent.clone().materialize(context)?);
    }
    let input_cotangents = operation.backward().interpret_in_context(context, backward_inputs)?;
    check_count!("output", input_cotangents, tangent_count, ProgramError);

    // The user's backward program is an opaque splice: its outputs come back as plain replayed values, so any
    // structural zero-ness a user backward expresses (a `zero`/`zero_like` output for a non-differentiated input)
    // would otherwise be lost at this boundary and stage wasted adjoint work upstream. Recover it here with one local
    // pass over the *backward program* itself: an output produced by a nullary canonical zero instruction is a
    // structural zero. This is the reverse-mode analogue of JAX's `custom_vjp` symbolic zeros, recovered
    // automatically instead of through an opt-in.
    let backward = operation.backward();
    let output_is_zero = backward
        .output_ids()
        .iter()
        .map(|output| {
            backward.instructions().iter().any(|instruction| {
                instruction.outputs().contains(output)
                    && instruction.inputs().is_empty()
                    && instruction.operation().is_zero_operation()
            })
        })
        .collect::<Vec<_>>();

    // Reassemble one cotangent per operand: the residuals carry structural zeros, while the input tangents receive the
    // backward program's outputs in order (recovered as structural zeros where the backward emitted canonical zeros).
    let mut input_cotangents = input_cotangents.into_iter().zip(output_is_zero).map(|(cotangent, is_zero)| {
        if is_zero { MaybeZero::Zero(cotangent.r#type().into_owned()) } else { MaybeZero::Value(cotangent) }
    });
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

/// Transpose rule for the opaque [`CustomVjpTangentOperation`] carrier, forwarding to
/// [`transpose_primal_custom_vjp`]. The recursion stays value-level (the user's backward program is replayed forward
/// into the pullback builder through [`Context::bind`]), so instantiating this implementation for a closed operation
/// enum introduces no recursive [`TransposableOperation`] obligation on `O`.
impl<V, O> TransposableOperation<V, O> for CustomVjpTangentOperation<V, O>
where
    V: Value,
    O: Clone + Operation<V::Type> + MaybeZeroOperation<V::Type> + From<ZeroOperation<V::Type>>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        transpose_primal_custom_vjp(self, context, inputs, outputs).map_err(DifferentiationError::from)
    }
}

/// Function with a user-supplied JVP rule — the ergonomic analogue of JAX's
/// [`jax.custom_jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html) /
/// [`defjvp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.defjvp.html) decorator pair, built by
/// [`custom_jvp`].
///
/// # When to use
///
/// Reach for a custom JVP when the function *is* forward-differentiable but its automatically derived tangent is
/// numerically unstable or wasteful and you want to supply a stable, efficient one by hand — classic cases are a
/// `log`-`sum`-`exp`, a softmax, or a normalization, where a hand-written tangent avoids the cancellation or
/// redundant work the generic rule incurs. A single custom JVP serves **both** differentiation modes: reverse mode
/// obtains its gradient by transposing the supplied tangent map, so the one rule composes with forward mode, reverse
/// mode, and their higher-order combinations. Prefer it over [`CustomVjp`] whenever the function is naturally
/// forward-differentiable, and reach for [`CustomVjp`] only when just the reverse rule is natural (for example
/// implicit differentiation or adjoint solvers).
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
pub struct CustomJvp<D: Domain, P, J, IT, OT> {
    /// Closure computing the primal output tree from the primal input tree.
    primal: P,

    /// Closure computing `(outputs, output_tangents)` from `(inputs, input_tangents)`.
    jvp: J,

    /// Phantom marker pinning the [`Domain`] and the input and output tracer-tree types named by the closure
    /// signatures. The domain is a pure type witness, so no domain value is stored.
    marker: PhantomData<fn() -> (D, IT, OT)>,
}

/// Creates a [`CustomJvp`] function from a primal closure and a JVP-rule closure over trees of the [`Domain`] `D`'s
/// tracers. Refer to the documentation of [`CustomJvp`] for the calling convention and tracing semantics.
pub fn custom_jvp<D, P, J, IT, OT>(primal: P, jvp: J) -> CustomJvp<D, P, J, IT, OT>
where
    D: Domain,
    P: Fn(IT) -> Result<OT, ProgramError>,
    J: Fn(IT, IT) -> Result<(OT, OT), ProgramError>,
{
    CustomJvp { primal, jvp, marker: PhantomData }
}

impl<D, P, J, IT, OT> CustomJvp<D, P, J, IT, OT>
where
    D: Domain<Type: PartialEq>,
    P: Fn(IT) -> Result<OT, ProgramError>,
    J: Fn(IT, IT) -> Result<(OT, OT), ProgramError>,
    D::Operation: Clone + Operation<D::Type> + From<CustomJvpOperation<D::Constant, D::Operation>>,
    IT: Parameterized<DomainTracer<D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    OT: Parameterized<DomainTracer<D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    IT::To<D::Type>: Clone
        + Parameterized<D::Type, Family = IT::Family, To<DomainTracer<D>> = IT, To<D::Constant> = IT::To<D::Constant>>,
    OT::To<D::Type>:
        Parameterized<D::Type, Family = OT::Family, To<DomainTracer<D>> = OT, To<D::Constant> = OT::To<D::Constant>>,
{
    /// Stages this custom-JVP function on the provided tracer input tree and returns its output tree, tracing the
    /// stored closures into programs specialized to the input types. Differentiation of the staged call replays the
    /// JVP rule instead of differentiating the primal body, in both forward and reverse mode.
    pub fn call<V, ICT>(&self, input: ICT) -> Result<<OT::To<D::Type> as Parameterized<D::Type>>::To<V>, ProgramError>
    where
        V: Value<Type = D::Type>,
        V::DispatchDomain: Context<Type = D::Type, Constant = D::Constant, Operation = D::Operation>,
        IT::Family: ParameterizedFamily<V>,
        OT::Family: ParameterizedFamily<V>,
        ICT: Parameterized<V, Family = IT::Family, To<D::Type> = IT::To<D::Type>>,
        <OT::To<D::Type> as Parameterized<D::Type>>::To<V>: Parameterized<
                V,
                Family = OT::Family,
                ParameterStructure = <OT::To<D::Type> as Parameterized<D::Type>>::ParameterStructure,
            >,
    {
        let mut input_values = Vec::new();
        let input_types = input
            .map_parameters(|value| {
                let r#type = value.r#type().into_owned();
                input_values.push(value);
                r#type
            })
            .map_err(ProgramError::from)?;
        let Some(first) = input_values.first() else {
            return Err(TypeError { message: "custom_jvp requires at least one input".to_string() }.into());
        };
        let (_, primal) = D::trace(|xs| (self.primal)(xs), input_types.clone())?;
        let (output_types, jvp) = D::trace(|(x, t)| (self.jvp)(x, t), (input_types.clone(), input_types))?;
        let operation = D::Operation::from(CustomJvpOperation::new(primal.to_flat_program(), jvp.to_flat_program())?);
        // The call binds through whatever context the input values flow (a staged trace, a batching context, or a
        // JVP context), so `custom_jvp` composes under `vmap`/`jvp` — the batch/JVP rule of the bound operation fires.
        let context = first.dispatch_domain();
        let outputs = context.bind(operation, &input_values)?;
        let output_structure = output_types.0.parameter_structure();
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

/// Function with user-supplied forward/backward (VJP) rules — the ergonomic analogue of JAX's
/// [`jax.custom_vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) /
/// [`defvjp`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html) decorator pair, built by
/// [`custom_vjp`].
///
/// # When to use
///
/// Reach for a custom VJP when only the *reverse* rule is natural, or when the function is not (efficiently)
/// forward-differentiable. Common cases:
///
///   - **Implicit differentiation** — differentiate through a solver, optimizer, or fixed point via the implicit
///     function theorem rather than unrolling its iterations.
///   - **Adjoint methods** — backpropagate through an ODE or PDE solve via the adjoint system instead of
///     differentiating the integrator's individual steps.
///   - **External or black-box calls** — supply the reverse rule for a custom kernel or a computation that does not
///     itself trace into `ryft` programs.
///   - **Numerical stability** — replace an unstable or wasteful automatically derived gradient with a hand-written
///     one.
///
/// A custom VJP is reverse-mode only (forward-mode differentiation of a staged call is rejected, as detailed below),
/// so second-order derivatives are reachable via reverse-over-reverse — the `backward` program is itself
/// differentiated — but not via forward-over-reverse. When the function is forward-differentiable and you want a
/// single rule that serves both modes, use [`CustomJvp`] instead.
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
pub struct CustomVjp<D: Domain, P, F, B, IT, OT, RT> {
    /// Closure computing the primal output tree from the primal input tree.
    primal: P,

    /// Closure computing `(outputs, residuals)` from the primal input tree.
    forward: F,

    /// Closure computing the input cotangent tree from `(residuals, output_cotangents)`.
    backward: B,

    /// Phantom marker pinning the [`Domain`] and the input, output, and residual tracer-tree types named by the
    /// closure signatures. The domain is a pure type witness, so no domain value is stored.
    marker: PhantomData<fn() -> (D, IT, OT, RT)>,
}

/// Creates a [`CustomVjp`] function from primal, forward, and backward closures over trees of the [`Domain`] `D`'s
/// tracers. Refer to the documentation of [`CustomVjp`] for the calling convention and tracing semantics.
pub fn custom_vjp<D, P, F, B, IT, OT, RT>(primal: P, forward: F, backward: B) -> CustomVjp<D, P, F, B, IT, OT, RT>
where
    D: Domain,
    P: Fn(IT) -> Result<OT, ProgramError>,
    F: Fn(IT) -> Result<(OT, RT), ProgramError>,
    B: Fn(RT, OT) -> Result<IT, ProgramError>,
{
    CustomVjp { primal, forward, backward, marker: PhantomData }
}

impl<D, P, F, B, IT, OT, RT> CustomVjp<D, P, F, B, IT, OT, RT>
where
    D: Domain<Type: PartialEq>,
    P: Fn(IT) -> Result<OT, ProgramError>,
    F: Fn(IT) -> Result<(OT, RT), ProgramError>,
    B: Fn(RT, OT) -> Result<IT, ProgramError>,
    D::Operation: Clone + Operation<D::Type> + From<CustomVjpOperation<D::Constant, D::Operation>>,
    IT: Parameterized<DomainTracer<D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    OT: Parameterized<DomainTracer<D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    RT: Parameterized<DomainTracer<D>, Family: ParameterizedFamily<D::Type> + ParameterizedFamily<D::Constant>>,
    IT::To<D::Type>: Clone
        + Parameterized<D::Type, Family = IT::Family, To<DomainTracer<D>> = IT, To<D::Constant> = IT::To<D::Constant>>,
    OT::To<D::Type>: Clone
        + Parameterized<D::Type, Family = OT::Family, To<DomainTracer<D>> = OT, To<D::Constant> = OT::To<D::Constant>>,
    RT::To<D::Type>:
        Parameterized<D::Type, Family = RT::Family, To<DomainTracer<D>> = RT, To<D::Constant> = RT::To<D::Constant>>,
{
    /// Stages this custom-VJP function on the provided tracer input tree and returns its output tree, tracing the
    /// stored closures into programs specialized to the input types. Reverse-mode differentiation of the staged
    /// call replays the backward rule on the forward rule's residuals instead of differentiating the primal body.
    pub fn call<V, ICT>(&self, input: ICT) -> Result<<OT::To<D::Type> as Parameterized<D::Type>>::To<V>, ProgramError>
    where
        V: Value<Type = D::Type>,
        V::DispatchDomain: Context<Type = D::Type, Constant = D::Constant, Operation = D::Operation>,
        IT::Family: ParameterizedFamily<V>,
        OT::Family: ParameterizedFamily<V>,
        ICT: Parameterized<V, Family = IT::Family, To<D::Type> = IT::To<D::Type>>,
        <OT::To<D::Type> as Parameterized<D::Type>>::To<V>: Parameterized<
                V,
                Family = OT::Family,
                ParameterStructure = <OT::To<D::Type> as Parameterized<D::Type>>::ParameterStructure,
            >,
    {
        let mut input_values = Vec::new();
        let input_types = input
            .map_parameters(|value| {
                let r#type = value.r#type().into_owned();
                input_values.push(value);
                r#type
            })
            .map_err(ProgramError::from)?;
        let Some(first) = input_values.first() else {
            return Err(TypeError { message: "custom_vjp requires at least one input".to_string() }.into());
        };
        let (output_types, primal) = D::trace(|xs| (self.primal)(xs), input_types.clone())?;
        let (forward_output_types, forward) = D::trace(|xs| (self.forward)(xs), input_types.clone())?;
        let (_, residual_types) = forward_output_types;
        let (_, backward) = D::trace(
            |(residuals, cotangents)| (self.backward)(residuals, cotangents),
            (residual_types, output_types.clone()),
        )?;
        let operation = D::Operation::from(CustomVjpOperation::new(
            primal.to_flat_program(),
            forward.to_flat_program(),
            backward.to_flat_program(),
        )?);
        // Bind through whatever context the inputs flow, so `custom_vjp` composes under `vmap`/`jvp`.
        let context = first.dispatch_domain();
        let outputs = context.bind(operation, &input_values)?;
        let output_structure = output_types.parameter_structure();
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::backends::scalars::Scalar;
    use crate::backends::scalars::ScalarOperation;
    use crate::batching::{Batch, BatchAxis};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::operations::math::MulOperation;
    use crate::operations::math::{Cos, CosOperation, Sin, SinOperation};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
    use crate::types::{DataType, Shape, Size};

    use super::*;

    /// Returns the canonical test array type with the provided dimensions.
    fn test_type(dimensions: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(dimensions.iter().map(|dimension| Size::Static(*dimension)).collect()))
    }

    /// Builds `f(x) = sin(x)` over one input of the provided type.
    fn sin_program(
        r#type: &ArrayType,
    ) -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(r#type.clone());
        let output = builder.add_instruction(SinOperation, vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong rule `jvp(x, dx) = (sin(x), 2 * cos(x) * dx)`, detectably different from the
    /// true derivative so tests can prove the custom rule is used.
    fn doubled_sin_jvp_program(
        r#type: &ArrayType,
    ) -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
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
    ) -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
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
    ) -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(r#type.clone());
        let cotangent = builder.add_input(r#type.clone());
        let three = builder.add_constant(TestArray::scalar(3.0));
        let scaled = builder.add_instruction(MulOperation, vec![three, residual]).unwrap()[0];
        let gradient = builder.add_instruction(MulOperation, vec![scaled, cotangent]).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    fn custom_jvp_sin(r#type: &ArrayType) -> ArrayOperation<TestArray> {
        ArrayOperation::CustomJvp(Box::new(
            CustomJvpOperation::new(sin_program(r#type), doubled_sin_jvp_program(r#type)).unwrap(),
        ))
    }

    fn custom_vjp_sin(r#type: &ArrayType) -> ArrayOperation<TestArray> {
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
    fn test_custom_derivative_calls_remain_opaque_to_partial_evaluation() {
        let scalar = test_type(&[]);
        for operation in [custom_jvp_sin(&scalar), custom_vjp_sin(&scalar)] {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let input = builder.add_input(scalar.clone());
            let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
            let program = builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();

            let evaluation = program.partially_evaluate(&[PartialValue::Unknown(scalar.clone())]).unwrap();

            assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
            assert_eq!(evaluation.program.instructions().len(), 1);
            assert!(matches!(
                evaluation.program.instructions()[0].operation(),
                ArrayOperation::CustomJvp(_) | ArrayOperation::CustomVjp(_),
            ));
        }
    }

    #[test]
    fn test_custom_vjp_tangent_carrier_remains_opaque_to_partial_evaluation() {
        let scalar = test_type(&[]);
        let operation = ArrayOperation::CustomVjpTangent(Box::new(CustomVjpTangentOperation::new(
            tripled_sin_backward_program(&scalar),
            1,
            false,
        )));
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let tangent = builder.add_input(scalar.clone());
        let residual = builder.add_input(scalar.clone());
        let output = builder.add_instruction(operation, vec![tangent, residual]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(scalar.clone()), PartialValue::Unknown(scalar)])
            .unwrap();

        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::CustomVjpTangent(_)));
    }

    #[test]
    fn test_custom_jvp_interprets_the_primal_program() {
        let scalar = test_type(&[]);
        let outputs = custom_jvp_sin(&scalar)
            .interpret(&crate::EagerContext::<TestArray>::new(), &[TestArray::scalar(2.0)])
            .unwrap();
        assert_abs_diff_eq!(outputs[0].values[0], 2.0f64.sin(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_governs_forward_mode() {
        let scalar = test_type(&[]);
        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                |x| {
                    let operation = custom_jvp_sin(&test_type(&[]));
                    Ok(x.context().bind(operation, &[x.clone()])?.into_iter().next().unwrap())
                },
                TestArray::scalar(2.0),
                TestArray::scalar(1.0),
            )
            .unwrap();
        let _ = scalar;
        assert_abs_diff_eq!(primal.values[0], 2.0f64.sin(), epsilon = 1e-9);
        // The custom rule doubles the true derivative, proving it is in control.
        assert_abs_diff_eq!(tangent.values[0], 2.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_governs_reverse_mode() {
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                |x| {
                    let operation = custom_jvp_sin(&test_type(&[]));
                    x.context().bind(operation, &[x.clone()]).unwrap().into_iter().next().unwrap()
                },
                TestArray::scalar(3.0),
            )
            .unwrap();
        assert_abs_diff_eq!(value.values[0], 3.0f64.sin(), epsilon = 1e-9);
        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        assert_abs_diff_eq!(gradient.values[0], 2.0 * 3.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_governs_reverse_mode() {
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                |x| {
                    let operation = custom_vjp_sin(&test_type(&[]));
                    x.context().bind(operation, &[x.clone()]).unwrap().into_iter().next().unwrap()
                },
                TestArray::scalar(2.0),
            )
            .unwrap();
        assert_abs_diff_eq!(value.values[0], 2.0f64.sin(), epsilon = 1e-9);
        // The custom backward rule triples the true gradient, proving it is in control.
        assert_abs_diff_eq!(gradient.values[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_rejects_forward_mode() {
        // The staged linear call refuses interpretation in its un-transposed (pushforward) form, which is exactly
        // the operation `jvp` would need to execute; reverse mode transposes it first and replays `backward`.
        let scalar = test_type(&[]);
        let call = CustomVjpCallOperation::<TestArray, ArrayOperation<TestArray>, TestArray>::new(
            tripled_sin_backward_program(&scalar),
            vec![TestArray::scalar(2.0f64.cos())],
            false,
        );
        assert!(matches!(
            call.interpret(&crate::EagerContext::<TestArray>::new(), &[TestArray::scalar(1.0)]),
            Err(ProgramError::Type(TypeError { message }))
                if message.starts_with("custom_vjp does not support forward-mode differentiation"),
        ));
    }

    #[test]
    fn test_jacrev_through_custom_vjp_uses_the_custom_backward_rule() {
        use crate::tracing_v2::jacrev;

        // jacrev interprets the pullback with batch-stacked cotangent bases, exercising the batched replay of the
        // custom backward program. The Jacobian of elementwise `sin` with the tripled rule is the diagonal matrix
        // `diag(3 * cos(x))`.
        let vector = test_type(&[2]);
        let jacobian = jacrev(
            &EagerContext::<TestArray, ArrayOperation<TestArray>>::new(),
            |x| {
                let operation = custom_vjp_sin(&test_type(&[2]));
                Ok(x.context().bind(operation, &[x.clone()])?.into_iter().next().unwrap())
            },
            TestArray::new(vector, vec![0.5, 1.0]),
        )
        .unwrap();
        let (_, _, block) = jacobian.iter_blocks().next().unwrap();
        assert_abs_diff_eq!(block.value().values()[0], 3.0 * 0.5f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[2], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[3], 3.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    /// Builds the scalar `f(x) = sin(x)` program.
    fn scalar_sin_program() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(SinOperation, vec![input]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong scalar rule `jvp(x, dx) = (sin(x), 2 * cos(x) * dx)`.
    fn scalar_doubled_sin_jvp_program() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(DataType::F64);
        let dx = builder.add_input(DataType::F64);
        let y = builder.add_instruction(SinOperation, vec![x]).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation, vec![x]).unwrap()[0];
        let two = builder.add_constant(Scalar::from(2.0));
        let scaled = builder.add_instruction(MulOperation, vec![two, cosine]).unwrap()[0];
        let tangent = builder.add_instruction(MulOperation, vec![scaled, dx]).unwrap()[0];
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the malformed rule `jvp(x, dx) = (sin(x), 1)`. Its tangent ignores `dx` and is therefore an affine
    /// constant rather than a linear tangent map.
    fn scalar_known_tangent_jvp_program() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(DataType::F64);
        builder.add_input(DataType::F64);
        let y = builder.add_instruction(SinOperation, vec![x]).unwrap()[0];
        let tangent = builder.add_constant(Scalar::from(1.0));
        builder
            .build(vec![y, tangent], vec![Placeholder, Placeholder], vec![Placeholder, Placeholder])
            .unwrap()
    }

    /// Builds the scalar forward rule `forward(x) = (sin(x), cos(x))`, with the cosine as the residual.
    fn scalar_sin_forward_program() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::new();
        let x = builder.add_input(DataType::F64);
        let y = builder.add_instruction(SinOperation, vec![x]).unwrap()[0];
        let residual = builder.add_instruction(CosOperation, vec![x]).unwrap()[0];
        builder.build(vec![y, residual], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap()
    }

    /// Builds the deliberately wrong scalar rule `backward(residual, cotangent) = 3 * residual * cotangent`.
    fn scalar_tripled_sin_backward_program() -> Program<Scalar, ScalarOperation<Scalar>, Vec<Scalar>, Vec<Scalar>> {
        let mut builder = ProgramBuilder::new();
        let residual = builder.add_input(DataType::F64);
        let cotangent = builder.add_input(DataType::F64);
        let three = builder.add_constant(Scalar::from(3.0));
        let scaled = builder.add_instruction(MulOperation, vec![three, residual]).unwrap()[0];
        let gradient = builder.add_instruction(MulOperation, vec![scaled, cotangent]).unwrap()[0];
        builder.build(vec![gradient], vec![Placeholder, Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_scalar_custom_jvp_governs_forward_mode() {
        let (primal, tangent) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .jvp(
                |x| {
                    let operation = ScalarOperation::CustomJvp(Box::new(
                        CustomJvpOperation::new(scalar_sin_program(), scalar_doubled_sin_jvp_program()).unwrap(),
                    ));
                    Ok(x.context().bind(operation, &[x.clone()])?.into_iter().next().unwrap())
                },
                Scalar::from(2.0),
                Scalar::from(1.0),
            )
            .unwrap();
        assert_abs_diff_eq!(primal, 2.0f64.sin(), epsilon = 1e-9);
        // The custom rule doubles the true derivative, proving it is in control.
        assert_abs_diff_eq!(tangent, 2.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_linearization_rejects_known_custom_jvp_tangents() {
        let operation = ScalarOperation::CustomJvp(Box::new(
            CustomJvpOperation::new(scalar_sin_program(), scalar_known_tangent_jvp_program()).unwrap(),
        ));
        let expected = "linearization produced a known tangent output; differentiation rules must represent \
                        input-independent zero tangents structurally";

        // Program-level direct linearization must reject the malformed rule rather than silently replacing its
        // constant tangent with zero.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(operation.clone(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            program.linearize(),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message))) if message == expected,
        ));

        // Value-level direct linearization enforces the same rule contract before exposing a reusable pushforward.
        let result = EagerContext::<Scalar, ScalarOperation<Scalar>>::new().linearize(
            |input| {
                let mut outputs = input.context().bind(operation, &[input.clone()])?;
                Ok(outputs.remove(0))
            },
            Scalar::from(2.0),
        );
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message))) if message == expected,
        ));
    }

    #[test]
    fn test_scalar_custom_vjp_governs_reverse_mode() {
        let (value, gradient) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .value_and_gradient(
                |x| {
                    let operation = ScalarOperation::CustomVjp(Box::new(
                        CustomVjpOperation::new(
                            scalar_sin_program(),
                            scalar_sin_forward_program(),
                            scalar_tripled_sin_backward_program(),
                        )
                        .unwrap(),
                    ));
                    x.context().bind(operation, &[x.clone()]).unwrap().into_iter().next().unwrap()
                },
                Scalar::from(2.0),
            )
            .unwrap();
        assert_abs_diff_eq!(value, 2.0f64.sin(), epsilon = 1e-9);
        // The custom backward rule triples the true gradient, proving it is in control.
        assert_abs_diff_eq!(gradient, 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_wrapper_traces_closures_lazily() {
        // No manual programs: the wrapper traces the closures at the call site, specialized to the input types.
        let function = custom_jvp::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, dx| {
                // The deliberately wrong rule `jvp(x, dx) = (sin(x), cos(x) * dx + cos(x) * dx)` doubles the true
                // derivative (expressed through addition to avoid constant lifting), proving the rule is in control.
                let tangent = x.cos()? * dx;
                Ok((x.sin()?, tangent.clone() + tangent))
            },
        );
        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(|x| function.call(x), TestArray::scalar(2.0), TestArray::scalar(1.0))
            .unwrap();
        assert_abs_diff_eq!(primal.values[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.values[0], 2.0 * 2.0f64.cos(), epsilon = 1e-9);
        // Reverse mode transposes the linearized custom rule, so the doubled derivative carries over.
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(|x| function.call(x).unwrap(), TestArray::scalar(3.0))
            .unwrap();
        assert_abs_diff_eq!(value.values[0], 3.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[0], 2.0 * 3.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_wrapper_governs_reverse_mode() {
        let function = custom_vjp::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| {
                // The deliberately wrong rule `backward(residual, cotangent) = 3 * residual * cotangent` triples the
                // true gradient (expressed through addition to avoid constant lifting).
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(|x| function.call(x).unwrap(), TestArray::scalar(2.0))
            .unwrap();
        assert_abs_diff_eq!(value.values[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_wrapper_supports_structured_signatures_and_captured_configuration() {
        // Tuple inputs and tuple residuals exercise the `Parameterized` (pytree) calling convention, and the
        // captured `triple` closure plays the role of a JAX `nondiff_argnums` argument: static configuration
        // visible to the rule closures without being differentiated or stored as a residual.
        let repeats = 3usize;
        let function = custom_vjp::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _, _, _, _>(
            |(x, y): (
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
            )| Ok(x * y),
            |(x, y): (
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
            )| { Ok((x.clone() * y.clone(), (x, y))) },
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
        let (value, (gradient_x, gradient_y)) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                |(x, y)| function.call((x, y)).unwrap(),
                (TestArray::scalar(2.0), TestArray::scalar(5.0)),
            )
            .unwrap();
        assert_abs_diff_eq!(value.values[0], 10.0, epsilon = 1e-9);
        // The custom rule triples the true gradients `(y, x)`.
        assert_abs_diff_eq!(gradient_x.values[0], 3.0 * 5.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient_y.values[0], 3.0 * 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_scalar_custom_vjp_wrapper_governs_reverse_mode() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let function = custom_vjp::<EagerContext<Scalar, ScalarOperation<Scalar>>, _, _, _, _, _, _>(
            |x: DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| {
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let (value, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), Scalar::from(2.0)).unwrap();
        assert_abs_diff_eq!(value, 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient, 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_jvp_wrapper_surfaces_rule_signature_mismatches() {
        // Arity mismatches are compile-time errors under the structured signatures, but shape mismatches remain
        // runtime concerns: this rule produces a scalar tangent for a vector-valued function, so the traced JVP
        // program fails the signature validation that `CustomJvpOperation::new` performs at the call site.
        let function = custom_jvp::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, dx| {
                Ok((x.sin()?, dx.dot(&dx, &DotDimensionNumbers::inner_product())))
            },
        );
        let error = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), test_type(&[2]))
            .unwrap_err();
        assert!(error.to_string().contains("custom_jvp rule output"));
    }

    #[test]
    fn test_custom_jvp_batches_by_rewrapping_the_call() {
        let scalar = test_type(&[]);
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |x| {
                    let operation = custom_jvp_sin(&scalar);
                    Ok(x.context().bind(operation, &[x.clone()])?.into_iter().next().unwrap())
                },
                TestArray::vector(vec![0.5, 1.0, 1.5]),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        for (actual, input) in output.values.iter().zip([0.5f64, 1.0, 1.5]) {
            assert_abs_diff_eq!(*actual, input.sin(), epsilon = 1e-9);
        }
    }

    #[test]
    fn test_custom_jvp_survives_batching_and_governs_the_batched_gradient() {
        use crate::batching::Batch;
        use crate::differentiation::LinearizationTracer;
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};

        // Differentiating *through* a batch of the custom call must still use the (deliberately doubled) custom
        // rule: traced batching re-wraps the call around batched programs instead of inlining the primal, so the
        // custom derivative survives `batch` — mirroring JAX's `vmap`-of-`custom_jvp` semantics.
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let mapped: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>> = Batch::batch(
                        &context,
                        |item| {
                            let operation = custom_jvp_sin(&test_type(&[]));
                            Ok(item.context().bind(operation, &[item.clone()])?.into_iter().next().unwrap())
                        },
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                TestArray::vector(vec![0.5, 1.0]),
            )
            .unwrap();
        assert_abs_diff_eq!(value.values[0], 0.5f64.sin() + 1.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[0], 2.0 * 0.5f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[1], 2.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_survives_batching_and_governs_the_batched_gradient() {
        use crate::batching::Batch;
        use crate::differentiation::LinearizationTracer;
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};

        // The reverse-mode analogue of the test above: the (deliberately tripled) custom backward rule governs the
        // gradient through the batched call — mirroring JAX's `vmap`-of-`custom_vjp` semantics.
        let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let mapped: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>> = Batch::batch(
                        &context,
                        |item| {
                            let operation = custom_vjp_sin(&test_type(&[]));
                            Ok(item.context().bind(operation, &[item.clone()])?.into_iter().next().unwrap())
                        },
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                TestArray::vector(vec![0.5, 1.0]),
            )
            .unwrap();
        assert_abs_diff_eq!(value.values[0], 0.5f64.sin() + 1.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[0], 3.0 * 0.5f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[1], 3.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_user_custom_vjp_pullback_applies_the_user_backward_program() {
        // First-order reverse mode through a user custom VJP applies the user-supplied backward program. The
        // reverse entry stages an opaque tangent carrier and the direct transpose replays the backward program forward
        // into the pullback, so seeding the pullback at `[cotangent ++ residuals]` recovers `residual * cotangent`. The
        // user backward defines the residual as `cos(x)`, so at `x = 0.7` and a unit cotangent the input cotangent is
        // `cos(0.7)`.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let function = custom_vjp::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| Ok(residual * cotangent),
        );
        let (_, pullback) = domain.vjp(|x| function.call(x), TestArray::scalar(0.7)).unwrap();
        let (pullback, residuals) = pullback.into_parts();
        let mut pullback_inputs = vec![TestArray::scalar(1.0)];
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret(pullback_inputs).unwrap();
        assert_abs_diff_eq!(input_cotangents[0].values[0], 0.7f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_custom_vjp_batching_broadcasts_replicated_inputs() {
        // Mapping only the first input exercises the replicated broadcast in the re-wrapping batch rule: the
        // unmapped operand is broadcast into the batch (the all-inputs-mapped-at-0 convention) and the batched call
        // still computes per-item products.
        let function = custom_vjp::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _, _, _, _>(
            |(x, y): (
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
            )| Ok(x * y),
            |(x, y): (
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
            )| { Ok((x.clone() * y.clone(), (x, y))) },
            |(x, y), cotangent| Ok((y * cotangent.clone(), x * cotangent)),
        );
        let output: TestArray = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .batch(
                |(x, y)| function.call((x, y)),
                (TestArray::vector(vec![2.0, 3.0, 4.0]), TestArray::scalar(5.0)),
                (BatchAxis::new(0), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
        assert_eq!(output.values, vec![10.0, 15.0, 20.0]);
    }
}
