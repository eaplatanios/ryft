use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDual, DifferentiationError, DifferentiationTracer,
    LinearizationTracer, TransposableOperation,
};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::{OneOperation, Zero, ZeroOperation};
use crate::parameters::{ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationInput, PartialEvaluationValue, PartialTracer, PartialValue,
    PartiallyEvaluatableOperation,
};
use crate::programs::{Atom, AtomId, Instruction, MaybeZero, Program, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{Type, Typed};

/// Extension trait carrying the forward- and reverse-mode differentiation transforms on every [`Context`], mirroring
/// how [`Batch`](crate::batching::Batch) carries batching.
///
/// This trait is blanket-implemented for all [`Context`]s and has no items of its own to implement: every entry
/// point is a defaulted method whose `where` clause carries its actual requirements (the operation family's
/// [`DifferentiableOperation`] rules, transposability for reverse mode, and so on), so whether a particular
/// transform is available on a particular context is decided per method at the call site, exactly as with
/// [`Batch::batch`](crate::batching::Batch::batch). Tangents and cotangents are ordinary values of the same universe
/// as the primals — [`Domain::Value`] — flowing through the same context (the descriptor-level tangent structure,
/// such as cotangent types, lives on [`DifferentiableType`] instead). Predicate-capable operations such as
/// `condition`, `while`, and `select` impose their own [`BooleanLike`] bounds through their operation-family
/// implementations; tangent carriers themselves do not need to be Boolean-like just to participate in
/// differentiation.
///
/// Whether a transform runs eagerly or stages a program is decided by the context's
/// [`Value`](Domain::Value) (concrete vs [`Tracer`]), not by a separate trait. Values from a *different* trace are
/// detected lazily, like everything else about staging: a foreign tracer fails the builder-identity check either
/// when an operation binds it ([`StagingContext::stage_operation`]) or when it escapes through a trace boundary
/// (the boundary output checks), with [`ProgramError::MismatchedProgramBuilders`].
pub trait Differentiate: Context {
    /// Evaluates `function` on the primal `primals` and propagates the tangent `tangents` forward by running the
    /// closure **directly on [`DifferentiationTracer`] duals** — the single forward-mode entry point, and the analogue of
    /// [JAX's `jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html).
    ///
    /// Like [`batch`](crate::batching::Batch::batch), this is a context-wrapping transform: each input is paired
    /// with its tangent as a dual over a [`DifferentiationContext`] wrapping this context, and `function` runs directly on those
    /// duals, with each operation the closure performs (`x.sin()`, `x * y`, …) dispatching its
    /// [`jvp`](DifferentiableOperation::jvp) rule through [`Context::bind`]. Eager-versus-staged behavior is
    /// absorbed entirely by this context:
    ///
    ///   - Over an **eager** context both dual halves are concrete, so the closure sees real primal values — it can
    ///     branch on them (`if x.boolean()? { … }`), print them, or otherwise use Rust control flow driven by the
    ///     primal — and a staged data-dependent `while` combinator differentiates by running directly at the
    ///     concrete primals, with no iteration bound needed.
    ///   - Over a **staging** context the same closure stages the primal and tangent operations into the enclosing
    ///     trace op by op (this is how a fused JVP computation is built under an outer trace), and branching on a
    ///     primal errors because it is a [`Tracer`] with no concrete payload.
    ///
    /// The closure executes exactly as written: no dead code is trimmed, and observable effects fire as the closure
    /// runs. Structural zero tangents stay symbolic between operations and are materialized through this context's
    /// [`Zero`] capability only at the output boundary. Transforms nest: inside the closure, an inner transform
    /// invoked on a dual's [`DifferentiationContext`] (a [`Differentiate`] itself) differentiates through the duals,
    /// composing reverse-over-forward and higher-order forward modes.
    fn jvp<F, Input, Output>(
        &self,
        function: F,
        primals: Input,
        tangents: Input::To<<Self as Domain>::Value>,
    ) -> Result<(Output::To<<Self as Domain>::Value>, Output::To<<Self as Domain>::Value>), ProgramError>
    where
        Self: Zero<<Self as Domain>::Value>,
        <Self as Domain>::Operation: Clone + DifferentiableOperation<Self>,
        F: FnOnce(Input::To<DifferentiationTracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                Family: ParameterizedFamily<DifferentiationTracer<Self>> + ParameterizedFamily<<Self as Domain>::Value>,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<DifferentiationTracer<Self>, Family: ParameterizedFamily<<Self as Domain>::Value>>,
    {
        if primals.parameters().next().is_none() {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
        }
        let primal_structure = primals.parameter_structure();
        let tangent_structure = tangents.parameter_structure();
        // Tangents are ordinary domain values, so each dual pairs values of the same type on both sides.
        if tangent_structure != primal_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }

        // Wrap each (primal, tangent) as a dual stamped with the forward-mode context so the closure's value sugar
        // dispatches through it, then run the closure directly on those duals.
        let context = DifferentiationContext::new(self.clone());
        let input_duals = primals
            .into_parameters()
            .zip(tangents.into_parameters())
            .map(|(primal, tangent)| {
                DifferentiationTracer::new(DifferentiationDual::new(primal, tangent), context.clone())
            })
            .collect::<Vec<_>>();
        let input = Input::To::<DifferentiationTracer<Self>>::from_parameters(primal_structure, input_duals)?;
        let output = function(input)?;

        // Split each output dual into its primal value and its materialized tangent.
        let output_structure = output.parameter_structure();
        let output_duals = output.into_parameters().collect::<Vec<_>>();
        let mut primal_outputs = Vec::with_capacity(output_duals.len());
        let mut tangent_outputs = Vec::with_capacity(output_duals.len());
        for dual in output_duals {
            let (primal, tangent) = dual.into_dual().into_parts();
            tangent_outputs.push(tangent.materialize(self)?);
            primal_outputs.push(primal);
        }
        let primal_output =
            Output::To::<<Self as Domain>::Value>::from_parameters(output_structure.clone(), primal_outputs)?;
        let tangent_output = Output::To::<<Self as Domain>::Value>::from_parameters(output_structure, tangent_outputs)?;

        Ok((primal_output, tangent_output))
    }

    /// Linearizes `function` at `primals`, returning the primal output and a reusable
    /// [`Pushforward`] — the JAX `linearize` analogue.
    ///
    /// This is the partial-evaluation sibling of [`jvp`](Self::jvp): where `jvp` runs the closure once per
    /// `(primal, tangent)` pair, this runs the closure once on [`DifferentiationTracer`] duals over a
    /// [`PartialEvaluationContext`] wrapping this context, with each dual's primal half *known* at its primal value
    /// and its tangent half *unknown*. Primal-side operations are then all-known and fold through this context itself
    /// — executing eagerly under an eager context or staging into the enclosing trace under a staging one, so
    /// linearization composes under an outer trace — while tangent-side operations residualize into the accumulated
    /// pushforward program `(ẋ, r) ↦ ẏ`, which is linear in `ẋ` with the linearization point entering only through
    /// the residuals `r` recovered along the way. The returned [`Pushforward`] closes that program over those
    /// residuals, so [`Pushforward::apply`] pushes any number of tangents through the function's Jacobian at this
    /// point without re-tracing or re-differentiating.
    ///
    /// Because the closure's primal halves carry concrete values under an eager context, host control flow on primals
    /// works exactly as under [`jvp`](Self::jvp): the closure can branch on a primal (`if x.boolean()? { … }`), the
    /// untaken branch is never traced at all, and a data-dependent `while` combinator differentiates by running
    /// directly at the concrete primals. This matches JAX's `linearize`/`grad` tracing semantics, where the same JVP
    /// interpreter runs over a partial-evaluation trace instead of the eval trace.
    ///
    /// Reverse mode is this transform plus transposition, literally: [`vjp`](Self::vjp) calls this, opens the
    /// returned [`Pushforward`] back up with [`Pushforward::into_parts`], and transposes its program into the
    /// pullback (and the forward-mode Jacobian transform batch-replays it the same way).
    fn linearize<F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        (
            TracedOutput::To<<Self as Domain>::Value>,
            Pushforward<Self, Input, TracedOutput::To<<Self as Domain>::Value>>,
        ),
        ProgramError,
    >
    where
        <Self as Domain>::Operation: Clone
            + PartiallyEvaluatableOperation<Self>
            + From<ZeroOperation<<Self as Domain>::Type>>
            + DifferentiableOperation<PartialEvaluationContext<Self>>,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Result<TracedOutput, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<<Self as Domain>::Value>,
            >,
        TracedOutput: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<<Self as Domain>::Value>>,
    {
        if primals.parameters().next().is_none() {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
        }
        let input_structure = primals.parameter_structure();
        let input_values = primals.into_parameters().collect::<Vec<_>>();
        let tangent_input_count = input_values.len();

        // Wrap each primal as a dual over a partial-evaluation context wrapping this context: the primal half is a
        // known value and the tangent half is an unknown seeded as a leading residual-program input, in primal-input
        // order.
        let evaluation_context = PartialEvaluationContext::new(self.clone());
        let differentiation_context = DifferentiationContext::new(evaluation_context.clone());
        let input_duals = input_values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let tangent = evaluation_context.unknown_input(value.r#type().into_owned(), index);
                let dual = DifferentiationDual::new(
                    PartialTracer::new(evaluation_context.clone(), PartialEvaluationValue::known_input(value)),
                    MaybeZero::Value(PartialTracer::new(evaluation_context.clone(), tangent)),
                );
                DifferentiationTracer::new(dual, differentiation_context.clone())
            })
            .collect::<Vec<_>>();
        let input = Input::To::<LinearizationTracer<Self>>::from_parameters(input_structure, input_duals)?;
        let output = function(input)?;

        // Split each output dual into its known primal value and its tangent. Primal work depends only on the known
        // primal inputs, so every primal half must have folded to a known value. Tangent halves that are structural
        // zeros — or that folded to known values, which a map linear in `ẋ` produces only for its constant-zero
        // components — are restored as staged zeros, so the pushforward program presents the canonical
        // one-tangent-output-per-primal-output arity (matching `Program::linearize`'s restoration).
        let output_structure = output.parameter_structure();
        let output_duals = output.into_parameters().collect::<Vec<_>>();
        let staged_zero = |r#type: <Self as Domain>::Type| {
            let mut outputs = evaluation_context.residualize(ZeroOperation::new(r#type), &[])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok::<_, ProgramError>(outputs.remove(0))
        };
        let mut primal_outputs = Vec::with_capacity(output_duals.len());
        let mut tangent_outputs = Vec::with_capacity(output_duals.len());
        for dual in output_duals {
            let (primal, tangent) = dual.into_dual().into_parts();
            let primal = match primal.into_value()?.value() {
                PartialValue::Known(value) => value.clone(),
                PartialValue::Unknown(_) => {
                    return Err(ProgramError::MalformedProgram(
                        "linearization produced an unknown primal output but primal work depends only on the known \
                         primal inputs"
                            .to_string(),
                    ));
                }
            };
            let tangent = match tangent {
                MaybeZero::Value(tracer) => {
                    let value = tracer.into_value()?;
                    match value.value() {
                        PartialValue::Unknown(_) => value,
                        PartialValue::Known(known) => staged_zero(known.r#type().into_owned())?,
                    }
                }
                MaybeZero::Zero(r#type) => staged_zero(r#type)?,
            };
            primal_outputs.push(primal);
            tangent_outputs.push(tangent);
        }
        let output =
            TracedOutput::To::<<Self as Domain>::Value>::from_parameters(output_structure.clone(), primal_outputs)?;

        // All tracer-stamped context clones are dropped here, so the accumulated pushforward program can be
        // finalized.
        drop(differentiation_context);
        let evaluation = evaluation_context.into_evaluation(tangent_outputs)?;

        // The pushforward program's inputs are the leading tangent unknowns followed by the residuals materialized
        // during the trace; collect the residual values in input order.
        let mut residuals = Vec::with_capacity(evaluation.inputs.len().saturating_sub(tangent_input_count));
        for (index, input) in evaluation.inputs.iter().enumerate() {
            match input {
                PartialEvaluationInput::Unknown(ordinal) if index < tangent_input_count && *ordinal == index => {}
                PartialEvaluationInput::Known(value) if index >= tangent_input_count => residuals.push(value.clone()),
                _ => {
                    return Err(ProgramError::MalformedProgram(
                        "linearization produced a pushforward program whose tangent inputs do not lead its residuals"
                            .to_string(),
                    ));
                }
            }
        }

        // Close the pushforward program over the linearization-point residuals behind the reusable callable.
        let pushforward = Pushforward {
            context: self.clone(),
            program: evaluation.program,
            residuals,
            output_structure,
            marker: PhantomData,
        };
        Ok((output, pushforward))
    }

    /// Reverse-mode-differentiates `function` at `primals`, returning the primal output and a reusable [`Pullback`] —
    /// the JAX `vjp` analogue.
    ///
    /// This is the value-level reverse-mode transform: [`linearize`](Self::linearize) followed by transposition,
    /// exactly JAX's `vjp = linearize + transpose`. The closure runs once on [`DifferentiationTracer`] duals over a
    /// [`PartialEvaluationContext`] wrapping this context (primal halves known, tangent halves unknown), which
    /// executes the primal work through this context — recovering the primal outputs and the residual values at the
    /// linearization point — while accumulating the linear pushforward program `(ẋ, r) ↦ ẏ`; that program is then
    /// transposed with respect to its leading tangent inputs, holding the trailing residuals as known parameters. The
    /// resulting pullback program stays in this context's staged [`Constant`](Domain::Constant) space; interpreting
    /// it through [`Program::interpret_in_context`] lifts its literal constants through this context's
    /// [`lift`](Context::lift) at replay time, which is what serves reverse mode *under tracing*: in an eager context
    /// the lift is the identity, while in a staging context (whose values are [`Tracer`]s) it records the pullback's
    /// constants in the enclosing trace, so the backward pass splices into that trace. Host control flow on primals
    /// works exactly as under [`linearize`](Self::linearize) (JAX's `grad`-allows-Python-control-flow property).
    ///
    /// The returned [`Pullback`] closes the transposed program over the linearization-point residuals, so
    /// [`Pullback::apply`] maps output cotangents to input cotangents — appending the residuals, interpreting the
    /// program, and reshaping the flat input cotangents against the closure's input structure — without the caller
    /// threading the residuals by hand. Consumers that need the open parts (e.g., to batch-replay or seed the
    /// pullback program manually) recover them with [`Pullback::into_parts`].
    ///
    /// Functions reaching operations outside the supported straight-line slice fail with an
    /// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
    fn vjp<F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        (TracedOutput::To<<Self as Domain>::Value>, Pullback<Self, Input, TracedOutput::To<<Self as Domain>::Value>>),
        ProgramError,
    >
    where
        <Self as Domain>::Type: DifferentiableType,
        <Self as Domain>::Constant: Value<Type = <Self as Domain>::Type>,
        <Self as Domain>::Operation: Clone
            + TransposableOperation<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + PartiallyEvaluatableOperation<Self>
            + From<ZeroOperation<<Self as Domain>::Type>>
            + From<AddOperation>
            + DifferentiableOperation<PartialEvaluationContext<Self>>,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Result<TracedOutput, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<LinearizationTracer<Self>>,
            >,
        TracedOutput: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<<Self as Domain>::Value>>,
    {
        let input_structure = primals.parameter_structure();
        let (output, pushforward) = self.linearize(function, primals)?;
        let (program, residuals) = pushforward.into_parts();

        // Transpose the pushforward program with respect to its leading tangent inputs, holding the trailing residual
        // inputs as known parameters. Partition-aware transposition threads each residual through to the pullback as
        // a pullback input rather than folding it into a captured factor, so the pullback maps
        // `(output_cotangents ++ residuals)` to the input cotangents.
        let with_respect_to = (0..program.input_ids().len() - residuals.len()).collect::<Vec<_>>();
        let program = program.transpose_with_respect_to(with_respect_to.as_slice())?;
        Ok((output, Pullback { context: self.clone(), program, residuals, input_structure, marker: PhantomData }))
    }

    /// Returns the traced scalar output and reverse-mode gradient for `function`.
    ///
    /// This is the active-context counterpart of [`crate::tracing_v2::value_and_gradient`]. It uses
    /// [`Differentiate::vjp`] directly, so nested reverse mode composes with any enclosing context that
    /// implements this trait instead of going through a separate tracer dispatch path.
    fn value_and_gradient<F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(<Self as Domain>::Value, Input::To<<Self as Domain>::Value>), DifferentiationError>
    where
        <Self as Domain>::Constant: Value<Type = <Self as Domain>::Type>,
        <Self as Domain>::Type: DifferentiableType,
        <Self as Domain>::Operation: Clone
            + TransposableOperation<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + PartiallyEvaluatableOperation<Self>
            + From<ZeroOperation<<Self as Domain>::Type>>
            + From<OneOperation<<Self as Domain>::Type>>
            + From<AddOperation>
            + DifferentiableOperation<PartialEvaluationContext<Self>>,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> LinearizationTracer<Self>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<LinearizationTracer<Self>>,
            >,
    {
        let (output, pullback) = self.vjp(|input| Ok(function(input)), primals)?;
        // Reverse mode only defines a gradient for scalar-output functions; reject non-scalar outputs before
        // seeding (see `DifferentiationError::NonScalarGradientOutput`).
        if !output.r#type().is_scalar() {
            return Err(DifferentiationError::NonScalarGradientOutput { output_type: output.r#type().to_string() });
        }
        // Seed the single output cotangent with the multiplicative identity of the scalar output, typed with the
        // output's cotangent type (e.g., swapping unreduced and reduced sharding axes for arrays) and staged through
        // `bind`. A non-differentiable scalar output (a Boolean or integer, the `float0` analogue) carries no cotangent
        // space and thus no "one" to seed, so reverse mode is degenerate and is rejected up front. The pullback then
        // pulls the seed back to the input cotangents, reshaped against the closure's input structure.
        let output_cotangent_type = output.r#type().cotangent().ok_or_else(|| {
            DifferentiationError::NonDifferentiableGradientOutput { output_type: output.r#type().to_string() }
        })?;
        let one_operation = <Self as Domain>::Operation::from(OneOperation::new(output_cotangent_type));
        let mut seeds = self.bind(one_operation, &[])?;
        check_count!("output", seeds, 1, ProgramError);
        let gradient = pullback.apply(seeds.pop().unwrap())?;
        Ok((output, gradient))
    }

    /// Returns the reverse-mode gradient of a traced scalar-output function. This is the gradient-only counterpart of
    /// [`value_and_gradient`](Self::value_and_gradient), discarding the primal output.
    #[inline]
    fn gradient<F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<Input::To<<Self as Domain>::Value>, DifferentiationError>
    where
        <Self as Domain>::Constant: Value<Type = <Self as Domain>::Type>,
        <Self as Domain>::Type: DifferentiableType,
        <Self as Domain>::Operation: Clone
            + TransposableOperation<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + PartiallyEvaluatableOperation<Self>
            + From<ZeroOperation<<Self as Domain>::Type>>
            + From<OneOperation<<Self as Domain>::Type>>
            + From<AddOperation>
            + DifferentiableOperation<PartialEvaluationContext<Self>>,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> LinearizationTracer<Self>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<LinearizationTracer<Self>>,
            >,
    {
        self.value_and_gradient(function, primals).map(|(_, gradient)| gradient)
    }
}

impl<C: Context> Differentiate for C {}

/// A forward-mode differentiation [`Context`] that interleaves [`DifferentiableOperation`] rules with an inner
/// [`Context`], without building a program: its values are [`DifferentiationTracer`] duals over the inner context's values, and
/// binding an operation dispatches the operation's [`jvp`](DifferentiableOperation::jvp) rule against the inner
/// context directly. Over an eager inner context this computes primal and tangent values operation by operation
/// (the analogue of [JAX's `jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) interpreter), while over
/// a staging inner context the rules stage the primal and tangent operations into the enclosing trace.
///
/// This is forward mode's counterpart of [`BatchingContext`](crate::batching::BatchingContext): a transform context
/// that wraps the receiver and runs the user's closure directly on transform tracers ([`DifferentiationTracer`] duals here,
/// [`BatchingTracer`](crate::batching::BatchingTracer)s there), with eager-versus-staged behavior absorbed entirely
/// by the wrapped context. It is what makes [`Differentiate::jvp`] the single forward-mode entry point.
///
/// Structural zero tangents stay symbolic [`MaybeZero::Zero`]s while they flow between rules: the
/// [`bind`](Context::bind) fast path skips an operation's rule entirely when every input tangent is a structural
/// zero, exactly like the program-level replay behind [`Program::linearize`], so no zero values are constructed and no
/// zero work is performed until a boundary [`materialize`](MaybeZero::materialize)s one through the inner
/// context's [`Zero`] capability.
#[derive(Clone)]
pub struct DifferentiationContext<C: Context> {
    /// Parent context that carries the primal and tangent values and executes (or stages) the operations that the
    /// forward-mode rules bind.
    parent: C,
}

impl<C: Context> DifferentiationContext<C> {
    /// Creates a new [`DifferentiationContext`] over the provided parent [`Context`].
    #[inline]
    pub fn new(parent: C) -> Self {
        Self { parent }
    }

    /// Returns the parent [`Context`].
    #[inline]
    pub fn parent(&self) -> &C {
        &self.parent
    }
}

impl<C: Context> Domain for DifferentiationContext<C> {
    type Type = C::Type;
    type Value = DifferentiationTracer<C>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C: Context<Operation: Clone + DifferentiableOperation<C>> + Zero<C::Value>> Context for DifferentiationContext<C> {
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<DifferentiationTracer<C>, ProgramError> {
        // Constants are independent of every differentiation input, so their tangents are structural zeros.
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent.lift(constant)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }

    fn bind<O: Into<C::Operation>>(
        &self,
        operation: O,
        inputs: &[DifferentiationTracer<C>],
    ) -> Result<Vec<DifferentiationTracer<C>>, ProgramError> {
        let operation = operation.into();
        // Unwrap the input tracers into context-free duals, run the rule against those, and rewrap the produced duals
        // with this context, mirroring how `BatchingContext::bind` unwraps to `ArrayBatch`es and rewraps.
        let input_duals = inputs.iter().map(|input| input.dual().clone()).collect::<Vec<_>>();
        // All-zero fast path mirroring `Program::jvp`: when an operation consumes at least one input and every
        // input tangent is a structural zero, the operation's tangent is zero by the chain rule, so the rule is
        // skipped and the primal operation binds directly. Zero-input operations are excluded so their dedicated
        // rules keep handling primal synthesis and tangent typing.
        let output_duals = if !input_duals.is_empty() && input_duals.iter().all(|dual| dual.tangent().is_zero()) {
            let primal_inputs = input_duals.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
            self.parent
                .bind(operation, &primal_inputs)?
                .into_iter()
                .map(DifferentiationDual::new_with_zero_tangent)
                .collect()
        } else {
            operation.jvp(&self.parent, input_duals.as_slice())?
        };
        // Stamp this context onto every value handed back to the caller so its capability sugar dispatches through this
        // forward-mode context (the `jvp` rules build their outputs context-free via `DifferentiationDual::new`).
        Ok(output_duals.into_iter().map(|dual| DifferentiationTracer::new(dual, self.clone())).collect())
    }

    /// A forward-mode context is eager exactly when the inner context carrying its duals' values is (never over a
    /// staging inner context, always over an eager one).
    #[inline]
    fn is_eager(&self) -> bool {
        self.parent.is_eager()
    }
}

impl<T, V, O, Input, Output> Program<V, O, Input, Output>
where
    T: Type,
    V: Value<Type = T>,
    O: Clone + Operation<T> + From<ZeroOperation<T>>,
    O: DifferentiableOperation<TracingContext<V, O>>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Builds the *fused* jvp program of this already-traced primal [`Program`].
    ///
    /// Read the input program as a function `f` from its flat inputs to its flat outputs, `x ↦ y = f(x)`. This returns
    /// the program that computes `f` together with its *pushforward* (the forward-mode Jacobian-vector product): given
    /// an input tangent (i.e., perturbation direction) `ẋ`, the pushforward produces the output tangent
    /// `ẏ = (∂f/∂x)(x) · ẋ`, the directional derivative of `f` at `x` along `ẋ`. As a single map, the returned program
    /// computes `(x, ẋ) ↦ (f(x), (∂f/∂x)(x) · ẋ) = (y, ẏ)`.
    ///
    /// In terms of the flat program boundaries: if the input program has inputs `[x_1, …, x_n]` and outputs
    /// `[y_1, …, y_m]` (so `y = f(x)`), the returned program has
    ///
    ///   - inputs `[x_1, …, x_n, ẋ_1, …, ẋ_n]` — the `n` primal inputs followed by one fresh tangent input `ẋ_i` per
    ///     primal input `x_i`, of the same type; and
    ///   - outputs `[y_1, …, y_m, ẏ_1, …, ẏ_m]` — the `m` primal outputs `y_j = f_j(x)` followed by the `m` tangent
    ///     outputs `ẏ = (∂f/∂x)(x) · ẋ`.
    ///
    /// Both halves stay over the same primal operation family; the program is *not* split into separate primal and
    /// tangent sub-programs (that is [`Self::linearize`], whose partial-evaluation known-ness split consumes this fused
    /// program as its front half). This un-split form is exposed for fused higher-order JVP rules and direct
    /// forward-mode interpretation.
    ///
    /// Each primal instruction is replayed once through its [`DifferentiableOperation`] rule, which returns the dual
    /// (primal result plus tangent) for the instruction's outputs; both are staged into the shared builder as ordinary
    /// primal operations, so the result contains no symbolic capture.
    ///
    /// Atoms that are not reached by any input tangent are structurally zero. Their tangents stay symbolic as typed
    /// [`MaybeZero::Zero`]s and stage nothing. The shared all-zero fast path below short-circuits the all-zero case (an
    /// operation consuming at least one input whose every input tangent is a structural zero) by staging the primal
    /// operation directly and pairing each primal output with a typed structural zero tangent, so zero-ness propagates
    /// transitively without staging or scanning instructions. Structural zero tangents are materialized as typed
    /// [`ZeroOperation`] instructions only at the output boundary, preserving the `(primal_outputs ++ tangent_outputs)`
    /// program contract.
    ///
    /// Operations outside the supported slice fail with the [`DifferentiableOperation`] default's
    /// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
    pub fn jvp(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        let primal_input_count = self.input_ids().len();

        // Hold a standalone `Rc` clone of the context's builder, and move the context itself into the block below, so that
        // scoping every tracer (and the context) inside that block makes the `Rc::try_unwrap` at the end a real ownership
        // check rather than depending on manual drops. Only raw output atom ids escape the block.
        let context = TracingContext::<V, O>::new();
        let builder = context.builder().clone();
        let output_atoms = {
            let context = context;

            // Track the primal tracer and symbolic tangent for each source atom. Tangents of atoms not connected to an
            // input tangent (constants and dead inputs) are derived lazily as structural zeros typed with the atom's
            // primal type.
            let mut primals: Vec<Option<Tracer<TracingContext<V, O>>>> = vec![None; self.atoms().len()];
            let mut tangents: Vec<Option<MaybeZero<Tracer<TracingContext<V, O>>>>> = vec![None; self.atoms().len()];

            // Primal inputs become the leading inputs; one fresh tangent input is added per primal input afterwards
            // so the input order is `(primals ++ tangents)`.
            for input_id in self.input_ids().iter().copied() {
                let r#type = self.atoms()[input_id.index()].r#type().into_owned();
                primals[input_id.index()] = Some(context.input(r#type));
            }
            for input_id in self.input_ids().iter().copied() {
                let r#type = self.atoms()[input_id.index()].r#type().into_owned();
                tangents[input_id.index()] = Some(MaybeZero::Value(context.input(r#type)));
            }

            // Constants are lifted into the builder as primal constants; their tangents are derived lazily as structural
            // zeros typed with the atom's primal type. The call is disambiguated to the staging method because the
            // `Constant` capability trait also provides a `constant` method.
            for (atom_index, atom) in self.atoms().iter().enumerate() {
                if let Atom::Constant(value) = atom {
                    primals[atom_index] = Some(StagingContext::constant(&context, value.clone()));
                }
            }

            // Replay each primal instruction in JVP form, staging both the primal result and the tangent operations
            // into the shared builder.
            for instruction in self.instructions() {
                let input_duals = instruction
                    .inputs()
                    .iter()
                    .copied()
                    .map(|input_atom| {
                        let primal = primals[input_atom.index()]
                            .clone()
                            .ok_or(ProgramError::UnboundAtomId { id: input_atom })?;
                        // Atoms not connected to an input tangent (constants and dead inputs) take a structural zero typed
                        // with the atom's primal type.
                        let tangent = match &tangents[input_atom.index()] {
                            Some(tangent) => tangent.clone(),
                            None => MaybeZero::Zero(primal.r#type().into_owned()),
                        };
                        Ok(DifferentiationDual::<Tracer<TracingContext<V, O>>>::new(primal, tangent))
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;

                // All-zero fast path: when an operation consumes at least one input and every input tangent is a
                // structural zero, the operation's tangent is zero by the chain rule, so the rule is skipped. The primal
                // outputs are staged directly and each output tangent is a typed structural zero. Zero-input operations
                // are excluded so their dedicated rules keep handling primal synthesis and tangent typing.
                let all_input_tangents_are_zero = input_duals.iter().all(|dual| dual.tangent().is_zero());
                let output_duals = if !input_duals.is_empty() && all_input_tangents_are_zero {
                    let primal_inputs = input_duals.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
                    context
                        .stage_operation(instruction.operation().clone(), primal_inputs.as_slice())?
                        .into_iter()
                        .map(DifferentiationDual::<Tracer<TracingContext<V, O>>>::new_with_zero_tangent)
                        .collect()
                } else {
                    instruction.operation().jvp(&context, input_duals.as_slice())?
                };

                check_count!("output", output_duals, instruction.outputs().len(), ProgramError);
                for (output_atom, dual) in instruction.outputs().iter().copied().zip(output_duals) {
                    let (primal, tangent) = dual.into_parts();
                    primals[output_atom.index()] = Some(primal);
                    tangents[output_atom.index()] = Some(tangent);
                }
            }

            // Collect the outputs: the primal outputs followed by the tangent outputs, in the original output order.
            // Structural zero tangents are materialized as typed `ZeroOperation` instructions here — the output boundary
            // is the only place the fused program requires a real atom for them.
            let primal_output_atoms = self
                .output_ids()
                .iter()
                .copied()
                .map(|output_atom| {
                    primals[output_atom.index()]
                        .as_ref()
                        .map(|primal| primal.atom_id())
                        .ok_or(ProgramError::UnboundAtomId { id: output_atom })?
                })
                .collect::<Result<Vec<_>, _>>()?;
            let tangent_output_atoms = self
                .output_ids()
                .iter()
                .copied()
                .map(|output_atom| {
                    // Atoms not connected to an input tangent (constants and dead inputs) take a structural zero typed
                    // with the atom's primal type.
                    let tangent = match &tangents[output_atom.index()] {
                        Some(tangent) => tangent.clone(),
                        None => MaybeZero::Zero(
                            primals[output_atom.index()]
                                .as_ref()
                                .ok_or(ProgramError::UnboundAtomId { id: output_atom })?
                                .r#type()
                                .into_owned(),
                        ),
                    };
                    tangent.materialize(&context)?.atom_id()
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut output_atoms = primal_output_atoms;
            output_atoms.extend(tangent_output_atoms);
            output_atoms
        };

        // All tracing handles are dropped here, so the builder can be recovered and finalized.
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let input_count = 2 * primal_input_count;
        let output_count = output_atoms.len();
        builder.build::<Vec<V>, Vec<V>>(output_atoms, vec![Placeholder; input_count], vec![Placeholder; output_count])
    }
}

/// Result of [`Program::linearize`]: the linearization of a program computing `y = f(x)`,
/// split into a nonlinear primal sub-program and a linear tangent sub-program that communicate through a residual
/// environment.
///
/// Linearization splits the program's fused jvp program `(x, ẋ) ↦ (f(x), (∂f/∂x)(x) · ẋ)` by known-ness into:
///
///   - the [`primal`](Self::primal) sub-program `x ↦ (y, r)`, computing the primal outputs `y = f(x)` together with
///     the residuals `r` — the intermediate values of the derivative computation that depend only on `x` (e.g.,
///     `cos(x)` when `f` is `sin`); and
///   - the [`tangent`](Self::tangent) sub-program `(ẋ, r) ↦ ẏ`, computing the tangent outputs `ẏ = (∂f/∂x)(x) · ẋ`.
///     It is linear in `ẋ`, with the linearization point `x` entering only through the residuals `r`.
///
/// This is the domain-free, interpretation-free core shared by every linearization entry point: it carries only the
/// two sub-programs and the residual count that relates them, leaving the concrete primal outputs to be recovered by
/// callers that interpret [`primal`](Self::primal) under a value semantics of their choice. The tangent sub-program
/// stays in the primal operation family `O` with the residuals as ordinary trailing inputs, which is why
/// [`pullback`](Self::pullback) can transpose it directly through
/// [`Program::transpose_with_respect_to`] without re-keying it into a
/// linear operation family.
///
/// The value type `V` and operation family `O` match the primal program being linearized.
pub struct Linearization<V: Value, O: Clone + Operation<V::Type>> {
    /// Nonlinear primal sub-program `x ↦ (y, r)`. It takes the primal inputs `x` and produces the primal outputs
    /// `y = f(x)` followed by the residuals `r`, its trailing [`residual_count`](Self::residual_count) outputs, which
    /// form the residual environment consumed by the tangent sub-program.
    primal: Program<V, O, Vec<V>, Vec<V>>,

    /// Linear tangent sub-program `(ẋ, r) ↦ ẏ`. It takes the tangent inputs `ẋ` followed by the residuals `r` and
    /// produces the tangent outputs `ẏ = (∂f/∂x)(x) · ẋ`.
    tangent: Program<V, O, Vec<V>, Vec<V>>,

    /// Number of residuals `r` threaded from the primal sub-program into the tangent sub-program — the count of the
    /// trailing outputs of [`primal`](Self::primal) and of the trailing inputs of [`tangent`](Self::tangent).
    residual_count: usize,
}

impl<V: Value, O: Clone + Operation<V::Type>> Linearization<V, O> {
    /// Returns the nonlinear primal sub-program `x ↦ (y, r)`. It takes the primal inputs `x` and produces the primal
    /// outputs `y = f(x)` followed by the residuals `r` — the intermediate values of the derivative computation that
    /// depend only on `x` — whose trailing [`residual_count`](Self::residual_count) outputs form the residual
    /// environment consumed by the [`tangent`](Self::tangent) sub-program.
    pub fn primal(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.primal
    }

    /// Returns the linear tangent sub-program `(ẋ, r) ↦ ẏ`. It takes the tangent inputs `ẋ` followed by the
    /// residuals `r` and produces the tangent outputs `ẏ = (∂f/∂x)(x) · ẋ`. The sub-program is linear in `ẋ`, with
    /// the linearization point `x` entering only through the residuals `r`.
    pub fn tangent(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.tangent
    }

    /// Returns the number of residuals `r` threaded from the primal sub-program into the tangent sub-program — the
    /// count of the trailing outputs of [`primal`](Self::primal) and of the trailing inputs of
    /// [`tangent`](Self::tangent).
    pub fn residual_count(&self) -> usize {
        self.residual_count
    }

    /// Consumes this [`Linearization`] and returns its [`primal`](Self::primal) sub-program,
    /// [`tangent`](Self::tangent) sub-program, and [`residual_count`](Self::residual_count), in that order.
    pub fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>, usize) {
        (self.primal, self.tangent, self.residual_count)
    }

    /// Interprets the [`primal`](Self::primal) sub-program at the primal inputs `x` through `context`, returning the
    /// primal outputs `y = f(x)` and the residuals `r`, split at [`residual_count`](Self::residual_count).
    ///
    /// This recovers the value-level half of a linearization point: the outputs and residuals flow as `context`'s
    /// [`Value`](crate::contexts::Domain::Value)s — concrete values under an eager context, enclosing-trace tracers
    /// under a staging one — while the sub-program's staged constants are lifted through the context's
    /// [`lift`](Context::lift) at replay time.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context whose [`lift`](Context::lift) and [`bind`](Context::bind) replay the primal
    ///     sub-program.
    ///   - `primals`: Flat primal inputs `x`, aligned with the primal sub-program's inputs.
    pub fn interpret_primal<C>(
        &self,
        context: &C,
        primals: Vec<C::Value>,
    ) -> Result<(Vec<C::Value>, Vec<C::Value>), ProgramError>
    where
        C: Context<Type = V::Type, Constant = V, Operation = O>,
    {
        let mut outputs = self.primal.interpret_in_context(context, primals)?;
        let output_count = outputs.len().checked_sub(self.residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "primal program produced {} outputs which is fewer than its {} residuals",
                outputs.len(),
                self.residual_count,
            ))
        })?;
        let residuals = outputs.split_off(output_count);
        Ok((outputs, residuals))
    }

    /// Returns the forward-mode pushforward program `(ẋ, r) ↦ ẏ`: it takes the tangent inputs `ẋ` followed by the
    /// residuals `r` and produces the tangent outputs `ẏ = (∂f/∂x)(x) · ẋ`. Because linearization already produces
    /// the pushforward as its unknown half, this is the [`tangent`](Self::tangent) sub-program itself, cloned — the
    /// identity counterpart of [`pullback`](Self::pullback), which derives its program by transposition.
    pub fn pushforward(&self) -> Program<V, O, Vec<V>, Vec<V>> {
        self.tangent.clone()
    }

    /// Builds the reverse-mode pullback program `(ȳ, r) ↦ x̄` by transposing the [`tangent`](Self::tangent)
    /// sub-program: it takes the output cotangents `ȳ` followed by the residuals `r` and produces the input
    /// cotangents `x̄ = (∂f/∂x)(x)ᵀ · ȳ`. It is the derived third member of this [`Linearization`]'s program family,
    /// alongside the stored [`primal`](Self::primal) and [`tangent`](Self::tangent) sub-programs.
    ///
    /// Rather than re-keying each bilinear operation of the tangent sub-program into a closed captured factor (for
    /// example, folding a scalar `Mul` against a known operand into a multiply-by-a-captured-constant) by folding the
    /// consuming residual value, this function leaves the tangent sub-program in the primal operation family `O` and
    /// transposes it through
    /// [`Program::transpose_with_respect_to`]. The tangent sub-program's
    /// inputs are `(ẋ, r)`, so it is transposed with respect to the leading tangent inputs `ẋ` while the trailing
    /// [`residual_count`](Self::residual_count) residual inputs are held as known parameters. Partition-aware
    /// transposition then threads each known residual through to the pullback as a pullback input (consumed by the
    /// adjoint operation that the bilinear operation's transpose rule stages), rather than folding it into a captured
    /// factor, so the returned pullback program stays over the primal operation family `O` and produces the
    /// cotangents of the linear tangent inputs only.
    pub fn pullback(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
    where
        V::Type: DifferentiableType,
        O: TransposableOperation<V, O> + From<ZeroOperation<V::Type>> + From<AddOperation>,
    {
        let tangent_input_count = self.tangent.input_ids().len().checked_sub(self.residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "tangent program has {} inputs which is fewer than its {} residuals",
                self.tangent.input_ids().len(),
                self.residual_count,
            ))
        })?;

        // Transpose with respect to the leading tangent inputs, holding the trailing residual inputs as known
        // parameters. Partial transposition exposes each known residual as a pullback input, so the residuals are not
        // folded into captured factors here.
        let with_respect_to = (0..tangent_input_count).collect::<Vec<_>>();
        self.tangent.transpose_with_respect_to(with_respect_to.as_slice())
    }
}

impl<T, V, O, Input, Output> Program<V, O, Input, Output>
where
    T: Type,
    V: Value<Type = T>,
    O: Clone + Operation<T> + From<ZeroOperation<T>>,
    O: DifferentiableOperation<TracingContext<V, O>>,
    O: PartiallyEvaluatableOperation<TracingContext<V, O>>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Builds the linearization core from this already-traced primal [`Program`] by fusing the forward-mode rules into
    /// one jvp program and splitting it into the primal (known) and tangent (unknown) halves through the
    /// partial-evaluation known-ness split.
    ///
    /// This is the domain-free, interpretation-free generic core of the linearization pipeline, shared by every
    /// concrete entry point. It builds the fused jvp program — replaying each primal instruction once in jvp form so
    /// the program stages both the primal computation and its pushforward over the primal operation family — and then
    /// partitions that program through [`Program::partition`](crate::Program::partition) with the leading primal
    /// inputs marked known and the trailing tangent inputs marked unknown. The split's fresh known-side staging
    /// trace becomes the primal program, so *linearity separation is known-ness separation*: the per-operation
    /// partial-evaluation rules
    /// own the split, higher-order operations (`scan`/`condition`) separate through their known-ness splits instead
    /// of needing linearize-specific handling, and effectful primal work lands in the primal program per the effect
    /// placement contract of
    /// [`PartialEvaluationContext::fold_or_residualize`](crate::partial::PartialEvaluationContext::fold_or_residualize).
    /// The known side computes the primal outputs followed by the residual edges and the residual side is the linear
    /// tangent map taking `(tangents ++ residuals)` — the JAX `linearize` shape, produced by the same machinery JAX
    /// uses (`partial_eval` of the jvp function). The tangent program's canonical input order is then rebuilt from
    /// the split's recorded per-input sources rather than assumed from the walk's input layout, so the tangent
    /// program always presents its full leading tangent inputs ahead of the residuals. No value semantics are
    /// applied: the returned [`Linearization`] carries only the two split sub-programs and the metadata needed to
    /// reassemble and transpose them, leaving interpretation of the primal side to callers.
    ///
    /// Linearization splits with the known-ness partial-evaluation rules rather than a value-free structural split:
    /// instruction-granular structural classification cannot separate a fused higher-order operation (a fused jvp
    /// `scan` mixes primal and tangent carries inside one instruction), while the known-ness rules split inside it.
    ///
    /// Operations outside the supported slice fail with the [`DifferentiableOperation`] default's
    /// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
    pub fn linearize(&self) -> Result<Linearization<V, O>, ProgramError> {
        let primal_input_count = self.input_ids().len();
        let primal_output_count = self.output_ids().len();

        // Build the fused jvp program over `[primals..., tangents...] -> [primal_outputs..., tangent_outputs...]`.
        let fused = self.jvp()?;

        // Split the fused program with the leading `primal_input_count` primal inputs known and the trailing tangent
        // inputs unknown. The split walks the fused program through the per-operation partial-evaluation rules
        // against a fresh known-side staging trace: known (primal) work folds by staging into that trace, and the
        // residual program that survives is the linear tangent map.
        let input_known = [vec![true; primal_input_count], vec![false; primal_input_count]].concat();
        let partition = fused.partition(input_known.as_slice())?;
        let residual_count = partition.residual_inputs().iter().filter(|input| input.is_known()).count();
        let known_output_indices = partition
            .outputs()
            .iter()
            .enumerate()
            .filter_map(|(index, output)| output.is_known().then_some(index))
            .collect::<Vec<_>>();
        let residual_output_indices = partition
            .outputs()
            .iter()
            .enumerate()
            .filter_map(|(index, output)| output.is_unknown().then_some(index))
            .collect::<Vec<_>>();
        let (mut known_program, residual_program, _, residual_inputs, _) = partition.into_parts();

        // The known program's outputs are the fully known fused outputs followed by the residual edges. Every primal
        // output must be known (the primals are all known, and effectful primal work folds into the known trace);
        // any *further* known outputs are structurally zero tangent outputs (for example the Boolean mask item of a
        // vmapped masked `while`, whose all-zero JVP fast path stages a fresh zero rather than threading the input
        // tangent), which belong to the tangent half and are restored there below.
        if known_output_indices.len() < primal_output_count
            || known_output_indices[..primal_output_count]
                .iter()
                .zip(0..primal_output_count)
                .any(|(&index, expected)| index != expected)
        {
            return Err(ProgramError::MalformedProgram(
                "a primal output did not fold to the known side during linearization".into(),
            ));
        }
        // Drop the stray tangent zeros from the known program's outputs so the primal program presents
        // `[primal_outputs..., residuals...]`: they occupy exactly the window between the primal outputs and the
        // residual edges.
        if known_output_indices.len() > primal_output_count {
            known_program.output_ids.drain(primal_output_count..known_output_indices.len());
            known_program.output_structure = vec![Placeholder; known_program.output_ids.len()];
        }

        // Restore the residual (tangent) program's canonical input order `[tangents..., residuals...]` from the
        // split's recorded per-input sources: each tangent input's atom lands at its original tangent position, a
        // tangent position missing from the sources is restored as a fresh dead atom of its fused type, and each
        // residual edge lands after the tangents at its edge ordinal. Today's walk seeds every unknown input up
        // front in original order, appends residual edges in first-use order, and never prunes residual-program
        // inputs, so this rebuild is an identity and no tangent position is ever missing; it stays source-driven
        // anyway because that layout is an implementation detail of the walk rather than part of the
        // partial-evaluation contract, and a walk that materialized unknown inputs lazily or pruned dead ones (a
        // structurally zero tangent whose input reaches no tangent output) would invalidate a layout-based rebuild
        // but not this one. The restored atoms are fresh program inputs that no instruction references, so the
        // direct program-field extensions preserve every [`Program`] invariant a [`ProgramBuilder`] would have
        // established.
        let mut tangent_program = residual_program;
        let surviving_input_ids = tangent_program.input_ids.split_off(0);
        let mut tangent_inputs: Vec<Option<AtomId>> = vec![None; primal_input_count];
        let mut edge_inputs: Vec<Option<AtomId>> = vec![None; residual_count];
        for (source, atom) in residual_inputs.iter().zip(surviving_input_ids) {
            match source {
                PartialEvaluationInput::Unknown(index) => {
                    let position = index.checked_sub(primal_input_count).ok_or_else(|| {
                        ProgramError::MalformedProgram(
                            "a known primal input survived as a residual-program input during linearization".into(),
                        )
                    })?;
                    tangent_inputs[position] = Some(atom);
                }
                PartialEvaluationInput::Known(ordinal) => edge_inputs[*ordinal] = Some(atom),
            }
        }
        for (position, atom) in tangent_inputs.into_iter().enumerate() {
            let restored = match atom {
                Some(atom) => atom,
                // The split recorded no source for this tangent position, so restore it as a fresh dead program input
                // (referenced by no instruction) typed from the corresponding fused-program tangent input. The fused
                // program's inputs are laid out as `[primals..., tangents...]`, so the tangent for `position` lives at
                // index `primal_input_count + position`.
                None => {
                    let fused_input_index = primal_input_count + position;
                    let Atom::Variable(tangent_type) = &fused.atoms[fused.input_ids[fused_input_index].index()] else {
                        return Err(ProgramError::MalformedProgram(format!(
                            "tangent input {fused_input_index} is not a variable",
                        )));
                    };
                    let restored = AtomId::new(tangent_program.atoms.len());
                    tangent_program.atoms.push(Atom::Variable(tangent_type.clone()));
                    restored
                }
            };
            tangent_program.input_ids.push(restored);
        }
        for atom in edge_inputs.into_iter() {
            tangent_program.input_ids.push(atom.ok_or_else(|| {
                ProgramError::MalformedProgram("a linearization residual edge has no residual-program input".into())
            })?);
        }
        tangent_program.input_structure = vec![Placeholder; tangent_program.input_ids.len()];

        // Restore the canonical tangent outputs: the residual program's outputs are the unknown fused outputs in
        // original order (all within the tangent half, since every primal output is known), and each structurally
        // zero tangent output that folded to the known side is restored as a fresh staged zero of its fused type.
        let surviving_outputs = tangent_program.output_ids.split_off(0);
        let mut survivors = residual_output_indices.into_iter().zip(surviving_outputs).peekable();
        for output in 0..primal_output_count {
            let fused_output_index = primal_output_count + output;
            match survivors.peek() {
                Some(&(index, atom)) if index == fused_output_index => {
                    survivors.next();
                    tangent_program.output_ids.push(atom);
                }
                _ => {
                    let zero_atom = fused.output_ids[fused_output_index];
                    let zero_type = fused.atoms[zero_atom.index()].r#type().into_owned();
                    let zero_output = AtomId::new(tangent_program.atoms.len());
                    tangent_program.atoms.push(Atom::Variable(zero_type.clone()));
                    tangent_program.instructions.push(Instruction::new(
                        O::from(ZeroOperation::new(zero_type)),
                        Vec::new(),
                        vec![zero_output],
                    ));
                    tangent_program.output_ids.push(zero_output);
                }
            }
        }
        tangent_program.output_structure = vec![Placeholder; tangent_program.output_ids.len()];

        Ok(Linearization { primal: known_program, tangent: tangent_program, residual_count })
    }
}

/// The *pushforward* of a function `f` at a linearization point `x` — the linear map `ẋ ↦ (∂f/∂x)(x) · ẋ` — as a
/// reusable callable produced by [`Differentiate::linearize`], the JAX `linearize` analogue. It is the forward-mode
/// dual of [`Pullback`], whose callable applies the transposed map `ȳ ↦ (∂f/∂x)(x)ᵀ · ȳ` instead:
/// [`apply`](Self::apply) pushes any tangent at the linearization point through the function's Jacobian without
/// re-tracing or re-differentiating.
///
/// It wraps the pushforward program `(ẋ, r) ↦ ẏ` accumulated while partially evaluating the closure, closed over the
/// residuals recovered at the linearization point: [`apply`](Self::apply) computes `ẏ = (∂f/∂x)(x) · ẋ` by appending the residuals `r` to the
/// flattened tangents `ẋ`, interpreting the pushforward program `(ẋ, r) ↦ ẏ`, and reshaping the flat tangent outputs
/// against the closure's output structure. The cost of differentiating once is thereby amortized over many tangent
/// applications (for example, replaying every coordinate basis tangent to build a Jacobian). It is the exact
/// forward-mode dual of [`Pullback`], which closes the transposed pullback program over the same residuals.
///
/// The pushforward program is over the primal operation family `<C as Domain>::Operation` in the staged constant
/// space `<C as Domain>::Constant`, while the residuals and tangents flow as `<C as Domain>::Value`s: under an eager
/// domain the residuals are concrete values and [`apply`](Self::apply) interprets the pushforward immediately, while
/// under a staging domain they are [`Tracer`]s into the enclosing trace and [`apply`](Self::apply) stages the
/// pushforward into that trace.
///
/// The differentiation context `C` supplies the value semantics and operation family; `Input` is the closure's
/// structured input type and `TracedOutput` its structured output type, whose
/// [`ParameterStructure`](crate::parameters::Parameterized::ParameterStructure) is retained so the flat tangent outputs
/// reshape back into `TracedOutput::To<<C as Domain>::Value>`. `Input` is carried as a type parameter so
/// [`apply`](Self::apply) infers the tangent family from the linearization itself rather than requiring a turbofish.
pub struct Pushforward<C, Input, TracedOutput>
where
    C: Domain,
    TracedOutput: Parameterized<<C as Domain>::Value>,
{
    /// Differentiation context the linearization was built in; [`apply`](Self::apply) replays the pushforward program
    /// in it, mirroring how [`Pullback`] replays its pullback program.
    context: C,

    /// Pushforward program over the primal operation family in the context's staged [`Constant`](Domain::Constant)
    /// space, mapping `[tangents ++ residuals]` to flat tangent outputs. Its literal constants are lifted through the
    /// context's [`lift`](Context::lift) when [`apply`](Self::apply) replays it.
    program: Program<
        <C as Domain>::Constant,
        <C as Domain>::Operation,
        Vec<<C as Domain>::Constant>,
        Vec<<C as Domain>::Constant>,
    >,

    /// Linearization-point residuals consumed by [`program`](Self::program), appended after the tangents when
    /// interpreting it.
    residuals: Vec<<C as Domain>::Value>,

    /// Parameter structure of the closure's output, used to reshape the flat tangent outputs.
    output_structure: TracedOutput::ParameterStructure,

    /// Encodes the closure's input family `Input` so [`apply`](Self::apply) can flatten the tangents without a
    /// turbofish. Covariant and ownership-free.
    marker: PhantomData<fn() -> Input>,
}

impl<C, Input, TracedOutput> Pushforward<C, Input, TracedOutput>
where
    C: Context,
    <C as Domain>::Operation: Clone,
    Input: Parameterized<<C as Domain>::Value>,
    TracedOutput: Parameterized<<C as Domain>::Value>,
    TracedOutput::Family: ParameterizedFamily<<C as Domain>::Value>,
{
    /// Returns the pushforward program `(ẋ, r) ↦ ẏ` this callable closes over. Its inputs are the flat tangents
    /// followed by the residuals carried by [`residuals`](Self::residuals).
    pub fn program(
        &self,
    ) -> &Program<
        <C as Domain>::Constant,
        <C as Domain>::Operation,
        Vec<<C as Domain>::Constant>,
        Vec<<C as Domain>::Constant>,
    > {
        &self.program
    }

    /// Returns the linearization-point residuals `r` this callable closes over, aligned with the trailing inputs of
    /// [`program`](Self::program).
    pub fn residuals(&self) -> &[<C as Domain>::Value] {
        &self.residuals
    }

    /// Consumes this [`Pushforward`] and returns its open parts: the pushforward program `(ẋ, r) ↦ ẏ` and the
    /// linearization-point residuals `r` its trailing inputs consume, in that order. This is how reverse mode opens
    /// the closure back up: [`Differentiate::vjp`] linearizes, takes the parts, and transposes the program.
    pub fn into_parts(
        self,
    ) -> (
        Program<
            <C as Domain>::Constant,
            <C as Domain>::Operation,
            Vec<<C as Domain>::Constant>,
            Vec<<C as Domain>::Constant>,
        >,
        Vec<<C as Domain>::Value>,
    ) {
        (self.program, self.residuals)
    }

    /// Pushes the structured tangents `tangents` through the linearized Jacobian, returning the tangent outputs.
    ///
    /// The tangents are flattened, the linearization-point residuals are appended, the pushforward program is
    /// interpreted at that vector in the differentiation context this linearization was built in — the single replay
    /// path for both context flavors: an eager domain interprets the pushforward immediately, while a staging domain
    /// stages it into the enclosing trace and returns tracers — and the flat tangent outputs are reshaped against the
    /// closure's output structure.
    ///
    /// # Parameters
    ///
    ///   - `tangents`: Structured tangents at the linearization point, matching the closure's input structure.
    pub fn apply(
        &self,
        tangents: Input::To<<C as Domain>::Value>,
    ) -> Result<TracedOutput::To<<C as Domain>::Value>, ProgramError> {
        let mut inputs = tangents.into_parameters().collect::<Vec<_>>();
        inputs.extend(self.residuals.iter().cloned());
        let tangent_outputs = self.program.interpret_in_context(&self.context, inputs)?;
        Ok(TracedOutput::To::<<C as Domain>::Value>::from_parameters(self.output_structure.clone(), tangent_outputs)?)
    }
}

/// A reusable reverse-mode linear map produced by [`Differentiate::vjp`]: it wraps the pullback program and
/// linearization-point residuals that [`Differentiate::vjp`] returns behind a callable that maps output
/// cotangents to input cotangents — the JAX `vjp` analogue.
///
/// The raw [`vjp`](Differentiate::vjp) returns a pullback program plus the residuals it consumes; reconstructing
/// the input cotangents means appending the residuals to the output cotangents, interpreting the pullback, and reshaping
/// the flat result against the closure's input structure. [`apply`](Self::apply) performs exactly those steps, so callers
/// hold one callable instead of threading the residuals by hand.
///
/// The differentiation context `C` supplies the value semantics and operation family; `Input` is the closure's
/// structured input type, whose [`ParameterStructure`](crate::parameters::Parameterized::ParameterStructure) is retained
/// so the flat input cotangents reshape back into `Input::To<<C as Domain>::Value>`. `TracedOutput` is the closure's structured
/// output type, carried as a type parameter so [`apply`](Self::apply) infers the cotangent family from the pullback
/// itself rather than requiring a turbofish.
pub struct Pullback<C, Input, TracedOutput>
where
    C: Domain,
    Input: Parameterized<<C as Domain>::Value>,
{
    /// Differentiation context the pullback was built in; [`apply`](Self::apply) replays the pullback program in it,
    /// mirroring how [`Pushforward`] replays its pushforward program.
    context: C,

    /// Pullback program over the primal operation family in the context's staged [`Constant`](Domain::Constant)
    /// space, mapping `[output_cotangents ++ residuals]` to flat input cotangents. Its literal constants are lifted
    /// through the context's [`lift`](Context::lift) when [`apply`](Self::apply) replays it.
    program: Program<
        <C as Domain>::Constant,
        <C as Domain>::Operation,
        Vec<<C as Domain>::Constant>,
        Vec<<C as Domain>::Constant>,
    >,

    /// Linearization-point residuals consumed by [`program`](Self::program), appended after the output cotangents when
    /// interpreting it.
    residuals: Vec<<C as Domain>::Value>,

    /// Parameter structure of the closure's input, used to reshape the flat input cotangents.
    input_structure: Input::ParameterStructure,

    /// Encodes the closure's output family `TracedOutput` so [`apply`](Self::apply) can flatten the cotangents without a
    /// turbofish. Covariant and ownership-free.
    marker: PhantomData<fn() -> TracedOutput>,
}

impl<C, Input, TracedOutput> Pullback<C, Input, TracedOutput>
where
    C: Context,
    <C as Domain>::Operation: Clone,
    Input: Parameterized<<C as Domain>::Value>,
    Input::Family: ParameterizedFamily<<C as Domain>::Value>,
    TracedOutput: Parameterized<<C as Domain>::Value>,
{
    /// Returns the pullback program `(ȳ, r) ↦ x̄` this callable closes over. Its inputs are the flat output
    /// cotangents followed by the residuals carried by [`residuals`](Self::residuals).
    pub fn program(
        &self,
    ) -> &Program<
        <C as Domain>::Constant,
        <C as Domain>::Operation,
        Vec<<C as Domain>::Constant>,
        Vec<<C as Domain>::Constant>,
    > {
        &self.program
    }

    /// Returns the linearization-point residuals `r` this callable closes over, aligned with the trailing inputs of
    /// [`program`](Self::program).
    pub fn residuals(&self) -> &[<C as Domain>::Value] {
        &self.residuals
    }

    /// Consumes this [`Pullback`] and returns its open parts: the pullback program `(ȳ, r) ↦ x̄` and the
    /// linearization-point residuals `r` its trailing inputs consume, in that order — the raw form that
    /// [`Differentiate::vjp`] returns directly, mirroring [`Pushforward::into_parts`].
    pub fn into_parts(
        self,
    ) -> (
        Program<
            <C as Domain>::Constant,
            <C as Domain>::Operation,
            Vec<<C as Domain>::Constant>,
            Vec<<C as Domain>::Constant>,
        >,
        Vec<<C as Domain>::Value>,
    ) {
        (self.program, self.residuals)
    }

    /// Pulls the structured output cotangents `cotangents` back to the closure's input cotangents.
    ///
    /// The cotangents are flattened, the linearization-point residuals are appended, the pullback program is
    /// interpreted at that vector in the differentiation context this pullback was built in — the single replay path
    /// for both context flavors: an eager domain (a stateless [`EagerContext`](crate::contexts::EagerContext) or a
    /// backend domain such as a PJRT-client-backed one) interprets the pullback immediately, while a staging domain
    /// stages it into the enclosing trace — and the flat input cotangents are reshaped against the closure's input
    /// structure.
    ///
    /// # Parameters
    ///
    ///   - `cotangents`: Structured output cotangents, matching the closure's output structure.
    pub fn apply(
        &self,
        cotangents: TracedOutput::To<<C as Domain>::Value>,
    ) -> Result<Input::To<<C as Domain>::Value>, ProgramError> {
        let mut inputs = cotangents.into_parameters().collect::<Vec<_>>();
        inputs.extend(self.residuals.iter().cloned());
        let input_cotangents = self.program.interpret_in_context(&self.context, inputs)?;
        Ok(Input::To::<<C as Domain>::Value>::from_parameters(self.input_structure.clone(), input_cotangents)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::Sin;
    use crate::scalars::Scalar;
    use crate::tracing_v2::test_util::assert_scalar_close;

    use super::{Differentiate, LinearizationTracer};
    use crate::contexts::EagerContext;
    use crate::operations::BooleanLike;

    #[test]
    fn test_gradient_and_linearize_support_host_control_flow_on_primals() {
        // JAX-parity marquee behavior: the closure branches on a *primal* with host control flow, which works
        // because the duals' primal halves carry concrete known values under an eager context — exactly like
        // branching on concrete primals under JAX's `grad`/`linearize`. For `x = 3` the predicate is true, so
        // `f(x) = x * x` with gradient `2 x = 6`; the untaken `sin(x)` branch is never traced at all, so no `sin`
        // (nor its `cos` derivative) can appear in the pushforward program.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let function = |x: LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| {
            if x.boolean().unwrap() { x.clone() * x } else { x.sin().unwrap() }
        };
        let (value, gradient) = domain.value_and_gradient(function, Scalar::from(3.0)).unwrap();
        assert_scalar_close(value, 9.0);
        assert_scalar_close(gradient, 6.0);

        let (output, pushforward) = domain.linearize(|x| Ok(function(x)), Scalar::from(3.0)).unwrap();
        assert_scalar_close(output, 9.0);
        let program = pushforward.program.to_string();
        assert!(program.contains("mul"), "{program}");
        assert!(
            !program.contains("sin") && !program.contains("cos"),
            "the untaken branch must never be traced: {program}"
        );
        let tangent = pushforward.apply(Scalar::from(1.0)).unwrap();
        assert_scalar_close(tangent, 6.0);
    }

    #[test]
    fn test_nested_value_and_grad_computes_the_analytic_second_derivative() {
        // Reverse-over-reverse through closure-level nesting: the outer transform differentiates a closure that
        // itself calls `value_and_gradient` on the nested tracing context its tracer flows in. For f(x) = sin(x²),
        // the outer value is f'(x) = 2x cos(x²) and the outer gradient is f''(x) = 2 cos(x²) - 4x² sin(x²).
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (value, gradient) = domain
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    context.gradient(|y| (y.clone() * y).sin().unwrap(), x).unwrap()
                },
                Scalar::from(0.7),
            )
            .unwrap();
        let x: f64 = 0.7;
        assert_scalar_close(value, 2.0 * x * (x * x).cos());
        assert_scalar_close(gradient, 2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin());
    }

    #[test]
    fn test_triple_nested_value_and_grad_computes_the_analytic_third_derivative() {
        // Three levels of closure nesting exercise the recursive `NestedTracingContext<NestedTracingContext<...>>`
        // types through the trait solver. For f(x) = sin(x²), f'''(x) = -12x sin(x²) - 8x³ cos(x²).
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (value, gradient) = domain
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    context
                        .gradient(
                            |y| {
                                let context = y.context().clone();
                                context.gradient(|z| (z.clone() * z).sin().unwrap(), y).unwrap()
                            },
                            x,
                        )
                        .unwrap()
                },
                Scalar::from(0.7),
            )
            .unwrap();
        let x: f64 = 0.7;
        assert_scalar_close(value, 2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin());
        assert_scalar_close(gradient, -12.0 * x * (x * x).sin() - 8.0 * x * x * x * (x * x).cos());
    }

    #[test]
    fn test_jvp_over_nested_gradient_computes_a_hessian_vector_product() {
        // Forward-over-reverse: pushing a tangent through the gradient of f computes the Hessian-vector product
        // f''(x)·v without materializing a dense Hessian. The `jvp` closure receives `DifferentiationTracer` duals whose
        // stamped `DifferentiationContext` is itself a `Differentiate`, so the inner reverse-mode transform nests on
        // it and differentiates through the duals. For f(x) = sin(x²) at x = 0.7 with v = 2, the primal is f'(0.7)
        // and the tangent is 2·f''(0.7).
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = domain
            .jvp(
                |x| {
                    let context = x.context().clone();
                    Ok(context.gradient(|y| (y.clone() * y).sin().unwrap(), x).unwrap())
                },
                Scalar::from(0.7),
                Scalar::from(2.0),
            )
            .unwrap();
        let x: f64 = 0.7;
        assert_scalar_close(primal, 2.0 * x * (x * x).cos());
        assert_scalar_close(tangent, 2.0 * (2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin()));
    }
}
