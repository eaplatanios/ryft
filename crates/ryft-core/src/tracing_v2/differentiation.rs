use std::fmt::Debug;

use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationError,
    DifferentiationTracer, LinearizationTracer, Pullback, Pushforward, TransposableOperation,
};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::{OneOperation, Zero, ZeroOperation};
use crate::parameters::{ParameterError, Parameterized, ParameterizedFamily};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationInput, PartialEvaluationValue, PartialTracer, PartialValue,
    PartiallyEvaluatableOperation,
};
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::types::{Type, Typed};

/// Extension trait carrying the value-level *forward-mode* differentiation transforms on every [`Context`], mirroring
/// how [`Batch`](crate::batching::Batch) carries batching. Its sibling [`ReverseModeDifferentiate`] builds reverse
/// mode on top of it (i.e., `vjp = linearize + transpose`).
///
/// This trait is blanket-implemented for all [`Context`]s and has no items of its own to implement: every entry point
/// is a defaulted method whose `where` clause carries its actual requirements (e.g., the operation family's
/// [`DifferentiableOperation`] rules), so whether a particular transform is available on a particular context is
/// decided per method at the call site, exactly as with [`Batch::batch`](crate::batching::Batch::batch). Tangents are
/// ordinary values of the same universe as the primals (i.e., [`Domain::Value`](crate::contexts::Domain::Value)) flowing through the same context;
/// the descriptor-level tangent structure, such as cotangent types, lives on [`DifferentiableType`] instead.
/// Predicate-capable operations such as `condition`, `while`, and `select` impose their own
/// [`BooleanLike`](crate::operations::BooleanLike) bounds through their operation-family implementations; tangent
/// carriers themselves do not need to be Boolean-like just to participate in differentiation.
///
/// Whether a transform runs eagerly or stages a program is decided by the context's [`Value`](crate::contexts::Domain::Value)
/// (i.e., concrete vs [`Tracer`](crate::tracing::Tracer)), not by a separate trait. Values from a *different* trace
/// are detected lazily, like everything else about staging: a foreign tracer fails the builder-identity check either
/// when an operation binds it ([`StagingContext::stage_operation`](crate::contexts::StagingContext::stage_operation))
/// or when it escapes through a trace boundary (i.e., the boundary output checks), with
/// [`ProgramError::MismatchedProgramBuilders`].
pub trait ForwardModeDifferentiate: Context {
    /// Evaluates `function` on the primal `primals` and propagates the tangent `tangents` forward, with this
    /// [`Context`] executing (or staging) the differentiated operations. Refer to the documentation of the [`jvp`]
    /// function for information on the forward-mode transform and its arguments. This method serves the call sites
    /// that must name the [`Context`] explicitly instead of recovering it from the input's leaf values.
    fn jvp<F, Input, Output>(
        &self,
        function: F,
        primals: Input,
        tangents: Input::To<Self::Value>,
    ) -> Result<(Output::To<Self::Value>, Output::To<Self::Value>), ProgramError>
    where
        Self: Zero<Self::Value>,
        Self::Operation: Clone + DifferentiableOperation<Self>,
        F: FnOnce(Input::To<DifferentiationTracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<DifferentiationTracer<Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<DifferentiationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
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
        let primal_output = Output::To::<Self::Value>::from_parameters(output_structure.clone(), primal_outputs)?;
        let tangent_output = Output::To::<Self::Value>::from_parameters(output_structure, tangent_outputs)?;

        Ok((primal_output, tangent_output))
    }

    /// Linearizes `function` at `primals`, returning the primal output and a reusable [`Pushforward`], with this
    /// [`Context`] executing (or staging) the primal-side operations. Refer to the documentation of the
    /// [`linearize`] function for information on the linearization transform and its arguments. This method serves
    /// the call sites that must name the [`Context`] explicitly instead of recovering it from the input's leaf
    /// values.
    fn linearize<F, Input, Output>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Output::To<Self::Value>, Pushforward<Self, Input, Output::To<Self::Value>>), ProgramError>
    where
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + From<ZeroOperation<Self::Type>>,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Result<Output, ProgramError>,
        Input:
            Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
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
        let staged_zero = |r#type: Self::Type| {
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
        let output = Output::To::<Self::Value>::from_parameters(output_structure.clone(), primal_outputs)?;

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
        let pushforward = Pushforward::new(self.clone(), evaluation.program, residuals, output_structure)?;
        Ok((output, pushforward))
    }
}

impl<C: Context> ForwardModeDifferentiate for C {}

/// Evaluates `function` on the primal `primals` and propagates the tangent `tangents` forward by running the closure
/// **directly on [`DifferentiationTracer`] duals** (i.e., the single forward-mode entry point, and the analogue of
/// [JAX's `jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html)).
///
/// The transform recovers a [`Context`] from the input's leaf values through
/// [`Value::ExecutionDomain`](crate::programs::Value::ExecutionDomain), exactly like [`batch`](crate::batching::batch):
/// staged [`Tracer`](crate::tracing::Tracer)s recover their trace, transform tracers recover their transform level,
/// and concrete values recover the eager backend domain they name, so the transform composes uniformly across the
/// whole stack. Each input is then paired with its tangent as a dual over a [`DifferentiationContext`] wrapping the
/// recovered context, and `function` runs directly on those duals, with each operation the closure performs (e.g.,
/// `x.sin()`, `x * y`, etc.) dispatching its [`jvp`](DifferentiableOperation::jvp) rule through [`Context::bind`].
/// Eager-versus-staged behavior is absorbed entirely by that context:
///
///   - Over an **eager** context both dual halves are concrete, so the closure sees real primal values (i.e., it can
///     branch on them with `if x.boolean()? { … }`, print them, or otherwise use Rust control flow driven by the
///     primal) and a staged data-dependent `while` combinator differentiates by running directly at the concrete
///     primals, with no iteration bound needed.
///   - Over a **staging** context the same closure stages the primal and tangent operations into the enclosing trace
///     operation by operation (this is how a fused JVP computation is built under an outer trace), and branching on a
///     primal errors because it is a [`Tracer`](crate::tracing::Tracer) with no concrete payload.
///
/// The closure executes exactly as written: no dead code is trimmed, and observable effects fire as the closure runs.
/// Structural zero tangents stay symbolic between operations and are materialized through the recovered context's
/// [`Zero`] capability only at the output boundary. Transforms nest: inside the closure, an inner transform invoked
/// on a dual's [`DifferentiationContext`] (a [`Context`] carrying these transforms itself) differentiates through the
/// duals, composing reverse-over-forward and higher-order forward modes.
///
/// Inputs with no leaf values are rejected with an [`InvalidInputCount`](ProgramError::InvalidInputCount) error:
/// there is nothing to recover a context from, and differentiating a function of no inputs is degenerate anyway.
/// [`ForwardModeDifferentiate::jvp`] is the explicit-context method form behind this function.
#[inline]
pub fn jvp<V, F, Input, Output>(
    function: F,
    primals: Input,
    tangents: Input::To<V>,
) -> Result<(Output::To<V>, Output::To<V>), ProgramError>
where
    V: Value<ExecutionDomain: Context + Zero<V>>,
    <V::ExecutionDomain as Domain>::Operation: Clone + DifferentiableOperation<V::ExecutionDomain>,
    F: FnOnce(Input::To<DifferentiationTracer<V::ExecutionDomain>>) -> Result<Output, ProgramError>,
    Input: Parameterized<
            V,
            To<V> = Input,
            Family: ParameterizedFamily<DifferentiationTracer<V::ExecutionDomain>>,
            ParameterStructure: Debug + PartialEq,
        >,
    Output: Parameterized<DifferentiationTracer<V::ExecutionDomain>, Family: ParameterizedFamily<V>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
    };
    context.jvp(function, primals, tangents)
}

/// Linearizes `function` at `primals`, returning the primal output and a reusable [`Pushforward`] (i.e., the analogue
/// of [JAX's `linearize`](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html)).
///
/// This is the partial-evaluation sibling of [`jvp`]: where `jvp` runs the closure once per `(primal, tangent)` pair,
/// this runs the closure once on [`DifferentiationTracer`] duals over a [`PartialEvaluationContext`] wrapping the
/// context recovered from the input's leaf values through
/// [`Value::ExecutionDomain`](crate::programs::Value::ExecutionDomain) (exactly like [`jvp`] and
/// [`batch`](crate::batching::batch)), with each dual's primal half *known* at its primal value and its tangent half
/// *unknown*. Primal-side operations are then all-known and fold through the recovered context itself (i.e.,
/// executing eagerly under an eager context or staging into the enclosing trace under a staging one, so that
/// linearization composes under an outer trace), while tangent-side operations residualize into the accumulated
/// pushforward program `(ẋ, r) ↦ ẏ`, which is linear in `ẋ` with the linearization point entering only through the
/// residuals `r` recovered along the way. The returned [`Pushforward`] closes that program over those residuals, so
/// that [`Pushforward::apply`] pushes any number of tangents through the function's Jacobian at this point without
/// re-tracing or re-differentiating.
///
/// Because the closure's primal halves carry concrete values under an eager context, host control flow on primals
/// works exactly as under [`jvp`]: the closure can branch on a primal (`if x.boolean()? { … }`), the untaken branch
/// is never traced at all, and a data-dependent `while` combinator differentiates by running directly at the concrete
/// primals. This matches JAX's `linearize`/`grad` tracing semantics, where the same JVP interpreter runs over a
/// partial-evaluation trace instead of the eval trace.
///
/// Reverse mode is this transform plus transposition, literally: [`vjp`] calls this, opens the returned
/// [`Pushforward`] back up with [`Pushforward::into_parts`], and transposes its program into the pullback (and the
/// forward-mode Jacobian transform batch-replays it the same way).
///
/// Inputs with no leaf values are rejected with an [`InvalidInputCount`](ProgramError::InvalidInputCount) error:
/// there is nothing to recover a context from, and linearizing a function of no inputs is degenerate anyway.
/// [`ForwardModeDifferentiate::linearize`] is the explicit-context method form behind this function.
#[inline]
pub fn linearize<V, F, Input, Output>(
    function: F,
    primals: Input,
) -> Result<(Output::To<V>, Pushforward<V::ExecutionDomain, Input, Output::To<V>>), ProgramError>
where
    V: Value<ExecutionDomain: Context>,
    <V::ExecutionDomain as Domain>::Operation: Clone
        + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + PartiallyEvaluatableOperation<V::ExecutionDomain>
        + From<ZeroOperation<V::Type>>,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<Output, ProgramError>,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: Parameterized<LinearizationTracer<V::ExecutionDomain>, Family: ParameterizedFamily<V>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
    };
    context.linearize(function, primals)
}

/// Extension trait carrying the value-level *reverse-mode* differentiation transforms on every [`Context`]. Reverse
/// mode is forward mode plus transposition (i.e., JAX's `vjp = linearize + transpose`), which the
/// [`ForwardModeDifferentiate`] supertrait states structurally: [`vjp`](Self::vjp) opens the [`Pushforward`] returned
/// by [`linearize`](ForwardModeDifferentiate::linearize) and transposes its program into a [`Pullback`], and the
/// gradient entry points seed that pullback with the scalar output's multiplicative identity.
///
/// Like its supertrait, this trait is blanket-implemented for all [`Context`]s and has no items of its own to
/// implement: every entry point is a defaulted method whose `where` clause carries its actual requirements. On top of
/// the forward-mode requirements, reverse mode needs the operation family's [`TransposableOperation`] rules, and the
/// gradient entry points additionally need a [`DifferentiableType`] whose scalar outputs carry a cotangent space to
/// seed. Cotangents, like tangents, are ordinary values of the same universe as the primals (i.e., [`Domain::Value`](crate::contexts::Domain::Value))
/// flowing through the same context.
pub trait ReverseModeDifferentiate: ForwardModeDifferentiate {
    /// Reverse-mode-differentiates `function` at `primals`, returning the primal output and a reusable [`Pullback`],
    /// with this [`Context`] executing (or staging) the primal-side operations. Refer to the documentation of the
    /// [`vjp`] function for information on the reverse-mode transform and its arguments. This method serves the call
    /// sites that must name the [`Context`] explicitly instead of recovering it from the input's leaf values.
    fn vjp<F, Input, Output>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Output::To<Self::Value>, Pullback<Self, Input, Output::To<Self::Value>>), ProgramError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<AddOperation>,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Result<Output, ProgramError>,
        Input:
            Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
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
        Ok((output, Pullback::new(self.clone(), program, residuals, input_structure)?))
    }

    /// Computes both the primal scalar output of `function` at `primals` and its reverse-mode gradient, with this
    /// [`Context`] executing (or staging) the primal-side operations and the pullback replay. Refer to the
    /// documentation of the [`value_and_gradient`] function for information on the transform and its arguments. This
    /// method serves the call sites that must name the [`Context`] explicitly instead of recovering it from the
    /// input's leaf values — which is also what lets nested reverse mode compose with any enclosing context.
    fn value_and_gradient<F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Input::To<Self::Value>), DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> LinearizationTracer<Self>,
        Input:
            Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
    {
        // Seed the single output cotangent with the multiplicative identity of the scalar output and pull it back to
        // the input cotangents, reshaped against the closure's input structure.
        let (output, pullback) = self.vjp(|input| Ok(function(input)), primals)?;
        let seed = self.gradient_seed(&output, false)?;
        let gradient = pullback.apply(seed)?;
        Ok((output, gradient))
    }

    /// Computes the reverse-mode gradient of `function` at `primals`, with this [`Context`] executing (or staging)
    /// the primal-side operations and the pullback replay. This is the gradient-only counterpart of
    /// [`value_and_gradient`](Self::value_and_gradient), discarding the primal output; refer to the documentation of
    /// the [`gradient`] function for information on the transform and its arguments.
    #[inline]
    fn gradient<F, Input>(&self, function: F, primals: Input) -> Result<Input::To<Self::Value>, DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> LinearizationTracer<Self>,
        Input:
            Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
    {
        self.value_and_gradient(function, primals).map(|(_, gradient)| gradient)
    }

    /// Computes both the primal scalar output of `function` at `primals` and its holomorphic reverse-mode
    /// gradient, with this [`Context`] executing (or staging) the primal-side operations and the pullback replay.
    /// Refer to the documentation of the [`value_and_gradient_holomorphic`] function for information on the transform,
    /// its arguments, and the holomorphy promise it relies on. This method serves the call sites that must name the
    /// [`Context`] explicitly instead of recovering it from the input's leaf values.
    fn value_and_gradient_holomorphic<F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Input::To<Self::Value>), DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> LinearizationTracer<Self>,
        Input:
            Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
    {
        // Identical to `value_and_gradient` except that the seed is gated on holomorphy: the output must be complex,
        // and under the caller's holomorphy promise the single seed recovers the complex derivative ∂f/∂z.
        let (output, pullback) = self.vjp(|input| Ok(function(input)), primals)?;
        let seed = self.gradient_seed(&output, true)?;
        let gradient = pullback.apply(seed)?;
        Ok((output, gradient))
    }

    /// Computes the holomorphic reverse-mode gradient of `function` at `primals`, with this [`Context`] executing
    /// (or staging) the primal-side operations and the pullback replay. This is the gradient-only counterpart of
    /// [`value_and_gradient_holomorphic`](Self::value_and_gradient_holomorphic), discarding the primal output; refer
    /// to the documentation of the [`gradient_holomorphic`] function for information on the transform, its arguments,
    /// and the holomorphy promise it relies on.
    #[inline]
    fn gradient_holomorphic<F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<Input::To<Self::Value>, DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> LinearizationTracer<Self>,
        Input:
            Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
    {
        self.value_and_gradient_holomorphic(function, primals).map(|(_, gradient)| gradient)
    }

    /// Computes the scalar output of `function` at `primals`, its auxiliary outputs, and its reverse-mode gradient,
    /// with this [`Context`] executing (or staging) the primal-side operations and the pullback replay. Refer to the
    /// documentation of the [`value_and_gradient_with_aux`] function for information on the transform and its
    /// arguments. This method serves the call sites that must name the [`Context`] explicitly instead of recovering
    /// it from the input's leaf values.
    fn value_and_gradient_with_aux<F, Input, Aux>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<((Self::Value, Aux), Input::To<Self::Value>), DifferentiationError>
    where
        Self: Zero<Self::Value>,
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        F: FnOnce(
            Input::To<LinearizationTracer<Self>>,
        ) -> (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>),
        Input:
            Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Aux: Parameterized<
                Self::Value,
                To<Self::Value> = Aux,
                Family: ParameterizedFamily<LinearizationTracer<Self>, To = Aux::To<LinearizationTracer<Self>>>,
            >,
        (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>): Parameterized<
                LinearizationTracer<Self>,
                To<Self::Value> = (Self::Value, Aux),
                Family: ParameterizedFamily<Self::Value>,
            >,
    {
        let input_structure = primals.parameter_structure();
        let ((output, aux), pullback): ((Self::Value, Aux), _) = self.vjp(|input| Ok(function(input)), primals)?;
        let (pullback, residuals) = pullback.into_parts();
        // The flat pullback consumes `[output_cotangents ++ residuals]`. The traced output flattens as the scalar
        // output leaf followed by the auxiliary leaves, so seed the output leaf with a one cotangent and every
        // auxiliary leaf with a zero cotangent, then append the linearization-point residuals. Both the seeds and the
        // replay go through this context itself: an eager context constructs and interprets concrete values, while a
        // staging context stages into its enclosing trace.
        let mut pullback_inputs = vec![self.gradient_seed(&output, false)?];
        for value in Parameterized::<Self::Value>::parameters(&aux) {
            pullback_inputs.push(self.zero(value.r#type().as_ref())?);
        }
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(self, pullback_inputs)?;
        let gradient =
            Input::To::<Self::Value>::from_parameters(input_structure, input_cotangents).map_err(ProgramError::from)?;
        Ok(((output, aux), gradient))
    }

    /// Computes the reverse-mode gradient of `function` at `primals` and its auxiliary outputs, with this [`Context`]
    /// executing (or staging) the primal-side operations and the pullback replay. This is the gradient-only
    /// counterpart of [`value_and_gradient_with_aux`](Self::value_and_gradient_with_aux), discarding the primal
    /// scalar output; refer to the documentation of the [`gradient_with_aux`] function for information on the
    /// transform and its arguments.
    #[inline]
    fn gradient_with_aux<F, Input, Aux>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Input::To<Self::Value>, Aux), DifferentiationError>
    where
        Self: Zero<Self::Value>,
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        F: FnOnce(
            Input::To<LinearizationTracer<Self>>,
        ) -> (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>),
        Input:
            Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Aux: Parameterized<
                Self::Value,
                To<Self::Value> = Aux,
                Family: ParameterizedFamily<LinearizationTracer<Self>, To = Aux::To<LinearizationTracer<Self>>>,
            >,
        (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>): Parameterized<
                LinearizationTracer<Self>,
                To<Self::Value> = (Self::Value, Aux),
                Family: ParameterizedFamily<Self::Value>,
            >,
    {
        self.value_and_gradient_with_aux(function, primals).map(|((_, aux), gradient)| (gradient, aux))
    }

    /// Computes the scalar output of `function` at `primals`, its auxiliary outputs, and its holomorphic
    /// reverse-mode gradient, with this [`Context`] executing (or staging) the primal-side operations and the
    /// pullback replay. Refer to the documentation of the [`value_and_gradient_holomorphic_with_aux`] function for
    /// information on the transform, its arguments, and the holomorphy promise it relies on. This method serves the
    /// call sites that must name the [`Context`] explicitly instead of recovering it from the input's leaf values.
    fn value_and_gradient_holomorphic_with_aux<F, Input, Aux>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<((Self::Value, Aux), Input::To<Self::Value>), DifferentiationError>
    where
        Self: Zero<Self::Value>,
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        F: FnOnce(
            Input::To<LinearizationTracer<Self>>,
        ) -> (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>),
        Input:
            Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Aux: Parameterized<
                Self::Value,
                To<Self::Value> = Aux,
                Family: ParameterizedFamily<LinearizationTracer<Self>, To = Aux::To<LinearizationTracer<Self>>>,
            >,
        (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>): Parameterized<
                LinearizationTracer<Self>,
                To<Self::Value> = (Self::Value, Aux),
                Family: ParameterizedFamily<Self::Value>,
            >,
    {
        // Identical to `value_and_gradient_with_aux` except that the seed is gated on holomorphy: the output must be
        // complex, and under the caller's holomorphy promise the single seed recovers the complex derivative ∂f/∂z.
        let input_structure = primals.parameter_structure();
        let ((output, aux), pullback): ((Self::Value, Aux), _) = self.vjp(|input| Ok(function(input)), primals)?;
        let (pullback, residuals) = pullback.into_parts();
        let mut pullback_inputs = vec![self.gradient_seed(&output, true)?];
        for value in Parameterized::<Self::Value>::parameters(&aux) {
            pullback_inputs.push(self.zero(value.r#type().as_ref())?);
        }
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(self, pullback_inputs)?;
        let gradient =
            Input::To::<Self::Value>::from_parameters(input_structure, input_cotangents).map_err(ProgramError::from)?;
        Ok(((output, aux), gradient))
    }

    /// Computes the holomorphic reverse-mode gradient of `function` at `primals` and its auxiliary outputs, with this
    /// [`Context`] executing (or staging) the primal-side operations and the pullback replay. This is the
    /// gradient-only counterpart of
    /// [`value_and_gradient_holomorphic_with_aux`](Self::value_and_gradient_holomorphic_with_aux), discarding the
    /// primal scalar output; refer to the documentation of the [`gradient_holomorphic_with_aux`] function for
    /// information on the transform, its arguments, and the holomorphy promise it relies on.
    #[inline]
    fn gradient_holomorphic_with_aux<F, Input, Aux>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Input::To<Self::Value>, Aux), DifferentiationError>
    where
        Self: Zero<Self::Value>,
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        F: FnOnce(
            Input::To<LinearizationTracer<Self>>,
        ) -> (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>),
        Input:
            Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Aux: Parameterized<
                Self::Value,
                To<Self::Value> = Aux,
                Family: ParameterizedFamily<LinearizationTracer<Self>, To = Aux::To<LinearizationTracer<Self>>>,
            >,
        (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>): Parameterized<
                LinearizationTracer<Self>,
                To<Self::Value> = (Self::Value, Aux),
                Family: ParameterizedFamily<Self::Value>,
            >,
    {
        self.value_and_gradient_holomorphic_with_aux(function, primals)
            .map(|((_, aux), gradient)| (gradient, aux))
    }

    /// Validates the scalar `output` of a gradient entry point and constructs its cotangent seed. The output must be
    /// a single rank-0 scalar (otherwise
    /// [`NonScalarGradientOutput`](DifferentiationError::NonScalarGradientOutput)) with a cotangent space (otherwise
    /// [`NonDifferentiableGradientOutput`](DifferentiationError::NonDifferentiableGradientOutput)), and complex
    /// outputs additionally require `holomorphic`: a single reverse-mode seed recovers the derivative of a
    /// complex-output function only when the function is holomorphic, so without that promise a complex output is
    /// rejected with [`ComplexGradientOutput`](DifferentiationError::ComplexGradientOutput) instead of silently
    /// computing a value that is not a derivative (`holomorphic` changes nothing for real outputs). The seed is the
    /// multiplicative identity typed with the output's cotangent type (e.g., swapping unreduced and reduced sharding
    /// axes for arrays) and bound through this context, so an eager context constructs a concrete value while a
    /// staging context stages into its enclosing trace.
    ///
    /// This is the shared seeding step behind [`value_and_gradient`](Self::value_and_gradient),
    /// [`value_and_gradient_holomorphic`](Self::value_and_gradient_holomorphic), and
    /// [`value_and_gradient_with_aux`](Self::value_and_gradient_with_aux), exposed so that custom gradient-style
    /// entry points built on [`vjp`](Self::vjp) can reuse the same validation and seeding contract.
    fn gradient_seed(&self, output: &Self::Value, holomorphic: bool) -> Result<Self::Value, DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: From<OneOperation<Self::Type>>,
    {
        let output_type = output.r#type();
        // Reverse mode only defines a gradient for scalar-output functions.
        if !output_type.is_scalar() {
            return Err(DifferentiationError::NonScalarGradientOutput { output_type: output_type.to_string() });
        }
        if !holomorphic && output_type.is_complex() {
            return Err(DifferentiationError::ComplexGradientOutput { output_type: output_type.to_string() });
        }
        // A non-differentiable scalar output (a Boolean or integer, the `float0` analogue) carries no cotangent space
        // and thus no "one" to seed, so reverse mode is degenerate and is rejected up front.
        let output_cotangent_type = output_type.cotangent().ok_or_else(|| {
            DifferentiationError::NonDifferentiableGradientOutput { output_type: output_type.to_string() }
        })?;
        let mut seeds = self.bind(OneOperation::new(output_cotangent_type), &[])?;
        check_count!("output", seeds, 1, ProgramError);
        Ok(seeds.pop().unwrap())
    }
}

impl<C: Context> ReverseModeDifferentiate for C {}

/// Reverse-mode-differentiates `function` at `primals`, returning the primal output and a reusable [`Pullback`]
/// (i.e., the analogue of [JAX's `vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html)).
///
/// This is the value-level reverse-mode transform: [`linearize`] followed by transposition, exactly JAX's
/// `vjp = linearize + transpose`. The closure runs once on [`DifferentiationTracer`] duals over a
/// [`PartialEvaluationContext`] wrapping the context recovered from the input's leaf values through
/// [`Value::ExecutionDomain`](crate::programs::Value::ExecutionDomain) (i.e., primal halves known, tangent halves
/// unknown), which executes the primal work through that context — recovering the primal outputs and the residual
/// values at the linearization point — while accumulating the linear pushforward program `(ẋ, r) ↦ ẏ`; that program
/// is then transposed with respect to its leading tangent inputs, holding the trailing residuals as known parameters.
/// The resulting pullback program stays in the recovered context's staged
/// [`Constant`](crate::contexts::Domain::Constant) space; interpreting it through
/// [`Program::interpret_in_context`](crate::Program::interpret_in_context) lifts its literal constants through that
/// context's [`lift`](Context::lift) at replay time, which is what serves reverse mode *under tracing*: in an eager
/// context the lift is the identity, while in a staging context (whose values are [`Tracer`](crate::tracing::Tracer)s)
/// it records the pullback's constants in the enclosing trace, so that the backward pass splices into that trace.
/// Host control flow on primals works exactly as under [`linearize`] (i.e., JAX's `grad`-allows-Python-control-flow
/// property).
///
/// The returned [`Pullback`] closes the transposed program over the linearization-point residuals, so that
/// [`Pullback::apply`] maps output cotangents to input cotangents — appending the residuals, interpreting the
/// program, and reshaping the flat input cotangents against the closure's input structure — without the caller
/// threading the residuals by hand. Consumers that need the open parts (e.g., to batch-replay or seed the pullback
/// program manually) recover them with [`Pullback::into_parts`].
///
/// Functions reaching operations outside the supported straight-line slice fail with an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error, and inputs with no leaf values are rejected
/// with an [`InvalidInputCount`](ProgramError::InvalidInputCount) error, exactly as under [`linearize`].
/// [`ReverseModeDifferentiate::vjp`] is the explicit-context method form behind this function.
#[inline]
pub fn vjp<V, F, Input, Output>(
    function: F,
    primals: Input,
) -> Result<(Output::To<V>, Pullback<V::ExecutionDomain, Input, Output::To<V>>), ProgramError>
where
    V: Value<Type: DifferentiableType, ExecutionDomain: Context>,
    <V::ExecutionDomain as Domain>::Operation: Clone
        + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + PartiallyEvaluatableOperation<V::ExecutionDomain>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<AddOperation>,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<Output, ProgramError>,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: Parameterized<LinearizationTracer<V::ExecutionDomain>, Family: ParameterizedFamily<V>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
    };
    context.vjp(function, primals)
}

/// Computes both the primal scalar output of `function` at `primals` and its reverse-mode gradient.
///
/// This is the most direct reverse-mode API when the caller needs both the function value and the gradient at the
/// same primal point: [`vjp`] followed by seeding the returned [`Pullback`] with the multiplicative identity of the
/// scalar output, typed with the output's cotangent type (e.g., swapping unreduced and reduced sharding axes for
/// arrays), and pulling that seed back to the input cotangents, reshaped against the closure's input structure. The
/// function must return exactly one rank-0 real scalar leaf: a non-scalar output is rejected with
/// [`DifferentiationError::NonScalarGradientOutput`], a non-differentiable scalar output (i.e., a Boolean or an
/// integer, the `float0` analogue, whose cotangent space carries no "one" to seed) with
/// [`DifferentiationError::NonDifferentiableGradientOutput`], and a complex scalar output with
/// [`DifferentiationError::ComplexGradientOutput`], because the single seed recovers a complex derivative only for
/// holomorphic functions (use [`value_and_gradient_holomorphic`] to promise holomorphy). Complex *inputs* with a real
/// scalar output are supported directly: under the bilinear transposition pairing documented on
/// [`Program::transpose`](crate::Program::transpose), the gradient returned for such a ℂ → ℝ function is `2 · ∂f/∂z̄`
/// (e.g., `2·z̄` for `f(z) = |z|²`), the steepest-ascent direction and the same value JAX's `grad` returns. Use
/// [`vjp`] directly for vector-valued functions that need an explicit output cotangent, and
/// [`value_and_gradient_with_aux`](crate::tracing_v2::value_and_gradient_with_aux) when the function carries
/// auxiliary outputs.
///
/// Like [`vjp`], the transform recovers a [`Context`] from the input's leaf values through
/// [`Value::ExecutionDomain`](crate::programs::Value::ExecutionDomain), and host control flow on primals works
/// exactly as under [`linearize`]. [`ReverseModeDifferentiate::value_and_gradient`] is the explicit-context method
/// form behind this function.
#[inline]
pub fn value_and_gradient<V, F, Input>(function: F, primals: Input) -> Result<(V, Input::To<V>), DifferentiationError>
where
    V: Value<Type: DifferentiableType, ExecutionDomain: Context>,
    <V::ExecutionDomain as Domain>::Operation: Clone
        + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + PartiallyEvaluatableOperation<V::ExecutionDomain>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<OneOperation<V::Type>>
        + From<AddOperation>,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> LinearizationTracer<V::ExecutionDomain>,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    context.value_and_gradient(function, primals)
}

/// Computes the reverse-mode gradient of `function` at `primals` (i.e., the analogue of
/// [JAX's `grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html)). This is the gradient-only counterpart
/// of [`value_and_gradient`], discarding the primal output; refer to its documentation for information on the
/// transform, its arguments, and its scalar-output requirements. Use
/// [`gradient_with_aux`](crate::tracing_v2::gradient_with_aux) when the function carries auxiliary outputs.
/// [`ReverseModeDifferentiate::gradient`] is the explicit-context method form behind this function.
#[inline]
pub fn gradient<V, F, Input>(function: F, primals: Input) -> Result<Input::To<V>, DifferentiationError>
where
    V: Value<Type: DifferentiableType, ExecutionDomain: Context>,
    <V::ExecutionDomain as Domain>::Operation: Clone
        + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + PartiallyEvaluatableOperation<V::ExecutionDomain>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<OneOperation<V::Type>>
        + From<AddOperation>,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> LinearizationTracer<V::ExecutionDomain>,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
{
    value_and_gradient(function, primals).map(|(_, gradient)| gradient)
}

/// Computes both the primal scalar output of `function` at `primals` and its holomorphic reverse-mode
/// gradient (i.e., the analogue of
/// [JAX's `grad(f, holomorphic=True)`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html)).
///
/// This is [`value_and_gradient`] with the complex-output guard lifted: the caller *promises* that `function` is
/// holomorphic (i.e., complex-differentiable), and under that promise the single reverse-mode seed `1` pulled back
/// through the transposed pushforward (under the bilinear pairing documented on
/// [`Program::transpose`](crate::Program::transpose)) recovers the complex derivative `∂f/∂z`, exactly as it recovers
/// the gradient of a real-valued function. The promise is not (and cannot be) checked: for a non-holomorphic function with a complex
/// output the result is not a derivative in any useful sense, so split such a function into its real and imaginary
/// parts and differentiate those instead. For real outputs the promise changes nothing and this behaves exactly like
/// [`value_and_gradient`]; the [`NonScalarGradientOutput`](DifferentiationError::NonScalarGradientOutput) rejection
/// applies as usual.
///
/// Like [`value_and_gradient`], the transform recovers a [`Context`] from the input's leaf values through
/// [`Value::ExecutionDomain`](crate::programs::Value::ExecutionDomain), and host control flow on primals works
/// exactly as under [`linearize`]. Use [`value_and_gradient_holomorphic_with_aux`] when the function carries
/// auxiliary outputs. [`ReverseModeDifferentiate::value_and_gradient_holomorphic`] is the explicit-context method
/// form behind this function.
#[inline]
pub fn value_and_gradient_holomorphic<V, F, Input>(
    function: F,
    primals: Input,
) -> Result<(V, Input::To<V>), DifferentiationError>
where
    V: Value<Type: DifferentiableType, ExecutionDomain: Context>,
    <V::ExecutionDomain as Domain>::Operation: Clone
        + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + PartiallyEvaluatableOperation<V::ExecutionDomain>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<OneOperation<V::Type>>
        + From<AddOperation>,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> LinearizationTracer<V::ExecutionDomain>,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    context.value_and_gradient_holomorphic(function, primals)
}

/// Computes the holomorphic reverse-mode gradient of `function` at `primals`. This is the gradient-only counterpart
/// of [`value_and_gradient_holomorphic`], discarding the primal output; refer to its documentation for information on
/// the transform, its arguments, and the holomorphy promise it relies on.
/// [`ReverseModeDifferentiate::gradient_holomorphic`] is the explicit-context method form behind this function.
#[inline]
pub fn gradient_holomorphic<V, F, Input>(function: F, primals: Input) -> Result<Input::To<V>, DifferentiationError>
where
    V: Value<Type: DifferentiableType, ExecutionDomain: Context>,
    <V::ExecutionDomain as Domain>::Operation: Clone
        + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + PartiallyEvaluatableOperation<V::ExecutionDomain>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<OneOperation<V::Type>>
        + From<AddOperation>,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> LinearizationTracer<V::ExecutionDomain>,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
{
    value_and_gradient_holomorphic(function, primals).map(|(_, gradient)| gradient)
}

/// Computes the scalar output of `function` at `primals`, its auxiliary outputs, and its reverse-mode gradient.
///
/// This is the `has_aux` counterpart of [`value_and_gradient`], spelled as a separate function because the auxiliary
/// outputs change the closure's return type: the differentiated value is the first element returned by `function` and
/// must be exactly one rank-0 scalar leaf (subject to the same
/// [`NonScalarGradientOutput`](DifferentiationError::NonScalarGradientOutput) and
/// [`NonDifferentiableGradientOutput`](DifferentiationError::NonDifferentiableGradientOutput) rejections), while the
/// auxiliary leaves are returned to the caller as ordinary primal values but seeded with typed zero cotangents when
/// the pullback is interpreted, so that they do not contribute to the gradient. The primal value and auxiliary data
/// are returned as `((value, aux), gradient)`.
///
/// Like [`value_and_gradient`], the transform recovers a [`Context`] from the input's leaf values through
/// [`Value::ExecutionDomain`](crate::programs::Value::ExecutionDomain), and host control flow on primals works
/// exactly as under [`linearize`]. [`ReverseModeDifferentiate::value_and_gradient_with_aux`] is the explicit-context
/// method form behind this function.
#[inline]
pub fn value_and_gradient_with_aux<V, F, Input, Aux>(
    function: F,
    primals: Input,
) -> Result<((V, Aux), Input::To<V>), DifferentiationError>
where
    V: Value<Type: DifferentiableType, ExecutionDomain: Context + Zero<V>>,
    <V::ExecutionDomain as Domain>::Operation: Clone
        + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + PartiallyEvaluatableOperation<V::ExecutionDomain>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<OneOperation<V::Type>>
        + From<AddOperation>,
    F: FnOnce(
        Input::To<LinearizationTracer<V::ExecutionDomain>>,
    ) -> (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>),
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Aux: Parameterized<
            V,
            To<V> = Aux,
            Family: ParameterizedFamily<
                LinearizationTracer<V::ExecutionDomain>,
                To = Aux::To<LinearizationTracer<V::ExecutionDomain>>,
            >,
        >,
    (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>):
        Parameterized<LinearizationTracer<V::ExecutionDomain>, To<V> = (V, Aux), Family: ParameterizedFamily<V>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    context.value_and_gradient_with_aux(function, primals)
}

/// Computes the reverse-mode gradient of `function` at `primals` and its auxiliary outputs. This is the gradient-only
/// counterpart of [`value_and_gradient_with_aux`], discarding the primal scalar output; refer to its documentation
/// for information on the transform, its arguments, and its scalar-output requirements. The return order is
/// `(gradient, aux)`, matching the common use case where auxiliary outputs are diagnostics or cached intermediates
/// and the gradient remains the primary result. [`ReverseModeDifferentiate::gradient_with_aux`] is the
/// explicit-context method form behind this function.
#[inline]
pub fn gradient_with_aux<V, F, Input, Aux>(
    function: F,
    primals: Input,
) -> Result<(Input::To<V>, Aux), DifferentiationError>
where
    V: Value<Type: DifferentiableType, ExecutionDomain: Context + Zero<V>>,
    <V::ExecutionDomain as Domain>::Operation: Clone
        + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + PartiallyEvaluatableOperation<V::ExecutionDomain>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<OneOperation<V::Type>>
        + From<AddOperation>,
    F: FnOnce(
        Input::To<LinearizationTracer<V::ExecutionDomain>>,
    ) -> (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>),
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Aux: Parameterized<
            V,
            To<V> = Aux,
            Family: ParameterizedFamily<
                LinearizationTracer<V::ExecutionDomain>,
                To = Aux::To<LinearizationTracer<V::ExecutionDomain>>,
            >,
        >,
    (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>):
        Parameterized<LinearizationTracer<V::ExecutionDomain>, To<V> = (V, Aux), Family: ParameterizedFamily<V>>,
{
    value_and_gradient_with_aux(function, primals).map(|((_, aux), gradient)| (gradient, aux))
}

/// Computes the scalar output of `function` at `primals`, its auxiliary outputs, and its holomorphic
/// reverse-mode gradient.
///
/// This is [`value_and_gradient_with_aux`] with the complex-output guard lifted, exactly as
/// [`value_and_gradient_holomorphic`] lifts it for [`value_and_gradient`]: the caller *promises* that the
/// differentiated first element returned by `function` is holomorphic in the inputs (refer to
/// [`value_and_gradient_holomorphic`] for what the promise means and when it can be made), it must be exactly one
/// rank-0 scalar leaf, and the auxiliary leaves are returned as ordinary primal values seeded with typed zero
/// cotangents, so that they do not contribute to the gradient. The primal value and auxiliary data are returned as
/// `((value, aux), gradient)`.
///
/// Like every free entry point in this module, the transform recovers a [`Context`] from the input's leaf values
/// through [`Value::ExecutionDomain`](crate::programs::Value::ExecutionDomain).
/// [`ReverseModeDifferentiate::value_and_gradient_holomorphic_with_aux`] is the explicit-context method form behind
/// this function.
#[inline]
pub fn value_and_gradient_holomorphic_with_aux<V, F, Input, Aux>(
    function: F,
    primals: Input,
) -> Result<((V, Aux), Input::To<V>), DifferentiationError>
where
    V: Value<Type: DifferentiableType, ExecutionDomain: Context + Zero<V>>,
    <V::ExecutionDomain as Domain>::Operation: Clone
        + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + PartiallyEvaluatableOperation<V::ExecutionDomain>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<OneOperation<V::Type>>
        + From<AddOperation>,
    F: FnOnce(
        Input::To<LinearizationTracer<V::ExecutionDomain>>,
    ) -> (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>),
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Aux: Parameterized<
            V,
            To<V> = Aux,
            Family: ParameterizedFamily<
                LinearizationTracer<V::ExecutionDomain>,
                To = Aux::To<LinearizationTracer<V::ExecutionDomain>>,
            >,
        >,
    (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>):
        Parameterized<LinearizationTracer<V::ExecutionDomain>, To<V> = (V, Aux), Family: ParameterizedFamily<V>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
    };
    context.value_and_gradient_holomorphic_with_aux(function, primals)
}

/// Computes the holomorphic reverse-mode gradient of `function` at `primals` and its auxiliary outputs. This is the
/// gradient-only counterpart of [`value_and_gradient_holomorphic_with_aux`], discarding the primal scalar output;
/// refer to its documentation for information on the transform, its arguments, and the holomorphy promise it relies
/// on. The return order is `(gradient, aux)`, matching [`gradient_with_aux`].
/// [`ReverseModeDifferentiate::gradient_holomorphic_with_aux`] is the explicit-context method form behind this
/// function.
#[inline]
pub fn gradient_holomorphic_with_aux<V, F, Input, Aux>(
    function: F,
    primals: Input,
) -> Result<(Input::To<V>, Aux), DifferentiationError>
where
    V: Value<Type: DifferentiableType, ExecutionDomain: Context + Zero<V>>,
    <V::ExecutionDomain as Domain>::Operation: Clone
        + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
        + PartiallyEvaluatableOperation<V::ExecutionDomain>
        + TransposableOperation<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>
        + From<ZeroOperation<V::Type>>
        + From<OneOperation<V::Type>>
        + From<AddOperation>,
    F: FnOnce(
        Input::To<LinearizationTracer<V::ExecutionDomain>>,
    ) -> (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>),
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Aux: Parameterized<
            V,
            To<V> = Aux,
            Family: ParameterizedFamily<
                LinearizationTracer<V::ExecutionDomain>,
                To = Aux::To<LinearizationTracer<V::ExecutionDomain>>,
            >,
        >,
    (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>):
        Parameterized<LinearizationTracer<V::ExecutionDomain>, To<V> = (V, Aux), Family: ParameterizedFamily<V>>,
{
    value_and_gradient_holomorphic_with_aux(function, primals).map(|((_, aux), gradient)| (gradient, aux))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::Sin;
    use crate::scalars::Scalar;
    use crate::tracing::{DomainTracer, DomainTracingContext};
    use crate::tracing_v2::test_util::assert_scalar_close;
    use crate::types::{DataType, Typed};

    use super::{ForwardModeDifferentiate, LinearizationTracer, ReverseModeDifferentiate};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::DifferentiationError;
    use crate::operations::BooleanLike;
    use crate::programs::ProgramError;

    #[test]
    fn test_traced_value_and_grad_requires_input_leaves() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let empty_primals: Vec<DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>> = Vec::new();

        let result = context.value_and_gradient(
            |_inputs: Vec<_>| panic!("closure should not run without traced inputs"),
            empty_primals,
        );

        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 }))
        ));
    }

    #[test]
    fn test_traced_value_and_grad_rejects_mismatched_program_builders() {
        // Mixing tracers of two different traces is rejected with `MismatchedProgramBuilders`. The closure runs on
        // differentiation duals whose operator sugar has no deferral point of its own, so the partial-evaluation
        // context defers the failed bind by poisoning its outputs, and the original error surfaces as a plain `Err`
        // at the evaluation boundary.
        let context_a = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let context_b = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let primal_a = context_a.input(DataType::F64);
        let primal_b = context_b.input(DataType::F64);

        let result =
            context_a.value_and_gradient(|inputs| inputs[0].clone() + inputs[1].clone(), vec![primal_a, primal_b]);

        assert!(matches!(result, Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders))));
    }

    #[test]
    fn test_traced_value_and_grad_invokes_function_once() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let primal = context.input(DataType::F64);
        let calls = Cell::new(0);

        let (_value, gradient): (
            DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
            Vec<DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>>,
        ) = context
            .value_and_gradient(
                |inputs| {
                    calls.set(calls.get() + 1);
                    inputs[0].clone() * inputs[0].clone()
                },
                vec![primal],
            )
            .unwrap();

        assert_eq!(calls.get(), 1);
        assert_eq!(gradient.len(), 1);
    }

    #[test]
    fn test_value_and_grad_with_aux_ignores_aux_cotangents() {
        // The free `value_and_gradient_with_aux` recovers the eager scalar domain from the concrete primals; the
        // auxiliary outputs are returned as primal values and receive zero cotangent seeds, so they do not
        // contribute to the gradient of f(x, y) = x * y.
        let ((value, aux), gradient): ((Scalar, (Scalar, Scalar)), (Scalar, Scalar)) =
            super::value_and_gradient_with_aux(
                |(x, y)| {
                    let value = x.clone() * y.clone();
                    let aux = (x.clone() + y, x.clone() * x);
                    (value, aux)
                },
                (Scalar::from(2.0), Scalar::from(3.0)),
            )
            .unwrap();

        assert_eq!(value, 6.0);
        assert_eq!(aux, (Scalar::from(5.0), Scalar::from(4.0)));
        assert_eq!(gradient, (Scalar::from(3.0), Scalar::from(2.0)));
    }

    #[test]
    fn test_grad_returns_only_the_gradient() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();

        let gradient: (Scalar, Scalar) =
            domain.gradient(|(x, y)| x.clone() * y.clone() + x, (Scalar::from(2.0), Scalar::from(3.0))).unwrap();

        assert_eq!(gradient, (Scalar::from(4.0), Scalar::from(2.0)));
    }

    #[test]
    fn test_gradient_routes_complex_outputs_through_the_holomorphic_entry_points() {
        // The identity function flows types without executing any complex arithmetic, so the complex-output guards
        // are exercised at the type level under a tracing context. A complex output through the plain entry point is
        // rejected toward the holomorphic one.
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let primal = context.input(DataType::C64);
        let result = context.value_and_gradient(|inputs: Vec<_>| inputs[0].clone(), vec![primal]);
        assert!(matches!(
            result,
            Err(DifferentiationError::ComplexGradientOutput { output_type }) if output_type == "c64",
        ));

        // The holomorphic entry point accepts the complex output and seeds `one` at the complex cotangent type.
        let primal = context.input(DataType::C64);
        let (value, gradient) =
            context.value_and_gradient_holomorphic(|inputs: Vec<_>| inputs[0].clone(), vec![primal]).unwrap();
        assert_eq!(*value.r#type(), DataType::C64);
        assert_eq!(gradient.len(), 1);
        assert_eq!(*gradient[0].r#type(), DataType::C64);

        // For real outputs the holomorphy promise changes nothing and the holomorphic entry point behaves exactly
        // like the plain one.
        let primal = context.input(DataType::F64);
        let (value, gradient) =
            context.value_and_gradient_holomorphic(|inputs: Vec<_>| inputs[0].clone(), vec![primal]).unwrap();
        assert_eq!(*value.r#type(), DataType::F64);
        assert_eq!(gradient.len(), 1);
        assert_eq!(*gradient[0].r#type(), DataType::F64);

        // The auxiliary-output cross variants share the same holomorphy gate: a complex output with an auxiliary
        // output is accepted end to end.
        type TestTracer = DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>;
        let primal = context.input(DataType::C64);
        let ((value, aux), gradient): ((TestTracer, TestTracer), Vec<TestTracer>) = context
            .value_and_gradient_holomorphic_with_aux(
                |inputs: Vec<_>| (inputs[0].clone(), inputs[0].clone()),
                vec![primal],
            )
            .unwrap();
        assert_eq!(*value.r#type(), DataType::C64);
        assert_eq!(*aux.r#type(), DataType::C64);
        assert_eq!(gradient.len(), 1);
        assert_eq!(*gradient[0].r#type(), DataType::C64);
    }

    #[test]
    fn test_grad_with_aux_returns_gradient_and_aux() {
        let (gradient, aux): ((Scalar, Scalar), Scalar) =
            super::gradient_with_aux(|(x, y)| (x.clone() * y.clone(), x + y), (Scalar::from(2.0), Scalar::from(3.0)))
                .unwrap();

        assert_eq!(gradient, (Scalar::from(3.0), Scalar::from(2.0)));
        assert_eq!(aux, 5.0);
    }

    #[test]
    fn test_holomorphic_gradient_computes_complex_derivatives() {
        use num_complex::Complex;

        // The holomorphic entry points recover the complex derivative ∂f/∂z from the single reverse-mode seed under
        // the holomorphy promise: d/dz z² = 2z and d/dz sin(z) = cos(z), evaluated at a genuinely complex point.
        let z = Complex::new(0.7f64, -0.3f64);
        let (value, gradient) = super::value_and_gradient_holomorphic(|x| x.clone() * x, Scalar::from(z)).unwrap();
        assert_eq!(value, Scalar::from(z * z));
        assert_eq!(gradient, Scalar::from(z + z));
        let gradient = super::gradient_holomorphic(|x| x.sin().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(gradient, Scalar::from(z.cos()));

        // Forward mode agrees: the jvp of z² pushes the tangent ż to 2z · ż through the same rules.
        let tangent_seed = Complex::new(1.0f64, 0.5f64);
        let (primal, tangent) = super::jvp(|x| Ok(x.clone() * x), Scalar::from(z), Scalar::from(tangent_seed)).unwrap();
        assert_eq!(primal, Scalar::from(z * z));
        assert_eq!(tangent, Scalar::from((z + z) * tangent_seed));

        // The plain entry point keeps rejecting the complex output toward the holomorphic ones.
        let result = super::value_and_gradient(|x| x.clone() * x, Scalar::from(z));
        assert!(matches!(
            result,
            Err(DifferentiationError::ComplexGradientOutput { output_type }) if output_type == "c128",
        ));
    }

    #[test]
    fn test_free_functions_recover_the_context_from_input_leaves() {
        // The free entry points recover their context from the inputs' `Value::ExecutionDomain` (a concrete `Scalar`
        // names the eager scalar domain), so no explicit context is threaded. Every entry point differentiates
        // f(x) = x * sin(x), whose derivative is f'(x) = sin(x) + x cos(x), at x = 0.7.
        let x: f64 = 0.7;
        let expected_value = x * x.sin();
        let expected_gradient = x.sin() + x * x.cos();
        let function = |input: LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| {
            input.clone() * input.sin().unwrap()
        };

        let (value, gradient) = super::value_and_gradient(function, Scalar::from(x)).unwrap();
        assert_scalar_close(value, expected_value);
        assert_scalar_close(gradient, expected_gradient);
        let gradient = super::gradient(function, Scalar::from(x)).unwrap();
        assert_scalar_close(gradient, expected_gradient);

        let (value, pushforward) = super::linearize(|input| Ok(function(input)), Scalar::from(x)).unwrap();
        assert_scalar_close(value, expected_value);
        assert_scalar_close(pushforward.apply(Scalar::from(2.0)).unwrap(), 2.0 * expected_gradient);

        let (value, pullback) = super::vjp(|input| Ok(function(input)), Scalar::from(x)).unwrap();
        assert_scalar_close(value, expected_value);
        assert_scalar_close(pullback.apply(Scalar::from(3.0)).unwrap(), 3.0 * expected_gradient);

        let (value, tangent) =
            super::jvp(|input| Ok(input.clone() * input.sin().unwrap()), Scalar::from(x), Scalar::from(2.0)).unwrap();
        assert_scalar_close(value, expected_value);
        assert_scalar_close(tangent, 2.0 * expected_gradient);

        // With no leaf value to recover a context from, the free entry points report an invalid input count (the
        // closure is never invoked).
        let empty = super::linearize::<Scalar, _, (), ()>(|_| Ok(()), ());
        assert!(matches!(empty, Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 })));
    }

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
        let program = pushforward.program().to_string();
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
        // stamped `DifferentiationContext` is itself a `ReverseModeDifferentiate`, so the inner reverse-mode transform nests on
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
