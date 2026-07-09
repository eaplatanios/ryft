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
/// [`Value`](Domain::Value) (concrete vs [`Tracer`](crate::tracing::Tracer)), not by a separate trait. Values from a *different* trace are
/// detected lazily, like everything else about staging: a foreign tracer fails the builder-identity check either
/// when an operation binds it ([`StagingContext::stage_operation`](crate::contexts::StagingContext::stage_operation)) or when it escapes through a trace boundary
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
    ///     primal errors because it is a [`Tracer`](crate::tracing::Tracer) with no concrete payload.
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
        let pushforward = Pushforward::new(self.clone(), evaluation.program, residuals, output_structure)?;
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
    /// the lift is the identity, while in a staging context (whose values are [`Tracer`](crate::tracing::Tracer)s) it records the pullback's
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
        Ok((output, Pullback::new(self.clone(), program, residuals, input_structure)?))
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
