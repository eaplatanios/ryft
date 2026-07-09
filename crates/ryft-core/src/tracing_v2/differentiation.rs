use std::fmt::Debug;
use std::rc::Rc;

use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDual, DifferentiationError,
    DifferentiationTracer, Linearization, LinearizationTracer, Pullback, Pushforward, TransposableOperation,
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

impl<
    V: Value,
    O: Clone + Operation<V::Type> + DifferentiableOperation<TracingContext<V, O>> + From<ZeroOperation<V::Type>>,
> Program<V, O, Vec<V>, Vec<V>>
{
    /// Builds the *fused* jvp program of this already-traced flat primal [`Program`].
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

impl<V, O> Program<V, O, Vec<V>, Vec<V>>
where
    V: Value,
    O: Clone
        + Operation<V::Type>
        + PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + From<ZeroOperation<V::Type>>,
{
    /// Builds the linearization core from this already-traced flat primal [`Program`] by fusing the forward-mode
    /// rules into one jvp program and splitting it into the primal (known) and tangent (unknown) halves through the
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
    /// [`PartialEvaluationContext::fold_or_residualize`].
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

        Linearization::new(known_program, tangent_program, residual_count)
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
