use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

use thiserror::Error;

use crate::compilation::context::CapturingContext;
use crate::contexts::{Context, Domain, EagerContext, StagingContext};
use crate::differentiation::{DifferentiableType, TransposableOperation};
use crate::interpretation::InterpretableOperation;
use crate::macros::{check_builders, check_count};
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::{Constant, Fill, Iota, One, OneOperation, Zero, ZeroOperation};
use crate::operations::control_flow::MaybeWhile;
use crate::operations::{BooleanLike, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{
    PartialEvaluation, PartialEvaluationInput, PartialEvaluationOutput, PartialValue, PartiallyEvaluatableOperation,
    PartitionedProgram,
};
use crate::programs::{Atom, AtomId, Instruction, MaybeZero, Program, ProgramError, Value};
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};
use crate::tracing_v2::unroll::unroll_concretizable_whiles;
use crate::types::{Type, Typed};

/// Errors emitted by the differentiation helpers in [`crate::tracing_v2`].
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
    /// Reverse-mode differentiation (`grad`/`value_and_grad`) was requested for a function whose output is not a
    /// single scalar. Reverse mode seeds the output cotangent with the multiplicative identity ("one") and pulls it
    /// back to the inputs, which yields a gradient only when the output is a rank-0 scalar. A non-scalar output
    /// describes a vector-valued function whose full derivative is a Jacobian; because program interpretation binds
    /// inputs positionally without checking their types, seeding such an output with a ones cotangent would not
    /// fail but would instead silently compute the gradient of the sum of the outputs, so the gradient entry points
    /// reject it up front. Use a Jacobian transform such as `jacrev`/`jacfwd` for non-scalar outputs.
    #[error("gradient output must be a rank-0 scalar but got {output_type}")]
    NonScalarGradientOutput {
        /// Rendered [`Type`] of the offending non-scalar output.
        output_type: String,
    },

    /// Reverse-mode differentiation (`grad`/`value_and_grad`) was requested for a function whose output type carries no
    /// cotangent space (a non-differentiable type such as a Boolean or integer scalar, the
    /// [`float0`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.float0.html) analogue). Reverse mode seeds the
    /// output cotangent with the multiplicative identity ("one"), but a non-differentiable output has no "one" to seed,
    /// so the gradient is degenerate and the gradient entry points reject it up front rather than fabricating a seed.
    #[error("gradient output type {output_type} is non-differentiable and carries no cotangent space")]
    NonDifferentiableGradientOutput {
        /// Rendered [`Type`] of the offending non-differentiable output.
        output_type: String,
    },

    /// A program-level error surfaced while differentiating.
    #[error(transparent)]
    Program(#[from] ProgramError),
}

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
/// [`DispatchDomain::Value`] (concrete vs [`Tracer`]), not by a separate trait. Values from a *different* trace are
/// detected lazily, like everything else about staging: a foreign tracer fails the builder-identity check either
/// when an operation binds it ([`StagingContext::stage_operation`]) or when it escapes through a trace boundary
/// (the boundary output checks), with [`ProgramError::MismatchedProgramBuilders`].
pub trait DifferentiationContext: Context {
    /// Traces `function` into a flat primal [`Program`] over this context's types.
    ///
    /// This is the shared tracing prologue of the program-level entry points that consume a primal program:
    /// [`linearize`](Self::linearize) and the reverse-mode [`vjp`](Self::vjp) family (transposition consumes a
    /// program, so reverse mode always traces first; forward-mode [`jvp`](Self::jvp) runs the closure directly on
    /// duals instead). The closure runs inside a [`NestedTracingContext`]
    /// over this context, so runtime captures registered while tracing delegate to this context, and every operation
    /// is staged without running any differentiation rule. The traced program is then simplified so closure dead code
    /// is dropped before linearization. Returns the simplified program, the closure's output structure, and the
    /// primal input values aligned with the program's input atoms.
    ///
    /// # Parameters
    ///
    ///   - `function`: Closure traced into a primal program.
    ///   - `primals`: Structured primal input values; their count must be non-zero.
    fn trace_into_primal_program<F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        (
            Program<
                <Self as Domain>::Constant,
                <Self as Domain>::Operation,
                Vec<<Self as Domain>::Constant>,
                Vec<<Self as Domain>::Constant>,
            >,
            Input::ParameterStructure,
            TracedOutput::ParameterStructure,
            Vec<<Self as Domain>::Value>,
        ),
        ProgramError,
    >
    where
        <Self as Domain>::Operation: Clone,
        F: FnOnce(Input::To<Tracer<NestedTracingContext<Self>>>) -> Result<TracedOutput, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                Family: ParameterizedFamily<Tracer<NestedTracingContext<Self>>>,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput: Parameterized<Tracer<NestedTracingContext<Self>>>,
    {
        if primals.parameters().next().is_none() {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        }
        let input_structure = primals.parameter_structure();
        let input_values = primals.into_parameters().collect::<Vec<_>>();

        // Trace the closure into a flat primal program over this context's types. Tracing stages every operation
        // without running any differentiation rule; simplification then drops staged dead code so the JVP replay
        // below does not pay for it.
        let context = NestedTracingContext::new(self.clone());
        let (output_structure, output_atoms) = {
            let input_tracers =
                input_values.iter().map(|value| context.input(value.r#type().into_owned())).collect::<Vec<_>>();
            let input = Input::To::<Tracer<NestedTracingContext<Self>>>::from_parameters(
                input_structure.clone(),
                input_tracers,
            )?;
            let output =
                function(input).map_err(|error| context.builder().borrow_mut().error.take().unwrap_or(error))?;
            context.builder().borrow_mut().error.take().map_or(Ok(()), Err)?;
            let output_structure = output.parameter_structure();
            // The outputs must belong to this trace: a foreign tracer's atom id would silently alias whichever atom
            // shares its index in this builder, so the boundary rejects it with a builder-identity check.
            check_builders!(context.builder(), [output.parameters().map(|output| output.builder())])?;
            let output_atoms = output.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
            (output_structure, output_atoms)
        };
        // Clone out the builder handle and drop the context so the clone is the sole owner, letting `Rc::try_unwrap`
        // recover the builder below unless a [`Tracer`] escaped the trace and still holds a reference.
        let builder = context.builder().clone();
        drop(context);
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let output_count = output_atoms.len();
        let program = builder
            .build::<Vec<<Self as Domain>::Constant>, Vec<<Self as Domain>::Constant>>(
                output_atoms,
                vec![Placeholder; input_values.len()],
                vec![Placeholder; output_count],
            )?
            .into_simplified()?;
        Ok((program, input_structure, output_structure, input_values))
    }

    /// Evaluates `function` on the primal `primals` and propagates the tangent `tangents` forward by running the
    /// closure **directly on [`JvpTracer`] duals** — the single forward-mode entry point, and the analogue of
    /// [JAX's `jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html).
    ///
    /// Like [`batch`](crate::batching::Batch::batch), this is a context-wrapping transform: each input is paired
    /// with its tangent as a dual over a [`JvpContext`] wrapping this context, and `function` runs directly on those
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
    /// invoked on a dual's [`JvpContext`] (a [`DifferentiationContext`] itself) differentiates through the duals,
    /// composing reverse-over-forward and higher-order forward modes.
    fn jvp<F, Input, Output>(
        &self,
        function: F,
        primals: Input,
        tangents: Input::To<<Self as Domain>::Value>,
    ) -> Result<(Output::To<<Self as Domain>::Value>, Output::To<<Self as Domain>::Value>), ProgramError>
    where
        Self: Zero<<Self as Domain>::Value>,
        JvpContext<Self>: Context<Value = JvpTracer<Self>>,
        <Self as Domain>::Operation: Clone + DifferentiableOperation<Self>,
        F: FnOnce(Input::To<JvpTracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                Family: ParameterizedFamily<JvpTracer<Self>> + ParameterizedFamily<<Self as Domain>::Value>,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<
                JvpTracer<Self>,
                Family: ParameterizedFamily<<Self as Domain>::Value> + ParameterizedFamily<<Self as Domain>::Value>,
            >,
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
        let context = JvpContext::new(self.clone());
        let input_duals = primals
            .into_parameters()
            .zip(tangents.into_parameters())
            .map(|(primal, tangent)| JvpTracer::new(primal, tangent).with_context(context.clone()))
            .collect::<Vec<_>>();
        let input = Input::To::<JvpTracer<Self>>::from_parameters(primal_structure, input_duals)?;
        let output = function(input)?;

        // Split each output dual into its primal value and its materialized tangent.
        let output_structure = output.parameter_structure();
        let output_duals = output.into_parameters().collect::<Vec<_>>();
        let mut primal_outputs = Vec::with_capacity(output_duals.len());
        let mut tangent_outputs = Vec::with_capacity(output_duals.len());
        for dual in output_duals {
            let JvpTracer { primal, tangent, .. } = dual;
            tangent_outputs.push(materialize(self, tangent)?);
            primal_outputs.push(primal);
        }
        let primal_output =
            Output::To::<<Self as Domain>::Value>::from_parameters(output_structure.clone(), primal_outputs)?;
        let tangent_output = Output::To::<<Self as Domain>::Value>::from_parameters(output_structure, tangent_outputs)?;

        Ok((primal_output, tangent_output))
    }

    /// Linearizes `function` at `primals`, returning the primal output and a reusable
    /// [`ForwardLinearization`] — the JAX `linearize` analogue.
    ///
    /// This is the program-level sibling of [`jvp`](Self::jvp). The closure is traced once into a primal
    /// [`Program`] and fused into a single JVP program over the ordinary primal operation family; that program is then
    /// partially evaluated with the primals known and the tangents unknown, folding the known half through this
    /// context itself: an eager context folds the primal computation and every residual factor to concrete values
    /// now, while a *staging* context stages the primal computation into the enclosing trace and carries the residual
    /// factors as [`Tracer`]s, so linearization composes under an outer trace. Either way only the
    /// linear tangent map survives as the residual program. The returned [`ForwardLinearization`] holds that tangent
    /// map, so [`ForwardLinearization::apply`] pushes any number of tangents through the function's Jacobian at this
    /// point without re-tracing or re-differentiating. In an eager context any concretizable `while` loop is unrolled
    /// at the primals beforehand.
    fn linearize<F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        (
            TracedOutput::To<<Self as Domain>::Value>,
            ForwardLinearization<Self, Input, TracedOutput::To<<Self as Domain>::Value>>,
        ),
        ProgramError,
    >
    where
        <Self as Domain>::Value: BooleanLike,
        <Self as Domain>::Operation: Clone
            + PartiallyEvaluatableOperation<Self>
            + MaybeWhile<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + From<ZeroOperation<<Self as Domain>::Type>>
            + DifferentiableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>,
        F: FnOnce(Input::To<Tracer<NestedTracingContext<Self>>>) -> TracedOutput,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<NestedTracingContext<Self>>>
                            + ParameterizedFamily<<Self as Domain>::Value>,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput:
            Parameterized<Tracer<NestedTracingContext<Self>>, Family: ParameterizedFamily<<Self as Domain>::Value>>,
    {
        let (program, _input_structure, output_structure, input_values) =
            self.trace_into_primal_program::<_, Input, TracedOutput>(|input| Ok(function(input)), primals)?;

        // Eager domains unroll any concretizable `while` loop at the concrete primals before fusing; staging domains
        // keep the bounded `while` rule.
        let program = unroll_concretizable_whiles(self, program, input_values.clone())?;
        let jvp_program = build_jvp_program(&program)?.into_simplified()?;

        // The fused program takes `[primals(n) ++ tangents(n)]` and produces `[primal(n) ++ tangent(n)]`. Mark the
        // primals known and the tangents unknown: the tangent of input `i` has the same type as primal input `i`.
        let primal_input_count = input_values.len();
        let mut partial_inputs =
            input_values.iter().map(|value| PartialValue::Known(value.clone())).collect::<Vec<_>>();
        partial_inputs.extend(input_values.iter().map(|value| PartialValue::Unknown(value.r#type().into_owned())));

        // Fold the known (primal) half through the differentiation context itself: an eager domain interprets it
        // immediately, while a staging domain stages it into the enclosing trace.
        let evaluation = jvp_program.partially_evaluate_in_context(self, partial_inputs.as_slice())?;

        // The fused program emits one tangent output per primal output, so its outputs split into two equal halves:
        // the leading half are the primal outputs and the trailing half are the tangent outputs. The split point is
        // the number of *function outputs*, which is the residual output count halved — not the primal *input* count,
        // which differs whenever the function's input and output arities differ.
        if evaluation.outputs.len() % 2 != 0 {
            return Err(ProgramError::MalformedProgram(format!(
                "fused jvp program produced {} outputs which is not an even split into primal and tangent halves",
                evaluation.outputs.len(),
            )));
        }
        let primal_output_count = evaluation.outputs.len() / 2;

        // The primals are all known, so each primal output folds to a known value; collect the folded values and
        // reshape them into the structured primal output.
        let primal_values = evaluation.outputs[..primal_output_count]
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(value) => Ok(value.clone()),
                PartialEvaluationOutput::Unknown(_) => Err(ProgramError::MalformedProgram(
                    "primal output did not fold to a known value during forward linearization".into(),
                )),
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let output =
            TracedOutput::To::<<Self as Domain>::Value>::from_parameters(output_structure.clone(), primal_values)?;

        let forward = ForwardLinearization::<Self, Input, TracedOutput::To<<Self as Domain>::Value>> {
            evaluation,
            domain: self.clone(),
            primal_input_count,
            primal_output_count,
            output_structure,
            marker: PhantomData,
        };
        Ok((output, forward))
    }

    /// Returns the traced primal output, a traced pullback program, and the linearization-point residuals.
    ///
    /// This is the value-level reverse-mode transform. The closure is traced once into a primal [`Program`] and
    /// differentiated on the capture-free path: the primal computation
    /// and its pushforward are staged into a single JVP program over the ordinary primal operation family, that
    /// program is partially evaluated into a known primal sub-program and an unknown linear tangent sub-program, the
    /// primal side is replayed to recover the primal outputs and the concrete residual values at the linearization
    /// point, and the tangent side is transposed in the primal operation family. The resulting pullback is then lifted
    /// into this context's value space so it can serve reverse mode *under tracing*: in an eager context the lift is
    /// the identity, while in a staging context (whose values are [`Tracer`]s) it records the pullback's literal
    /// constants as constants in the enclosing trace, so the backward pass splices into that trace.
    ///
    /// The returned pullback is a flat program over the primal operation family that maps
    /// `(output_cotangents ++ residuals)` to flat input cotangents. Because the direct-transpose pullback consumes the
    /// residuals as ordinary inputs rather than folding them in at transpose time, the recovered residuals are returned
    /// alongside the pullback so a caller appends them to the output cotangents when interpreting it, then reshapes the
    /// flat input cotangents through [`Parameterized::from_parameters`] against the closure's input structure.
    ///
    /// Functions reaching operations outside the supported straight-line slice fail with an
    /// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
    fn vjp<F, Input, TracedOutput>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        (
            TracedOutput::To<<Self as Domain>::Value>,
            Program<
                <Self as Domain>::Value,
                <Self as Domain>::Operation,
                Vec<<Self as Domain>::Value>,
                Vec<<Self as Domain>::Value>,
            >,
            Vec<<Self as Domain>::Value>,
        ),
        ProgramError,
    >
    where
        <Self as Domain>::Type: DifferentiableType,
        <Self as Domain>::Constant: Value<Type = <Self as Domain>::Type>,
        <Self as Domain>::Value: BooleanLike,
        <Self as Domain>::Operation: Clone
            + TransposableOperation<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + MaybeWhile<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + From<ZeroOperation<<Self as Domain>::Type>>
            + From<AddOperation>
            + DifferentiableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>
            + PartiallyEvaluatableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>,
        F: FnOnce(Input::To<Tracer<NestedTracingContext<Self>>>) -> Result<TracedOutput, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<NestedTracingContext<Self>>>,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput:
            Parameterized<Tracer<NestedTracingContext<Self>>, Family: ParameterizedFamily<<Self as Domain>::Value>>,
    {
        // Flatten the structured primals and wrap the structured closure into the flat closure the universal
        // reverse entry expects, recording the closure's output structure so the flat primal outputs can be reshaped
        // back into `TracedOutput::To<Value>` afterwards. The closure runs exactly once, so the recorded structure is
        // always present when the entry returns successfully.
        let input_structure = primals.parameter_structure();
        let flat_primals = primals.into_parameters().collect::<Vec<_>>();
        let output_structure: RefCell<Option<TracedOutput::ParameterStructure>> = RefCell::new(None);
        let flat_function = |input_tracers: Vec<Tracer<NestedTracingContext<Self>>>| {
            let input = Input::To::<Tracer<NestedTracingContext<Self>>>::from_parameters(
                input_structure.clone(),
                input_tracers,
            )?;
            let output = function(input)?;
            *output_structure.borrow_mut() = Some(output.parameter_structure());
            Ok(output.into_parameters().collect::<Vec<_>>())
        };

        let (program, _flat_input_structure, _flat_output_structure, input_values) =
            self.trace_into_primal_program::<_, Vec<<Self as Domain>::Value>, Vec<_>>(flat_function, flat_primals)?;

        // Eager domains unroll any concretizable `while` loop at the concrete primals before fusing, so reverse mode
        // through unbounded / data-dependent loops lowers to a control-flow-free tangent program that transposes via
        // the partitioned transposition; staging domains keep the bounded `while` rule.
        let program = unroll_concretizable_whiles(self, program, input_values.clone())?;
        let linearization = program.linearize()?;

        // Replay the primal side to recover the primal outputs followed by the residuals at the linearization point.
        // Under tracing these are enclosing-trace values, so they are returned for the caller to append to the output
        // cotangents.
        let primal_side = replay_via_bind(self, &linearization.primal_program, input_values)?;
        let primal_output_count = primal_side.len().checked_sub(linearization.residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "primal program produced {} outputs which is fewer than its {} residuals",
                primal_side.len(),
                linearization.residual_count,
            ))
        })?;
        let residuals = primal_side[primal_output_count..].to_vec();
        let primal_outputs = primal_side[..primal_output_count].to_vec();

        // Transpose the tangent sub-program in `Constant` space, then lift the resulting pullback into the active
        // value space so reverse-mode-under-tracing consumers can interpret it into the enclosing trace. The only
        // place a value flows into a transposed pullback is its constant atoms, so lifting those constants is the
        // complete conversion; under an enclosing `TracingContext` the lift records each constant in the enclosing
        // trace, while for an eager context it is the identity.
        let constant_pullback = transpose_tangent_partitioned(&linearization)?;
        let Program { atoms, input_ids, output_ids, instructions, .. } = constant_pullback;
        let atoms = atoms
            .into_iter()
            .map(|atom| match atom {
                Atom::Constant(constant) => Ok(Atom::Constant(self.lift(constant)?)),
                Atom::Variable(r#type) => Ok(Atom::Variable(r#type)),
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let input_count = input_ids.len();
        let output_count = output_ids.len();
        let pullback = Program {
            atoms,
            input_ids,
            output_ids,
            instructions,
            input_structure: vec![Placeholder; input_count],
            output_structure: vec![Placeholder; output_count],
            marker: PhantomData,
        };
        let output_structure = output_structure
            .into_inner()
            .ok_or_else(|| ProgramError::MalformedProgram("vjp closure did not record an output structure".into()))?;
        let output = TracedOutput::To::<<Self as Domain>::Value>::from_parameters(output_structure, primal_outputs)?;
        Ok((output, pullback, residuals))
    }

    /// Reverse-mode-differentiates `function` at `primals`, returning the primal output and a reusable [`Pullback`] —
    /// the JAX `vjp` analogue.
    ///
    /// This is the callable-surface sibling of [`vjp`](Self::vjp). It calls [`vjp`](Self::vjp) once and wraps the
    /// returned pullback program and linearization-point residuals in a [`Pullback`], so [`Pullback::apply`] maps
    /// output cotangents to input cotangents — appending the residuals, interpreting the pullback, and reshaping the
    /// flat input cotangents against the closure's input structure — without the caller threading the residuals by
    /// hand.
    fn vjp_fn<F, Input, TracedOutput>(
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
        <Self as Domain>::Value: BooleanLike,
        <Self as Domain>::Operation: Clone
            + TransposableOperation<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + MaybeWhile<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + From<ZeroOperation<<Self as Domain>::Type>>
            + From<AddOperation>
            + DifferentiableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>
            + PartiallyEvaluatableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>,
        F: FnOnce(Input::To<Tracer<NestedTracingContext<Self>>>) -> Result<TracedOutput, ProgramError>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<NestedTracingContext<Self>>>,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput:
            Parameterized<Tracer<NestedTracingContext<Self>>, Family: ParameterizedFamily<<Self as Domain>::Value>>,
    {
        let input_structure = primals.parameter_structure();
        let (output, program, residuals) = self.vjp(function, primals)?;
        Ok((output, Pullback { program, residuals, input_structure, marker: PhantomData }))
    }

    /// Returns the traced scalar output and reverse-mode gradient for `function`.
    ///
    /// This is the active-context counterpart of [`crate::tracing_v2::value_and_grad`]. It uses
    /// [`DifferentiationContext::vjp`] directly, so nested reverse mode composes with any enclosing context that
    /// implements this trait instead of going through a separate tracer dispatch path.
    fn value_and_grad<F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(<Self as Domain>::Value, Input::To<<Self as Domain>::Value>), DifferentiationError>
    where
        <Self as Domain>::Value: BooleanLike,
        <Self as Domain>::Constant: Value<Type = <Self as Domain>::Type>,
        <Self as Domain>::Type: DifferentiableType,
        <Self as Domain>::Operation: Clone
            + InterpretableOperation<<Self as Domain>::Value, Self>
            + TransposableOperation<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + MaybeWhile<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + From<ZeroOperation<<Self as Domain>::Type>>
            + From<OneOperation<<Self as Domain>::Type>>
            + From<AddOperation>
            + DifferentiableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>
            + PartiallyEvaluatableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>,
        F: FnOnce(Input::To<Tracer<NestedTracingContext<Self>>>) -> Tracer<NestedTracingContext<Self>>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<NestedTracingContext<Self>>>,
                ParameterStructure: Debug + PartialEq,
            >,
    {
        let input_structure = primals.parameter_structure();
        let (output, pullback, residuals) = self.vjp(|input| Ok(function(input)), primals)?;
        // Reverse mode only defines a gradient for scalar-output functions; reject non-scalar outputs before
        // seeding (see `DifferentiationError::NonScalarGradientOutput`).
        if !output.r#type().is_scalar() {
            return Err(DifferentiationError::NonScalarGradientOutput { output_type: output.r#type().to_string() });
        }
        // Seed the single output cotangent with the multiplicative identity of the scalar output, typed with the
        // output's cotangent type (e.g., swapping unreduced and reduced sharding axes for arrays) and staged through
        // `bind`. A non-differentiable scalar output (a Boolean or integer, the `float0` analogue) carries no cotangent
        // space and thus no "one" to seed, so reverse mode is degenerate and is rejected up front. The direct-transpose
        // pullback consumes `[output_cotangents ++ residuals]`, so the seed is followed by the linearization-point
        // residuals; its flat input cotangents are reshaped against the closure's input structure.
        let output_cotangent_type = output.r#type().cotangent().ok_or_else(|| {
            DifferentiationError::NonDifferentiableGradientOutput { output_type: output.r#type().to_string() }
        })?;
        let one_operation = <Self as Domain>::Operation::from(OneOperation::new(output_cotangent_type));
        let mut seeds = self.bind(one_operation, &[])?;
        check_count!("output", seeds, 1, ProgramError);
        let mut pullback_inputs = vec![seeds.pop().unwrap()];
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(self, pullback_inputs)?;
        let gradient = Input::To::<<Self as Domain>::Value>::from_parameters(input_structure, input_cotangents)
            .map_err(ProgramError::from)?;
        Ok((output, gradient))
    }

    /// Returns the reverse-mode gradient of a traced scalar-output function.
    #[inline]
    fn value_and_gradient<F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<Input::To<<Self as Domain>::Value>, DifferentiationError>
    where
        <Self as Domain>::Value: BooleanLike,
        <Self as Domain>::Constant: Value<Type = <Self as Domain>::Type>,
        <Self as Domain>::Type: DifferentiableType,
        <Self as Domain>::Operation: Clone
            + InterpretableOperation<<Self as Domain>::Value, Self>
            + TransposableOperation<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + MaybeWhile<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + From<ZeroOperation<<Self as Domain>::Type>>
            + From<OneOperation<<Self as Domain>::Type>>
            + From<AddOperation>
            + DifferentiableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>
            + PartiallyEvaluatableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>,
        F: FnOnce(Input::To<Tracer<NestedTracingContext<Self>>>) -> Tracer<NestedTracingContext<Self>>,
        Input: Parameterized<
                <Self as Domain>::Value,
                To<<Self as Domain>::Value> = Input,
                Family: ParameterizedFamily<Tracer<NestedTracingContext<Self>>>,
                ParameterStructure: Debug + PartialEq,
            >,
    {
        self.value_and_grad(function, primals).map(|(_, gradient)| gradient)
    }
}

impl<C: Context> DifferentiationContext for C {}

/// A dual value: a primal value paired with its symbolic tangent, both flowing through the same [`Context`] `C`.
///
/// Under a staging context both components are tracers in the *one* shared builder, so a tangent coefficient is
/// produced by ordinary tracer arithmetic on the primal tracer
/// (for example, `primal.cos()` stages a fresh `Cos` operation) rather than by capturing a residual factor. Both the
/// primal SSA values and the tangent SSA values of a linearization live in that one [`TracingContext`],
/// whose [`DispatchDomain::Operation`] is the ordinary primal operation family `O`, so a tangent tracer staged there is an
/// ordinary primal operation (a `Mul`, `Add`, `Sin`, ...) rather than a linear operation with capture factors — which
/// is precisely how the front end avoids symbolic capture entirely. Under an eager context (through a
/// [`JvpContext`]) both components are concrete runtime values instead, and the same rules compute them directly.
/// The tangent is a [`MaybeZero`]: structural zeros stay symbolic between rules and are materialized only at
/// boundaries (see [`materialize`]).
pub struct JvpTracer<C: Context> {
    /// Primal value of this dual, staged in the staging context `C`.
    primal: C::Value,

    /// Tangent of this dual: a tangent value staged in the staging context `C`, or a structural
    /// [`MaybeZero::Zero`] carrying only the tangent's [`Type`](crate::Type).
    tangent: MaybeZero<C::Value>,

    /// [`JvpContext`] this dual flows through, stamped by [`JvpContext::bind`] and [`JvpContext::lift`] so that the
    /// value-capability sugar (`x + y`, `x.sin()`, …) can dispatch each operation through the forward-mode rule. It is
    /// `None` only for duals a [`jvp`](DifferentiableOperation::jvp) rule constructs internally (via [`JvpTracer::new`])
    /// before `bind` stamps the context onto the values it hands back to the user closure; rule bodies never call the
    /// sugar on those intermediates, so the stamp is always present by the time a dual reaches the closure.
    context: Option<JvpContext<C>>,
}

impl<C: Context> JvpTracer<C> {
    /// Creates a dual from its primal staged value and its symbolic tangent. The tangent may be passed either as a
    /// staged tangent value directly or as a [`MaybeZero`].
    ///
    /// Public so backend crates can author [`DifferentiableOperation`] rules for their own operations, pairing
    /// each staged primal output with its tangent output.
    #[inline]
    pub fn new<Tangent: Into<MaybeZero<C::Value>>>(primal: C::Value, tangent: Tangent) -> Self {
        Self { primal, tangent: tangent.into(), context: None }
    }

    /// Creates a dual from its primal staged value and a structural zero tangent typed with the primal's own
    /// [`Type`].
    #[inline]
    pub fn with_zero_tangent(primal: C::Value) -> Self {
        let tangent = MaybeZero::Zero(primal.r#type().into_owned());
        Self { primal, tangent, context: None }
    }

    /// Returns the primal staged value of this dual.
    #[inline]
    pub fn primal(&self) -> &C::Value {
        &self.primal
    }

    /// Returns the symbolic tangent of this dual.
    #[inline]
    pub fn tangent(&self) -> &MaybeZero<C::Value> {
        &self.tangent
    }

    /// Returns the [`JvpContext`] this dual flows through, stamped by [`JvpContext::bind`]/[`JvpContext::lift`]. The
    /// value-capability sugar dispatches operations through it. This is only ever called on closure-facing duals, which
    /// are always stamped (see the [`context`](Self::context) field documentation).
    #[inline]
    pub(crate) fn context(&self) -> &JvpContext<C> {
        self.context
            .as_ref()
            .expect("a `JvpTracer` reached value-capability sugar without a stamped `JvpContext`")
    }

    /// Returns this dual with `context` stamped as its flowing [`JvpContext`]. [`JvpContext::bind`] and
    /// [`JvpContext::lift`] use this to stamp the values they hand back to the user closure.
    #[inline]
    fn with_context(mut self, context: JvpContext<C>) -> Self {
        self.context = Some(context);
        self
    }
}

impl<C: Context> Clone for JvpTracer<C> {
    #[inline]
    fn clone(&self) -> Self {
        Self { primal: self.primal.clone(), tangent: self.tangent.clone(), context: self.context.clone() }
    }
}

impl<C: Context> Debug for JvpTracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JvpTracer")
            .field("primal", &self.primal)
            .field("tangent", &self.tangent)
            .finish()
    }
}

impl<C: Context> std::fmt::Display for JvpTracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Dual-number notation: the primal plus the tangent-scaled infinitesimal.
        match &self.tangent {
            MaybeZero::Zero(_) => write!(formatter, "{} + 0ε", self.primal),
            MaybeZero::Value(tangent) => write!(formatter, "{} + {}ε", self.primal, tangent),
        }
    }
}

impl<C: Context> Typed for JvpTracer<C> {
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> std::borrow::Cow<'_, C::Type> {
        self.primal.r#type()
    }
}

impl<C: Context> Parameter for JvpTracer<C> {}

/// A JVP dual flows through the [`JvpContext`] stamped on it.
impl<C: Context> Value for JvpTracer<C> {
    type DispatchDomain = JvpContext<C>;
    type ExecutionDomain = JvpContext<C>;

    #[inline]
    fn dispatch_domain(&self) -> JvpContext<C> {
        self.context().clone()
    }

    #[inline]
    fn execution_domain(&self) -> JvpContext<C> {
        self.context().clone()
    }
}

// The elementwise/arithmetic/trigonometric and other operation-specific capability implementations for `JvpTracer`
// are hand-written in each operation's own module (bind-forwarding through the stamped `JvpContext`). The
// capabilities whose signatures do not fit that shape are implemented by hand below.

/// A dual's Boolean view uses its primal's: [`as_boolean`](BooleanLike::as_boolean) reinterprets the primal with a
/// structural zero tangent, and [`boolean`](BooleanLike::boolean) decodes the primal — so branching on a dual in a
/// closure succeeds exactly when the primal is a concrete (eager) value and errors when it is a staged tracer.
impl<C: Context> BooleanLike for JvpTracer<C>
where
    C::Value: BooleanLike,
{
    #[inline]
    fn as_boolean(&self) -> Self {
        let primal = self.primal.as_boolean();
        let tangent = MaybeZero::Zero(primal.r#type().into_owned());
        Self { primal, tangent, context: self.context.clone() }
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        self.primal.boolean()
    }
}

/// A forward-mode differentiation [`Context`] that interleaves [`DifferentiableOperation`] rules with an inner
/// [`Context`], without building a program: its values are [`JvpTracer`] duals over the inner context's values, and
/// binding an operation dispatches the operation's [`jvp`](DifferentiableOperation::jvp) rule against the inner
/// context directly. Over an eager inner context this computes primal and tangent values operation by operation
/// (the analogue of [JAX's `jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) interpreter), while over
/// a staging inner context the rules stage the primal and tangent operations into the enclosing trace.
///
/// This is forward mode's counterpart of [`BatchingContext`](crate::batching::BatchingContext): a transform context
/// that wraps the receiver and runs the user's closure directly on transform tracers ([`JvpTracer`] duals here,
/// [`BatchingTracer`](crate::batching::BatchingTracer)s there), with eager-versus-staged behavior absorbed entirely
/// by the wrapped context. It is what makes [`DifferentiationContext::jvp`] the single forward-mode entry point.
///
/// Structural zero tangents stay symbolic [`MaybeZero::Zero`]s while they flow between rules: the
/// [`bind`](Context::bind) fast path skips an operation's rule entirely when every input tangent is a structural
/// zero, exactly like the program-level replay behind [`Program::linearize`], so no zero values are constructed and no
/// zero work is performed until a boundary [`materialize`]s one through the inner context's [`Zero`] capability.
#[derive(Clone)]
pub struct JvpContext<C: Context> {
    /// Inner context that carries the primal and tangent values and executes (or stages) the operations that the
    /// forward-mode rules bind.
    context: C,
}

impl<C: Context> JvpContext<C> {
    /// Creates a new [`JvpContext`] over the provided inner [`Context`].
    #[inline]
    pub fn new(context: C) -> Self {
        Self { context }
    }

    /// Returns the inner [`Context`].
    #[inline]
    pub fn context(&self) -> &C {
        &self.context
    }
}

impl<C: Context> Domain for JvpContext<C> {
    type Type = C::Type;
    type Value = JvpTracer<C>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

/// A [`JvpContext`] binds no named axes of its own: axis-name resolution passes through to the inner context, so
/// collectives inside a differentiated closure resolve against the enclosing batching levels and mesh regions.
impl<C> crate::axes::NamedAxes for JvpContext<C>
where
    C: Context + crate::axes::NamedAxes + Zero<C::Value>,
    C::Operation: Clone + DifferentiableOperation<C>,
{
    #[inline]
    fn named_axis(&self, name: &str) -> Option<crate::axes::NamedAxis> {
        self.context().named_axis(name)
    }
}

impl<C> Context for JvpContext<C>
where
    C: Context + Zero<C::Value>,
    C::Operation: Clone + DifferentiableOperation<C>,
{
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<JvpTracer<C>, ProgramError> {
        // Constants are independent of every differentiation input, so their tangents are structural zeros.
        Ok(JvpTracer::with_zero_tangent(self.context.lift(constant)?).with_context(self.clone()))
    }

    fn bind<O: Into<C::Operation>>(
        &self,
        operation: O,
        inputs: &[JvpTracer<C>],
    ) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        let operation = operation.into();
        // All-zero fast path mirroring `build_jvp_program`: when an operation consumes at least one input and every
        // input tangent is a structural zero, the operation's tangent is zero by the chain rule, so the rule is
        // skipped and the primal operation binds directly. Zero-input operations are excluded so their dedicated
        // rules keep handling primal synthesis and tangent typing.
        let outputs = if !inputs.is_empty() && inputs.iter().all(|dual| dual.tangent().is_zero()) {
            let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
            self.context
                .bind(operation, primal_inputs.as_slice())?
                .into_iter()
                .map(JvpTracer::with_zero_tangent)
                .collect()
        } else {
            operation.jvp(&self.context, inputs)?
        };
        // Stamp this context onto every value handed back to the caller so its capability sugar dispatches through this
        // forward-mode context (the `jvp` rules build their outputs context-free via `JvpTracer::new`).
        Ok(outputs.into_iter().map(|dual| dual.with_context(self.clone())).collect())
    }

    /// A forward-mode context is eager exactly when the inner context carrying its duals' values is (never over a
    /// staging inner context, always over an eager one).
    #[inline]
    fn is_eager(&self) -> bool {
        self.context.is_eager()
    }
}

/// A zero synthesized inside a forward-mode context is independent of every differentiation input, so it is the
/// inner context's zero paired with a structural zero tangent.
impl<C> Zero<JvpTracer<C>> for JvpContext<C>
where
    C: Context + Zero<C::Value>,
    C::Operation: Clone + DifferentiableOperation<C>,
{
    #[inline]
    fn zero(&self, r#type: &C::Type) -> Result<JvpTracer<C>, ProgramError> {
        Ok(JvpTracer::with_zero_tangent(self.context().zero(r#type)?).with_context(self.clone()))
    }
}

/// A one synthesized inside a forward-mode context is independent of every differentiation input, so it is the
/// inner context's one paired with a structural zero tangent.
impl<C> One<JvpTracer<C>> for JvpContext<C>
where
    C: Context + Zero<C::Value> + One<C::Value>,
    C::Operation: Clone + DifferentiableOperation<C>,
{
    #[inline]
    fn one(&self, r#type: &C::Type) -> Result<JvpTracer<C>, ProgramError> {
        Ok(JvpTracer::with_zero_tangent(self.context().one(r#type)?).with_context(self.clone()))
    }
}

/// A fill synthesized inside a forward-mode context is independent of every differentiation input, so it is the
/// inner context's fill paired with a structural zero tangent.
impl<C, S> Fill<S, JvpTracer<C>> for JvpContext<C>
where
    C: Context + Zero<C::Value> + Fill<S, C::Value>,
    C::Operation: Clone + DifferentiableOperation<C>,
{
    #[inline]
    fn fill(&self, r#type: &C::Type, value: S) -> Result<JvpTracer<C>, ProgramError> {
        Ok(JvpTracer::with_zero_tangent(self.context().fill(r#type, value)?).with_context(self.clone()))
    }
}

/// An iota synthesized inside a forward-mode context is independent of every differentiation input, so it is the
/// inner context's iota paired with a structural zero tangent.
impl<C> Iota<JvpTracer<C>> for JvpContext<C>
where
    C: Context + Zero<C::Value> + Iota<C::Value>,
    C::Operation: Clone + DifferentiableOperation<C>,
{
    #[inline]
    fn iota(&self, r#type: &C::Type, dimension: usize) -> Result<JvpTracer<C>, ProgramError> {
        Ok(JvpTracer::with_zero_tangent(self.context().iota(r#type, dimension)?).with_context(self.clone()))
    }
}

/// A captured constant lifted inside a forward-mode context is independent of every differentiation input, so it is
/// the inner context's lifted value paired with a structural zero tangent (via [`Context::lift`]).
impl<C> Constant<JvpTracer<C>, C::Constant> for JvpContext<C>
where
    C: Context + Zero<C::Value>,
    C::Operation: Clone + DifferentiableOperation<C>,
{
    #[inline]
    fn constant(&self, value: C::Constant) -> Result<JvpTracer<C>, ProgramError> {
        self.lift(value)
    }
}

/// Runtime capture registration inside a forward-mode context delegates to the inner context, so values captured
/// while differentiating flow into the same capture table as ordinary staging.
impl<C, Capture> CapturingContext<Capture> for JvpContext<C>
where
    C: CapturingContext<Capture> + Zero<C::Value>,
    C::Operation: Clone + DifferentiableOperation<C>,
    Capture: Value<Type = C::Type>,
{
    #[inline]
    fn capture(&self, value: Capture) -> Result<Self::Constant, ProgramError> {
        self.context().capture(value)
    }
}

/// Operation-level contract for capture-free forward-mode (JVP) staging.
///
/// In a [`DifferentiableOperation`] each primitive operation owns its forward-mode rule, and the `ScalarOperation` /
/// [`ArrayOperation`](crate::tracing_v2::ArrayOperation) enums forward to the active variant. The rule is keyed by the
/// ordinary primal operation family `O` rather than by a differentiation context: it consumes [`JvpTracer`] inputs and
/// stages both the primal result and the tangent operations into the one shared [`TracingContext`] as ordinary
/// primal operations (a `Mul`, `Add`, `Sin`, ...), so no symbolic capture is ever introduced.
///
/// Implementing this trait is what gives an operation its forward-mode behavior: [`jvp`](Self::jvp) is a required
/// method, and an operation that has no capture-free forward-mode form — such as the scalar `while` loop or the
/// reverse-mode-only custom-VJP tangent carrier — implements it with a rule that reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error. Operation enums such as `ScalarOperation` and
/// [`ArrayOperation`](crate::tracing_v2::ArrayOperation) implement the trait through
/// `#[derive(DifferentiableOperation)]` (see the derive contract on [`Operation`]), whose generated dispatcher
/// forwards every variant to its payload's rule uniformly, so the unsupported payloads' own erroring rules report
/// those failures. Each per-operation rule supplies the operation's own primal-enum operand arithmetic; the
/// [`ConditionOperation`](crate::operations::control_flow::ConditionOperation) rule is the higher-order case,
/// linearizing both branches capture-free through [`DifferentiableProgramOperation`] and staging an ordinary
/// primal-enum `condition` for each of the primal and the tangent side.
///
/// The context `C` carries the type, constant, value, and operation universe of the primal program being
/// linearized: `C::Type` is the type descriptor, `C::Constant` the program's constant payload, `C::Value` the value
/// flowing through the rules (a staged [`Tracer`] under a staging context, a concrete runtime value under an eager
/// one), and `C::Operation` the primal operation family. Rules bind the operations they synthesize through
/// [`Context::bind`], which stages them under a staging context and executes them under an eager context, so one
/// rule serves program-building replay (behind [`Program::linearize`]) and interleaved forward mode ([`JvpContext`])
/// uniformly.
///
/// ## Deriving Differentiable Operation Enums
///
/// Ryft also provides a `#[derive(DifferentiableOperation)]` procedural macro for operation enums whose variants own
/// forward-mode (JVP) rules through
/// [`DifferentiableOperation`](crate::tracing_v2::differentiation::DifferentiableOperation). This derive enables
/// forward-mode differentiation only; enums that also need reverse-mode differentiation additionally derive
/// `TransposableOperation` (see the derive contract on
/// [`TransposableOperation`](crate::differentiation::TransposableOperation)), whose transposition dispatchers
/// reverse mode is built on. It follows the same enum-shape and operation-type-inference rules as
/// `#[derive(Operation)]` and generates:
///
///   - An `impl DifferentiableOperation<C> for Enum` that is generic over a
///     [`StagingContext`](crate::StagingContext) `C` pinned to the enum's primary type, program constant type, and
///     the enum itself as its operation family. Every variant forwards
///     [`jvp`](crate::tracing_v2::differentiation::DifferentiableOperation::jvp) to its payload's own rule, so payloads
///     without a capture-free forward-mode form must still implement the trait with a rule that reports an
///     [`UnsupportedOperation`](crate::ProgramError::UnsupportedOperation) error (e.g., the scalar `while` rule).
///   - A `where` clause following the same shape as the generated interpretation and partial-evaluation impls: a
///     per-variant `Payload: DifferentiableOperation<C>` predicate for every *non-recursive* payload — the
///     predicate transports each rule's own capability requirements (e.g., `C::Value: Sin` for the sine rule) to
///     the use site, so the enum does not spell them — plus a `Self: From<Payload>` conversion for every concrete
///     payload (the rules stage ordinary primal-enum operations for both the primal and the tangent side) and the
///     `Self: MaybeZeroOperation<T> + From<ZeroOperation<T>> +
///     DifferentiableProgramOperation<C::Constant, Self>` fixed-point witnesses that higher-order payload rules
///     (condition/while/scan) use to linearize their nested programs. *Recursive* payloads (those mentioning
///     `Self`) are skipped — such a predicate would re-enter the enum's own obligation and overflow the trait
///     solver — and their rules are discharged as definition-time body obligations against the witnesses instead.
///     The enum must therefore supply its own
///     [`DifferentiableProgramOperation`](crate::tracing_v2::differentiation::DifferentiableProgramOperation)
///     implementation, spelling only the leaf capabilities that
///     [`Program::linearize`](crate::Program::linearize) needs.
///
/// The derive supports no `#[ryft(bounds(...))]` kind of its own: the per-variant predicates always forward each
/// payload rule's capability requirements, so there is nothing for the enum to add. It tolerates (parses and
/// discards) the `interpretation(...)` and `partial_evaluation(...)` kinds owned by a sibling `#[derive(Operation)]`
/// sharing the `#[ryft(...)]` attribute namespace.
pub trait DifferentiableOperation<C: Context<Operation: Clone>>: Operation<C::Type> {
    /// Applies this operation's capture-free forward-mode (JVP) rule.
    ///
    /// The returned vector must be aligned with this operation's outputs, each carrying the primal output value and
    /// the staged tangent value for that output, both staged in the shared builder.
    ///
    /// # Parameters
    ///
    ///   - `context`: Shared context into which both primal and tangent operations are staged.
    ///   - `inputs`: Input duals aligned with this operation's operands.
    fn jvp(&self, context: &C, inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError>;
}

/// Replays `operation` on the primal tracers of `inputs` and pairs each primal output with a structural zero
/// tangent typed with that output's own [`Type`](crate::Type).
///
/// This is the shared rule for operations whose outputs carry no tangent — the nullary and exemplar-derived
/// constants, discrete-valued comparisons, the logical operations, and `stop_gradient` (which severs an incoming
/// tangent). The primal is synthesized by staging the original primal-enum operation so the program reproduces it
/// exactly; the zero tangents stay symbolic and stage nothing.
///
/// # Parameters
///
///   - `context`: Shared context into which the primal operation is staged.
///   - `operation`: Primal operation to replay on the input primals.
///   - `inputs`: Input duals whose primal tracers feed the replayed operation.
pub(crate) fn replay_zero_tangent<C, P>(
    context: &C,
    operation: P,
    inputs: &[JvpTracer<C>],
) -> Result<Vec<JvpTracer<C>>, ProgramError>
where
    C: Context,
    P: Into<C::Operation>,
{
    let primal_inputs = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
    Ok(context
        .bind(operation, primal_inputs.as_slice())?
        .into_iter()
        .map(JvpTracer::with_zero_tangent)
        .collect())
}

/// Combines the two optional tangent terms of a (bi)linear differentiation rule, falling back to a structural zero
/// tangent of `primal`'s type when both terms were dropped as zero.
///
/// This is the shared term-combination step of the product, quotient, and contraction rules: each surviving term is a
/// live tangent tracer staged in the shared builder, and the all-zero fallback stays symbolic and stages nothing.
///
/// # Parameters
///
///   - `left_term`: Live left tangent term, or [`None`] if it was dropped as a structural zero.
///   - `right_term`: Live right tangent term, or [`None`] if it was dropped as a structural zero.
///   - `primal`: Primal tracer whose type the zero fallback adopts.
pub(crate) fn combine_terms<T, V>(left_term: Option<V>, right_term: Option<V>, primal: &V) -> MaybeZero<V>
where
    T: Type,
    V: Typed<Type = T> + std::ops::Add<Output = V>,
{
    match (left_term, right_term) {
        (Some(left_term), Some(right_term)) => MaybeZero::Value(left_term + right_term),
        (Some(term), None) | (None, Some(term)) => MaybeZero::Value(term),
        (None, None) => MaybeZero::Zero(primal.r#type().into_owned()),
    }
}

/// Builds the JVP program from an already-traced primal [`Program`].
///
/// The returned program stages both the primal computation and its pushforward into one program over the primal
/// operation family: its inputs are the primal inputs followed by one fresh tangent input per primal input (same
/// types), and its outputs are the original primal outputs followed by the tangent outputs. Each primal instruction is
/// replayed once through its [`DifferentiableOperation`] rule, which returns the dual (primal result plus tangent)
/// for the instruction's outputs; both are staged into the shared builder as ordinary primal operations, so the result
/// contains no symbolic capture.
///
/// Atoms that are not reached by any input tangent are structurally zero. Their tangents stay symbolic as typed
/// [`MaybeZero::Zero`]s and stage nothing. The shared all-zero fast path below short-circuits the all-zero case (an
/// operation consuming at least one input whose every input tangent is a structural zero) by staging the primal
/// operation directly and pairing each primal output with a typed structural zero tangent, so zero-ness propagates
/// transitively without staging or scanning instructions. Structural zero tangents are materialized as typed
/// [`ZeroOperation`] instructions only at the output boundary, preserving the `(primal_outputs ++ tangent_outputs)`
/// program contract.
///
/// # Parameters
///
///   - `program`: Already-traced primal program over the primal operation family `O`. Its constants are lifted into
///     the builder and its instructions are replayed in order. Operations outside the supported slice fail with
///     the [`DifferentiableOperation`] default's
///     [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
pub(crate) fn build_jvp_program<T, V, O, Input, Output>(
    program: &Program<V, O, Input, Output>,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    T: Type,
    V: Value<Type = T>,
    O: Clone + Operation<T> + From<ZeroOperation<T>>,
    O: DifferentiableOperation<TracingContext<V, O>>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    let primal_input_count = program.input_ids().len();

    // Hold a standalone `Rc` clone of the context's builder, and move the context itself into the block below, so that
    // scoping every tracer (and the context) inside that block makes the `Rc::try_unwrap` at the end a real ownership
    // check rather than depending on manual drops. Only raw output atom ids escape the block.
    let context = TracingContext::<V, O>::new();
    let builder = context.builder().clone();
    let output_atoms = {
        let context = context;

        // Track the primal tracer and symbolic tangent for each source atom. Tangents of atoms not connected to an
        // input tangent (constants and dead inputs) are derived lazily as structural zeros by `recorded_tangent`.
        let mut primals: Vec<Option<Tracer<TracingContext<V, O>>>> = vec![None; program.atoms().len()];
        let mut tangents: Vec<Option<MaybeZero<Tracer<TracingContext<V, O>>>>> = vec![None; program.atoms().len()];

        // Primal inputs become the leading inputs; one fresh tangent input is added per primal input afterwards
        // so the input order is `(primals ++ tangents)`.
        for input_id in program.input_ids().iter().copied() {
            let r#type = program.atoms()[input_id.index()].r#type().into_owned();
            primals[input_id.index()] = Some(context.input(r#type));
        }
        let tangent_inputs = program
            .input_ids()
            .iter()
            .copied()
            .map(|input_id| {
                let r#type = program.atoms()[input_id.index()].r#type().into_owned();
                context.input(r#type)
            })
            .collect::<Vec<_>>();
        for (input_id, tangent) in program.input_ids().iter().copied().zip(tangent_inputs) {
            tangents[input_id.index()] = Some(MaybeZero::Value(tangent));
        }

        // Constants are lifted into the builder as primal constants; their tangents are derived lazily as structural
        // zeros by `recorded_tangent`. The call is disambiguated to the staging method because the `Constant`
        // capability trait also provides a `constant` method.
        for (atom_index, atom) in program.atoms().iter().enumerate() {
            if let Atom::Constant(value) = atom {
                primals[atom_index] = Some(StagingContext::constant(&context, value.clone()));
            }
        }

        // Replay each primal instruction in JVP form, staging both the primal result and the tangent operations
        // into the shared builder.
        for instruction in program.instructions() {
            let input_duals = instruction
                .inputs()
                .iter()
                .copied()
                .map(|input_atom| {
                    let primal =
                        primals[input_atom.index()].clone().ok_or(ProgramError::UnboundAtomId { id: input_atom })?;
                    let tangent = recorded_tangent(&primals, &tangents, input_atom)?;
                    Ok(JvpTracer::<TracingContext<V, O>>::new(primal, tangent))
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
                    .map(JvpTracer::<TracingContext<V, O>>::with_zero_tangent)
                    .collect()
            } else {
                instruction.operation().jvp(&context, input_duals.as_slice())?
            };

            check_count!("output", output_duals, instruction.outputs().len(), ProgramError);
            for (output_atom, dual) in instruction.outputs().iter().copied().zip(output_duals) {
                primals[output_atom.index()] = Some(dual.primal);
                tangents[output_atom.index()] = Some(dual.tangent);
            }
        }

        // Collect the outputs: the primal outputs followed by the tangent outputs, in the original output order.
        // Structural zero tangents are materialized as typed `ZeroOperation` instructions here — the output boundary
        // is the only place the fused program requires a real atom for them.
        let primal_output_atoms = program
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
        let tangent_output_atoms = program
            .output_ids()
            .iter()
            .copied()
            .map(|output_atom| {
                let tangent = recorded_tangent(&primals, &tangents, output_atom)?;
                materialize(&context, tangent)?.atom_id()
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

/// Returns the symbolic tangent recorded for `atom`, deriving a structural zero typed with the atom's primal type
/// for atoms (constants and dead inputs) not connected to any input tangent.
fn recorded_tangent<V: Typed + Clone>(
    primals: &[Option<V>],
    tangents: &[Option<MaybeZero<V>>],
    atom: AtomId,
) -> Result<MaybeZero<V>, ProgramError> {
    if let Some(tangent) = &tangents[atom.index()] {
        return Ok(tangent.clone());
    }
    let primal = primals[atom.index()].as_ref().ok_or(ProgramError::UnboundAtomId { id: atom })?;
    Ok(MaybeZero::Zero(primal.r#type().into_owned()))
}

/// Returns the value inside the provided symbolic `value`, materializing a structural zero as a real typed zero in
/// the provided [`Context`] through its [`Zero`] capability: a staging context stages a typed
/// [`ZeroOperation`] instruction, while an eager context constructs a concrete zero value. This is the
/// instantiate-zeros boundary shared by forward-mode replay and transposition: call it exactly where a symbolic zero
/// must become a real value — a nested sub-program operand, a program output, an eagerly returned tangent — and
/// match on the [`MaybeZero`] everywhere else so zeros stay symbolic. Public so backend crates can materialize zeros
/// in their own higher-order [`DifferentiableOperation`] and [`TransposableOperation`]
/// rules.
pub fn materialize<C>(context: &C, value: MaybeZero<C::Value>) -> Result<C::Value, ProgramError>
where
    C: Context + Zero<C::Value>,
{
    match value {
        MaybeZero::Value(value) => Ok(value),
        MaybeZero::Zero(r#type) => context.zero(&r#type),
    }
}

/// Result of [`Program::linearize`](crate::Program::linearize): the partially evaluated known (primal) and unknown
/// (tangent) sub-programs together with the metadata needed to reassemble and transpose them.
///
/// This is the domain-free, interpretation-free core shared by every linearization entry point: it carries only
/// the two split sub-programs and the structural metadata that relates them, leaving the concrete primal outputs to be
/// recovered by callers that interpret [`primal_program`](Self::primal_program) under a value semantics of their
/// choice.
///
/// Its tangent sub-program is expressed in the primal operation family `O` with inputs `(tangents ++ residuals)`, which
/// is why [`transpose_tangent_partitioned`] can transpose it directly through
/// [`Program::transpose_with_respect_to`](crate::Program::transpose_with_respect_to) without re-keying it into a linear
/// operation family.
///
/// The value type `V` and operation family `O` match the primal program that was
/// program being linearized.
pub struct Linearization<V: Value, O: Clone + Operation<V::Type>> {
    /// Known sub-program. It takes the primal inputs and produces the primal outputs followed by the residuals; its
    /// trailing [`residual_count`](Self::residual_count) outputs are the residual environment consumed by the tangent
    /// sub-program.
    pub primal_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Unknown sub-program. It takes the tangent inputs followed by the residuals and produces the tangent outputs.
    pub tangent_program: Program<V, O, Vec<V>, Vec<V>>,

    /// For each original function output, `true` if it is produced by the tangent sub-program and `false` if by the
    /// primal sub-program. Used to reassemble the combined program's outputs from the two sides.
    pub output_unknowns: Vec<bool>,

    /// Number of residuals threaded from the primal sub-program into the tangent sub-program — the count of trailing
    /// outputs of [`primal_program`](Self::primal_program) and trailing inputs of
    /// [`tangent_program`](Self::tangent_program).
    pub residual_count: usize,
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
    /// Builds the fused jvp program of this already-traced primal [`Program`] over `[primals..., tangents...]`,
    /// producing `[primal_outputs..., tangent_outputs...]`, without splitting it into primal and tangent halves;
    /// this is the un-split front half of [`Self::linearize`], exposed for fused higher-order JVP rules and
    /// direct forward-mode interpretation.
    pub fn jvp_program(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        build_jvp_program(self)
    }

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
    /// [`PartialEvaluator::fold_or_residualize`](crate::partial::PartialEvaluator::fold_or_residualize).
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
        let fused = build_jvp_program(self)?;

        // Split the fused program with the leading `primal_input_count` primal inputs known and the trailing tangent
        // inputs unknown. The split walks the fused program through the per-operation partial-evaluation rules
        // against a fresh known-side staging trace: known (primal) work folds by staging into that trace, and the
        // residual program that survives is the linear tangent map.
        let input_known = std::iter::repeat(true)
            .take(primal_input_count)
            .chain(std::iter::repeat(false).take(primal_input_count))
            .collect::<Vec<bool>>();
        let partition = fused.partition(input_known.as_slice())?;
        let residual_count = partition.residual_inputs.iter().filter(|input| input.is_known()).count();
        let known_output_indices = partition
            .outputs
            .iter()
            .enumerate()
            .filter_map(|(index, output)| output.is_known().then_some(index))
            .collect::<Vec<_>>();
        let residual_output_indices = partition
            .outputs
            .iter()
            .enumerate()
            .filter_map(|(index, output)| output.is_unknown().then_some(index))
            .collect::<Vec<_>>();
        let PartitionedProgram { mut known_program, residual_program, residual_inputs, .. } = partition;

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
                None => restored_input_atom(&mut tangent_program.atoms, &fused, primal_input_count + position)?,
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

        Ok(Linearization {
            primal_program: known_program,
            tangent_program,
            // Every consumer sees canonical arity: the primal half is known and the tangent half is unknown.
            output_unknowns: std::iter::repeat(false)
                .take(primal_output_count)
                .chain(std::iter::repeat(true).take(primal_output_count))
                .collect(),
            residual_count,
        })
    }
}

/// Returns a fresh tangent input atom of the type of the `jvp_input_index`-th input of `jvp_program`, pushing it onto
/// `atoms`. Used by the linearization input rebuild to restore a tangent position missing from the split's recorded
/// residual-program input sources — unreachable under the current walk, which seeds every unknown input up front and
/// never prunes residual-program inputs, but kept so the rebuild stays correct for any source layout.
fn restored_input_atom<T, V, O>(
    atoms: &mut Vec<Atom<V>>,
    jvp_program: &Program<V, O, Vec<V>, Vec<V>>,
    jvp_input_index: usize,
) -> Result<AtomId, ProgramError>
where
    T: Type,
    V: Value<Type = T>,
    O: Operation<T>,
{
    let Atom::Variable(tangent_type) = &jvp_program.atoms[jvp_program.input_ids[jvp_input_index].index()] else {
        return Err(ProgramError::MalformedProgram(format!("tangent input {jvp_input_index} is not a variable")));
    };
    let restored = AtomId::new(atoms.len());
    atoms.push(Atom::Variable(tangent_type.clone()));
    Ok(restored)
}

/// Operation families whose captured flat programs can be linearized capture-free on behalf of an enclosing JVP rule.
///
/// Higher-order forward-mode rules, such as the control-flow rules, must linearize the captured branch or
/// body programs whose operation family is the same closed enum currently being proven
/// [`DifferentiableOperation`]. Writing that need directly as a recursive `DifferentiableOperation` bound at every
/// recursive payload boundary makes Rust's trait solver re-enter the same enum and overflow.
/// [`DifferentiableProgramOperation`] names that recursive fixed point once: the value `V`
/// and operation family `O` stay fixed across the recursion, so a closed operation enum implements this trait directly
/// — calling [`Program::linearize`](crate::Program::linearize) in the body while spelling only the *leaf* closure of
/// capabilities that body needs in the impl's `where` clause, rather than the recursive
/// `Self: DifferentiableOperation<…>` bound
/// itself. That recursive obligation is then discharged once, as a definition-time body check, which is what lets a
/// higher-order rule require `Self: DifferentiableProgramOperation<V, Self>` without sending the trait
/// solver into an unbounded recursion. Higher-order payloads depend on this semantic witness instead of reproducing the
/// full linearization obligation.
///
/// This trait is intentionally about complete operation families rather than individual primitive payloads, and is
/// implemented explicitly per
/// operation enum rather than through a blanket impl (a blanket
/// `impl DifferentiableProgramOperation for O where O: DifferentiableOperation` would reintroduce exactly the
/// recursion this trait exists to break).
///
/// The value type `V` (whose carried type descriptor types the programs) and operation family `O` match the primal
/// program being linearized.
pub trait DifferentiableProgramOperation<V: Value, O>: Clone + Operation<V::Type> + Sized
where
    O: Clone + Operation<V::Type> + From<ZeroOperation<V::Type>>,
{
    /// Linearizes `program` capture-free; refer to the documentation of
    /// [`Program::linearize`](crate::Program::linearize) for the returned packaging.
    ///
    /// # Parameters
    ///
    ///   - `program`: Already-traced flat sub-program over this operation family, with [`Vec`]-parameterized inputs and
    ///     outputs.
    fn linearize_program(program: &Program<V, Self, Vec<V>, Vec<V>>) -> Result<Linearization<V, Self>, ProgramError>;

    /// Builds the *fused* jvp program of `program` over `[primals..., tangents...]`, producing
    /// `[primal_outputs..., tangent_outputs...]`, without splitting it into primal and tangent halves.
    ///
    /// This is what the fused higher-order JVP rules (`scan`/`condition`) stage as their nested jvp bodies: keeping
    /// the body fused defers the primal/tangent separation to the partial-evaluation known-ness split that
    /// [`Program::linearize`](crate::Program::linearize) performs, so pure forward mode stages no residual stacks
    /// and pays a single loop pass.
    ///
    /// # Parameters
    ///
    ///   - `program`: Already-traced flat sub-program over this operation family, with [`Vec`]-parameterized inputs
    ///     and outputs.
    fn jvp_program(
        program: &Program<V, Self, Vec<V>, Vec<V>>,
    ) -> Result<Program<V, Self, Vec<V>, Vec<V>>, ProgramError>;
}

/// Replays a flat sub-program of a [`Linearization`] through a context's [`bind`](Context::bind) by interpreting
/// each instruction with [`Program::interpret_with`].
///
/// This is the value-level driver shared by the forward and reverse paths. Both sub-programs are expressed
/// in the primal operation family `<C as Domain>::Operation` over constants `<C as Domain>::Constant`, so a sub-program
/// can be replayed exactly like the primal program: its constants are lifted with [`Context::lift`] and each
/// instruction is bound with [`Context::bind`]. This gives the eager/staging duality for free — an eager `bind`
/// computes the operation immediately, while a staging `bind` splices it into the active trace. Because tangents are
/// ordinary [`Value`](DispatchDomain::Value)s of the same universe as the primals, the tangent sub-program (whose
/// leaves are tangents) is replayed through the same `bind` with no tangent-context bridging.
///
/// This is the plain-program sibling of [`PartialEvaluation::interpret`], which replays a partial-evaluation
/// residual program by additionally wiring its residual-input feeders; both share the same
/// [`interpret_with`](Program::interpret_with) + [`lift`](Context::lift)/[`bind`](Context::bind) shape.
///
/// # Parameters
///
///   - `context`: Context whose [`lift`](Context::lift) and [`bind`](Context::bind) interpret the sub-program.
///   - `program`: Flat sub-program over the primal operation family, taking and producing flat
///     [`Vec`]s of constants.
///   - `inputs`: Flat input values aligned with the sub-program's input atoms.
pub(crate) fn replay_via_bind<C, Input, Output>(
    context: &C,
    program: &Program<<C as Domain>::Constant, <C as Domain>::Operation, Input, Output>,
    inputs: Vec<<C as Domain>::Value>,
) -> Result<Vec<<C as Domain>::Value>, ProgramError>
where
    C: Context,
    <C as Domain>::Constant: Clone,
    <C as Domain>::Operation: Clone,
    Input: Parameterized<<C as Domain>::Constant>,
    Output: Parameterized<<C as Domain>::Constant>,
{
    program.interpret_with(
        inputs,
        |_, constant| context.lift(constant.clone()),
        |instruction, inputs| context.bind(instruction.operation().clone(), inputs),
    )
}

/// Transposes a [`Linearization`]'s tangent sub-program into the reverse-mode pullback directly, without re-keying
/// it into a linear operation enum.
///
/// Rather than re-keying each bilinear operation of the tangent sub-program into a closed captured factor (for example,
/// folding a scalar `Mul` against a known operand into a multiply-by-a-captured-constant) by folding the consuming
/// residual value, this function leaves the tangent sub-program in the primal operation family `O` and transposes it
/// through
/// [`Program::transpose_with_respect_to`](crate::Program::transpose_with_respect_to). The tangent sub-program's inputs are
/// `[tangents..., residuals...]`, so the partition mask marks the leading `tangent_input_count` inputs linear and the
/// trailing [`residual_count`](Linearization::residual_count) inputs known. Partition-aware transposition then
/// threads each known residual through to the pullback as a pullback input (consumed by the adjoint operation that the
/// bilinear operation's transpose rule stages), rather than folding it into a captured factor. The returned pullback is
/// therefore over the primal operation family `O` and maps `(output_cotangents ++ residuals)` to the cotangents of the
/// linear tangent inputs only.
///
/// The value type `V` and operation family `O` match the primal program that was
/// program being linearized.
///
/// # Parameters
///
///   - `linearization`: Linearization whose tangent sub-program is transposed.
pub fn transpose_tangent_partitioned<T, V, O>(
    linearization: &Linearization<V, O>,
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
where
    T: DifferentiableType,
    V: Value<Type = T>,
    O: Clone + Operation<T> + TransposableOperation<V, O>,
    O: From<ZeroOperation<T>> + From<AddOperation>,
{
    let tangent_program = &linearization.tangent_program;
    let residual_count = linearization.residual_count;
    let tangent_input_count = tangent_program.input_ids().len().checked_sub(residual_count).ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "tangent program has {} inputs which is fewer than its {} residuals",
            tangent_program.input_ids().len(),
            residual_count,
        ))
    })?;

    // Transpose with respect to the leading tangent inputs, holding the trailing residual inputs as known
    // parameters. Partial transposition exposes each known residual as a pullback input, so the residuals are not
    // folded into captured factors here.
    let with_respect_to = (0..tangent_input_count).collect::<Vec<_>>();
    tangent_program.transpose_with_respect_to(with_respect_to.as_slice())
}

/// A reusable forward-mode linear map produced by [`DifferentiationContext::linearize`], the JAX `linearize` analogue:
/// it pairs the primal output with a callable that pushes any tangent at the linearization point through the function's
/// Jacobian.
///
/// Where [`jvp`](DifferentiationContext::jvp) interprets the fused JVP program once per `(primal, tangent)` pair,
/// linearization partially evaluates that program against the *known* primals up front — folding the primal computation
/// and every residual factor to concrete values — and keeps only the residual program over the still-*unknown*
/// tangents. That residual program is the linear tangent map `f'(x)`: interpreting it at a flat tangent vector yields
/// the directional derivative, so the cost of differentiating once is amortized over many tangent applications (for
/// example, replaying every coordinate basis tangent to build a Jacobian).
///
/// The residual (tangent-map) program is over the primal operation family `<C as Domain>::Operation` in the staged
/// constant space `<C as Domain>::Constant`, while its feeders and outputs flow as `<C as Domain>::Value`s: under an
/// eager domain the folded residual factors are concrete values and [`apply`](Self::apply) interprets the tangent map
/// immediately, while under a staging domain they are [`Tracer`]s into the enclosing trace and
/// [`apply`](Self::apply) stages the tangent map into that trace. Each [`PartialEvaluationOutput::Known`] tangent
/// output is a structurally zero tangent that partial evaluation folded away; each
/// [`PartialEvaluationOutput::Unknown`] indexes a residual-program output.
///
/// The differentiation context `C` supplies the value semantics and operation family; `Input` is the closure's
/// structured input type and `TracedOutput` its structured output type, whose
/// [`ParameterStructure`](crate::parameters::Parameterized::ParameterStructure) is retained so the flat tangent outputs
/// reshape back into `TracedOutput::To<<C as Domain>::Value>`. `Input` is carried as a type parameter so
/// [`apply`](Self::apply) infers the tangent family from the linearization itself rather than requiring a turbofish.
pub struct ForwardLinearization<C, Input, TracedOutput>
where
    C: Context,
    TracedOutput: Parameterized<<C as Domain>::Value>,
{
    /// Partial evaluation of the fused JVP program against the known primals: its residual program is the linear
    /// tangent map over the primal operation family, its
    /// [`Known`](crate::partial::PartialEvaluationInput::Known) feeders are the folded residual factors, and its
    /// outputs split into the folded primal half followed by the tangent half.
    evaluation: PartialEvaluation<C>,

    /// Differentiation context the linearization was built in; [`apply`](Self::apply) replays the tangent map in it.
    domain: C,

    /// Number of primal inputs `n`. The fused JVP program takes `[primals(n) ++ tangents(n)]`, so
    /// [`apply`](Self::apply) expects exactly `n` flat tangents.
    primal_input_count: usize,

    /// Number of primal outputs; the fused JVP program's outputs split into the leading primal half and the trailing
    /// tangent half, and [`apply`](Self::apply) returns the tangent half.
    primal_output_count: usize,

    /// Parameter structure of the closure's output, used to reshape the flat tangent outputs.
    output_structure: TracedOutput::ParameterStructure,

    /// Encodes the closure's input family `Input` so [`apply`](Self::apply) can reshape the flat tangents without a
    /// turbofish. Covariant and ownership-free.
    marker: PhantomData<fn() -> Input>,
}

impl<C, Input, TracedOutput> ForwardLinearization<C, Input, TracedOutput>
where
    C: Context,
    <C as Domain>::Operation: Clone,
    Input: Parameterized<<C as Domain>::Value>,
    TracedOutput: Parameterized<<C as Domain>::Value>,
    TracedOutput::Family: ParameterizedFamily<<C as Domain>::Value>,
{
    /// Pushes the structured tangents `tangents` through the linearized Jacobian, returning the tangent outputs.
    ///
    /// The tangents are flattened and the residual tangent map is replayed through
    /// [`PartialEvaluation::interpret`] in the differentiation context this linearization was built in — the single
    /// replay path for both known-side flavors: an eager domain interprets the tangent map immediately, while a
    /// staging domain stages it into the enclosing trace and returns tracers. Each tangent output is mapped from its
    /// source — a folded structural zero returns its folded value and the rest index the replayed residual-program
    /// outputs — before being reshaped against the closure's output structure.
    ///
    /// # Parameters
    ///
    ///   - `tangents`: Structured tangents at the linearization point, matching the closure's input structure.
    pub fn apply(
        &self,
        tangents: Input::To<<C as Domain>::Value>,
    ) -> Result<TracedOutput::To<<C as Domain>::Value>, ProgramError> {
        let tangents = tangents.into_parameters().collect::<Vec<_>>();
        if tangents.len() != self.primal_input_count {
            return Err(ProgramError::InvalidInputCount { expected: self.primal_input_count, actual: tangents.len() });
        }

        // Replay the residual program: the fused JVP program's unknown inputs are exactly the tangents, in order, so
        // the flat tangent vector feeds the unknown residual inputs directly. The replayed outputs are the fused
        // program's outputs — the folded primal half followed by the tangent half — and `apply` returns the latter.
        let mut outputs = self.evaluation.interpret(&self.domain, tangents.as_slice())?;
        let tangent_values = outputs.split_off(self.primal_output_count);
        Ok(TracedOutput::To::<<C as Domain>::Value>::from_parameters(self.output_structure.clone(), tangent_values)?)
    }
}

/// A reusable reverse-mode linear map produced by [`DifferentiationContext::vjp_fn`]: it wraps the pullback program and
/// linearization-point residuals that [`DifferentiationContext::vjp`] returns behind a callable that maps output
/// cotangents to input cotangents — the JAX `vjp` analogue.
///
/// The raw [`vjp`](DifferentiationContext::vjp) returns a pullback program plus the residuals it consumes; reconstructing
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
    /// Pullback program over the primal operation family, mapping `[output_cotangents ++ residuals]` to flat input
    /// cotangents.
    pub(crate) program:
        Program<<C as Domain>::Value, <C as Domain>::Operation, Vec<<C as Domain>::Value>, Vec<<C as Domain>::Value>>,

    /// Linearization-point residuals consumed by [`program`](Self::program), appended after the output cotangents when
    /// interpreting it.
    pub(crate) residuals: Vec<<C as Domain>::Value>,

    /// Parameter structure of the closure's input, used to reshape the flat input cotangents.
    pub(crate) input_structure: Input::ParameterStructure,

    /// Encodes the closure's output family `TracedOutput` so [`apply`](Self::apply) can flatten the cotangents without a
    /// turbofish. Covariant and ownership-free.
    pub(crate) marker: PhantomData<fn() -> TracedOutput>,
}

impl<C, Input, TracedOutput> Pullback<C, Input, TracedOutput>
where
    C: Context,
    <C as Domain>::Operation: Clone
        + InterpretableOperation<<C as Domain>::Value, EagerContext<<C as Domain>::Value, <C as Domain>::Operation>>,
    Input: Parameterized<<C as Domain>::Value>,
    Input::Family: ParameterizedFamily<<C as Domain>::Value>,
    TracedOutput: Parameterized<<C as Domain>::Value>,
{
    /// Pulls the structured output cotangents `cotangents` back to the closure's input cotangents.
    ///
    /// The cotangents are flattened, the linearization-point residuals are appended, the pullback program is
    /// interpreted at that vector under the domain's eager [`EagerContext`], and the flat
    /// input cotangents are reshaped against the closure's input structure.
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
        let context = EagerContext::<<C as Domain>::Value, <C as Domain>::Operation>::new();
        let input_cotangents = self.program.interpret_in_context(&context, inputs)?;
        Ok(Input::To::<<C as Domain>::Value>::from_parameters(self.input_structure.clone(), input_cotangents)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::Sin;
    use crate::scalars::Scalar;
    use crate::tracing_v2::test_util::assert_scalar_close;

    use super::DifferentiationContext;
    use crate::contexts::EagerContext;

    #[test]
    fn test_nested_value_and_grad_computes_the_analytic_second_derivative() {
        // Reverse-over-reverse through closure-level nesting: the outer transform differentiates a closure that
        // itself calls `value_and_gradient` on the nested tracing context its tracer flows in. For f(x) = sin(x²),
        // the outer value is f'(x) = 2x cos(x²) and the outer gradient is f''(x) = 2 cos(x²) - 4x² sin(x²).
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (value, gradient) = domain
            .value_and_grad(
                |x| {
                    let context = x.context().clone();
                    context.value_and_gradient(|y| (y.clone() * y).sin().unwrap(), x).unwrap()
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
            .value_and_grad(
                |x| {
                    let context = x.context().clone();
                    context
                        .value_and_gradient(
                            |y| {
                                let context = y.context().clone();
                                context.value_and_gradient(|z| (z.clone() * z).sin().unwrap(), y).unwrap()
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
        // f''(x)·v without materializing a dense Hessian. The `jvp` closure receives `JvpTracer` duals whose
        // stamped `JvpContext` is itself a `DifferentiationContext`, so the inner reverse-mode transform nests on
        // it and differentiates through the duals. For f(x) = sin(x²) at x = 0.7 with v = 2, the primal is f'(0.7)
        // and the tangent is 2·f''(0.7).
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) = domain
            .jvp(
                |x| {
                    let context = x.context().clone();
                    Ok(context.value_and_gradient(|y| (y.clone() * y).sin().unwrap(), x).unwrap())
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
