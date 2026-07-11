//! Gradient rematerialization / rematerialization — the analogue of JAX's
//! [`jax.checkpoint` / `jax.remat`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html).
//!
//! [`rematerialize`] wraps a function so that reverse-mode differentiation through it trades memory for compute:
//! instead of storing every linearization residual produced inside the wrapped region, only the region's inputs
//! (plus any values selected by a [`RematerializationPolicy`]) are saved, and everything else is recomputed from them in
//! the backward pass.
//!
//! Each [`Rematerialize::call`] stages a [`RematerializeOperation`] by deriving forward, backward, and tangent
//! programs symbolically: the forward program computes the region outputs plus the saved values, the backward program
//! recomputes the remaining linearization residuals before replaying the transposed tangent map, and the tangent
//! program replays the pushforward for forward mode.
//!
//! The name-based [`RematerializationPolicy`] members classify residuals by the
//! [`tag`](crate::operations::tag::Tag::tag) key carried by [`TagOperation`](crate::operations::tag::TagOperation).

use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use crate::batching::ArrayBatch;
use crate::batching::BatchingContext;
use crate::batching::BatchingError;
use crate::batching::{BatchableOperation, BatchableProgramOperation};
use crate::contexts::{Context, Domain, EagerContext, StagingContext};
use crate::differentiation::DifferentiationDual;
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationError, TransposableOperation,
};
use crate::effects::Effects;
use crate::interpretation::{InterpretableOperation, InterpretableProgramOperation};
use crate::macros::{check_count, check_types};
use crate::operations::Operation;
use crate::operations::constants::{Constant as ConstantCapability, Zero, ZeroOperation};
use crate::operations::control_flow::MaybeScan;
use crate::operations::manipulation::{Broadcast, BroadcastOperation, Transpose, TransposeOperation};
use crate::operations::math::AddOperation;
use crate::operations::tag::MaybeTag;
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::payloads::Captured;
use crate::programs::{AtomId, MaybeZero, Program, ProgramError, Value};
use crate::tracing::{DomainTracer, Trace, Tracer, TracingContext};
use crate::tracing_v2::operations::custom_derivatives::{
    CustomVjpResidual, batch_rewrapped_program, stage_rewrapped_custom_call,
};
use crate::tracing_v2::operations::dot::{DotDimensionNumbers, MaybeDot};
use crate::tracing_v2::operations::memory::{TransferToMemory, TransferToMemoryOperation};
use crate::types::{ArrayType, Memory, Type, TypeError, Typed};

/// Higher-order operation used by checkpointing/rematerialization.
///
/// [`RematerializeOperation`] has the same primal/forward/backward structure as
/// [`CustomVjpOperation`](crate::tracing_v2::operations::custom_derivatives::CustomVjpOperation), but it also carries
/// a derived tangent program. That extra program is not user-authored custom-VJP state: it is produced by
/// [`Rematerialize`] so forward-mode differentiation can replay the rematerialized pushforward while reverse mode
/// replays the rematerialized pullback.
///
/// The `prevent_cse` flag is likewise rematerialization-specific. Backends may lower it as an optimization barrier
/// around rematerialized tangent/pullback outputs so compiler common-subexpression elimination does not undo the
/// requested memory/computation tradeoff.
#[derive(Clone, Debug)]
pub struct RematerializeOperation<V: Value, O> {
    /// Program computing the primal outputs from the primal inputs.
    primal: Program<V, O, Vec<V>, Vec<V>>,

    /// Program computing `(outputs..., residuals...)` from the primal inputs.
    forward: Program<V, O, Vec<V>, Vec<V>>,

    /// Program computing one input cotangent per primal input from `(residuals..., output_cotangents...)`.
    backward: Program<V, O, Vec<V>, Vec<V>>,

    /// Program computing one output tangent per primal output from `(residuals..., input_tangents...)`.
    tangent: Program<V, O, Vec<V>, Vec<V>>,

    /// Backend lowering hint requesting an optimization barrier around rematerialized backward/tangent outputs.
    prevent_cse: bool,
}

impl<V: Value, O: Operation<V::Type>> RematerializeOperation<V, O> {
    /// Creates a rematerialization operation after validating the forward, backward, and tangent program signatures.
    pub fn new(
        primal: Program<V, O, Vec<V>, Vec<V>>,
        forward: Program<V, O, Vec<V>, Vec<V>>,
        backward: Program<V, O, Vec<V>, Vec<V>>,
        tangent: Program<V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        let input_types = primal.input_types();
        let output_types = primal.output_types();
        check_types!("rematerialize forward input", &input_types, &forward.input_types());
        let forward_output_types = forward.output_types();
        if forward_output_types.len() < output_types.len() {
            return Err(TypeError {
                message: format!(
                    "rematerialize forward must produce at least the {} primal output(s) but produced {} value(s)",
                    output_types.len(),
                    forward_output_types.len(),
                ),
            });
        }
        check_types!("rematerialize forward output", &output_types, &forward_output_types[..output_types.len()]);
        let residual_types = &forward_output_types[output_types.len()..];
        let expected_backward_input_types: Vec<V::Type> =
            residual_types.iter().chain(output_types.iter()).cloned().collect();
        check_types!("rematerialize backward input", &expected_backward_input_types, &backward.input_types(),);
        check_types!("rematerialize backward output", &input_types, &backward.output_types());
        let expected_tangent_input_types: Vec<V::Type> =
            residual_types.iter().chain(input_types.iter()).cloned().collect();
        check_types!("rematerialize tangent input", &expected_tangent_input_types, &tangent.input_types());
        check_types!("rematerialize tangent output", &output_types, &tangent.output_types());
        Ok(Self { primal, forward, backward, tangent, prevent_cse: false })
    }

    /// Sets whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier
    /// (e.g., StableHLO's `optimization_barrier`). Without a barrier, a compiler may common-subexpression-eliminate
    /// values recomputed by the backward or tangent program against the forward pass, silently restoring the memory
    /// cost the rematerialization was meant to avoid.
    pub fn with_prevent_cse(mut self, prevent_cse: bool) -> Self {
        self.prevent_cse = prevent_cse;
        self
    }

    /// Returns whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier.
    #[inline]
    pub fn prevent_cse(&self) -> bool {
        self.prevent_cse
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

    /// Returns the tangent-producing program.
    #[inline]
    pub fn tangent(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.tangent
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

impl<V: Value, O> Display for RematerializeOperation<V, O>
where
    Self: Operation<V::Type>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value, O: Operation<V::Type>> Operation<V::Type> for RematerializeOperation<V, O> {
    #[inline]
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn infer_output_types(&self, input_types: &[V::Type]) -> Result<Vec<V::Type>, TypeError> {
        check_types!("rematerialize input", &self.input_types(), input_types);
        Ok(self.output_types())
    }

    #[inline]
    fn effects(&self) -> Effects {
        // The forward, backward, and tangent programs replay the primal, so the primal's summary covers them; the
        // union guards derived programs that stage extra effectful work (none do today).
        self.primal
            .effects()
            .union(self.forward.effects())
            .union(self.backward.effects())
            .union(self.tangent.effects())
    }
}

impl<Constant, O, V, C> InterpretableOperation<V, C> for RematerializeOperation<Constant, O>
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
/// [`RematerializeOperation`]: a call with all-known operands folds by interpreting its primal, and otherwise
/// residualizes unchanged.
impl<V: Value, O: Clone + Operation<V::Type>, C: Context<Type = V::Type>> PartiallyEvaluatableOperation<C>
    for RematerializeOperation<V, O>
where
    C::Operation: From<RematerializeOperation<V, O>>,
{
}

/// Capture-free forward-mode (JVP) rule for [`RematerializeOperation`]: splices the derived forward and tangent
/// programs directly into the shared builder.
///
/// Both derived programs are ordinary primal-enum programs, so the rule replays them through
/// [`Program::interpret_in_context`](crate::Program::interpret_in_context):
///
///   1. The forward program maps `inputs -> (outputs..., forward_tail...)`, where the tail is the region inputs
///      followed by the policy-saved residuals. Replaying it on the dual primals yields the primal outputs and the
///      forward tail; the tail is split off after the primal outputs.
///   2. The tangent program maps `(forward_tail..., input_tangents...) -> output_tangents`, exactly the forward tail
///      followed by the input tangents (per [`RematerializeOperation::new`]'s signature validation), so the tail is
///      spliced verbatim ahead of the dual tangents and replayed to produce the output tangents. The tangent program
///      recomputes any unsaved residuals from the tail internally, so no residual reconstruction is needed here.
///   3. Each primal output is paired with its staged output tangent into a [`DifferentiationDual`].
///
/// Because both spliced programs are straight-line primal-enum operations referencing the staged tracers directly,
/// the rule introduces no symbolic capture and the enclosing partial-evaluation split discovers the residual
/// operand edges structurally — so
/// this is a leaf rule needing no [`DifferentiableProgramOperation`](crate::differentiation::DifferentiableProgramOperation)
/// or [`LinearizableProgramOperation`](crate::differentiation::LinearizableProgramOperation)
/// witness, and reverse mode transposes the spliced recompute-and-pushforward operations like any other straight-line
/// tangent program. The [`prevent_cse`](RematerializeOperation::prevent_cse) optimization-barrier hint is
/// dropped in the forward (it is a backend lowering hint with no value-level semantics).
impl<C: Context + Zero<C::Value>> DifferentiableOperation<C> for RematerializeOperation<C::Constant, C::Operation>
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

        // Splice the forward program on the dual primals, recovering the primal outputs followed by the forward tail
        // (region inputs plus policy-saved residuals) that the tangent program consumes.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut forward_outputs = self.forward().interpret_in_context(context, primal_operands)?;
        if forward_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "rematerialize forward program produced {} outputs which is fewer than its {output_count} \
                 primal output(s)",
                forward_outputs.len(),
            ))
            .into());
        }
        let forward_tail = forward_outputs.split_off(output_count);
        let primal_outputs = forward_outputs;

        // Splice the tangent program on `(forward_tail..., input_tangents...)`, yielding one output tangent per primal
        // output.
        let mut tangent_operands = forward_tail;
        // The rematerialize call takes every input tangent as a real operand, so materialize structural zeros.
        for input in inputs {
            tangent_operands.push(input.tangent().clone().materialize(context)?);
        }
        let tangent_outputs = self.tangent().interpret_in_context(context, tangent_operands)?;
        check_count!("output", tangent_outputs, output_count, ProgramError);

        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect())
    }
}

/// Transpose rule for [`RematerializeOperation`]: the call is a higher-order primal boundary rather than a linear
/// map, so a tangent program never contains it on a linear operand (linearization splices the derived forward and
/// tangent programs instead) and the rule reports an
/// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
impl<V, O, W, OLinear> TransposableOperation<W, OLinear> for RematerializeOperation<V, O>
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

/// Value-level batching for [`RematerializeOperation`]: inlines the primal program.
impl<V, O> BatchableOperation<V, EagerContext<V, O>> for RematerializeOperation<V, O>
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

/// Traced batching for [`RematerializeOperation`]: re-wraps the call around batched primal/forward/backward/tangent
/// programs so the rematerialization boundary survives `batch`; see `stage_rewrapped_custom_call`.
impl<C, O> BatchableOperation<<C as Domain>::Value, BatchingContext<C>> for RematerializeOperation<C::Constant, O>
where
    C: Context<Type = ArrayType, Operation = O>,
    C::Constant: Value<Type = ArrayType>,
    <C as Domain>::Value: Broadcast + Transpose,
    O: Clone
        + Operation<ArrayType>
        + From<TransposeOperation>
        + From<BroadcastOperation>
        + From<RematerializeOperation<C::Constant, O>>
        + BatchableProgramOperation<C::Constant>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        stage_rewrapped_custom_call(context, inputs, |batched| match batched {
            None => Ok(O::from(self.clone())),
            Some(axis_size) => Ok(O::from(
                RematerializeOperation::new(
                    batch_rewrapped_program(&self.primal, axis_size)?,
                    batch_rewrapped_program(&self.forward, axis_size)?,
                    batch_rewrapped_program(&self.backward, axis_size)?,
                    batch_rewrapped_program(&self.tangent, axis_size)?,
                )?
                .with_prevent_cse(self.prevent_cse),
            )),
        })
    }
}

/// Linear operation staged by [`RematerializeOperation`]'s JVP rule.
///
/// Unlike [`CustomVjpCallOperation`](crate::tracing_v2::operations::custom_derivatives::CustomVjpCallOperation),
/// both directions are executable: the un-transposed form replays the derived tangent program, and the transposed
/// form replays the derived pullback program. This is what lets rematerialized regions support forward mode while
/// user-authored [`CustomVjpOperation`](crate::tracing_v2::operations::custom_derivatives::CustomVjpOperation)
/// remains reverse-mode-only.
#[derive(Clone, Debug)]
pub struct RematerializeCallOperation<V: Value, O, F: Value<Type = V::Type>> {
    /// Derived backward program, mapping `(residuals..., output_cotangents...)` to input cotangents.
    backward: Program<V, O, Vec<V>, Vec<V>>,

    /// Derived tangent program, mapping `(residuals..., input_tangents...)` to output tangents.
    tangent: Program<V, O, Vec<V>, Vec<V>>,

    /// Captured residual factors consumed by the tangent or backward program.
    residuals: Vec<F>,

    /// Whether this call has been transposed into its pullback form.
    transposed: bool,

    /// Backend lowering hint requesting an optimization barrier around the lowered program outputs.
    prevent_cse: bool,
}

impl<V: Value, F: Value<Type = V::Type>, O> RematerializeCallOperation<V, O, F> {
    /// Creates a rematerialization call. Use `transposed = false` for the tangent form and `transposed = true` for
    /// the pullback form.
    pub fn new(
        backward: Program<V, O, Vec<V>, Vec<V>>,
        tangent: Program<V, O, Vec<V>, Vec<V>>,
        residuals: Vec<F>,
        transposed: bool,
        prevent_cse: bool,
    ) -> Self {
        Self { backward, tangent, residuals, transposed, prevent_cse }
    }

    /// Returns the derived backward program.
    #[inline]
    pub fn backward(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.backward
    }

    /// Returns the derived tangent program.
    #[inline]
    pub fn tangent(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.tangent
    }

    /// Returns the captured residual factors.
    #[inline]
    pub fn residuals(&self) -> &[F] {
        self.residuals.as_slice()
    }

    /// Returns whether this call is in its transposed (pullback) form.
    #[inline]
    pub fn transposed(&self) -> bool {
        self.transposed
    }

    /// Returns whether backends should wrap this call's lowered program outputs in an optimization barrier.
    #[inline]
    pub fn prevent_cse(&self) -> bool {
        self.prevent_cse
    }

    /// Maps the residual factor payloads with `map_factor`, preserving the captured programs and direction.
    pub fn map_captures<MappedFactor: Value<Type = V::Type>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<RematerializeCallOperation<V, O, MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
        O: Clone,
    {
        Ok(RematerializeCallOperation {
            backward: self.backward.clone(),
            tangent: self.tangent.clone(),
            residuals: self.residuals.iter().map(map_factor).collect::<Result<Vec<_>, _>>()?,
            transposed: self.transposed,
            prevent_cse: self.prevent_cse,
        })
    }
}

impl<V: Value, F: Value<Type = V::Type>, O: Operation<V::Type>> RematerializeCallOperation<V, O, F> {
    /// Returns the cotangent types flowing *into* the backward program (one per primal output).
    fn cotangent_types(&self) -> Vec<V::Type> {
        self.backward.input_types().split_off(self.residuals.len())
    }

    /// Returns the tangent types flowing *into* the tangent program (one per primal input).
    fn tangent_input_types(&self) -> Vec<V::Type> {
        self.tangent.input_types().split_off(self.residuals.len())
    }
}

impl<V: Value, F: Value<Type = V::Type>, O> Display for RematerializeCallOperation<V, O, F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.transposed {
            formatter.write_str("rematerialize_backward")
        } else {
            formatter.write_str("rematerialize_tangent")
        }
    }
}

impl<V: Value, F: Value<Type = V::Type>, O: Operation<V::Type>> Operation<V::Type>
    for RematerializeCallOperation<V, O, F>
{
    #[inline]
    fn name(&self) -> &'static str {
        if self.transposed { "rematerialize_backward" } else { "rematerialize_tangent" }
    }

    fn infer_output_types(&self, input_types: &[V::Type]) -> Result<Vec<V::Type>, TypeError> {
        if self.transposed {
            check_types!("rematerialize backward cotangent", &self.cotangent_types(), input_types);
            Ok(self.backward.output_types())
        } else {
            check_types!("rematerialize tangent", &self.tangent_input_types(), input_types);
            Ok(self.tangent.output_types())
        }
    }

    #[inline]
    fn effects(&self) -> Effects {
        self.backward.effects().union(self.tangent.effects())
    }
}

impl<Constant, O, F, V, C> InterpretableOperation<V, C> for RematerializeCallOperation<Constant, O, F>
where
    Constant: Value,
    V: Value<Type = Constant::Type>,
    F: CustomVjpResidual<V>,
    O: InterpretableOperation<V, C> + Operation<Constant::Type>,
    C: ConstantCapability<V, Constant, Captured>,
    Vec<Constant>: Parameterized<Constant, ParameterStructure: Debug + PartialEq>,
{
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let program = if self.transposed { &self.backward } else { &self.tangent };
        let mut values =
            self.residuals.iter().map(|residual| residual.residual_value()).collect::<Result<Vec<_>, _>>()?;
        values.extend(inputs.iter().cloned());
        program.interpret_with(
            values,
            |_, constant| context.constant(constant.clone()),
            |instruction, inputs| instruction.operation().interpret(context, inputs),
        )
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`] for a
/// [`RematerializeCallOperation`]. The residual operation family `O` is independent of the call's primal operation
/// family `CallOperation`, because partial evaluation never inlines a nested program here and so never builds a
/// residual program of its own.
impl<V, CallOperation, F, C> PartiallyEvaluatableOperation<C> for RematerializeCallOperation<V, CallOperation, F>
where
    V: Value,
    CallOperation: Clone + Operation<V::Type>,
    F: Value<Type = V::Type>,
    C: Context<Type = V::Type>,
    C::Operation: From<RematerializeCallOperation<V, CallOperation, F>>,
{
}

/// Transpose rule for [`RematerializeCallOperation`]: stages the flipped-direction form of the call.
impl<V, O, F, W, OLinear> TransposableOperation<W, OLinear> for RematerializeCallOperation<V, O, F>
where
    V: Value,
    F: Value<Type = V::Type>,
    W: Value<Type = V::Type>,
    O: Clone + Operation<V::Type>,
    OLinear: Operation<V::Type> + From<ZeroOperation<V::Type>> + From<RematerializeCallOperation<V, O, F>>,
{
    fn transpose(
        &self,
        context: &mut TracingContext<W, OLinear>,
        _inputs: &[PartialValue<Tracer<TracingContext<W, OLinear>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<W, OLinear>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<W, OLinear>>>>, DifferentiationError> {
        let cotangent_types = if self.transposed { self.tangent_input_types() } else { self.cotangent_types() };
        check_count!("output", outputs, cotangent_types.len(), ProgramError);
        let cotangent_tracers = outputs
            .iter()
            .map(|cotangent| cotangent.clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        let call = OLinear::from(RematerializeCallOperation {
            backward: self.backward.clone(),
            tangent: self.tangent.clone(),
            residuals: self.residuals.to_vec(),
            transposed: !self.transposed,
            prevent_cse: self.prevent_cse,
        });
        let outputs = context.stage_operation(call, cotangent_tracers.as_slice())?;
        Ok(outputs.into_iter().map(MaybeZero::Value).collect())
    }
}

/// Value-level batching for [`RematerializeCallOperation`]: replays the selected tangent or backward program.
impl<V, O, F> BatchableOperation<V, EagerContext<V, O>> for RematerializeCallOperation<V, O, F>
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
        let program = if self.transposed { &self.backward } else { &self.tangent };
        let mut values = self
            .residuals
            .iter()
            .map(|residual| Ok(ArrayBatch::replicated(residual.residual_value()?)))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        values.extend(inputs.iter().cloned());
        program.interpret_with(
            values,
            |_, constant: &V| Ok(ArrayBatch::replicated(constant.clone())),
            |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
        )
    }
}

/// Classification facts about one linearization residual, exposed to [`RematerializationPolicy::Custom`] policies.
///
/// A candidate describes the instruction that produced the residual — its operation name, dot classification, and
/// [`tag`](crate::operations::tag::Tag::tag) key — together with the residual's staged type, mirroring the
/// information JAX exposes to custom `jax.checkpoint` policies (the primitive and its abstract values).
#[derive(Clone, Debug)]
pub struct RematerializationCandidate<'a, T: Type> {
    /// Name of the operation that produced the residual.
    operation_name: &'a str,

    /// Dimension numbers of the producing operation when it is a dot-like contraction.
    dot_dimensions: Option<&'a DotDimensionNumbers>,

    /// [`tag`](crate::operations::tag::Tag::tag) key on the producing operation, if any.
    key: Option<&'a str>,

    /// Staged type of the residual value.
    residual_type: T,
}

impl<'a, T: Type> RematerializationCandidate<'a, T> {
    /// Returns the name of the operation that produced the residual.
    #[inline]
    pub fn operation_name(&self) -> &'a str {
        self.operation_name
    }

    /// Returns whether the producing operation is a dot-like contraction.
    #[inline]
    pub fn is_dot(&self) -> bool {
        self.dot_dimensions.is_some()
    }

    /// Returns the dimension numbers of the producing operation when it is a dot-like contraction.
    #[inline]
    pub fn dot_dimensions(&self) -> Option<&'a DotDimensionNumbers> {
        self.dot_dimensions
    }

    /// Returns the [`tag`](crate::operations::tag::Tag::tag) key on the producing operation, if any.
    #[inline]
    pub fn key(&self) -> Option<&'a str> {
        self.key
    }

    /// Returns the staged type of the residual value.
    #[inline]
    pub fn residual_type(&self) -> &T {
        &self.residual_type
    }
}

impl<'a, T: Type> RematerializationCandidate<'a, T> {
    /// Builds the classification candidate for the residual produced at `residual_atom` of `program` by walking back to
    /// the instruction that defines it. Returns `None` for residuals that are not produced by an instruction (region
    /// inputs and constants), which policies never save; see the [`RematerializationPolicy`] documentation.
    ///
    /// This recovers the same producing-operation provenance the symbolic linearization residual tracers carried — the
    /// operation name, dot dimension numbers, and [`tag`](crate::operations::tag::Tag::tag) key — directly from
    /// the residual-producing instruction of the linearization's primal sub-program, whose trailing outputs are
    /// the residuals.
    ///
    /// # Parameters
    ///
    ///   - `program`: Primal sub-program whose trailing outputs are the residuals.
    ///   - `residual_atom`: The residual output atom whose producing instruction is classified.
    ///   - `residual_type`: Staged type of the residual value.
    fn from_program_residual<V, O>(
        program: &'a Program<V, O, Vec<V>, Vec<V>>,
        residual_atom: AtomId,
        residual_type: T,
    ) -> Option<RematerializationCandidate<'a, T>>
    where
        V: Value<Type = T>,
        O: Operation<T> + MaybeDot + MaybeTag + MaybeScan<V, O>,
    {
        let mut program = program;
        let mut atom = residual_atom;
        let operation = loop {
            let instruction =
                program.instructions().iter().rev().find(|instruction| instruction.outputs().contains(&atom))?;
            let operation = instruction.operation();
            match operation.scan_body() {
                // Look through the scan boundary: scan outputs and body outputs are index-aligned
                // (`[final_carries..., stacked...]` versus `[next_carries..., slices...]`), so the stacked residual
                // at this scan output is produced per iteration by the body instruction defining the same-index
                // body output. Nested scans recurse naturally, and a body output that is a pass-through input or a
                // constant has no producing instruction, so the residual is never saved — the same contract as at
                // the top level.
                Some(body) => {
                    let index = instruction.outputs().iter().position(|output| *output == atom).unwrap();
                    atom = body.output_ids()[index];
                    program = body;
                }
                None => break operation,
            }
        };
        Some(RematerializationCandidate {
            operation_name: operation.name(),
            dot_dimensions: operation.dot_dimensions(),
            key: operation.key(),
            residual_type,
        })
    }
}

/// Policy selecting which linearization residuals a [`Rematerialize`] saves instead of recomputing — the analogue of
/// the named members of JAX's
/// [`jax.checkpoint_policies`](https://docs.jax.dev/en/latest/gradient-checkpointing.html#custom-policies-for-what-s-saveable).
///
/// A residual is a value captured during linearization as a coefficient of the staged linear (tangent) map — for
/// example, `cos(x)` for `sin`, or the operand values for `mul`. Saved residuals are emitted as extra outputs of the
/// rematerialization's forward program and consumed directly by its backward program; unsaved residuals are recomputed in
/// the backward program from the saved values. Residuals that are region inputs or constants are never stored: the
/// backward program always receives the region inputs, and constants are re-created in place.
///
/// Custom-derivative boundaries stay opaque to policies: candidates are classified from the operations that produce
/// residuals in the rematerialized region's own linearization, so a
/// [`CustomVjpOperation`](crate::tracing_v2::operations::custom_derivatives::CustomVjpOperation) (or
/// [`CustomJvpOperation`](crate::tracing_v2::operations::custom_derivatives::CustomJvpOperation)) call inside the
/// body keeps its user-supplied derivative, the only residuals a custom-VJP call contributes are the ones its
/// user-authored forward rule declares, and the call's user-owned backward program is neither inspected nor
/// re-classified by the policy.
///
/// The name-based members classify residuals by [`tag`](crate::operations::tag::Tag::tag)
/// keys applied inside the body, [`Custom`](Self::Custom) policies classify them through a
/// [`RematerializationCandidate`], and [`SaveFromBothPolicies`](Self::SaveFromBothPolicies) combines two policies.
/// Every member answers save-or-recompute only; policies that can also *offload* saved residuals into another
/// memory space (JAX's `save_and_offload_only_these_names` / `offload_dot_with_no_batch_dims`) live in the separate
/// [`OffloadingRematerializationPolicy`] vocabulary, which is available exactly in the domains whose operation
/// types can represent memory transfers.
#[derive(Clone, Default)]
pub enum RematerializationPolicy<T: Type = ArrayType> {
    /// Save nothing beyond the region inputs; recompute every residual in the backward pass. This is the default,
    /// matching JAX's `nothing_saveable` (the default policy of `jax.checkpoint`).
    #[default]
    NothingSaveable,

    /// Save every instruction-produced residual, making the rematerialization inert: the backward pass recomputes nothing.
    /// Matches JAX's `everything_saveable`.
    EverythingSaveable,

    /// Save residuals produced by dot-like contractions (classified via [`MaybeDot`]) and recompute the rest.
    /// Matches JAX's `dots_saveable`.
    DotsSaveable,

    /// Save residuals produced by dot-like contractions whose [`DotDimensionNumbers`] have no batch dimensions and
    /// recompute the rest. Batched contractions behave more like cheap elementwise work per batch element, so
    /// saving only the unbatched ones targets the genuinely expensive matrix products. Matches JAX's
    /// `dots_with_no_batch_dims_saveable`.
    DotsWithNoBatchDimsSaveable,

    /// Save only residuals tagged with one of the provided
    /// [`tag`](crate::operations::tag::Tag::tag) keys and recompute everything else.
    /// Matches JAX's `save_only_these_names`.
    SaveOnlyTheseNames(Vec<String>),

    /// Save every *named* residual except those tagged with one of the provided names; unnamed residuals are
    /// recomputed. Matches JAX's `save_any_names_but_these`.
    SaveAnyNamesButThese(Vec<String>),

    /// Save every instruction-produced residual except those tagged with one of the provided names. Matches JAX's
    /// `save_anything_except_these_names`.
    SaveAnythingExceptTheseNames(Vec<String>),

    /// Save every residual that either of the two combined policies saves. Matches JAX's
    /// `save_from_both_policies`.
    SaveFromBothPolicies(Box<RematerializationPolicy<T>>, Box<RematerializationPolicy<T>>),

    /// Save exactly the residuals for which the provided callable returns `true`, classifying each through a
    /// [`RematerializationCandidate`]. This is the analogue of passing an arbitrary callable as a JAX
    /// `jax.checkpoint` policy. The callable is shared by reference counting, so cloned policies (and the wrappers
    /// holding them) observe the same classification function; policies never travel inside staged programs.
    Custom(Rc<dyn Fn(&RematerializationCandidate<'_, T>) -> bool>),
}

impl<T: Type> Debug for RematerializationPolicy<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NothingSaveable => formatter.write_str("NothingSaveable"),
            Self::EverythingSaveable => formatter.write_str("EverythingSaveable"),
            Self::DotsSaveable => formatter.write_str("DotsSaveable"),
            Self::DotsWithNoBatchDimsSaveable => formatter.write_str("DotsWithNoBatchDimsSaveable"),
            Self::SaveOnlyTheseNames(names) => formatter.debug_tuple("SaveOnlyTheseNames").field(names).finish(),
            Self::SaveAnyNamesButThese(names) => formatter.debug_tuple("SaveAnyNamesButThese").field(names).finish(),
            Self::SaveAnythingExceptTheseNames(names) => {
                formatter.debug_tuple("SaveAnythingExceptTheseNames").field(names).finish()
            }
            Self::SaveFromBothPolicies(first, second) => {
                formatter.debug_tuple("SaveFromBothPolicies").field(first).field(second).finish()
            }
            Self::Custom(_) => formatter.write_str("Custom(..)"),
        }
    }
}

/// [`Custom`](RematerializationPolicy::Custom) policies compare by callable identity ([`Rc::ptr_eq`]): two custom
/// policies are equal exactly when they share the same classification function. Every other variant compares
/// structurally.
impl<T: Type> PartialEq for RematerializationPolicy<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NothingSaveable, Self::NothingSaveable)
            | (Self::EverythingSaveable, Self::EverythingSaveable)
            | (Self::DotsSaveable, Self::DotsSaveable)
            | (Self::DotsWithNoBatchDimsSaveable, Self::DotsWithNoBatchDimsSaveable) => true,
            (Self::SaveOnlyTheseNames(left), Self::SaveOnlyTheseNames(right))
            | (Self::SaveAnyNamesButThese(left), Self::SaveAnyNamesButThese(right))
            | (Self::SaveAnythingExceptTheseNames(left), Self::SaveAnythingExceptTheseNames(right)) => left == right,
            (
                Self::SaveFromBothPolicies(left_first, left_second),
                Self::SaveFromBothPolicies(right_first, right_second),
            ) => left_first == right_first && left_second == right_second,
            (Self::Custom(left), Self::Custom(right)) => Rc::ptr_eq(left, right),
            _ => false,
        }
    }
}

impl<T: Type> Eq for RematerializationPolicy<T> {}

impl<T: Type> RematerializationPolicy<T> {
    /// Returns whether this policy saves the residual described by `candidate`.
    fn saves_candidate(&self, candidate: &RematerializationCandidate<'_, T>) -> bool {
        match self {
            Self::NothingSaveable => false,
            Self::EverythingSaveable => true,
            Self::DotsSaveable => candidate.is_dot(),
            Self::DotsWithNoBatchDimsSaveable => candidate.dot_dimensions().is_some_and(|dimensions| {
                dimensions.lhs_batching_dimensions().is_empty() && dimensions.rhs_batching_dimensions().is_empty()
            }),
            Self::SaveOnlyTheseNames(names) => candidate.key().is_some_and(|name| names.iter().any(|n| n == name)),
            Self::SaveAnyNamesButThese(names) => candidate.key().is_some_and(|name| !names.iter().any(|n| n == name)),
            Self::SaveAnythingExceptTheseNames(names) => {
                !candidate.key().is_some_and(|name| names.iter().any(|n| n == name))
            }
            Self::SaveFromBothPolicies(first, second) => {
                first.saves_candidate(candidate) || second.saves_candidate(candidate)
            }
            Self::Custom(policy) => policy(candidate),
        }
    }
}

/// Three-way disposition of one linearization residual under an [`OffloadingRematerializationPolicy`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RematerializationVerdict {
    /// Save the residual as a forward-program output in the memory space it was produced in.
    Save,

    /// Save the residual as a forward-program output, parked in `destination` between the forward and backward
    /// passes: the forward program transfers it right after it is produced, and the backward (and tangent) programs
    /// transfer it back to its source memory right before consuming it.
    Offload {
        /// Destination [`Memory`] the saved residual is parked in.
        destination: Memory,
    },

    /// Recompute the residual in the backward program instead of saving it.
    Recompute,
}

/// [`RematerializationPolicy`] companion vocabulary whose verdicts can also *offload* saved residuals — park them
/// in another memory space (canonically pinned host memory) between the forward and backward passes — covering the
/// offloading members of JAX's
/// [`jax.checkpoint_policies`](https://docs.jax.dev/en/latest/gradient-checkpointing.html#custom-policies-for-offloadable-and-saveable-names).
///
/// Offloaded residuals are still saved (not recomputed): the forward program emits them as outputs behind a staged
/// memory transfer, so the saved types carry the destination space, and the backward and tangent programs transfer
/// them back to their source memory right before consuming them. Backends then legalize the staged transfers into
/// their native placement machinery (the XLA backend lowers them to the device-placement annotations consumed by
/// its host-offloading pipeline), so the residuals do not occupy device memory between the two passes.
///
/// This is a separate type from [`RematerializationPolicy`] — rather than a set of extra members — because
/// offloading requires the domain's operation types to represent memory transfers: the
/// [`ResidualHandling`] impl for this type is bounded on
/// [`From<TransferToMemoryOperation>`](TransferToMemoryOperation), so
/// configuring an offloading policy in a domain without that capability (for example, a scalar domain) is a
/// compile-time error at the configuration site rather than a runtime failure.
/// Plain policies compose through [`Base`](Self::Base).
#[derive(Clone)]
pub enum OffloadingRematerializationPolicy {
    /// Classifies with a plain save-or-recompute [`RematerializationPolicy`], letting the two vocabularies compose
    /// (for example, inside [`SaveFromBothPolicies`](Self::SaveFromBothPolicies)).
    Base(RematerializationPolicy<ArrayType>),

    /// Saves residuals tagged with one of the `saveable` [`tag`](crate::operations::tag::Tag::tag) keys in
    /// place, offloads residuals tagged with one of the `offloadable` names to `destination`, and recomputes
    /// everything else (including unnamed residuals). Matches JAX's `save_and_offload_only_these_names`.
    SaveAndOffloadOnlyTheseNames {
        /// Names whose residuals are saved in their own memory space.
        saveable: Vec<String>,

        /// Names whose residuals are saved behind a transfer into `destination`.
        offloadable: Vec<String>,

        /// Destination [`Memory`] for the offloaded residuals.
        destination: Memory,
    },

    /// Offloads residuals produced by dot-like contractions whose [`DotDimensionNumbers`] have no batch dimensions
    /// to `destination` and recomputes the rest. Matches JAX's `offload_dot_with_no_batch_dims`.
    OffloadDotsWithNoBatchDims {
        /// Destination [`Memory`] for the offloaded residuals.
        destination: Memory,
    },

    /// Combines two offloading policies: the first non-[`Recompute`](RematerializationVerdict::Recompute) verdict
    /// wins, mirroring the boolean-`or` short-circuit of [`RematerializationPolicy::SaveFromBothPolicies`]. In
    /// particular, when the first policy saves a residual in place, the second policy never gets to offload it.
    SaveFromBothPolicies(Box<OffloadingRematerializationPolicy>, Box<OffloadingRematerializationPolicy>),

    /// Returns an arbitrary three-way [`RematerializationVerdict`] for each residual, classifying it through a
    /// [`RematerializationCandidate`]. The callable is shared by reference counting, so cloned policies observe the
    /// same classification function; policies never travel inside staged programs.
    Custom(Rc<dyn Fn(&RematerializationCandidate<'_, ArrayType>) -> RematerializationVerdict>),
}

impl Debug for OffloadingRematerializationPolicy {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base(policy) => formatter.debug_tuple("Base").field(policy).finish(),
            Self::SaveAndOffloadOnlyTheseNames { saveable, offloadable, destination } => formatter
                .debug_struct("SaveAndOffloadOnlyTheseNames")
                .field("saveable", saveable)
                .field("offloadable", offloadable)
                .field("destination", destination)
                .finish(),
            Self::OffloadDotsWithNoBatchDims { destination } => {
                formatter.debug_struct("OffloadDotsWithNoBatchDims").field("destination", destination).finish()
            }
            Self::SaveFromBothPolicies(first, second) => {
                formatter.debug_tuple("SaveFromBothPolicies").field(first).field(second).finish()
            }
            Self::Custom(_) => formatter.write_str("Custom(..)"),
        }
    }
}

/// [`Custom`](OffloadingRematerializationPolicy::Custom) policies compare by callable identity ([`Rc::ptr_eq`]):
/// two custom policies are equal exactly when they share the same classification function. Every other variant
/// compares structurally.
impl PartialEq for OffloadingRematerializationPolicy {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Base(left), Self::Base(right)) => left == right,
            (
                Self::SaveAndOffloadOnlyTheseNames {
                    saveable: left_saveable,
                    offloadable: left_offloadable,
                    destination: left_destination,
                },
                Self::SaveAndOffloadOnlyTheseNames {
                    saveable: right_saveable,
                    offloadable: right_offloadable,
                    destination: right_destination,
                },
            ) => {
                left_saveable == right_saveable
                    && left_offloadable == right_offloadable
                    && left_destination == right_destination
            }
            (
                Self::OffloadDotsWithNoBatchDims { destination: left },
                Self::OffloadDotsWithNoBatchDims { destination: right },
            ) => left == right,
            (
                Self::SaveFromBothPolicies(left_first, left_second),
                Self::SaveFromBothPolicies(right_first, right_second),
            ) => left_first == right_first && left_second == right_second,
            (Self::Custom(left), Self::Custom(right)) => Rc::ptr_eq(left, right),
            _ => false,
        }
    }
}

impl Eq for OffloadingRematerializationPolicy {}

impl OffloadingRematerializationPolicy {
    /// Returns the [`RematerializationVerdict`] for the residual described by `candidate`.
    fn classify_candidate(&self, candidate: &RematerializationCandidate<'_, ArrayType>) -> RematerializationVerdict {
        match self {
            Self::Base(policy) => match policy.saves_candidate(candidate) {
                true => RematerializationVerdict::Save,
                false => RematerializationVerdict::Recompute,
            },
            Self::SaveAndOffloadOnlyTheseNames { saveable, offloadable, destination } => match candidate.key() {
                Some(name) if saveable.iter().any(|n| n == name) => RematerializationVerdict::Save,
                Some(name) if offloadable.iter().any(|n| n == name) => {
                    RematerializationVerdict::Offload { destination: *destination }
                }
                _ => RematerializationVerdict::Recompute,
            },
            Self::OffloadDotsWithNoBatchDims { destination } => {
                let unbatched_dot = candidate.dot_dimensions().is_some_and(|dimensions| {
                    dimensions.lhs_batching_dimensions().is_empty() && dimensions.rhs_batching_dimensions().is_empty()
                });
                match unbatched_dot {
                    true => RematerializationVerdict::Offload { destination: *destination },
                    false => RematerializationVerdict::Recompute,
                }
            }
            Self::SaveFromBothPolicies(first, second) => match first.classify_candidate(candidate) {
                RematerializationVerdict::Recompute => second.classify_candidate(candidate),
                verdict => verdict,
            },
            Self::Custom(policy) => policy(candidate),
        }
    }
}

/// Per-policy residual handling seam consumed by [`Rematerialize::call`].
///
/// [`Rematerialize`] is generic over its policy type, and every residual-placement decision is routed through this
/// trait so that capability bounds travel with the *policy* instead of with the rematerialization machinery: the
/// [`RematerializationPolicy`] impl covers every domain with no bounds beyond what plain rematerialization already
/// needs, while the [`OffloadingRematerializationPolicy`] impl — which stages memory transfers from inside its
/// hooks — is bounded on
/// [`From<TransferToMemoryOperation>`](TransferToMemoryOperation) and
/// therefore only exists in domains whose operation types can represent transfers. Configuring an offloading
/// policy anywhere else is a compile-time error, and
/// [`Rematerialize::call`] itself never sees an offload case.
pub trait ResidualHandling<D: Domain> {
    /// Returns whether the residual described by `candidate` is saved as a forward-program output (saved in place or
    /// offloaded) rather than recomputed by the backward program. This is the pure-provenance classification used to
    /// size the derived programs' residual region before any value is materialized.
    fn saves_residual(&self, candidate: &RematerializationCandidate<'_, <D as Domain>::Type>) -> bool;

    /// Materializes the saved value the forward program should emit for the residual described by `candidate` — the
    /// residual itself, or the residual behind a staged memory transfer for offload verdicts — or [`None`] when the
    /// backward program should recompute it.
    ///
    /// The save/recompute decision agrees with [`saves_residual`](Self::saves_residual); `residual` is the concrete
    /// residual value (replayed into the forward derivation's trace) that an offload verdict moves behind a staged
    /// transfer.
    fn process_residual(
        &self,
        candidate: &RematerializationCandidate<'_, <D as Domain>::Type>,
        residual: &DomainTracer<D>,
    ) -> Result<Option<DomainTracer<D>>, ProgramError>;

    /// Restores one saved residual to the form the recomputation graph expects before it is substituted into the
    /// residual table: the identity for residuals saved in place, and a staged transfer back to the source memory
    /// for offloaded residuals. `original` is the recomputed residual tracer the saved value replaces, which
    /// carries the source type.
    fn restore_saved(
        &self,
        saved: DomainTracer<D>,
        original: &DomainTracer<D>,
    ) -> Result<DomainTracer<D>, ProgramError>;
}

impl<D> ResidualHandling<D> for RematerializationPolicy<<D as Domain>::Type>
where
    D: Domain<Operation: MaybeDot + MaybeTag>,
{
    fn saves_residual(&self, candidate: &RematerializationCandidate<'_, <D as Domain>::Type>) -> bool {
        self.saves_candidate(candidate)
    }

    fn process_residual(
        &self,
        candidate: &RematerializationCandidate<'_, <D as Domain>::Type>,
        residual: &DomainTracer<D>,
    ) -> Result<Option<DomainTracer<D>>, ProgramError> {
        Ok(match self.saves_candidate(candidate) {
            true => Some(residual.clone()),
            false => None,
        })
    }

    fn restore_saved(
        &self,
        saved: DomainTracer<D>,
        _original: &DomainTracer<D>,
    ) -> Result<DomainTracer<D>, ProgramError> {
        Ok(saved)
    }
}

impl<D: Domain<Type = ArrayType, Operation: MaybeDot + MaybeTag + From<TransferToMemoryOperation>>> ResidualHandling<D>
    for OffloadingRematerializationPolicy
{
    fn saves_residual(&self, candidate: &RematerializationCandidate<'_, ArrayType>) -> bool {
        !matches!(self.classify_candidate(candidate), RematerializationVerdict::Recompute)
    }

    fn process_residual(
        &self,
        candidate: &RematerializationCandidate<'_, ArrayType>,
        residual: &DomainTracer<D>,
    ) -> Result<Option<DomainTracer<D>>, ProgramError> {
        Ok(match self.classify_candidate(candidate) {
            RematerializationVerdict::Save => Some(residual.clone()),
            RematerializationVerdict::Offload { destination } => Some(residual.transfer_to_memory(destination)),
            RematerializationVerdict::Recompute => None,
        })
    }

    fn restore_saved(
        &self,
        saved: DomainTracer<D>,
        original: &DomainTracer<D>,
    ) -> Result<DomainTracer<D>, ProgramError> {
        // Whether a saved residual was offloaded is recoverable from the types alone: the saved type carries the
        // offload destination while the recomputed residual carries the source memory, so a mismatch is exactly an
        // offloaded residual that must move back before the recomputation graph consumes it.
        let source = original.r#type().memory();
        Ok(match saved.r#type().memory() == source {
            true => saved,
            false => saved.transfer_to_memory(source),
        })
    }
}

/// Function whose reverse-mode differentiation rematerializes interior values instead of storing them — the
/// ergonomic analogue of JAX's [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html),
/// built by [`rematerialize`].
///
/// The wrapped body is stored as a plain closure over [`DomainTracer`]s and nothing is derived at construction
/// time: each [`call`](Self::call) reads the input types off its tracer arguments, traces the body, derives the
/// forward and backward programs symbolically (specialized to those types and to the configured
/// [`RematerializationPolicy`]), and stages one [`RematerializeOperation`]. The derived forward program returns the
/// body outputs followed by the region inputs and the policy-saved residual values; the derived backward program
/// recomputes the remaining residuals from those saved values and replays the transposed tangent map. Reverse-mode
/// differentiation through the staged call therefore stores exactly the saved values — nothing interior — and both
/// derived programs are pruned of unreachable instructions, so saved residuals are genuinely not recomputed.
///
/// Unlike user-authored custom VJPs, the expansion also carries a derived *tangent program*, so forward-mode
/// differentiation works through rematerialized calls — matching `jax.checkpoint`, which supports `jvp`.
/// Un-differentiated calls replay the lean primal program and pay for neither residual computation nor saving.
///
/// Each [`call`](Self::call) caches its derivation inside the wrapper keyed by the flat input types — the analogue
/// of JAX caching traced rules on `(function, avals)` — so repeated calls with equal input types stage the
/// previously derived operation without re-tracing anything. Nothing invalidates the cache: the wrapper owns both
/// the body closure and the policy, and both are immutable after construction ([`with_policy`](Self::with_policy)
/// and [`with_prevent_cse`](Self::with_prevent_cse) consume `self`).
///
/// The policy is a type parameter (defaulting to the plain [`RematerializationPolicy`]) and every
/// residual-placement decision goes through its [`ResidualHandling`] impl, so capability-requiring policy
/// vocabularies — [`OffloadingRematerializationPolicy`] stages memory transfers — bring their own operation-type
/// bounds without imposing them on plain rematerialization.
pub struct Rematerialize<D: Domain, B, IT, OT, P = RematerializationPolicy<<D as Domain>::Type>>
where
    D::Type: PartialEq,
    OT: Parameterized<DomainTracer<D>>,
{
    /// Closure computing the region output tree from the region input tree.
    body: B,

    /// Policy selecting which linearization residuals are saved (possibly offloaded) instead of recomputed.
    policy: P,

    /// Whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier;
    /// see [`Self::with_prevent_cse`].
    prevent_cse: bool,

    /// Derivations already produced by [`call`](Self::call), keyed by the flat input types they were specialized
    /// to. Entries hold the staged operation together with the body's output-tree structure. The handful of
    /// distinct input signatures a wrapper sees makes a linear scan cheaper and simpler than hashing types.
    cache: RefCell<Vec<CachedDerivation<D, OT>>>,

    /// Phantom marker pinning the [`Domain`] and the input and output tracer-tree types named by the body's
    /// signature. The domain is a pure type witness, so no domain value is stored.
    marker: PhantomData<fn() -> (D, IT, OT)>,
}

/// One [`Rematerialize`] cache entry: the flat input types a derivation was specialized to, the derived
/// rematerialization operation, and the structure of the body's output tree.
type CachedDerivation<D, OT> = (
    Vec<<D as Domain>::Type>,
    RematerializeOperation<<D as Domain>::Constant, <D as Domain>::Operation>,
    <OT as Parameterized<DomainTracer<D>>>::ParameterStructure,
);

/// Creates a [`Rematerialize`] function from a body closure over the [`Domain`] `D`'s tracers, with the default
/// [`RematerializationPolicy::NothingSaveable`] policy. Use [`Rematerialize::with_policy`] to select a different policy.
/// Refer to the documentation of [`Rematerialize`] for the derivation and rematerialization semantics.
pub fn rematerialize<D, B, IT, OT>(body: B) -> Rematerialize<D, B, IT, OT>
where
    D: Domain<Type: PartialEq>,
    B: Fn(IT) -> Result<OT, ProgramError>,
    OT: Parameterized<DomainTracer<D>>,
{
    Rematerialize {
        body,
        policy: RematerializationPolicy::NothingSaveable,
        prevent_cse: true,
        cache: RefCell::new(Vec::new()),
        marker: PhantomData,
    }
}

impl<D, B, IT, OT, P> Rematerialize<D, B, IT, OT, P>
where
    D: Domain<Type: PartialEq>,
    OT: Parameterized<DomainTracer<D>>,
{
    /// Replaces this rematerialization's policy, re-typing the wrapper to the new policy type — pass a plain
    /// [`RematerializationPolicy`] or an [`OffloadingRematerializationPolicy`]. Offloading policies require the
    /// domain's operation type to support memory transfers (see [`ResidualHandling`]), so configuring one in a
    /// domain without that capability fails to compile when the rematerialization is [called](Self::call). The
    /// derivation cache starts empty under the new policy.
    #[inline]
    pub fn with_policy<P2>(self, policy: P2) -> Rematerialize<D, B, IT, OT, P2> {
        Rematerialize {
            body: self.body,
            policy,
            prevent_cse: self.prevent_cse,
            cache: RefCell::new(Vec::new()),
            marker: PhantomData,
        }
    }

    /// Sets whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier
    /// (e.g., StableHLO's `optimization_barrier`), preventing the compiler from common-subexpression-eliminating
    /// the recomputed values against the forward pass — which would silently restore the memory cost the
    /// rematerialization was meant to avoid. Enabled by default, mirroring `jax.checkpoint`'s `prevent_cse=True`;
    /// disable it when the rematerialized region is staged somewhere CSE cannot reach (for example, under
    /// `jax.checkpoint`'s documented `scan` carve-out) and the barrier would inhibit useful optimizations.
    /// Offloaded residuals are unaffected either way: they cross through another memory space, which the compiler
    /// cannot common-subexpression-eliminate against the forward pass.
    #[inline]
    pub fn with_prevent_cse(mut self, prevent_cse: bool) -> Self {
        self.prevent_cse = prevent_cse;
        self
    }
}

impl<D, B, IT, OT, P> Rematerialize<D, B, IT, OT, P>
where
    D: Context<Type: PartialEq>,
    B: Fn(IT) -> Result<OT, ProgramError>,
    P: ResidualHandling<D>,
    IT: Parameterized<
            DomainTracer<D>,
            Family: ParameterizedFamily<D::Type> + ParameterizedFamily<<D as Domain>::Constant>,
        >,
    OT: Parameterized<
            DomainTracer<D>,
            Family: ParameterizedFamily<D::Type> + ParameterizedFamily<<D as Domain>::Constant>,
        >,
    IT::To<D::Type>: Clone
        + Parameterized<
            D::Type,
            Family = IT::Family,
            To<DomainTracer<D>> = IT,
            To<<D as Domain>::Constant> = IT::To<<D as Domain>::Constant>,
        >,
    OT::To<D::Type>: Clone
        + Parameterized<
            D::Type,
            Family = OT::Family,
            To<DomainTracer<D>> = OT,
            To<<D as Domain>::Constant> = OT::To<<D as Domain>::Constant>,
        >,
    <D as Domain>::Operation: Clone
        + MaybeDot
        + MaybeTag
        + MaybeScan<<D as Domain>::Constant, <D as Domain>::Operation>
        + From<RematerializeOperation<<D as Domain>::Constant, <D as Domain>::Operation>>
        + From<ZeroOperation<D::Type>>
        + From<AddOperation>
        + TransposableOperation<<D as Domain>::Constant, <D as Domain>::Operation>
        + DifferentiableOperation<TracingContext<<D as Domain>::Constant, <D as Domain>::Operation>>
        + DifferentiableOperation<
            PartialEvaluationContext<TracingContext<<D as Domain>::Constant, <D as Domain>::Operation>>,
        > + PartiallyEvaluatableOperation<TracingContext<<D as Domain>::Constant, <D as Domain>::Operation>>,
    Vec<D::Type>: Parameterized<
            D::Type,
            Family: ParameterizedFamily<<D as Domain>::Constant> + ParameterizedFamily<DomainTracer<D>>,
            To<DomainTracer<D>> = Vec<DomainTracer<D>>,
            To<<D as Domain>::Constant> = Vec<<D as Domain>::Constant>,
        >,
    Vec<DomainTracer<D>>: Parameterized<
            DomainTracer<D>,
            Family: ParameterizedFamily<D::Type> + ParameterizedFamily<<D as Domain>::Constant>,
            To<D::Type> = Vec<D::Type>,
            To<<D as Domain>::Constant> = Vec<<D as Domain>::Constant>,
            ParameterStructure: Debug + PartialEq,
        >,
    Vec<<D as Domain>::Constant>: Parameterized<
            <D as Domain>::Constant,
            Family: ParameterizedFamily<DomainTracer<D>>,
            To<DomainTracer<D>> = Vec<DomainTracer<D>>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    /// Stages this rematerialized function on the provided tracer inputs and returns its outputs, deriving the
    /// forward/backward programs specialized to the inputs' types. Reverse-mode differentiation of the staged call
    /// stores only the region inputs plus the policy-saved residuals and recomputes everything else.
    pub fn call<V, ICT>(&self, input: ICT) -> Result<<OT::To<D::Type> as Parameterized<D::Type>>::To<V>, ProgramError>
    where
        <D as Domain>::Type: DifferentiableType,
        V: Value<Type = D::Type>,
        V::DispatchDomain:
            Context<Type = D::Type, Constant = <D as Domain>::Constant, Operation = <D as Domain>::Operation>,
        IT::Family: ParameterizedFamily<V>,
        OT::Family: ParameterizedFamily<V>,
        ICT: Parameterized<V, Family = IT::Family, To<D::Type> = IT::To<D::Type>>,
        <OT::To<D::Type> as Parameterized<D::Type>>::To<V>: Parameterized<
                V,
                Family = OT::Family,
                ParameterStructure = <OT::To<D::Type> as Parameterized<D::Type>>::ParameterStructure,
            >,
    {
        let mut input_tracers = Vec::new();
        let structured_input_types = input
            .map_parameters(|tracer| {
                let r#type = tracer.r#type().into_owned();
                input_tracers.push(tracer);
                r#type
            })
            .map_err(ProgramError::from)?;
        let Some(first) = input_tracers.first() else {
            return Err(TypeError { message: "rematerialization requires at least one input".to_string() }.into());
        };
        let input_types = structured_input_types.parameters().cloned().collect::<Vec<_>>();

        // Stage a previously cached derivation when one exists for these input types, without re-tracing anything.
        let cached = self
            .cache
            .borrow()
            .iter()
            .find(|(cached_input_types, ..)| *cached_input_types == input_types)
            .map(|(_, operation, output_structure)| (operation.clone(), output_structure.clone()));
        if let Some((operation, output_structure)) = cached {
            let operation = <D as Domain>::Operation::from(operation);
            let context = first.dispatch_domain();
            let outputs = context.bind(operation, &input_tracers)?;
            return Ok(Parameterized::from_parameters(output_structure, outputs)?);
        }

        let (structured_output_types, primal) = D::trace(|xs| (self.body)(xs), structured_input_types.clone())?;
        let primal = primal.to_flat_program();
        let output_types = structured_output_types.parameters().cloned().collect::<Vec<_>>();
        let output_count = output_types.len();
        let input_count = input_types.len();

        // Build the capture-free linearization of the body once. Its primal sub-program computes the body
        // outputs followed by every linearization residual (its trailing `residual_count` outputs), and its tangent
        // sub-program is the linear tangent map over `[input_tangents..., residuals...]`. The three derived programs
        // below all replay these two sub-programs, so the residual order is fixed once here and shared across them.
        let linearization = primal.linearize()?;
        let residual_count = linearization.residual_count();
        let residual_atoms = linearization.primal().output_ids()[output_count..].to_vec();
        let residual_types = linearization.primal().output_types().split_off(output_count);

        // Classify each residual from the producing-operation provenance recovered from the primal sub-program (the
        // operation that defines the residual atom), recording which residual indices the policy saves. Residuals
        // that are region inputs or constants have no producing instruction, so `from_program_residual` returns
        // `None` and they are never saved — the backward program always receives the region inputs and recomputes
        // everything else, exactly as before.
        let saved_indices = (0..residual_count)
            .filter(|&index| {
                RematerializationCandidate::from_program_residual(
                    linearization.primal(),
                    residual_atoms[index],
                    residual_types[index].clone(),
                )
                .is_some_and(|candidate| self.policy.saves_residual(&candidate))
            })
            .collect::<Vec<_>>();
        let saved_count = saved_indices.len();

        // Pass 1: derive the forward program — the body outputs followed by the region inputs and the policy-saved
        // residual values. Replaying the primal sub-program recomputes the body outputs and every residual; the
        // outputs and region inputs lead the forward outputs and the policy-saved residual subset follows. Offloading
        // policies emit the saved value behind a staged memory transfer, so the forward output types naturally carry
        // the destination space.
        let (forward_output_types, forward) = D::trace(
            |xs: Vec<DomainTracer<D>>| {
                let context = xs.first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?.context();
                let context = context.clone();
                let mut primal_side = linearization.primal().interpret_in_context(&context, xs.clone())?;
                let residuals = primal_side.split_off(output_count);
                let mut outputs = primal_side;
                outputs.extend(xs.iter().cloned());
                for (slot, &index) in saved_indices.iter().enumerate() {
                    let candidate = RematerializationCandidate::from_program_residual(
                        linearization.primal(),
                        residual_atoms[index],
                        residual_types[index].clone(),
                    )
                    .ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "rematerialization saved residual slot {slot} has no producing instruction"
                        ))
                    })?;
                    if let Some(saved) = self.policy.process_residual(&candidate, &residuals[index])? {
                        outputs.push(saved);
                    }
                }
                Ok(outputs)
            },
            input_types.clone(),
        )?;
        let forward = forward.into_simplified()?;

        // The saved residual types are the forward program's trailing outputs, after the body outputs and the region
        // inputs. They reflect any offload transfer the policy staged, so an offloaded residual carries its
        // destination memory here and the backward/tangent program signatures below match the forward outputs.
        let saved_types = forward_output_types[output_count + input_count..].to_vec();

        // Pass 2: derive the backward program over `(inputs..., saved..., cotangents...)`. Replaying the primal
        // sub-program inside this trace recomputes every residual from the inputs (region-input residuals just
        // re-read the inputs, with no storage); substituting the saved residuals by index short-circuits exactly the
        // policy-saved values. The tangent sub-program is then transposed with the residuals marked known, so
        // its pullback consumes `[output_cotangents..., residuals...]` and produces one cotangent per region input.
        let backward_input_types =
            input_types.iter().chain(saved_types.iter()).chain(output_types.iter()).cloned().collect::<Vec<_>>();
        let pullback = linearization.pullback()?;
        let (_, backward) = D::trace(
            |flat: Vec<DomainTracer<D>>| {
                let context = flat.first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?.context();
                let context = context.clone();
                let primal_tracers = flat[..input_count].to_vec();
                let saved_tracers = &flat[input_count..input_count + saved_count];
                let cotangent_tracers = flat[input_count + saved_count..].to_vec();
                let mut primal_side = linearization.primal().interpret_in_context(&context, primal_tracers)?;
                let mut residuals = primal_side.split_off(output_count);
                for (slot, &index) in saved_indices.iter().enumerate() {
                    residuals[index] = self.policy.restore_saved(saved_tracers[slot].clone(), &residuals[index])?;
                }
                let mut pullback_inputs = cotangent_tracers;
                pullback_inputs.extend(residuals);
                pullback.interpret_in_context(&context, pullback_inputs)
            },
            backward_input_types,
        )?;
        let backward = backward.into_simplified()?;

        // Pass 3: derive the tangent program over `(inputs..., saved..., input_tangents...)` so that forward-mode
        // differentiation works through the rematerialized call (JAX's `jax.checkpoint` also supports `jvp`). The
        // derivation mirrors the backward pass without the transposition: recompute the residuals, substitute the
        // saved ones, and replay the tangent sub-program over `[input_tangents..., residuals...]`.
        let tangent_input_types =
            input_types.iter().chain(saved_types.iter()).chain(input_types.iter()).cloned().collect::<Vec<_>>();
        let (_, tangent) = D::trace(
            |flat: Vec<DomainTracer<D>>| {
                let context = flat.first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?.context();
                let context = context.clone();
                let primal_tracers = flat[..input_count].to_vec();
                let saved_tracers = &flat[input_count..input_count + saved_count];
                let tangent_tracers = flat[input_count + saved_count..].to_vec();
                let mut primal_side = linearization.primal().interpret_in_context(&context, primal_tracers)?;
                let mut residuals = primal_side.split_off(output_count);
                for (slot, &index) in saved_indices.iter().enumerate() {
                    residuals[index] = self.policy.restore_saved(saved_tracers[slot].clone(), &residuals[index])?;
                }
                let mut tangent_inputs = tangent_tracers;
                tangent_inputs.extend(residuals);
                linearization.tangent().interpret_in_context(&context, tangent_inputs)
            },
            tangent_input_types,
        )?;
        let tangent = tangent.into_simplified()?;

        let operation =
            RematerializeOperation::new(primal, forward, backward, tangent)?.with_prevent_cse(self.prevent_cse);
        let output_structure = structured_output_types.parameter_structure();
        self.cache.borrow_mut().push((input_types, operation.clone(), output_structure.clone()));
        let context = first.dispatch_domain();
        let outputs = context.bind(<D as Domain>::Operation::from(operation), &input_tracers)?;
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::backends::scalars::Scalar;
    use crate::backends::scalars::ScalarOperation;
    use crate::batching::BatchAxis;
    use crate::contexts::EagerContext;
    use crate::operations::math::{Cos, Sin};
    use crate::operations::tag::Tag;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::tests::TestArray;
    use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate, ReverseModeDifferentiate};
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    /// Computes `f(x) = u * sin(u)` with `u = x · x`, whose linearization residuals span all three policy classes:
    /// `u` is produced by a dot, `sin(u)` by a sine, and the sine rule's `cos(u)` factor by a cosine.
    fn dot_sine<V>(x: V) -> V
    where
        V: Clone + Sin + Dot + std::ops::Mul<Output = V>,
    {
        let u = x.dot(&x, &DotDimensionNumbers::inner_product());
        u.clone() * u.sin().unwrap()
    }

    /// [`dot_sine`] in the closure shape consumed by [`rematerialization`].
    fn dot_sine_body(
        input: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
    ) -> Result<DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, ProgramError> {
        Ok(dot_sine(input))
    }

    /// Reference gradient of [`dot_sine_body`]: `∇f(x) = (sin(u) + u * cos(u)) * 2x` with `u = x · x`.
    fn dot_sine_gradient(x: &[f64]) -> Vec<f64> {
        let u: f64 = x.iter().map(|value| value * value).sum();
        x.iter().map(|value| (u.sin() + u * u.cos()) * 2.0 * value).collect()
    }

    fn vector_type(size: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(size)]))
    }

    #[test]
    fn test_rematerialization_matches_the_unrematerialized_gradient_under_every_policy() {
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        let (direct_value, direct_gradient) = domain.value_and_gradient(|x| dot_sine(x), input.clone()).unwrap();
        for policy in [
            RematerializationPolicy::NothingSaveable,
            RematerializationPolicy::EverythingSaveable,
            RematerializationPolicy::DotsSaveable,
        ] {
            let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(dot_sine_body)
                .with_policy(policy);
            let (value, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input.clone()).unwrap();
            assert_abs_diff_eq!(value.values[0], direct_value.values[0], epsilon = 1e-9);
            for (index, expected) in expected_gradient.iter().enumerate() {
                assert_abs_diff_eq!(gradient.values[index], *expected, epsilon = 1e-9);
                assert_abs_diff_eq!(direct_gradient.values[index], *expected, epsilon = 1e-9);
            }
        }
    }

    /// Candidate classification follows stacked residual edges through `scan` boundaries to the body instructions
    /// that produce them, so structural policies see loop-interior operations: `DotsSaveable` saves exactly the
    /// per-iteration dot stack of a loop body (one more forward output than `NothingSaveable`), and every policy
    /// still produces the reference gradient (unsaved loop residuals recompute through the replayed known scan).
    #[test]
    fn test_rematerialization_policies_classify_residuals_inside_scan_bodies() {
        use crate::operations::control_flow::ScanOperation;
        use crate::tracing_v2::ArrayOperation;
        use crate::types::Shape;

        // Loop body `[c, x] -> [c * (x · x)]` over three two-element rows: `f(c0) = c0 * Π |xᵢ|²`.
        let rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let squared_norms: Vec<f64> = rows.iter().map(|row| row.iter().map(|value| value * value).sum()).collect();
        let expected_gradient: f64 = squared_norms.iter().product();

        let body = {
            use crate::parameters::Placeholder;
            use crate::programs::ProgramBuilder;

            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let carry = builder.add_input(ArrayType::scalar(DataType::F64));
            let row = builder.add_input(vector_type(2));
            let dot = builder
                .add_instruction(
                    crate::tracing_v2::operations::DotOperation::new(DotDimensionNumbers::inner_product()),
                    vec![row, row],
                )
                .unwrap()[0];
            let next = builder.add_instruction(crate::operations::math::MulOperation, vec![carry, dot]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![next], vec![Placeholder; 2], vec![Placeholder; 1])
                .unwrap()
        };
        let stacked = TestArray::new(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(2)])),
            rows.iter().flatten().copied().collect(),
        );
        let scan_body = |carry: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| -> Result<DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, ProgramError> {
            let context = carry.context().clone();
            let xs = StagingContext::constant(&context, stacked.clone());
            let scan = ScanOperation::new(body.clone(), 1, 3)?;
            let outputs = context.stage_operation(ArrayOperation::Scan(Box::new(scan)), &[carry, xs])?;
            Ok(outputs.into_iter().next().unwrap())
        };

        let mut forward_output_counts = Vec::new();
        for policy in [RematerializationPolicy::NothingSaveable, RematerializationPolicy::DotsSaveable] {
            let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(scan_body)
                .with_policy(policy);
            let (_, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
                |carry| function.call(carry),
                ArrayType::scalar(DataType::F64),
            )
            .unwrap();
            let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a rematerialize call");
            };
            forward_output_counts.push(operation.forward().output_types().len());

            let (value, gradient) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
                .value_and_gradient(|carry| function.call(carry).unwrap(), TestArray::scalar(2.0))
                .unwrap();
            assert_abs_diff_eq!(value.values[0], 2.0 * expected_gradient, epsilon = 1e-9);
            assert_abs_diff_eq!(gradient.values[0], expected_gradient, epsilon = 1e-9);
        }
        // `DotsSaveable` saves exactly the stacked per-iteration dot outputs (one `[3]`-shaped residual) beyond the
        // `NothingSaveable` baseline of region output plus region input.
        assert_eq!(forward_output_counts[1], forward_output_counts[0] + 1);
    }

    #[test]
    fn test_rematerialization_policies_control_the_saved_residuals() {
        // `dot_sine_body` has one output and one input, and three instruction-produced residuals: the dot output
        // `u`, the sine output `sin(u)`, and the sine rule's `cos(u)` factor. The forward program therefore outputs
        // 2 values under `NothingSaveable` (output + input), 3 under `DotsSaveable` (+`u`), and 5 under
        // `EverythingSaveable`; and the backward program shrinks as more residuals are saved instead of recomputed.
        let mut forward_output_counts = Vec::new();
        let mut backward_instruction_counts = Vec::new();
        for policy in [
            RematerializationPolicy::NothingSaveable,
            RematerializationPolicy::DotsSaveable,
            RematerializationPolicy::EverythingSaveable,
        ] {
            let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(dot_sine_body)
                .with_policy(policy);
            let (_, program) =
                EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type(2))
                    .unwrap();
            assert_eq!(program.instructions().len(), 1);
            let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a rematerialize call");
            };
            forward_output_counts.push(operation.forward().output_types().len());
            backward_instruction_counts.push(operation.backward().instructions().len());
        }
        assert_eq!(forward_output_counts, vec![2, 3, 5]);
        // Saving everything prunes the whole recomputation graph from the backward program. Saving only the dot
        // output does not shrink it here because the unsaved `sin(u)` and `cos(u)` residuals still recompute from
        // `u`, keeping the dot instruction reachable; the saved value only short-circuits the factor use itself.
        assert!(
            backward_instruction_counts[0] >= backward_instruction_counts[1]
                && backward_instruction_counts[1] > backward_instruction_counts[2],
            "saving more residuals should never grow the backward program and saving everything should shrink it, \
             but instruction counts were {backward_instruction_counts:?}",
        );
    }

    #[test]
    fn test_tag_is_transparent_to_differentiation() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (primal, tangent) =
            domain.jvp(|x| Ok((x.clone() * x).tag("square")), Scalar::from(2.0), Scalar::from(1.0)).unwrap();
        assert_eq!(primal, 4.0);
        assert_eq!(tangent, 4.0);
        let (value, gradient) =
            domain.value_and_gradient(|x| (x.clone() * x).tag("square"), Scalar::from(3.0)).unwrap();
        assert_eq!(value, 9.0);
        assert_eq!(gradient, 6.0);
    }

    #[test]
    fn test_name_based_rematerialization_policies_classify_tagged_residuals() {
        // `f(x) = u * sin(u)` with `u = (x · x).tag("u")`: the tagged dot output is one of the three
        // instruction-produced residuals (`u`, `sin(u)`, and the sine rule's `cos(u)` factor), so name-based
        // policies can select it (or its complement) by tag.
        fn body(
            x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
        ) -> Result<DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, ProgramError> {
            let u = x.dot(&x, &DotDimensionNumbers::inner_product()).tag("u");
            Ok(u.clone() * u.sin()?)
        }
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        // Forward output counts: 2 base outputs (output + input), plus the residuals each policy saves.
        let cases = [
            (RematerializationPolicy::SaveOnlyTheseNames(vec!["u".to_string()]), 3),
            (RematerializationPolicy::SaveOnlyTheseNames(vec!["other".to_string()]), 2),
            (RematerializationPolicy::SaveAnyNamesButThese(vec!["u".to_string()]), 2),
            (RematerializationPolicy::SaveAnyNamesButThese(vec!["other".to_string()]), 3),
            (RematerializationPolicy::SaveAnythingExceptTheseNames(vec!["u".to_string()]), 4),
            (RematerializationPolicy::SaveAnythingExceptTheseNames(vec!["other".to_string()]), 5),
        ];
        for (policy, expected_forward_outputs) in cases {
            let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(body)
                .with_policy(policy.clone());
            let (_, program) =
                EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type(2))
                    .unwrap();
            let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a rematerialize call");
            };
            assert_eq!(
                operation.forward().output_types().len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
            // Every policy preserves the gradient; only the save/recompute split changes.
            let (_, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input.clone()).unwrap();
            for (index, expected) in expected_gradient.iter().enumerate() {
                assert_abs_diff_eq!(gradient.values[index], *expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_scalar_rematerialization_matches_the_unrematerialized_gradient() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        for policy in [RematerializationPolicy::NothingSaveable, RematerializationPolicy::EverythingSaveable] {
            let function = rematerialize::<EagerContext<Scalar, ScalarOperation<Scalar>>, _, _, _>(
                |x: DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok((x.clone() * x).sin()?),
            )
            .with_policy(policy);
            let (value, gradient) =
                domain.value_and_gradient(|x| function.call(x).unwrap(), Scalar::from(2.0)).unwrap();
            assert_abs_diff_eq!(value, 4.0f64.sin(), epsilon = 1e-9);
            assert_abs_diff_eq!(gradient, 4.0f64.cos() * 4.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_jvp_through_rematerialization_uses_the_derived_tangent_program() {
        // Unlike user-authored custom VJPs (which reject forward mode, matching JAX), rematerialized calls carry a
        // derived tangent program, so `jvp` works through them — matching `jax.checkpoint`.
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(dot_sine_body);
        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                |x| function.call(x),
                TestArray::new(vector_type(2), vec![0.5, 1.5]),
                TestArray::new(vector_type(2), vec![1.0, 0.0]),
            )
            .unwrap();
        // f(x) = u * sin(u) with u = x · x; the tangent against seed e_0 is the first gradient component.
        let u: f64 = 0.5 * 0.5 + 1.5 * 1.5;
        assert_abs_diff_eq!(primal.values[0], u * u.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.values[0], dot_sine_gradient(&[0.5, 1.5])[0], epsilon = 1e-9);
    }

    #[test]
    fn test_rematerialization_preserves_custom_vjp_semantics_and_keeps_the_boundary_opaque() {
        use crate::tracing_v2::operations::custom_derivatives::custom_vjp;

        // The custom backward rule triples the true gradient (expressed through addition to avoid constant lifting),
        // so a matching gradient proves the user-authored rule — not the true derivative — governs reverse mode
        // through the rematerialized region.
        let custom = custom_vjp::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok(x.sin()?),
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok((x.sin()?, x.cos()?)),
            |residual, cotangent| {
                let product = residual * cotangent;
                Ok(product.clone() + product.clone() + product)
            },
        );
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            move |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| custom.call(x),
        )
        .with_policy(RematerializationPolicy::EverythingSaveable);
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let (value, gradient) =
            domain.value_and_gradient(|x| function.call(x).unwrap(), TestArray::scalar(2.0)).unwrap();
        assert_abs_diff_eq!(value.values[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        // The custom-VJP boundary stays opaque to the policy: the rematerialized primal program preserves the
        // custom_vjp call intact, and even `EverythingSaveable` saves only the residual the user's forward rule
        // declares (`cos(x)`) — never values from inside the user-owned backward program — so the forward program
        // outputs exactly the body output, the region input, and that one residual.
        let scalar_type = ArrayType::new(DataType::F64, Shape::new(Vec::new()));
        let (_, program) =
            EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), scalar_type).unwrap();
        assert_eq!(program.instructions().len(), 1);
        let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
            panic!("rematerialization should stage a rematerialize call");
        };
        assert!(
            operation
                .primal()
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayOperation::CustomVjp(_))),
            "the rematerialized primal program should preserve the custom_vjp call",
        );
        assert_eq!(operation.forward().output_types().len(), 3);
    }

    #[test]
    fn test_prevent_cse_is_carried_on_the_staged_rematerialize_operation() {
        // `prevent_cse` defaults to enabled (JAX parity) and is carried on the staged operation as a backend
        // lowering hint; user-authored custom VJPs (constructed directly) leave it disabled.
        for (function, expected) in [
            (rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(dot_sine_body), true),
            (
                rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(dot_sine_body)
                    .with_prevent_cse(false),
                false,
            ),
        ] {
            let (_, program) =
                EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type(2))
                    .unwrap();
            let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a rematerialize call");
            };
            assert_eq!(operation.prevent_cse(), expected);
        }
    }

    #[test]
    fn test_rematerialize_remains_opaque_to_partial_evaluation() {
        let input_type = vector_type(2);
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(dot_sine_body);
        let (_, program) =
            EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), input_type.clone())
                .unwrap();
        let program = program.to_flat_program();

        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(input_type)]).unwrap();

        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        let ArrayOperation::Rematerialize(operation) = evaluation.program.instructions()[0].operation() else {
            panic!("partial evaluation should preserve the rematerialize boundary");
        };
        assert!(operation.prevent_cse());
    }

    /// Under a *staging* known-side context, a mixed rematerialized call stays fully opaque: nothing folds across
    /// the boundary (no intermediate crosses it as a residual edge), nothing is staged into the live outer trace,
    /// and the whole call residualizes with the symbolic known input as a residual-input feeder. This is precisely
    /// the memory profile rematerialization asks for — the residual side recomputes from the saved *inputs* instead
    /// of storing intermediates — so the conservative default rule is also the semantically correct online behavior;
    /// finer-grained save-versus-recompute choices belong to the policy-driven structural split (see
    /// `.tasks/plan_partition_policies.md`).
    #[test]
    fn test_rematerialize_remains_opaque_to_partial_evaluation_under_staging() {
        use crate::contexts::StagingContext;
        use crate::partial::PartialEvaluationInput;
        use crate::tracing::TracingContext;

        let input_type = vector_type(2);
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |(a, x): (
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
                DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
            )| { Ok((a * x.clone()).sin()?.dot(&x, &DotDimensionNumbers::inner_product())) },
        );
        let (_, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |inputs| function.call(inputs),
            (input_type.clone(), input_type.clone()),
        )
        .unwrap();
        let program = program.to_flat_program();

        let outer = TracingContext::<TestArray, ArrayOperation<TestArray>>::new();
        let known = outer.input(input_type.clone());
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(input_type)])
            .unwrap();

        assert!(outer.builder().borrow().instructions().is_empty());
        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        let ArrayOperation::Rematerialize(operation) = evaluation.program.instructions()[0].operation() else {
            panic!("staged partial evaluation should preserve the rematerialize boundary");
        };
        assert!(operation.prevent_cse());
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs[1], PartialEvaluationInput::Known(value) if value.atom_id().is_ok()));
    }

    #[test]
    fn test_nested_rematerialization_matches_the_unrematerialized_gradient() {
        // The analogue of JAX's sqrt-schedule idiom: rematerialized regions nest, with each level storing only its
        // own region inputs. Differentiating the outer call replays the inner call's backward program inside the
        // outer backward derivation, which interprets the inner (transposed) rematerialize call over tracers.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let inner = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok((x.clone() * x).sin()?),
        );
        let outer = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                let y = inner.call(x.clone())?;
                Ok(y.dot(&x, &DotDimensionNumbers::inner_product()))
            },
        );
        // f(x) = Σᵢ sin(xᵢ²) xᵢ, so ∂f/∂xⱼ = sin(xⱼ²) + 2 xⱼ² cos(xⱼ²).
        let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
        let expected_value: f64 = [0.5f64, 1.5].iter().map(|x| (x * x).sin() * x).sum();
        let expected_gradient = [0.5f64, 1.5].map(|x| (x * x).sin() + 2.0 * x * x * (x * x).cos());
        let (value, gradient) = domain.value_and_gradient(|x| outer.call(x).unwrap(), input).unwrap();
        assert_abs_diff_eq!(value.values[0], expected_value, epsilon = 1e-9);
        for (index, expected) in expected_gradient.iter().enumerate() {
            assert_abs_diff_eq!(gradient.values[index], *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_nested_rematerialization_preserves_the_nested_call_structure_and_residual_accounting() {
        let inner = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok((x.clone() * x).sin()?),
        );
        let outer = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                let y = inner.call(x.clone())?;
                Ok(y.dot(&x, &DotDimensionNumbers::inner_product()))
            },
        );
        let (_, program) =
            EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| outer.call(x), vector_type(2)).unwrap();
        assert_eq!(program.instructions().len(), 1);
        let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
            panic!("nested rematerialization should stage a rematerialize call");
        };
        // The outer primal program preserves the inner rematerialized call instead of inlining its body.
        assert!(
            operation
                .primal()
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayOperation::Rematerialize(_))),
            "the outer primal program should contain the inner rematerialized call",
        );
        // `NothingSaveable` everywhere: the outer forward program outputs only the body output plus the region
        // input, storing no interior residuals — in particular nothing produced inside the inner region.
        assert_eq!(operation.forward().output_types().len(), 2);
    }

    #[test]
    fn test_jvp_through_nested_rematerialization_uses_the_derived_tangent_programs() {
        // Forward mode through nested rematerialized calls exercises the un-transposed rematerialize call replay over
        // tracers: deriving the outer tangent program interprets the inner call's tangent program inside the outer
        // tangent trace.
        let inner = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok((x.clone() * x).sin()?),
        );
        let outer = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                let y = inner.call(x.clone())?;
                Ok(y.dot(&x, &DotDimensionNumbers::inner_product()))
            },
        );
        let (primal, tangent) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .jvp(
                |x| outer.call(x),
                TestArray::new(vector_type(2), vec![0.5, 1.5]),
                TestArray::new(vector_type(2), vec![1.0, 0.0]),
            )
            .unwrap();
        let expected_value: f64 = [0.5f64, 1.5].iter().map(|x| (x * x).sin() * x).sum();
        let expected_tangent = {
            let x = 0.5f64;
            (x * x).sin() + 2.0 * x * x * (x * x).cos()
        };
        assert_abs_diff_eq!(primal.values[0], expected_value, epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.values[0], expected_tangent, epsilon = 1e-9);
    }

    #[test]
    fn test_nested_scalar_rematerialization_matches_the_unrematerialized_gradient() {
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let inner = rematerialize::<EagerContext<Scalar, ScalarOperation<Scalar>>, _, _, _>(
            |x: DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok((x.clone() * x).sin()?),
        );
        let outer = rematerialize::<EagerContext<Scalar, ScalarOperation<Scalar>>, _, _, _>(
            |x: DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| {
                let y = inner.call(x.clone())?;
                Ok(y * x)
            },
        );
        // f(x) = sin(x²) x, so f'(x) = sin(x²) + 2 x² cos(x²).
        let (value, gradient) = domain.value_and_gradient(|x| outer.call(x).unwrap(), Scalar::from(0.7)).unwrap();
        assert_abs_diff_eq!(value, 0.49f64.sin() * 0.7, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient, 0.49f64.sin() + 2.0 * 0.49 * 0.49f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_rematerialization_survives_batching_with_preserved_residual_structure() {
        use crate::batching::Batch;

        // Batching a rematerialized call re-wraps it around batched programs instead of inlining the primal, so
        // the memory-saving structure survives `vmap`: the staged program holds exactly one rematerialize call whose
        // batched forward program still stores only the body output and the region input plus the policy-saved
        // residuals — each now carrying the batch axis.
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        for (policy, expected_forward_outputs) in [
            (RematerializationPolicy::NothingSaveable, 2),
            (RematerializationPolicy::DotsSaveable, 3),
            (RematerializationPolicy::EverythingSaveable, 5),
        ] {
            let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(dot_sine_body)
                .with_policy(policy.clone());
            let (_, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
                |x| {
                    let context = x.context().clone();
                    Batch::batch(&context, |item| function.call(item), x, BatchAxis::new(0), BatchAxis::new(0), None)
                        .map_err(ProgramError::from)
                },
                matrix_type.clone(),
            )
            .unwrap();
            assert_eq!(program.instructions().len(), 1, "unexpected batched program shape for policy {policy:?}");
            let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
                panic!("batching a rematerialized call should re-wrap the staged rematerialize call");
            };
            let forward_output_types = operation.forward().output_types();
            assert_eq!(
                forward_output_types.len(),
                expected_forward_outputs,
                "unexpected batched forward output count for policy {policy:?}",
            );
            // Every batched forward output carries the batch axis at position 0.
            for output_type in &forward_output_types {
                assert_eq!(
                    output_type.shape().dimensions().first().copied(),
                    Some(Size::Static(2)),
                    "batched forward outputs should carry the batch axis for policy {policy:?}",
                );
            }
        }
    }

    #[test]
    fn test_rematerialized_gradients_are_correct_through_batching() {
        use crate::batching::Batch;
        use crate::differentiation::LinearizationTracer;
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};

        // `grad(vmap(rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(f)))`: the gradient flows through the re-wrapped batched call's derived
        // backward program and matches the analytic per-item gradients.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok((x.clone() * x).sin()?),
        );
        let (value, gradient) = domain
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let mapped: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>> = Batch::batch(
                        &context,
                        |item| function.call(item),
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                TestArray::new(vector_type(2), vec![0.5, 1.0]),
            )
            .unwrap();
        // f(x) = Σᵢ sin(xᵢ²), so ∂f/∂xⱼ = 2 xⱼ cos(xⱼ²).
        assert_abs_diff_eq!(value.values[0], 0.25f64.sin() + 1.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[0], 2.0 * 0.5 * 0.25f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.values[1], 2.0 * 1.0 * 1.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_call_caches_derivations_per_input_types() {
        use std::cell::Cell;

        // The body closure runs only while deriving (the primal trace; the remaining passes replay the traced
        // program), so the closure invocation count equals the number of derivations.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let trace_count = Rc::new(Cell::new(0));
        let counter = trace_count.clone();
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            move |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                counter.set(counter.get() + 1);
                Ok((x.clone() * x).sin()?)
            },
        );

        // Two calls with equal input types derive once, both within one trace and across separate traces.
        let (_, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                let first = function.call(x.clone())?;
                let second = function.call(x)?;
                Ok(first + second)
            },
            vector_type(2),
        )
        .unwrap();
        assert_eq!(trace_count.get(), 1);
        let rematerialize_count = program
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), ArrayOperation::Rematerialize(_)))
            .count();
        assert_eq!(rematerialize_count, 2, "both calls should stage their own rematerialize instruction");
        EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type(2)).unwrap();
        assert_eq!(trace_count.get(), 1);

        // A different input type re-derives.
        EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type(3)).unwrap();
        assert_eq!(trace_count.get(), 2);

        // Cache hits still differentiate correctly: the second gradient call reuses the derivation staged by the
        // first one.
        let (_, first_gradient) =
            domain.value_and_gradient(|x| function.call(x).unwrap(), TestArray::scalar(0.7)).unwrap();
        let derivations_after_first_gradient = trace_count.get();
        let (_, second_gradient) =
            domain.value_and_gradient(|x| function.call(x).unwrap(), TestArray::scalar(0.7)).unwrap();
        assert_eq!(trace_count.get(), derivations_after_first_gradient);
        assert_abs_diff_eq!(first_gradient.values[0], 2.0 * 0.7 * 0.49f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(second_gradient.values[0], first_gradient.values[0], epsilon = 1e-9);
    }

    #[test]
    fn test_dots_with_no_batch_dims_saveable_skips_batched_contractions() {
        // The body stages two dots: a batched per-row inner product `u = dot(x, x; batch=[0])` and an unbatched
        // inner product `v = u · u`. `DotsSaveable` saves both dot residuals while
        // `DotsWithNoBatchDimsSaveable` saves only the unbatched one, so the forward output counts differ by one
        // (2 base outputs = body output + region input).
        fn body(
            x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
        ) -> Result<DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, ProgramError> {
            let batched = DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]);
            let u = x.dot(&x, &batched);
            let v = u.dot(&u, &DotDimensionNumbers::inner_product());
            Ok(v.clone() * v.sin()?)
        }
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let cases =
            [(RematerializationPolicy::DotsSaveable, 4), (RematerializationPolicy::DotsWithNoBatchDimsSaveable, 3)];
        for (policy, expected_forward_outputs) in cases {
            let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(body)
                .with_policy(policy.clone());
            let (_, program) =
                EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), matrix_type.clone())
                    .unwrap();
            let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a rematerialize call");
            };
            assert_eq!(
                operation.forward().output_types().len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
        }
    }

    #[test]
    fn test_save_from_both_policies_saves_the_union_of_both_policies() {
        // The body produces one dot residual `u`, one named residual `s`, and one unnamed `cos` residual; the
        // combinator saves the union of what its two members save.
        fn body(
            x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
        ) -> Result<DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, ProgramError> {
            let u = x.dot(&x, &DotDimensionNumbers::inner_product());
            let s = u.sin()?.tag("s");
            Ok(u * s)
        }
        let cases = [
            (RematerializationPolicy::DotsSaveable, 3),
            (RematerializationPolicy::SaveOnlyTheseNames(vec!["s".to_string()]), 3),
            (
                RematerializationPolicy::SaveFromBothPolicies(
                    Box::new(RematerializationPolicy::DotsSaveable),
                    Box::new(RematerializationPolicy::SaveOnlyTheseNames(vec!["s".to_string()])),
                ),
                4,
            ),
        ];
        for (policy, expected_forward_outputs) in cases {
            let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(body)
                .with_policy(policy.clone());
            let (_, program) =
                EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type(2))
                    .unwrap();
            let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a rematerialize call");
            };
            assert_eq!(
                operation.forward().output_types().len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
        }
    }

    #[test]
    fn test_custom_policies_classify_residuals_through_candidates() {
        // Custom policies see each residual's classification facts. The first policy reproduces the
        // `SaveFromBothPolicies` union from the test above through candidate queries; the second selects by
        // operation name; and both observe the residuals' staged types.
        fn body(
            x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
        ) -> Result<DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, ProgramError> {
            let u = x.dot(&x, &DotDimensionNumbers::inner_product());
            let s = u.sin()?.tag("s");
            Ok(u * s)
        }
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let dot_or_named =
            RematerializationPolicy::Custom(Rc::new(|candidate: &RematerializationCandidate<'_, ArrayType>| {
                assert!(candidate.residual_type().shape().dimensions().is_empty());
                candidate.is_dot() || candidate.key() == Some("s")
            }));
        let cosines_only =
            RematerializationPolicy::Custom(Rc::new(|candidate: &RematerializationCandidate<'_, ArrayType>| {
                candidate.operation_name() == "cos"
            }));
        let expected_gradient = {
            // f(x) = u sin(u) with u = x · x, so ∇f(x) = (sin(u) + u cos(u)) · 2x.
            let u: f64 = 0.5 * 0.5 + 1.5 * 1.5;
            [0.5f64, 1.5].map(|value| (u.sin() + u * u.cos()) * 2.0 * value)
        };
        for (policy, expected_forward_outputs) in [(dot_or_named, 4), (cosines_only, 3)] {
            let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(body)
                .with_policy(policy.clone());
            let (_, program) =
                EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type(2))
                    .unwrap();
            let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a rematerialize call");
            };
            assert_eq!(
                operation.forward().output_types().len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
            // Custom policies only change the save/recompute split, never the gradient.
            let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
            let (_, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input).unwrap();
            for (index, expected) in expected_gradient.iter().enumerate() {
                assert_abs_diff_eq!(gradient.values[index], *expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_second_order_reverse_through_rematerialization_matches_the_analytic_second_derivative() {
        use crate::tracing_v2::DifferentiableDomainExtension;

        // Second-order differentiation through a rematerialized call: the inner reverse pass replays the derived
        // backward program over tracers (inlining it into the gradient program), and the outer pass differentiates
        // the result. f(x) = sin(x²), so f''(x) = 2 cos(x²) - 4x² sin(x²).
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok((x.clone() * x).sin()?),
        );
        let hessian = domain.hessian(|x| function.call(x).unwrap(), TestArray::scalar(0.7)).unwrap();
        let (_, _, block) = hessian.iter_blocks().next().unwrap();
        let x: f64 = 0.7;
        assert_abs_diff_eq!(
            block.value().values()[0],
            2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_scalar_second_order_through_rematerialization_matches_the_analytic_second_derivative() {
        use crate::backends::scalars::ScalarOperation;

        // The scalar counterpart of the test above, composed through nested transforms: the outer reverse pass
        // differentiates a closure that takes the rematerialized gradient on its nested tracing context.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let function = rematerialize::<EagerContext<Scalar, ScalarOperation<Scalar>>, _, _, _>(
            |x: DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>| Ok((x.clone() * x).sin()?),
        );
        let (gradient, second_derivative) = domain
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    context.gradient(|y| function.call(y).unwrap(), x).unwrap()
                },
                Scalar::from(0.7),
            )
            .unwrap();
        let x: f64 = 0.7;
        assert_abs_diff_eq!(gradient, 2.0 * x * (x * x).cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(second_derivative, 2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin(), epsilon = 1e-9);
    }

    #[test]
    fn test_rematerialized_pullback_recovers_the_tangent_map() {
        // The pullback of a rematerialized call carries a derived tangent program. For a scalar function the input
        // cotangent at a unit output cotangent equals the tangent-map coefficient `f'(x)`, so seeding the
        // direct-transpose pullback at `[1.0 ++ residuals]` recovers the tangent map. f(x) = sin(x²), so the recovered
        // value is f'(0.7) = 2·0.7·cos(0.7²).
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok((x.clone() * x).sin()?),
        );
        let (_, pullback) = domain.vjp(|x| function.call(x), TestArray::scalar(0.7)).unwrap();
        let (pullback, residuals) = pullback.into_parts();
        let mut pullback_inputs = vec![TestArray::scalar(1.0)];
        pullback_inputs.extend(residuals);
        let output = pullback.interpret(pullback_inputs).unwrap();
        let x: f64 = 0.7;
        assert_abs_diff_eq!(output[0].values[0], 2.0 * x * (x * x).cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_jacrev_through_rematerialization_uses_the_rematerializing_backward_program() {
        use crate::tracing_v2::jacrev;

        // The Jacobian of elementwise `sin(x * x)` is the diagonal matrix `diag(cos(x²) * 2x)`; `jacrev` exercises
        // the batched replay of the derived backward program.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(
            |x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| Ok((x.clone() * x).sin()?),
        );
        let jacobian = jacrev(&domain, |x| function.call(x), TestArray::new(vector_type(2), vec![0.5, 1.0])).unwrap();
        let (_, _, block) = jacobian.iter_blocks().next().unwrap();
        assert_abs_diff_eq!(block.value().values()[0], 0.25f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[2], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().values()[3], 1.0f64.cos() * 2.0, epsilon = 1e-9);
    }

    /// Canonical offload destination used by the offloading policy tests.
    const PINNED_HOST: Memory = Memory::Host { pinned: true };

    /// [`dot_sine`] with the dot output tagged `"u"`, so name-based offloading policies can select it.
    fn tagged_dot_sine_body(
        x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
    ) -> Result<DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, ProgramError> {
        let u = x.dot(&x, &DotDimensionNumbers::inner_product()).tag("u");
        Ok(u.clone() * u.sin()?)
    }

    /// Returns whether `program` stages any memory transfers.
    fn contains_memory_transfers(
        program: &crate::programs::Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>>,
    ) -> bool {
        program
            .instructions()
            .iter()
            .any(|instruction| instruction.operation().name() == "transfer_to_memory")
    }

    #[test]
    fn test_offloading_policies_park_saved_residuals_in_the_destination_memory() {
        // The tagged body has three instruction-produced residuals (`u`, `sin(u)`, and the sine rule's `cos(u)`
        // factor), so saving or offloading `u` always yields three forward outputs (body output + region input +
        // `u`). Offloaded residuals are emitted behind a staged transfer — the saved forward output carries the
        // destination memory, and the backward and tangent programs transfer it back before consuming it — while
        // residuals saved in place stay in their own memory with no transfers anywhere.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        let offload_u = OffloadingRematerializationPolicy::SaveAndOffloadOnlyTheseNames {
            saveable: Vec::new(),
            offloadable: vec!["u".to_string()],
            destination: PINNED_HOST,
        };
        let save_u = OffloadingRematerializationPolicy::SaveAndOffloadOnlyTheseNames {
            saveable: vec!["u".to_string()],
            offloadable: Vec::new(),
            destination: PINNED_HOST,
        };
        // The first non-`Recompute` verdict wins, so a save-side hit shields `u` from the offload side, while a
        // recompute-only first side defers to the offload side.
        let save_beats_offload = OffloadingRematerializationPolicy::SaveFromBothPolicies(
            Box::new(save_u.clone()),
            Box::new(offload_u.clone()),
        );
        let offload_after_recompute = OffloadingRematerializationPolicy::SaveFromBothPolicies(
            Box::new(OffloadingRematerializationPolicy::Base(RematerializationPolicy::NothingSaveable)),
            Box::new(offload_u.clone()),
        );
        let cases = [
            (offload_u, PINNED_HOST),
            (save_u, Memory::Device),
            (save_beats_offload, Memory::Device),
            (offload_after_recompute, PINNED_HOST),
        ];
        for (policy, expected_memory) in cases {
            let function =
                rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(tagged_dot_sine_body)
                    .with_policy(policy.clone());
            let (_, program) =
                EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type(2))
                    .unwrap();
            let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a rematerialize call");
            };
            let forward_output_types = operation.forward().output_types();
            assert_eq!(forward_output_types.len(), 3, "unexpected forward output count for policy {policy:?}");
            assert_eq!(forward_output_types[0].memory(), Memory::Device);
            assert_eq!(forward_output_types[1].memory(), Memory::Device);
            assert_eq!(
                forward_output_types[2].memory(),
                expected_memory,
                "unexpected saved-residual memory for policy {policy:?}",
            );
            // Transfers appear exactly when the policy offloads: once in the forward program (to the destination)
            // and once in each of the backward and tangent programs (back to the source).
            let expects_transfers = expected_memory != Memory::Device;
            assert_eq!(
                contains_memory_transfers(operation.forward()),
                expects_transfers,
                "unexpected forward transfers for policy {policy:?}",
            );
            assert_eq!(
                contains_memory_transfers(operation.backward()),
                expects_transfers,
                "unexpected backward transfers for policy {policy:?}",
            );
            assert_eq!(
                contains_memory_transfers(operation.tangent()),
                expects_transfers,
                "unexpected tangent transfers for policy {policy:?}",
            );
            // Offloading changes placement, never values: gradients match the direct computation.
            let (_, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input.clone()).unwrap();
            for (index, expected) in expected_gradient.iter().enumerate() {
                assert_abs_diff_eq!(gradient.values[index], *expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_offload_dots_with_no_batch_dims_offloads_unbatched_contractions() {
        // `dot_sine_body`'s only dot residual is the unbatched inner product `u`, so the policy offloads exactly
        // that residual — `DotsSaveable`'s split, with the saved value parked in pinned host memory.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let function = rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(dot_sine_body)
            .with_policy(OffloadingRematerializationPolicy::OffloadDotsWithNoBatchDims { destination: PINNED_HOST });
        let (_, program) =
            EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type(2)).unwrap();
        let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
            panic!("rematerialization should stage a rematerialize call");
        };
        let forward_output_types = operation.forward().output_types();
        assert_eq!(forward_output_types.len(), 3);
        assert_eq!(forward_output_types[2].memory(), PINNED_HOST);

        let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        let (value, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input).unwrap();
        let u: f64 = 0.5 * 0.5 + 1.5 * 1.5;
        assert_abs_diff_eq!(value.values[0], u * u.sin(), epsilon = 1e-9);
        for (index, expected) in expected_gradient.iter().enumerate() {
            assert_abs_diff_eq!(gradient.values[index], *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_custom_offloading_policies_mix_all_three_verdicts() {
        // `u` is saved in place, `v = sin(u)` is offloaded, and the sine rule's `cos(u)` factor is recomputed, so
        // the forward program emits four outputs whose final two are the device-resident `u` and the host-parked
        // `v`.
        fn body(
            x: DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>,
        ) -> Result<DomainTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>, ProgramError> {
            let u = x.dot(&x, &DotDimensionNumbers::inner_product()).tag("u");
            let v = u.sin()?.tag("v");
            Ok(u * v)
        }
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let policy = OffloadingRematerializationPolicy::Custom(Rc::new(
            |candidate: &RematerializationCandidate<'_, ArrayType>| match candidate.key() {
                Some("u") => RematerializationVerdict::Save,
                Some("v") => RematerializationVerdict::Offload { destination: PINNED_HOST },
                _ => RematerializationVerdict::Recompute,
            },
        ));
        let function =
            rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(body).with_policy(policy);
        let (_, program) =
            EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(|x| function.call(x), vector_type(2)).unwrap();
        let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
            panic!("rematerialization should stage a rematerialize call");
        };
        let forward_output_types = operation.forward().output_types();
        assert_eq!(forward_output_types.len(), 4);
        // The two saved residuals are `u` (saved in place, device-resident) and `v` (offloaded to pinned host);
        // their relative order follows the linearization's residual enumeration, which the test does not pin.
        let saved_memories =
            forward_output_types[2..].iter().map(|output_type| output_type.memory()).collect::<Vec<_>>();
        assert_eq!(saved_memories.len(), 2);
        assert!(saved_memories.contains(&Memory::Device), "expected a device-resident saved residual");
        assert!(saved_memories.contains(&PINNED_HOST), "expected a host-parked saved residual");

        // f(x) = u sin(u) with u = x · x, so the gradient matches `dot_sine`'s.
        let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        let (_, gradient) = domain.value_and_gradient(|x| function.call(x).unwrap(), input).unwrap();
        for (index, expected) in expected_gradient.iter().enumerate() {
            assert_abs_diff_eq!(gradient.values[index], *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_offloaded_rematerialization_survives_batching_with_host_parked_saved_types() {
        use crate::batching::Batch;
        use crate::differentiation::LinearizationTracer;
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};

        // `vmap` re-wraps the rematerialized call around batched programs, and the offloaded saved residual keeps
        // its host placement with the batch axis prepended to its shape.
        let domain = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let function =
            rematerialize::<EagerContext<TestArray, ArrayOperation<TestArray>>, _, _, _>(tagged_dot_sine_body)
                .with_policy(OffloadingRematerializationPolicy::SaveAndOffloadOnlyTheseNames {
                    saveable: Vec::new(),
                    offloadable: vec!["u".to_string()],
                    destination: PINNED_HOST,
                });
        let (_, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |x| {
                let context = x.context().clone();
                Batch::batch(&context, |item| function.call(item), x, BatchAxis::new(0), BatchAxis::new(0), None)
                    .map_err(ProgramError::from)
            },
            matrix_type.clone(),
        )
        .unwrap();
        assert_eq!(program.instructions().len(), 1);
        let ArrayOperation::Rematerialize(operation) = program.instructions()[0].operation() else {
            panic!("batching a rematerialized call should re-wrap the staged rematerialize call");
        };
        let forward_output_types = operation.forward().output_types();
        assert_eq!(forward_output_types.len(), 3);
        let saved_type = &forward_output_types[2];
        assert_eq!(saved_type.shape().dimensions().first().copied(), Some(Size::Static(2)));
        assert_eq!(saved_type.memory(), PINNED_HOST);

        // `grad(vmap(...))` through the offloaded call matches the analytic per-item gradients.
        let rows = [[0.5, 1.5, 1.0], [0.25, 0.75, 1.25]];
        let (_, gradient) = domain
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    let mapped: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>> = Batch::batch(
                        &context,
                        |item| function.call(item),
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )
                    .unwrap();
                    mapped.reduce(&[0], ReductionKind::Sum)
                },
                TestArray::new(matrix_type, rows.as_flattened().to_vec()),
            )
            .unwrap();
        for (row, values) in rows.iter().enumerate() {
            let expected_row_gradient = dot_sine_gradient(values);
            for (column, expected) in expected_row_gradient.iter().enumerate() {
                assert_abs_diff_eq!(gradient.values[row * 3 + column], *expected, epsilon = 1e-9);
            }
        }
    }
}
