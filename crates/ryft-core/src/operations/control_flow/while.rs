//! Contains the `while` control-flow operation: [`WhileOperation`], which repeatedly applies a body
//! [`Region`](crate::Region) to a loop-carried state while a condition [`Region`](crate::Region) over that same
//! state produces a true predicate, together with its interpretation, partial-evaluation, batching, forward-mode
//! differentiation, and transposition rules. This is the analogue of
//! [JAX's `lax.while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html) and lowers to
//! [StableHLO's `while`](https://openxla.org/stablehlo/spec#while), extended in two ways JAX does not support:
//! predicates may be *batched* (a prefix-shaped Boolean whose consumers own the per-item masking semantics via
//! [`WhilePredicate`]), and loops constructed with a semantic
//! [`iteration_bound`](WhileOperation::with_iteration_bound) support reverse-mode differentiation through a staged
//! masked tangent scan.

use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::arrays::batching::align_array_batch;
use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, DataType,
    DimensionType, DimensionValue,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    BatchingTracer, ProgramBatchingOutputAxesPolicy,
};
use crate::captures::CaptureReference;
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    DifferentiationTracer, ResidualZeroProvider, TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::one::OneOperation;
use crate::operations::constants::zero::{Zero, ZeroOperation, ZeroOperationProvider};
use crate::operations::control_flow::condition::ConditionOperation;
use crate::operations::control_flow::scan::{ScanOperation, stacked_scan_type};
use crate::operations::control_flow::select::SelectOperation;
use crate::operations::control_flow::{TemporalResidualOperation, TemporalResidualType};
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::logical::and::AndOperation;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation, DynamicBroadcastOperation};
use crate::operations::manipulation::slicing::DynamicUpdateSliceOperation;
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::operations::math::add::AddOperation;
use crate::operations::math::reduce::{ReduceOperation, ReductionKind};
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput,
    PartialEvaluationOutput, PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    AtomId, CalleeRegionDriver, Concretizable, MaybeZero, Operation, OperationFormatter, OperationProjection,
    OutputRegionProvenance, Program, ProgramBuilder, ProgramError, ReferenceAccessMode, RegionInterface, RegionRef,
    RegionSlot, Type, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`WhileOperation`].
pub const WHILE_OPERATION_NAME: &str = "while";

/// [`Operation`] that repeatedly applies a body [`Region`](crate::Region) to a loop-carried state while a condition
/// [`Region`](crate::Region) over that same state produces a true scalar Boolean predicate. The condition and body
/// consume identical state type signatures, the body produces the next state with that same signature, and the
/// operation outputs the final state once the condition produces false.
///
/// The condition and body computations are not part of this payload: they are [`Region`](crate::Region)s attached to
/// the [`Instruction`](crate::Instruction) applying the operation, in the [`region_slots`](Operation::region_slots)
/// order `["condition", "body"]`, and semantic rules reach them through their driver-granted region access. Owned
/// loops supply the two [`Program`]s through the `driver` argument of
/// [`Context::bind`]. [`Operation::infer_output_types`] validates the loop contract over the
/// attached [`RegionInterface`]s: the condition and body share the loop-carried state input signature, the condition
/// returns exactly one Boolean predicate, the body returns the state signature, and a batched (per-item) predicate
/// requires both regions to be pure because observable effects cannot be masked for finished batch items.
///
/// **Iteration bounds are semantic.** A while loop built with [`Self::with_iteration_bound`] runs **at most** `bound`
/// iterations *by definition*: a loop whose condition would keep it running longer is truncated after `bound` body
/// applications. This is visible, defined behavior — not an unchecked promise — and every consumer enforces it:
/// interpretation exits the loop once the bound is reached even while the condition is still true, and the XLA
/// lowering threads an iteration counter through the `stablehlo.while` state and conjoins `counter < bound` into the
/// loop condition.
///
/// Differentiation through `while` follows one of three regimes:
///
///   - **Eager (direct dual execution).** When the differentiation context's primal values are concrete, the JVP rule
///     evaluates each condition on the primal carries and each body once over the current dual carries (respecting any
///     iteration bound). During linearization, partial evaluation records the executed tangent operations as a
///     straight-line — and therefore transposable — pushforward, so eager reverse mode works without replaying primal
///     body effects.
///   - **Bounded staged (stored stacks + masked scan, reverse-capable).** When primal values are tracers and the
///     loop carries an iteration bound `B`, the rule stages an augmented primal while that *stores* every
///     per-iteration pushforward residual into a preallocated `[B, …]` stack (plus a Boolean validity mask), and the
///     tangent side becomes one masked linear [`scan`](super::scan::ScanOperation) of length `B` whose per-iteration
///     `select` passes tangents through unchanged on the iterations beyond the actual trip count. The linear scan
///     transposes totally, so reverse mode composes through staged bounded loops.
///   - **Unbounded staged (recompute loop, forward-only).** Without a bound, no statically shaped residual
///     stack exists, so the rule stages a doubled-state linear loop that recomputes its residuals forward; that loop
///     rejects transposition, exactly like JAX's `while_loop`.
///
/// The `T` parameter fixes the loop's type universe in the payload itself. Consequently, one concrete
/// [`WhileOperation<T>`](WhileOperation) has exactly one [`Operation<Type = T>`](Operation) contract even though the
/// shared implementation supports homogeneous array and composite array IR loops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhileOperation<T: Type> {
    /// Optional semantic iteration bound: when present, the loop runs at most this many iterations by definition,
    /// truncating even while the condition still produces true.
    pub(crate) iteration_bound: Option<usize>,

    /// Type universe of the loop-carried state and attached regions.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> Copy for WhileOperation<T> {}

impl<T: Type> WhileOperation<T> {
    /// Creates a new [`WhileOperation`] with no semantic iteration bound. The condition and body [`Program`]s are
    /// supplied separately as the operation's attached regions (via the region driver passed to
    /// [`Context::bind`]); [`Operation::infer_output_types`] validates the loop contract over
    /// their interfaces.
    #[inline]
    pub fn new() -> Self {
        Self { iteration_bound: None, marker: PhantomData }
    }

    /// Returns the semantic iteration bound of this [`WhileOperation`], if any. Refer to the documentation of
    /// [`Self::with_iteration_bound`] for the truncation semantics a bound carries.
    #[inline]
    pub fn iteration_bound(&self) -> Option<usize> {
        self.iteration_bound
    }

    /// Returns this [`WhileOperation`] with its semantic iteration bound set to `bound` (or cleared when `bound` is
    /// `None`). The bound must be at least `1` when present.
    ///
    /// **A bounded while runs at most `bound` iterations by definition.** The bound is not a hint and not an
    /// unchecked promise: a loop whose condition would keep it running longer is *truncated* after `bound` body
    /// applications, which is visible, defined behavior. Interpretation exits the loop once the bound is reached
    /// even while the condition still produces true, and the XLA lowering threads an iteration counter through the
    /// `stablehlo.while` state and conjoins `counter < bound` into the loop condition. The bound is also what makes
    /// staged reverse-mode differentiation possible: it gives the loop's linearization a static residual-stack
    /// length (see the bounded staged regime described on [`WhileOperation`]).
    pub fn with_iteration_bound(mut self, bound: impl Into<Option<usize>>) -> Result<Self, ProgramError> {
        let bound = bound.into();
        if bound == Some(0) {
            return Err(TypeError::invalid(format!("{WHILE_OPERATION_NAME} iteration bound must be at least 1")).into());
        }
        self.iteration_bound = bound;
        Ok(self)
    }
}

impl<T: Type> Default for WhileOperation<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: WhileTypeSemantics> Display for WhileOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

/// Type-family semantics for [`WhileOperation`].
///
/// [`ArrayType`] conditions must produce a Boolean array whose shape is a *prefix* of every loop-carried state shape.
/// A rank-0 predicate is the ordinary whole-loop condition, while a batched (non-scalar) predicate carries one
/// termination decision per leading-axes item: the loop continues while *any* item's predicate is true, and items
/// whose predicate is false keep their carried state (refer to the documentation of [`WhilePredicate`]). The prefix
/// requirement is what makes the per-item masking well-defined — the predicate broadcasts against every state element
/// along its leading axes. This mirrors JAX's batched `while_p` contract, where the batching transform emits a loop
/// whose condition returns a batched predicate and the loop's consumers implement the masked semantics.
///
/// [`ArrayIrType`] conditions must produce a Boolean array member and may carry mixed array/dimension state. Reference
/// state is structurally representable in the composite signature but remains unsupported until validation and
/// discharge make it explicit as arrays. Under a batched predicate the prefix requirement applies to the array members
/// only, since a first-class dimension carries no shape, and such a dimension carry must additionally be
/// *loop-invariant*: one dimension value cannot represent independently masked per-item extents, but masking a carry
/// that the body forwards unchanged is the identity. Eager
/// interpretation enforces that invariance dynamically through
/// [`ArrayIrValue`](crate::arrays::ArrayIrValue)'s
/// [`mask_select`](WhilePredicate::mask_select), which returns equal dimension carries unchanged and falls back to
/// scalar-predicate concretization — an error under a batched predicate — for distinct ones. Structural composite
/// batching relies on this relaxation to thread its loop-invariant mapped extent through batch-varying loops.
///
/// The loop-carried state rule is otherwise identical for every type family: the condition and body consume the same
/// state signature, and the body returns the next state with that same signature.
pub trait WhileTypeSemantics: Type {
    /// Validates the condition output type of a while loop against the loop-carried state types.
    ///
    /// # Parameters
    ///
    ///   - `condition_output`: The single output type produced by the condition program.
    ///   - `state_types`: The loop-carried state types the condition and body consume.
    fn validate_while_condition_output(condition_output: &Self, state_types: &[Self]) -> Result<(), TypeError>;

    /// Returns whether `condition_output` is a *batched* (per-item) predicate carrying one termination decision per
    /// leading-axes item, rather than a whole-loop scalar predicate. This is `true` for a non-scalar Boolean
    /// [`ArrayType`] predicate. It gates the purity requirement on batched-predicate loops:
    /// a batched-predicate loop keeps running for still-active items after others have finished, so it re-evaluates
    /// the condition and body over *every* item each iteration, and observable effects cannot be masked back out for
    /// the finished items the way values can (see [`WhilePredicate`]).
    fn is_batched_predicate(condition_output: &Self) -> bool;
}

impl WhileTypeSemantics for ArrayType {
    fn validate_while_condition_output(condition_output: &Self, state_types: &[Self]) -> Result<(), TypeError> {
        if !condition_output.data_type().is_boolean() {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got {condition_output}"
            )));
        }
        let predicate_shape = condition_output.shape();
        for state_type in state_types {
            let state_shape = state_type.shape();
            let is_prefix = predicate_shape.rank() <= state_shape.rank()
                && predicate_shape.dimensions().iter().zip(state_shape.dimensions()).all(|(p, s)| p == s);
            if !is_prefix {
                return Err(TypeError::invalid(format!(
                    "`{WHILE_OPERATION_NAME}` condition predicate shape must be a prefix of every state shape, but predicate \
                         {condition_output} is not a prefix of state {state_type}",
                )));
            }
        }
        Ok(())
    }

    fn is_batched_predicate(condition_output: &Self) -> bool {
        condition_output.rank() > 0
    }
}

impl WhileTypeSemantics for ArrayIrType {
    fn validate_while_condition_output(condition_output: &Self, state_types: &[Self]) -> Result<(), TypeError> {
        let Self::Array(condition_output) = condition_output else {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got {condition_output}"
            )));
        };
        if !condition_output.data_type().is_boolean() {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got {condition_output}"
            )));
        }
        let predicate_shape = condition_output.shape();
        if predicate_shape.rank() == 0 {
            return Ok(());
        }
        for state_type in state_types {
            let state_type = match state_type {
                Self::Array(state_type) => state_type,
                // A first-class dimension carries no shape, so the predicate-prefix requirement does not apply to it.
                // Such a carry must instead be loop-invariant, which `WhilePredicate::mask_select` enforces dynamically
                // (refer to the documentation of `WhileTypeSemantics`).
                Self::Dimension(_) => continue,
                Self::Reference(state_type) => {
                    return Err(TypeError::invalid(format!(
                        "`{WHILE_OPERATION_NAME}` condition with a batched predicate requires references to be \
                         discharged, but state type is `{state_type}`",
                    )));
                }
            };
            let state_shape = state_type.shape();
            let is_prefix = predicate_shape.rank() <= state_shape.rank()
                && predicate_shape.dimensions().iter().zip(state_shape.dimensions()).all(|(p, s)| p == s);
            if !is_prefix {
                return Err(TypeError::invalid(format!(
                    "`{WHILE_OPERATION_NAME}` condition predicate shape must be a prefix of every array state shape, but predicate \
                     {condition_output} is not a prefix of state {state_type}",
                )));
            }
        }
        Ok(())
    }

    fn is_batched_predicate(condition_output: &Self) -> bool {
        matches!(condition_output, Self::Array(r#type) if r#type.rank() > 0)
    }
}

/// Validates the loop contract over the two attached region interfaces (`["condition", "body"]` region order) and
/// returns them. The loop-carried state signature is the body's input signature: the condition must consume the same
/// state and return exactly one Boolean predicate valid for that state under [`WhileTypeSemantics`], the body must
/// return the state signature, and a batched (per-item) predicate requires both regions to be pure — the loop keeps
/// running for still-active items after others finish, so the condition and body re-execute over every item each
/// iteration and observable effects cannot be masked back out for the finished items the way values can. This
/// mirrors JAX's `_while_loop_batching_rule`, which rejects IO effects once the predicate is batched.
fn validated_while_interfaces<'i, T: WhileTypeSemantics>(
    region_interfaces: &'i [RegionInterface<T>],
) -> Result<(&'i RegionInterface<T>, &'i RegionInterface<T>), TypeError> {
    check_count!("region", region_interfaces, 2, TypeError);
    let condition_interface = &region_interfaces[0];
    let body_interface = &region_interfaces[1];
    let state_types = body_interface.input_types();
    check_types!(@same, format!("{WHILE_OPERATION_NAME} condition/body input"), [state_types, condition_interface.input_types()]);
    let condition_output_types = condition_interface.output_types();
    if condition_output_types.len() != 1 {
        return Err(TypeError::invalid(format!(
            "{} condition must return exactly one predicate leaf but returned {}",
            WHILE_OPERATION_NAME,
            condition_output_types.len(),
        )));
    }
    T::validate_while_condition_output(&condition_output_types[0], state_types)?;
    check_types!(@same, format!("{WHILE_OPERATION_NAME} body output"), [state_types, body_interface.output_types()]);
    if T::is_batched_predicate(&condition_output_types[0])
        && (!condition_interface.effects().is_pure() || !body_interface.effects().is_pure())
    {
        return Err(TypeError::invalid(format!(
            "`{WHILE_OPERATION_NAME}` loop with a batched predicate must be pure because observable effects cannot be \
                      masked for finished batch items"
        )));
    }
    Ok((condition_interface, body_interface))
}

impl<T: WhileTypeSemantics> Operation for WhileOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        WHILE_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("condition"), RegionSlot::computation("body")] }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        check_count!("region", region_interfaces, 2, TypeError);
        if region_interfaces.iter().all(|interface| interface.input_types() == input_types) {
            return Ok(vec![None, None]);
        }
        Ok(vec![Some(input_types.to_vec()), Some(input_types.to_vec())])
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        let (_, body_interface) = validated_while_interfaces(region_interfaces)?;
        let state_types = body_interface.input_types();
        check_count!("input", input_types, state_types.len(), TypeError);
        check_types!(@same, format!("{WHILE_OPERATION_NAME} input"), [state_types, input_types]);
        Ok(state_types.to_vec())
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<usize> {
        (region_index < 2).then_some(input_index)
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 1, output_index }]
    }

    #[inline]
    fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
        // Every `while` output is a loop carry aligned with the same-position input, so the root constraint maps
        // positionally for any output the operation actually declares.
        Some(output_index)
    }

    #[inline]
    fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
        region_index != 0 || mode == ReferenceAccessMode::Read
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let operation = OperationFormatter::new(formatter, indentation, WHILE_OPERATION_NAME)?;
        match self.iteration_bound {
            Some(iteration_bound) => {
                operation.bracketed(|operation| operation.field("iteration_bound", iteration_bound))
            }
            None => Ok(()),
        }
    }
}

impl<C: Domain> InterpretableOperation<C> for WhileOperation<C::Type>
where
    C::Value: WhilePredicate,
    C::Type: WhileTypeSemantics,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let state_count = inputs.len();
        let mut state = inputs.to_vec();
        let mut completed_iterations = 0;
        loop {
            // The iteration bound is semantic: a bounded loop runs at most `bound` iterations by definition, so the
            // loop exits here even while the condition still produces true.
            if self.iteration_bound.is_some_and(|bound| completed_iterations >= bound) {
                return Ok(state);
            }
            let condition_outputs = driver.interpret_region(context, 0, state.clone())?;
            check_count!("output", condition_outputs, 1, ProgramError);
            let predicate = &condition_outputs[0];
            if !predicate.any_true()? {
                return Ok(state);
            }
            // Masked state update: items whose predicate is true take the body's candidate update, the rest keep
            // their carried state. For a scalar predicate this reduces to taking the candidates wholesale, since a
            // false scalar predicate exits above.
            let candidates = driver.interpret_region(context, 1, state.clone())?;
            check_count!("output", candidates, state_count, ProgramError);
            state = candidates
                .iter()
                .zip(state.iter())
                .map(|(candidate, carried)| predicate.mask_select(candidate, carried))
                .collect::<Result<Vec<_>, _>>()?;
            completed_iterations += 1;
        }
    }
}

/// Partial-evaluation override for [`WhileOperation`], dispatching to the loop's type family through
/// [`WhilePartialEvaluation`]. Homogeneous array loops fold loop-invariant-known state, while composite array IR loops
/// use the closed-knownness split when applicable.
impl<C: Context> PartiallyEvaluatableOperation<C> for WhileOperation<C::Type>
where
    C::Type: WhilePartialEvaluation<C>,
    C::Operation: From<WhileOperation<C::Type>>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        <C::Type>::partially_evaluate_while(self, context, driver, inputs)
    }
}

/// Type-family partial-evaluation semantics for [`WhileOperation`]s. The known-side context parameter rides as a
/// trait input (with the type family as the implementing type, mirroring [`ScanPayload`](super::scan::ScanPayload)) so
/// that each family implementation can carry exactly the capability bounds its rule needs.
pub(crate) trait WhilePartialEvaluation<C: Context>: WhileTypeSemantics {
    /// Partially evaluates the provided [`WhileOperation`]; refer to the documentation of
    /// [`PartiallyEvaluatableOperation::partially_evaluate`] for the contract.
    fn partially_evaluate_while<D: PartialEvaluationDriver<C>>(
        operation: &WhileOperation<Self>,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>;
}

/// Partial-evaluation rule for a [`WhileOperation`] over [`ArrayType`].
///
/// A while's inputs are the initial loop state and its outputs are the final loop state (the same arity). Partial
/// evaluation folds the known value of every *loop-invariant-known* state element into both nested programs: a state
/// element is loop-invariant-known iff its init input is [`Known`](PartialValue::Known) and, with the
/// loop-invariant-known state bound to its init values and everything else [`Unknown`](PartialValue::Unknown), its body
/// next-state output is itself a known value equal to that init. Such an element holds its init value on every
/// iteration, so binding it to that constant inside the condition and body is sound and collapses every subcomputation
/// that depended only on it.
///
/// The invariant set is found by the same monotonic fixed point as the [`scan`](super::scan::ScanOperation) rule (a
/// state element can only be demoted from invariant to non-invariant as more are admitted, so it converges), recursing
/// through the partial-evaluation driver's split requests on the *body* (the condition produces no state and so
/// cannot affect whether a state element reproduces its init). After the fixed point, both the body and the condition
/// are partially evaluated with the invariant-known state knowledge — the condition reads the state too, so folding
/// an invariant element can shrink it as well.
///
/// The residual while keeps the *same* state set and therefore the same output arity as the original operation. A
/// loop-invariant-known element is not dropped; instead its body next-state output is rebuilt as the constant init
/// value and its uses fold away inside both programs. Because the condition and body run on the same loop-carried
/// state each iteration, both residual programs are rebuilt over the loop's full state signature (in state order):
/// each surviving unknown state element feeds the matching state input, and every known residual a program closed over
/// is rebuilt as an inline residual-program constant (so the residual while needs no captures). The
/// [`iteration_bound`](WhileOperation::iteration_bound) is preserved. The rewrite is emitted over the original while
/// inputs unchanged.
///
/// If no state element is loop-invariant-known and neither nested program shrank, the rule attempts the
/// *closed-knownness split* before residualizing unchanged: when a known state subset's next values and the trip
/// predicate fold from known state alone, the loop separates into a known loop bound on the known side and the
/// residual loop kept whole (see `split_while_by_closed_knownness`). This is the split that makes
/// [`Program::linearize`] total over the fused doubled-state loops staged by the unbounded `while` forward-mode
/// rule.
impl<V, O, C> WhilePartialEvaluation<C> for ArrayType
where
    V: Value<Type = ArrayType>,
    C: Context<Type = ArrayType, Constant = V, Operation = O>,
    C::Value: PartialEq,
    O: Operation<Type = ArrayType> + From<WhileOperation<ArrayType>>,
{
    fn partially_evaluate_while<D: PartialEvaluationDriver<C>>(
        operation: &WhileOperation<ArrayType>,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // The rule requests all nested-computation work through its region access (region 0 is the condition and
        // region 1 the body), which keeps its bounds free of the operation family's own semantic traits.
        //
        // When every input is known the whole loop folds by binding it in the known-side context; defer to that
        // default behavior.
        if inputs.iter().all(PartialEvaluationValue::is_known) {
            return context.fold_or_residualize(
                O::from(*operation),
                driver.regions().map(|region| region.to_program()).collect(),
                inputs,
            );
        }

        let condition = driver.region(0)?;
        let body = driver.region(1)?;
        let state_types = body.input_types();
        let state_count = state_types.len();

        // The invariance fixed point below probes by folding the condition and body through the *live* known-side
        // context, and the closed-knownness split's known loop re-runs the known part of every iteration. For an
        // effectful loop the probes would execute (eager) or stage (staging) the loop's effects once more and the
        // split would run them twice, so effectful loops skip both and residualize unchanged (see the effect
        // placement contract on `PartialEvaluationContext::fold_or_residualize`).
        if !condition.effects().is_pure() || !body.effects().is_pure() {
            return context.fold_or_residualize(
                O::from(*operation),
                vec![condition.to_program(), body.to_program()],
                inputs,
            );
        }

        // Every pure fallback below first attempts the closed-knownness split — a known state subset whose next
        // values and trip predicate fold from known state alone separates into a known loop bound on the known side
        // and the residual loop kept whole (see `split_while_by_closed_knownness`) — and only residualizes the loop
        // unchanged when the split does not apply.
        let split_or_residualize = |context: &PartialEvaluationContext<C>| match split_while_by_closed_knownness(
            context, operation, condition, body, inputs, driver,
        )? {
            Some(outputs) => Ok(outputs),
            None => context.fold_or_residualize(
                O::from(*operation),
                vec![condition.to_program(), body.to_program()],
                inputs,
            ),
        };

        // A state element can only fold if its init input is known *and* resolves to a constant in the known-side
        // context: the folded value must be embeddable as a rebuilt-program constant, and skipping symbolic knowns
        // also keeps the fixed point's probe rounds from folding symbolic known work into a live staging context.
        let state_inits = (0..state_count)
            .map(|index| {
                inputs[index].as_known().filter(|value| context.parent().resolve(value).is_constant()).cloned()
            })
            .collect::<Vec<Option<C::Value>>>();

        // Monotonically narrow the set of loop-invariant-known state elements to a fixed point. A round binds each
        // invariant element to its init, leaves everything else unknown, and keeps an element only if the body
        // reproduces its init as the next-state value. With no invariance candidates at all there is nothing the
        // rebuild below could embed, so skip the live-context probe entirely — in particular, under a staging
        // known-side context every symbolic known init lands here, which is where the closed-knownness split serves
        // `Program::linearize`.
        let mut invariant = state_inits.iter().map(Option::is_some).collect::<Vec<bool>>();
        if invariant.iter().all(|candidate| !candidate) {
            return split_or_residualize(context);
        }
        let state_knowledge = |invariant: &[bool]| -> Vec<PartialValue<C::Value>> {
            (0..state_count)
                .map(|index| match (invariant[index], &state_inits[index]) {
                    (true, Some(value)) => PartialValue::Known(value.clone()),
                    _ => PartialValue::Unknown(state_types[index].clone()),
                })
                .collect()
        };

        // A probe failure falls back through `split_or_residualize`: the body may never run at runtime (the
        // condition can be false on entry), so an erroring known-side fold (e.g., an integer division by a known
        // zero) must not fail partial evaluation — the branch's work, and its error if the loop is ever entered,
        // stays behind the condition. Both programs are pure here (the effects gate above), so partially completed
        // probe folds are safe to discard, and the split stays error-consistent: its partitions stage through fresh
        // contexts without executing known work, and its known loop replays the loop's exact runtime semantics
        // (running nothing when the condition is false on entry).
        let Ok(mut body_evaluation) = driver.partially_evaluate_program(context, body, &state_knowledge(&invariant))
        else {
            return split_or_residualize(context);
        };
        loop {
            let refined = (0..state_count)
                .map(|index| {
                    invariant[index]
                        && matches!(
                            &body_evaluation.outputs[index],
                            PartialEvaluationOutput::Known(value) if Some(value) == state_inits[index].as_ref()
                        )
                })
                .collect::<Vec<bool>>();
            if refined == invariant {
                break;
            }
            invariant = refined;
            body_evaluation = match driver.partially_evaluate_program(context, body, &state_knowledge(&invariant)) {
                Ok(evaluation) => evaluation,
                Err(_) => return split_or_residualize(context),
            };
        }

        // The condition reads the loop state too, so folding the invariant-known state can shrink it as well.
        let condition_evaluation =
            match driver.partially_evaluate_program(context, condition, &state_knowledge(&invariant)) {
                Ok(evaluation) => evaluation,
                Err(_) => return split_or_residualize(context),
            };

        // Nothing folded: defer to the split-or-residualize fallback. A loop-invariant-known element always shrinks
        // the body (its uses fold to constants), so the only way nothing folds is an empty invariant set whose
        // residual condition and body did not shrink either — a time-varying known chain lands here and is what the
        // closed-knownness split recovers. The rebuild below embeds the probes' known values as inline program
        // constants, which is only possible when they all resolve to constants — under a staging known-side context a
        // probe can fold a constant-only chain into a live-trace tracer — so a non-constant probe takes the same
        // fallback.
        if (invariant.iter().all(|folded| !folded)
            && body_evaluation.program.instructions().len() >= body.instructions().len()
            && condition_evaluation.program.instructions().len() >= condition.instructions().len())
            || !context.all_knowns_are_constants(&body_evaluation)
            || !context.all_knowns_are_constants(&condition_evaluation)
        {
            return split_or_residualize(context);
        }

        // The residual while keeps the same state set, so its output arity matches the original while. The condition
        // and body run on the same loop-carried state each iteration, so both are rebuilt over the loop's full state
        // signature.
        let mut condition_builder = ProgramBuilder::<V, O>::new();
        let condition_state_atoms = state_types
            .iter()
            .map(|state_type| condition_builder.add_input(state_type.clone()))
            .collect::<Vec<_>>();
        let condition_outputs =
            rebuild_while_program(context, &mut condition_builder, &condition_state_atoms, &condition_evaluation)?;
        let residual_condition = condition_builder.build::<Vec<V>, Vec<V>>(
            condition_outputs,
            vec![Placeholder; state_count],
            vec![Placeholder; 1],
        )?;

        let mut body_builder = ProgramBuilder::<V, O>::new();
        let body_state_atoms =
            state_types.iter().map(|state_type| body_builder.add_input(state_type.clone())).collect::<Vec<_>>();
        let body_outputs = rebuild_while_program(context, &mut body_builder, &body_state_atoms, &body_evaluation)?;
        let residual_body = body_builder.build::<Vec<V>, Vec<V>>(
            body_outputs,
            vec![Placeholder; state_count],
            vec![Placeholder; state_count],
        )?;

        let while_operation = WhileOperation::new().with_iteration_bound(operation.iteration_bound)?;

        // The residual while's inputs are exactly the original while's inputs: each state element's init value (now a
        // known residual for the folded elements) in state order.
        context.fold_or_residualize(O::from(while_operation), vec![residual_condition, residual_body], inputs)
    }
}

impl<C: Context<Type = ArrayIrType>> WhilePartialEvaluation<C> for ArrayIrType
where
    C::Operation: From<WhileOperation<ArrayIrType>>,
{
    fn partially_evaluate_while<D: PartialEvaluationDriver<C>>(
        operation: &WhileOperation<ArrayIrType>,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        partially_evaluate_while_by_closed_knownness(operation, context, driver, inputs)
    }
}

/// Partially evaluates a loop type family that supports the closed-knownness split but no invariant-state rewrite.
fn partially_evaluate_while_by_closed_knownness<C: Context, D: PartialEvaluationDriver<C>>(
    operation: &WhileOperation<C::Type>,
    context: &PartialEvaluationContext<C>,
    driver: &D,
    inputs: &[PartialEvaluationValue<C::Value>],
) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
where
    C::Type: WhileTypeSemantics,
    C::Operation: From<WhileOperation<C::Type>>,
{
    let condition = driver.region(0)?;
    let body = driver.region(1)?;
    // The split's known loop re-runs the known part of every iteration, so it is only sound for pure loops.
    if inputs.iter().any(PartialEvaluationValue::is_known)
        && !inputs.iter().all(PartialEvaluationValue::is_known)
        && condition.effects().is_pure()
        && body.effects().is_pure()
        && let Some(outputs) = split_while_by_closed_knownness(context, operation, condition, body, inputs, driver)?
    {
        return Ok(outputs);
    }
    context.fold_or_residualize(C::Operation::from(*operation), vec![condition.to_program(), body.to_program()], inputs)
}

/// Rebuilds one partially-evaluated nested `while` program (the condition or the body) over the loop's full state
/// signature; see the loop-invariant-known [`PartiallyEvaluatableOperation`] implementation for [`WhileOperation`].
///
/// The condition and the body both run on the *same* loop-carried state every iteration, so each is rebuilt over the
/// identical `state_atoms` input signature rather than over its own residual-input signature. Each residual input of
/// `evaluation` is fed either by the matching state atom (for an unknown state element the sub-program still reads) or
/// by an inline constant (for a known residual the sub-program closed over, such as a folded loop-invariant state
/// value). The sub-program's residual program is then spliced over those inputs and its outputs are reassembled: a
/// folded [`Known`](PartialEvaluationOutput::Known) output becomes an inline constant and an
/// [`Unknown`](PartialEvaluationOutput::Unknown) output reads the spliced residual program's corresponding output.
///
/// # Parameters
///
///   - `builder`: Builder accumulating the rebuilt program.
///   - `state_atoms`: Input atoms holding the loop state, in state order, that this sub-program's residual inputs map
///     back to.
///   - `evaluation`: Partial evaluation of this sub-program against the loop-invariant-known state knowledge.
fn rebuild_while_program<C: Context<Type = ArrayType>>(
    context: &PartialEvaluationContext<C>,
    builder: &mut ProgramBuilder<C::Constant, C::Operation>,
    state_atoms: &[AtomId],
    evaluation: &PartialEvaluation<C>,
) -> Result<Vec<AtomId>, ProgramError> {
    let mut residual_inputs = Vec::with_capacity(evaluation.inputs.len());
    for residual_input in evaluation.inputs.iter() {
        match residual_input {
            PartialEvaluationInput::Unknown(state_index) => residual_inputs.push(state_atoms[*state_index]),
            PartialEvaluationInput::Known(value) => {
                residual_inputs.push(builder.add_constant(context.known_constant(value)?))
            }
        }
    }
    let spliced_outputs = builder.splice_program(&evaluation.program, &residual_inputs)?;
    evaluation
        .outputs
        .iter()
        .map(|output| match output {
            PartialEvaluationOutput::Known(value) => Ok(builder.add_constant(context.known_constant(value)?)),
            PartialEvaluationOutput::Unknown(index) => Ok(spliced_outputs[*index]),
        })
        .collect()
}

/// Splits a pure `while` loop whose known state subset is *closed* — every known state element's next-state value
/// and the trip predicate fold from the known state alone — into a *known* loop bound in the enclosing known-side
/// context and the *residual* loop kept whole, returning `None` when the split does not apply.
///
/// The known state subset is found by a monotonic fixed point mirroring the
/// [scan known-ness split](super::scan)'s: a state element stays known iff its init is known and the body computes
/// its next value from known state alone, with each round partitioning the body through a **fresh** staging context
/// (via [`PartialEvaluationDriver::partition_program`]) so no probe work leaks into the caller's context. Unlike the
/// loop-*invariance* rewrite, known-ness needs neither constant resolution nor value equality, so symbolic known
/// inits (tracers into a live outer trace) participate fully — this is what makes the split fire under
/// [`Program::linearize`]. After convergence the split additionally requires the *predicate* to fold from the known
/// state alone; only then does the known loop run the original trip count, since its trip decision is byte-for-byte
/// the original one over state it computes itself.
///
/// The known loop is the projection of the original loop onto the known subset: its body maps the known state
/// elements to their known next values and its condition is the known projection of the original condition. It is
/// bound whole into the enclosing known-side context over the original known inits (executing under an eager
/// context and staging into the outer program under a staging one), and its outputs are the known state elements'
/// final values. The residual loop is the **original loop unchanged**: its unknown outputs are the unknown state
/// elements' finals, and any known per-iteration values the unknown side reads are *recomputed inside the loop*
/// rather than streamed as residual edges, because a loop with a data-dependent trip count has no statically shaped
/// residual stream. This primal duplication is exactly what
/// [JAX's `_while_partial_eval`](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/loops.py)
/// accepts when linearizing `lax.while_loop`, and it is what makes [`Program::linearize`] total over the fused
/// doubled-state loops staged by the unbounded `while` forward-mode rule: the known (primal) side recovers the
/// primal outputs while the tangent program keeps the fused loop whole. The known-state outputs of the residual
/// loop are left dead.
///
/// The split does not apply — and the caller residualizes unchanged — when the converged known subset is empty or
/// complete (an all-known loop folds whole through the default rule), or when the predicate reads unknown state.
/// Callers must ensure both regions are pure: the known loop re-runs the known part of every iteration and the
/// residual loop runs all of it again, so an effectful loop would observe its effects twice.
fn split_while_by_closed_knownness<V, O, C, D>(
    context: &PartialEvaluationContext<C>,
    operation: &WhileOperation<V::Type>,
    condition: RegionRef<'_, V, O>,
    body: RegionRef<'_, V, O>,
    inputs: &[PartialEvaluationValue<C::Value>],
    driver: &D,
) -> Result<Option<Vec<PartialEvaluationValue<C::Value>>>, ProgramError>
where
    V: Value,
    V::Type: WhileTypeSemantics,
    C: Context<Constant = V, Operation = O>,
    O: Operation<Type = V::Type> + From<WhileOperation<V::Type>>,
    D: PartialEvaluationDriver<C>,
{
    let state_types = body.input_types();
    let state_count = state_types.len();

    // Fixed point over state known-ness: a state element can only be demoted as more are demoted, so the loop
    // converges in at most `state_count` rounds.
    let mut state_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
    let partition = loop {
        let partition = driver.partition_program(body, state_known.as_slice())?;
        let refined = (0..state_count)
            .map(|index| {
                state_known[index] && matches!(partition.outputs().get(index), Some(PartialEvaluationOutput::Known(_)))
            })
            .collect::<Vec<bool>>();
        if refined == state_known {
            break partition;
        }
        state_known = refined;
    };

    // The split only applies to a genuinely mixed converged state: an all-known loop folds whole through the
    // default rule and an all-unknown state leaves no known side to recover.
    if state_known.iter().all(|&known| known) || !state_known.iter().any(|&known| known) {
        return Ok(None);
    }

    // The predicate must fold from the known state alone; otherwise the known loop cannot reproduce the original
    // trip count and the split does not apply.
    let condition_partition = driver.partition_program(condition, state_known.as_slice())?;
    let (condition_known_program, _, condition_known_input_indices, _, condition_outputs) =
        condition_partition.into_parts();
    check_count!("output", condition_outputs, 1, ProgramError);
    let PartialEvaluationOutput::Known(predicate_output) = condition_outputs[0] else {
        return Ok(None);
    };

    let (known_program, _, known_input_indices, _, partition_outputs) = partition.into_parts();
    check_count!("output", partition_outputs, state_count, ProgramError);
    let known_state_indices = (0..state_count).filter(|&index| state_known[index]).collect::<Vec<_>>();
    if known_input_indices != known_state_indices || condition_known_input_indices != known_state_indices {
        return Err(ProgramError::MalformedProgram(format!(
            "{WHILE_OPERATION_NAME} body partition reported known input indices {known_input_indices:?} and its condition partition \
             reported {condition_known_input_indices:?} but the converged known state expects {known_state_indices:?}",
        )));
    }

    // Project the known body onto the known state subset: its inputs are the known state elements in state order
    // and its outputs are their known next values (the partition's trailing feeder-edge outputs are dropped — the
    // residual loop recomputes them).
    let known_state_types = known_state_indices.iter().map(|&index| state_types[index].clone()).collect::<Vec<_>>();
    let mut known_body_builder = ProgramBuilder::<V, O>::new();
    let known_body_inputs = known_state_types
        .iter()
        .map(|state_type| known_body_builder.add_input(state_type.clone()))
        .collect::<Vec<_>>();
    let known_program_outputs = known_body_builder.splice_program(&known_program, known_body_inputs.as_slice())?;
    let known_body_outputs = known_state_indices
        .iter()
        .map(|&index| match &partition_outputs[index] {
            PartialEvaluationOutput::Known(output) => known_program_outputs.get(*output).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "{WHILE_OPERATION_NAME} body partition output {index} references missing known-program output {output}",
                ))
            }),
            PartialEvaluationOutput::Unknown(_) => Err(ProgramError::MalformedProgram(
                format!("{WHILE_OPERATION_NAME} known-ness fixed point converged with an unknown next value for a known state element"),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let known_count = known_body_outputs.len();
    let known_body = known_body_builder.build::<Vec<V>, Vec<V>>(
        known_body_outputs,
        vec![Placeholder; known_count],
        vec![Placeholder; known_count],
    )?;

    // Project the known condition the same way: the known state elements in state order to the folded predicate.
    let mut known_condition_builder = ProgramBuilder::<V, O>::new();
    let known_condition_inputs = known_state_types
        .iter()
        .map(|state_type| known_condition_builder.add_input(state_type.clone()))
        .collect::<Vec<_>>();
    let known_condition_outputs =
        known_condition_builder.splice_program(&condition_known_program, known_condition_inputs.as_slice())?;
    let predicate_atom = known_condition_outputs.get(predicate_output).copied().ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "{WHILE_OPERATION_NAME} condition partition references missing known-program output {predicate_output}",
        ))
    })?;
    let known_condition = known_condition_builder.build::<Vec<V>, Vec<V>>(
        vec![predicate_atom],
        vec![Placeholder; known_count],
        vec![Placeholder; 1],
    )?;

    // Bind the known loop into the enclosing known-side context over the original known inits, and emit the
    // original loop unchanged into the residual program for the unknown state elements' finals.
    let known_while = WhileOperation::new().with_iteration_bound(operation.iteration_bound())?;
    let known_inputs = known_state_indices.iter().map(|&index| inputs[index].clone()).collect::<Vec<_>>();
    let known_outputs = context.fold_or_residualize(
        O::from(known_while),
        vec![known_condition, known_body],
        known_inputs.as_slice(),
    )?;
    check_count!("output", known_outputs, known_count, ProgramError);
    let residual_outputs =
        context.residualize(O::from(*operation), vec![condition.to_program(), body.to_program()], inputs)?;
    check_count!("output", residual_outputs, state_count, ProgramError);

    // Assemble the loop's outputs in state order: known finals from the known loop, unknown finals from the
    // residual loop.
    let mut known_outputs = known_outputs.into_iter();
    let outputs = residual_outputs
        .into_iter()
        .enumerate()
        .map(|(index, residual_output)| match state_known[index] {
            true => known_outputs.next().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "{WHILE_OPERATION_NAME} known loop is missing the final value of known state element {index}",
                ))
            }),
            false => Ok(residual_output),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(outputs))
}

/// Batching rule for [`WhileOperation`]. The rule builds batched loop *structure* and binds it into the parent
/// context — interpreted eagerly under an eager parent (whose relaxed-predicate interpretation owns the per-item
/// masked semantics) and staged into the enclosing trace under a staging parent:
///
///   1. Every batched state input is realigned to batch axis `0` in the parent context, and the body is batched at
///      the state batch axes via [`Program::batched`](crate::Program::batched),
///      iterating the axes to a fixed point: a while loop's state types are loop-invariant, so a replicated state
///      element whose update depends on a batched element *becomes* batched, and the rule widens that element's
///      input axis and re-batches until the body is axis-invariant (the iteration count is bounded by the state
///      count because every non-final pass widens at least one element). Each pass instantiates the body outputs at
///      the current state axes ([`ProgramBatchingOutputAxesPolicy::AlignEachTo`], mirroring JAX's
///      `instantiate=carry_bat`), so the converged body is already aligned to the loop-invariant state layout, and
///      widened parent inputs gain their batch axis through staged broadcasts.
///   2. The condition is batched at the stabilized axes. When its predicate output stays *replicated*, one
///      [`WhileOperation`] over the batched condition and body is bound into the parent directly, preserving any
///      semantic [`iteration_bound`](WhileOperation::with_iteration_bound) (so bounded loops stay reverse-capable
///      under `batch`).
///   3. When the predicate output is *batched* (per-item termination), every state element is widened to a batched
///      element, the condition's predicate output is instantiated at axis `0` (re-batching the condition only when its
///      natural predicate axis is not already `0`), and one
///      [`WhileOperation`] is bound directly with that batched predicate (mirroring JAX's
///      `_while_loop_batching_rule`). The predicate's `[axis_size]` shape is a prefix of every widened state shape,
///      so the bound loop satisfies the relaxed predicate contract and its consumers own the masked semantics:
///      eager interpretation continues while any per-item predicate is true and freezes finished items, and the XLA
///      lowering reduces the predicate with `or` and masks carry updates with a broadcast select. The iteration
///      bound is preserved (batch items share masked iterations, so capping the loop matches per-item truncation
///      exactly).
impl<C, O, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for WhileOperation<ArrayType>
where
    C: Context<Type = ArrayType, Operation = O>,
    <C as Domain>::Value: Broadcast + Transpose,
    O: Operation<Type = ArrayType>
        + From<TransposeOperation>
        + From<BroadcastOperation>
        + From<ReduceOperation>
        + From<SelectOperation<ArrayType>>
        + From<AndOperation<ArrayType>>
        + From<WhileOperation<ArrayType>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        // The rule requests all nested-computation work through its region access (region 0 is the condition and
        // region 1 the body), which keeps its bounds free of the operation family's own semantic traits.
        let state_count = inputs.len();
        let axis_size = P::axis_size(context)?;
        let condition_region = driver.region(0)?;
        let body_region = driver.region(1)?;

        // Realign every batched state input to batch axis 0 in the parent context, so the loop-invariance fixed
        // point below only ever distinguishes replicated (`None`) from batched-at-0 (`Some(0)`) state elements.
        let mut state = inputs.iter().map(|input| input.move_axis(0)).collect::<Result<Vec<_>, _>>()?;
        let mut state_axes = state.iter().map(|input| input.batch_axis()).collect::<Vec<_>>();

        // Iterate the body's batch axes to a fixed point: a replicated state element whose update is batched
        // becomes batched. Every non-final pass widens at least one of the `state_count` elements, so the loop
        // stabilizes within `state_count + 1` passes by construction; the trailing error guards the contract that
        // separately implemented batching rules report widening monotonically. Each pass instantiates the body's
        // outputs at the current state axes (JAX's `instantiate=carry_bat`), so the body that stabilizes the fixed
        // point is already aligned to the loop-invariant state layout and needs no further normalization.
        let mut batched_body = None;
        for _ in 0..=state_count {
            let (candidate_body, body_axes) = driver
                .batch_program(
                    context,
                    body_region,
                    state_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::AlignEachTo(state_axes.clone()),
                )?
                .into_parts();
            check_count!("output", body_axes, state_count, ProgramError);
            let mut widened = false;
            for (state_axis, body_axis) in state_axes.iter_mut().zip(body_axes.iter()) {
                if state_axis.is_replicated() && !body_axis.is_replicated() {
                    *state_axis = BatchAxis::new(0);
                    widened = true;
                }
            }
            if !widened {
                batched_body = Some(candidate_body);
                break;
            }
        }
        let Some(mut batched_body) = batched_body else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "{WHILE_OPERATION_NAME} loop batching failed to stabilize the loop state batch axes within {state_count} \
                     widening passes",
                ),
            });
        };

        // Batch the condition at the stabilized axes; a batched predicate output means per-item termination, in
        // which case every state element participates in per-item masking and is therefore widened to a batched
        // element before the masked loop structure is built.
        let mut batched_condition = driver.batch_program(
            context,
            condition_region,
            state_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        check_count!("output", batched_condition.output_axes(), 1, ProgramError);
        let batch_varying = !batched_condition.output_axes()[0].is_replicated();
        if batch_varying && state_axes.iter().any(|axis| axis.is_replicated()) {
            state_axes = vec![BatchAxis::new(0); state_count];
            let (widened_body, body_axes) = driver
                .batch_program(
                    context,
                    body_region,
                    state_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::AlignEachTo(state_axes.clone()),
                )?
                .into_parts();
            check_count!("output", body_axes, state_count, ProgramError);
            batched_body = widened_body;
            batched_condition = driver.batch_program(
                context,
                condition_region,
                state_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?;
            check_count!("output", batched_condition.output_axes(), 1, ProgramError);
        }

        // Widen the parent state values whose elements became batched (their batch axis is materialized through a
        // staged broadcast); the batched body's outputs are already aligned to the state axes by the fixed point.
        for (element, state_axis) in state.iter_mut().zip(state_axes.iter()) {
            if !state_axis.is_replicated() && element.batch_axis().is_replicated() {
                *element = element.broadcast(0, axis_size, context.axis_sharding().clone())?;
            }
        }
        let state_values = state.iter().map(|element| element.value().clone()).collect::<Vec<_>>();

        // Replicated predicate: stage one while over the batched condition and body directly.
        if !batch_varying {
            let batched_condition = batched_condition.into_parts().0;
            let batched_while = WhileOperation::new().with_iteration_bound(self.iteration_bound())?;
            let outputs = context.parent().bind(batched_while, vec![batched_condition, batched_body], &state_values)?;
            check_count!("output", outputs, state_count, ProgramError);
            return Ok(outputs
                .into_iter()
                .zip(state_axes)
                .map(|(output, axis)| ArrayBatch::new(output, axis))
                .collect::<Result<Vec<_>, _>>()?
                .into());
        }

        // Batch-varying predicate (per-item termination): re-batch the condition with its predicate output
        // instantiated at axis 0 and stage the while directly with that batched predicate, mirroring JAX's
        // `_while_loop_batching_rule` (which re-batches the cond jaxpr with the predicate at dimension 0 and binds
        // `while_p` directly). The predicate's `[axis_size]` shape is a prefix of every (widened) state shape, so the
        // staged loop satisfies the relaxed predicate contract, and the loop's consumers own the masked semantics:
        // eager interpretation continues while any per-item predicate is true and freezes finished items through
        // `WhilePredicate::mask_select`, and the XLA lowering
        // reduces the predicate with `or` and masks carry updates with a broadcast select. The condition batched above
        // used these same state axes, so a predicate that naturally landed on axis 0 keeps that program instead of
        // being replayed for an alignment that would stage nothing.
        let batched_condition = context.align_batched_program_outputs(
            driver,
            condition_region,
            state_axes.as_slice(),
            batched_condition,
            &[BatchAxis::new(0)],
        )?;
        let batched_while = WhileOperation::new().with_iteration_bound(self.iteration_bound())?;
        let outputs = context.parent().bind(batched_while, vec![batched_condition, batched_body], &state_values)?;
        check_count!("output", outputs, state_count, ProgramError);
        Ok(outputs
            .into_iter()
            .map(|output| ArrayBatch::new(output, Some(0)))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

/// Composite array IR batching rule for [`WhileOperation`].
///
/// Array carries use the same monotonic mapped-axis fixed point as homogeneous array loops. First-class dimensions
/// remain replicated loop state, and the transformed condition and body explicitly thread the mapped extent through
/// their region boundaries. Under a batch-varying predicate the replicated dimension carries ride through the loop as
/// loop-invariant state that per-item masking never touches (the relaxed [`WhileTypeSemantics`] contract documents that
/// invariance requirement), while *mapped* dimension inputs and dimension outputs that would widen into per-item
/// extents remain rejected.
impl<C> BatchableOperation<C, ArrayIrBatching> for WhileOperation<ArrayIrType>
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + From<WhileOperation<ArrayIrType>>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatching>, BatchingError> {
        let condition_region = driver.region(0)?;
        let body_region = driver.region(1)?;
        let state_count = inputs.len();

        // Canonicalize every mapped array carry to the leading axis. Dimension carries remain replicated.
        let mut state = inputs
            .iter()
            .cloned()
            .map(|input| match input.unbatched_type() {
                ArrayIrType::Array(_) if !input.batch_axis().is_replicated() => {
                    align_array_batch(context, input, Axis::from(0))
                }
                ArrayIrType::Array(_) => Ok(input),
                ArrayIrType::Dimension(_) => {
                    input.validate_replicated_dimension()?;
                    Ok(input)
                }
                ArrayIrType::Reference(_) => Err(BatchingError::UnsupportedOperation {
                    message: "references must be discharged before batching a `while`".to_string(),
                }),
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        let mut state_axes = state.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>();

        // Iterate array carry axes to the same monotonic fixed point as the homogeneous rule. A dimension output can
        // never widen because structural composite batching rejects mapped dimensions at its boundary.
        let mut batched_body = None;
        for _ in 0..=state_count {
            let candidate = driver.batch_program(
                context,
                body_region,
                state_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?;
            check_count!("output", candidate.output_axes(), state_count, ProgramError);
            let mut widened = false;
            for (index, (state_axis, body_axis)) in
                state_axes.iter_mut().zip(candidate.output_axes().iter()).enumerate()
            {
                if state_axis.is_replicated() && !body_axis.is_replicated() {
                    if matches!(inputs[index].unbatched_type(), ArrayIrType::Dimension(_)) {
                        return Err(BatchingError::MappedDimension {
                            r#type: Box::new(<&DimensionType>::try_from(&inputs[index].unbatched_type())?.clone()),
                            axis: *body_axis,
                        });
                    }
                    *state_axis = BatchAxis::new(0);
                    widened = true;
                }
            }
            if !widened {
                // The stabilizing pass already used these input axes, so when its discovered (normalized) output axes
                // equal the state axes it *is* the aligned body and is kept as-is instead of being rebuilt.
                batched_body = Some(context.align_batched_program_outputs(
                    driver,
                    body_region,
                    state_axes.as_slice(),
                    candidate,
                    state_axes.as_slice(),
                )?);
                break;
            }
        }
        let Some(mut batched_body) = batched_body else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "{WHILE_OPERATION_NAME} loop batching failed to stabilize the loop state batch axes within {state_count} \
                     widening passes",
                ),
            });
        };

        let mut batched_condition = driver.batch_program(
            context,
            condition_region,
            state_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        check_count!("output", batched_condition.output_axes(), 1, ProgramError);
        let batch_varying = !batched_condition.output_axes()[0].is_replicated();
        if batch_varying {
            // Per-item termination masks every array carry, so the masked invariant boundary widens every array carry
            // to the leading axis and forces the predicate itself to axis 0, and both regions are rebuilt there.
            // Dimension carries stay replicated — they are loop-invariant passthrough state that masking never touches
            // — so they are neither widened nor a reason to rebuild. When that boundary already equals the discovered
            // one and the predicate naturally landed on axis 0, the rebuilds would replay identical programs, so the
            // discovery body and condition are kept instead.
            let masked_state_axes = state_axes
                .iter()
                .zip(inputs.iter())
                .map(|(axis, input)| match input.unbatched_type() {
                    ArrayIrType::Array(_) => BatchAxis::new(0),
                    ArrayIrType::Dimension(_) => *axis,
                    ArrayIrType::Reference(_) => {
                        unreachable!("reference state inputs are rejected before while batching discovery")
                    }
                })
                .collect::<Vec<_>>();
            if masked_state_axes != state_axes || batched_condition.output_axes()[0] != BatchAxis::new(0) {
                state_axes = masked_state_axes;
                batched_body = driver
                    .batch_program(
                        context,
                        body_region,
                        state_axes.as_slice(),
                        ProgramBatchingOutputAxesPolicy::AlignEachTo(state_axes.clone()),
                    )?
                    .into_parts()
                    .0;
                batched_condition = driver.batch_program(
                    context,
                    condition_region,
                    state_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(0)]),
                )?;
            }
        }

        for (value, axis) in state.iter_mut().zip(state_axes.iter()) {
            if !axis.is_replicated() && value.batch_axis().is_replicated() {
                *value = align_array_batch(context, value.clone(), Axis::from(0))?;
            }
        }
        let (batched_condition, _) = batched_condition.into_parts();

        // Composite region batching forwards the extent as its leading output. The while condition consumes that
        // extent as loop state but returns only its predicate, so project the bookkeeping output from this boundary.
        if batched_condition.output_count() < 2 {
            return Err(ProgramError::MalformedProgram(format!(
                "a structurally batched {WHILE_OPERATION_NAME} condition must return its threaded extent and predicate"
            ))
            .into());
        }
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        let condition_inputs = batched_condition
            .input_types()
            .into_iter()
            .map(|r#type| builder.add_input(r#type))
            .collect::<Vec<_>>();
        let condition_outputs = builder.splice_program(&batched_condition, condition_inputs.as_slice())?;
        let condition_output_count = condition_outputs.len() - 1;
        let batched_condition = builder.build(
            condition_outputs[1..].to_vec(),
            vec![Placeholder; condition_inputs.len()],
            vec![Placeholder; condition_output_count],
        )?;

        let mut packed_inputs = Vec::with_capacity(state.len() + 1);
        packed_inputs.push(context.axis_extent().clone());
        packed_inputs.extend(state.iter().map(|value| value.value().clone()));
        let mut outputs =
            context.parent().bind(*self, vec![batched_condition, batched_body], packed_inputs.as_slice())?;
        check_count!("output", outputs, state_count + 1, ProgramError);
        outputs.remove(0);
        Ok(outputs
            .into_iter()
            .zip(state_axes)
            .map(|(output, axis)| ArrayIrBatch::new(output, axis))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

/// Forward-mode (JVP) rule for [`WhileOperation`]. An [eager](Context::is_eager) context
/// runs the loop directly at the concrete duals (see the crate-private `jvp_while_eagerly`), so eager forward mode is
/// total over data-dependent `while` loops with no iteration bound. Staging contexts — and eager contexts whose loop
/// predicate is batched and therefore has no single trip decision — dispatch to the loop's type family through the
/// `WhileJvp` trait: bounded array loops stage the reverse-capable hybrid rule documented on that trait's
/// [`ArrayType`] implementation, while unbounded loops stage the forward-only fused doubled-state loop (see the
/// crate-private `jvp_while_fused`).
impl<C> DifferentiableOperation<C> for WhileOperation<C::Type>
where
    C: Context + Zero<C::Value>,
    C::Type: WhileJvp<C>,
    C::Value: Concretizable<bool>,
    C::Operation: ZeroOperationProvider<C::Type>,
    for<'operation> &'operation WhileOperation<C::Type>: TryFrom<&'operation C::Operation>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The rule requests all nested-computation work through its driver (region 0 is the condition and
        // region 1 the body), which keeps its bounds free of the operation family's own semantic traits.
        let condition = driver.region(0)?.to_program();
        let body = driver.region(1)?.to_program();
        if context.is_eager()
            && let Some(outputs) = jvp_while_eagerly(self, &condition, &body, context, driver, inputs)?
        {
            return Ok(outputs);
        }
        <C::Type>::jvp_while(self, &condition, &body, context, driver, inputs)
    }
}

/// Type-family forward-mode (JVP) semantics for [`WhileOperation`], with the differentiation context riding as a
/// trait input and the type family as the implementing type (mirroring the partial-evaluation dispatch in the
/// `while` module), so that each family implementation carries exactly the capability bounds its rule needs.
pub(crate) trait WhileJvp<C>: DifferentiableType + WhileTypeSemantics
where
    C: Context<Type = Self>,
{
    /// Applies the type family's `while` forward-mode rule over the loop's materialized condition and body region
    /// programs; refer to the documentation of [`DifferentiableOperation::jvp`] for the contract. The scoped
    /// `driver` serves the rule's nested forward-mode and linearization requests over rebuilt body forms,
    /// keeping the rule free of operation-family semantic bounds.
    fn jvp_while<D: DifferentiationDriver<C>>(
        operation: &WhileOperation<Self>,
        condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>;
}

/// Array-backed storage semantics used by bounded `while` differentiation.
///
/// A bounded loop stores each per-iteration residual in an ordinary leading-axis array stack. Homogeneous array
/// programs use their residual type directly. Array IR programs additionally map first-class dimension
/// residuals to checked scalar-array storage and restore them at the tangent-body boundary. Keeping this contract at
/// the type-family boundary lets both operation families share the bounded-loop algorithm without making the generic
/// control-flow rule inspect composite variants.
pub(crate) trait WhileResidualStackType: DifferentiableType + TemporalResidualType + WhileTypeSemantics {
    /// Wraps an array type in this type family.
    fn from_array_type(r#type: ArrayType) -> Self;

    /// Projects an array member from this type family.
    fn array_type(&self) -> Result<&ArrayType, TypeError>;

    /// Returns the array member that a batched-predicate mask/select applies to, or `None` for a first-class dimension
    /// carry. Such a carry is loop-invariant under a batched predicate, so masking it is the identity and it is exempt
    /// from the per-element selection the masked normal form builds (refer to the documentation of
    /// [`WhileTypeSemantics`]).
    fn maskable_array_type(&self) -> Option<&ArrayType>;
}

/// Operation-family constructors required by the shared bounded-`while` residual-stack algorithm.
///
/// The methods return operations rather than binding values, so the same contract works in the parent
/// differentiation context and in the nested tracing contexts used to build the augmented primal body and masked
/// tangent scan body. Composite implementations own the visible dimension/scalar gateway operations; homogeneous
/// array implementations return `None` for both gateway hooks.
pub(crate) trait WhileResidualStackOperation<T: WhileResidualStackType, A>:
    Operation<Type = T> + TemporalResidualOperation<T>
{
    /// Constructs a zero for a statically shaped array-backed state type.
    fn residual_stack_zero(r#type: T) -> Self;

    /// Constructs a one for a scalar array-backed state type.
    fn residual_stack_one(r#type: T) -> Self;

    /// Constructs an array broadcast used to insert a leading stack axis or expand a mask.
    fn residual_stack_broadcast(output_type: ArrayType, output_axes: Vec<usize>) -> Self;

    /// Constructs an array dynamic-update-slice operation.
    fn residual_stack_update() -> Self;

    /// Constructs an array addition operation.
    fn residual_stack_add() -> Self;

    /// Constructs an array selection operation.
    fn residual_stack_select() -> Self;

    /// Constructs a Boolean `any` reduction used to scalarize a batched loop predicate.
    fn mask_reduce_any(axes: Vec<usize>) -> Self;

    /// Constructs a Boolean conjunction used to retain only active loop items.
    fn mask_and() -> Self;
}

impl TemporalResidualType for ArrayType {
    #[inline]
    fn temporal_storage_type(&self) -> Result<Self, TypeError> {
        Ok(self.clone())
    }
}

impl<O: Operation<Type = ArrayType>> TemporalResidualOperation<ArrayType> for O {
    #[inline]
    fn residual_to_storage(_residual_type: &ArrayType) -> Result<Option<Self>, TypeError> {
        Ok(None)
    }

    #[inline]
    fn residual_from_storage(_residual_type: &ArrayType) -> Result<Option<Self>, TypeError> {
        Ok(None)
    }
}

impl WhileResidualStackType for ArrayType {
    #[inline]
    fn from_array_type(r#type: ArrayType) -> Self {
        r#type
    }

    #[inline]
    fn array_type(&self) -> Result<&ArrayType, TypeError> {
        Ok(self)
    }

    #[inline]
    fn maskable_array_type(&self) -> Option<&ArrayType> {
        Some(self)
    }
}

impl<A, O> WhileResidualStackOperation<ArrayType, A> for O
where
    O: Operation<Type = ArrayType>
        + From<ZeroOperation<ArrayType>>
        + From<OneOperation<ArrayType>>
        + From<AddOperation<ArrayType>>
        + From<BroadcastOperation>
        + From<DynamicUpdateSliceOperation>
        + From<SelectOperation<ArrayType>>
        + From<ReduceOperation>
        + From<AndOperation<ArrayType>>,
{
    #[inline]
    fn residual_stack_zero(r#type: ArrayType) -> Self {
        Self::from(ZeroOperation::new(r#type))
    }

    #[inline]
    fn residual_stack_one(r#type: ArrayType) -> Self {
        Self::from(OneOperation::new(r#type))
    }

    #[inline]
    fn residual_stack_broadcast(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        Self::from(BroadcastOperation::new(output_type, output_axes))
    }

    #[inline]
    fn residual_stack_update() -> Self {
        Self::from(DynamicUpdateSliceOperation)
    }

    #[inline]
    fn residual_stack_add() -> Self {
        Self::from(AddOperation::new())
    }

    #[inline]
    fn residual_stack_select() -> Self {
        Self::from(SelectOperation::<ArrayType>::new())
    }

    #[inline]
    fn mask_reduce_any(axes: Vec<usize>) -> Self {
        Self::from(ReduceOperation::new(axes, ReductionKind::Any))
    }

    #[inline]
    fn mask_and() -> Self {
        Self::from(AndOperation::new())
    }
}

/// Capture-free forward-mode (JVP) rule for the bounded [`WhileOperation`], staging an augmented primal `while`
/// and one masked length-`bound` tangent `scan` as ordinary primal-enum operations over the shared builder.
///
/// In the bounded regime the rule keeps every per-iteration residual and the validity mask as plain primal operand
/// edges: they leave the augmented primal while as ordinary stacked outputs and re-enter the tangent scan as ordinary
/// stacked scanned inputs, so no symbolic capture is ever introduced. The enclosing partial-evaluation split then
/// discovers the residual operand edges structurally, exactly as it does for the scan and condition rules.
///
/// **The unbounded case stages the fused doubled-state loop instead.** This staged rule is only reached when the
/// context is not [eager](Context::is_eager) (eager contexts run the loop directly through [`jvp_while_eagerly`],
/// with no bound needed), and without a semantic
/// [`iteration_bound`](crate::operations::control_flow::WhileOperation::with_iteration_bound) there is no statically
/// shaped residual stack for the bounded strategy below, so the rule defers to [`jvp_while_fused`]: one `while` over
/// `[primal_state..., tangent_state...]` whose trip decision reads the primal half, forward-total but not
/// transposable — reverse mode through a staged unbounded loop still reports the transposition error, exactly like
/// JAX's `lax.while_loop`.
///
/// For a bound `B`, the rule linearizes the body capture-free through its instruction-scoped driver,
/// giving a primal body `[state] -> [next_state, residuals...]` and a tangent body
/// `[state_tangent, residuals...] -> [next_state_tangent]` together with the residual count. It then:
///
///   1. Builds the augmented primal `while` over the state `[original_state..., counter (i64 scalar), residual_stacks
///      (one zero-initialized [B, ...] stack per residual), mask_stack (a false-initialized Boolean [B] stack)]`
///      with `build_bounded_while_programs` from the residual-extended primal body, keeping `iteration_bound = B` so
///      the per-item writes can never clamp, and stages it over the operand primals followed by the staged counter and
///      stack zeros. Its outputs split into the original state outputs (the primal outputs), the dropped counter, the
///      stacked residual outputs, and the mask stack.
///   2. Stages a length-`B` tangent [`ScanOperation`] over the live tangent carries, residual storage stacks, and one
///      Boolean validity stack. Each iteration reads one scalar validity item and selects between the pushforward
///      output and the carried tangent, relying on ordinary scalar broadcasting. Dimension residuals that vary across
///      iterations cross the scan boundary through checked scalar-array gateways; forwarded invariant dimensions are
///      threaded directly as known carries. Gateway restoration is placed behind a condition so inactive placeholder
///      storage after early exit is never interpreted as a dimension value.
///   3. Pairs each primal output tracer with its tangent output tracer into a [`DifferentiationDual`].
///
/// Reverse mode is total with no while-specific transpose code: the staged tangent scan re-keys through the existing
/// scan re-key path into a captured-stack linear scan whose body re-keys the per-iteration `select` over its mask-item
/// capture, and the single outer transpose flips the scan direction and transposes the body — the masked pushforward
/// side receives a zero cotangent on inactive batch items while the carried side receives the full cotangent, so
/// cotangents pass through inactive batch items unchanged.
fn jvp_array_backed_while<C, D: DifferentiationDriver<C>, A>(
    operation: &WhileOperation<C::Type>,
    condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    context: &C,
    driver: &D,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    C: Context<Type: WhileResidualStackType> + Zero<C::Value>,
    C::Value: Concretizable<bool>,
    C::Operation: ResidualZeroProvider<C::Type>
        + From<ConditionOperation<C::Constant>>
        + From<WhileOperation<C::Type>>
        + From<ScanOperation<C::Constant>>
        + WhileResidualStackOperation<C::Type, A>,
{
    let state_count = body.input_count();
    check_count!("input", inputs, state_count, ProgramError);

    // An unbounded loop has no statically shaped residual stacks for the bounded hybrid rule below, so it stages
    // the fused doubled-state loop instead (see `jvp_while_fused`). The fused path needs no batched-predicate
    // rewrite: the fused state doubles every prefix-shaped state element, so a per-item predicate still satisfies
    // the relaxed predicate contract over the doubled state.
    let Some(bound) = operation.iteration_bound() else {
        return jvp_while_fused(operation, condition, context, driver, inputs);
    };

    // A batched (per-item) predicate cannot thread the bounded rule's augmented differentiation state through the
    // predicate-prefix contract (the scalar iteration counter and the `[bound, ...]` residual stacks are not
    // predicate-prefixed), so the loop is first rewritten into its scalar-predicate masked normal form over
    // `[state..., active_mask]` (see `masked_while_programs`) and differentiated recursively — the masked loop's
    // forward mode is this same rule. The initial mask is the condition replayed on the operand primals, carried
    // with a zero tangent since a Boolean mask has no derivative.
    if C::Type::is_batched_predicate(&condition.output_types()[0]) {
        let (masked_condition, masked_body) = masked_while_programs::<_, _, A>(condition, body)?;
        let masked_while = WhileOperation::new().with_iteration_bound(operation.iteration_bound())?;
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut initial_mask = condition.interpret_in_context(context, primal_operands)?;
        check_count!("output", initial_mask, 1, ProgramError);
        let mut extended_inputs = inputs.to_vec();
        extended_inputs.push(DifferentiationDual::new_with_zero_tangent(initial_mask.remove(0))?);
        // The masked loop's condition and body are freshly built region programs, so the recursive `jvp` is
        // requested through the instruction-scoped driver over them.
        let mut outputs = driver.jvp_operation(
            &C::Operation::from(masked_while),
            vec![masked_condition, masked_body],
            context,
            extended_inputs.as_slice(),
        )?;
        check_count!("output", outputs, state_count + 1, ProgramError);
        outputs.truncate(state_count);
        return Ok(outputs);
    }

    jvp_while_bounded::<_, _, A>(condition, context, driver, inputs, bound)
}

impl<C: Context<Type = ArrayType> + Zero<C::Value>> WhileJvp<C> for ArrayType
where
    C::Value: Concretizable<bool>,
    C::Operation: ResidualZeroProvider<ArrayType>
        + From<ConditionOperation<C::Constant>>
        + From<WhileOperation<ArrayType>>
        + From<ScanOperation<C::Constant>>
        + WhileResidualStackOperation<ArrayType, C::Constant>,
{
    fn jvp_while<D: DifferentiationDriver<C>>(
        operation: &WhileOperation<ArrayType>,
        condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        jvp_array_backed_while::<_, _, C::Constant>(operation, condition, body, context, driver, inputs)
    }
}

/// Shared reverse-capable forward rule for a bounded, scalar-predicate, array-backed `while` loop.
fn jvp_while_bounded<C, D: DifferentiationDriver<C>, A>(
    condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    context: &C,
    driver: &D,
    inputs: &[DifferentiationDual<C::Value>],
    bound: usize,
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    C: Context<Type: WhileResidualStackType> + Zero<C::Value>,
    C::Operation: ResidualZeroProvider<C::Type>
        + WhileResidualStackOperation<C::Type, A>
        + From<ConditionOperation<C::Constant>>
        + From<WhileOperation<C::Type>>
        + From<ScanOperation<C::Constant>>,
{
    let state_types = driver.region(1)?.input_types();
    let state_count = state_types.len();

    // Linearize the body once. The nonlinear half returns the next state followed by ordinary residuals, and the
    // linear half consumes the live state tangents followed by those residuals.
    let (primal_program, tangent_program, residual_count) = driver.linearize_program(driver.region(1)?)?.into_parts();
    let residual_types = primal_program.output_types().split_off(state_count);
    check_count!("output", residual_types, residual_count, ProgramError);

    // A dimension residual that is exactly a forwarded dimension-state input is invariant across the loop. Thread
    // it directly as known tangent-scan state instead of storing one scalar copy per iteration. This preserves its
    // boundary identity for dynamic tangent-array carries and limits scalar gateway storage to varying dimensions.
    let primal_input_ids = primal_program.input_ids().to_vec();
    let primal_output_ids = primal_program.output_ids().to_vec();
    let invariant_residual_sources = (0..residual_count)
        .map(|residual_index| {
            if C::Operation::residual_from_storage(&residual_types[residual_index])?.is_none() {
                return Ok(None);
            }
            let residual = primal_output_ids[state_count + residual_index];
            Ok((0..state_count).find(|&state_index| {
                primal_input_ids[state_index] == residual && primal_output_ids[state_index] == residual
            }))
        })
        .collect::<Result<Vec<Option<usize>>, TypeError>>()?;
    let stacked_residual_indices = invariant_residual_sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| source.is_none().then_some(index))
        .collect::<Vec<_>>();
    let stacked_residual_types =
        stacked_residual_indices.iter().map(|&index| residual_types[index].clone()).collect::<Vec<_>>();
    let stacked_primal_program = if stacked_residual_indices.len() == residual_count {
        primal_program
    } else {
        let input_types = primal_program.input_types();
        let mut builder = ProgramBuilder::new();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
        let outputs = builder.splice_program(&primal_program, inputs.as_slice())?;
        let selected_outputs = outputs[..state_count]
            .iter()
            .copied()
            .chain(stacked_residual_indices.iter().map(|&index| outputs[state_count + index]))
            .collect::<Vec<_>>();
        builder.build(
            selected_outputs,
            vec![Placeholder; inputs.len()],
            vec![Placeholder; state_count + stacked_residual_indices.len()],
        )?
    };

    // Run the primal loop over its original state plus a counter, one stack per stored residual, and a validity mask.
    let counter_type = C::Type::from_array_type(ArrayType::scalar(DataType::I64));
    let boolean_scalar_type = C::Type::from_array_type(ArrayType::scalar(DataType::Boolean));
    let mask_stack_type = C::Type::from_array_type(stacked_scan_type(boolean_scalar_type.array_type()?, bound));
    let (extended_condition, augmented_body, stack_types) = build_bounded_while_programs::<_, _, A>(
        condition,
        &stacked_primal_program,
        stacked_residual_types.as_slice(),
        bound,
    )?;
    let mut primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    for zero_state_type in
        std::iter::once(&counter_type).chain(stack_types.iter()).chain(std::iter::once(&mask_stack_type))
    {
        let mut zeros = context.bind(C::Operation::residual_stack_zero(zero_state_type.clone()), Vec::new(), &[])?;
        check_count!("output", zeros, 1, ProgramError);
        primal_operands.push(zeros.remove(0));
    }
    let mut while_outputs = context.bind(
        C::Operation::from(WhileOperation::new().with_iteration_bound(bound)?),
        vec![extended_condition, augmented_body],
        primal_operands.as_slice(),
    )?;
    check_count!("output", while_outputs, state_count + 2 + stack_types.len(), ProgramError);
    let mask_stack = while_outputs.pop().unwrap();
    let residual_stacks = while_outputs.split_off(state_count + 1);
    while_outputs.truncate(state_count);
    let primal_outputs = while_outputs;

    let element_has_tangent = state_types
        .iter()
        .map(|state_type| Ok(!state_type.tangent()?.is_zero_space()))
        .collect::<Result<Vec<_>, DifferentiationError>>()?;
    let tangent_state_count = element_has_tangent.iter().filter(|&&has_tangent| has_tangent).count();
    if tangent_state_count == 0 {
        return primal_outputs.into_iter().map(DifferentiationDual::new_with_zero_tangent).collect();
    }

    check_count!("input", tangent_program.input_ids(), tangent_state_count + residual_count, ProgramError);
    check_count!("output", tangent_program.output_ids(), tangent_state_count, ProgramError);
    let storage_types = stacked_residual_types
        .iter()
        .map(TemporalResidualType::temporal_storage_type)
        .collect::<Result<Vec<_>, _>>()?;
    let invariant_residual_indices = invariant_residual_sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| source.is_some().then_some(index))
        .collect::<Vec<_>>();
    let invariant_residual_count = invariant_residual_indices.len();
    let tangent_input_types = tangent_program.input_types();
    let iteration_input_types = invariant_residual_indices
        .iter()
        .map(|&index| residual_types[index].clone())
        .chain(tangent_input_types[..tangent_state_count].iter().cloned())
        .chain(storage_types.iter().cloned())
        .collect::<Vec<_>>();
    let uses_gateway = stacked_residual_types
        .iter()
        .map(C::Operation::residual_from_storage)
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(Option::is_some);
    let active_body = TracingContext::<C::Constant, C::Operation>::trace(
        |inputs| -> Result<Vec<_>, ProgramError> {
            let trace_context = inputs[0].context().clone();
            let invariant_inputs = &inputs[..invariant_residual_count];
            let tangent_carry_start = invariant_residual_count;
            let tangent_carry_end = tangent_carry_start + tangent_state_count;
            let carried_tangents = &inputs[tangent_carry_start..tangent_carry_end];
            let mut tangent_inputs = carried_tangents.to_vec();
            let mut restored_residuals: Vec<(C::Type, Tracer<TracingContext<C::Constant, C::Operation>>)> = Vec::new();
            let storage_inputs = &inputs[tangent_carry_end..tangent_carry_end + stacked_residual_indices.len()];
            let mut invariant_residuals = invariant_inputs.iter();
            let mut storage_inputs = storage_inputs.iter().zip(storage_types.iter());
            for (residual_index, residual_type) in residual_types.iter().enumerate() {
                if invariant_residual_sources[residual_index].is_some() {
                    tangent_inputs.push(invariant_residuals.next().unwrap().clone());
                    continue;
                }
                let (storage_input, storage_type) = storage_inputs.next().unwrap();
                if storage_input.r#type().as_ref() != storage_type {
                    return Err(TypeError::invalid(format!(
                        "bounded {} residual stack item has type {} but expected {}",
                        WHILE_OPERATION_NAME,
                        storage_input.r#type().as_ref(),
                        storage_type,
                    ))
                    .into());
                }
                if let Some(operation) = C::Operation::residual_from_storage(residual_type)? {
                    if let Some((_, restored)) =
                        restored_residuals.iter().find(|(restored_type, _)| restored_type == residual_type)
                    {
                        tangent_inputs.push(restored.clone());
                        continue;
                    }
                    let mut restored =
                        trace_context.bind(operation, Vec::new(), std::slice::from_ref(storage_input))?;
                    check_count!("output", restored, 1, ProgramError);
                    let restored = restored.remove(0);
                    restored_residuals.push((residual_type.clone(), restored.clone()));
                    tangent_inputs.push(restored);
                } else {
                    tangent_inputs.push(storage_input.clone());
                }
            }
            let pushforward_outputs = tangent_program.interpret_in_context(&trace_context, tangent_inputs)?;
            check_count!("output", pushforward_outputs, tangent_state_count, ProgramError);
            let mut outputs = invariant_inputs.to_vec();
            outputs.extend(pushforward_outputs);
            Ok(outputs)
        },
        iteration_input_types.clone(),
    )?
    .1;
    let mut scan_body_input_types = iteration_input_types.clone();
    scan_body_input_types.push(boolean_scalar_type);
    let scan_body = if uses_gateway {
        // Inactive bounded iterations contain only placeholder storage. Guard the restoration branch itself so a
        // checked dimension gateway never observes an unused out-of-bounds placeholder value.
        let mut passive_builder = ProgramBuilder::new();
        let passive_inputs = iteration_input_types
            .iter()
            .map(|r#type| passive_builder.add_input(r#type.clone()))
            .collect::<Vec<_>>();
        let passive_output_count = invariant_residual_count + tangent_state_count;
        let passive_body = passive_builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            passive_inputs[..passive_output_count].to_vec(),
            vec![Placeholder; passive_inputs.len()],
            vec![Placeholder; passive_output_count],
        )?;
        TracingContext::<C::Constant, C::Operation>::trace(
            |inputs| {
                let trace_context = inputs[0].context().clone();
                let (iteration_inputs, mask) = inputs.split_at(inputs.len() - 1);
                let mut operands = Vec::with_capacity(inputs.len());
                operands.push(mask[0].clone());
                operands.extend_from_slice(iteration_inputs);
                trace_context.bind(
                    C::Operation::from(ConditionOperation::new()),
                    vec![active_body, passive_body],
                    operands.as_slice(),
                )
            },
            scan_body_input_types,
        )?
        .1
    } else {
        TracingContext::<C::Constant, C::Operation>::trace(
            |inputs| -> Result<Vec<_>, ProgramError> {
                let trace_context = inputs[0].context().clone();
                let (iteration_inputs, mask) = inputs.split_at(inputs.len() - 1);
                let active_outputs = active_body.interpret_in_context(&trace_context, iteration_inputs.to_vec())?;
                let mut outputs = active_outputs[..invariant_residual_count].to_vec();
                outputs.extend(
                    active_outputs[invariant_residual_count..]
                        .iter()
                        .zip(
                            &iteration_inputs[invariant_residual_count..invariant_residual_count + tangent_state_count],
                        )
                        .map(|(pushforward_output, carried_input)| {
                            let mut selected = trace_context.bind(
                                C::Operation::residual_stack_select(),
                                Vec::new(),
                                &[mask[0].clone(), pushforward_output.clone(), carried_input.clone()],
                            )?;
                            check_count!("output", selected, 1, ProgramError);
                            Ok(selected.remove(0))
                        })
                        .collect::<Result<Vec<_>, ProgramError>>()?,
                );
                Ok(outputs)
            },
            scan_body_input_types,
        )?
        .1
    };

    // The tangent scan consumes invariant dimension residuals, live tangent carries, stored residual stacks, and one
    // scalar validity stack.
    let mut tangent_operands = invariant_residual_sources
        .iter()
        .flatten()
        .map(|&state_index| inputs[state_index].primal().clone())
        .collect::<Vec<_>>();
    // Each state element's own primal names every runtime quantity a reference-bearing tangent type omits, because
    // the tangent type derivation preserves geometry exactly.
    tangent_operands.extend(
        inputs
            .iter()
            .zip(&element_has_tangent)
            .filter_map(|(input, &has_tangent)| {
                has_tangent.then(|| {
                    C::Operation::materialize_zero_from_residual_sources(
                        context,
                        input.tangent().clone(),
                        std::iter::once(input.primal()),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    tangent_operands.extend(residual_stacks);
    tangent_operands.push(mask_stack);
    let tangent_scan = ScanOperation::<C::Constant>::new(invariant_residual_count + tangent_state_count, bound);
    let tangent_outputs =
        context.bind(C::Operation::from(tangent_scan), vec![scan_body], tangent_operands.as_slice())?;
    check_count!("output", tangent_outputs, invariant_residual_count + tangent_state_count, ProgramError);

    let mut tangent_outputs = tangent_outputs.into_iter().skip(invariant_residual_count);
    primal_outputs
        .into_iter()
        .zip(element_has_tangent)
        .map(|(primal, has_tangent)| {
            if has_tangent {
                DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
            } else {
                DifferentiationDual::new_with_zero_tangent(primal)
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

impl<C: Context<Type = ArrayIrType> + Zero<C::Value>> WhileJvp<C> for ArrayIrType
where
    C::Constant: ValueProjection<ArrayType>,
    C::Value: Concretizable<bool>,
    C::Operation: ResidualZeroProvider<ArrayIrType>
        + From<ConditionOperation<C::Constant>>
        + From<WhileOperation<ArrayIrType>>
        + From<ScanOperation<C::Constant>>
        + WhileResidualStackOperation<ArrayIrType, <C::Constant as ValueProjection<ArrayType>>::Projected>,
{
    fn jvp_while<D: DifferentiationDriver<C>>(
        operation: &WhileOperation<ArrayIrType>,
        condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        jvp_array_backed_while::<_, _, <C::Constant as ValueProjection<ArrayType>>::Projected>(
            operation, condition, body, context, driver, inputs,
        )
    }
}

/// Stages **one fused** doubled-state forward-mode `while` as an ordinary primal-enum operation over the shared
/// builder — the analogue of
/// [JAX's `jvp` of `lax.while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html), which
/// runs the pushforward alongside the primal loop instead of storing per-iteration residuals. The fused state is
/// `[primal_state..., tangent_state...]`: the body is the loop body's fused forward-mode program (built through the
/// instruction-scoped driver) and the condition is the original condition extended with ignored tangent-state
/// inputs, so the trip decision reads the primal state alone and the fused loop runs exactly as long as the primal
/// loop. The fused state is compact: state elements whose tangent type is a zero differential space carry no
/// tangent boundary input or output, and their output duals are restored as structural zeros. Because no residuals
/// are stored, the rule applies to loops with *no* [`WhileOperation::iteration_bound`].
///
/// The primal/tangent separation that linearization needs is recovered by partial evaluation rather than by this
/// rule: the fused loop's primal half is *closed* (its next state and the predicate fold from primal state alone),
/// so the `while` closed-knownness split (see the crate-private `split_while_by_closed_knownness`) rebinds a known
/// primal-only loop on the known side and keeps the fused loop whole on the residual side, recomputing primal state
/// there — the same primal duplication JAX's linearize-of-`while_loop` performs. Because the fused loop stores no
/// per-iteration residuals, its linearized form is **not transposable**: reverse mode through a staged unbounded
/// loop still reports the `while` transposition error, exactly like JAX's `lax.while_loop`.
fn jvp_while_fused<C, D: DifferentiationDriver<C>>(
    operation: &WhileOperation<C::Type>,
    condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    context: &C,
    driver: &D,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    C: Context + Zero<C::Value>,
    C::Type: DifferentiableType + WhileTypeSemantics,
    C::Operation: ResidualZeroProvider<C::Type> + From<WhileOperation<C::Type>>,
{
    let state_count = inputs.len();

    // Build the fused body over the compact state `[primal_state..., live(tangent_state)...]` through the
    // instruction-scoped driver (region 1 is the loop body). The fused body carries no tangent boundary inputs or
    // outputs for state elements whose tangent type is a zero differential space, so the liveness mask below is
    // derived from the same body state types that the fused-body construction filtered on.
    let element_has_tangent = driver
        .region(1)?
        .input_types()
        .iter()
        .map(|r#type| Ok(!r#type.tangent()?.is_zero_space()))
        .collect::<Result<Vec<_>, DifferentiationError>>()?;
    check_count!("input", element_has_tangent, state_count, ProgramError);
    let tangent_state_count = element_has_tangent.iter().filter(|&&has_tangent| has_tangent).count();
    let fused_state_count = state_count + tangent_state_count;
    // The body is differentiated through its region's retained transform cache, so a body shared by several programs
    // is differentiated once and repeated attachments of the result intern by `Arc` identity.
    let fused_body = driver.jvp_program(driver.region(1)?)?;
    let fused_state_types = fused_body.input_types();
    check_count!("input", fused_state_types, fused_state_count, ProgramError);

    // Extend the condition over the compact fused state: the original condition reads the primal prefix and the
    // live tangent-state inputs are ignored, so the fused loop's trip count is driven by the primal state alone.
    let mut condition_builder = ProgramBuilder::<C::Constant, C::Operation>::new();
    let condition_inputs =
        fused_state_types.into_iter().map(|r#type| condition_builder.add_input(r#type)).collect::<Vec<_>>();
    let condition_outputs = condition_builder.splice_program(condition, &condition_inputs[..state_count])?;
    let condition_output_count = condition_outputs.len();
    let fused_condition = condition_builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
        condition_outputs,
        vec![Placeholder; fused_state_count],
        vec![Placeholder; condition_output_count],
    )?;

    // Stage the fused loop over the operand primals followed by the materialized tangents of the live differential
    // state elements — the fused body takes each of those tangents as a real program input, so their structural
    // zeros are materialized against their own primal, which supplies the runtime geometry a reference-bearing
    // tangent type omits — and zip the output halves back into `DifferentiationDual`s in the original state order,
    // restoring structural zeros for the omitted zero-space state elements.
    let fused_while = WhileOperation::new().with_iteration_bound(operation.iteration_bound())?;
    let mut operands = Vec::with_capacity(fused_state_count);
    operands.extend(inputs.iter().map(|input| input.primal().clone()));
    for (input, &has_tangent) in inputs.iter().zip(&element_has_tangent) {
        if has_tangent {
            operands.push(C::Operation::materialize_zero_from_residual_sources(
                context,
                input.tangent().clone(),
                std::iter::once(input.primal()),
            )?);
        }
    }
    let fused_regions = [Arc::new(fused_condition), fused_body];
    let outputs = context.bind(C::Operation::from(fused_while), CalleeRegionDriver::new(&fused_regions), &operands)?;
    check_count!("output", outputs, fused_state_count, ProgramError);
    let (primal_outputs, tangent_outputs) = outputs.split_at(state_count);
    let mut tangent_outputs = tangent_outputs.iter().cloned();
    Ok(primal_outputs
        .iter()
        .cloned()
        .zip(element_has_tangent)
        .map(|(primal, has_tangent)| {
            if has_tangent {
                DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
            } else {
                DifferentiationDual::new_with_zero_tangent(primal)
            }
        })
        .collect::<Result<Vec<_>, _>>()?)
}

/// Runs a `while` loop's forward-mode rule directly at concrete duals for an
/// [eager](Context::is_eager) context, returning `None` when the loop's predicate does
/// not concretize to one scalar Boolean (e.g., a batched per-item predicate) and the caller must therefore fall back
/// to the type family's staged strategy.
///
/// Each iteration evaluates the condition on the concrete primal carries and interprets the body once over the
/// current dual carries. Body instructions re-enter their JVP rules through the differentiation driver, so nested
/// data-dependent `while` operations recurse through this same eager path without a program-level unroll pre-pass.
/// Data-dependent trip counts therefore need no iteration bound — this is the analogue of
/// [JAX's `jvp` through an eagerly executed loop](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) — while a
/// semantic [`iteration_bound`](WhileOperation::with_iteration_bound) truncates the loop once it is reached, matching
/// the bounded-`while` truncation semantics. Each body effect executes exactly once per loop iteration.
///
/// This rule derives no program from the body region and therefore consults no per-region transform cache: it
/// interprets the body once per iteration at the concrete duals reached by the loop, so its work is a function of the
/// runtime values rather than of the region's contents alone.
fn jvp_while_eagerly<C, D: DifferentiationDriver<C>>(
    operation: &WhileOperation<C::Type>,
    condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    context: &C,
    driver: &D,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Option<Vec<DifferentiationDual<C::Value>>>, ProgramError>
where
    C: Context + Zero<C::Value>,
    C::Type: DifferentiableType + WhileTypeSemantics,
    C::Value: Concretizable<bool>,
    C::Operation: ZeroOperationProvider<C::Type>,
    for<'operation> &'operation WhileOperation<C::Type>: TryFrom<&'operation C::Operation>,
{
    let state_count = inputs.len();
    let mut primal_carries = Vec::with_capacity(state_count);
    let mut tangent_carries = Vec::with_capacity(state_count);
    for input in inputs {
        primal_carries.push(input.primal().clone());
        tangent_carries.push(input.tangent().clone());
    }

    let mut completed_iterations = 0;
    loop {
        if operation.iteration_bound().is_some_and(|bound| completed_iterations >= bound) {
            break;
        }

        // Concretize the condition on the current concrete primal carries to decide whether another iteration runs.
        let mut condition_outputs = condition.interpret_in_context(context, primal_carries.clone())?;
        check_count!("output", condition_outputs, 1, ProgramError);
        let predicate = match condition_outputs.remove(0).concretize() {
            Ok(predicate) => predicate,
            // The predicate does not concretize to one scalar Boolean — e.g., a batched per-item predicate, whose
            // items stop on different iterations, has no single trip decision. Report the loop as non-concretizable
            // so the caller falls back to the type family's staged strategy; nothing has been advanced yet on the
            // first iteration. The predicate type is loop-invariant, so a later-iteration failure cannot occur once
            // the first concretization succeeds, and any such error is surfaced.
            Err(_) if completed_iterations == 0 => return Ok(None),
            Err(error) => return Err(error),
        };
        if !predicate {
            break;
        }

        // Advance one iteration by replaying the body directly over dual values. This is deliberately not routed
        // through a program-level unroll pre-pass: such a rewrite interprets the body to discover concrete nested-loop
        // trip counts, after which replaying its fused JVP would execute primal effects a second time. Direct dual
        // interpretation lets a nested while recurse through this eager rule and executes every primal operation once.
        let input_duals = primal_carries
            .drain(..)
            .zip(tangent_carries.drain(..))
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()?;
        let body_region = body.entry_region_ref();
        let output_duals = body_region.interpret_with::<_, ProgramError, _, _>(
            input_duals,
            |_, constant| Ok(DifferentiationDual::new_with_zero_tangent(context.lift(constant.clone())?)?),
            |instruction, input_duals| {
                let programs = instruction
                    .regions()
                    .iter()
                    .map(|region| body_region.with_id(*region).map(RegionRef::to_program))
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                let output_duals = if !input_duals.is_empty() && input_duals.iter().all(|dual| dual.tangent().is_zero())
                {
                    let primal_inputs = input_duals.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
                    context
                        .bind(instruction.operation().clone(), programs, primal_inputs.as_slice())?
                        .into_iter()
                        .map(DifferentiationDual::new_with_zero_tangent)
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    driver.jvp_operation(instruction.operation(), programs, context, input_duals)?
                };
                check_count!("output", output_duals, instruction.outputs().len(), ProgramError);
                Ok(output_duals)
            },
        )?;
        check_count!("output", output_duals, state_count, ProgramError);
        for dual in output_duals {
            let (primal, tangent) = dual.into_parts();
            primal_carries.push(primal);
            tangent_carries.push(tangent);
        }
        completed_iterations += 1;
    }

    Ok(Some(
        primal_carries
            .into_iter()
            .zip(tangent_carries)
            .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}

/// Builds the augmented condition and body programs of the bounded staged while loop (see the [`WhileOperation`] JVP
/// rule below). The body replays the original body through its tracing context before adding the augmented-state
/// operations, while the condition structurally relocates the original condition into a builder with the extended
/// input boundary.
///
/// The augmented loop state is `[original_state..., counter (i64 scalar), residual_stacks..., mask_stack]`:
///
///   - The body runs the residual-extended primal body (which outputs `[next_state..., residuals...]`) on the
///     original state slots, then *stores* instead of returning each per-iteration residual: residual `k` is
///     broadcast to `[1, …]` and written into stack `k` at batch index `counter` via `dynamic_update_slice`, a scalar
///     Boolean `one` (true) is written into the Boolean `[bound]` mask stack at batch index `counter`, and the counter
///     advances by an i64 `one`. Because the enclosing while keeps `iteration_bound = bound`, the counter is always
///     strictly below `bound` whenever the body runs, so the writes can never clamp.
///   - The condition is the original loop condition extended with ignored extra-state inputs.
///
/// Returns the extended condition, the augmented body, and the `[bound, …]` residual stack types.
fn build_bounded_while_programs<V, O, A>(
    condition: &Program<V, O, Vec<V>, Vec<V>>,
    primal_body: &Program<V, O, Vec<V>, Vec<V>>,
    residual_types: &[V::Type],
    bound: usize,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>, Vec<V::Type>), ProgramError>
where
    V: Value<Type: WhileResidualStackType>,
    O: WhileResidualStackOperation<V::Type, A>,
{
    let state_count = condition.input_types().len();
    let counter_type = V::Type::from_array_type(ArrayType::scalar(DataType::I64));
    let boolean_scalar_type = V::Type::from_array_type(ArrayType::scalar(DataType::Boolean));
    let mask_stack_type = V::Type::from_array_type(stacked_scan_type(boolean_scalar_type.array_type()?, bound));
    let storage_types = residual_types
        .iter()
        .map(TemporalResidualType::temporal_storage_type)
        .collect::<Result<Vec<_>, _>>()?;
    for storage_type in &storage_types {
        let storage_type = storage_type.array_type()?;
        if storage_type.static_shape().is_none() {
            return Err(TypeError::invalid(format!(
                "jvp of a bounded {WHILE_OPERATION_NAME} loop requires statically shaped residual storage but got {storage_type}",
            ))
            .into());
        }
    }
    let stack_types = storage_types
        .iter()
        .map(|storage_type| {
            storage_type
                .array_type()
                .map(|storage_type| V::Type::from_array_type(stacked_scan_type(storage_type, bound)))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let body_input_types = primal_body
        .input_types()
        .into_iter()
        .chain(std::iter::once(counter_type.clone()))
        .chain(stack_types.iter().cloned())
        .chain(std::iter::once(mask_stack_type.clone()))
        .collect::<Vec<_>>();
    let body = TracingContext::<V, O>::trace(
        |inputs| {
            let context = inputs[0].context().clone();
            let original_input_count = primal_body.input_ids().len();
            let mut extra_inputs = inputs[original_input_count..].iter();
            let counter_input = extra_inputs.next().cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "bounded {WHILE_OPERATION_NAME} body adapter is missing the counter input"
                ))
            })?;
            let stack_inputs = extra_inputs.by_ref().take(stack_types.len()).cloned().collect::<Vec<_>>();
            check_count!("input", stack_inputs, stack_types.len(), ProgramError);
            let mask_input = extra_inputs.next().cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "bounded {WHILE_OPERATION_NAME} body adapter is missing the mask input"
                ))
            })?;
            let mut body_outputs =
                primal_body.interpret_in_context(&context, inputs[..original_input_count].to_vec())?;
            let residual_outputs = body_outputs.split_off(state_count);
            check_count!("output", residual_outputs, residual_types.len(), ProgramError);
            let zero_index = if storage_types
                .iter()
                .any(|storage_type| storage_type.array_type().map(ArrayType::rank).unwrap_or_default() > 0)
            {
                let mut outputs = context.bind(O::residual_stack_zero(counter_type.clone()), Vec::new(), &[])?;
                check_count!("output", outputs, 1, ProgramError);
                Some(outputs.remove(0))
            } else {
                None
            };
            let mut next_stacks = Vec::with_capacity(stack_types.len());
            for (((residual_output, residual_type), storage_type), stack_input) in
                residual_outputs.iter().zip(residual_types).zip(storage_types.iter()).zip(stack_inputs.iter())
            {
                let storage_output = if let Some(operation) = O::residual_to_storage(residual_type)? {
                    let mut outputs = context.bind(operation, Vec::new(), std::slice::from_ref(residual_output))?;
                    check_count!("output", outputs, 1, ProgramError);
                    outputs.remove(0)
                } else {
                    residual_output.clone()
                };
                let storage_array_type = storage_type.array_type()?;
                let batch_item_type = stacked_scan_type(storage_array_type, 1);
                let output_axes = (1..=storage_array_type.rank()).collect::<Vec<_>>();
                let mut expanded = context.bind(
                    O::residual_stack_broadcast(batch_item_type, output_axes),
                    Vec::new(),
                    std::slice::from_ref(&storage_output),
                )?;
                check_count!("output", expanded, 1, ProgramError);
                let mut start_indices = vec![counter_input.clone()];
                if let Some(zero_index) = &zero_index {
                    start_indices.extend((0..storage_array_type.rank()).map(|_| zero_index.clone()));
                }
                let mut next_stack = context.bind(
                    O::residual_stack_update(),
                    Vec::new(),
                    &std::iter::once(stack_input.clone())
                        .chain(expanded.drain(..))
                        .chain(start_indices)
                        .collect::<Vec<_>>(),
                )?;
                check_count!("output", next_stack, 1, ProgramError);
                next_stacks.push(next_stack.remove(0));
            }
            let true_scalar = context.bind(O::residual_stack_one(boolean_scalar_type.clone()), Vec::new(), &[])?;
            check_count!("output", true_scalar, 1, ProgramError);
            let true_item_type = stacked_scan_type(boolean_scalar_type.array_type()?, 1);
            let mut true_item = context.bind(
                O::residual_stack_broadcast(true_item_type, Vec::new()),
                Vec::new(),
                std::slice::from_ref(&true_scalar[0]),
            )?;
            check_count!("output", true_item, 1, ProgramError);
            let mut next_mask = context.bind(
                O::residual_stack_update(),
                Vec::new(),
                &[mask_input.clone(), true_item.remove(0), counter_input.clone()],
            )?;
            check_count!("output", next_mask, 1, ProgramError);
            let mut one_i64 = context.bind(O::residual_stack_one(counter_type.clone()), Vec::new(), &[])?;
            check_count!("output", one_i64, 1, ProgramError);
            let mut next_counter =
                context.bind(O::residual_stack_add(), Vec::new(), &[counter_input, one_i64.remove(0)])?;
            check_count!("output", next_counter, 1, ProgramError);
            body_outputs.push(next_counter.remove(0));
            body_outputs.extend(next_stacks);
            body_outputs.push(next_mask.remove(0));
            Ok(body_outputs)
        },
        body_input_types,
    )?
    .1;

    // Condition: the original loop condition extended with ignored extra-state inputs.
    let condition_input_types = condition
        .input_types()
        .into_iter()
        .chain(std::iter::once(counter_type))
        .chain(stack_types.iter().cloned())
        .chain(std::iter::once(mask_stack_type))
        .collect::<Vec<_>>();
    let condition_input_count = condition_input_types.len();
    let mut condition_builder = ProgramBuilder::new();
    let condition_inputs = condition_input_types
        .into_iter()
        .map(|r#type| condition_builder.add_input(r#type))
        .collect::<Vec<_>>();
    let condition_outputs = condition_builder.splice_program(condition, &condition_inputs[..state_count])?;
    let condition_output_count = condition_outputs.len();
    let extended_condition = condition_builder.build(
        condition_outputs,
        vec![Placeholder; condition_input_count],
        vec![Placeholder; condition_output_count],
    )?;
    Ok((extended_condition, body, stack_types))
}

/// Rewrites a while loop's condition and body into the scalar-predicate *masked form* over the augmented state
/// `[state..., active_mask]`: the masked condition reduces the mask with a Boolean `any` over every predicate axis,
/// and the masked body replays the original body for candidate updates, selects per array state element between the
/// candidate and the carried state under the (broadcast) mask, recomputes the per-item predicate on the new state, and
/// ANDs it into the mask. A first-class dimension carry is loop-invariant under a batched predicate, so it carries no
/// selection and its candidate is forwarded directly (refer to the documentation of [`WhileTypeSemantics`]). The
/// bounded while forward-mode rule uses this normal form for batched-predicate loops, whose counter- and
/// stack-augmented differentiation state is not predicate-prefixed and therefore needs the loop's masking made
/// explicit as program data hanging off a scalar predicate.
fn masked_while_programs<V, O, A>(
    condition: &Program<V, O, Vec<V>, Vec<V>>,
    body: &Program<V, O, Vec<V>, Vec<V>>,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>), ProgramError>
where
    V: Value<Type: WhileResidualStackType>,
    O: WhileResidualStackOperation<V::Type, A>,
{
    let state_types = body.input_types();
    let state_count = state_types.len();
    let mask_type = condition.output_types()[0].clone();
    let mask_axes = (0..mask_type.array_type()?.rank()).collect::<Vec<_>>();
    let mut masked_state_types = state_types.clone();
    masked_state_types.push(mask_type.clone());

    // Masked condition: `any(active_mask)` over every predicate axis, ignoring the state inputs.
    let (_, masked_condition) = TracingContext::<V, O>::trace(
        |inputs| {
            inputs[0]
                .context()
                .bind(O::mask_reduce_any(mask_axes.clone()), Vec::new(), &[inputs[state_count].clone()])
        },
        masked_state_types.clone(),
    )?;

    // Masked body: candidate updates from the replayed body, per-element masked selection between the candidate
    // update and the carried state, the per-item predicate recomputed on the new state, and the mask narrowed via
    // AND.
    let (_, masked_body) = TracingContext::<V, O>::trace(
        |inputs| {
            let (mask, state) = inputs.split_last().unwrap();
            let trace_context = mask.context().clone();
            let candidates = body.interpret_in_context(&trace_context, state.to_vec())?;
            check_count!("output", candidates, state_count, ProgramError);
            let mut next_state = Vec::with_capacity(state_count);
            for ((candidate, carried), state_type) in candidates.iter().zip(state).zip(state_types.iter()) {
                // A first-class dimension carry is loop-invariant under a batched predicate (the contract documented
                // on `WhileTypeSemantics`), so masking it is the identity and the body's candidate result is threaded
                // on directly. This matches eager interpretation, whose `mask_select` returns equal dimension carries
                // unchanged, and the XLA lowering's batched-predicate masking.
                let Some(state_array_type) = state_type.maskable_array_type() else {
                    next_state.push(candidate.clone());
                    continue;
                };
                // The mask broadcasts to each state element's shape so the selection is per predicate item; a state
                // element already shaped like the mask reuses it directly.
                let element_mask_type = ArrayType::new(DataType::Boolean, state_array_type.shape().clone());
                let element_mask = if element_mask_type == *mask_type.array_type()? {
                    mask.clone()
                } else {
                    trace_context
                        .bind(
                            O::residual_stack_broadcast(element_mask_type, mask_axes.clone()),
                            Vec::new(),
                            std::slice::from_ref(mask),
                        )?
                        .remove(0)
                };
                next_state.push(
                    trace_context
                        .bind(
                            O::residual_stack_select(),
                            Vec::new(),
                            &[element_mask, candidate.clone(), carried.clone()],
                        )?
                        .remove(0),
                );
            }
            let next_predicate = condition.interpret_in_context(&trace_context, next_state.clone())?;
            check_count!("output", next_predicate, 1, ProgramError);
            let mut outputs = next_state;
            outputs.extend(trace_context.bind(
                O::mask_and(),
                Vec::new(),
                &[mask.clone(), next_predicate.into_iter().next().unwrap()],
            )?);
            Ok(outputs)
        },
        masked_state_types,
    )?;
    Ok((masked_condition, masked_body))
}

impl<V: Value, O> TransposableOperation<V, O> for WhileOperation<V::Type>
where
    V::Type: WhileTypeSemantics,
    O: Operation<Type = V::Type>,
{
    /// Rejects transposition. This rule is only reachable for *unbounded* staged while loops — the doubled-state
    /// linear loop staged by the [`WhileOperation`] JVP rule, which recomputes primal state *forward* through
    /// the iterations, so transposing it would have to run that recomputation backwards, which a while loop cannot
    /// express. Two paths avoid it entirely: eager domains execute the loop directly over concrete duals and record a
    /// straight-line pushforward during linearization, and bounded loops ([`WhileOperation::with_iteration_bound`])
    /// never stage a linear `while` — their tangent side is a masked linear scan whose transpose is total, so reverse
    /// mode through staged bounded loops flows through the scan transpose without reaching this rule.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("{WHILE_OPERATION_NAME} does not support transposition (reverse-mode differentiation through staged unbounded \
                      {WHILE_OPERATION_NAME} loops is not supported; eager differentiation executes concrete duals, and loops built \
                      with `with_iteration_bound` stage a transposable masked scan)"),
        }
        .into())
    }
}

/// Value-level predicate capability backing [`WhileOperation`]'s masked loop semantics.
///
/// A while condition may produce a *batched* Boolean predicate — one termination decision per leading-axes item, with
/// the predicate shape a prefix of every state shape (see [`WhileTypeSemantics`]). Interpretation then continues while
/// [`any_true`](Self::any_true) holds and updates each state element with [`mask_select`](Self::mask_select), so items
/// whose predicate is false keep their carried state while active items take the body's candidate update. A frozen
/// item's predicate is recomputed from its frozen state, so a finished item can never rejoin the loop.
///
/// The default implementations are the scalar-predicate semantics, expressed through [`Concretizable<bool>`]: the predicate's
/// own truth decides continuation, and a true predicate takes the candidate wholesale. Value types with genuinely
/// batched payloads (e.g. [`Array`](crate::arrays::Array)) override both methods with per-item semantics, and
/// symbolic values (tracers and capture references) inherit the defaults, which surface [`Concretizable::concretize`]'s
/// concretization errors — a staged while is consumed by staging and lowering rather than by this eager loop.
pub trait WhilePredicate: Concretizable<bool> + Clone + Sized {
    /// Returns `true` when any element of this Boolean predicate is true — the loop-continuation decision.
    fn any_true(&self) -> Result<bool, ProgramError> {
        self.concretize()
    }

    /// Selects between `on_true` and `on_false` per element under this Boolean predicate, broadcasting the predicate
    /// against their shape along its leading (prefix) axes.
    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        Ok(if self.concretize()? { on_true.clone() } else { on_false.clone() })
    }
}

// Symbolic values inherit the scalar `WhilePredicate` defaults, which surface `Concretizable::concretize`'s
// concretization errors: none of these carry a concrete predicate payload, and staged whiles are consumed by staging
// and lowering rather than by the eager masked loop.
impl<C: Context> WhilePredicate for Tracer<C> {}

impl WhilePredicate for CaptureReference<ArrayType> {}

impl<C: Context<Type = ArrayType>> WhilePredicate for BatchingTracer<C, ArrayBatching> where
    C::Value: Concretizable<bool>
{
}

impl<C: Context<Type: DifferentiableType>> WhilePredicate for DifferentiationTracer<C> where
    C::Value: Concretizable<bool>
{
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow as StdCow;
    use std::cell::Cell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameter;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, Dimension, DimensionBounds, DimensionType,
        DimensionVariable, Shape,
    };
    use crate::batching::batch;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        Differentiate, ForwardModeDifferentiate, LinearizationTracer, ReverseModeDifferentiate, differentiate_at,
    };
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::one::One;
    use crate::operations::constants::one_like::{OneLike, OneLikeOperation};
    use crate::operations::constants::zero_like::{ZeroLike, ZeroLikeOperation};
    use crate::operations::debugging::PrintOperation;
    use crate::operations::math::add::AddOperation;
    use crate::operations::math::div::DivOperation;
    use crate::operations::math::mul::MulOperation;
    use crate::operations::math::sub::{SUB_OPERATION_NAME, SubOperation};
    use crate::parameters::Parameter;
    use crate::programs::{Effects, ReferenceType};
    use crate::tests::CountingBatchingDriver;
    use crate::tracing::DomainTracingContext;

    use super::*;

    /// Builds a condition program that maps a scalar `f64` state to the scalar Boolean predicate `state > 0`.
    fn greater_than_zero_condition() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![state]).unwrap()[0];
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![state, zero])
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a body program that maps a scalar `f64` state to `state - 1`.
    fn subtract_one_body() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![state]).unwrap()[0];
        let next_state = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one]).unwrap()[0];
        builder.build(vec![next_state], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns the [`RegionInterface`] of the provided flat region program.
    fn region_interface(
        program: &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>,
    ) -> RegionInterface<ArrayType> {
        program.interface()
    }

    #[test]
    fn test_while_composite_type_contract() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = ArrayIrType::Dimension(DimensionType::new(extent.clone()));
        let array_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())])));
        let state_types = vec![dimension_type.clone(), array_type.clone()];
        let condition_interface = RegionInterface::new(
            state_types.clone(),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))],
            Effects::PURE,
        );
        let body_interface = RegionInterface::new(state_types.clone(), state_types.clone(), Effects::PURE);
        let operation = WhileOperation::new();

        assert_eq!(
            operation.infer_output_types(state_types.as_slice(), &[condition_interface, body_interface.clone()]),
            Ok(state_types.clone()),
        );

        // A batched predicate is legal alongside first-class dimension state: the dimension member carries no shape, so
        // the prefix requirement applies to the array member only, and the loop-invariance requirement on a dimension
        // carry is enforced dynamically during interpretation rather than by this contract.
        let batched_condition_interface = RegionInterface::new(
            state_types.clone(),
            vec![ArrayIrType::Array(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(extent)])))],
            Effects::PURE,
        );
        assert_eq!(
            operation
                .infer_output_types(state_types.as_slice(), &[batched_condition_interface, body_interface.clone()]),
            Ok(state_types.clone()),
        );

        // Reference state has no per-item mask semantics. Reject it at this local type boundary when the predicate is
        // batched rather than silently treating it like a loop-invariant first-class dimension.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(2)]),
        )));
        let reference_condition_interface = RegionInterface::new(
            vec![reference_type.clone()],
            vec![ArrayIrType::Array(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])))],
            Effects::PURE,
        );
        let reference_body_interface =
            RegionInterface::new(vec![reference_type.clone()], vec![reference_type.clone()], Effects::PURE);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&reference_type),
                &[reference_condition_interface, reference_body_interface],
            ),
            Err(TypeError::invalid(
                "`while` condition with a batched predicate requires references to be discharged, but state type is \
                 `ref<f32[2]>`",
            )),
        );

        // The prefix requirement still holds for every array member under a batched predicate.
        let mismatched_condition_interface = RegionInterface::new(
            state_types.clone(),
            vec![ArrayIrType::Array(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])))],
            Effects::PURE,
        );
        assert_eq!(
            operation.infer_output_types(state_types.as_slice(), &[mismatched_condition_interface, body_interface]),
            Err(TypeError::invalid(
                "`while` condition predicate shape must be a prefix of every array state shape, but predicate bool[2] \
                 is not a prefix of state f32[extent]"
                    .to_string(),
            )),
        );

        // A body-local gateway may define a fresh dimension, but that identity cannot replace the declared carry
        // identity on each iteration. Bounds widening is deliberately not inferred for loop-carried state.
        let carry = DimensionVariable::new("carry", DimensionBounds::positive(Some(8)).unwrap());
        let next = DimensionVariable::new("next", DimensionBounds::positive(Some(8)).unwrap());
        let carry_type = ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(carry)])));
        let next_type = ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(next)])));
        let condition_interface = RegionInterface::new(
            vec![carry_type.clone()],
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))],
            Effects::PURE,
        );
        let shape_varying_body = RegionInterface::new(vec![carry_type.clone()], vec![next_type], Effects::PURE);
        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&carry_type), &[condition_interface, shape_varying_body],),
            Err(TypeError::invalid(
                "while body output type signature mismatch: expected [f32[carry]] but got [f32[next]]".to_string(),
            )),
        );
    }

    #[test]
    fn test_while() {
        let state_type = ArrayType::scalar(DataType::F64);
        let operation = WhileOperation::new();
        let condition = greater_than_zero_condition();
        let body = subtract_one_body();
        let interfaces = vec![region_interface(&condition), region_interface(&body)];

        // Operation identity, declared region slots, reference flow, and payload-free rendering.
        assert_eq!(operation.name(), WHILE_OPERATION_NAME);
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("condition"), RegionSlot::computation("body")]);
        assert_eq!(operation.input_region_provenance(0, 0), Some(0));
        assert_eq!(operation.input_region_provenance(1, 0), Some(0));
        assert_eq!(
            operation.output_region_provenance(0),
            vec![OutputRegionProvenance { region_index: 1, output_index: 0 }],
        );
        assert_eq!(operation.reference_output_identity_input(0), Some(0));
        assert!(operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::Read));
        assert!(!operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::Write));
        assert!(!operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::Accumulate));
        assert!(!operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::Consume));
        assert!(operation.allows_reference_access_through_region_input(1, ReferenceAccessMode::Write));
        assert_eq!(format!("{operation}"), "while");

        // Type inference validates the region interfaces and the input types, and returns the state types.
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&state_type), interfaces.as_slice()),
            Ok(vec![state_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&state_type), &[]),
            Err(TypeError::invalid("expected 2 regions but got 0")),
        );
        assert_eq!(
            operation.infer_output_types(&[], interfaces.as_slice()),
            Err(TypeError::invalid("expected 1 input but got 0".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid(
                "while input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string()
            )),
        );

        // Inference rejects mismatched condition/body state signatures, non-Boolean condition outputs,
        // multi-output conditions, and body outputs that do not match the state signature.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])));
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![state]).unwrap()[0];
        let vector_body = builder.build(vec![zero], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&state_type),
                &[region_interface(&condition), region_interface(&vector_body)],
            ),
            Err(TypeError::invalid(
                "while condition/body input type signature mismatch: expected [f64[2]] but got [f64[]]".to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&state_type),
                &[region_interface(&subtract_one_body()), region_interface(&body)],
            ),
            Err(TypeError::invalid("`while` condition output type must be a Boolean array, but got f64[]".to_string())),
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let multi_output_condition =
            builder.build(vec![state, state], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap();
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&state_type),
                &[region_interface(&multi_output_condition), region_interface(&body)],
            ),
            Err(TypeError::invalid(
                "while condition must return exactly one predicate leaf but returned 2".to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&state_type),
                &[region_interface(&condition), region_interface(&greater_than_zero_condition())],
            ),
            Err(TypeError::invalid(
                "while body output type signature mismatch: expected [f64[]] but got [bool[]]".to_string()
            )),
        );

        // The semantic iteration bound defaults to absent, must be at least one, may be cleared with `None`, and is
        // reported by the accessor.
        assert_eq!(operation.iteration_bound(), None);
        let bounded = WhileOperation::new().with_iteration_bound(2).unwrap();
        assert_eq!(bounded.iteration_bound(), Some(2));
        assert_eq!(bounded.with_iteration_bound(None).unwrap().iteration_bound(), None);
        assert_eq!(
            WhileOperation::<ArrayType>::new().with_iteration_bound(0).map(|_| ()),
            Err(ProgramError::Type(TypeError::invalid("while iteration bound must be at least 1".to_string()))),
        );

        // The bound renders as an `iteration_bound=` field on the operation itself.
        assert_eq!(format!("{bounded}"), "while [iteration_bound=2]");

        // Eager binding iterates the body until the condition produces false.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let outputs = context.bind(operation, [condition.clone(), body.clone()], &[Array::scalar(3.0)]).unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![0.0]);
        let outputs = context.bind(operation, vec![condition.clone(), body.clone()], &[Array::scalar(-1.0)]).unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![-1.0]);

        // A bounded while runs at most `bound` iterations by definition: the subtract-one loop at 5 would run five
        // iterations on its own, but the bound of 2 truncates it at 3 even though the condition is still true.
        let outputs = context.bind(bounded, vec![condition.clone(), body.clone()], &[Array::scalar(5.0)]).unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![3.0]);
        // A loop that exits before reaching the bound is unaffected by it.
        let outputs = context.bind(bounded, vec![condition.clone(), body.clone()], &[Array::scalar(1.0)]).unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![0.0]);

        // Staging imports the condition and body programs as attached regions of the staged instruction instead of
        // trying to drive the loop with a concrete predicate.
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = context.builder().clone();
        let staged_state = context.input(state_type.clone());
        let outputs = context
            .stage_operation(operation, [condition.clone(), body.clone()], std::slice::from_ref(&staged_state))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let builder = builder.borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), ArrayOperation::While(_)));
        assert_eq!(builder.instructions()[0].regions().len(), 2);
        assert_eq!(builder.instructions()[0].inputs(), &[staged_state.atom_id().unwrap()]);
        assert_eq!(outputs[0].atom_id(), Ok(builder.instructions()[0].outputs()[0]));

        // Program rendering shows the attached condition and body regions at the instruction with their declared
        // slot names, with the iteration bound rendered on the operation itself.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let program_state = builder.add_input(state_type);
        let program_output = builder
            .add_instruction(ArrayOperation::While(bounded), vec![condition_region, body_region], vec![program_state])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![program_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = while [iteration_bound=2] %0 [
                    condition={
                        lambda %0:f64[] .
                        let %1:f64[] = zero_like %0
                            %2:bool[] = compare [direction=GreaterThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = one_like %0
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_unbounded_while_eager_linearization_and_transposition_follow_the_executed_iterations() {
        type TestContext = EagerContext<Array, ArrayOperation<Array>>;
        type TestTracer = LinearizationTracer<TestContext>;

        let function = |state: TestTracer| {
            let mut outputs = state.context().bind(
                ArrayOperation::While(WhileOperation::new()),
                vec![greater_than_zero_condition(), subtract_one_body()],
                &[state.clone()],
            )?;
            Ok(outputs.remove(0))
        };
        let (output, pushforward) = differentiate_at(Array::scalar(3.5)).linearize(function).unwrap();
        assert_eq!(output, Array::scalar(-0.5));
        assert_eq!(pushforward.apply(Array::scalar(2.0)), Ok(Array::scalar(2.0)));

        let (output, pullback) = differentiate_at(Array::scalar(3.5)).vjp(function).unwrap();
        assert_eq!(output, Array::scalar(-0.5));
        assert_eq!(pullback.apply(Array::scalar(2.0)), Ok(Array::scalar(2.0)));
    }

    /// With a *loop-invariant known* state element, a `while` partially evaluates by folding that element's value into
    /// the condition and body: the residual while keeps the same state set (so its output arity is preserved) but its
    /// body shrinks because every subcomputation that depended only on the known element collapses to a constant.
    ///
    /// The loop carries `[counter, acc, k]` and runs while `counter > 0`. Its body computes `ksq = k * k`,
    /// `next_acc = acc + ksq`, `next_counter = counter - 1`, and returns `[next_counter, next_acc, k]`: `counter` is a
    /// down-counter, `acc` accumulates `k * k` each iteration, and `k` is forwarded unchanged (loop-invariant). With
    /// `k` known (`3`) and `counter` and `acc` unknown, the `k` state element is loop-invariant-known (its next-state
    /// equals its init), so `ksq` folds to the constant `9` and the body shrinks from four instructions to three, with
    /// the final `k` folded to the constant `3` inside the residual while body. A bound terminates the loop
    /// deterministically. Interpreting the residual program reproduces the original while over the same inputs.
    #[test]
    fn test_while_partial_evaluation_folds_loop_invariant_known_state() {
        let scalar = || ArrayType::scalar(DataType::F64);

        // Condition `[counter, acc, k] -> [counter > 0]` (reads only the counter).
        let condition = || {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            let _acc = builder.add_input(scalar());
            let _k = builder.add_input(scalar());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![counter]).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::GreaterThan),
                    Vec::new(),
                    vec![counter, zero],
                )
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };

        // Body `[counter, acc, k] -> [counter - 1, acc + k * k, k]`.
        let body = || {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            let acc = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter]).unwrap()[0];
            let next_counter = builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one]).unwrap()[0];
            let ksq = builder.add_instruction(MulOperation::new(), Vec::new(), vec![k, k]).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation::new(), Vec::new(), vec![acc, ksq]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(
                    vec![next_counter, next_acc, k],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        // Flat program over `[counter_init, acc_init, k_init]` staging the bounded while; its outputs are the final
        // `[counter, acc, k]` state.
        let operation = WhileOperation::new().with_iteration_bound(8).unwrap();
        let original_body_instructions = body().instructions().len();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let condition_region = builder.import_region(condition().entry_region_ref());
        let body_region = builder.import_region(body().entry_region_ref());
        let counter_init = builder.add_input(scalar());
        let acc_init = builder.add_input(scalar());
        let k_init = builder.add_input(scalar());
        let outputs = builder
            .add_instruction(
                ArrayOperation::While(operation),
                vec![condition_region, body_region],
                vec![counter_init, acc_init, k_init],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let knowledge = vec![
            PartialValue::Unknown(scalar()),
            PartialValue::Unknown(scalar()),
            PartialValue::Known(Array::scalar(3.0)),
        ];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        // The while instruction residualizes (its inputs are not all known), so every state output is produced by
        // the residual program — even the loop-invariant `k`, whose residual while body folds it to the constant 3.
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(_)));
        assert!(matches!(&evaluation.outputs[1], PartialEvaluationOutput::Unknown(_)));
        assert!(matches!(&evaluation.outputs[2], PartialEvaluationOutput::Unknown(_)));

        // The residual program's only instruction is the rewritten while, carrying its rewritten condition and body
        // as attached regions.
        assert_eq!(evaluation.program.instructions().len(), 1);
        let residual_instruction = &evaluation.program.instructions()[0];
        let ArrayOperation::While(residual_while) = residual_instruction.operation() else {
            panic!("expected the residual program to contain a rewritten while");
        };

        // The state set is preserved (so output arity matches) and the iteration bound is carried over, but the body
        // shrank: `k * k` folded to a constant, so the body drops from four instructions to three.
        assert_eq!(residual_instruction.regions().len(), 2);
        let residual_body = evaluation.program.region_ref(residual_instruction.regions()[1]).unwrap().to_program();
        assert_eq!(residual_body.input_types().len(), 3);
        assert_eq!(residual_while.iteration_bound(), Some(8));
        assert!(residual_body.instructions().len() < original_body_instructions);
        assert_eq!(residual_body.instructions().len(), 3);

        // Correctness: interpreting the residual program reproduces the original program on the same concrete inputs.
        let runtime = |counter: f64, acc: f64| -> Vec<Array> {
            let arguments = evaluation
                .inputs
                .iter()
                .map(|residual_input| match residual_input {
                    PartialEvaluationInput::Known(value) => value.clone(),
                    PartialEvaluationInput::Unknown(index) => match index {
                        0 => Array::scalar(counter),
                        _ => Array::scalar(acc),
                    },
                })
                .collect::<Vec<_>>();
            let residual_outputs = evaluation.program.interpret(arguments).unwrap();
            evaluation
                .outputs
                .iter()
                .map(|output| match output {
                    PartialEvaluationOutput::Known(value) => value.clone(),
                    PartialEvaluationOutput::Unknown(index) => residual_outputs[*index].clone(),
                })
                .collect()
        };
        let original = |counter: f64, acc: f64, k: f64| {
            program.interpret(vec![Array::scalar(counter), Array::scalar(acc), Array::scalar(k)]).unwrap()
        };

        let reassembled = runtime(4.0, 1.0);
        let expected = original(4.0, 1.0, 3.0);
        assert_eq!(
            reassembled.iter().map(|value| value.to_f64s()).collect::<Vec<_>>(),
            expected.iter().map(|value| value.to_f64s()).collect::<Vec<_>>(),
        );
        // The loop runs four times (counter `4 -> 0`): `counter` lands at `0`, `acc` threads
        // `1 -> 1 + 9 -> 19 -> 28 -> 37`, and the loop-invariant `k` final state stays `3`.
        assert_eq!(reassembled[0].to_f64s(), vec![0.0]);
        assert_eq!(reassembled[1].to_f64s(), vec![37.0]);
        assert_eq!(reassembled[2].to_f64s(), vec![3.0]);
    }

    /// A loop whose invariance probe fails (here an integer division by a known zero state element in a body that
    /// runtime interpretation may never enter, because the condition can be false on entry) keeps the loop whole
    /// instead of failing partial evaluation, so the body's error surfaces only if the loop actually runs.
    #[test]
    fn test_while_partial_evaluation_keeps_erroring_body_folds_behind_the_condition() {
        let state_type = ArrayType::scalar(DataType::I32);
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let x = builder.add_input(state_type.clone());
            let _k = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![x]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![x, zero])
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let x = builder.add_input(state_type.clone());
            let k = builder.add_input(state_type.clone());
            let one = builder.add_constant(Array::from_f64s(state_type.clone(), vec![1.0]));
            let inverse = builder.add_instruction(DivOperation::new(), Vec::new(), vec![one, k]).unwrap()[0];
            let next_x = builder.add_instruction(AddOperation::new(), Vec::new(), vec![x, inverse]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next_x, k], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let x = builder.add_input(state_type.clone());
        let k = builder.add_input(state_type.clone());
        let outputs = builder
            .add_instruction(
                ArrayOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![x, k],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs.to_vec(), vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // The known zero divisor is an invariance candidate, so probing would fold `1 / 0`; the rule must fall back
        // to residualizing the loop whole.
        let knowledge = vec![
            PartialValue::Unknown(state_type.clone()),
            PartialValue::Known(Array::from_f64s(state_type, vec![0.0])),
        ];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::While(_)));

        // Interpreting the residual program with a non-positive entry state never enters the body.
        let inputs = evaluation
            .inputs
            .iter()
            .map(|input| match input {
                PartialEvaluationInput::Unknown(_) => Array::from_f64s(ArrayType::scalar(DataType::I32), vec![-1.0]),
                PartialEvaluationInput::Known(value) => value.clone(),
            })
            .collect::<Vec<_>>();
        let outputs = evaluation.program.interpret(inputs).unwrap();
        assert_eq!(outputs[0].elements::<i32>(), Ok(vec![-1]));
        assert_eq!(outputs[1].elements::<i32>(), Ok(vec![0]));
    }

    #[test]
    fn test_while_partial_evaluation_splits_closed_known_state_from_the_residual_loop() {
        // The loop carries `[counter, acc]` and runs while `counter > 0`; its body computes
        // `next_counter = counter - 1` and `next_acc = acc + counter`. The `counter` element is *time-varying* known
        // (its value changes every iteration, so the loop-invariant rewrite cannot fold it) but *closed*: its next
        // value and the trip predicate fold from it alone. With `counter` known (`3`) and `acc` unknown, the
        // closed-knownness split runs the known counter loop on the known side — folding the final counter to the
        // known value `0` — and keeps the whole loop residual for `acc`, recomputing the counter chain inside it
        // (there is no statically shaped residual stream to feed `acc + counter` with).
        let scalar = || ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            let _acc = builder.add_input(scalar());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![counter]).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::GreaterThan),
                    Vec::new(),
                    vec![counter, zero],
                )
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            let acc = builder.add_input(scalar());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter]).unwrap()[0];
            let next_counter = builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one]).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation::new(), Vec::new(), vec![acc, counter]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(
                    vec![next_counter, next_acc],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let counter_init = builder.add_input(scalar());
        let acc_init = builder.add_input(scalar());
        let outputs = builder
            .add_instruction(
                ArrayOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![counter_init, acc_init],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let knowledge = vec![PartialValue::Known(Array::scalar(3.0)), PartialValue::Unknown(scalar())];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        // The known side ran the projected counter loop to completion (3 -> 2 -> 1 -> 0), so the final counter is a
        // *known* output, while the final accumulator stays residual.
        assert!(
            matches!(&evaluation.outputs[0], PartialEvaluationOutput::Known(value) if *value == Array::scalar(0.0))
        );
        assert!(matches!(&evaluation.outputs[1], PartialEvaluationOutput::Unknown(_)));

        // The residual program keeps the original loop whole: one while instruction whose body still carries the
        // full two-element state and all three body instructions, recomputing the known counter chain internally.
        assert_eq!(evaluation.program.instructions().len(), 1);
        let residual_instruction = &evaluation.program.instructions()[0];
        assert!(matches!(residual_instruction.operation(), ArrayOperation::While(_)));
        let residual_body = evaluation.program.region_ref(residual_instruction.regions()[1]).unwrap().to_program();
        assert_eq!(residual_body.input_types().len(), 2);
        assert_eq!(residual_body.instructions().len(), 3);

        // Interpreting the residual program reproduces the original loop's accumulator: from `acc = 10` the loop
        // adds 3, 2, and 1, so the final accumulator is 16.
        let arguments = evaluation
            .inputs
            .iter()
            .map(|residual_input| match residual_input {
                PartialEvaluationInput::Known(value) => value.clone(),
                PartialEvaluationInput::Unknown(_) => Array::scalar(10.0),
            })
            .collect::<Vec<_>>();
        let residual_outputs = evaluation.program.interpret(arguments).unwrap();
        assert_eq!(residual_outputs.last().unwrap().to_f64s(), vec![16.0]);
    }

    #[test]
    fn test_while_accepts_batched_predicate_and_interprets_with_masked_semantics() {
        // A `bool[3]` predicate over an `f64[3]` state satisfies the predicate-prefix rule, and interpretation runs
        // the masked loop: it continues while any per-item predicate is true, and items whose predicate is false
        // keep their carried state. Items [3, 1, 2] count down independently, terminating after 3, 1, and 2
        // iterations.
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![state]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![state, zero])
                .unwrap()[0];
            builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![state]).unwrap()[0];
            let next_state = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one]).unwrap()[0];
            builder.build(vec![next_state], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let operation = WhileOperation::new();
        assert_eq!(region_interface(&condition).output_types()[0].shape().rank(), 1);

        let context = crate::contexts::EagerContext::<Array, ArrayOperation<Array>>::new();
        let outputs = context
            .bind(operation, vec![condition.clone(), body.clone()], &[Array::vector(vec![3.0, 1.0, 2.0])])
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![0.0, 0.0, 0.0]);

        // The semantic iteration bound truncates the shared masked iterations: item 0 stops at 1.0 after two body
        // applications while items 1 and 2 finish on their own predicates first.
        let bounded = operation.with_iteration_bound(2).unwrap();
        let outputs = context.bind(bounded, vec![condition, body], &[Array::vector(vec![3.0, 1.0, 2.0])]).unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_while_rejects_batched_predicate_with_effects() {
        // A batched predicate keeps the loop running for still-active items after others finish, re-executing the
        // body over every item each iteration. Values are masked back for finished items, but observable effects
        // (here a `print` in the body) cannot be, so type inference rejects an effectful batched-predicate loop
        // through the attached region interfaces' declared effects. A scalar predicate imposes no such restriction
        // (the loop exits for all items at once).
        use crate::operations::debugging::PrintOperation;
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![state]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![state, zero])
                .unwrap()[0];
            builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let effectful_body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![state]).unwrap()[0];
            let next_state = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one]).unwrap()[0];
            let printed =
                builder.add_instruction(PrintOperation::new("state"), Vec::new(), vec![next_state]).unwrap()[0];
            builder.build(vec![printed], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        assert_eq!(
            WhileOperation::new().infer_output_types(
                std::slice::from_ref(&state_type),
                &[region_interface(&condition), region_interface(&effectful_body)],
            ),
            Err(TypeError::invalid(
                "`while` loop with a batched predicate must be pure because observable effects cannot be \
                          masked for finished batch items"
                    .to_string()
            )),
        );
    }

    #[test]
    fn test_while_rejects_batched_predicate_that_is_not_a_state_shape_prefix() {
        // A `bool[3]` predicate over an `f64[2]` state violates the predicate-prefix rule: item masking would be
        // ill-defined, so type inference over the attached region interfaces fails.
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            builder.add_input(state_type.clone());
            let predicate = builder
                .add_instruction(crate::operations::constants::ZeroOperation::new(predicate_type), Vec::new(), vec![])
                .unwrap()[0];
            builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(state_type.clone());
            builder.build(vec![state], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        assert_eq!(
            WhileOperation::new().infer_output_types(
                std::slice::from_ref(&state_type),
                &[region_interface(&condition), region_interface(&body)],
            ),
            Err(TypeError::invalid(
                "`while` condition predicate shape must be a prefix of every state shape, but predicate \
                          bool[3] is not a prefix of state f64[2]"
                    .to_string()
            )),
        );
    }

    #[derive(Clone, Debug, Parameter, PartialEq)]
    enum TestValue {
        Bool(bool),
        Number(f64),
    }

    impl Display for TestValue {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Bool(value) => Display::fmt(value, formatter),
                Self::Number(value) => Display::fmt(value, formatter),
            }
        }
    }

    impl Typed for TestValue {
        type Type = ArrayType;

        fn r#type(&self) -> StdCow<'_, ArrayType> {
            match self {
                Self::Bool(_) => StdCow::Owned(ArrayType::scalar(DataType::Boolean)),
                Self::Number(_) => StdCow::Owned(ArrayType::scalar(DataType::F64)),
            }
        }
    }

    impl Value for TestValue {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> EagerContext<Self> {
            EagerContext::new()
        }

        fn execution_domain(&self) -> EagerContext<Self> {
            EagerContext::new()
        }
    }

    impl ZeroLike for TestValue {
        fn zero_like(&self) -> Self {
            match self {
                Self::Bool(_) => Self::Bool(false),
                Self::Number(_) => Self::Number(0.0),
            }
        }
    }

    impl OneLike for TestValue {
        fn one_like(&self) -> Self {
            match self {
                Self::Bool(_) => Self::Bool(true),
                Self::Number(_) => Self::Number(1.0),
            }
        }
    }

    impl<O: Operation<Type = ArrayType>> Zero<TestValue> for EagerContext<TestValue, O> {
        fn zero(&self, value_type: &ArrayType) -> Result<TestValue, ProgramError> {
            match value_type.data_type() {
                DataType::Boolean => Ok(TestValue::Bool(false)),
                DataType::F64 => Ok(TestValue::Number(0.0)),
                _ => Err(crate::programs::types::TypeError::invalid(format!(
                    "test value cannot synthesize zero for {value_type}"
                ))
                .into()),
            }
        }
    }

    impl<O: Operation<Type = ArrayType>> One<TestValue> for EagerContext<TestValue, O> {
        fn one(&self, value_type: &ArrayType) -> Result<TestValue, ProgramError> {
            match value_type.data_type() {
                DataType::Boolean => Ok(TestValue::Bool(true)),
                DataType::F64 => Ok(TestValue::Number(1.0)),
                _ => Err(crate::programs::types::TypeError::invalid(format!(
                    "test value cannot synthesize one for {value_type}"
                ))
                .into()),
            }
        }
    }

    impl Concretizable<bool> for TestValue {
        fn concretize(&self) -> Result<bool, ProgramError> {
            match self {
                Self::Bool(value) => Ok(*value),
                value => Err(ProgramError::Concretization {
                    message: format!(
                        "cannot extract a concrete boolean from a value of type {}; expected bool[]",
                        value.r#type().as_ref(),
                    ),
                }),
            }
        }
    }

    // `TestValue` predicates are scalar, so the scalar `WhilePredicate` defaults apply.
    impl crate::operations::control_flow::WhilePredicate for TestValue {}

    #[derive(Clone, Debug)]
    enum TestOperation {
        Sub,
        IsPositive,
        While(WhileOperation<ArrayType>),
    }

    impl Display for TestOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.name())
        }
    }

    impl Operation for TestOperation {
        type Type = ArrayType;

        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Sub => SUB_OPERATION_NAME,
                Self::IsPositive => "is_positive",
                Self::While(while_operation) => while_operation.name(),
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::While(while_operation) => while_operation.region_slots(),
                _ => &[],
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayType],
            region_interfaces: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Sub => {
                    check_count!("input", input_types, 2, TypeError);
                    check_types!(@same, self.name(), [&input_types[..1], &input_types[1..]]);
                    Ok(vec![input_types[0].clone()])
                }
                Self::IsPositive => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![ArrayType::scalar(DataType::Boolean)])
                }
                Self::While(while_operation) => while_operation.infer_output_types(input_types, region_interfaces),
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::While(while_operation) => while_operation.render(formatter, indentation),
                _ => Display::fmt(self, formatter),
            }
        }
    }

    impl<C: Domain<Type = ArrayType, Value = TestValue>> InterpretableOperation<C> for TestOperation
    where
        C: crate::operations::constants::Constant<TestValue, TestValue>,
    {
        fn interpret<D: InterpretationDriver<C>>(
            &self,
            context: &C,
            driver: &D,
            inputs: &[TestValue],
        ) -> Result<Vec<TestValue>, ProgramError> {
            match self {
                Self::Sub => match (&inputs[0], &inputs[1]) {
                    (TestValue::Number(left), TestValue::Number(right)) => Ok(vec![TestValue::Number(left - right)]),
                    _ => Err(TypeError::invalid("sub expected numeric inputs").into()),
                },
                Self::IsPositive => match &inputs[0] {
                    TestValue::Number(value) => Ok(vec![TestValue::Bool(*value > 0.0)]),
                    _ => Err(TypeError::invalid("is_positive expected a numeric input").into()),
                },
                Self::While(while_operation) => while_operation.interpret(context, driver, inputs),
            }
        }
    }

    fn subtract_one_branch() -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(TestValue::Number(1.0));
        let output = builder.add_instruction(TestOperation::Sub, Vec::new(), vec![input, one]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_while_interprets_until_condition_is_false() {
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output = condition_builder
            .add_instruction(TestOperation::IsPositive, Vec::new(), vec![condition_input])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new();

        assert_eq!(
            EagerContext::<TestValue, TestOperation>::new().bind(
                TestOperation::While(while_operation),
                vec![condition, subtract_one_branch()],
                &[TestValue::Number(3.0)],
            ),
            Ok(vec![TestValue::Number(0.0)]),
        );
    }

    #[test]
    fn test_while_program_rendering_includes_condition_and_body() {
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_input = condition_builder.add_input(ArrayType::scalar(DataType::F64));
        let condition_output = condition_builder
            .add_instruction(TestOperation::IsPositive, Vec::new(), vec![condition_input])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![condition_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(subtract_one_branch().entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestOperation::While(while_operation), vec![condition_region, body_region], vec![input])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = while %0 [
                    condition={
                        lambda %0:f64[] .
                        let %1:bool[] = is_positive %0
                        in (%1)
                    },
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = const 1
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
    }

    /// Test array operation enum used by the while tests below.
    type TestDomainOperation = ArrayOperation<Array>;

    /// Eager interpreting domain over [`Array`] values that reports no support for primal concretization. Hybrid
    /// rules (in particular the while JVP rule) therefore take their staged, non-concretizing paths while every
    /// primal bind still computes concrete values, which lets the tests below interpret linear while bodies
    /// numerically without abstract tracers.
    #[derive(Copy, Clone, Debug)]
    struct StagedDispatchTestDomain;

    impl Domain for StagedDispatchTestDomain {
        type Type = ArrayType;
        type Value = Array;
        type Constant = Array;
        type Operation = TestDomainOperation;
    }

    impl Context for StagedDispatchTestDomain {
        fn lift(&self, constant: Array) -> Result<Array, ProgramError> {
            Ok(constant)
        }

        fn bind<P: Into<Self::Operation>, D: crate::BindingRegionDriver<Self::Constant, Self::Operation>>(
            &self,
            operation: P,
            driver: D,
            inputs: &[Self::Value],
        ) -> Result<Vec<Self::Value>, ProgramError> {
            // Region-carrying binds route through the eager context's own bind, which grants application-scoped region
            // access.
            crate::EagerContext::<Array, Self::Operation>::new().bind(operation, driver, inputs)
        }

        fn resolve(&self, value: &Array) -> crate::ValueResolution<Array> {
            crate::ValueResolution::Constant(value.clone())
        }

        fn is_eager(&self) -> bool {
            false
        }
    }

    /// Eager-domain context capabilities, delegating to the zero-state [`crate::EagerContext`] exactly like
    /// `EagerContext<Array, ArrayOperation<Array>>`'s.
    impl crate::operations::constants::Zero<Array> for StagedDispatchTestDomain {
        fn zero(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
            crate::operations::constants::Zero::zero(&crate::EagerContext::<Array>::new(), r#type)
        }
    }

    impl crate::operations::constants::One<Array> for StagedDispatchTestDomain {
        fn one(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
            crate::operations::constants::One::one(&crate::EagerContext::<Array>::new(), r#type)
        }
    }

    impl<S: crate::arrays::ArrayElement> crate::operations::constants::Fill<S, Array> for StagedDispatchTestDomain {
        fn fill(&self, r#type: &ArrayType, value: S) -> Result<Array, ProgramError> {
            crate::operations::constants::Fill::fill(&crate::EagerContext::<Array>::new(), r#type, value)
        }
    }

    impl crate::operations::constants::Iota<Array> for StagedDispatchTestDomain {
        fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Array, ProgramError> {
            crate::operations::constants::Iota::iota(&crate::EagerContext::<Array>::new(), r#type, dimension)
        }
    }

    impl crate::operations::constants::Constant<Array, Array> for StagedDispatchTestDomain {
        fn constant(&self, value: Array) -> Result<Array, ProgramError> {
            Ok(value)
        }
    }

    /// Eager test context that counts each interpreted [`PrintOperation`]. Clones share the counter so nested
    /// transformation contexts and region replay contribute to one observation.
    #[derive(Clone, Debug)]
    struct CountingPrintContext {
        /// Number of print operations bound through this context.
        print_count: Rc<Cell<usize>>,
    }

    impl CountingPrintContext {
        /// Creates a context whose print count starts at zero.
        fn new() -> Self {
            Self { print_count: Rc::new(Cell::new(0)) }
        }

        /// Returns the number of print operations bound so far.
        fn print_count(&self) -> usize {
            self.print_count.get()
        }
    }

    impl Domain for CountingPrintContext {
        type Type = ArrayType;
        type Value = Array;
        type Constant = Array;
        type Operation = TestDomainOperation;
    }

    impl Context for CountingPrintContext {
        fn lift(&self, constant: Array) -> Result<Array, ProgramError> {
            Ok(constant)
        }

        fn bind<P: Into<Self::Operation>, D: crate::BindingRegionDriver<Self::Constant, Self::Operation>>(
            &self,
            operation: P,
            driver: D,
            inputs: &[Self::Value],
        ) -> Result<Vec<Self::Value>, ProgramError> {
            let operation = operation.into();
            if matches!(operation, ArrayOperation::Print(_)) {
                self.print_count.set(self.print_count.get() + 1);
            }
            EagerContext::<Array, Self::Operation>::new().bind(operation, driver, inputs)
        }

        fn resolve(&self, value: &Array) -> crate::ValueResolution<Array> {
            crate::ValueResolution::Constant(value.clone())
        }

        fn is_eager(&self) -> bool {
            true
        }
    }

    impl crate::operations::constants::Zero<Array> for CountingPrintContext {
        fn zero(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
            crate::operations::constants::Zero::zero(&EagerContext::<Array>::new(), r#type)
        }
    }

    impl crate::operations::constants::One<Array> for CountingPrintContext {
        fn one(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
            crate::operations::constants::One::one(&EagerContext::<Array>::new(), r#type)
        }
    }

    impl<S: crate::arrays::ArrayElement> crate::operations::constants::Fill<S, Array> for CountingPrintContext {
        fn fill(&self, r#type: &ArrayType, value: S) -> Result<Array, ProgramError> {
            crate::operations::constants::Fill::fill(&EagerContext::<Array>::new(), r#type, value)
        }
    }

    impl crate::operations::constants::Iota<Array> for CountingPrintContext {
        fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Array, ProgramError> {
            crate::operations::constants::Iota::iota(&EagerContext::<Array>::new(), r#type, dimension)
        }
    }

    impl crate::operations::constants::Constant<Array, Array> for CountingPrintContext {
        fn constant(&self, value: Array) -> Result<Array, ProgramError> {
            Ok(value)
        }
    }

    /// Builds the `state < threshold` while condition program over one scalar state element.
    fn scalar_threshold_condition(threshold: f64) -> Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let threshold = builder.add_constant(Array::scalar(threshold));
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![state, threshold])
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds the `while (x < threshold) { x = 2 * x }` loop with the provided semantic iteration bound.
    fn bounded_doubling_while_operation(
        threshold: f64,
        bound: usize,
    ) -> (WhileOperation<ArrayType>, Vec<Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>>>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = builder.add_input(scalar_f64);
        let two = builder.add_constant(Array::scalar(2.0));
        let doubled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, two]).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = WhileOperation::new().with_iteration_bound(bound).unwrap();
        (operation, vec![scalar_threshold_condition(threshold), body])
    }

    /// Builds `while (x < 8) { print(x); x = 2 * x }`, whose input `1` executes exactly three body iterations.
    fn effectful_doubling_while_operation()
    -> (WhileOperation<ArrayType>, Vec<Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>>>) {
        let condition = scalar_threshold_condition(8.0);
        let mut body_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let input = body_builder.add_input(ArrayType::scalar(DataType::F64));
        let printed = body_builder.add_instruction(PrintOperation::new("state"), Vec::new(), vec![input]).unwrap()[0];
        let two = body_builder.add_constant(Array::scalar(2.0));
        let output = body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![printed, two]).unwrap()[0];
        let body = body_builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        (WhileOperation::new(), vec![condition, body])
    }

    /// Builds the `while (x < threshold) { x = x * x }` loop with the provided semantic iteration bound. Squaring
    /// captures the loop state itself as a loop-varying residual, so differentiating this loop exercises the
    /// per-iteration residual stacks of the bounded staged path.
    fn bounded_squaring_while_operation(
        threshold: f64,
        bound: usize,
    ) -> (WhileOperation<ArrayType>, Vec<Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>>>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = builder.add_input(scalar_f64);
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state]).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = WhileOperation::new().with_iteration_bound(bound).unwrap();
        (operation, vec![scalar_threshold_condition(threshold), body])
    }

    /// Builds the *unbounded* `while (x < threshold) { x = x * x }` loop. Squaring makes the pushforward read the
    /// primal state every iteration, so staged forward mode through this loop exercises the fused doubled-state
    /// rule's primal/tangent coupling, and linearizing it exercises the closed-knownness split's primal
    /// recomputation inside the residual loop.
    fn unbounded_squaring_while_operation(
        threshold: f64,
    ) -> (WhileOperation<ArrayType>, Vec<Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>>>) {
        let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state]).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        (WhileOperation::new(), vec![scalar_threshold_condition(threshold), body])
    }

    #[test]
    fn test_bounded_while_value_and_grad_computes_gradient_through_staged_masked_scan() {
        // The headline bounded-while capability: end-to-end reverse mode through a *staged* while loop.
        // `f(x) = while (x < 8, iteration_bound = 5) { x = 2 * x }` at `x = 1` runs three iterations (`x` visits 1,
        // 2, 4), so the actual trip count 3 is strictly below the bound 5 and the two trailing batch items matter:
        // their mask entries are false, so they must pass tangents through unchanged in the forward scan and cotangents
        // through unchanged in the transposed scan. Locally `f(x) = 8 x`: value 8, gradient 8.
        let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
        let (output, pullback) = StagedDispatchTestDomain
            .vjp(
                move |x, ()| {
                    let mut outputs = x.context().bind(
                        TestDomainOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(1.0),
                (),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.to_f64s(), vec![8.0]);

        // The pullback contains the transposed (reversed) linear scan and no while loop, and every cotangent seed
        // scales the hand-computed gradient 8. The direct-transpose pullback consumes `[cotangent ++ residuals]`.
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("scan"), "{rendered_pullback}");
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        assert!(!rendered_pullback.contains("while"), "{rendered_pullback}");
        let pullback_inputs = |cotangent: Array| {
            let mut inputs = vec![cotangent];
            inputs.extend(residuals.iter().cloned());
            inputs
        };
        assert_eq!(
            pullback.interpret(pullback_inputs(Array::scalar(1.0))).map(|cotangents| cotangents[0].to_f64s()),
            Ok(vec![8.0]),
        );
        assert_eq!(
            pullback.interpret(pullback_inputs(Array::scalar(2.0))).map(|cotangents| cotangents[0].to_f64s()),
            Ok(vec![16.0]),
        );

        // `value_and_gradient` composes the same machinery end to end.
        let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
        let (value, gradient) = StagedDispatchTestDomain
            .differentiate_at(Array::scalar(1.0))
            .value_and_gradient(move |x| {
                let mut outputs = x
                    .context()
                    .bind(TestDomainOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                    .unwrap();
                outputs.remove(0)
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![8.0]);
        assert_eq!(gradient.to_f64s(), vec![8.0]);
    }

    #[test]
    fn test_bounded_while_value_and_grad_stores_loop_varying_residual_stacks() {
        // The store-instead-of-recompute proof: `while (x < 100, iteration_bound = 4) { x = x * x }` at `x = 2`
        // squares three times (`x` visits 2, 4, 16 → 256, trip count 3 < bound 4), and the product rule references
        // the *per-iteration* state as a loop-varying residual, so the gradient depends on the stored stack batch
        // items `[2, 4, 16, 0]` — including the zero batch item beyond the trip count, which the mask must keep inert
        // in both directions. Locally `f(x) = x⁸`: value 256 and gradient `8 x⁷ = 1024`.
        let (while_operation, while_regions) = bounded_squaring_while_operation(100.0, 4);
        let (output, pullback) = StagedDispatchTestDomain
            .vjp(
                move |x, ()| {
                    let mut outputs = x.context().bind(
                        TestDomainOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(2.0),
                (),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_parts();
        assert_eq!(output.to_f64s(), vec![256.0]);
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        let mut pullback_inputs = vec![Array::scalar(1.0)];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs).map(|cotangents| cotangents[0].to_f64s()), Ok(vec![1024.0]),);

        // The eager-domain reverse-mode entry point produces the same value and gradient numbers.
        let (while_operation, while_regions) = bounded_squaring_while_operation(100.0, 4);
        let (value, gradient) = differentiate_at(Array::scalar(2.0))
            .value_and_gradient(move |x| {
                let mut outputs = x
                    .context()
                    .bind(TestDomainOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                    .unwrap();
                outputs.remove(0)
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![256.0]);
        assert_eq!(gradient.to_f64s(), vec![1024.0]);
    }

    #[test]
    fn test_bounded_while_value_and_grad_supports_vector_state() {
        // Vector-state coverage for the bounded staged path: the residual stacks gain trailing axes (written at
        // `[counter, 0]` through the staged zero index), while each scan iteration uses its scalar validity item as a
        // broadcastable select condition for the vector tangent. The loop
        // `while (sum(x) < 20, iteration_bound = 4) { x = x * x }` at `x = [1.5, 2]` squares twice (sums visit 3.5
        // and 6.25 before reaching 21.0625), so `f(x) = sum(x⁴)` locally: value `1.5⁴ + 2⁴ = 21.0625` and gradient
        // `4 x³ = [13.5, 32]`, with trip count 2 strictly below the bound 4.
        use crate::operations::math::reduce::ReductionKind;

        let vector_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let mut condition_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let condition_state = condition_builder.add_input(vector_f64.clone());
        let summed = condition_builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![condition_state])
            .unwrap()[0];
        let threshold = condition_builder.add_constant(Array::scalar(20.0));
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![summed, threshold])
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let body_state = body_builder.add_input(vector_f64.clone());
        let squared =
            body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![body_state, body_state]).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new().with_iteration_bound(4).unwrap();
        let while_regions = vec![condition, body];

        let (value, gradient) = StagedDispatchTestDomain
            .differentiate_at(Array::vector(vec![1.5, 2.0]))
            .value_and_gradient(move |x| {
                let mut outputs = x
                    .context()
                    .bind(TestDomainOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                    .unwrap();
                let state = outputs.remove(0);
                let mut outputs = state
                    .context()
                    .bind(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), &[state.clone()])
                    .unwrap();
                outputs.remove(0)
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![21.0625]);
        assert_eq!(gradient.to_f64s(), vec![13.5, 32.0]);
    }

    #[test]
    fn test_bounded_while_eager_value_and_grad_matches_staged_numbers() {
        // The eager-domain entry point differentiates the same bounded loop to identical numbers: the loop exits
        // through its condition after three iterations, well below the bound of five.
        let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
        let (value, gradient) = differentiate_at(Array::scalar(1.0))
            .value_and_gradient(move |x| {
                let mut outputs = x
                    .context()
                    .bind(TestDomainOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                    .unwrap();
                outputs.remove(0)
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![8.0]);
        assert_eq!(gradient.to_f64s(), vec![8.0]);
    }

    #[test]
    fn test_unbounded_while_eager_jvp_executes_body_effects_once_per_iteration() {
        let context = CountingPrintContext::new();
        let observed_context = context.clone();
        let (while_operation, while_regions) = effectful_doubling_while_operation();
        let (primal, tangent) = context
            .jvp(
                move |x, ()| {
                    let mut outputs = x.context().bind(
                        TestDomainOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(1.0),
                Array::scalar(1.0),
                (),
            )
            .unwrap();

        assert_eq!(primal.to_f64s(), vec![8.0]);
        assert_eq!(tangent.to_f64s(), vec![8.0]);
        assert_eq!(observed_context.print_count(), 3);
    }

    #[test]
    fn test_bounded_while_truncation_differentiates_consistently_across_paths() {
        // A loop whose condition never turns false truncates at the bound by definition: with bound 3 the doubling
        // loop computes `f(x) = 8 x`, so at `x = 2` the value is 16 and the gradient is 8 — identical between plain
        // interpretation, the eager-domain entry point, and the staged dispatch domain (where every mask batch
        // item is true).
        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let outputs = crate::EagerContext::<Array, TestDomainOperation>::new()
            .bind(TestDomainOperation::While(while_operation), while_regions, &[Array::scalar(2.0)])
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![16.0]);

        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = differentiate_at(Array::scalar(2.0))
            .value_and_gradient(move |x| {
                let mut outputs = x
                    .context()
                    .bind(TestDomainOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                    .unwrap();
                outputs.remove(0)
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![16.0]);
        assert_eq!(gradient.to_f64s(), vec![8.0]);

        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = StagedDispatchTestDomain
            .differentiate_at(Array::scalar(2.0))
            .value_and_gradient(move |x| {
                let mut outputs = x
                    .context()
                    .bind(TestDomainOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                    .unwrap();
                outputs.remove(0)
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![16.0]);
        assert_eq!(gradient.to_f64s(), vec![8.0]);
    }

    /// Builds the per-item countdown loop `while (x > 0) { x = x - 1 }` over one scalar state element.
    fn countdown_while_operation()
    -> (WhileOperation<ArrayType>, Vec<Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>>>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder
            .add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![condition_state])
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![condition_state, zero],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let one = body_builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![body_state]).unwrap()[0];
        let next = body_builder.add_instruction(SubOperation::new(), Vec::new(), vec![body_state, one]).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        (WhileOperation::new(), vec![condition, body])
    }

    /// Stages `while_operation` over one batched item (mapped at axis 0 with `batch_size` batch items) under tracing
    /// and returns the staged batched program for structural and numeric assertions.
    fn batch_while_under_tracing(
        while_operation: WhileOperation<ArrayType>,
        while_regions: Vec<Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>>>,
        batch_size: usize,
    ) -> Program<Array, TestDomainOperation, Array, Array> {
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(batch_size)]));
        let input_atom = builder.borrow_mut().add_input(input_type);
        let input_tracer = parent.tracer(input_atom, None);
        let output = batch(
            |item| {
                let mut outputs = item.context().bind(
                    TestDomainOperation::While(while_operation),
                    while_regions.clone(),
                    &[item.clone()],
                )?;
                Ok(outputs.remove(0))
            },
            input_tracer,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        builder.borrow().clone().build::<Array, Array>(vec![output_atom], Placeholder, Placeholder).unwrap()
    }

    #[test]
    fn test_while_batching_stages_batched_predicate_loops_under_tracing() {
        // vmap-under-tracing of the per-item countdown loop: the predicate `x > 0` is per batch item, so the staged
        // batching rule stages exactly one `while` whose condition returns the batched `bool[3]` predicate directly
        // (the relaxed predicate contract, mirroring JAX's `_while_loop_batching_rule`) instead of unrolling (the
        // body's single `sub` appears exactly once in the staged trace) and without building any masking program data
        // (no `reduce_any` in the staged form; interpretation and lowering own the masked semantics). Batch items
        // [3, 1, 2] terminate after 3, 1, and 2 iterations, and inactive batch items carry their final state,
        // matching the eager operational path batch item for batch item.
        let (countdown_operation, countdown_regions) = countdown_while_operation();
        let program = batch_while_under_tracing(countdown_operation, countdown_regions, 3);
        let rendered = program.to_string();
        assert_eq!(rendered.matches("= while").count(), 1, "{rendered}");
        assert!(!rendered.contains("reduce_any"), "{rendered}");
        assert!(rendered.contains("%2:bool[3] = compare"), "{rendered}");
        assert_eq!(rendered.matches("sub").count(), 1, "{rendered}");
        let output = program.interpret(Array::vector(vec![3.0, 1.0, 2.0])).unwrap();
        assert_eq!(output.to_f64s(), vec![0.0, 0.0, 0.0]);

        // The semantic iteration bound is preserved on the staged batched-predicate while: every batch item performs
        // at most two body applications, so batch item 0 truncates at 1.0 — the numbers of the eager operational
        // bounded path.
        let (countdown_operation, countdown_regions) = countdown_while_operation();
        let program =
            batch_while_under_tracing(countdown_operation.with_iteration_bound(2).unwrap(), countdown_regions, 3);
        let rendered = program.to_string();
        assert!(rendered.contains("iteration_bound=2"), "{rendered}");
        let output = program.interpret(Array::vector(vec![3.0, 1.0, 2.0])).unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 0.0, 0.0]);
    }

    /// The rule batches the condition with natural output axes to detect per-item termination and then instantiates its
    /// predicate at axis 0. `AlignEachTo` stages axis movement only where a natural axis differs from a mapped target,
    /// so a predicate that already landed on axis 0 over unchanged state axes keeps its discovery program: batching the
    /// countdown loop performs one structural pass per region instead of re-batching the condition a second time.
    #[test]
    fn test_while_batching_reuses_the_naturally_aligned_batched_predicate_program() {
        let (countdown_operation, countdown_regions) = countdown_while_operation();
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let packed_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let state_atom = builder.borrow_mut().add_input(packed_type.clone());
        let state = parent.tracer(state_atom, None);
        let context = BatchingContext::new(parent, 3);
        let inputs = vec![ArrayBatch::new(state, BatchAxis::new(0)).unwrap()];
        let driver = CountingBatchingDriver::new(&countdown_regions);
        let outputs = countdown_operation.batch(&context, &driver, inputs.as_slice()).unwrap().into_parts().0;
        assert_eq!(driver.batch_program_calls(), 2);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        let program = builder
            .borrow()
            .clone()
            .build::<Array, Array>(vec![outputs[0].value().atom_id().unwrap()], Placeholder, Placeholder)
            .unwrap();

        // The staged loop keeps the relaxed batched predicate and its masked per-item semantics.
        let rendered = program.to_string();
        assert_eq!(rendered.matches("= while").count(), 1, "{rendered}");
        assert!(rendered.contains("%2:bool[3] = compare"), "{rendered}");
        assert_eq!(program.interpret(Array::vector(vec![3.0, 1.0, 2.0])).unwrap().to_f64s(), vec![0.0, 0.0, 0.0]);
    }

    /// Composite array IR region program over the reference [`Array`] backend.
    type CompositeRegionProgram =
        Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>;

    /// Builds the composite per-item countdown loop `while (x > 0) { x = x - 1 }` over one scalar `f64` array state
    /// element, returning its condition and body programs in `while` region order.
    fn composite_countdown_while_regions() -> Vec<CompositeRegionProgram> {
        let scalar_f64 = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let mut condition_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::ZeroLike(ZeroLikeOperation::new())),
                Vec::new(),
                vec![condition_state],
            )
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                    ComparisonDirection::GreaterThan,
                ))),
                Vec::new(),
                vec![condition_state, zero],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![predicate],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let mut body_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let one = body_builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::OneLike(OneLikeOperation::new())),
                Vec::new(),
                vec![body_state],
            )
            .unwrap()[0];
        let next = body_builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Sub(SubOperation::new())),
                Vec::new(),
                vec![body_state, one],
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![next],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        vec![condition, body]
    }

    /// The composite rule batches the condition with natural output axes to detect per-item termination and rebuilds
    /// both regions only when that widening changes the invariant boundary: with every array carry already mapped at
    /// axis 0 and a predicate that naturally landed on axis 0, the rebuilds would replay the exact same programs, so
    /// the discovery body and condition are kept and batching the per-item countdown loop performs one structural
    /// pass per region.
    #[test]
    fn test_composite_while_batching_reuses_naturally_aligned_batch_varying_programs() {
        type EagerParent = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let countdown_regions = composite_countdown_while_regions();

        // Batching batch items [3, 1, 2] under an eager parent performs exactly one structural pass per region (the
        // fixed-point discovery body and the natural condition) and interprets the loop with its masked per-item
        // semantics: every batch item counts down to 0 even though the items terminate after different trip counts.
        let context = BatchingContext::<_, ArrayIrBatching>::new(
            EagerParent::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        );
        let inputs = vec![
            ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![3.0, 1.0, 2.0])), BatchAxis::new(0)).unwrap(),
        ];
        let driver = CountingBatchingDriver::new(&countdown_regions);
        let countdown_operation = WhileOperation::<ArrayIrType>::new();
        let outputs = countdown_operation.batch(&context, &driver, inputs.as_slice()).unwrap().into_parts().0;
        assert_eq!(driver.batch_program_calls(), 2);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &ArrayIrValue::Array(Array::vector(vec![0.0, 0.0, 0.0])));

        // The skipped alignment rebuilds are byte-identical to the discovery programs the rule kept: replaying each
        // region with `AlignEachTo` at the already-aligned axes (each region returns one output — the predicate or
        // the state — whose target is axis 0) renders exactly the natural discovery program.
        let state_axes = [BatchAxis::new(0)];
        for region in &countdown_regions {
            let natural = driver
                .batch_program(
                    &context,
                    region.entry_region_ref(),
                    state_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap();
            let aligned = driver
                .batch_program(
                    &context,
                    region.entry_region_ref(),
                    state_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::AlignEachTo(vec![BatchAxis::new(0)]),
                )
                .unwrap();
            assert_eq!(natural.into_parts().0.to_string(), aligned.into_parts().0.to_string());
        }
    }

    /// Pins that composite batching of a batch-varying `while` loop stages under tracing. The rule threads the
    /// loop-invariant mapped extent as leading loop state next to the batched `bool[batch]` predicate, which the type
    /// contract used to reject outright ("a batched predicate cannot carry first-class dimension state"), so staging
    /// this shape previously failed. The relaxed contract accepts a loop-invariant dimension carry, and interpreting
    /// the staged loop exercises the masked eager path over that carry.
    #[test]
    fn test_composite_while_batching_stages_batched_predicate_loops_under_tracing() {
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_type = DimensionType::new(batch.clone());
        let trace = TraceContext::new();
        let batch_extent = trace.input(batch_type.clone().into());
        let state =
            trace.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let input_ids = [batch_extent.clone(), state.clone()].map(|input| input.atom_id().unwrap());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent);
        let outputs = context
            .bind(
                ArrayIrOperation::While(WhileOperation::new()),
                composite_countdown_while_regions(),
                &[BatchingTracer::new(context.clone(), ArrayIrBatch::new(state, BatchAxis::new(0)).unwrap())],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));

        // Exactly one `while` is staged, and it carries the threaded extent ahead of the mapped array state. The
        // condition returns the batched per-item predicate directly: no mask machinery is staged, because the loop's
        // consumers own the masked semantics.
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![outputs[0].batch().value().atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let [r#while] = program.entry_region().instructions() else {
            panic!("composite while batching should stage exactly one instruction");
        };
        assert!(matches!(r#while.operation(), ArrayIrOperation::While(_)));
        assert_eq!(r#while.inputs(), &[input_ids[0], input_ids[1]]);
        let rendered = program.to_string();
        assert!(rendered.contains("bool[batch]"), "{rendered}");
        assert!(!rendered.contains("reduce_any"), "{rendered}");

        // Interpreting the staged loop over batch items [3, 1, 2] counts every item down to 0 under the masked
        // semantics while the loop-invariant extent carry rides through unchanged.
        let outputs = program
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(batch_type, 3).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![3.0, 1.0, 2.0])),
            ])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], ArrayIrValue::Array(Array::vector(vec![0.0, 0.0, 0.0])));
    }

    /// Builds the `while (counter > 0) { (counter, value) = (counter - 1, value + value) }` loop whose predicate
    /// depends only on the counter state element.
    fn counter_doubling_while_operation()
    -> (WhileOperation<ArrayType>, Vec<Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>>>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let condition_counter = condition_builder.add_input(scalar_f64.clone());
        condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder
            .add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![condition_counter])
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![condition_counter, zero],
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let body_counter = body_builder.add_input(scalar_f64.clone());
        let body_value = body_builder.add_input(scalar_f64);
        let one = body_builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![body_counter]).unwrap()[0];
        let next_counter =
            body_builder.add_instruction(SubOperation::new(), Vec::new(), vec![body_counter, one]).unwrap()[0];
        let doubled =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![body_value, body_value]).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_counter, doubled], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        (WhileOperation::new(), vec![condition, body])
    }

    #[test]
    fn test_while_batching_stages_plain_loops_for_replicated_predicates_under_tracing() {
        // vmap-under-tracing of a loop whose predicate depends only on a replicated counter: the staged batching
        // rule batches the condition and body at the state batch axes and stages one plain `while` — no mask
        // machinery (`reduce_any` / per-element `select`) appears in the staged program. Two iterations double the
        // batched value twice: [1, 2, 3] -> [4, 8, 12], with the replicated counter ending at 0.
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let counter_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let value_atom = builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        let counter_tracer = parent.tracer(counter_atom, None);
        let value_tracer = parent.tracer(value_atom, None);
        let (counter_output, value_output) = batch(
            |(counter, value)| {
                let (while_operation, while_regions) = counter_doubling_while_operation();
                let mut outputs = counter.context().bind(
                    TestDomainOperation::While(while_operation),
                    while_regions,
                    &[counter.clone(), value.clone()],
                )?;
                let value_output = outputs.remove(1);
                Ok((outputs.remove(0), value_output))
            },
            (counter_tracer, value_tracer),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        let output_atoms = vec![counter_output.atom_id().unwrap(), value_output.atom_id().unwrap()];
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), (Array, Array)>(
                output_atoms,
                (Placeholder, Placeholder),
                (Placeholder, Placeholder),
            )
            .unwrap();
        let rendered = program.to_string();
        assert_eq!(rendered.matches("= while").count(), 1, "{rendered}");
        assert!(!rendered.contains("reduce_any"), "{rendered}");
        assert!(!rendered.contains("select"), "{rendered}");
        let (counter_output, value_output) =
            program.interpret((Array::scalar(2.0), Array::vector(vec![1.0, 2.0, 3.0]))).unwrap();
        assert_eq!(counter_output.to_f64s(), vec![0.0]);
        assert_eq!(value_output.to_f64s(), vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_bounded_while_jvp_after_batching_composes_with_masked_scan() {
        use crate::batching::{Batch, BatchableOperation, BatchingTracer};

        // F5 x F6 composition: jvp of a *vmapped bounded* while under the non-concretizing staged dispatch domain.
        // Batching stages one masked bounded while (the predicate `x < 8` is per batch item and the iteration bound 5
        // survives the staged rewrite), so the while JVP rule takes the bounded staged path: stored residual
        // stacks plus a masked linear scan on the tangent side. Batch items [1, 5, 9] double 3, 1, and 0 times, so the
        // primal is [8, 10, 9] and the per-item tangent scale is 2^iterations = [8, 2, 1].
        fn batched_bounded_while<V>(x: V) -> Result<V, ProgramError>
        where
            V: Value<Type = ArrayType> + crate::operations::manipulation::Transpose,
            V::DispatchDomain: Context<Type = ArrayType, Value = V, Constant = Array, Operation = TestDomainOperation>,
            TestDomainOperation: BatchableOperation<V::DispatchDomain, ArrayBatching>
                + crate::batching::BatchableOperation<
                    crate::TracingContext<
                        <V::DispatchDomain as crate::Domain>::Constant,
                        <V::DispatchDomain as crate::Domain>::Operation,
                    >,
                    ArrayBatching,
                > + From<crate::operations::manipulation::TransposeOperation>
                + From<crate::operations::manipulation::BroadcastOperation>,
        {
            let context = x.dispatch_domain();
            let mapped = Batch::batch(
                &context,
                |item: BatchingTracer<V::DispatchDomain, ArrayBatching>| {
                    let batching_context = item.context().clone();
                    let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
                    let mut outputs = batching_context.bind(while_operation, while_regions, &[item])?;
                    Ok(outputs.remove(0))
                },
                x,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )?;
            Ok(mapped)
        }
        let (primal, tangent) = StagedDispatchTestDomain
            .jvp(
                |input, ()| batched_bounded_while(input),
                Array::vector(vec![1.0, 5.0, 9.0]),
                Array::vector(vec![1.0, 1.0, 1.0]),
                (),
            )
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![8.0, 10.0, 9.0]);
        assert_eq!(tangent.to_f64s(), vec![8.0, 2.0, 1.0]);

        // The plain eager domain produces the same numbers...
        let (primal, tangent) = differentiate_at(Array::vector(vec![1.0, 5.0, 9.0]))
            .jvp(Array::vector(vec![1.0, 1.0, 1.0]), batched_bounded_while)
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![8.0, 10.0, 9.0]);
        assert_eq!(tangent.to_f64s(), vec![8.0, 2.0, 1.0]);

        // ... and reverse mode composes through the masked linear scan: the pullback contains the reversed scan
        // and no while loop, and the per-item gradients match the tangent scales.
        let (output, pullback) = StagedDispatchTestDomain
            .vjp(|input, ()| batched_bounded_while(input), Array::vector(vec![1.0, 5.0, 9.0]), ())
            .unwrap();
        assert_eq!(output.to_f64s(), vec![8.0, 10.0, 9.0]);
        let rendered_pullback = pullback.program().to_string();
        assert!(rendered_pullback.contains("scan"), "{rendered_pullback}");
        assert!(rendered_pullback.contains("reverse=true"), "{rendered_pullback}");
        assert!(!rendered_pullback.contains("while"), "{rendered_pullback}");
        let cotangent = pullback.apply(Array::vector(vec![1.0, 1.0, 1.0])).unwrap();
        assert_eq!(cotangent.to_f64s(), vec![8.0, 2.0, 1.0]);
    }

    #[test]
    fn test_unbounded_while_staged_jvp_stages_one_fused_doubled_state_loop() {
        // JAX-parity forward mode through a *staged* unbounded while loop: the rule stages one fused doubled-state
        // loop whose trip decision reads the primal half. For `f(x) = while (x < 16) { x = x * x }` at `x = 2` the
        // loop runs twice (2 -> 4 -> 16), so locally `f(x) = x^4`: primal 16, tangent `4 x^3 = 32`.
        let (while_operation, while_regions) = unbounded_squaring_while_operation(16.0);
        let (output, tangent) = StagedDispatchTestDomain
            .jvp(
                move |x, ()| {
                    let mut outputs = x.context().bind(
                        TestDomainOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(2.0),
                Array::scalar(1.0),
                (),
            )
            .unwrap();
        assert_eq!(output.to_f64s(), vec![16.0]);
        assert_eq!(tangent.to_f64s(), vec![32.0]);

        // Structurally, the fused forward-mode program contains exactly one while loop over the doubled state
        // `[primal, tangent]` — no unroll and no residual stacks — and its trip count stays data-dependent:
        // interpreting the same fused program at `x = 0.5` never enters the loop (0.5 * 0.5 < 0.5 is off-path), so
        // the state passes through unchanged.
        let (while_operation, while_regions) = unbounded_squaring_while_operation(16.0);
        let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let condition_region = builder.import_region(while_regions[0].entry_region_ref());
        let body_region = builder.import_region(while_regions[1].entry_region_ref());
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                TestDomainOperation::While(while_operation),
                vec![condition_region, body_region],
                vec![input],
            )
            .unwrap()[0];
        let fused = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        let rendered = fused.to_string();
        assert_eq!(rendered.matches("= while").count(), 1, "{rendered}");
        assert_eq!(fused.input_types().len(), 2);
        assert_eq!(fused.output_types().len(), 2);
        let outputs = fused.interpret(vec![Array::scalar(2.0), Array::scalar(1.0)]).unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![16.0]);
        assert_eq!(outputs[1].to_f64s(), vec![32.0]);
        let outputs = fused.interpret(vec![Array::scalar(16.0), Array::scalar(1.0)]).unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![16.0]);
        assert_eq!(outputs[1].to_f64s(), vec![1.0]);
    }

    #[test]
    fn test_unbounded_while_staged_jvp_with_zero_space_counter_state() {
        // A staged unbounded loop whose state mixes a differentiable carry with a zero-differential-space element —
        // here an `i64` iteration counter, the shape of every keyed or counted training loop. The compact fused-JVP
        // contract omits the counter's tangent entirely, so the fused loop state is `[x, i, live(ẋ)]` and the counter's
        // output dual is restored as a structural zero. `f(x) = while (i < 3) { x = x * x; i = i + 1 }` at `x = 2`
        // squares three times: primal `x^8 = 256` with tangent `8 x^7 = 1024`.
        let float_type = ArrayType::scalar(DataType::F64);
        let counter_type = ArrayType::scalar(DataType::I64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
            builder.add_input(float_type.clone());
            let counter = builder.add_input(counter_type.clone());
            let bound = builder.add_constant(Array::from_elements(counter_type.clone(), &[3i64]).unwrap());
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![counter, bound])
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
            let state = builder.add_input(float_type.clone());
            let counter = builder.add_input(counter_type.clone());
            let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state]).unwrap()[0];
            let one = builder.add_constant(Array::from_elements(counter_type.clone(), &[1i64]).unwrap());
            let incremented = builder.add_instruction(AddOperation::new(), Vec::new(), vec![counter, one]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![squared, incremented], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(float_type);
        let counter = builder.add_input(counter_type.clone());
        let outputs = builder
            .add_instruction(
                TestDomainOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                vec![state, counter],
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // The fused program's boundary is compact: `[x, i, live(ẋ)] -> [x', i', live(ẋ')]` with no counter tangent.
        let fused = program.jvp().unwrap();
        assert_eq!(fused.input_types().len(), 3);
        assert_eq!(fused.output_types().len(), 3);
        let rendered = fused.to_string();
        assert_eq!(rendered.matches("= while").count(), 1, "{rendered}");
        let outputs = fused
            .interpret(vec![
                Array::scalar(2.0),
                Array::from_elements(counter_type, &[0i64]).unwrap(),
                Array::scalar(1.0),
            ])
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![256.0]);
        assert_eq!(outputs[1].elements::<i64>(), Ok(vec![3]));
        assert_eq!(outputs[2].to_f64s(), vec![1024.0]);
    }

    #[test]
    fn test_unbounded_while_staged_linearization_recovers_the_primal_loop_through_the_closed_knownness_split() {
        // Linearization of a staged unbounded loop composes the fused doubled-state forward-mode rule with the
        // `while` closed-knownness split: the fused loop's primal half is closed under the body and the condition
        // reads only it, so the split rebinds the primal loop on the known (primal) side while the tangent program
        // keeps the fused loop whole, recomputing primal state internally. Same function as above: primal 16 and
        // pushforward scaling 32 at `x = 2`.
        let (while_operation, while_regions) = unbounded_squaring_while_operation(16.0);
        let (output, pushforward) = StagedDispatchTestDomain
            .linearize(
                move |x, ()| {
                    let mut outputs = x.context().bind(
                        TestDomainOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(2.0),
                (),
            )
            .unwrap();
        assert_eq!(output.to_f64s(), vec![16.0]);
        assert_eq!(pushforward.apply(Array::scalar(1.0)).map(|tangent| tangent.to_f64s()), Ok(vec![32.0]));
        assert_eq!(pushforward.apply(Array::scalar(2.0)).map(|tangent| tangent.to_f64s()), Ok(vec![64.0]));
    }

    #[test]
    fn test_unbounded_while_staged_reverse_mode_reports_the_transposition_error() {
        // Reverse mode through a staged unbounded loop linearizes (through the fused rule and the closed-knownness
        // split) but has no transposable tangent loop — the fused loop stores no per-iteration residuals — so the
        // `while` transposition rule reports its error, exactly like JAX's `lax.while_loop`.
        let (while_operation, while_regions) = unbounded_squaring_while_operation(16.0);
        assert!(matches!(
            StagedDispatchTestDomain.vjp(
                move |x, ()| {
                    let mut outputs = x.context().bind(
                        TestDomainOperation::While(while_operation),
                        while_regions.clone(),
                        &[x.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(2.0),
                (),
            ),
            Err(crate::differentiation::DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message,
            })) if message
                == "while does not support transposition (reverse-mode differentiation through staged unbounded \
                    while loops is not supported; eager differentiation executes concrete duals, and loops built \
                    with `with_iteration_bound` stage a transposable masked scan)",
        ));
    }
}
