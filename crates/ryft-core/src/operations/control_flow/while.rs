use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::batching::BatchingTracer;
use crate::contexts::Context;
use crate::differentiation::DifferentiationTracer;
use crate::effects::Effects;
use crate::interpretation::{InterpretableOperation, InterpretableProgramOperation};
use crate::macros::{check_count, check_types};
use crate::operations::{BooleanLike, Operation, OperationFormatter};
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationInput, PartialEvaluationOutput,
    PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, PartiallyEvaluatableProgramOperation,
};
use crate::payloads::{Captured, Input};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::types::{ArrayType, DataType, Type, TypeError};
use crate::{CaptureReference, Tracer};

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`WhileOperation`].
pub const WHILE_OPERATION_NAME: &'static str = "while";

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
/// [`DataType`] conditions must produce a scalar Boolean data type. The loop-carried state rule is otherwise identical
/// for both type families: the condition and body consume the same state signature, and the body returns the next
/// state with that same signature.
pub trait WhileTypeSemantics: Type {
    /// Validates the condition output type of a while loop against the loop-carried state types.
    ///
    /// # Parameters
    ///
    ///   - `condition_output`: The single output type produced by the condition program.
    ///   - `state_types`: The loop-carried state types the condition and body consume.
    fn validate_while_condition_output(condition_output: &Self, state_types: &[Self]) -> Result<(), TypeError>;

    /// Returns whether `condition_output` is a *batched* (per-item) predicate carrying one termination decision per
    /// leading-axes item, rather than a whole-loop scalar predicate. This is `false` for scalar
    /// [`DataType`](crate::types::DataType) predicates and `true` for a non-scalar Boolean
    /// [`ArrayType`](crate::types::ArrayType) predicate. It gates the purity requirement on batched-predicate loops:
    /// a batched-predicate loop keeps running for still-active items after others have finished, so it re-evaluates
    /// the condition and body over *every* item each iteration, and observable effects cannot be masked back out for
    /// the finished items the way values can (see [`WhilePredicate`]).
    fn is_batched_predicate(condition_output: &Self) -> bool;
}

impl WhileTypeSemantics for ArrayType {
    fn validate_while_condition_output(condition_output: &Self, state_types: &[Self]) -> Result<(), TypeError> {
        if condition_output.data_type() != DataType::Boolean {
            return Err(TypeError {
                message: format!("'while' condition output type must be a Boolean array, but got {condition_output}"),
            });
        }
        let predicate_shape = condition_output.shape();
        for state_type in state_types {
            let state_shape = state_type.shape();
            let is_prefix = predicate_shape.rank() <= state_shape.rank()
                && predicate_shape.dimensions().iter().zip(state_shape.dimensions()).all(|(p, s)| p == s);
            if !is_prefix {
                return Err(TypeError {
                    message: format!(
                        "'while' condition predicate shape must be a prefix of every state shape, but predicate \
                         {condition_output} is not a prefix of state {state_type}",
                    ),
                });
            }
        }
        Ok(())
    }

    fn is_batched_predicate(condition_output: &Self) -> bool {
        condition_output.rank() > 0
    }
}

impl WhileTypeSemantics for DataType {
    fn validate_while_condition_output(condition_output: &Self, _state_types: &[Self]) -> Result<(), TypeError> {
        if condition_output != &DataType::Boolean {
            return Err(TypeError {
                message: format!("'while' condition output type must be bool, but got {condition_output}"),
            });
        }
        Ok(())
    }

    fn is_batched_predicate(_condition_output: &Self) -> bool {
        false
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
/// The default implementations are the scalar-predicate semantics, expressed through [`BooleanLike`]: the predicate's
/// own truth decides continuation, and a true predicate takes the candidate wholesale. Value types with genuinely
/// batched payloads (e.g. [`TestArray`](crate::tests::TestArray)) override both methods with per-item semantics, and
/// symbolic values (tracers and capture references) inherit the defaults, which surface [`BooleanLike::boolean`]'s
/// concretization errors — a staged while is consumed by staging and lowering rather than by this eager loop.
pub trait WhilePredicate: BooleanLike + Clone + Sized {
    /// Returns `true` when any element of this Boolean predicate is true — the loop-continuation decision.
    fn any_true(&self) -> Result<bool, ProgramError> {
        self.boolean()
    }

    /// Selects between `on_true` and `on_false` per element under this Boolean predicate, broadcasting the predicate
    /// against their shape along its leading (prefix) axes.
    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        Ok(if self.boolean()? { on_true.clone() } else { on_false.clone() })
    }
}

// Type metadata and symbolic values inherit the scalar [`WhilePredicate`] defaults, which surface
// [`BooleanLike::boolean`]'s concretization errors: none of these carry a concrete predicate payload, and staged
// whiles are consumed by staging and lowering rather than by the eager masked loop.
impl WhilePredicate for DataType {}

impl WhilePredicate for ArrayType {}

impl<C: Context> WhilePredicate for Tracer<C> {}

impl WhilePredicate for CaptureReference<ArrayType> {}

impl<C: Context<Type = ArrayType>> WhilePredicate for BatchingTracer<C> where C::Value: BooleanLike {}
impl<C: Context> WhilePredicate for DifferentiationTracer<C> where C::Value: BooleanLike {}

/// [`Operation`] that repeatedly applies a nested body [`Program`] to a loop-carried state while a nested condition
/// [`Program`] over that same state produces a true scalar Boolean predicate. The condition and body consume identical
/// state type signatures, the body produces the next state with that same signature, and the operation outputs the
/// final state once the condition produces false.
///
/// The nested condition and body are stored as flat `Vec`-parameter [`Program`]s because they consume the
/// loop-carried state directly. Structured Rust parameters are flattened before a region is captured and
/// reconstructed by the surrounding API when needed; the operation itself only needs the ordered leaf signature for
/// type checking, interpretation, JVP, batching, and transposition.
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
///   - **Eager (unrolled).** When the differentiation context's primal values are concrete, the hybrid JVP rule
///     unrolls the loop (respecting any iteration bound), producing a straight-line — and therefore transposable —
///     pushforward, so eager reverse mode works.
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
/// The `Payload` type parameter is a zero-sized semantic tag for nested program constants. The default
/// [`Captured`] payload means nested constants are carried by the program and must be materialized through the active
/// interpretation context. Direct linear programs use [`Input`] when their nested constants are already runtime
/// tangent/cotangent values and should be reused rather than lifted as primal constants.
#[derive(Clone)]
pub struct WhileOperation<V: Value, O, Payload = Captured> {
    /// Condition [`Program`] of this [`WhileOperation`] that maps the current loop state to one scalar Boolean
    /// predicate.
    pub(crate) condition: Program<V, O, Vec<V>, Vec<V>>,

    /// Body [`Program`] of this [`WhileOperation`] that maps the current loop state to the next loop state.
    pub(crate) body: Program<V, O, Vec<V>, Vec<V>>,

    /// Optional semantic iteration bound: when present, the loop runs at most this many iterations by definition,
    /// truncating even while the condition still produces true.
    pub(crate) iteration_bound: Option<usize>,

    /// [`PhantomData`] marker tying this operation to the nested constant payload role.
    marker: PhantomData<fn() -> Payload>,
}

impl<V: Value, O: Operation<V::Type>, Payload> WhileOperation<V, O, Payload>
where
    V::Type: WhileTypeSemantics,
{
    /// Creates a new [`WhileOperation`] with the provided condition and body programs.
    ///
    /// # Parameters
    ///
    ///   - `condition`: Condition [`Program`] that maps the loop-carried state to one scalar Boolean predicate.
    ///   - `body`: Body [`Program`] that maps the loop-carried state to the next loop state. This program must
    ///     consume and produce the same state type signature that `condition` consumes.
    pub fn new(
        condition: Program<V, O, Vec<V>, Vec<V>>,
        body: Program<V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        let state_types = condition.input_types();
        check_types!("while condition/body input", &state_types, &body.input_types());
        let condition_output_types = condition.output_types();
        if condition_output_types.len() != 1 {
            return Err(TypeError {
                message: format!(
                    "while condition must return exactly one predicate leaf but returned {}",
                    condition_output_types.len()
                ),
            });
        }
        <V::Type>::validate_while_condition_output(&condition_output_types[0], state_types.as_slice())?;
        check_types!("while body output", &state_types, &body.output_types());
        // A batched (per-item) predicate keeps the loop running for still-active items after others finish, so the
        // condition and body re-execute over every item each iteration. Values are masked back for finished items,
        // but observable effects cannot be, so a batched-predicate loop must be pure. This mirrors JAX's
        // `_while_loop_batching_rule`, which rejects IO effects once the predicate is batched.
        if <V::Type>::is_batched_predicate(&condition_output_types[0])
            && (!condition.effects().is_pure() || !body.effects().is_pure())
        {
            return Err(TypeError {
                message: "'while' loop with a batched predicate must be pure because observable effects cannot be \
                          masked for finished batch items"
                    .to_string(),
            });
        }
        Ok(Self { condition, body, iteration_bound: None, marker: PhantomData })
    }
}

impl<V: Value, O: Operation<V::Type>, Payload> WhileOperation<V, O, Payload> {
    /// Returns the condition [`Program`] of this [`WhileOperation`] that is evaluated before each loop iteration.
    #[inline]
    pub fn condition(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.condition
    }

    /// Returns the body [`Program`] of this [`WhileOperation`] that computes the next loop-carried state.
    #[inline]
    pub fn body(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.body
    }

    /// Returns the loop-carried state types of this [`WhileOperation`].
    #[inline]
    pub fn state_types(&self) -> Vec<V::Type> {
        self.body.input_types()
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
            return Err(TypeError { message: "while iteration bound must be at least 1".to_string() }.into());
        }
        self.iteration_bound = bound;
        Ok(self)
    }
}

/// Borrowed view of the pieces of a [`WhileOperation`] that a concrete loop driver needs: the condition program, the
/// body program, and the optional semantic iteration bound. It is returned by [`MaybeWhile::as_while`] so generic
/// transform code can unroll a `while` loop without naming the concrete [`WhileOperation`] payload type.
pub struct WhileParts<'operation, V: Value, O> {
    /// Condition [`Program`] evaluated before each iteration; maps the loop state to one scalar Boolean predicate.
    pub condition: &'operation Program<V, O, Vec<V>, Vec<V>>,

    /// Body [`Program`] computing the next loop state from the current one.
    pub body: &'operation Program<V, O, Vec<V>, Vec<V>>,

    /// Semantic iteration bound that truncates the loop when present. Refer to the documentation of
    /// [`WhileOperation::with_iteration_bound`] for the truncation semantics.
    pub iteration_bound: Option<usize>,
}

/// Query trait classifying operations as `while` loops. Closed operation enums implement this trait so that generic
/// transform code — most notably the eager unroll-then-fuse pass — can recognize and decompose a `while` instruction
/// without knowing the concrete operation enum or naming the [`WhileOperation`] payload directly. Operations that
/// are not `while` loops return [`None`].
///
/// The trait is generic over the value type `V` and operation type `O` of the nested condition and body programs so
/// that the returned [`WhileParts`] can borrow them at the operation's own program parameterization.
pub trait MaybeWhile<V: Value, O> {
    /// Returns a borrowed [`WhileParts`] view when this operation is a `while` loop, and [`None`] otherwise.
    fn as_while(&self) -> Option<WhileParts<'_, V, O>>;

    /// Returns whether this operation is a `while` loop.
    #[inline]
    fn is_while(&self) -> bool {
        self.as_while().is_some()
    }
}

impl<V: Value, O, Payload> MaybeWhile<V, O> for WhileOperation<V, O, Payload> {
    #[inline]
    fn as_while(&self) -> Option<WhileParts<'_, V, O>> {
        Some(WhileParts { condition: &self.condition, body: &self.body, iteration_bound: self.iteration_bound })
    }
}

impl<V: Value, O, Payload> Debug for WhileOperation<V, O, Payload>
where
    O: Debug,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WhileOperation")
            .field("condition", &self.condition)
            .field("body", &self.body)
            .field("iteration_bound", &self.iteration_bound)
            .finish()
    }
}

impl<V: Value, O, Payload> Display for WhileOperation<V, O, Payload>
where
    Self: Operation<V::Type>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value, O: Operation<V::Type>, Payload> Operation<V::Type> for WhileOperation<V, O, Payload> {
    #[inline]
    fn name(&self) -> &'static str {
        WHILE_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[V::Type]) -> Result<Vec<V::Type>, TypeError> {
        let state_types = self.state_types();
        check_count!("input", input_types, state_types.len(), TypeError);
        check_types!("while input", &state_types, input_types);
        Ok(state_types)
    }

    #[inline]
    fn effects(&self) -> Effects {
        self.condition.effects().union(self.body.effects())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, WHILE_OPERATION_NAME)?.bracketed(|operation| {
            if let Some(iteration_bound) = self.iteration_bound {
                operation.field("iteration_bound", iteration_bound)?;
            }
            operation.program("condition", &self.condition)?;
            operation.program("body", &self.body)
        })
    }
}

impl<Constant, V, O, Payload, C> InterpretableOperation<V, C> for WhileOperation<Constant, O, Payload>
where
    Constant: Value,
    Constant::Type: WhileTypeSemantics,
    V: Value<Type = Constant::Type> + WhilePredicate,
    O: InterpretableProgramOperation<V, C, Constant>,
{
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let mut state = inputs.to_vec();
        let mut completed_iterations = 0;
        loop {
            // The iteration bound is semantic: a bounded loop runs at most `bound` iterations by definition, so the
            // loop exits here even while the condition still produces true.
            if self.iteration_bound.is_some_and(|bound| completed_iterations >= bound) {
                return Ok(state);
            }
            let condition_outputs = O::interpret_program(context, &self.condition, state.clone())?;
            check_count!("output", condition_outputs, 1, ProgramError);
            let predicate = &condition_outputs[0];
            if !predicate.any_true()? {
                return Ok(state);
            }
            // Masked state update: items whose predicate is true take the body's candidate update, the rest keep
            // their carried state. For a scalar predicate this reduces to taking the candidates wholesale, since a
            // false scalar predicate exits above.
            let candidates = O::interpret_program(context, &self.body, state.clone())?;
            check_count!("output", candidates, self.state_types().len(), ProgramError);
            state = candidates
                .iter()
                .zip(state.iter())
                .map(|(candidate, carried)| predicate.mask_select(candidate, carried))
                .collect::<Result<Vec<_>, _>>()?;
            completed_iterations += 1;
        }
    }
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
) -> Result<Vec<AtomId>, ProgramError>
where
    C::Operation: Clone,
{
    let mut residual_inputs = Vec::with_capacity(evaluation.inputs.len());
    for residual_input in evaluation.inputs.iter() {
        match residual_input {
            PartialEvaluationInput::Unknown(state_index) => residual_inputs.push(state_atoms[*state_index]),
            PartialEvaluationInput::Known(value) => {
                residual_inputs.push(builder.add_constant(context.known_constant(value)?))
            }
        }
    }
    let spliced_outputs = builder.add_program(&evaluation.program, &residual_inputs)?;
    evaluation
        .outputs
        .iter()
        .map(|output| match output {
            PartialEvaluationOutput::Known(value) => Ok(builder.add_constant(context.known_constant(value)?)),
            PartialEvaluationOutput::Unknown(index) => Ok(spliced_outputs[*index]),
        })
        .collect()
}

/// Type-family partial-evaluation semantics for [`Captured`]-payload [`WhileOperation`]s. The loop's value,
/// operation, and known-side context parameters ride as trait inputs (with the type family as the implementing type,
/// mirroring [`ScanPayload`](super::scan::ScanPayload)) so that the [`ArrayType`] and [`DataType`] implementations
/// stay coherent now that [`WhileOperation`] no longer names its type family as a struct parameter, and so that each
/// family implementation can carry exactly the capability bounds its rule needs.
pub(crate) trait WhilePartialEvaluation<V, O, C>: Type
where
    V: Value<Type = Self>,
    C: Context,
{
    /// Partially evaluates the provided [`Captured`]-payload [`WhileOperation`]; refer to the documentation of
    /// [`PartiallyEvaluatableOperation::partially_evaluate`] for the contract.
    fn partially_evaluate_while(
        operation: &WhileOperation<V, O, Captured>,
        context: &PartialEvaluationContext<C>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>;
}

/// Partial-evaluation rule for a [`Captured`]-payload [`WhileOperation`] over [`ArrayType`].
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
/// through the [`PartiallyEvaluatableProgramOperation`] witness on the *body* (the condition produces no state and so
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
/// If no state element is loop-invariant-known and neither nested program shrank, the rule defers to the default
/// residualize-unchanged behavior.
impl<V, O, C> WhilePartialEvaluation<V, O, C> for ArrayType
where
    V: Value<Type = ArrayType>,
    C: Context<Type = ArrayType, Constant = V, Operation = O>,
    C::Value: PartialEq,
    O: Clone
        + Operation<ArrayType>
        + PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableProgramOperation<C>
        + From<WhileOperation<V, O>>,
{
    fn partially_evaluate_while(
        operation: &WhileOperation<V, O, Captured>,
        context: &PartialEvaluationContext<C>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // When every input is known the whole loop folds by binding it in the known-side context; defer to that
        // default behavior.
        if inputs.iter().all(PartialEvaluationValue::is_known) {
            return context.fold_or_residualize(O::from(operation.clone()), inputs);
        }

        let state_types = operation.state_types();
        let state_count = state_types.len();

        // The invariance fixed point below probes by folding the condition and body through the *live* known-side
        // context. For an effectful loop each probe round would execute (eager) or stage (staging) the loop's
        // effects once more, so effectful loops skip invariance probing and residualize unchanged (see the effect
        // placement contract on `PartialEvaluationContext::fold_or_residualize`); `while` has no known-ness
        // split to fall back to.
        if !operation.condition.effects().is_pure() || !operation.body.effects().is_pure() {
            return context.fold_or_residualize(O::from(operation.clone()), inputs);
        }

        // A state element can only fold if its init input is known *and* concretizable in the known-side context:
        // the folded value must be embeddable as a rebuilt-program constant, and skipping symbolic knowns also keeps
        // the fixed point's probe rounds from folding symbolic known work into a live staging context.
        let state_inits = (0..state_count)
            .map(|index| {
                inputs[index].as_known().filter(|value| context.parent().resolve(value).is_concrete()).cloned()
            })
            .collect::<Vec<Option<C::Value>>>();

        // Monotonically narrow the set of loop-invariant-known state elements to a fixed point. A round binds each
        // invariant element to its init, leaves everything else unknown, and keeps an element only if the body
        // reproduces its init as the next-state value. With no invariance candidates at all there is nothing the
        // rebuild below could embed, so skip the live-context probe entirely and residualize unchanged.
        let mut invariant = state_inits.iter().map(Option::is_some).collect::<Vec<bool>>();
        if invariant.iter().all(|candidate| !candidate) {
            return context.fold_or_residualize(O::from(operation.clone()), inputs);
        }
        let state_knowledge = |invariant: &[bool]| -> Vec<PartialValue<C::Value>> {
            (0..state_count)
                .map(|index| match (invariant[index], &state_inits[index]) {
                    (true, Some(value)) => PartialValue::Known(value.clone()),
                    _ => PartialValue::Unknown(state_types[index].clone()),
                })
                .collect()
        };

        let mut body_evaluation =
            O::partially_evaluate_program(context.parent(), &operation.body, &state_knowledge(&invariant))?;
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
            body_evaluation =
                O::partially_evaluate_program(context.parent(), &operation.body, &state_knowledge(&invariant))?;
        }

        // The condition reads the loop state too, so folding the invariant-known state can shrink it as well.
        let condition_evaluation =
            O::partially_evaluate_program(context.parent(), &operation.condition, &state_knowledge(&invariant))?;

        // Nothing folded: defer to the default residualize-unchanged behavior. A loop-invariant-known element always
        // shrinks the body (its uses fold to constants), so the only way nothing folds is an empty invariant set whose
        // residual condition and body did not shrink either. The rebuild below embeds the probes' known values as
        // inline program constants, which is only possible when they are all concrete — under a staging known-side
        // context a probe can fold a constant-only chain into a live-trace tracer — so a non-concrete probe takes
        // the same fallback.
        if (invariant.iter().all(|folded| !folded)
            && body_evaluation.program.instructions().len() >= operation.body.instructions().len()
            && condition_evaluation.program.instructions().len() >= operation.condition.instructions().len())
            || !context.all_knowns_are_concrete(&body_evaluation)
            || !context.all_knowns_are_concrete(&condition_evaluation)
        {
            return context.fold_or_residualize(O::from(operation.clone()), inputs);
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

        let while_operation = WhileOperation::<V, O>::new(residual_condition, residual_body)?
            .with_iteration_bound(operation.iteration_bound)?;

        // The residual while's inputs are exactly the original while's inputs: each state element's init value (now a
        // known residual for the folded elements) in state order.
        context.fold_or_residualize(O::from(while_operation), inputs)
    }
}

/// Partial evaluation of a scalar [`Captured`]-payload [`WhileOperation`] over [`DataType`] defers to the default
/// fold-or-residualize behavior of [`Program::partially_evaluate`]. Scalar `DataType` has no array-stack metadata for
/// the loop-invariant folding rewrite to rebuild residual state with, so a scalar while folds entirely when its
/// inputs are all known and otherwise residualizes unchanged.
impl<V, O, C> WhilePartialEvaluation<V, O, C> for DataType
where
    V: Value<Type = DataType>,
    O: Clone + Operation<DataType>,
    C: Context<Type = DataType>,
    C::Operation: From<WhileOperation<V, O>>,
{
    fn partially_evaluate_while(
        operation: &WhileOperation<V, O, Captured>,
        context: &PartialEvaluationContext<C>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        context.fold_or_residualize(C::Operation::from(operation.clone()), inputs)
    }
}

/// Partial-evaluation override for a [`Captured`]-payload [`WhileOperation`], dispatching to the loop's type family
/// through [`WhilePartialEvaluation`]: array loops fold loop-invariant-known state, and scalar loops defer to the
/// default fold-or-residualize behavior.
impl<V: Value, O: Clone, C: Context> PartiallyEvaluatableOperation<C> for WhileOperation<V, O, Captured>
where
    V::Type: WhilePartialEvaluation<V, O, C>,
    C::Operation: From<WhileOperation<V, O>>,
{
    fn partially_evaluate(
        &self,
        context: &PartialEvaluationContext<C>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        <V::Type>::partially_evaluate_while(self, context, inputs)
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`] for an
/// [`Input`]-payload [`WhileOperation`] — the linearized loops whose nested constants are already runtime tangent or
/// cotangent values. The loop-invariant-folding rule above applies only to the ordinary [`Captured`]-payload while;
/// keying this default on the [`Input`] payload keeps the two implementations disjoint.
impl<V: Value, O: Clone + Operation<V::Type>, C: Context> PartiallyEvaluatableOperation<C>
    for WhileOperation<V, O, Input>
where
    C::Operation: From<WhileOperation<V, O, Input>>,
{
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::contexts::StagingContext;
    use crate::operations::arithmetic::{AddOperation, MulOperation, SubOperation};
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::{OneLikeOperation, ZeroLikeOperation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::tracing::DomainTracingContext;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{DataType, Shape, Size};

    use super::*;

    /// Builds a condition program that maps a scalar `f64` state to the scalar Boolean predicate `state > 0`.
    fn greater_than_zero_condition() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(ZeroLikeOperation, vec![state]).unwrap()[0];
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![state, zero])
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a body program that maps a scalar `f64` state to `state - 1`.
    fn subtract_one_body() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_instruction(OneLikeOperation, vec![state]).unwrap()[0];
        let next_state = builder.add_instruction(SubOperation, vec![state, one]).unwrap()[0];
        builder.build(vec![next_state], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_while() {
        let state_type = ArrayType::scalar(DataType::F64);
        let operation = WhileOperation::new(greater_than_zero_condition(), subtract_one_body()).unwrap();

        // Operation identity and accessors.
        assert_eq!(operation.name(), WHILE_OPERATION_NAME);
        assert_eq!(operation.condition().output_types(), vec![ArrayType::scalar(DataType::Boolean)]);
        assert_eq!(operation.body().output_types(), vec![state_type.clone()]);
        assert_eq!(operation.state_types(), vec![state_type.clone()]);
        assert_eq!(
            format!("{operation}"),
            indoc! {"
                while [
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
            "}
            .trim_end(),
        );

        // Type inference validates the state types and returns them as the output types.
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&state_type)), Ok(vec![state_type.clone()]));
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))]),
            Err(TypeError {
                message: "while input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            }),
        );

        // Construction rejects mismatched condition/body state signatures, non-scalar-Boolean condition outputs,
        // multi-output conditions, and body outputs that do not match the state signature.
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let state = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])));
        let zero = builder.add_instruction(ZeroLikeOperation, vec![state]).unwrap()[0];
        let vector_body = builder.build(vec![zero], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            WhileOperation::<TestArray, ArrayOperation<TestArray>>::new(greater_than_zero_condition(), vector_body,)
                .map(|_| ()),
            Err(TypeError {
                message: "while condition/body input type signature mismatch: expected [f64[]] but got [f64[2]]"
                    .to_string(),
            }),
        );
        assert_eq!(
            WhileOperation::<TestArray, ArrayOperation<TestArray>>::new(subtract_one_body(), subtract_one_body(),)
                .map(|_| ()),
            Err(TypeError {
                message: "'while' condition output type must be a Boolean array, but got f64[]".to_string(),
            }),
        );
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let multi_output_condition =
            builder.build(vec![state, state], vec![Placeholder], vec![Placeholder, Placeholder]).unwrap();
        assert_eq!(
            WhileOperation::<TestArray, ArrayOperation<TestArray>>::new(multi_output_condition, subtract_one_body(),)
                .map(|_| ()),
            Err(TypeError {
                message: "while condition must return exactly one predicate leaf but returned 2".to_string(),
            }),
        );
        assert_eq!(
            WhileOperation::<TestArray, ArrayOperation<TestArray>>::new(
                greater_than_zero_condition(),
                greater_than_zero_condition(),
            )
            .map(|_| ()),
            Err(TypeError {
                message: "while body output type signature mismatch: expected [f64[]] but got [bool[]]".to_string(),
            }),
        );

        // The semantic iteration bound defaults to absent, must be at least one, may be cleared with `None`, and is
        // reported by the accessor.
        assert_eq!(operation.iteration_bound(), None);
        let bounded: WhileOperation<TestArray, ArrayOperation<TestArray>> =
            WhileOperation::new(greater_than_zero_condition(), subtract_one_body())
                .unwrap()
                .with_iteration_bound(2)
                .unwrap();
        assert_eq!(bounded.iteration_bound(), Some(2));
        assert_eq!(bounded.clone().with_iteration_bound(None).unwrap().iteration_bound(), None);
        assert_eq!(
            WhileOperation::<TestArray, ArrayOperation<TestArray>>::new(
                greater_than_zero_condition(),
                subtract_one_body(),
            )
            .unwrap()
            .with_iteration_bound(0)
            .map(|_| ()),
            Err(ProgramError::Type(TypeError { message: "while iteration bound must be at least 1".to_string() })),
        );

        // The bound renders as an `iteration_bound=` field ahead of the nested programs.
        assert_eq!(
            format!("{bounded}"),
            indoc! {"
                while [
                    iteration_bound=2,
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
            "}
            .trim_end(),
        );

        // Interpretation iterates the body until the condition produces false.
        let outputs = operation.interpret(&crate::EagerContext::<TestArray>::new(), &[TestArray::scalar(3.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![0.0]);
        let outputs =
            operation.interpret(&crate::EagerContext::<TestArray>::new(), &[TestArray::scalar(-1.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![-1.0]);
        assert_eq!(
            operation.interpret(&crate::EagerContext::<TestArray>::new(), &[] as &[TestArray]),
            Err(ProgramError::Type(TypeError { message: "expected 1 input but got 0".to_string() })),
        );

        // A bounded while runs at most `bound` iterations by definition: the subtract-one loop at 5 would run five
        // iterations on its own, but the bound of 2 truncates it at 3 even though the condition is still true.
        let outputs = bounded.interpret(&crate::EagerContext::<TestArray>::new(), &[TestArray::scalar(5.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![3.0]);
        // A loop that exits before reaching the bound is unaffected by it.
        let outputs = bounded.interpret(&crate::EagerContext::<TestArray>::new(), &[TestArray::scalar(1.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![0.0]);

        // Staging records the while payload into the active program instead of trying to drive the loop with a
        // concrete predicate.
        let context = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::new();
        let builder = context.builder().clone();
        let staged_state = context.input(state_type.clone());
        let outputs = context.stage_operation(operation.clone(), std::slice::from_ref(&staged_state)).unwrap();
        assert_eq!(outputs.len(), 1);
        let builder = builder.borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), ArrayOperation::While(_)));
        assert_eq!(builder.instructions()[0].inputs(), &[staged_state.atom_id().unwrap()]);
        assert_eq!(outputs[0].atom_id(), Ok(builder.instructions()[0].outputs()[0]));

        // Program rendering uses the canonical operation name and includes the nested condition and body programs.
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let program_state = builder.add_input(state_type);
        let program_output =
            builder.add_instruction(ArrayOperation::While(Box::new(operation)), vec![program_state]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![program_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = while [
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
                ] %0
                in (%1)
            "}
            .trim_end(),
        );
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
    fn test_partially_evaluate_folds_loop_invariant_known_state() {
        let scalar = || ArrayType::scalar(DataType::F64);

        // Condition `[counter, acc, k] -> [counter > 0]` (reads only the counter).
        let condition = || {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let counter = builder.add_input(scalar());
            let _acc = builder.add_input(scalar());
            let _k = builder.add_input(scalar());
            let zero = builder.add_instruction(ZeroLikeOperation, vec![counter]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![counter, zero])
                .unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };

        // Body `[counter, acc, k] -> [counter - 1, acc + k * k, k]`.
        let body = || {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let counter = builder.add_input(scalar());
            let acc = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let one = builder.add_instruction(OneLikeOperation, vec![counter]).unwrap()[0];
            let next_counter = builder.add_instruction(SubOperation, vec![counter, one]).unwrap()[0];
            let ksq = builder.add_instruction(MulOperation, vec![k, k]).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation, vec![acc, ksq]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(
                    vec![next_counter, next_acc, k],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };

        // Flat program over `[counter_init, acc_init, k_init]` staging the bounded while; its outputs are the final
        // `[counter, acc, k]` state.
        let operation = WhileOperation::<TestArray, ArrayOperation<TestArray>>::new(condition(), body())
            .unwrap()
            .with_iteration_bound(8)
            .unwrap();
        let original_body_instructions = body().instructions().len();
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let counter_init = builder.add_input(scalar());
        let acc_init = builder.add_input(scalar());
        let k_init = builder.add_input(scalar());
        let outputs = builder.add_instruction(operation, vec![counter_init, acc_init, k_init]).unwrap().to_vec();
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let knowledge = vec![
            PartialValue::Unknown(scalar()),
            PartialValue::Unknown(scalar()),
            PartialValue::Known(TestArray::scalar(3.0)),
        ];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        // The while instruction residualizes (its inputs are not all known), so every state output is produced by
        // the residual program — even the loop-invariant `k`, whose residual while body folds it to the constant 3.
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(_)));
        assert!(matches!(&evaluation.outputs[1], PartialEvaluationOutput::Unknown(_)));
        assert!(matches!(&evaluation.outputs[2], PartialEvaluationOutput::Unknown(_)));

        // The residual program's only instruction is the rewritten while.
        assert_eq!(evaluation.program.instructions().len(), 1);
        let ArrayOperation::While(residual_while) = evaluation.program.instructions()[0].operation() else {
            panic!("expected the residual program to contain a rewritten while");
        };

        // The state set is preserved (so output arity matches) and the iteration bound is carried over, but the body
        // shrank: `k * k` folded to a constant, so the body drops from four instructions to three.
        assert_eq!(residual_while.state_types().len(), 3);
        assert_eq!(residual_while.iteration_bound(), Some(8));
        assert!(residual_while.body().instructions().len() < original_body_instructions);
        assert_eq!(residual_while.body().instructions().len(), 3);

        // Correctness: interpreting the residual program reproduces the original program on the same concrete inputs.
        let runtime = |counter: f64, acc: f64| -> Vec<TestArray> {
            let arguments = evaluation
                .inputs
                .iter()
                .map(|residual_input| match residual_input {
                    PartialEvaluationInput::Known(value) => value.clone(),
                    PartialEvaluationInput::Unknown(index) => match index {
                        0 => TestArray::scalar(counter),
                        _ => TestArray::scalar(acc),
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
            program
                .interpret(vec![TestArray::scalar(counter), TestArray::scalar(acc), TestArray::scalar(k)])
                .unwrap()
        };

        let reassembled = runtime(4.0, 1.0);
        let expected = original(4.0, 1.0, 3.0);
        assert_eq!(
            reassembled.iter().map(|value| value.values.clone()).collect::<Vec<_>>(),
            expected.iter().map(|value| value.values.clone()).collect::<Vec<_>>(),
        );
        // The loop runs four times (counter `4 -> 0`): `counter` lands at `0`, `acc` threads
        // `1 -> 1 + 9 -> 19 -> 28 -> 37`, and the loop-invariant `k` final state stays `3`.
        assert_eq!(reassembled[0].values, vec![0.0]);
        assert_eq!(reassembled[1].values, vec![37.0]);
        assert_eq!(reassembled[2].values, vec![3.0]);
    }

    #[test]
    fn test_while_accepts_batched_predicate_and_interprets_with_masked_semantics() {
        // A `bool[3]` predicate over an `f64[3]` state satisfies the predicate-prefix rule, and interpretation runs
        // the masked loop: it continues while any per-item predicate is true, and items whose predicate is false
        // keep their carried state. Items [3, 1, 2] count down independently, terminating after 3, 1, and 2
        // iterations.
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let condition = {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation, vec![state]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![state, zero])
                .unwrap()[0];
            builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation, vec![state]).unwrap()[0];
            let next_state = builder.add_instruction(SubOperation, vec![state, one]).unwrap()[0];
            builder.build(vec![next_state], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let operation = WhileOperation::<TestArray, ArrayOperation<TestArray>>::new(condition, body).unwrap();
        assert_eq!(operation.condition().output_types()[0].shape().rank(), 1);

        let context = crate::contexts::EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let outputs = operation.interpret(&context, &[TestArray::vector(vec![3.0, 1.0, 2.0])]).unwrap();
        assert_eq!(outputs[0].values, vec![0.0, 0.0, 0.0]);

        // The semantic iteration bound truncates the shared masked iterations: item 0 stops at 1.0 after two body
        // applications while items 1 and 2 finish on their own predicates first.
        let bounded = operation.with_iteration_bound(2).unwrap();
        let outputs = bounded.interpret(&context, &[TestArray::vector(vec![3.0, 1.0, 2.0])]).unwrap();
        assert_eq!(outputs[0].values, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_while_rejects_batched_predicate_with_effects() {
        // A batched predicate keeps the loop running for still-active items after others finish, re-executing the
        // body over every item each iteration. Values are masked back for finished items, but observable effects
        // (here a `print` in the body) cannot be, so construction rejects an effectful batched-predicate loop. A
        // scalar predicate imposes no such restriction (the loop exits for all items at once).
        use crate::operations::debugging::PrintOperation;
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let condition = {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation, vec![state]).unwrap()[0];
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![state, zero])
                .unwrap()[0];
            builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let effectful_body = {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation, vec![state]).unwrap()[0];
            let next_state = builder.add_instruction(SubOperation, vec![state, one]).unwrap()[0];
            let printed = builder.add_instruction(PrintOperation::new("state"), vec![next_state]).unwrap()[0];
            builder.build(vec![printed], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        assert_eq!(
            WhileOperation::<TestArray, ArrayOperation<TestArray>>::new(condition, effectful_body).map(|_| ()),
            Err(TypeError {
                message: "'while' loop with a batched predicate must be pure because observable effects cannot be \
                          masked for finished batch items"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_while_rejects_batched_predicate_that_is_not_a_state_shape_prefix() {
        // A `bool[3]` predicate over an `f64[2]` state violates the predicate-prefix rule: item masking would be
        // ill-defined, so construction fails eagerly.
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]));
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(3)]));
        let condition = {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            builder.add_input(state_type.clone());
            let predicate = builder
                .add_instruction(crate::operations::constants::ZeroOperation::new(predicate_type), vec![])
                .unwrap()[0];
            builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let state = builder.add_input(state_type);
            builder.build(vec![state], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        assert_eq!(
            WhileOperation::<TestArray, ArrayOperation<TestArray>>::new(condition, body).map(|_| ()),
            Err(TypeError {
                message: "'while' condition predicate shape must be a prefix of every state shape, but predicate \
                          bool[3] is not a prefix of state f64[2]"
                    .to_string(),
            }),
        );
    }
}
