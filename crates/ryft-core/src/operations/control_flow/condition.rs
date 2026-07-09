use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::contexts::{Context, StagingContext};
use crate::effects::Effects;
use crate::interpretation::{InterpretableOperation, InterpretableProgramOperation};
use crate::macros::{check_count, check_types};
use crate::operations::constants::ZeroOperation;
use crate::operations::{BooleanLike, Operation, OperationFormatter};
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationInput, PartialEvaluationOutput,
    PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, PartiallyEvaluatableProgramOperation,
};
use crate::payloads::{Captured, Input};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::TracingContext;
use crate::types::{ArrayType, Type, TypeError, Typed};

/// Canonical operation name for [`ConditionOperation`].
pub const CONDITION_OPERATION_NAME: &'static str = "condition";

// TODO(eaplatanios): Review from here onwards.

/// [`Operation`] that evaluates one of two nested branch [`Program`]s depending on a Boolean predicate. Ordinary
/// conditions use the [`Input`] predicate payload: the predicate is supplied as the first operation input (a scalar
/// Boolean input) and the remaining operation inputs are forwarded to the selected branch. Linearized conditions use
/// the [`Captured`] predicate payload: the predicate is stored in the operation payload as a residual value and the
/// operation inputs are exactly the branch input tangents or cotangents.
///
/// A predicate that is already known while *building* a program is naturally expressed with a plain Rust `if` that
/// chooses which operations to stage, so no `condition` operation is needed for it. A predicate that is staged as a
/// constant still lowers to a `stablehlo.if` operation whose constant predicate the backend folds away (via
/// [StableHLO canonicalization](https://openxla.org/stablehlo/generated/stablehlo_passes) and XLA's conditional
/// simplification), so `ryft` performs no predicate folding of its own.
///
/// The nested branches are stored as flat `Vec`-parameter [`Program`]s because they consume the operation inputs
/// directly. Structured Rust parameters are flattened before a branch is captured (i.e., via
/// [`Parameterized`](crate::parameters::Parameterized) helpers) and reconstructed later as needed. The operation
/// itself only needs the ordered parameter signature for type checking, interpretation, batching, differentiation,
/// transposition, and other transforms.
#[derive(Clone)]
pub struct ConditionOperation<V: Value, O, F: Value<Type = V::Type> = V, PredicatePayload = Input> {
    /// Branch [`Program`] of this [`ConditionOperation`] that is evaluated when the predicate is true.
    pub(crate) true_branch: Program<V, O, Vec<V>, Vec<V>>,

    /// Branch [`Program`] of this [`ConditionOperation`] that is evaluated when the predicate is false.
    pub(crate) false_branch: Program<V, O, Vec<V>, Vec<V>>,

    /// Captured predicate for captured-predicate conditions, or `None` for input-predicate conditions.
    pub(crate) predicate: Option<F>,

    /// Marker describing where the predicate value lives.
    pub(crate) predicate_payload: PhantomData<PredicatePayload>,
}

impl<V: Value, O: Debug, F: Value<Type = V::Type>, PredicatePayload> Debug
    for ConditionOperation<V, O, F, PredicatePayload>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("ConditionOperation");
        debug.field("true_branch", &self.true_branch);
        debug.field("false_branch", &self.false_branch);
        if let Some(predicate) = &self.predicate {
            debug.field("predicate", predicate);
        }
        debug.finish()
    }
}

impl<V: Value<Type = ArrayType>, O: Operation<ArrayType>> ConditionOperation<V, O> {
    /// Creates a new [`ConditionOperation`] whose predicate is supplied as the first operation input. The predicate
    /// input is not described by the operation itself: it must simply be a scalar Boolean type, which
    /// [`Operation::infer_output_types`] validates structurally against the actual first input type.
    ///
    /// # Parameters
    ///
    ///   - `true_branch`: Branch [`Program`] evaluated when the predicate is true.
    ///   - `false_branch`: Branch [`Program`] evaluated when the predicate is false. This program must have the same
    ///     input and output type signatures as `true_branch`.
    pub fn new(
        true_branch: Program<V, O, Vec<V>, Vec<V>>,
        false_branch: Program<V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        let input_types = true_branch.input_types();
        check_types!("condition branch input", &input_types, &false_branch.input_types());
        let output_types = true_branch.output_types();
        check_types!("condition branch output", &output_types, &false_branch.output_types());
        Ok(Self { true_branch, false_branch, predicate: None, predicate_payload: PhantomData })
    }
}

impl<V: Value, O: Operation<V::Type>, F: Value<Type = V::Type>> ConditionOperation<V, O, F, Captured> {
    /// Creates a new [`ConditionOperation`] whose predicate is captured in the operation payload rather than supplied
    /// as an operation input.
    ///
    /// # Parameters
    ///
    ///   - `predicate`: Captured Boolean predicate that selects the branch program to run.
    ///   - `true_branch`: Branch [`Program`] evaluated when the predicate is true.
    ///   - `false_branch`: Branch [`Program`] evaluated when the predicate is false. This program must have the same
    ///     input and output type signatures as `true_branch`.
    pub fn new_captured(
        predicate: F,
        true_branch: Program<V, O, Vec<V>, Vec<V>>,
        false_branch: Program<V, O, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        let input_types = true_branch.input_types();
        check_types!("condition branch input", &input_types, &false_branch.input_types());
        let output_types = true_branch.output_types();
        check_types!("condition branch output", &output_types, &false_branch.output_types());
        Ok(Self { true_branch, false_branch, predicate: Some(predicate), predicate_payload: PhantomData })
    }

    /// Returns the captured Boolean predicate that selects the branch to run.
    #[inline]
    pub fn predicate(&self) -> &F {
        self.predicate.as_ref().unwrap()
    }
}

impl<V: Value, O: Operation<V::Type>, F: Value<Type = V::Type>, PredicatePayload>
    ConditionOperation<V, O, F, PredicatePayload>
{
    /// Returns the branch [`Program`] of this [`ConditionOperation`] that is evaluated when the predicate is true.
    #[inline]
    pub fn true_branch(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.true_branch
    }

    /// Returns the branch [`Program`] of this [`ConditionOperation`] that is evaluated when the predicate is false.
    #[inline]
    pub fn false_branch(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.false_branch
    }

    /// Returns the output types produced by both branches of this [`ConditionOperation`].
    #[inline]
    pub fn output_types(&self) -> Vec<V::Type> {
        self.true_branch.output_types()
    }
}

impl<V: Value, O, F: Value<Type = V::Type>, PredicatePayload> Display for ConditionOperation<V, O, F, PredicatePayload>
where
    Self: Operation<V::Type>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<V: Value, O: Operation<V::Type>> Operation<V::Type> for ConditionOperation<V, O, V, Input>
where
    V::Type: BooleanLike,
{
    #[inline]
    fn name(&self) -> &'static str {
        CONDITION_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[V::Type]) -> Result<Vec<V::Type>, TypeError> {
        let branch_input_types = self.true_branch.input_types();
        check_count!("input", input_types, branch_input_types.len() + 1, TypeError);
        if !input_types[0].is_scalar() || input_types[0] != input_types[0].as_boolean() {
            return Err(TypeError {
                message: format!("condition predicate type must be a scalar boolean, but got {}", input_types[0]),
            });
        }
        check_types!("condition input", &branch_input_types, &input_types[1..]);
        Ok(self.output_types())
    }

    #[inline]
    fn effects(&self) -> Effects {
        self.true_branch.effects().union(self.false_branch.effects())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONDITION_OPERATION_NAME)?.bracketed(|operation| {
            operation.program("true_branch", &self.true_branch)?;
            operation.program("false_branch", &self.false_branch)
        })
    }
}

impl<V: Value<Type = ArrayType>, F: Value<Type = ArrayType>, O: Operation<ArrayType>> Operation<ArrayType>
    for ConditionOperation<V, O, F, Captured>
{
    #[inline]
    fn name(&self) -> &'static str {
        CONDITION_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let branch_input_types = self.true_branch.input_types();
        check_types!("condition branch input", &branch_input_types, &self.false_branch.input_types());
        let output_types = self.true_branch.output_types();
        check_types!("condition branch output", &output_types, &self.false_branch.output_types());
        check_count!("input", input_types, branch_input_types.len(), TypeError);
        check_types!("condition input", &branch_input_types, input_types);
        Ok(output_types)
    }

    #[inline]
    fn effects(&self) -> Effects {
        self.true_branch.effects().union(self.false_branch.effects())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONDITION_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("predicate", self.predicate())?;
            operation.program("true_branch", &self.true_branch)?;
            operation.program("false_branch", &self.false_branch)
        })
    }
}

impl<Constant, V, O, C> InterpretableOperation<V, C> for ConditionOperation<Constant, O, Constant, Input>
where
    Constant: Value,
    Constant::Type: BooleanLike,
    V: Value<Type = Constant::Type> + BooleanLike,
    O: InterpretableProgramOperation<V, C, Constant>,
{
    fn interpret(&self, context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice())?;
        let (predicate, branch_inputs) = (inputs[0].boolean()?, &inputs[1..]);
        O::interpret_program(
            context,
            if predicate { &self.true_branch } else { &self.false_branch },
            branch_inputs.to_vec(),
        )
    }
}

/// Partial-evaluation override for an [`Input`]-predicate [`ConditionOperation`], whose predicate is the operation's
/// first input.
///
/// With a [`Known`](PartialValue::Known) predicate that the known-side context can
/// [`resolve`](Context::resolve) to a [`Concrete`](crate::ValueResolution::Concrete) constant it selects the taken
/// branch and inlines it via
/// [`PartialEvaluationContext::inline_program`], so the condition disappears from the residual program; the inlined
/// branch is fed the remaining inputs. A known predicate that is *not* concretizable — under a staging known-side
/// context, a genuine [`Tracer`](crate::Tracer) into the outer program — cannot select a branch at
/// partial-evaluation time; the condition is instead split by `split_condition_by_knownness` into a *known*
/// condition bound in the enclosing known-side context (so known branch work stays behind the conditional instead of
/// being staged speculatively for both branches) and a *residual* condition over the unknown work, connected by
/// per-branch residual edges.
///
/// With an [`Unknown`](PartialValue::Unknown) predicate no known branch work can be hoisted at all — there is no
/// predicate to select which branch's work would run — so the condition must survive whole. It is nonetheless
/// *shrunk*: each branch is partially evaluated against the input knowledge (inputs `1..`), folding away each
/// branch's known subcomputation, and the two residual branch programs are reconciled into a single rewritten
/// `condition` emitted through the active context. Because the two branches generally need different residual
/// inputs, the rewritten condition takes the *concatenation* of the true branch's residual inputs followed by the
/// false branch's; the reconciled true branch consumes the first half and the false branch the second half, leaving
/// the other half unused so both branches share one input signature. A branch residual input fed by a folded known
/// value (a [`PartialEvaluationInput::Known`]) is propagated outward as a fresh known trace value, and one fed by an
/// unknown branch input (a [`PartialEvaluationInput::Unknown`] of branch input `k`) maps back to condition input
/// `k + 1`.
impl<V, O, C> PartiallyEvaluatableOperation<C> for ConditionOperation<V, O, V, Input>
where
    V: Value<Type = ArrayType> + BooleanLike,
    C: Context<Type = ArrayType, Constant = V, Operation = O>,
    O: Clone
        + Operation<ArrayType>
        + PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableProgramOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<V, O>>
        + From<ConditionOperation<V, O>>
        + From<ZeroOperation<ArrayType>>,
{
    fn partially_evaluate(
        &self,
        context: &PartialEvaluationContext<C>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Input 0 is the predicate; inputs 1.. feed both branches.
        if let PartialValue::Known(predicate) = inputs[0].value() {
            // A known predicate selects a branch only when it resolves to a concrete constant: under a staging
            // known-side context "known" means known to the outer program, and a genuine tracer carries no boolean
            // to branch on. A known-but-symbolic predicate — or a concrete constant payload that exposes no concrete
            // boolean, such as an abstract backend capture reference — keeps the conditional on both sides of the
            // split instead.
            if let Some(predicate) = context.parent().resolve(predicate).into_concrete() {
                if let Ok(predicate) = predicate.boolean() {
                    let branch = if predicate { self.true_branch() } else { self.false_branch() };
                    return context.inline_program(branch, inputs[1..].to_vec());
                }
            }
            if inputs.iter().all(PartialEvaluationValue::is_known) {
                return context.fold_or_residualize(O::from(self.clone()), inputs);
            }
            return split_condition_by_knownness(context, self, inputs);
        }

        // Unknown predicate: partially evaluate each branch against the input knowledge and reconcile the two
        // residual branch programs into a single rewritten condition. The recursive branch partial evaluation goes
        // through the `PartiallyEvaluatableProgramOperation` witness rather than `Program::partially_evaluate`
        // directly, so this impl avoids re-entering the operation-enum trait-solver cycle.
        //
        // Two conservative gates keep the conditional whole instead: effectful branches, because the branch folds
        // below run through the *live* known-side context and would execute or stage a branch's effects
        // speculatively (the predicate is unknown, so neither branch is selected yet); and symbolic knowns, because
        // the reconciled branch programs must embed folded known values as inline constants, which a live-trace
        // tracer cannot be.
        if !self.true_branch().effects().is_pure()
            || !self.false_branch().effects().is_pure()
            || context.any_known_is_symbolic(&inputs[1..])
        {
            return context.fold_or_residualize(O::from(self.clone()), inputs);
        }
        let branch_knowledge = inputs[1..].iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let true_evaluation = O::partially_evaluate_program(context.parent(), self.true_branch(), &branch_knowledge)?;
        let false_evaluation = O::partially_evaluate_program(context.parent(), self.false_branch(), &branch_knowledge)?;

        // Map each branch's residual inputs (true then false) back to a source feeding the rewritten condition.
        let source = |residual_input: &PartialEvaluationInput<C::Value>| match residual_input {
            PartialEvaluationInput::Unknown(input) => inputs[*input + 1].clone(),
            PartialEvaluationInput::Known(value) => PartialEvaluationValue::known(value.clone()),
        };
        let combined_inputs =
            true_evaluation.inputs.iter().chain(false_evaluation.inputs.iter()).map(source).collect::<Vec<_>>();

        // Reconcile both branches over the same concatenated input signature: the true branch consumes the leading
        // inputs and the false branch the trailing ones.
        let true_count = true_evaluation.inputs.len();
        let mut combined_input_types = true_evaluation.program.input_types();
        combined_input_types.extend(false_evaluation.program.input_types());
        let reconciled_true = reconcile_branch(context, &combined_input_types, 0, &true_evaluation)?;
        let reconciled_false = reconcile_branch(context, &combined_input_types, true_count, &false_evaluation)?;

        let condition = ConditionOperation::new(reconciled_true, reconciled_false)?;
        let mut rewritten_inputs = Vec::with_capacity(combined_inputs.len() + 1);
        rewritten_inputs.push(inputs[0].clone());
        rewritten_inputs.extend(combined_inputs);
        context.fold_or_residualize(O::from(condition), rewritten_inputs.as_slice())
    }
}

/// Bookkeeping for one branch of [`split_condition_by_knownness`]: the branch's fresh known-side context, its
/// split, and the positions of its residual edges.
struct ConditionBranchSplit<V: Value<Type = ArrayType>, O: Operation<ArrayType>> {
    /// Fresh known-side context the branch was split through; its builder holds the branch's known work.
    fresh: TracingContext<V, O>,

    /// Partial evaluation of the branch against the boundary knowledge.
    evaluation: PartialEvaluation<TracingContext<V, O>>,

    /// Per-edge local types, in edge order (feeders first, then instantiated known outputs of residual-owned slots).
    edge_types: Vec<ArrayType>,

    /// For each of the branch evaluation's residual inputs, the edge ordinal it maps to when it is a known feeder.
    feeder_edge_ordinals: Vec<Option<usize>>,

    /// For each branch output, the edge ordinal carrying its folded value when the output is residual-owned but this
    /// branch folded it (the instantiation case).
    instantiated_edge_ordinals: Vec<Option<usize>>,

    /// Fresh-context atoms of the branch's edges, in edge order.
    edge_atoms: Vec<AtomId>,
}

/// Splits an [`Input`]-predicate `condition` with a known-but-symbolic predicate into a *known* condition bound in
/// the enclosing known-side context and a *residual* condition emitted into the residual program — ryft's analogue
/// of JAX's `_cond_partial_eval` for a known branch index.
///
/// Each branch is partially evaluated through its own **fresh** staging context whose inputs stand in for the known
/// boundary inputs, so no branch work is staged speculatively into the caller's live context. An output is known
/// only when *both* branches folded it; a residual-owned output that one branch nonetheless folded is instantiated
/// as one more of that branch's residual edges, which the residual branch passes through — mirroring JAX's
/// `instantiate` flag. The known condition's branches share the signature
/// `[known inputs...] -> [known outputs..., true edges..., false edges...]`, each branch producing typed zeros for
/// the *other* branch's edge slots (only the taken branch's edges are ever consumed downstream, so the zeros are
/// dead outputs that keep the signatures aligned). The residual condition's branches share the signature
/// `[unknown inputs..., true edges..., false edges...] -> [residual outputs...]`, each branch reading only its own
/// edges.
fn split_condition_by_knownness<V, O, C>(
    context: &PartialEvaluationContext<C>,
    condition: &ConditionOperation<V, O, V, Input>,
    inputs: &[PartialEvaluationValue<C::Value>],
) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
where
    V: Value<Type = ArrayType>,
    C: Context<Type = ArrayType, Constant = V, Operation = O>,
    O: Clone
        + Operation<ArrayType>
        + PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<V, O>>
        + From<ConditionOperation<V, O>>
        + From<ZeroOperation<ArrayType>>,
{
    let branch_inputs = &inputs[1..];
    let branch_input_types = condition.true_branch.input_types();
    check_count!("input", branch_inputs, branch_input_types.len(), ProgramError);
    let input_known = branch_inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
    let output_count = condition.true_branch.output_types().len();

    // Split each branch through its own fresh known-side context.
    let split_branch = |branch: &Program<V, O, Vec<V>, Vec<V>>| -> Result<
        (TracingContext<V, O>, PartialEvaluation<TracingContext<V, O>>),
        ProgramError,
    > {
        let fresh = TracingContext::<V, O>::new();
        let knowledge = branch_input_types
            .iter()
            .zip(input_known.iter())
            .map(|(input_type, &known)| {
                if known {
                    PartialValue::Known(fresh.input(input_type.clone()))
                } else {
                    PartialValue::Unknown(input_type.clone())
                }
            })
            .collect::<Vec<_>>();
        let evaluation = branch.partially_evaluate_in_context(&fresh, knowledge.as_slice())?;
        Ok((fresh, evaluation))
    };
    let (true_fresh, true_evaluation) = split_branch(&condition.true_branch)?;
    let (false_fresh, false_evaluation) = split_branch(&condition.false_branch)?;

    // An output is known only when both branches folded it.
    let out_known = (0..output_count)
        .map(|index| {
            matches!(&true_evaluation.outputs[index], PartialEvaluationOutput::Known(_))
                && matches!(&false_evaluation.outputs[index], PartialEvaluationOutput::Known(_))
        })
        .collect::<Vec<bool>>();

    // Collect each branch's residual edges: its known feeders plus the instantiated folded values of residual-owned
    // outputs.
    let collect_branch = |fresh: TracingContext<V, O>,
                          evaluation: PartialEvaluation<TracingContext<V, O>>|
     -> Result<ConditionBranchSplit<V, O>, ProgramError> {
        let mut edge_types = Vec::new();
        let mut edge_atoms = Vec::new();
        let mut feeder_edge_ordinals = Vec::with_capacity(evaluation.inputs.len());
        for input in evaluation.inputs.iter() {
            match input {
                PartialEvaluationInput::Known(value) => {
                    feeder_edge_ordinals.push(Some(edge_types.len()));
                    edge_types.push(value.r#type().into_owned());
                    edge_atoms.push(value.atom_id()?);
                }
                PartialEvaluationInput::Unknown(_) => feeder_edge_ordinals.push(None),
            }
        }
        let mut instantiated_edge_ordinals = vec![None; output_count];
        for (index, output) in evaluation.outputs.iter().enumerate() {
            if !out_known[index] {
                if let PartialEvaluationOutput::Known(value) = output {
                    instantiated_edge_ordinals[index] = Some(edge_types.len());
                    edge_types.push(value.r#type().into_owned());
                    edge_atoms.push(value.atom_id()?);
                }
            }
        }
        Ok(ConditionBranchSplit {
            fresh,
            evaluation,
            edge_types,
            feeder_edge_ordinals,
            instantiated_edge_ordinals,
            edge_atoms,
        })
    };
    let true_split = collect_branch(true_fresh, true_evaluation)?;
    let false_split = collect_branch(false_fresh, false_evaluation)?;

    // An empty known side (no known output and no edge on either branch) means the split folds nothing; residualize
    // the condition unchanged through the default rule, with the symbolic predicate as a known feeder.
    let known_side_is_empty =
        !out_known.iter().any(|&known| known) && true_split.edge_atoms.is_empty() && false_split.edge_atoms.is_empty();
    if known_side_is_empty {
        return context.fold_or_residualize(O::from(condition.clone()), inputs);
    }

    // Build each known branch over the shared `[known outputs..., true edges..., false edges...]` output signature,
    // producing typed zeros for the other branch's edge slots.
    let build_known_branch = |own: &ConditionBranchSplit<V, O>,
                              other: &ConditionBranchSplit<V, O>,
                              own_first: bool|
     -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        let mut output_atoms = Vec::new();
        for (index, output) in own.evaluation.outputs.iter().enumerate() {
            if out_known[index] {
                match output {
                    PartialEvaluationOutput::Known(value) => output_atoms.push(value.atom_id()?),
                    PartialEvaluationOutput::Unknown(_) => {
                        return Err(ProgramError::MalformedProgram(
                            "condition known-ness split lost a known output".to_string(),
                        ));
                    }
                }
            }
        }
        let mut zero_atoms = Vec::with_capacity(other.edge_types.len());
        for edge_type in other.edge_types.iter() {
            let zeros = own.fresh.bind(ZeroOperation::new(edge_type.clone()), &[])?;
            check_count!("output", zeros, 1, ProgramError);
            zero_atoms.push(zeros[0].atom_id()?);
        }
        if own_first {
            output_atoms.extend(own.edge_atoms.iter().copied());
            output_atoms.extend(zero_atoms);
        } else {
            output_atoms.extend(zero_atoms);
            output_atoms.extend(own.edge_atoms.iter().copied());
        }
        let known_input_count = input_known.iter().filter(|&&known| known).count();
        let output_count = output_atoms.len();
        own.fresh
            .builder()
            .borrow()
            .clone()
            .build::<Vec<V>, Vec<V>>(
                output_atoms,
                vec![Placeholder; known_input_count],
                vec![Placeholder; output_count],
            )?
            .into_simplified()
    };
    let known_true = build_known_branch(&true_split, &false_split, true)?;
    let known_false = build_known_branch(&false_split, &true_split, false)?;

    // Bind the known condition into the enclosing known-side context over the predicate and the known inputs.
    let known_condition = ConditionOperation::new(known_true, known_false)?;
    let mut known_condition_inputs = Vec::with_capacity(inputs.len());
    known_condition_inputs.push(inputs[0].clone());
    known_condition_inputs.extend(
        branch_inputs
            .iter()
            .zip(input_known.iter())
            .filter(|(_, known)| **known)
            .map(|(input, _)| input.clone()),
    );
    let known_outputs = context.fold_or_residualize(O::from(known_condition), known_condition_inputs.as_slice())?;
    let known_output_count = out_known.iter().filter(|&&known| known).count();
    let true_edge_offset = known_output_count;
    let false_edge_offset = known_output_count + true_split.edge_types.len();

    // Build each residual branch over the shared `[unknown inputs..., true edges..., false edges...]` input
    // signature, each branch reading only its own edges, with instantiated folded values passed through from their
    // edge slots.
    let residual_output_ordinals = {
        let mut ordinals = vec![None; output_count];
        let mut next = 0;
        for (index, &known) in out_known.iter().enumerate() {
            if !known {
                ordinals[index] = Some(next);
                next += 1;
            }
        }
        ordinals
    };
    let has_residual_outputs = residual_output_ordinals.iter().any(Option::is_some);
    let residual_outputs = if has_residual_outputs {
        let build_residual_branch = |own: &ConditionBranchSplit<V, O>,
                                     own_edges_first: bool|
         -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
            let mut builder = ProgramBuilder::<V, O>::new();
            let mut unknown_input_atoms = vec![None; branch_input_types.len()];
            for (index, input_type) in branch_input_types.iter().enumerate() {
                if !input_known[index] {
                    unknown_input_atoms[index] = Some(builder.add_input(input_type.clone()));
                }
            }
            // The shared input signature always lists the true branch's edges before the false branch's; the branch
            // being built reads only its own group.
            let leading_edge_atoms = true_split
                .edge_types
                .iter()
                .map(|edge_type| builder.add_input(edge_type.clone()))
                .collect::<Vec<_>>();
            let trailing_edge_atoms = false_split
                .edge_types
                .iter()
                .map(|edge_type| builder.add_input(edge_type.clone()))
                .collect::<Vec<_>>();
            let own_edge_atoms = if own_edges_first { &leading_edge_atoms } else { &trailing_edge_atoms };

            let mut spliced_inputs = Vec::with_capacity(own.evaluation.inputs.len());
            for (input, edge_ordinal) in own.evaluation.inputs.iter().zip(own.feeder_edge_ordinals.iter()) {
                match (input, edge_ordinal) {
                    (PartialEvaluationInput::Unknown(slot), _) => {
                        spliced_inputs.push(unknown_input_atoms[*slot].ok_or_else(|| {
                            ProgramError::MalformedProgram(
                                "condition known-ness split saw a residual feeder for a known input".to_string(),
                            )
                        })?);
                    }
                    (PartialEvaluationInput::Known(_), Some(edge)) => spliced_inputs.push(own_edge_atoms[*edge]),
                    (PartialEvaluationInput::Known(_), None) => {
                        return Err(ProgramError::MalformedProgram(
                            "condition known-ness split lost a residual edge".to_string(),
                        ));
                    }
                }
            }
            let spliced_outputs = builder.add_program(&own.evaluation.program, &spliced_inputs)?;

            let mut output_atoms = Vec::new();
            for (index, output) in own.evaluation.outputs.iter().enumerate() {
                if out_known[index] {
                    continue;
                }
                match output {
                    PartialEvaluationOutput::Unknown(spliced) => output_atoms.push(spliced_outputs[*spliced]),
                    PartialEvaluationOutput::Known(_) => {
                        let edge = own.instantiated_edge_ordinals[index].ok_or_else(|| {
                            ProgramError::MalformedProgram(
                                "condition known-ness split lost an instantiated output edge".to_string(),
                            )
                        })?;
                        output_atoms.push(own_edge_atoms[edge]);
                    }
                }
            }
            let input_count = unknown_input_atoms.iter().filter(|atom| atom.is_some()).count()
                + leading_edge_atoms.len()
                + trailing_edge_atoms.len();
            let output_count = output_atoms.len();
            builder.build::<Vec<V>, Vec<V>>(
                output_atoms,
                vec![Placeholder; input_count],
                vec![Placeholder; output_count],
            )
        };
        let residual_true = build_residual_branch(&true_split, true)?;
        let residual_false = build_residual_branch(&false_split, false)?;
        let residual_condition = ConditionOperation::new(residual_true, residual_false)?;

        let mut residual_condition_inputs = Vec::new();
        residual_condition_inputs.push(inputs[0].clone());
        residual_condition_inputs.extend(
            branch_inputs
                .iter()
                .zip(input_known.iter())
                .filter(|(_, known)| !**known)
                .map(|(input, _)| input.clone()),
        );
        for edge in 0..true_split.edge_types.len() {
            residual_condition_inputs.push(known_outputs.get(true_edge_offset + edge).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(
                    "condition known-ness split known side produced no output for a true-branch edge".to_string(),
                )
            })?);
        }
        for edge in 0..false_split.edge_types.len() {
            residual_condition_inputs.push(known_outputs.get(false_edge_offset + edge).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(
                    "condition known-ness split known side produced no output for a false-branch edge".to_string(),
                )
            })?);
        }
        context.residualize(O::from(residual_condition), residual_condition_inputs.as_slice())?
    } else {
        Vec::new()
    };

    // Reassemble the original output order from the two sides.
    let mut known_output_ordinal = 0;
    (0..output_count)
        .map(|index| {
            if out_known[index] {
                let value = known_outputs.get(known_output_ordinal).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(
                        "condition known-ness split known side produced no output for a known result".to_string(),
                    )
                });
                known_output_ordinal += 1;
                value
            } else {
                let ordinal = residual_output_ordinals[index].ok_or_else(|| {
                    ProgramError::MalformedProgram(
                        "condition known-ness split produced a result owned by neither side".to_string(),
                    )
                })?;
                residual_outputs.get(ordinal).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(
                        "condition known-ness split residual side produced no output for a residual result".to_string(),
                    )
                })
            }
        })
        .collect()
}

/// Reconciles one partially-evaluated `condition` branch into a branch program over the shared concatenated input
/// signature; see the unknown-predicate [`PartiallyEvaluatableOperation`] implementation for
/// [`ConditionOperation`].
///
/// The reconciled program takes one input per combined source (in `combined_input_types`), splices the branch's
/// residual program over the `offset..offset + evaluation.inputs.len()` inputs (leaving the rest unused), and
/// produces the original condition's outputs by reading each [`PartialEvaluationOutput`]: a folded
/// [`Known`](PartialEvaluationOutput::Known) output becomes an inline constant (its staged payload recovered through
/// [`PartialEvaluationContext::known_constant`]), and an [`Unknown`](PartialEvaluationOutput::Unknown) output
/// reads the spliced residual program's corresponding output.
///
/// # Parameters
///
///   - `context`: Active [`PartialEvaluationContext`], used to recover constant payloads for folded known outputs.
///   - `combined_input_types`: Shared input signature both reconciled branches are built over.
///   - `offset`: Index of the first of this branch's inputs within `combined_input_types`.
///   - `evaluation`: Partial evaluation of this branch against the condition's input knowledge.
fn reconcile_branch<C: Context>(
    context: &PartialEvaluationContext<C>,
    combined_input_types: &[C::Type],
    offset: usize,
    evaluation: &PartialEvaluation<C>,
) -> Result<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, ProgramError>
where
    C::Operation: Clone,
{
    let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
    let input_atoms = combined_input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
    let branch_inputs = &input_atoms[offset..offset + evaluation.inputs.len()];
    let residual_outputs = builder.add_program(&evaluation.program, branch_inputs)?;
    let output_atoms = evaluation
        .outputs
        .iter()
        .map(|output| match output {
            PartialEvaluationOutput::Known(value) => Ok(builder.add_constant(context.known_constant(value)?)),
            PartialEvaluationOutput::Unknown(index) => Ok(residual_outputs[*index]),
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let output_count = output_atoms.len();
    builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
        output_atoms,
        vec![Placeholder; combined_input_types.len()],
        vec![Placeholder; output_count],
    )
}

/// Partial-evaluation override for a [`Captured`]-predicate [`ConditionOperation`], whose predicate is stored in the
/// operation payload rather than supplied as an input. Because the predicate is not part of the inputs offered to
/// partial evaluation, this defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`].
impl<V, F, O, C> PartiallyEvaluatableOperation<C> for ConditionOperation<V, O, F, Captured>
where
    V: Value<Type = ArrayType>,
    F: Value<Type = ArrayType>,
    O: Clone + Operation<ArrayType>,
    C: Context<Type = ArrayType>,
    C::Operation: From<ConditionOperation<V, O, F, Captured>>,
{
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::contexts::StagingContext;
    use crate::operations::arithmetic::AddOperation;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::ZeroLikeOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::tracing::DomainTracingContext;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{DataType, Shape, Size};

    use super::*;

    /// Builds a single-input flat program that maps its scalar `f64` input through `operation`.
    fn scalar_branch(
        operation: ArrayOperation<TestArray>,
    ) -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let inputs = if matches!(operation, ArrayOperation::Add(_)) { vec![input, input] } else { vec![input] };
        let output = builder.add_instruction(operation, inputs).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_condition() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::scalar(DataType::F64);
        let operation = ConditionOperation::new(
            scalar_branch(ArrayOperation::Add(AddOperation)),
            scalar_branch(ArrayOperation::ZeroLike(ZeroLikeOperation)),
        )
        .unwrap();

        // Operation identity and accessors.
        assert_eq!(operation.name(), CONDITION_OPERATION_NAME);
        assert_eq!(operation.true_branch().input_types(), vec![operand_type.clone()]);
        assert_eq!(operation.true_branch().output_types(), vec![operand_type.clone()]);
        assert_eq!(operation.false_branch().output_types(), vec![operand_type.clone()]);
        assert_eq!(operation.output_types(), vec![operand_type.clone()]);
        assert_eq!(
            format!("{operation}"),
            indoc! {"
                condition [
                    true_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = add %0 %0
                        in (%1)
                    },
                    false_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = zero_like %0
                        in (%1)
                    },
                ]
            "}
            .trim_end(),
        );

        // Type inference validates the predicate and input types and returns the branch output types.
        assert_eq!(
            operation.infer_output_types(&[predicate_type.clone(), operand_type.clone()]),
            Ok(vec![operand_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(&[]),
            Err(TypeError { message: "expected 2 inputs but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[operand_type.clone(), operand_type.clone()]),
            Err(TypeError { message: "condition predicate type must be a scalar boolean, but got f64[]".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)])),
                operand_type.clone(),
            ]),
            Err(TypeError {
                message: "condition predicate type must be a scalar boolean, but got bool[2]".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[
                predicate_type.clone(),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)])),
            ]),
            Err(TypeError {
                message: "condition input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            }),
        );

        // Construction rejects mismatched branch signatures.
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(ZeroLikeOperation, vec![input]).unwrap()[0];
        let boolean_output = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), vec![input, zero])
            .unwrap()[0];
        let boolean_branch = builder.build(vec![boolean_output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            ConditionOperation::new(scalar_branch(ArrayOperation::Add(AddOperation)), boolean_branch).map(|_| ()),
            Err(TypeError {
                message: "condition branch output type signature mismatch: expected [f64[]] but got [bool[]]"
                    .to_string(),
            }),
        );

        // Interpretation extracts the predicate from the first input and selects between the two branches.
        let predicate = |value: f64| TestArray::new(predicate_type.clone(), vec![value]);
        let outputs = operation
            .interpret(&crate::EagerContext::<TestArray>::new(), &[predicate(1.0), TestArray::scalar(4.0)])
            .unwrap();
        assert_eq!(outputs[0].values, vec![8.0]);
        let outputs = operation
            .interpret(&crate::EagerContext::<TestArray>::new(), &[predicate(0.0), TestArray::scalar(4.0)])
            .unwrap();
        assert_eq!(outputs[0].values, vec![0.0]);
        assert_eq!(
            operation.interpret(&crate::EagerContext::<TestArray>::new(), &[] as &[TestArray]),
            Err(ProgramError::Type(TypeError { message: "expected 2 inputs but got 0".to_string() })),
        );

        // Staging records the condition payload into the active program instead of trying to concretize the staged
        // predicate.
        let context = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::new();
        let builder = context.builder().clone();
        let staged_predicate = context.input(predicate_type.clone());
        let staged_operand = context.input(operand_type.clone());
        let outputs = context
            .stage_operation(operation.clone(), &[staged_predicate.clone(), staged_operand.clone()])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let builder = builder.borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), ArrayOperation::Condition(_)));
        assert_eq!(
            builder.instructions()[0].inputs(),
            &[staged_predicate.atom_id().unwrap(), staged_operand.atom_id().unwrap()],
        );
        assert_eq!(outputs[0].atom_id(), Ok(builder.instructions()[0].outputs()[0]));

        // Program rendering uses the canonical operation name and includes the nested branch programs.
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let program_predicate = builder.add_input(predicate_type);
        let program_operand = builder.add_input(operand_type);
        let program_output = builder
            .add_instruction(ArrayOperation::Condition(Box::new(operation)), vec![program_predicate, program_operand])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![program_output],
                vec![Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition [
                    true_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = add %0 %0
                        in (%1)
                    },
                    false_branch={
                        lambda %0:f64[] .
                        let %1:f64[] = zero_like %0
                        in (%1)
                    },
                ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }
}
