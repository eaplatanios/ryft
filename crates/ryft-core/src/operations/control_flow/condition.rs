//! Contains the `condition` control-flow operation: [`ConditionOperation`], which evaluates one of its two attached
//! branch [`Region`](crate::Region)s depending on a scalar Boolean predicate, together with its interpretation,
//! partial-evaluation, batching, forward-mode differentiation, and transposition rules. This is the analogue of
//! [JAX's `lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) (restricted to two branches)
//! and lowers to [StableHLO's `if`](https://openxla.org/stablehlo/spec#if).

use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::backends::array_programs::ArrayIrValue;
use crate::backends::dimensions::{DimensionOperation, DimensionValue};
use crate::batching::array_ir::{ArrayIrBatch, ArrayIrBatching, require_equal_dimensions};
use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchAxis, BatchableOperation, BatchedProgram, BatchingContext,
    BatchingDriver, BatchingError, ProgramBatchingOutputAxesPolicy, batch_projected_operation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::constants::{Zero, ZeroOperationProvider};
use crate::operations::control_flow::{Select, SelectOperation};
use crate::operations::dimensions::{DimensionRequirementOperation, DimensionSizeOperation};
use crate::operations::manipulation::{
    BroadcastOperation, LegacyBroadcast, LegacyBroadcastOperation, Transpose, TransposeOperation,
};
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput,
    PartialEvaluationOutput, PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, PartitionedProgram,
};
use crate::programs::builders::ProgramBuilder;
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::programs::Program;
use crate::programs::regions::{OutputRegionProvenance, RegionInterface, RegionSlot};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Concretizable, Value, ValueProjection};
use crate::programs::{MaybeZero, ProgramError};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayIrType, ArrayType, DimensionType};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`ConditionOperation`].
pub const CONDITION_OPERATION_NAME: &str = "condition";

/// [`Operation`] that evaluates one of its two attached branch [`Region`](crate::Region)s depending on a Boolean
/// predicate supplied as the first operation input (a scalar Boolean); the remaining operation inputs are forwarded
/// to the selected branch.
///
/// The branch computations are not part of this payload: they are [`Region`](crate::Region)s attached to the
/// [`Instruction`](crate::Instruction) applying the operation, in the [`region_slots`](Operation::region_slots)
/// order `["true", "false"]`, and semantic rules reach them through their driver-granted region access. Conditions
/// with owned branches supply the two branch [`Program`]s through the region driver passed to [`Context::bind`].
///
/// A predicate that is already known while *building* a program is naturally expressed with a plain Rust `if` that
/// chooses which operations to stage, so no `condition` operation is needed for it. A predicate that is staged as a
/// constant still lowers to a `stablehlo.if` operation whose constant predicate the backend folds away (via
/// [StableHLO canonicalization](https://openxla.org/stablehlo/generated/stablehlo_passes) and XLA's conditional
/// simplification), so `ryft` performs no predicate folding of its own. This is the analogue of
/// [JAX's `lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) restricted to two branches.
#[derive(Clone)]
pub struct ConditionOperation<F: Value> {
    /// Marker tying the condition to the value family whose programs its enclosing operation family stages.
    value_family: PhantomData<F>,
}

impl<F: Value> Debug for ConditionOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ConditionOperation").finish()
    }
}

impl<F: Value> ConditionOperation<F> {
    /// Creates a new [`ConditionOperation`]. The two branch
    /// [`Program`]s are supplied separately as the operation's attached regions (via the region driver passed to
    /// [`Context::bind`]); [`Operation::infer_output_types`] validates that the branch
    /// interfaces agree and that the predicate input is a scalar Boolean.
    #[inline]
    pub fn new() -> Self {
        Self { value_family: PhantomData }
    }
}

impl<F: Value> Default for ConditionOperation<F> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Value<Type: ConditionTypeSemantics>> Display for ConditionOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

/// Type-family predicate semantics for [`ConditionOperation`].
///
/// Conditions always branch on ordinary Boolean data. [`ArrayType`] accepts rank-zero Boolean predicates, while a
/// composite [`ArrayIrType`] accepts only its rank-zero Boolean array member. A first-class dimension describes an
/// array extent rather than Boolean data, even though its runtime representation is scalar.
pub trait ConditionTypeSemantics: Type {
    /// Returns whether this type is a valid condition predicate.
    fn is_condition_predicate(&self) -> bool;
}

impl ConditionTypeSemantics for ArrayType {
    #[inline]
    fn is_condition_predicate(&self) -> bool {
        self.is_scalar() && self.data_type().is_boolean()
    }
}

impl ConditionTypeSemantics for ArrayIrType {
    #[inline]
    fn is_condition_predicate(&self) -> bool {
        matches!(self, Self::Array(r#type) if r#type.is_condition_predicate())
    }
}

/// Validates that the two condition branch interfaces agree on their input and output boundary types and returns
/// them, so both predicate payloads share one interface contract.
fn validated_branch_interfaces<'i, T: Type>(
    region_interfaces: &'i [RegionInterface<T>],
) -> Result<(&'i RegionInterface<T>, &'i RegionInterface<T>), TypeError> {
    if region_interfaces.len() != 2 {
        return Err(TypeError::invalid(format!(
            "condition expects 2 attached regions but got {}",
            region_interfaces.len()
        )));
    }
    let true_interface = &region_interfaces[0];
    let false_interface = &region_interfaces[1];
    check_types!(@same, "condition branch input", [
        true_interface.input_types(),
        false_interface.input_types(),
    ]);
    check_types!(@same, "condition branch output", [
        true_interface.output_types(),
        false_interface.output_types(),
    ]);
    Ok((true_interface, false_interface))
}

impl<F: Value> Operation for ConditionOperation<F>
where
    F::Type: ConditionTypeSemantics,
{
    type Type = F::Type;

    #[inline]
    fn name(&self) -> &'static str {
        CONDITION_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("true"), RegionSlot::computation("false")] }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[F::Type],
        region_interfaces: &[RegionInterface<F::Type>],
    ) -> Result<Vec<Option<Vec<F::Type>>>, TypeError> {
        if region_interfaces.len() != 2 {
            return Err(TypeError::invalid(format!(
                "condition expects 2 attached regions but got {}",
                region_interfaces.len(),
            )));
        }
        if input_types.is_empty() {
            return Err(TypeError::invalid("condition expects at least one input but got 0"));
        }
        if region_interfaces.iter().all(|interface| interface.input_types() == &input_types[1..]) {
            return Ok(vec![None, None]);
        }
        let branch_input_types = input_types[1..].to_vec();
        Ok(vec![Some(branch_input_types.clone()), Some(branch_input_types)])
    }

    fn infer_output_types(
        &self,
        input_types: &[F::Type],
        region_interfaces: &[RegionInterface<F::Type>],
    ) -> Result<Vec<F::Type>, TypeError> {
        let (true_interface, _) = validated_branch_interfaces(region_interfaces)?;
        check_count!("input", input_types, true_interface.input_types().len() + 1, TypeError);
        if !input_types[0].is_condition_predicate() {
            return Err(TypeError::invalid(format!(
                "condition predicate type must be a scalar boolean, but got {}",
                input_types[0]
            )));
        }
        check_types!(@same, "condition input", [true_interface.input_types(), &input_types[1..]]);
        Ok(true_interface.output_types().to_vec())
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![
            OutputRegionProvenance { region_index: 0, output_index },
            OutputRegionProvenance { region_index: 1, output_index },
        ]
    }
}

/// Interpretation rule for [`ConditionOperation`]: extracts the concrete Boolean predicate from the first input and
/// interprets only the selected branch region over the remaining inputs (region 0 for `true` and region 1 for
/// `false`), so the untaken branch never runs.
impl<F, C> InterpretableOperation<C> for ConditionOperation<F>
where
    F: Value,
    F::Type: ConditionTypeSemantics,
    C: Domain<Type = F::Type, Value: Concretizable<bool>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        if inputs.is_empty() {
            return Err(ProgramError::MalformedProgram(
                "condition interpretation requires a predicate input".to_string(),
            ));
        }
        let (predicate, branch_inputs) = (inputs[0].concretize()?, &inputs[1..]);
        driver.interpret_region(context, if predicate { 0 } else { 1 }, branch_inputs.to_vec())
    }
}

/// Partial-evaluation override for [`ConditionOperation`], whose predicate is the operation's first input.
///
/// With a [`Known`](PartialValue::Known) predicate that the known-side context can
/// [`resolve`](Context::resolve) to a [`Constant`](crate::ValueResolution::Constant) payload whose value can be
/// concretized as a Boolean, it selects the taken branch and inlines it via
/// [`PartialEvaluationContext::inline_program`], so the condition disappears from the residual program; the inlined
/// branch is fed the remaining inputs. A known predicate that is *not* concretizable — under a staging known-side
/// context, a genuine [`Tracer`] into the outer program — cannot select a branch at
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
impl<V, O, C> PartiallyEvaluatableOperation<C> for ConditionOperation<V>
where
    V: Value + Concretizable<bool>,
    C: Context<Type = V::Type, Constant = V, Operation = O>,
    O: Operation<Type = V::Type> + From<ConditionOperation<V>> + ZeroOperationProvider<V::Type>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // The rule requests all nested-computation work through its region access (region 0 is the `true` branch and
        // region 1 the `false` branch), which keeps its bounds free of the operation family's own semantic traits.
        // Input 0 is the predicate; inputs 1.. feed both branches.
        if let PartialValue::Known(predicate) = inputs[0].value() {
            // A known predicate selects a branch only when it resolves to a program constant: under a staging
            // known-side context "known" means known to the outer program, and a genuine tracer carries no boolean
            // to branch on. A known-but-symbolic predicate — or a program constant payload that exposes no concrete
            // boolean, such as an abstract backend capture reference — keeps the conditional on both sides of the
            // split instead.
            if let Some(predicate) = context.parent().resolve(predicate).into_constant() {
                if let Ok(predicate) = predicate.concretize() {
                    let index = if predicate { 0 } else { 1 };
                    return driver.partially_evaluate_region(context, index, inputs[1..].to_vec());
                }
            }
            if inputs.iter().all(PartialEvaluationValue::is_known) {
                return context.fold_or_residualize(
                    O::from(self.clone()),
                    driver.regions().map(|region| region.to_program()).collect(),
                    inputs,
                );
            }
            return split_condition_by_knownness(context, driver, self, inputs);
        }

        // Unknown predicate: partially evaluate each branch against the input knowledge and reconcile the two
        // residual branch programs into a single rewritten condition. The recursive branch partial evaluation goes
        // through the partial-evaluation driver's split requests rather than `Program::partially_evaluate` directly, so
        // this impl carries no operation-enum semantic bounds of its own.
        //
        // Two conservative gates keep the conditional whole instead: effectful branches, because the branch folds
        // below run through the *live* known-side context and would execute or stage a branch's effects
        // speculatively (the predicate is unknown, so neither branch is selected yet); and symbolic knowns, because
        // the reconciled branch programs must embed folded known values as inline constants, which a live-trace
        // tracer cannot be.
        let true_branch = driver.region(0)?;
        let false_branch = driver.region(1)?;
        if !true_branch.effects().is_pure()
            || !false_branch.effects().is_pure()
            || context.any_known_is_symbolic(&inputs[1..])
        {
            return context.fold_or_residualize(
                O::from(self.clone()),
                vec![true_branch.to_program(), false_branch.to_program()],
                inputs,
            );
        }
        let branch_knowledge = inputs[1..].iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let true_evaluation =
            driver.partially_evaluate_program(context, driver.region(0)?, branch_knowledge.as_slice());
        let false_evaluation =
            driver.partially_evaluate_program(context, driver.region(1)?, branch_knowledge.as_slice());
        let (true_evaluation, false_evaluation) = match (true_evaluation, false_evaluation) {
            (Ok(true_evaluation), Ok(false_evaluation)) => (true_evaluation, false_evaluation),
            // A failed branch fold must not fail the whole partial evaluation: the predicate is unknown, so the
            // branch whose known subcomputation errors when evaluated speculatively (e.g., an integer division by a
            // known zero) may never run at runtime. The conditional is kept whole instead of shrunk, deferring the
            // branch's work — and its error, if that branch is ever actually taken — to runtime, which is the
            // semantics interpretation gives the original program. Both branches are pure here (the effects gate
            // above), so the partially completed folds are safe to discard.
            _ => {
                return context.fold_or_residualize(
                    O::from(self.clone()),
                    vec![true_branch.to_program(), false_branch.to_program()],
                    inputs,
                );
            }
        };

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

        let condition = ConditionOperation::new();
        let mut rewritten_inputs = Vec::with_capacity(combined_inputs.len() + 1);
        rewritten_inputs.push(inputs[0].clone());
        rewritten_inputs.extend(combined_inputs);
        context.fold_or_residualize(
            O::from(condition),
            vec![reconciled_true, reconciled_false],
            rewritten_inputs.as_slice(),
        )
    }
}

/// Bookkeeping for one branch of [`split_condition_by_knownness`]: the branch's partitioned programs, boundary
/// mappings, and residual edges.
struct ConditionBranchSplit<V: Value, O: Operation<Type = V::Type>> {
    /// Known-side program reified by partitioning the branch through a fresh staging context.
    known_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Residual-side program produced by partitioning the branch.
    residual_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Source of each residual-program input.
    residual_inputs: Vec<PartialEvaluationInput<usize>>,

    /// Source of each original branch output.
    outputs: Vec<PartialEvaluationOutput<usize>>,

    /// Per-edge local types, in edge order (feeders first, then instantiated known outputs of residual-owned slots).
    edge_types: Vec<V::Type>,

    /// Known-program output providing each edge, in edge order.
    edge_program_outputs: Vec<usize>,

    /// For each branch output, the edge ordinal carrying its folded value when the output is residual-owned but this
    /// branch folded it (the instantiation case).
    instantiated_edge_ordinals: Vec<Option<usize>>,
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
fn split_condition_by_knownness<V, O, C, D: PartialEvaluationDriver<C>>(
    context: &PartialEvaluationContext<C>,
    driver: &D,
    condition: &ConditionOperation<V>,
    inputs: &[PartialEvaluationValue<C::Value>],
) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
where
    V: Value,
    C: Context<Type = V::Type, Constant = V, Operation = O>,
    O: Operation<Type = V::Type> + From<ConditionOperation<V>> + ZeroOperationProvider<V::Type>,
{
    let true_branch = driver.region(0)?;
    let false_branch = driver.region(1)?;
    let branch_inputs = &inputs[1..];
    let branch_input_types = true_branch.input_types();
    check_count!("input", branch_inputs, branch_input_types.len(), ProgramError);
    let input_known = branch_inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
    let output_count = true_branch.output_types().len();

    // Partition each branch through its own fresh known-side context, requested through the driver so that this rule
    // carries no fresh-trace semantic bounds of its own.
    let true_partition = driver.partition_program(true_branch, input_known.as_slice())?;
    let false_partition = driver.partition_program(false_branch, input_known.as_slice())?;

    // An output is known only when both branches folded it.
    let out_known = (0..output_count)
        .map(|index| {
            matches!(true_partition.outputs().get(index), Some(PartialEvaluationOutput::Known(_)))
                && matches!(false_partition.outputs().get(index), Some(PartialEvaluationOutput::Known(_)))
        })
        .collect::<Vec<bool>>();

    // Collect each branch's residual edges: its known feeders plus the instantiated folded values of residual-owned
    // outputs.
    let collect_branch = |partition: PartitionedProgram<V, O>| -> Result<ConditionBranchSplit<V, O>, ProgramError> {
        let (known_program, residual_program, known_input_indices, residual_inputs, outputs) = partition.into_parts();
        check_count!("output", outputs, output_count, ProgramError);
        let expected_known_input_indices = input_known
            .iter()
            .enumerate()
            .filter_map(|(index, &known)| known.then_some(index))
            .collect::<Vec<_>>();
        if known_input_indices != expected_known_input_indices {
            return Err(ProgramError::MalformedProgram(format!(
                "condition branch partition reported known input indices {known_input_indices:?} but expected \
                 {expected_known_input_indices:?}",
            )));
        }
        check_count!("input", residual_program.input_ids(), residual_inputs.len(), ProgramError);

        let known_result_count =
            outputs.iter().filter(|output| matches!(output, PartialEvaluationOutput::Known(_))).count();
        let feeder_edge_count =
            residual_inputs.iter().filter(|input| matches!(input, PartialEvaluationInput::Known(_))).count();
        check_count!("output", known_program.output_ids(), known_result_count + feeder_edge_count, ProgramError);
        let known_program_output_types = known_program.output_types();

        let mut edge_types = Vec::new();
        let mut edge_program_outputs = Vec::new();
        for input in residual_inputs.iter() {
            if let PartialEvaluationInput::Known(edge) = input {
                if *edge != edge_types.len() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "condition branch partition reported residual edge {edge} out of order",
                    )));
                }
                let output = known_result_count + edge;
                let output_type = known_program_output_types.get(output).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "condition branch partition residual edge {edge} has no known-program output",
                    ))
                })?;
                edge_types.push(output_type.clone());
                edge_program_outputs.push(output);
            }
        }
        let mut instantiated_edge_ordinals = vec![None; output_count];
        for (index, output) in outputs.iter().enumerate() {
            if !out_known[index] {
                if let PartialEvaluationOutput::Known(output) = output {
                    let output_type = known_program_output_types.get(*output).ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "condition branch partition output {index} references missing known-program output \
                             {output}",
                        ))
                    })?;
                    instantiated_edge_ordinals[index] = Some(edge_types.len());
                    edge_types.push(output_type.clone());
                    edge_program_outputs.push(*output);
                }
            }
        }
        Ok(ConditionBranchSplit {
            known_program,
            residual_program,
            residual_inputs,
            outputs,
            edge_types,
            edge_program_outputs,
            instantiated_edge_ordinals,
        })
    };
    let true_split = collect_branch(true_partition)?;
    let false_split = collect_branch(false_partition)?;

    // An empty known side (no known output and no edge on either branch) means the split folds nothing; residualize
    // the condition unchanged through the default rule, with the symbolic predicate as a known feeder.
    let known_side_is_empty = !out_known.iter().any(|&known| known)
        && true_split.edge_program_outputs.is_empty()
        && false_split.edge_program_outputs.is_empty();
    if known_side_is_empty {
        return context.fold_or_residualize(
            O::from(condition.clone()),
            vec![true_branch.to_program(), false_branch.to_program()],
            inputs,
        );
    }

    // Reconciling the known branches requires each branch to produce placeholder zeros for the other branch's
    // residual edges. A type that carries an identity cannot be constructed from its type alone: a dimension needs a
    // real producer and a dynamic array needs explicit extent values. Keep the original condition residual in that
    // case; its known operands become ordinary residual inputs and no placeholder value is needed.
    if true_split
        .edge_types
        .iter()
        .chain(&false_split.edge_types)
        .any(|r#type| r#type.identities().next().is_some())
    {
        return context.fold_or_residualize(
            O::from(condition.clone()),
            vec![true_branch.to_program(), false_branch.to_program()],
            inputs,
        );
    }

    // Build each known branch over the shared `[known outputs..., true edges..., false edges...]` output signature,
    // producing typed zeros for the other branch's edge slots.
    let build_known_branch = |own: &ConditionBranchSplit<V, O>,
                              other: &ConditionBranchSplit<V, O>,
                              own_first: bool|
     -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        let mut builder = ProgramBuilder::<V, O>::new();
        let known_inputs = own
            .known_program
            .input_types()
            .into_iter()
            .map(|input_type| builder.add_input(input_type))
            .collect::<Vec<_>>();
        let known_outputs = builder.splice_program(&own.known_program, known_inputs.as_slice())?;
        let mut output_atoms = Vec::new();
        for (index, output) in own.outputs.iter().enumerate() {
            if out_known[index] {
                match output {
                    PartialEvaluationOutput::Known(output) => {
                        output_atoms.push(*known_outputs.get(*output).ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "condition branch partition references missing known-program output {output}",
                            ))
                        })?)
                    }
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
            let zeros = builder.add_instruction(O::zero_operation(edge_type.clone())?, Vec::new(), Vec::new())?;
            check_count!("output", zeros, 1, ProgramError);
            zero_atoms.push(zeros[0]);
        }
        let edge_atoms = own
            .edge_program_outputs
            .iter()
            .map(|&output| {
                known_outputs.get(output).copied().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "condition branch partition references missing edge output {output}",
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if own_first {
            output_atoms.extend(edge_atoms);
            output_atoms.extend(zero_atoms);
        } else {
            output_atoms.extend(zero_atoms);
            output_atoms.extend(edge_atoms);
        }
        let output_count = output_atoms.len();
        builder
            .build::<Vec<V>, Vec<V>>(
                output_atoms,
                vec![Placeholder; known_inputs.len()],
                vec![Placeholder; output_count],
            )?
            .into_simplified()
    };
    let known_true = build_known_branch(&true_split, &false_split, true)?;
    let known_false = build_known_branch(&false_split, &true_split, false)?;

    // Bind the known condition into the enclosing known-side context over the predicate and the known inputs.
    let known_condition = ConditionOperation::new();
    let mut known_condition_inputs = Vec::with_capacity(inputs.len());
    known_condition_inputs.push(inputs[0].clone());
    known_condition_inputs.extend(
        branch_inputs
            .iter()
            .zip(input_known.iter())
            .filter(|(_, known)| **known)
            .map(|(input, _)| input.clone()),
    );
    let known_outputs = context.fold_or_residualize(
        O::from(known_condition),
        vec![known_true, known_false],
        known_condition_inputs.as_slice(),
    )?;
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
    let needs_residual_condition = residual_output_ordinals.iter().any(Option::is_some)
        || !true_split.residual_program.effects().is_pure()
        || !false_split.residual_program.effects().is_pure();
    let residual_outputs = if needs_residual_condition {
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

            let mut spliced_inputs = Vec::with_capacity(own.residual_inputs.len());
            for input in own.residual_inputs.iter() {
                match input {
                    PartialEvaluationInput::Unknown(index) => {
                        spliced_inputs.push(unknown_input_atoms.get(*index).copied().flatten().ok_or_else(|| {
                            ProgramError::MalformedProgram(
                                "condition known-ness split saw a residual feeder for a known input".to_string(),
                            )
                        })?);
                    }
                    PartialEvaluationInput::Known(edge) => {
                        spliced_inputs.push(*own_edge_atoms.get(*edge).ok_or_else(|| {
                            ProgramError::MalformedProgram(
                                "condition known-ness split lost a residual edge".to_string(),
                            )
                        })?)
                    }
                }
            }
            let spliced_outputs = builder.splice_program(&own.residual_program, &spliced_inputs)?;

            let mut output_atoms = Vec::new();
            for (index, output) in own.outputs.iter().enumerate() {
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
        let residual_condition = ConditionOperation::new();

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
        context.residualize(
            O::from(residual_condition),
            vec![residual_true, residual_false],
            residual_condition_inputs.as_slice(),
        )?
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
) -> Result<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, ProgramError> {
    let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
    let input_atoms = combined_input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
    let branch_inputs = &input_atoms[offset..offset + evaluation.inputs.len()];
    let residual_outputs = builder.splice_program(&evaluation.program, branch_inputs)?;
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

/// Batching rule for [`ConditionOperation`]. The rule builds batched condition *structure* and binds it into the
/// parent context — interpreted eagerly under an eager parent and staged into the enclosing trace under a staging
/// parent:
///
///   - **Replicated predicate.** Both branch programs are batched at the operand batch axes via
///     [`Program::batched`](crate::Program::batched) (the batching analog of symbolic program
///     linearization), their per-output batch axes are
///     normalized to a common layout by appending staged axis-moving operations at the branch tails when they
///     disagree (a transpose for a mismatched axis, a broadcast for a replicated output paired with a batched
///     one), and one [`ConditionOperation`] over the batched branches is bound into the parent context with the
///     unbatched predicate passed through as its scalar Boolean operand. A staging parent therefore keeps one
///     `condition` operation whose branches run whole batches per batch item, while an eager parent concretizes the
///     predicate and interprets the chosen batched branch.
///   - **Batch-varying predicate.** Both pure branches are interpreted over the operand inputs and merged per batch
///     item via [`Select`]: every per-item primitive re-enters this operation
///     family's batching rules against the same active context, so the multi-operation rewrite composes for eager
///     and staging parents alike. Effectful branches are rejected because evaluating both branches would perform
///     effects that the per-item selection cannot mask.
impl<C, O, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for ConditionOperation<C::Constant>
where
    C: Context<Type = ArrayType, Operation = O>,
    <C as Domain>::Value: Concretizable<bool> + LegacyBroadcast + Transpose + Select,
    O: Operation<Type = ArrayType>
        + From<TransposeOperation>
        + From<LegacyBroadcastOperation>
        + From<SelectOperation<ArrayType>>
        + From<ConditionOperation<C::Constant>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        let Some((predicate_batch, operand_inputs)) = inputs.split_first() else {
            return Err(BatchingError::UnsupportedOperation {
                message: "cannot batch a condition operation with no predicate input".to_string(),
            });
        };
        if !predicate_batch.batch_axis().is_replicated() {
            let true_region = driver.region(0)?;
            let false_region = driver.region(1)?;
            if !true_region.effects().is_pure() || !false_region.effects().is_pure() {
                return Err(BatchingError::UnsupportedOperation {
                    message: "cannot batch a condition with a batch-varying predicate and effectful branches because \
                              observable effects cannot be selected per batch item"
                        .to_string(),
                });
            }
            // Batch-varying predicate: batch both branches item-agnostically through the region access and merge
            // their outputs per batch item via `Select`.
            return batch_condition_with_interpreter(
                context,
                predicate_batch,
                operand_inputs,
                |index, region_inputs| driver.batch_region(context, index, region_inputs),
            );
        }

        // Replicated (abstract) predicate: batch both branches at the operand batch axes with natural output axes to
        // discover which outputs each branch batches, join the two answers into one output layout — preferring the
        // true branch's natural axis when both are batched — and instantiate each branch at the joined targets so the
        // branch signatures agree. This is the two-pass shape of JAX's `_cond_batching_rule` (`batch_jaxpr` with
        // `instantiate=out_bat`). Each branch is instantiated independently through
        // `BatchingContext::align_batched_program_outputs`, which keeps a discovery program whose (normalized) natural
        // axes already equal the joined targets because an aligned replay of it would rebuild the identical program.
        let operand_axes = operand_inputs.iter().map(|input| input.batch_axis()).collect::<Vec<_>>();
        let true_region = driver.region(0)?;
        let false_region = driver.region(1)?;
        let true_program = driver.batch_program(
            context,
            true_region,
            operand_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let false_program = driver.batch_program(
            context,
            false_region,
            operand_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        check_count!("output", false_program.output_axes(), true_program.output_axes().len(), ProgramError);
        let output_axes: Vec<BatchAxis> = true_program
            .output_axes()
            .iter()
            .zip(false_program.output_axes())
            .map(|(true_axis, false_axis)| if true_axis.is_replicated() { *false_axis } else { *true_axis })
            .collect();
        let batched_true_branch = context.align_batched_program_outputs(
            driver,
            true_region,
            operand_axes.as_slice(),
            true_program,
            output_axes.as_slice(),
        )?;
        let batched_false_branch = context.align_batched_program_outputs(
            driver,
            false_region,
            operand_axes.as_slice(),
            false_program,
            output_axes.as_slice(),
        )?;

        // Stage one condition over the batched branches with the unbatched predicate passed through.
        let batched_condition = ConditionOperation::new();
        let mut staged_inputs = Vec::with_capacity(inputs.len());
        staged_inputs.push(predicate_batch.value().clone());
        staged_inputs.extend(operand_inputs.iter().map(|input| input.value().clone()));
        let outputs = context.parent().bind(
            batched_condition,
            vec![batched_true_branch, batched_false_branch],
            &staged_inputs,
        )?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| {
                let batched_type = output.r#type().into_owned();
                ArrayBatch::new(batched_type, output, axis)
            })
            .collect()
    }
}

/// Batches a condition whose predicate is *batch-varying* by replaying both attached regions over the operand inputs
/// through `batch_branch` and merging their outputs per batch item via
/// [`Select`](crate::operations::control_flow::Select). The ordinary [`SelectOperation`] batching rule aligns every
/// branch output with the predicate's mapped axis, broadcasts replicated branch outputs across the batch, and expands
/// the per-item scalar predicate across non-scalar branch output shapes. The predicate must carry a mapped batch axis;
/// the replicated case is the caller's structural staging path.
pub(crate) fn batch_condition_with_interpreter<C, P: ArrayBatchingPolicy<C>, F>(
    context: &BatchingContext<C, ArrayBatching<P>>,
    predicate_batch: &ArrayBatch<C::Value>,
    operand_inputs: &[ArrayBatch<C::Value>],
    mut batch_branch: F,
) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    C::Value: LegacyBroadcast + Transpose + Select,
    C::Operation: From<LegacyBroadcastOperation> + From<SelectOperation<ArrayType>> + From<TransposeOperation>,
    F: FnMut(usize, Vec<ArrayBatch<C::Value>>) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError>,
{
    let true_outputs = batch_branch(0, operand_inputs.to_vec())?;
    let false_outputs = batch_branch(1, operand_inputs.to_vec())?;
    check_count!("output", true_outputs, false_outputs.len(), ProgramError);
    true_outputs
        .into_iter()
        .zip(false_outputs)
        .map(|(true_output, false_output)| -> Result<ArrayBatch<C::Value>, BatchingError> {
            let mut selected = SelectOperation::<ArrayType>::new().batch(
                context,
                &crate::EmptyRegionDriver,
                &[predicate_batch.clone(), true_output, false_output],
            )?;
            check_count!("output", selected, 1, ProgramError);
            Ok(selected.remove(0))
        })
        .collect()
}

/// Composite array IR batching rule for [`ConditionOperation`].
///
/// A replicated predicate preserves one structural condition whose transformed branches explicitly thread the
/// mapped extent. A mapped predicate replays both pure branches and selects their array outputs per item. First-class
/// dimension outputs remain replicated, so the mapped-predicate path requires both branches to produce the same
/// dimension value.
impl<A, C> BatchableOperation<C, ArrayIrBatching> for ConditionOperation<ArrayIrValue<A>>
where
    A: Value<Type = ArrayType>,
    C: Context<
            Type = ArrayIrType,
            Operation: From<BroadcastOperation>
                           + From<ConditionOperation<ArrayIrValue<A>>>
                           + From<DimensionOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>
                           + OperationProjection<DimensionType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    C::Value: ValueProjection<ArrayType, Projected: LegacyBroadcast + Select + Transpose + Value<Type = ArrayType>>
        + ValueProjection<DimensionType, Projected: Value<Type = DimensionType>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected:
        From<LegacyBroadcastOperation> + From<SelectOperation<ArrayType>> + From<TransposeOperation>,
    <C::Operation as OperationProjection<DimensionType>>::Projected: From<DimensionRequirementOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        let Some((predicate, operands)) = inputs.split_first() else {
            return Err(BatchingError::UnsupportedOperation {
                message: "cannot batch a condition operation with no predicate input".to_string(),
            });
        };
        <&ArrayType>::try_from(predicate.unbatched_type())?;

        if predicate.batch_axis().is_replicated() {
            let operand_axes = operands.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>();
            let true_region = driver.region(0)?;
            let false_region = driver.region(1)?;
            let true_program = driver.batch_program(
                context,
                true_region,
                operand_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?;
            let false_program = driver.batch_program(
                context,
                false_region,
                operand_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?;
            check_count!("output", false_program.output_axes(), true_program.output_axes().len(), ProgramError);
            let output_axes = true_program
                .output_axes()
                .iter()
                .zip(false_program.output_axes())
                .map(|(true_axis, false_axis)| if true_axis.is_replicated() { *false_axis } else { *true_axis })
                .collect::<Vec<_>>();

            // Each branch is instantiated at the joined targets independently. A branch whose discovered (normalized)
            // axes already equal those targets keeps its discovery program because an aligned replay of it would
            // rebuild the identical program.
            let true_branch = context.align_batched_program_outputs(
                driver,
                true_region,
                operand_axes.as_slice(),
                true_program,
                output_axes.as_slice(),
            )?;
            let false_branch = context.align_batched_program_outputs(
                driver,
                false_region,
                operand_axes.as_slice(),
                false_program,
                output_axes.as_slice(),
            )?;

            let mut packed_inputs = Vec::with_capacity(inputs.len() + 1);
            packed_inputs.push(predicate.value().clone());
            packed_inputs.push(context.axis_extent().clone());
            packed_inputs.extend(operands.iter().map(|operand| operand.value().clone()));
            let mut outputs =
                context.parent().bind(self.clone(), vec![true_branch, false_branch], packed_inputs.as_slice())?;
            check_count!("output", outputs, output_axes.len() + 1, ProgramError);
            outputs.remove(0);
            return outputs
                .into_iter()
                .zip(output_axes)
                .map(|(output, axis)| ArrayIrBatch::new(output, axis))
                .collect();
        }

        let true_region = driver.region(0)?;
        let false_region = driver.region(1)?;
        if !true_region.effects().is_pure() || !false_region.effects().is_pure() {
            return Err(BatchingError::UnsupportedOperation {
                message: "cannot batch a condition with a batch-varying predicate and effectful branches because \
                          observable effects cannot be selected per batch item"
                    .to_string(),
            });
        }
        let true_outputs = driver.batch_region(context, 0, operands.to_vec())?;
        let false_outputs = driver.batch_region(context, 1, operands.to_vec())?;
        check_count!("output", false_outputs, true_outputs.len(), ProgramError);
        true_outputs
            .into_iter()
            .zip(false_outputs)
            .map(|(true_output, false_output)| match true_output.unbatched_type() {
                ArrayIrType::Array(_) => {
                    <&ArrayType>::try_from(false_output.unbatched_type())?;
                    let mut selected = batch_projected_operation(
                        context,
                        &SelectOperation::<ArrayType>::new(),
                        &[predicate.clone(), true_output, false_output],
                    )?;
                    check_count!("output", selected, 1, ProgramError);
                    Ok(selected.remove(0))
                }
                ArrayIrType::Dimension(_) => {
                    true_output.validate_replicated_dimension()?;
                    false_output.validate_replicated_dimension()?;
                    require_equal_dimensions(context.parent(), true_output.value(), false_output.value())?;
                    Ok(true_output)
                }
            })
            .collect()
    }
}

/// Capture-free forward-mode (JVP) rule for [`ConditionOperation`], staging **one fused** jvp `condition` as an
/// ordinary primal-enum operation over the shared builder.
///
/// The rule builds each branch's fused jvp program through its instruction-scoped differentiation driver — both
/// branches share a signature, so their compact `[primal_operands..., live_tangent_operands...] ->
/// [primal_outputs..., live_tangent_outputs...]` signatures also match with no joining or padding — and stages one
/// `condition` over the predicate primal followed by the operand primals and live tangents. Pure forward mode
/// therefore stages a single conditional and no residual plumbing.
///
/// The primal/tangent separation that reverse mode needs is deferred to partial evaluation: under the known-ness
/// split of [`Program::linearize`](crate::Program::linearize) the predicate is a known (symbolic) primal, so the
/// condition composite split (ryft's `_cond_partial_eval` analogue) separates the fused conditional into a known
/// primal condition — producing each branch's known→unknown edges with typed zero-padding for the peer's slots —
/// and a residual tangent condition over the operand tangents and those edges.
///
/// The predicate is the first operand and carries no tangent (Boolean predicates have no tangent space); the fused
/// conditional selects the same branch for both halves because they share the same primal predicate edge.
impl<C: Context<Type: ConditionTypeSemantics + DifferentiableType> + Zero<C::Value>> DifferentiableOperation<C>
    for ConditionOperation<C::Constant>
where
    C::Operation: ZeroOperationProvider<C::Type> + From<ConditionOperation<C::Constant>>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The rule requests all nested-computation work through its driver (region 0 is the `true` branch
        // and region 1 the `false` branch); the true branch's boundary is materialized for the arity checks.
        let true_branch = driver.region(0)?;
        check_count!("input", inputs, true_branch.input_types().len() + 1, ProgramError);
        let predicate_primal = inputs[0].primal().clone();
        let operands = &inputs[1..];
        let output_types = true_branch.output_types();
        let output_count = output_types.len();
        let tangent_output_count = output_types.iter().filter(|r#type| !r#type.tangent().is_zero_space()).count();

        // Build both fused jvp branches and stage one fused conditional over the predicate primal followed by the
        // operand primals and tangents.
        let fused_true = driver.jvp_program(true_branch)?;
        let fused_false = driver.jvp_program(driver.region(1)?)?;
        let fused_condition = ConditionOperation::new();
        let mut condition_operands = Vec::with_capacity(2 * operands.len() + 1);
        condition_operands.push(predicate_primal);
        condition_operands.extend(operands.iter().map(|operand| operand.primal().clone()));
        for operand in operands {
            if !operand.tangent().r#type().is_zero_space() {
                condition_operands.push(operand.tangent().clone().materialize(context)?);
            }
        }
        let outputs = context.bind(fused_condition, vec![fused_true, fused_false], &condition_operands)?;
        check_count!("output", outputs, output_count + tangent_output_count, ProgramError);

        // The fused conditional's outputs are the primal outputs followed by only the live tangent outputs. Restore
        // structural zeros at the dual boundary for zero differential spaces.
        let (primal_outputs, tangent_outputs) = outputs.split_at(output_count);
        let mut tangent_outputs = tangent_outputs.iter().cloned();
        Ok(primal_outputs
            .iter()
            .cloned()
            .zip(output_types)
            .map(|(primal, output_type)| {
                if output_type.tangent().is_zero_space() {
                    Ok(DifferentiationDual::new_with_zero_tangent(primal))
                } else {
                    DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
                }
            })
            .collect::<Result<Vec<_>, _>>()?)
    }
}

/// Partition-aware transpose rule for a *primal* input-predicate [`ConditionOperation`], forwarding to
/// [`transpose_primal_condition`]. The predicate and the per-branch residuals ride as ordinary known operands, and the
/// branch recursion happens through the instruction-scoped driver's transposition requests, so instantiating this
/// implementation for a closed operation enum introduces no recursive [`TransposableOperation`] obligation on `O`.
impl<V, O> TransposableOperation<V, O> for ConditionOperation<V>
where
    V: Value<Type: ConditionTypeSemantics + DifferentiableType>,
    O: Operation<Type = V::Type> + ZeroOperationProvider<V::Type> + From<ConditionOperation<V>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        transpose_primal_condition(context, driver, inputs, outputs).map_err(DifferentiationError::from)
    }
}

/// Partition-aware transpose rule for a *primal* [`ConditionOperation`], used when the direct reverse transposes a
/// tangent program in the primal operation family `O`. The predicate and the per-branch residuals are ordinary
/// *operands* (known values supplied through the pullback), so the rule reads them from the pullback and threads
/// them back through as known operands of a transposed condition.
///
/// The forward stages the tangent condition over `[predicate, branch_tangents..., residuals...]` with the
/// predicate and the joined residual set marked known and the branch tangents marked linear, and with both branches
/// already joined to the same input signature `[branch_tangents..., residuals...]` and output signature
/// `[branch_tangent_outputs...]`. This rule therefore:
///
///   1. Splits the operands by `operand_linear` into the known predicate (operand `0`), the leading linear run of
///      branch tangents, and the trailing known residuals.
///   2. Transposes each branch through the driver's region-transposition request, marking the branch tangent inputs
///      linear and the residual inputs known. Each transposed branch maps
///      `[branch_tangent_output_cotangents..., residuals...]` to `[branch_tangent_input_cotangents...]`; because both
///      branches shared the joined signature, their transposes share it too and form a well-typed condition.
///   3. Re-stages a primal input-predicate [`ConditionOperation`] selecting between the two transposed branches by the
///      same known predicate, over `[predicate, outputs..., residuals...]`. Its outputs are the branch-tangent
///      input cotangents.
///
/// The returned cotangents place those branch-tangent cotangents at the linear-operand positions and a structural
/// [`MaybeZero::Zero`] at the predicate and residual positions, which carry no cotangent. The branch recursion happens
/// through the instruction-scoped driver in the same operation family, so it introduces no
/// recursive [`TransposableOperation`] obligation on `O`.
///
/// # Parameters
///
///   - `operation`: Primal input-predicate condition staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-operand [`PartialValue`] knowledge. The [`Unknown`](PartialValue::Unknown) entries are the branch
///     tangents; the [`Known`](PartialValue::Known) entries carry the predicate and residual tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the condition's outputs.
pub fn transpose_primal_condition<V, O, D: TranspositionDriver<V, O>>(
    context: &mut TracingContext<V, O>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>
where
    V: Value<Type: ConditionTypeSemantics + DifferentiableType>,
    O: Operation<Type = V::Type> + ZeroOperationProvider<V::Type> + From<ConditionOperation<V>>,
{
    // A condition with no live output cotangents is a zero linear map, so every operand cotangent is zero.
    if outputs.iter().all(MaybeZero::is_zero) {
        return Ok(inputs
            .iter()
            .map(|input| {
                let input_type = input.r#type();
                MaybeZero::Zero(input_type.cotangent())
            })
            .collect());
    }

    // The rule operates on the attached branch regions through its driver (region 0 is the `true` branch and region 1
    // the `false` branch), which keeps its bounds free of the operation family's own semantic traits.
    let true_branch = driver.region(0)?;

    // Operand layout is `[predicate(known), branch_tangents(linear)..., residuals(known)...]`. The branch tangents are
    // exactly the linear operands, and the residuals are the trailing known operands after the predicate and tangents.
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let branch_input_count = true_branch.input_types().len();
    let branch_tangent_count = operand_linear.iter().filter(|&&linear| linear).count();
    let residual_count = branch_input_count.checked_sub(branch_tangent_count).ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "condition transpose found {branch_tangent_count} linear operands but its branches take only \
             {branch_input_count} inputs",
        ))
    })?;
    check_count!("input", operand_linear, 1 + branch_tangent_count + residual_count, ProgramError);

    // The predicate is operand `0` and the residuals are the trailing operands; both are known values read from the
    // pullback. The dispatch guarantees a `Known` operand carries its pullback value, so each tracer is read directly.
    let read_known = |index: usize| -> Result<Tracer<TracingContext<V, O>>, ProgramError> {
        inputs[index]
            .as_known()
            .ok_or_else(|| {
                ProgramError::MalformedProgram(format!("condition transpose operand {index} has no known value"))
            })
            .cloned()
    };
    let predicate = read_known(0)?;
    let residuals = (1 + branch_tangent_count..inputs.len()).map(read_known).collect::<Result<Vec<_>, _>>()?;

    // Transpose each branch with the branch tangents marked linear and the residual inputs marked known. Each
    // transposed branch maps `[branch_output_cotangents..., residuals...]` to `[branch_tangent_cotangents...]`.
    let mut branch_linear = vec![true; branch_tangent_count];
    branch_linear.extend(std::iter::repeat(false).take(residual_count));
    let transposed_true = driver.transpose_program(driver.region(0)?, branch_linear.as_slice())?;
    let transposed_false = driver.transpose_program(driver.region(1)?, branch_linear.as_slice())?;
    let transposed_condition = ConditionOperation::new();

    // Stage the transposed condition over `[predicate, outputs..., residuals...]`. Its outputs are the
    // branch-tangent input cotangents.
    let output_types = true_branch.output_types();
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(1 + output_types.len() + residuals.len());
    operands.push(predicate);
    for cotangent in outputs {
        operands.push(cotangent.clone().materialize(context)?);
    }
    operands.extend(residuals);
    let branch_cotangents =
        context.bind(O::from(transposed_condition), vec![transposed_true, transposed_false], operands.as_slice())?;
    check_count!("output", branch_cotangents, branch_tangent_count, ProgramError);

    // Reassemble one cotangent per operand: the predicate and residuals carry structural zeros, while the branch
    // tangents receive the transposed condition's outputs in order.
    let mut branch_cotangents = branch_cotangents.into_iter().map(MaybeZero::Value);
    let cotangents = operand_linear
        .iter()
        .zip(inputs)
        .map(|(&linear, input)| {
            if linear {
                branch_cotangents.next().unwrap()
            } else {
                let input_type = input.r#type();
                MaybeZero::Zero(input_type.cotangent())
            }
        })
        .collect();
    Ok(cotangents)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use std::borrow::Cow;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{ArrayBatch, BatchAxis, BatchingContext, BatchingTracer, batch};
    use crate::captures::CaptureReference;
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::jacobian::JacobianDifferentiate;
    use crate::differentiation::{
        DifferentiationTracer, LinearizationTracer, ReverseModeDifferentiate, jvp, linearize,
    };
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::ZeroLikeOperation;
    use crate::operations::math::{AddOperation, DivOperation, MulOperation, SinOperation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::programs::effects::Effects;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tests::CountingBatchingDriver;
    use crate::tracing::DomainTracingContext;
    use crate::tracing::Trace;
    use crate::types::{DataType, Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};

    use super::*;

    /// Builds a single-input flat program that maps its scalar `f64` input through `operation`.
    fn scalar_branch(
        operation: ArrayOperation<Array>,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let inputs = if matches!(operation, ArrayOperation::Add(_)) { vec![input, input] } else { vec![input] };
        let output = builder.add_instruction(operation, Vec::new(), inputs).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns the [`RegionInterface`] of the provided flat branch program.
    fn branch_interface(
        program: &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>,
    ) -> RegionInterface<ArrayType> {
        program.interface()
    }

    /// Builds a scalar branch that returns whether its input is greater than zero.
    fn boolean_branch() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_constant(Array::scalar(0.0));
        let output = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![input, zero])
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a single-input branch that scales its scalar input by `factor`.
    fn scalar_scale_branch(factor: f64) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let factor = builder.add_constant(Array::scalar(factor));
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, factor]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a single-input branch that scales a vector input by `factor`.
    fn vector_scale_branch(size: usize, factor: f64) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(size)])));
        let factor = builder.add_constant(Array::scalar(factor));
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, factor]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a vector-input branch that returns a replicated constant vector.
    fn constant_vector_branch(values: Vec<f64>) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(values.len())])));
        let output = builder.add_constant(Array::vector(values));
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Batches a vector-valued condition whose branches scale their input by two and three.
    fn batch_vector_condition(batch_size: usize, item_size: usize, input_values: Vec<f64>) -> ArrayBatch<Array> {
        let batched_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(batch_size), Dimension::Static(item_size)]),
        );
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(batch_size)]));
        let predicate_values = (0..batch_size).map(|index| if index == 0 { 1.0 } else { 0.0 }).collect();
        let predicate = ArrayBatch::new(
            predicate_type.clone(),
            Array::from_f64s(predicate_type, predicate_values),
            BatchAxis::new(0),
        )
        .unwrap();
        let operand =
            ArrayBatch::new(batched_type.clone(), Array::from_f64s(batched_type, input_values), BatchAxis::new(0))
                .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), batch_size);
        let mut outputs = context
            .bind(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![vector_scale_branch(item_size, 2.0), vector_scale_branch(item_size, 3.0)],
                &[BatchingTracer::new(context.clone(), predicate), BatchingTracer::new(context.clone(), operand)],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        outputs.remove(0).into_batch()
    }

    /// Applies a condition whose predicate is computed from `input`, retaining both attached regions during replay.
    fn stage_runtime_predicate_condition<V: Value<Type = ArrayType>>(input: V) -> Result<V, ProgramError>
    where
        V::DispatchDomain: Context<Type = ArrayType, Constant = Array, Operation = ArrayOperation<Array>>,
    {
        let context = input.dispatch_domain();
        let zero = context.lift(Array::scalar(0.0))?;
        let mut predicates = context.bind(
            ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan)),
            Vec::new(),
            &[input.clone(), zero],
        )?;
        let predicate = predicates.remove(0);
        let mut outputs = context.bind(
            ArrayOperation::Condition(ConditionOperation::new()),
            vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)],
            &[predicate, input],
        )?;
        Ok(outputs.remove(0))
    }

    #[test]
    fn test_condition_composite_type_contract() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = ArrayIrType::Dimension(DimensionType::new(extent.clone()));
        let array_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])));
        let branch_inputs = vec![dimension_type.clone(), array_type.clone()];
        let branch_outputs = vec![array_type.clone(), dimension_type.clone()];
        let branch_interface = RegionInterface::new(branch_inputs.clone(), branch_outputs.clone(), Effects::PURE);
        let operation = ConditionOperation::<CaptureReference<ArrayIrType>>::new();
        let mut input_types = vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))];
        input_types.extend(branch_inputs);

        assert_eq!(
            operation.infer_output_types(input_types.as_slice(), &[branch_interface.clone(), branch_interface]),
            Ok(branch_outputs),
        );
        assert_eq!(
            operation.infer_output_types(
                &[dimension_type],
                &[
                    RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE),
                    RegionInterface::new(Vec::new(), Vec::new(), Effects::PURE),
                ],
            ),
            Err(TypeError::invalid(
                "condition predicate type must be a scalar boolean, but got dimension<extent ∈ [1, 8)>".to_string(),
            )),
        );
    }

    #[test]
    fn test_condition() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::scalar(DataType::F64);
        let operation = ConditionOperation::<Array>::new();
        let true_branch = scalar_branch(ArrayOperation::Add(AddOperation::new()));
        let false_branch = scalar_branch(ArrayOperation::ZeroLike(ZeroLikeOperation::new()));
        let interfaces = vec![branch_interface(&true_branch), branch_interface(&false_branch)];

        // Operation identity, declared region slots, output provenance, and payload-free rendering.
        assert_eq!(operation.name(), CONDITION_OPERATION_NAME);
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("true"), RegionSlot::computation("false")],);
        assert_eq!(
            operation.output_region_provenance(0),
            vec![
                OutputRegionProvenance { region_index: 0, output_index: 0 },
                OutputRegionProvenance { region_index: 1, output_index: 0 },
            ],
        );
        assert_eq!(format!("{operation}"), "condition");

        // Type inference validates the branch interfaces, the predicate, and the input types, and returns the
        // branch output types.
        assert_eq!(
            operation.infer_output_types(&[predicate_type.clone(), operand_type.clone()], interfaces.as_slice()),
            Ok(vec![operand_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(&[predicate_type.clone(), operand_type.clone()], &[]),
            Err(TypeError::invalid("condition expects 2 attached regions but got 0".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(&[], interfaces.as_slice()),
            Err(TypeError::invalid("expected 2 inputs but got 0".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(&[operand_type.clone(), operand_type.clone()], interfaces.as_slice()),
            Err(TypeError::invalid("condition predicate type must be a scalar boolean, but got f64[]".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])), operand_type.clone()],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid("condition predicate type must be a scalar boolean, but got bool[2]".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(
                &[predicate_type.clone(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid(
                "condition input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string()
            )),
        );

        // Inference rejects branch interfaces with mismatched output signatures.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![input]).unwrap()[0];
        let boolean_output = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![input, zero])
            .unwrap()[0];
        let boolean_branch = builder.build(vec![boolean_output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            operation.infer_output_types(
                &[predicate_type.clone(), operand_type.clone()],
                &[branch_interface(&true_branch), branch_interface(&boolean_branch)],
            ),
            Err(TypeError::invalid(
                "condition branch output type signature mismatch: expected [f64[]] but got [bool[]]".to_string()
            )),
        );

        // Eager binding interprets the predicate-selected branch through detached region access, and interpretation
        // without a predicate input is rejected.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let predicate = |value: f64| Array::from_f64s(predicate_type.clone(), vec![value]);
        let outputs = context
            .bind(
                operation.clone(),
                vec![true_branch.clone(), false_branch.clone()],
                &[predicate(1.0), Array::scalar(4.0)],
            )
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![8.0]);
        let outputs = context
            .bind(
                operation.clone(),
                vec![true_branch.clone(), false_branch.clone()],
                &[predicate(0.0), Array::scalar(4.0)],
            )
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![0.0]);
        assert_eq!(
            operation.interpret(&context.clone(), &crate::EmptyRegionDriver, &[] as &[Array]),
            Err(ProgramError::MalformedProgram("condition interpretation requires a predicate input".to_string(),)),
        );

        // Staging imports the branch programs as attached regions of the staged instruction instead of trying to
        // concretize the staged predicate.
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = context.builder().clone();
        let staged_predicate = context.input(predicate_type.clone());
        let staged_operand = context.input(operand_type.clone());
        let outputs = context
            .stage_operation(
                operation.clone(),
                vec![true_branch.clone(), false_branch.clone()],
                &[staged_predicate.clone(), staged_operand.clone()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let builder = builder.borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), ArrayOperation::Condition(_)));
        assert_eq!(builder.instructions()[0].regions().len(), 2);
        assert_eq!(
            builder.instructions()[0].inputs(),
            &[staged_predicate.atom_id().unwrap(), staged_operand.atom_id().unwrap()],
        );
        assert_eq!(outputs[0].atom_id(), Ok(builder.instructions()[0].outputs()[0]));

        // Program rendering shows the attached branch regions at the instruction with their declared slot names.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let program_predicate = builder.add_input(predicate_type);
        let program_operand = builder.add_input(operand_type);
        let program_output = builder
            .add_instruction(
                ArrayOperation::Condition(operation),
                vec![true_region, false_region],
                vec![program_predicate, program_operand],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![program_output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[] .
                let %2:f64[] = condition %0 %1 [
                    true={
                        lambda %0:f64[] .
                        let %1:f64[] = add %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:f64[] .
                        let %1:f64[] = zero_like %0
                        in (%1)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    /// A known-symbolic predicate splits known branch results from residual branch work without dropping an
    /// effectful residual condition whose branches have no data outputs.
    #[test]
    fn test_condition_partial_evaluation_preserves_zero_output_residual_effects() {
        use crate::operations::debugging::PrintOperation;
        use crate::partial::{PartialEvaluationOutput, PartialValue};
        use crate::tracing::TracingContext;

        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::scalar(DataType::F64);
        let branch = |label| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(operand_type.clone());
            builder.add_instruction(PrintOperation::new(label), Vec::new(), vec![input]).unwrap();
            let output = builder.add_constant(Array::scalar(1.0));
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let true_branch = branch("true");
        let false_branch = branch("false");
        assert!(true_branch.partition(&[false]).unwrap().residual_program().effects().is_ordered());
        assert!(false_branch.partition(&[false]).unwrap().residual_program().effects().is_ordered());

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let operand = builder.add_input(operand_type.clone());
        let output = builder
            .add_instruction(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, operand],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = TracingContext::<Array, ArrayOperation<Array>>::new();
        let symbolic_predicate = outer.input(predicate_type);
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(symbolic_predicate), PartialValue::Unknown(operand_type)],
            )
            .unwrap();

        assert!(matches!(evaluation.outputs.as_slice(), [PartialEvaluationOutput::Known(_)]));
        assert!(evaluation.program.effects().is_ordered());
        assert_eq!(evaluation.program.output_ids().len(), 0);
        assert_eq!(evaluation.program.instructions().len(), 1);
        let residual_condition = &evaluation.program.instructions()[0];
        assert!(matches!(residual_condition.operation(), ArrayOperation::Condition(_)));
        assert_eq!(residual_condition.outputs().len(), 0);
        assert!(
            residual_condition.regions().iter().all(|&region| evaluation
                .program
                .region_ref(region)
                .unwrap()
                .effects()
                .is_ordered())
        );
    }

    /// A branch whose fold fails under an unknown predicate (here an integer division by a known zero divisor in a
    /// branch that interpretation may never take) keeps the conditional whole instead of failing partial evaluation,
    /// so the branch's error surfaces only if that branch actually runs.
    #[test]
    fn test_condition_partial_evaluation_keeps_erroring_branch_folds_behind_the_predicate() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::scalar(DataType::I32);
        let divide_branch = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(operand_type.clone());
            let one = builder.add_constant(Array::from_f64s(operand_type.clone(), vec![1.0]));
            let output = builder
                .add_instruction(ArrayOperation::Div(DivOperation::new()), Vec::new(), vec![one, input])
                .unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let identity_branch = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(operand_type.clone());
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_region(divide_branch.entry_region_ref());
        let false_region = builder.import_region(identity_branch.entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let operand = builder.add_input(operand_type.clone());
        let output = builder
            .add_instruction(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, operand],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // The zero divisor is known, so shrinking the branches would fold `1 / 0` speculatively; the rule must fall
        // back to residualizing the conditional whole.
        let knowledge = vec![
            PartialValue::Unknown(predicate_type),
            PartialValue::Known(Array::from_f64s(operand_type.clone(), vec![0.0])),
        ];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();
        assert!(matches!(evaluation.outputs.as_slice(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Condition(_)));

        // Interpreting the residual program with a false predicate takes the identity branch and never divides.
        let inputs = evaluation
            .inputs
            .iter()
            .map(|input| match input {
                PartialEvaluationInput::Unknown(_) => Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![0.0]),
                PartialEvaluationInput::Known(value) => value.clone(),
            })
            .collect::<Vec<_>>();
        let outputs = evaluation.program.interpret(inputs).unwrap();
        assert_eq!(outputs[0].elements::<i32>(), Ok(vec![0]));
    }

    #[test]
    fn test_condition_infers_output_types_through_operation_enum() {
        // Inference dispatches through the closed operation enum exactly like through the bare operation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let identity_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let operation = ArrayOperation::Condition(ConditionOperation::<Array>::new());
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64)],
                &[identity_branch.interface(), identity_branch.interface()],
            ),
            Ok(vec![ArrayType::scalar(DataType::F64)]),
        );
    }

    #[test]
    fn test_condition_region_batching_preserves_mapped_axis_sharding() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let batched_sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let batched_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
                    .with_sharding(batched_sharding)
                    .unwrap();
            let operand = ArrayBatch::new(
                batched_type.clone(),
                Array::from_f64s(batched_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                BatchAxis::new(0),
            )
            .unwrap();
            let unbatched_type = operand.unbatched_type();
            let (_, branch) =
                EagerContext::<Array, ArrayOperation<Array>>::trace(|inputs: Vec<_>| Ok(inputs), vec![unbatched_type])
                    .unwrap();
            let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));
            let predicate = ArrayBatch::replicated(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0]));

            let outputs = context
                .bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![branch.clone(), branch],
                    &[BatchingTracer::new(context.clone(), predicate), BatchingTracer::new(context.clone(), operand)],
                )
                .unwrap();

            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].batch().r#type(), Cow::Borrowed(&batched_type));
        }
    }

    #[test]
    fn test_condition_batching_stages_replicated_predicates() {
        // A replicated *abstract* condition predicate under trace-time batching cannot be concretized to pick one
        // branch (previously this surfaced a `Concretization` error), so the staged batching rule batches both
        // branch programs at the operand batch axes and stages exactly one `condition` operation over them, with the
        // unbatched predicate passed through. Interpreting the staged batched program with both concrete predicate
        // values matches the eager operational path item for item (scale by 2 when true and by 3 when false).
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let predicate_atom = builder.borrow_mut().add_input(predicate_type.clone());
        let operand_atom = builder.borrow_mut().add_input(operand_type);
        let predicate_tracer = parent.tracer(predicate_atom, None);
        let operand_tracer = parent.tracer(operand_atom, None);
        let output = batch(
            |(predicate, x)| {
                let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
                let condition = ConditionOperation::new();
                let op = ArrayOperation::Condition(condition);
                let outputs = x.context().bind(op, condition_regions, &[predicate.clone(), x.clone()])?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (predicate_tracer, operand_tracer),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let condition_count = program
            .instructions()
            .iter()
            .filter(|instruction| instruction.operation().name() == "condition")
            .count();
        assert_eq!(condition_count, 1, "{program}");
        let truthy = Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0]);
        let falsy = Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        let operand = Array::vector(vec![1.0, 4.0, 9.0]);
        assert_eq!(program.interpret((truthy, operand.clone())).unwrap().to_f64s(), vec![2.0, 8.0, 18.0]);
        assert_eq!(program.interpret((falsy, operand)).unwrap().to_f64s(), vec![3.0, 12.0, 27.0]);
    }

    /// The replicated-predicate rule discovers each branch's natural output axes before instantiating both branches at
    /// the joined layout. `AlignEachTo` stages axis movement only where a natural axis differs from a mapped target, so
    /// a branch whose discovered axes already equal the joined targets keeps its discovery program and the rule
    /// performs one structural pass for it instead of two.
    #[test]
    fn test_condition_batching_reuses_naturally_aligned_branch_programs() {
        let packed_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let truthy = Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0]);
        let falsy = Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        let operand_values = Array::from_f64s(packed_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Both branches scale the batched operand per batch item, so both discover axis 0 and the joined layout equals
        // each branch's discovered layout: the rule batches each branch exactly once.
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let operand_atom = builder.borrow_mut().add_input(packed_type.clone());
        let predicate = parent.tracer(predicate_atom, None);
        let operand = parent.tracer(operand_atom, None);
        let context = BatchingContext::new(parent, 2);
        let inputs = vec![
            ArrayBatch::replicated(predicate),
            ArrayBatch::new(packed_type.clone(), operand, BatchAxis::new(0)).unwrap(),
        ];
        let regions = vec![vector_scale_branch(3, 2.0), vector_scale_branch(3, 3.0)];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = ConditionOperation::new().batch(&context, &driver, inputs.as_slice()).unwrap();
        assert_eq!(driver.batch_program_calls(), 2);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(
                vec![outputs[0].value().atom_id().unwrap()],
                (Placeholder, Placeholder),
                Placeholder,
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[2, 3] .
                let %2:f64[2, 3] = condition %0 %1 [
                    true={
                        lambda %0:f64[2, 3] .
                        let %1:f64[] = const
                            %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[]] %1
                            %3:f64[2, 3] = mul %0 %2
                        in (%3)
                    },
                    false={
                        lambda %0:f64[2, 3] .
                        let %1:f64[] = const
                            %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[]] %1
                            %3:f64[2, 3] = mul %0 %2
                        in (%3)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            program.interpret((truthy.clone(), operand_values.clone())).unwrap().to_f64s(),
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        );
        assert_eq!(
            program.interpret((falsy.clone(), operand_values.clone())).unwrap().to_f64s(),
            vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
        );

        // A replicated false-branch output disagrees with the joined layout, so only that branch is re-batched to
        // broadcast its output across the batch: three structural passes in total.
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let operand_atom = builder.borrow_mut().add_input(packed_type.clone());
        let predicate = parent.tracer(predicate_atom, None);
        let operand = parent.tracer(operand_atom, None);
        let context = BatchingContext::new(parent, 2);
        let inputs = vec![
            ArrayBatch::replicated(predicate),
            ArrayBatch::new(packed_type.clone(), operand, BatchAxis::new(0)).unwrap(),
        ];
        let regions = vec![vector_scale_branch(3, 2.0), constant_vector_branch(vec![10.0, 20.0, 30.0])];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = ConditionOperation::new().batch(&context, &driver, inputs.as_slice()).unwrap();
        assert_eq!(driver.batch_program_calls(), 3);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(
                vec![outputs[0].value().atom_id().unwrap()],
                (Placeholder, Placeholder),
                Placeholder,
            )
            .unwrap();
        assert_eq!(
            program.interpret((truthy, operand_values.clone())).unwrap().to_f64s(),
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        );

        // The re-batched false branch broadcasts its replicated constant across the batch, so every batch item
        // receives the same constant vector.
        assert_eq!(
            program.interpret((falsy, operand_values)).unwrap().to_f64s(),
            vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
        );
    }

    #[test]
    fn test_condition_batching_normalizes_replicated_branch_output_axes() {
        // The two branches of a staged batched condition may disagree on their natural output batch axes: here the
        // true branch scales the batched operand per batch item (axis 0) while the false branch returns a replicated
        // constant (no batch axis). The staged rule normalizes the false branch by appending a broadcast at its
        // tail, so the staged condition stays well-typed and both predicate values interpret correctly per batch item.
        let mut constant_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        constant_builder.add_input(ArrayType::scalar(DataType::F64));
        let constant_output = constant_builder.add_constant(Array::scalar(7.0));
        let constant_branch = constant_builder
            .build::<Vec<Array>, Vec<Array>>(vec![constant_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let operand_atom = builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        let predicate_tracer = parent.tracer(predicate_atom, None);
        let operand_tracer = parent.tracer(operand_atom, None);
        let output = batch(
            |(predicate, x)| {
                let condition_regions = vec![scalar_scale_branch(2.0), constant_branch];
                let condition = ConditionOperation::new();
                let op = ArrayOperation::Condition(condition);
                let outputs = x.context().bind(op, condition_regions, &[predicate.clone(), x.clone()])?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (predicate_tracer, operand_tracer),
            (BatchAxis::replicated(), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let output_atom = output.atom_id().unwrap();
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let rendered = program.to_string();
        assert!(rendered.contains("broadcast"), "{rendered}");
        let truthy = Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0]);
        let falsy = Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![0.0]);
        let operand = Array::vector(vec![1.0, 4.0, 9.0]);
        assert_eq!(program.interpret((truthy, operand.clone())).unwrap().to_f64s(), vec![2.0, 8.0, 18.0]);
        assert_eq!(program.interpret((falsy, operand)).unwrap().to_f64s(), vec![7.0, 7.0, 7.0]);
    }

    /// A batch-varying predicate cannot select one branch for the whole batch, so batching runs both pure branches
    /// and merges their outputs per batch item through the `Select` batching rule.
    #[test]
    fn test_condition_batching_selects_branch_outputs_per_item_for_batch_varying_predicates() {
        let output = batch(
            |(predicate, x)| {
                let outputs = x.context().bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)],
                    &[predicate.clone(), x.clone()],
                )?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (Array::vector(vec![true, false, true]), Array::vector(vec![1.0, 4.0, 9.0])),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output.to_f64s(), vec![2.0, 12.0, 18.0]);
    }

    #[test]
    fn test_condition_batching_selects_non_scalar_outputs_per_item() {
        // The batch size differs from the per-item vector length. The Boolean `[2]` predicate must become `[2, 1]`
        // before selecting between the `[2, 3]` branch values.
        let output = batch_vector_condition(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        assert_eq!(output.value().to_f64s(), vec![2.0, 4.0, 6.0, 12.0, 15.0, 18.0]);

        // Equal batch and item sizes previously allowed trailing-axis broadcasting to select columns rather than
        // rows. Pin the row-wise result explicitly.
        let output = batch_vector_condition(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        assert_eq!(output.value().to_f64s(), vec![2.0, 4.0, 9.0, 12.0]);
    }

    #[test]
    fn test_condition_batching_aligns_replicated_and_mapped_branch_outputs() {
        let batch_size = 2;
        let item_size = 3;
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(batch_size)]));
        let predicate = ArrayBatch::new(
            predicate_type.clone(),
            Array::from_f64s(predicate_type, vec![1.0, 0.0]),
            BatchAxis::new(0),
        )
        .unwrap();
        let operand = Array::matrix(batch_size, item_size, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let operand = ArrayBatch::new(operand.r#type().into_owned(), operand, BatchAxis::new(0)).unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), batch_size);

        let outputs = context
            .bind(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![constant_vector_branch(vec![10.0, 20.0, 30.0]), vector_scale_branch(item_size, 3.0)],
                &[BatchingTracer::new(context.clone(), predicate), BatchingTracer::new(context.clone(), operand)],
            )
            .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].batch().value().to_f64s(), vec![10.0, 20.0, 30.0, 12.0, 15.0, 18.0]);
    }

    /// Effectful branches cannot be batched under a batch-varying predicate: both branches would run for the whole
    /// batch and their observable effects cannot be selected per batch item.
    #[test]
    fn test_condition_batching_rejects_batch_varying_predicates_with_effectful_branches() {
        use crate::operations::debugging::PrintOperation;

        let effectful_branch = |label| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            builder
                .add_instruction(ArrayOperation::Print(PrintOperation::new(label)), Vec::new(), vec![input])
                .unwrap();
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let result: Result<Array, BatchingError> = batch(
            |(predicate, x)| {
                let outputs = x.context().bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![effectful_branch("true"), effectful_branch("false")],
                    &[predicate.clone(), x.clone()],
                )?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (Array::vector(vec![true, false]), Array::vector(vec![1.0, 2.0])),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        let error = result.unwrap_err();
        assert!(
            error.to_string().contains(
                "cannot batch a condition with a batch-varying predicate and effectful branches because observable \
                 effects cannot be selected per batch item"
            ),
            "{error}",
        );
    }

    #[test]
    fn test_condition_linearization_replays_the_selected_branch() {
        type TestContext = EagerContext<Array, ArrayOperation<Array>>;
        type TestTracer = LinearizationTracer<TestContext>;

        for (predicate, expected_value, expected_tangent) in
            [(true, 1.4, 3.0), (false, 0.7f64.sin(), 1.5 * 0.7f64.cos())]
        {
            let (value, pushforward) = linearize(
                move |input: TestTracer| {
                    let predicate = input.context().lift(Array::from_f64s(
                        ArrayType::scalar(DataType::Boolean),
                        vec![if predicate { 1.0 } else { 0.0 }],
                    ))?;
                    let mut outputs = input.context().bind(
                        ArrayOperation::Condition(ConditionOperation::new()),
                        vec![
                            scalar_branch(ArrayOperation::Add(AddOperation::new())),
                            scalar_branch(ArrayOperation::Sin(SinOperation::new())),
                        ],
                        &[predicate, input.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(0.7),
            )
            .unwrap();
            assert_eq!(value, Array::scalar(expected_value));
            assert_eq!(pushforward.apply(Array::scalar(1.5)), Ok(Array::scalar(expected_tangent)));
        }
    }

    #[test]
    fn test_condition_jvp_preserves_zero_space_output_tangents() {
        type TestContext = EagerContext<Array, ArrayOperation<Array>>;
        type TestTracer = DifferentiationTracer<TestContext>;

        let (primal, tangent) = jvp(
            |input: TestTracer| {
                let predicate =
                    input.context().lift(Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0]))?;
                let mut outputs = input.context().bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![boolean_branch(), boolean_branch()],
                    &[predicate, input.clone()],
                )?;
                Ok(outputs.remove(0))
            },
            Array::scalar(2.0),
            Array::scalar(3.0),
        )
        .unwrap();
        assert_eq!(primal, Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0]));
        assert_eq!(tangent, Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap());
    }

    #[test]
    fn test_condition_dense_jacobians_replay_runtime_regions() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();

        let forward = context.jacobian_forward(stage_runtime_predicate_condition, Array::scalar(4.0)).unwrap();
        let reverse = context.jacobian_reverse(stage_runtime_predicate_condition, Array::scalar(4.0)).unwrap();
        assert_eq!(forward.iter_blocks().next().unwrap().value().to_f64s(), vec![2.0]);
        assert_eq!(reverse.iter_blocks().next().unwrap().value().to_f64s(), vec![2.0]);

        let forward = context.jacobian_forward(stage_runtime_predicate_condition, Array::scalar(-4.0)).unwrap();
        let reverse = context.jacobian_reverse(stage_runtime_predicate_condition, Array::scalar(-4.0)).unwrap();
        assert_eq!(forward.iter_blocks().next().unwrap().value().to_f64s(), vec![3.0]);
        assert_eq!(reverse.iter_blocks().next().unwrap().value().to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_condition_vjp_selects_runtime_branch_cotangents() {
        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |(predicate, operand)| {
                    let mut outputs = predicate.context().bind(
                        ArrayOperation::Condition(ConditionOperation::new()),
                        vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)],
                        &[predicate.clone(), operand],
                    )?;
                    Ok(outputs.remove(0))
                },
                (Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![1.0]), Array::scalar(4.0)),
            )
            .unwrap();
        let cotangents = pullback.apply(Array::scalar(5.0)).unwrap();
        assert_eq!(output.to_f64s(), vec![8.0]);
        assert!(cotangents.0.storage_bytes().is_empty());
        assert_eq!(cotangents.1.to_f64s(), vec![10.0]);

        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |(predicate, operand)| {
                    let mut outputs = predicate.context().bind(
                        ArrayOperation::Condition(ConditionOperation::new()),
                        vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)],
                        &[predicate.clone(), operand],
                    )?;
                    Ok(outputs.remove(0))
                },
                (Array::from_f64s(ArrayType::scalar(DataType::Boolean), vec![0.0]), Array::scalar(4.0)),
            )
            .unwrap();
        let cotangents = pullback.apply(Array::scalar(5.0)).unwrap();
        assert_eq!(output.to_f64s(), vec![12.0]);
        assert!(cotangents.0.storage_bytes().is_empty());
        assert_eq!(cotangents.1.to_f64s(), vec![15.0]);
    }
}
