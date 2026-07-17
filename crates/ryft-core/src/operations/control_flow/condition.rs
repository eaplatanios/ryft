use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::BooleanLike;
use crate::operations::constants::ZeroOperation;
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput,
    PartialEvaluationOutput, PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, PartitionedProgram,
};
use crate::payloads::{Captured, Input};
use crate::programs::ProgramError;
use crate::programs::builders::ProgramBuilder;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::programs::Program;
use crate::programs::regions::{OutputRegionProvenance, RegionInterface};
use crate::programs::types::{Type, TypeError};
use crate::programs::values::Value;
use crate::types::ArrayType;

/// Canonical operation name for [`ConditionOperation`].
pub const CONDITION_OPERATION_NAME: &str = "condition";

// TODO(eaplatanios): Review from here onwards.

/// [`Operation`] that evaluates one of its two attached branch [`Region`](crate::Region)s depending on a Boolean
/// predicate. Ordinary conditions use the [`Input`] predicate payload: the predicate is supplied as the first
/// operation input (a scalar Boolean input) and the remaining operation inputs are forwarded to the selected branch.
/// Linearized conditions use the [`Captured`] predicate payload: the predicate is stored in the operation payload as
/// a residual value and the operation inputs are exactly the branch input tangents or cotangents.
///
/// The branch computations are not part of this payload: they are [`Region`](crate::Region)s attached to the
/// [`Instruction`](crate::Instruction) applying the operation, in the [`region_names`](Operation::region_names)
/// order `["true", "false"]`, and semantic rules reach them through their driver-granted region access. Conditions
/// with owned branches supply the two branch [`Program`]s through the region driver passed to [`Context::bind`].
///
/// A predicate that is already known while *building* a program is naturally expressed with a plain Rust `if` that
/// chooses which operations to stage, so no `condition` operation is needed for it. A predicate that is staged as a
/// constant still lowers to a `stablehlo.if` operation whose constant predicate the backend folds away (via
/// [StableHLO canonicalization](https://openxla.org/stablehlo/generated/stablehlo_passes) and XLA's conditional
/// simplification), so `ryft` performs no predicate folding of its own.
#[derive(Clone)]
pub struct ConditionOperation<F: Value, PredicatePayload = Input> {
    /// Captured predicate for captured-predicate conditions, or `None` for input-predicate conditions.
    pub(crate) predicate: Option<F>,

    /// Marker describing where the predicate value lives.
    pub(crate) predicate_payload: PhantomData<PredicatePayload>,
}

impl<F: Value, PredicatePayload> Debug for ConditionOperation<F, PredicatePayload> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("ConditionOperation");
        if let Some(predicate) = &self.predicate {
            debug.field("predicate", predicate);
        }
        debug.finish()
    }
}

impl<F: Value> ConditionOperation<F> {
    /// Creates a new [`ConditionOperation`] whose predicate is supplied as the first operation input. The two branch
    /// [`Program`]s are supplied separately as the operation's attached regions (via the region driver passed to
    /// [`Context::bind`]); [`Operation::infer_output_types`] validates that the branch
    /// interfaces agree and that the predicate input is a scalar Boolean.
    #[inline]
    pub fn new() -> Self {
        Self { predicate: None, predicate_payload: PhantomData }
    }
}

impl<F: Value> Default for ConditionOperation<F> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Value> ConditionOperation<F, Captured> {
    /// Creates a new [`ConditionOperation`] whose predicate is captured in the operation payload rather than supplied
    /// as an operation input. The two branch [`Program`]s are supplied separately as the operation's attached regions
    /// (via the region driver passed to [`Context::bind`]).
    #[inline]
    pub fn new_captured(predicate: F) -> Self {
        Self { predicate: Some(predicate), predicate_payload: PhantomData }
    }

    /// Returns the captured Boolean predicate that selects the branch to run.
    #[inline]
    pub fn predicate(&self) -> &F {
        self.predicate.as_ref().unwrap()
    }
}

impl<F: Value<Type: BooleanLike>> Display for ConditionOperation<F, Input> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Value<Type = ArrayType>> Display for ConditionOperation<F, Captured> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

/// Validates that the two condition branch interfaces agree on their input and output boundary types and returns
/// them, so both predicate payloads share one interface contract.
fn validated_branch_interfaces<'i, T: Type>(
    region_interfaces: &'i [RegionInterface<T>],
) -> Result<(&'i RegionInterface<T>, &'i RegionInterface<T>), TypeError> {
    if region_interfaces.len() != 2 {
        return Err(TypeError {
            message: format!("condition expects 2 attached regions but got {}", region_interfaces.len()),
        });
    }
    let true_interface = &region_interfaces[0];
    let false_interface = &region_interfaces[1];
    check_types!("condition branch input", true_interface.input_types(), false_interface.input_types());
    check_types!("condition branch output", true_interface.output_types(), false_interface.output_types());
    Ok((true_interface, false_interface))
}

impl<F: Value> Operation<F::Type> for ConditionOperation<F, Input>
where
    F::Type: BooleanLike,
{
    #[inline]
    fn name(&self) -> &'static str {
        CONDITION_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[F::Type],
        region_interfaces: &[RegionInterface<F::Type>],
    ) -> Result<Vec<F::Type>, TypeError> {
        let (true_interface, _) = validated_branch_interfaces(region_interfaces)?;
        check_count!("input", input_types, true_interface.input_types().len() + 1, TypeError);
        if !input_types[0].is_scalar() || input_types[0] != input_types[0].as_boolean() {
            return Err(TypeError {
                message: format!("condition predicate type must be a scalar boolean, but got {}", input_types[0]),
            });
        }
        check_types!("condition input", true_interface.input_types(), &input_types[1..]);
        Ok(true_interface.output_types().to_vec())
    }

    #[inline]
    fn region_names(&self) -> &'static [&'static str] {
        &["true", "false"]
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![
            OutputRegionProvenance { region_index: 0, output_index },
            OutputRegionProvenance { region_index: 1, output_index },
        ]
    }
}

impl<F: Value<Type = ArrayType>> Operation<ArrayType> for ConditionOperation<F, Captured> {
    #[inline]
    fn name(&self) -> &'static str {
        CONDITION_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        let (true_interface, _) = validated_branch_interfaces(region_interfaces)?;
        check_count!("input", input_types, true_interface.input_types().len(), TypeError);
        check_types!("condition input", true_interface.input_types(), input_types);
        Ok(true_interface.output_types().to_vec())
    }

    #[inline]
    fn region_names(&self) -> &'static [&'static str] {
        &["true", "false"]
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![
            OutputRegionProvenance { region_index: 0, output_index },
            OutputRegionProvenance { region_index: 1, output_index },
        ]
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, CONDITION_OPERATION_NAME)?
            .bracketed(|operation| operation.field("predicate", self.predicate()))
    }
}

impl<F, C> InterpretableOperation<C> for ConditionOperation<F, Input>
where
    F: Value,
    F::Type: BooleanLike,
    C: Domain<Type = F::Type, Value: BooleanLike>,
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
        let (predicate, branch_inputs) = (inputs[0].boolean()?, &inputs[1..]);
        driver.interpret_region(context, if predicate { 0 } else { 1 }, branch_inputs.to_vec())
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
impl<V, O, C> PartiallyEvaluatableOperation<C> for ConditionOperation<V, Input>
where
    V: Value<Type = ArrayType> + BooleanLike,
    C: Context<Type = ArrayType, Constant = V, Operation = O>,
    O: Operation<ArrayType> + From<ConditionOperation<V>> + From<ZeroOperation<ArrayType>>,
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
            // A known predicate selects a branch only when it resolves to a concrete constant: under a staging
            // known-side context "known" means known to the outer program, and a genuine tracer carries no boolean
            // to branch on. A known-but-symbolic predicate — or a concrete constant payload that exposes no concrete
            // boolean, such as an abstract backend capture reference — keeps the conditional on both sides of the
            // split instead.
            if let Some(predicate) = context.parent().resolve(predicate).into_concrete() {
                if let Ok(predicate) = predicate.boolean() {
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
            driver.partially_evaluate_program(context, driver.region(0)?, branch_knowledge.as_slice())?;
        let false_evaluation =
            driver.partially_evaluate_program(context, driver.region(1)?, branch_knowledge.as_slice())?;

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
struct ConditionBranchSplit<V: Value<Type = ArrayType>, O: Operation<ArrayType>> {
    /// Known-side program reified by partitioning the branch through a fresh staging context.
    known_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Residual-side program produced by partitioning the branch.
    residual_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Source of each residual-program input.
    residual_inputs: Vec<PartialEvaluationInput<usize>>,

    /// Source of each original branch output.
    outputs: Vec<PartialEvaluationOutput<usize>>,

    /// Per-edge local types, in edge order (feeders first, then instantiated known outputs of residual-owned slots).
    edge_types: Vec<ArrayType>,

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
    condition: &ConditionOperation<V, Input>,
    inputs: &[PartialEvaluationValue<C::Value>],
) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
where
    V: Value<Type = ArrayType>,
    C: Context<Type = ArrayType, Constant = V, Operation = O>,
    O: Operation<ArrayType> + From<ConditionOperation<V>> + From<ZeroOperation<ArrayType>>,
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
            let zeros = builder.add_instruction(ZeroOperation::new(edge_type.clone()), Vec::new(), Vec::new())?;
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

/// Partial-evaluation override for a [`Captured`]-predicate [`ConditionOperation`], whose predicate is stored in the
/// operation payload rather than supplied as an input. Because the predicate is not part of the inputs offered to
/// partial evaluation, this defers to the default fold-or-residualize behavior of [`Program::partially_evaluate`].
impl<F, C> PartiallyEvaluatableOperation<C> for ConditionOperation<F, Captured>
where
    F: Value<Type = ArrayType>,
    C: Context<Type = ArrayType>,
    C::Operation: From<ConditionOperation<F, Captured>>,
{
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiationTracer, LinearizationTracer};
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::ZeroLikeOperation;
    use crate::operations::math::{AddOperation, SinOperation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::tracing::DomainTracingContext;
    use crate::tracing_v2::{ArrayOperation, ForwardModeDifferentiate};
    use crate::types::{DataType, Shape, Size};

    use super::*;

    /// Builds a single-input flat program that maps its scalar `f64` input through `operation`.
    fn scalar_branch(
        operation: ArrayOperation<TestArray>,
    ) -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let inputs = if matches!(operation, ArrayOperation::Add(_)) { vec![input, input] } else { vec![input] };
        let output = builder.add_instruction(operation, Vec::new(), inputs).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Returns the [`RegionInterface`] of the provided flat branch program.
    fn branch_interface(
        program: &Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>>,
    ) -> RegionInterface<ArrayType> {
        program.interface()
    }

    /// Builds a scalar branch that returns whether its input is greater than zero.
    fn boolean_branch() -> Program<TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_constant(TestArray::scalar(0.0));
        let output = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![input, zero])
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_condition() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::scalar(DataType::F64);
        let operation = ConditionOperation::<TestArray>::new();
        let true_branch = scalar_branch(ArrayOperation::Add(AddOperation));
        let false_branch = scalar_branch(ArrayOperation::ZeroLike(ZeroLikeOperation));
        let interfaces = vec![branch_interface(&true_branch), branch_interface(&false_branch)];

        // Operation identity, declared region slots, output provenance, and payload-free rendering.
        assert_eq!(operation.name(), CONDITION_OPERATION_NAME);
        assert_eq!(operation.region_names(), &["true", "false"]);
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
            Err(TypeError { message: "condition expects 2 attached regions but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[], interfaces.as_slice()),
            Err(TypeError { message: "expected 2 inputs but got 0".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[operand_type.clone(), operand_type.clone()], interfaces.as_slice()),
            Err(TypeError { message: "condition predicate type must be a scalar boolean, but got f64[]".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)])), operand_type.clone()],
                interfaces.as_slice(),
            ),
            Err(TypeError {
                message: "condition predicate type must be a scalar boolean, but got bool[2]".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(
                &[predicate_type.clone(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))],
                interfaces.as_slice(),
            ),
            Err(TypeError {
                message: "condition input type signature mismatch: expected [f64[]] but got [f64[2]]".to_string(),
            }),
        );

        // Inference rejects branch interfaces with mismatched output signatures.
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(ZeroLikeOperation, Vec::new(), vec![input]).unwrap()[0];
        let boolean_output = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::GreaterThan), Vec::new(), vec![input, zero])
            .unwrap()[0];
        let boolean_branch = builder.build(vec![boolean_output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            operation.infer_output_types(
                &[predicate_type.clone(), operand_type.clone()],
                &[branch_interface(&true_branch), branch_interface(&boolean_branch)],
            ),
            Err(TypeError {
                message: "condition branch output type signature mismatch: expected [f64[]] but got [bool[]]"
                    .to_string(),
            }),
        );

        // Eager binding interprets the predicate-selected branch through detached region access, and interpretation
        // without a predicate input is rejected.
        let context = EagerContext::<TestArray, ArrayOperation<TestArray>>::new();
        let predicate = |value: f64| TestArray::new(predicate_type.clone(), vec![value]);
        let outputs = context
            .bind(
                operation.clone(),
                vec![true_branch.clone(), false_branch.clone()],
                &[predicate(1.0), TestArray::scalar(4.0)],
            )
            .unwrap();
        assert_eq!(outputs[0].values, vec![8.0]);
        let outputs = context
            .bind(
                operation.clone(),
                vec![true_branch.clone(), false_branch.clone()],
                &[predicate(0.0), TestArray::scalar(4.0)],
            )
            .unwrap();
        assert_eq!(outputs[0].values, vec![0.0]);
        assert_eq!(
            operation.interpret(&context.clone(), &crate::EmptyRegionDriver, &[] as &[TestArray]),
            Err(ProgramError::MalformedProgram("condition interpretation requires a predicate input".to_string(),)),
        );

        // Staging imports the branch programs as attached regions of the staged instruction instead of trying to
        // concretize the staged predicate.
        let context = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::new();
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
        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
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

    #[test]
    fn test_condition_linearization_replays_the_selected_branch() {
        type TestContext = EagerContext<TestArray, ArrayOperation<TestArray>>;
        type TestTracer = LinearizationTracer<TestContext>;

        for (predicate, expected_value, expected_tangent) in
            [(true, 1.4, 3.0), (false, 0.7f64.sin(), 1.5 * 0.7f64.cos())]
        {
            let (value, pushforward) = TestContext::new()
                .linearize(
                    move |input: TestTracer| {
                        let predicate = input.context().lift(TestArray::new(
                            ArrayType::scalar(DataType::Boolean),
                            vec![if predicate { 1.0 } else { 0.0 }],
                        ))?;
                        let mut outputs = input.context().bind(
                            ArrayOperation::Condition(ConditionOperation::new()),
                            vec![
                                scalar_branch(ArrayOperation::Add(AddOperation)),
                                scalar_branch(ArrayOperation::Sin(SinOperation)),
                            ],
                            &[predicate, input.clone()],
                        )?;
                        Ok(outputs.remove(0))
                    },
                    TestArray::scalar(0.7),
                )
                .unwrap();
            assert_eq!(value, TestArray::scalar(expected_value));
            assert_eq!(pushforward.apply(TestArray::scalar(1.5)), Ok(TestArray::scalar(expected_tangent)));
        }
    }

    #[test]
    fn test_condition_jvp_preserves_zero_space_output_tangents() {
        type TestContext = EagerContext<TestArray, ArrayOperation<TestArray>>;
        type TestTracer = DifferentiationTracer<TestContext>;

        let (primal, tangent) = TestContext::new()
            .jvp(
                |input: TestTracer| {
                    let predicate =
                        input.context().lift(TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]))?;
                    let mut outputs = input.context().bind(
                        ArrayOperation::Condition(ConditionOperation::new()),
                        vec![boolean_branch(), boolean_branch()],
                        &[predicate, input.clone()],
                    )?;
                    Ok(outputs.remove(0))
                },
                TestArray::scalar(2.0),
                TestArray::scalar(3.0),
            )
            .unwrap();
        assert_eq!(primal, TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]));
        assert_eq!(tangent, TestArray::new(ArrayType::scalar(DataType::Zero), vec![0.0]));
    }

    /// A known-symbolic predicate splits known branch results from residual branch work without dropping an
    /// effectful residual condition whose branches have no data outputs.
    #[test]
    fn test_partially_evaluate_condition_preserves_zero_output_residual_effects() {
        use crate::operations::debugging::PrintOperation;
        use crate::partial::{PartialEvaluationOutput, PartialValue};
        use crate::tracing::TracingContext;

        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let operand_type = ArrayType::scalar(DataType::F64);
        let branch = |label| {
            let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
            let input = builder.add_input(operand_type.clone());
            builder.add_instruction(PrintOperation::new(label), Vec::new(), vec![input]).unwrap();
            let output = builder.add_constant(TestArray::scalar(1.0));
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let true_branch = branch("true");
        let false_branch = branch("false");
        assert!(true_branch.partition(&[false]).unwrap().residual_program().effects().is_ordered());
        assert!(false_branch.partition(&[false]).unwrap().residual_program().effects().is_ordered());

        let mut builder = ProgramBuilder::<TestArray, ArrayOperation<TestArray>>::new();
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
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = TracingContext::<TestArray, ArrayOperation<TestArray>>::new();
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
}
