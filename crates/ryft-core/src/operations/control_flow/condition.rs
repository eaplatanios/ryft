//! Contains the `condition` control-flow operation: [`ConditionOperation`], which evaluates one of its two attached
//! branch [`Region`](crate::Region)s depending on a scalar Boolean predicate, together with its interpretation,
//! partial-evaluation, batching, forward-mode differentiation, and transposition rules. This is the analogue of
//! [JAX's `lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) and lowers to
//! [StableHLO's `if`](https://openxla.org/stablehlo/spec#if).

use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::arrays::{
    ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrType,
    ArrayType, DimensionType, DimensionValue, Sharding,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchedProgram, BatchingContext, BatchingDriver, BatchingError,
    ProgramBatchingOutputAxesPolicy, batch_projected_operation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, CotangentDestinationKind, CotangentDestinations, DifferentiableOperation, DifferentiableType,
    DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy,
    ResidualZeroProvider, TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::arithmetic::AddOperation;
use crate::operations::assertions::Assert;
use crate::operations::collectives::parallel_vary::{ManualVariationAlignment, PARALLEL_VARY_OPERATION_NAME};
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::control_flow::select::{Select, SelectOperation};
use crate::operations::control_flow::{refine_output_types, region_input_mismatch, validate_output_identities};
use crate::operations::differentiation::stop_gradient::StopGradient;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::{
    Broadcast, BroadcastOperation, DynamicBroadcast, DynamicBroadcastOperation,
};
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::operations::references::ReferenceNewOperation;
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput,
    PartialEvaluationOutput, PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation, PartitionedProgram,
};
use crate::programs::{
    AtomId, CalleeRegionDriver, Concretizable, EmptyRegionDriver, InputRegionProvenance, MaybeZero, Operation,
    OperationBoundaryPruning, OperationProjection, OperationProvider, OutputRegionProvenance, Program, ProgramBuilder,
    ProgramError, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceRoot, RegionInterface, RegionLiveness,
    RegionSlot, Type, TypeError, Typed, Value, ValueProjection, discharge_positional_region_operation,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this since it is mostly vibe coded.

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
/// constant remains a conditional during ordinary staging, and the backend can fold its `stablehlo.if` away via
/// [StableHLO canonicalization](https://openxla.org/stablehlo/generated/stablehlo_passes) and XLA's conditional
/// simplification. Partial evaluation specializes a concrete known predicate by inlining the selected branch; a known
/// symbolic predicate retains conditional execution of both the known and residual branch programs.
///
/// Batching a mapped predicate evaluates both pure branches and selects their outputs per item. The primal computations
/// of inactive branches still execute and can fail, but gradient propagation into their inputs is stopped before branch
/// evaluation so inactive non-finite derivatives do not contaminate the selected derivative. The selected branch must
/// still have a defined derivative.
#[derive(Clone)]
pub struct ConditionOperation<T: Type> {
    /// Type universe of the predicate and of the attached branch regions.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> Copy for ConditionOperation<T> {}

impl<T: Type> ConditionOperation<T> {
    /// Creates a new [`ConditionOperation`]. The two branch [`Program`]s are supplied separately as the operation's
    /// attached regions (via the region driver passed to [`Context::bind`]); [`Operation::infer_output_types`]
    /// validates that the branch interfaces agree and that the predicate input is a scalar Boolean.
    #[inline]
    pub fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl<T: Type> Debug for ConditionOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("ConditionOperation").finish()
    }
}

impl<T: Type> Default for ConditionOperation<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// A condition carries no attributes besides its type-universe marker (its branches are attached regions), so every two
// conditions of one universe are identical. These implementations are written by hand so that they do not require the
// marker's type to be hashable or totally comparable.
impl<T: Type> PartialEq for ConditionOperation<T> {
    #[inline]
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<T: Type> Eq for ConditionOperation<T> {}

impl<T: Type> Hash for ConditionOperation<T> {
    #[inline]
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

impl<T: ConditionType> Display for ConditionOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: ConditionType> Operation for ConditionOperation<T> {
    type Type = T;

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
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        check_count!("region", region_interfaces, 2, TypeError);
        if input_types.is_empty() {
            return Err(TypeError::invalid(format!(
                "`{CONDITION_OPERATION_NAME}` expects at least one input but got 0"
            )));
        }
        if region_interfaces.iter().all(|interface| interface.input_types() == &input_types[1..]) {
            return Ok(vec![None, None]);
        }
        let branch_input_types = input_types[1..].to_vec();
        Ok(vec![Some(branch_input_types.clone()), Some(branch_input_types)])
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        check_count!("region", region_interfaces, 2, TypeError);
        let true_interface = &region_interfaces[0];
        let false_interface = &region_interfaces[1];
        check_types!(@same, format!("`{CONDITION_OPERATION_NAME}` branch input"), [
            true_interface.input_types(),
            false_interface.input_types(),
        ]);
        check_types!(@same, format!("`{CONDITION_OPERATION_NAME}` branch output"), [
            true_interface.output_types(),
            false_interface.output_types(),
        ]);
        check_count!("input", input_types, true_interface.input_types().len() + 1, TypeError);
        if !input_types[0].is_condition_predicate() {
            return Err(TypeError::invalid(format!(
                "`{}` predicate type must be a scalar boolean, but got `{}`",
                CONDITION_OPERATION_NAME, input_types[0],
            )));
        }
        for output_type in true_interface.output_types() {
            output_type.validate_condition_output(&input_types[0])?;
        }
        // Branch value inputs are validated with the directional declared-vs-actual `Type::is_refined_by` relation
        // rather than strict type equality, as for `while` and `scan`, so actual inputs that carry metadata the branch
        // input types leave unspecified (e.g., the normalized shardings of concrete backend array types) or static
        // extents within the bounds of dynamic branch input dimensions are accepted. Reference inputs must equal their
        // branch input types (refer to `region_input_mismatch`). The outputs are the branch output types refined by the
        // facts that the inputs establish, except for identities that an output defines.
        for (index, (branch_input_type, input_type)) in
            true_interface.input_types().iter().zip(&input_types[1..]).enumerate()
        {
            if let Some(relation) = region_input_mismatch(branch_input_type, input_type) {
                return Err(TypeError::invalid(format!(
                    "`{CONDITION_OPERATION_NAME}` input {} has type `{input_type}`, which {relation} its branch input \
                     type `{branch_input_type}`",
                    index + 1,
                )));
            }
        }
        // A branch may forward any of its reference inputs, and which one is only visible in its body, so reference
        // outputs keep their declared types (refer to `refine_output_types`).
        let output_types = refine_output_types(
            true_interface.input_types(),
            &input_types[1..],
            true_interface.output_types(),
            |_| None,
        )?;
        validate_output_identities(CONDITION_OPERATION_NAME, input_types, output_types.as_slice())?;
        Ok(output_types)
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        if region_index < 2 {
            InputRegionProvenance::Input { index: input_index + 1 }
        } else {
            InputRegionProvenance::None
        }
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![
            OutputRegionProvenance { region_index: 0, output_index },
            OutputRegionProvenance { region_index: 1, output_index },
        ]
    }

    fn prune_boundary(
        &self,
        input_count: usize,
        used_outputs: &[bool],
        regions: &mut dyn RegionLiveness,
    ) -> Result<Option<OperationBoundaryPruning<Self>>, ProgramError> {
        // Both branches keep one shared boundary, whose region inputs the instruction inputs after the predicate supply
        // positionally, so an instruction input is kept when either branch uses the region input it supplies (as in
        // JAX's `_cond_dce_rule`). The predicate is always kept.
        let mut used_inputs = vec![false; input_count.saturating_sub(1)];
        for region_index in 0..2 {
            let branch_inputs = regions.used_region_inputs(region_index, used_outputs)?;
            for (used, branch_used) in used_inputs.iter_mut().zip(branch_inputs) {
                *used |= branch_used;
            }
        }
        Ok(Some(OperationBoundaryPruning {
            operation: *self,
            kept_inputs: std::iter::once(true).chain(used_inputs).collect(),
            kept_outputs: used_outputs.to_vec(),
        }))
    }
}

// A condition's branches mirror its input list after the leading predicate, and its results are each branch's own
// outputs, which is exactly the positionally forwarding shape the shared structured rewrite serves. Both branches
// therefore receive one shared state boundary: every root either branch touches enters, and only the roots one of them
// mutates are published back, so a condition whose branches merely read keeps its source boundary unchanged.
impl<C, P> ReferenceDischargeableOperation<C, P> for ConditionOperation<C::Type>
where
    C: Context<Type: ConditionType, Operation: From<ConditionOperation<C::Type>>>,
    C::Type: From<P::Referent>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        discharge_positional_region_operation(self, context, driver, inputs, 1, |_| *self)
    }
}

// Interpretation rule for [`ConditionOperation`]: extracts the concrete Boolean predicate from the first input and
// interprets only the selected branch region over the remaining inputs (region 0 for `true` and region 1 for
// `false`), so the untaken branch never runs.
impl<C> InterpretableOperation<C> for ConditionOperation<C::Type>
where
    C: Domain<Type: ConditionType, Value: Concretizable<bool>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        if inputs.is_empty() {
            return Err(ProgramError::MalformedProgram(format!(
                "`{CONDITION_OPERATION_NAME}` interpretation requires a predicate input"
            )));
        }
        let (predicate, branch_inputs) = (inputs[0].concretize()?, &inputs[1..]);
        driver.interpret_region(context, if predicate { 0 } else { 1 }, branch_inputs.to_vec())
    }
}

// Partial-evaluation override for [`ConditionOperation`], whose predicate is the operation's first input.
//
// With a [`Known`](PartialValue::Known) predicate that the known-side context can
// [`resolve`](Context::resolve) to a [`Constant`](crate::ValueResolution::Constant) payload whose value can be
// concretized as a Boolean, it selects the taken branch and inlines it via
// [`PartialEvaluationContext::inline_program`], so the condition disappears from the residual program; the inlined
// branch is fed the remaining inputs. A known predicate that is *not* concretizable — under a staging known-side
// context, a genuine [`Tracer`] into the outer program — cannot select a branch at
// partial-evaluation time; the condition is instead split by `split_condition_by_knownness` into a *known*
// condition bound in the enclosing known-side context (so known branch work stays behind the conditional instead of
// being staged speculatively for both branches) and a *residual* condition over the unknown work, connected by
// per-branch residual edges.
//
// With an [`Unknown`](PartialValue::Unknown) predicate no known branch work can be hoisted at all — there is no
// predicate to select which branch's work would run — so the condition must survive whole. It is nonetheless
// *shrunk*: each branch is partially evaluated against the input knowledge (inputs `1..`), folding away each
// branch's known subcomputation, and the two residual branch programs are reconciled into a single rewritten
// `condition` emitted through the active context. Because the two branches generally need different residual
// inputs, the rewritten condition takes the *concatenation* of the true branch's residual inputs followed by the
// false branch's; the reconciled true branch consumes the first half and the false branch the second half, leaving
// the other half unused so both branches share one input signature. A branch residual input fed by a folded known
// value (a [`PartialEvaluationInput::Known`]) is propagated outward as a fresh known trace value, and one fed by an
// unknown branch input (a [`PartialEvaluationInput::Unknown`] of branch input `k`) maps back to condition input
// `k + 1`.
impl<O, C> PartiallyEvaluatableOperation<C> for ConditionOperation<C::Type>
where
    C: Context<Constant: Concretizable<bool>, Operation = O>,
    O: Operation<Type = C::Type>
        + From<ConditionOperation<C::Type>>
        + OperationProvider<C::Type, ZeroOperation<C::Type>, Operation = O>,
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
            if let Some(predicate) = context.parent().resolve(predicate).into_constant()
                && let Ok(predicate) = predicate.concretize()
            {
                let index = if predicate { 0 } else { 1 };
                return driver.partially_evaluate_region(context, index, inputs[1..].to_vec());
            }
            if inputs.iter().all(PartialEvaluationValue::is_known) {
                return context.fold_or_residualize(
                    O::from(*self),
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
        // tracer cannot be. Residualizing the whole conditional records that later ordered effects must remain
        // residual if either branch has ordered effects. Pure arithmetic or pure views do not impose that restriction;
        // analyzing the mutually exclusive branches does not change the active context's ordering state.
        let true_branch = driver.region(0)?;
        let false_branch = driver.region(1)?;
        if !true_branch.effects().classes().is_empty()
            || !false_branch.effects().classes().is_empty()
            || context.any_known_is_symbolic(&inputs[1..])
        {
            return context.fold_or_residualize(
                O::from(*self),
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
                    O::from(*self),
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

// Batching binds conditional structure into the parent context for a replicated predicate and selects both branches'
// candidate outputs per item for a mapped predicate:
//
//   - **Replicated predicate.** Both branch programs are batched at the batch axes of the non-predicate inputs via
//     [`Program::batched`](crate::Program::batched) (the batching analog of symbolic program linearization), their
//     per-output batch axes are normalized to a common layout by appending staged axis-moving operations at the
//     branch tails when they disagree (a transpose for a mismatched axis, a broadcast for a replicated output paired
//     with a batched one), and one [`ConditionOperation`] over the batched branches is bound into the parent context
//     with the unbatched predicate passed through as its scalar Boolean input. A staging parent therefore keeps one
//     `condition` operation whose selected branch runs the whole batch, while an eager parent concretizes the
//     predicate and interprets the chosen batched branch.
//   - **Batch-varying predicate.** Both pure branches are batched over gated non-predicate inputs via
//     `driver.batch_region`, and their outputs are merged per batch item via [`Select`]. Each input `x` enters the
//     `true` branch as `select(predicate, x, stop_gradient(x))` and the `false` branch with the two select candidates
//     swapped, so primal values are unchanged but an inactive branch contributes no derivatives, and in particular no
//     non-finite ones (the fix in JAX's `_cond_batching_rule`). Every per-item primitive re-enters this operation
//     family's batching rules against the same active context, so the multi-operation rewrite composes for eager and
//     staging parents alike. Effectful branches are rejected because evaluating both branches would perform effects
//     that the per-item selection cannot mask.
impl<C, O, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for ConditionOperation<ArrayType>
where
    C: Context<Type = ArrayType, Operation = O>,
    <C as Domain>::Value: Broadcast + Transpose + Select + StopGradient + ManualVariationAlignment<ArrayType>,
    O: Operation<Type = ArrayType>
        + From<TransposeOperation>
        + From<BroadcastOperation>
        + From<SelectOperation<ArrayType>>
        + From<ConditionOperation<ArrayType>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        let Some((predicate_batch, branch_inputs)) = inputs.split_first() else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("cannot batch a `{CONDITION_OPERATION_NAME}` operation with no predicate input"),
            });
        };
        if !predicate_batch.batch_axis().is_replicated() {
            let true_region = driver.region(0)?;
            let false_region = driver.region(1)?;
            if !true_region.effects().classes().is_empty() || !false_region.effects().classes().is_empty() {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "cannot batch a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate and effectful \
                         branches because observable effects cannot be selected per batch item",
                    ),
                });
            }
            if true_region.effects().has_deferred_work() || false_region.effects().has_deferred_work() {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "cannot batch a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate and branches that \
                         carry deferred work because transformation obligations cannot be selected per batch item",
                    ),
                });
            }
            // Preserve primal inputs while severing the derivative path into each inactive branch. Selecting only
            // branch outputs would send zero cotangents through inactive non-finite derivatives and produce NaNs.
            let stopped_inputs = branch_inputs
                .iter()
                .map(|input| {
                    ArrayBatch::new(input.value().stop_gradient()?, input.batch_axis())?
                        .with_ragged_axes(input.ragged_axes().to_vec())
                })
                .collect::<Result<Vec<_>, BatchingError>>()?;
            // `Context::bind` does not align manual variation, so each selection first aligns the predicate with its
            // candidates, which may vary over manual mesh axes that the predicate does not vary over. This stages
            // `parallel_vary` on the Boolean predicate, which carries no tangent, or on a candidate that varies less
            // than the predicate, whose transition then owns the adjoint. Values without manual variation pass
            // through unchanged.
            let select = |on_true: &ArrayBatch<<C as Domain>::Value>, on_false: &ArrayBatch<<C as Domain>::Value>| {
                let candidates = [predicate_batch, on_true, on_false];
                let values = candidates.iter().map(|batch| batch.value().clone()).collect::<Vec<_>>();
                let aligned_inputs = ManualVariationAlignment::align_manual_variation(&values)?
                    .into_iter()
                    .zip(candidates)
                    .map(|(value, batch)| {
                        ArrayBatch::new(value, batch.batch_axis())?.with_ragged_axes(batch.ragged_axes().to_vec())
                    })
                    .collect::<Result<Vec<_>, BatchingError>>()?;
                let (mut selected, _) = SelectOperation::<ArrayType>::new()
                    .batch(context, &EmptyRegionDriver, &aligned_inputs)?
                    .into_parts();
                check_count!("output", selected, 1, ProgramError);
                Ok::<_, BatchingError>(selected.remove(0))
            };
            let batch_branch = |branch_index: usize| {
                let gated_inputs = branch_inputs
                    .iter()
                    .zip(&stopped_inputs)
                    .map(|(input, stopped)| {
                        let (on_true, on_false) = if branch_index == 0 { (input, stopped) } else { (stopped, input) };
                        select(on_true, on_false)
                    })
                    .collect::<Result<Vec<_>, BatchingError>>()?;
                driver.batch_region(context, branch_index, gated_inputs)
            };
            let true_outputs = batch_branch(0)?;
            let false_outputs = batch_branch(1)?;
            check_count!("output", true_outputs, false_outputs.len(), ProgramError);
            return Ok(true_outputs
                .iter()
                .zip(&false_outputs)
                .map(|(true_output, false_output)| select(true_output, false_output))
                .collect::<Result<Vec<_>, BatchingError>>()?
                .into());
        }

        // Replicated (abstract) predicate: batch both branches at the batch axes of the non-predicate inputs with
        // natural output axes to discover which outputs each branch batches, join the two answers into one output
        // layout — preferring the true branch's natural axis when both are batched — and instantiate each branch at the
        // joined targets so the branch signatures agree. This is the two-pass shape of JAX's `_cond_batching_rule`
        // (`batch_jaxpr` with `instantiate=out_bat`). Each branch is instantiated independently through
        // `BatchingContext::align_batched_program_outputs`, which keeps a discovery program whose (normalized) natural
        // axes already equal the joined targets because an aligned replay of it would rebuild the identical program.
        let branch_input_axes = branch_inputs.iter().map(|input| input.batch_axis()).collect::<Vec<_>>();
        let true_region = driver.region(0)?;
        let false_region = driver.region(1)?;
        let true_program = driver.batch_program(
            context,
            true_region,
            branch_input_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        let false_program = driver.batch_program(
            context,
            false_region,
            branch_input_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        check_count!("output", false_program.output_axes(), true_program.output_axes().len(), ProgramError);
        let output_axes: Vec<BatchAxis> = true_program
            .output_axes()
            .iter()
            .zip(false_program.output_axes())
            .map(|(true_axis, false_axis)| if true_axis.is_replicated() { *false_axis } else { *true_axis })
            .collect();
        let mut branches = Vec::with_capacity(2);
        for (region, program) in [(true_region, true_program), (false_region, false_program)] {
            branches.push(context.align_batched_program_outputs(
                driver,
                region,
                &branch_input_axes,
                program,
                &output_axes,
            )?);
        }

        // Stage one condition over the batched branches with the unbatched predicate passed through.
        let batched_condition = ConditionOperation::new();
        let mut staged_inputs = Vec::with_capacity(inputs.len());
        staged_inputs.push(predicate_batch.value().clone());
        staged_inputs.extend(branch_inputs.iter().map(|input| input.value().clone()));
        let outputs = context.parent().bind(batched_condition, branches, &staged_inputs)?;
        check_count!("output", outputs, output_axes.len(), ProgramError);
        Ok(outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output, axis))
            .collect::<Result<Vec<_>, _>>()?
            .into())
    }
}

// A replicated predicate preserves one structural condition whose transformed branches explicitly thread the mapped
// extent. A mapped predicate batches both pure branches via `driver.batch_region` and selects their array outputs per
// item. Each array input `x` is gated before batching, entering the `true` branch as
// `select(predicate, x, stop_gradient(x))` and the `false` branch with the two select candidates swapped, so an
// inactive branch contributes no non-finite derivatives (the fix in JAX's `_cond_batching_rule`); dimension and
// reference inputs pass through unchanged. First-class dimension outputs remain replicated, so the mapped-predicate
// path requires both branches to produce the same dimension value.
impl<C> BatchableOperation<C, ArrayIrBatchingPolicy> for ConditionOperation<ArrayIrType>
where
    C: Context<
            Type = ArrayIrType,
            Operation: From<DynamicBroadcastOperation>
                           + From<ConditionOperation<ArrayIrType>>
                           + From<ConstantOperation<DimensionValue>>
                           + From<DimensionSizeOperation>
                           + OperationProjection<ArrayType>,
        >,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: DimensionSize
        + Assert
        + DynamicBroadcast
        + ValueProjection<
            ArrayType,
            Projected: Broadcast
                           + Select
                           + StopGradient
                           + Transpose
                           + ManualVariationAlignment<ArrayType>
                           + Value<Type = ArrayType>,
        > + ValueProjection<DimensionType, Projected: Compare<C::Value>>,
    <C::Operation as OperationProjection<ArrayType>>::Projected:
        From<BroadcastOperation> + From<SelectOperation<ArrayType>> + From<TransposeOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let Some((predicate, branch_inputs)) = inputs.split_first() else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("cannot batch a `{CONDITION_OPERATION_NAME}` operation with no predicate input"),
            });
        };
        <&ArrayType>::try_from(&predicate.unbatched_type())?;

        if predicate.batch_axis().is_replicated() {
            let branch_input_axes = branch_inputs.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>();
            let true_region = driver.region(0)?;
            let false_region = driver.region(1)?;
            let true_program = driver.batch_program(
                context,
                true_region,
                branch_input_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?;
            let false_program = driver.batch_program(
                context,
                false_region,
                branch_input_axes.as_slice(),
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
            let mut branches = Vec::with_capacity(2);
            for (region, program) in [(true_region, true_program), (false_region, false_program)] {
                branches.push(context.align_batched_program_outputs(
                    driver,
                    region,
                    &branch_input_axes,
                    program,
                    &output_axes,
                )?);
            }

            let mut packed_inputs = Vec::with_capacity(inputs.len() + 1);
            packed_inputs.push(predicate.value().clone());
            packed_inputs.push(context.axis_extent().clone());
            packed_inputs.extend(branch_inputs.iter().map(|input| input.value().clone()));
            let mut outputs = context.parent().bind(*self, branches, packed_inputs.as_slice())?;
            check_count!("output", outputs, output_axes.len() + 1, ProgramError);
            outputs.remove(0);
            return Ok(outputs
                .into_iter()
                .zip(output_axes)
                .map(|(output, axis)| ArrayIrBatch::new(output, axis))
                .collect::<Result<Vec<_>, _>>()?
                .into());
        }

        // A batch-varying predicate lowers to running both branches and selecting per item, so no branch may touch a
        // reference in any mode (a read in the untaken branch would still be ordered against the taken branch's
        // writes, and a local allocation is a state effect of its own). A branch that merely forwards a reference it
        // never accesses is fine. This check runs ahead of the general purity check so that the diagnostic names the
        // actual cause.
        let true_region = driver.region(0)?;
        let false_region = driver.region(1)?;
        let true_analysis = true_region.reference_analysis(0).map_err(ProgramError::from)?;
        let false_analysis = false_region.reference_analysis(0).map_err(ProgramError::from)?;
        for analysis in [&true_analysis, &false_analysis] {
            let allocates = analysis.roots().any(|root| matches!(root, ReferenceRoot::Allocation { .. }));
            if !analysis.access_modes().is_empty() || allocates {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "cannot batch a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate whose branches \
                         access references because select lowering runs both branches and reference effects cannot be \
                         selected per batch item",
                    ),
                });
            }
        }
        if !true_region.effects().classes().is_empty() || !false_region.effects().classes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "cannot batch a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate and effectful branches \
                     because observable effects cannot be selected per batch item",
                ),
            });
        }
        if true_region.effects().has_deferred_work() || false_region.effects().has_deferred_work() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "cannot batch a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate and branches that \
                     carry deferred work because transformation obligations cannot be selected per batch item",
                ),
            });
        }
        // Array inputs keep their primal values in both branches but gradients enter only the selected one.
        // Dimensions and untouched references describe shared structural state and pass through unchanged.
        let stopped_inputs = branch_inputs
            .iter()
            .map(|input| {
                if matches!(input.unbatched_type(), ArrayIrType::Array(_)) {
                    let stopped =
                        ValueProjection::<ArrayType>::into_projected(input.value().clone())?.stop_gradient()?;
                    ArrayIrBatch::new(ValueProjection::<ArrayType>::from_projected(stopped), input.batch_axis())?
                        .with_ragged_axes(input.ragged_axes().to_vec())
                } else {
                    Ok(input.clone())
                }
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;
        // `Context::bind` does not align manual variation, so each selection first aligns the predicate with its
        // array candidates, which may vary over manual mesh axes that the predicate does not vary over. This stages
        // `parallel_vary` on the Boolean predicate, which carries no tangent, or on a candidate that varies less than
        // the predicate, whose transition then owns the adjoint. Values without manual variation pass through
        // unchanged.
        let select = |on_true: &ArrayIrBatch<C::Value>, on_false: &ArrayIrBatch<C::Value>| {
            let candidates = [predicate, on_true, on_false];
            let values = candidates.iter().map(|batch| batch.value().clone()).collect::<Vec<_>>();
            let aligned_inputs = ManualVariationAlignment::<ArrayIrType>::align_manual_variation(&values)?
                .into_iter()
                .zip(candidates)
                .map(|(value, batch)| {
                    ArrayIrBatch::new(value, batch.batch_axis())?.with_ragged_axes(batch.ragged_axes().to_vec())
                })
                .collect::<Result<Vec<_>, BatchingError>>()?;
            let (mut selected, _) =
                batch_projected_operation(context, &SelectOperation::<ArrayType>::new(), &aligned_inputs)?.into_parts();
            check_count!("output", selected, 1, ProgramError);
            Ok::<_, BatchingError>(selected.remove(0))
        };
        let batch_branch = |branch_index: usize| {
            let gated_inputs = branch_inputs
                .iter()
                .zip(&stopped_inputs)
                .map(|(input, stopped)| {
                    if !matches!(input.unbatched_type(), ArrayIrType::Array(_)) {
                        return Ok(input.clone());
                    }
                    let (on_true, on_false) = if branch_index == 0 { (input, stopped) } else { (stopped, input) };
                    select(on_true, on_false)
                })
                .collect::<Result<Vec<_>, BatchingError>>()?;
            driver.batch_region(context, branch_index, gated_inputs)
        };
        let true_outputs = batch_branch(0)?;
        let false_outputs = batch_branch(1)?;
        check_count!("output", false_outputs, true_outputs.len(), ProgramError);
        Ok(true_outputs
            .into_iter()
            .zip(false_outputs)
            .enumerate()
            .map(|(index, (true_output, false_output))| match true_output.unbatched_type() {
                ArrayIrType::Array(_) => {
                    <&ArrayType>::try_from(&false_output.unbatched_type())?;
                    select(&true_output, &false_output)
                }
                ArrayIrType::Dimension(_) => {
                    true_output.validate_replicated_dimension()?;
                    false_output.validate_replicated_dimension()?;
                    ValueProjection::<DimensionType>::into_projected(true_output.value().clone())?
                        .compare(
                            &ValueProjection::<DimensionType>::into_projected(false_output.value().clone())?,
                            ComparisonDirection::Equal,
                        )?
                        .assert(
                            "branch dimensions must agree",
                            &[("true", true_output.value().clone()), ("false", false_output.value().clone())],
                        )?;
                    Ok(true_output)
                }
                ArrayIrType::Reference(_) => {
                    // A reference output is never selected: both branches must forward the same region input, and the
                    // batch of the instruction input that supplies it then passes through unchanged with the axis its
                    // referent fixes. The root must belong to the branch's own input boundary, so a root of some other
                    // region fails loudly instead of indexing the branch inputs with a foreign region input position.
                    let true_root = true_analysis.output_roots().get(index).copied().flatten();
                    let false_root = false_analysis.output_roots().get(index).copied().flatten();
                    match (true_root, false_root) {
                        (
                            Some(ReferenceRoot::RegionInput { region: true_root_region, input_index: true_input }),
                            Some(ReferenceRoot::RegionInput { region: false_root_region, input_index: false_input }),
                        ) if true_root_region == true_region.id()
                            && false_root_region == false_region.id()
                            && true_input == false_input
                            && true_input < branch_inputs.len() =>
                        {
                            Ok(branch_inputs[true_input].clone())
                        }
                        _ => Err(BatchingError::UnsupportedOperation {
                            message: format!(
                                "cannot batch a `{CONDITION_OPERATION_NAME}` with a batch-varying predicate whose \
                                 reference output {index} does not forward the same region input in both branches",
                            ),
                        }),
                    }
                }
            })
            .collect::<Result<Vec<_>, BatchingError>>()?
            .into())
    }
}

// Forward-mode (JVP) rule for [`ConditionOperation`]. A shared primal/tangent context stages one fused condition;
// separate contexts stage a primal condition that records residuals and a tangent condition that consumes them.
//
// The rule builds each branch's fused jvp program through its instruction-scoped differentiation driver — both
// branches share a signature, so their compact `[primal_inputs..., live_tangent_inputs...] ->
// [primal_outputs..., live_tangent_outputs...]` signatures also match with no joining or padding — and stages one
// `condition` over the predicate primal followed by the primals and live tangents of the branch inputs. Pure forward
// mode therefore stages a single conditional and no residual plumbing.
//
// Separate linearization partitions each branch through the differentiation driver and joins their residual boundaries.
// The primal condition produces the selected branch's residuals and placeholders for the peer's slots; the tangent
// condition consumes only its selected branch's residuals. Zeroable placeholders use live input geometry, and
// non-zeroable placeholders forward a known input of the identical type. Partial evaluation uses the same
// reconstruction helper, retaining the original condition when it cannot construct a safe placeholder.
//
// The predicate is the first input and carries no tangent (Boolean predicates have no tangent space); the fused
// conditional selects the same branch for both halves because they share the same primal predicate edge.
impl<C: Context<Type: ConditionType + DifferentiableType> + Zero<C::Value>> DifferentiableOperation<C>
    for ConditionOperation<C::Type>
where
    C::Operation: ResidualZeroProvider<C::Type, Operation = C::Operation> + From<ConditionOperation<C::Type>>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The rule requests all nested-computation work through its driver (region 0 is the `true` branch
        // and region 1 the `false` branch); the true branch's boundary is materialized for the arity checks.
        let true_branch = driver.region(0)?;
        check_count!("input", inputs, true_branch.input_types().len() + 1, ProgramError);
        let predicate_primal = inputs[0].primal().clone();
        let branch_inputs = &inputs[1..];
        let output_types = true_branch.output_types();
        let output_count = output_types.len();

        // The instruction inputs after the predicate map onto the branch region inputs positionally, so each branch's
        // activity mask is the liveness of their tangents: an input with a live tangent is active, while a structural
        // zero, a plumbing reference input (a captured or inactive reference reaching the branch at any region input
        // position), and a zero-space input are inactive and receive no tangent input. An output tangent is live when
        // it depends on an active input in either branch. The others are structural zeros, so the branch derivatives
        // are projected to the live output tangents and linearization never sees a tangent that depends on no tangent
        // input.
        let activity = branch_inputs
            .iter()
            .map(|input| input.is_tangent_active() && !input.tangent().is_zero())
            .collect::<Vec<_>>();
        let primal_input_count = branch_inputs.len();
        let input_indices = activity
            .iter()
            .enumerate()
            .filter_map(|(index, &active)| active.then_some(index))
            .collect::<Vec<_>>();
        let false_branch = driver.region(1)?;
        let true_jvp = driver.jvp_program(true_branch, &input_indices)?;
        let false_jvp = driver.jvp_program(false_branch, &input_indices)?;
        let tangent_slots = true_branch.tangent_output_mask(&input_indices)?;
        let tangent_inputs = (primal_input_count..true_jvp.input_count()).collect::<Vec<_>>();
        let mut true_depends = true_jvp.output_dependence(&tangent_inputs)?.into_iter().skip(output_count);
        let mut false_depends = false_jvp.output_dependence(&tangent_inputs)?.into_iter().skip(output_count);
        let output_activity = tangent_slots
            .iter()
            // Both branch iterators yield one entry per tangent slot, so neither may be skipped by short-circuiting.
            .map(|&has_slot| has_slot && (true_depends.next().unwrap() | false_depends.next().unwrap()))
            .collect::<Vec<_>>();
        let live_tangent_slots = (0..output_count)
            .filter(|&index| tangent_slots[index])
            .enumerate()
            .filter(|(_, index)| output_activity[*index])
            .map(|(slot, _)| slot)
            .collect::<Vec<_>>();
        let tangent_output_count = live_tangent_slots.len();
        let live_input_count = input_indices.len();
        let mut condition_inputs = vec![predicate_primal];
        condition_inputs.extend(branch_inputs.iter().map(|input| input.primal().clone()));
        for (input, &active) in branch_inputs.iter().zip(&activity) {
            if active {
                let primal = context.primal_to_tangent(input.primal().clone())?;
                condition_inputs.push(C::Operation::materialize_zero_from_residual_sources(
                    context.tangent(),
                    input.tangent().clone(),
                    std::iter::once(&primal),
                )?);
            }
        }
        let outputs = if std::ptr::eq(context.primal(), context.tangent()) {
            let jvp_outputs = (0..output_count)
                .chain(live_tangent_slots.iter().map(|&slot| output_count + slot))
                .collect::<Vec<_>>();
            let branches = [true_jvp, false_jvp]
                .map(|program| {
                    if jvp_outputs.iter().copied().eq(0..program.output_count()) {
                        Ok(program)
                    } else {
                        program.with_outputs(&jvp_outputs).map(Arc::new)
                    }
                })
                .into_iter()
                .collect::<Result<Vec<_>, ProgramError>>()?;
            context
                .primal()
                .bind(ConditionOperation::new(), CalleeRegionDriver::new(&branches), &condition_inputs)?
        } else {
            let mut partitions = Vec::with_capacity(2);
            let mut branch_input_types = true_branch.input_types();
            for branch in [true_branch, false_branch] {
                let (primal, tangent, residual_count) = driver.linearize_program(branch, &input_indices)?.into_parts();
                if partitions.is_empty() {
                    branch_input_types.extend(tangent.input_types().into_iter().take(live_input_count));
                }
                partitions.push(PartitionedProgram::from_parts(
                    Arc::unwrap_or_clone(primal),
                    tangent.with_outputs(&live_tangent_slots)?,
                    (0..primal_input_count).collect(),
                    (primal_input_count..primal_input_count + live_input_count)
                        .map(PartialEvaluationInput::Unknown)
                        .chain((0..residual_count).map(PartialEvaluationInput::Known))
                        .collect(),
                    (0..output_count)
                        .map(PartialEvaluationOutput::Known)
                        .chain((0..tangent_output_count).map(PartialEvaluationOutput::Unknown))
                        .collect(),
                ));
            }
            let input_known = (0..primal_input_count + live_input_count)
                .map(|index| index < primal_input_count)
                .collect::<Vec<_>>();
            let false_partition = partitions.pop().unwrap();
            let true_partition = partitions.pop().unwrap();
            reconstruct_partitioned_condition(
                &branch_input_types,
                output_count + tangent_output_count,
                &condition_inputs,
                &input_known,
                true_partition,
                false_partition,
                |builder, edge_type, known_inputs| {
                    // A non-zeroable edge type (e.g., a first-class dimension, a reference, a token, or an `F8E8M0FNU`
                    // array) has no typed zero to stand in for the peer branch's edge, so its placeholder forwards a
                    // known input of the identical type instead. The placeholder is a dead output of the untaken
                    // branch, so any value of that type preserves the semantics; when no such input exists, the shared
                    // residual boundary cannot be constructed. Zeroable edges instead use the residual-zero protocol,
                    // which reads their live geometry from the known inputs.
                    if edge_type.validate_zero().is_err() {
                        return Ok(known_inputs
                            .iter()
                            .copied()
                            .find(|input| builder.atoms()[input.index()].r#type().as_ref() == edge_type));
                    }
                    let placeholder_context = TracingContext::<C::Constant, C::Operation>::new();
                    let sources = known_inputs
                        .iter()
                        .map(|input| placeholder_context.input(builder.atoms()[input.index()].r#type().into_owned()))
                        .collect::<Vec<_>>();
                    let placeholder = match C::Operation::materialize_zero_from_residual_sources(
                        &placeholder_context,
                        MaybeZero::Zero(edge_type.clone()),
                        &sources,
                    ) {
                        Ok(placeholder) => placeholder,
                        Err(ProgramError::UnsupportedOperation { .. }) => return Ok(None),
                        Err(error) => return Err(error),
                    };
                    let placeholder_program =
                        placeholder_context.builder().borrow().clone().build::<Vec<C::Constant>, Vec<C::Constant>>(
                            vec![placeholder.atom_id()?],
                            vec![Placeholder; known_inputs.len()],
                            vec![Placeholder],
                        )?;
                    let placeholders = builder.splice_program(&placeholder_program, known_inputs)?;
                    check_count!("output", placeholders, 1, ProgramError);
                    Ok(Some(placeholders[0]))
                },
                |operation, programs, inputs| context.primal().bind(operation, programs, inputs),
                |operation, programs, inputs| {
                    let tangent_inputs = inputs
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            if index == 0 || index > live_input_count {
                                context.primal_to_tangent(value.clone()).map_err(ProgramError::from)
                            } else {
                                Ok(value.clone())
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    context.tangent().bind(operation, programs, &tangent_inputs)
                },
            )?
            .ok_or_else(|| ProgramError::UnsupportedOperation {
                message: format!(
                    "`{CONDITION_OPERATION_NAME}` linearization cannot construct a shared residual boundary for its \
                     branches",
                ),
            })?
        };
        check_count!("output", outputs, output_count + tangent_output_count, ProgramError);

        let (primal_outputs, tangent_outputs) = outputs.split_at(output_count);
        let mut tangent_outputs = tangent_outputs.iter().cloned();
        primal_outputs
            .iter()
            .cloned()
            .zip(output_activity)
            .map(|(primal, active)| {
                if active {
                    DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
                } else {
                    DifferentiationDual::new_with_zero_tangent(primal)
                }
            })
            .collect()
    }
}

// Partition-aware transpose rules for a *primal* [`ConditionOperation`], forwarding to [`transpose_primal_condition`].
// The predicate and the per-branch residuals ride as ordinary known inputs, and the branch recursion happens through
// the instruction-scoped driver's transposition requests, so instantiating these implementations for a closed
// operation enum introduces no recursive [`TransposableOperation`] obligation on `O`. The two type universes differ
// only in where input cotangents go, so each implementation carries only the bounds its destinations need.
//
// The array universe has no reference types, so no input carries a cotangent reference and every needed cotangent is
// returned as a value.
impl<V, O> TransposableOperation<V, O> for ConditionOperation<ArrayType>
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + ResidualZeroProvider<ArrayType, Operation = O>
        + From<AddOperation<ArrayType>>
        + From<ConditionOperation<ArrayType>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        let branch = driver.region(0)?;
        check_count!("input", inputs, 1 + branch.input_types().len(), ProgramError);
        check_count!("output", outputs, branch.output_types().len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let cotangents =
            CotangentDestinations::without_references(accumulators.iter().map(CotangentAccumulator::is_needed));
        let contributions = transpose_primal_condition(context, driver, inputs, outputs, &cotangents)?;
        check_count!("input", contributions, accumulators.len(), ProgramError);
        for (accumulator, contribution) in accumulators.iter().zip(contributions) {
            accumulator.accumulate(context, contribution)?;
        }
        Ok(())
    }
}

// Reference inputs accumulate through the enclosing context's cotangent references, which are resolved (and allocated
// on first use when their state cotangent is live) before the shared rule passes them into both transposed branches.
impl<V, O> TransposableOperation<V, O> for ConditionOperation<ArrayIrType>
where
    V: Value<Type = ArrayIrType>,
    O: Operation<Type = ArrayIrType>
        + ResidualZeroProvider<ArrayIrType, Operation = O>
        + From<AddOperation<ArrayIrType>>
        + From<ConditionOperation<ArrayIrType>>
        + From<ReferenceNewOperation<ArrayType, ArrayIrType>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        let branch = driver.region(0)?;
        check_count!("input", inputs, 1 + branch.input_types().len(), ProgramError);
        check_count!("output", outputs, branch.output_types().len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let cotangents = context.cotangent_destinations(driver, inputs, accumulators)?;
        let contributions = transpose_primal_condition(context, driver, inputs, outputs, &cotangents)?;
        check_count!("input", contributions, accumulators.len(), ProgramError);
        for (accumulator, contribution) in accumulators.iter().zip(contributions) {
            accumulator.accumulate(context, contribution)?;
        }
        Ok(())
    }
}

/// Type-family predicate semantics for [`ConditionOperation`].
///
/// [`ArrayType`] accepts rank-zero Boolean predicates, while a composite [`ArrayIrType`] accepts only its rank-zero
/// Boolean array member. A first-class dimension describes an array extent rather than Boolean data, even though its
/// runtime representation is scalar, and a reference is a mutable state handle rather than a predicate value.
///
/// Inside a manual region, a predicate that varies over manual mesh axes lets devices take different branches. Such a
/// condition is accepted only when its outputs are typed accordingly: every branch output must vary over each manual
/// axis that the predicate varies over (refer to [`validate_condition_output`](Self::validate_condition_output)). An
/// invariant predicate keeps every device on the same branch. As for a `while` loop with a varying predicate, keeping
/// collectives out of branches that devices may take differently is the program's responsibility. This includes
/// collectives that transposition introduces: a branch that applies `parallel_vary` to an invariant differentiable
/// input transposes into a mesh sum inside that branch, so such inputs should be varied before the condition instead.
pub trait ConditionType: Type {
    /// Returns whether this type is a valid condition predicate.
    fn is_condition_predicate(&self) -> bool;

    /// Validates that a branch output of this type is well-typed under a predicate of type `predicate`: because
    /// devices whose predicates differ may take different branches, the output must vary over every manual mesh axis
    /// that the predicate varies over.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if the output lacks one of the predicate's varying manual axes, or if it cannot record
    /// manual variation at all (e.g., a first-class dimension) while the predicate varies.
    fn validate_condition_output(&self, predicate: &Self) -> Result<(), TypeError>;
}

impl ConditionType for ArrayType {
    #[inline]
    fn is_condition_predicate(&self) -> bool {
        self.is_scalar() && self.data_type().is_boolean()
    }

    fn validate_condition_output(&self, predicate: &Self) -> Result<(), TypeError> {
        let Some(predicate_axes) = predicate.sharding().map(Sharding::varying_manual_axes) else {
            return Ok(());
        };
        let output_axes = self.sharding().map(Sharding::varying_manual_axes);
        if predicate_axes.iter().all(|axis| output_axes.is_some_and(|output_axes| output_axes.contains(axis))) {
            return Ok(());
        }
        Err(TypeError::invalid(format!(
            "`{CONDITION_OPERATION_NAME}` output `{self}` must vary over every manual axis that the predicate \
             `{predicate}` varies over, because devices may take different branches; insert \
             `{PARALLEL_VARY_OPERATION_NAME}` on the branch outputs",
        )))
    }
}

impl ConditionType for ArrayIrType {
    #[inline]
    fn is_condition_predicate(&self) -> bool {
        matches!(self, Self::Array(r#type) if r#type.is_condition_predicate())
    }

    fn validate_condition_output(&self, predicate: &Self) -> Result<(), TypeError> {
        let Self::Array(predicate) = predicate else {
            return Ok(());
        };
        match self {
            Self::Array(output) => output.validate_condition_output(predicate),
            Self::Reference(output) => output.referent().validate_condition_output(predicate),
            Self::Dimension(_) => {
                if predicate.sharding().is_none_or(|sharding| sharding.varying_manual_axes().is_empty()) {
                    return Ok(());
                }
                Err(TypeError::invalid(format!(
                    "`{CONDITION_OPERATION_NAME}` output `{self}` cannot record manual variation, so it cannot be \
                     produced under the varying predicate `{predicate}`",
                )))
            }
        }
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

/// Splits a `condition` with a known-but-symbolic predicate into a *known* condition bound in
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
    condition: &ConditionOperation<V::Type>,
    inputs: &[PartialEvaluationValue<C::Value>],
) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
where
    V: Value,
    C: Context<Type = V::Type, Constant = V, Operation = O>,
    O: Operation<Type = V::Type>
        + From<ConditionOperation<V::Type>>
        + OperationProvider<V::Type, ZeroOperation<V::Type>, Operation = O>,
{
    let true_branch = driver.region(0)?;
    let false_branch = driver.region(1)?;
    let branch_inputs = &inputs[1..];
    let branch_input_types = true_branch.input_types();
    check_count!("input", branch_inputs, branch_input_types.len(), ProgramError);
    let input_known = branch_inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
    // Partition each branch through its own fresh known-side context, requested through the driver so that this rule
    // carries no fresh-trace semantic bounds of its own. Unlike the branches' derived forward-mode and transposed
    // programs, a partition is not retained by the branch region's transform cache: it carries known outputs that are
    // values of the live parent context, so the branch and the known-ness mask alone do not determine it.
    let true_partition = driver.partition_program(context, true_branch, input_known.as_slice())?;
    let false_partition = driver.partition_program(context, false_branch, input_known.as_slice())?;

    if let Some(outputs) = reconstruct_partitioned_condition(
        &branch_input_types,
        true_branch.output_types().len(),
        inputs,
        &input_known,
        true_partition,
        false_partition,
        |builder, edge_type, _known_inputs| {
            // Ordinary specialization does not invent values for identity-bearing edges or non-zeroable types.
            if edge_type.identities().next().is_some() || edge_type.validate_zero().is_err() {
                return Ok(None);
            }
            let zeros = builder.add_instruction(
                O::provide(ZeroOperation::new(edge_type.clone()), &[])?,
                Vec::new(),
                Vec::new(),
                None,
            )?;
            check_count!("output", zeros, 1, ProgramError);
            Ok(Some(zeros[0]))
        },
        |operation, programs, inputs| context.fold_or_residualize(operation, programs, inputs),
        |operation, programs, inputs| context.residualize(operation, programs, inputs),
    )? {
        return Ok(outputs);
    }
    context.fold_or_residualize(O::from(*condition), vec![true_branch.to_program(), false_branch.to_program()], inputs)
}

/// Rebuilds both halves of a conditional over one shared residual signature: a known condition bound through
/// `bind_known` and a residual condition bound through `bind_residual`. Unit-returning splits retain residual effects
/// and deferred work without binding an empty known condition.
///
/// Returns `None`, so that the caller keeps the original condition intact, when:
///
///   - either branch partition feeds a known reference value into its residual program or embeds a reference-typed
///     constant in its known or residual program, because splitting the branches must not expose a reference through an
///     edge or a captured constant without accounting for its identity,
///   - a value-returning split contains no known work, or
///   - `build_placeholder` cannot construct a placeholder for one of the peer branch's residual edges.
///
/// Both known branches are constructed before either `bind_known` or `bind_residual` can stage work into the caller's
/// context, so returning `None` never leaves partially staged work behind.
///
/// # Parameters
///
///   - `branch_input_types`: Types of the branch region inputs (i.e., the condition inputs after the predicate).
///   - `output_count`: Number of outputs of the original condition, which both partitions must report.
///   - `inputs`: Condition inputs, starting with the predicate, in the caller's value representation.
///   - `input_known`: Known-ness mask over the branch inputs, which both partitions were split with.
///   - `true_partition`: Partition of the `true` branch into its known and residual programs.
///   - `false_partition`: Partition of the `false` branch into its known and residual programs.
///   - `build_placeholder`: Stages a placeholder of the given edge type for one of the peer branch's residual edges
///     into the detached builder of a known branch, given that branch's known input atoms. Returning `None` keeps the
///     original condition intact.
///   - `bind_known`: Binds the known condition over the predicate and the known inputs into the caller's known-side
///     context and returns its outputs (i.e., the known outputs followed by both branches' residual edges).
///   - `bind_residual`: Binds the residual condition over the predicate, the unknown inputs, and both branches'
///     residual edges, and returns the residual outputs.
fn reconstruct_partitioned_condition<V, O, Input, PlaceholderBuild, KnownBind, ResidualBind>(
    branch_input_types: &[V::Type],
    output_count: usize,
    inputs: &[Input],
    input_known: &[bool],
    true_partition: PartitionedProgram<V, O>,
    false_partition: PartitionedProgram<V, O>,
    mut build_placeholder: PlaceholderBuild,
    mut bind_known: KnownBind,
    mut bind_residual: ResidualBind,
) -> Result<Option<Vec<Input>>, ProgramError>
where
    V: Value,
    O: Operation<Type = V::Type>
        + From<ConditionOperation<V::Type>>
        + OperationProvider<V::Type, ZeroOperation<V::Type>, Operation = O>,
    Input: Clone,
    PlaceholderBuild: FnMut(&mut ProgramBuilder<V, O>, &V::Type, &[AtomId]) -> Result<Option<AtomId>, ProgramError>,
    KnownBind: FnMut(O, Vec<Program<V, O, Vec<V>, Vec<V>>>, &[Input]) -> Result<Vec<Input>, ProgramError>,
    ResidualBind: FnMut(O, Vec<Program<V, O, Vec<V>, Vec<V>>>, &[Input]) -> Result<Vec<Input>, ProgramError>,
{
    let branch_inputs = &inputs[1..];
    // Reject reference-typed known feeders and executable reference constants. A reference feeder has no typed zero for
    // the other branch's edge slot, and splitting branches must not expose a reference through either an edge or a
    // captured constant without accounting for its identity.
    if [&true_partition, &false_partition].into_iter().any(|partition| {
        partition.known_reference_inputs().next().is_some()
            || [partition.known_program(), partition.residual_program()].into_iter().any(|program| {
                program.entry_region_ref().computation_regions().any(|region| {
                    region.atoms().iter().any(|atom| atom.as_constant().is_some() && atom.r#type().is_reference())
                })
            })
    }) {
        return Ok(None);
    }

    // An output is known only when both branches folded it.
    let output_known = (0..output_count)
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
                "`{CONDITION_OPERATION_NAME}` branch partition reported known input indices {known_input_indices:?} \
                 but expected {expected_known_input_indices:?}",
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
                        "`{CONDITION_OPERATION_NAME}` branch partition reported residual edge {edge} out of order",
                    )));
                }
                let output = known_result_count + edge;
                let output_type = known_program_output_types.get(output).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{CONDITION_OPERATION_NAME}` branch partition residual edge {edge} has no \
                         known-program output",
                    ))
                })?;
                edge_types.push(output_type.clone());
                edge_program_outputs.push(output);
            }
        }
        let mut instantiated_edge_ordinals = vec![None; output_count];
        for (index, output) in outputs.iter().enumerate() {
            if !output_known[index]
                && let PartialEvaluationOutput::Known(output) = output
            {
                let output_type = known_program_output_types.get(*output).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{CONDITION_OPERATION_NAME}` branch partition output {index} references missing \
                         known-program output {output}",
                    ))
                })?;
                instantiated_edge_ordinals[index] = Some(edge_types.len());
                edge_types.push(output_type.clone());
                edge_program_outputs.push(*output);
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

    // An empty known side (no outputs, edges, effects, or deferred work on either branch) folds nothing; retain a
    // value-returning condition unchanged. Unit-returning branches still need their residual effects reconstructed.
    let known_side_is_empty = !output_known.iter().any(|&known| known)
        && true_split.edge_program_outputs.is_empty()
        && false_split.edge_program_outputs.is_empty()
        && true_split.known_program.effects().classes().is_empty()
        && false_split.known_program.effects().classes().is_empty()
        && !true_split.known_program.effects().has_deferred_work()
        && !false_split.known_program.effects().has_deferred_work();
    if known_side_is_empty && output_count != 0 {
        return Ok(None);
    }

    // Build each known branch over the shared `[known outputs..., true edges..., false edges...]` output signature,
    // filling the peer branch's edge slots with placeholders. The caller decides how to construct them: partial
    // evaluation accepts only identity-free typed zeros, while differentiation can obtain runtime geometry from the
    // known branch inputs or forward a known input of the identical type. Both branches are built before binding either
    // condition so an unsupported placeholder can retain the original conditional without staging any work in the
    // caller's context.
    let mut build_known_branch = |own: &ConditionBranchSplit<V, O>,
                                  other: &ConditionBranchSplit<V, O>,
                                  own_first: bool|
     -> Result<Option<Program<V, O, Vec<V>, Vec<V>>>, ProgramError> {
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
            if output_known[index] {
                match output {
                    PartialEvaluationOutput::Known(output) => {
                        output_atoms.push(*known_outputs.get(*output).ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "`{CONDITION_OPERATION_NAME}` branch partition references missing known-program output \
                                 {output}",
                            ))
                        })?)
                    }
                    PartialEvaluationOutput::Unknown(_) => {
                        return Err(ProgramError::MalformedProgram(format!(
                            "`{CONDITION_OPERATION_NAME}` known-ness split lost a known output"
                        )));
                    }
                }
            }
        }
        let mut zero_atoms = Vec::with_capacity(other.edge_types.len());
        for edge_type in other.edge_types.iter() {
            let Some(placeholder) = build_placeholder(&mut builder, edge_type, &known_inputs)? else {
                return Ok(None);
            };
            zero_atoms.push(placeholder);
        }
        let edge_atoms = own
            .edge_program_outputs
            .iter()
            .map(|&output| {
                known_outputs.get(output).copied().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{CONDITION_OPERATION_NAME}` branch partition references missing edge output {output}",
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
            .map(Some)
    };

    // A unit-returning condition may have no known work at all. It still needs any residual branch effects, but
    // building and binding an empty pure known condition serves no purpose and must not make separate-context JVP
    // unsupported. Otherwise, the known condition is bound into the enclosing known-side context over the predicate and
    // the known inputs.
    let known_outputs = if known_side_is_empty {
        Vec::new()
    } else {
        let Some(known_true) = build_known_branch(&true_split, &false_split, true)? else {
            return Ok(None);
        };
        let Some(known_false) = build_known_branch(&false_split, &true_split, false)? else {
            return Ok(None);
        };
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
        bind_known(O::from(known_condition), vec![known_true, known_false], known_condition_inputs.as_slice())?
    };
    let known_output_count = output_known.iter().filter(|&&known| known).count();
    let true_edge_offset = known_output_count;
    let false_edge_offset = known_output_count + true_split.edge_types.len();

    // Build each residual branch over the shared `[unknown inputs..., true edges..., false edges...]` input
    // signature, each branch reading only its own edges, with instantiated folded values passed through from their
    // edge slots.
    let residual_output_ordinals = {
        let mut ordinals = vec![None; output_count];
        let mut next = 0;
        for (index, &known) in output_known.iter().enumerate() {
            if !known {
                ordinals[index] = Some(next);
                next += 1;
            }
        }
        ordinals
    };
    // A branch residual with effects or deferred work must survive even when the condition returns no residual values,
    // because simplification retains that work and dropping the condition would discard it.
    let needs_residual_condition = residual_output_ordinals.iter().any(Option::is_some)
        || !true_split.residual_program.effects().classes().is_empty()
        || !false_split.residual_program.effects().classes().is_empty()
        || true_split.residual_program.effects().has_deferred_work()
        || false_split.residual_program.effects().has_deferred_work();
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
                            ProgramError::MalformedProgram(format!(
                                "`{CONDITION_OPERATION_NAME}` known-ness split saw a residual feeder for a known input",
                            ))
                        })?);
                    }
                    PartialEvaluationInput::Known(edge) => {
                        spliced_inputs.push(*own_edge_atoms.get(*edge).ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "`{CONDITION_OPERATION_NAME}` known-ness split lost a residual edge"
                            ))
                        })?)
                    }
                }
            }
            let spliced_outputs = builder.splice_program(&own.residual_program, &spliced_inputs)?;

            let mut output_atoms = Vec::new();
            for (index, output) in own.outputs.iter().enumerate() {
                if output_known[index] {
                    continue;
                }
                match output {
                    PartialEvaluationOutput::Unknown(spliced) => output_atoms.push(spliced_outputs[*spliced]),
                    PartialEvaluationOutput::Known(_) => {
                        let edge = own.instantiated_edge_ordinals[index].ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "`{CONDITION_OPERATION_NAME}` known-ness split lost an instantiated output edge",
                            ))
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
                ProgramError::MalformedProgram(format!(
                    "`{CONDITION_OPERATION_NAME}` known-ness split known side produced no output for a true-branch \
                     edge",
                ))
            })?);
        }
        for edge in 0..false_split.edge_types.len() {
            residual_condition_inputs.push(known_outputs.get(false_edge_offset + edge).cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{CONDITION_OPERATION_NAME}` known-ness split known side produced no output for a false-branch \
                     edge",
                ))
            })?);
        }
        bind_residual(
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
            if output_known[index] {
                let value = known_outputs.get(known_output_ordinal).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{CONDITION_OPERATION_NAME}` known-ness split known side produced no output for a known \
                         result",
                    ))
                });
                known_output_ordinal += 1;
                value
            } else {
                let ordinal = residual_output_ordinals[index].ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{CONDITION_OPERATION_NAME}` known-ness split produced a result owned by neither side"
                    ))
                })?;
                residual_outputs.get(ordinal).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{CONDITION_OPERATION_NAME}` known-ness split residual side produced no output for a \
                         residual result",
                    ))
                })
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
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

/// Partition-aware transpose rule for a *primal* [`ConditionOperation`], used when the direct reverse transposes a
/// tangent program in the primal operation family `O`. The predicate and the per-branch residuals are ordinary
/// *instruction inputs* (known values supplied through the pullback), so the rule reads them from the pullback and
/// threads them back through as known inputs of a transposed condition.
///
/// The forward stages the tangent condition over `[predicate, branch_tangents..., residuals...]`, with the predicate
/// and the joined residual set marked known, the branch tangents marked linear, and both branches already joined to the
/// same input signature `[branch_tangents..., residuals...]` and output signature `[branch_tangent_outputs...]`. This
/// rule therefore:
///
///   1. Splits the inputs by `input_linear` into the known predicate (input `0`), the linear branch inputs, and the
///      known branch residuals, preserving source order within each group. The split does not rely on the forward's
///      grouping because direct transposition also supports known and linear branch inputs interleaved in any order.
///   2. Transposes each branch through the driver's region-transposition request, marking the branch tangent inputs
///      linear and the residual inputs known. Each transposed branch maps
///      `[branch_tangent_output_cotangents..., branch_cotangent_references..., residuals...]` to
///      `[branch_tangent_input_cotangents...]`, where only the branch inputs with a `Reference` cotangent destination
///      own a cotangent reference slot; because both branches shared the joined signature, their transposes share it
///      too and form a well-typed condition.
///   3. Re-stages a primal [`ConditionOperation`] selecting between the two transposed branches by the same known
///      predicate, over `[predicate, outputs..., cotangent_references..., residuals...]`, where `outputs` holds the
///      cotangents of the non-reference condition outputs. Its outputs are the branch-tangent input cotangents.
///
/// The returned cotangents place those branch-tangent cotangents at the linear-input positions and a structural
/// [`MaybeZero::Zero`] at the predicate and residual positions, which carry no cotangent. The branch recursion happens
/// through the instruction-scoped driver in the same operation family, so it introduces no recursive
/// [`TransposableOperation`] obligation on `O`.
///
/// # Parameters
///
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `inputs`: Per-input [`PartialValue`] knowledge. The [`Unknown`](PartialValue::Unknown) entries are the branch
///     tangents; the [`Known`](PartialValue::Known) entries carry the predicate and residual tracers the pullback
///     reads.
///   - `outputs`: Symbolic cotangents for the condition's outputs.
///   - `cotangents`: Cotangent destinations of the inputs (refer to the documentation of
///     [`TranspositionContext::cotangent_destinations`]). Both branches are transposed with the destination kinds of
///     the inputs after the predicate. A `Reference` destination passes the input's cotangent reference into both
///     transposed branches, and the selected branch accumulates into it in place. For a reference-state input, the
///     branch also returns that reference by identity; an ordinary value using a cotangent buffer produces no
///     corresponding output. Both receive structural-zero contributions from this function. An `Ignore`-kind input has
///     no slot in either transposed branch and receives a structural zero as well.
pub fn transpose_primal_condition<V, O, D: TranspositionDriver<V, O>>(
    context: &mut TracingContext<V, O>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    cotangents: &CotangentDestinations<Tracer<TracingContext<V, O>>>,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>
where
    V: Value<Type: ConditionType + DifferentiableType>,
    O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type, Operation = O> + From<ConditionOperation<V::Type>>,
{
    // A condition with no live output cotangents and no live reference input is a zero linear map, so every input
    // cotangent is zero. A live reference input keeps the rule live, because its accumulated state cotangent flows
    // through the transposed branches even when no ordinary output cotangent does. Branches with deferred work or
    // observable rule effects also keep it live, so that their obligations run with structural-zero seeds.
    check_count!("input", cotangents.kinds(), inputs.len(), ProgramError);
    if outputs.iter().all(MaybeZero::is_zero)
        && !cotangents.has_reference_state_destinations()
        && !driver.region(0)?.must_transpose()
        && !driver.region(1)?.must_transpose()
    {
        return inputs
            .iter()
            .map(|input| {
                let input_type = input.r#type();
                Ok(MaybeZero::Zero(input_type.cotangent()?))
            })
            .collect();
    }

    // The rule operates on the attached branch regions through its driver (region 0 is the `true` branch and region 1
    // the `false` branch), which keeps its bounds free of the operation family's own semantic traits.
    let true_branch = driver.region(0)?;

    // Linear branch inputs can occur at any boundary position. Preserve source order separately for the linear branch
    // inputs and the known residual inputs, as the branch transposition driver does.
    let input_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    let branch_input_count = true_branch.input_types().len();
    check_count!("input", inputs, 1 + branch_input_count, ProgramError);
    let branch_input_indices = input_linear[1..]
        .iter()
        .enumerate()
        .filter_map(|(index, &linear)| linear.then_some(index))
        .collect::<Vec<_>>();

    // The predicate is input `0` and the residuals are the known branch inputs; both are known values read from the
    // pullback. The dispatch guarantees a `Known` input carries its pullback value, so each tracer is read directly.
    let read_known = |index: usize| -> Result<Tracer<TracingContext<V, O>>, ProgramError> {
        inputs[index]
            .as_known()
            .ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{CONDITION_OPERATION_NAME}` transpose input {index} has no known value",
                ))
            })
            .cloned()
    };
    let predicate = read_known(0)?;
    let residuals = (1..inputs.len())
        .filter(|&index| !input_linear[index])
        .map(read_known)
        .collect::<Result<Vec<_>, _>>()?;

    // Transpose each branch with the branch tangents marked linear and the residual inputs marked known, through each
    // branch region's retained transform cache so that a branch shared by several programs is transposed once per
    // selection of linear inputs. A live reference-typed branch tangent is transposed with a `Reference` destination
    // and a dead one with an `Ignore` destination, so each transposed branch maps
    // `[branch_output_cotangents..., branch_cotangent_references..., residuals...]` to
    // `[branch_tangent_cotangents...]`, where a live reference tangent's cotangent is its cotangent reference itself
    // and a dead reference tangent has no cotangent slot at all.
    let branch_destination_kinds =
        branch_input_indices.iter().map(|&index| cotangents.kind(index + 1)).collect::<Vec<_>>();
    let transposed_branches = [
        driver.transpose_program(driver.region(0)?, &branch_input_indices, &branch_destination_kinds)?,
        driver.transpose_program(driver.region(1)?, &branch_input_indices, &branch_destination_kinds)?,
    ];
    let transposed_condition = ConditionOperation::new();

    // Stage the transposed condition over `[predicate, outputs..., cotangent_references..., residuals...]`. Its
    // outputs are the branch-tangent input cotangents.
    let output_types = true_branch.output_types();
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut transposed_inputs = Vec::with_capacity(1 + output_types.len() + residuals.len());
    transposed_inputs.push(predicate);
    for (cotangent, output_type) in outputs.iter().zip(&output_types) {
        // A reference output forwards a branch region input root whose state cotangent lives in the cotangent reference
        // of the instruction input that supplies that root, so it owns no cotangent slot.
        if output_type.is_reference() {
            continue;
        }
        // A dead output's structural-zero cotangent still becomes a real input of the transposed condition. Its type
        // alone cannot construct it when it references runtime identities, but the boundary already carries that
        // geometry: at least one peer cotangent is live here (the all-zero case returned above), and the known
        // residuals are live too.
        transposed_inputs.push(O::materialize_zero_from_residual_sources(
            context,
            cotangent.clone(),
            outputs.iter().filter_map(MaybeZero::as_value).chain(&residuals),
        )?);
    }
    transposed_inputs.extend(cotangents.references().iter().cloned());
    transposed_inputs.extend(residuals);
    // The shared transposed-branch handles are attached directly, so repeated binds of one transposed branch intern
    // by `Arc` identity instead of copying it again.
    let branch_cotangents = context.bind(
        O::from(transposed_condition),
        CalleeRegionDriver::new(&transposed_branches),
        transposed_inputs.as_slice(),
    )?;
    let output_count = branch_input_indices.iter().filter(|&&index| cotangents.returns_cotangent(index + 1)).count();
    check_count!("output", branch_cotangents, output_count, ProgramError);

    // Reassemble one cotangent per input: the predicate and residuals carry structural zeros, while the branch tangents
    // receive the transposed condition's outputs in order. A live reference tangent's output is its cotangent
    // reference, whose contents were accumulated in place, and a dead reference tangent has no output, so every
    // reference input receives a structural zero.
    let mut branch_cotangents = branch_cotangents.into_iter();
    input_linear
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(index, (&linear, input))| -> Result<_, ProgramError> {
            match cotangents.kind(index) {
                CotangentDestinationKind::Return if linear => Ok(MaybeZero::Value(branch_cotangents.next().unwrap())),
                CotangentDestinationKind::Reference => {
                    if linear && cotangents.is_reference_input(index) {
                        branch_cotangents.next();
                    }
                    Ok(MaybeZero::Zero(input.r#type().cotangent()?))
                }
                CotangentDestinationKind::Return | CotangentDestinationKind::Ignore => {
                    Ok(MaybeZero::Zero(input.r#type().cotangent()?))
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayOperation,
        ArrayReference, ArrayReferenceTransform, ArrayReferenceTransformIndex, DataType, Dimension, DimensionBounds,
        DimensionType, DimensionValue, DimensionVariable, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
        ShardingDimension,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, BatchingContext, BatchingTracer, batch};
    use crate::captures::{CaptureReference, CapturingContext, ClosedProgram};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::reverse::tests::{run_transposed_with_destinations, transposition_statistics};
    use crate::differentiation::{Differentiate, ForwardModeDifferentiate, ReverseModeDifferentiate, differentiate_at};
    use crate::operations::arithmetic::{AddOperation, DivOperation, MulOperation, SqrtOperation};
    use crate::operations::assertions::AssertionError;
    use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::zero_like::ZeroLikeOperation;
    use crate::operations::control_flow::tests::{CountingBatchingDriver, array, dimension, resolve_captures};
    use crate::operations::differentiation::stop_gradient::StopGradientOperation;
    use crate::operations::references::{
        ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, ReferenceRead,
        ReferenceReadOperation, ReferenceSwapOperation, ReferenceWriteOperation,
    };
    use crate::operations::trigonometric::SinOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{
        EffectClasses, EmptyRegionDriver, ExternalReferenceBinding, ProgramBuilder, ReferenceAccessOperation,
        ReferenceDischargeResult, ReferenceSource, ReferenceType, ReferenceView, TypeError,
    };
    use crate::tracing::{DomainTracingContext, NestedTracingContext, Trace, Tracer, TracingContext};

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;

    /// Builds the canonical array IR test program whose whole-array state crosses a [`ConditionOperation`] boundary,
    /// shared by tests comparing direct reference transforms with explicit discharge. The program takes a Boolean
    /// predicate and an `f32[]` initial value, allocates one local reference from that initial value, and passes the
    /// reference into a condition whose branches access it with unequal modes. The `true` branch accumulates `1.0` and
    /// reads the reference, while the `false` branch swaps in `9.0` and yields the replaced value. Its two outputs are
    /// the condition's snapshot followed by the frozen final state, so a discharged program must thread identical state
    /// through both branches and keep both public outputs interpretable. On `[true, 4.0]` the outputs are `[5.0, 5.0]`,
    /// and on `[false, 4.0]` they are `[4.0, 9.0]`.
    fn test_condition_program()
    -> Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());

        let mut true_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let update = true_builder.add_constant(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()));
        true_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![snapshot],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut false_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = false_builder.add_input(reference_type.into());
        let replacement = false_builder.add_constant(ArrayIrValue::Array(Array::scalar(9.0f32).unwrap()));
        let snapshot = false_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![snapshot],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let initial = builder.add_input(ArrayIrType::Array(scalar_type));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![snapshot, frozen],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap()
    }

    /// Builds a single-input flat program that maps its scalar `f64` input through `operation`.
    fn scalar_branch(
        operation: ArrayOperation<Array>,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let inputs = if matches!(operation, ArrayOperation::Add(_)) { vec![input, input] } else { vec![input] };
        let output = builder.add_instruction(operation, Vec::new(), inputs, None).unwrap()[0];
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
        let zero = builder.add_constant(Array::scalar(0.0).unwrap());
        let output = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![input, zero],
                None,
            )
            .unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a single-input branch that scales its scalar input by `factor`.
    fn scalar_scale_branch(factor: f64) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let factor = builder.add_constant(Array::scalar(factor).unwrap());
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, factor], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a single-input branch that scales a vector input by `factor`.
    fn vector_scale_branch(size: usize, factor: f64) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(size)])));
        let factor = builder.add_constant(Array::scalar(factor).unwrap());
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, factor], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a vector-input branch that returns a replicated constant vector.
    fn constant_vector_branch(values: Vec<f64>) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(values.len())])));
        let output = builder.add_constant(Array::vector(values).unwrap());
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Batches a vector-valued condition whose branches scale their input by two and three.
    fn batch_vector_condition(batch_size: usize, item_size: usize, input_values: Vec<f64>) -> ArrayBatch<Array> {
        let batched_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(batch_size), Dimension::Static(item_size)]),
        );
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(batch_size)]));
        let predicate_values = (0..batch_size).map(|index| index == 0).collect::<Vec<_>>();
        let predicate = ArrayBatch::new(
            Array::from_elements::<bool>(predicate_type, &predicate_values).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        let branch_input =
            ArrayBatch::new(Array::from_elements::<f64>(batched_type, &input_values).unwrap(), BatchAxis::new(0))
                .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), batch_size);
        let mut outputs = context
            .bind(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![vector_scale_branch(item_size, 2.0), vector_scale_branch(item_size, 3.0)],
                &[BatchingTracer::new(context.clone(), predicate), BatchingTracer::new(context.clone(), branch_input)],
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
        let zero = context.lift(Array::scalar(0.0).unwrap())?;
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

    /// Builds a scalar condition with a square root in its positive branch and the identity otherwise.
    fn square_root_or_identity_condition_program() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut identity_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let identity_input = identity_builder.add_input(scalar_type.clone());
        let identity_branch = identity_builder
            .build::<Vec<Array>, Vec<Array>>(vec![identity_input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(scalar_type);
        let zero = builder.add_constant(Array::scalar(0f64).unwrap());
        let predicate = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![input, zero],
                None,
            )
            .unwrap()[0];
        let regions = vec![
            builder.import_program(scalar_branch(ArrayOperation::Sqrt(SqrtOperation::new()))),
            builder.import_program(identity_branch),
        ];
        let output =
            builder.add_instruction(ConditionOperation::new(), regions, vec![predicate, input], None).unwrap()[0];
        builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a scalar condition over `[predicate, branch_inputs...]` whose branches sum the products of the
    /// `(scale, linear)` branch input pairs in `pairs` and scale that sum by `1.0` in the `true` branch and by `2.0` in
    /// the `false` branch, so that known scale inputs and linear inputs can be interleaved in any order.
    fn interleaved_product_condition_program(
        pairs: &[(usize, usize)],
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let input_count = pairs.len() * 2;
        let branch = |factor: f64| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs =
                (0..input_count).map(|_| builder.add_input(ArrayType::scalar(DataType::F64))).collect::<Vec<_>>();
            let products = pairs
                .iter()
                .map(|&(scale, linear)| {
                    builder
                        .add_instruction(MulOperation::new(), Vec::new(), vec![inputs[scale], inputs[linear]], None)
                        .unwrap()[0]
                })
                .collect::<Vec<_>>();
            let mut output = products[0];
            for product in &products[1..] {
                output =
                    builder.add_instruction(AddOperation::new(), Vec::new(), vec![output, *product], None).unwrap()[0];
            }
            let factor = builder.add_constant(Array::scalar(factor).unwrap());
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![output, factor], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; input_count], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_branch = builder.import_program(branch(1.0));
        let false_branch = builder.import_program(branch(2.0));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let mut inputs = vec![predicate];
        inputs.extend((0..input_count).map(|_| builder.add_input(ArrayType::scalar(DataType::F64))));
        let outputs = builder
            .add_instruction(ConditionOperation::new(), vec![true_branch, false_branch], inputs, None)
            .unwrap()
            .to_vec();
        builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; input_count + 1], vec![Placeholder])
            .unwrap()
    }

    /// Builds a composite branch that forwards its `dimension_type` extent input and scales its scalar `f64` input by
    /// `factor`.
    fn scale_branch(
        dimension_type: DimensionType,
        factor: f64,
    ) -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(dimension_type));
        let input = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let factor = builder.add_constant(array(Array::scalar(factor).unwrap()));
        let output = builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(MulOperation::new())),
                Vec::new(),
                vec![input, factor],
                None,
            )
            .unwrap()[0];
        builder.build(vec![extent, output], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap()
    }

    /// Builds a composite condition over `[predicate, extent, input]`, where `input` is an `f64` vector whose dynamic
    /// size is `extent`. The `true` branch squares the input and the `false` branch doubles it. Returns the program
    /// together with the extent's dimension type and the input type.
    fn dynamic_extent_condition_program()
    -> (Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>, DimensionType, ArrayType) {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let input_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let branch = |squares| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            builder.add_input(extent_type.clone().into());
            let input = builder.add_input(input_type.clone().into());
            let operation = if squares {
                TestOperation::Array(ArrayOperation::Mul(MulOperation::new()))
            } else {
                TestOperation::Array(ArrayOperation::Add(AddOperation::new()))
            };
            let output = builder.add_instruction(operation, Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let extent = builder.add_input(extent_type.clone().into());
        let input = builder.add_input(input_type.clone().into());
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, extent, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        (program, extent_type, input_type)
    }

    /// Builds a program whose `condition` selects between the unspecialized branches `x * x + y` and `x + y` over
    /// `[x: f64[rows], y: f64[rows]]` while the program feeds it static `f64[3]` inputs, so the condition's output is
    /// refined by its inputs rather than by re-typed regions.
    fn refined_vector_condition_program() -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows)])));
        let branch = |squares| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let x = builder.add_input(vector_type.clone());
            let y = builder.add_input(vector_type.clone());
            let x = if squares {
                builder
                    .add_instruction(
                        TestOperation::Array(ArrayOperation::Mul(MulOperation::new())),
                        Vec::new(),
                        vec![x, x],
                        None,
                    )
                    .unwrap()[0]
            } else {
                x
            };
            let output = builder
                .add_instruction(
                    TestOperation::Array(ArrayOperation::Add(AddOperation::new())),
                    Vec::new(),
                    vec![x, y],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let x = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let y = builder.add_input(ArrayType::new_static(DataType::F64, [3]).into());
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![true_region, false_region], vec![predicate, x, y], None)
            .unwrap()[0];
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap()
    }

    /// Captured composite value in the reference discharge fixtures.
    type DischargeCapture = CaptureReference<ArrayIrType>;
    /// Captured array payload in the reference discharge fixtures.
    type DischargeArrayCapture = CaptureReference<ArrayType>;
    /// Operation family used by captured array discharge programs.
    type DischargeCaptureOperation = ArrayIrOperation<DischargeArrayCapture>;

    #[test]
    fn test_condition() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch_input_type = ArrayType::scalar(DataType::F64);
        let operation = ConditionOperation::<ArrayType>::new();
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
            operation.infer_output_types(&[predicate_type.clone(), branch_input_type.clone()], interfaces.as_slice()),
            Ok(vec![branch_input_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(&[predicate_type.clone(), branch_input_type.clone()], &[]),
            Err(TypeError::invalid("expected 2 regions but got 0")),
        );
        assert_eq!(
            operation.infer_output_types(&[], interfaces.as_slice()),
            Err(TypeError::invalid("expected 2 inputs but got 0".to_string())),
        );
        assert_eq!(
            operation
                .infer_output_types(&[branch_input_type.clone(), branch_input_type.clone()], interfaces.as_slice()),
            Err(TypeError::invalid("`condition` predicate type must be a scalar boolean, but got `f64[]`".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])), branch_input_type.clone()],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid(
                "`condition` predicate type must be a scalar boolean, but got `bool[2]`".to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(
                &[predicate_type.clone(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid(
                "`condition` input 1 has type `f64[2]`, which does not refine its branch input type `f64[]`"
            )),
        );

        // Inference rejects branch interfaces with mismatched output signatures.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let boolean_output = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![input, zero],
                None,
            )
            .unwrap()[0];
        let boolean_branch = builder.build(vec![boolean_output], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            operation.infer_output_types(
                &[predicate_type.clone(), branch_input_type.clone()],
                &[branch_interface(&true_branch), branch_interface(&boolean_branch)],
            ),
            Err(TypeError::invalid(
                "`condition` branch output type signature mismatch: expected [f64[]] but got [bool[]]".to_string()
            )),
        );

        // Eager binding interprets the predicate-selected branch through detached region access, and interpretation
        // without a predicate input is rejected.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let predicate = |value: bool| Array::from_elements::<bool>(predicate_type.clone(), &[value]).unwrap();
        let outputs = context
            .bind(
                operation.clone(),
                vec![true_branch.clone(), false_branch.clone()],
                &[predicate(true), Array::scalar(4.0).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![8.0]);
        let outputs = context
            .bind(
                operation.clone(),
                vec![true_branch.clone(), false_branch.clone()],
                &[predicate(false), Array::scalar(4.0).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![0.0]);
        assert_eq!(
            operation.interpret(&context.clone(), &EmptyRegionDriver, &[] as &[Array]),
            Err(ProgramError::MalformedProgram("`condition` interpretation requires a predicate input".to_string(),)),
        );

        // Staging imports the branch programs as attached regions of the staged instruction instead of trying to
        // concretize the staged predicate.
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = context.builder().clone();
        let staged_predicate = context.input(predicate_type.clone());
        let staged_branch_input = context.input(branch_input_type.clone());
        let outputs = context
            .stage_operation(
                operation.clone(),
                vec![true_branch.clone(), false_branch.clone()],
                &[staged_predicate.clone(), staged_branch_input.clone()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let builder = builder.borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), ArrayOperation::Condition(_)));
        assert_eq!(builder.instructions()[0].regions().len(), 2);
        assert_eq!(
            builder.instructions()[0].inputs(),
            &[staged_predicate.atom_id().unwrap(), staged_branch_input.atom_id().unwrap()],
        );
        assert_eq!(outputs[0].atom_id(), Ok(builder.instructions()[0].outputs()[0]));

        // Program rendering shows the attached branch regions at the instruction with their declared slot names.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let program_predicate = builder.add_input(predicate_type);
        let program_branch_input = builder.add_input(branch_input_type);
        let program_output = builder
            .add_instruction(
                ArrayOperation::Condition(operation),
                vec![true_region, false_region],
                vec![program_predicate, program_branch_input],
                None,
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

    #[test]
    fn test_condition_type_inference_refines_input_types() {
        // Branch inputs only need to refine the branch input types, so actual types that carry metadata the branch
        // input types leave unspecified (e.g., the normalized shardings of concrete backend array types) are accepted,
        // and the outputs do not inherit that metadata.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch_input_type = ArrayType::scalar(DataType::F64);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_type = branch_input_type.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let interfaces = vec![
            branch_interface(&scalar_branch(ArrayOperation::Add(AddOperation::new()))),
            branch_interface(&scalar_branch(ArrayOperation::ZeroLike(ZeroLikeOperation::new()))),
        ];
        assert_eq!(
            ConditionOperation::<ArrayType>::new()
                .infer_output_types(&[predicate_type.clone(), sharded_type.clone()], interfaces.as_slice()),
            Ok(vec![branch_input_type.clone()]),
        );
        assert_eq!(
            ConditionOperation::<ArrayType>::new()
                .infer_region_input_types(&[predicate_type, sharded_type.clone()], interfaces.as_slice()),
            Ok(vec![Some(vec![sharded_type.clone()]), Some(vec![sharded_type])]),
        );

        // A static extent within the bounds of a dynamic branch input dimension refines it, and the outputs take the
        // extent that the inputs establish for that dimension's identity. Inputs that share the identity share its
        // extent, so this holds even when another input still carries the identity dynamically.
        let predicate_type = ArrayIrType::Array(ArrayType::scalar(DataType::Boolean));
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])));
        let static_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [5]));
        let interface = RegionInterface::new(
            vec![vector_type.clone(), vector_type.clone()],
            vec![vector_type.clone()],
            EffectClasses::NONE,
        );
        let interfaces = vec![interface.clone(), interface];
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[predicate_type.clone(), vector_type.clone(), static_type.clone()],
                interfaces.as_slice(),
            ),
            Ok(vec![static_type.clone()]),
        );
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[
                    predicate_type.clone(),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [9])),
                    static_type.clone()
                ],
                interfaces.as_slice(),
            ),
            Err(TypeError::invalid(
                "`condition` input 1 has type `f32[9]`, which does not refine its branch input type `f32[extent]`",
            )),
        );
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[predicate_type.clone(), static_type.clone(), static_type.clone()],
                interfaces.as_slice(),
            ),
            Ok(vec![static_type.clone()]),
        );

        // A reference input must equal its branch input type exactly, because its allocation keeps its type when
        // references are discharged.
        let ArrayIrType::Array(referent_type) = &vector_type else { unreachable!() };
        let reference_type = ArrayIrType::Reference(ReferenceType::new(referent_type.clone()));
        let refined_reference_type =
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [5])));
        let interface = RegionInterface::new(
            vec![reference_type.clone(), vector_type.clone()],
            vec![reference_type.clone()],
            EffectClasses::NONE,
        );
        assert_eq!(
            ConditionOperation::<ArrayIrType>::new().infer_output_types(
                &[predicate_type, refined_reference_type, vector_type],
                &[interface.clone(), interface],
            ),
            Err(TypeError::invalid(
                "`condition` input 1 has type `ref<f32[5]>`, which does not equal its branch input type \
                 `ref<f32[extent]>`",
            )),
        );
    }

    #[test]
    fn test_condition_composite_type_contract() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = ArrayIrType::Dimension(DimensionType::from(extent.clone()));
        let array_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])));
        let branch_inputs = vec![dimension_type.clone(), array_type.clone()];
        let branch_outputs = vec![array_type.clone(), dimension_type.clone()];
        let branch_interface = RegionInterface::new(branch_inputs.clone(), branch_outputs.clone(), EffectClasses::NONE);
        let operation = ConditionOperation::<ArrayIrType>::new();
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
                    RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE),
                    RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE),
                ],
            ),
            Err(TypeError::invalid(
                "`condition` predicate type must be a scalar boolean, but got `dimension<extent ∈ [1, 8)>`".to_string(),
            )),
        );
    }

    #[test]
    fn test_condition_type_semantics_manual_predicate() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant =
            ArrayType::scalar(DataType::Boolean).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let varying = ArrayType::scalar(DataType::Boolean)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        assert!(invariant.is_condition_predicate());
        assert!(varying.is_condition_predicate());
        assert!(ArrayIrType::Array(varying.clone()).is_condition_predicate());

        // Outputs must vary over every manual axis that the predicate varies over.
        let varying_output = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        let invariant_output = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        assert_eq!(varying_output.validate_condition_output(&varying), Ok(()));
        assert_eq!(invariant_output.validate_condition_output(&invariant), Ok(()));
        assert_eq!(ArrayType::scalar(DataType::F32).validate_condition_output(&invariant), Ok(()));
        assert_eq!(
            invariant_output.validate_condition_output(&varying),
            Err(TypeError::invalid(
                "`condition` output `f32[][sharding={mesh<['devices'=2:manual]>, []}]` must vary over every manual \
                 axis that the predicate \
                 `bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]` varies over, because \
                 devices may take different branches; insert `parallel_vary` on the branch outputs",
            )),
        );
        assert_eq!(
            ArrayIrType::Array(varying_output).validate_condition_output(&ArrayIrType::Array(varying.clone())),
            Ok(()),
        );
        assert_eq!(
            ArrayIrType::Dimension(DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap()))
                .validate_condition_output(&ArrayIrType::Array(varying)),
            Err(TypeError::invalid(
                "`condition` output `dimension<extent ∈ [1, 8)>` cannot record manual variation, so it cannot be \
                 produced under the varying predicate \
                 `bool[][sharding={mesh<['devices'=2:manual]>, [], varying_manual={'devices'}}]`",
            )),
        );
    }

    #[test]
    fn test_condition_infers_output_types_through_operation_enum() {
        // Inference dispatches through the closed operation enum exactly like through the bare operation.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let identity_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let operation = ArrayOperation::<Array>::Condition(ConditionOperation::new());
        assert_eq!(
            operation.infer_output_types(
                &[ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::F64)],
                &[identity_branch.interface(), identity_branch.interface()],
            ),
            Ok(vec![ArrayType::scalar(DataType::F64)]),
        );
    }

    #[test]
    fn test_condition_differentiation_keeps_tangents_of_inactive_outputs_structural() {
        // The true branch maps `(a, b, c)` to `(sin(a), sin(c))` and the false branch maps it to `(sin(b), sin(c))`.
        // Only `a` has a live tangent, so the first output's tangent is live because the true branch depends on it,
        // while the second output depends on `a` in neither branch and keeps a structural-zero tangent. The branch
        // derivatives therefore take one tangent input and return one tangent output, and linearization succeeds.
        let scalar = ArrayType::scalar(DataType::F64);
        let branch = |first: usize| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs = (0..3).map(|_| builder.add_input(scalar.clone())).collect::<Vec<_>>();
            let outputs = [inputs[first], inputs[2]]
                .map(|input| builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0]);
            builder
                .build::<Vec<Array>, Vec<Array>>(outputs.to_vec(), vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let inputs = (0..3).map(|_| builder.add_input(scalar.clone())).collect::<Vec<_>>();
        let true_branch = builder.import_program(branch(0));
        let false_branch = builder.import_program(branch(1));
        let outputs = builder
            .add_instruction(
                ConditionOperation::<ArrayType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, inputs[0], inputs[1], inputs[2]],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.jvp_with_respect_to(&[1]).unwrap().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                let %5:f64[], %6:f64[], %7:f64[] = condition %0 %1 %2 %3 %4 [
                    true={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = sin %0
                            %5:f64[] = sin %2
                            %6:f64[] = cos %0
                            %7:f64[] = mul %6 %3
                        in (%4, %5, %7)
                    },
                    false={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = sin %1
                            %5:f64[] = sin %2
                            %6:f64[] = zero [type=f64[]]
                        in (%4, %5, %6)
                    },
                ]
                    %8:f64[] = zero [type=f64[]]
                in (%5, %6, %7, %8)
            "}
            .trim_end(),
        );

        let linearization = program.linearize_with_respect_to(&[1]).unwrap();
        for (predicate, expected_tangent) in [(true, 0.5f64.cos()), (false, 0.0)] {
            let mut primal_outputs = linearization
                .primal()
                .interpret(vec![
                    Array::scalar(predicate).unwrap(),
                    Array::scalar(0.5f64).unwrap(),
                    Array::scalar(1.5f64).unwrap(),
                    Array::scalar(2.5f64).unwrap(),
                ])
                .unwrap();
            let residuals = primal_outputs.split_off(2);
            let mut tangent_inputs = vec![Array::scalar(1f64).unwrap()];
            tangent_inputs.extend(residuals);
            assert_eq!(
                linearization.tangent().interpret(tangent_inputs),
                Ok(vec![Array::scalar(expected_tangent).unwrap(), Array::scalar(0f64).unwrap()]),
            );
        }
    }

    #[test]
    fn test_condition_boundary_pruning() {
        // The true branch maps `(a, b, c)` to `(sin(a), sin(c))` and the false branch maps it to `(sin(b), sin(c))`.
        // Using only the first output keeps the inputs that either branch needs for it, `a` and `b`, together with the
        // predicate, and drops `c` and the second output from both branches.
        let scalar = ArrayType::scalar(DataType::F64);
        let branch = |first: usize| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let inputs = (0..3).map(|_| builder.add_input(scalar.clone())).collect::<Vec<_>>();
            let outputs = [inputs[first], inputs[2]]
                .map(|input| builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0]);
            builder
                .build::<Vec<Array>, Vec<Array>>(outputs.to_vec(), vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let p = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let inputs = (0..3).map(|_| builder.add_input(scalar.clone())).collect::<Vec<_>>();
        let true_branch = builder.import_program(branch(0));
        let false_branch = builder.import_program(branch(1));
        let outputs = builder
            .add_instruction(
                ConditionOperation::<ArrayType>::new(),
                vec![true_branch, false_branch],
                vec![p, inputs[0], inputs[1], inputs[2]],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![outputs[0]], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let pruned = program.clone().into_pruned().unwrap();
        assert_eq!(
            pruned.to_string(),
            indoc! {"
                lambda %0:bool[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %0
                        in (%2)
                    },
                    false={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = sin %1
                        in (%2)
                    },
                ]
                in (%4)"},
        );
        let inputs = vec![
            Array::scalar(false).unwrap(),
            Array::scalar(0.5f64).unwrap(),
            Array::scalar(1.5f64).unwrap(),
            Array::scalar(2.5f64).unwrap(),
        ];
        assert_eq!(pruned.interpret(inputs.clone()).unwrap(), program.interpret(inputs).unwrap());
    }

    #[test]
    fn test_condition_boundary_pruning_keeps_inputs_that_refine_kept_outputs() {
        // Neither branch reads its second or third input. When the second input is the static `f64[3]`, it is the only
        // input that fixes `rows = 3`, which refines the output to `f64[3]`, so dropping it would change the type of
        // the kept output. The pruning that the condition proposes is then rejected as a whole and the instruction
        // keeps its boundary. When the second input is dynamic too, it establishes no fact and both unread inputs are
        // pruned.
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows)])));
        let static_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        let branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let first = builder.add_input(vector_type.clone());
            builder.add_input(vector_type.clone());
            builder.add_input(vector_type.clone());
            let output = builder
                .add_instruction(
                    TestOperation::Array(ArrayOperation::from(SinOperation::new())),
                    Vec::new(),
                    vec![first],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let program = |second_type: ArrayIrType| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
            let first = builder.add_input(vector_type.clone());
            let second = builder.add_input(second_type);
            let third = builder.add_input(vector_type.clone());
            let true_branch = builder.import_program(branch.clone());
            let false_branch = builder.import_program(branch.clone());
            let output = builder
                .add_instruction(
                    TestOperation::Condition(ConditionOperation::new()),
                    vec![true_branch, false_branch],
                    vec![predicate, first, second, third],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 4], vec![Placeholder])
                .unwrap()
        };

        let refined = program(static_type.clone());
        assert_eq!(refined.output_types(), vec![static_type.clone()]);
        let pruned = refined.clone().into_pruned().unwrap();
        assert_eq!(pruned.to_string(), refined.to_string());
        assert_eq!(pruned.output_types(), vec![static_type]);

        let unrefined = program(vector_type.clone());
        let pruned = unrefined.into_pruned().unwrap();
        assert_eq!(pruned.output_types(), vec![vector_type]);
        assert_eq!(pruned.instructions()[0].inputs().len(), 2);
    }

    #[test]
    fn test_condition_interprets_branch_local_reference_allocations() {
        // Only the taken branch allocates and reads its local reference, and that allocation never leaves the branch,
        // so both predicates interpret to the input value.
        let array_type = ArrayType::new_static(DataType::F32, [2]);

        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_input = true_builder.add_input(array_type.clone().into());
        let true_reference = true_builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![true_input], None)
            .unwrap()[0];
        let true_output = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![true_reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![true_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let false_input = false_builder.add_input(array_type.clone().into());
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![false_input], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(array_type.into());
        let output = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let value = TestValue::Array(Array::vector(vec![2.0f32, 4.0]).unwrap());
        assert_eq!(
            program.interpret(vec![TestValue::Array(Array::scalar(true).unwrap()), value.clone()]),
            Ok(vec![value.clone()]),
        );
        assert_eq!(
            program.interpret(vec![TestValue::Array(Array::scalar(false).unwrap()), value.clone()]),
            Ok(vec![value]),
        );
    }

    #[test]
    fn test_condition_interprets_references_forwarded_into_branches() {
        // A reference allocated before the condition enters each branch as an input, so both branches read the
        // caller's allocation, through both program interpretation and region interpretation in an explicit context.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let reference_type = ReferenceType::new(array_type.clone());
        let build_branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let true_branch = build_branch();
        let false_branch = build_branch();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(array_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let output = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let context = EagerContext::<TestValue, TestOperation>::new();
        let value = TestValue::Array(Array::vector(vec![2.0f32, 4.0]).unwrap());
        for predicate in [true, false] {
            assert_eq!(
                program.interpret(vec![TestValue::Array(Array::scalar(predicate).unwrap()), value.clone()]),
                Ok(vec![value.clone()]),
            );
            assert_eq!(
                program.entry_region_ref().interpret_in_context(
                    &context,
                    vec![TestValue::Array(Array::scalar(predicate).unwrap()), value.clone()],
                ),
                Ok(vec![value.clone()]),
            );
        }
    }

    #[test]
    fn test_composite_condition_tracing_rendering_and_eager_execution() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let input = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let regions = vec![
            builder.import_region(scale_branch(extent_type.clone(), 2.0).entry_region_ref()),
            builder.import_region(scale_branch(extent_type.clone(), 3.0).entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();

        // A dimension carried through a condition is an ordinary structural value: it appears in both branch
        // interfaces and in the composite output signature exactly like the array beside it.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)>, %2:f64[] .
                let %3:dimension<extent ∈ [1, 8)>, %4:f64[] = condition %0 %1 %2 [
                    true={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[] .
                        let %2:f64[] = const 2.0
                            %3:f64[] = mul %1 %2
                        in (%0, %3)
                    },
                    false={
                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[] .
                        let %2:f64[] = const 3.0
                            %3:f64[] = mul %1 %2
                        in (%0, %3)
                    },
                ]
                in (%3, %4)"},
        );

        // Eager interpretation selects one branch per predicate value and forwards the same dimension either way.
        let boolean =
            |value: bool| array(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[value]).unwrap());
        assert_eq!(
            program.interpret(vec![boolean(true), dimension(&extent_type, 4), array(Array::scalar(5.0).unwrap())]),
            Ok(vec![dimension(&extent_type, 4), array(Array::scalar(10.0).unwrap())]),
        );
        assert_eq!(
            program.interpret(vec![boolean(false), dimension(&extent_type, 4), array(Array::scalar(5.0).unwrap())]),
            Ok(vec![dimension(&extent_type, 4), array(Array::scalar(15.0).unwrap())]),
        );

        // Relocating the composite program imports both branch regions unchanged, so it renders and executes exactly
        // like its source.
        let mut relocated_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let relocated_inputs = vec![
            relocated_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))),
            relocated_builder.add_input(ArrayIrType::Dimension(extent_type.clone())),
            relocated_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64))),
        ];
        let relocated_outputs = relocated_builder.splice_program(&program, &relocated_inputs).unwrap();
        let relocated = relocated_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(relocated_outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(relocated.to_string(), program.to_string());
        assert_eq!(
            relocated.interpret(vec![boolean(true), dimension(&extent_type, 4), array(Array::scalar(5.0).unwrap())]),
            Ok(vec![dimension(&extent_type, 4), array(Array::scalar(10.0).unwrap())]),
        );
    }

    #[test]
    fn test_condition_reference_discharge() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let boolean_type = ArrayType::scalar(DataType::Boolean);
        let reference_type = ReferenceType::new(scalar_type.clone());

        // Read-only pruning. Both branches merely read the root, so the shared state boundary enters both branches but
        // publishes nothing back: the rebuilt branches gain no appended output, the discharged condition keeps the
        // source output boundary exactly, and the entry root receives no hidden final-state output.
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let doubled = true_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![snapshot, snapshot], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_program(true_branch);
        let false_branch = builder.import_program(false_branch);
        let predicate = builder.add_input(boolean_type.clone().into());
        let reference = builder.add_input(reference_type.clone().into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = add %0 %0
                        in (%1)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.program().output_types().len(), 1);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);

        // A root one branch mutates is published instead. Exactly one output is appended, and both branches receive
        // the identical widened boundary even though only the true branch writes: the reading branch republishes the
        // state it received, which is what keeps the two branch interfaces agreeing after the rewrite.
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let replacement = true_builder.add_constant(TestValue::Array(Array::scalar(7.0f32).unwrap()));
        let snapshot = true_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_program(true_branch);
        let false_branch = builder.import_program(false_branch);
        let predicate = builder.add_input(boolean_type.clone().into());
        let reference = builder.add_input(reference_type.clone().into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[], %3:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = const 7.0
                        in (%0, %1)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0, %0)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(1));

        // Pruning is decided per root rather than for the operation as a whole. Two roots enter as instruction inputs
        // and both branches reach both of them, but only the second is ever written, so only the second gains an
        // appended output while the first keeps crossing the boundary as a read-only state.
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = true_builder.add_input(reference_type.clone().into());
        let second = true_builder.add_input(reference_type.clone().into());
        let update = true_builder.add_constant(TestValue::Array(Array::scalar(1.0f32).unwrap()));
        let snapshot =
            true_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        true_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![second, update], None)
            .unwrap();
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = false_builder.add_input(reference_type.clone().into());
        let second = false_builder.add_input(reference_type.clone().into());
        false_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap();
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_program(true_branch);
        let false_branch = builder.import_program(false_branch);
        let predicate = builder.add_input(boolean_type.into());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, first, second],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings().len(), 2);
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);
        assert_eq!(discharged.external_reference_bindings()[1].output_index(), Some(1));

        // The true branch reports the first root's state and accumulates into the second; the false branch reports the
        // second root's state and writes nothing, so its appended final state is the value that entered.
        let inputs = vec![
            TestValue::Array(Array::scalar(true).unwrap()),
            TestValue::Array(Array::scalar(10.0f32).unwrap()),
            TestValue::Array(Array::scalar(20.0f32).unwrap()),
        ];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![
                TestValue::Array(Array::scalar(10.0f32).unwrap()),
                TestValue::Array(Array::scalar(21.0f32).unwrap())
            ]),
        );
        let inputs = vec![
            TestValue::Array(Array::scalar(false).unwrap()),
            TestValue::Array(Array::scalar(10.0f32).unwrap()),
            TestValue::Array(Array::scalar(20.0f32).unwrap()),
        ];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![
                TestValue::Array(Array::scalar(20.0f32).unwrap()),
                TestValue::Array(Array::scalar(20.0f32).unwrap())
            ]),
        );
    }

    #[test]
    fn test_condition_reference_discharge_threads_a_preserved_allocation_through_condition_branches() {
        // A condition's shared state boundary carries both kinds of allocation: the selected one crosses as immutable
        // state and is widened with a published successor, while the preserved one crosses as the reference it already
        // is, at its own declared input position, and is read inside each branch exactly as the source read it.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let branch = |accumulates: bool| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let pipeline = builder.add_input(reference_type.clone().into());
            let kernel = builder.add_input(reference_type.clone().into());
            let observed =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
            if accumulates {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![pipeline, observed], None)
                    .unwrap();
            }
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let true_branch = branch(true);
        let false_branch = branch(false);

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let pipeline_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let kernel_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let pipeline = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![pipeline_initial], None)
            .unwrap()[0];
        let kernel = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![kernel_initial], None)
            .unwrap()[0];
        let observed = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, pipeline, kernel],
                None,
            )
            .unwrap()[0];
        let pipeline_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![pipeline], None).unwrap()[0];
        let kernel_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![observed, pipeline_final, kernel_final],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &targets[..1]).unwrap();
        assert_eq!(discharged.output_count(), 3);
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:ref<f32[]> = reference_new %2
                    %4:f32[], %5:f32[] = condition %0 %1 %3 [
                        true={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = reference_read %1
                                %3:f32[] = add %0 %2
                            in (%2, %3)
                        },
                        false={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = reference_read %1
                            in (%2, %0)
                        },
                    ]
                    %6:f32[] = reference_freeze %3
                in (%4, %5, %6)"},
        );

        // Eager reference semantics stay the oracle on both sides of the rewrite.
        for (predicate, expected) in [(true, 13.0f32), (false, 10.0)] {
            let inputs = vec![
                TestValue::Array(Array::scalar(predicate).unwrap()),
                TestValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(3.0).unwrap()),
            ];
            let outputs = vec![
                TestValue::Array(Array::scalar::<f32>(3.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(expected).unwrap()),
                TestValue::Array(Array::scalar::<f32>(3.0).unwrap()),
            ];
            assert_eq!(source.clone().interpret(inputs.clone()), Ok(outputs.clone()));
            assert_eq!(discharged.program().interpret(inputs), Ok(outputs));
        }
    }

    #[test]
    fn test_condition_reference_discharge_threads_a_preserved_allocation_through_nested_structured_boundaries() {
        // A rebuilt region is discharged against its own isolated environment, so a preserved reference crossing two
        // boundaries is bound as a preserved reference of the outer fork and then threaded again into the inner one.
        // The reference therefore reaches the innermost access as the caller's own, and the discharged reference beside
        // it is widened independently at each level.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let inner = |accumulates: bool| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let pipeline = builder.add_input(reference_type.clone().into());
            let kernel = builder.add_input(reference_type.clone().into());
            let observed =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![kernel], None).unwrap()[0];
            if accumulates {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![pipeline, observed], None)
                    .unwrap();
            }
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![observed], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let inner_true = inner(true);
        let inner_false = inner(false);

        let mut outer_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let inner_true = outer_builder.import_region(inner_true.entry_region_ref());
        let inner_false = outer_builder.import_region(inner_false.entry_region_ref());
        let predicate = outer_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let pipeline = outer_builder.add_input(reference_type.clone().into());
        let kernel = outer_builder.add_input(reference_type.clone().into());
        let observed = outer_builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![inner_true, inner_false],
                vec![predicate, pipeline, kernel],
                None,
            )
            .unwrap()[0];
        let outer = outer_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![observed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let outer_true = builder.import_region(outer.entry_region_ref());
        let outer_false = builder.import_region(outer.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let pipeline_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let kernel_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let pipeline = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![pipeline_initial], None)
            .unwrap()[0];
        let kernel = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![kernel_initial], None)
            .unwrap()[0];
        let observed = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![outer_true, outer_false],
                vec![predicate, predicate, pipeline, kernel],
                None,
            )
            .unwrap()[0];
        let pipeline_final =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![pipeline], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![observed, pipeline_final],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &targets[..1]).unwrap();
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:ref<f32[]> = reference_new %2
                    %4:f32[], %5:f32[] = condition %0 %0 %1 %3 [
                        true={
                            lambda %0:bool[], %1:f32[], %2:ref<f32[]> .
                            let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                                true={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                        %3:f32[] = add %0 %2
                                    in (%2, %3)
                                },
                                false={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                    in (%2, %0)
                                },
                            ]
                            in (%3, %4)
                        },
                        false={
                            lambda %0:bool[], %1:f32[], %2:ref<f32[]> .
                            let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                                true={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                        %3:f32[] = add %0 %2
                                    in (%2, %3)
                                },
                                false={
                                    lambda %0:f32[], %1:ref<f32[]> .
                                    let %2:f32[] = reference_read %1
                                    in (%2, %0)
                                },
                            ]
                            in (%3, %4)
                        },
                    ]
                in (%4, %5)"},
        );

        let inputs = vec![
            TestValue::Array(Array::scalar(true).unwrap()),
            TestValue::Array(Array::scalar::<f32>(10.0).unwrap()),
            TestValue::Array(Array::scalar::<f32>(3.0).unwrap()),
        ];
        let outputs = vec![
            TestValue::Array(Array::scalar::<f32>(3.0).unwrap()),
            TestValue::Array(Array::scalar::<f32>(13.0).unwrap()),
        ];
        assert_eq!(source.interpret(inputs.clone()), Ok(outputs.clone()));
        assert_eq!(discharged.program().interpret(inputs), Ok(outputs));
    }

    #[test]
    fn test_condition_reference_discharge_selecting_nothing_is_the_identity_on_a_structured_program() {
        // Preserving every allocation is the opposite extreme from full discharge, and it must be the identity: every
        // access, every transform, and every structured boundary replays exactly as the source declared it. This is the
        // sharpest statement of what "preserved" means, and it holds through a condition's attached regions.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = branch_builder.add_input(reference_type.clone().into());
        let replacement = branch_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let previous = branch_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![previous], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(branch.entry_region_ref());
        let false_branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let replacement = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let allocation =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let previous = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, allocation, replacement],
                None,
            )
            .unwrap()[0];
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![allocation], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![previous, frozen], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        let preserved = source.clone().partially_discharge_references(0, &[]).unwrap();
        assert_eq!(preserved.output_count(), 2);
        assert_eq!(preserved.external_reference_bindings(), &[]);
        assert_eq!(preserved.program().to_string(), source.to_string());

        // Selecting the one target instead is full discharge, which is the other extreme of the same rewrite.
        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = ReferenceDischargeResult::try_from(
            source.clone().partially_discharge_references(0, targets.as_slice()).unwrap(),
        )
        .unwrap();
        assert_eq!(discharged.program().to_string(), source.discharge_references(0).unwrap().program().to_string());
    }

    #[test]
    fn test_condition_reference_discharge_threads_identical_state_through_unequal_branch_accesses() {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let replacement = true_builder.add_input(ArrayType::scalar(DataType::F32).into());
        true_builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        false_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type.into());
        let replacement = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, replacement],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // Both branches receive the entering state and return their own final state after the source output, so the
        // writing branch returns its replacement while the reading branch returns the state unchanged.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[], %2:f32[] .
                let %3:f32[], %4:f32[] = condition %0 %1 %2 [
                    true={
                        lambda %0:f32[], %1:f32[] .
                        in (%1, %1)
                    },
                    false={
                        lambda %0:f32[], %1:f32[] .
                        in (%0, %0)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(1));

        // The true branch writes and then reads, so both the public snapshot and final state are the replacement.
        assert_eq!(
            discharged.program().interpret(vec![
                TestValue::Array(Array::scalar(true).unwrap()),
                TestValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(7.0).unwrap())
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar::<f32>(7.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(7.0).unwrap())
            ]),
        );

        // The false branch only reads, so the entering state is both the snapshot and the final state.
        assert_eq!(
            discharged.program().interpret(vec![
                TestValue::Array(Array::scalar(false).unwrap()),
                TestValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(7.0).unwrap())
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(10.0).unwrap())
            ]),
        );
    }

    #[test]
    fn test_condition_reference_discharge_orders_multiple_allocations_by_parent_boundary() {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = true_builder.add_input(reference_type.clone().into());
        let second = true_builder.add_input(reference_type.clone().into());
        true_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let second_replacement = true_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let first_snapshot =
            true_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        let second_snapshot = true_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![second, second_replacement], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![first_snapshot, second_snapshot],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = false_builder.add_input(reference_type.clone().into());
        let second = false_builder.add_input(reference_type.clone().into());
        let first_replacement = false_builder.add_input(ArrayType::scalar(DataType::F32).into());
        false_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let first_snapshot = false_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![first, first_replacement], None)
            .unwrap()[0];
        let second_snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![first_snapshot, second_snapshot],
                vec![Placeholder; 4],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        let first_replacement = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let second_replacement = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let outputs = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, first, second, first_replacement, second_replacement],
                None,
            )
            .unwrap()
            .to_vec();
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 5], vec![Placeholder; 2])
            .unwrap();

        // Both branches write a different allocation, so both allocations cross the boundary; the appended final-state
        // outputs follow parent entry-boundary order rather than the order in which either branch happens to access
        // them.
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.external_reference_bindings().len(), 2);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Input { index: 1 });
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(2));
        assert_eq!(discharged.external_reference_bindings()[1].source(), ReferenceSource::Input { index: 2 });
        assert_eq!(discharged.external_reference_bindings()[1].output_index(), Some(3));

        // The true branch swaps only the second allocation, leaving the first allocation's final state at its entering
        // value.
        let inputs = vec![
            TestValue::Array(Array::scalar(true).unwrap()),
            TestValue::Array(Array::scalar::<f32>(10.0).unwrap()),
            TestValue::Array(Array::scalar::<f32>(20.0).unwrap()),
            TestValue::Array(Array::scalar::<f32>(11.0).unwrap()),
            TestValue::Array(Array::scalar::<f32>(22.0).unwrap()),
        ];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![
                TestValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(20.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(22.0).unwrap())
            ]),
        );

        // The false branch swaps only the first allocation, which mirrors the same contract on the other position.
        let inputs = vec![
            TestValue::Array(Array::scalar(false).unwrap()),
            TestValue::Array(Array::scalar::<f32>(10.0).unwrap()),
            TestValue::Array(Array::scalar::<f32>(20.0).unwrap()),
            TestValue::Array(Array::scalar::<f32>(11.0).unwrap()),
            TestValue::Array(Array::scalar::<f32>(22.0).unwrap()),
        ];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![
                TestValue::Array(Array::scalar::<f32>(10.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(20.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(11.0).unwrap()),
                TestValue::Array(Array::scalar::<f32>(20.0).unwrap())
            ]),
        );
    }

    #[test]
    fn test_condition_reference_discharge_isolates_its_branches() {
        // Both branches accumulate a different amount into the same allocation and return the state they observe. If
        // either branch's staging leaked into the other's, the second branch would start from the first's successor
        // state.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let branch = |amount: f32| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let update = builder.add_constant(TestValue::Array(Array::scalar::<f32>(amount).unwrap()));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                .unwrap();
            let snapshot =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_program(branch(1.0));
        let false_branch = builder.import_program(branch(10.0));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let allocation =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, allocation],
                None,
            )
            .unwrap()[0];

        // The condition's outputs are bound in the *parent*, so a later parent instruction consumes them directly. A
        // value stamped with a branch's own destination builder would be rejected here instead of staged.
        let doubled = builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![snapshot, snapshot], None)
            .unwrap()[0];
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![allocation], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![doubled, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let discharged = source.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[], %3:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                    false={
                        lambda %0:f32[] .
                        let %1:f32[] = const 10.0
                            %2:f32[] = add %0 %1
                        in (%2, %2)
                    },
                ]
                    %4:f32[] = add %2 %2
                in (%4, %3)"},
        );
        for (predicate, expected) in [
            (
                true,
                vec![
                    TestValue::Array(Array::scalar::<f32>(6.0).unwrap()),
                    TestValue::Array(Array::scalar::<f32>(3.0).unwrap()),
                ],
            ),
            (
                false,
                vec![
                    TestValue::Array(Array::scalar::<f32>(24.0).unwrap()),
                    TestValue::Array(Array::scalar::<f32>(12.0).unwrap()),
                ],
            ),
        ] {
            let inputs = vec![
                TestValue::Array(Array::scalar(predicate).unwrap()),
                TestValue::Array(Array::scalar::<f32>(2.0).unwrap()),
            ];
            assert_eq!(source.clone().interpret(inputs.clone()), Ok(expected.clone()));
            assert_eq!(discharged.program().interpret(inputs), Ok(expected));
        }
    }

    #[test]
    fn test_condition_reference_discharge_rejects_a_branch_local_allocation_that_escapes() {
        // Both branches allocate an allocation of their own and return it, so the condition's output denotes a
        // reference its caller never threaded in. Merging that output would hand the caller a handle into an
        // environment that no longer exists, so the rewrite rejects it instead.
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
            let allocation =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![allocation], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_program(branch());
        let false_branch = builder.import_program(branch());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let escaped = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, initial],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![escaped], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            source.discharge_references(0),
            Err(ProgramError::MalformedProgram(message))
                if message.ends_with("whose caller did not thread that allocation"),
        ));
    }

    #[test]
    fn test_condition_reference_discharge_read_only_adds_no_final_state_output() {
        // A closure that only reads an external allocation needs the state to enter both branches, but the allocation's
        // value never changes, so no branch gains a final-state result and the parent condition keeps exactly its
        // public outputs instead of carrying a dead state output.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let make_branch = || {
            let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = branch_builder.add_input(reference_type.clone().into());
            let snapshot = branch_builder
                .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
                .unwrap()[0];
            branch_builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(make_branch().entry_region_ref());
        let false_branch = builder.import_region(make_branch().entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type.into());
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        in (%0)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.program().output_types().len(), 1);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);
        assert_eq!(
            discharged.program().interpret(vec![
                TestValue::Array(Array::scalar(true).unwrap()),
                TestValue::Array(Array::scalar::<f32>(4.0).unwrap())
            ]),
            Ok(vec![TestValue::Array(Array::scalar::<f32>(4.0).unwrap())])
        );
        assert_eq!(
            discharged.program().interpret(vec![
                TestValue::Array(Array::scalar(false).unwrap()),
                TestValue::Array(Array::scalar::<f32>(4.0).unwrap())
            ]),
            Ok(vec![TestValue::Array(Array::scalar::<f32>(4.0).unwrap())])
        );
    }

    #[test]
    fn test_condition_captures_a_lazy_view_root_and_dynamic_binding() {
        // Branches traced as nested regions read a captured root through lazy views. The views are not program values,
        // so the regions capture only the root and the dynamic index, and each access re-applies its own transforms.
        let context = TracingContext::<DischargeCapture, TestOperation, TestValue>::new();
        let predicate = context.input(ArrayType::scalar(DataType::Boolean).into());
        let offset = context.input(ArrayType::scalar(DataType::F32).into());
        let root = TestValue::Reference(ArrayReference::new(Array::vector(vec![10f32, 20., 30.]).unwrap()));
        let index = TestValue::Array(Array::scalar(2i32).unwrap());
        let (_, then_branch) = NestedTracingContext::trace(
            context.clone(),
            |inputs: Vec<Tracer<_>>| {
                let branch = inputs[0].context().clone();
                let root = StagingContext::constant(&branch, branch.capture(root.clone())?);
                let index = StagingContext::constant(&branch, branch.capture(index.clone())?);
                let element = ReferenceView::new(root)?.dynamic_index(0, &index)?.read()?;
                Ok(branch
                    .bind(AddOperation::<ArrayIrType>::new(), Vec::new(), &[element, inputs[0].clone()])?
                    .remove(0))
            },
            vec![ArrayType::scalar(DataType::F32).into()],
        )
        .unwrap();
        let (_, else_branch) = NestedTracingContext::trace(
            context.clone(),
            |inputs: Vec<Tracer<_>>| Ok(inputs[0].clone()),
            vec![ArrayType::scalar(DataType::F32).into()],
        )
        .unwrap();
        let access = &then_branch.instructions()[0];
        assert_eq!(access.inputs().len(), 2);
        assert_eq!(access.operation().reference_access_descriptor(0).unwrap().bindings(), 1..2);
        assert_eq!(
            access.operation().reference_access_descriptor(0).unwrap().transforms(),
            &[ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }],
        );
        let output = context
            .bind(ConditionOperation::<ArrayIrType>::new(), vec![then_branch, else_branch], &[predicate, offset])
            .unwrap()
            .remove(0);
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let captures = context.captures().borrow().clone();
        assert_eq!(captures, vec![root, index.clone()]);

        // Discharge threads the captured root's state as an array input while the index capture stays an ordinary
        // value, and executing the result selects the element the captured index names when the branch runs.
        let closed = ClosedProgram::new(program, captures.clone()).unwrap();
        let discharged = closed.discharge_references().unwrap();
        assert_eq!(
            discharged.program().input_types(),
            vec![
                ArrayType::new_static(DataType::F32, [3]).into(),
                ArrayType::scalar(DataType::I32).into(),
                ArrayType::scalar(DataType::Boolean).into(),
                ArrayType::scalar(DataType::F32).into(),
            ],
        );
        assert!(!discharged.program().entry_region_ref().contains_reference_accesses_in_closure());
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Capture { index: 0 });
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        let executable = resolve_captures(discharged.program(), &captures);
        let state = TestValue::Array(Array::vector(vec![10f32, 20., 30.]).unwrap());
        assert_eq!(
            executable.interpret(vec![
                state.clone(),
                index.clone(),
                TestValue::Array(Array::scalar(true).unwrap()),
                TestValue::Array(Array::scalar(0.5f32).unwrap()),
            ]),
            Ok(vec![TestValue::Array(Array::scalar(30.5f32).unwrap())]),
        );
        assert_eq!(
            executable.interpret(vec![
                state,
                index,
                TestValue::Array(Array::scalar(false).unwrap()),
                TestValue::Array(Array::scalar(0.5f32).unwrap()),
            ]),
            Ok(vec![TestValue::Array(Array::scalar(0.5f32).unwrap())]),
        );
    }

    #[test]
    fn test_condition_reference_discharge_resolves_reference_captures_inside_condition_regions() {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut branch_builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let reference = branch_builder.add_constant(DischargeCapture::new(0, reference_type.into()));
        let value = branch_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let value = builder
            .add_instruction(ConditionOperation::<ArrayIrType>::new(), vec![branch, branch], vec![predicate], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let reference = ArrayReference::new(Array::scalar(4.0f32).unwrap());
        let closed = ClosedProgram::new(program, vec![ArrayIrValue::Reference(reference)]).unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(
            discharged.external_reference_bindings(),
            &[ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None)],
        );
        assert_eq!(
            serde_json::to_string(discharged.external_reference_bindings()).unwrap(),
            r#"[{"source":{"capture":{"index":0}},"output_index":null}]"#,
        );
        assert_eq!(
            discharged.program().input_types(),
            vec![ArrayType::scalar(DataType::F32).into(), ArrayType::scalar(DataType::Boolean).into()],
        );
    }

    #[test]
    fn test_condition_reference_discharge_matches_eager_reference_execution() {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let update = true_builder.add_constant(TestValue::Array(Array::scalar::<f32>(1.0).unwrap()));
        true_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.clone().into());
        let replacement = false_builder.add_constant(TestValue::Array(Array::scalar::<f32>(9.0).unwrap()));
        let snapshot = false_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // Each branch mutates the shared allocation differently, so the eager reference interpreter and the discharged
        // program must agree on the branch snapshot as well as on the state observed after the condition.
        let discharged = source.clone().discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings(), &[]);
        for (predicate, expected) in [
            (
                true,
                vec![
                    TestValue::Array(Array::scalar::<f32>(5.0).unwrap()),
                    TestValue::Array(Array::scalar::<f32>(5.0).unwrap()),
                ],
            ),
            (
                false,
                vec![
                    TestValue::Array(Array::scalar::<f32>(4.0).unwrap()),
                    TestValue::Array(Array::scalar::<f32>(9.0).unwrap()),
                ],
            ),
        ] {
            let inputs = vec![
                TestValue::Array(Array::scalar(predicate).unwrap()),
                TestValue::Array(Array::scalar::<f32>(4.0).unwrap()),
            ];
            let eager = source.clone().interpret(inputs.clone()).unwrap();
            assert_eq!(eager, expected);
            assert_eq!(discharged.program().interpret(inputs), Ok(eager));
        }
    }

    #[test]
    fn test_condition_reference_discharge_preserves_folded_path_inside_region() {
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3)]));
        let reference_type = ReferenceType::new(vector_type.clone());
        let true_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let update = builder.add_constant(TestValue::Array(Array::scalar::<f32>(1.0).unwrap()));
            builder
                .add_instruction(
                    ReferenceAddUpdateOperation::new().with_transforms(vec![ArrayReferenceTransform::Index {
                        axis: 0,
                        index: ArrayReferenceTransformIndex::Static(1),
                    }]),
                    Vec::new(),
                    vec![reference, update],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(vector_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let reference = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let true_inputs = vec![
            TestValue::Array(Array::scalar(true).unwrap()),
            TestValue::Array(Array::vector::<f32>(vec![1.0, 2.0, 3.0]).unwrap()),
        ];
        let false_inputs = vec![
            TestValue::Array(Array::scalar(false).unwrap()),
            TestValue::Array(Array::vector::<f32>(vec![1.0, 2.0, 3.0]).unwrap()),
        ];
        assert_eq!(
            source.clone().interpret(true_inputs.clone()),
            Ok(vec![TestValue::Array(Array::vector::<f32>(vec![1.0, 3.0, 3.0]).unwrap())])
        );
        assert_eq!(
            source.clone().interpret(false_inputs.clone()),
            Ok(vec![TestValue::Array(Array::vector::<f32>(vec![1.0, 2.0, 3.0]).unwrap())])
        );

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().interpret(true_inputs),
            Ok(vec![TestValue::Array(Array::vector::<f32>(vec![1.0, 3.0, 3.0]).unwrap())])
        );
        assert_eq!(
            discharged.program().interpret(false_inputs),
            Ok(vec![TestValue::Array(Array::vector::<f32>(vec![1.0, 2.0, 3.0]).unwrap())])
        );
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[3] .
                let %2:f32[3] = condition %0 %1 [
                    true={
                        lambda %0:f32[3] .
                        let %1:f32[] = const 1.0
                            %2:f32[1] = slice [start_indices=[1], limits=[2]] %0
                            %3:f32[] = reshape [shape=[]] %2
                            %4:f32[] = add %3 %1
                            %5:f32[1] = reshape [shape=[1]] %4
                            %6:f32[3] = update_slice [start_indices=[1]] %0 %5
                        in (%6)
                    },
                    false={
                        lambda %0:f32[3] .
                        in (%0)
                    },
                ]
                in (%2)"},
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
        let branch_input_type = ArrayType::scalar(DataType::F64);
        let branch = |label| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(branch_input_type.clone());
            builder.add_instruction(PrintOperation::new(label), Vec::new(), vec![input], None).unwrap();
            let output = builder.add_constant(Array::scalar(1.0).unwrap());
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let true_branch = branch("true");
        let false_branch = branch("false");
        assert!(true_branch.partition(&[false]).unwrap().residual_program().effects().classes().is_ordered());
        assert!(false_branch.partition(&[false]).unwrap().residual_program().effects().classes().is_ordered());

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_region(true_branch.entry_region_ref());
        let false_region = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let branch_input = builder.add_input(branch_input_type.clone());
        let output = builder
            .add_instruction(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, branch_input],
                None,
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
                &[PartialValue::Known(symbolic_predicate), PartialValue::Unknown(branch_input_type)],
            )
            .unwrap();

        assert!(matches!(evaluation.outputs.as_slice(), [PartialEvaluationOutput::Known(_)]));
        assert!(evaluation.program.effects().classes().is_ordered());
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
                .classes()
                .is_ordered())
        );
    }

    /// A branch whose known-ness split would leave one reference root reachable from both sides (here a known reference
    /// that the residual side writes through a residual edge) keeps the conditional whole instead of hoisting the known
    /// branch work ahead of it, so that the accesses stay in program order and no typed zero is needed for the
    /// reference-typed edge slot of the other branch.
    #[test]
    fn test_condition_partial_evaluation_residualizes_whole_when_a_branch_shares_a_reference_root() {
        use crate::operations::references::ReferenceWriteOperation;
        use crate::partial::{PartialEvaluationOutput, PartialValue};
        use crate::tracing::TracingContext;

        // `f(p, r, x) = if p { write(r, x); x } else { x }`.
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let branch = |writes: bool| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            let x = builder.add_input(scalar_type.clone().into());
            if writes {
                builder
                    .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, x], None)
                    .unwrap();
            }
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![x], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_region = builder.import_program(branch(true));
        let false_region = builder.import_program(branch(false));
        let predicate = builder.add_input(predicate_type.clone().into());
        let reference = builder.add_input(reference_type.clone().into());
        let x = builder.add_input(scalar_type.clone().into());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_region, false_region],
                vec![predicate, reference, x],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // A symbolic known predicate cannot select a branch, and the true branch's partition would need the known
        // reference as a residual edge, so the whole condition residualizes with the reference as a residual reference
        // and nothing is hoisted into the outer program.
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(predicate_type.into())),
                    PartialValue::Known(outer.input(reference_type.into())),
                    PartialValue::Unknown(scalar_type.into()),
                ],
            )
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        assert!(matches!(evaluation.outputs(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![2]);
        assert_eq!(
            evaluation
                .program()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["condition"],
        );
    }

    /// A branch whose fold fails under an unknown predicate (here an integer division by a known zero divisor in a
    /// branch that interpretation may never take) keeps the conditional whole instead of failing partial evaluation,
    /// so the branch's error surfaces only if that branch actually runs.
    #[test]
    fn test_condition_partial_evaluation_keeps_erroring_branch_folds_behind_the_predicate() {
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch_input_type = ArrayType::scalar(DataType::I32);
        let divide_branch = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(branch_input_type.clone());
            let one = builder.add_constant(Array::from_elements::<i32>(branch_input_type.clone(), &[1]).unwrap());
            let output = builder
                .add_instruction(ArrayOperation::Div(DivOperation::new()), Vec::new(), vec![one, input], None)
                .unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let identity_branch = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(branch_input_type.clone());
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let true_region = builder.import_region(divide_branch.entry_region_ref());
        let false_region = builder.import_region(identity_branch.entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let branch_input = builder.add_input(branch_input_type.clone());
        let output = builder
            .add_instruction(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate, branch_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // The zero divisor is known, so shrinking the branches would fold `1 / 0` speculatively; the rule must fall
        // back to residualizing the conditional whole.
        let knowledge = vec![
            PartialValue::Unknown(predicate_type),
            PartialValue::Known(Array::from_elements::<i32>(branch_input_type.clone(), &[0]).unwrap()),
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
                PartialEvaluationInput::Unknown(_) => {
                    Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap()
                }
                PartialEvaluationInput::Known(value) => value.clone(),
            })
            .collect::<Vec<_>>();
        let outputs = evaluation.program.interpret(inputs).unwrap();
        assert_eq!(outputs[0].elements::<i32>(), Ok(vec![0]));
    }

    #[test]
    fn test_composite_condition_partial_evaluation_refines_outputs_of_unspecialized_branches() {
        // Partially evaluating a condition whose refined output comes from unspecialized branches keeps it refined, and
        // the residual program reproduces the original one, whether the predicate is concrete or unknown.
        let program = refined_vector_condition_program();
        let vector_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        assert_eq!(program.output_types(), vec![vector_type.clone()]);
        let arguments = vec![
            array(Array::scalar(true).unwrap()),
            array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
            array(Array::vector(vec![10.0, 20.0, 30.0]).unwrap()),
        ];
        let expected = vec![array(Array::vector(vec![11.0, 24.0, 39.0]).unwrap())];
        for partial_inputs in [
            vec![
                PartialValue::Known(arguments[0].clone()),
                PartialValue::Known(arguments[1].clone()),
                PartialValue::Unknown(vector_type.clone()),
            ],
            vec![
                PartialValue::Unknown(ArrayType::scalar(DataType::Boolean).into()),
                PartialValue::Known(arguments[1].clone()),
                PartialValue::Unknown(vector_type.clone()),
            ],
        ] {
            let evaluation = program.partially_evaluate(&partial_inputs).unwrap();
            assert_eq!(evaluation.program.output_types(), vec![vector_type.clone()]);
            let residual_arguments = evaluation
                .inputs
                .iter()
                .map(|input| match input {
                    PartialEvaluationInput::Known(value) => value.clone(),
                    PartialEvaluationInput::Unknown(index) => arguments[*index].clone(),
                })
                .collect::<Vec<_>>();
            assert_eq!(evaluation.program.interpret(residual_arguments), Ok(expected.clone()));
        }

        // A symbolic known predicate cannot select a branch. The branch partitions are computed over the unspecialized
        // branches, so the residual edge `x * x` has the identity-bearing type `f64[rows]`, for which no peer-branch
        // placeholder is invented: the condition remains whole, and its output stays refined.
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(ArrayType::scalar(DataType::Boolean).into())),
                    PartialValue::Known(outer.input(vector_type.clone())),
                    PartialValue::Unknown(vector_type.clone()),
                ],
            )
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), TestOperation::Condition(_)));
        assert_eq!(evaluation.program.output_types(), vec![vector_type]);
    }

    #[test]
    fn test_composite_condition_partial_evaluation_retains_dynamic_residual_edges() {
        // Ordinary partial evaluation remains conservative: a symbolic known predicate and dynamic residual edge
        // retain the whole condition, without staging either branch's arithmetic in the outer known context.
        let (program, extent_type, input_type) = dynamic_extent_condition_program();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(ArrayType::scalar(DataType::Boolean).into())),
                    PartialValue::Known(outer.input(extent_type.into())),
                    PartialValue::Unknown(input_type.into()),
                ],
            )
            .unwrap();
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), TestOperation::Condition(_)));
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
            let branch_input = ArrayBatch::new(
                Array::from_elements::<f64>(batched_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                BatchAxis::new(0),
            )
            .unwrap();
            let unbatched_type = branch_input.unbatched_type();
            let (_, branch) =
                EagerContext::<Array, ArrayOperation<Array>>::trace(|inputs: Vec<_>| Ok(inputs), vec![unbatched_type])
                    .unwrap();
            let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));
            let predicate = ArrayBatch::replicated(
                Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap(),
            );

            let outputs = context
                .bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![branch.clone(), branch],
                    &[
                        BatchingTracer::new(context.clone(), predicate),
                        BatchingTracer::new(context.clone(), branch_input),
                    ],
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
        // branch (previously this surfaced a `Concretization` error), so the staged batching rule batches both branch
        // programs at the batch axes of the non-predicate inputs and stages exactly one `condition` operation over
        // them, with the unbatched predicate passed through. Interpreting the staged batched program with both concrete
        // predicate values matches the eager operational path item for item (scale by 2 when true and by 3 when false).
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let branch_input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let predicate_atom = builder.borrow_mut().add_input(predicate_type.clone());
        let branch_input_atom = builder.borrow_mut().add_input(branch_input_type);
        let predicate_tracer = parent.tracer(predicate_atom, None);
        let branch_input_tracer = parent.tracer(branch_input_atom, None);
        let output = batch(
            |(predicate, x)| {
                let condition_regions = vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)];
                let condition = ConditionOperation::new();
                let op = ArrayOperation::Condition(condition);
                let outputs = x.context().bind(op, condition_regions, &[predicate.clone(), x.clone()])?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (predicate_tracer, branch_input_tracer),
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
        let truthy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap();
        let falsy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap();
        let branch_input = Array::vector(vec![1.0, 4.0, 9.0]).unwrap();
        assert_eq!(program.interpret((truthy, branch_input.clone())).unwrap().to_f64s(), vec![2.0, 8.0, 18.0]);
        assert_eq!(program.interpret((falsy, branch_input)).unwrap().to_f64s(), vec![3.0, 12.0, 27.0]);
    }

    /// The replicated-predicate rule discovers each branch's natural output axes before instantiating both branches at
    /// the joined layout. `AlignEachTo` stages axis movement only where a natural axis differs from a mapped target, so
    /// a branch whose discovered axes already equal the joined targets keeps its discovery program and the rule
    /// performs one structural pass for it instead of two.
    #[test]
    fn test_condition_batching_reuses_naturally_aligned_branch_programs() {
        let packed_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let truthy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap();
        let falsy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap();
        let branch_input_values =
            Array::from_elements::<f64>(packed_type.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Both branches scale the batched input per batch item, so both discover axis 0 and the joined layout equals
        // each branch's discovered layout: the rule batches each branch exactly once.
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let branch_input_atom = builder.borrow_mut().add_input(packed_type.clone());
        let predicate = parent.tracer(predicate_atom, None);
        let branch_input = parent.tracer(branch_input_atom, None);
        let context = BatchingContext::new(parent, 2);
        let inputs = vec![ArrayBatch::replicated(predicate), ArrayBatch::new(branch_input, BatchAxis::new(0)).unwrap()];
        let regions = vec![vector_scale_branch(3, 2.0), vector_scale_branch(3, 3.0)];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = ConditionOperation::new().batch(&context, &driver, inputs.as_slice()).unwrap().into_parts().0;
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
                        let %1:f64[] = const 2.0
                            %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[]] %1
                            %3:f64[2, 3] = mul %0 %2
                        in (%3)
                    },
                    false={
                        lambda %0:f64[2, 3] .
                        let %1:f64[] = const 3.0
                            %2:f64[2, 3] = broadcast [output_type=f64[2, 3], output_axes=[]] %1
                            %3:f64[2, 3] = mul %0 %2
                        in (%3)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            program.interpret((truthy.clone(), branch_input_values.clone())).unwrap().to_f64s(),
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        );
        assert_eq!(
            program.interpret((falsy.clone(), branch_input_values.clone())).unwrap().to_f64s(),
            vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
        );

        // A replicated false-branch output disagrees with the joined layout, so only that branch is re-batched to
        // broadcast its output across the batch: three structural passes in total.
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let branch_input_atom = builder.borrow_mut().add_input(packed_type.clone());
        let predicate = parent.tracer(predicate_atom, None);
        let branch_input = parent.tracer(branch_input_atom, None);
        let context = BatchingContext::new(parent, 2);
        let inputs = vec![ArrayBatch::replicated(predicate), ArrayBatch::new(branch_input, BatchAxis::new(0)).unwrap()];
        let regions = vec![vector_scale_branch(3, 2.0), constant_vector_branch(vec![10.0, 20.0, 30.0])];
        let driver = CountingBatchingDriver::new(&regions);
        let outputs = ConditionOperation::new().batch(&context, &driver, inputs.as_slice()).unwrap().into_parts().0;
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
            program.interpret((truthy, branch_input_values.clone())).unwrap().to_f64s(),
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        );

        // The re-batched false branch broadcasts its replicated constant across the batch, so every batch item
        // receives the same constant vector.
        assert_eq!(
            program.interpret((falsy, branch_input_values)).unwrap().to_f64s(),
            vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
        );
    }

    #[test]
    fn test_condition_batching_normalizes_replicated_branch_output_axes() {
        // The two branches of a staged batched condition may disagree on their natural output batch axes: here the true
        // branch scales the batched input per batch item (axis 0) while the false branch returns a replicated constant
        // (no batch axis). The staged rule normalizes the false branch by appending a broadcast at its tail, so the
        // staged condition stays well-typed and both predicate values interpret correctly per batch item.
        let mut constant_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        constant_builder.add_input(ArrayType::scalar(DataType::F64));
        let constant_output = constant_builder.add_constant(Array::scalar(7.0).unwrap());
        let constant_branch = constant_builder
            .build::<Vec<Array>, Vec<Array>>(vec![constant_output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let predicate_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::Boolean));
        let branch_input_atom = builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])));
        let predicate_tracer = parent.tracer(predicate_atom, None);
        let branch_input_tracer = parent.tracer(branch_input_atom, None);
        let output = batch(
            |(predicate, x)| {
                let condition_regions = vec![scalar_scale_branch(2.0), constant_branch];
                let condition = ConditionOperation::new();
                let op = ArrayOperation::Condition(condition);
                let outputs = x.context().bind(op, condition_regions, &[predicate.clone(), x.clone()])?;
                Ok(outputs.into_iter().next().unwrap())
            },
            (predicate_tracer, branch_input_tracer),
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
        let truthy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap();
        let falsy = Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap();
        let branch_input = Array::vector(vec![1.0, 4.0, 9.0]).unwrap();
        assert_eq!(program.interpret((truthy, branch_input.clone())).unwrap().to_f64s(), vec![2.0, 8.0, 18.0]);
        assert_eq!(program.interpret((falsy, branch_input)).unwrap().to_f64s(), vec![7.0, 7.0, 7.0]);
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
            (Array::vector(vec![true, false, true]).unwrap(), Array::vector(vec![1.0, 4.0, 9.0]).unwrap()),
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
    fn test_condition_batching_aligns_manual_variation_for_batch_varying_predicates() {
        // Inside a manual region, branch values may vary over manual axes while the predicate stays invariant. The
        // selections that gate the branch inputs and merge the branch outputs per batch item first give the Boolean
        // predicate, which carries no tangent, the variation of the branch values through `parallel_vary`.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = |rank| Sharding::replicated(mesh.clone(), rank).with_varying_manual_axes(["x"]).unwrap();
        let predicate_type = ArrayType::new_static(DataType::Boolean, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let value_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(varying_sharding(1)).unwrap();
        let branch_type = ArrayType::scalar(DataType::F64).with_sharding(varying_sharding(0)).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(branch_type.clone());
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let sine_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sine], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(branch_type);
        let identity_branch =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<Tracer<TracingContext<Array, ArrayOperation<Array>>>>| {
                let context = BatchingContext::<_, ArrayBatchingPolicy>::new(inputs[0].dispatch_domain(), 2);
                let predicate =
                    BatchingTracer::new(context.clone(), ArrayBatch::new(inputs[0].clone(), BatchAxis::new(0))?);
                let value =
                    BatchingTracer::new(context.clone(), ArrayBatch::new(inputs[1].clone(), BatchAxis::new(0))?);
                let outputs = context.bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![sine_branch, identity_branch],
                    &[predicate, value],
                )?;
                Ok(outputs.into_iter().next().unwrap().into_batch().into_value())
            },
            vec![predicate_type, value_type.clone()],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(output_type, value_type);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[2][sharding={mesh<['x'=2:manual]>, [{}]}], \
                    %1:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %2:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = stop_gradient %1
                    %3:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %4:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %3 %1 %2
                    %5:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sin %4
                    %6:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %7:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %6 %2 %1
                    %8:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %9:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %8 %5 %7
                in (%9)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_condition_batching_aligns_replicated_and_mapped_branch_outputs() {
        let batch_size = 2;
        let item_size = 3;
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(batch_size)]));
        let predicate =
            ArrayBatch::new(Array::from_elements::<bool>(predicate_type, &[true, false]).unwrap(), BatchAxis::new(0))
                .unwrap();
        let branch_input = Array::matrix(batch_size, item_size, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let branch_input = ArrayBatch::new(branch_input, BatchAxis::new(0)).unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), batch_size);

        let outputs = context
            .bind(
                ArrayOperation::Condition(ConditionOperation::new()),
                vec![constant_vector_branch(vec![10.0, 20.0, 30.0]), vector_scale_branch(item_size, 3.0)],
                &[BatchingTracer::new(context.clone(), predicate), BatchingTracer::new(context.clone(), branch_input)],
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
                .add_instruction(ArrayOperation::Print(PrintOperation::new(label)), Vec::new(), vec![input], None)
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
            (Array::vector(vec![true, false]).unwrap(), Array::vector(vec![1.0, 2.0]).unwrap()),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(
            result,
            Err(BatchingError::UnsupportedOperation {
                message: "cannot batch a `condition` with a batch-varying predicate and effectful branches because \
                          observable effects cannot be selected per batch item"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_condition_batching_rejects_batch_varying_dimension_results() -> Result<(), ProgramError> {
        // Under a batch-varying predicate, dimension results of the two branches stay replicated and are guarded by an
        // equality assertion, so branches whose dimension results differ fail during execution instead of becoming
        // ragged values.
        let mut true_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let true_extent = true_builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(2)?));
        let true_branch = true_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![true_extent],
            Vec::new(),
            vec![Placeholder],
        )?;
        let mut false_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let false_extent = false_builder.add_constant(ArrayIrValue::Dimension(DimensionValue::constant(3)?));
        let false_branch = false_builder.build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            vec![false_extent],
            Vec::new(),
            vec![Placeholder],
        )?;
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let predicate =
            trace.input(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let outputs = context
            .bind(
                ArrayIrOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                &[BatchingTracer::new(context.clone(), ArrayIrBatch::new(predicate, BatchAxis::new(0))?)],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert!(
            trace
                .builder()
                .borrow()
                .instructions()
                .iter()
                .any(|instruction| { instruction.operation().name() == "assert" })
        );
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            Vec::new(),
            vec![Placeholder; 2],
            Vec::new(),
        )?;
        let error = program
            .interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(DimensionType::from(batch), 2)?),
                ArrayIrValue::Array(Array::vector(vec![true, false])?),
            ])
            .unwrap_err();
        assert_eq!(
            error.downcast_custom::<AssertionError>(),
            Some(&AssertionError::Failed {
                message: "branch dimensions must agree".to_owned(),
                observations: vec![("true".to_owned(), "2".to_owned()), ("false".to_owned(), "3".to_owned())],
            }),
        );
        Ok(())
    }

    #[test]
    fn test_composite_condition_batching_refines_outputs_of_unspecialized_branches() {
        // Batching a condition whose refined output comes from unspecialized branches keeps its batched output refined,
        // under replicated and mapped predicates alike.
        let matrix = |values: &[f64]| {
            array(Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 3]), values).unwrap())
        };
        let x = matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = matrix(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let extent = DimensionValue::constant(2).unwrap();
        for (predicate_axis, predicate, expected) in [
            (
                BatchAxis::replicated(),
                array(Array::scalar(true).unwrap()),
                matrix(&[11.0, 24.0, 39.0, 56.0, 75.0, 96.0]),
            ),
            (
                BatchAxis::new(0),
                array(Array::vector(vec![true, false]).unwrap()),
                matrix(&[11.0, 24.0, 39.0, 44.0, 55.0, 66.0]),
            ),
        ] {
            let batched = refined_vector_condition_program()
                .batched_with_threaded_extent(
                    extent.r#type().into_owned(),
                    ShardingDimension::Replicated,
                    &[predicate_axis, BatchAxis::new(0), BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .unwrap()
                .into_parts()
                .0;
            assert_eq!(batched.output_types()[1], ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3])));
            assert_eq!(
                batched.interpret(vec![TestValue::Dimension(extent.clone()), predicate, x.clone(), y.clone()]),
                Ok(vec![TestValue::Dimension(extent.clone()), expected]),
            );
        }
    }

    #[test]
    fn test_composite_condition_batching_threads_reference_carries_under_replicated_predicate() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let current = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        true_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, current], None)
            .unwrap();
        let snapshot = true_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.into());
        let snapshot = false_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_program(true_branch);
        let false_branch = builder.import_program(false_branch);
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot, frozen], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // A replicated predicate keeps one structural condition through which the mapped reference carry flows like an
        // array carry: the taken branch doubles every item in place and its snapshot comes back at the referent's axis,
        // so batching the stateful program directly agrees with batching its discharged counterpart for either branch.
        let axis_extent = DimensionValue::constant(2).unwrap();
        let extent_type = axis_extent.r#type().into_owned();
        let direct = source
            .batched_with_threaded_extent(
                extent_type.clone(),
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        let discharged = source
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .batched_with_threaded_extent(
                extent_type,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(direct.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(discharged.output_axes(), direct.output_axes());
        let (direct, _) = direct.into_parts();
        let (discharged, _) = discharged.into_parts();
        assert_eq!(
            direct.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:bool[], %2:f32[2] .
                let %3:ref<f32[2]> = reference_new %2
                    %4:dimension<2>, %5:f32[2] = condition %1 %0 %3 [
                        true={
                            lambda %0:dimension<2>, %1:ref<f32[2]> .
                            let %2:f32[2] = reference_read %1
                                () = reference_add_update %1 %2
                                %3:f32[2] = reference_read %1
                            in (%0, %3)
                        },
                        false={
                            lambda %0:dimension<2>, %1:ref<f32[2]> .
                            let %2:f32[2] = reference_read %1
                            in (%0, %2)
                        },
                    ]
                    %6:f32[2] = reference_freeze %3
                in (%0, %5, %6)"},
        );

        // Generic program interpretation cannot bind a reference entering a branch region (stateful compilation
        // domains own that boundary), so the runtime agreement is checked through the discharged program.
        for (predicate, expected) in [(true, vec![2.0f32, 4.0]), (false, vec![1.0f32, 2.0])] {
            let inputs = vec![
                TestValue::Dimension(axis_extent.clone()),
                TestValue::Array(Array::scalar(predicate).unwrap()),
                TestValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
            ];
            let expected = vec![
                TestValue::Dimension(axis_extent.clone()),
                TestValue::Array(Array::vector(expected.clone()).unwrap()),
                TestValue::Array(Array::vector(expected).unwrap()),
            ];
            assert_eq!(discharged.interpret(inputs), Ok(expected));
        }
    }

    #[test]
    fn test_composite_condition_batching_rejects_reference_access_under_mapped_predicate() {
        type Parent = EagerContext<TestValue, TestOperation>;

        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = branch_builder.add_input(reference_type.into());
        let snapshot = branch_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // Select lowering runs both branches, so even a read-only access in one branch is rejected ahead of the general
        // purity check, naming the reference access as the cause.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            Parent::new(),
            TestValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let reference = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let error = context
            .bind(
                TestOperation::Condition(ConditionOperation::new()),
                vec![branch.clone(), branch],
                &[
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(
                            TestValue::Array(Array::vector(vec![true, false]).unwrap()),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(TestValue::Reference(reference), BatchAxis::new(0)).unwrap(),
                    ),
                ],
            )
            .unwrap_err();
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::UnsupportedOperation { message })
                if message == "cannot batch a `condition` with a batch-varying predicate whose branches access \
                               references because select lowering runs both branches and reference effects cannot be \
                               selected per batch item",
        ));
    }

    #[test]
    fn test_composite_condition_batching_forwards_untouched_reference_under_mapped_predicate() {
        type Parent = EagerContext<TestValue, TestOperation>;

        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut true_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = true_builder.add_input(reference_type.clone().into());
        let value = true_builder.add_input(scalar_type.clone().into());
        let doubled = true_builder
            .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![value, value], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![reference, doubled],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut false_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = false_builder.add_input(reference_type.into());
        let value = false_builder.add_input(scalar_type.into());
        let false_branch = false_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference, value], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // A branch that merely forwards a reference it never accesses is fine under a batch-varying predicate: the
        // reference output is never selected but passes through as the region input both branches forward, while the
        // array output is selected per item.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            Parent::new(),
            TestValue::Dimension(DimensionValue::constant(2).unwrap()),
        );
        let reference = ArrayReference::new(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let outputs = context
            .bind(
                TestOperation::Condition(ConditionOperation::new()),
                vec![true_branch, false_branch],
                &[
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(
                            TestValue::Array(Array::vector(vec![true, false]).unwrap()),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(TestValue::Reference(reference.clone()), BatchAxis::new(0)).unwrap(),
                    ),
                    BatchingTracer::new(
                        context.clone(),
                        ArrayIrBatch::new(
                            TestValue::Array(Array::vector(vec![3.0f32, 4.0]).unwrap()),
                            BatchAxis::new(0),
                        )
                        .unwrap(),
                    ),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].batch().value(), &TestValue::Reference(reference.clone()));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().value(), &TestValue::Array(Array::vector(vec![6.0f32, 4.0]).unwrap()));
        assert_eq!(reference.read(), Ok(Array::vector(vec![1.0f32, 2.0]).unwrap()));
    }

    #[test]
    fn test_composite_condition_batching_aligns_manual_variation_for_batch_varying_predicates() {
        // Composite batching gives the invariant Boolean predicate the variation of the array branch values in the same
        // way as array batching, for both the gating and the merging selections.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = |rank| Sharding::replicated(mesh.clone(), rank).with_varying_manual_axes(["x"]).unwrap();
        let predicate_type = ArrayType::new_static(DataType::Boolean, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let value_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(varying_sharding(1)).unwrap();
        let branch_type = ArrayType::scalar(DataType::F64).with_sharding(varying_sharding(0)).unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(branch_type.clone().into());
        let sine = builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(SinOperation::new())),
                Vec::new(),
                vec![input],
                None,
            )
            .unwrap()[0];
        let sine_branch = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sine], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(branch_type.into());
        let identity_branch = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![input], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let (output_type, program) = TracingContext::<TestValue, TestOperation>::trace_with_named_axes(
            |inputs: Vec<Tracer<TracingContext<TestValue, TestOperation>>>| {
                let parent = inputs[0].dispatch_domain();
                let extent = parent.constant(TestValue::Dimension(DimensionValue::constant(2).unwrap()));
                let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent, extent);
                let predicate =
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(inputs[0].clone(), BatchAxis::new(0))?);
                let value =
                    BatchingTracer::new(context.clone(), ArrayIrBatch::new(inputs[1].clone(), BatchAxis::new(0))?);
                let outputs = context.bind(
                    TestOperation::Condition(ConditionOperation::new()),
                    vec![sine_branch, identity_branch],
                    &[predicate, value],
                )?;
                Ok(outputs.into_iter().next().unwrap().into_batch().into_value())
            },
            vec![ArrayIrType::Array(predicate_type), ArrayIrType::Array(value_type.clone())],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(value_type));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:bool[2][sharding={mesh<['x'=2:manual]>, [{}]}], \
                    %1:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %2:dimension<2> = const 2
                    %3:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = stop_gradient %1
                    %4:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %5:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %4 %1 %3
                    %6:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sin %5
                    %7:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %8:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %7 %3 %1
                    %9:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %0
                    %10:f64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %9 %6 %8
                in (%10)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_condition_batching_after_local_reference_discharge() {
        // Discharge across a condition: the mapped predicate turns the condition into a select over both discharged
        // branch states while the batched program stays pure and reference-free.
        let program = test_condition_program();

        let axis_extent = DimensionValue::constant(2).unwrap();
        let batched = program
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .batched_with_threaded_extent(
                axis_extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        let (batched, _) = batched.into_parts();
        assert!(batched.effects().classes().is_empty());
        assert!(batched.regions().iter().flat_map(|region| region.atoms()).all(|atom| !atom.r#type().is_reference()));

        // Mixed predicates pin that each batch item selects its own branch's state: the accumulating true branch
        // yields `4 + 1` for both outputs, while the overwriting false branch yields the pre-swap `7` snapshot and
        // the replacement `9` as the final state.
        assert_eq!(
            batched
                .interpret(vec![
                    ArrayIrValue::Dimension(axis_extent.clone()),
                    ArrayIrValue::Array(Array::vector(vec![true, false]).unwrap()),
                    ArrayIrValue::Array(Array::vector(vec![4.0f32, 7.0]).unwrap()),
                ])
                .unwrap(),
            vec![
                ArrayIrValue::Dimension(axis_extent),
                ArrayIrValue::Array(Array::vector(vec![5.0f32, 7.0]).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![5.0f32, 9.0]).unwrap()),
            ],
        );
    }

    #[test]
    fn test_condition_linearization_replays_the_selected_branch() {
        for (predicate, expected_value, expected_tangent) in
            [(true, 1.4, 3.0), (false, 0.7f64.sin(), 1.5 * 0.7f64.cos())]
        {
            let (value, pushforward) = differentiate_at(Array::scalar(0.7).unwrap())
                .linearize(move |input| {
                    let predicate = input.context().lift(
                        Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[predicate]).unwrap(),
                    )?;
                    let mut outputs = input.context().bind(
                        ArrayOperation::Condition(ConditionOperation::new()),
                        vec![
                            scalar_branch(ArrayOperation::Add(AddOperation::new())),
                            scalar_branch(ArrayOperation::Sin(SinOperation::new())),
                        ],
                        &[predicate, input.clone()],
                    )?;
                    Ok(outputs.remove(0))
                })
                .unwrap();
            assert_eq!(value, Array::scalar(expected_value).unwrap());
            assert_eq!(pushforward.apply(Array::scalar(1.5).unwrap()), Ok(Array::scalar(expected_tangent).unwrap()));
        }
    }

    #[test]
    fn test_condition_linearization_accepts_unit_returning_branches() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let branch = || {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            builder.add_input(scalar_type.clone());
            builder.build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(scalar_type.clone());
        let true_region = builder.import_program(branch());
        let false_region = builder.import_program(branch());
        builder
            .add_instruction(ConditionOperation::new(), vec![true_region, false_region], vec![predicate, input], None)
            .unwrap();
        let program = builder.build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder; 2], Vec::new()).unwrap();
        assert_eq!(program.instructions().len(), 1);

        // The active numeric input makes linearization replay the operation through its separate-context JVP rule,
        // even though neither branch returns any values. No primal or tangent result is required in this case.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        for predicate in [true, false] {
            assert_eq!(
                linearization
                    .primal()
                    .interpret(vec![Array::scalar(predicate).unwrap(), Array::scalar(3.0).unwrap()]),
                Ok(vec![]),
            );
        }
        assert_eq!(linearization.tangent().interpret(vec![Array::scalar(1.0).unwrap()]), Ok(vec![]));
    }

    #[test]
    fn test_composite_condition_linearization_preserves_dynamic_residual_geometry() {
        let (program, extent_type, _) = dynamic_extent_condition_program();

        // Separate primal and tangent contexts require the untaken branch's residual slots to retain the same live
        // extent. Only the selected branch consumes its residuals, so geometry-aware zero placeholders are sufficient.
        let linearization = program.linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        for (predicate, expected_value, expected_gradient) in
            [(true, vec![4.0f64, 9.0, 16.0], vec![4.0f64, 6.0, 8.0]), (false, vec![4.0f64, 6.0, 8.0], vec![2.0f64; 3])]
        {
            let mut outputs = linearization
                .primal()
                .interpret(vec![
                    array(Array::scalar(predicate).unwrap()),
                    dimension(&extent_type, 3),
                    array(Array::vector(vec![2.0f64, 3.0, 4.0]).unwrap()),
                ])
                .unwrap();
            let residuals = outputs.split_off(1);
            assert_eq!(outputs, vec![array(Array::vector(expected_value).unwrap())]);
            let mut pullback_inputs = vec![array(Array::vector(vec![1.0f64; 3]).unwrap())];
            pullback_inputs.extend(residuals);
            assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![array(Array::vector(expected_gradient).unwrap())]));
        }
    }

    #[test]
    fn test_condition_differentiation_passes_plumbing_references_into_branches() {
        // Both branches read the reference they receive at input position 1, after a numeric input, and add it to
        // that numeric input: `f(p, x, r) = x + read(r)`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let value = builder.add_input(scalar_type.clone().into());
            let reference = builder.add_input(reference_type.clone().into());
            let current =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let output = builder
                .add_instruction(AddOperation::<ArrayIrType>::new(), Vec::new(), vec![value, current], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(branch().entry_region_ref());
        let false_branch = builder.import_region(branch().entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let value = builder.add_input(scalar_type.clone().into());
        let reference = builder.add_input(reference_type.into());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, value, reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        // With the reference inactive it reaches the branches as plumbing at a non-prefix position: the branches
        // receive no tangent input for it (`[x, r, ẋ]`), the read's tangent is zero, and the program still
        // differentiates with `ẏ = ẋ`. The fused program is spliced behind local allocations of the reference inputs so
        // that the interpreted program owns the state it mutates.
        let jvp = program.entry_region_ref().jvp(&[1]).unwrap();
        assert_eq!(jvp.input_types().len(), 4);
        assert_eq!(jvp.output_types().len(), 2);
        let condition =
            jvp.instructions().iter().find(|instruction| instruction.operation().name() == "condition").unwrap();
        assert_eq!(jvp.region_ref(condition.regions()[0]).unwrap().input_types().len(), 3);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let value = builder.add_input(scalar_type.clone().into());
        let state = builder.add_input(scalar_type.clone().into());
        let tangent = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let outputs = builder.splice_program(&jvp, &[predicate, value, reference, tangent]).unwrap();
        let runnable = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            runnable.interpret(vec![
                TestValue::Array(Array::scalar(true).unwrap()),
                TestValue::Array(Array::scalar(2.0f32).unwrap()),
                TestValue::Array(Array::scalar(5.0f32).unwrap()),
                TestValue::Array(Array::scalar(3.0f32).unwrap()),
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar(7.0f32).unwrap()),
                TestValue::Array(Array::scalar(3.0f32).unwrap())
            ]),
        );

        // With the reference active the branches receive its tangent reference too (`[x, r, ẋ, ṛ]`) and the read's
        // tangent is the referenced tangent: `ẏ = ẋ + read(ṛ)`.
        let jvp = program.jvp().unwrap();
        let condition =
            jvp.instructions().iter().find(|instruction| instruction.operation().name() == "condition").unwrap();
        assert_eq!(jvp.region_ref(condition.regions()[0]).unwrap().input_types().len(), 4);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let value = builder.add_input(scalar_type.clone().into());
        let state = builder.add_input(scalar_type.clone().into());
        let tangent = builder.add_input(scalar_type.clone().into());
        let state_tangent = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let reference_tangent = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state_tangent], None)
            .unwrap()[0];
        let outputs = builder.splice_program(&jvp, &[predicate, value, reference, tangent, reference_tangent]).unwrap();
        let runnable = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 5], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            runnable.interpret(vec![
                TestValue::Array(Array::scalar(false).unwrap()),
                TestValue::Array(Array::scalar(2.0f32).unwrap()),
                TestValue::Array(Array::scalar(5.0f32).unwrap()),
                TestValue::Array(Array::scalar(3.0f32).unwrap()),
                TestValue::Array(Array::scalar(0.5f32).unwrap()),
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar(7.0f32).unwrap()),
                TestValue::Array(Array::scalar(3.5f32).unwrap())
            ]),
        );
    }

    #[test]
    fn test_condition_differentiation_forwards_inactive_reference_outputs() {
        // Both branches forward the reference they receive, and the program reads through the conditional's result,
        // so the public output is numeric while the branch outputs are references.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(branch().entry_region_ref());
        let false_branch = builder.import_region(branch().entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(reference_type.into());
        let forwarded = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference],
                None,
            )
            .unwrap()[0];
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![forwarded], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // An inactive reference is forwarded through the branch without a tangent slot. Reading it produces a
        // numeric structural zero, which is materialized at the outer output boundary.
        let inactive_jvp = program.entry_region_ref().jvp(&[]).unwrap();
        assert_eq!(inactive_jvp.input_ids().len(), 2);
        assert_eq!(inactive_jvp.output_ids().len(), 2);
        assert_eq!(
            inactive_jvp.interpret(vec![
                TestValue::Array(Array::scalar(true).unwrap()),
                TestValue::Reference(ArrayReference::new(Array::scalar(5.0f32).unwrap())),
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar(5.0f32).unwrap()),
                TestValue::Array(Array::scalar(0.0f32).unwrap())
            ]),
        );

        // An active reference is forwarded together with its tangent reference, and the read pairs them back up. The
        // fused program is spliced behind local allocations of the reference inputs so that the interpreted program
        // owns the state it mutates.
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_types().len(), 3);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let state = builder.add_input(scalar_type.clone().into());
        let state_tangent = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let reference_tangent = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state_tangent], None)
            .unwrap()[0];
        let outputs = builder.splice_program(&jvp, &[predicate, reference, reference_tangent]).unwrap();
        let runnable = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            runnable.interpret(vec![
                TestValue::Array(Array::scalar(true).unwrap()),
                TestValue::Array(Array::scalar(5.0f32).unwrap()),
                TestValue::Array(Array::scalar(0.5f32).unwrap()),
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar(5.0f32).unwrap()),
                TestValue::Array(Array::scalar(0.5f32).unwrap())
            ]),
        );
    }

    #[test]
    fn test_condition_differentiation_after_local_reference_discharge() {
        let source = test_condition_program();

        // Forward mode, linearization, and transposition all consume the discharged program, so every derived
        // program must be pure and reference-free even though the source threads state through both branches.
        let jvp = source
            .clone()
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .jvp()
            .unwrap();
        let linearization = source
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .linearize()
            .unwrap();
        let pullback = linearization.pullback().unwrap();
        for program in [&jvp, linearization.primal(), linearization.tangent(), &pullback] {
            assert!(!program.entry_region_ref().contains_atom_type_in_closure(Type::is_reference));
            assert!(program.effects().classes().is_empty());
        }

        // The true branch accumulates the input, so both public outputs remain differentiable.
        let predicate = ArrayIrValue::Array(Array::scalar(true).unwrap());
        let initial = ArrayIrValue::Array(Array::scalar(4.0f32).unwrap());
        assert_eq!(
            jvp.interpret(vec![
                predicate.clone(),
                initial.clone(),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(5.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(5.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ]),
        );
        let primal_outputs = linearization.primal().interpret(vec![predicate, initial]).unwrap();
        assert_eq!(
            primal_outputs[..2],
            [ArrayIrValue::Array(Array::scalar(5.0f32).unwrap()), ArrayIrValue::Array(Array::scalar(5.0f32).unwrap())],
        );
        let mut pullback_inputs = vec![
            ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
        ];
        pullback_inputs.extend_from_slice(&primal_outputs[2..]);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![ArrayIrValue::Array(Array::scalar(5.0f32).unwrap())]));

        // The false branch replaces the state with a constant, so the frozen output has zero tangent and contributes
        // no cotangent to the input.
        let predicate = ArrayIrValue::Array(Array::scalar(false).unwrap());
        let initial = ArrayIrValue::Array(Array::scalar(4.0f32).unwrap());
        assert_eq!(
            jvp.interpret(vec![
                predicate.clone(),
                initial.clone(),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(4.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(9.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()),
            ]),
        );
        let primal_outputs = linearization.primal().interpret(vec![predicate, initial]).unwrap();
        assert_eq!(
            primal_outputs[..2],
            [ArrayIrValue::Array(Array::scalar(4.0f32).unwrap()), ArrayIrValue::Array(Array::scalar(9.0f32).unwrap())],
        );
        let mut pullback_inputs = vec![
            ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
        ];
        pullback_inputs.extend_from_slice(&primal_outputs[2..]);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())]));
    }

    #[test]
    fn test_condition_jvp_preserves_zero_space_output_tangents() {
        let (primal, tangent) = differentiate_at(Array::scalar(2.0).unwrap())
            .jvp(Array::scalar(3.0).unwrap(), |input| {
                let predicate = input
                    .context()
                    .lift(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap())?;
                let mut outputs = input.context().bind(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    vec![boolean_branch(), boolean_branch()],
                    &[predicate, input.clone()],
                )?;
                Ok(outputs.remove(0))
            })
            .unwrap();
        assert_eq!(primal, Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap());
        assert_eq!(tangent, Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap());
    }

    #[test]
    fn test_condition_dense_jacobians_replay_runtime_regions() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();

        let forward = context
            .differentiate_at(Array::scalar(4.0).unwrap())
            .jacobian_forward(stage_runtime_predicate_condition)
            .unwrap();
        let reverse = context
            .differentiate_at(Array::scalar(4.0).unwrap())
            .jacobian_reverse(stage_runtime_predicate_condition)
            .unwrap();
        assert_eq!(forward.iter_blocks().next().unwrap().value().to_f64s(), vec![2.0]);
        assert_eq!(reverse.iter_blocks().next().unwrap().value().to_f64s(), vec![2.0]);

        let forward = context
            .differentiate_at(Array::scalar(-4.0).unwrap())
            .jacobian_forward(stage_runtime_predicate_condition)
            .unwrap();
        let reverse = context
            .differentiate_at(Array::scalar(-4.0).unwrap())
            .jacobian_reverse(stage_runtime_predicate_condition)
            .unwrap();
        assert_eq!(forward.iter_blocks().next().unwrap().value().to_f64s(), vec![3.0]);
        assert_eq!(reverse.iter_blocks().next().unwrap().value().to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_condition_vjp_selects_runtime_branch_cotangents() {
        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |(predicate, branch_input), ()| {
                    let mut outputs = predicate.context().bind(
                        ArrayOperation::Condition(ConditionOperation::new()),
                        vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)],
                        &[predicate.clone(), branch_input],
                    )?;
                    Ok(outputs.remove(0))
                },
                (
                    Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap(),
                    Array::scalar(4.0).unwrap(),
                ),
                (),
            )
            .unwrap();
        let cotangents = pullback.apply(Array::scalar(5.0).unwrap()).unwrap();
        assert_eq!(output.to_f64s(), vec![8.0]);
        assert!(cotangents.0.storage_bytes().is_empty());
        assert_eq!(cotangents.1.to_f64s(), vec![10.0]);

        let (output, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |(predicate, branch_input), ()| {
                    let mut outputs = predicate.context().bind(
                        ArrayOperation::Condition(ConditionOperation::new()),
                        vec![scalar_scale_branch(2.0), scalar_scale_branch(3.0)],
                        &[predicate.clone(), branch_input],
                    )?;
                    Ok(outputs.remove(0))
                },
                (
                    Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[false]).unwrap(),
                    Array::scalar(4.0).unwrap(),
                ),
                (),
            )
            .unwrap();
        let cotangents = pullback.apply(Array::scalar(5.0).unwrap()).unwrap();
        assert_eq!(output.to_f64s(), vec![12.0]);
        assert!(cotangents.0.storage_bytes().is_empty());
        assert_eq!(cotangents.1.to_f64s(), vec![15.0]);
    }

    /// Gate measurement for extending the per-[`Region`](crate::Region) transform cache to the `condition`
    /// differentiation rules. Several distinct outer programs attach *one shared* pair of branch regions, which is
    /// exactly the sharing a region-keyed cache can serve, and each outer program is then linearized and transposed
    /// from cold. The printed table reports the frontend cost of each transform per outer program and how it scales
    /// with the branch instruction count, which is the input to deciding whether retaining the branches' derived
    /// programs is worth its complexity.
    #[test]
    #[ignore = "region transform cache gate measurement"]
    fn test_baseline_repeated_condition_branch_transformation() {
        /// Per-branch instruction counts swept by the measurement.
        const BRANCH_OPERATION_COUNTS: [usize; 2] = [2, 200];

        /// Number of distinct outer programs that attach the one shared pair of branch regions.
        const OUTER_SPECIALIZATIONS: usize = 4;

        /// Builds a branch that scales its scalar input by `factor` and then applies a chain of sines.
        fn chained_branch(
            factor: f64,
            operation_count: usize,
        ) -> Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>> {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let factor = builder.add_constant(Array::scalar(factor).unwrap());
            let mut value =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, factor], None).unwrap()[0];
            for _ in 1..operation_count {
                value = builder.add_instruction(SinOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
            }
            Arc::new(
                builder.build::<Vec<Array>, Vec<Array>>(vec![value], vec![Placeholder], vec![Placeholder]).unwrap(),
            )
        }

        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut measurements = Vec::new();
        for branch_operations in BRANCH_OPERATION_COUNTS {
            let true_branch = chained_branch(2.0, branch_operations);
            let false_branch = chained_branch(3.0, branch_operations);

            // Each outer program interns that one pair of branches and differs only in the length of its sine
            // epilogue, so their derived programs are genuinely distinct while the branch regions are shared.
            let outers = (0..OUTER_SPECIALIZATIONS)
                .map(|index| {
                    let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
                    let predicate = builder.add_input(predicate_type.clone());
                    let branch_input = builder.add_input(scalar_type.clone());
                    let regions = vec![
                        builder.intern_callee(&true_branch, None).unwrap(),
                        builder.intern_callee(&false_branch, None).unwrap(),
                    ];
                    let mut value = builder
                        .add_instruction(
                            ArrayOperation::Condition(ConditionOperation::new()),
                            regions,
                            vec![predicate, branch_input],
                            None,
                        )
                        .unwrap()[0];
                    for _ in 0..=index {
                        value = builder.add_instruction(SinOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
                    }
                    builder
                        .build::<Vec<Array>, Vec<Array>>(vec![value], vec![Placeholder, Placeholder], vec![Placeholder])
                        .unwrap()
                })
                .collect::<Vec<_>>();

            let mut rows = Vec::with_capacity(OUTER_SPECIALIZATIONS);
            for outer in &outers {
                let start = Instant::now();
                let linearization = outer.linearize().unwrap();
                let linearized = start.elapsed();
                let start = Instant::now();
                linearization
                    .tangent()
                    .entry_region_ref()
                    .transpose(
                        &(0..linearization.tangent().input_ids().len() - linearization.residual_count())
                            .collect::<Vec<_>>(),
                        &[],
                        &[],
                    )
                    .unwrap();
                rows.push((linearized, start.elapsed()));
            }
            measurements.push((branch_operations, rows));
        }

        println!("condition branch transform gate: one shared branch pair, {OUTER_SPECIALIZATIONS} outer programs");
        for (branch_operations, rows) in &measurements {
            println!("  branches with {branch_operations} operations each (all times in milliseconds):");
            println!("    outer |    linearize |    transpose |        total");
            for (index, (linearized, transposed)) in rows.iter().enumerate() {
                println!(
                    "    {index:>5} | {:>12.3} | {:>12.3} | {:>12.3}",
                    linearized.as_secs_f64() * 1e3,
                    transposed.as_secs_f64() * 1e3,
                    (*linearized + *transposed).as_secs_f64() * 1e3,
                );
            }
        }

        // Repeated-outer cost is the mean over the outer programs after the first, which is what retained branch
        // transforms could serve; the per-operation column reports how much of it is branch-proportional.
        let repeated_mean = |rows: &[(Duration, Duration)]| {
            rows[1..]
                .iter()
                .map(|(linearized, transposed)| (*linearized + *transposed).as_secs_f64() * 1e3)
                .sum::<f64>()
                / (rows.len() - 1) as f64
        };
        let (small_operations, small_rows) = &measurements[0];
        let (large_operations, large_rows) = &measurements[1];
        let small_mean = repeated_mean(small_rows);
        let large_mean = repeated_mean(large_rows);
        println!(
            "  repeated-outer summary (mean over outers 1..{}, milliseconds): {small_operations}-op branches \
             {small_mean:.3}, {large_operations}-op branches {large_mean:.3}, per branch operation {:.4}",
            OUTER_SPECIALIZATIONS,
            (large_mean - small_mean) / (large_operations - small_operations) as f64,
        );
    }

    /// The `condition` differentiation rules reach their branches through the per-[`Region`](crate::Region) transform
    /// cache, so several programs attaching one shared pair of branches derive each branch's fused forward-mode
    /// program once and each branch's transposition once per linearity mask, while staging exactly the programs the
    /// uncached path stages from independently built copies of the same branches.
    #[test]
    fn test_condition_differentiation_reuses_shared_branch_transforms() {
        /// Builds a program that applies a condition over the provided branches followed by `epilogue` sines, so that
        /// programs sharing one pair of branches still have distinct derived programs.
        fn conditional_program(
            true_branch: &Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>,
            false_branch: &Arc<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>>,
            epilogue: usize,
        ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
            let branch_input = builder.add_input(ArrayType::scalar(DataType::F64));
            let regions = vec![
                builder.intern_callee(true_branch, None).unwrap(),
                builder.intern_callee(false_branch, None).unwrap(),
            ];
            let mut value = builder
                .add_instruction(
                    ArrayOperation::Condition(ConditionOperation::new()),
                    regions,
                    vec![predicate, branch_input],
                    None,
                )
                .unwrap()[0];
            for _ in 0..epilogue {
                value = builder.add_instruction(SinOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
            }
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![value], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        }

        let true_branch = Arc::new(scalar_scale_branch(2.0));
        let false_branch = Arc::new(scalar_scale_branch(3.0));
        let first = conditional_program(&true_branch, &false_branch, 1).linearize().unwrap();
        let second = conditional_program(&true_branch, &false_branch, 2).linearize().unwrap();
        assert_ne!(first.tangent().to_string(), second.tangent().to_string());

        // Independently built copies of the same branches share no retained transforms, so they exercise the uncached
        // path and pin that caching changed nothing about what is staged.
        let uncached = conditional_program(&Arc::new(scalar_scale_branch(2.0)), &Arc::new(scalar_scale_branch(3.0)), 1)
            .linearize()
            .unwrap();
        assert_eq!(first.primal().to_string(), uncached.primal().to_string());
        assert_eq!(first.tangent().to_string(), uncached.tangent().to_string());
        assert_eq!(first.residual_count(), uncached.residual_count());

        // Transposing the tangent program twice transposes its condition's branches once: the second pass is served
        // from the branch regions' retained transpositions and produces the identical pullback.
        // Build a fresh outer transpose on both calls so each reaches the nested region's cache. These static
        // tangent types need no residual dimension mappings for zeros; trailing residual inputs remain known.
        let tangent_input_indices = (0..first.tangent().input_ids().len() - first.residual_count()).collect::<Vec<_>>();
        let pullback = first.tangent().entry_region_ref().transpose(&tangent_input_indices, &[], &[]).unwrap();
        let repeated = first.tangent().entry_region_ref().transpose(&tangent_input_indices, &[], &[]).unwrap();
        assert_eq!(pullback.to_string(), repeated.to_string());
        let tangent_condition = first
            .tangent()
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), ArrayOperation::Condition(_)))
            .unwrap();
        for region in tangent_condition.regions() {
            let statistics = transposition_statistics(first.tangent().region_ref(*region).unwrap()).unwrap();
            assert_eq!((statistics.productions, statistics.hits), (1, 1));
        }
        assert_eq!(
            pullback.to_string(),
            uncached
                .tangent()
                .entry_region_ref()
                .transpose(
                    &(0..uncached.tangent().input_ids().len() - uncached.residual_count()).collect::<Vec<_>>(),
                    &[],
                    &[],
                )
                .unwrap()
                .to_string(),
        );
    }

    #[test]
    fn test_composite_condition_differentiation_refines_outputs_of_unspecialized_branches() {
        // Differentiating a condition whose refined output comes from unspecialized branches keeps the primal, tangent,
        // and cotangent types refined. The true branch computes `x * x + y`.
        let program = refined_vector_condition_program();
        let predicate = array(Array::scalar(true).unwrap());
        let x = array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let y = array(Array::vector(vec![10.0, 20.0, 30.0]).unwrap());

        let jvp = program.jvp().unwrap();
        assert!(jvp.output_types().iter().all(|r#type| r#type.identities().next().is_none()));
        assert_eq!(
            jvp.interpret(vec![
                predicate.clone(),
                x.clone(),
                y.clone(),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
                array(Array::vector(vec![0.0, 0.0, 1.0]).unwrap()),
            ]),
            Ok(vec![
                array(Array::vector(vec![11.0, 24.0, 39.0]).unwrap()),
                array(Array::vector(vec![2.0, 4.0, 7.0]).unwrap()),
            ]),
        );

        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![predicate, x, y]).unwrap();
        let residuals = primal_outputs.split_off(1);
        let pullback = linearization.pullback().unwrap();
        assert!(pullback.output_types().iter().all(|r#type| r#type.identities().next().is_none()));
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![
                array(Array::vector(vec![2.0, 4.0, 6.0]).unwrap()),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_composite_condition_jvp_preserves_dimension_outputs_without_tangent_slots() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let input = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let true_branch = scale_branch(extent_type.clone(), 2.0);
        let false_branch = scale_branch(extent_type.clone(), 3.0);
        let regions = vec![
            builder.import_region(true_branch.entry_region_ref()),
            builder.import_region(false_branch.entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 3], vec![Placeholder; 2]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_count(), 4);
        assert_eq!(jvp.output_count(), 3);
        assert_eq!(
            jvp.interpret(vec![
                array(Array::scalar(true).unwrap()),
                dimension(&extent_type, 4),
                array(Array::scalar(5.0).unwrap()),
                array(Array::scalar(7.0).unwrap()),
            ]),
            Ok(vec![
                dimension(&extent_type, 4),
                array(Array::scalar(10.0).unwrap()),
                array(Array::scalar(14.0).unwrap())
            ]),
        );

        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                array(Array::scalar(true).unwrap()),
                dimension(&extent_type, 4),
                array(Array::scalar(5.0).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        let mut pullback_inputs = vec![array(Array::scalar(1.0).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::scalar(2.0).unwrap())]),
        );
    }

    #[test]
    fn test_composite_condition_all_zero_jvp_materializes_a_dynamic_output_tangent() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let output_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let extent = builder.add_input(extent_type.clone().into());
            let output = builder
                .add_instruction(ZeroOperation::new(output_type.clone()), Vec::new(), vec![extent], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let extent = builder.add_input(extent_type.clone().into());
        let regions = vec![
            builder.import_region(branch().entry_region_ref()),
            builder.import_region(branch().entry_region_ref()),
        ];
        let output =
            builder.add_instruction(ConditionOperation::new(), regions, vec![predicate, extent], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Both inputs have zero tangent spaces, but the dynamic floating-point result does not. Its zero tangent must
        // therefore consume the selected primal result's explicit runtime extent instead of using a nullary zero.
        let jvp = program.jvp().unwrap();
        assert_eq!(jvp.input_count(), 2);
        assert_eq!(jvp.output_count(), 2);
        assert_eq!(
            jvp.interpret(vec![
                array(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap()),
                dimension(&extent_type, 3),
            ]),
            Ok(vec![array(Array::vector(vec![0.0f64; 3]).unwrap()), array(Array::vector(vec![0.0f64; 3]).unwrap()),]),
        );

        // Eager direct JVP keeps the operation's all-zero region fast path and derives the concrete output tangent
        // extent from the selected primal result at the public boundary.
        let eager = EagerContext::<TestValue, TestOperation>::new();
        let (primal, tangent) = eager
            .jvp(
                |inputs, ()| {
                    let context = inputs[0].context().clone();
                    context.bind(ConditionOperation::new(), vec![branch(), branch()], inputs.as_slice())
                },
                vec![
                    array(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap()),
                    dimension(&extent_type, 3),
                ],
                vec![
                    array(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap()),
                    array(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap()),
                ],
                (),
            )
            .unwrap();
        assert_eq!(primal, vec![array(Array::vector(vec![0.0f64; 3]).unwrap())]);
        assert_eq!(tangent, vec![array(Array::vector(vec![0.0f64; 3]).unwrap())]);

        // Split program linearization stages the same extent read on the primal side and forces the shaped zero into
        // the tangent program, rather than folding it into an affine known tangent.
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                array(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap()),
                dimension(&extent_type, 3),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(
            linearization.tangent().interpret(residuals),
            Ok(vec![array(Array::vector(vec![0.0f64; 3]).unwrap())]),
        );

        // A known symbolic predicate cannot select a branch during partial evaluation. Because the dynamic output
        // edge refers to the extent identity, the condition remains whole instead of fabricating an opposite-branch
        // placeholder with arbitrary geometry.
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let symbolic_predicate = outer.input(ArrayType::scalar(DataType::Boolean).into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(symbolic_predicate), PartialValue::Unknown(extent_type.clone().into())],
            )
            .unwrap();
        assert!(matches!(evaluation.outputs.as_slice(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayIrOperation::Condition(_),));

        // Direct transform dispatch must make the same decision before it has a staged instruction whose result type
        // it can inspect. The condition rule retains the selected branch's extent and constructs the tangent there.
        let context = TracingContext::<TestValue, TestOperation>::new();
        let predicate = context.input(ArrayType::scalar(DataType::Boolean).into());
        let extent = context.input(extent_type.clone().into());
        let predicate_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let (_, tangent) = context
            .jvp(
                |inputs, ()| {
                    let context = inputs[0].context().clone();
                    Ok(context.bind(ConditionOperation::new(), vec![branch(), branch()], inputs.as_slice())?.remove(0))
                },
                vec![predicate, extent],
                vec![predicate_tangent, extent_tangent],
                (),
            )
            .unwrap();
        assert_eq!(tangent.r#type().as_ref(), &ArrayIrType::Array(output_type.clone()));

        // Reusable linearization follows the same ordinary region rule and closes over the dynamic result geometry;
        // applying its null linear map therefore reconstructs the shaped tangent without a type-only zero.
        let predicate = context.input(ArrayType::scalar(DataType::Boolean).into());
        let extent = context.input(extent_type.clone().into());
        let (_, pushforward) = context
            .linearize(
                |inputs, ()| {
                    let context = inputs[0].context().clone();
                    Ok(context.bind(ConditionOperation::new(), vec![branch(), branch()], inputs.as_slice())?.remove(0))
                },
                vec![predicate, extent],
                (),
            )
            .unwrap();
        let predicate_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        let extent_tangent = context.input(ArrayType::scalar(DataType::Zero).into());
        assert_eq!(pushforward.apply(vec![predicate_tangent, extent_tangent]).unwrap().r#type(), tangent.r#type(),);
    }

    #[test]
    fn test_composite_condition_jvp_shapes_a_disconnected_dynamic_input_tangent_from_its_primal() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
            let left = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let right = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let sum = builder
                .add_instruction(
                    TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                    Vec::new(),
                    vec![left, right],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![extent, sum], vec![Placeholder; 3], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let left = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let right = builder.add_input(ArrayIrType::Array(array_type.clone()));
        // Severing the tangent of the conditional's last instruction input leaves the fused conditional with one live
        // and one structurally zero dynamic tangent input, which is exactly the case a type-only nullary zero cannot
        // construct.
        let severed = builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(StopGradientOperation::<ArrayType>::new())),
                Vec::new(),
                vec![right],
                None,
            )
            .unwrap()[0];
        let regions = vec![
            builder.import_region(branch().entry_region_ref()),
            builder.import_region(branch().entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, left, severed],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![outputs[1]], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();

        // The severed input's tangent reads its own primal's runtime extent before constructing the dynamic zero.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:bool[], %1:dimension<extent ∈ [1, 8)>, %2:f64[extent], %3:f64[extent], \
                    %4:f64[extent], %5:f64[extent] .
                let %6:f64[extent] = stop_gradient %3
                    %7:dimension<extent ∈ [1, 8)> = dimension_size [axis=0] %6
                    %8:f64[extent] = zero [type=f64[extent]] %7
                    %9:dimension<extent ∈ [1, 8)>, %10:f64[extent], %11:f64[extent] = condition %0 %1 %2 %6 %4 %8 [
                        true={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent], %2:f64[extent], \
                                %3:f64[extent], %4:f64[extent] .
                            let %5:f64[extent] = add %1 %2
                                %6:f64[extent] = add %3 %4
                            in (%0, %5, %6)
                        },
                        false={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent], %2:f64[extent], \
                                %3:f64[extent], %4:f64[extent] .
                            let %5:f64[extent] = add %1 %2
                                %6:f64[extent] = add %3 %4
                            in (%0, %5, %6)
                        },
                    ]
                in (%10, %11)
            "}
            .trim_end(),
        );
        assert_eq!(
            jvp.interpret(vec![
                array(Array::from_elements::<bool>(ArrayType::scalar(DataType::Boolean), &[true]).unwrap()),
                dimension(&extent_type, 3),
                array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                array(Array::vector(vec![10.0, 20.0, 30.0]).unwrap()),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
                array(Array::vector(vec![5.0, 5.0, 5.0]).unwrap()),
            ]),
            Ok(vec![
                array(Array::vector(vec![11.0, 22.0, 33.0]).unwrap()),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())
            ]),
        );
    }

    #[test]
    fn test_composite_condition_pullback_shapes_a_dead_dynamic_output_cotangent_from_a_live_peer() {
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type =
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_type.variable().clone())]));
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
            let input = builder.add_input(ArrayIrType::Array(array_type.clone()));
            let doubled = builder
                .add_instruction(
                    TestOperation::Array(ArrayOperation::from(AddOperation::new())),
                    Vec::new(),
                    vec![input, input],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![extent, doubled, input],
                    vec![Placeholder; 2],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let input = builder.add_input(ArrayIrType::Array(array_type.clone()));
        let regions = vec![
            builder.import_region(branch().entry_region_ref()),
            builder.import_region(branch().entry_region_ref()),
        ];
        let outputs = builder
            .add_instruction(
                TestOperation::Condition(ConditionOperation::new()),
                regions,
                vec![predicate, extent, input],
                None,
            )
            .unwrap()
            .to_vec();
        // Keeping only the doubled output leaves the third branch output dead, so its dynamic cotangent reaches the
        // transposed condition as a structural zero that no type-only constructor can build. The transpose boundary
        // reads the runtime extent it names off the live peer cotangent and stages the mixed dynamic zero.
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![outputs[1]], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let linearization = program.linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:bool[] .
                let %2:dimension<extent \u{2208} [1, 8)> = dimension_size [axis=0] %0
                    %3:f64[extent] = zero [type=f64[extent]] %2
                    %4:f64[extent] = condition %1 %0 %3 [
                        true={
                            lambda %0:f64[extent], %1:f64[extent] .
                            let %2:f64[extent] = add %1 %0
                                %3:f64[extent] = add %2 %0
                            in (%3)
                        },
                        false={
                            lambda %0:f64[extent], %1:f64[extent] .
                            let %2:f64[extent] = add %1 %0
                                %3:f64[extent] = add %2 %0
                            in (%3)
                        },
                    ]
                in (%4)"}
            .trim_end(),
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                array(Array::scalar(true).unwrap()),
                dimension(&extent_type, 3),
                array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![array(Array::vector(vec![2.0, 2.0, 2.0]).unwrap())]));
    }

    #[test]
    fn test_condition_differentiation_after_batching_blocks_inactive_non_finite_derivatives() {
        // The inactive square root produces NaN at -1 and an infinite derivative at zero. Input gradient barriers
        // must discard both contributions, while preserving the selected identity derivative for those items.
        let (batched, _) = square_root_or_identity_condition_program()
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        let linearization = batched.linearize().unwrap();
        let mut primal_outputs =
            linearization.primal().interpret(vec![Array::vector(vec![4.0f64, -1.0, 0.0]).unwrap()]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![Array::vector(vec![2.0f64, -1.0, 0.0]).unwrap()]);
        let mut pullback_inputs = vec![Array::vector(vec![1.0f64, 1.0, 1.0]).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![Array::vector(vec![0.25f64, 1.0, 1.0]).unwrap()]),
        );
    }

    #[test]
    fn test_composite_condition_differentiation_after_batching_blocks_inactive_non_finite_derivatives() {
        let program =
            square_root_or_identity_condition_program().into_unprojected::<TestValue, TestOperation>().unwrap();
        let extent = DimensionValue::constant(3).unwrap();
        let (batched, _) = program
            .batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        let linearization = batched.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                TestValue::Dimension(extent.clone()),
                array(Array::vector(vec![4.0f64, -1.0, 0.0]).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        assert_eq!(
            primal_outputs,
            vec![TestValue::Dimension(extent), array(Array::vector(vec![2.0f64, -1.0, 0.0]).unwrap())],
        );
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0f64, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![0.25f64, 1.0, 1.0]).unwrap())]),
        );
    }

    #[test]
    fn test_condition_transposition_accepts_interleaved_known_inputs() {
        // A known scale precedes the linear input.
        let transposed = interleaved_product_condition_program(&[(0, 1)]).transpose_with_respect_to(&[2], &[]).unwrap();
        assert_eq!(
            transposed.interpret(vec![
                Array::scalar(3.0f64).unwrap(),
                Array::scalar(true).unwrap(),
                Array::scalar(5.0f64).unwrap(),
            ]),
            Ok(vec![Array::scalar(15.0f64).unwrap()]),
        );
        assert_eq!(
            transposed.interpret(vec![
                Array::scalar(3.0f64).unwrap(),
                Array::scalar(false).unwrap(),
                Array::scalar(5.0f64).unwrap(),
            ]),
            Ok(vec![Array::scalar(30.0f64).unwrap()]),
        );

        // Known scales and linear inputs alternate. The cotangents are requested in reverse input order to check that
        // the branch's source order is reassembled correctly.
        let transposed = interleaved_product_condition_program(&[(1, 0), (3, 2)])
            .transpose_with_respect_to(&[3, 1], &[])
            .unwrap();
        assert_eq!(
            transposed.interpret(vec![
                Array::scalar(3.0f64).unwrap(),
                Array::scalar(true).unwrap(),
                Array::scalar(2.0f64).unwrap(),
                Array::scalar(5.0f64).unwrap(),
            ]),
            Ok(vec![Array::scalar(15.0f64).unwrap(), Array::scalar(6.0f64).unwrap()]),
        );
        assert_eq!(
            transposed.interpret(vec![
                Array::scalar(3.0f64).unwrap(),
                Array::scalar(false).unwrap(),
                Array::scalar(2.0f64).unwrap(),
                Array::scalar(5.0f64).unwrap(),
            ]),
            Ok(vec![Array::scalar(30.0f64).unwrap(), Array::scalar(12.0f64).unwrap()]),
        );
    }

    #[test]
    fn test_condition_transposition_reference_input_destinations() {
        // Both branches receive the cotangent reference of the reference input: the taken branch's transpose acts on it
        // in place (`add_update` reads the destination into `x̄`, `write` swaps a zero into it).
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let true_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(scalar_type.clone());
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, value],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 3], Vec::<Placeholder>::new())
            .unwrap();

        // The predicate is a known parameter of the linear map, so the transposed program consumes `[r̄, p]`.
        let transposed = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))
            ],
        );
        assert_eq!(
            transposed.output_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))), scalar_type]
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::scalar(5.0f32).unwrap()],
                vec![Array::scalar(true).unwrap()]
            ),
            vec![Array::scalar(5.0f32).unwrap(), Array::scalar(5.0f32).unwrap()],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::scalar(5.0f32).unwrap()],
                vec![Array::scalar(false).unwrap()]
            ),
            vec![Array::scalar(5.0f32).unwrap(), Array::scalar(0.0f32).unwrap()],
        );
    }

    #[test]
    fn test_condition_transposition_write_only_reference_input_destinations() {
        // Both branches only store into the reference input (`write` when taken, `add_update` otherwise) and forward
        // `x` as the live output. Under an `Ignore` destination for the reference no later instruction accumulated into
        // its root and neither branch reads it, so its state cotangent is provably zero: the branches are transposed
        // with an `Ignore` destination as well and the pullback stages no cotangent reference at all instead of
        // allocating, zeroing, and freezing a dead accumulator around the transposed condition.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let predicate_type = ArrayIrType::Array(ArrayType::scalar(DataType::Boolean));
        let true_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let transposed = program
            .transpose_with_respect_to(&[1, 2], &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return])
            .unwrap();
        assert_eq!(transposed.input_types(), vec![scalar_type.clone(), predicate_type.clone()]);
        assert_eq!(transposed.output_types(), vec![scalar_type.clone()]);
        let names = transposed
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["condition"]);
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(3.0f32).unwrap()],
                vec![],
                vec![Array::scalar(true).unwrap()],
            ),
            vec![Array::scalar(3.0f32).unwrap()],
        );

        // Under a `Reference` destination the reference input's state cotangent is live, so both branches receive the
        // cotangent reference and their stores transpose against it.
        let transposed = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![
                scalar_type.clone(),
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                predicate_type
            ],
        );
        assert_eq!(
            transposed.output_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))), scalar_type]
        );
        let names = transposed
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert!(names.contains(&"reference_swap"), "{names:?}");
        assert!(names.contains(&"reference_read"), "{names:?}");
        assert!(!names.contains(&"reference_new"), "{names:?}");
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(3.0f32).unwrap()],
                vec![Array::scalar(5.0f32).unwrap()],
                vec![Array::scalar(true).unwrap()],
            ),
            vec![Array::scalar(8.0f32).unwrap(), Array::scalar(0.0f32).unwrap()],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(3.0f32).unwrap()],
                vec![Array::scalar(5.0f32).unwrap()],
                vec![Array::scalar(false).unwrap()],
            ),
            vec![Array::scalar(8.0f32).unwrap(), Array::scalar(5.0f32).unwrap()],
        );
    }

    #[test]
    fn test_condition_transposition_reference_access_with_enclosing_binding() {
        // Each branch accesses the reference root through a dynamic index that the enclosing region computes. The
        // transposed branches apply the same transforms to the root's cotangent reference, so the index reaches them as
        // an ordinary known input recomputed in the enclosing region: `add_update(r[i], x)` transposes into
        // `x̄ = read(r̄[i])`, and `write(r[i], x)` additionally clears `r̄[i]`.
        let vector_reference_type = ArrayIrType::from(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let index_type = ArrayIrType::Array(ArrayType::scalar(DataType::I32));
        let element_transforms =
            vec![ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic }];
        let true_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(vector_reference_type.clone());
            let value = builder.add_input(scalar_type.clone());
            let index = builder.add_input(index_type.clone());
            builder
                .add_instruction(
                    ReferenceAddUpdateOperation::new().with_transforms(element_transforms.clone()),
                    Vec::new(),
                    vec![reference, value, index],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let reference = builder.add_input(vector_reference_type.clone());
            let value = builder.add_input(scalar_type.clone());
            let index = builder.add_input(index_type.clone());
            builder
                .add_instruction(
                    ReferenceWriteOperation::new().with_transforms(element_transforms),
                    Vec::new(),
                    vec![reference, value, index],
                    None,
                )
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let reference = builder.add_input(vector_reference_type.clone());
        let value = builder.add_input(scalar_type.clone());
        let offset = builder.add_input(index_type);
        let one = builder.add_constant(TestValue::Array(Array::scalar(1i32).unwrap()));
        let index = builder.add_instruction(AddOperation::new(), Vec::new(), vec![offset, one], None).unwrap()[0];
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, value, index],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 4], Vec::<Placeholder>::new())
            .unwrap();

        let transposed = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(transposed.output_types(), vec![vector_reference_type, scalar_type]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:ref<f32[3]>, %1:bool[], %2:i32[] .
                let %3:i32[] = const 1
                    %4:i32[] = add %2 %3
                    %5:ref<f32[3]>, %6:f32[] = condition %1 %0 %4 [
                        true={
                            lambda %0:ref<f32[3]>, %1:i32[] .
                            let %2:f32[] = reference_read [transforms=[index(axis=0, index=dynamic)]] %0 %1
                            in (%0, %2)
                        },
                        false={
                            lambda %0:ref<f32[3]>, %1:i32[] .
                            let %2:f32[] = zero [type=f32[]]
                                %3:f32[] = reference_swap [transforms=[index(axis=0, index=dynamic)]] %0 %2 %1
                            in (%0, %3)
                        },
                    ]
                in (%0, %6)"},
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::vector(vec![1f32, 2., 3.]).unwrap()],
                vec![Array::scalar(true).unwrap(), Array::scalar(0i32).unwrap()],
            ),
            vec![Array::scalar(2f32).unwrap(), Array::vector(vec![1f32, 2., 3.]).unwrap()],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::vector(vec![1f32, 2., 3.]).unwrap()],
                vec![Array::scalar(false).unwrap(), Array::scalar(1i32).unwrap()],
            ),
            vec![Array::scalar(3f32).unwrap(), Array::vector(vec![1f32, 2., 0.]).unwrap()],
        );
    }

    #[test]
    fn test_condition_transposition_preserves_shared_gradient_buffers() {
        type TestValue = ArrayIrValue<Array>;
        type TestOperation = ArrayIrOperation<Array>;

        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let left = branch.add_input(scalar_type.clone());
        let right = branch.add_input(scalar_type.clone());
        let sum = branch.add_instruction(AddOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let branch = branch
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_program(branch);
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![branch, branch],
                vec![predicate, input, input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[1], &[CotangentDestinationKind::Reference]).unwrap();
        let condition = transposed
            .instructions()
            .iter()
            .find(|instruction| instruction.operation().name() == CONDITION_OPERATION_NAME)
            .unwrap();
        assert_eq!(condition.inputs().len(), 4);
        assert_eq!(condition.inputs()[2], condition.inputs()[3]);

        // Both nested instruction input positions share one caller-owned buffer. Test through a local allocation so
        // reference discharge must preserve that internal alias rather than treating the two branch inputs as
        // independent state.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type.clone());
        let seed = builder.add_input(scalar_type);
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        assert!(builder.splice_program(&transposed, &[seed, reference, predicate]).unwrap().is_empty());
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let staged = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let inputs = vec![
            Array::scalar(5.0f32).unwrap().into(),
            Array::scalar(3.0f32).unwrap().into(),
            Array::scalar(true).unwrap().into(),
        ];
        let expected = vec![Array::scalar(11.0f32).unwrap().into()];
        assert_eq!(staged.interpret(inputs.clone()).unwrap(), expected);
        let discharged = staged.discharge_references(0).unwrap().into_program_without_external_references().unwrap();
        assert_eq!(discharged.interpret(inputs).unwrap(), expected);
    }
}
