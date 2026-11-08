use std::cell::RefCell;
use std::collections::HashSet;

use crate::contexts::Context;
use crate::operations::GuardOperation;
use crate::partial::contexts::{PartialEvaluationContext, ReferencePlacement};
use crate::partial::evaluations::PartialEvaluation;
use crate::partial::partitions::{EffectOrdering, PartitionedProgram};
use crate::partial::values::{PartialEvaluationValue, PartialValue};
use crate::programs::{
    AttachedRegions, EmptyRegionDriver, InstructionId, Operation, OperationProvider, ProgramError, ReferenceAnalysis,
    ReferenceRoot, RegionDriver, RegionRef, Value, ValueId,
};
use crate::tracing::TracingContext;

/// [`RegionDriver`] that provides [`Instruction`](crate::Instruction)-scoped access to [`Region`](crate::Region)s
/// attached to a partially evaluated [`Operation`] application. A [`PartialEvaluationDriver`] borrows the current
/// instruction's regions and supports recursive partial evaluation. Operation rules receive it separately from their
/// durable [`PartialEvaluationContext`], so the borrowed region access cannot escape through a
/// [`PartialTracer`](crate::PartialTracer). [`RegionDriver`] provides structural region access,
/// while this trait adds partial-evaluation-specific recursion.
pub trait PartialEvaluationDriver<C: Context>: RegionDriver<C::Constant, C::Operation> {
    /// Partially evaluates the [`Region`](crate::Region) at `index` over the provided partial-evaluation values
    /// by re-entering the active partial-evaluation transform.
    fn partially_evaluate_region(
        &self,
        context: &PartialEvaluationContext<C>,
        index: usize,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>;

    /// Partially evaluates `region` against the provided input knowledge through the active known-side context and
    /// returns the region's residual split.
    fn partially_evaluate_program(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        knowledge: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError>;

    /// Builds separate known and residual programs for `region`, treating input `i` as known when `input_known[i]`
    /// is `true`. Both programs are constructed in a fresh staging context, so even known operations are recorded
    /// rather than executed through the active context. Partitioning performs an analysis where both programs are
    /// seeded at the declared input types of `region`, without checking the relationships that its entry checks
    /// between the actual inputs. A rule binds a partition's known program over the evaluation's inputs only after
    /// guarding them, which [`PartialEvaluationContext::inline_partitioned_program`] does (refer to
    /// [`PartialEvaluationContext::guard_inputs`] for rules that bind operations they rebuild from partitions).
    ///
    /// The new evaluation uses `context`'s execution configuration but tracks effect ordering and pending errors
    /// separately. For example, a loop rule can try a partition, discover that a loop-carried value must be unknown,
    /// and try again without executing reference writes or changing which later operations the active evaluation
    /// must defer. Effects already deferred by the active evaluation likewise do not constrain this fresh partition.
    fn partition_program(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_known: &[bool],
    ) -> Result<PartitionedProgram<C::Constant, C::Operation>, ProgramError>;

    /// Splits the all-known region-carrying instruction at `instruction_index` of `region` on residual demand for the
    /// outputs that `demanded_outputs` marks, through [`PartiallyEvaluatableOperation::split_on_residual_demand`] of
    /// its operation and with the residual policy of `context`, and returns the split as a partition of that
    /// instruction. The partition receives every input of the instruction as a known input. Its known program returns
    /// the outputs of the instruction followed by its residual edges, and its residual program returns the demanded
    /// outputs, in output order. Operations use this when they plan residual work over a known program of their own
    /// (e.g., the carry recomputation of a `scan` under a residual policy), so that the instructions of that program
    /// are split like those of a partition whose residuals are placed (refer to
    /// [`PartitionedProgram::with_top_level_residual_policy`] for more information). This is a driver function rather
    /// than a function of [`PartialEvaluationContext`] because such rules run in
    /// [`partially_evaluate`](PartiallyEvaluatableOperation::partially_evaluate), whose bounds do not include the
    /// partial evaluation of the operation family in a [`TracingContext`]. The driver carries that capability from
    /// where it was constructed.
    ///
    /// Returns [`None`] when `context` has no residual policy, when this driver does not support such splits, or when
    /// the rule keeps the instruction whole, in which case the planner saves its demanded outputs instead.
    ///
    /// # Errors
    ///
    /// Returns the errors of the partial-evaluation rule of the instruction and of partitioning its regions.
    fn split_instruction_on_residual_demand(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        instruction_index: usize,
        demanded_outputs: &[bool],
    ) -> Result<Option<PartitionedProgram<C::Constant, C::Operation>>, ProgramError> {
        let _ = (context, region, instruction_index, demanded_outputs);
        Ok(None)
    }

    /// Returns whether recursive partitioning analyzes one known invocation followed by repeated residual invocations.
    /// In this mode, allocation and reference-ordering analysis determines effect placement rather than inheriting the
    /// active context's effect-folding restriction. Rules that retain structural partitions must include this mode in
    /// their cache key when their regions have effects.
    ///
    /// The default is ordinary specialization, which preserves the active context's effect-folding policy. A driver
    /// that overrides this query must use the reported mode in [`partition_program`](Self::partition_program).
    #[inline]
    fn partitions_for_repeated_residual_calls(&self) -> bool {
        false
    }
}

impl<C: Context> PartialEvaluationDriver<C> for EmptyRegionDriver {
    #[inline]
    fn partially_evaluate_region(
        &self,
        _context: &PartialEvaluationContext<C>,
        _index: usize,
        _inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot partially evaluate a region".to_string()))
    }

    #[inline]
    fn partially_evaluate_program(
        &self,
        _context: &PartialEvaluationContext<C>,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _knowledge: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot partially evaluate a program".to_string()))
    }

    #[inline]
    fn partition_program(
        &self,
        _context: &PartialEvaluationContext<C>,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _input_known: &[bool],
    ) -> Result<PartitionedProgram<C::Constant, C::Operation>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot partition a program".to_string()))
    }
}

/// [`PartialEvaluationDriver`] scoped to one [`Operation`] application. It borrows the application's complete
/// [`AttachedRegions`], preserving the operation-defined ordering of owned [`Region`](crate::Region)s, borrowed
/// regions, and shared callees without collecting [`Program`](crate::Program)s or region views. Recursive requests
/// re-enter partial evaluation for a selected region or partition it into known and residual programs.
pub(super) struct RecursivePartialEvaluationDriver<'r, V: Value, O: Operation<Type = V::Type>> {
    /// Application-scoped [`AttachedRegions`], in [`Operation`]-defined order.
    pub(super) regions: &'r AttachedRegions<'r, V, O>,

    /// Specifies whether partitioning must account for a known computation running once and its results being reused
    /// across separate residual calls. The same requirement applies when recursively partitioning nested computations.
    ///
    /// For example, linearizing a fused custom Jacobian-Vector Product (JVP) function at a fixed primal input computes
    /// the primal result and reusable coefficients once, then produces a pushforward callable with different tangent
    /// inputs. With this flag set, an accumulator used only by the pushforward must be allocated afresh on each call,
    /// even if its initial value is a known zero. Nested partitions determine which allocations belong to each call
    /// and use reference analysis to preserve ordering between accesses to the same allocation.
    ///
    /// Without this flag, nested computations use the standard partial evaluation rules. For example, specializing a
    /// function with a fixed scalar argument can fold pure arithmetic involving that argument while leaving work that
    /// depends on other arguments in the residual program. Ordered effects retain their relative order, and reference
    /// placement follows the context's configuration rather than the repeated-call allocation analysis. The resulting
    /// specialized program can still be called more than once (this flag selects how partitioning assigns state and
    /// preserves effect ordering, and does not impose a limit on the number of residual calls).
    pub(super) repeated_residual: bool,
}

impl<V: Value, O: Operation<Type = V::Type>> RecursivePartialEvaluationDriver<'_, V, O> {
    /// Evaluates one source instruction while preserving the ordering and allocation requirements of its replay.
    /// Without reference analysis, this dispatches through the active context unless the caller explicitly requires
    /// residual execution. With analysis, independent allocations may be evaluated separately, but an instruction
    /// cannot move ahead of earlier deferred work on the same allocation or work requiring global ordering.
    ///
    /// Nested partitioning replaces arguments with fresh symbolic inputs. If two source arguments name the same
    /// allocation, their alias relationship would be lost in that replacement. Such an ordered application is kept
    /// whole (i.e., it can fold with known inputs, or remain residual, but cannot be split recursively).
    ///
    /// # Parameters
    ///
    ///   - `context`: Active evaluation whose residual builder receives emitted work. Only replay with reference
    ///     analysis creates a clone with instruction-local ordering state; ordinary replay uses this context directly.
    ///   - `region`: Source region containing the instruction and its reference identities.
    ///   - `instruction_index`: Instruction index within the source region.
    ///   - `operation`: Operation of the instruction with its type identities renamed for this replay (refer to
    ///     [`ReplayTypeIdentities::instruction_renaming`](crate::ReplayTypeIdentities::instruction_renaming)).
    ///     The source instruction at `instruction_index` still drives the effect and reference analysis.
    ///   - `inputs`: Partially evaluated instruction inputs in source input order.
    ///   - `deferred_instructions`: Source instructions explicitly required to remain residual, including allocations
    ///     that must be created afresh for each residual call.
    ///   - `deferred_effect_ordering`: Accumulated ordering constraints of earlier deferred source work. Updated only
    ///     when reference analysis is supplied; nested emission still enforces ordering within this instruction.
    ///   - `reference_analysis`: Canonical source-reference analysis for repeated residual calls. Omitting it retains
    ///     the active context's ordering rules without performing source-reference analysis.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn partially_evaluate_instruction<C: Context<Constant = V, Operation = O>>(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        instruction_index: usize,
        operation: &C::Operation,
        inputs: &[PartialEvaluationValue<C::Value>],
        deferred_instructions: &HashSet<InstructionId>,
        deferred_effect_ordering: &RefCell<EffectOrdering<ReferenceRoot>>,
        reference_analysis: Option<&ReferenceAnalysis>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        O: OperationProvider<V::Type, GuardOperation<V::Type>, Operation = O>,
        C::Operation:
            PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    {
        let instruction = &region.instructions()[instruction_index];
        let instruction_id = InstructionId::new(region.id(), instruction_index);
        let mut shared_reference_boundary = false;
        let effect_ordering = if let Some(analysis) = reference_analysis {
            let effects = region.instruction_effects(instruction_index)?;
            if effects.classes().is_ordered() {
                // Resolve input roots once for both ordering and alias detection. Allocation identity includes views,
                // even when the views select different elements of the allocation.
                let mut roots = HashSet::new();
                for &atom in instruction.inputs() {
                    if let Some(root) = analysis.root_of(ValueId::new(region.id(), atom)) {
                        shared_reference_boundary |= !roots.insert(root);
                    }
                }
                roots.extend(
                    instruction.outputs().iter().filter_map(|&atom| analysis.root_of(ValueId::new(region.id(), atom))),
                );
                roots.extend(analysis.transitive_access(instruction_id).into_iter().flat_map(|access| access.roots()));

                // Captured handles cannot prove independence from symbolic incoming references. Empty reference
                // sets conservatively give ordered work global ordering, rather than inventing independent identities.
                if roots.iter().any(|root| matches!(root, ReferenceRoot::Constant { .. })) {
                    roots.clear();
                }

                Some(effects.effect_ordering(roots))
            } else {
                Some(EffectOrdering::default())
            }
        } else {
            None
        };

        let instruction_context = effect_ordering.as_ref().map(|ordering| {
            // An entry guard precedes this source-reference walk, so its global barrier survives local ordering.
            context.clone().with_defer_ordered_effects(
                context.defer_ordered_effects() || deferred_effect_ordering.borrow().conflicts(ordering),
            )
        });

        let context = instruction_context.as_ref().unwrap_or(context);
        let must_defer = deferred_instructions.contains(&instruction_id);
        let outputs = if must_defer || shared_reference_boundary {
            let programs = self.region_programs();
            if must_defer {
                context.residualize(operation.clone(), programs, inputs)
            } else {
                context.fold_or_residualize(operation.clone(), programs, inputs)
            }
        } else {
            operation.partially_evaluate(context, self, inputs)
        }?;

        if let Some(ordering) = effect_ordering
            && context.defer_ordered_effects()
        {
            deferred_effect_ordering.borrow_mut().extend(&ordering);
        }

        Ok(outputs)
    }
}

impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for RecursivePartialEvaluationDriver<'_, V, O> {
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.regions.regions()
    }

    #[inline]
    fn region_count(&self) -> usize {
        self.regions.region_count()
    }

    #[inline]
    fn region(&self, index: usize) -> Result<RegionRef<'_, V, O>, ProgramError> {
        self.regions.region(index)
    }
}

impl<C: Context> PartialEvaluationDriver<C> for RecursivePartialEvaluationDriver<'_, C::Constant, C::Operation>
where
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
{
    fn partially_evaluate_region(
        &self,
        context: &PartialEvaluationContext<C>,
        index: usize,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let region = self.region(index)?;

        // Both replays below take the place of the region's entry, so they follow the shared entry contract.
        let inputs = context.guard_inputs(&region.input_types(), inputs, false)?;

        // Inlining during ordinary specialization shares the caller's residual builder and accumulated effect
        // ordering. It needs no separate allocation discovery or per-reference ordering analysis.
        if !self.repeated_residual {
            return context.inline_region(region, inputs, &HashSet::new(), None, None);
        }

        // Analyze a staged copy using only input knownness to discover allocations that must be fresh on each
        // residual invocation. Discard the staged programs as the replay below must use the caller's actual values.
        let knowledge = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<_>>();
        let (_, deferred_instructions) = region.partition_with_configuration(&knowledge, true, true, None, None)?;

        // Replay into the active context, explicitly deferring the discovered allocations. Source reference roots
        // let replay distinguish independent accesses while keeping accesses to the same state in order.
        context.inline_region(
            region,
            inputs,
            &deferred_instructions,
            None,
            Some(region.reference_analysis_with_configuration(None, true, &[])?.as_ref()),
        )
    }

    fn partially_evaluate_program(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        knowledge: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError> {
        // Unlike inlining a region, constructing a separate residual program needs fresh builder and ordering
        // state. Ordinary specialization still inherits the caller's permission to fold effectful work.
        if !self.repeated_residual {
            return region.partially_evaluate_in_context(context.parent(), knowledge, context.allow_effect_folding());
        }

        // Give the nested evaluation its own residual program while folding through the same known-side parent.
        // Stage placement keeps live reference operations residual when that parent is eager.
        let nested = PartialEvaluationContext::new(context.parent().clone())
            .with_reference_placement(ReferencePlacement::Stage)
            .with_residual_placement(context.residual_placement());

        // Repeated residual calls use their own allocation placement and reference-ordering analysis. Keep the
        // fresh context's effect folding enabled so that this analysis determines which effects must be deferred.
        let known = knowledge.iter().map(PartialValue::is_known).collect::<Vec<_>>();
        let (_, deferred_instructions) = region.partition_with_configuration(&known, true, true, None, None)?;

        // Retain actual known values and create residual inputs for unknowns. Their indices refer to the original
        // region inputs so the returned evaluation can reconstruct its residual arguments in the correct order.
        let inputs = nested.prepare_root_inputs(region, knowledge)?;

        // Apply the allocation decisions to replay of the original region, using its reference analysis to preserve
        // dependencies between accesses. The discovery pass already recorded which allocations need deferring, so
        // this replay does not need to collect another list of residual source instructions.
        let outputs = nested.inline_region(
            region,
            inputs,
            &deferred_instructions,
            None,
            Some(region.reference_analysis_with_configuration(None, true, &[])?.as_ref()),
        )?;

        // Finalize the residual program and report how its inputs and outputs relate to the original computation.
        nested.into_evaluation(outputs)
    }

    fn partition_program(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_known: &[bool],
    ) -> Result<PartitionedProgram<C::Constant, C::Operation>, ProgramError> {
        // Both paths stage fresh known and residual programs without executing effects or changing the caller's
        // accumulated ordering state. Only the programs are needed here; allocation IDs are for replaying source
        // regions, whereas these returned programs already incorporate the allocation decisions. Also, a residual
        // policy set on the caller's context places the residuals of the partition and of the partitions nested
        // within it.
        let residual_placement = context.residual_placement();
        if self.repeated_residual {
            // Let repeated-call allocation and reference-ordering analysis decide which effects can remain known,
            // rather than inheriting a restriction intended for the caller's current residual program.
            region
                .partition_with_configuration(input_known, true, true, None, residual_placement)
                .map(|(partition, _)| partition)
        } else {
            // Ordinary specialization preserves the caller's effect-folding policy and uses a single partition pass.
            region
                .partition_with_configuration(
                    input_known,
                    context.allow_effect_folding(),
                    false,
                    None,
                    residual_placement,
                )
                .map(|(partition, _)| partition)
        }
    }

    fn split_instruction_on_residual_demand(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        instruction_index: usize,
        demanded_outputs: &[bool],
    ) -> Result<Option<PartitionedProgram<C::Constant, C::Operation>>, ProgramError> {
        match context.residual_placement() {
            Some(residual_placement) => {
                region.split_on_residual_demand(instruction_index, demanded_outputs, residual_placement)
            }
            None => Ok(None),
        }
    }

    #[inline]
    fn partitions_for_repeated_residual_calls(&self) -> bool {
        self.repeated_residual
    }
}

/// [`Operation`] that supports partial evaluation via
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate). This trait lets an individual operation
/// decide how partial evaluation treats it. It can be implemented with an empty implementation block, deferring
/// to [`PartialEvaluationContext::fold_or_residualize`], which is what most operations do, or its behavior can be
/// customized by overriding the [`PartiallyEvaluatableOperation::partially_evaluate`] function.
///
/// Inlining and region specialization preserve boundary checks by staging [`GuardOperation`]s through
/// the guard provider that every [`Context`]'s operation family carries, so neither this trait nor
/// [`partially_evaluate`](Self::partially_evaluate) states a separate guard-provider bound.
///
/// # Type Parameters
///
///   - `C`: Known-side [`Context`] that partial evaluation folds known work through. Its
///     [`Operation`](crate::Domain::Operation) is the operation family of the residual [`Program`](crate::Program)
///     and of any inlined nested programs (e.g., the enum this operation may belong to). Its
///     [`Constant`](crate::Domain::Constant) is the staged constant space those programs store. Finally, its
///     [`Value`](crate::Domain::Value) is the space known values flow in (i.e., concrete values under eager contexts
///     and [`Tracer`](crate::Tracer)s into the outer program under [`StagingContext`](crate::StagingContext)s).
///
/// # Deriving Partially Evaluatable Operation Enums
///
/// The `#[derive(Operation)]` macro generates a [`PartiallyEvaluatableOperation`] implementation for operation enums.
/// Native variants forward to their payload's own rule, and the generated per-payload predicates transport that rule's
/// value and context requirements to the enum's use site. Declared member variants instead use the enclosing enum's
/// canonical fold-or-residualize path because member-side partial values cannot represent values belonging to other
/// members of the heterogeneous universe. This preserves correct folding and residualization without a second projected
/// partial-value protocol. Refer to the documentation of [`Operation`] for the full derive contract. Partial evaluation
/// is always generated and does not require a `dispatch(...)` selection.
pub trait PartiallyEvaluatableOperation<C: Context>: Clone + Into<C::Operation> {
    /// Partially evaluates this [`PartiallyEvaluatableOperation`] for the provided [`PartialEvaluationValue`]s. Unless
    /// overridden, this function will default to calling [`PartialEvaluationContext::fold_or_residualize`] which uses
    /// the following semantics:
    ///
    ///   - When *all* of the operation's inputs are [`Known`](PartialValue::Known), it **folds** the operation by
    ///     [`bind`](Context::bind)ing it in the known-side context, interpreting it immediately under an eager context,
    ///     and staging it into the outer program under a [`StagingContext`](crate::StagingContext), so that the
    ///     operation's outputs become known values and the operation contributes nothing to the residual
    ///     [`Program`](crate::Program). Pure regionless operations without references remain residual
    ///     when eager execution returns [`ProgramError::UnsupportedOperation`].
    ///   - Otherwise, it **residualizes** the operation unchanged, meaning that it emits the operation into the
    ///     residual program over its inputs' residual program [`Atom`](crate::Atom)s, materializing each known input as
    ///     a residual input for a known variable or as an inlined residual program constant for a literal, so that the
    ///     operation runs at residual program execution time.
    ///   - An operation with an ordered effect (i.e., [`EffectClasses::is_ordered`](crate::EffectClasses::is_ordered)
    ///     over the operation and its executable computation regions) also follows the ordering rules on
    ///     [`PartialEvaluationContext`] where once an ordered operation has been staged, every later ordered operation
    ///     is staged too, even when all of its inputs are known, and reference operations under an eager known side
    ///     follow the context's [`ReferencePlacement`].
    ///   - An operation that carries deferred work over the same scope is always residualized, even when all of its
    ///     inputs are known. Refer to [`PartialEvaluationContext::fold_or_residualize`] for more information.
    ///
    /// There are situations where overriding this function can result in improved performance and better partitioning
    /// of a computation into known and unknown parts. For example, a `condition` instruction whose predicate is
    /// [`Known`](PartialValue::Known) and Boolean-concretizable may ask the context to inline the selected branch and
    /// return that branch's output trace values, so that the condition disappears from the residual program and only
    /// the taken branch's work survives. Rules that inspect known *payloads* must gate that inspection on a
    /// [`Constant`](crate::ValueResolution::Constant) [`Context::resolve`] resolution because a known value under a
    /// staging known-side context may be a [`Tracer`](crate::Tracer) into the outer program rather than a program
    /// constant, and partial evaluation should fall back to a conservative rewrite otherwise. Resolving to a constant
    /// alone does not guarantee that the payload is host-inspectable; rules that inspect it require the corresponding
    /// capability separately.
    ///
    /// # Parameters
    ///
    ///   - `context`: Durable [`PartialEvaluationContext`] that owns residual emission, inlining, and materialization.
    ///   - `driver`: [`PartialEvaluationDriver`] that provides [`Instruction`](crate::Instruction)-scoped access to the
    ///     application [`Region`](crate::Region)s.
    ///   - `inputs`: [`PartialEvaluationValue`] for each of this [`Operation`]'s inputs, in input order.
    #[inline]
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        context.fold_or_residualize(self.clone(), driver.region_programs(), inputs)
    }

    /// Splits this [`PartiallyEvaluatableOperation`]'s application, whose inputs are all known, on residual demand for
    /// the outputs that `demanded_outputs` marks, or returns [`None`] to keep the application whole, which is what the
    /// default implementation does. The residual planner asks for such a split when the residual program of a partition
    /// demands an output of an all-known region-carrying application whose regions hold work that the residual policy
    /// saves (refer to [`PartitionedProgram::with_top_level_residual_policy`] for more information). Replaying the
    /// application whole would recompute that work, and saving its demanded outputs would save more than the policy
    /// asks for, so a split saves only the interior values that the policy saves, as JAX's `partial_eval_jaxpr_custom`
    /// rules for `cond`, `scan`, and `pjit` do.
    ///
    /// A rule that supports such a split binds a known part in the known-side context and emits a replay part into the
    /// residual program (refer to [`ResidualDemandSplit`]). [`PartialEvaluationContext::partition_residual_demand`]
    /// provides the partitions of the regions that such a split needs. A rule must keep an application whole when it
    /// has effects, accesses references, carries deferred work, or would expose nominal type identities.
    ///
    /// # Parameters
    ///
    ///   - `context`: Durable [`PartialEvaluationContext`], whose residual policy decides which interior values the
    ///     known part saves.
    ///   - `driver`: [`PartialEvaluationDriver`] that provides [`Instruction`](crate::Instruction)-scoped access
    ///     to the application [`Region`](crate::Region)s.
    ///   - `inputs`: [`PartialEvaluationValue`] for each of this [`Operation`]'s inputs, in input order, all of which
    ///     are known.
    ///   - `demanded_outputs`: Outputs that the residual program demands, marked in output order.
    #[inline]
    fn split_on_residual_demand<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
        demanded_outputs: &[bool],
    ) -> Result<Option<ResidualDemandSplit<C::Value>>, ProgramError>
    where
        C::Operation: PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    {
        let _ = (context, driver, inputs, demanded_outputs);
        Ok(None)
    }
}

/// Split of an [`Operation`] application whose inputs are all known on residual demand, which
/// [`PartiallyEvaluatableOperation::split_on_residual_demand`] returns. Its _known part_ is bound in the known-side
/// context and computes every output of the application together with the interior values that the residual policy
/// saves, and its _replay part_ is emitted into the residual program and recomputes the demanded outputs from those
/// saved values and from the known inputs that it reads.
#[derive(Clone)]
pub struct ResidualDemandSplit<V: Value> {
    /// Outputs of the known part, which are every output of the application, in output order.
    known_outputs: Vec<PartialEvaluationValue<V>>,

    /// Outputs of the replay part, which are one residual value per demanded output, in output order.
    replayed_outputs: Vec<PartialEvaluationValue<V>>,
}

impl<V: Value> ResidualDemandSplit<V> {
    /// Creates a new [`ResidualDemandSplit`] from the outputs of its known part and of its replay part.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::InvalidArgument`] when one of `known_outputs` is unknown or one of `replayed_outputs`
    /// is known, because the known part computes known values and the replay part computes residual values.
    pub fn new(
        known_outputs: Vec<PartialEvaluationValue<V>>,
        replayed_outputs: Vec<PartialEvaluationValue<V>>,
    ) -> Result<Self, ProgramError> {
        if let Some(index) = known_outputs.iter().position(PartialEvaluationValue::is_unknown) {
            return Err(ProgramError::InvalidArgument {
                message: format!("output {index} of the known part of a residual demand split is unknown"),
            });
        }
        if let Some(index) = replayed_outputs.iter().position(PartialEvaluationValue::is_known) {
            return Err(ProgramError::InvalidArgument {
                message: format!("output {index} of the replay part of a residual demand split is known"),
            });
        }
        Ok(Self { known_outputs, replayed_outputs })
    }

    /// Returns the outputs of the known part, which are every output of the application, in output order.
    #[inline]
    pub fn known_outputs(&self) -> &[PartialEvaluationValue<V>] {
        &self.known_outputs
    }

    /// Returns the outputs of the replay part, which are one residual value per demanded output, in output order.
    #[inline]
    pub fn replayed_outputs(&self) -> &[PartialEvaluationValue<V>] {
        &self.replayed_outputs
    }

    /// Returns the outputs of the known part and of the replay part, in that order.
    #[inline]
    pub fn into_parts(self) -> (Vec<PartialEvaluationValue<V>>, Vec<PartialEvaluationValue<V>>) {
        (self.known_outputs, self.replayed_outputs)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrType, ArrayOperation, ArrayReference, ArrayType, DataType, DimensionBounds, DimensionError,
        DimensionType, DimensionValue, Shape,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::NothingSavable;
    use crate::operations::{ConditionOperation, NegOperation, ReferenceReadOperation, ReferenceWriteOperation};
    use crate::parameters::Placeholder;
    use crate::partial::contexts::PartialEvaluationContext;
    use crate::partial::residuals::ResidualPolicyReference;
    use crate::partial::tests::{TestOperation, TestValue};
    use crate::partial::values::{PartialEvaluationOutput, PartialValueMaterialization};
    use crate::programs::{AtomId, ProgramBuilder, ReferenceType, TypeError};

    use super::*;

    #[test]
    fn test_recursive_partial_evaluation_driver_partially_evaluate_program_preserves_entry() {
        let dimension = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![dimension.dimension().clone()]));
        let reference_type = ReferenceType::new(array_type.clone());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(dimension.into());
        let array = builder.add_input(array_type.clone().into());
        let reference = builder.add_input(reference_type.clone().into());
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, array], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
            .unwrap();
        let regions = AttachedRegions::from(vec![program.clone()]);
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let knowledge = [
            PartialValue::Known(TestValue::Dimension(DimensionValue::constant(2).unwrap())),
            PartialValue::Unknown(array_type.into()),
            PartialValue::Unknown(reference_type.into()),
        ];

        // Both single and repeated residual evaluation check the known extent before the write. The array and the
        // reference keep their supplied types, which share the identity `extent` (a reference cannot be cast, so the
        // array is not narrowed either), so the residual entry relates them, and an entry guard relates them to the
        // known extent. A matching array is written, and a mismatching array or reference is rejected before the write.
        let check = |repeated_residual: bool| {
            let driver = RecursivePartialEvaluationDriver { regions: &regions, repeated_residual };
            let evaluation =
                driver.partially_evaluate_program(&context, program.entry_region_ref(), &knowledge).unwrap();
            let reference = ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap());
            let inputs =
                [TestValue::Array(Array::vector(vec![4f32; 2]).unwrap()), TestValue::Reference(reference.clone())];
            assert_eq!(evaluation.interpret(context.parent(), &inputs), Ok(Vec::new()));
            assert_eq!(reference.read().unwrap(), Array::vector(vec![4f32; 2]).unwrap());

            let reference = ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap());
            let inputs =
                [TestValue::Array(Array::vector(vec![4f32; 3]).unwrap()), TestValue::Reference(reference.clone())];
            assert_eq!(
                evaluation.interpret(context.parent(), &inputs),
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "extent".to_string(),
                    expected: 3,
                    actual: 2,
                }))),
            );
            assert_eq!(reference.read().unwrap(), Array::vector(vec![1f32; 2]).unwrap());

            let reference = ArrayReference::new(Array::vector(vec![1f32; 3]).unwrap());
            let inputs =
                [TestValue::Array(Array::vector(vec![4f32; 3]).unwrap()), TestValue::Reference(reference.clone())];
            assert_eq!(
                evaluation.interpret(context.parent(), &inputs),
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "extent".to_string(),
                    expected: 2,
                    actual: 3,
                }))),
            );
            assert_eq!(reference.read().unwrap(), Array::vector(vec![1f32; 3]).unwrap());
        };

        check(false);
        check(true);
    }

    #[test]
    fn test_partial_evaluation_driver_partitions_for_repeated_residual_calls() {
        type TestContext = EagerContext<TestValue, TestOperation>;

        assert!(!PartialEvaluationDriver::<TestContext>::partitions_for_repeated_residual_calls(&EmptyRegionDriver));
        let regions = AttachedRegions::<TestValue, TestOperation>::Empty;
        let ordinary = RecursivePartialEvaluationDriver { regions: &regions, repeated_residual: false };
        let repeated = RecursivePartialEvaluationDriver { regions: &regions, repeated_residual: true };
        assert!(!PartialEvaluationDriver::<TestContext>::partitions_for_repeated_residual_calls(&ordinary));
        assert!(PartialEvaluationDriver::<TestContext>::partitions_for_repeated_residual_calls(&repeated));
    }

    #[test]
    fn test_recursive_partial_evaluation_driver_partition_program() {
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let destination = builder.add_input(reference_type.clone());
        let source = builder.add_input(reference_type);
        let update = builder.add_input(scalar_type);
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![destination, update], None)
            .unwrap();
        let read = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![source], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let regions = AttachedRegions::from(vec![program.clone()]);
        let driver = RecursivePartialEvaluationDriver { regions: &regions, repeated_residual: false };
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new())
            .with_defer_ordered_effects(true);

        // A fresh partition inherits configuration, but not the active context's recorded ordering constraints.
        let partition = driver.partition_program(&context, program.entry_region_ref(), &[true, true, true]).unwrap();
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Known(0)]);
        assert!(context.defer_ordered_effects());

        // The same driver uses the effect placement passed to each call.
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new()).deferred_sibling();
        let partition = driver.partition_program(&context, program.entry_region_ref(), &[true, true, true]).unwrap();
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert!(!context.defer_ordered_effects());
    }

    #[test]
    fn test_recursive_partial_evaluation_driver_split_instruction_on_residual_demand() {
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = branch.add_input(scalar_type.clone());
        let output = branch
            .add_instruction(ArrayOperation::from(NegOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        let branch = branch
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let input = builder.add_input(scalar_type);
        let first = builder.import_program(branch.clone());
        let second = builder.import_program(branch);
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![first, second], vec![predicate, input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let regions = AttachedRegions::from(vec![program.clone()]);
        let driver = RecursivePartialEvaluationDriver { regions: &regions, repeated_residual: false };
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        assert!(
            driver
                .split_instruction_on_residual_demand(&context, program.entry_region_ref(), 0, &[true])
                .unwrap()
                .is_none()
        );
        let context = context
            .with_residual_policy(&ResidualPolicyReference::new(NothingSavable))
            .with_defer_ordered_effects(true);
        assert!(
            EmptyRegionDriver
                .split_instruction_on_residual_demand(&context, program.entry_region_ref(), 0, &[true])
                .unwrap()
                .is_none()
        );
        let partition = driver
            .split_instruction_on_residual_demand(&context, program.entry_region_ref(), 0, &[true])
            .unwrap()
            .unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Known(0), Unknown(0)],
                ]
                known={
                    lambda %0:bool[], %1:f32[] .
                    let %2:f32[], %3:f32[] = condition %0 %1 [
                        true={
                            lambda %0:f32[] .
                            let %1:f32[] = neg %0
                            in (%1, %0)
                        },
                        false={
                            lambda %0:f32[] .
                            let %1:f32[] = neg %0
                            in (%1, %0)
                        },
                    ]
                    in (%2, %0, %3)
                }
                residual={
                    lambda %0:bool[], %1:f32[] .
                    let %2:f32[] = condition %0 %1 [
                        true={
                            lambda %0:f32[] .
                            let %1:f32[] = neg %0
                            in (%1)
                        },
                        false={
                            lambda %0:f32[] .
                            let %1:f32[] = neg %0
                            in (%1)
                        },
                    ]
                    in (%2)
                }"},
        );
        assert!(context.defer_ordered_effects());
    }

    #[test]
    fn test_residual_demand_split_new() {
        let value = Array::scalar(2f32).unwrap();
        let split = ResidualDemandSplit::new(
            vec![PartialEvaluationValue::known(value.clone())],
            vec![PartialEvaluationValue::<Array>::variable(ArrayType::scalar(DataType::F32), AtomId::new(3))],
        )
        .unwrap();
        assert_eq!(split.known_outputs[0].as_known(), Some(&value));
        assert_eq!(
            split.replayed_outputs[0].materialization(),
            PartialValueMaterialization::Variable { residual_atom: AtomId::new(3) },
        );
        let empty = ResidualDemandSplit::<Array>::new(Vec::new(), Vec::new()).unwrap();
        assert!(empty.known_outputs.is_empty());
        assert!(empty.replayed_outputs.is_empty());
    }

    #[test]
    fn test_residual_demand_split_new_rejects_misplaced_knownness() {
        let known = PartialEvaluationValue::known(Array::scalar(1f32).unwrap());
        let unknown = PartialEvaluationValue::<Array>::variable(ArrayType::scalar(DataType::F32), AtomId::new(0));
        assert_eq!(
            ResidualDemandSplit::new(vec![known.clone(), unknown.clone()], Vec::new()).map(|_| ()),
            Err(ProgramError::InvalidArgument {
                message: "output 1 of the known part of a residual demand split is unknown".to_string(),
            }),
        );
        assert_eq!(
            ResidualDemandSplit::new(vec![known.clone()], vec![unknown, known]).map(|_| ()),
            Err(ProgramError::InvalidArgument {
                message: "output 1 of the replay part of a residual demand split is known".to_string(),
            }),
        );
    }

    #[test]
    fn test_residual_demand_split_known_outputs() {
        let first = Array::scalar(2f32).unwrap();
        let second = Array::scalar(3f32).unwrap();
        let split = ResidualDemandSplit::new(
            vec![PartialEvaluationValue::known(first.clone()), PartialEvaluationValue::known(second.clone())],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            split.known_outputs().iter().map(PartialEvaluationValue::as_known).collect::<Vec<_>>(),
            vec![Some(&first), Some(&second)],
        );
    }

    #[test]
    fn test_residual_demand_split_replayed_outputs() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let split = ResidualDemandSplit::new(
            Vec::new(),
            vec![
                PartialEvaluationValue::<Array>::variable(scalar_type.clone(), AtomId::new(3)),
                PartialEvaluationValue::<Array>::variable(scalar_type, AtomId::new(1)),
            ],
        )
        .unwrap();
        assert_eq!(
            split.replayed_outputs().iter().map(PartialEvaluationValue::materialization).collect::<Vec<_>>(),
            vec![
                PartialValueMaterialization::Variable { residual_atom: AtomId::new(3) },
                PartialValueMaterialization::Variable { residual_atom: AtomId::new(1) },
            ],
        );
    }

    #[test]
    fn test_residual_demand_split_into_parts() {
        let value = Array::scalar(2f32).unwrap();
        let split = ResidualDemandSplit::new(
            vec![PartialEvaluationValue::known(value.clone())],
            vec![PartialEvaluationValue::<Array>::variable(ArrayType::scalar(DataType::F32), AtomId::new(3))],
        )
        .unwrap();
        let (known, replayed) = split.into_parts();
        assert_eq!(known.iter().map(PartialEvaluationValue::as_known).collect::<Vec<_>>(), vec![Some(&value)]);
        assert_eq!(
            replayed[0].materialization(),
            PartialValueMaterialization::Variable { residual_atom: AtomId::new(3) },
        );
    }
}
