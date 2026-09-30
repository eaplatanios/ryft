use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;

use crate::contexts::Context;
use crate::partial::contexts::{PartialEvaluationContext, ReferencePlacement};
use crate::partial::evaluations::PartialEvaluation;
use crate::partial::partitions::{EffectOrdering, PartitionedProgram};
use crate::partial::values::{PartialEvaluationValue, PartialValue};
use crate::programs::{
    EmptyRegionDriver, InstructionId, Operation, ProgramError, ReferenceAnalysis, ReferenceRoot, RegionDriver,
    RegionRef, Value, ValueId,
};
use crate::tracing::TracingContext;

#[cfg(doc)]
use crate::contexts::{Domain, StagingContext, ValueResolution};
#[cfg(doc)]
use crate::partial::contexts::PartialTracer;
#[cfg(doc)]
use crate::programs::{EffectClasses, Program};

/// [`RegionDriver`] that provides [`Instruction`](crate::Instruction)-scoped access to [`Region`](crate::Region)s
/// attached to a partially evaluated [`Operation`] application. A [`PartialEvaluationDriver`] borrows the current
/// instruction's regions and supports recursive partial evaluation. Operation rules receive it separately from their
/// durable [`PartialEvaluationContext`], so the borrowed region access cannot escape through a [`PartialTracer`].
/// [`RegionDriver`] provides structural region access, while this trait adds partial-evaluation-specific recursion.
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
    /// rather than executed through the active context.
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
/// [`RegionDriver`], preserving the operation-defined ordering of owned [`Region`](crate::Region)s, borrowed regions,
/// and shared callees without collecting [`Program`]s or region views. Recursive requests re-enter partial evaluation
/// for a selected region or partition it into known and residual programs.
pub(super) struct RecursivePartialEvaluationDriver<'r, D> {
    /// Application-scoped [`RegionDriver`], in [`Operation`]-defined order.
    pub(super) driver: &'r D,

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

impl<D> RecursivePartialEvaluationDriver<'_, D> {
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
    ///   - `inputs`: Partially evaluated operands in source operand order.
    ///   - `deferred_instructions`: Source instructions explicitly required to remain residual, including allocations
    ///     that must be created afresh for each residual call.
    ///   - `deferred_effect_ordering`: Accumulated ordering constraints of earlier deferred source work. Updated only
    ///     when reference analysis is supplied; nested emission still enforces ordering within this instruction.
    ///   - `reference_analysis`: Canonical source-reference analysis for repeated residual calls. Omitting it retains
    ///     the active context's ordering rules without performing source-reference analysis.
    pub(super) fn partially_evaluate_instruction<C: Context>(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        instruction_index: usize,
        inputs: &[PartialEvaluationValue<C::Value>],
        deferred_instructions: &HashSet<InstructionId>,
        deferred_effect_ordering: &RefCell<EffectOrdering<ReferenceRoot>>,
        reference_analysis: Option<&ReferenceAnalysis>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        D: RegionDriver<C::Constant, C::Operation>,
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
            let mut instruction_context = context.clone();
            instruction_context.defer_ordered_effects =
                Rc::new(Cell::new(deferred_effect_ordering.borrow().conflicts(ordering)));
            instruction_context
        });

        let context = instruction_context.as_ref().unwrap_or(context);
        let must_defer = deferred_instructions.contains(&instruction_id);
        let outputs = if must_defer || shared_reference_boundary {
            let programs = self.regions().map(RegionRef::to_program).collect();
            if must_defer {
                context.residualize(instruction.operation().clone(), programs, inputs)
            } else {
                context.fold_or_residualize(instruction.operation().clone(), programs, inputs)
            }
        } else {
            instruction.operation().partially_evaluate(context, self, inputs)
        }?;

        if let Some(ordering) = effect_ordering
            && context.defer_ordered_effects.get()
        {
            deferred_effect_ordering.borrow_mut().extend(&ordering);
        }

        Ok(outputs)
    }
}

impl<V: Value, O: Operation<Type = V::Type>, D: RegionDriver<V, O>> RegionDriver<V, O>
    for RecursivePartialEvaluationDriver<'_, D>
{
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.driver.regions()
    }
}

impl<C: Context, D: RegionDriver<C::Constant, C::Operation>> PartialEvaluationDriver<C>
    for RecursivePartialEvaluationDriver<'_, D>
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
            return region.partially_evaluate_in_context(context.parent(), knowledge, context.allow_effect_folding);
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
        let inputs = knowledge
            .iter()
            .enumerate()
            .map(|(index, value)| match value {
                PartialValue::Known(value) => PartialEvaluationValue::known_input(value.clone()),
                PartialValue::Unknown(r#type) => nested.unknown_input(r#type.clone(), index),
            })
            .collect();

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
                    context.allow_effect_folding,
                    false,
                    None,
                    residual_placement,
                )
                .map(|(partition, _)| partition)
        }
    }
}

/// [`Operation`] that supports partial evaluation via [`Program::partially_evaluate`]. This trait lets an individual
/// operation decide how partial evaluation treats it. It can be implemented with an empty implementation block,
/// deferring to [`PartialEvaluationContext::fold_or_residualize`], which is what most operations do, or its behavior
/// can be customized by overriding the [`PartiallyEvaluatableOperation::partially_evaluate`] function.
///
/// # Type Parameters
///
///   - `C`: Known-side [`Context`] that partial evaluation folds known work through. Its
///     [`Operation`](Domain::Operation) is the operation family of the residual [`Program`] and of any inlined nested
///     programs (e.g., the enum this operation may belong to). Its [`Constant`](Domain::Constant) is the staged
///     constant space those programs store. Finally, its [`Value`](Domain::Value) is the space known values flow in
///     (i.e., concrete values under eager contexts and [`Tracer`](crate::Tracer)s into the outer program under
///     [`StagingContext`]s).
///
/// # Deriving Partially Evaluatable Operation Enums
///
/// The `#[derive(Operation)]` macro generates a [`PartiallyEvaluatableOperation`] implementation for operation enums.
/// Native variants forward to their payload's own rule, and the generated per-payload predicates transport that rule's
/// value and context requirements to the enum's use site. Declared member variants instead use the enclosing enum's
/// canonical fold-or-residualize path because member-side partial values cannot represent values belonging to other
/// members of the composite universe. This preserves correct folding and residualization without a second projected
/// partial-value protocol. Refer to the documentation of [`Operation`] for the full derive contract. Partial evaluation
/// is always generated and does not require a `dispatch(...)` selection.
pub trait PartiallyEvaluatableOperation<C: Context>: Clone + Into<C::Operation> {
    /// Partially evaluates this [`PartiallyEvaluatableOperation`] for the provided [`PartialEvaluationValue`]s. Unless
    /// overridden, this function will default to calling [`PartialEvaluationContext::fold_or_residualize`] which uses
    /// the following semantics:
    ///
    ///   - When *all* of the operation's inputs are [`Known`](PartialValue::Known), it **folds** the operation by
    ///     [`bind`](Context::bind)ing it in the known-side context, interpreting it immediately under an eager context,
    ///     and staging it into the outer program under a [`StagingContext`], so that the operation's outputs become
    ///     known values and the operation contributes nothing to the residual [`Program`]. Pure regionless operations
    ///     without references remain residual when eager execution returns [`ProgramError::UnsupportedOperation`].
    ///   - Otherwise, it **residualizes** the operation unchanged, meaning that it emits the operation into the
    ///     residual program over its inputs' residual program [`Atom`](crate::Atom)s, materializing each known input as
    ///     a residual input for a known variable or as an inlined residual program constant for a literal, so that the
    ///     operation runs at residual program execution time.
    ///   - An operation with an ordered effect (i.e., [`EffectClasses::is_ordered`] over the operation and its
    ///     executable computation regions) also follows the ordering rules on [`PartialEvaluationContext`] where once
    ///     an ordered operation has been staged, every later ordered operation is staged too, even when all of its
    ///     inputs are known, and reference operations under an eager known side follow the context's
    ///     [`ReferencePlacement`].
    ///   - An operation that carries deferred work over the same scope is always residualized, even when all of its
    ///     inputs are known. Refer to [`PartialEvaluationContext::fold_or_residualize`] for more information.
    ///
    /// There are situations where overriding this function can result in improved performance and better partitioning
    /// of a computation into known and unknown parts. For example, a `condition` instruction whose predicate is
    /// [`Known`](PartialValue::Known) and Boolean-concretizable may ask the context to inline the selected branch and
    /// return that branch's output trace values, so that the condition disappears from the residual program and only
    /// the taken branch's work survives. Rules that inspect known *payloads* must gate that inspection on a
    /// [`Constant`](ValueResolution::Constant) [`Context::resolve`] resolution because a known value under a staging
    /// known-side context may be a [`Tracer`](crate::Tracer) into the outer program rather than a program constant,
    /// and partial evaluation should fall back to a conservative rewrite otherwise. Resolving to a constant alone
    /// does not guarantee that the payload is host-inspectable; rules that inspect it require the corresponding
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
        context.fold_or_residualize(self.clone(), driver.regions().map(|region| region.to_program()).collect(), inputs)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayIrType, ArrayType, DataType};
    use crate::contexts::EagerContext;
    use crate::operations::{ReferenceReadOperation, ReferenceWriteOperation};
    use crate::parameters::Placeholder;
    use crate::partial::contexts::PartialEvaluationContext;
    use crate::partial::tests::{TestOperation, TestValue};
    use crate::partial::values::PartialEvaluationOutput;
    use crate::programs::{ProgramBuilder, ReferenceType};

    use super::*;

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
        let regions = vec![program];
        let driver = RecursivePartialEvaluationDriver { driver: &regions, repeated_residual: false };
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        context.defer_ordered_effects.set(true);

        // A fresh partition inherits configuration, but not the active context's recorded ordering constraints.
        let partition = driver.partition_program(&context, regions[0].entry_region_ref(), &[true, true, true]).unwrap();
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Known(0)]);
        assert!(context.defer_ordered_effects.get());

        // The same driver uses the effect placement passed to each call.
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new()).deferred_sibling();
        let partition = driver.partition_program(&context, regions[0].entry_region_ref(), &[true, true, true]).unwrap();
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert!(!context.defer_ordered_effects.get());
    }
}
