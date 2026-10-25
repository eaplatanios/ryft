//! Loops over a loop-carried state through [`WhileOperation`], which applies an attached body
//! [`Region`](crate::Region) to the state while an attached condition [`Region`](crate::Region) over that same state
//! produces a true predicate, and returns the final state.
//!
//! A while application takes the initial state and returns the final state with the same signature. The condition
//! and the body receive the state, the condition returns one Boolean predicate, and the body returns the next state;
//! [`WhileType`] states this contract once per type universe, including that a batched (prefix-shaped) predicate
//! masks the state per item and that a reference carry leaves the body as the reference it entered with. An optional
//! [`iteration_bound`](WhileOperation::with_iteration_bound) truncates the loop after that many body applications by
//! definition, which every rule honors. Every rule validates this complete contract before it executes, rewrites,
//! discharges, or stages the regions (refer to [`WhileOperation::validate_application`]), so whether an application
//! is accepted depends neither on the transformation nor on the trip count. The rules are:
//!
//!   - **Interpretation:** The condition is evaluated on the current state and the body is applied while any item of
//!     the predicate is true, each item of the state taking the body's update exactly when its predicate is true,
//!     until the bound is reached.
//!   - **Reference Discharge:** Discharged references become state carries; a condition that mutates a discharged
//!     allocation rotates the loop into do-while form, which evaluates the condition exactly as often as the source.
//!   - **Partial Evaluation:** Known loop-invariant state folds into both regions, and a known state subset whose
//!     next values and predicate fold from known state alone splits into a known loop and the original loop.
//!   - **Batching:** A fixed point over the state batch axes batches the regions once; a predicate that becomes
//!     batched widens every array state element and the bound loop masks per item.
//!   - **Differentiation:** An eager loop runs its duals directly; a bounded staged loop stores per-iteration
//!     residuals in `[bound, …]` stacks and stages one masked tangent `scan`, whose transposition gives reverse mode;
//!     an unbounded staged loop stages one fused doubled-state loop, which forward mode supports and transposition
//!     rejects.
//!
//! XLA lowers a while loop to [StableHLO `while`](https://openxla.org/stablehlo/spec#while).

use std::collections::BTreeSet;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch,
    ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayMemberType, ArrayType, DataType, MAX_DIMENSION_EXTENT,
    ShardableType, Sharding,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
    BatchingTracer, ProgramBatchingOutputAxesPolicy,
};
use crate::captures::CaptureReference;
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    CotangentAccumulator, DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationDriver,
    DifferentiationDual, DifferentiationError, DifferentiationPolicy, DifferentiationTracer, ResidualZeroProvider,
    TangentActivity, TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::arithmetic::AddOperation;
use crate::operations::collectives::parallel_vary::{PARALLEL_VARY_OPERATION_NAME, ParallelVaryOperation};
use crate::operations::constants::one::OneOperation;
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::control_flow::condition::ConditionOperation;
use crate::operations::control_flow::scan::ScanOperation;
use crate::operations::control_flow::select::SelectOperation;
use crate::operations::control_flow::{
    TemporalResidualOperation, TemporalResidualType, canonicalize_carry_axes, passed_through_carries,
    refine_output_types, stabilize_carry_axes, validate_output_identities, validate_reference_carry_roots,
    validate_region_input,
};
use crate::operations::manipulation::broadcasting::BroadcastOperation;
use crate::operations::manipulation::slicing::DynamicUpdateSliceOperation;
use crate::operations::reductions::{ReduceOperation, ReductionKind};
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput,
    PartialEvaluationOutput, PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    AtomId, CalleeRegionDriver, Concretizable, InputRegionProvenance, MaybeZero, Operation, OperationFormatter,
    OperationProjection, OperationProvider, OutputRegionProvenance, Program, ProgramBuilder, ProgramError,
    ReferenceAccessMode, ReferenceDischargeAllocationId, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeRegionBoundary, ReferenceDischargeRegionBoundaryInsertion,
    ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceType, RegionDataFlow, RegionDataFlowBoundary,
    RegionDataFlowRule, RegionDataFlowSource, RegionDataFlowSources, RegionInterface, RegionLiveness, RegionRef,
    RegionSlot, Type, TypeError, Typed, Value,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this since it is mostly vibe coded.

/// Canonical operation name for [`WhileOperation`].
pub const WHILE_OPERATION_NAME: &str = "while";

/// [`Operation`] that repeatedly applies a body [`Region`](crate::Region) to a loop-carried state while a condition
/// [`Region`](crate::Region) over that same state produces a true Boolean predicate. The condition and body
/// consume identical state type signatures, the body produces the next state with that same signature, and the
/// operation outputs the final state once the condition produces false.
///
/// The condition and body computations are not part of this payload: they are [`Region`](crate::Region)s attached to
/// the [`Instruction`](crate::Instruction) applying the operation, in the [`region_slots`](Operation::region_slots)
/// order `["condition", "body"]`, and semantic rules reach them through their driver-granted region access. Owned
/// loops supply the two [`Program`]s through the `driver` argument of [`Context::bind`].
/// [`Operation::infer_output_types`] validates the loop contract over the attached [`RegionInterface`]s: the condition
/// and body share the loop-carried state input signature, the condition returns exactly one Boolean predicate, and the
/// body returns the state signature. A batched (per-item) predicate requires both regions to be pure and free of
/// deferred work because neither observable effects nor transformation obligations can be masked for finished items.
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
///     condition forwards tangents unchanged without evaluating residuals beyond the actual trip count. Reverse mode
///     transposes that scan when its residual types support the required storage and transposition operations.
///   - **Unbounded staged (forward-only).** Shared forward-mode execution stages one fused loop over primal and
///     tangent state. Separate linearization can split a pure loop when its predicate depends only on known primal
///     state: the tangent loop recomputes the primal values it needs because no statically shaped residual stack is
///     available. Loops that cannot satisfy this partition are rejected, and unbounded tangent loops do not support
///     transposition.
///
/// The `T` parameter fixes the loop's type universe in the payload itself. Consequently, one concrete
/// [`WhileOperation<T>`](WhileOperation) has exactly one [`Operation<Type = T>`](Operation) contract even though the
/// shared implementation supports homogeneous array and array IR loops.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WhileOperation<T: Type> {
    /// Optional semantic iteration bound: when present, the loop runs at most this many iterations by definition,
    /// truncating even while the condition still produces true.
    pub(crate) iteration_bound: Option<usize>,

    /// Type universe of the loop-carried state and attached regions.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> WhileOperation<T> {
    /// Creates a new [`WhileOperation`] with no semantic iteration bound. The condition and body [`Program`]s are
    /// supplied separately as the operation's attached regions via the region driver passed to [`Context::bind`];
    /// [`Operation::infer_output_types`] validates the loop contract over their interfaces.
    #[inline]
    pub fn new() -> Self {
        Self { iteration_bound: None, marker: PhantomData }
    }

    /// Returns this [`WhileOperation`] with its semantic iteration bound set to `bound` (or cleared when `bound` is
    /// `None`). When present, the bound must be at least `1` and at most [`MAX_DIMENSION_EXTENT`], so it fits the
    /// signed 64-bit counters used by compiled loops and the static residual-stack length used by differentiation.
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
            return Err(
                TypeError::invalid(format!("`{WHILE_OPERATION_NAME}` iteration bound must be at least 1")).into()
            );
        }
        if let Some(bound) = bound
            && bound > MAX_DIMENSION_EXTENT
        {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` iteration bound {bound} exceeds backend maximum {MAX_DIMENSION_EXTENT}",
            ))
            .into());
        }
        self.iteration_bound = bound;
        Ok(self)
    }

    /// Returns the semantic iteration bound of this [`WhileOperation`], if any. Refer to the documentation of
    /// [`Self::with_iteration_bound`] for the truncation semantics a bound carries.
    #[inline]
    pub fn iteration_bound(&self) -> Option<usize> {
        self.iteration_bound
    }

    /// Validates that the `body` of this loop returns every reference carry as its own body input (refer to
    /// [`validate_reference_carry_roots`]), which is part of the contract that [`Self::validate_application`] checks
    /// before every rule runs, including when no iteration runs, so that whether a loop is accepted does not depend
    /// on the transform that reaches it or on how its inputs alias.
    pub(crate) fn validate_reference_carries<V: Value<Type = T>, O: Operation<Type = T>>(
        &self,
        body: RegionRef<'_, V, O>,
    ) -> Result<(), ProgramError> {
        validate_reference_carry_roots(WHILE_OPERATION_NAME, body, 0..body.input_ids().len())
    }

    /// Validates the shared state positions and the condition's predicate boundary used by region data flow.
    fn validate_data_flow_boundary(&self, boundary: RegionDataFlowBoundary<'_>) -> Result<(), ProgramError> {
        if boundary.regions.len() != 2
            || boundary.input_count != boundary.output_count
            || boundary.regions[0].input_count != boundary.input_count
            || boundary.regions[0].output_count != 1
            || boundary.regions[1].input_count != boundary.input_count
            || boundary.regions[1].output_count != boundary.output_count
        {
            return Err(ProgramError::MalformedProgram(format!(
                "`{WHILE_OPERATION_NAME}` region data flow requires matching state boundaries and one condition output",
            )));
        }
        Ok(())
    }
}

impl<T: WhileType> WhileOperation<T> {
    /// Validates the application of this loop to inputs of the types `input_types` with the `condition` and `body`
    /// regions, and returns its output types. Every rule validates this complete contract before it executes,
    /// rewrites, discharges, or stages the regions, so that whether an application is accepted does not depend on the
    /// transform that reaches it. The contract consists of the boundary that [`Operation::infer_output_types`]
    /// validates, of the reference contract that only the body reveals (refer to [`Self::validate_reference_carries`]),
    /// and of the loop invariance of the state that [`WhileType::is_loop_invariant_state`] requires under a batched
    /// predicate: such state has no per-item value that masking could select, so the body must pass it through
    /// unchanged, which makes masking it the identity on every path (eager interpretation, the masked form of
    /// differentiation, and the lowering all forward it unmasked).
    ///
    /// # Parameters
    ///
    ///   - `input_types`: Types of the operation inputs (i.e., the initial state).
    ///   - `condition`: Condition region of the loop.
    ///   - `body`: Body region of the loop.
    pub(crate) fn validate_application<V: Value<Type = T>, O: Operation<Type = T>>(
        &self,
        input_types: &[T],
        condition: RegionRef<'_, V, O>,
        body: RegionRef<'_, V, O>,
    ) -> Result<Vec<T>, ProgramError> {
        let output_types = self.infer_output_types(input_types, &[condition.interface(), body.interface()])?;
        self.validate_reference_carries(body)?;
        let state_types = body.input_types();
        if T::is_batched_predicate(&condition.output_types()[0]) {
            let passes_through = passed_through_carries(body, 0..state_types.len());
            for (index, state_type) in state_types.iter().enumerate() {
                if T::is_loop_invariant_state(state_type) && !passes_through[index] {
                    return Err(TypeError::invalid(format!(
                        "`{WHILE_OPERATION_NAME}` state {index} of type `{state_type}` must be passed through \
                         unchanged by the body under a batched predicate, because it has no per-item value to mask",
                    ))
                    .into());
                }
            }
        }
        Ok(output_types)
    }
}

impl<T: Type> Copy for WhileOperation<T> {}

impl<T: Type> Default for WhileOperation<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: WhileType> Display for WhileOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: WhileType> Operation for WhileOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        WHILE_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("condition"), RegionSlot::computation("body")] }
    }

    fn validate_region_interfaces(&self, region_interfaces: &[RegionInterface<T>]) -> Result<(), ProgramError> {
        // The condition and the body both receive the loop state, and the body must map it to itself. Validating these
        // declared signatures here means that specialization cannot make a mismatched body agree with its state types,
        // and that eager execution rejects such a body even when the loop would not iterate.
        self.validate_region_count(region_interfaces.len())?;
        validated_while_interfaces(region_interfaces)?;
        Ok(())
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
        // Value inputs are validated with the directional declared-vs-actual `Type::is_refined_by` relation rather than
        // strict type equality, so actual inputs that carry metadata the state types leave unspecified (e.g., the
        // normalized shardings of concrete backend array types) or static extents within the bounds of dynamic state
        // dimensions are accepted. Reference inputs refine their state types by their referents (refer to
        // `validate_region_input`).
        // The outputs are the state types refined by the facts that the inputs establish (refer to
        // `refine_output_types`): an identity that the inputs fix to a static extent and that no state element defines
        // is loop-invariant, so every iteration carries values of that extent, while an identity that a first-class
        // dimension carry defines may change across iterations and stays symbolic. Metadata such as shardings is not
        // propagated, because the body may change it.
        let (_, body_interface) = validated_while_interfaces(region_interfaces)?;
        let state_types = body_interface.input_types();
        check_count!("input", input_types, state_types.len(), TypeError);
        for (index, (state_type, input_type)) in state_types.iter().zip(input_types).enumerate() {
            validate_region_input(WHILE_OPERATION_NAME, index, input_type, state_type, "state type")?;
        }
        let output_types = refine_output_types(state_types, input_types, state_types, |index| {
            self.reference_output_identity_input(index)
        })?;
        validate_output_identities(WHILE_OPERATION_NAME, input_types, output_types.as_slice())?;
        Ok(output_types)
    }

    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> InputRegionProvenance {
        if region_index < 2 { InputRegionProvenance::Input { index: input_index } } else { InputRegionProvenance::None }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 1, output_index }]
    }

    #[inline]
    fn region_data_flow(&self) -> RegionDataFlow<'_> {
        RegionDataFlow::Custom(self)
    }

    #[inline]
    fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
        // Every `while` output is a loop carry aligned with the same-position input, so the allocation constraint maps
        // positionally for any output the operation actually declares.
        Some(output_index)
    }

    #[inline]
    fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
        // The condition may read and mutate entering references: reference discharge rotates a loop whose condition
        // mutates a discharged allocation into do-while form, where the condition runs at the tail of the body and
        // publishes updates through the body's state carries, and writes to preserved references stay explicit effects.
        // Consumption leaves no successor state for a carry to hold, so the condition's policy names it as the one
        // access it refuses; the body keeps the permissive default and relies on the region summary, which rejects
        // consumption of an entering reference in every region.
        region_index != 0 || !mode.is_consuming()
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

impl<T: WhileType> RegionDataFlowRule for WhileOperation<T> {
    fn input_sources(
        &self,
        _region_index: usize,
        input_index: usize,
        boundary: RegionDataFlowBoundary<'_>,
    ) -> Result<RegionDataFlowSources, ProgramError> {
        // The region data-flow dispatcher validates `region_index` and `input_index` before calling this rule.
        self.validate_data_flow_boundary(boundary)?;
        // Both condition and body see the initial state or a state produced by the preceding body invocation.
        // Producer correspondence does not alter the separate reference-access and output-identity contracts.
        Ok(RegionDataFlowSources::Known(vec![
            RegionDataFlowSource::InstructionInput(input_index),
            RegionDataFlowSource::RegionOutput(OutputRegionProvenance { region_index: 1, output_index: input_index }),
        ]))
    }

    fn output_sources(
        &self,
        output_index: usize,
        boundary: RegionDataFlowBoundary<'_>,
    ) -> Result<RegionDataFlowSources, ProgramError> {
        self.validate_data_flow_boundary(boundary)?;
        // A condition that is initially false forwards the initial state. Expanding the body input also accounts for
        // every later state through its feedback source, without assuming any particular number of iterations.
        Ok(RegionDataFlowSources::Known(vec![RegionDataFlowSource::RegionInput {
            region_index: 1,
            input_index: output_index,
        }]))
    }

    fn execution_demands(
        &self,
        _used_outputs: &[bool],
        boundary: RegionDataFlowBoundary<'_>,
        _regions: &mut dyn RegionLiveness,
    ) -> Result<Vec<Option<Vec<bool>>>, ProgramError> {
        self.validate_data_flow_boundary(boundary)?;
        // The predicate executes even though it supplies no instruction output. While preserves its complete state
        // boundary, so replay must classify every body result, including results the surrounding region does not use.
        Ok(vec![Some(vec![true]), Some(vec![true; boundary.output_count])])
    }
}

// A `while`'s boundaries are all one state signature: its own inputs and outputs, its condition's inputs, and its
// body's inputs and outputs carry the same carries at the same positions. Threading discharged state therefore appends
// the same carries to every one of them, and the loop deliberately applies no read-only pruning: an allocation the loop
// merely reads still occupies a carry position, because dropping it would leave the body's boundary disagreeing with
// the condition's. The one asymmetry is forced by the operation's own contract — the condition returns only a
// predicate, so it receives the entering state and, as long as it only reads, publishes none. A condition that mutates
// a discharged (threaded) allocation has nowhere to publish its updates through that boundary, so the rule rotates such
// a loop into do-while form: the original condition is discharged once into the parent, producing the initial predicate
// and applying its effects exactly once before the loop; the predicate becomes a trailing carry; the rebuilt body runs
// the original body and then the original condition, returning the states the condition published in place of the
// body's and the fresh predicate; and the rebuilt condition merely projects the predicate carry. The condition is
// thereby emitted twice, and the loop evaluates it exactly as often and in the same order as the source did. Because
// rotation fixes the order to body then condition, a root written by both regions is accepted as well. A bounded loop
// rotates with its bound lowered by one, because the bound truncates the loop before the condition's next evaluation,
// whereas every rotated iteration evaluates the condition after the body; the last body application then runs after
// the loop under the final predicate, through a `condition` that otherwise forwards the state. A condition that writes
// only *preserved* references needs no rotation, because those writes remain explicit effects with no immutable
// successor state.
// A carry that partial reference discharge *preserved* keeps its declared position on every one of those boundaries and
// widens nothing at all. A preserved allocation reached only through an inherited capture gains a reference-typed carry
// so the rebuilt regions can bind that capture; in either case it enters as the reference the caller already holds, its
// accesses replay as the operations the source performed, and it publishes no state successor.
impl<T, C, P> ReferenceDischargeableOperation<C, P> for WhileOperation<T>
where
    T: WhileType + From<P::Referent> + From<ReferenceType<P::Referent>>,
    C: Context<Type = T, Operation: From<WhileOperation<T>> + From<ConditionOperation<T>>>,
    P: ReferenceDischargePolicy<C>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        let name = self.name();
        let carries = inputs.iter().map(|input| context.boundary_allocation(input)).collect::<Result<Vec<_>, _>>()?;

        // Both regions observe the same entering state, so one summary of the two sizes one boundary.
        let condition = driver.region(0)?;
        let body = driver.region(1)?;
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.validate_application(&input_types, condition, body)?;
        let condition_summary = context.region_summary(self, 0, condition, carries.as_slice())?;
        let body_summary = context.region_summary(self, 1, body, carries.as_slice())?;
        let mut summary = body_summary;
        summary.merge(&condition_summary);

        // An allocation the body returns is threaded even if the body never accesses it, because the rebuilt body must
        // resolve every reference that it returns, and the structural carry check above already guarantees that it
        // returns each reference carry as the reference that it entered with. A preserved reference already in the
        // carry list stays at its declared position; one reached only through a capture gains a reference-typed carry
        // rather than a state carry.
        let carried = carries.iter().copied().flatten().collect::<BTreeSet<_>>();
        let widening = context.boundary_widening(&summary, &carried)?;
        let entering = widening.entering().to_vec();

        // A read-only condition publishes nothing — it returns only its predicate — so its declared output allocations
        // need no `validate_predicted_output_allocations` pass; the structural carry check above (refer to
        // `validate_reference_carry_roots`) is the stronger version of that agreement for the one region that does
        // return references. A condition that mutates a discharged allocation is rotated to the tail of the body, so it
        // publishes the final state of every discharged allocation it mutates after its predicate, and the rotated body
        // forwards those states in place of the body's own.
        let condition_published = widening
            .threaded()
            .iter()
            .copied()
            .filter(|allocation| condition_summary.is_mutated(*allocation))
            .collect::<Vec<_>>();
        // Preserved reference writes remain explicit effects and need no immutable successor state. Rotate only
        // when the condition must publish a discharged allocation through its predicate-only boundary.
        let rotate = !condition_published.is_empty();
        let condition_result = driver.rebuild_region(
            context,
            driver.region(0)?,
            &ReferenceDischargeRegionBoundary::new(
                self,
                0,
                carries.clone(),
                ReferenceDischargeRegionBoundaryInsertion::new(entering.clone(), inputs.len()),
                [ReferenceDischargeRegionBoundaryInsertion::new(
                    condition_published.clone(),
                    condition.output_ids().len(),
                )
                .into()],
            ),
        )?;
        condition_result.validate_predicted_mutations(condition_published.as_slice(), name)?;
        let body_result = driver.rebuild_region(
            context,
            driver.region(1)?,
            &ReferenceDischargeRegionBoundary::symmetric(
                self,
                1,
                carries.clone(),
                ReferenceDischargeRegionBoundaryInsertion::new(entering.clone(), inputs.len()),
            ),
        )?;
        body_result.validate_predicted_mutations(widening.published(), name)?;

        // Rotation discharges the original condition once into the parent before any instruction input is read: its
        // effects and state updates land in this context exactly once, and the loop then enters with the updated state
        // and the resulting predicate as a trailing carry.
        let initial_predicate = match rotate {
            true => {
                let initial = driver.inline_region(context, 0, inputs.to_vec())?;
                check_count!("output", initial, 1, ProgramError);
                Some(initial[0].try_as_value("the `while` condition predicate")?.clone())
            }
            false => None,
        };
        let mut discharged_inputs = Vec::with_capacity(inputs.len() + entering.len() + 1);
        for input in inputs {
            discharged_inputs.push(context.boundary_value(input)?);
        }
        for allocation in &entering {
            discharged_inputs.push(
                context
                    .boundary_value(&ReferenceDischargeValue::Reference(context.allocation_reference(*allocation)?))?,
            );
        }
        let condition_program = condition_result.into_program();
        let body_program = body_result.into_program();
        let outputs = match initial_predicate {
            Some(predicate) => {
                discharged_inputs.push(predicate);
                let state_allocations =
                    carries.iter().copied().chain(entering.iter().copied().map(Some)).collect::<Vec<_>>();
                let (rotated_condition, rotated_body) = rotated_discharge_regions(
                    &condition_program,
                    &body_program,
                    state_allocations.as_slice(),
                    condition_published.as_slice(),
                )?;
                // The bound truncates the source loop after its last body application, before the condition's next
                // evaluation, whereas every rotated iteration evaluates the condition after the body. The rotated loop
                // therefore runs one body application fewer (none at all for a bound of one), and the last one runs
                // after the loop under the final predicate, which selects the rebuilt body or the state unchanged.
                let rotated_bound = self.iteration_bound.map(|bound| bound - 1);
                let mut outputs = match rotated_bound {
                    Some(0) => discharged_inputs,
                    rotated_bound => context.parent().bind(
                        WhileOperation::<T>::new().with_iteration_bound(rotated_bound)?,
                        vec![rotated_condition, rotated_body],
                        discharged_inputs.as_slice(),
                    )?,
                };
                check_count!("output", outputs, inputs.len() + entering.len() + 1, ProgramError);
                let predicate = outputs.pop().unwrap();
                if self.iteration_bound.is_some() {
                    let state_types = body_program.input_types();
                    let mut unchanged = ProgramBuilder::<C::Constant, C::Operation>::new();
                    let state = state_types.into_iter().map(|r#type| unchanged.add_input(r#type)).collect::<Vec<_>>();
                    let state_count = state.len();
                    let unchanged = unchanged.build::<Vec<C::Constant>, Vec<C::Constant>>(
                        state,
                        vec![Placeholder; state_count],
                        vec![Placeholder; state_count],
                    )?;
                    let mut conditional_inputs = Vec::with_capacity(outputs.len() + 1);
                    conditional_inputs.push(predicate);
                    conditional_inputs.extend(outputs);
                    outputs = context.parent().bind(
                        C::Operation::from(ConditionOperation::<T>::new()),
                        vec![body_program, unchanged],
                        conditional_inputs.as_slice(),
                    )?;
                }
                outputs
            }
            None => {
                context.parent().bind(*self, vec![condition_program, body_program], discharged_inputs.as_slice())?
            }
        };
        check_count!("output", outputs, inputs.len() + entering.len(), ProgramError);

        // A symmetric boundary returns a successor state for every carried allocation, including ones the loop only
        // read, so the merge records a mutation exactly where the summary saw one. Marking a read-only carry as written
        // would publish a hidden final-state output for an allocation the program never writes. A carry that survives
        // as a reference came back out as the same reference and has no state to merge.
        let mut results = Vec::with_capacity(inputs.len());
        for (position, output) in outputs.into_iter().enumerate() {
            match carries.get(position).copied().flatten() {
                Some(allocation) => {
                    context.merge_boundary_state(&summary, &widening, allocation, output)?;
                    results.push(inputs[position].clone());
                }
                None if position < inputs.len() => results.push(ReferenceDischargeValue::Value(output)),
                None => {
                    let allocation = entering[position - inputs.len()];
                    context.merge_boundary_state(&summary, &widening, allocation, output)?;
                }
            }
        }
        Ok(results)
    }
}

// The eager rule validates the application before the first condition (refer to `validate_application`), so that a
// malformed application is reported as such even when no iteration runs. It then evaluates the condition and masks
// each body update with its predicate. The `shard_map` emulator uses the same rule over one value per device, whose
// `WhilePredicate` implementation requires device agreement and applies each device's own mask.
impl<C: Domain> InterpretableOperation<C> for WhileOperation<C::Type>
where
    C::Value: WhilePredicate,
    C::Type: WhileType,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let (condition, body) = (driver.region(0)?, driver.region(1)?);
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.validate_application(&input_types, condition, body)?;
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

// Partial-evaluation rule for [`WhileOperation`].
//
// A while's inputs are the initial loop state and its outputs are the final loop state (the same arity). Partial
// evaluation folds the known value of every *loop-invariant-known* state element into both nested programs: a state
// element is loop-invariant-known iff the body passes it through unchanged (its next-state output is its own state
// input), its init input is [`Known`](PartialValue::Known), and its type is not a reference and mentions no type
// identity. Such an element holds its init value on every iteration, so binding it to that constant inside the
// condition and body is sound and collapses every subcomputation that depended only on it. Invariance is structural
// rather than decided by comparing the body's next-state values with the init values, because value equality cannot
// distinguish values that the body changes without changing their equality class (e.g., a body that negates a state
// element of `0.0` returns `-0.0`, which compares equal to `0.0`). A folded element is rebuilt as a program constant,
// which excludes two kinds of state even when the body only passes them through: a reference carry must keep its
// allocation flowing through the loop boundary, and a type that mentions an identity cannot be rebuilt as a constant. A
// type that defines an identity (e.g., a first-class dimension) is already defined by the state input that the rebuilt
// condition and body keep, so a constant of that type would define the identity twice, and a type that refers to an
// identity (e.g., `f32[extent]`) would be replaced by the concrete static type of the constant, which no longer matches
// the loop's state signature.
//
// Both the body and the condition are then partially evaluated with the invariant-known state knowledge through the
// partial-evaluation driver's split requests. The condition reads the state too, so folding an invariant element can
// shrink it as well.
//
// The residual while keeps the *same* state set and therefore the same output arity as the original operation. A
// loop-invariant-known element is not dropped; instead its body next-state output is rebuilt as the constant init
// value and its uses fold away inside both programs. Because the condition and body run on the same loop-carried
// state each iteration, both residual programs are rebuilt over the loop's full state signature (in state order):
// each surviving unknown state element feeds the matching state input, and every known residual a program closed over
// is rebuilt as an inline residual-program constant (so the residual while needs no captures). The
// [`iteration_bound`](WhileOperation::iteration_bound) is preserved. The rewrite is emitted over the original while
// inputs unchanged.
//
// If no state element is loop-invariant-known, or a probe fails or folds to a value that is not a constant, the rule
// attempts the *closed-knownness split* before residualizing unchanged (provided that some input is known): when a
// known state subset's next values and the trip predicate fold from known state alone, the loop separates into a known
// loop bound on the known side and the residual loop kept whole (see `split_while_by_closed_knownness`). This is the
// split that makes [`Program::linearize`] total over the fused doubled-state loops staged by the unbounded `while`
// forward-mode rule.
impl<C: Context> PartiallyEvaluatableOperation<C> for WhileOperation<C::Type>
where
    C::Type: WhileType,
    C::Operation: From<WhileOperation<C::Type>>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // The rule requests all nested-computation work through its region access (region 0 is the condition and
        // region 1 the body), which keeps its bounds free of the operation family's own semantic traits.
        //
        // When every input is known the whole loop folds by binding it in the known-side context; defer to that
        // default behavior, after the validation that every rule performs, which a staging known-side context would
        // otherwise defer to the consumer of the staged loop.
        let condition = driver.region(0)?;
        let body = driver.region(1)?;
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.validate_application(&input_types, condition, body)?;
        if inputs.iter().all(PartialEvaluationValue::is_known) {
            return context.fold_or_residualize(
                C::Operation::from(*self),
                vec![condition.to_program(), body.to_program()],
                inputs,
            );
        }

        let state_types = body.input_types();
        let state_count = state_types.len();

        // The invariance probes below fold the condition and body through the *live* known-side context, and the
        // closed-knownness split's known loop re-runs the known part of every iteration. For an effectful loop the
        // probes would execute (eager) or stage (staging) the loop's effects once more and the split would run them
        // twice, so effectful loops skip both and residualize unchanged (see the effect placement contract on
        // `PartialEvaluationContext::fold_or_residualize`). Every reference operation is `OrderedState`, so a loop
        // touching references is never pure and no probe below can execute a reference operation, fold a reference
        // carry across the loop boundary, or change the active context's effect-ordering state. Deferred work skips
        // both for the same reason: every probe would residualize its obligation again.
        if !condition.effects().classes().is_empty()
            || !body.effects().classes().is_empty()
            || condition.effects().has_deferred_work()
            || body.effects().has_deferred_work()
        {
            return context.fold_or_residualize(
                C::Operation::from(*self),
                vec![condition.to_program(), body.to_program()],
                inputs,
            );
        }

        // Every pure fallback below first attempts the closed-knownness split — a known state subset whose next
        // values and trip predicate fold from known state alone separates into a known loop bound on the known side
        // and the residual loop kept whole (see `split_while_by_closed_knownness`) — and only residualizes the loop
        // unchanged when the split does not apply.
        let split_or_residualize = |context: &PartialEvaluationContext<C>| {
            if inputs.iter().any(PartialEvaluationValue::is_known)
                && let Some(outputs) = split_while_by_closed_knownness(context, self, condition, body, inputs, driver)?
            {
                return Ok(outputs);
            }
            context.fold_or_residualize(
                C::Operation::from(*self),
                vec![condition.to_program(), body.to_program()],
                inputs,
            )
        };

        // A state element can only fold if its type is not a reference and mentions no type identity (see the rule
        // documentation above), the body passes it through unchanged, and its init input is known *and* resolves to a
        // constant in the known-side context: the folded value must be embeddable as a rebuilt-program constant, and
        // skipping symbolic knowns also keeps the probes from folding symbolic known work into a live staging context.
        let passes_through = passed_through_carries(body, 0..state_count);
        let state_inits = (0..state_count)
            .map(|index| {
                inputs[index]
                    .as_known()
                    .filter(|value| {
                        !state_types[index].is_reference()
                            && state_types[index].identities().next().is_none()
                            && passes_through[index]
                            && context.parent().resolve(value).is_constant()
                    })
                    .cloned()
            })
            .collect::<Vec<Option<C::Value>>>();

        // With no invariant elements there is nothing the rebuild below could embed, so skip the live-context probes
        // entirely. In particular, under a staging known-side context every symbolic known init lands here, which is
        // where the closed-knownness split serves `Program::linearize`.
        let invariant = state_inits.iter().map(Option::is_some).collect::<Vec<bool>>();
        if invariant.iter().all(|folded| !folded) {
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
        // condition can be false on entry), so a known-side fold that rejects its inputs must not fail partial
        // evaluation — the branch's work, and its error if the loop is ever entered, stays behind the condition.
        // Both programs are pure here (the effects gate above), so partially completed probe folds are safe to
        // discard, and the split stays error-consistent: its partitions stage through fresh contexts without
        // executing known work, and its known loop replays the loop's exact runtime semantics
        // (running nothing when the condition is false on entry).
        let Ok(body_evaluation) = driver.partially_evaluate_program(context, body, &state_knowledge(&invariant)) else {
            return split_or_residualize(context);
        };

        // The condition reads the loop state too, so folding the invariant-known state can shrink it as well.
        let condition_evaluation =
            match driver.partially_evaluate_program(context, condition, &state_knowledge(&invariant)) {
                Ok(evaluation) => evaluation,
                Err(_) => return split_or_residualize(context),
            };

        // The rebuild below embeds the probes' known values as inline program constants, which is only possible when
        // they all resolve to constants. Under a staging known-side context a probe can fold a constant-only chain into
        // a live-trace tracer, so a non-constant probe defers to the split-or-residualize fallback.
        if !context.all_knowns_are_constants(&body_evaluation)
            || !context.all_knowns_are_constants(&condition_evaluation)
        {
            return split_or_residualize(context);
        }

        // The residual while keeps the same state set, so its output arity matches the original while. The condition
        // and body run on the same loop-carried state each iteration, so both are rebuilt over the loop's full state
        // signature.
        let mut condition_builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        let condition_state_atoms = state_types
            .iter()
            .map(|state_type| condition_builder.add_input(state_type.clone()))
            .collect::<Vec<_>>();
        let condition_outputs =
            rebuild_while_program(context, &mut condition_builder, &condition_state_atoms, &condition_evaluation)?;
        let residual_condition = condition_builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            condition_outputs,
            vec![Placeholder; state_count],
            vec![Placeholder; 1],
        )?;

        let mut body_builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        let body_state_atoms =
            state_types.iter().map(|state_type| body_builder.add_input(state_type.clone())).collect::<Vec<_>>();
        let body_outputs = rebuild_while_program(context, &mut body_builder, &body_state_atoms, &body_evaluation)?;
        let residual_body = body_builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            body_outputs,
            vec![Placeholder; state_count],
            vec![Placeholder; state_count],
        )?;

        let while_operation = WhileOperation::new().with_iteration_bound(self.iteration_bound)?;

        // The folded loop may still have a closed known state subset (e.g., a counter that the folded limit bounds),
        // which the split recovers on the known side; the two rewrites compose, so the split runs over the rebuilt
        // regions. Otherwise the residual while's inputs are exactly the original while's inputs: each state element's
        // init value (now a known residual for the folded elements) in state order.
        if let Some(outputs) = split_while_by_closed_knownness(
            context,
            &while_operation,
            residual_condition.entry_region_ref(),
            residual_body.entry_region_ref(),
            inputs,
            driver,
        )? {
            return Ok(outputs);
        }
        context.fold_or_residualize(
            C::Operation::from(while_operation),
            vec![residual_condition, residual_body],
            inputs,
        )
    }
}

// Batching rule for [`WhileOperation`]. The rule builds batched loop *structure* and binds it into the parent
// context — interpreted eagerly under an eager parent (whose relaxed-predicate interpretation owns the per-item
// masked semantics) and staged into the enclosing trace under a staging parent:
//
//   1. Every batched array state input is realigned to batch axis `0` in the parent context, and the body is batched at
//      the state batch axes, iterating the axes to a fixed point (refer to `stabilize_carry_axes`): a while loop's
//      state types are loop-invariant, so a replicated state element whose update depends on a batched element
//      *becomes* batched. Each pass instantiates the body outputs at the current state axes (mirroring JAX's
//      `instantiate=carry_bat`), so the converged body is already aligned to the loop-invariant state layout, and
//      widened parent inputs gain their batch axis through the driver. First-class dimension state stays replicated,
//      because array IR batching admits no mapped dimension values, and reference state keeps the batch axis that its
//      referent fixes.
//   2. The condition is batched at the stabilized axes. When its predicate output stays *replicated*, one
//      [`WhileOperation`] over the batched condition and body is bound into the parent directly, preserving any
//      semantic [`iteration_bound`](WhileOperation::with_iteration_bound) (so bounded loops stay reverse-capable
//      under `batch`). A replicated predicate that already masks a prefix of the array state keeps that prefix ahead
//      of the new mapped carry axis, so its per-item selection remains valid.
//   3. When the predicate output is *batched* (per-item termination), every array state element is widened to a
//      batched element, the condition's predicate output is instantiated at axis `0` (re-batching the condition only
//      when its natural predicate axis is not already `0`), and one [`WhileOperation`] is bound directly with that
//      batched predicate (mirroring JAX's `_while_loop_batching_rule`). The predicate's `[axis_size]` shape is a prefix
//      of every widened array state shape, so the bound loop satisfies the relaxed predicate contract and its consumers
//      own the masked semantics: eager interpretation continues while any per-item predicate is true and freezes
//      finished items, and the XLA lowering reduces the predicate with `or` and masks carry updates with a broadcast
//      select. Dimension state rides through the loop as loop-invariant state that masking never touches (the relaxed
//      [`WhileType`] contract documents that invariance requirement), and the rebuilt loop's type contract decides
//      whether reference state may ride under a batched predicate. The iteration bound is preserved (batch items share
//      masked iterations, so capping the loop matches per-item truncation exactly).
//
// The policy's bookkeeping inputs (refer to [`BatchingPolicy::boundary_inputs`]) lead the state of the bound loop. The
// condition consumes them as loop state but returns only its predicate, so their forwarded outputs are projected out
// of its boundary.
impl<C, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>> for WhileOperation<ArrayType>
where
    C: Context<Type = ArrayType, Operation: From<WhileOperation<ArrayType>>>,
{
    #[inline]
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        batch_while(self, context, driver, inputs)
    }
}

impl<C> BatchableOperation<C, ArrayIrBatchingPolicy> for WhileOperation<ArrayIrType>
where
    C: Context<Type = ArrayIrType, Operation: From<WhileOperation<ArrayIrType>>>,
{
    #[inline]
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        batch_while(self, context, driver, inputs)
    }
}

/// Batches an application of `operation` to the batched `inputs` structurally (refer to the documentation of its
/// batching rules for the algorithm).
fn batch_while<T, C, P, D>(
    operation: &WhileOperation<T>,
    context: &BatchingContext<C, P>,
    driver: &D,
    inputs: &[P::Batch],
) -> Result<Vec<P::Batch>, BatchingError>
where
    T: WhileType + ArrayMemberType,
    C: Context<Type = T, Operation: From<WhileOperation<T>>>,
    P: BatchingPolicy<C>,
    D: BatchingDriver<C, P>,
{
    // The rule requests all nested-computation work through its region access (region 0 is the condition and region 1
    // the body), which keeps its bounds free of the operation family's own semantic traits.
    let condition = driver.region(0)?;
    let body = driver.region(1)?;
    let input_types = inputs.iter().map(|input| P::unbatched_type(input).into_owned()).collect::<Vec<_>>();
    operation.validate_application(&input_types, condition, body)?;
    let state_count = inputs.len();
    let mut state = canonicalize_carry_axes(context, driver, inputs)?;
    let mut state_axes = state.iter().map(P::batch_axis).collect::<Vec<_>>();
    let batched_body =
        stabilize_carry_axes(WHILE_OPERATION_NAME, context, driver, body, &mut state_axes, 0..state_count)?
            .ok_or_else(|| BatchingError::UnsupportedOperation {
                message: format!(
                    "`{WHILE_OPERATION_NAME}` loop batching failed to stabilize the loop state batch axes within \
                 {state_count} widening passes",
                ),
            })?;
    let mut batched_body = batched_body.into_parts().0;

    // A batched predicate means per-item termination, in which case every array state element participates in
    // per-item masking and is therefore widened to a batched element before the masked loop structure is built.
    let is_array = inputs.iter().map(|input| P::unbatched_type(input).as_array_type().is_ok()).collect::<Vec<_>>();
    let mut batched_condition =
        driver.batch_program(context, condition, state_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)?;
    let batch_varying = !batched_condition.output_axes()[0].is_replicated();
    let rebatched_state_axes = if batch_varying {
        state_axes
            .iter()
            .zip(&is_array)
            .map(|(axis, &array)| if array { BatchAxis::new(0) } else { *axis })
            .collect()
    } else {
        // A replicated predicate can already mask a prefix of the source state. Insert the mapped axis after that
        // prefix so the existing per-item predicate still broadcasts over every mapped carry.
        let predicate_rank = condition.output_types()[0].as_array_type().map_or(0, ArrayType::rank);
        state_axes
            .iter()
            .zip(&is_array)
            .map(|(axis, &array)| match array && predicate_rank > 0 && !axis.is_replicated() {
                true => BatchAxis::new(predicate_rank),
                false => *axis,
            })
            .collect::<Vec<_>>()
    };
    if rebatched_state_axes != state_axes {
        state_axes = rebatched_state_axes;
        batched_body = driver
            .batch_program(
                context,
                body,
                state_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::AlignEachTo(state_axes.clone()),
            )?
            .into_parts()
            .0;
        batched_condition = driver.batch_program(
            context,
            condition,
            state_axes.as_slice(),
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
    }
    // The condition batched above used these same state axes, so a predicate that naturally landed on axis 0 keeps that
    // program instead of being replayed for an alignment that would stage nothing.
    let batched_condition = match batch_varying {
        true => context.align_batched_program_outputs(
            driver,
            condition,
            state_axes.as_slice(),
            batched_condition,
            &[BatchAxis::new(0)],
        )?,
        false => batched_condition.into_parts().0,
    };

    // Align the parent state values with the final state axes: elements that became batched materialize their batch
    // axis, and already batched elements move their batch axis when it follows a replicated predicate prefix.
    for (value, axis) in state.iter_mut().zip(&state_axes) {
        if let Some(axis) = axis.axis()
            && P::batch_axis(value) != BatchAxis::new(axis)
        {
            *value = driver.align_batch_axis(context, value.clone(), axis)?;
        }
    }
    let bookkeeping_inputs = P::boundary_inputs(context.axis_extent());
    let bookkeeping_count = bookkeeping_inputs.len();
    let batched_condition = match bookkeeping_count {
        0 => batched_condition,
        _ => {
            let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
            let inputs = batched_condition
                .input_types()
                .into_iter()
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let outputs = builder.splice_program(&batched_condition, inputs.as_slice())?;
            let output_count = outputs.len() - bookkeeping_count;
            builder.build(
                outputs[bookkeeping_count..].to_vec(),
                vec![Placeholder; inputs.len()],
                vec![Placeholder; output_count],
            )?
        }
    };
    let mut packed_inputs = bookkeeping_inputs;
    packed_inputs.extend(state.iter().map(|value| P::value(value).clone()));
    let mut outputs =
        context.parent().bind(*operation, vec![batched_condition, batched_body], packed_inputs.as_slice())?;
    check_count!("output", outputs, bookkeeping_count + state_count, ProgramError);
    outputs.drain(..bookkeeping_count);
    Ok(outputs
        .into_iter()
        .zip(state_axes)
        .map(|(output, axis)| P::batch(output, axis))
        .collect::<Result<Vec<_>, _>>()?)
}

// Forward-mode (JVP) rule for [`WhileOperation`]. An [eager](Context::is_eager) context runs the loop directly at the
// concrete duals (see the crate-private `jvp_while_eagerly`), so eager forward mode is total over data-dependent
// `while` loops with no iteration bound. Staging contexts, and eager contexts whose loop predicate is batched and
// therefore has no single trip decision, stage the shared array-backed rule (see the crate-private
// `jvp_array_backed_while`): bounded loops stage the reverse-capable hybrid of an augmented primal `while` and a masked
// tangent `scan`, while unbounded loops stage the forward-only fused doubled-state loop (see the crate-private
// `jvp_while_fused`). Both type families share that rule; they differ only in their array-backed storage, which
// [`WhileResidualStackType`] and [`WhileResidualStackOperation`] describe.
impl<C> DifferentiableOperation<C> for WhileOperation<C::Type>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    C: Context<Type: WhileResidualStackType> + Zero<C::Value>,
    C::Value: Concretizable<bool>,
    C::Operation: OperationProvider<C::Type, ZeroOperation<C::Type>, Operation = C::Operation>
        + ResidualZeroProvider<C::Type, Operation = C::Operation>
        + From<ConditionOperation<C::Type>>
        + From<WhileOperation<C::Type>>
        + From<ScanOperation<C::Type>>
        + WhileResidualStackOperation<C::Type>,
    for<'o> &'o WhileOperation<C::Type>: TryFrom<&'o C::Operation>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // The rule requests all nested-computation work through its driver (region 0 is the condition and
        // region 1 the body), which keeps its bounds free of the operation family's own semantic traits.
        let input_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
        self.validate_application(&input_types, driver.region(0)?, driver.region(1)?)?;
        let condition = driver.region(0)?.to_program();
        let body = driver.region(1)?.to_program();
        if context.primal().is_eager()
            && let Some(outputs) = jvp_while_eagerly(self, &condition, &body, context, driver, inputs)?
        {
            return Ok(outputs);
        }
        jvp_array_backed_while(self, &condition, &body, context, driver, inputs)
    }
}

impl<V: Value, O> TransposableOperation<V, O> for WhileOperation<V::Type>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V::Type: WhileType,
    O: Operation<Type = V::Type>,
{
    // Rejects transposition. This rule is only reachable for *unbounded* staged while loops — the doubled-state
    // linear loop staged by the [`WhileOperation`] JVP rule, which recomputes primal state *forward* through
    // the iterations, so transposing it would have to run that recomputation backwards, which a while loop cannot
    // express. Two paths avoid it entirely: eager domains execute the loop directly over concrete duals and record a
    // straight-line pushforward during linearization, and bounded loops ([`WhileOperation::with_iteration_bound`])
    // never stage a linear `while` — their tangent side is a masked linear scan whose transpose is total, so reverse
    // mode through staged bounded loops flows through the scan transpose without reaching this rule.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        _accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // The loop is validated as every rule validates it before the unsupported transposition is reported.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.validate_application(&input_types, driver.region(0)?, driver.region(1)?)?;
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{WHILE_OPERATION_NAME}` does not support transposition (reverse-mode differentiation through staged \
                 unbounded `{WHILE_OPERATION_NAME}` loops is not supported; eager differentiation executes concrete \
                 duals, and loops built with `with_iteration_bound` stage a transposable masked scan)",
            ),
        }
        .into())
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
/// [`ArrayIrType`] conditions must produce a Boolean array member and may carry mixed array, dimension, and reference
/// state. Reference state is supported under scalar predicates; batched predicates require reference discharge first
/// because reference effects cannot be masked per item. Under a batched predicate the prefix requirement applies to the
/// array members only, since a first-class dimension carries no shape, and such a dimension carry must additionally be
/// *loop-invariant*: one dimension value cannot represent independently masked per-item extents, but masking a carry
/// that the body forwards unchanged is the identity. [`WhileOperation::validate_application`] checks that invariance
/// structurally (the body must return the carry's own input), so every path accepts the same loops, and
/// [`ArrayIrValue`]'s [`mask_select`](WhilePredicate::mask_select) then returns such equal dimension carries unchanged.
/// Structural array IR batching relies on this relaxation to thread its loop-invariant mapped extent through
/// batch-varying loops.
///
/// The loop-carried state rule is otherwise identical for every type family: the condition and body consume the same
/// state signature, and the body returns the next state with that same signature.
pub trait WhileType: Type {
    /// Validates the condition output type of a while loop against the loop-carried state types.
    ///
    /// # Parameters
    ///
    ///   - `condition_output`: The single output type produced by the condition program.
    ///   - `state_types`: The loop-carried state types the condition and body consume.
    fn validate_while_condition_output(condition_output: &Self, state_types: &[Self]) -> Result<(), TypeError>;

    /// Returns whether `condition_output` is a *batched* (per-item) predicate carrying one termination decision per
    /// leading-axes item, rather than a whole-loop scalar predicate. This is `true` for a non-scalar Boolean
    /// [`ArrayType`] predicate. It gates the purity requirement on batched-predicate loops: a batched-predicate loop
    /// keeps running for still-active items after others have finished, so it re-evaluates the condition and body over
    /// *every* item each iteration, and observable effects cannot be masked back out for the finished items the way
    /// values can (see [`WhilePredicate`]).
    fn is_batched_predicate(condition_output: &Self) -> bool;

    /// Returns whether a state element of type `state_type` has no per-item value that a batched predicate could
    /// mask (e.g., a first-class dimension), so that the body must pass it through unchanged under a batched
    /// predicate (refer to [`WhileOperation::validate_application`]). Array state is always maskable.
    fn is_loop_invariant_state(state_type: &Self) -> bool;
}

impl WhileType for ArrayType {
    #[inline]
    fn validate_while_condition_output(condition_output: &Self, state_types: &[Self]) -> Result<(), TypeError> {
        validate_while_array_predicate(condition_output, state_types)?;
        validate_while_state_variation(condition_output, state_types)
    }

    #[inline]
    fn is_batched_predicate(condition_output: &Self) -> bool {
        condition_output.rank() > 0
    }

    #[inline]
    fn is_loop_invariant_state(_state_type: &Self) -> bool {
        false
    }
}

impl WhileType for ArrayIrType {
    fn validate_while_condition_output(condition_output: &Self, state_types: &[Self]) -> Result<(), TypeError> {
        let Self::Array(predicate) = condition_output else {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got {condition_output}"
            )));
        };
        // A first-class dimension carries no shape, so the predicate-prefix requirement does not apply to it. Such a
        // carry must instead be passed through the body unchanged, which `WhileOperation::validate_application`
        // enforces structurally (refer to the documentation of `WhileType`).
        validate_while_array_predicate(
            predicate,
            state_types.iter().filter_map(|state_type| match state_type {
                Self::Array(state_type) => Some(state_type),
                Self::Dimension(_) | Self::Reference(_) => None,
            }),
        )?;
        // A reference's referent changes with every iteration that writes it, so it follows the same variation rule
        // as array state.
        validate_while_state_variation(
            predicate,
            state_types.iter().filter_map(|state_type| match state_type {
                Self::Array(state_type) => Some(state_type),
                Self::Reference(state_type) => Some(state_type.referent()),
                Self::Dimension(_) => None,
            }),
        )?;
        // Reference operations are effectful, and a batched predicate masks carries per batch item after the body ran
        // for the whole batch. Effectful state cannot be masked that way, because the items whose predicate is already
        // false would still observe the body's writes.
        if predicate.rank() > 0
            && let Some(Self::Reference(state_type)) =
                state_types.iter().find(|state_type| matches!(state_type, Self::Reference(_)))
        {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition with a batched predicate cannot carry reference state \
                 `{state_type}` because effectful state cannot be masked per batch item; discharge the reference first",
            )));
        }
        Ok(())
    }

    #[inline]
    fn is_batched_predicate(condition_output: &Self) -> bool {
        matches!(condition_output, Self::Array(r#type) if r#type.rank() > 0)
    }

    #[inline]
    fn is_loop_invariant_state(state_type: &Self) -> bool {
        matches!(state_type, Self::Dimension(_))
    }
}

/// Validates that `predicate` is a Boolean [`while`](WhileOperation) predicate for array state of types
/// `state_types`: a scalar predicate is always valid, and a batched predicate's shape must be a prefix of every array
/// state shape, so that it broadcasts against each state element along its leading axes.
fn validate_while_array_predicate<'t>(
    predicate: &ArrayType,
    state_types: impl IntoIterator<Item = &'t ArrayType>,
) -> Result<(), TypeError> {
    if !predicate.data_type().is_boolean() {
        return Err(TypeError::invalid(format!(
            "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got {predicate}"
        )));
    }
    let predicate_dimensions = predicate.shape().dimensions();
    for state_type in state_types {
        if !state_type.shape().dimensions().starts_with(predicate_dimensions) {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition predicate shape must be a prefix of every array state shape, but \
                 predicate {predicate} is not a prefix of state {state_type}",
            )));
        }
    }
    Ok(())
}

/// Validates that every array state element of a [`while`](WhileOperation) loop varies over each manual mesh axis that
/// its `predicate` varies over. Devices whose predicates differ run different numbers of iterations, so any state that
/// the body may change can differ across those devices, and its type must say so. JAX does not enforce this, which
/// leaves such state typed as identical across devices. A first-class dimension carries no manual variation and is not
/// checked.
///
/// # Errors
///
/// Returns a [`TypeError`] naming the first state type that lacks one of the predicate's varying manual axes.
fn validate_while_state_variation<'t>(
    predicate: &ArrayType,
    state_types: impl IntoIterator<Item = &'t ArrayType>,
) -> Result<(), TypeError> {
    if !predicate.is_varying() {
        return Ok(());
    }
    let predicate_axes = predicate.sharding().unwrap().varying_manual_axes();
    for state_type in state_types {
        let state_axes = state_type.sharding().map(Sharding::varying_manual_axes);
        if !predicate_axes.iter().all(|axis| state_axes.is_some_and(|state_axes| state_axes.contains(axis))) {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` state `{state_type}` must vary over every manual axis that the predicate \
                 `{predicate}` varies over, because devices may run different numbers of iterations; insert \
                 `{PARALLEL_VARY_OPERATION_NAME}` on the initial state",
            )));
        }
    }
    Ok(())
}

/// Validates the loop contract over the two attached region interfaces (`["condition", "body"]` region order) and
/// returns them. The loop-carried state signature is the body's input signature: the condition must consume the same
/// state and return exactly one Boolean predicate valid for that state under [`WhileType`], the body must return the
/// state signature, and a batched (per-item) predicate requires both regions to be pure and free of deferred work — the
/// loop keeps running for still-active items after others finish, so the condition and body re-execute over every item
/// each iteration, and neither observable effects nor transformation obligations (refer to the
/// [Deferred Work](crate::Effects#deferred-work) section of [`Effects`](crate::Effects)) can be masked back out for the
/// finished items the way values can. This mirrors JAX's `_while_loop_batching_rule`, which rejects IO effects once the
/// predicate is batched.
fn validated_while_interfaces<'i, T: WhileType>(
    region_interfaces: &'i [RegionInterface<T>],
) -> Result<(&'i RegionInterface<T>, &'i RegionInterface<T>), TypeError> {
    check_count!("region", region_interfaces, 2, TypeError);
    let condition_interface = &region_interfaces[0];
    let body_interface = &region_interfaces[1];
    let state_types = body_interface.input_types();
    check_types!(@same, format!("`{WHILE_OPERATION_NAME}` condition/body input"), [
        state_types,
        condition_interface.input_types(),
    ]);
    let condition_output_types = condition_interface.output_types();
    if condition_output_types.len() != 1 {
        return Err(TypeError::invalid(format!(
            "`{}` condition must return exactly one predicate leaf but returned {}",
            WHILE_OPERATION_NAME,
            condition_output_types.len(),
        )));
    }
    T::validate_while_condition_output(&condition_output_types[0], state_types)?;
    check_types!(@same, format!("`{WHILE_OPERATION_NAME}` body output"), [state_types, body_interface.output_types()]);
    if T::is_batched_predicate(&condition_output_types[0])
        && (!condition_interface.effects().is_empty() || !body_interface.effects().is_empty())
    {
        return Err(TypeError::invalid(format!(
            "`{WHILE_OPERATION_NAME}` loop with a batched predicate must be pure because observable effects cannot be \
             masked for finished batch items",
        )));
    }
    if T::is_batched_predicate(&condition_output_types[0])
        && (condition_interface.has_deferred_work() || body_interface.has_deferred_work())
    {
        return Err(TypeError::invalid(format!(
            "`{WHILE_OPERATION_NAME}` loop with a batched predicate cannot carry deferred work because finished batch \
             items would repeat its transformation obligation"
        )));
    }
    Ok((condition_interface, body_interface))
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
fn rebuild_while_program<C: Context>(
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
/// residual stream. For an eligible fused forward-mode loop, this duplication lets the known side recover the
/// primal outputs while the tangent program keeps the complete loop. The residual loop's known-state outputs are
/// left dead. This does not support arbitrary unbounded-loop linearization: the split requires pure regions and a
/// predicate that depends only on the closed known state.
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
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value,
    V::Type: WhileType,
    C: Context<Type = V::Type, Constant = V, Operation = O>,
    O: Operation<Type = V::Type> + From<WhileOperation<V::Type>>,
    D: PartialEvaluationDriver<C>,
{
    let state_types = body.input_types();
    let state_count = state_types.len();

    // Fixed point over state known-ness: a state element can only be demoted as more are demoted, so the loop
    // converges in at most `state_count` rounds.
    let mut state_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();

    // The residual loop re-runs every iteration and consumes no saved values, so the partitions below ignore any
    // residual policy of the context: a `while` loop saves nothing.
    let partition_context = context.clone().without_residual_policy();
    let partition = loop {
        let partition = driver.partition_program(&partition_context, body, state_known.as_slice())?;
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
    let condition_partition = driver.partition_program(&partition_context, condition, state_known.as_slice())?;
    condition_partition.validate_known_input_indices(&state_known)?;
    condition_partition.validate_canonical_residual_edges()?;
    let PartialEvaluationOutput::Known(predicate_output) = condition_partition.outputs()[0] else {
        return Ok(None);
    };
    let (condition_known_program, ..) = condition_partition.into_parts();
    partition.validate_known_input_indices(&state_known)?;
    partition.validate_canonical_residual_edges()?;
    let (known_program, _, _, _, partition_outputs) = partition.into_parts();
    let known_state_indices = (0..state_count).filter(|&index| state_known[index]).collect::<Vec<_>>();

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
                    "`{WHILE_OPERATION_NAME}` body partition output {index} references missing known-program output \
                     {output}",
                ))
            }),
            PartialEvaluationOutput::Unknown(_) => Err(ProgramError::MalformedProgram(format!(
                "`{WHILE_OPERATION_NAME}` known-ness fixed point converged with an unknown next value for a known \
                 state element",
            ))),
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
            "`{WHILE_OPERATION_NAME}` condition partition references missing known-program output {predicate_output}",
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
                    "`{WHILE_OPERATION_NAME}` known loop is missing the final value of known state element {index}",
                ))
            }),
            false => Ok(residual_output),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(outputs))
}

/// Rotates the discharged regions of a `while` loop whose condition mutates a discharged allocation into do-while form
/// and returns the rotated `(condition, body)` programs. Both take the rebuilt body's state boundary followed by one
/// predicate input. The rotated body splices the rebuilt body over the state, then splices the rebuilt condition over
/// the states the body returned, and returns those states, with the final states the condition published replacing the
/// body's at every position carrying the same allocation, followed by the fresh predicate. The rotated condition merely
/// returns its predicate input. The caller supplies the initial predicate by discharging the original condition once
/// before the loop, so the loop evaluates the condition exactly as often and in the same order as the source did.
///
/// # Parameters
///
///   - `condition`: Rebuilt condition program returning its predicate followed by the final state of every allocation
///     in `condition_published`.
///   - `body`: Rebuilt body program with a symmetric state boundary.
///   - `state_allocations`: Allocation carried at each position of the body's boundary, or [`None`] for a value.
///   - `condition_published`: Allocations whose final states the rebuilt condition publishes after its predicate.
fn rotated_discharge_regions<V: Value, O: Operation<Type = V::Type>>(
    condition: &Program<V, O, Vec<V>, Vec<V>>,
    body: &Program<V, O, Vec<V>, Vec<V>>,
    state_allocations: &[Option<ReferenceDischargeAllocationId>],
    condition_published: &[ReferenceDischargeAllocationId],
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>), ProgramError> {
    let state_types = body.input_types();
    check_count!("input", state_allocations, state_types.len(), ProgramError);
    let condition_output_types = condition.output_types();
    check_count!("output", condition_output_types, 1 + condition_published.len(), ProgramError);
    let predicate_type = condition_output_types[0].clone();

    let mut body_builder = ProgramBuilder::<V, O>::new();
    let state = state_types.iter().cloned().map(|r#type| body_builder.add_input(r#type)).collect::<Vec<_>>();
    body_builder.add_input(predicate_type.clone());
    let state = body_builder.splice_program(body, state.as_slice())?;
    check_count!("output", state, state_types.len(), ProgramError);
    let condition_outputs = body_builder.splice_program(condition, state.as_slice())?;
    let mut outputs = Vec::with_capacity(state.len() + 1);
    for (position, current) in state.iter().enumerate() {
        let published = state_allocations[position]
            .and_then(|allocation| condition_published.iter().position(|published| *published == allocation));
        outputs.push(match published {
            Some(index) => condition_outputs[1 + index],
            None => *current,
        });
    }
    outputs.push(condition_outputs[0]);
    let rotated_body = body_builder.build::<Vec<V>, Vec<V>>(
        outputs,
        vec![Placeholder; state_types.len() + 1],
        vec![Placeholder; state_types.len() + 1],
    )?;

    let mut condition_builder = ProgramBuilder::<V, O>::new();
    for r#type in state_types.iter().cloned() {
        condition_builder.add_input(r#type);
    }
    let predicate = condition_builder.add_input(predicate_type);
    let rotated_condition = condition_builder.build::<Vec<V>, Vec<V>>(
        vec![predicate],
        vec![Placeholder; state_types.len() + 1],
        vec![Placeholder],
    )?;
    Ok((rotated_condition, rotated_body))
}

/// Array-backed storage semantics used by bounded `while` differentiation.
///
/// A bounded loop stores each per-iteration residual in an ordinary leading-axis array stack. Homogeneous array
/// programs use their residual type directly. Array IR programs additionally map first-class dimension residuals to
/// checked scalar-array storage and restore them at the tangent-body boundary. Keeping this contract at the type-family
/// boundary lets both operation families share the bounded-loop algorithm without making the generic control-flow rule
/// inspect array IR variants.
pub(crate) trait WhileResidualStackType:
    DifferentiableType + TemporalResidualType + WhileType + ArrayMemberType + From<ArrayType>
{
}

impl<T: DifferentiableType + TemporalResidualType + WhileType + ArrayMemberType + From<ArrayType>>
    WhileResidualStackType for T
{
}

/// Operation-family constructors required by the shared bounded-`while` residual-stack algorithm.
///
/// The functions return operations rather than binding values, so the same contract works in the parent
/// differentiation context and in the nested tracing contexts used to build the augmented primal body and masked
/// tangent scan body. The dimension/scalar gateway operations come from the [`TemporalResidualOperation`] supertrait.
pub(crate) trait WhileResidualStackOperation<T: WhileResidualStackType>:
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

    /// Constructs the transition that adds variation over the manual mesh axis `axis_name` to a loop index or mask of
    /// type `input_type`, so that it can be combined with stacks or state that vary over that axis. Operation
    /// families that cannot stage variation transitions return an error.
    fn residual_stack_vary(axis_name: &str, input_type: &ArrayType) -> Result<Self, ProgramError>;
}

impl<O> WhileResidualStackOperation<ArrayType> for O
where
    O: Operation<Type = ArrayType>
        + From<ZeroOperation<ArrayType>>
        + From<OneOperation<ArrayType>>
        + From<AddOperation<ArrayType>>
        + From<BroadcastOperation>
        + From<DynamicUpdateSliceOperation>
        + From<SelectOperation<ArrayType>>
        + From<ReduceOperation>
        + OperationProvider<ArrayType, ParallelVaryOperation, Operation = O>,
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
        Self::from(DynamicUpdateSliceOperation::new())
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
    fn residual_stack_vary(axis_name: &str, input_type: &ArrayType) -> Result<Self, ProgramError> {
        O::provide(ParallelVaryOperation::new(axis_name.to_string()), &[input_type])
    }
}

impl<O> WhileResidualStackOperation<ArrayIrType> for O
where
    O: Operation<Type = ArrayIrType>
        + From<ZeroOperation<ArrayType>>
        + From<OneOperation<ArrayType>>
        + OperationProjection<
            ArrayType,
            Projected: From<AddOperation<ArrayType>>
                           + From<BroadcastOperation>
                           + From<DynamicUpdateSliceOperation>
                           + From<SelectOperation<ArrayType>>
                           + From<ReduceOperation>,
        > + TemporalResidualOperation<ArrayIrType>,
    O::Projected: OperationProvider<ArrayType, ParallelVaryOperation, Operation = O::Projected>,
{
    fn residual_stack_zero(r#type: ArrayIrType) -> Self {
        let ArrayIrType::Array(r#type) = r#type else { unreachable!("bounded-while stack zeros are always arrays") };
        Self::from(ZeroOperation::new(r#type))
    }

    fn residual_stack_one(r#type: ArrayIrType) -> Self {
        let ArrayIrType::Array(r#type) = r#type else { unreachable!("bounded-while stack ones are always arrays") };
        Self::from(OneOperation::new(r#type))
    }

    #[inline]
    fn residual_stack_broadcast(output_type: ArrayType, output_axes: Vec<usize>) -> Self {
        Self::from(O::Projected::from(BroadcastOperation::new(output_type, output_axes)))
    }

    #[inline]
    fn residual_stack_update() -> Self {
        Self::from(O::Projected::from(DynamicUpdateSliceOperation::new()))
    }

    #[inline]
    fn residual_stack_add() -> Self {
        Self::from(O::Projected::from(AddOperation::<ArrayType>::new()))
    }

    #[inline]
    fn residual_stack_select() -> Self {
        Self::from(O::Projected::from(SelectOperation::<ArrayType>::new()))
    }

    #[inline]
    fn mask_reduce_any(axes: Vec<usize>) -> Self {
        Self::from(O::Projected::from(ReduceOperation::new(axes, ReductionKind::Any)))
    }

    #[inline]
    fn residual_stack_vary(axis_name: &str, input_type: &ArrayType) -> Result<Self, ProgramError> {
        Ok(Self::from(O::Projected::provide(ParallelVaryOperation::new(axis_name.to_string()), &[input_type])?))
    }
}

/// Capture-free forward-mode (JVP) rule for the bounded [`WhileOperation`], staging an augmented primal `while`
/// and one masked length-`bound` tangent `scan` as ordinary primal-enum operations over the shared builder.
///
/// In the bounded regime the rule keeps every per-iteration residual and the validity mask as plain primal input edges:
/// they leave the augmented primal while as ordinary stacked outputs and re-enter the tangent scan as ordinary stacked
/// scanned inputs, so no symbolic capture is ever introduced. The enclosing partial-evaluation split then discovers the
/// residual input edges structurally, exactly as it does for the scan and condition rules.
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
///      the per-item writes can never clamp, and stages it over the input primals followed by the staged counter and
///      stack zeros. Its outputs split into the original state outputs (the primal outputs), the dropped counter, the
///      stacked residual outputs, and the mask stack.
///   2. Stages a length-`B` tangent [`ScanOperation`] over the live tangent carries, residual storage stacks, and one
///      Boolean validity stack. Each iteration reads one scalar validity item and conditionally evaluates the
///      pushforward or forwards the carried tangent unchanged. Dimension residuals that vary across iterations cross
///      the scan boundary through checked scalar-array gateways; forwarded invariant dimensions are threaded directly
///      as known carries. The condition guards both derivative arithmetic and gateway restoration, so placeholder
///      storage after early exit is never evaluated as a residual.
///   3. Pairs each primal output tracer with its tangent output tracer into a [`DifferentiationDual`].
///
/// Reverse mode is total with no while-specific transpose code: the staged tangent scan reads its residual and validity
/// stacks as known stacked inputs and its invariant residuals as known carries that its body passes through unchanged,
/// so the ordinary scan transposition flips the scan direction and transposes the body. The per-iteration `condition`
/// operation that guards each tangent step forwards cotangents through inactive iterations without evaluating their
/// placeholder residuals.
fn jvp_array_backed_while<C, D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
    operation: &WhileOperation<C::Type>,
    condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    context: &DifferentiationContext<C, P>,
    driver: &D,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    C: Context<Type: WhileResidualStackType> + Zero<C::Value>,
    C::Value: Concretizable<bool>,
    C::Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>
        + From<ConditionOperation<C::Type>>
        + From<WhileOperation<C::Type>>
        + From<ScanOperation<C::Type>>
        + WhileResidualStackOperation<C::Type>,
{
    let state_count = body.input_count();
    check_count!("input", inputs, state_count, ProgramError);

    // A loop without a live state tangent after the carry fixed point (refer to `while_tangent_activity`) is bound
    // as it is, with structural-zero tangents: no residual stack, mask, or doubled tangent state would ever be read.
    let activity = while_tangent_activity(driver, driver.region(1)?, inputs)?;
    if activity.1.input_indices().is_empty() {
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let outputs = context.primal().bind(
            C::Operation::from(*operation),
            vec![condition.clone(), body.clone()],
            primal_inputs.as_slice(),
        )?;
        check_count!("output", outputs, state_count, ProgramError);
        return outputs.into_iter().map(DifferentiationDual::new_with_zero_tangent).collect();
    }

    // An unbounded loop has no statically shaped residual stacks for the bounded hybrid rule below, so it stages
    // the fused doubled-state loop instead (see `jvp_while_fused`). The fused path needs no batched-predicate
    // rewrite: the fused state doubles every prefix-shaped state element, so a per-item predicate still satisfies
    // the relaxed predicate contract over the doubled state.
    let Some(bound) = operation.iteration_bound() else {
        return jvp_while_fused(operation, condition, context, driver, inputs, activity);
    };

    // A batched (per-item) predicate cannot thread the bounded rule's augmented differentiation state through the
    // predicate-prefix contract (the scalar iteration counter and the `[bound, ...]` residual stacks are not
    // predicate-prefixed), so the loop is first rewritten into its scalar-predicate masked normal form over
    // `[state..., active_mask]` (see `masked_while_programs`) and differentiated recursively — the masked loop's
    // forward mode is this same rule. The initial mask is the condition replayed on the input primals, carried with a
    // zero tangent since a Boolean mask has no derivative.
    if C::Type::is_batched_predicate(&condition.output_types()[0]) {
        let (masked_condition, masked_body) = masked_while_programs(condition, body)?;
        let masked_while = WhileOperation::new().with_iteration_bound(operation.iteration_bound())?;
        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let mut initial_mask = condition.interpret_in_context(context.primal(), primal_inputs)?;
        check_count!("output", initial_mask, 1, ProgramError);
        let mut extended_inputs = inputs.to_vec();
        extended_inputs.push(DifferentiationDual::new_with_zero_tangent(initial_mask.remove(0))?);
        // The masked loop's condition and body are freshly built region programs, so the recursive `jvp` is
        // requested through the instruction-scoped driver over them.
        let mut outputs = driver.bind_jvp_operation(
            context,
            &C::Operation::from(masked_while),
            vec![masked_condition, masked_body],
            extended_inputs.as_slice(),
        )?;
        check_count!("output", outputs, state_count + 1, ProgramError);
        outputs.truncate(state_count);
        return Ok(outputs);
    }

    jvp_while_bounded(condition, context, driver, inputs, bound, activity)
}

/// Shared reverse-capable forward rule for a bounded, scalar-predicate, array-backed `while` loop with at least one
/// live state tangent (`activity` is the converged tangent activity of the loop body from `while_tangent_activity`).
fn jvp_while_bounded<C, D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
    condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    context: &DifferentiationContext<C, P>,
    driver: &D,
    inputs: &[DifferentiationDual<C::Value>],
    bound: usize,
    activity: (Vec<bool>, TangentActivity<C>),
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    C: Context<Type: WhileResidualStackType> + Zero<C::Value>,
    C::Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>
        + WhileResidualStackOperation<C::Type>
        + From<ConditionOperation<C::Type>>
        + From<WhileOperation<C::Type>>
        + From<ScanOperation<C::Type>>,
{
    let state_types = driver.region(1)?.input_types();
    let state_count = state_types.len();

    // Linearize the body once under the converged tangent liveness of the state (state elements keep positional
    // identity, so the body's region input mask is the live state mask). The nonlinear half returns the next state
    // followed by ordinary residuals, and the linear half consumes the live state tangents followed by those residuals
    // and returns the tangents of the live state elements alone.
    check_count!("input", inputs, state_count, ProgramError);
    let body = driver.region(1)?;
    let (input_tangent_liveness, activity) = activity;
    let input_indices = activity.input_indices();
    let live_tangent_slots = activity.live_tangent_slots(activity.output_tangent_liveness());
    let (primal_program, tangent_program, residual_count) = driver.linearize_program(body, input_indices)?.into_parts();
    let tangent_program = tangent_program.with_outputs(&live_tangent_slots)?;
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
        Arc::new(builder.build(
            selected_outputs,
            vec![Placeholder; inputs.len()],
            vec![Placeholder; state_count + stacked_residual_indices.len()],
        )?)
    };

    // Run the primal loop over its original state plus a counter, one stack per stored residual, and a validity mask.
    let predicate_type = condition.output_types()[0].as_array_type()?.clone();
    let counter_type = C::Type::from(bounded_while_scalar_type(DataType::I64, &predicate_type)?);
    let boolean_scalar_type = C::Type::from(bounded_while_scalar_type(DataType::Boolean, &predicate_type)?);
    let mask_stack_type =
        C::Type::from(boolean_scalar_type.as_array_type()?.with_inserted_dimension(0, bound.into()).unwrap());
    let (extended_condition, augmented_body, stack_types) =
        build_bounded_while_programs(condition, &stacked_primal_program, stacked_residual_types.as_slice(), bound)?;
    let mut primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
    for zero_state_type in
        std::iter::once(&counter_type).chain(stack_types.iter()).chain(std::iter::once(&mask_stack_type))
    {
        let mut zeros =
            context.primal().bind(C::Operation::residual_stack_zero(zero_state_type.clone()), Vec::new(), &[])?;
        check_count!("output", zeros, 1, ProgramError);
        primal_inputs.push(zeros.remove(0));
    }
    let mut while_outputs = context.primal().bind(
        C::Operation::from(WhileOperation::new().with_iteration_bound(bound)?),
        vec![extended_condition, augmented_body],
        primal_inputs.as_slice(),
    )?;
    check_count!("output", while_outputs, state_count + 2 + stack_types.len(), ProgramError);
    let mask_stack = while_outputs.pop().unwrap();
    let residual_stacks = while_outputs.split_off(state_count + 1);
    while_outputs.truncate(state_count);
    let primal_outputs = while_outputs;

    let tangent_state_count = input_indices.len();

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
    // Conditional transposition requires linear tangent inputs before known residuals. The surrounding scan keeps
    // invariant residual carries first, so its branch boundary reorders only the inputs consumed by the condition.
    let branch_input_types = tangent_input_types[..tangent_state_count]
        .iter()
        .cloned()
        .chain(iteration_input_types[..invariant_residual_count].iter().cloned())
        .chain(storage_types.iter().cloned())
        .collect::<Vec<_>>();
    let active_body = TracingContext::<C::Constant, C::Operation>::trace(
        |inputs| -> Result<Vec<_>, ProgramError> {
            let trace_context = inputs[0].context().clone();
            let residual_start = tangent_state_count + invariant_residual_count;
            let invariant_inputs = &inputs[tangent_state_count..residual_start];
            let carried_tangents = &inputs[..tangent_state_count];
            let mut tangent_inputs = carried_tangents.to_vec();
            let mut restored_residuals: Vec<(C::Type, Tracer<TracingContext<C::Constant, C::Operation>>)> = Vec::new();
            let storage_inputs = &inputs[residual_start..residual_start + stacked_residual_indices.len()];
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
                        "bounded `{WHILE_OPERATION_NAME}` residual stack item has type `{}` but expected `{}`",
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
            Ok(pushforward_outputs)
        },
        branch_input_types.clone(),
    )?
    .1;
    let mut scan_body_input_types = vec![C::Type::from(ArrayType::scalar(DataType::I64))];
    scan_body_input_types.extend(iteration_input_types.iter().cloned());
    scan_body_input_types.push(boolean_scalar_type);
    let scan_body = {
        // Inactive iterations contain zero-filled placeholder residuals. Guard the whole pushforward so neither
        // forward execution nor transposition can evaluate singular derivatives or dimension gateways on them.
        let mut passive_builder = ProgramBuilder::new();
        let passive_inputs = branch_input_types
            .iter()
            .map(|r#type| passive_builder.add_input(r#type.clone()))
            .collect::<Vec<_>>();
        let passive_body = passive_builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            passive_inputs[..tangent_state_count].to_vec(),
            vec![Placeholder; passive_inputs.len()],
            vec![Placeholder; tangent_state_count],
        )?;
        TracingContext::<C::Constant, C::Operation>::trace(
            |inputs| {
                let trace_context = inputs[0].context().clone();
                let (iteration_inputs, mask) = inputs[1..].split_at(inputs.len() - 2);
                let mut conditional_inputs = Vec::with_capacity(inputs.len());
                conditional_inputs.push(mask[0].clone());
                conditional_inputs.extend_from_slice(
                    &iteration_inputs[invariant_residual_count..invariant_residual_count + tangent_state_count],
                );
                conditional_inputs.extend_from_slice(&iteration_inputs[..invariant_residual_count]);
                conditional_inputs
                    .extend_from_slice(&iteration_inputs[invariant_residual_count + tangent_state_count..]);
                let conditional_outputs = trace_context.bind(
                    C::Operation::from(ConditionOperation::new()),
                    vec![active_body, passive_body],
                    conditional_inputs.as_slice(),
                )?;
                // Invariant residuals remain structurally passed through at the scan boundary, which lets scan
                // transposition treat them as invariant known carries.
                let mut outputs = iteration_inputs[..invariant_residual_count].to_vec();
                outputs.extend(conditional_outputs);
                Ok::<_, ProgramError>(outputs)
            },
            scan_body_input_types,
        )?
        .1
    };

    // The tangent scan consumes invariant dimension residuals, live tangent carries, stored residual stacks, and one
    // scalar validity stack.
    let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
    let mut tangent_scan_inputs = invariant_residual_sources
        .iter()
        .flatten()
        .map(|&state_index| tangent_inputs[state_index].primal().clone())
        .collect::<Vec<_>>();
    // Each state element's own primal names every runtime quantity a reference-bearing tangent type omits, because the
    // tangent type derivation preserves shapes exactly.
    for (input, &active) in inputs.iter().zip(&input_tangent_liveness) {
        if active {
            tangent_scan_inputs.push(context.materialize_dual_tangent(input)?);
        }
    }
    tangent_scan_inputs.extend(
        residual_stacks
            .into_iter()
            .map(|value| context.primal_to_tangent(value))
            .collect::<Result<Vec<_>, _>>()?,
    );
    tangent_scan_inputs.push(context.primal_to_tangent(mask_stack)?);
    let tangent_scan = ScanOperation::<C::Type>::new(invariant_residual_count + tangent_state_count, bound);
    let tangent_outputs =
        context
            .tangent()
            .bind(C::Operation::from(tangent_scan), vec![scan_body], tangent_scan_inputs.as_slice())?;
    check_count!("output", tangent_outputs, invariant_residual_count + tangent_state_count, ProgramError);
    let tangent_outputs = tangent_outputs.into_iter().skip(invariant_residual_count).collect();
    DifferentiationDual::from_live_tangents(primal_outputs, tangent_outputs, &input_tangent_liveness)
}

/// Returns the forward-mode activity of the `body` of a `while` loop whose state enters as the instruction input duals
/// `inputs`, together with whether each state element carries a live tangent after the carry fixed point (refer to
/// [`TangentActivity::from_loop_body`]). The state elements keep their positions across the body boundary, so every
/// state element is a carry and the body's region input mask is the live state mask. A state element starts live when
/// its tangent is live (structural zeros are not).
fn while_tangent_activity<C: Context<Type: DifferentiableType>, D: DifferentiationDriver<C>>(
    driver: &D,
    body: RegionRef<'_, C::Constant, C::Operation>,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<(Vec<bool>, TangentActivity<C>), DifferentiationError> {
    let state_count = inputs.len();
    let input_tangent_liveness = inputs.iter().map(|input| !input.tangent().is_zero()).collect();
    TangentActivity::from_loop_body(driver, body, input_tangent_liveness, 0..state_count, 0..state_count)
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
/// Shared construction binds the fused loop directly. Separate construction partitions the generated loop while
/// requiring every primal state output to remain known; loops that require an unbounded history are rejected.
/// `activity` is the converged tangent activity of the loop body from `while_tangent_activity`.
fn jvp_while_fused<C, D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
    operation: &WhileOperation<C::Type>,
    condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    context: &DifferentiationContext<C, P>,
    driver: &D,
    inputs: &[DifferentiationDual<C::Value>],
    activity: (Vec<bool>, TangentActivity<C>),
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    C: Context + Zero<C::Value>,
    C::Type: DifferentiableType + WhileType,
    C::Operation: ResidualZeroProvider<C::Type, Operation = C::Operation> + From<WhileOperation<C::Type>>,
{
    let state_count = inputs.len();

    // Build the fused body over the compact state `[primal_state..., live(tangent_state)...]` through the
    // instruction-scoped driver (region 1 is the loop body). The fused body carries a tangent exactly for the state
    // elements whose tangent is live after the carry fixed point (refer to `while_tangent_activity`), so inactive
    // elements, such as a pass-through loop limit, keep structural-zero tangents instead of threading materialized
    // zeros through the loop.
    let body = driver.region(1)?;
    check_count!("input", inputs, body.input_types().len(), ProgramError);
    let (input_tangent_liveness, activity) = activity;
    let live_tangent_slots = activity.live_tangent_slots(activity.output_tangent_liveness());
    let tangent_state_count = activity.input_indices().len();
    let fused_state_count = state_count + tangent_state_count;
    // The body is differentiated through its region's retained transform cache, so a body shared by several programs
    // is differentiated once and repeated attachments of the result intern by `Arc` identity. Its tangent outputs are
    // projected to the live state elements only when some state element is inactive.
    let fused_body = activity.jvp_program().clone();
    let fused_outputs = (0..state_count)
        .chain(live_tangent_slots.iter().map(|&slot| state_count + slot))
        .collect::<Vec<_>>();
    let fused_body = if fused_outputs.iter().copied().eq(0..fused_body.output_count()) {
        fused_body
    } else {
        Arc::new(fused_body.with_outputs(&fused_outputs)?)
    };
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

    // Stage the fused loop over the instruction input primals followed by the materialized tangents of the live
    // differential state elements — the fused body takes each of those tangents as a real program input, so their
    // structural zeros are materialized against their own primal, which supplies the runtime extents a
    // reference-bearing tangent type omits — and zip the output halves back into `DifferentiationDual`s in the original
    // state order, restoring structural zeros for the omitted zero-space state elements.
    let fused_while = WhileOperation::new().with_iteration_bound(operation.iteration_bound())?;
    let mut fused_inputs = Vec::with_capacity(fused_state_count);
    fused_inputs.extend(inputs.iter().map(|input| input.primal().clone()));
    for (input, &has_tangent) in inputs.iter().zip(&input_tangent_liveness) {
        if has_tangent {
            fused_inputs.push(context.materialize_dual_tangent(input)?);
        }
    }
    let fused_regions = [Arc::new(fused_condition), fused_body];
    let outputs = if std::ptr::eq(context.primal(), context.tangent()) {
        context.primal().bind(
            C::Operation::from(fused_while),
            CalleeRegionDriver::new(&fused_regions),
            &fused_inputs,
        )?
    } else {
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        let arguments =
            fused_inputs.iter().map(|value| builder.add_input(value.r#type().into_owned())).collect::<Vec<_>>();
        let regions = fused_regions.iter().map(|program| builder.import_region(program.entry_region_ref())).collect();
        let outputs = builder.add_instruction(C::Operation::from(fused_while), regions, arguments, None)?.to_vec();
        let program = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            outputs,
            vec![Placeholder; fused_state_count],
            vec![Placeholder; fused_state_count],
        )?;
        let input_known = (0..fused_state_count).map(|index| index < state_count).collect::<Vec<_>>();
        let required_outputs = (0..state_count).collect::<Vec<_>>();
        let partition = driver.partition_jvp_program(program.entry_region_ref(), &input_known, &required_outputs)?;
        partition.interpret_in_context(context, &fused_inputs, state_count)?
    };
    check_count!("output", outputs, fused_state_count, ProgramError);
    let mut primal_outputs = outputs;
    let tangent_outputs = primal_outputs.split_off(state_count);
    DifferentiationDual::from_live_tangents(primal_outputs, tangent_outputs, &input_tangent_liveness)
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
fn jvp_while_eagerly<C, D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
    operation: &WhileOperation<C::Type>,
    condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
    context: &DifferentiationContext<C, P>,
    driver: &D,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Option<Vec<DifferentiationDual<C::Value>>>, ProgramError>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    C: Context + Zero<C::Value>,
    C::Type: DifferentiableType + WhileType,
    C::Value: Concretizable<bool>,
    C::Operation: OperationProvider<C::Type, ZeroOperation<C::Type>, Operation = C::Operation>,
    for<'o> &'o WhileOperation<C::Type>: TryFrom<&'o C::Operation>,
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
        let mut condition_outputs = condition.interpret_in_context(context.primal(), primal_carries.clone())?;
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
            |_, constant| Ok(DifferentiationDual::new_with_zero_tangent(context.primal().lift(constant.clone())?)?),
            |instruction, input_duals| {
                let programs = instruction
                    .regions()
                    .iter()
                    .map(|region| body_region.with_id(*region).map(RegionRef::to_program))
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                let output_duals =
                    driver.bind_jvp_operation(context, instruction.operation(), programs, input_duals)?;
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

/// Returns the scalar type of a loop counter (for `DataType::I64`) or of a validity-mask item (for `DataType::Boolean`)
/// in a bounded `while` program whose predicate has type `predicate_type`. Devices whose predicates differ run
/// different numbers of iterations, so these values vary over every manual mesh axis that the predicate varies over.
/// They are non-differentiable integers and Booleans, so they are created with that variation directly instead of
/// through `parallel_vary` transitions. Outside manual regions, the scalar type carries no sharding.
fn bounded_while_scalar_type(data_type: DataType, predicate_type: &ArrayType) -> Result<ArrayType, TypeError> {
    let scalar_type = ArrayType::scalar(data_type);
    if !predicate_type.is_varying() {
        return Ok(scalar_type);
    }
    let sharding = predicate_type.sharding().unwrap();
    let scalar_sharding = Sharding::replicated(sharding.mesh().clone(), 0)
        .with_varying_manual_axes(sharding.varying_manual_axes().iter().cloned())
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    scalar_type.with_sharding(scalar_sharding).map_err(|error| TypeError::invalid(error.to_string()))
}

/// Stages the transitions that make `value`, a loop index or Boolean mask inside a bounded or masked `while` program
/// that carries no tangent, vary over every manual mesh axis that `target` varies over, so that it can be combined with
/// stacks or state of type `target`. A value without a sharding is first placed, replicated, on the mesh of `target`.
/// The value is returned unchanged when it already varies over those axes, which keeps programs outside manual regions
/// unchanged. The transitions are bound as raw operations because the tracing contexts that build these programs bind
/// no named axes, and they are never transposed because their inputs are integer or Boolean values.
fn vary_while_index_or_mask<V, O>(
    context: &TracingContext<V, O>,
    value: Tracer<TracingContext<V, O>>,
    target: &ArrayType,
) -> Result<Tracer<TracingContext<V, O>>, ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: WhileResidualStackType>,
    O: WhileResidualStackOperation<V::Type>,
{
    let Some(target_sharding) = target.sharding() else {
        return Ok(value);
    };
    let value_type = value.r#type().as_array_type()?.clone();
    let missing_axes = target_sharding
        .varying_manual_axes()
        .iter()
        .filter(|axis| !value_type.sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(*axis)))
        .collect::<Vec<_>>();
    if missing_axes.is_empty() {
        return Ok(value);
    }
    let mut value = value;
    if value_type.sharding().is_none() {
        let rank = value_type.rank();
        let placed_type = value_type
            .with_sharding(Sharding::replicated(target_sharding.mesh().clone(), rank))
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let mut outputs = context.bind(
            O::residual_stack_broadcast(placed_type, (0..rank).collect()),
            Vec::new(),
            std::slice::from_ref(&value),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        value = outputs.remove(0);
    }
    for axis in missing_axes {
        let operation = O::residual_stack_vary(axis, value.r#type().as_array_type()?)?;
        let mut outputs = context.bind(operation, Vec::new(), std::slice::from_ref(&value))?;
        check_count!("output", outputs, 1, ProgramError);
        value = outputs.remove(0);
    }
    Ok(value)
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
fn build_bounded_while_programs<V, O>(
    condition: &Program<V, O, Vec<V>, Vec<V>>,
    primal_body: &Program<V, O, Vec<V>, Vec<V>>,
    residual_types: &[V::Type],
    bound: usize,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>, Vec<V::Type>), ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: WhileResidualStackType>,
    O: WhileResidualStackOperation<V::Type>,
{
    let state_count = condition.input_types().len();
    let predicate_type = condition.output_types()[0].as_array_type()?.clone();
    let counter_type = V::Type::from(bounded_while_scalar_type(DataType::I64, &predicate_type)?);
    let boolean_scalar_type = V::Type::from(bounded_while_scalar_type(DataType::Boolean, &predicate_type)?);
    let mask_stack_type =
        V::Type::from(boolean_scalar_type.as_array_type()?.with_inserted_dimension(0, bound.into()).unwrap());
    let storage_types = residual_types
        .iter()
        .map(TemporalResidualType::temporal_storage_type)
        .collect::<Result<Vec<_>, _>>()?;
    let predicate_axes = predicate_type.sharding().map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
    for storage_type in &storage_types {
        let storage_type = storage_type.as_array_type()?;
        if storage_type.static_shape().is_none() {
            return Err(TypeError::invalid(format!(
                "`jvp` of a bounded `{WHILE_OPERATION_NAME}` loop requires statically shaped residual storage but got \
                 `{storage_type}`",
            ))
            .into());
        }

        // Devices whose predicates differ fill different numbers of stack slots, so every stack must vary over the
        // predicate's manual axes. A residual that depends on loop state already does, because all array state varies
        // over those axes. A residual that does not cannot be retyped without changing the tangent program that
        // consumes it, so it is rejected instead of being mistyped.
        if !predicate_axes
            .iter()
            .all(|axis| storage_type.sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(axis)))
        {
            return Err(TypeError::invalid(format!(
                "`jvp` of a bounded `{WHILE_OPERATION_NAME}` loop requires every residual to vary over the manual axes \
                 that the predicate `{predicate_type}` varies over, but got residual storage `{storage_type}`",
            ))
            .into());
        }
    }
    let stack_types = storage_types
        .iter()
        .map(|storage_type| {
            storage_type
                .as_array_type()
                .map(|storage_type| V::Type::from(storage_type.with_inserted_dimension(0, bound.into()).unwrap()))
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
                    "bounded `{WHILE_OPERATION_NAME}` body adapter is missing the counter input"
                ))
            })?;
            let stack_inputs = extra_inputs.by_ref().take(stack_types.len()).cloned().collect::<Vec<_>>();
            check_count!("input", stack_inputs, stack_types.len(), ProgramError);
            let mask_input = extra_inputs.next().cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "bounded `{WHILE_OPERATION_NAME}` body adapter is missing the mask input"
                ))
            })?;
            let mut body_outputs =
                primal_body.interpret_in_context(&context, inputs[..original_input_count].to_vec())?;
            let residual_outputs = body_outputs.split_off(state_count);
            check_count!("output", residual_outputs, residual_types.len(), ProgramError);
            // Integer indices carry no tangent, but `dynamic_update_slice` requires them to vary over the same manual
            // mesh axes as the stack that they index. Invariant stacks use the counter and one shared zero index. For
            // each distinct stack variation, the counter, which is loop state, takes that variation through staged
            // transitions, while the zero index is a non-differentiable constant and is therefore created directly
            // with the variation of that varied counter.
            let manual_variation = |storage_type: &ArrayType| {
                storage_type.sharding().map(Sharding::varying_manual_axes).cloned().unwrap_or_default()
            };
            let zero_index = if storage_types.iter().any(|storage_type| {
                storage_type
                    .as_array_type()
                    .is_ok_and(|storage_type| storage_type.rank() > 0 && manual_variation(storage_type).is_empty())
            }) {
                let mut outputs = context.bind(O::residual_stack_zero(counter_type.clone()), Vec::new(), &[])?;
                check_count!("output", outputs, 1, ProgramError);
                Some(outputs.remove(0))
            } else {
                None
            };
            let mut varying_indices = Vec::<(BTreeSet<String>, Tracer<TracingContext<V, O>>, Option<_>)>::new();
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
                let storage_array_type = storage_type.as_array_type()?;
                let batch_item_type = storage_array_type.with_inserted_dimension(0, 1.into()).unwrap();
                let output_axes = (1..=storage_array_type.rank()).collect::<Vec<_>>();
                let mut expanded = context.bind(
                    O::residual_stack_broadcast(batch_item_type, output_axes),
                    Vec::new(),
                    std::slice::from_ref(&storage_output),
                )?;
                check_count!("output", expanded, 1, ProgramError);
                let variation = manual_variation(storage_array_type);
                let (counter_index, zero_index) = if variation.is_empty() {
                    (counter_input.clone(), zero_index.clone())
                } else {
                    let position = match varying_indices.iter().position(|(axes, ..)| *axes == variation) {
                        Some(position) => position,
                        None => {
                            let counter =
                                vary_while_index_or_mask(&context, counter_input.clone(), storage_array_type)?;
                            varying_indices.push((variation, counter, None));
                            varying_indices.len() - 1
                        }
                    };
                    let (_, counter, zero) = &mut varying_indices[position];
                    if zero.is_none() && storage_array_type.rank() > 0 {
                        let mut outputs =
                            context.bind(O::residual_stack_zero(counter.r#type().into_owned()), Vec::new(), &[])?;
                        check_count!("output", outputs, 1, ProgramError);
                        *zero = Some(outputs.remove(0));
                    }
                    (counter.clone(), zero.clone())
                };
                let mut start_indices = vec![counter_index];
                if let Some(zero_index) = zero_index {
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
            let true_item_type = boolean_scalar_type.as_array_type()?.with_inserted_dimension(0, 1.into()).unwrap();
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
/// `[state..., active_mask]`: the masked condition reduces the mask with a Boolean `any` over every predicate axis, and
/// the masked body replays the original body for candidate updates, selects per array state element between the
/// candidate and the carried state under the (broadcast) mask, and recomputes the per-item predicate on the new state
/// as the next mask. A first-class dimension carry is loop-invariant under a batched predicate, so it carries no
/// selection and its candidate is forwarded directly (refer to the documentation of [`WhileType`]). The bounded while
/// forward-mode rule uses this normal form for batched-predicate loops, whose counter- and stack-augmented
/// differentiation state is not predicate-prefixed and therefore needs the loop's masking made explicit as program data
/// hanging off a scalar predicate.
fn masked_while_programs<V, O>(
    condition: &Program<V, O, Vec<V>, Vec<V>>,
    body: &Program<V, O, Vec<V>, Vec<V>>,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>), ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    V: Value<Type: WhileResidualStackType>,
    O: WhileResidualStackOperation<V::Type>,
{
    let state_types = body.input_types();
    let state_count = state_types.len();
    let mask_type = condition.output_types()[0].clone();
    let mask_axes = (0..mask_type.as_array_type()?.rank()).collect::<Vec<_>>();
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
    // update and the carried state, and the per-item predicate recomputed on the new state as the next mask.
    let (_, masked_body) = TracingContext::<V, O>::trace(
        |inputs| {
            let (mask, state) = inputs.split_last().unwrap();
            let trace_context = mask.context().clone();
            let candidates = body.interpret_in_context(&trace_context, state.to_vec())?;
            check_count!("output", candidates, state_count, ProgramError);
            let mut next_state = Vec::with_capacity(state_count);
            for ((candidate, carried), state_type) in candidates.iter().zip(state).zip(state_types.iter()) {
                // A first-class dimension carry is loop-invariant under a batched predicate (the contract documented on
                // `WhileType`), so masking it is the identity and the body's candidate result is threaded on directly.
                // This matches eager interpretation, whose `mask_select` returns equal dimension carries unchanged, and
                // the XLA lowering's batched-predicate masking.
                let Some(state_array_type) = state_type.as_array_type().ok() else {
                    next_state.push(candidate.clone());
                    continue;
                };
                // The Boolean mask carries no tangent, so it first takes the manual variation of the state element
                // that it selects through staged transitions. It then broadcasts to the state element's shape with its
                // own sharding extended over the trailing axes, so that the selection is per predicate item. A state
                // element already shaped like the mask reuses it directly.
                let varied_mask = vary_while_index_or_mask(&trace_context, mask.clone(), state_array_type)?;
                let varied_mask_type = varied_mask.r#type().as_array_type()?.clone();
                let element_mask_sharding = varied_mask_type
                    .sharding()
                    .map(|sharding| sharding.with_broadcasted_dimensions(state_array_type.rank(), &mask_axes))
                    .transpose()
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                let element_mask_type = ArrayType::new(DataType::Boolean, state_array_type.shape().clone())
                    .with_sharding(element_mask_sharding)
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                let element_mask = if element_mask_type == varied_mask_type {
                    varied_mask
                } else {
                    trace_context
                        .bind(
                            O::residual_stack_broadcast(element_mask_type, mask_axes.clone()),
                            Vec::new(),
                            std::slice::from_ref(&varied_mask),
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
            // The next mask is the predicate of the next state, exactly as interpretation, the lowering, and the fused
            // forward-mode rule recompute it: an item whose predicate reads its own state alone stays finished, because
            // its state is frozen, while an item whose predicate reads other items may rejoin on every path alike.
            let next_predicate = condition.interpret_in_context(&trace_context, next_state.clone())?;
            check_count!("output", next_predicate, 1, ProgramError);
            let mut outputs = next_state;
            outputs.extend(next_predicate);
            Ok(outputs)
        },
        masked_state_types,
    )?;
    Ok((masked_condition, masked_body))
}

/// Value-level predicate capability backing [`WhileOperation`]'s masked loop semantics.
///
/// A while condition may produce a *batched* Boolean predicate — one termination decision per leading-axes item, with
/// the predicate shape a prefix of every state shape (see [`WhileType`]). Interpretation then continues while
/// [`any_true`](Self::any_true) holds and updates each state element with [`mask_select`](Self::mask_select), so items
/// whose predicate is false keep their carried state while active items take the body's candidate update. The
/// predicate is recomputed from the whole state on every iteration, so a finished item whose predicate reads its own
/// state alone can never rejoin the loop (its state is frozen), whereas an item whose predicate also reads other items
/// rejoins whenever that predicate turns true again; every path (interpretation, the masked form of differentiation,
/// and the lowering) decides this the same way.
///
/// Per-item masking selects values after evaluating the body for every item; it does not skip inactive items' body
/// computations. The body must therefore remain defined for their frozen state too. During differentiation, undefined
/// derivatives in those computations can produce non-finite cotangents even when their selected output is inactive.
///
/// The default implementations are the scalar-predicate semantics, expressed through [`Concretizable<bool>`]: the
/// predicate's own truth decides continuation, and a true predicate takes the candidate wholesale. Value types with
/// genuinely batched payloads (e.g. [`Array`]) override both functions with per-item semantics, and symbolic values
/// (tracers and capture references) inherit the defaults, which surface [`Concretizable::concretize`]'s concretization
/// errors — a staged while is consumed by staging and lowering rather than by this eager loop.
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

// Batched while-predicate semantics for [`Array`]: `any_true` reduces the whole Boolean payload with `or`, and
// `mask_select` broadcasts the predicate against its other inputs along its leading (prefix) axes, so predicate item
// `i` masks the contiguous per-item block of `on_true` / `on_false` elements it governs.
impl WhilePredicate for Array {
    fn any_true(&self) -> Result<bool, ProgramError> {
        if !self.r#type().data_type().is_boolean() {
            return Err(ProgramError::Concretization {
                message: format!("cannot use a value of type `{}` as a Boolean `while` predicate", self.r#type()),
            });
        }
        let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        Ok((0..addressing.element_count())
            .any(|index| self.storage_bytes()[addressing.byte_range_for_flat_index(index).start] != 0))
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let predicate_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let true_addressing = ArrayAddressing::new(on_true.r#type().into_owned())?;
        if on_true.r#type() != on_false.r#type() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`mask_select` requires congruent inputs, but got `{}` and `{}`",
                    on_true.r#type(),
                    on_false.r#type(),
                ),
            });
        }
        validate_while_array_predicate(self.r#type().as_ref(), [on_true.r#type().as_ref()])?;
        // Both candidates share the state type, which the selection keeps. A scalar predicate selects one candidate
        // wholesale, and the `while` interpreter only calls this function once that predicate is true, so its manual
        // variation never mixes candidate items and does not constrain the state type. A batched predicate selects per
        // item, so the state type must already vary over every manual axis that the predicate varies over. Over an axis
        // that it is invariant over, a batched predicate selects the same items on every device, so it can select
        // between candidates that vary over that axis.
        if self.r#type().rank() > 0 {
            let predicate_axes =
                self.r#type().sharding().map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
            let state_axes =
                on_true.r#type().sharding().map(Sharding::varying_manual_axes).cloned().unwrap_or_default();
            if !predicate_axes.is_subset(&state_axes) {
                return Err(TypeError::invalid(format!(
                    "batched `while` predicate `{}` varies over manual axes that state `{}` does not vary over",
                    self.r#type(),
                    on_true.r#type(),
                ))
                .into());
            }
        }
        if predicate_addressing.element_count() == 0 {
            return Ok(on_false.clone());
        }
        let block = true_addressing.element_count() / predicate_addressing.element_count();
        let mut output_bytes = vec![0; true_addressing.storage_byte_len()];
        for index in 0..true_addressing.element_count() {
            let predicate_range = predicate_addressing.byte_range_for_flat_index(index / block);
            let source = if self.storage_bytes()[predicate_range.start] != 0 {
                on_true.storage_bytes()
            } else {
                on_false.storage_bytes()
            };
            let source_range = true_addressing.byte_range_for_flat_index(index);
            output_bytes[source_range.clone()].copy_from_slice(&source[source_range]);
        }
        Ok(Self::new_unchecked(on_true.r#type().into_owned(), Arc::new(output_bytes)))
    }
}

// Array IR predicates delegate to their array payload. Under a batched predicate, equal dimension and reference carries
// pass through unchanged; the eager `while` interpreter selects distinct reference carries wholesale under a scalar
// predicate and rejects a batched predicate over them, since effectful state has no per-item value to mask.
impl<A: Value<Type = ArrayType> + WhilePredicate> WhilePredicate for ArrayIrValue<A> {
    fn any_true(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Array(predicate) => predicate.any_true(),
            Self::Dimension(value) => Err(ProgramError::Concretization {
                message: format!("cannot use first-class dimension `{value}` as a `while` predicate"),
            }),
            Self::Reference(value) => Err(ProgramError::Concretization {
                message: format!("cannot use reference `{value}` as a `while` predicate"),
            }),
        }
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let predicate = match self {
            Self::Array(predicate) => predicate,
            Self::Dimension(value) => {
                return Err(ProgramError::Concretization {
                    message: format!("cannot use first-class dimension `{value}` as a `while` predicate"),
                });
            }
            Self::Reference(value) => {
                return Err(ProgramError::Concretization {
                    message: format!("cannot use reference `{value}` as a `while` predicate"),
                });
            }
        };
        match (on_true, on_false) {
            (Self::Array(on_true), Self::Array(on_false)) => Ok(Self::Array(predicate.mask_select(on_true, on_false)?)),
            (Self::Dimension(on_true), Self::Dimension(on_false)) => {
                // Selecting between equal dimension carries (e.g., the loop-invariant mapped extent that structural
                // batching threads through a while loop's state) is the identity for every item, so it needs no
                // scalar predicate. Distinct dimension carries fall back to the scalar-predicate semantics because
                // one dimension value cannot represent independently masked per-item extents.
                if on_true == on_false {
                    return Ok(Self::Dimension(on_true.clone()));
                }
                Ok(Self::Dimension(if predicate.concretize()? { on_true.clone() } else { on_false.clone() }))
            }
            (Self::Reference(on_true), Self::Reference(on_false)) => {
                // A reference carry has one state per allocation rather than one per batch item, so it is never
                // masked: identical carries (the common loop-invariant case) need no selection, a scalar predicate
                // selects one carry wholesale, and a batched predicate over distinct carries has no per-item meaning.
                if on_true == on_false {
                    return Ok(Self::Reference(on_true.clone()));
                }
                if predicate.r#type().rank() != 0 {
                    return Err(ProgramError::UnsupportedOperation {
                        message: "a batched `while` predicate cannot select between distinct reference carries per \
                                  batch item; reference state has one value per allocation"
                            .to_string(),
                    });
                }
                Ok(Self::Reference(if predicate.concretize()? { on_true.clone() } else { on_false.clone() }))
            }
            _ => Err(TypeError::invalid(format!(
                "`while` predicate cannot select between mismatched state types `{}` and `{}`",
                on_true.r#type().as_ref(),
                on_false.r#type().as_ref(),
            ))
            .into()),
        }
    }
}

// Symbolic values inherit the scalar `WhilePredicate` defaults, which surface `Concretizable::concretize`'s
// concretization errors: none of these carry a concrete predicate payload, and staged whiles are consumed by staging
// and lowering rather than by the eager masked loop.
impl<C: Context> WhilePredicate for Tracer<C> {}

impl WhilePredicate for CaptureReference<ArrayType> {}

impl<C: Context<Type = ArrayType>> WhilePredicate for BatchingTracer<C, ArrayBatchingPolicy> where
    C::Value: Concretizable<bool>
{
}

impl<C, P> WhilePredicate for DifferentiationTracer<C, P>
where
    C: Context<Type: DifferentiableType>,
    C::Value: Concretizable<bool>,
    P: DifferentiationPolicy<C>,
{
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashSet;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayElement, ArrayIrOperation, ArrayIrValue, ArrayOperation, ArrayReference, Dimension,
        DimensionBounds, DimensionError, DimensionType, DimensionValue, DimensionVariable, Layout, LogicalMesh,
        MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension, StridedLayout,
    };
    use crate::axes::{NamedAxes, NamedAxis};
    use crate::batching::{Batch, batch};
    use crate::captures::ClosedProgram;
    use crate::contexts::{Domain, EagerContext, StagingContext, ValueResolution};
    use crate::differentiation::{
        Differentiate, ForwardModeDifferentiate, LinearizationTracer, ReverseModeDifferentiate, differentiate_at,
    };
    use crate::kernels::{KernelOperation, NoKernelExtension};
    use crate::macros::{check_gradient, check_operation_batching};
    use crate::operations::arithmetic::{AddOperation, DivOperation, MulOperation, NegOperation, SubOperation};
    use crate::operations::collectives::parallel_reduce::ParallelReduceOperation;
    use crate::operations::comparisons::{CompareOperation, ComparisonDirection};
    use crate::operations::composites::operations::CompositeTransposeOperation;
    use crate::operations::constants::constant::Constant;
    use crate::operations::constants::fill::Fill;
    use crate::operations::constants::iota::Iota;
    use crate::operations::constants::one::One;
    use crate::operations::constants::one_like::OneLikeOperation;
    use crate::operations::constants::zero::Zero;
    use crate::operations::constants::zero_like::ZeroLikeOperation;
    use crate::operations::control_flow::tests::{
        CountingBatchingDriver, DischargeCapture, DischargeCaptureOperation, TestIrOperation, TestIrValue,
        TestRegionLiveness, array, dimension,
    };
    use crate::operations::debugging::PrintOperation;
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::logical::{AndOperation, OrOperation};
    use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcastOperation};
    use crate::operations::manipulation::reshaping::DynamicReshapeOperation;
    use crate::operations::manipulation::reversing::ReverseOperation;
    use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
    use crate::operations::references::{
        ReferenceAddUpdateOperation, ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation,
        ReferenceRead, ReferenceReadOperation, ReferenceWriteOperation,
    };
    use crate::operations::sharding::shard_map::{ShardMap, ShardMapOperation};
    use crate::operations::trigonometric::SinOperation;
    use crate::partial::{PartialTracer, ResidualInputSource};
    use crate::programs::{
        BindingRegionDriver, EffectClasses, InstructionId, Provenance, ProvenanceScope, ReferenceAnalysisError,
        ReferenceRoot, ReferenceSource, ReferenceType, RegionDataFlowRegionBoundary,
    };
    use crate::tracing::{DomainTracingContext, Tracer, TracingContext};

    use super::*;

    /// Builds a `while` condition and body over an `f64[extent]` state whose condition is always false and whose body
    /// returns a constant `f64[3]`. The body maps its state to itself only when `extent` is 3, so its declared
    /// signature is invalid even though an `f64[3]` input refines it to a valid one.
    fn refinement_only_while_regions() -> [Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>; 2] {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(state_type.clone());
        let predicate = builder.add_constant(Array::scalar(false).unwrap());
        let condition = builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(state_type);
        let output = builder.add_constant(Array::vector(vec![2.0f64; 3]).unwrap());
        let body = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        [condition, body]
    }

    /// Builds a condition program that maps a scalar `f64` state to the scalar Boolean predicate `state > 0`.
    fn greater_than_zero_condition() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let predicate = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![state, zero],
                None,
            )
            .unwrap()[0];
        builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a body program that maps a scalar `f64` state to `state - 1`.
    fn subtract_one_body() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let next_state = builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
        builder.build(vec![next_state], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a loop over `[counter, first, second, source]` whose body returns
    /// `[counter - 1, second, source, source]` while the counter is positive, under the optional `iteration_bound`.
    fn chained_carry_while_program(
        iteration_bound: Option<usize>,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let condition_inputs = (0..4).map(|_| condition_builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
        let zero = condition_builder.add_constant(Array::scalar(0.0).unwrap());
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![condition_inputs[0], zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body_inputs = (0..4).map(|_| body_builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
        let one = body_builder.add_constant(Array::scalar(1.0).unwrap());
        let next_counter = body_builder
            .add_instruction(SubOperation::new(), Vec::new(), vec![body_inputs[0], one], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![next_counter, body_inputs[2], body_inputs[3], body_inputs[3]],
                vec![Placeholder; 4],
                vec![Placeholder; 4],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = (0..4).map(|_| builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let operation = WhileOperation::new().with_iteration_bound(iteration_bound).unwrap();
        let outputs = builder
            .add_instruction(operation, vec![condition_region, body_region], inputs, None)
            .unwrap()
            .to_vec();
        builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 4], vec![Placeholder; 4])
            .unwrap()
    }

    /// Builds a program whose `while` loop runs over the states `[x: f64[], limit: f64[]]`, squaring `x` while it is
    /// below `limit` and passing `limit` through unchanged, under the optional `iteration_bound`.
    fn pass_through_while_program(
        iteration_bound: Option<usize>,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let condition_inputs = (0..2).map(|_| condition_builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                condition_inputs.clone(),
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body_inputs = (0..2).map(|_| body_builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
        let square = body_builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![body_inputs[0], body_inputs[0]], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![square, body_inputs[1]], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = (0..2).map(|_| builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let operation = WhileOperation::new().with_iteration_bound(iteration_bound).unwrap();
        let outputs = builder
            .add_instruction(operation, vec![condition_region, body_region], inputs, None)
            .unwrap()
            .to_vec();
        builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
    }

    /// Builds a while condition that increments reference state before comparing it with `limit`.
    fn incrementing_condition(limit: f32) -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let update = builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let counter =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let limit = builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(limit).unwrap()));
        let predicate = builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    /// Builds a program that runs a `while` loop with the provided `condition` over one scalar `f32` reference whose
    /// body passes the reference through unchanged, and then reads the final reference value.
    fn reference_while_program(
        condition: Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>,
        iteration_bound: Option<usize>,
    ) -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(iteration_bound).unwrap();
        let reference = builder.add_instruction(operation, vec![condition, body], vec![reference], None).unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    /// Array operation family of the homogeneous `while` test programs.
    type TestDomainOperation = ArrayOperation<Array>;

    /// Eager interpreting domain over [`Array`] values that reports no support for primal concretization. Hybrid
    /// rules (in particular the while JVP rule) therefore take their staged, non-concretizing paths while every
    /// primal bind still computes concrete values, which lets the tests interpret linear while bodies numerically
    /// without abstract tracers.
    #[derive(Copy, Clone, Debug)]
    struct StagedDispatchTestDomain;

    impl Domain for StagedDispatchTestDomain {
        type Type = ArrayType;
        type Value = Array;
        type Constant = Array;
        type Operation = TestDomainOperation;
    }

    impl NamedAxes for StagedDispatchTestDomain {
        fn named_axis(&self, _name: &str) -> Option<NamedAxis> {
            None
        }

        fn named_axes(&self) -> Vec<(String, NamedAxis)> {
            Vec::new()
        }
    }

    impl Context for StagedDispatchTestDomain {
        fn lift(&self, constant: Array) -> Result<Array, ProgramError> {
            Ok(constant)
        }

        fn bind<P: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
            &self,
            operation: P,
            driver: D,
            inputs: &[Self::Value],
        ) -> Result<Vec<Self::Value>, ProgramError> {
            // Region-carrying binds route through the eager context's own bind, which grants application-scoped region
            // access.
            EagerContext::<Array, Self::Operation>::new().bind(operation, driver, inputs)
        }

        fn is_eager(&self) -> bool {
            false
        }

        // This test domain executes values directly and records no instructions, so provenance is a no-op.
        fn provenance(&self) -> Provenance {
            Provenance::unknown()
        }

        fn resolve(&self, value: &Array) -> ValueResolution<Array> {
            ValueResolution::Constant(value.clone())
        }

        fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
            function()
        }

        fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
            function()
        }
    }

    // Eager-domain context capabilities, delegating to the zero-state [`EagerContext`] exactly like
    // `EagerContext<Array, ArrayOperation<Array>>`'s.
    impl Zero<Array> for StagedDispatchTestDomain {
        fn zero(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
            Zero::zero(&EagerContext::<Array>::new(), r#type)
        }
    }

    impl One<Array> for StagedDispatchTestDomain {
        fn one(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
            One::one(&EagerContext::<Array>::new(), r#type)
        }
    }

    impl<S: ArrayElement> Fill<S, Array> for StagedDispatchTestDomain {
        fn fill(&self, r#type: &ArrayType, value: S) -> Result<Array, ProgramError> {
            Fill::fill(&EagerContext::<Array>::new(), r#type, value)
        }
    }

    impl Iota<Array> for StagedDispatchTestDomain {
        fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Array, ProgramError> {
            Iota::iota(&EagerContext::<Array>::new(), r#type, dimension)
        }
    }

    impl Constant<Array, Array> for StagedDispatchTestDomain {
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

        fn bind<P: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
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

        fn is_eager(&self) -> bool {
            true
        }

        // This test context executes eagerly and records no instructions, so provenance is a no-op.
        fn provenance(&self) -> Provenance {
            Provenance::unknown()
        }

        fn resolve(&self, value: &Array) -> ValueResolution<Array> {
            ValueResolution::Constant(value.clone())
        }

        fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
            function()
        }

        fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
            function()
        }
    }

    impl Zero<Array> for CountingPrintContext {
        fn zero(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
            Zero::zero(&EagerContext::<Array>::new(), r#type)
        }
    }

    impl One<Array> for CountingPrintContext {
        fn one(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
            One::one(&EagerContext::<Array>::new(), r#type)
        }
    }

    impl<S: ArrayElement> Fill<S, Array> for CountingPrintContext {
        fn fill(&self, r#type: &ArrayType, value: S) -> Result<Array, ProgramError> {
            Fill::fill(&EagerContext::<Array>::new(), r#type, value)
        }
    }

    impl Iota<Array> for CountingPrintContext {
        fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Array, ProgramError> {
            Iota::iota(&EagerContext::<Array>::new(), r#type, dimension)
        }
    }

    impl Constant<Array, Array> for CountingPrintContext {
        fn constant(&self, value: Array) -> Result<Array, ProgramError> {
            Ok(value)
        }
    }

    /// Binds the `while` `operation` with its condition and body `regions` to `inputs` in the domain of the inputs and
    /// returns the final loop state.
    fn bind_while<V: Value<Type = ArrayType>>(
        operation: WhileOperation<ArrayType>,
        regions: Vec<Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>>>,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError>
    where
        V::Domain: Context<Type = ArrayType, Constant = Array, Operation = TestDomainOperation>,
    {
        inputs[0].domain().bind(TestDomainOperation::While(operation), regions, inputs)
    }

    /// Builds the `state < threshold` while condition program over one scalar state element.
    fn scalar_threshold_condition(threshold: f64) -> Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let threshold = builder.add_constant(Array::scalar(threshold).unwrap());
        let predicate = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![state, threshold],
                None,
            )
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
        let two = builder.add_constant(Array::scalar(2.0).unwrap());
        let doubled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, two], None).unwrap()[0];
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
        let printed =
            body_builder.add_instruction(PrintOperation::new("state"), Vec::new(), vec![input], None).unwrap()[0];
        let two = body_builder.add_constant(Array::scalar(2.0).unwrap());
        let output =
            body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![printed, two], None).unwrap()[0];
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
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
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
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        (WhileOperation::new(), vec![scalar_threshold_condition(threshold), body])
    }

    /// Builds the per-item countdown loop `while (x > 0) { x = x - 1 }` over one scalar state element.
    fn countdown_while_operation()
    -> (WhileOperation<ArrayType>, Vec<Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>>>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder
            .add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![condition_state], None)
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![condition_state, zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let one = body_builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![body_state], None).unwrap()[0];
        let next =
            body_builder.add_instruction(SubOperation::new(), Vec::new(), vec![body_state, one], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        (WhileOperation::new(), vec![condition, body])
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
            .add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![condition_counter], None)
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![condition_counter, zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let body_counter = body_builder.add_input(scalar_f64.clone());
        let body_value = body_builder.add_input(scalar_f64);
        let one =
            body_builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![body_counter], None).unwrap()[0];
        let next_counter = body_builder
            .add_instruction(SubOperation::new(), Vec::new(), vec![body_counter, one], None)
            .unwrap()[0];
        let doubled = body_builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![body_value, body_value], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_counter, doubled], vec![Placeholder; 2], vec![Placeholder; 2])
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

    /// Builds a program whose `while` loop runs over the states `[counter: f64[], acc: f64[]]` while `counter > 0`,
    /// computing `(counter - 1, acc + counter)` under the optional `iteration_bound`. The counter is a closed, but not
    /// loop-invariant, state element: its next value and the predicate depend on it alone.
    fn counter_accumulator_while_program(
        iteration_bound: Option<usize>,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let counter = condition_builder.add_input(scalar_type.clone());
        condition_builder.add_input(scalar_type.clone());
        let zero = condition_builder
            .add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![counter], None)
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![counter, zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let counter = body_builder.add_input(scalar_type.clone());
        let acc = body_builder.add_input(scalar_type.clone());
        let one = body_builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
        let next_counter =
            body_builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
        let next_acc =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![acc, counter], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_counter, next_acc], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let inputs = vec![builder.add_input(scalar_type.clone()), builder.add_input(scalar_type)];
        let operation = WhileOperation::new().with_iteration_bound(iteration_bound).unwrap();
        let outputs = builder.add_instruction(operation, regions, inputs, None).unwrap().to_vec();
        builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
    }

    /// Builds a program that applies the `while` `operation` with its condition and body `regions` to inputs of the
    /// provided `input_types` and returns the final loop state.
    fn while_program(
        operation: WhileOperation<ArrayType>,
        regions: &[Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>>],
        input_types: Vec<ArrayType>,
    ) -> Program<Array, TestDomainOperation, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let regions = regions.iter().map(|region| builder.import_region(region.entry_region_ref())).collect();
        let state_count = input_types.len();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect();
        let outputs = builder.add_instruction(operation, regions, inputs, None).unwrap().to_vec();
        builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; state_count], vec![Placeholder; state_count])
            .unwrap()
    }

    /// Array IR region program over the reference [`Array`] backend.
    type ArrayIrRegionProgram =
        Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>;

    /// Builds the array IR per-item countdown loop `while (x > 0) { x = x - 1 }` over one scalar `f64` array state
    /// element, returning its condition and body programs in `while` region order.
    fn array_ir_countdown_while_regions() -> Vec<ArrayIrRegionProgram> {
        let scalar_f64 = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let mut condition_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::ZeroLike(ZeroLikeOperation::new())),
                Vec::new(),
                vec![condition_state],
                None,
            )
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                    ComparisonDirection::GreaterThan,
                ))),
                Vec::new(),
                vec![condition_state, zero],
                None,
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
                None,
            )
            .unwrap()[0];
        let next = body_builder
            .add_instruction(
                ArrayIrOperation::Array(ArrayOperation::Sub(SubOperation::new())),
                Vec::new(),
                vec![body_state, one],
                None,
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

    /// Builds the array IR loop `while (x < 8) { (extent, x) = (extent, x + x) }` over a forwarded first-class
    /// dimension of type `extent_type` and one scalar `f64` array state element, returning its condition and body
    /// programs in `while` region order.
    fn doubling_while_regions(
        extent_type: DimensionType,
    ) -> Vec<Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>> {
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let state = condition_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let limit = condition_builder.add_constant(array(Array::scalar(8.0).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![state, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder.build(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let extent = body_builder.add_input(ArrayIrType::Dimension(extent_type));
        let state = body_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let doubled = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![state, state],
                None,
            )
            .unwrap()[0];
        let body = body_builder.build(vec![extent, doubled], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        vec![condition, body]
    }

    /// Builds a program whose `while` loop runs over the states `[counter: f64[], vector: f64[rows]]` from
    /// unspecialized regions (`counter > 0`, and `[counter - 1, vector + vector]`) while the program feeds it a static
    /// `f64[3]` vector, so the loop's outputs are refined by its inputs rather than by re-typed regions.
    fn refined_vector_while_program(
        iteration_bound: Option<usize>,
    ) -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let counter_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let vector_type = ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(rows)])));
        let state_types = vec![counter_type.clone(), vector_type];

        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_inputs =
            state_types.iter().map(|r#type| condition_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let zero = condition_builder.add_constant(array(Array::scalar(0.0).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::GreaterThan))),
                Vec::new(),
                vec![condition_inputs[0], zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body_inputs = state_types.iter().map(|r#type| body_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let one = body_builder.add_constant(array(Array::scalar(1.0).unwrap()));
        let next_counter = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(SubOperation::new())),
                Vec::new(),
                vec![body_inputs[0], one],
                None,
            )
            .unwrap()[0];
        let next_vector = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![body_inputs[1], body_inputs[1]],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_counter, next_vector],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let inputs = vec![
            builder.add_input(counter_type),
            builder.add_input(ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]))),
        ];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(iteration_bound).unwrap();
        let outputs = builder.add_instruction(operation, regions, inputs, None).unwrap().to_vec();
        builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap()
    }

    #[test]
    fn test_while() {
        // A `while` payload carries only its optional iteration bound, so loops compare, hash, and copy by that bound,
        // and the default loop is the unbounded one.
        let operation = WhileOperation::<ArrayType>::new();
        let bounded = WhileOperation::<ArrayType>::new().with_iteration_bound(2).unwrap();
        assert_eq!(operation.name(), WHILE_OPERATION_NAME);
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("condition"), RegionSlot::computation("body")]);
        assert_eq!(WhileOperation::<ArrayType>::default(), operation);
        assert_eq!(operation, operation);
        assert_ne!(operation, bounded);
        let copy = bounded;
        assert_eq!(copy, bounded);
        let operations = HashSet::from([operation, bounded]);
        assert!(operations.contains(&WhileOperation::new()));
        assert!(operations.contains(&WhileOperation::new().with_iteration_bound(2).unwrap()));
        assert!(!operations.contains(&WhileOperation::new().with_iteration_bound(3).unwrap()));
    }

    #[test]
    fn test_while_render() {
        // The operation renders its iteration bound as a field, and a program renders the attached condition and body
        // regions at the instruction with their declared slot names.
        let bounded = WhileOperation::<ArrayType>::new().with_iteration_bound(2).unwrap();
        assert_eq!(WhileOperation::<ArrayType>::new().to_string(), "while");
        assert_eq!(bounded.to_string(), "while [iteration_bound=2]");
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let regions =
            vec![builder.import_program(greater_than_zero_condition()), builder.import_program(subtract_one_body())];
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(bounded, regions, vec![state], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
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
                in (%1)"},
        );
    }

    #[test]
    fn test_while_with_iteration_bound() {
        // A bound must be at least one and at most the backend maximum, and `None` clears it.
        let bounded = WhileOperation::<ArrayType>::new().with_iteration_bound(2).unwrap();
        assert_eq!(bounded.iteration_bound(), Some(2));
        assert_eq!(bounded.with_iteration_bound(None), Ok(WhileOperation::new()));
        assert_eq!(
            WhileOperation::<ArrayType>::new().with_iteration_bound(0),
            Err(ProgramError::Type(TypeError::invalid("`while` iteration bound must be at least 1"))),
        );
        assert_eq!(
            WhileOperation::<ArrayType>::new()
                .with_iteration_bound(MAX_DIMENSION_EXTENT)
                .map(|operation| operation.iteration_bound()),
            Ok(Some(MAX_DIMENSION_EXTENT)),
        );

        // On hosts whose `usize` cannot exceed the backend maximum, no bound is rejected for being too large.
        if let Some(bound) = MAX_DIMENSION_EXTENT.checked_add(1) {
            assert_eq!(
                WhileOperation::<ArrayType>::new().with_iteration_bound(bound),
                Err(ProgramError::Type(TypeError::invalid(format!(
                    "`while` iteration bound {bound} exceeds backend maximum {MAX_DIMENSION_EXTENT}",
                )))),
            );
        }
    }

    #[test]
    fn test_while_iteration_bound() {
        // The iteration bound reads back as configured, and a loop without one reports `None`.
        assert_eq!(WhileOperation::<ArrayType>::new().iteration_bound(), None);
        assert_eq!(WhileOperation::<ArrayType>::new().with_iteration_bound(5).unwrap().iteration_bound(), Some(5));
    }

    #[test]
    fn test_while_validate_application() {
        // A valid application returns the output types that type inference infers for it, and the boundary that type
        // inference validates (e.g., the input count) is part of the validated contract.
        let state_type = ArrayType::scalar(DataType::F64);
        let regions = [greater_than_zero_condition(), subtract_one_body()];
        let operation = WhileOperation::<ArrayType>::new();
        assert_eq!(
            operation.validate_application(
                std::slice::from_ref(&state_type),
                regions[0].entry_region_ref(),
                regions[1].entry_region_ref(),
            ),
            Ok(vec![state_type]),
        );
        assert_eq!(
            operation.validate_application(&[], regions[0].entry_region_ref(), regions[1].entry_region_ref()),
            Err(ProgramError::Type(TypeError::invalid("expected 1 input but got 0"))),
        );

        // So is the reference contract that only the body reveals: a body that swaps two reference carries gives the
        // loop state no fixed point.
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(reference_type.clone());
        condition_builder.add_input(reference_type.clone());
        let predicate = condition_builder.add_constant(array(Array::scalar(false).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = body_builder.add_input(reference_type.clone());
        let second = body_builder.add_input(reference_type.clone());
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![second, first],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            WhileOperation::<ArrayIrType>::new().validate_application(
                &[reference_type.clone(), reference_type],
                condition.entry_region_ref(),
                body.entry_region_ref(),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`while` body does not return carry 0 as the reference it entered with, so the loop state has no fixed \
                 point",
            ))),
        );

        // So is the loop invariance of state that a batched predicate cannot mask: a first-class dimension carry has no
        // per-item value, so the body must pass it through unchanged, and replacing it even by an equal constant fails.
        let dimension_type = DimensionType::from(Dimension::Static(2));
        let state_types =
            vec![ArrayIrType::Dimension(dimension_type.clone()), ArrayType::new_static(DataType::F32, [2]).into()];
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(state_types[0].clone());
        condition_builder.add_input(state_types[1].clone());
        let predicate = condition_builder.add_constant(array(Array::vector(vec![true, false]).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(state_types[0].clone());
        let vector = body_builder.add_input(state_types[1].clone());
        let extent = body_builder.add_constant(dimension(&dimension_type, 2));
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![extent, vector],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(
            WhileOperation::<ArrayIrType>::new().validate_application(
                &state_types,
                condition.entry_region_ref(),
                body.entry_region_ref(),
            ),
            Err(ProgramError::Type(TypeError::invalid(
                "`while` state 0 of type `dimension<2>` must be passed through unchanged by the body under a batched \
                 predicate, because it has no per-item value to mask",
            ))),
        );
    }

    #[test]
    fn test_while_validate_region_interfaces() {
        // Region interface validation checks the region count and the declared (unrefined) body signature, so a body
        // that maps `f64[extent]` to `f64[3]` is rejected through every operation-family dispatch.
        let regions = refinement_only_while_regions();
        let interfaces = regions.each_ref().map(Program::interface);
        let expected = Err(ProgramError::Type(TypeError::invalid(
            "`while` body output type signature mismatch: expected [f64[extent]] but got [f64[3]]",
        )));
        assert_eq!(WhileOperation::<ArrayType>::new().validate_region_interfaces(&interfaces), expected);
        assert_eq!(Box::new(WhileOperation::<ArrayType>::new()).validate_region_interfaces(&interfaces), expected);
        assert_eq!(
            ArrayOperation::<Array>::While(WhileOperation::new()).validate_region_interfaces(&interfaces),
            expected,
        );
        assert_eq!(
            WhileOperation::<ArrayType>::new().validate_region_interfaces(&interfaces[..1]),
            Err(ProgramError::MalformedProgram(
                "operation `while` declares 2 region slots but 1 regions were attached".to_string(),
            )),
        );

        // Array IR operation families validate the same contract through their derived and kernel dispatch.
        let regions = regions.map(|region| region.into_unprojected::<TestIrValue, TestIrOperation>().unwrap());
        let interfaces = regions.each_ref().map(Program::interface);
        assert_eq!(TestIrOperation::While(WhileOperation::new()).validate_region_interfaces(&interfaces), expected);
        assert_eq!(
            KernelOperation::<NoKernelExtension>::Portable(TestIrOperation::While(WhileOperation::new()))
                .validate_region_interfaces(&interfaces),
            expected,
        );
    }

    #[test]
    fn test_while_rejects_bodies_that_only_fit_after_refinement() {
        // An `f64[3]` input refines the body's declared `f64[extent] -> f64[3]` signature to a valid one, but the
        // declared body does not map its state to itself. Builder construction, eager binding (even though the loop
        // never iterates), staging (which could otherwise specialize the body), partial evaluation, and forward-mode
        // differentiation all reject it.
        let regions = refinement_only_while_regions();
        let expected = ProgramError::Type(TypeError::invalid(
            "`while` body output type signature mismatch: expected [f64[extent]] but got [f64[3]]",
        ));
        let input_type = ArrayType::new_static(DataType::F64, [3]);
        let input = Array::vector(vec![1.0f64; 3]).unwrap();

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let region_ids = regions.each_ref().map(|region| builder.import_region(region.entry_region_ref()));
        let state = builder.add_input(input_type.clone());
        assert_eq!(
            builder.add_instruction(WhileOperation::new(), region_ids.to_vec(), vec![state], None).map(|_| ()),
            Err(expected.clone()),
        );

        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            context.bind(WhileOperation::new(), regions.to_vec(), std::slice::from_ref(&input)),
            Err(expected.clone()),
        );

        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let state = context.input(input_type.clone());
        assert_eq!(context.bind(WhileOperation::new(), regions.to_vec(), &[state]).map(|_| ()), Err(expected.clone()));
        assert!(context.builder().borrow().instructions().is_empty());

        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let state = PartialTracer::new(context.clone(), context.unknown_input(input_type, 0));
        assert_eq!(context.bind(WhileOperation::new(), regions.to_vec(), &[state]).map(|_| ()), Err(expected.clone()));

        let result = differentiate_at(input.clone()).jvp(input, |state| {
            let context = state.context().clone();
            Ok(context.bind(WhileOperation::new(), regions.to_vec(), &[state])?.remove(0))
        });
        assert!(matches!(result, Err(DifferentiationError::Program(error)) if error == expected));
    }

    #[test]
    fn test_while_infer_region_input_types() {
        // The regions are requested at the input types only when those differ from the declared state types, so that
        // staging can specialize them, and inputs that match the state types leave the declared regions in place.
        let state_type = ArrayType::scalar(DataType::F64);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_type = state_type.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let interfaces = vec![greater_than_zero_condition().interface(), subtract_one_body().interface()];
        let operation = WhileOperation::<ArrayType>::new();
        assert_eq!(
            operation.infer_region_input_types(std::slice::from_ref(&state_type), &interfaces),
            Ok(vec![None, None]),
        );
        assert_eq!(
            operation.infer_region_input_types(std::slice::from_ref(&sharded_type), &interfaces),
            Ok(vec![Some(vec![sharded_type.clone()]), Some(vec![sharded_type])]),
        );
        assert_eq!(
            operation.infer_region_input_types(std::slice::from_ref(&state_type), &interfaces[..1]),
            Err(TypeError::invalid("expected 2 regions but got 1")),
        );
    }

    #[test]
    fn test_while_type_inference() {
        // Type inference validates the region interfaces and the input types, and returns the state types.
        let state_type = ArrayType::scalar(DataType::F64);
        let operation = WhileOperation::<ArrayType>::new();
        let condition = greater_than_zero_condition().interface();
        let body = subtract_one_body().interface();
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&state_type), &[condition.clone(), body.clone()]),
            Ok(vec![state_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&state_type), &[]),
            Err(TypeError::invalid("expected 2 regions but got 0")),
        );
        assert_eq!(
            operation.infer_output_types(&[], &[condition.clone(), body.clone()]),
            Err(TypeError::invalid("expected 1 input but got 0")),
        );
        assert_eq!(
            operation
                .infer_output_types(&[ArrayType::new_static(DataType::F64, [2])], &[condition.clone(), body.clone()]),
            Err(TypeError::invalid("`while` input 0 has type `f64[2]`, which does not refine its state type `f64[]`")),
        );

        // Inference rejects mismatched condition and body state signatures, non-Boolean condition outputs, conditions
        // with several outputs, and body outputs that do not match the state signature.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(ArrayType::new_static(DataType::F64, [2]));
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let vector_body = builder.build::<Vec<Array>, Vec<Array>>(vec![zero], vec![Placeholder], vec![Placeholder]);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&state_type),
                &[condition.clone(), vector_body.unwrap().interface()],
            ),
            Err(TypeError::invalid(
                "`while` condition/body input type signature mismatch: expected [f64[2]] but got [f64[]]",
            )),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&state_type), &[body.clone(), body.clone()]),
            Err(TypeError::invalid("`while` condition output type must be a Boolean array, but got f64[]")),
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let multi_output_condition = builder
            .build::<Vec<Array>, Vec<Array>>(vec![state, state], vec![Placeholder], vec![Placeholder, Placeholder])
            .unwrap();
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&state_type),
                &[multi_output_condition.interface(), body.clone()],
            ),
            Err(TypeError::invalid("`while` condition must return exactly one predicate leaf but returned 2")),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&state_type), &[condition.clone(), condition]),
            Err(TypeError::invalid("`while` body output type signature mismatch: expected [f64[]] but got [bool[]]")),
        );

        // The final `validate_output_identities` check cannot fail for a `while`: its outputs are its state types
        // refined by the input facts, and every identity that a state type refers to is either defined by a state
        // type (a first-class dimension carry) or fixed by the input that refines it. An input that refers to a
        // different identity does not refine the state type at all.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(4)).unwrap());
        let state_types =
            vec![ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])))];
        let interfaces = [
            RegionInterface::new(
                state_types.clone(),
                vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))],
                EffectClasses::NONE,
            ),
            RegionInterface::new(state_types.clone(), state_types, EffectClasses::NONE),
        ];
        assert_eq!(
            WhileOperation::<ArrayIrType>::new().infer_output_types(
                &[ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)])))],
                &interfaces,
            ),
            Err(TypeError::invalid(
                "`while` input 0 has type `f32[rows]`, which does not refine its state type `f32[extent]`",
            )),
        );
    }

    #[test]
    fn test_while_type_inference_array_ir_state() {
        // A loop over array IR state, a first-class dimension carry and an array sized by it, infers its state types
        // under a scalar predicate.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = ArrayIrType::Dimension(DimensionType::from(extent.clone()));
        let array_type =
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())])));
        let state_types = vec![dimension_type.clone(), array_type.clone()];
        let condition_interface = RegionInterface::new(
            state_types.clone(),
            vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))],
            EffectClasses::NONE,
        );
        let body_interface = RegionInterface::new(state_types.clone(), state_types.clone(), EffectClasses::NONE);
        let operation = WhileOperation::new();

        assert_eq!(
            operation.infer_output_types(state_types.as_slice(), &[condition_interface, body_interface.clone()]),
            Ok(state_types.clone()),
        );

        // A batched predicate is legal alongside first-class dimension state: the dimension member carries no shape, so
        // the prefix requirement applies to the array member only, and the loop invariance that a dimension carry needs
        // is checked over the body itself (refer to `test_while_validate_application`) rather than by this contract.
        let batched_condition_interface = RegionInterface::new(
            state_types.clone(),
            vec![ArrayIrType::Array(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Dynamic(extent)])))],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation
                .infer_output_types(state_types.as_slice(), &[batched_condition_interface, body_interface.clone()]),
            Ok(state_types.clone()),
        );

        // Reference state is effectful and has no per-item mask semantics. Reject it at this local type boundary when
        // the predicate is batched rather than silently treating it like a loop-invariant first-class dimension.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Static(2)]),
        )));
        let reference_condition_interface = RegionInterface::new(
            vec![reference_type.clone()],
            vec![ArrayIrType::Array(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])))],
            EffectClasses::NONE,
        );
        let reference_body_interface =
            RegionInterface::new(vec![reference_type.clone()], vec![reference_type.clone()], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&reference_type),
                &[reference_condition_interface, reference_body_interface],
            ),
            Err(TypeError::invalid(
                "`while` condition with a batched predicate cannot carry reference state `ref<f32[2]>` because \
                 effectful state cannot be masked per batch item; discharge the reference first",
            )),
        );

        // The prefix requirement still holds for every array member under a batched predicate.
        let mismatched_condition_interface = RegionInterface::new(
            state_types.clone(),
            vec![ArrayIrType::Array(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])))],
            EffectClasses::NONE,
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
            EffectClasses::NONE,
        );
        let shape_varying_body = RegionInterface::new(vec![carry_type.clone()], vec![next_type], EffectClasses::NONE);
        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&carry_type), &[condition_interface, shape_varying_body],),
            Err(TypeError::invalid(
                "`while` body output type signature mismatch: expected [f32[carry]] but got [f32[next]]".to_string(),
            )),
        );
    }

    #[test]
    fn test_while_type_inference_rejects_batched_predicates_with_effects() {
        // A batched predicate keeps the loop running for still-active items after others finish, re-executing the
        // body over every item each iteration. Values are masked back for finished items, but observable effects
        // (here a `print` in the body) cannot be, so type inference rejects an effectful batched-predicate loop
        // through the attached region interfaces' declared effects. A scalar predicate imposes no such restriction
        // (the loop exits for all items at once).
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::GreaterThan),
                    Vec::new(),
                    vec![state, zero],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let effectful_body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            let next_state =
                builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            let printed =
                builder.add_instruction(PrintOperation::new("state"), Vec::new(), vec![next_state], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![printed], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        assert_eq!(
            WhileOperation::new().infer_output_types(
                std::slice::from_ref(&state_type),
                &[condition.interface(), effectful_body.interface()],
            ),
            Err(TypeError::invalid(
                "`while` loop with a batched predicate must be pure because observable effects cannot be \
                 masked for finished batch items"
                    .to_string()
            )),
        );
    }

    #[test]
    fn test_while_type_inference_rejects_batched_predicates_with_deferred_work() {
        // Finished batch items keep re-executing the condition and body, so a region carrying deferred work would
        // repeat its transformation obligation for items whose values are masked back. Type inference therefore
        // rejects deferred work in either region of a batched-predicate loop, while a scalar predicate (the loop exits
        // for all items at once) imposes no such restriction.
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let batched_predicate = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(3)]));
        let condition = |predicate: &ArrayType| {
            RegionInterface::new(vec![state_type.clone()], vec![predicate.clone()], EffectClasses::NONE)
        };
        let body = RegionInterface::new(vec![state_type.clone()], vec![state_type.clone()], EffectClasses::NONE);
        let error = Err(TypeError::invalid(
            "`while` loop with a batched predicate cannot carry deferred work because finished batch items would \
             repeat its transformation obligation"
                .to_string(),
        ));
        assert_eq!(
            WhileOperation::new().infer_output_types(
                std::slice::from_ref(&state_type),
                &[condition(&batched_predicate), body.clone().with_deferred_work(true)],
            ),
            error,
        );
        assert_eq!(
            WhileOperation::new().infer_output_types(
                std::slice::from_ref(&state_type),
                &[condition(&batched_predicate).with_deferred_work(true), body.clone()],
            ),
            error,
        );
        assert_eq!(
            WhileOperation::new().infer_output_types(
                std::slice::from_ref(&state_type),
                &[condition(&ArrayType::scalar(DataType::Boolean)), body.with_deferred_work(true)],
            ),
            Ok(vec![state_type.clone()]),
        );
    }

    #[test]
    fn test_while_type_inference_rejects_batched_predicates_that_are_not_state_shape_prefixes() {
        // A `bool[3]` predicate over an `f64[2]` state violates the predicate-prefix rule: item masking would be
        // ill-defined, so type inference over the attached region interfaces fails.
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            builder.add_input(state_type.clone());
            let predicate =
                builder.add_instruction(ZeroOperation::new(predicate_type), Vec::new(), vec![], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(state_type.clone());
            builder.build::<Vec<Array>, Vec<Array>>(vec![state], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        assert_eq!(
            WhileOperation::new()
                .infer_output_types(std::slice::from_ref(&state_type), &[condition.interface(), body.interface()]),
            Err(TypeError::invalid(
                "`while` condition predicate shape must be a prefix of every array state shape, but predicate \
                 bool[3] is not a prefix of state f64[2]"
                    .to_string()
            )),
        );
    }

    #[test]
    fn test_while_type_inference_manual_variation() {
        // Devices whose predicates differ run different numbers of iterations, so every state element must vary over
        // the predicate's manual axes. The rule covers array state and reference referents but not dimensions.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant = |data_type| ArrayType::scalar(data_type).with_sharding(Sharding::replicated(mesh.clone(), 0));
        let varying = |data_type| {
            ArrayType::scalar(data_type)
                .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["m"]).unwrap())
        };
        let infer = |predicate_type: ArrayIrType, state_types: Vec<ArrayIrType>| {
            WhileOperation::new().infer_output_types(
                state_types.as_slice(),
                &[
                    RegionInterface::new(state_types.clone(), vec![predicate_type], EffectClasses::NONE),
                    RegionInterface::new(state_types.clone(), state_types.clone(), EffectClasses::NONE),
                ],
            )
        };
        let varying_predicate = ArrayIrType::Array(varying(DataType::Boolean).unwrap());
        let invariant_predicate = ArrayIrType::Array(invariant(DataType::Boolean).unwrap());
        let varying_state = ArrayIrType::Array(varying(DataType::F32).unwrap());
        let invariant_state = ArrayIrType::Array(invariant(DataType::F32).unwrap());
        let dimension_state =
            ArrayIrType::Dimension(DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let state_types = vec![varying_state.clone(), dimension_state];
        assert_eq!(infer(varying_predicate.clone(), state_types.clone()), Ok(state_types));
        let state_types = vec![invariant_state.clone(), ArrayIrType::Array(ArrayType::scalar(DataType::F32))];
        assert_eq!(infer(invariant_predicate, state_types.clone()), Ok(state_types));
        assert_eq!(
            infer(varying_predicate.clone(), vec![varying_state.clone(), invariant_state]),
            Err(TypeError::invalid(
                "`while` state `f32[][sharding={mesh<['m'=2:manual]>, []}]` must vary over every manual axis that the \
                 predicate `bool[][sharding={mesh<['m'=2:manual]>, [], varying_manual={'m'}}]` varies over, because \
                 devices may run different numbers of iterations; insert `parallel_vary` on the initial state",
            )),
        );
        assert_eq!(
            infer(varying_predicate.clone(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]),
            Err(TypeError::invalid(
                "`while` state `f32[]` must vary over every manual axis that the predicate \
                 `bool[][sharding={mesh<['m'=2:manual]>, [], varying_manual={'m'}}]` varies over, because devices may \
                 run different numbers of iterations; insert `parallel_vary` on the initial state",
            )),
        );
        let varying_reference = ArrayIrType::Reference(ReferenceType::new(varying(DataType::F32).unwrap()));
        assert_eq!(infer(varying_predicate.clone(), vec![varying_reference.clone()]), Ok(vec![varying_reference]),);
        assert_eq!(
            infer(
                varying_predicate,
                vec![ArrayIrType::Reference(ReferenceType::new(invariant(DataType::F32).unwrap()))],
            ),
            Err(TypeError::invalid(
                "`while` state `f32[][sharding={mesh<['m'=2:manual]>, []}]` must vary over every manual axis that the \
                 predicate `bool[][sharding={mesh<['m'=2:manual]>, [], varying_manual={'m'}}]` varies over, because \
                 devices may run different numbers of iterations; insert `parallel_vary` on the initial state",
            )),
        );
    }

    #[test]
    fn test_while_type_inference_refines_input_types() {
        // Inputs only need to refine the state types, so actual types that carry metadata the state types leave
        // unspecified (e.g., the normalized shardings of concrete backend array types) are accepted, and the outputs
        // do not inherit that metadata.
        let state_type = ArrayType::scalar(DataType::F64);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_type = state_type.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let operation = WhileOperation::new();
        let interfaces = vec![greater_than_zero_condition().interface(), subtract_one_body().interface()];
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&sharded_type), interfaces.as_slice()),
            Ok(vec![state_type.clone()]),
        );

        // A static extent within the bounds of a dynamic state dimension refines it, while an extent outside those
        // bounds or a different data type does not.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let state_types = vec![
            ArrayIrType::Dimension(DimensionType::from(extent.clone())),
            ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]))),
        ];
        let interfaces = vec![
            RegionInterface::new(
                state_types.clone(),
                vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))],
                EffectClasses::NONE,
            ),
            RegionInterface::new(state_types.clone(), state_types.clone(), EffectClasses::NONE),
        ];
        let refined_input_types = |data_type: DataType, extent: usize| {
            vec![state_types[0].clone(), ArrayIrType::Array(ArrayType::new_static(data_type, [extent]))]
        };
        assert_eq!(
            WhileOperation::new().infer_output_types(&refined_input_types(DataType::F32, 5), interfaces.as_slice()),
            Ok(state_types.clone()),
        );
        assert_eq!(
            WhileOperation::new().infer_output_types(&refined_input_types(DataType::F32, 9), interfaces.as_slice()),
            Err(TypeError::invalid(
                "`while` input 1 has type `f32[9]`, which does not refine its state type `f32[extent]`",
            )),
        );
        assert_eq!(
            WhileOperation::new().infer_output_types(&refined_input_types(DataType::F64, 5), interfaces.as_slice()),
            Err(TypeError::invalid(
                "`while` input 1 has type `f64[5]`, which does not refine its state type `f32[extent]`",
            )),
        );

        // Without the first-class dimension state, nothing defines `extent`, so it is loop-invariant and the output
        // takes the static extent that the input establishes for it.
        let array_state_types = vec![state_types[1].clone()];
        let array_interfaces = vec![
            RegionInterface::new(
                array_state_types.clone(),
                vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))],
                EffectClasses::NONE,
            ),
            RegionInterface::new(array_state_types.clone(), array_state_types, EffectClasses::NONE),
        ];
        assert_eq!(
            WhileOperation::new().infer_output_types(
                &[ArrayIrType::Array(ArrayType::new_static(DataType::F32, [5]))],
                array_interfaces.as_slice(),
            ),
            Ok(vec![ArrayIrType::Array(ArrayType::new_static(DataType::F32, [5]))]),
        );

        // Facts are established from the complete input signature, so two states that observe different extents for
        // `extent` conflict. A partial refinement fixes `rows` and keeps `columns` symbolic in the output.
        let pair_state_types = vec![state_types[1].clone(), state_types[1].clone()];
        let pair_interfaces = vec![
            RegionInterface::new(
                pair_state_types.clone(),
                vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))],
                EffectClasses::NONE,
            ),
            RegionInterface::new(pair_state_types.clone(), pair_state_types, EffectClasses::NONE),
        ];
        let conflict = WhileOperation::new()
            .infer_output_types(
                &[
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [5])),
                    ArrayIrType::Array(ArrayType::new_static(DataType::F32, [6])),
                ],
                pair_interfaces.as_slice(),
            )
            .unwrap_err();
        assert_eq!(
            conflict.downcast_custom::<DimensionError>(),
            Some(&DimensionError::InputDimensionMismatch { dimension: "extent".to_string(), expected: 5, actual: 6 }),
        );
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::positive(Some(8)).unwrap());
        let matrix_type = |rows: Dimension| {
            ArrayIrType::Array(ArrayType::new(
                DataType::F32,
                Shape::new(vec![rows, Dimension::Dynamic(columns.clone())]),
            ))
        };
        let matrix_state_types = vec![matrix_type(Dimension::Dynamic(rows))];
        let matrix_interfaces = vec![
            RegionInterface::new(
                matrix_state_types.clone(),
                vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))],
                EffectClasses::NONE,
            ),
            RegionInterface::new(matrix_state_types.clone(), matrix_state_types, EffectClasses::NONE),
        ];
        assert_eq!(
            WhileOperation::new()
                .infer_output_types(&[matrix_type(Dimension::Static(3))], matrix_interfaces.as_slice()),
            Ok(vec![matrix_type(Dimension::Static(3))]),
        );

        // A reference input refines its state type by its referent, and the carry output keeps the input's own type,
        // because it denotes the input's allocation.
        let ArrayIrType::Array(referent_type) = &state_types[1] else { unreachable!() };
        let reference_state_types = vec![ArrayIrType::Reference(ReferenceType::new(referent_type.clone()))];
        let reference_interfaces = vec![
            RegionInterface::new(
                reference_state_types.clone(),
                vec![ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))],
                EffectClasses::NONE,
            ),
            RegionInterface::new(reference_state_types.clone(), reference_state_types, EffectClasses::NONE),
        ];
        let refined_reference_type =
            ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, [5])));
        assert_eq!(
            WhileOperation::new()
                .infer_output_types(std::slice::from_ref(&refined_reference_type), reference_interfaces.as_slice()),
            Ok(vec![refined_reference_type]),
        );
    }

    #[test]
    fn test_while_traced_carry_joins_input_alias_family() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestContext>;

        // A `while` declares no reference effects of its own; it states that its carry output denotes the same
        // reference as its carry input through `reference_output_identity_input` instead. Tracing honors that hook, so
        // the loop's result belongs to its input's alias family, and an access through it after the allocation is
        // frozen is reported at the access that performs it.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(scalar_type.clone()));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(reference_type.clone());
        let predicate = condition_builder.add_constant(TestIrValue::Array(Array::scalar(false).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let carry = body_builder.add_input(reference_type);
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![carry], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let error = TestContext::trace(
            |input: TestTracer| {
                let context = input.context().clone();
                let reference = input.reference_new()?;
                let carried = context
                    .bind(
                        WhileOperation::new(),
                        vec![condition.clone(), body.clone()],
                        std::slice::from_ref(&reference),
                    )?
                    .remove(0);
                reference.freeze()?;
                carried.read()
            },
            ArrayIrType::Array(scalar_type),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::MalformedProgram(
                "`reference_read` reads a reference whose alias family `reference_freeze` already consumed".to_string(),
            ),
        );
    }

    #[test]
    fn test_while_staging() {
        type TestContext = TracingContext<Array, ArrayOperation<Array>>;

        // Staging imports the condition and body programs as attached regions of the staged instruction instead of
        // driving the loop with a concrete predicate.
        let (output_type, program) = TestContext::trace(
            |state: Tracer<TestContext>| {
                let context = state.context().clone();
                let regions = vec![greater_than_zero_condition(), subtract_one_body()];
                Ok(context.bind(WhileOperation::new(), regions, &[state])?.remove(0))
            },
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::scalar(DataType::F64));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = while %0 [
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
                in (%1)"},
        );
    }

    #[test]
    fn test_while_staging_falls_back_to_declared_regions_when_specialization_drops_a_refinement() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestContext>;

        // The body replaces its carry with an unsharded constant, so the carry is unsharded after every iteration even
        // when it enters the loop with a sharding. The body specialized at the sharded carry no longer maps its carry
        // to itself, so staging falls back to the declared body, and the carry keeps its declared, unsharded type.
        let state_type = ArrayType::scalar(DataType::F32);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_type = state_type.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(state_type.clone().into());
        let predicate = condition_builder.add_constant(array(Array::scalar(false).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(state_type.clone().into());
        let replacement = body_builder.add_constant(array(Array::scalar(1f32).unwrap()));
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![replacement], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let (output_type, program) = TestContext::trace(
            |input: TestTracer| {
                let context = input.context().clone();
                Ok(context
                    .bind(WhileOperation::new(), vec![condition.clone(), body.clone()], std::slice::from_ref(&input))?
                    .remove(0))
            },
            ArrayIrType::Array(sharded_type),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(state_type));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:auto]>, []}] .
                let %1:f32[] = while %0 [
                    condition={
                        lambda %0:f32[] .
                        let %1:bool[] = const false
                        in (%1)
                    },
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                        in (%1)
                    },
                ]
                in (%1)"},
        );
    }

    #[test]
    fn test_while_region_data_flow() {
        // Both regions receive the loop state at the instruction input positions, every output is the body output at
        // its position, and each output carry denotes the reference of its same-position input. The condition may
        // read and mutate an entering reference but never consume it, while the body keeps the permissive default.
        let operation = WhileOperation::<ArrayType>::new();
        assert!(matches!(operation.region_data_flow(), RegionDataFlow::Custom(_)));
        assert_eq!(operation.input_region_provenance(0, 1), InputRegionProvenance::Input { index: 1 });
        assert_eq!(operation.input_region_provenance(1, 1), InputRegionProvenance::Input { index: 1 });
        assert_eq!(operation.input_region_provenance(2, 1), InputRegionProvenance::None);
        assert_eq!(
            operation.output_region_provenance(1),
            vec![OutputRegionProvenance { region_index: 1, output_index: 1 }],
        );
        assert_eq!(operation.reference_output_identity_input(1), Some(1));
        assert!(operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::Read));
        assert!(operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::Write));
        assert!(operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::ReadWrite));
        assert!(operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::Accumulate));
        assert!(operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::AtomicAccumulate));
        assert!(!operation.allows_reference_access_through_region_input(0, ReferenceAccessMode::Consume));
        assert!(operation.allows_reference_access_through_region_input(1, ReferenceAccessMode::Write));
        assert!(operation.allows_reference_access_through_region_input(1, ReferenceAccessMode::ReadWrite));
        assert!(operation.allows_reference_access_through_region_input(1, ReferenceAccessMode::Consume));
    }

    #[test]
    fn test_while_region_data_flow_input_sources() {
        // Each region input is sourced from the same-position instruction input on entry and from the same-position
        // body output on every later iteration, and an out-of-range region input is rejected.
        let operation = WhileOperation::<ArrayType>::new();
        let regions = [
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 1 },
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 2 },
        ];
        let boundary = RegionDataFlowBoundary { input_count: 2, output_count: 2, regions: &regions };
        let sources = Ok(RegionDataFlowSources::Known(vec![
            RegionDataFlowSource::InstructionInput(1),
            RegionDataFlowSource::RegionOutput(OutputRegionProvenance { region_index: 1, output_index: 1 }),
        ]));
        assert_eq!(operation.input_sources(0, 1, boundary), sources);
        assert_eq!(operation.input_sources(1, 1, boundary), sources);
        assert_eq!(
            operation.region_data_flow().input_sources(&operation, 1, 2, boundary),
            Err(ProgramError::MalformedProgram(
                "operation `while` declares input 2 of region 1 with 2 inputs".to_owned()
            )),
        );

        // The rule itself rejects a boundary that does not share one state signature across the instruction and both
        // regions: a missing region, a condition or a body whose inputs do not match the instruction inputs.
        let malformed = Err(ProgramError::MalformedProgram(
            "`while` region data flow requires matching state boundaries and one condition output".to_owned(),
        ));
        assert_eq!(
            operation.input_sources(0, 1, RegionDataFlowBoundary { regions: &regions[..1], ..boundary }),
            malformed,
        );
        let condition_regions = [
            RegionDataFlowRegionBoundary { input_count: 1, output_count: 1 },
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 2 },
        ];
        assert_eq!(
            operation.input_sources(0, 1, RegionDataFlowBoundary { regions: &condition_regions, ..boundary }),
            malformed,
        );
        let body_regions = [
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 1 },
            RegionDataFlowRegionBoundary { input_count: 1, output_count: 2 },
        ];
        assert_eq!(
            operation.input_sources(1, 1, RegionDataFlowBoundary { regions: &body_regions, ..boundary }),
            malformed,
        );
    }

    #[test]
    fn test_while_region_data_flow_output_sources() {
        // Each instruction output is sourced from the same-position body input, which holds the final state when the
        // loop exits, and an out-of-range output is rejected.
        let operation = WhileOperation::<ArrayType>::new();
        let regions = [
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 1 },
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 2 },
        ];
        let boundary = RegionDataFlowBoundary { input_count: 2, output_count: 2, regions: &regions };
        assert_eq!(
            operation.output_sources(1, boundary),
            Ok(RegionDataFlowSources::Known(vec![RegionDataFlowSource::RegionInput {
                region_index: 1,
                input_index: 1,
            }])),
        );
        assert_eq!(
            operation.region_data_flow().output_sources(&operation, 2, boundary),
            Err(ProgramError::MalformedProgram("operation `while` declares output 2 with 2 outputs".to_owned())),
        );

        // The rule itself rejects an instruction whose outputs do not match its inputs and a body whose outputs do not
        // match the instruction outputs.
        let malformed = Err(ProgramError::MalformedProgram(
            "`while` region data flow requires matching state boundaries and one condition output".to_owned(),
        ));
        assert_eq!(operation.output_sources(0, RegionDataFlowBoundary { output_count: 1, ..boundary }), malformed);
        let body_regions = [
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 1 },
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 1 },
        ];
        assert_eq!(
            operation.output_sources(0, RegionDataFlowBoundary { regions: &body_regions, ..boundary }),
            malformed,
        );
    }

    #[test]
    fn test_while_region_data_flow_execution_demands() {
        // The condition's predicate is an execution-only root and every state output stays demanded even when the
        // instruction is retained solely for effects. An iteration bound leaves these conservative roots intact. While
        // data flow never prunes its complete state boundary, so it never queries region liveness.
        let operation = WhileOperation::<ArrayType>::new();
        let regions = [
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 1 },
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 2 },
        ];
        let boundary = RegionDataFlowBoundary { input_count: 2, output_count: 2, regions: &regions };
        let mut liveness = TestRegionLiveness::new(|_: &[bool]| -> Vec<bool> {
            panic!("while data flow must retain the complete state boundary")
        });
        assert_eq!(
            operation.execution_demands(&[false, true], boundary, &mut liveness),
            Ok(vec![Some(vec![true]), Some(vec![true, true])]),
        );
        assert_eq!(
            operation
                .with_iteration_bound(3)
                .unwrap()
                .execution_demands(&[false, false], boundary, &mut liveness,),
            Ok(vec![Some(vec![true]), Some(vec![true, true])]),
        );
        assert_eq!(
            operation.region_data_flow().execution_demands(&operation, &[true], boundary, &mut liveness),
            Err(ProgramError::MalformedProgram(
                "operation `while` receives used outputs for 1 positions but has 2 outputs".to_owned(),
            )),
        );
        let malformed_regions = [
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 2 },
            RegionDataFlowRegionBoundary { input_count: 2, output_count: 2 },
        ];
        assert_eq!(
            operation.execution_demands(
                &[true, true],
                RegionDataFlowBoundary { regions: &malformed_regions, ..boundary },
                &mut liveness,
            ),
            Err(ProgramError::MalformedProgram(
                "`while` region data flow requires matching state boundaries and one condition output".to_owned(),
            )),
        );
    }

    #[test]
    fn test_while_interpretation() {
        // Eager interpretation applies the body while the condition holds, and returns the initial state unchanged when
        // the condition is false on entry.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let regions = vec![greater_than_zero_condition(), subtract_one_body()];
        assert_eq!(
            context.bind(WhileOperation::new(), regions.clone(), &[Array::scalar(3.0).unwrap()]),
            Ok(vec![Array::scalar(0.0).unwrap()]),
        );
        assert_eq!(
            context.bind(WhileOperation::new(), regions.clone(), &[Array::scalar(-1.0).unwrap()]),
            Ok(vec![Array::scalar(-1.0).unwrap()]),
        );

        // A bounded loop runs at most `bound` iterations by definition: a bound of 2 truncates the count down from 5 at
        // 3 even though the condition still holds, and leaves a loop that exits earlier unaffected.
        let bounded = WhileOperation::new().with_iteration_bound(2).unwrap();
        assert_eq!(
            context.bind(bounded, regions.clone(), &[Array::scalar(5.0).unwrap()]),
            Ok(vec![Array::scalar(3.0).unwrap()]),
        );
        assert_eq!(
            context.bind(bounded, regions, &[Array::scalar(1.0).unwrap()]),
            Ok(vec![Array::scalar(0.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_interpretation_masks_finished_items_under_batched_predicates() {
        // A `bool[3]` predicate over an `f64[3]` state satisfies the predicate-prefix rule, and interpretation runs
        // the masked loop: it continues while any per-item predicate is true, and items whose predicate is false
        // keep their carried state. Items [3, 1, 2] count down independently, terminating after 3, 1, and 2
        // iterations.
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::GreaterThan),
                    Vec::new(),
                    vec![state, zero],
                    None,
                )
                .unwrap()[0];
            builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let state = builder.add_input(state_type.clone());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            let next_state =
                builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
            builder.build(vec![next_state], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let operation = WhileOperation::new();
        assert_eq!(condition.output_types(), vec![ArrayType::new_static(DataType::Boolean, [3])]);

        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let outputs = context
            .bind(operation, vec![condition.clone(), body.clone()], &[Array::vector(vec![3.0, 1.0, 2.0]).unwrap()])
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![0.0, 0.0, 0.0]);

        // The semantic iteration bound truncates the shared masked iterations: item 0 stops at 1.0 after two body
        // applications, item 1 finishes on its own predicate after one, and item 2 reaches 0 exactly at the bound.
        let bounded = operation.with_iteration_bound(2).unwrap();
        let outputs = context
            .bind(bounded, vec![condition, body], &[Array::vector(vec![3.0, 1.0, 2.0]).unwrap()])
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_while_interpretation_passes_dimension_carries_through_batched_predicates() {
        // A batched predicate masks every array carry per item, while a first-class dimension carry has no per-item
        // value to mask: the body must pass it through unchanged, which makes masking it the identity. The loop carries
        // `[extent, x]` with `x: f32[2]`, runs while any item of `x < 3` holds, and returns `[extent, x + 1]`, so the
        // items of `[0, 2]` both stop at `3`, after three and one iterations. A dimension carry can never take a
        // different value, under any predicate: a symbolic carry's type is defined by the body input alone, so any
        // other body output of that type would define its identity twice, and a static carry has a single value.
        let dimension_type = DimensionType::from(Dimension::Static(2));
        let vector_type = ArrayType::new_static(DataType::F32, [2]);
        let state_types = vec![ArrayIrType::Dimension(dimension_type.clone()), ArrayIrType::Array(vector_type)];
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_inputs =
            state_types.iter().map(|r#type| condition_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let limit = condition_builder.add_constant(array(Array::vector(vec![3f32, 3.0]).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![condition_inputs[1], limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let body = |replace_extent: bool| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
            let one = builder.add_constant(array(Array::vector(vec![1f32, 1.0]).unwrap()));
            let next = builder
                .add_instruction(
                    TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                    Vec::new(),
                    vec![inputs[1], one],
                    None,
                )
                .unwrap()[0];
            let extent = match replace_extent {
                true => builder.add_constant(dimension(&dimension_type, 2)),
                false => inputs[0],
            };
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![extent, next],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let inputs = [dimension(&dimension_type, 2), array(Array::vector(vec![0f32, 2.0]).unwrap())];
        assert_eq!(
            context.bind(WhileOperation::<ArrayIrType>::new(), vec![condition.clone(), body(false)], &inputs),
            Ok(vec![dimension(&dimension_type, 2), array(Array::vector(vec![3f32, 3.0]).unwrap())]),
        );

        // A body that replaces the dimension carry, even by an equal constant, does not pass it through, so the
        // application is rejected before the first iteration (refer to `WhileOperation::validate_application`).
        assert_eq!(
            context.bind(WhileOperation::<ArrayIrType>::new(), vec![condition, body(true)], &inputs),
            Err(ProgramError::from(TypeError::invalid(
                "`while` state 0 of type `dimension<2>` must be passed through unchanged by the body under a batched \
                 predicate, because it has no per-item value to mask",
            ))),
        );
    }

    #[test]
    fn test_while_interpretation_allocates_fresh_local_references_per_invocation() {
        // Both regions allocate a local reference on every invocation. The body initializes its reference from a
        // constant and accumulates the carried state into it, so an allocation that persisted across iterations would
        // accumulate into a stale value and diverge from `state + 1` after the first iteration. The carried predicates
        // shift by one position per iteration, so the loop runs exactly three times.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let boolean_type = ArrayType::scalar(DataType::Boolean);
        let condition = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let state = builder.add_input(array_type.clone().into());
            let predicate = builder.add_input(boolean_type.clone().into());
            builder.add_input(boolean_type.clone().into());
            builder.add_input(boolean_type.clone().into());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 4], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let state = builder.add_input(array_type.clone().into());
            builder.add_input(boolean_type.clone().into());
            let second = builder.add_input(boolean_type.clone().into());
            let third = builder.add_input(boolean_type.clone().into());
            let one = builder.add_constant(TestIrValue::Array(Array::vector(vec![1.0f32, 1.0]).unwrap()));
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![one], None).unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, state], None)
                .unwrap();
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let done = builder.add_constant(TestIrValue::Array(Array::scalar(false).unwrap()));
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![state, second, third, done],
                    vec![Placeholder; 4],
                    vec![Placeholder; 4],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let inputs = vec![
            builder.add_input(array_type.into()),
            builder.add_input(boolean_type.clone().into()),
            builder.add_input(boolean_type.clone().into()),
            builder.add_input(boolean_type.into()),
        ];
        let outputs = builder
            .add_instruction(
                TestIrOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                inputs,
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 4], vec![Placeholder; 4])
            .unwrap();
        let predicate = |value| TestIrValue::Array(Array::scalar(value).unwrap());
        assert_eq!(
            program.interpret(vec![
                TestIrValue::Array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
                predicate(true),
                predicate(true),
                predicate(true),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::vector(vec![4.0f32, 5.0]).unwrap()),
                predicate(false),
                predicate(false),
                predicate(false),
            ]),
        );
    }

    #[test]
    fn test_while_interpretation_rejects_bodies_that_swap_reference_carries() {
        // A body that returns each of two reference carries at the other's position gives the loop state no fixed
        // point, so eager interpretation rejects it before any iteration runs, exactly as reference discharge does,
        // including when both carries enter as one allocation (refer to
        // `test_while_transforms_agree_on_aliased_reference_carries`).
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut condition = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition.add_input(reference_type.clone().into());
        condition.add_input(reference_type.clone().into());
        let predicate = condition.add_constant(TestIrValue::Array(Array::scalar(false).unwrap()));
        let condition = condition
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = body.add_input(reference_type.clone().into());
        let second = body.add_input(reference_type.clone().into());
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![second, first],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let error = TypeError::invalid(
            "`while` body does not return carry 0 as the reference it entered with, so the loop state has no fixed \
             point",
        );
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let first = TestIrValue::Reference(ArrayReference::new(Array::scalar(1f32).unwrap()));
        let second = TestIrValue::Reference(ArrayReference::new(Array::scalar(2f32).unwrap()));
        assert_eq!(
            context.bind(
                WhileOperation::<ArrayIrType>::new(),
                vec![condition.clone(), body.clone()],
                &[first.clone(), second],
            ),
            Err(ProgramError::from(error.clone())),
        );
        assert_eq!(
            context.bind(
                WhileOperation::<ArrayIrType>::new(),
                vec![condition.clone(), body.clone()],
                &[first.clone(), first],
            ),
            Err(ProgramError::from(error.clone())),
        );

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let first = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let second = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let first = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        let second = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![second], None).unwrap()[0];
        let outputs = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), vec![condition, body], vec![first, second], None)
            .unwrap()
            .to_vec();
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[0]], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(program.discharge_references(0).map(|_| ()), Err(ProgramError::from(error)));
    }

    #[test]
    fn test_while_reference_discharge() {
        // A loop applies no read-only pruning. The condition merely reads the allocation and the body never touches it,
        // yet the allocation keeps a carry position in every boundary, because a carry must exist in the condition's
        // and the body's boundaries or in neither. The asymmetry that the operation's own contract forces is visible
        // beside it: the condition region receives the entering state and publishes none, returning exactly one
        // Boolean.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = condition_builder.add_input(scalar_type.clone().into());
        let reference = condition_builder.add_input(reference_type.clone().into());
        let threshold = condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan)),
                Vec::new(),
                vec![counter, threshold],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = body_builder.add_input(scalar_type.clone().into());
        let reference = body_builder.add_input(reference_type.clone().into());
        let step = body_builder.add_constant(TestIrValue::Array(Array::scalar(-1.0f32).unwrap()));
        let next_counter = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, step], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_counter, reference],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_program(condition);
        let body = builder.import_program(body);
        let reference = builder.add_input(reference_type.clone().into());
        let counter = builder.add_input(scalar_type.clone().into());
        let final_counter = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new(),
                vec![condition, body],
                vec![counter, reference],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![final_counter], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[] = while %1 %0 [
                    condition={
                        lambda %0:f32[], %1:f32[] .
                        let %2:bool[] = compare [direction=GreaterThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = const -1.0
                            %3:f32[] = add %0 %2
                        in (%3, %1)
                    },
                ]
                in (%2)"},
        );

        // The carry survives inside the loop, but the *entry* boundary is a different question: nothing writes the
        // allocation, so it publishes no hidden final-state output and its caller's holder is left alone. Symmetry is a
        // property of the loop's own boundaries, not a claim that the loop wrote what it carried.
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.program().output_types().len(), 1);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(5.0f32).unwrap())
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar(2.0f32).unwrap())]),
        );
    }

    #[test]
    fn test_while_reference_discharge_threads_a_preserved_carry_through_a_loop() {
        // A loop's boundaries stay symmetric with a preserved carry in them: the carry occupies its declared position
        // in the instruction input list, in both region boundaries, and in the output list, carrying a reference rather
        // than state, and the loop publishes no successor for it.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = condition_builder.add_input(reference_type.clone().into());
        condition_builder.add_input(reference_type.clone().into());
        let observed = condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![counter], None)
            .unwrap()[0];
        let limit = condition_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![observed, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = body_builder.add_input(reference_type.clone().into());
        let step = body_builder.add_input(reference_type.clone().into());
        let increment =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![step], None).unwrap()[0];
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![counter, increment], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![counter, step],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let counter_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let step_initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let counter = builder
            .add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![counter_initial], None)
            .unwrap()[0];
        let step =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![step_initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(Some(8)).unwrap();
        let carried = builder.add_instruction(operation, vec![condition, body], vec![counter, step], None).unwrap();
        let (carried_counter, carried_step) = (carried[0], carried[1]);
        let total = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![carried_counter], None)
            .unwrap()[0];
        let remaining = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![carried_step], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![total, remaining],
                vec![Placeholder; 2],
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
                lambda %0:f32[], %1:f32[] .
                let %2:ref<f32[]> = reference_new %1
                    %3:f32[], %4:ref<f32[]> = while [iteration_bound=8] %0 %2 [
                        condition={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = const 3.0
                                %3:bool[] = compare [direction=LessThan] %0 %2
                            in (%3)
                        },
                        body={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = reference_read %1
                                %3:f32[] = add %0 %2
                            in (%3, %1)
                        },
                    ]
                    %5:f32[] = reference_freeze %2
                in (%3, %5)"},
        );

        // The eager interpreter carries a reference through the loop by selecting the handle wholesale under the scalar
        // predicate, so the mixed program runs exactly like the source it came from: the counter accumulates
        // `0 + 2 + 2` before the condition fails, and the untouched step still holds its initial value.
        let inputs = vec![
            TestIrValue::Array(Array::scalar::<f32>(0.0).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(2.0).unwrap()),
        ];
        let expected = Ok(vec![
            TestIrValue::Array(Array::scalar::<f32>(4.0).unwrap()),
            TestIrValue::Array(Array::scalar::<f32>(2.0).unwrap()),
        ]);
        assert_eq!(source.interpret(inputs.clone()), expected);
        assert_eq!(discharged.program().interpret(inputs), expected);
    }

    #[test]
    fn test_while_reference_discharge_nested_condition_keeps_local_state_inside_its_creation_scope() {
        // An allocation that is created and frozen inside the true branch of a `condition` crosses no entry boundary,
        // even though a loop nested in that branch mutates it, and the false branch never sees the allocation at all.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let predicate = condition_builder.add_constant(TestIrValue::Array(Array::scalar(true).unwrap()));
        let loop_condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let loop_body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut true_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let loop_condition = true_builder.import_region(loop_condition.entry_region_ref());
        let loop_body = true_builder.import_region(loop_body.entry_region_ref());
        let initial = true_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference =
            true_builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
        let reference = true_builder
            .add_instruction(operation, vec![loop_condition, loop_body], vec![reference], None)
            .unwrap()[0];
        let value = true_builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let true_branch = true_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut false_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let value = false_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let false_branch = false_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrType>::new(),
                vec![true_branch, false_branch],
                vec![predicate, initial],
                None,
            )
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:bool[], %1:f32[] .
                let %2:f32[] = condition %0 %1 [
                    true={
                        lambda %0:f32[] .
                        let %1:f32[] = while [iteration_bound=2] %0 [
                            condition={
                                lambda %0:f32[] .
                                let %1:bool[] = const true
                                in (%1)
                            },
                            body={
                                lambda %0:f32[] .
                                let %1:f32[] = const 1.0
                                    %2:f32[] = add %0 %1
                                in (%2)
                            },
                        ]
                        in (%1)
                    },
                    false={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar(true).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap())
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar::<f32>(5.0).unwrap())])
        );
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar(false).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap())
            ]),
            Ok(vec![TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap())])
        );
    }

    #[test]
    fn test_while_reference_discharge_preserves_zero_iteration_and_threads_mutated_state() {
        // The external reference carry becomes an ordinary array carry whose final state is also published as the
        // hidden output: the condition region observes the state without returning it, while the body returns the
        // accumulated state in the carry position. A predicate that is immediately false leaves the entering state
        // untouched in both the public read and the appended final state.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let condition = condition_builder.add_constant(TestIrValue::Array(Array::scalar(false).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![condition], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let reference = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), vec![condition, body], vec![reference], None)
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = while %0 [
                    condition={
                        lambda %0:f32[] .
                        let %1:bool[] = const false
                        in (%1)
                    },
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = const 1.0
                            %2:f32[] = add %0 %1
                        in (%2)
                    },
                ]
                in (%1, %1)"},
        );
        assert_eq!(
            discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(2.0).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(2.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(2.0).unwrap())
            ])
        );
    }

    #[test]
    fn test_while_reference_discharge_accepts_the_same_allocation_at_repeated_carry_positions() {
        // Each declared carry position still exists in the rebuilt loop even though both source inputs name one
        // allocation. The allocation has one canonical state, so either carry observes the same update and only one
        // hidden final state is published.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first_reference = condition_builder.add_input(reference_type.clone().into());
        let second_reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first_reference], None)
            .unwrap();
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second_reference], None)
            .unwrap();
        let condition = condition_builder.add_constant(TestIrValue::Array(Array::scalar(true).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![condition], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first_reference = body_builder.add_input(reference_type.clone().into());
        let second_reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![second_reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![first_reference, second_reference],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(Some(1)).unwrap();
        let outputs =
            builder.add_instruction(operation, vec![condition, body], vec![reference, reference], None).unwrap();
        let first_reference = outputs[0];
        let second_reference = outputs[1];
        let first_value = builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first_reference], None)
            .unwrap()[0];
        let second_value = builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second_reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![first_value, second_value],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[], %2:f32[] = while [iteration_bound=1] %0 %0 [
                    condition={
                        lambda %0:f32[], %1:f32[] .
                        let %2:bool[] = const true
                        in (%2)
                    },
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = const 1.0
                            %3:f32[] = add %0 %2
                        in (%3, %3)
                    },
                ]
                in (%2, %2, %2)"},
        );
        assert_eq!(
            discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(2.0).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap())
            ]),
        );
    }

    #[test]
    fn test_while_reference_discharge_rotates_an_initially_false_mutating_condition() {
        // The condition increments the counter and compares it against zero, so from a non-negative start it is false
        // on its very first evaluation. Rotation discharges that evaluation once before the loop, threads its predicate
        // as a trailing carry, and runs the body followed by the condition inside the rebuilt body, so the condition's
        // instructions appear exactly twice and its increment happens exactly once when the body never runs.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(incrementing_condition(0.0).entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let reference = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), vec![condition, body], vec![reference], None)
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert!(discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(1));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = const 1.0
                    %2:f32[] = const 0.0
                    %3:f32[] = add %0 %1
                    %4:bool[] = compare [direction=LessThan] %3 %2
                    %5:f32[], %6:bool[] = while %3 %4 [
                        condition={
                            lambda %0:f32[], %1:bool[] .
                            in (%1)
                        },
                        body={
                            lambda %0:f32[], %1:bool[] .
                            let %2:f32[] = const 10.0
                                %3:f32[] = add %0 %2
                                %4:f32[] = const 1.0
                                %5:f32[] = const 0.0
                                %6:f32[] = add %3 %4
                                %7:bool[] = compare [direction=LessThan] %6 %5
                            in (%6, %7)
                        },
                    ]
                in (%5, %5)"},
        );

        // From 0, zero body iterations still keep the single increment. From -5, the first evaluation tests -4 and runs
        // the body once (reaching 6), after which the second condition evaluation increments to 7 and exits.
        assert_eq!(
            discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(0.0).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap())
            ])
        );
        assert_eq!(
            discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(-5.0).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(7.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(7.0).unwrap())
            ])
        );
    }

    #[test]
    fn test_while_reference_discharge_matches_hand_rotated_do_while_for_a_condition_only_mutation() {
        // Only the condition mutates the counter reference, while the body doubles an ordinary value carry. The oracle
        // is the reference-free do-while loop written by hand: the condition evaluated once up front, the predicate
        // carried, and the body followed by the condition inside the loop.
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let accumulator = body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let doubled = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![accumulator, accumulator], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, doubled],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = condition_builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        condition_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let condition = condition_builder.splice_program(&incrementing_condition(5.0), &[reference]).unwrap();
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(condition, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let accumulator = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let outputs = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new(),
                vec![condition, body],
                vec![reference, accumulator],
                None,
            )
            .unwrap()
            .to_vec();
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![outputs[0]], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![frozen, outputs[1]],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut oracle_condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        oracle_condition_builder.add_input(ArrayType::scalar(DataType::F32).into());
        oracle_condition_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let predicate = oracle_condition_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let oracle_condition = oracle_condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut oracle_body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = oracle_body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let accumulator = oracle_body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        oracle_body_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let doubled = oracle_body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![accumulator, accumulator], None)
            .unwrap()[0];
        let step = oracle_body_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()));
        let counter = oracle_body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, step], None)
            .unwrap()[0];
        let limit = oracle_body_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(5.0).unwrap()));
        let predicate = oracle_body_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let oracle_body = oracle_body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![counter, doubled, predicate],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let mut oracle_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let oracle_condition = oracle_builder.import_region(oracle_condition.entry_region_ref());
        let oracle_body = oracle_builder.import_region(oracle_body.entry_region_ref());
        let initial = oracle_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let accumulator = oracle_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let step = oracle_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()));
        let counter = oracle_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![initial, step], None)
            .unwrap()[0];
        let limit = oracle_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(5.0).unwrap()));
        let predicate = oracle_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let outputs = oracle_builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new(),
                vec![oracle_condition, oracle_body],
                vec![counter, accumulator, predicate],
                None,
            )
            .unwrap()
            .to_vec();
        let oracle = oracle_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![outputs[0], outputs[1]],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = const 1.0
                    %3:f32[] = const 5.0
                    %4:f32[] = add %0 %2
                    %5:bool[] = compare [direction=LessThan] %4 %3
                    %6:f32[], %7:f32[], %8:bool[] = while %4 %1 %5 [
                        condition={
                            lambda %0:f32[], %1:f32[], %2:bool[] .
                            in (%2)
                        },
                        body={
                            lambda %0:f32[], %1:f32[], %2:bool[] .
                            let %3:f32[] = add %1 %1
                                %4:f32[] = const 1.0
                                %5:f32[] = const 5.0
                                %6:f32[] = add %0 %4
                                %7:bool[] = compare [direction=LessThan] %6 %5
                            in (%6, %3, %7)
                        },
                    ]
                in (%6, %7)"},
        );
        for initial in [0.0f32, 3.0, 4.0, 7.0, -2.0] {
            let expected = oracle
                .clone()
                .interpret(vec![
                    TestIrValue::Array(Array::scalar::<f32>(initial).unwrap()),
                    TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()),
                ])
                .unwrap();
            assert_eq!(
                discharged.program().interpret(vec![
                    TestIrValue::Array(Array::scalar::<f32>(initial).unwrap()),
                    TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap())
                ]),
                Ok(expected)
            );
        }
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar::<f32>(0.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap())
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(5.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(16.0).unwrap())
            ])
        );
        assert_eq!(
            discharged.program().interpret(vec![
                TestIrValue::Array(Array::scalar::<f32>(7.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap())
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(8.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap())
            ])
        );
    }

    #[test]
    fn test_while_reference_discharge_rotates_a_root_written_by_both_condition_and_body() {
        // The body doubles the carried counter and the condition increments it, so the rotation's fixed body-then-
        // condition order is observable in the final value. The oracle is the hand-rotated reference-free loop.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let counter = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let doubled = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, counter], None)
            .unwrap()[0];
        body_builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, doubled], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(incrementing_condition(10.0).entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let reference = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), vec![condition, body], vec![reference], None)
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut oracle_condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        oracle_condition_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let predicate = oracle_condition_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let oracle_condition = oracle_condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut oracle_body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = oracle_body_builder.add_input(ArrayType::scalar(DataType::F32).into());
        oracle_body_builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let doubled = oracle_body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, counter], None)
            .unwrap()[0];
        let step = oracle_body_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()));
        let counter = oracle_body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![doubled, step], None)
            .unwrap()[0];
        let limit = oracle_body_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()));
        let predicate = oracle_body_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let oracle_body = oracle_body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![counter, predicate],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut oracle_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let oracle_condition = oracle_builder.import_region(oracle_condition.entry_region_ref());
        let oracle_body = oracle_builder.import_region(oracle_body.entry_region_ref());
        let initial = oracle_builder.add_input(ArrayType::scalar(DataType::F32).into());
        let step = oracle_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()));
        let counter = oracle_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![initial, step], None)
            .unwrap()[0];
        let limit = oracle_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(10.0).unwrap()));
        let predicate = oracle_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let counter = oracle_builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new(),
                vec![oracle_condition, oracle_body],
                vec![counter, predicate],
                None,
            )
            .unwrap()[0];
        let oracle = oracle_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![counter, counter],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert!(discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = const 1.0
                    %2:f32[] = const 10.0
                    %3:f32[] = add %0 %1
                    %4:bool[] = compare [direction=LessThan] %3 %2
                    %5:f32[], %6:bool[] = while %3 %4 [
                        condition={
                            lambda %0:f32[], %1:bool[] .
                            in (%1)
                        },
                        body={
                            lambda %0:f32[], %1:bool[] .
                            let %2:f32[] = add %0 %0
                                %3:f32[] = const 1.0
                                %4:f32[] = const 10.0
                                %5:f32[] = add %2 %3
                                %6:bool[] = compare [direction=LessThan] %5 %4
                            in (%5, %6)
                        },
                    ]
                in (%5, %5)"},
        );
        for initial in [0.0f32, 4.0, 9.0, 20.0] {
            let expected =
                oracle.clone().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(initial).unwrap())]).unwrap();
            assert_eq!(
                discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(initial).unwrap())]),
                Ok(expected)
            );
        }
        assert_eq!(
            discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(0.0).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(15.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(15.0).unwrap())
            ])
        );
    }

    #[test]
    fn test_while_reference_discharge_rotates_a_mutated_reference_capture_inside_the_condition() {
        // A capture the condition accumulates into reaches the loop only through a synthesized carry. Rotation
        // discharges the condition once in the parent, where the capture's state is updated in place, and again at the
        // tail of the body, where the published state replaces the carried one. The capture value family carries no
        // data, so the rendered program pins the state flow.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let reference = condition_builder.add_constant(DischargeCapture::new(0, reference_type.into()));
        let update = condition_builder.add_constant(DischargeCapture::new(1, ArrayType::scalar(DataType::F32).into()));
        condition_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let predicate =
            condition_builder.add_constant(DischargeCapture::new(2, ArrayType::scalar(DataType::Boolean).into()));
        let condition = condition_builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![predicate], Vec::new(), vec![Placeholder])
            .unwrap();
        let body = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new()
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), vec![condition, body], Vec::new(), None)
            .unwrap();
        let while_program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        let closed = ClosedProgram::new(
            while_program,
            vec![
                ArrayIrValue::Reference(ArrayReference::new(Array::scalar(4.0f32).unwrap())),
                TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()),
                TestIrValue::Array(Array::scalar(false).unwrap()),
            ],
        )
        .unwrap();

        let discharged = closed.discharge_references().unwrap();
        assert_eq!(discharged.output_count(), 0);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Capture { index: 0 });
        assert!(discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(0));
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:bool[] .
                let %3:f32[] = const capture#1:f32[]
                    %4:bool[] = const capture#2:bool[]
                    %5:f32[] = add %0 %3
                    %6:f32[], %7:bool[] = while %5 %4 [
                        condition={
                            lambda %0:f32[], %1:bool[] .
                            in (%1)
                        },
                        body={
                            lambda %0:f32[], %1:bool[] .
                            let %2:f32[] = const capture#1:f32[]
                                %3:bool[] = const capture#2:bool[]
                                %4:f32[] = add %0 %2
                            in (%4, %3)
                        },
                    ]
                in (%6)"},
        );
    }

    #[test]
    fn test_while_reference_discharge_rotates_an_aliased_root_mutated_inside_the_condition() {
        // Both carry positions name one allocation. The condition increments it through the first position and tests it
        // through the second, so the rotation must keep the two positions in agreement after every evaluation.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first_reference = condition_builder.add_input(reference_type.clone().into());
        let second_reference = condition_builder.add_input(reference_type.clone().into());
        let update = condition_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap()));
        condition_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![first_reference, update], None)
            .unwrap();
        let counter = condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second_reference], None)
            .unwrap()[0];
        let limit = condition_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first_reference = body_builder.add_input(reference_type.clone().into());
        let second_reference = body_builder.add_input(reference_type.clone().into());
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![first_reference, second_reference],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let outputs = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new(),
                vec![condition, body],
                vec![reference, reference],
                None,
            )
            .unwrap()
            .to_vec();
        let first_value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[0]], None).unwrap()[0];
        let second_value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[1]], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![first_value, second_value],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let discharged = source.discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert!(discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = const 1.0
                    %2:f32[] = const 3.0
                    %3:f32[] = add %0 %1
                    %4:bool[] = compare [direction=LessThan] %3 %2
                    %5:f32[], %6:f32[], %7:bool[] = while %3 %3 %4 [
                        condition={
                            lambda %0:f32[], %1:f32[], %2:bool[] .
                            in (%2)
                        },
                        body={
                            lambda %0:f32[], %1:f32[], %2:bool[] .
                            let %3:f32[] = const 1.0
                                %4:f32[] = const 3.0
                                %5:f32[] = add %0 %3
                                %6:bool[] = compare [direction=LessThan] %5 %4
                            in (%5, %5, %6)
                        },
                    ]
                in (%6, %6, %6)"},
        );
        assert_eq!(
            discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(0.0).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(3.0).unwrap())
            ])
        );
        assert_eq!(
            discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(7.0).unwrap())]),
            Ok(vec![
                TestIrValue::Array(Array::scalar::<f32>(8.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(8.0).unwrap()),
                TestIrValue::Array(Array::scalar::<f32>(8.0).unwrap())
            ])
        );
    }

    #[test]
    fn test_while_reference_discharge_rejects_a_consuming_condition() {
        // A condition borrows its entering reference and cannot consume it. The shared reference analysis reports the
        // consuming instruction, the region input it consumes, and the external input that the region input aliases.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = condition_builder.add_input(reference_type.into());
        condition_builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let predicate = condition_builder.add_constant(TestIrValue::Array(Array::scalar(true).unwrap()));
        let consuming_condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let source = reference_while_program(consuming_condition, None);
        let condition = source.entry_region_ref().instructions()[0].regions()[0];
        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ReferenceAnalysisError::ExternalReferenceConsumption {
                operation: "reference_freeze",
                instruction: InstructionId::new(condition, 0),
                root: ReferenceRoot::RegionInput { region: condition, input_index: 0 },
                external_source: ReferenceSource::Input { index: 0 },
            }
            .into(),
        );
    }

    #[test]
    fn test_while_reference_discharge_rotates_a_bounded_mutating_condition() {
        // A bounded loop evaluates its condition exactly `bound` times when the bound truncates it, whereas every
        // rotated iteration evaluates the condition after the body. The rotated loop therefore runs one iteration
        // fewer and the last body application runs after the loop under the final predicate; a bound of one leaves
        // no loop at all. The body doubles the counter that the condition increments before testing it against 10,
        // so from 0 the source tests 1, 3, 7, and 15 and doubles to 2, 6, and 14: a bound of 1 truncates it at 2, a
        // bound of 2 truncates it at 6, and a bound of 5 lets it exit at 15. A trailing step that skipped or repeated
        // the last body application would change each of these values.
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.clone());
        let counter = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let doubled = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, counter], None)
            .unwrap()[0];
        body_builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, doubled], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let source = |bound: usize| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let regions =
                vec![builder.import_program(incrementing_condition(10.0)), builder.import_program(body.clone())];
            let reference = builder.add_input(reference_type.clone());
            let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(bound).unwrap();
            let reference = builder.add_instruction(operation, regions, vec![reference], None).unwrap()[0];
            let value =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        for (bound, expected) in [(1, 2.0f32), (2, 6.0), (5, 15.0)] {
            let source = source(bound);
            let initial = ArrayReference::new(Array::scalar(0f32).unwrap());
            assert_eq!(
                source.interpret(vec![TestIrValue::Reference(initial.clone())]),
                Ok(vec![TestIrValue::Array(Array::scalar(expected).unwrap())]),
            );
            assert_eq!(initial.read().unwrap(), Array::scalar(expected).unwrap());
            let discharged = source.discharge_references(0).unwrap();
            assert_eq!(
                discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(0.0).unwrap())]),
                Ok(vec![
                    TestIrValue::Array(Array::scalar(expected).unwrap()),
                    TestIrValue::Array(Array::scalar(expected).unwrap()),
                ]),
            );
        }
        // A bound of one leaves no rotated loop: the condition runs once in the parent and the only body application
        // is the trailing one. A bound of two rotates one iteration into the loop before that trailing application.
        assert_eq!(
            source(1).discharge_references(0).unwrap().program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = const 1.0
                    %2:f32[] = const 10.0
                    %3:f32[] = add %0 %1
                    %4:bool[] = compare [direction=LessThan] %3 %2
                    %5:f32[] = condition %4 %3 [
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
                in (%5, %5)"},
        );
        assert_eq!(
            source(2).discharge_references(0).unwrap().program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = const 1.0
                    %2:f32[] = const 10.0
                    %3:f32[] = add %0 %1
                    %4:bool[] = compare [direction=LessThan] %3 %2
                    %5:f32[], %6:bool[] = while [iteration_bound=1] %3 %4 [
                        condition={
                            lambda %0:f32[], %1:bool[] .
                            in (%1)
                        },
                        body={
                            lambda %0:f32[], %1:bool[] .
                            let %2:f32[] = add %0 %0
                                %3:f32[] = const 1.0
                                %4:f32[] = const 10.0
                                %5:f32[] = add %2 %3
                                %6:bool[] = compare [direction=LessThan] %5 %4
                            in (%5, %6)
                        },
                    ]
                    %7:f32[] = condition %6 %5 [
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
                in (%7, %7)"},
        );
    }

    #[test]
    fn test_while_reference_discharge_rotates_a_condition_that_mutates_preserved_and_discharged_references() {
        // The condition increments both carried allocations, but only the first one is discharged. The loop still
        // rotates to publish the discharged state, while the writes to the preserved reference stay explicit effects in
        // both copies of the condition and its carry publishes no state. From `[0, 10]` the condition tests the first
        // counter at 1, 2, and 3, so both counters end three increments later.
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = condition_builder.add_input(reference_type.clone());
        let second = condition_builder.add_input(reference_type.clone());
        let one = condition_builder.add_constant(array(Array::scalar(1.0f32).unwrap()));
        condition_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![first, one], None)
            .unwrap();
        condition_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![second, one], None)
            .unwrap();
        let counter = condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None)
            .unwrap()[0];
        let limit = condition_builder.add_constant(array(Array::scalar(3.0f32).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = body_builder.add_input(reference_type.clone());
        let second = body_builder.add_input(reference_type);
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![first, second],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let initial = (0..2).map(|_| builder.add_input(ArrayType::scalar(DataType::F32).into())).collect::<Vec<_>>();
        let references = initial
            .into_iter()
            .map(|initial| {
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), regions, references, None)
            .unwrap()
            .to_vec();
        let values = outputs
            .into_iter()
            .map(|output| {
                builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![output], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(values, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let targets = source.reference_discharge_targets(0).unwrap();
        let discharged = source.clone().partially_discharge_references(0, &targets[..1]).unwrap();
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:ref<f32[]> = reference_new %1
                    %3:f32[] = const 1.0
                    %4:f32[] = const 3.0
                    %5:f32[] = add %0 %3
                    () = reference_add_update %2 %3
                    %6:bool[] = compare [direction=LessThan] %5 %4
                    %7:f32[], %8:ref<f32[]>, %9:bool[] = while %5 %2 %6 [
                        condition={
                            lambda %0:f32[], %1:ref<f32[]>, %2:bool[] .
                            in (%2)
                        },
                        body={
                            lambda %0:f32[], %1:ref<f32[]>, %2:bool[] .
                            let %3:f32[] = const 1.0
                                %4:f32[] = const 3.0
                                %5:f32[] = add %0 %3
                                () = reference_add_update %1 %3
                                %6:bool[] = compare [direction=LessThan] %5 %4
                            in (%5, %1, %6)
                        },
                    ]
                    %10:f32[] = reference_freeze %2
                in (%7, %10)"},
        );
        let inputs = vec![array(Array::scalar(0.0f32).unwrap()), array(Array::scalar(10.0f32).unwrap())];
        let expected = Ok(vec![array(Array::scalar(3.0f32).unwrap()), array(Array::scalar(13.0f32).unwrap())]);
        assert_eq!(source.interpret(inputs.clone()), expected);
        assert_eq!(discharged.program().interpret(inputs), expected);
    }

    #[test]
    fn test_while_reference_discharge_rotated_loops_support_jvp_and_batching() {
        // Rotation threads the predicate as a trailing Boolean carry, which both transformations of the discharged loop
        // keep: it has no tangent in forward mode, and it becomes a batched predicate under batching. The body doubles
        // a local counter that the condition increments before testing it against 10, so the loop computes
        // `f(x) = 8 x + 15` from `x = 0.5` (testing 1.5, 4, 9, and 19) and `f(x) = 2 x + 3` from `x = 5` (testing 6
        // and 13).
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type);
        let counter = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let doubled = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, counter], None)
            .unwrap()[0];
        body_builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, doubled], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![builder.import_program(incrementing_condition(10.0)), builder.import_program(body)];
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let reference = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), regions, vec![reference], None)
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let discharged = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap()
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap();

        let jvp = discharged.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = const 1.0
                    %3:f32[] = const 10.0
                    %4:f32[] = add %0 %2
                    %5:bool[] = compare [direction=LessThan] %4 %3
                    %6:f32[], %7:bool[], %8:f32[] = while %4 %5 %1 [
                        condition={
                            lambda %0:f32[], %1:bool[], %2:f32[] .
                            in (%1)
                        },
                        body={
                            lambda %0:f32[], %1:bool[], %2:f32[] .
                            let %3:f32[] = const 1.0
                                %4:f32[] = const 10.0
                                %5:f32[] = add %0 %0
                                %6:f32[] = add %2 %2
                                %7:f32[] = add %5 %3
                                %8:bool[] = compare [direction=LessThan] %7 %4
                            in (%7, %8, %6)
                        },
                    ]
                in (%6, %8)"},
        );
        assert_eq!(
            jvp.interpret(vec![array(Array::scalar(0.5f32).unwrap()), array(Array::scalar(1.0f32).unwrap())]),
            Ok(vec![array(Array::scalar(19.0f32).unwrap()), array(Array::scalar(8.0f32).unwrap())]),
        );

        let axis_extent = DimensionValue::constant(2).unwrap();
        let batched = discharged
            .batched_with_threaded_extent(
                axis_extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[2] .
                let %2:f32[] = const 1.0
                    %3:f32[2] = broadcast [output_axes=[]] %2 %0
                    %4:f32[2] = add %1 %3
                    %5:f32[] = const 10.0
                    %6:f32[2] = broadcast [output_axes=[]] %5 %0
                    %7:bool[2] = compare [direction=LessThan] %4 %6
                    %8:dimension<2>, %9:f32[2], %10:bool[2] = while %0 %4 %7 [
                        condition={
                            lambda %0:dimension<2>, %1:f32[2], %2:bool[2] .
                            in (%2)
                        },
                        body={
                            lambda %0:dimension<2>, %1:f32[2], %2:bool[2] .
                            let %3:f32[2] = add %1 %1
                                %4:f32[] = const 1.0
                                %5:f32[2] = broadcast [output_axes=[]] %4 %0
                                %6:f32[2] = add %3 %5
                                %7:f32[] = const 10.0
                                %8:f32[2] = broadcast [output_axes=[]] %7 %0
                                %9:bool[2] = compare [direction=LessThan] %6 %8
                            in (%0, %6, %9)
                        },
                    ]
                in (%0, %9)"},
        );
        assert_eq!(
            batched.interpret(vec![
                TestIrValue::Dimension(axis_extent.clone()),
                array(Array::vector(vec![0.5f32, 5.0]).unwrap()),
            ]),
            Ok(vec![TestIrValue::Dimension(axis_extent), array(Array::vector(vec![19.0f32, 13.0]).unwrap())]),
        );
    }

    #[test]
    fn test_while_reference_discharge_preserves_a_reference_that_the_condition_mutates() {
        // Preserving the reference leaves the condition's writes explicit, so no rotation is needed even with a
        // bound. The condition runs twice and the same allocation remains available to its caller.
        let preserved = reference_while_program(incrementing_condition(3.0), Some(2))
            .partially_discharge_references(0, &[])
            .unwrap();
        assert_eq!(
            preserved.program().to_string(),
            indoc! {"
                lambda %0:ref<f32[]> .
                let %1:ref<f32[]> = while [iteration_bound=2] %0 [
                    condition={
                        lambda %0:ref<f32[]> .
                        let %1:f32[] = const 1.0
                            %2:f32[] = const 3.0
                            () = reference_add_update %0 %1
                            %3:f32[] = reference_read %0
                            %4:bool[] = compare [direction=LessThan] %3 %2
                        in (%4)
                    },
                    body={
                        lambda %0:ref<f32[]> .
                        in (%0)
                    },
                ]
                    %2:f32[] = reference_read %0
                in (%2)"},
        );
        let initial = ArrayReference::new(Array::scalar(0f32).unwrap());
        assert_eq!(
            preserved.program().interpret(vec![TestIrValue::Reference(initial.clone())]),
            Ok(vec![TestIrValue::Array(Array::scalar(2f32).unwrap())]),
        );
        assert_eq!(initial.read().unwrap(), Array::scalar(2f32).unwrap());
    }

    #[test]
    fn test_while_reference_discharge_threads_reference_captures_through_while() {
        // A capture read only by the loop condition still needs a synthesized state input on both while regions, but no
        // final-state output because nothing writes it.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let reference = condition_builder.add_constant(DischargeCapture::new(0, reference_type.into()));
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let predicate =
            condition_builder.add_constant(DischargeCapture::new(1, ArrayType::scalar(DataType::Boolean).into()));
        let condition = condition_builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![predicate], Vec::new(), vec![Placeholder])
            .unwrap();
        let body = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new()
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), vec![condition, body], Vec::new(), None)
            .unwrap();
        let while_program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        let concrete_reference = ArrayReference::new(Array::scalar(4.0f32).unwrap());
        let closed = ClosedProgram::new(
            while_program,
            vec![ArrayIrValue::Reference(concrete_reference), TestIrValue::Array(Array::scalar(false).unwrap())],
        )
        .unwrap();
        let discharged = closed.discharge_references().unwrap();
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:bool[] .
                let %2:f32[] = while %0 [
                    condition={
                        lambda %0:f32[] .
                        let %1:bool[] = const capture#1:bool[]
                        in (%1)
                    },
                    body={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in ()"},
        );
    }

    #[test]
    fn test_while_reference_discharge_matches_the_interpretation_of_the_source() {
        // Eager interpretation executes the reference-carrying source loop directly, selecting the reference carry
        // wholesale under its scalar predicate, so the discharged loop, which threads the same state through an
        // ordinary array carry, must agree with it for every initial state.
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let predicate = condition_builder.add_constant(TestIrValue::Array(Array::scalar(true).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(TestIrValue::Array(Array::scalar::<f32>(2.0).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(3).unwrap();
        let reference = builder.add_instruction(operation, vec![condition, body], vec![reference], None).unwrap()[0];
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The condition region observes the carried state while the body accumulates into it.
        let discharged = source.clone().discharge_references(0).unwrap();
        assert_eq!(discharged.external_reference_bindings(), &[]);
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = while [iteration_bound=3] %0 [
                    condition={
                        lambda %0:f32[] .
                        let %1:bool[] = const true
                        in (%1)
                    },
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = const 2.0
                            %2:f32[] = add %0 %1
                        in (%2)
                    },
                ]
                in (%1)"},
        );
        for initial in [0.0f32, 1.0, -4.5] {
            let expected = source.interpret(vec![TestIrValue::Array(Array::scalar::<f32>(initial).unwrap())]).unwrap();
            assert_eq!(
                discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(initial).unwrap())]),
                Ok(expected)
            );
        }
        assert_eq!(
            discharged.program().interpret(vec![TestIrValue::Array(Array::scalar::<f32>(1.0).unwrap())]),
            Ok(vec![TestIrValue::Array(Array::scalar::<f32>(7.0).unwrap())])
        );
    }

    #[test]
    fn test_while_reference_discharge_rotated_bounded_loops_support_jvp_and_batching() {
        // A bounded rotated loop ends with the trailing body application under the final predicate, which both
        // transformations of the discharged loop must keep. The body doubles a local counter that the condition
        // increments before testing it against 10, under a bound of 2. From `x = 0.5` the source tests 1.5 and 4 and
        // the bound truncates it after its second doubling, so it computes `f(x) = 4 x + 6 = 8`. From `x = 5` it tests
        // 6 and 13 and exits after one doubling, so it computes `f(x) = 2 x + 3 = 13`.
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type);
        let counter = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let doubled = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, counter], None)
            .unwrap()[0];
        body_builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, doubled], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![builder.import_program(incrementing_condition(10.0)), builder.import_program(body)];
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
        let reference = builder.add_instruction(operation, regions, vec![reference], None).unwrap()[0];
        let value =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let discharged =
            source.clone().discharge_references(0).unwrap().into_program_without_external_references().unwrap();
        assert_eq!(
            discharged.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = const 1.0
                    %2:f32[] = const 10.0
                    %3:f32[] = add %0 %1
                    %4:bool[] = compare [direction=LessThan] %3 %2
                    %5:f32[], %6:bool[] = while [iteration_bound=1] %3 %4 [
                        condition={
                            lambda %0:f32[], %1:bool[] .
                            in (%1)
                        },
                        body={
                            lambda %0:f32[], %1:bool[] .
                            let %2:f32[] = add %0 %0
                                %3:f32[] = const 1.0
                                %4:f32[] = const 10.0
                                %5:f32[] = add %2 %3
                                %6:bool[] = compare [direction=LessThan] %5 %4
                            in (%5, %6)
                        },
                    ]
                    %7:f32[] = condition %6 %5 [
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
                in (%7)"},
        );
        for (x, expected) in [(0.5f32, 8.0f32), (5.0, 13.0)] {
            let expected = Ok(vec![array(Array::scalar(expected).unwrap())]);
            assert_eq!(source.interpret(vec![array(Array::scalar(x).unwrap())]), expected);
            assert_eq!(discharged.interpret(vec![array(Array::scalar(x).unwrap())]), expected);
        }

        // Eager forward mode runs the rotated loop and its trailing step directly, so the tangent is the slope of the
        // affine piece that each input selects.
        for (x, expected_primal, expected_tangent) in [(0.5f32, 8.0f32, 4.0f32), (5.0, 13.0, 2.0)] {
            assert_eq!(
                differentiate_at(array(Array::scalar(x).unwrap())).jvp(array(Array::scalar(1.0f32).unwrap()), |x| {
                    let context = x.context().clone();
                    Ok(discharged.interpret_in_context(&context, vec![x])?.remove(0))
                }),
                Ok((array(Array::scalar(expected_primal).unwrap()), array(Array::scalar(expected_tangent).unwrap()))),
            );
        }

        // Batching makes the threaded predicate batched, so the trailing step selects the doubled state per item.
        let axis_extent = DimensionValue::constant(2).unwrap();
        let batched = discharged
            .batched_with_threaded_extent(
                axis_extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[2] .
                let %2:f32[] = const 1.0
                    %3:f32[2] = broadcast [output_axes=[]] %2 %0
                    %4:f32[2] = add %1 %3
                    %5:f32[] = const 10.0
                    %6:f32[2] = broadcast [output_axes=[]] %5 %0
                    %7:bool[2] = compare [direction=LessThan] %4 %6
                    %8:dimension<2>, %9:f32[2], %10:bool[2] = while [iteration_bound=1] %0 %4 %7 [
                        condition={
                            lambda %0:dimension<2>, %1:f32[2], %2:bool[2] .
                            in (%2)
                        },
                        body={
                            lambda %0:dimension<2>, %1:f32[2], %2:bool[2] .
                            let %3:f32[2] = add %1 %1
                                %4:f32[] = const 1.0
                                %5:f32[2] = broadcast [output_axes=[]] %4 %0
                                %6:f32[2] = add %3 %5
                                %7:f32[] = const 10.0
                                %8:f32[2] = broadcast [output_axes=[]] %7 %0
                                %9:bool[2] = compare [direction=LessThan] %6 %8
                            in (%0, %6, %9)
                        },
                    ]
                    %11:f32[2] = stop_gradient %9
                    %12:f32[2] = select %10 %9 %11
                    %13:f32[2] = add %12 %12
                    %14:f32[2] = select %10 %11 %9
                    %15:f32[2] = select %10 %13 %14
                in (%0, %15)"},
        );
        assert_eq!(
            batched.interpret(vec![
                TestIrValue::Dimension(axis_extent.clone()),
                array(Array::vector(vec![0.5f32, 5.0]).unwrap()),
            ]),
            Ok(vec![TestIrValue::Dimension(axis_extent), array(Array::vector(vec![8.0f32, 13.0]).unwrap())]),
        );
    }

    #[test]
    fn test_while_partial_evaluation() {
        // A loop-invariant known state element folds into the condition and the body: the residual while keeps the
        // same state set (so its output arity is preserved), and every subcomputation that depended only on the known
        // element collapses to a constant. The loop carries `[counter, acc, k]`, runs while `counter > 0`, and its body
        // returns `[counter - 1, acc + k * k, k]`. With `k` known (`3`) and `counter` and `acc` unknown, the body
        // passes `k` through unchanged, so `k * k` folds to `9`, the final `k` is rebuilt as the constant `3`, and the
        // bound carries over to the residual while.
        let scalar = || ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            builder.add_input(scalar());
            builder.add_input(scalar());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::GreaterThan),
                    Vec::new(),
                    vec![counter, zero],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            let acc = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
            let next_counter =
                builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
            let ksq = builder.add_instruction(MulOperation::new(), Vec::new(), vec![k, k], None).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation::new(), Vec::new(), vec![acc, ksq], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(
                    vec![next_counter, next_acc, k],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let inputs = (0..3).map(|_| builder.add_input(scalar())).collect::<Vec<_>>();
        let operation = WhileOperation::new().with_iteration_bound(8).unwrap();
        let outputs = builder.add_instruction(operation, regions, inputs, None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(scalar()),
                PartialValue::Unknown(scalar()),
                PartialValue::Known(Array::scalar(3.0).unwrap()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[], %4:f64[], %5:f64[] = while [iteration_bound=8] %0 %1 %2 [
                    condition={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = zero_like %0
                            %4:bool[] = compare [direction=GreaterThan] %0 %3
                        in (%4)
                    },
                    body={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = one_like %0
                            %4:f64[] = sub %0 %3
                            %5:f64[] = const 9.0
                            %6:f64[] = add %1 %5
                            %7:f64[] = const 3.0
                        in (%4, %6, %7)
                    },
                ]
                in (%3, %4, %5)"},
        );

        // The loop runs four times (counter `4 -> 0`), so `acc` threads `1 -> 10 -> 19 -> 28 -> 37` and the
        // loop-invariant `k` stays `3`, exactly as the source program computes.
        let expected = vec![Array::scalar(0.0).unwrap(), Array::scalar(37.0).unwrap(), Array::scalar(3.0).unwrap()];
        let inputs = vec![Array::scalar(4.0).unwrap(), Array::scalar(1.0).unwrap(), Array::scalar(3.0).unwrap()];
        assert_eq!(program.interpret(inputs), Ok(expected.clone()));
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<Array, ArrayOperation<Array>>::new(),
                &[Array::scalar(4.0).unwrap(), Array::scalar(1.0).unwrap()],
            ),
            Ok(expected),
        );
    }

    #[test]
    fn test_while_partial_evaluation_folds_invariant_state_into_the_condition() {
        // The condition reads the loop state too, so folding a loop-invariant known element shrinks the condition as
        // well: the loop carries `[counter, k]`, runs while `counter < k`, and returns `[counter + 1, k]`, and with `k`
        // known (`3`) the residual condition compares the counter against the constant `3`.
        let scalar = || ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::LessThan),
                    Vec::new(),
                    vec![counter, k],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
            let next_counter =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next_counter, k], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let inputs = (0..2).map(|_| builder.add_input(scalar())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(WhileOperation::new(), regions, inputs, None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(scalar()), PartialValue::Known(Array::scalar(3.0).unwrap())])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = while %0 %1 [
                    condition={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = const 3.0
                            %3:bool[] = compare [direction=LessThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = one_like %0
                            %3:f64[] = add %0 %2
                            %4:f64[] = const 3.0
                        in (%3, %4)
                    },
                ]
                in (%2, %3)"},
        );
        let expected = vec![Array::scalar(3.0).unwrap(), Array::scalar(3.0).unwrap()];
        assert_eq!(
            program.interpret(vec![Array::scalar(0.0).unwrap(), Array::scalar(3.0).unwrap()]),
            Ok(expected.clone()),
        );
        assert_eq!(
            evaluation.interpret(&EagerContext::<Array, ArrayOperation<Array>>::new(), &[Array::scalar(0.0).unwrap()]),
            Ok(expected),
        );
    }

    #[test]
    fn test_while_partial_evaluation_residualizes_reference_carry_whole() {
        // `f(n, r) = { while n > 0 { add_update(r, 1); n -= 1 }; read(r) }` with the reference carried through the
        // loop.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = condition_builder.add_input(scalar_type.clone().into());
        condition_builder.add_input(reference_type.clone().into());
        let zero = condition_builder.add_constant(TestIrValue::Array(Array::scalar(0.0f32).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan)),
                Vec::new(),
                vec![counter, zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = body_builder.add_input(scalar_type.clone().into());
        let reference = body_builder.add_input(reference_type.clone().into());
        let one = body_builder.add_constant(TestIrValue::Array(Array::scalar(1.0f32).unwrap()));
        let step = body_builder.add_constant(TestIrValue::Array(Array::scalar(-1.0f32).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, one], None)
            .unwrap();
        let next_counter = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, step], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_counter, reference],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_program(condition);
        let body = builder.import_program(body);
        let counter = builder.add_input(scalar_type.clone().into());
        let reference = builder.add_input(reference_type.into());
        let outputs = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new(),
                vec![condition, body],
                vec![counter, reference],
                None,
            )
            .unwrap()
            .to_vec();
        let read =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[1]], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![read], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // A loop touching references is never pure, so neither the invariance probes nor the closed-knownness split
        // run: the loop residualizes whole, its reference carry stays a residual reference, and the live state stays
        // untouched until the residual program runs.
        let live = ArrayReference::new(Array::scalar(2.0f32).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(scalar_type.into()),
                PartialValue::Known(TestIrValue::Reference(live.clone())),
            ])
            .unwrap();
        assert_eq!(live.read(), Ok(Array::scalar(2.0f32).unwrap()));
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]> .
                let %2:f32[], %3:ref<f32[]> = while %0 %1 [
                    condition={
                        lambda %0:f32[], %1:ref<f32[]> .
                        let %2:f32[] = const 0.0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f32[], %1:ref<f32[]> .
                        let %2:f32[] = const 1.0
                            %3:f32[] = const -1.0
                            () = reference_add_update %1 %2
                            %4:f32[] = add %0 %3
                        in (%4, %1)
                    },
                ]
                    %4:f32[] = reference_read %3
                in (%4)"},
        );
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<TestIrValue, TestIrOperation>::new(),
                &[TestIrValue::Array(Array::scalar(3.0f32).unwrap())],
            ),
            Ok(vec![TestIrValue::Array(Array::scalar(5.0f32).unwrap())]),
        );
        assert_eq!(live.read(), Ok(Array::scalar(5.0f32).unwrap()));
    }

    #[test]
    fn test_while_partial_evaluation_specializes_known_reference_carry() {
        // A known reference handle refines the symbolic `f32[extent]` carry to `f32[3]` in the residual loop, but
        // preparation must not run any iteration's state update: the handle is only written when the residual runs.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent)]),
        )));
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = condition_builder.add_input(scalar_type.clone());
        condition_builder.add_input(reference_type.clone());
        let zero = condition_builder.add_constant(array(Array::scalar(0f32).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan)),
                Vec::new(),
                vec![counter, zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = body_builder.add_input(scalar_type.clone());
        let reference = body_builder.add_input(reference_type.clone());
        let value = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let step = body_builder.add_constant(array(Array::scalar(-1f32).unwrap()));
        let next_counter = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, step], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_counter, reference],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_program(condition);
        let body = builder.import_program(body);
        let counter = builder.add_input(scalar_type.clone());
        let reference = builder.add_input(reference_type);
        let outputs = builder
            .add_instruction(WhileOperation::new(), vec![condition, body], vec![counter, reference], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let live = ArrayReference::new(Array::vector(vec![1f32, 2., 3.]).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(scalar_type),
                PartialValue::Known(TestIrValue::Reference(live.clone())),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[3]> .
                let %2:f32[], %3:ref<f32[3]> = while %0 %1 [
                    condition={
                        lambda %0:f32[], %1:ref<f32[3]> .
                        let %2:f32[] = const 0.0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f32[], %1:ref<f32[3]> .
                        let %2:f32[3] = reference_read %1
                            %3:f32[] = const -1.0
                            () = reference_add_update %1 %2
                            %4:f32[] = add %0 %3
                        in (%4, %1)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![1f32, 2., 3.]).unwrap()));
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<TestIrValue, TestIrOperation>::new(),
                &[array(Array::scalar(2f32).unwrap())],
            ),
            Ok(vec![array(Array::scalar(0f32).unwrap()), TestIrValue::Reference(live.clone())]),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![4f32, 8., 12.]).unwrap()));
    }

    #[test]
    fn test_while_partial_evaluation_folds_loop_invariant_known_array_state() {
        // Array IR loops fold loop-invariant known array state exactly as homogeneous loops do, while a known
        // first-class dimension keeps flowing through the loop because its state input already defines its identity.
        // Body `[extent, counter, accumulator, factor] -> [extent, counter - 1, accumulator + factor * factor, factor]`
        // over a known extent and a known `factor`.
        let extent_type =
            DimensionType::from(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let scalar = || ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let state_types = vec![ArrayIrType::Dimension(extent_type.clone()), scalar(), scalar(), scalar()];

        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_inputs =
            state_types.iter().map(|r#type| condition_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let zero = condition_builder.add_constant(array(Array::scalar(0.0).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::GreaterThan))),
                Vec::new(),
                vec![condition_inputs[1], zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body_inputs = state_types.iter().map(|r#type| body_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let (extent, counter, accumulator, factor) = (body_inputs[0], body_inputs[1], body_inputs[2], body_inputs[3]);
        let one = body_builder.add_constant(array(Array::scalar(1.0).unwrap()));
        let next_counter = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(SubOperation::new())),
                Vec::new(),
                vec![counter, one],
                None,
            )
            .unwrap()[0];
        let factor_squared = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(MulOperation::new())),
                Vec::new(),
                vec![factor, factor],
                None,
            )
            .unwrap()[0];
        let next_accumulator = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![accumulator, factor_squared],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![extent, next_counter, next_accumulator, factor],
                vec![Placeholder; 4],
                vec![Placeholder; 4],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(
                TestIrOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                inputs,
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 4], vec![Placeholder; 4])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(dimension(&extent_type, 5)),
                PartialValue::Unknown(scalar()),
                PartialValue::Unknown(scalar()),
                PartialValue::Known(array(Array::scalar(3.0).unwrap())),
            ])
            .unwrap();

        // The residual while keeps all four state elements, but its body folds `factor * factor` to a constant and
        // rebuilds the invariant `factor` output as a constant, while the dimension still flows from its state input to
        // its output. The known dimension is an eager value, typed exactly, so the residual regions are specialized to
        // it.
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:dimension<5>, %3:f64[] .
                let %4:dimension<5>, %5:f64[], %6:f64[], %7:f64[] = while %2 %0 %1 %3 [
                    condition={
                        lambda %0:dimension<5>, %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = const 0.0
                            %5:bool[] = compare [direction=GreaterThan] %1 %4
                        in (%5)
                    },
                    body={
                        lambda %0:dimension<5>, %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = const 1.0
                            %5:f64[] = sub %1 %4
                            %6:f64[] = const 9.0
                            %7:f64[] = add %2 %6
                            %8:f64[] = const 3.0
                        in (%0, %5, %7, %8)
                    },
                ]
                in (%4, %5, %6, %7)"},
        );

        // The residual program reproduces the source program: the loop runs four times, so `accumulator` threads
        // `1 -> 10 -> 19 -> 28 -> 37`.
        let expected = vec![
            dimension(&extent_type, 5),
            array(Array::scalar(0.0).unwrap()),
            array(Array::scalar(37.0).unwrap()),
            array(Array::scalar(3.0).unwrap()),
        ];
        assert_eq!(
            program.interpret(vec![
                dimension(&extent_type, 5),
                array(Array::scalar(4.0).unwrap()),
                array(Array::scalar(1.0).unwrap()),
                array(Array::scalar(3.0).unwrap()),
            ]),
            Ok(expected.clone()),
        );
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<TestIrValue, TestIrOperation>::new(),
                &[array(Array::scalar(4.0).unwrap()), array(Array::scalar(1.0).unwrap())],
            ),
            Ok(expected),
        );
    }

    #[test]
    fn test_while_partial_evaluation_keeps_known_state_that_refers_to_an_identity() {
        // A known array whose state type refers to a dimension identity (`f64[extent]`) is never folded, even when the
        // body only passes it through: a rebuilt constant would carry the concrete static type `f64[5]` of the known
        // value rather than the state type, which the loop's state signature rejects. The loop keeps carrying it, and
        // its refined known initial value is accepted as a loop input.
        let extent_type =
            DimensionType::from(DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap()));
        let counter_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let vector_type =
            ArrayIrType::Array(ArrayType::new(DataType::F64, Shape::new(vec![extent_type.dimension().clone()])));
        let state_types = vec![ArrayIrType::Dimension(extent_type.clone()), counter_type.clone(), vector_type];

        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_inputs =
            state_types.iter().map(|r#type| condition_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let zero = condition_builder.add_constant(array(Array::scalar(0.0).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::GreaterThan))),
                Vec::new(),
                vec![condition_inputs[1], zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body_inputs = state_types.iter().map(|r#type| body_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let one = body_builder.add_constant(array(Array::scalar(1.0).unwrap()));
        let next_counter = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(SubOperation::new())),
                Vec::new(),
                vec![body_inputs[1], one],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![body_inputs[0], next_counter, body_inputs[2]],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(
                TestIrOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                inputs,
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let vector = array(Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(dimension(&extent_type, 5)),
                PartialValue::Unknown(counter_type),
                PartialValue::Known(vector.clone()),
            ])
            .unwrap();

        // The residual while keeps the vector state, which still flows from its state input to its output. The known
        // dimension and vector are eager values, typed exactly, so the residual regions are specialized to them.
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:dimension<5>, %2:f64[5] .
                let %3:dimension<5>, %4:f64[], %5:f64[5] = while %1 %0 %2 [
                    condition={
                        lambda %0:dimension<5>, %1:f64[], %2:f64[5] .
                        let %3:f64[] = const 0.0
                            %4:bool[] = compare [direction=GreaterThan] %1 %3
                        in (%4)
                    },
                    body={
                        lambda %0:dimension<5>, %1:f64[], %2:f64[5] .
                        let %3:f64[] = const 1.0
                            %4:f64[] = sub %1 %3
                        in (%0, %4, %2)
                    },
                ]
                in (%3, %4, %5)"},
        );

        // The residual program reproduces the source program.
        let expected = vec![dimension(&extent_type, 5), array(Array::scalar(0.0).unwrap()), vector.clone()];
        assert_eq!(
            program.interpret(vec![dimension(&extent_type, 5), array(Array::scalar(3.0).unwrap()), vector]),
            Ok(expected.clone()),
        );
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<TestIrValue, TestIrOperation>::new(),
                &[array(Array::scalar(3.0).unwrap())],
            ),
            Ok(expected),
        );
    }

    #[test]
    fn test_while_partial_evaluation_refines_outputs_of_unspecialized_regions() {
        // Partially evaluating a loop whose refined outputs come from unspecialized regions keeps them refined
        // whichever state is known, and the residual program reproduces the original one.
        let program = refined_vector_while_program(None);
        let static_vector_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        assert_eq!(program.output_types()[1], static_vector_type);
        let counter = array(Array::scalar(2.0).unwrap());
        let vector = array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());
        let expected = vec![array(Array::scalar(0.0).unwrap()), array(Array::vector(vec![4.0, 8.0, 12.0]).unwrap())];
        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(counter.clone()), PartialValue::Unknown(static_vector_type)])
            .unwrap();
        assert_eq!(
            evaluation.program.output_types(),
            vec![ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]))],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[] .
                let %2:f64[], %3:f64[3] = while %1 %0 [
                    condition={
                        lambda %0:f64[], %1:f64[3] .
                        let %2:f64[] = const 0.0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[3] .
                        let %2:f64[] = const 1.0
                            %3:f64[] = sub %0 %2
                            %4:f64[3] = add %1 %1
                        in (%3, %4)
                    },
                ]
                in (%3)"},
        );
        assert_eq!(evaluation.interpret(&context, std::slice::from_ref(&vector)), Ok(expected.clone()));
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(ArrayIrType::Array(ArrayType::scalar(DataType::F64))),
                PartialValue::Known(vector),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.output_types(),
            vec![
                ArrayIrType::Array(ArrayType::scalar(DataType::F64)),
                ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3])),
            ],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3] = while %0 %1 [
                    condition={
                        lambda %0:f64[], %1:f64[3] .
                        let %2:f64[] = const 0.0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[3] .
                        let %2:f64[] = const 1.0
                            %3:f64[] = sub %0 %2
                            %4:f64[3] = add %1 %1
                        in (%3, %4)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(evaluation.interpret(&context, &[counter]), Ok(expected));
    }

    #[test]
    fn test_while_partial_evaluation_never_folds_a_reference_carry() {
        // A pure loop that only passes a known reference through is still never folded, because a folded carry would
        // be rebuilt as a program constant instead of keeping its allocation flowing through the loop boundary.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let state_types: Vec<ArrayIrType> = vec![scalar_type.clone().into(), reference_type.into()];

        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_inputs =
            state_types.iter().map(|r#type| condition_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let zero = condition_builder.add_constant(array(Array::scalar(0.0f32).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::GreaterThan))),
                Vec::new(),
                vec![condition_inputs[0], zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body_inputs = state_types.iter().map(|r#type| body_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let one = body_builder.add_constant(array(Array::scalar(1.0f32).unwrap()));
        let next_counter = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(SubOperation::new())),
                Vec::new(),
                vec![body_inputs[0], one],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_counter, body_inputs[1]],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert!(body.effects().classes().is_empty());

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_region = builder.import_region(condition.entry_region_ref());
        let body_region = builder.import_region(body.entry_region_ref());
        let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(
                TestIrOperation::While(WhileOperation::new()),
                vec![condition_region, body_region],
                inputs,
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let reference = TestIrValue::Reference(ArrayReference::new(Array::scalar(5.0f32).unwrap()));
        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(scalar_type.into()), PartialValue::Known(reference.clone())])
            .unwrap();

        // The residual regions are the source regions: the reference still flows from its state input to its state
        // output.
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]> .
                let %2:f32[], %3:ref<f32[]> = while %0 %1 [
                    condition={
                        lambda %0:f32[], %1:ref<f32[]> .
                        let %2:f32[] = const 0.0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f32[], %1:ref<f32[]> .
                        let %2:f32[] = const 1.0
                            %3:f32[] = sub %0 %2
                        in (%3, %1)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_while_partial_evaluation_preserves_signed_zero_state() {
        // Body `[counter, value] -> [counter - 1, -value]` over an unknown counter and a known value of `0.0`. Negating
        // `0.0` yields `-0.0`, which compares equal to `0.0`, so a value-based invariance check would wrongly fold the
        // value to its initial `0.0`. After an odd number of iterations, the final value must be `-0.0`.
        let scalar = || ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            let _value = builder.add_input(scalar());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::GreaterThan),
                    Vec::new(),
                    vec![counter, zero],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            let value = builder.add_input(scalar());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
            let next_counter =
                builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
            let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(
                    vec![next_counter, negated],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let counter = builder.add_input(scalar());
        let value = builder.add_input(scalar());
        let outputs = builder
            .add_instruction(
                ArrayOperation::While(WhileOperation::new()),
                vec![condition, body],
                vec![counter, value],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(scalar()), PartialValue::Known(Array::scalar(0.0f64).unwrap())])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = while %0 %1 [
                    condition={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = zero_like %0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = one_like %0
                            %3:f64[] = sub %0 %2
                            %4:f64[] = neg %1
                        in (%3, %4)
                    },
                ]
                in (%2, %3)"},
        );
        let outputs = evaluation
            .interpret(&EagerContext::<Array, ArrayOperation<Array>>::new(), &[Array::scalar(3.0f64).unwrap()])
            .unwrap();
        // `Array` equality compares decoded values, under which the two signed zeros are equal, so the sign of the
        // final value is checked through its bits.
        assert_eq!(outputs, vec![Array::scalar(0.0f64).unwrap(), Array::scalar(-0.0f64).unwrap()]);
        assert_eq!(outputs[1].to_f64s()[0].to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn test_while_partial_evaluation_folds_integer_division_by_a_known_zero() {
        // The invariance probes fold the body although interpretation may never enter it, because the condition can be
        // false on entry. Folding is sound here because integer division by zero is defined (it produces all ones, so
        // `1 / 0 = -1` for `i32`), so the probe of `1 / k` with the known invariant `k = 0` folds to `-1` instead of
        // failing, and the residual body adds that constant. The residual program then agrees with the source program
        // whether or not the loop runs.
        let state_type = ArrayType::scalar(DataType::I32);
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let x = builder.add_input(state_type.clone());
            let _k = builder.add_input(state_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![x], None).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::GreaterThan),
                    Vec::new(),
                    vec![x, zero],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let x = builder.add_input(state_type.clone());
            let k = builder.add_input(state_type.clone());
            let one = builder.add_constant(Array::from_elements::<i32>(state_type.clone(), &[1]).unwrap());
            let inverse = builder.add_instruction(DivOperation::new(), Vec::new(), vec![one, k], None).unwrap()[0];
            let next_x = builder.add_instruction(AddOperation::new(), Vec::new(), vec![x, inverse], None).unwrap()[0];
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
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs.to_vec(), vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let knowledge = vec![
            PartialValue::Unknown(state_type.clone()),
            PartialValue::Known(Array::from_elements::<i32>(state_type, &[0]).unwrap()),
        ];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:i32[], %1:i32[] .
                let %2:i32[], %3:i32[] = while %0 %1 [
                    condition={
                        lambda %0:i32[], %1:i32[] .
                        let %2:i32[] = zero_like %0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:i32[], %1:i32[] .
                        let %2:i32[] = const -1
                            %3:i32[] = add %0 %2
                            %4:i32[] = const 0
                        in (%3, %4)
                    },
                ]
                in (%2, %3)"},
        );

        // A non-positive entry state never enters the body, and a positive one counts down to zero.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let scalar = |value: i32| Array::from_elements::<i32>(ArrayType::scalar(DataType::I32), &[value]).unwrap();
        assert_eq!(program.interpret(vec![scalar(-1), scalar(0)]), Ok(vec![scalar(-1), scalar(0)]));
        assert_eq!(evaluation.interpret(&context, &[scalar(-1)]), Ok(vec![scalar(-1), scalar(0)]));
        assert_eq!(program.interpret(vec![scalar(2), scalar(0)]), Ok(vec![scalar(0), scalar(0)]));
        assert_eq!(evaluation.interpret(&context, &[scalar(2)]), Ok(vec![scalar(0), scalar(0)]));
    }

    #[test]
    fn test_while_partial_evaluation_residualizes_effectful_loops_unchanged() {
        // The invariance probes fold the regions through the live known-side context, so for an effectful loop they
        // would run its effects once more, and the closed-knownness split would run the known part of every iteration
        // a second time. One effects gate therefore skips both, and the loop residualizes unchanged with every output
        // unknown even though `k` is loop-invariant known: the loop carries `[counter, k]`, runs while `counter > 0`,
        // and its body prints `k` and returns `[counter - k * k, k]`, so a probe would print the known `k`, whereas
        // partial evaluation prints nothing and the residual body still squares `k`.
        let scalar = || ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            builder.add_input(scalar());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::GreaterThan),
                    Vec::new(),
                    vec![counter, zero],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let printed = builder.add_instruction(PrintOperation::new("k"), Vec::new(), vec![k], None).unwrap()[0];
            let ksq =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![printed, printed], None).unwrap()[0];
            let next_counter =
                builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, ksq], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next_counter, k], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let inputs = (0..2).map(|_| builder.add_input(scalar())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(WhileOperation::new(), regions, inputs, None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let context = CountingPrintContext::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &context,
                &[PartialValue::Unknown(scalar()), PartialValue::Known(Array::scalar(1.0).unwrap())],
            )
            .unwrap();
        assert_eq!(context.print_count(), 0);
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = while %0 %1 [
                    condition={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = zero_like %0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = print [label=k] %1
                            %3:f64[] = mul %2 %2
                            %4:f64[] = sub %0 %3
                        in (%4, %1)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(3.0).unwrap()]),
            Ok(vec![Array::scalar(0.0).unwrap(), Array::scalar(1.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_partial_evaluation_residualizes_loops_with_deferred_work_unchanged() {
        // Deferred work skips the invariance probes for the same reason as effects, because every probe would
        // residualize its obligation once more. The body carries a transposition carrier whose attached backward rule
        // writes the reference carry, so the body has deferred work but no effects of its own, and the loop
        // residualizes unchanged even though `k` is loop-invariant known.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let reference_type: ArrayIrType = ReferenceType::new(scalar_type.clone()).into();
        let state_types: Vec<ArrayIrType> =
            vec![scalar_type.clone().into(), scalar_type.clone().into(), reference_type];
        let mut backward_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = backward_builder.add_input(state_types[2].clone());
        let update = backward_builder.add_constant(array(Array::scalar(1.0f64).unwrap()));
        backward_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let backward = backward_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_inputs =
            state_types.iter().map(|r#type| condition_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let zero = condition_builder.add_constant(array(Array::scalar(0.0f64).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::GreaterThan))),
                Vec::new(),
                vec![condition_inputs[0], zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body_inputs = state_types.iter().map(|r#type| body_builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let backward = body_builder.import_program(backward);
        body_builder
            .add_instruction(
                TestIrOperation::CompositeTranspose(CompositeTransposeOperation::from_backward_region(
                    1,
                    Vec::new(),
                    Vec::new(),
                )),
                vec![backward],
                vec![body_inputs[2]],
                None,
            )
            .unwrap();
        let ksq = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(MulOperation::new())),
                Vec::new(),
                vec![body_inputs[1], body_inputs[1]],
                None,
            )
            .unwrap()[0];
        let next_counter = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(SubOperation::new())),
                Vec::new(),
                vec![body_inputs[0], ksq],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_counter, body_inputs[1], body_inputs[2]],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        assert!(body.effects().classes().is_empty());
        assert!(body.effects().has_deferred_work());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let inputs = state_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let outputs = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), regions, inputs, None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(state_types[0].clone()),
                PartialValue::Known(array(Array::scalar(1.0f64).unwrap())),
                PartialValue::Unknown(state_types[2].clone()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:ref<f64[]>, %2:f64[] .
                let %3:f64[], %4:f64[], %5:ref<f64[]> = while %0 %2 %1 [
                    condition={
                        lambda %0:f64[], %1:f64[], %2:ref<f64[]> .
                        let %3:f64[] = const 0.0
                            %4:bool[] = compare [direction=GreaterThan] %0 %3
                        in (%4)
                    },
                    body={
                        lambda %0:f64[], %1:f64[], %2:ref<f64[]> .
                        let () = composite_transpose [leading_input_count=1] %2 [
                            backward={
                                lambda %0:ref<f64[]> .
                                let %1:f64[] = const 1.0
                                    () = reference_add_update %0 %1
                                in ()
                            },
                        ]
                            %3:f64[] = mul %1 %1
                            %4:f64[] = sub %0 %3
                        in (%4, %1, %2)
                    },
                ]
                in (%3, %4, %5)"},
        );
    }

    #[test]
    fn test_while_partial_evaluation_splits_closed_known_state() {
        // The loop carries `[counter, acc]` and runs while `counter > 0`; its body computes
        // `next_counter = counter - 1` and `next_acc = acc + counter`. The `counter` element is *time-varying* known
        // (its value changes every iteration, so the loop-invariant rewrite cannot fold it) but *closed*: its next
        // value and the trip predicate fold from it alone. With `counter` known (`3`) and `acc` unknown, the
        // closed-knownness split runs the known counter loop on the known side, folding the final counter to the
        // known value `0` (3 -> 2 -> 1 -> 0), and keeps the whole loop residual for `acc`, recomputing the counter
        // chain inside it (there is no statically shaped residual stream to feed `acc + counter` with).
        let program = counter_accumulator_while_program(None);
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(3.0).unwrap()),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Known(Array::scalar(0.0).unwrap()), PartialEvaluationOutput::Unknown(0)],
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = while %1 %0 [
                    condition={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = zero_like %0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = one_like %0
                            %3:f64[] = sub %0 %2
                            %4:f64[] = add %1 %0
                        in (%3, %4)
                    },
                ]
                in (%3)"},
        );

        // From `acc = 10`, the residual loop adds 3, 2, and 1.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(10.0).unwrap()]),
            Ok(vec![Array::scalar(0.0).unwrap(), Array::scalar(16.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_partial_evaluation_split_truncates_both_loops_at_the_iteration_bound() {
        // A bound of two iterations, below the trip count of three, truncates the known counter loop at `3 - 2 = 1`
        // and the residual loop after adding 3 and 2, because both loops keep the bound of the source loop.
        let program = counter_accumulator_while_program(Some(2));
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(3.0).unwrap()),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Known(Array::scalar(1.0).unwrap()), PartialEvaluationOutput::Unknown(0)],
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = while [iteration_bound=2] %1 %0 [
                    condition={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = zero_like %0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = one_like %0
                            %3:f64[] = sub %0 %2
                            %4:f64[] = add %1 %0
                        in (%3, %4)
                    },
                ]
                in (%3)"},
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(10.0).unwrap()]),
            Ok(vec![Array::scalar(1.0).unwrap(), Array::scalar(15.0).unwrap()]),
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(3.0).unwrap(), Array::scalar(10.0).unwrap()]),
            Ok(vec![Array::scalar(1.0).unwrap(), Array::scalar(15.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_partial_evaluation_split_runs_zero_iterations_for_a_known_counter_at_its_limit() {
        // A known counter that already fails the predicate makes the known loop run no iteration, so the final counter
        // folds to its initial value and the residual loop returns the unknown accumulator unchanged.
        let program = counter_accumulator_while_program(None);
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(0.0).unwrap()),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![PartialEvaluationOutput::Known(Array::scalar(0.0).unwrap()), PartialEvaluationOutput::Unknown(0)],
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = while %1 %0 [
                    condition={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = zero_like %0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = one_like %0
                            %3:f64[] = sub %0 %2
                            %4:f64[] = add %1 %0
                        in (%3, %4)
                    },
                ]
                in (%3)"},
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(10.0).unwrap()]),
            Ok(vec![Array::scalar(0.0).unwrap(), Array::scalar(10.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_partial_evaluation_split_declines_when_the_predicate_reads_unknown_state() {
        // The counter of `while (x < 10) { (counter, x) = (counter - 1, x + x) }` is closed known, but the predicate
        // reads the unknown `x`, so a known counter loop could not reproduce the trip count. Like JAX's
        // `_while_partial_eval`, the rule then keeps every output unknown and residualizes the loop unchanged.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        condition_builder.add_input(scalar_type.clone());
        let x = condition_builder.add_input(scalar_type.clone());
        let limit = condition_builder.add_constant(Array::scalar(10.0).unwrap());
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![x, limit], None)
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let counter = body_builder.add_input(scalar_type.clone());
        let x = body_builder.add_input(scalar_type.clone());
        let one = body_builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
        let next_counter =
            body_builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
        let doubled = body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![x, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_counter, doubled], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let inputs = vec![builder.add_input(scalar_type.clone()), builder.add_input(scalar_type.clone())];
        let outputs = builder.add_instruction(WhileOperation::new(), regions, inputs, None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(Array::scalar(0.0).unwrap()), PartialValue::Unknown(scalar_type)])
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = while %1 %0 [
                    condition={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = const 10.0
                            %3:bool[] = compare [direction=LessThan] %1 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = one_like %0
                            %3:f64[] = sub %0 %2
                            %4:f64[] = add %1 %1
                        in (%3, %4)
                    },
                ]
                in (%2, %3)"},
        );

        // From `x = 3`, the loop doubles `x` twice (3 -> 6 -> 12) and counts the counter down from 0 to -2.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(3.0).unwrap()]),
            Ok(vec![Array::scalar(-2.0).unwrap(), Array::scalar(12.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_partial_evaluation_folds_invariant_state_then_splits_closed_known_state() {
        // The loop carries `[i, n, x]`, runs while `i < n`, and computes `(i + 1, n, 2 * x)`. With `i = 0` and `n = 5`
        // known and `x` unknown, the invariant `n` folds into both regions first, and the split then recovers the
        // closed known counter of the folded loop: the known loop runs five iterations, so the finals of both `i`
        // and `n` stay known, and the residual loop over the folded regions computes `x · 2⁵`.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let i = condition_builder.add_input(scalar_type.clone());
        let n = condition_builder.add_input(scalar_type.clone());
        condition_builder.add_input(scalar_type.clone());
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![i, n], None)
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let i = body_builder.add_input(scalar_type.clone());
        let n = body_builder.add_input(scalar_type.clone());
        let x = body_builder.add_input(scalar_type.clone());
        let one = body_builder.add_constant(Array::scalar(1.0).unwrap());
        let next_i = body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![i, one], None).unwrap()[0];
        let two = body_builder.add_constant(Array::scalar(2.0).unwrap());
        let next_x = body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![two, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_i, n, next_x], vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let inputs = (0..3).map(|_| builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
        let outputs = builder.add_instruction(WhileOperation::new(), regions, inputs, None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(0.0).unwrap()),
                PartialValue::Known(Array::scalar(5.0).unwrap()),
                PartialValue::Unknown(scalar_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Array::scalar(5.0).unwrap()),
                PartialEvaluationOutput::Known(Array::scalar(5.0).unwrap()),
                PartialEvaluationOutput::Unknown(0),
            ],
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[], %4:f64[], %5:f64[] = while %1 %2 %0 [
                    condition={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = const 5.0
                            %4:bool[] = compare [direction=LessThan] %0 %3
                        in (%4)
                    },
                    body={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = const 1.0
                            %4:f64[] = add %0 %3
                            %5:f64[] = const 5.0
                            %6:f64[] = const 2.0
                            %7:f64[] = mul %6 %2
                        in (%4, %5, %7)
                    },
                ]
                in (%5)"},
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(1.5).unwrap()]),
            Ok(vec![Array::scalar(5.0).unwrap(), Array::scalar(5.0).unwrap(), Array::scalar(48.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_partial_evaluation_preserves_entry_relationships_across_the_closed_knownness_split() {
        // The loop carries `[counter, x, acc]` with `x` and `acc` declared as `f64[n]`, runs while `counter > 0`, and
        // computes `(counter - 1, x, acc + x)`. A known `x: f64[3]` next to an unknown `acc: f64[n]` requires `n = 3`
        // at the loop entry. The split runs the closed known `[counter, x]` loop on the known side, and the residual
        // program must still reject an `acc` of any other length, like the source program does.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let static_type = ArrayType::new_static(DataType::F64, [3]);
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar_type.clone());
            builder.add_input(vector_type.clone());
            builder.add_input(vector_type.clone());
            let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
            let predicate = builder
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::GreaterThan),
                    Vec::new(),
                    vec![counter, zero],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let counter = builder.add_input(scalar_type.clone());
            let x = builder.add_input(vector_type.clone());
            let acc = builder.add_input(vector_type.clone());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
            let next_counter =
                builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
            let next_acc = builder.add_instruction(AddOperation::new(), Vec::new(), vec![acc, x], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(
                    vec![next_counter, x, next_acc],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let counter = builder.add_input(scalar_type.clone());
        let x = builder.add_input(static_type.clone());
        let acc = builder.add_input(vector_type.clone());
        let outputs = builder
            .add_instruction(
                ArrayOperation::While(WhileOperation::new()),
                vec![condition, body],
                vec![counter, x, acc],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();

        let counter = Array::scalar(2.0f64).unwrap();
        let known = Array::vector(vec![1.0f64; 3]).unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(counter.clone()),
                PartialValue::Known(known.clone()),
                PartialValue::Unknown(vector_type),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[n], %1:f64[], %2:f64[3] .
                let %3:f64[], %4:f64[3], %5:f64[3] = while %1 %2 %0 [
                    condition={
                        lambda %0:f64[], %1:f64[n], %2:f64[n] .
                        let %3:f64[] = zero_like %0
                            %4:bool[] = compare [direction=GreaterThan] %0 %3
                        in (%4)
                    },
                    body={
                        lambda %0:f64[], %1:f64[n], %2:f64[n] .
                        let %3:f64[] = one_like %0
                            %4:f64[] = sub %0 %3
                            %5:f64[n] = add %2 %1
                        in (%4, %1, %5)
                    },
                ]
                in (%5)"},
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let conforming = Array::vector(vec![5.0f64; 3]).unwrap();
        assert_eq!(
            evaluation.interpret(&context, std::slice::from_ref(&conforming)),
            program.interpret(vec![counter.clone(), known.clone(), conforming]),
        );
        let nonconforming = Array::vector(vec![5.0f64; 4]).unwrap();
        assert_eq!(
            program.interpret(vec![counter.clone(), known.clone(), nonconforming.clone()]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_owned(),
                expected: 3,
                actual: 4,
            }))),
        );
        assert_eq!(
            evaluation.interpret(&context, std::slice::from_ref(&nonconforming)),
            program.interpret(vec![counter, known, nonconforming]),
        );
    }

    #[test]
    fn test_while_partial_evaluation_propagates_unknown_carries_across_multiple_rounds() {
        // With initially known `first` and `second`, the body first discovers that `second` depends on the unknown
        // `source`. Only another round discovers that `first` must also become unknown. Loop invariance is structural
        // and the only carry that the body passes through is the unknown `source`, so nothing folds and the
        // closed-knownness split alone decides that only the counter stays known.
        let program = chained_carry_while_program(None);
        let arguments = vec![
            Array::scalar(3.0).unwrap(),
            Array::scalar(1.0).unwrap(),
            Array::scalar(1.0).unwrap(),
            Array::scalar(7.0).unwrap(),
        ];
        let expected = vec![
            Array::scalar(0.0).unwrap(),
            Array::scalar(7.0).unwrap(),
            Array::scalar(7.0).unwrap(),
            Array::scalar(7.0).unwrap(),
        ];
        assert_eq!(program.interpret(arguments.clone()), Ok(expected.clone()));

        // Eager specialization must not retain either initial value after the dependency propagates through carries.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(arguments[0].clone()),
                PartialValue::Known(arguments[1].clone()),
                PartialValue::Known(arguments[2].clone()),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Array::scalar(0.0).unwrap()),
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
                PartialEvaluationOutput::Unknown(2),
            ],
        );
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[], %5:f64[], %6:f64[], %7:f64[] = while %1 %2 %3 %0 [
                    condition={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = const 0.0
                            %5:bool[] = compare [direction=GreaterThan] %0 %4
                        in (%5)
                    },
                    body={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = const 1.0
                            %5:f64[] = sub %0 %4
                        in (%5, %2, %3, %3)
                    },
                ]
                in (%5, %6, %7)"},
        );
        assert_eq!(evaluation.interpret(&EagerContext::new(), &[arguments[3].clone()]), Ok(expected.clone()));
    }

    #[test]
    fn test_while_partition_propagates_unknown_carries_across_multiple_rounds() {
        // Partitioning uses symbolic known inputs, so its separate knownness loop must reach the same result as partial
        // evaluation (refer to the preceding test): only the counter stays known, and the known and residual programs
        // together compute the source outputs.
        let program = chained_carry_while_program(None);
        let arguments = vec![
            Array::scalar(3.0).unwrap(),
            Array::scalar(1.0).unwrap(),
            Array::scalar(1.0).unwrap(),
            Array::scalar(7.0).unwrap(),
        ];
        let expected = vec![
            Array::scalar(0.0).unwrap(),
            Array::scalar(7.0).unwrap(),
            Array::scalar(7.0).unwrap(),
            Array::scalar(7.0).unwrap(),
        ];
        let partition = program.partition(&[true, true, true, false]).unwrap();
        assert_eq!(
            partition.outputs().iter().map(PartialEvaluationOutput::is_known).collect::<Vec<_>>(),
            vec![true, false, false, false],
        );
        assert_eq!(
            partition.known_program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = while %0 [
                    condition={
                        lambda %0:f64[] .
                        let %1:f64[] = const 0.0
                            %2:bool[] = compare [direction=GreaterThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = const 1.0
                            %2:f64[] = sub %0 %1
                        in (%2)
                    },
                ]
                in (%3, %0, %1, %2)"},
        );
        assert_eq!(
            partition.residual_program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                let %4:f64[], %5:f64[], %6:f64[], %7:f64[] = while %1 %2 %3 %0 [
                    condition={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = const 0.0
                            %5:bool[] = compare [direction=GreaterThan] %0 %4
                        in (%5)
                    },
                    body={
                        lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                        let %4:f64[] = const 1.0
                            %5:f64[] = sub %0 %4
                        in (%5, %2, %3, %3)
                    },
                ]
                in (%5, %6, %7)"},
        );
        let known_inputs = partition.known_input_indices().iter().map(|&index| arguments[index].clone()).collect();
        let known_outputs = partition.known_program().interpret(known_inputs).unwrap();
        // Feeder indices count only the residual edges, which follow the fully known outputs in the known program.
        let known_output_count = partition.outputs().iter().filter(|output| output.is_known()).count();
        let residual_inputs = partition
            .residual_inputs()
            .iter()
            .map(|source| match source {
                ResidualInputSource::ResidualEdge(index) => known_outputs[known_output_count + index].clone(),
                ResidualInputSource::UnknownInput(index) => arguments[*index].clone(),
                source => panic!("ordinary partitions have no `{source:?}` sources"),
            })
            .collect();
        let residual_outputs = partition.residual_program().interpret(residual_inputs).unwrap();
        let outputs = partition
            .outputs()
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(index) => known_outputs[*index].clone(),
                PartialEvaluationOutput::Unknown(index) => residual_outputs[*index].clone(),
            })
            .collect::<Vec<_>>();
        assert_eq!(outputs, expected);
    }

    #[test]
    fn test_while_batching() {
        // Batching a loop whose predicate reads only a replicated counter keeps that predicate replicated and binds one
        // plain loop over the batched body, in which every batch item doubles its mapped value once per iteration.
        let (operation, regions) = counter_doubling_while_operation();
        check_operation_batching!(
            @exact,
            context = EagerContext::<Array, ArrayOperation<Array>>::new(),
            driver = &CountingBatchingDriver::new(&regions),
            operation = operation,
            axis_size = 3,
            axis_sharding = ShardingDimension::Replicated,
            cases = [
                {
                    inputs = [
                        (@replicated, Array::scalar(2.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                    ],
                    outputs = [
                        (@replicated, Array::scalar(0.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![4.0, 8.0, 12.0]).unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@replicated, Array::scalar(0.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                    ],
                    outputs = [
                        (@replicated, Array::scalar(0.0).unwrap()),
                        (@mapped(axis = 0), Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                    ],
                },
            ],
        );

        // The staged form of the same batched loop is one plain `while` whose condition reads only the counter.
        let program = while_program(operation, &regions, vec![ArrayType::scalar(DataType::F64); 2]);
        let batched = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::replicated(), BatchAxis::new(0)]);
        let batched = batched.into_parts().0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:f64[], %3:f64[3] = while %0 %1 [
                    condition={
                        lambda %0:f64[], %1:f64[3] .
                        let %2:f64[] = zero_like %0
                            %3:bool[] = compare [direction=GreaterThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[3] .
                        let %2:f64[] = one_like %0
                            %3:f64[] = sub %0 %2
                            %4:f64[3] = add %1 %1
                        in (%3, %4)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            batched.interpret(vec![Array::scalar(2.0).unwrap(), Array::vector(vec![1.0, 2.0, 3.0]).unwrap()]),
            Ok(vec![Array::scalar(0.0).unwrap(), Array::vector(vec![4.0, 8.0, 12.0]).unwrap()]),
        );
    }

    #[test]
    fn test_while_batching_freezes_every_item_whose_predicate_is_false_at_entry() {
        // When the batched predicate is false for every item at entry, the staged loop runs no iteration and returns
        // each item's state unchanged, exactly like the source loop applied to each item.
        let (countdown_operation, countdown_regions) = countdown_while_operation();
        let program = batch_while_under_tracing(countdown_operation, countdown_regions, 3);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = while %0 [
                    condition={
                        lambda %0:f64[3] .
                        let %1:f64[3] = zero_like %0
                            %2:bool[3] = compare [direction=GreaterThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64[3] .
                        let %1:f64[3] = one_like %0
                            %2:f64[3] = sub %0 %1
                        in (%2)
                    },
                ]
                in (%1)"},
        );
        assert_eq!(
            program.interpret(Array::vector(vec![0.0, -1.0, -2.5]).unwrap()),
            Ok(Array::vector(vec![0.0, -1.0, -2.5]).unwrap()),
        );
    }

    #[test]
    fn test_while_batching_stages_batched_predicate_loops_under_tracing() {
        // Batching the per-item countdown loop under tracing makes the predicate `x > 0` per batch item, so the rule
        // stages exactly one `while`, without unrolling it, whose condition returns the batched `bool[3]` predicate
        // directly (the relaxed predicate contract, mirroring JAX's `_while_loop_batching_rule`) and without building
        // any masking program data: interpretation and lowering own the masked semantics. Batch items [3, 1, 2]
        // terminate after 3, 1, and 2 iterations, and finished batch items keep their final state, matching the eager
        // path batch item for batch item. Staging needs only one structural batching pass per region.
        let (countdown_operation, countdown_regions) = countdown_while_operation();
        let program = batch_while_under_tracing(countdown_operation, countdown_regions, 3);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = while %0 [
                    condition={
                        lambda %0:f64[3] .
                        let %1:f64[3] = zero_like %0
                            %2:bool[3] = compare [direction=GreaterThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64[3] .
                        let %1:f64[3] = one_like %0
                            %2:f64[3] = sub %0 %1
                        in (%2)
                    },
                ]
                in (%1)"},
        );
        assert_eq!(
            program.interpret(Array::vector(vec![3.0, 1.0, 2.0]).unwrap()),
            Ok(Array::vector(vec![0.0, 0.0, 0.0]).unwrap()),
        );

        // The rule batches the condition with natural output axes to detect per-item termination and then instantiates
        // its predicate at axis 0. `AlignEachTo` stages axis movement only where a natural axis differs from a mapped
        // target, so a predicate that already landed on axis 0 over unchanged state axes keeps its discovery program:
        // binding the rule directly performs one structural pass per region instead of re-batching the condition, and
        // stages the same loop.
        let (countdown_operation, countdown_regions) = countdown_while_operation();
        let parent = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = parent.builder().clone();
        let packed_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let state_atom = builder.borrow_mut().add_input(packed_type.clone());
        let state = parent.tracer(state_atom, None);
        let context = BatchingContext::new(parent, 3);
        let inputs = vec![ArrayBatch::new(state, BatchAxis::new(0)).unwrap()];
        let driver = CountingBatchingDriver::new(&countdown_regions);
        let outputs = countdown_operation.batch(&context, &driver, inputs.as_slice()).unwrap();
        assert_eq!(driver.batch_program_calls(), 2);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        let direct_program = builder
            .borrow()
            .clone()
            .build::<Array, Array>(vec![outputs[0].value().atom_id().unwrap()], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(direct_program.to_string(), program.to_string());

        // The semantic iteration bound is preserved on the staged batched-predicate while: every batch item performs
        // at most two body applications, so batch item 0 truncates at 1.0, like the eager bounded path.
        let (countdown_operation, countdown_regions) = countdown_while_operation();
        let program =
            batch_while_under_tracing(countdown_operation.with_iteration_bound(2).unwrap(), countdown_regions, 3);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = while [iteration_bound=2] %0 [
                    condition={
                        lambda %0:f64[3] .
                        let %1:f64[3] = zero_like %0
                            %2:bool[3] = compare [direction=GreaterThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64[3] .
                        let %1:f64[3] = one_like %0
                            %2:f64[3] = sub %0 %1
                        in (%2)
                    },
                ]
                in (%1)"},
        );
        assert_eq!(
            program.interpret(Array::vector(vec![3.0, 1.0, 2.0]).unwrap()),
            Ok(Array::vector(vec![1.0, 0.0, 0.0]).unwrap()),
        );
    }

    #[test]
    fn test_while_batching_reuses_naturally_aligned_batch_varying_array_ir_programs() {
        // The array IR rule batches the condition with natural output axes to detect per-item termination and rebuilds
        // both regions only when that widening changes the invariant boundary: with every array carry already mapped at
        // axis 0 and a predicate that naturally landed on axis 0, the rebuilds would replay the exact same programs, so
        // the discovery body and condition are kept and batching the per-item countdown loop performs one structural
        // pass per region.
        type EagerParent = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let countdown_regions = array_ir_countdown_while_regions();

        // Batching batch items [3, 1, 2] under an eager parent performs exactly one structural pass per region (the
        // fixed-point discovery body and the natural condition) and interprets the loop with its masked per-item
        // semantics: every batch item counts down to 0 even though the items terminate after different trip counts.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            EagerParent::new(),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        );
        let inputs = vec![
            ArrayIrBatch::new(ArrayIrValue::Array(Array::vector(vec![3.0, 1.0, 2.0]).unwrap()), BatchAxis::new(0))
                .unwrap(),
        ];
        let driver = CountingBatchingDriver::new(&countdown_regions);
        let countdown_operation = WhileOperation::<ArrayIrType>::new();
        let outputs = countdown_operation.batch(&context, &driver, inputs.as_slice()).unwrap();
        assert_eq!(driver.batch_program_calls(), 2);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &ArrayIrValue::Array(Array::vector(vec![0.0, 0.0, 0.0]).unwrap()));

        // The skipped alignment rebuilds are byte-identical to the discovery programs the rule kept: replaying each
        // region with `AlignEachTo` at the already-aligned axes (each region returns one output — the predicate or
        // the state — whose target is axis 0) renders exactly the natural discovery program.
        let state_axes = [BatchAxis::new(0)];
        let expected = [
            indoc! {"
                lambda %0:dimension<3>, %1:f64[3] .
                let %2:f64[3] = zero_like %1
                    %3:bool[3] = compare [direction=GreaterThan] %1 %2
                in (%0, %3)"},
            indoc! {"
                lambda %0:dimension<3>, %1:f64[3] .
                let %2:f64[3] = one_like %1
                    %3:f64[3] = sub %1 %2
                in (%0, %3)"},
        ];
        for (region, expected) in countdown_regions.iter().zip(expected) {
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
            assert_eq!(natural.into_parts().0.to_string(), expected);
            assert_eq!(aligned.into_parts().0.to_string(), expected);
        }
    }

    #[test]
    fn test_while_batching_stages_batched_predicate_array_ir_loops_under_tracing() {
        // Array IR batching of a batch-varying `while` loop stages under tracing. The rule threads the loop-invariant
        // mapped extent as leading loop state next to the batched `bool[batch]` predicate, which the type contract
        // accepts because the body passes that dimension carry through unchanged, and interpreting the staged loop
        // exercises the masked eager path over that carry.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_type = DimensionType::from(batch.clone());
        let trace = TraceContext::new();
        let batch_extent = trace.input(batch_type.clone().into());
        let state =
            trace.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let input_ids = [batch_extent.clone(), state.clone()].map(|input| input.atom_id().unwrap());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let outputs = context
            .bind(
                ArrayIrOperation::While(WhileOperation::new()),
                array_ir_countdown_while_regions(),
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
            panic!("array IR while batching should stage exactly one instruction");
        };
        assert_eq!(r#while.inputs(), &[input_ids[0], input_ids[1]]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                let %2:dimension<batch ∈ [1, 9)>, %3:f64[batch] = while %0 %1 [
                    condition={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                        let %2:f64[batch] = zero_like %1
                            %3:bool[batch] = compare [direction=GreaterThan] %1 %2
                        in (%3)
                    },
                    body={
                        lambda %0:dimension<batch ∈ [1, 9)>, %1:f64[batch] .
                        let %2:f64[batch] = one_like %1
                            %3:f64[batch] = sub %1 %2
                        in (%0, %3)
                    },
                ]
                in (%3)"},
        );

        // Interpreting the staged loop over batch items [3, 1, 2] counts every item down to 0 under the masked
        // semantics while the loop-invariant extent carry rides through unchanged.
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Dimension(DimensionValue::new(batch_type, 3).unwrap()),
                ArrayIrValue::Array(Array::vector(vec![3.0, 1.0, 2.0]).unwrap()),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![0.0, 0.0, 0.0]).unwrap())]),
        );
    }

    #[test]
    fn test_while_batching_preserves_a_replicated_predicate_prefix() {
        // A replicated `[true, false]` predicate already masks the leading state axis, so batching inserts the mapped
        // axis after that prefix: the first state item doubles twice for every batch item and the second never does.
        let state_type = ArrayType::new_static(DataType::F64, [2]);
        let mut condition_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        condition_builder.add_input(state_type.clone());
        let predicate = condition_builder.add_constant(Array::vector(vec![true, false]).unwrap());
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = body_builder.add_input(state_type.clone());
        let doubled =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let condition = builder.import_program(condition);
        let body = builder.import_program(body);
        let state = builder.add_input(state_type);
        let outputs = builder
            .add_instruction(
                WhileOperation::new().with_iteration_bound(2).unwrap(),
                vec![condition, body],
                vec![state],
                None,
            )
            .unwrap()
            .to_vec();
        let source = builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap();
        let batched = source
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(1)]);
        let batched = batched.into_parts().0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3, 2] .
                let %1:f64[2, 3] = transpose [permutation=[1, 0]] %0
                    %2:f64[2, 3] = while [iteration_bound=2] %1 [
                        condition={
                            lambda %0:f64[2, 3] .
                            let %1:bool[2] = const [true, false]
                            in (%1)
                        },
                        body={
                            lambda %0:f64[2, 3] .
                            let %1:f64[2, 3] = add %0 %0
                            in (%1)
                        },
                    ]
                in (%2)"},
        );
        let input = Array::from_elements(ArrayType::new_static(DataType::F64, [3, 2]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        assert_eq!(
            batched.interpret(vec![input]),
            Ok(vec![
                Array::from_elements(ArrayType::new_static(DataType::F64, [2, 3]), &[4.0, 12.0, 20.0, 2.0, 4.0, 6.0])
                    .unwrap(),
            ]),
        );
    }

    #[test]
    fn test_while_batching_preserves_a_replicated_predicate_prefix_of_array_ir_state() {
        // Array IR batching with a threaded extent preserves a replicated predicate prefix exactly as array batching
        // does: the mapped axis lands after the `[true, false]` prefix, so the first state item doubles twice for every
        // batch item and the second never does.
        let state_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2]));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(state_type.clone());
        let predicate = condition_builder.add_constant(array(Array::vector(vec![true, false]).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = body_builder.add_input(state_type.clone());
        let doubled = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::Add(AddOperation::new())),
                Vec::new(),
                vec![state, state],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_program(condition);
        let body = builder.import_program(body);
        let state = builder.add_input(state_type);
        let outputs = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap(),
                vec![condition, body],
                vec![state],
                None,
            )
            .unwrap()
            .to_vec();
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();
        let extent = DimensionValue::constant(3).unwrap();
        let batched = source
            .batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(1)]);
        let batched = batched.into_parts().0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<3>, %1:f64[3, 2] .
                let %2:f64[2, 3] = transpose [permutation=[1, 0]] %1
                    %3:dimension<3>, %4:f64[2, 3] = while [iteration_bound=2] %0 %2 [
                        condition={
                            lambda %0:dimension<3>, %1:f64[2, 3] .
                            let %2:bool[2] = const [true, false]
                            in (%2)
                        },
                        body={
                            lambda %0:dimension<3>, %1:f64[2, 3] .
                            let %2:f64[2, 3] = add %1 %1
                            in (%0, %2)
                        },
                    ]
                in (%0, %4)"},
        );
        let input = Array::from_elements(ArrayType::new_static(DataType::F64, [3, 2]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        assert_eq!(
            batched.interpret(vec![TestIrValue::Dimension(extent.clone()), array(input)]),
            Ok(vec![
                TestIrValue::Dimension(extent),
                array(
                    Array::from_elements(
                        ArrayType::new_static(DataType::F64, [2, 3]),
                        &[4.0, 12.0, 20.0, 2.0, 4.0, 6.0],
                    )
                    .unwrap(),
                ),
            ]),
        );
    }

    #[test]
    fn test_while_batching_refines_outputs_of_unspecialized_regions() {
        // Batching a loop whose refined outputs come from unspecialized regions keeps its batched outputs refined.
        let extent = DimensionValue::constant(2).unwrap();
        let batched = refined_vector_while_program(None)
            .batched_with_threaded_extent(
                extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(batched.output_types()[2], ArrayIrType::Array(ArrayType::new_static(DataType::F64, [2, 3])));
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f64[], %2:f64[2, 3] .
                let %3:dimension<2>, %4:f64[], %5:f64[2, 3] = while %0 %1 %2 [
                    condition={
                        lambda %0:dimension<2>, %1:f64[], %2:f64[2, 3] .
                        let %3:f64[] = const 0.0
                            %4:bool[] = compare [direction=GreaterThan] %1 %3
                        in (%4)
                    },
                    body={
                        lambda %0:dimension<2>, %1:f64[], %2:f64[2, 3] .
                        let %3:f64[] = const 1.0
                            %4:f64[] = sub %1 %3
                            %5:f64[2, 3] = add %2 %2
                        in (%0, %4, %5)
                    },
                ]
                in (%0, %4, %5)"},
        );
        let vectors =
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let outputs = batched
            .interpret(vec![TestIrValue::Dimension(extent), array(Array::scalar(2.0).unwrap()), array(vectors)])
            .unwrap();
        assert_eq!(
            outputs[2],
            array(
                Array::from_elements::<f64>(
                    ArrayType::new_static(DataType::F64, [2, 3]),
                    &[4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
                )
                .unwrap(),
            ),
        );
    }

    #[test]
    fn test_while_batching_threads_reference_carries() {
        // The replicated counter drives one structural loop for the whole batch while the reference carry keeps the
        // batch axis of its allocation and doubles its per-item contents on every iteration, so batching the stateful
        // loop directly agrees with batching its discharged counterpart.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = condition_builder.add_input(scalar_type.clone().into());
        condition_builder.add_input(reference_type.clone().into());
        let zero = condition_builder
            .add_instruction(ArrayOperation::ZeroLike(ZeroLikeOperation::new()), Vec::new(), vec![counter], None)
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::GreaterThan)),
                Vec::new(),
                vec![counter, zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = body_builder.add_input(scalar_type.clone().into());
        let reference = body_builder.add_input(reference_type.into());
        let current = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, current], None)
            .unwrap();
        let step = body_builder.add_constant(TestIrValue::Array(Array::scalar(-1.0f32).unwrap()));
        let next_counter = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, step], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_counter, reference],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_program(condition);
        let body = builder.import_program(body);
        let counter = builder.add_input(scalar_type.clone().into());
        let initial = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let outputs = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new(),
                vec![condition, body],
                vec![counter, reference],
                None,
            )
            .unwrap();
        let final_reference = outputs[1];
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![final_reference], None)
            .unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![frozen], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
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
        assert_eq!(direct.output_axes(), &[BatchAxis::new(0)]);
        assert_eq!(discharged.output_axes(), direct.output_axes());
        let (direct, _) = direct.into_parts();
        assert_eq!(
            direct.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[], %2:f32[2] .
                let %3:ref<f32[2]> = reference_new %2
                    %4:dimension<2>, %5:f32[], %6:ref<f32[2]> = while %0 %1 %3 [
                        condition={
                            lambda %0:dimension<2>, %1:f32[], %2:ref<f32[2]> .
                            let %3:f32[] = zero_like %1
                                %4:bool[] = compare [direction=GreaterThan] %1 %3
                            in (%4)
                        },
                        body={
                            lambda %0:dimension<2>, %1:f32[], %2:ref<f32[2]> .
                            let %3:f32[2] = reference_read %2
                                %4:f32[] = const -1.0
                                () = reference_add_update %2 %3
                                %5:f32[] = add %1 %4
                            in (%0, %5, %2)
                        },
                    ]
                    %7:f32[2] = reference_freeze %6
                in (%0, %7)"},
        );

        // Both batched programs run the loop to the same values: three iterations double each item's referent.
        let inputs = vec![
            TestIrValue::Dimension(axis_extent.clone()),
            TestIrValue::Array(Array::scalar(3.0f32).unwrap()),
            TestIrValue::Array(Array::vector(vec![1.0f32, 10.0]).unwrap()),
        ];
        let expected =
            vec![TestIrValue::Dimension(axis_extent), TestIrValue::Array(Array::vector(vec![8.0f32, 80.0]).unwrap())];
        assert_eq!(direct.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.into_parts().0.interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_while_batching_rejects_mapped_shape_computation() {
        // Computing a first-class extent from a mapped carry would give each batch item its own array shape, so the
        // scalar-to-dimension gateway rejects the first structural pass over the body and its error propagates
        // unchanged before the parent stages anything.
        let iteration_variable = DimensionVariable::new("iteration", DimensionBounds::positive(Some(4)).unwrap());
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let scalar_u64 = ArrayType::scalar(DataType::U64);

        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(ArrayIrType::Array(scalar_f64.clone()));
        let counter = condition_builder.add_input(ArrayIrType::Array(scalar_u64.clone()));
        let limit = condition_builder.add_constant(array(Array::scalar(3u64).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = body_builder.add_input(ArrayIrType::Array(scalar_f64.clone()));
        let counter = body_builder.add_input(ArrayIrType::Array(scalar_u64.clone()));
        let iteration = body_builder
            .add_instruction(
                TestIrOperation::from(DimensionFromScalarOperation::new(iteration_variable.clone().into())),
                Vec::new(),
                vec![counter],
                None,
            )
            .unwrap()[0];
        let repeated = body_builder
            .add_instruction(
                TestIrOperation::from(DynamicBroadcastOperation::new(Vec::new())),
                Vec::new(),
                vec![state, iteration],
                None,
            )
            .unwrap()[0];
        let next_state = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum))),
                Vec::new(),
                vec![repeated],
                None,
            )
            .unwrap()[0];
        let one = body_builder.add_constant(array(Array::scalar(1u64).unwrap()));
        let next_counter = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![counter, one],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_state, next_counter],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let regions = vec![condition, body];
        let driver = CountingBatchingDriver::new(&regions);

        // Both carries are mapped under a staging parent, whose trace must stay empty after the rejection.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let trace = TracingContext::<TestIrValue, TestIrOperation>::new();
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let state =
            trace.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let counter = trace.input(ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(batch)])).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let inputs = [
            ArrayIrBatch::new(state, BatchAxis::new(0)).unwrap(),
            ArrayIrBatch::new(counter, BatchAxis::new(0)).unwrap(),
        ];
        assert_eq!(
            WhileOperation::<ArrayIrType>::new().batch(&context, &driver, &inputs).map(|_| ()),
            Err(BatchingError::MappedDimension {
                r#type: Box::new(DimensionType::from(iteration_variable)),
                axis: BatchAxis::new(0),
            }),
        );
        assert!(trace.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_while_differentiation() {
        // An eager loop runs its duals directly: `while (x < 100, iteration_bound = 4) { x = x * x }` at `x = 2`
        // squares three times (2 -> 4 -> 16 -> 256), so locally `f(x) = x⁸` with value 256 and derivative
        // `8 x⁷ = 1024`, which finite differences confirm independently.
        let (operation, regions) = bounded_squaring_while_operation(100.0, 4);
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                bind_while(operation, regions.clone(), &[x]).map(|mut outputs| outputs.remove(0))
            }),
            Ok((Array::scalar(256.0).unwrap(), Array::scalar(1024.0).unwrap())),
        );
        check_gradient!(
            |x| bind_while(operation, regions.clone(), &[x]).map(|mut outputs| outputs.remove(0)),
            at = Array::scalar(2.0).unwrap(),
            step = 1e-4,
            tolerance = 1e-3,
        );
    }

    #[test]
    fn test_while_differentiation_eager_linearization_and_transposition_follow_the_executed_iterations() {
        // Eager linearization and reverse mode follow the iterations that actually ran: `while (x > 0) { x = x - 1 }`
        // at `x = 3.5` runs four iterations to `-0.5`, so locally `f(x) = x - 4`, and both the pushforward and the
        // pullback pass a tangent of 2 through unchanged.
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
        let (output, pushforward) = differentiate_at(Array::scalar(3.5).unwrap()).linearize(function).unwrap();
        assert_eq!(output, Array::scalar(-0.5).unwrap());
        assert_eq!(pushforward.apply(Array::scalar(2.0).unwrap()), Ok(Array::scalar(2.0).unwrap()));

        let (output, pullback) = differentiate_at(Array::scalar(3.5).unwrap()).vjp(function).unwrap();
        assert_eq!(output, Array::scalar(-0.5).unwrap());
        assert_eq!(pullback.apply(Array::scalar(2.0).unwrap()), Ok(MaybeZero::Value(Array::scalar(2.0).unwrap())));
        check_gradient!(
            |x| {
                let regions = vec![greater_than_zero_condition(), subtract_one_body()];
                bind_while(WhileOperation::new(), regions, &[x]).map(|mut outputs| outputs.remove(0))
            },
            at = Array::scalar(3.5).unwrap(),
            step = 1e-4,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_while_differentiation_eager_jvp_executes_body_effects_once_per_iteration() {
        // Eager forward mode runs the loop once over its duals instead of re-executing the primal, so the body's
        // `print` runs exactly once per iteration: from `x = 1`, `while (x < 8) { print(x); x = 2 * x }` prints three
        // times and computes `f(x) = 8 x`.
        let context = CountingPrintContext::new();
        let observed_context = context.clone();
        let (while_operation, while_regions) = effectful_doubling_while_operation();
        let (primal, tangent) = context
            .jvp(
                move |x, ()| {
                    let mut outputs = x.context().bind(while_operation, while_regions.clone(), &[x.clone()])?;
                    Ok(outputs.remove(0))
                },
                Array::scalar(1.0).unwrap(),
                Array::scalar(1.0).unwrap(),
                (),
            )
            .unwrap();

        assert_eq!(primal.to_f64s(), vec![8.0]);
        assert_eq!(tangent.to_f64s(), vec![8.0]);
        assert_eq!(observed_context.print_count(), 3);
    }

    #[test]
    fn test_while_differentiation_eager_jvp_allocates_zero_initialized_tangent_references() {
        // Each body iteration allocates a reference from a constant zero, whose tangent is therefore a structural zero,
        // but eager forward mode must still give the reference its own zero-initialized tangent allocation, because
        // the two updates that follow add the live input tangent into it. The body maps `x` to `2 x`, so from `x = 1`
        // the loop `while (x < 4)` runs twice (`f(x) = 4 x`) and the tangent 3 becomes 12.
        let scalar = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = condition_builder.add_input(scalar.clone().into());
        let limit = condition_builder.add_constant(TestIrValue::Array(Array::scalar(4.0f64).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::<Array>::from(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![input, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let input = body_builder.add_input(scalar.into());
        let zero = body_builder.add_constant(TestIrValue::Array(Array::scalar(0.0f64).unwrap()));
        let reference =
            body_builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, input], None)
            .unwrap();
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, input], None)
            .unwrap();
        let output = body_builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let context = EagerContext::<TestIrValue, TestIrOperation>::new();
        let (primal, tangent) = context
            .jvp(
                move |input, ()| {
                    Ok(input
                        .context()
                        .bind(WhileOperation::new(), vec![condition.clone(), body.clone()], &[input.clone()])?
                        .remove(0))
                },
                TestIrValue::Array(Array::scalar(1.0f64).unwrap()),
                TestIrValue::Array(Array::scalar(3.0f64).unwrap()),
                (),
            )
            .unwrap();
        assert_eq!(primal, TestIrValue::Array(Array::scalar(4.0f64).unwrap()));
        assert_eq!(tangent, TestIrValue::Array(Array::scalar(12.0f64).unwrap()));
    }

    #[test]
    fn test_while_differentiation_keeps_tangents_of_inactive_state_structural() {
        // `limit` passes through the loop without a tangent, so both forward-mode rules keep its tangent a structural
        // zero instead of threading a materialized zero through the loop, and linearization succeeds. At `x = 2` with
        // limit `10`, the loop squares twice, so `f(x) = x⁴` with value `16` and derivative `4 x³ = 32`.
        let check = |iteration_bound: Option<usize>, expected_jvp: &str| {
            let program = pass_through_while_program(iteration_bound);
            assert_eq!(program.jvp_with_respect_to(&[0]).unwrap().to_string(), expected_jvp);
            let linearization = program.linearize_with_respect_to(&[0]).unwrap();
            let mut primal_outputs = linearization
                .primal()
                .interpret(vec![Array::scalar(2.0).unwrap(), Array::scalar(10.0).unwrap()])
                .unwrap();
            let residuals = primal_outputs.split_off(2);
            assert_eq!(primal_outputs, vec![Array::scalar(16.0).unwrap(), Array::scalar(10.0).unwrap()]);
            let mut tangent_inputs = vec![Array::scalar(1.0).unwrap()];
            tangent_inputs.extend(residuals);
            assert_eq!(
                linearization.tangent().interpret(tangent_inputs),
                Ok(vec![Array::scalar(32.0).unwrap(), Array::scalar(0.0).unwrap()]),
            );
        };

        // The bounded rule stores residuals and carries tangents for `x` alone.
        check(
            Some(3),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:i64[] = zero [type=i64[]]
                    %4:f64[3] = zero [type=f64[3]]
                    %5:bool[3] = zero [type=bool[3]]
                    %6:f64[], %7:f64[], %8:i64[], %9:f64[3], %10:bool[3] = while [iteration_bound=3] %0 %1 %3 %4 %5 [
                        condition={
                            lambda %0:f64[], %1:f64[], %2:i64[], %3:f64[3], %4:bool[3] .
                            let %5:bool[] = compare [direction=LessThan] %0 %1
                            in (%5)
                        },
                        body={
                            lambda %0:f64[], %1:f64[], %2:i64[], %3:f64[3], %4:bool[3] .
                            let %5:f64[] = mul %0 %0
                                %6:f64[1] = broadcast [output_type=f64[1], output_axes=[]] %0
                                %7:f64[3] = dynamic_update_slice %3 %6 %2
                                %8:bool[] = one [type=bool[]]
                                %9:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %8
                                %10:bool[3] = dynamic_update_slice %4 %9 %2
                                %11:i64[] = one [type=i64[]]
                                %12:i64[] = add %2 %11
                            in (%5, %1, %12, %7, %10)
                        },
                    ]
                    %11:f64[] = scan [carry_count=1, length=3, reverse=false] %2 %9 %10 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:bool[] .
                            let %4:f64[] = condition %3 %1 %2 [
                                true={
                                    lambda %0:f64[], %1:f64[] .
                                    let %2:f64[] = mul %1 %0
                                        %3:f64[] = mul %1 %0
                                        %4:f64[] = add %2 %3
                                    in (%4)
                                },
                                false={
                                    lambda %0:f64[], %1:f64[] .
                                    in (%0)
                                },
                            ]
                            in (%4)
                        },
                    ]
                    %12:f64[] = zero [type=f64[]]
                in (%6, %7, %11, %12)
            "}
            .trim_end(),
        );

        // The unbounded rule's fused loop doubles only `x`.
        check(
            None,
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[], %4:f64[], %5:f64[] = while %0 %1 %2 [
                    condition={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:bool[] = compare [direction=LessThan] %0 %1
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = mul %0 %0
                            %4:f64[] = mul %0 %2
                            %5:f64[] = mul %0 %2
                            %6:f64[] = add %4 %5
                        in (%3, %1, %6)
                    },
                ]
                    %6:f64[] = zero [type=f64[]]
                in (%3, %4, %5, %6)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_while_differentiation_enlivens_state_whose_tangents_become_live() {
        // Only `source` has a live tangent. It reaches `second` after one iteration and `first` after two, so the carry
        // fixed point takes two rounds to enliven them, while `counter` never depends on a live tangent and keeps a
        // structural-zero tangent. Both forward-mode rules carry tangents for exactly `first`, `second`, and `source`,
        // and the linearized body retains that entire dependent carry chain across repeated applications.
        let check = |iteration_bound: Option<usize>, expected_jvp: &str| {
            let program = chained_carry_while_program(iteration_bound);
            assert_eq!(program.jvp_with_respect_to(&[3]).unwrap().to_string(), expected_jvp);
            let linearization = program.linearize_with_respect_to(&[3]).unwrap();
            let mut primal_outputs = linearization
                .primal()
                .interpret(vec![
                    Array::scalar(3.0).unwrap(),
                    Array::scalar(1.0).unwrap(),
                    Array::scalar(1.0).unwrap(),
                    Array::scalar(7.0).unwrap(),
                ])
                .unwrap();
            let residuals = primal_outputs.split_off(4);
            assert_eq!(
                primal_outputs,
                vec![
                    Array::scalar(0.0).unwrap(),
                    Array::scalar(7.0).unwrap(),
                    Array::scalar(7.0).unwrap(),
                    Array::scalar(7.0).unwrap(),
                ],
            );
            for tangent in [2.0, 5.0] {
                let mut tangent_inputs = vec![Array::scalar(tangent).unwrap()];
                tangent_inputs.extend(residuals.clone());
                assert_eq!(
                    linearization.tangent().interpret(tangent_inputs),
                    Ok(vec![
                        Array::scalar(0.0).unwrap(),
                        Array::scalar(tangent).unwrap(),
                        Array::scalar(tangent).unwrap(),
                        Array::scalar(tangent).unwrap(),
                    ]),
                );
            }
        };

        // The bounded rule's tangent scan carries the three live tangents through its guarded steps.
        check(
            Some(4),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                let %5:i64[] = zero [type=i64[]]
                    %6:bool[4] = zero [type=bool[4]]
                    %7:f64[], %8:f64[], %9:f64[], %10:f64[], %11:i64[], %12:bool[4] = while [iteration_bound=4] %0 %1 \
                        %2 %3 %5 %6 [
                        condition={
                            lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:i64[], %5:bool[4] .
                            let %6:f64[] = const 0.0
                                %7:bool[] = compare [direction=GreaterThan] %0 %6
                            in (%7)
                        },
                        body={
                            lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:i64[], %5:bool[4] .
                            let %6:f64[] = const 1.0
                                %7:f64[] = sub %0 %6
                                %8:bool[] = one [type=bool[]]
                                %9:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %8
                                %10:bool[4] = dynamic_update_slice %5 %9 %4
                                %11:i64[] = one [type=i64[]]
                                %12:i64[] = add %4 %11
                            in (%7, %2, %3, %3, %12, %10)
                        },
                    ]
                    %13:f64[] = zero [type=f64[]]
                    %14:f64[] = zero [type=f64[]]
                    %15:f64[], %16:f64[], %17:f64[] = scan [carry_count=3, length=4, reverse=false] %13 %14 %4 %12 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[], %3:f64[], %4:bool[] .
                            let %5:f64[], %6:f64[], %7:f64[] = condition %4 %1 %2 %3 [
                                true={
                                    lambda %0:f64[], %1:f64[], %2:f64[] .
                                    in (%1, %2, %2)
                                },
                                false={
                                    lambda %0:f64[], %1:f64[], %2:f64[] .
                                    in (%0, %1, %2)
                                },
                            ]
                            in (%5, %6, %7)
                        },
                    ]
                    %18:f64[] = zero [type=f64[]]
                in (%7, %8, %9, %10, %18, %15, %16, %17)
            "}
            .trim_end(),
        );

        // The unbounded rule's fused loop doubles the three live state elements.
        check(
            None,
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[] .
                let %5:f64[] = zero [type=f64[]]
                    %6:f64[] = zero [type=f64[]]
                    %7:f64[], %8:f64[], %9:f64[], %10:f64[], %11:f64[], %12:f64[], %13:f64[] = while %0 %1 %2 %3 %5 %6 \
                        %4 [
                        condition={
                            lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[], %5:f64[], %6:f64[] .
                            let %7:f64[] = const 0.0
                                %8:bool[] = compare [direction=GreaterThan] %0 %7
                            in (%8)
                        },
                        body={
                            lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[], %4:f64[], %5:f64[], %6:f64[] .
                            let %7:f64[] = const 1.0
                                %8:f64[] = sub %0 %7
                            in (%8, %2, %3, %3, %5, %6, %6)
                        },
                    ]
                    %14:f64[] = zero [type=f64[]]
                in (%7, %8, %9, %10, %14, %11, %12, %13)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_while_differentiation_computes_gradients_through_staged_masked_scans() {
        // The headline bounded-while capability: end-to-end reverse mode through a *staged* while loop.
        // `f(x) = while (x < 8, iteration_bound = 5) { x = 2 * x }` at `x = 1` runs three iterations (`x` visits 1,
        // 2, 4), so the actual trip count 3 is strictly below the bound 5 and the two trailing stack slots matter:
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
                Array::scalar(1.0).unwrap(),
                (),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();
        assert_eq!(output.to_f64s(), vec![8.0]);

        // The pullback contains the transposed (reversed) linear scan and no while loop, and every cotangent seed
        // scales the hand-computed gradient 8. The direct-transpose pullback consumes `[cotangent ++ residuals]`.
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[5] .
                let %2:f64[] = scan [carry_count=1, length=5, reverse=true] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:bool[] .
                        let %3:f64[] = condition %2 %1 [
                            true={
                                lambda %0:f64[] .
                                let %1:f64[] = const 2.0
                                    %2:f64[] = mul %1 %0
                                in (%2)
                            },
                            false={
                                lambda %0:f64[] .
                                in (%0)
                            },
                        ]
                        in (%3)
                    },
                ]
                in (%2)"},
        );
        let pullback_inputs = |cotangent: Array| {
            let mut inputs = vec![cotangent];
            inputs.extend(residuals.iter().cloned());
            inputs
        };
        assert_eq!(
            pullback
                .interpret(pullback_inputs(Array::scalar(1.0).unwrap()))
                .map(|cotangents| cotangents[0].to_f64s()),
            Ok(vec![8.0]),
        );
        assert_eq!(
            pullback
                .interpret(pullback_inputs(Array::scalar(2.0).unwrap()))
                .map(|cotangents| cotangents[0].to_f64s()),
            Ok(vec![16.0]),
        );

        // `value_and_gradient` composes the same machinery end to end.
        let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
        let (value, gradient) = StagedDispatchTestDomain
            .differentiate_at(Array::scalar(1.0).unwrap())
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
    fn test_while_differentiation_stores_loop_varying_residual_stacks() {
        // The store-instead-of-recompute proof: `while (x < 100, iteration_bound = 4) { x = x * x }` at `x = 2`
        // squares three times (`x` visits 2, 4, 16 → 256, trip count 3 < bound 4), and the product rule references
        // the *per-iteration* state as a loop-varying residual, so the gradient depends on the stored stack slots
        // `[2, 4, 16, 0]`, including the zero slot beyond the trip count, which the mask must keep inert in both
        // directions. Locally `f(x) = x⁸`: value 256 and gradient `8 x⁷ = 1024`.
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
                Array::scalar(2.0).unwrap(),
                (),
            )
            .unwrap();
        let (pullback, residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();
        assert_eq!(output.to_f64s(), vec![256.0]);
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[4], %2:f64[4] .
                let %3:f64[] = scan [carry_count=1, length=4, reverse=true] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:bool[], %3:f64[] .
                        let %4:f64[] = condition %2 %1 %3 [
                            true={
                                lambda %0:f64[], %1:f64[] .
                                let %2:f64[] = mul %1 %0
                                    %3:f64[] = mul %1 %0
                                    %4:f64[] = add %2 %3
                                in (%4)
                            },
                            false={
                                lambda %0:f64[], %1:f64[] .
                                in (%0)
                            },
                        ]
                        in (%4)
                    },
                ]
                in (%3)"},
        );
        assert_eq!(
            residuals,
            vec![
                Array::vector(vec![true, true, true, false]).unwrap(),
                Array::vector(vec![2.0, 4.0, 16.0, 0.0]).unwrap(),
            ],
        );
        let mut pullback_inputs = vec![Array::scalar(1.0).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs).map(|cotangents| cotangents[0].to_f64s()), Ok(vec![1024.0]));

        // The eager-domain reverse-mode entry point produces the same value and gradient numbers.
        let (while_operation, while_regions) = bounded_squaring_while_operation(100.0, 4);
        let (value, gradient) = differentiate_at(Array::scalar(2.0).unwrap())
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
        let (while_operation, while_regions) = bounded_squaring_while_operation(100.0, 4);
        check_gradient!(
            |x| bind_while(while_operation, while_regions.clone(), &[x]).map(|mut outputs| outputs.remove(0)),
            at = Array::scalar(2.0).unwrap(),
            step = 1e-4,
            tolerance = 1e-3,
        );
    }

    #[test]
    fn test_while_differentiation_skips_inactive_singular_residuals() {
        // Quotient derivatives read the loop-varying denominator. Unused residual slots contain zero, so evaluating
        // their pushforward or pullback would divide by zero even though the actual loop never visits zero.
        for (initial, expected_value, expected_gradient) in [(0.5, 1.0, 0.0), (2.0, 2.0, 1.0)] {
            let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
            let state = builder.add_input(ArrayType::scalar(DataType::F64));
            let quotient =
                builder.add_instruction(DivOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
            let body = builder.build(vec![quotient], vec![Placeholder], vec![Placeholder]).unwrap();
            let regions = vec![scalar_threshold_condition(1.0), body];
            let operation = WhileOperation::new().with_iteration_bound(3).unwrap();
            check_gradient!(
                |x| bind_while(operation, regions.clone(), &[x]).map(|mut outputs| outputs.remove(0)),
                at = Array::scalar(initial).unwrap(),
                step = 1e-4,
                tolerance = 1e-6,
            );
            let (value, gradient) = StagedDispatchTestDomain
                .differentiate_at(Array::scalar(initial).unwrap())
                .value_and_gradient(move |state| {
                    state.context().bind(operation, regions.clone(), &[state.clone()]).unwrap().remove(0)
                })
                .unwrap();
            assert_eq!(value.to_f64s(), vec![expected_value]);
            assert_eq!(gradient.to_f64s(), vec![expected_gradient]);
        }
    }

    #[test]
    fn test_while_differentiation_supports_vector_state() {
        // The bounded staged path differentiates loops over vector state, whose per-iteration residuals are vectors
        // rather than scalars. The loop `while (sum(x) < 20, iteration_bound = 4) { x = x * x }` at `x = [1.5, 2]`
        // squares twice (sums visit 3.5 and 6.25 before reaching 21.0625), so `f(x) = sum(x⁴)` locally: value
        // `1.5⁴ + 2⁴ = 21.0625` and gradient `4 x³ = [13.5, 32]`, with trip count 2 strictly below the bound 4.
        let vector_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let mut condition_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let condition_state = condition_builder.add_input(vector_f64.clone());
        let summed = condition_builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![condition_state], None)
            .unwrap()[0];
        let threshold = condition_builder.add_constant(Array::scalar(20.0).unwrap());
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![summed, threshold],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let body_state = body_builder.add_input(vector_f64.clone());
        let squared = body_builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![body_state, body_state], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let while_operation = WhileOperation::new().with_iteration_bound(4).unwrap();
        let while_regions = vec![condition, body];
        check_gradient!(
            |x| {
                let state = bind_while(while_operation, while_regions.clone(), &[x])?.remove(0);
                let sum = ReduceOperation::new(vec![0], ReductionKind::Sum);
                state.domain().bind(sum, Vec::new(), &[state.clone()]).map(|mut outputs| outputs.remove(0))
            },
            at = Array::vector(vec![1.5, 2.0]).unwrap(),
            step = 1e-4,
            tolerance = 1e-3,
        );

        let (value, gradient) = StagedDispatchTestDomain
            .differentiate_at(Array::vector(vec![1.5, 2.0]).unwrap())
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
    fn test_while_differentiation_never_reads_the_placeholder_residuals_of_zero_trip_loops() {
        // At `x = 200`, the condition of `while (x < 100, iteration_bound = 4) { x = x * x }` is false at entry, so the
        // staged loop runs no iteration: every mask entry stays false and every residual stack slot keeps its zero
        // placeholder, which the guarded tangent steps never read. Locally `f(x) = x`: value 200 and derivative 1.
        let (operation, regions) = bounded_squaring_while_operation(100.0, 4);
        let program = while_program(operation, &regions, vec![ArrayType::scalar(DataType::F64)]);
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![Array::scalar(200.0).unwrap()]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![Array::scalar(200.0).unwrap()]);
        assert_eq!(residuals, vec![Array::vector(vec![false; 4]).unwrap(), Array::vector(vec![0.0f64; 4]).unwrap()],);
        let mut tangent_inputs = vec![Array::scalar(3.0).unwrap()];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![Array::scalar(3.0).unwrap()]));
        let mut pullback_inputs = vec![Array::scalar(1.0).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![Array::scalar(1.0).unwrap()]),
        );

        // The staged domain composes the same masked scan end to end, and finite differences confirm the identity.
        let staged_regions = regions.clone();
        assert_eq!(
            StagedDispatchTestDomain
                .differentiate_at(Array::scalar(200.0).unwrap())
                .value_and_gradient(move |x| {
                    bind_while(operation, staged_regions.clone(), &[x]).unwrap().remove(0)
                }),
            Ok((Array::scalar(200.0).unwrap(), Array::scalar(1.0).unwrap())),
        );
        check_gradient!(
            |x| bind_while(operation, regions.clone(), &[x]).map(|mut outputs| outputs.remove(0)),
            at = Array::scalar(200.0).unwrap(),
            step = 1e-4,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_while_differentiation_truncates_bounded_loops_consistently_across_paths() {
        // A loop whose condition never turns false truncates at the bound by definition: with bound 3 the doubling
        // loop computes `f(x) = 8 x`, so at `x = 2` plain interpretation computes the value 16, and the eager-domain
        // entry point and the staged domain (whose validity mask is true for every iteration) both compute the value
        // 16 and the gradient 8.
        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let outputs = EagerContext::<Array, ArrayOperation<Array>>::new()
            .bind(while_operation, while_regions, &[Array::scalar(2.0).unwrap()])
            .unwrap();
        assert_eq!(outputs[0].to_f64s(), vec![16.0]);

        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = differentiate_at(Array::scalar(2.0).unwrap())
            .value_and_gradient(move |x| {
                let mut outputs = x.context().bind(while_operation, while_regions.clone(), &[x.clone()]).unwrap();
                outputs.remove(0)
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![16.0]);
        assert_eq!(gradient.to_f64s(), vec![8.0]);

        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = StagedDispatchTestDomain
            .differentiate_at(Array::scalar(2.0).unwrap())
            .value_and_gradient(move |x| {
                let mut outputs = x.context().bind(while_operation, while_regions.clone(), &[x.clone()]).unwrap();
                outputs.remove(0)
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![16.0]);
        assert_eq!(gradient.to_f64s(), vec![8.0]);
        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        check_gradient!(
            |x| bind_while(while_operation, while_regions.clone(), &[x]).map(|mut outputs| outputs.remove(0)),
            at = Array::scalar(2.0).unwrap(),
            step = 1e-4,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_while_differentiation_truncates_masked_loops_at_the_iteration_bound() {
        // Under the batched predicate `x < [100, 3]`, `while (iteration_bound = 2) { x = x * x }` at `x = [2, 2]` stops
        // item 1 after one iteration (2 -> 4) and truncates item 0 after two (2 -> 4 -> 16) while its predicate is
        // still true. Locally `f(x) = [x₀⁴, x₁²]`: value `[16, 4]` and per-item derivatives `[4 x₀³, 2 x₁] = [32, 4]`.
        let state_type = ArrayType::new_static(DataType::F64, [2]);
        let mut condition_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = condition_builder.add_input(state_type.clone());
        let limits = condition_builder.add_constant(Array::vector(vec![100.0, 3.0]).unwrap());
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![state, limits],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = body_builder.add_input(state_type.clone());
        let squared =
            body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = WhileOperation::new().with_iteration_bound(2).unwrap();
        let regions = vec![condition, body];
        let program = while_program(operation, &regions, vec![state_type]);

        // The staged masked normal form reproduces the truncated values in both directions.
        let linearization = program.linearize().unwrap();
        let mut primal_outputs =
            linearization.primal().interpret(vec![Array::vector(vec![2.0, 2.0]).unwrap()]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![Array::vector(vec![16.0, 4.0]).unwrap()]);
        let mut tangent_inputs = vec![Array::vector(vec![1.0, 1.0]).unwrap()];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(
            linearization.tangent().interpret(tangent_inputs),
            Ok(vec![Array::vector(vec![32.0, 4.0]).unwrap()]),
        );
        let mut pullback_inputs = vec![Array::vector(vec![1.0, 2.0]).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![Array::vector(vec![32.0, 8.0]).unwrap()]),
        );

        // The eager rule cannot concretize the batched predicate, so it differentiates the same masked normal form.
        assert_eq!(
            differentiate_at(Array::vector(vec![2.0, 2.0]).unwrap()).jvp(Array::vector(vec![1.0, 1.0]).unwrap(), |x| {
                bind_while(operation, regions.clone(), &[x]).map(|mut outputs| outputs.remove(0))
            }),
            Ok((Array::vector(vec![16.0, 4.0]).unwrap(), Array::vector(vec![32.0, 4.0]).unwrap())),
        );
        check_gradient!(
            |x| {
                let state = bind_while(operation, regions.clone(), &[x])?.remove(0);
                let sum = ReduceOperation::new(vec![0], ReductionKind::Sum);
                state.domain().bind(sum, Vec::new(), &[state.clone()]).map(|mut outputs| outputs.remove(0))
            },
            at = Array::vector(vec![2.0, 2.0]).unwrap(),
            step = 1e-4,
            tolerance = 1e-3,
        );
    }

    #[test]
    fn test_while_differentiation_indexes_varying_residual_stacks_in_manual_regions() {
        // Inside a manual region, the residual stacks vary like the residuals that they store, while the loop counter
        // and the zero index are invariant integers. The augmented body therefore varies the counter through
        // `parallel_vary` and creates the zero index, a non-differentiable constant, with that variation directly
        // before it writes into a varying stack.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let counter_type = ArrayType::scalar(DataType::I64);
        let state_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let counter = builder.add_input(counter_type.clone());
        builder.add_input(state_type.clone());
        let limit = builder.add_constant(Array::scalar(100i64).unwrap());
        let predicate = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let condition = builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let counter = builder.add_input(counter_type.clone());
        let state = builder.add_input(state_type.clone());
        let one = builder.add_constant(Array::scalar(1i64).unwrap());
        let next_counter =
            builder.add_instruction(AddOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
        let next_state = builder.add_instruction(SinOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_counter, next_state], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<Tracer<TracingContext<Array, ArrayOperation<Array>>>>| {
                inputs[0].domain().bind(
                    ArrayOperation::While(WhileOperation::new().with_iteration_bound(3)?),
                    vec![condition, body],
                    &inputs,
                )
            },
            vec![counter_type, state_type],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        let jvp = program.to_flat_program().jvp_with_respect_to(&[1]).unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:i64[], %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %2:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %3:i64[] = zero [type=i64[]]
                    %4:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        zero [type=f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}]]
                    %5:bool[3] = zero [type=bool[3]]
                    %6:i64[], %7:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], %8:i64[], \
                        %9:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], %10:bool[3] = \
                        while [iteration_bound=3] %0 %1 %3 %4 %5 [
                        condition={
                            lambda %0:i64[], %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                                %2:i64[], \
                                %3:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                                %4:bool[3] .
                            let %5:i64[] = const 100
                                %6:bool[] = compare [direction=LessThan] %0 %5
                            in (%6)
                        },
                        body={
                            lambda %0:i64[], %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                                %2:i64[], \
                                %3:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                                %4:bool[3] .
                            let %5:i64[] = const 1
                                %6:i64[] = add %0 %5
                                %7:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = sin %1
                                %8:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = cos %1
                                %9:f32[1, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                                    broadcast [
                                    output_type=f32[1, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], \
                                        varying_manual={'x'}}],
                                    output_axes=[1],
                                ] %8
                                %10:i64[][sharding={mesh<['x'=2:manual]>, []}] = \
                                    broadcast [output_type=i64[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] \
                                    %2
                                %11:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                                    parallel_vary [axis_name=\"x\"] %10
                                %12:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                                    zero [type=i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]]
                                %13:f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                                    dynamic_update_slice %3 %9 %11 %12
                                %14:bool[] = one [type=bool[]]
                                %15:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %14
                                %16:bool[3] = dynamic_update_slice %4 %15 %2
                                %17:i64[] = one [type=i64[]]
                                %18:i64[] = add %2 %17
                            in (%6, %7, %18, %13, %16)
                        },
                    ]
                    %11:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        scan [carry_count=1, length=3, reverse=false] %2 %9 %10 [
                        body={
                            lambda %0:i64[], %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                                %2:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], %3:bool[] .
                            let %4:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                                condition %3 %1 %2 [
                                true={
                                    lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                                        %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                                    let %2:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                                        mul %1 %0
                                    in (%2)
                                },
                                false={
                                    lambda %0:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                                        %1:f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                                    in (%0)
                                },
                            ]
                            in (%4)
                        },
                    ]
                in (%6, %7, %11)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_while_differentiation_types_counter_and_mask_with_varying_predicates() {
        // A varying predicate makes devices run different numbers of iterations, so the loop counter and the validity
        // mask, which are non-differentiable integers and Booleans, are created varying like the predicate. The tangent
        // scan's per-step `condition` guard is then a varying predicate whose outputs, the tangent carries, vary too.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let state_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(state_type.clone());
        let limit = builder.add_constant(Array::from_elements(state_type.clone(), &[10f32]).unwrap());
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![state, limit], None)
            .unwrap()[0];
        let condition = builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(state_type.clone());
        let next_state = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_state], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<Tracer<TracingContext<Array, ArrayOperation<Array>>>>| {
                inputs[0].domain().bind(
                    ArrayOperation::While(WhileOperation::new().with_iteration_bound(3)?),
                    vec![condition, body],
                    &inputs,
                )
            },
            vec![state_type.clone()],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        let program = program.to_flat_program();
        let jvp = program.jvp_with_respect_to(&[0]).unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                    %1:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
                let %2:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = zero \
                    [type=i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]]
                    %3:f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero \
                        [type=f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]]
                    %4:bool[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = zero \
                        [type=bool[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]]
                    %5:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                        %6:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                        %7:f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                        %8:bool[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = while \
                        [iteration_bound=3] %0 %2 %3 %4 [
                        condition={
                            lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                                %1:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                                %2:f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                                %3:bool[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                            let %4:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = const 10.0
                                %5:bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = compare \
                                    [direction=LessThan] %0 %4
                            in (%5)
                        },
                        body={
                            lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                                %1:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                                %2:f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                                %3:bool[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                            let %4:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = mul %0 %0
                                %5:f32[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                    output_type=f32[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                    output_axes=[],
                                ] %0
                                %6:f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                                    dynamic_update_slice %2 %5 %1
                                %7:bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = one \
                                    [type=bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]]
                                %8:bool[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = broadcast [
                                    output_type=bool[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}],
                                    output_axes=[],
                                ] %7
                                %9:bool[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                                    dynamic_update_slice %3 %8 %1
                                %10:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = one \
                                    [type=i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]]
                                %11:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = add %1 %10
                            in (%4, %11, %6, %9)
                        },
                    ]
                    %9:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = scan [carry_count=1, \
                        length=3, reverse=false] %1 %7 %8 [
                        body={
                            lambda %0:i64[], %1:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                                %2:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                                %3:bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
                            let %4:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = condition %3 %1 \
                                %2 [
                                true={
                                    lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                                        %1:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
                                    let %2:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = mul %1 %0
                                        %3:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = mul %1 %0
                                        %4:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = add %2 %3
                                    in (%4)
                                },
                                false={
                                    lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                                        %1:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
                                    in (%0)
                                },
                            ]
                            in (%4)
                        },
                    ]
                in (%5, %9)
            "}
            .trim_end(),
        );

        // Reverse mode transposes the tangent scan, including its varying `condition` guard. At `x = 2` with limit
        // `10`, the loop squares twice, so `f(x) = x⁴` with value `16` and gradient `4 x³ = 32`.
        let linearization = program.linearize_with_respect_to(&[0]).unwrap();
        let pullback = linearization.pullback().unwrap().into_program();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                    %1:bool[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
                    %2:f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %3:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = scan [carry_count=1, \
                    length=3, reverse=true] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                            %2:bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                            %3:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
                        let %4:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = condition %2 %1 %3 [
                            true={
                                lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                                    %1:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
                                let %2:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = mul %1 %0
                                    %3:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = mul %1 %0
                                    %4:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = add %2 %3
                                in (%4)
                            },
                            false={
                                lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}], \
                                    %1:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
                                in (%0)
                            },
                        ]
                        in (%4)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        let state_value = |value: f32| Array::from_elements(state_type.clone(), &[value]).unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![state_value(2.0)]).unwrap();
        assert_eq!(primal_outputs[0], state_value(16.0));
        let mut pullback_inputs = vec![state_value(1.0)];
        pullback_inputs.extend(primal_outputs.split_off(1));
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![state_value(32.0)]));
    }

    #[test]
    fn test_while_differentiation_masks_varying_state_under_invariant_batched_predicates() {
        // The masked normal form of a batched-predicate loop selects every array state element under the loop mask.
        // Inside a manual region, the Boolean mask, which carries no tangent, first takes the variation of the state
        // element that it selects through `parallel_vary`, and its broadcast keeps the resulting sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let counter_type = ArrayType::new_static(DataType::I64, [2]);
        let state_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let counter = builder.add_input(counter_type.clone());
        builder.add_input(state_type.clone());
        let limit = builder.add_constant(Array::vector(vec![1i64, 2]).unwrap());
        let predicate = builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let condition = builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let counter = builder.add_input(counter_type.clone());
        let state = builder.add_input(state_type.clone());
        let one = builder.add_constant(Array::vector(vec![1i64, 1]).unwrap());
        let next_counter =
            builder.add_instruction(AddOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
        let next_state = builder.add_instruction(SinOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_counter, next_state], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<Tracer<TracingContext<Array, ArrayOperation<Array>>>>| {
                inputs[0].domain().bind(
                    ArrayOperation::While(WhileOperation::new().with_iteration_bound(3)?),
                    vec![condition, body],
                    &inputs,
                )
            },
            vec![counter_type, state_type],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        let jvp = program.to_flat_program().jvp_with_respect_to(&[1]).unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:i64[2], %1:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                    %2:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                let %3:i64[2] = const [1, 2]
                    %4:bool[2] = compare [direction=LessThan] %0 %3
                    %5:i64[] = zero [type=i64[]]
                    %6:f32[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}] = zero [
                        type=f32[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}],
                    ]
                    %7:bool[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}] = zero [
                        type=bool[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}],
                    ]
                    %8:bool[3] = zero [type=bool[3]]
                    %9:i64[2], %10:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                        %11:bool[2], %12:i64[], \
                        %13:f32[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}], \
                        %14:bool[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}], \
                        %15:bool[3] = while [iteration_bound=3] %0 %1 %4 %5 %6 %7 %8 [
                        condition={
                            lambda %0:i64[2], \
                                %1:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                                %2:bool[2], %3:i64[], \
                                %4:f32[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}], \
                                %5:bool[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}], \
                                %6:bool[3] .
                            let %7:bool[] = reduce [kind=any, axes=[0]] %2
                            in (%7)
                        },
                        body={
                            lambda %0:i64[2], \
                                %1:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                                %2:bool[2], %3:i64[], \
                                %4:f32[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}], \
                                %5:bool[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}], \
                                %6:bool[3] .
                            let %7:i64[2] = const [1, 1]
                                %8:i64[2] = const [1, 2]
                                %9:i64[2] = add %0 %7
                                %10:i64[2] = select %2 %9 %0
                                %11:bool[2][sharding={mesh<['x'=2:manual]>, [{}]}] = \
                                    broadcast [output_type=bool[2][sharding={mesh<['x'=2:manual]>, [{}]}], \
                                    output_axes=[0]] %2
                                %12:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                                    parallel_vary [axis_name=\"x\"] %11
                                %13:bool[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                                    broadcast [
                                    output_type=bool[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], \
                                        varying_manual={'x'}}],
                                    output_axes=[0],
                                ] %12
                                %14:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = sin %1
                                %15:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                                    select %13 %14 %1
                                %16:bool[2] = compare [direction=LessThan] %10 %8
                                %17:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = cos %1
                                %18:f32[1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}] \
                                    = broadcast [
                                    output_type=f32[1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], \
                                        varying_manual={'x'}}],
                                    output_axes=[1, 2],
                                ] %17
                                %19:i64[][sharding={mesh<['x'=2:manual]>, []}] = \
                                    broadcast [output_type=i64[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] \
                                    %3
                                %20:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                                    parallel_vary [axis_name=\"x\"] %19
                                %21:i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                                    zero [type=i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]]
                                %22:f32[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}] \
                                    = dynamic_update_slice %4 %18 %20 %21 %21
                                %23:bool[1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}] \
                                    = broadcast [
                                    output_type=bool[1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], \
                                        varying_manual={'x'}}],
                                    output_axes=[1, 2],
                                ] %13
                                %24:bool[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}] \
                                    = dynamic_update_slice %5 %23 %20 %21 %21
                                %25:bool[] = one [type=bool[]]
                                %26:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %25
                                %27:bool[3] = dynamic_update_slice %6 %26 %3
                                %28:i64[] = one [type=i64[]]
                                %29:i64[] = add %3 %28
                            in (%10, %15, %16, %29, %22, %24, %27)
                        },
                    ]
                    %16:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                        scan [carry_count=1, length=3, reverse=false] %2 %13 %14 %15 [
                        body={
                            lambda %0:i64[], \
                                %1:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                                %2:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                                %3:bool[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                                %4:bool[] .
                            let %5:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] = \
                                condition %4 %1 %2 %3 [
                                true={
                                    lambda %0:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], \
                                        varying_manual={'x'}}], \
                                        %1:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                                        %2:bool[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                                    let %3:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] \
                                        = mul %1 %0
                                        %4:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] \
                                            = select %2 %3 %0
                                    in (%4)
                                },
                                false={
                                    lambda %0:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], \
                                        varying_manual={'x'}}], \
                                        %1:f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}], \
                                        %2:bool[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}] .
                                    in (%0)
                                },
                            ]
                            in (%5)
                        },
                    ]
                in (%9, %10, %16)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_while_differentiation_runs_bounded_loops_in_shard_map_bodies() {
        /// Builds a program that applies `shard_map` to inputs of the `global_types`, with a body that runs the
        /// bounded `while` loop over the condition and body `regions` on the local values of those inputs.
        fn mapped_while_program(
            shard_map: ShardMap,
            global_types: Vec<ArrayType>,
            regions: Vec<Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>>,
        ) -> Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>> {
            let state_count = global_types.len();
            let mut map_body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let regions = regions.into_iter().map(|region| map_body.import_program(region)).collect();
            let inputs = global_types
                .iter()
                .enumerate()
                .map(|(index, r#type)| map_body.add_input(shard_map.local_input_type(index, r#type).unwrap().into()))
                .collect();
            let while_operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(3).unwrap();
            let outputs = map_body.add_instruction(while_operation, regions, inputs, None).unwrap().to_vec();
            let map_body = map_body
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    outputs,
                    vec![Placeholder; state_count],
                    vec![Placeholder; state_count],
                )
                .unwrap();
            let global_types = global_types.into_iter().map(ArrayIrType::from).collect::<Vec<_>>();
            let operation = ShardMapOperation::from_program(&map_body, global_types.clone(), shard_map).unwrap();
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let map_body = builder.import_program(map_body);
            let inputs = global_types.into_iter().map(|r#type| builder.add_input(r#type)).collect();
            let outputs = builder.add_instruction(operation, vec![map_body], inputs, None).unwrap().to_vec();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    outputs,
                    vec![Placeholder; state_count],
                    vec![Placeholder; state_count],
                )
                .unwrap()
        }

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_type = ArrayType::new_static(DataType::F32, [4]);
        let compare =
            || TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan)));
        let mul = || TestIrOperation::Array(ArrayOperation::from(MulOperation::new()));
        let sharded_type = global_type.clone().with_sharding(sharded.clone()).unwrap();
        let sharded_array = |values: &[f32]| array(Array::from_elements(sharded_type.clone(), values).unwrap());

        // Each device runs `while (x < 10) { x = x * x }` over its own `f32[2]` shard under a varying batched
        // predicate, so its items stop after different numbers of iterations: device 0 holds `[2, 5]` (two iterations
        // and one) and device 1 holds `[5, 3]` (one and two). The emulator runs the devices in lockstep, so both run
        // two iterations of the masked normal form, whose per-item masks differ across devices. Locally
        // `f = [x⁴, x², x², x⁴]`: value `[16, 25, 25, 81]` and derivative `[4 x³, 2 x, 2 x, 4 x³] = [32, 10, 10, 108]`.
        let shard_map = ShardMap::new(mesh.clone(), vec![sharded.clone()], vec![sharded.clone()], Vec::new()).unwrap();
        let local_type = shard_map.local_input_type(0, &global_type).unwrap();
        let mut condition = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = condition.add_input(local_type.clone().into());
        let limit = condition.add_constant(array(Array::from_elements(local_type.clone(), &[10f32, 10.0]).unwrap()));
        let predicate = condition.add_instruction(compare(), Vec::new(), vec![state, limit], None).unwrap()[0];
        let condition = condition
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = body.add_input(local_type.into());
        let squared = body.add_instruction(mul(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let program = mapped_while_program(shard_map, vec![global_type.clone()], vec![condition, body]);
        let x = array(Array::vector(vec![2f32, 5.0, 5.0, 3.0]).unwrap());
        assert_eq!(
            program.jvp().unwrap().interpret(vec![x.clone(), array(Array::vector(vec![1f32; 4]).unwrap())]),
            Ok(vec![sharded_array(&[16.0, 25.0, 25.0, 81.0]), sharded_array(&[32.0, 10.0, 10.0, 108.0])]),
        );
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![x]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![sharded_array(&[16.0, 25.0, 25.0, 81.0])]);
        let mut pullback_inputs = vec![sharded_array(&[1.0, 1.0, 2.0, 1.0])];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![32f32, 10.0, 20.0, 108.0]).unwrap())]),
        );

        // A replicated counter drives `while (counter < 2) { (counter, x) = (counter + 1, x * x) }`, so the predicate
        // is invariant while the stored residual `x` varies, and forward mode, linearization, and the pullback still
        // read each device's own residual stack. Two iterations give `[x⁴, …] = [16, 625, 625, 81]` with derivative
        // `4 x³ = [32, 500, 500, 108]`.
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![replicated.clone(), sharded.clone()],
            vec![replicated.clone(), sharded],
            Vec::new(),
        )
        .unwrap();
        let counter_type = shard_map.local_input_type(0, &ArrayType::scalar(DataType::I64)).unwrap();
        let local_type = shard_map.local_input_type(1, &global_type).unwrap();
        let mut condition = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = condition.add_input(counter_type.clone().into());
        condition.add_input(local_type.clone().into());
        let limit = condition.add_constant(array(Array::from_elements(counter_type.clone(), &[2i64]).unwrap()));
        let predicate = condition.add_instruction(compare(), Vec::new(), vec![counter, limit], None).unwrap()[0];
        let condition = condition
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = body.add_input(counter_type.clone().into());
        let state = body.add_input(local_type.into());
        let one = body.add_constant(array(Array::from_elements(counter_type, &[1i64]).unwrap()));
        let next_counter = body
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![counter, one],
                None,
            )
            .unwrap()[0];
        let squared = body.add_instruction(mul(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_counter, squared],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let program =
            mapped_while_program(shard_map, vec![ArrayType::scalar(DataType::I64), global_type], vec![condition, body]);
        let counter = array(Array::scalar(0i64).unwrap());
        let x = array(Array::vector(vec![2f32, 5.0, 5.0, 3.0]).unwrap());
        let final_counter = array(
            Array::from_elements(ArrayType::scalar(DataType::I64).with_sharding(replicated).unwrap(), &[2i64]).unwrap(),
        );
        assert_eq!(
            program.jvp_with_respect_to(&[1]).unwrap().interpret(vec![
                counter.clone(),
                x.clone(),
                array(Array::vector(vec![1f32; 4]).unwrap()),
            ]),
            Ok(vec![
                final_counter.clone(),
                sharded_array(&[16.0, 625.0, 625.0, 81.0]),
                sharded_array(&[32.0, 500.0, 500.0, 108.0]),
            ]),
        );
        let linearization = program.linearize_with_respect_to(&[1]).unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![counter, x]).unwrap();
        let residuals = primal_outputs.split_off(2);
        assert_eq!(primal_outputs, vec![final_counter, sharded_array(&[16.0, 625.0, 625.0, 81.0])]);
        let mut pullback_inputs = vec![sharded_array(&[1.0; 4])];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![32f32, 500.0, 500.0, 108.0]).unwrap())]),
        );
    }

    #[test]
    fn test_while_differentiation_rejects_dynamically_shaped_residual_storage() {
        // The bounded rule stores one residual per iteration in a `[bound, …]` stack, which needs a statically shaped
        // storage type, but squaring an `f64[n]` state stores the dynamically shaped state itself.
        let n = DimensionVariable::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(n)]));
        let mut condition_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = condition_builder.add_input(state_type.clone());
        let sum = condition_builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![state], None)
            .unwrap()[0];
        let limit = condition_builder.add_constant(Array::scalar(100.0).unwrap());
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![sum, limit], None)
            .unwrap()[0];
        let condition = condition_builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = body_builder.add_input(state_type.clone());
        let squared =
            body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = body_builder.build(vec![squared], vec![Placeholder], vec![Placeholder]).unwrap();
        let operation = WhileOperation::new().with_iteration_bound(3).unwrap();
        let program = while_program(operation, &[condition, body], vec![state_type]);
        assert_eq!(
            program.jvp().map(|_| ()),
            Err(DifferentiationError::from(ProgramError::from(TypeError::invalid(
                "`jvp` of a bounded `while` loop requires statically shaped residual storage but got `f64[n]`",
            )))),
        );
    }

    #[test]
    fn test_while_differentiation_rejects_invariant_residuals_under_varying_predicates() {
        // Devices whose predicates differ fill different numbers of stack slots, so every residual stack must vary
        // over the manual axes of the predicate. In `while (s < 10) { s = s + parallel_vary(sin(parallel_sum(s))) }`,
        // the predicate varies with `s` while the residual `cos(parallel_sum(s))` of the `sin` derivative is invariant,
        // so the bounded rule rejects the loop instead of mistyping that stack.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let state_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh.clone(), 0).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(state_type.clone());
        let limit = builder.add_constant(Array::from_elements(state_type.clone(), &[10f32]).unwrap());
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![state, limit], None)
            .unwrap()[0];
        let condition = builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = builder.add_input(state_type.clone());
        let sum = ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()).with_mesh(mesh.clone());
        let total = builder.add_instruction(sum, Vec::new(), vec![state], None).unwrap()[0];
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![total], None).unwrap()[0];
        let varied = builder
            .add_instruction(ParallelVaryOperation::new("x".to_string()), Vec::new(), vec![sine], None)
            .unwrap()[0];
        let next_state =
            builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, varied], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_state], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |inputs: Vec<Tracer<TracingContext<Array, ArrayOperation<Array>>>>| {
                inputs[0].domain().bind(
                    ArrayOperation::While(WhileOperation::new().with_iteration_bound(3)?),
                    vec![condition, body],
                    &inputs,
                )
            },
            vec![state_type],
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh })],
        )
        .unwrap();
        assert_eq!(
            program.to_flat_program().jvp().map(|_| ()),
            Err(DifferentiationError::from(ProgramError::from(TypeError::invalid(
                "`jvp` of a bounded `while` loop requires every residual to vary over the manual axes that the \
                 predicate `bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]` varies over, but got \
                 residual storage `f32[][sharding={mesh<['x'=2:manual]>, []}]`",
            )))),
        );
    }

    #[test]
    fn test_while_differentiation_composes_with_earlier_batching() {
        // Forward mode through a batched bounded loop under the non-concretizing staged domain: batching stages one
        // bounded loop with the per-item predicate `x < 8` and the iteration bound 5, so the JVP rule takes the
        // bounded staged path over its masked normal form, storing residual stacks and staging a masked linear scan.
        // Batch items [1, 5, 9] double 3, 1, and 0 times, so the primal is [8, 10, 9] and the per-item tangent scale
        // is 2^iterations = [8, 2, 1].
        fn batched_bounded_while<V>(x: V) -> Result<V, ProgramError>
        where
            V: Value<Type = ArrayType> + Broadcast + Transpose,
            V::Domain: Context<Type = ArrayType, Value = V, Constant = Array, Operation = TestDomainOperation>,
            TestDomainOperation: BatchableOperation<V::Domain, ArrayBatchingPolicy>
                + BatchableOperation<
                    TracingContext<<V::Domain as Domain>::Constant, <V::Domain as Domain>::Operation>,
                    ArrayBatchingPolicy,
                > + From<TransposeOperation>
                + From<BroadcastOperation>,
        {
            let context = x.domain();
            let mapped = Batch::batch(
                &context,
                |item: BatchingTracer<V::Domain, ArrayBatchingPolicy>| {
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
                Array::vector(vec![1.0, 5.0, 9.0]).unwrap(),
                Array::vector(vec![1.0, 1.0, 1.0]).unwrap(),
                (),
            )
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![8.0, 10.0, 9.0]);
        assert_eq!(tangent.to_f64s(), vec![8.0, 2.0, 1.0]);

        // The plain eager domain produces the same numbers.
        let (primal, tangent) = differentiate_at(Array::vector(vec![1.0, 5.0, 9.0]).unwrap())
            .jvp(Array::vector(vec![1.0, 1.0, 1.0]).unwrap(), batched_bounded_while)
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![8.0, 10.0, 9.0]);
        assert_eq!(tangent.to_f64s(), vec![8.0, 2.0, 1.0]);

        // Reverse mode composes through the masked linear scan: the pullback transposes the scan without any `while`
        // loop, and the per-item gradients match the tangent scales.
        let (output, pullback) = StagedDispatchTestDomain
            .vjp(|input, ()| batched_bounded_while(input), Array::vector(vec![1.0, 5.0, 9.0]).unwrap(), ())
            .unwrap();
        assert_eq!(output.to_f64s(), vec![8.0, 10.0, 9.0]);
        assert_eq!(
            pullback
                .linear_program()
                .transpose_with_trailing_residuals_shared(pullback.residuals().len(), &[])
                .unwrap()
                .program()
                .to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[5], %2:f64[5, 3], %3:bool[5, 3] .
                let %4:f64[3] = scan [carry_count=1, length=5, reverse=true] %0 %1 %2 %3 [
                    body={
                        lambda %0:i64[], %1:f64[3], %2:bool[], %3:f64[3], %4:bool[3] .
                        let %5:f64[3] = condition %2 %1 %3 %4 [
                            true={
                                lambda %0:f64[3], %1:f64[3], %2:bool[3] .
                                let %3:f64[3] = zero [type=f64[3]]
                                    %4:f64[3] = select %2 %0 %3
                                    %5:f64[3] = select %2 %3 %0
                                    %6:f64[3] = mul %1 %4
                                    %7:f64[3] = add %5 %6
                                in (%7)
                            },
                            false={
                                lambda %0:f64[3], %1:f64[3], %2:bool[3] .
                                in (%0)
                            },
                        ]
                        in (%5)
                    },
                ]
                in (%4)"},
        );
        let cotangent = pullback.apply_materialized(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()).unwrap();
        assert_eq!(cotangent.to_f64s(), vec![8.0, 2.0, 1.0]);
    }

    #[test]
    fn test_while_differentiation_composes_with_later_batching() {
        // Batching a while JVP program batches the augmented loop together with its masked tangent scan, or the fused
        // loop of an unbounded `while`. `while (x < 10) { x = x * x }` over batch items `[2, 3, 200]` runs 2, 2, and 0
        // iterations, so the primal is `[16, 81, 200]` and unit tangents scale to `[4 x³, 4 x³, 1] = [32, 108, 1]`.
        let x = Array::vector(vec![2.0, 3.0, 200.0]).unwrap();
        let tangent = Array::vector(vec![1.0, 1.0, 1.0]).unwrap();
        let expected =
            Ok(vec![Array::vector(vec![16.0, 81.0, 200.0]).unwrap(), Array::vector(vec![32.0, 108.0, 1.0]).unwrap()]);

        let (operation, regions) = bounded_squaring_while_operation(10.0, 4);
        let jvp = while_program(operation, &regions, vec![ArrayType::scalar(DataType::F64)]).jvp().unwrap();
        let batched = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        let batched = batched.into_parts().0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:i64[] = zero [type=i64[]]
                    %3:i64[3] = broadcast [output_type=i64[3], output_axes=[]] %2
                    %4:f64[4] = zero [type=f64[4]]
                    %5:f64[3, 4] = broadcast [output_type=f64[3, 4], output_axes=[1]] %4
                    %6:bool[4] = zero [type=bool[4]]
                    %7:bool[3, 4] = broadcast [output_type=bool[3, 4], output_axes=[1]] %6
                    %8:f64[3], %9:i64[3], %10:f64[3, 4], %11:bool[3, 4] = while [iteration_bound=4] %0 %3 %5 %7 [
                        condition={
                            lambda %0:f64[3], %1:i64[3], %2:f64[3, 4], %3:bool[3, 4] .
                            let %4:f64[] = const 10.0
                                %5:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %4
                                %6:bool[3] = compare [direction=LessThan] %0 %5
                            in (%6)
                        },
                        body={
                            lambda %0:f64[3], %1:i64[3], %2:f64[3, 4], %3:bool[3, 4] .
                            let %4:f64[3] = mul %0 %0
                                %5:i64[] = one [type=i64[]]
                                %6:i64[3] = broadcast [output_type=i64[3], output_axes=[]] %5
                                %7:i64[3] = add %1 %6
                                %8:f64[1, 4] = slice [start_indices=[0, 0], limits=[1, 4]] %2
                                %9:f64[4] = reshape [shape=[4]] %8
                                %10:f64[3, 1] = broadcast [output_type=f64[3, 1], output_axes=[0]] %0
                                %11:f64[1, 1] = slice [start_indices=[0, 0], limits=[1, 1]] %10
                                %12:f64[1] = reshape [shape=[1]] %11
                                %13:i64[1] = slice [start_indices=[0], limits=[1]] %1
                                %14:i64[] = reshape [shape=[]] %13
                                %15:f64[4] = dynamic_update_slice %9 %12 %14
                                %16:f64[1, 4] = reshape [shape=[1, 4]] %15
                                %17:f64[1, 4] = slice [start_indices=[1, 0], limits=[2, 4]] %2
                                %18:f64[4] = reshape [shape=[4]] %17
                                %19:f64[1, 1] = slice [start_indices=[1, 0], limits=[2, 1]] %10
                                %20:f64[1] = reshape [shape=[1]] %19
                                %21:i64[1] = slice [start_indices=[1], limits=[2]] %1
                                %22:i64[] = reshape [shape=[]] %21
                                %23:f64[4] = dynamic_update_slice %18 %20 %22
                                %24:f64[1, 4] = reshape [shape=[1, 4]] %23
                                %25:f64[1, 4] = slice [start_indices=[2, 0], limits=[3, 4]] %2
                                %26:f64[4] = reshape [shape=[4]] %25
                                %27:f64[1, 1] = slice [start_indices=[2, 0], limits=[3, 1]] %10
                                %28:f64[1] = reshape [shape=[1]] %27
                                %29:i64[1] = slice [start_indices=[2], limits=[3]] %1
                                %30:i64[] = reshape [shape=[]] %29
                                %31:f64[4] = dynamic_update_slice %26 %28 %30
                                %32:f64[1, 4] = reshape [shape=[1, 4]] %31
                                %33:f64[3, 4] = concatenate [axis=0] %16 %24 %32
                                %34:bool[1, 4] = slice [start_indices=[0, 0], limits=[1, 4]] %3
                                %35:bool[4] = reshape [shape=[4]] %34
                                %36:bool[] = one [type=bool[]]
                                %37:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %36
                                %38:i64[1] = slice [start_indices=[0], limits=[1]] %1
                                %39:i64[] = reshape [shape=[]] %38
                                %40:bool[4] = dynamic_update_slice %35 %37 %39
                                %41:bool[1, 4] = reshape [shape=[1, 4]] %40
                                %42:bool[1, 4] = slice [start_indices=[1, 0], limits=[2, 4]] %3
                                %43:bool[4] = reshape [shape=[4]] %42
                                %44:i64[1] = slice [start_indices=[1], limits=[2]] %1
                                %45:i64[] = reshape [shape=[]] %44
                                %46:bool[4] = dynamic_update_slice %43 %37 %45
                                %47:bool[1, 4] = reshape [shape=[1, 4]] %46
                                %48:bool[1, 4] = slice [start_indices=[2, 0], limits=[3, 4]] %3
                                %49:bool[4] = reshape [shape=[4]] %48
                                %50:i64[1] = slice [start_indices=[2], limits=[3]] %1
                                %51:i64[] = reshape [shape=[]] %50
                                %52:bool[4] = dynamic_update_slice %49 %37 %51
                                %53:bool[1, 4] = reshape [shape=[1, 4]] %52
                                %54:bool[3, 4] = concatenate [axis=0] %41 %47 %53
                            in (%4, %7, %33, %54)
                        },
                    ]
                    %12:f64[4, 3] = transpose [permutation=[1, 0]] %10
                    %13:bool[4, 3] = transpose [permutation=[1, 0]] %11
                    %14:f64[3] = scan [carry_count=1, length=4, reverse=false] %1 %12 %13 [
                        body={
                            lambda %0:i64[], %1:f64[3], %2:f64[3], %3:bool[3] .
                            let %4:f64[3] = stop_gradient %2
                                %5:f64[3] = select %3 %2 %4
                                %6:f64[3] = stop_gradient %1
                                %7:f64[3] = select %3 %1 %6
                                %8:f64[3] = mul %5 %7
                                %9:f64[3] = mul %5 %7
                                %10:f64[3] = add %8 %9
                                %11:f64[3] = select %3 %6 %1
                                %12:f64[3] = select %3 %10 %11
                            in (%12)
                        },
                    ]
                in (%8, %14)"},
        );
        assert_eq!(batched.interpret(vec![x.clone(), tangent.clone()]), expected);

        let (operation, regions) = unbounded_squaring_while_operation(10.0);
        let jvp = while_program(operation, &regions, vec![ArrayType::scalar(DataType::F64)]).jvp().unwrap();
        let batched = jvp
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap();
        assert_eq!(batched.output_axes(), &[BatchAxis::new(0), BatchAxis::new(0)]);
        let batched = batched.into_parts().0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3], %3:f64[3] = while %0 %1 [
                    condition={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[] = const 10.0
                            %3:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %2
                            %4:bool[3] = compare [direction=LessThan] %0 %3
                        in (%4)
                    },
                    body={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = mul %0 %0
                            %3:f64[3] = mul %0 %1
                            %4:f64[3] = mul %0 %1
                            %5:f64[3] = add %3 %4
                        in (%2, %5)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(batched.interpret(vec![x, tangent]), expected);
    }

    #[test]
    fn test_while_differentiation_stages_one_fused_doubled_state_loop() {
        // JAX-parity forward mode through a *staged* unbounded while loop: the rule stages one fused doubled-state
        // loop whose trip decision reads the primal half. For `f(x) = while (x < 16) { x = x * x }` at `x = 2` the
        // loop runs twice (2 -> 4 -> 16), so locally `f(x) = x^4`: primal 16, tangent `4 x^3 = 32`.
        let (while_operation, while_regions) = unbounded_squaring_while_operation(16.0);
        let staged_regions = while_regions.clone();
        let (output, tangent) = StagedDispatchTestDomain
            .jvp(
                move |x, ()| Ok(bind_while(while_operation, staged_regions.clone(), &[x])?.remove(0)),
                Array::scalar(2.0).unwrap(),
                Array::scalar(1.0).unwrap(),
                (),
            )
            .unwrap();
        assert_eq!(output.to_f64s(), vec![16.0]);
        assert_eq!(tangent.to_f64s(), vec![32.0]);

        // The fused forward-mode program contains exactly one while loop over the doubled state `[primal, tangent]`,
        // with no unrolling and no residual stacks, and its trip count stays data-dependent: interpreting the same
        // fused program at `x = 16` never enters the loop, so the state passes through unchanged.
        let fused = while_program(while_operation, &while_regions, vec![ArrayType::scalar(DataType::F64)])
            .jvp()
            .unwrap();
        assert_eq!(
            fused.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = while %0 %1 [
                    condition={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = const 16.0
                            %3:bool[] = compare [direction=LessThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %0 %0
                            %3:f64[] = mul %0 %1
                            %4:f64[] = mul %0 %1
                            %5:f64[] = add %3 %4
                        in (%2, %5)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            fused.interpret(vec![Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap()]),
            Ok(vec![Array::scalar(16.0).unwrap(), Array::scalar(32.0).unwrap()]),
        );
        assert_eq!(
            fused.interpret(vec![Array::scalar(16.0).unwrap(), Array::scalar(1.0).unwrap()]),
            Ok(vec![Array::scalar(16.0).unwrap(), Array::scalar(1.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_differentiation_omits_tangents_of_zero_space_state_from_fused_loops() {
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
                .add_instruction(
                    CompareOperation::new(ComparisonDirection::LessThan),
                    Vec::new(),
                    vec![counter, bound],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, TestDomainOperation>::new();
            let state = builder.add_input(float_type.clone());
            let counter = builder.add_input(counter_type.clone());
            let squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
            let one = builder.add_constant(Array::from_elements(counter_type.clone(), &[1i64]).unwrap());
            let incremented =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
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
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // The fused program's boundary is compact: `[x, i, live(ẋ)] -> [x', i', live(ẋ')]` with no counter tangent.
        let fused = program.jvp().unwrap();
        assert_eq!(
            fused.to_string(),
            indoc! {"
                lambda %0:f64[], %1:i64[], %2:f64[] .
                let %3:f64[], %4:i64[], %5:f64[] = while %0 %1 %2 [
                    condition={
                        lambda %0:f64[], %1:i64[], %2:f64[] .
                        let %3:i64[] = const 3
                            %4:bool[] = compare [direction=LessThan] %1 %3
                        in (%4)
                    },
                    body={
                        lambda %0:f64[], %1:i64[], %2:f64[] .
                        let %3:i64[] = const 1
                            %4:f64[] = mul %0 %0
                            %5:f64[] = mul %0 %2
                            %6:f64[] = mul %0 %2
                            %7:f64[] = add %5 %6
                            %8:i64[] = add %1 %3
                        in (%4, %8, %7)
                    },
                ]
                in (%3, %4, %5)"},
        );
        assert_eq!(
            fused.interpret(vec![
                Array::scalar(2.0).unwrap(),
                Array::from_elements(counter_type.clone(), &[0i64]).unwrap(),
                Array::scalar(1.0).unwrap(),
            ]),
            Ok(vec![
                Array::scalar(256.0).unwrap(),
                Array::from_elements(counter_type, &[3i64]).unwrap(),
                Array::scalar(1024.0).unwrap(),
            ]),
        );
    }

    #[test]
    fn test_while_differentiation_threads_reference_state() {
        // `f(s) = while (i < 2) { state += state; i += 1 }` doubles the referenced state twice, so `f(s) = 4s` with
        // tangent `4ṡ`. The counter is an integer, so its zero differential space contributes no tangent state.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let counter_type = ArrayType::scalar(DataType::I64);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let condition = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            builder.add_input(reference_type.clone().into());
            let counter = builder.add_input(counter_type.clone().into());
            let bound = builder.add_constant(TestIrValue::Array(Array::scalar(2i64).unwrap()));
            let predicate = builder
                .add_instruction(
                    ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                    Vec::new(),
                    vec![counter, bound],
                    None,
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let reference = builder.add_input(reference_type.into());
            let counter = builder.add_input(counter_type.clone().into());
            let current =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, current], None)
                .unwrap();
            let one = builder.add_constant(TestIrValue::Array(Array::scalar(1i64).unwrap()));
            let incremented = builder
                .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, one], None)
                .unwrap()[0];
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![reference, incremented],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(scalar_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let counter = builder.add_constant(TestIrValue::Array(Array::scalar(0i64).unwrap()));
        let outputs = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new(),
                vec![condition, body],
                vec![reference, counter],
                None,
            )
            .unwrap()
            .to_vec();
        let frozen = builder
            .add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![outputs[0]], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The fused loop state is `[ref, i, ṙef]`: the reference state keeps its position with its tangent reference
        // appended, and the zero-space counter contributes no tangent state.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:i64[] = const 0
                    %3:ref<f32[]> = reference_new %0
                    %4:ref<f32[]> = reference_new %1
                    %5:ref<f32[]>, %6:i64[], %7:ref<f32[]> = while %3 %2 %4 [
                        condition={
                            lambda %0:ref<f32[]>, %1:i64[], %2:ref<f32[]> .
                            let %3:i64[] = const 2
                                %4:bool[] = compare [direction=LessThan] %1 %3
                            in (%4)
                        },
                        body={
                            lambda %0:ref<f32[]>, %1:i64[], %2:ref<f32[]> .
                            let %3:i64[] = const 1
                                %4:f32[] = reference_read %0
                                %5:f32[] = reference_read %2
                                () = reference_add_update %0 %4
                                () = reference_add_update %2 %5
                                %6:i64[] = add %1 %3
                            in (%0, %6, %2)
                        },
                    ]
                    %8:f32[] = reference_freeze %5
                    %9:f32[] = reference_freeze %7
                in (%8, %9)"},
        );

        let inputs = vec![
            TestIrValue::Array(Array::scalar(1.5f32).unwrap()),
            TestIrValue::Array(Array::scalar(1.0f32).unwrap()),
        ];
        let expected = program
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .jvp()
            .unwrap()
            .interpret(inputs.clone())
            .unwrap();
        assert_eq!(
            expected,
            vec![
                TestIrValue::Array(Array::scalar(6.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(4.0f32).unwrap())
            ]
        );
        // Direct reference execution and reference discharge must produce the same derivative.
        assert_eq!(jvp.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(
            jvp.discharge_references(0)
                .unwrap()
                .into_program_without_external_references()
                .unwrap()
                .interpret(inputs),
            Ok(expected),
        );
    }

    #[test]
    fn test_while_differentiation_preserves_inactive_reference_state() {
        // Differentiating only with respect to the value carry leaves the reference carry inactive: the loop
        // `while (i < 2) { value += read(state); i += 1 }` reads the reference without giving it a tangent, so from
        // `value = 3` with the reference holding 5 it computes 13, and the value tangent 2 passes through unchanged.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let counter_type = ArrayType::scalar(DataType::I64);
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(reference_type.clone().into());
        let counter = condition_builder.add_input(counter_type.clone().into());
        condition_builder.add_input(scalar_type.clone().into());
        let bound = condition_builder.add_constant(TestIrValue::Array(Array::scalar(2i64).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![counter, bound],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let counter = body_builder.add_input(counter_type.into());
        let value = body_builder.add_input(scalar_type.clone().into());
        let state = body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let next_value = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![value, state], None)
            .unwrap()[0];
        let one = body_builder.add_constant(TestIrValue::Array(Array::scalar(1i64).unwrap()));
        let next_counter = body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, one], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, next_counter, next_value],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(reference_type.into());
        let value = builder.add_input(scalar_type.into());
        let counter = builder.add_constant(TestIrValue::Array(Array::scalar(0i64).unwrap()));
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let outputs = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new(),
                vec![condition, body],
                vec![reference, counter, value],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![outputs[2]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let jvp = program.entry_region_ref().jvp(&[1]).unwrap();
        assert_eq!(
            jvp.interpret(vec![
                TestIrValue::Reference(ArrayReference::new(Array::scalar(5.0f32).unwrap())),
                TestIrValue::Array(Array::scalar(3.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ]),
            Ok(vec![
                TestIrValue::Array(Array::scalar(13.0f32).unwrap()),
                TestIrValue::Array(Array::scalar(2.0f32).unwrap())
            ])
        );
    }

    #[test]
    fn test_while_differentiation_after_local_reference_discharge() {
        // Discharging a local allocation turns the loop's mutated state into an ordinary carry, so the derived forward,
        // linearized, and pullback programs are pure and reference-free, and reverse mode remains available through
        // the bounded loop.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut condition_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(true).unwrap()));
        let condition = condition_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![predicate],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut body_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(ArrayIrValue::Array(Array::scalar(1.0f32).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![reference],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(ArrayIrType::Array(scalar_type));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
        let reference = builder.add_instruction(operation, vec![condition, body], vec![reference], None).unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
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
        let pullback = linearization.pullback().unwrap().into_program();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:i64[] = zero [type=i64[]]
                    %3:bool[2] = zero [type=bool[2]]
                    %4:f32[], %5:i64[], %6:bool[2] = while [iteration_bound=2] %0 %2 %3 [
                        condition={
                            lambda %0:f32[], %1:i64[], %2:bool[2] .
                            let %3:bool[] = const true
                            in (%3)
                        },
                        body={
                            lambda %0:f32[], %1:i64[], %2:bool[2] .
                            let %3:f32[] = const 1.0
                                %4:f32[] = add %0 %3
                                %5:bool[] = one [type=bool[]]
                                %6:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %5
                                %7:bool[2] = dynamic_update_slice %2 %6 %1
                                %8:i64[] = one [type=i64[]]
                                %9:i64[] = add %1 %8
                            in (%4, %9, %7)
                        },
                    ]
                    %7:f32[] = scan [carry_count=1, length=2, reverse=false] %1 %6 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:bool[] .
                            let %3:f32[] = condition %2 %1 [
                                true={
                                    lambda %0:f32[] .
                                    in (%0)
                                },
                                false={
                                    lambda %0:f32[] .
                                    in (%0)
                                },
                            ]
                            in (%3)
                        },
                    ]
                in (%4, %7)"},
        );
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:i64[] = zero [type=i64[]]
                    %2:bool[2] = zero [type=bool[2]]
                    %3:f32[], %4:i64[], %5:bool[2] = while [iteration_bound=2] %0 %1 %2 [
                        condition={
                            lambda %0:f32[], %1:i64[], %2:bool[2] .
                            let %3:bool[] = const true
                            in (%3)
                        },
                        body={
                            lambda %0:f32[], %1:i64[], %2:bool[2] .
                            let %3:f32[] = const 1.0
                                %4:f32[] = add %0 %3
                                %5:i64[] = one [type=i64[]]
                                %6:i64[] = add %1 %5
                                %7:bool[] = one [type=bool[]]
                                %8:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %7
                                %9:bool[2] = dynamic_update_slice %2 %8 %1
                            in (%4, %6, %9)
                        },
                    ]
                in (%3, %5)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[], %1:bool[2] .
                let %2:f32[] = scan [carry_count=1, length=2, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:bool[] .
                        let %3:f32[] = condition %2 %1 [
                            true={
                                lambda %0:f32[] .
                                in (%0)
                            },
                            false={
                                lambda %0:f32[] .
                                in (%0)
                            },
                        ]
                        in (%3)
                    },
                ]
                in (%2)"},
        );
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f32[], %1:bool[2] .
                let %2:f32[] = scan [carry_count=1, length=2, reverse=true] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:bool[] .
                        let %3:f32[] = condition %2 %1 [
                            true={
                                lambda %0:f32[] .
                                in (%0)
                            },
                            false={
                                lambda %0:f32[] .
                                in (%0)
                            },
                        ]
                        in (%3)
                    },
                ]
                in (%2)"},
        );

        // Two iterations accumulate the constant `1.0` into the state, so `f(x) = x + 2` and the tangent passes
        // through unscaled in both directions.
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::scalar(3.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap()),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(5.0f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(2.0f32).unwrap())
            ]),
        );
        let primal_outputs =
            linearization.primal().interpret(vec![ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())]).unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::scalar(5.0f32).unwrap()));
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::scalar(4.0f32).unwrap())];
        pullback_inputs.extend_from_slice(&primal_outputs[1..]);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![ArrayIrValue::Array(Array::scalar(4.0f32).unwrap())]));
    }

    #[test]
    fn test_while_differentiation_fuses_unbounded_loops_with_batched_predicates() {
        // The eager rule cannot concretize the batched predicate `x < [16, 4, 2]` of an unbounded loop, so it falls
        // back to the fused doubled-state loop, whose doubled state still satisfies the predicate-prefix contract.
        // Squaring `x = [2, 2, 2]` runs 2, 1, and 0 iterations, so the value is `[16, 4, 2]` and the per-item
        // derivatives are `[4 x³, 2 x, 1] = [32, 4, 1]`.
        let state_type = ArrayType::new_static(DataType::F64, [3]);
        let mut condition_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = condition_builder.add_input(state_type.clone());
        let limits = condition_builder.add_constant(Array::vector(vec![16.0, 4.0, 2.0]).unwrap());
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![state, limits],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = body_builder.add_input(state_type.clone());
        let squared =
            body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = body_builder.build(vec![squared], vec![Placeholder], vec![Placeholder]).unwrap();
        let regions = vec![condition, body];
        let x = Array::vector(vec![2.0, 2.0, 2.0]).unwrap();
        let tangent = Array::vector(vec![1.0, 1.0, 1.0]).unwrap();
        let expected = (Array::vector(vec![16.0, 4.0, 2.0]).unwrap(), Array::vector(vec![32.0, 4.0, 1.0]).unwrap());
        assert_eq!(
            differentiate_at(x.clone()).jvp(tangent.clone(), |x| {
                bind_while(WhileOperation::new(), regions.clone(), &[x]).map(|mut outputs| outputs.remove(0))
            }),
            Ok(expected.clone()),
        );

        // The staged rule stages the same fused loop, whose interpretation applies the masked semantics.
        let fused = while_program(WhileOperation::new(), &regions, vec![state_type]).jvp().unwrap();
        assert_eq!(
            fused.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3], %3:f64[3] = while %0 %1 [
                    condition={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = const [16.0, 4.0, 2.0]
                            %3:bool[3] = compare [direction=LessThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = mul %0 %0
                            %3:f64[3] = mul %0 %1
                            %4:f64[3] = mul %0 %1
                            %5:f64[3] = add %3 %4
                        in (%2, %5)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(fused.interpret(vec![x, tangent]), Ok(vec![expected.0, expected.1]));
    }

    #[test]
    fn test_while_differentiation_recovers_the_primal_loop_of_staged_linearizations() {
        // Linearization of a staged unbounded loop composes the fused doubled-state forward-mode rule with the
        // `while` closed-knownness split: the fused loop's primal half is closed under the body and the condition
        // reads only it, so the split rebinds the primal loop on the known (primal) side while the tangent program
        // keeps the fused loop whole, recomputing primal state internally. For `while (x < 16) { x = x * x }` at
        // `x = 2`, the primal is 16 and the pushforward scales tangents by 32.
        let (while_operation, while_regions) = unbounded_squaring_while_operation(16.0);
        let (output, pushforward) = StagedDispatchTestDomain
            .linearize(
                move |x, ()| Ok(bind_while(while_operation, while_regions.clone(), &[x])?.remove(0)),
                Array::scalar(2.0).unwrap(),
                (),
            )
            .unwrap();
        assert_eq!(output.to_f64s(), vec![16.0]);
        assert_eq!(pushforward.apply(Array::scalar(1.0).unwrap()).map(|tangent| tangent.to_f64s()), Ok(vec![32.0]));
        assert_eq!(pushforward.apply(Array::scalar(2.0).unwrap()).map(|tangent| tangent.to_f64s()), Ok(vec![64.0]));
    }

    #[test]
    fn test_while_differentiation_rejects_staged_linearization_of_effectful_unbounded_loops() {
        // Separate linearization must keep every primal output of the fused loop known, but the effects gate of the
        // partial-evaluation rule residualizes an effectful loop unchanged, so the `print` loop cannot be linearized.
        let (operation, regions) = effectful_doubling_while_operation();
        let program = while_program(operation, &regions, vec![ArrayType::scalar(DataType::F64)]);
        assert_eq!(
            program.linearize().map(|_| ()),
            Err(DifferentiationError::from(ProgramError::MalformedProgram(
                "required output 0 depends on deferred work in a repeated residual computation".to_string(),
            ))),
        );
    }

    #[test]
    fn test_while_differentiation_falls_back_to_declared_regions_when_specialization_drops_a_refinement() {
        type TestContext = TracingContext<TestIrValue, TestIrOperation>;
        type TestTracer = Tracer<TestContext>;

        // Forward-mode differentiation specializes attached regions before applying the `while` rule. A body that
        // replaces its sharded carry with an unsharded constant cannot be specialized usefully, so differentiation uses
        // the declared regions, exactly like staging does.
        let state_type = ArrayType::scalar(DataType::F32);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_type = state_type.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(state_type.clone().into());
        let predicate = condition_builder.add_constant(array(Array::scalar(false).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(state_type.clone().into());
        let replacement = body_builder.add_constant(array(Array::scalar(1f32).unwrap()));
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![replacement], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let (output_types, program) = TestContext::trace(
            |input: TestTracer| {
                let context = input.context().clone();
                let (primal, tangent) = context.jvp(
                    |state: DifferentiationTracer<TestContext>, ()| {
                        let context = state.context().clone();
                        Ok(context
                            .bind(WhileOperation::new(), vec![condition.clone(), body.clone()], &[state])?
                            .remove(0))
                    },
                    input.clone(),
                    input,
                    (),
                )?;
                Ok(vec![primal, tangent])
            },
            ArrayIrType::Array(sharded_type),
        )
        .unwrap();
        assert_eq!(output_types, vec![ArrayIrType::Array(state_type); 2]);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:auto]>, []}] .
                let %1:f32[], %2:f32[] = while %0 %0 [
                    condition={
                        lambda %0:f32[], %1:f32[] .
                        let %2:bool[] = const false
                        in (%2)
                    },
                    body={
                        lambda %0:f32[], %1:f32[] .
                        let %2:f32[] = const 1.0
                            %3:f32[] = zero [type=f32[]]
                        in (%2, %3)
                    },
                ]
                in (%1, %2)"},
        );
    }

    #[test]
    fn test_while_differentiation_refines_outputs_of_unspecialized_regions() {
        // Differentiating a loop whose refined outputs come from unspecialized regions keeps the primal, tangent, and
        // cotangent types refined. The loop computes `[counter - 2, 4 * vector]` for a counter of 2.
        let static_vector_type = ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]));
        let counter = array(Array::scalar(2.0).unwrap());
        let vector = array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());

        let jvp = refined_vector_while_program(None).jvp().unwrap();
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        assert_eq!(
            jvp.output_types(),
            vec![scalar_type.clone(), static_vector_type.clone(), scalar_type.clone(), static_vector_type.clone()],
        );
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:f64[], %3:f64[3] .
                let %4:f64[], %5:f64[3], %6:f64[], %7:f64[3] = while %0 %1 %2 %3 [
                    condition={
                        lambda %0:f64[], %1:f64[3], %2:f64[], %3:f64[3] .
                        let %4:f64[] = const 0.0
                            %5:bool[] = compare [direction=GreaterThan] %0 %4
                        in (%5)
                    },
                    body={
                        lambda %0:f64[], %1:f64[3], %2:f64[], %3:f64[3] .
                        let %4:f64[] = const 1.0
                            %5:f64[] = sub %0 %4
                            %6:f64[3] = add %1 %1
                            %7:f64[3] = add %3 %3
                        in (%5, %6, %2, %7)
                    },
                ]
                in (%4, %5, %6, %7)"},
        );
        assert_eq!(
            jvp.interpret(vec![
                counter.clone(),
                vector.clone(),
                array(Array::scalar(0.0).unwrap()),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
            ]),
            Ok(vec![
                array(Array::scalar(0.0).unwrap()),
                array(Array::vector(vec![4.0, 8.0, 12.0]).unwrap()),
                array(Array::scalar(0.0).unwrap()),
                array(Array::vector(vec![4.0, 4.0, 4.0]).unwrap()),
            ]),
        );

        let linearization = refined_vector_while_program(Some(4)).linearize().unwrap();
        assert_eq!(linearization.primal().output_types()[1], static_vector_type);
        let mut primal_outputs = linearization.primal().interpret(vec![counter, vector]).unwrap();
        assert_eq!(primal_outputs[1], array(Array::vector(vec![4.0, 8.0, 12.0]).unwrap()));
        let residuals = primal_outputs.split_off(2);
        let pullback = linearization.pullback().unwrap().into_program();
        assert_eq!(pullback.output_types(), vec![scalar_type, static_vector_type]);
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:bool[4] .
                let %3:f64[], %4:f64[3] = scan [carry_count=2, length=4, reverse=true] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:f64[3], %3:bool[] .
                        let %4:f64[], %5:f64[3] = condition %3 %1 %2 [
                            true={
                                lambda %0:f64[], %1:f64[3] .
                                let %2:f64[3] = add %1 %1
                                in (%0, %2)
                            },
                            false={
                                lambda %0:f64[], %1:f64[3] .
                                in (%0, %1)
                            },
                        ]
                        in (%4, %5)
                    },
                ]
                in (%3, %4)"},
        );
        let mut pullback_inputs =
            vec![array(Array::scalar(1.0).unwrap()), array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            pullback.interpret(pullback_inputs),
            Ok(vec![array(Array::scalar(1.0).unwrap()), array(Array::vector(vec![4.0, 4.0, 4.0]).unwrap())]),
        );
    }

    #[test]
    fn test_while_differentiation_omits_the_dimension_state_tangent() {
        // The forwarded dimension carry has an empty tangent space, so it has neither a tangent input nor a tangent
        // output. From `x = 1`, the loop doubles `x` three times, so its value is 8 and its derivative is 8.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let state = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F64)));
        let regions = doubling_while_regions(extent_type.clone());
        let regions = regions.iter().map(|region| builder.import_region(region.entry_region_ref())).collect();
        let outputs = builder
            .add_instruction(
                TestIrOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![extent, state],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[], %2:f64[] .
                let %3:i64[] = zero [type=i64[]]
                    %4:bool[4] = zero [type=bool[4]]
                    %5:dimension<extent ∈ [1, 8)>, %6:f64[], %7:i64[], %8:bool[4] = while \
                        [iteration_bound=4] %0 %1 %3 %4 [
                        condition={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[], %2:i64[], %3:bool[4] .
                            let %4:f64[] = const 8.0
                                %5:bool[] = compare [direction=LessThan] %1 %4
                            in (%5)
                        },
                        body={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[], %2:i64[], %3:bool[4] .
                            let %4:f64[] = add %1 %1
                                %5:bool[] = one [type=bool[]]
                                %6:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %5
                                %7:bool[4] = dynamic_update_slice %3 %6 %2
                                %8:i64[] = one [type=i64[]]
                                %9:i64[] = add %2 %8
                            in (%0, %4, %9, %7)
                        },
                    ]
                    %9:f64[] = scan [carry_count=1, length=4, reverse=false] %2 %8 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:bool[] .
                            let %3:f64[] = condition %2 %1 [
                                true={
                                    lambda %0:f64[] .
                                    let %1:f64[] = add %0 %0
                                    in (%1)
                                },
                                false={
                                    lambda %0:f64[] .
                                    in (%0)
                                },
                            ]
                            in (%3)
                        },
                    ]
                in (%5, %6, %9)"},
        );
        assert_eq!(
            jvp.interpret(vec![
                dimension(&extent_type, 4),
                array(Array::scalar(1.0).unwrap()),
                array(Array::scalar(3.0).unwrap()),
            ]),
            Ok(vec![
                dimension(&extent_type, 4),
                array(Array::scalar(8.0).unwrap()),
                array(Array::scalar(24.0).unwrap()),
            ]),
        );

        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![dimension(&extent_type, 4), array(Array::scalar(1.0).unwrap())])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        assert_eq!(primal_outputs, vec![dimension(&extent_type, 4), array(Array::scalar(8.0).unwrap())]);
        let mut pullback_inputs = vec![array(Array::scalar(1.0).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![array(Array::scalar(8.0).unwrap())]),
        );
    }

    #[test]
    fn test_while_differentiation_pulls_back_array_ir_loops_with_batched_predicates() {
        // Reverse mode runs through a bounded array IR loop with a batched predicate:
        // `while (x < [2, 4, 8], iteration_bound = 4) { x = x + x }` from `[1, 1, 1]` doubles item `i` exactly `i + 1`
        // times before its own predicate turns false, so the primal is `[2, 4, 8]` and the pullback scales a unit
        // cotangent by the matching per-item factors `[2, 4, 8]`.
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));

        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = condition_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let limits = condition_builder.add_constant(array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![state, limits],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = body_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let doubled = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![state, state],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = builder.add_input(ArrayIrType::Array(vector_type));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestIrOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![state],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap();

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = const [2.0, 4.0, 8.0]
                    %2:bool[3] = compare [direction=LessThan] %0 %1
                    %3:i64[] = zero [type=i64[]]
                    %4:bool[4, 3] = zero [type=bool[4, 3]]
                    %5:bool[4] = zero [type=bool[4]]
                    %6:f64[3], %7:bool[3], %8:i64[], %9:bool[4, 3], %10:bool[4] = while \
                        [iteration_bound=4] %0 %2 %3 %4 %5 [
                        condition={
                            lambda %0:f64[3], %1:bool[3], %2:i64[], %3:bool[4, 3], %4:bool[4] .
                            let %5:bool[] = reduce [kind=any, axes=[0]] %1
                            in (%5)
                        },
                        body={
                            lambda %0:f64[3], %1:bool[3], %2:i64[], %3:bool[4, 3], %4:bool[4] .
                            let %5:f64[3] = add %0 %0
                                %6:f64[3] = select %1 %5 %0
                                %7:f64[3] = const [2.0, 4.0, 8.0]
                                %8:bool[3] = compare [direction=LessThan] %6 %7
                                %9:i64[] = one [type=i64[]]
                                %10:i64[] = add %2 %9
                                %11:bool[1, 3] = broadcast [output_type=bool[1, 3], output_axes=[1]] %1
                                %12:i64[] = zero [type=i64[]]
                                %13:bool[4, 3] = dynamic_update_slice %3 %11 %2 %12
                                %14:bool[] = one [type=bool[]]
                                %15:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %14
                                %16:bool[4] = dynamic_update_slice %4 %15 %2
                            in (%6, %8, %10, %13, %16)
                        },
                    ]
                in (%6, %10, %9)"},
        );
        assert_eq!(
            linearization.pullback().unwrap().program().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[4], %2:bool[4, 3] .
                let %3:f64[3] = scan [carry_count=1, length=4, reverse=true] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[3], %2:bool[], %3:bool[3] .
                        let %4:f64[3] = condition %2 %1 %3 [
                            true={
                                lambda %0:f64[3], %1:bool[3] .
                                let %2:f64[3] = zero [type=f64[3]]
                                    %3:f64[3] = select %1 %0 %2
                                    %4:f64[3] = select %1 %2 %0
                                    %5:f64[3] = add %4 %3
                                    %6:f64[3] = add %5 %3
                                in (%6)
                            },
                            false={
                                lambda %0:f64[3], %1:bool[3] .
                                in (%0)
                            },
                        ]
                        in (%4)
                    },
                ]
                in (%3)"},
        );
        let mut primal_outputs =
            linearization.primal().interpret(vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())]).unwrap();
        assert_eq!(primal_outputs[0], array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap()));
        let residuals = primal_outputs.split_off(1);
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap())]),
        );
    }

    #[test]
    fn test_while_differentiation_supports_batched_predicates_with_dimension_state() {
        // Forward mode, linearization, and the pullback support a batched predicate next to a loop-invariant dimension
        // carry. Starting from `[1, 1, 1]`, item `i` doubles `i + 1` times before its own predicate turns false, so the
        // primal outputs are `[2, 4, 8]` and the per-item tangent scale factors are the matching `[2, 4, 8]`. The
        // dimension carry has an empty tangent space, so it contributes neither a tangent input nor a tangent output.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));

        // Condition: a per-item predicate `state < [2, 4, 8]` that ignores the loop-invariant dimension carry.
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let state = condition_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let limits = condition_builder.add_constant(array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![state, limits],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Body: the dimension carry is forwarded unchanged and the array carry doubles.
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let extent = body_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let state = body_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let doubled = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![state, state],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![extent, doubled],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let state = builder.add_input(ArrayIrType::Array(vector_type));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestIrOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![extent, state],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[3], %2:f64[3] .
                let %3:f64[3] = const [2.0, 4.0, 8.0]
                    %4:bool[3] = compare [direction=LessThan] %1 %3
                    %5:i64[] = zero [type=i64[]]
                    %6:bool[4, 3] = zero [type=bool[4, 3]]
                    %7:bool[4] = zero [type=bool[4]]
                    %8:dimension<extent ∈ [1, 8)>, %9:f64[3], %10:bool[3], %11:i64[], \
                        %12:bool[4, 3], %13:bool[4] = while [iteration_bound=4] %0 %1 %4 %5 %6 %7 [
                        condition={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[3], %2:bool[3], %3:i64[], \
                                %4:bool[4, 3], %5:bool[4] .
                            let %6:bool[] = reduce [kind=any, axes=[0]] %2
                            in (%6)
                        },
                        body={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[3], %2:bool[3], %3:i64[], \
                                %4:bool[4, 3], %5:bool[4] .
                            let %6:f64[3] = const [2.0, 4.0, 8.0]
                                %7:f64[3] = add %1 %1
                                %8:f64[3] = select %2 %7 %1
                                %9:bool[3] = compare [direction=LessThan] %8 %6
                                %10:i64[] = zero [type=i64[]]
                                %11:bool[1, 3] = broadcast [output_type=bool[1, 3], output_axes=[1]] %2
                                %12:bool[4, 3] = dynamic_update_slice %4 %11 %3 %10
                                %13:bool[] = one [type=bool[]]
                                %14:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %13
                                %15:bool[4] = dynamic_update_slice %5 %14 %3
                                %16:i64[] = one [type=i64[]]
                                %17:i64[] = add %3 %16
                            in (%0, %8, %9, %17, %12, %15)
                        },
                    ]
                    %14:f64[3] = scan [carry_count=1, length=4, reverse=false] %2 %12 %13 [
                        body={
                            lambda %0:i64[], %1:f64[3], %2:bool[3], %3:bool[] .
                            let %4:f64[3] = condition %3 %1 %2 [
                                true={
                                    lambda %0:f64[3], %1:bool[3] .
                                    let %2:f64[3] = add %0 %0
                                        %3:f64[3] = select %1 %2 %0
                                    in (%3)
                                },
                                false={
                                    lambda %0:f64[3], %1:bool[3] .
                                    in (%0)
                                },
                            ]
                            in (%4)
                        },
                    ]
                in (%8, %9, %14)"},
        );
        let outputs = jvp
            .interpret(vec![
                dimension(&extent_type, 4),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
            ])
            .unwrap();
        assert_eq!(outputs[0], dimension(&extent_type, 4));
        assert_eq!(outputs[1], array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap()));
        assert_eq!(outputs[2], array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap()));

        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![dimension(&extent_type, 4), array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())])
            .unwrap();
        assert_eq!(primal_outputs[0], dimension(&extent_type, 4));
        assert_eq!(primal_outputs[1], array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap()));
        let residuals = primal_outputs.split_off(2);
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap())]),
        );
    }

    #[test]
    fn test_while_differentiation_threads_invariant_dimension_residuals_as_scan_carries() {
        // A body that reshapes its `f64[extent]` carry by the loop-invariant `extent` carry leaves that extent as a
        // residual of every iteration. The residual scan threads the invariant extent as a carry instead of storing it
        // through scalar gateways, and the known side of its split only forwards the stacked mask that the residual
        // scan consumes, so the primal program needs no known scan.
        let extent_variable = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let extent_type = DimensionType::from(extent_variable.clone());
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent_variable)]));

        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        condition_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let counter = condition_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let limit = condition_builder.add_constant(array(Array::scalar(2i64).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let extent = body_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let vector = body_builder.add_input(ArrayIrType::Array(vector_type.clone()));
        let counter = body_builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let reshaped = body_builder
            .add_instruction(
                TestIrOperation::from(DynamicReshapeOperation::new()),
                Vec::new(),
                vec![vector, extent],
                None,
            )
            .unwrap()[0];
        let one = body_builder.add_constant(array(Array::scalar(1i64).unwrap()));
        let next_counter = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![counter, one],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![extent, reshaped, next_counter],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let vector = builder.add_input(ArrayIrType::Array(vector_type));
        let counter = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestIrOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![extent, vector, counter],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent], %2:i64[] .
                let %3:i64[] = zero [type=i64[]]
                    %4:bool[4] = zero [type=bool[4]]
                    %5:dimension<extent ∈ [1, 8)>, %6:f64[extent], %7:i64[], %8:i64[], %9:bool[4] = while \
                        [iteration_bound=4] %0 %1 %2 %3 %4 [
                        condition={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent], %2:i64[], %3:i64[], %4:bool[4] .
                            let %5:i64[] = const 2
                                %6:bool[] = compare [direction=LessThan] %2 %5
                            in (%6)
                        },
                        body={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent], %2:i64[], %3:i64[], %4:bool[4] .
                            let %5:f64[extent] = reshape [requires_runtime_assertion=false] %1 %0
                                %6:i64[] = const 1
                                %7:i64[] = add %2 %6
                                %8:i64[] = one [type=i64[]]
                                %9:i64[] = add %3 %8
                                %10:bool[] = one [type=bool[]]
                                %11:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %10
                                %12:bool[4] = dynamic_update_slice %4 %11 %3
                            in (%0, %5, %7, %9, %12)
                        },
                    ]
                in (%5, %6, %7, %0, %9)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[extent], %1:dimension<extent ∈ [1, 8)>, %2:bool[4] .
                let %3:dimension<extent ∈ [1, 8)>, %4:f64[extent] = scan [carry_count=2, length=4, reverse=false] %1 \
                    %0 %2 [
                    body={
                        lambda %0:i64[], %1:dimension<extent ∈ [1, 8)>, %2:f64[extent], %3:bool[] .
                        let %4:f64[extent] = condition %3 %2 %1 [
                            true={
                                lambda %0:f64[extent], %1:dimension<extent ∈ [1, 8)> .
                                let %2:f64[extent] = linear_call [residual_count=1] %1 %0 [
                                    forward={
                                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent] .
                                        let %2:f64[extent] = reshape [requires_runtime_assertion=false] %1 %0
                                        in (%2)
                                    },
                                    transpose={
                                        lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[extent] .
                                        let %2:f64[extent] = reshape [requires_runtime_assertion=false] %1 %0
                                        in (%2)
                                    },
                                ]
                                in (%2)
                            },
                            false={
                                lambda %0:f64[extent], %1:dimension<extent ∈ [1, 8)> .
                                in (%0)
                            },
                        ]
                        in (%1, %4)
                    },
                ]
                in (%4)"},
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![
                dimension(&extent_type, 3),
                array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                array(Array::scalar(0i64).unwrap()),
            ])
            .unwrap();
        let residuals = primal_outputs.split_off(3);
        assert_eq!(
            primal_outputs,
            vec![
                dimension(&extent_type, 3),
                array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap()),
                array(Array::scalar(2i64).unwrap()),
            ],
        );
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())]),
        );
    }

    #[test]
    fn test_while_differentiation_stacks_varying_dimension_residuals_through_scalar_gateways() {
        // The body broadcasts its state to an extent computed from the counter and sums it back, so the loop-varying
        // extent is stored per iteration through a scalar gateway on the primal side and restored on the tangent side.
        // From `x = 2` and counter 1, the iterations with extents 1 and 2 give `x = 4` and the derivative 2.
        let iteration_variable = DimensionVariable::new("iteration", DimensionBounds::positive(Some(4)).unwrap());
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let scalar_u64 = ArrayType::scalar(DataType::U64);

        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(ArrayIrType::Array(scalar_f64.clone()));
        let counter = condition_builder.add_input(ArrayIrType::Array(scalar_u64.clone()));
        let limit = condition_builder.add_constant(array(Array::scalar(3u64).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(CompareOperation::new(ComparisonDirection::LessThan))),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = body_builder.add_input(ArrayIrType::Array(scalar_f64.clone()));
        let counter = body_builder.add_input(ArrayIrType::Array(scalar_u64.clone()));
        let iteration = body_builder
            .add_instruction(
                TestIrOperation::from(DimensionFromScalarOperation::new(iteration_variable.into())),
                Vec::new(),
                vec![counter],
                None,
            )
            .unwrap()[0];
        let repeated = body_builder
            .add_instruction(
                TestIrOperation::from(DynamicBroadcastOperation::new(Vec::new())),
                Vec::new(),
                vec![state, iteration],
                None,
            )
            .unwrap()[0];
        let next_state = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum))),
                Vec::new(),
                vec![repeated],
                None,
            )
            .unwrap()[0];
        let one = body_builder.add_constant(array(Array::scalar(1u64).unwrap()));
        let next_counter = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![counter, one],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![next_state, next_counter],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = builder.add_input(ArrayIrType::Array(scalar_f64));
        let counter = builder.add_input(ArrayIrType::Array(scalar_u64));
        let regions =
            vec![builder.import_region(condition.entry_region_ref()), builder.import_region(body.entry_region_ref())];
        let outputs = builder
            .add_instruction(
                TestIrOperation::While(WhileOperation::new().with_iteration_bound(4).unwrap()),
                regions,
                vec![state, counter],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[], %1:u64[] .
                let %2:i64[] = zero [type=i64[]]
                    %3:i64[4] = zero [type=i64[4]]
                    %4:i64[4] = zero [type=i64[4]]
                    %5:bool[4] = zero [type=bool[4]]
                    %6:f64[], %7:u64[], %8:i64[], %9:i64[4], %10:i64[4], %11:bool[4] = while [iteration_bound=4] %0 %1 \
                        %2 %3 %4 %5 [
                        condition={
                            lambda %0:f64[], %1:u64[], %2:i64[], %3:i64[4], %4:i64[4], %5:bool[4] .
                            let %6:u64[] = const 3
                                %7:bool[] = compare [direction=LessThan] %1 %6
                            in (%7)
                        },
                        body={
                            lambda %0:f64[], %1:u64[], %2:i64[], %3:i64[4], %4:i64[4], %5:bool[4] .
                            let %6:dimension<iteration ∈ [1, 4)> = dimension_from_scalar [bounds=[1, 4)] %1
                                %7:f64[iteration] = broadcast [output_axes=[]] %0 %6
                                %8:f64[] = reduce [kind=sum, axes=[0]] %7
                                %9:u64[] = const 1
                                %10:u64[] = add %1 %9
                                %11:i64[] = one [type=i64[]]
                                %12:i64[] = add %2 %11
                                %13:i64[] = dimension_to_scalar %6
                                %14:i64[1] = broadcast [output_type=i64[1], output_axes=[]] %13
                                %15:i64[4] = dynamic_update_slice %3 %14 %2
                                %16:dimension<iteration ∈ [1, 4)> = dimension_size [axis=0] %7
                                %17:i64[] = dimension_to_scalar %16
                                %18:i64[1] = broadcast [output_type=i64[1], output_axes=[]] %17
                                %19:i64[4] = dynamic_update_slice %4 %18 %2
                                %20:bool[] = one [type=bool[]]
                                %21:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %20
                                %22:bool[4] = dynamic_update_slice %5 %21 %2
                            in (%8, %10, %12, %15, %19, %22)
                        },
                    ]
                in (%6, %7, %11, %9, %10)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[4], %2:i64[4], %3:i64[4] .
                let %4:f64[] = scan [carry_count=1, length=4, reverse=false] %0 %1 %2 %3 [
                    body={
                        lambda %0:i64[], %1:f64[], %2:bool[], %3:i64[], %4:i64[] .
                        let %5:f64[] = condition %2 %1 %3 %4 [
                            true={
                                lambda %0:f64[], %1:i64[], %2:i64[] .
                                let %3:dimension<iteration ∈ [1, 4)> = dimension_from_scalar [bounds=[1, 4)] %1
                                    %4:f64[iteration] = broadcast [output_axes=[]] %0 %3
                                    %5:f64[] = linear_call [residual_count=1] %3 %4 [
                                        forward={
                                            lambda %0:dimension<iteration ∈ [1, 4)>, %1:f64[iteration] .
                                            let %2:f64[] = reduce [kind=sum, axes=[0]] %1
                                            in (%2)
                                        },
                                        transpose={
                                            lambda %0:dimension<iteration ∈ [1, 4)>, %1:f64[] .
                                            let %2:f64[iteration] = broadcast [output_axes=[]] %1 %0
                                            in (%2)
                                        },
                                    ]
                                in (%5)
                            },
                            false={
                                lambda %0:f64[], %1:i64[], %2:i64[] .
                                in (%0)
                            },
                        ]
                        in (%5)
                    },
                ]
                in (%4)"},
        );
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![array(Array::scalar(2.0).unwrap()), array(Array::scalar(1u64).unwrap())])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        assert_eq!(primal_outputs, vec![array(Array::scalar(4.0).unwrap()), array(Array::scalar(3u64).unwrap())]);
        let mut pullback_inputs = vec![array(Array::scalar(1.0).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![array(Array::scalar(2.0).unwrap())])
        );
    }

    #[test]
    fn test_while_differentiation_lets_batched_predicate_items_rejoin() {
        // Item 0 of the batched predicate `[x₀ < 1 ∨ 2 ≤ x₁ < 3, x₁ < 3]` reads item 1, so it rejoins the loop once
        // item 1 reaches the window `[2, 3)`. The body maps each item to `1.5 x + 1` under the bound 5. From `[0, 0]`
        // the predicate is `[true, true]`, then `[false, true]` at `[1, 1]`, then `[true, true]` at `[1, 2.5]`, where
        // item 0 rejoins, and finally `[false, false]` at `[2.5, 4.75]`. Item 0 takes two updates and item 1 three,
        // so the per-item derivatives are `[1.5², 1.5³] = [2.25, 3.375]`. A masked form that kept finished items
        // finished would instead stop at `[1, 4.75]` with derivatives `[1.5, 3.375]`.
        let state_type = ArrayType::new_static(DataType::F64, [2]);
        let mut condition_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = condition_builder.add_input(state_type.clone());
        let reversed = condition_builder
            .add_instruction(ReverseOperation::new([0]), Vec::new(), vec![state], None)
            .unwrap()[0];
        let own_limits = condition_builder.add_constant(Array::vector(vec![1.0, 3.0]).unwrap());
        let own = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![state, own_limits],
                None,
            )
            .unwrap()[0];
        // The lower limit of item 1 is never reached within the bound, so only item 0 reads the other item.
        let lower_limits = condition_builder.add_constant(Array::vector(vec![2.0, 10.0]).unwrap());
        let above = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThanOrEqual),
                Vec::new(),
                vec![reversed, lower_limits],
                None,
            )
            .unwrap()[0];
        let upper_limits = condition_builder.add_constant(Array::vector(vec![3.0, 3.0]).unwrap());
        let below = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![reversed, upper_limits],
                None,
            )
            .unwrap()[0];
        let window = condition_builder
            .add_instruction(AndOperation::<ArrayType>::new(), Vec::new(), vec![above, below], None)
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(OrOperation::<ArrayType>::new(), Vec::new(), vec![own, window], None)
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let state = body_builder.add_input(state_type.clone());
        let scale = body_builder.add_constant(Array::vector(vec![1.5, 1.5]).unwrap());
        let scaled =
            body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, scale], None).unwrap()[0];
        let offset = body_builder.add_constant(Array::vector(vec![1.0, 1.0]).unwrap());
        let next =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, offset], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = WhileOperation::new().with_iteration_bound(5).unwrap();
        let regions = vec![condition, body];
        let program = while_program(operation, &regions, vec![state_type]);
        let input = Array::vector(vec![0.0, 0.0]).unwrap();
        let expected = Array::vector(vec![2.5, 4.75]).unwrap();
        assert_eq!(program.interpret(vec![input.clone()]), Ok(vec![expected.clone()]));

        // The masked normal form recomputes the next mask as the predicate of the next state instead of narrowing the
        // current mask, so a finished item rejoins whenever its predicate turns true again.
        let (masked_condition, masked_body) = masked_while_programs(&regions[0], &regions[1]).unwrap();
        assert_eq!(
            masked_condition.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:bool[2] .
                let %2:bool[] = reduce [kind=any, axes=[0]] %1
                in (%2)"},
        );
        assert_eq!(
            masked_body.to_string(),
            indoc! {"
                lambda %0:f64[2], %1:bool[2] .
                let %2:f64[2] = const [1.5, 1.5]
                    %3:f64[2] = const [1.0, 1.0]
                    %4:f64[2] = mul %0 %2
                    %5:f64[2] = add %4 %3
                    %6:f64[2] = select %1 %5 %0
                    %7:f64[2] = const [1.0, 3.0]
                    %8:f64[2] = const [2.0, 10.0]
                    %9:f64[2] = const [3.0, 3.0]
                    %10:f64[2] = reverse [axes=[0]] %6
                    %11:bool[2] = compare [direction=LessThan] %6 %7
                    %12:bool[2] = compare [direction=GreaterThanOrEqual] %10 %8
                    %13:bool[2] = compare [direction=LessThan] %10 %9
                    %14:bool[2] = and %12 %13
                    %15:bool[2] = or %11 %14
                in (%6, %15)"},
        );

        // The staged bounded forward mode differentiates that masked form.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.interpret(vec![input.clone(), Array::vector(vec![1.0, 1.0]).unwrap()]),
            Ok(vec![expected.clone(), Array::vector(vec![2.25, 3.375]).unwrap()]),
        );

        // Reverse mode transposes the masked tangent scan of the same masked form.
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![input]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![expected]);
        let mut pullback_inputs = vec![Array::vector(vec![1.0, 2.0]).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![Array::vector(vec![2.25, 6.75]).unwrap()]),
        );
    }

    #[test]
    fn test_while_differentiation_linearizes_unbounded_loops_with_batched_predicates() {
        // An unbounded loop with the batched predicate `x < [10, 3, 1]` squares `x = [2, 2, 2]` 2, 1, and 0 times, so
        // its value is `[16, 4, 2]` and its per-item derivatives are `[4 x³, 2 x, 1] = [32, 4, 1]`.
        let state_type = ArrayType::new_static(DataType::F64, [3]);
        let mut condition_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = condition_builder.add_input(state_type.clone());
        let limits = condition_builder.add_constant(Array::vector(vec![10.0, 3.0, 1.0]).unwrap());
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![state, limits],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestDomainOperation>::new();
        let state = body_builder.add_input(state_type.clone());
        let squared =
            body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = body_builder.build(vec![squared], vec![Placeholder], vec![Placeholder]).unwrap();
        let program = while_program(WhileOperation::new(), &[condition, body], vec![state_type]);
        let linearization = program.linearize().unwrap();

        // The closed-knownness split binds the primal loop on the known side and forwards its initial state as the
        // residual, from which the tangent program recomputes the primal state inside the fused doubled-state loop.
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = while %0 [
                    condition={
                        lambda %0:f64[3] .
                        let %1:f64[3] = const [10.0, 3.0, 1.0]
                            %2:bool[3] = compare [direction=LessThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64[3] .
                        let %1:f64[3] = mul %0 %0
                        in (%1)
                    },
                ]
                in (%1, %0)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3], %3:f64[3] = while %1 %0 [
                    condition={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = const [10.0, 3.0, 1.0]
                            %3:bool[3] = compare [direction=LessThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:f64[3], %1:f64[3] .
                        let %2:f64[3] = mul %0 %0
                            %3:f64[3] = mul %0 %1
                            %4:f64[3] = mul %0 %1
                            %5:f64[3] = add %3 %4
                        in (%2, %5)
                    },
                ]
                in (%3)"},
        );
        let x = Array::vector(vec![2.0, 2.0, 2.0]).unwrap();
        assert_eq!(
            linearization.primal().interpret(vec![x.clone()]),
            Ok(vec![Array::vector(vec![16.0, 4.0, 2.0]).unwrap(), x.clone()]),
        );
        assert_eq!(
            linearization.tangent().interpret(vec![Array::vector(vec![1.0, 1.0, 1.0]).unwrap(), x]),
            Ok(vec![Array::vector(vec![32.0, 4.0, 1.0]).unwrap()]),
        );

        // The tangent program keeps the unbounded fused loop, which has no transposition.
        assert!(matches!(
            linearization.pullback().map(|_| ()),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`while` does not support transposition (reverse-mode differentiation through staged \
                               unbounded `while` loops is not supported; eager differentiation executes concrete \
                               duals, and loops built with `with_iteration_bound` stage a transposable masked scan)",
        ));
    }

    #[test]
    fn test_while_differentiation_composes_staged_linearizations_with_later_batching() {
        // Batching the tangent and pullback programs of a staged bounded linearization over tangents and cotangents
        // that share one primal point keeps their residual stacks replicated. `while (x < 10) { x = x * x }` at
        // `x = 2` squares twice to 16, so locally `f(x) = x⁴` and both programs scale by `4 x³ = 32`.
        let (operation, regions) = bounded_squaring_while_operation(10.0, 4);
        let program = while_program(operation, &regions, vec![ArrayType::scalar(DataType::F64)]);
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![Array::scalar(2.0).unwrap()]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![Array::scalar(16.0).unwrap()]);
        assert_eq!(
            residuals,
            vec![
                Array::vector(vec![true, true, false, false]).unwrap(),
                Array::vector(vec![2.0, 4.0, 0.0, 0.0]).unwrap(),
            ],
        );
        let mut axes = vec![BatchAxis::new(0)];
        axes.extend(residuals.iter().map(|_| BatchAxis::replicated()));

        let tangent = linearization
            .tangent()
            .batched(3, ShardingDimension::Replicated, axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        assert_eq!(tangent.output_axes(), &[BatchAxis::new(0)]);
        let tangent = tangent.into_parts().0;
        assert_eq!(
            tangent.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[4], %2:f64[4] .
                let %3:f64[3] = scan [carry_count=1, length=4, reverse=false] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[3], %2:bool[], %3:f64[] .
                        let %4:f64[3] = condition %2 %1 %3 [
                            true={
                                lambda %0:f64[3], %1:f64[] .
                                let %2:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                                    %3:f64[3] = mul %2 %0
                                    %4:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                                    %5:f64[3] = mul %4 %0
                                    %6:f64[3] = add %3 %5
                                in (%6)
                            },
                            false={
                                lambda %0:f64[3], %1:f64[] .
                                in (%0)
                            },
                        ]
                        in (%4)
                    },
                ]
                in (%3)"},
        );
        let mut tangent_inputs = vec![Array::vector(vec![1.0, 2.0, -1.0]).unwrap()];
        tangent_inputs.extend(residuals.iter().cloned());
        assert_eq!(tangent.interpret(tangent_inputs), Ok(vec![Array::vector(vec![32.0, 64.0, -32.0]).unwrap()]));

        let pullback = linearization
            .pullback()
            .unwrap()
            .program()
            .batched(3, ShardingDimension::Replicated, axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        assert_eq!(pullback.output_axes(), &[BatchAxis::new(0)]);
        let pullback = pullback.into_parts().0;
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[4], %2:f64[4] .
                let %3:f64[3] = scan [carry_count=1, length=4, reverse=true] %0 %1 %2 [
                    body={
                        lambda %0:i64[], %1:f64[3], %2:bool[], %3:f64[] .
                        let %4:f64[3] = condition %2 %1 %3 [
                            true={
                                lambda %0:f64[3], %1:f64[] .
                                let %2:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                                    %3:f64[3] = mul %2 %0
                                    %4:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                                    %5:f64[3] = mul %4 %0
                                    %6:f64[3] = add %3 %5
                                in (%6)
                            },
                            false={
                                lambda %0:f64[3], %1:f64[] .
                                in (%0)
                            },
                        ]
                        in (%4)
                    },
                ]
                in (%3)"},
        );
        let mut pullback_inputs = vec![Array::vector(vec![1.0, 2.0, -1.0]).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![Array::vector(vec![32.0, 64.0, -32.0]).unwrap()]));
    }

    #[test]
    fn test_while_transposition() {
        // A staged unbounded loop has no transposable form: its tangent program keeps the fused loop, which recomputes
        // primal state forward through the iterations, so transposing that program, or any other linear `while`,
        // reports the error of the `while` transposition rule, like JAX's `lax.while_loop`.
        let error = DifferentiationError::from(ProgramError::UnsupportedOperation {
            message: "`while` does not support transposition (reverse-mode differentiation through staged unbounded \
                      `while` loops is not supported; eager differentiation executes concrete duals, and loops built \
                      with `with_iteration_bound` stage a transposable masked scan)"
                .to_string(),
        });
        let (operation, regions) = unbounded_squaring_while_operation(16.0);
        let linearization =
            while_program(operation, &regions, vec![ArrayType::scalar(DataType::F64)]).linearize().unwrap();
        assert_eq!(linearization.pullback().map(|_| ()), Err(error.clone()));

        // `while (counter > 0) { (counter, x) = (counter - 1, x + x) }` is linear in `x` and reaches the rule directly.
        let (operation, regions) = counter_doubling_while_operation();
        let linear = while_program(operation, &regions, vec![ArrayType::scalar(DataType::F64); 2]);
        assert_eq!(linear.transpose_with_respect_to(&[1], &[]).map(|_| ()), Err(error));
    }

    #[test]
    fn test_while_transposition_reports_the_error_through_staged_reverse_mode() {
        // Reverse mode through a staged unbounded loop linearizes (through the fused rule and the closed-knownness
        // split) but has no transposable tangent loop, because the fused loop stores no per-iteration residuals.
        // Transposition is late-bound, so the error surfaces on the pullback's first application rather than from
        // `vjp` itself.
        let (while_operation, while_regions) = unbounded_squaring_while_operation(16.0);
        let (_, pullback) = StagedDispatchTestDomain
            .vjp(
                move |x, ()| Ok(bind_while(while_operation, while_regions.clone(), &[x])?.remove(0)),
                Array::scalar(2.0).unwrap(),
                (),
            )
            .unwrap();
        assert_eq!(
            pullback.apply(Array::scalar(1.0).unwrap()),
            Err(ProgramError::UnsupportedOperation {
                message: "`while` does not support transposition (reverse-mode differentiation through staged \
                          unbounded `while` loops is not supported; eager differentiation executes concrete duals, \
                          and loops built with `with_iteration_bound` stage a transposable masked scan)"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_while_transforms_agree_on_aliased_reference_carries() {
        // The body returns its two reference carries swapped, and both carries enter as one allocation, so the swapped
        // handles still denote the allocation that each carry entered with. Whether the loop is accepted must not
        // depend on the transform that reaches it, and every rule rejects it because the body does not return each
        // reference carry as its own body input.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let state_types: Vec<ArrayIrType> =
            vec![reference_type.clone().into(), reference_type.clone().into(), scalar_type.clone().into()];
        let mut condition = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        for state_type in &state_types {
            condition.add_input(state_type.clone());
        }
        let predicate = condition.add_constant(TestIrValue::Array(Array::scalar(false).unwrap()));
        let condition = condition
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = state_types.iter().map(|state_type| body.add_input(state_type.clone())).collect::<Vec<_>>();
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![state[1], state[0], state[2]],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![builder.import_program(condition.clone()), builder.import_program(body.clone())];
        let initial = builder.add_input(scalar_type.clone().into());
        let carry = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let outputs = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), regions, vec![reference, reference, carry], None)
            .unwrap()
            .to_vec();
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[0]], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![value, outputs[2]],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        // A linear program over a reference input that it passes as both reference carries reaches transposition,
        // which validates the loop before it reports that it cannot transpose it.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![builder.import_program(condition.clone()), builder.import_program(body)];
        let reference = builder.add_input(reference_type.into());
        let carry = builder.add_input(scalar_type.clone().into());
        let outputs = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), regions, vec![reference, reference, carry], None)
            .unwrap()
            .to_vec();
        let linear = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![outputs[2]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // The `shard_map` emulator runs the loop in lockstep when its body also reduces over a manual axis of the map
        // (here, through a reduction whose result the body does not use), so a map whose body applies such a loop
        // reaches the rule through the emulator.
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = state_types.iter().map(|state_type| body.add_input(state_type.clone())).collect::<Vec<_>>();
        let sum = ArrayOperation::ParallelReduce(ParallelReduceOperation::new(ReductionKind::Sum, "x".to_string()));
        body.add_instruction(sum, Vec::new(), vec![state[2]], None).unwrap();
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![state[1], state[0], state[2]],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let initial = builder.add_input(scalar_type.clone().into());
        let carry = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), regions, vec![reference, reference, carry], None)
            .unwrap();
        let map_body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let shard_map = ShardMap::new(mesh, vec![replicated.clone(), replicated], Vec::new(), Vec::new()).unwrap();
        let input_types = vec![ArrayIrType::from(scalar_type.clone()); 2];
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let map_body = builder.import_program(map_body);
        let inputs = input_types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let operation = ShardMapOperation::from_boundary(shard_map, input_types, Vec::<ArrayIrType>::new());
        builder.add_instruction(operation, vec![map_body], inputs, None).unwrap();
        let mapped = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();

        let error = ProgramError::Type(TypeError::invalid(
            "`while` body does not return carry 0 as the reference it entered with, so the loop state has no fixed \
             point",
        ));
        let initial = array(Array::scalar(1.0f32).unwrap());
        let carry = array(Array::scalar(2.0f32).unwrap());
        assert_eq!(program.interpret(vec![initial.clone(), carry.clone()]), Err(error.clone()));
        assert_eq!(mapped.interpret(vec![initial.clone(), carry.clone()]), Err(error.clone()));
        assert_eq!(program.clone().discharge_references(0).map(|_| ()), Err(error.clone()));
        assert_eq!(
            program
                .partially_evaluate(&[PartialValue::Known(initial.clone()), PartialValue::Known(carry)])
                .map(|_| ()),
            Err(error.clone()),
        );
        assert_eq!(
            program
                .partially_evaluate(&[PartialValue::Known(initial), PartialValue::Unknown(scalar_type.into())])
                .map(|_| ()),
            Err(error.clone()),
        );
        let axis_extent = DimensionValue::constant(2).unwrap();
        assert_eq!(
            program
                .batched_with_threaded_extent(
                    axis_extent.r#type().into_owned(),
                    ShardingDimension::Replicated,
                    &[BatchAxis::new(0), BatchAxis::new(0)],
                    ProgramBatchingOutputAxesPolicy::Natural,
                )
                .map(|_| ()),
            Err(BatchingError::from(error.clone())),
        );
        assert_eq!(program.jvp().map(|_| ()), Err(DifferentiationError::from(error.clone())));
        assert_eq!(linear.transpose().map(|_| ()), Err(DifferentiationError::from(error)));
    }

    #[test]
    fn test_while_transforms_agree_on_aliased_unswapped_reference_carries() {
        // Both reference carries enter as one allocation, and the body writes through the first one, reads through the
        // second one, and returns both in place, so every read observes the writes of the earlier iterations: after
        // its bound of two iterations, the outputs are `[2 · initial + increment, initial + increment]`. Every rule
        // that supports the loop accepts it and agrees on its values, reference discharge threads the state of the
        // shared allocation through both carries, which therefore stay equal, and transposition validates the loop
        // before it reports that it cannot transpose a `while`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let state_types: Vec<ArrayIrType> =
            vec![reference_type.clone().into(), reference_type.into(), scalar_type.clone().into()];
        let mut condition = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        for state_type in &state_types {
            condition.add_input(state_type.clone());
        }
        let predicate = condition.add_constant(TestIrValue::Array(Array::scalar(true).unwrap()));
        let condition = condition
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut body = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let state = state_types.iter().map(|state_type| body.add_input(state_type.clone())).collect::<Vec<_>>();
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![state[1]], None).unwrap()[0];
        body.add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![state[0], state[2]], None)
            .unwrap();
        let body = body
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![state[0], state[1], value],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let initial = builder.add_input(scalar_type.clone().into());
        let increment = builder.add_input(scalar_type.clone().into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
        let outputs = builder
            .add_instruction(operation, regions, vec![reference, reference, increment], None)
            .unwrap()
            .to_vec();
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[0]], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![value, outputs[2]],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let scalar = |value: f32| array(Array::scalar(value).unwrap());
        let vector = |values: Vec<f32>| array(Array::vector(values).unwrap());
        let expected = vec![scalar(12.0), scalar(11.0)];
        assert_eq!(program.interpret(vec![scalar(1.0), scalar(10.0)]), Ok(expected.clone()));

        let discharged = program.clone().discharge_references(0).unwrap();
        assert_eq!(
            discharged.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[], %3:f32[], %4:f32[] = while [iteration_bound=2] %0 %0 %1 [
                    condition={
                        lambda %0:f32[], %1:f32[], %2:f32[] .
                        let %3:bool[] = const true
                        in (%3)
                    },
                    body={
                        lambda %0:f32[], %1:f32[], %2:f32[] .
                        let %3:f32[] = add %0 %2
                        in (%3, %3, %0)
                    },
                ]
                in (%3, %4)"},
        );
        assert_eq!(discharged.program().interpret(vec![scalar(1.0), scalar(10.0)]), Ok(expected.clone()));

        let known = program
            .partially_evaluate(&[PartialValue::Known(scalar(1.0)), PartialValue::Known(scalar(10.0))])
            .unwrap();
        // An allocation stays on the residual side even when every input is known, as do the loop and the reads that
        // depend on its effects, so the residual program is the source program over the known values.
        assert_eq!(
            known.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:ref<f32[]> = reference_new %0
                    %3:ref<f32[]>, %4:ref<f32[]>, %5:f32[] = while [iteration_bound=2] %2 %2 %1 [
                        condition={
                            lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[] .
                            let %3:bool[] = const true
                            in (%3)
                        },
                        body={
                            lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[] .
                            let %3:f32[] = reference_read %1
                                () = reference_add_update %0 %2
                            in (%0, %1, %3)
                        },
                    ]
                    %6:f32[] = reference_read %3
                in (%6, %5)"},
        );
        assert_eq!(known.interpret(&EagerContext::<TestIrValue, TestIrOperation>::new(), &[]), Ok(expected.clone()));
        let mixed = program
            .partially_evaluate(&[PartialValue::Known(scalar(1.0)), PartialValue::Unknown(scalar_type.clone().into())])
            .unwrap();
        assert_eq!(
            mixed.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:ref<f32[]> = reference_new %1
                    %3:ref<f32[]>, %4:ref<f32[]>, %5:f32[] = while [iteration_bound=2] %2 %2 %0 [
                        condition={
                            lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[] .
                            let %3:bool[] = const true
                            in (%3)
                        },
                        body={
                            lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[] .
                            let %3:f32[] = reference_read %1
                                () = reference_add_update %0 %2
                            in (%0, %1, %3)
                        },
                    ]
                    %6:f32[] = reference_read %3
                in (%6, %5)"},
        );
        assert_eq!(
            mixed.interpret(&EagerContext::<TestIrValue, TestIrOperation>::new(), &[scalar(10.0)]),
            Ok(expected.clone()),
        );

        let axis_extent = DimensionValue::constant(2).unwrap();
        let batched = program
            .batched_with_threaded_extent(
                axis_extent.r#type().into_owned(),
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[2], %2:f32[2] .
                let %3:ref<f32[2]> = reference_new %1
                    %4:dimension<2>, %5:ref<f32[2]>, %6:ref<f32[2]>, %7:f32[2] = while [iteration_bound=2] %0 %3 %3 %2 [
                        condition={
                            lambda %0:dimension<2>, %1:ref<f32[2]>, %2:ref<f32[2]>, %3:f32[2] .
                            let %4:bool[] = const true
                            in (%4)
                        },
                        body={
                            lambda %0:dimension<2>, %1:ref<f32[2]>, %2:ref<f32[2]>, %3:f32[2] .
                            let %4:f32[2] = reference_read %2
                                () = reference_add_update %1 %3
                            in (%0, %1, %2, %4)
                        },
                    ]
                    %8:f32[2] = reference_read %5
                in (%0, %8, %7)"},
        );
        assert_eq!(
            batched.interpret(vec![
                TestIrValue::Dimension(axis_extent.clone()),
                vector(vec![1.0, 2.0]),
                vector(vec![10.0, 20.0]),
            ]),
            Ok(vec![TestIrValue::Dimension(axis_extent), vector(vec![12.0, 24.0]), vector(vec![11.0, 22.0])]),
        );

        let jvp = program.jvp().unwrap();
        // The bounded loop records which iterations ran, and the tangent loop replays them as a masked scan.
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[], %2:f32[], %3:f32[] .
                let %4:ref<f32[]> = reference_new %0
                    %5:ref<f32[]> = reference_new %2
                    %6:i64[] = zero [type=i64[]]
                    %7:bool[2] = zero [type=bool[2]]
                    %8:ref<f32[]>, %9:ref<f32[]>, %10:f32[], %11:i64[], %12:bool[2] = while [iteration_bound=2] \
                %4 %4 %1 %6 %7 [
                        condition={
                            lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[], %3:i64[], %4:bool[2] .
                            let %5:bool[] = const true
                            in (%5)
                        },
                        body={
                            lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[], %3:i64[], %4:bool[2] .
                            let %5:f32[] = reference_read %1
                                () = reference_add_update %0 %2
                                %6:bool[] = one [type=bool[]]
                                %7:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %6
                                %8:bool[2] = dynamic_update_slice %4 %7 %3
                                %9:i64[] = one [type=i64[]]
                                %10:i64[] = add %3 %9
                            in (%0, %1, %5, %10, %8)
                        },
                    ]
                    %13:ref<f32[]>, %14:ref<f32[]>, %15:f32[] = scan [carry_count=3, length=2, reverse=false] %5 %5 %3 \
                %12 [
                        body={
                            lambda %0:i64[], %1:ref<f32[]>, %2:ref<f32[]>, %3:f32[], %4:bool[] .
                            let %5:ref<f32[]>, %6:ref<f32[]>, %7:f32[] = condition %4 %1 %2 %3 [
                                true={
                                    lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[] .
                                    let %3:f32[] = reference_read %1
                                        () = reference_add_update %0 %2
                                    in (%0, %1, %3)
                                },
                                false={
                                    lambda %0:ref<f32[]>, %1:ref<f32[]>, %2:f32[] .
                                    in (%0, %1, %2)
                                },
                            ]
                            in (%5, %6, %7)
                        },
                    ]
                    %16:f32[] = reference_read %8
                    %17:f32[] = reference_read %13
                in (%16, %10, %17, %15)"},
        );
        assert_eq!(
            jvp.interpret(vec![scalar(1.0), scalar(10.0), scalar(1.0), scalar(0.0)]),
            Ok(vec![scalar(12.0), scalar(11.0), scalar(2.0), scalar(1.0)]),
        );

        assert_eq!(
            program.transpose().map(|_| ()),
            Err(DifferentiationError::from(ProgramError::UnsupportedOperation {
                message: "`while` does not support transposition (reverse-mode differentiation through staged \
                          unbounded `while` loops is not supported; eager differentiation executes concrete duals, and \
                          loops built with `with_iteration_bound` stage a transposable masked scan)"
                    .to_string(),
            })),
        );
    }

    #[test]
    fn test_while_type_validate_while_condition_output() {
        // An array predicate must be Boolean, and a batched one must be a prefix of every state shape.
        let state_types = vec![ArrayType::new_static(DataType::F32, [2])];
        assert_eq!(
            ArrayType::validate_while_condition_output(&ArrayType::scalar(DataType::Boolean), &state_types),
            Ok(()),
        );
        assert_eq!(
            ArrayType::validate_while_condition_output(&ArrayType::new_static(DataType::Boolean, [2]), &state_types),
            Ok(()),
        );
        assert_eq!(
            ArrayType::validate_while_condition_output(&ArrayType::scalar(DataType::F32), &state_types),
            Err(TypeError::invalid("`while` condition output type must be a Boolean array, but got f32[]")),
        );
        assert_eq!(
            ArrayType::validate_while_condition_output(&ArrayType::new_static(DataType::Boolean, [3]), &state_types),
            Err(TypeError::invalid(
                "`while` condition predicate shape must be a prefix of every array state shape, but predicate bool[3] \
                 is not a prefix of state f32[2]",
            )),
        );

        // An array IR predicate must be a Boolean array member, so a first-class dimension or a reference predicate is
        // rejected, and the prefix requirement applies to the array state members only.
        let dimension_type = ArrayIrType::Dimension(DimensionType::from(Dimension::Static(3)));
        let state_types = vec![dimension_type.clone(), ArrayType::new_static(DataType::F32, [2]).into()];
        assert_eq!(
            ArrayIrType::validate_while_condition_output(
                &ArrayType::new_static(DataType::Boolean, [2]).into(),
                &state_types,
            ),
            Ok(()),
        );
        assert_eq!(
            ArrayIrType::validate_while_condition_output(&dimension_type, &state_types),
            Err(TypeError::invalid("`while` condition output type must be a Boolean array, but got dimension<3>")),
        );
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Boolean)));
        assert_eq!(
            ArrayIrType::validate_while_condition_output(&reference_type, &state_types),
            Err(TypeError::invalid("`while` condition output type must be a Boolean array, but got ref<bool[]>")),
        );
    }

    #[test]
    fn test_while_type_is_batched_predicate() {
        // Only a non-scalar Boolean array predicate is batched, so a scalar predicate, a first-class dimension, and a
        // reference to a non-scalar Boolean array are not.
        let batched_type = ArrayType::new_static(DataType::Boolean, [2]);
        assert!(!ArrayType::is_batched_predicate(&ArrayType::scalar(DataType::Boolean)));
        assert!(ArrayType::is_batched_predicate(&batched_type));
        assert!(!ArrayIrType::is_batched_predicate(&ArrayType::scalar(DataType::Boolean).into()));
        assert!(ArrayIrType::is_batched_predicate(&batched_type.clone().into()));
        assert!(!ArrayIrType::is_batched_predicate(&ArrayIrType::Dimension(DimensionType::from(Dimension::Static(2)))));
        assert!(!ArrayIrType::is_batched_predicate(&ArrayIrType::Reference(ReferenceType::new(batched_type))));
    }

    #[test]
    fn test_while_type_is_loop_invariant_state() {
        // Only a first-class dimension has no per-item value to mask. Array state is maskable, and reference state is
        // rejected under a batched predicate by its own rule (refer to `test_while_type_inference_array_ir_state`).
        let scalar_type = ArrayType::scalar(DataType::F32);
        assert!(!ArrayType::is_loop_invariant_state(&scalar_type));
        assert!(!ArrayIrType::is_loop_invariant_state(&scalar_type.clone().into()));
        assert!(!ArrayIrType::is_loop_invariant_state(&ArrayIrType::Reference(ReferenceType::new(scalar_type))));
        let dimension_type = ArrayIrType::Dimension(DimensionType::from(Dimension::Static(2)));
        assert!(ArrayIrType::is_loop_invariant_state(&dimension_type));
    }

    #[test]
    fn test_while_predicate_any_true() {
        // An array predicate continues the loop when any of its items is true, independently of its layout.
        assert_eq!(Array::scalar(true).unwrap().any_true(), Ok(true));
        assert_eq!(Array::scalar(false).unwrap().any_true(), Ok(false));
        assert_eq!(Array::vector(vec![false, true]).unwrap().any_true(), Ok(true));
        assert_eq!(Array::vector(vec![false, false]).unwrap().any_true(), Ok(false));
        assert_eq!(Array::vector(Vec::<bool>::new()).unwrap().any_true(), Ok(false));
        let predicate_type =
            ArrayType::new_static(DataType::Boolean, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        assert_eq!(Array::from_elements(predicate_type, &[false, true]).unwrap().any_true(), Ok(true));
        assert_eq!(
            Array::vector(vec![1.0]).unwrap().any_true(),
            Err(ProgramError::Concretization {
                message: "cannot use a value of type `f64[1]` as a Boolean `while` predicate".to_string(),
            }),
        );

        // Array IR predicates delegate to their array payload and reject dimension and reference predicates.
        assert_eq!(ArrayIrValue::Array(Array::vector(vec![false, true]).unwrap()).any_true(), Ok(true));
        assert_eq!(
            ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3).unwrap()).any_true(),
            Err(ProgramError::Concretization {
                message: "cannot use first-class dimension `3` as a `while` predicate".to_string(),
            }),
        );
        assert_eq!(
            ArrayIrValue::<Array>::Reference(ArrayReference::new(Array::scalar(true).unwrap())).any_true(),
            Err(ProgramError::Concretization {
                message: "cannot use reference `ref<bool[]>` as a `while` predicate".to_string(),
            }),
        );

        // Symbolic values inherit the scalar default, which surfaces their concretization error.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            trace.input(ArrayType::scalar(DataType::Boolean)).any_true(),
            Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from a tracer".to_string(),
            }),
        );
    }

    #[test]
    fn test_while_predicate_mask_select() {
        // A scalar array predicate selects one candidate wholesale.
        let on_true = Array::vector(vec![1.0f64, 2.0]).unwrap();
        let on_false = Array::vector(vec![-1.0f64, -2.0]).unwrap();
        assert_eq!(Array::scalar(true).unwrap().mask_select(&on_true, &on_false), Ok(on_true.clone()));
        assert_eq!(Array::scalar(false).unwrap().mask_select(&on_true, &on_false), Ok(on_false.clone()));

        // Predicate item `i` masks the contiguous per-item block of `on_true` and `on_false` elements it governs.
        let predicate = Array::vector(vec![false, true]).unwrap();
        let on_true =
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2]), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let on_false =
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2]), &[-1.0, -2.0, -3.0, -4.0])
                .unwrap();
        assert_eq!(
            predicate.mask_select(&on_true, &on_false),
            Ok(Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [2, 2]), &[-1.0, -2.0, 3.0, 4.0])
                .unwrap()),
        );

        // Matching or divisible element counts do not make incompatible shapes broadcastable along prefix axes.
        let flattened = Array::vector(vec![1.0f64, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            predicate.mask_select(&flattened, &flattened),
            Err(ProgramError::Type(TypeError::invalid(
                "`while` condition predicate shape must be a prefix of every array state shape, but predicate bool[2] \
                 is not a prefix of state f64[4]",
            ))),
        );
        let matrix_predicate =
            Array::from_elements(ArrayType::new_static(DataType::Boolean, [2, 2]), &[false, true, true, false])
                .unwrap();
        assert_eq!(
            matrix_predicate.mask_select(&flattened, &flattened),
            Err(ProgramError::Type(TypeError::invalid(
                "`while` condition predicate shape must be a prefix of every array state shape, but predicate \
                 bool[2, 2] is not a prefix of state f64[4]",
            ))),
        );
        let mismatched = Array::vector(vec![1.0f64, 2.0]).unwrap();
        assert_eq!(
            predicate.mask_select(&on_true, &mismatched),
            Err(ProgramError::UnsupportedOperation {
                message: "`mask_select` requires congruent inputs, but got `f64[2, 2]` and `f64[2]`".to_string(),
            }),
        );

        // An empty predicate has a valid empty prefix and selects an empty state without dividing by zero.
        let empty_predicate = Array::vector(Vec::<bool>::new()).unwrap();
        let empty_state = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0, 2]), &[]).unwrap();
        assert_eq!(empty_predicate.mask_select(&empty_state, &empty_state), Ok(empty_state));

        // Predicate and branch layouts are independent of logical masking. The output preserves the congruent branch
        // layout, including its hole, while selecting exact element bytes in logical order.
        let predicate_type =
            ArrayType::new_static(DataType::Boolean, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let predicate = Array::from_elements(predicate_type, &[false, true]).unwrap();
        let branch_type =
            ArrayType::new_static(DataType::U16, [2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![-6, 2])));
        let on_true = Array::from_elements(branch_type.clone(), &[0x1111u16, 0x2222, 0x3333, 0x4444]).unwrap();
        let on_false = Array::from_elements(branch_type.clone(), &[0xaaaau16, 0xbbbb, 0xcccc, 0xdddd]).unwrap();
        let selected = predicate.mask_select(&on_true, &on_false).unwrap();
        assert_eq!(selected.r#type().as_ref(), &branch_type);
        assert_eq!(selected.elements::<u16>(), Ok(vec![0xaaaa, 0xbbbb, 0x3333, 0x4444]));
        assert_eq!(selected.storage_bytes(), [0x33, 0x33, 0x44, 0x44, 0, 0, 0xaa, 0xaa, 0xbb, 0xbb]);

        // Array members of array IR carries delegate to the array payload.
        let scalar_true = ArrayIrValue::Array(Array::scalar(true).unwrap());
        let scalar_false = ArrayIrValue::Array(Array::scalar(false).unwrap());
        let batched = ArrayIrValue::Array(Array::vector(vec![false, true]).unwrap());
        assert_eq!(
            batched.mask_select(
                &ArrayIrValue::Array(Array::vector(vec![1.0f64, 2.0]).unwrap()),
                &ArrayIrValue::Array(Array::vector(vec![-1.0f64, -2.0]).unwrap()),
            ),
            Ok(ArrayIrValue::Array(Array::vector(vec![-1.0f64, 2.0]).unwrap())),
        );

        // Reference carries are selected wholesale by a scalar predicate: the handle itself is selected, never its
        // contents, so the selected carry aliases the chosen input. A batched predicate cannot mask a reference per
        // item, so it accepts only identical carries.
        let on_true = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0f32).unwrap()));
        let on_false = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(2.0f32).unwrap()));
        assert_eq!(scalar_true.mask_select(&on_true, &on_false), Ok(on_true.clone()));
        assert_eq!(scalar_false.mask_select(&on_true, &on_false), Ok(on_false.clone()));
        assert_eq!(batched.mask_select(&on_true, &on_true), Ok(on_true.clone()));
        assert_eq!(
            batched.mask_select(&on_true, &on_false),
            Err(ProgramError::UnsupportedOperation {
                message: "a batched `while` predicate cannot select between distinct reference carries per batch \
                          item; reference state has one value per allocation"
                    .to_string(),
            }),
        );

        // Dimension carries are selected wholesale by a scalar predicate. A batched predicate returns equal dimension
        // carries unchanged, because masking a loop-invariant carry is the identity, but one dimension value cannot
        // represent independently masked per-item extents, so distinct dimension carries need a scalar predicate.
        let three = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let four = ArrayIrValue::Dimension(DimensionValue::constant(4).unwrap());
        assert_eq!(scalar_true.mask_select(&three, &four), Ok(three.clone()));
        assert_eq!(scalar_false.mask_select(&three, &four), Ok(four.clone()));
        assert_eq!(batched.mask_select(&three, &three), Ok(three.clone()));
        assert_eq!(
            batched.mask_select(&three, &four),
            Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from a value of type `bool[2]`; expected `bool[]`"
                    .to_string(),
            }),
        );

        // A state element can only be selected against an element of the same kind, and only an array predicate can
        // select at all.
        assert_eq!(
            scalar_true.mask_select(&three, &on_true),
            Err(ProgramError::Type(TypeError::invalid(
                "`while` predicate cannot select between mismatched state types `dimension<3>` and `ref<f32[]>`",
            ))),
        );
        assert_eq!(
            three.mask_select(&on_true, &on_false),
            Err(ProgramError::Concretization {
                message: "cannot use first-class dimension `3` as a `while` predicate".to_string(),
            }),
        );
        assert_eq!(
            on_true.mask_select(&three, &four),
            Err(ProgramError::Concretization {
                message: "cannot use reference `ref<f32[]>` as a `while` predicate".to_string(),
            }),
        );

        // Symbolic values inherit the scalar default, which surfaces their concretization error.
        let trace = TracingContext::<Array, ArrayOperation<Array>>::new();
        let predicate = trace.input(ArrayType::scalar(DataType::Boolean));
        let state = trace.input(ArrayType::scalar(DataType::F64));
        assert_eq!(
            predicate.mask_select(&state, &state).map(|_| ()),
            Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from a tracer".to_string(),
            }),
        );
    }

    #[test]
    fn test_while_predicate_mask_select_manual_variation() {
        // Masked selection checks the manual variation of a batched predicate against the state: a scalar predicate
        // selects a whole candidate, so its variation does not constrain the state.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = |rank| Sharding::replicated(mesh.clone(), rank).with_varying_manual_axes(["m"]).unwrap();
        let varying_scalar_type = ArrayType::scalar(DataType::Boolean).with_sharding(varying_sharding(0)).unwrap();
        let varying_vector_type =
            ArrayType::new_static(DataType::Boolean, [2]).with_sharding(varying_sharding(1)).unwrap();
        let invariant = Array::vector(vec![false, true]).unwrap();
        let varying = Array::from_elements(varying_vector_type.clone(), &[true, false]).unwrap();
        let predicate = Array::from_elements(varying_scalar_type, &[true]).unwrap();
        assert_eq!(predicate.mask_select(&invariant, &invariant), Ok(invariant.clone()));
        assert_eq!(predicate.mask_select(&varying, &varying), Ok(varying.clone()));

        // A batched predicate may be invariant over axes that the state varies over, but not the other way around.
        let predicate = Array::vector(vec![true, false]).unwrap();
        let other = Array::from_elements(varying_vector_type.clone(), &[false, true]).unwrap();
        assert_eq!(
            predicate.mask_select(&varying, &other),
            Ok(Array::from_elements(varying_vector_type.clone(), &[true, true]).unwrap()),
        );
        let predicate = Array::from_elements(varying_vector_type.clone(), &[false, true]).unwrap();
        assert_eq!(
            predicate.mask_select(&varying, &other),
            Ok(Array::from_elements(varying_vector_type, &[false, false]).unwrap()),
        );
        assert_eq!(
            predicate.mask_select(&invariant, &invariant),
            Err(TypeError::invalid(
                "batched `while` predicate `bool[2][sharding={mesh<['m'=2:manual]>, [{}], varying_manual={'m'}}]` \
                 varies over manual axes that state `bool[2]` does not vary over"
            )
            .into()),
        );
    }
}
