//! Loops over loop-carried state through [`WhileOperation`], which applies an attached body [`Region`](crate::Region)
//! to the state while an attached condition [`Region`](crate::Region) over the same state produces a true predicate,
//! and returns the final state. [`While::r#while`](While::while) traces a condition and a body over structured state
//! using an explicit context. The global [`r#while()`](fn@while) function recovers that context from the state values.
//! [`WhileOptions`] selects the iteration bound. Values that a loop uses as predicates implement [`WhilePredicate`],
//! which decides whether the loop continues and masks state updates per item when the loop is interpreted.
//!
//! A while application takes the initial state and returns the final state with the same signature. The condition and
//! the body receive the state, the condition returns one Boolean predicate, and the body returns the next state, in
//! which each reference carry must be the reference that it entered with. [`WhileType`] states which predicates are
//! valid for which state in a type universe, including that a batched (i.e., prefix-shaped) predicate masks the state
//! per item. An optional [`iteration_bound`](WhileOperation::with_iteration_bound) truncates the loop after that many
//! body applications by definition, which every rule honors. Every transformation rule validates this complete contract
//! before it executes, rewrites, discharges, or stages the regions, so whether an application is accepted depends
//! neither on the transformation nor on the iteration count. The rules are:
//!
//!   - **Interpretation:** The condition is evaluated on the current state, and the body is applied while any item of
//!     the predicate is true and the bound is not reached. Each item of the state takes the body's update exactly when
//!     its predicate is true.
//!   - **Reference Discharge:** Discharged references become state carries. A condition that mutates a discharged
//!     allocation rotates the loop into do-while form, which evaluates the condition exactly as often and in the same
//!     order as the source.
//!   - **Partial Evaluation:** An all-known loop folds whole. Otherwise, a pure loop first checks the relationships
//!     between its inputs that their types leave unproven in an entry guard. Known loop-invariant state then folds into
//!     both regions as constants, and a known state subset whose next values and predicate fold from known state alone
//!     splits the loop into a known loop and the original loop (i.e., the closed-knownness split). An effectful loop
//!     is residualized unchanged.
//!   - **Batching:** A fixed point over the state batch axes batches the regions once. A predicate that becomes
//!     batched widens every array state element, and the bound loop masks its updates per item.
//!   - **Differentiation:** An eager loop runs its duals directly and resumes with the staged rule from the current
//!     duals once its predicate stops concretizing to one scalar Boolean. A bounded staged loop stores per-iteration
//!     residuals in `[bound, ...]` stacks and stages one masked tangent `scan`, whose transposition gives reverse mode,
//!     after rewriting a batched predicate into a scalar-predicate masked form over the same state. An unbounded staged
//!     loop stages one fused doubled-state loop, which supports forward mode only, because staged linear `while` loops
//!     do not support transposition.
//!
//! The XLA backend lowers a while loop to [StableHLO `while`](https://openxla.org/stablehlo/spec#while).

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
use crate::operations::collectives::parallel_vary::{
    ManualVariationAlignment, PARALLEL_VARY_OPERATION_NAME, ParallelVaryOperation,
};
use crate::operations::constants::one::OneOperation;
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::control_flow::condition::ConditionOperation;
use crate::operations::control_flow::scan::ScanOperation;
use crate::operations::control_flow::select::SelectOperation;
use crate::operations::control_flow::{
    ResidualStackOperation, canonicalize_carry_axes, passed_through_carries, refine_output_types, stabilize_carry_axes,
    validate_output_identities, validate_reference_carry_roots, validate_region_input,
};
use crate::operations::manipulation::broadcasting::BroadcastOperation;
use crate::operations::manipulation::slicing::DynamicUpdateSliceOperation;
use crate::operations::reductions::{ReduceOperation, ReductionKind};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{
    PartialEvaluation, PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput,
    PartialEvaluationOutput, PartialEvaluationValue, PartialValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    AtomId, AttachedRegions, Concretizable, InputRegionProvenance, MaybeZero, Operation, OperationFormatter,
    OperationProjection, OperationProvider, OutputRegionProvenance, Program, ProgramBuilder, ProgramError,
    ReferenceAccessMode, ReferenceDischargeAllocationId, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeRegionBoundary, ReferenceDischargeRegionBoundaryInsertion,
    ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceType, RegionDataFlow, RegionDataFlowBoundary,
    RegionDataFlowRule, RegionDataFlowSource, RegionDataFlowSources, RegionInterface, RegionLiveness, RegionRef,
    RegionSlot, Type, TypeError, Typed, Value,
};
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};

/// Type-family predicate and state semantics for [`WhileOperation`].
///
/// Implementations define which types represent valid predicates for a given loop-carried state, which predicates are
/// batched (i.e., carry one termination decision per item of their leading axes rather than one for the whole loop),
/// and which state has no per-item value that a batched predicate could mask. These rules are independent of the
/// execution domain, so inference, transforms, and lowering can validate loops without requiring any runtime value
/// representation. The [`WhilePredicate`] values of a domain supply the matching runtime semantics: the loop continues
/// while any item of its predicate is `true`, and the items whose predicate is `false` keep their carried state.
///
/// [`WhileOperation`] validates the rest of the loop contract separately and identically for every type family. The
/// condition and the body consume the same state signature, the condition returns exactly one predicate, and the body
/// returns the next state with that same signature, in which each reference carry must be the reference that it
/// entered with. A loop with a batched predicate must be pure and free of deferred work, because neither observable
/// effects nor transformation obligations can be masked back out for the items that have finished, and its body must
/// pass every [loop-invariant](Self::is_loop_invariant_state) state element through unchanged, which makes masking
/// that element the identity on every path.
///
/// # Array Type Families
///
/// [`ArrayType`] predicates are Boolean arrays whose shape is a prefix of every state shape. A rank-zero predicate
/// is the ordinary whole-loop condition, while a batched predicate broadcasts against every state element along its
/// leading axes, which makes the per-item masking well-defined. Batching produces such loops when a predicate becomes
/// batched, and the consumers of the loop (i.e., interpretation, differentiation, and lowering) implement the masked
/// semantics. Array state is always maskable.
///
/// [`ArrayIrType`] predicates are Boolean array members, and the state may mix arrays, first-class dimensions, and
/// references. The prefix requirement applies to the array members only, because a first-class dimension has no
/// shape. Under a batched predicate, a dimension carry is loop-invariant state: one dimension value cannot represent
/// independently masked per-item extents, but masking a carry that the body passes through unchanged is the identity,
/// and the [`mask_select`](WhilePredicate::mask_select) function of [`ArrayIrValue`] returns such equal dimension
/// carries unchanged. Structural array IR batching relies on this to thread its loop-invariant mapped extent through
/// batch-varying loops. Reference state is supported under scalar predicates only: a batched predicate requires
/// reference discharge first, because reference effects cannot be masked per item.
///
/// A manual region is a [`ShardMapOperation`](crate::ShardMapOperation) body. Devices whose predicates differ along
/// its manual mesh axes run different numbers of iterations, so every array state element (and, for [`ArrayIrType`],
/// every reference referent) must vary over each manual mesh axis that the predicate varies over (refer to
/// [`Sharding::varying_manual_axes`]). A first-class dimension carries no manual variation and is not checked.
pub trait WhileType: Type + ArrayMemberType + From<ArrayType> {
    // TODO(eaplatanios): Review from here onwards.
    /// Validates the type family's restrictions on using a value of type `predicate_type` as the predicate of a loop
    /// over state of types `state_types`. [`WhileOperation`] validates the predicate count and the state signatures
    /// separately.
    ///
    /// For example, [`ArrayType`] requires a Boolean predicate whose shape is a prefix of every state shape, and state
    /// that varies over every manual mesh axis that the predicate varies over.
    ///
    /// # Parameters
    ///
    ///   - `predicate_type`: Type of the predicate returned by the condition region.
    ///   - `state_types`: Types of the loop-carried state that the condition and the body consume.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if `predicate_type` is not a valid predicate type for that state according to the type
    /// family's rules.
    fn validate_while_predicate(predicate_type: &Self, state_types: &[Self]) -> Result<(), TypeError>;

    /// Returns whether `predicate_type` describes a batched predicate, which carries one termination decision per item
    /// of its leading axes rather than one for the whole loop (e.g., a non-scalar Boolean [`ArrayType`] predicate). A
    /// loop with a batched predicate keeps running for active items after others have finished and re-evaluates the
    /// condition and the body over every item on each iteration, so [`WhileOperation`] requires both regions to be pure
    /// and masks the state updates per item (refer to [`WhilePredicate`]). Type families without batched predicates can
    /// return `false`.
    fn is_batched_while_predicate(predicate_type: &Self) -> bool;

    /// Returns whether state of type `state_type` has no per-item value that a batched predicate could mask (e.g., a
    /// first-class dimension). Under a batched predicate, [`WhileOperation`] requires the body to pass such state
    /// through unchanged, which makes masking it the identity. Type families whose state is always maskable (e.g.,
    /// [`ArrayType`]) return `false`.
    fn is_loop_invariant_state(state_type: &Self) -> bool;
}

impl WhileType for ArrayType {
    #[inline]
    fn validate_while_predicate(predicate_type: &Self, state_types: &[Self]) -> Result<(), TypeError> {
        validate_while_array_predicate(predicate_type, state_types)?;
        validate_while_state_variation(predicate_type, state_types)
    }

    #[inline]
    fn is_batched_while_predicate(predicate_type: &Self) -> bool {
        predicate_type.rank() > 0
    }

    #[inline]
    fn is_loop_invariant_state(_state_type: &Self) -> bool {
        false
    }
}

impl WhileType for ArrayIrType {
    fn validate_while_predicate(predicate_type: &Self, state_types: &[Self]) -> Result<(), TypeError> {
        let Self::Array(predicate) = predicate_type else {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got `{predicate_type}`",
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
    fn is_batched_while_predicate(predicate_type: &Self) -> bool {
        matches!(predicate_type, Self::Array(r#type) if r#type.rank() > 0)
    }

    #[inline]
    fn is_loop_invariant_state(state_type: &Self) -> bool {
        matches!(state_type, Self::Dimension(_))
    }
}

/// Value-level predicate contract that backs the masked loop semantics of [`WhileOperation`] during interpretation.
/// The shared [`InterpretableOperation`] rule of [`WhileOperation`] requires it of the values of the interpreting
/// domain, so custom value representations implement it to reuse that rule.
///
/// A while condition may produce a batched Boolean predicate (i.e., one termination decision per item of its leading
/// axes), whose shape is a prefix of every state shape (refer to [`WhileType`]). Interpretation then continues while
/// [`any_true`](Self::any_true) holds and updates each state element with [`mask_select`](Self::mask_select), so the
/// items whose predicate is false keep their carried state while the active items take the candidate update of the
/// body. The predicate is recomputed from the whole state on every iteration, so a finished item whose predicate reads
/// its own state alone can never rejoin the loop (its state is frozen), whereas an item whose predicate also reads
/// other items rejoins whenever that predicate turns true again. Every path (i.e., interpretation, the masked form of
/// differentiation, and the lowering) decides this the same way.
///
/// Per-item masking selects values after evaluating the body for every item; it does not skip the body computations of
/// inactive items. The body must therefore remain defined for their frozen state too. During differentiation,
/// undefined derivatives in those computations can produce non-finite cotangents even when their selected output is
/// inactive.
///
/// The default implementations are the scalar-predicate semantics, expressed through [`Concretizable<bool>`]: the
/// truth of the predicate decides continuation, and a true predicate takes the candidate wholesale. Value types with
/// genuinely batched payloads (e.g., [`Array`]) override both functions with per-item semantics, and symbolic values
/// (e.g., tracers and capture references) inherit the defaults, which surface the errors of
/// [`Concretizable::concretize`], because a staged loop is consumed by staging and lowering rather than by eager
/// interpretation.
pub trait WhilePredicate: Concretizable<bool> + Clone + Sized {
    /// Returns `true` when any element of this Boolean predicate is true, which is the loop-continuation decision.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when this value is not a Boolean predicate or does not have a concrete value (e.g.,
    /// the concretization error of a symbolic value).
    fn any_true(&self) -> Result<bool, ProgramError> {
        self.concretize()
    }

    /// Selects between `on_true` and `on_false` per element under this Boolean predicate, broadcasting the predicate
    /// against their shape along its leading (i.e., prefix) axes.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when this value is not a Boolean predicate whose shape is a prefix of the shape of
    /// `on_true` and `on_false`, when the two candidates have different types or (under a batched predicate) do not
    /// vary over every manual mesh axis that the predicate varies over, or when this value does not have a concrete
    /// value.
    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        Ok(if self.concretize()? { on_true.clone() } else { on_false.clone() })
    }
}

// Batched while-predicate semantics for `Array`: `any_true` reduces the whole Boolean payload with `or`, and
// `mask_select` broadcasts the predicate against its other inputs along its leading (prefix) axes, so predicate item
// `i` masks the contiguous per-item block of `on_true` / `on_false` elements it governs.
impl WhilePredicate for Array {
    fn any_true(&self) -> Result<bool, ProgramError> {
        if !self.r#type().data_type().is_boolean() {
            return Err(ProgramError::Concretization {
                message: format!(
                    "cannot use a value of type `{}` as a Boolean `{}` predicate",
                    self.r#type(),
                    WHILE_OPERATION_NAME,
                ),
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
                    "batched `{}` predicate `{}` varies over manual axes that state `{}` does not vary over",
                    WHILE_OPERATION_NAME,
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
                message: format!("cannot use first-class dimension `{value}` as a `{WHILE_OPERATION_NAME}` predicate"),
            }),
            Self::Reference(value) => Err(ProgramError::Concretization {
                message: format!("cannot use reference `{value}` as a `{WHILE_OPERATION_NAME}` predicate"),
            }),
        }
    }

    fn mask_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let predicate = match self {
            Self::Array(predicate) => predicate,
            Self::Dimension(value) => {
                return Err(ProgramError::Concretization {
                    message: format!(
                        "cannot use first-class dimension `{value}` as a `{WHILE_OPERATION_NAME}` predicate",
                    ),
                });
            }
            Self::Reference(value) => {
                return Err(ProgramError::Concretization {
                    message: format!("cannot use reference `{value}` as a `{WHILE_OPERATION_NAME}` predicate"),
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
                        message: format!(
                            "a batched `{WHILE_OPERATION_NAME}` predicate cannot select between distinct reference \
                             carries per batch item; reference state has one value per allocation",
                        ),
                    });
                }
                Ok(Self::Reference(if predicate.concretize()? { on_true.clone() } else { on_false.clone() }))
            }
            _ => Err(TypeError::invalid(format!(
                "`{}` predicate cannot select between mismatched state types `{}` and `{}`",
                WHILE_OPERATION_NAME,
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

impl<C: Context<Type = ArrayType, Value: Concretizable<bool>>> WhilePredicate
    for BatchingTracer<C, ArrayBatchingPolicy>
{
}

impl<C: Context<Type: DifferentiableType, Value: Concretizable<bool>>, P: DifferentiationPolicy<C>> WhilePredicate
    for DifferentiationTracer<C, P>
{
}

/// Canonical operation name for [`WhileOperation`].
pub const WHILE_OPERATION_NAME: &str = "while";

/// [`Operation`] that repeatedly applies a body [`Region`](crate::Region) to loop-carried state while a condition
/// [`Region`](crate::Region) over the same state produces a true Boolean predicate. The condition and the body consume
/// identical state type signatures, the body produces the next state with that same signature, and the operation
/// outputs the final state once the condition produces false or the optional semantic iteration bound truncates the
/// loop (refer to [`Self::with_iteration_bound`]).
///
/// The condition and the body are not part of this payload: they are [`Region`](crate::Region)s attached to the
/// [`Instruction`](crate::Instruction) that applies the operation, in the [`region_slots`](Operation::region_slots)
/// order `["condition", "body"]`, and semantic rules reach them through their driver-granted region access. Owned
/// loops supply the two [`Program`]s through the `driver` argument of [`Context::bind`].
/// [`Operation::infer_output_types`] validates the loop contract over the attached [`RegionInterface`]s: the condition
/// and the body share the loop-carried state input signature, the condition returns exactly one Boolean predicate that
/// [`WhileType`] admits for that state, and the body returns the state signature. A batched (i.e., per-item) predicate
/// requires both regions to be pure and free of deferred work, because neither observable effects nor transformation
/// obligations can be masked for the items that have finished.
///
/// Differentiation through a while loop follows one of three regimes:
///
///   - **Eager:** When the primal values of the differentiation context are concrete, the forward-mode rule evaluates
///     the condition on the primal state and applies the body once over the current dual state on each iteration,
///     respecting any iteration bound. During linearization, partial evaluation records the executed tangent
///     operations as a straight-line (and therefore transposable) pushforward, so eager reverse mode works without
///     replaying primal body effects. Once the predicate stops concretizing to one scalar Boolean (e.g., because it
///     becomes batched), a loop whose condition is pure and free of deferred work resumes from the current duals with
///     the staged regimes below and the remaining iteration bound.
///   - **Bounded Staged:** When the primal values are tracers and the loop has an iteration bound `B`, the rule stages
///     an augmented primal loop that stores every per-iteration residual in a preallocated `[B, …]` stack, together
///     with a Boolean validity mask, and the tangent side becomes one masked linear [`ScanOperation`] of length `B`,
///     whose per-iteration condition forwards tangents unchanged without evaluating residuals beyond the actual
///     iteration count. Reverse mode transposes that scan when its residual types support the required storage and
///     transposition operations. A loop with a batched predicate is first rewritten into a scalar-predicate masked form
///     over the same state, whose body recomputes the predicate to select its updates, so the rewritten loop carries
///     no mask of its own.
///   - **Unbounded Staged:** Shared forward mode stages one fused loop over the primal and tangent state. Separate
///     linearization can split a pure loop when its predicate depends only on known primal state, in which case the
///     tangent loop recomputes the primal values that it needs because no statically shaped residual stack is
///     available. Loops that cannot satisfy this partition are rejected, and unbounded tangent loops do not support
///     transposition.
///
/// The `T` parameter fixes the type universe of the loop in the payload itself. Consequently, one concrete
/// [`WhileOperation<T>`](WhileOperation) has exactly one [`Operation<Type = T>`](Operation) contract even though the
/// shared implementation supports homogeneous array and array IR loops.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WhileOperation<T: Type> {
    /// Optional semantic iteration bound of the loop (refer to [`Self::with_iteration_bound`]).
    iteration_bound: Option<usize>,

    /// [`PhantomData`] marker tying this [`Operation`] to the [`Type`] universe in which it is valid.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> WhileOperation<T> {
    /// Creates a new [`WhileOperation`] with no semantic iteration bound. The condition and body [`Program`]s are
    /// supplied separately as the attached regions of the operation, through the region driver passed to
    /// [`Context::bind`], and [`Operation::infer_output_types`] validates the loop contract over their interfaces.
    pub const fn new() -> Self {
        Self { iteration_bound: None, marker: PhantomData }
    }

    /// Returns this [`WhileOperation`] with its semantic iteration bound set to `bound`. Passing [`None`] restores the
    /// default unbounded loop.
    ///
    /// The bound is semantic rather than a hint or an unchecked promise: a bounded loop runs at most `bound` iterations
    /// by definition, so a loop whose condition would keep it running longer is truncated after `bound` body
    /// applications. Every rule honors this truncation. For example, interpretation exits the loop once the bound is
    /// reached even while the condition still produces true, and the XLA lowering threads an iteration counter through
    /// the `stablehlo.while` state and conjoins `counter < bound` into the loop condition. The bound also makes staged
    /// reverse-mode differentiation possible, because it gives the linearization of the loop a static residual-stack
    /// length (refer to the bounded staged regime described on [`WhileOperation`]).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Type`] when `bound` is zero or exceeds [`MAX_DIMENSION_EXTENT`], which keeps it within
    /// the signed 64-bit counters of compiled loops and the static residual-stack length used by differentiation.
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
                "`{WHILE_OPERATION_NAME}` iteration bound {bound} exceeds \
                 the maximum supported extent {MAX_DIMENSION_EXTENT}",
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
}

impl<T: WhileType> WhileOperation<T> {
    /// Validates the application of this loop to inputs of the types `input_types` with the `condition` and `body`
    /// regions, and returns its output types. Every rule validates this complete contract before it executes, rewrites,
    /// discharges, or stages the regions, so that whether an application is accepted does not depend on the transform
    /// that reaches it. The contract consists of the boundary that [`Operation::infer_output_types`] validates, of the
    /// reference contract that only the body reveals (refer to [`validate_reference_carry_roots`]), and of the loop
    /// invariance of the state that [`WhileType::is_loop_invariant_state`] requires under a batched predicate: such
    /// state has no per-item value that masking could select, so the body must pass it through unchanged, which makes
    /// masking it the identity on every path (eager interpretation, the masked form of differentiation, and the
    /// lowering all forward it unmasked).
    ///
    /// # Parameters
    ///
    ///   - `input_types`: Types of the operation inputs (i.e., the initial state).
    ///   - `condition`: Condition region of the loop.
    ///   - `body`: Body region of the loop.
    fn validate_application<V: Value<Type = T>, O: Operation<Type = T>>(
        &self,
        input_types: &[T],
        condition: RegionRef<'_, V, O>,
        body: RegionRef<'_, V, O>,
    ) -> Result<Vec<T>, ProgramError> {
        let output_types = self.infer_output_types(input_types, &[condition.interface(), body.interface()])?;
        // The body must return every reference carry as its own body input. This is checked even when no iteration
        // runs, so that whether a loop is accepted does not depend on the transform that reaches it or on how its
        // inputs alias.
        validate_reference_carry_roots(WHILE_OPERATION_NAME, body, 0..body.input_ids().len())?;
        let state_types = body.input_types();
        if T::is_batched_while_predicate(&condition.output_types()[0]) {
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

    /// Validates the loop contract over the two attached region interfaces (`["condition", "body"]` region order) and
    /// returns them. The loop-carried state signature is the body's input signature: the condition must consume the
    /// same state and return exactly one Boolean predicate valid for that state under [`WhileType`], and the body must
    /// return the state signature. A batched (per-item) predicate additionally requires both regions to be pure and
    /// free of deferred work: the loop keeps running for still-active items after others finish, so the condition and
    /// body re-execute over every item each iteration, and neither observable effects nor transformation obligations
    /// (refer to the [Deferred Work](crate::Effects#deferred-work) section of [`Effects`](crate::Effects)) can be
    /// masked back out for the finished items the way values can. This mirrors JAX's `_while_loop_batching_rule`, which
    /// rejects IO effects once the predicate is batched.
    fn validate_interfaces<'i>(
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
        T::validate_while_predicate(&condition_output_types[0], state_types)?;
        check_types!(@same, format!("`{WHILE_OPERATION_NAME}` body output"), [
            state_types,
            body_interface.output_types(),
        ]);
        if T::is_batched_while_predicate(&condition_output_types[0])
            && (!condition_interface.effects().is_empty() || !body_interface.effects().is_empty())
        {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` loop with a batched predicate must be pure because observable effects \
                 cannot be masked for finished batch items",
            )));
        }
        if T::is_batched_while_predicate(&condition_output_types[0])
            && (condition_interface.has_deferred_work() || body_interface.has_deferred_work())
        {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` loop with a batched predicate cannot carry deferred work because finished \
                 batch items would repeat its transformation obligation",
            )));
        }
        Ok((condition_interface, body_interface))
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

    /// Splits a pure `while` loop whose known state subset is *closed* (i.e., every known state element's next-state
    /// value and the predicate fold from the known state alone) into a *known* loop bound in the enclosing known-side
    /// context and the *residual* loop kept whole, returning [`None`] when the split does not apply.
    ///
    /// The known state subset is found by a monotonic fixed point like the known-ness split of [`ScanOperation`]: a
    /// state element stays known iff its init is known and the body computes its next value from known state alone,
    /// with each round partitioning the body through a **fresh** staging context (via
    /// [`PartialEvaluationDriver::partition_program`]) so no probe work leaks into the caller's context. Unlike the
    /// loop-*invariance* rewrite, known-ness needs neither constant resolution nor value equality, so symbolic known
    /// inits (tracers into a live outer trace) participate fully, which is what makes the split fire under
    /// [`Program::linearize`]. After convergence the split additionally requires the *predicate* to fold from the known
    /// state alone; only then does the known loop run the original iteration count, since its continuation decision is
    /// byte-for-byte the original one over state it computes itself.
    ///
    /// The known loop is the projection of the original loop onto the known subset: its body maps the known state
    /// elements to their known next values and its condition is the known projection of the original condition. It is
    /// bound whole into the enclosing known-side context over the original known inits (executing under an eager
    /// context and staging into the outer program under a staging one), and its outputs are the known state elements'
    /// final values. The residual loop is the **original loop unchanged**: its unknown outputs are the unknown state
    /// elements' finals, and any known per-iteration values the unknown side reads are *recomputed inside the loop*
    /// rather than streamed as residual edges, because a loop with a data-dependent iteration count has no statically
    /// shaped residual stream. For an eligible fused forward-mode loop, this duplication lets the known side recover
    /// the primal outputs while the tangent program keeps the complete loop. The residual loop's known-state outputs
    /// are left dead. This does not support arbitrary unbounded-loop linearization: the split requires pure regions and
    /// a predicate that depends only on the closed known state.
    ///
    /// The split does not apply (and the caller residualizes unchanged) when the converged known subset is empty or
    /// complete (an all-known loop folds whole through the default rule), or when the predicate reads unknown state.
    /// Callers must ensure both regions are pure: the known loop re-runs the known part of every iteration and the
    /// residual loop runs all of it again, so an effectful loop would observe its effects twice.
    fn split_by_closed_knownness<
        V: Value<Type = T>,
        O: Operation<Type = T> + From<WhileOperation<T>>,
        C: Context<Type = T, Constant = V, Operation = O>,
        D: PartialEvaluationDriver<C>,
    >(
        &self,
        context: &PartialEvaluationContext<C>,
        condition: RegionRef<'_, V, O>,
        body: RegionRef<'_, V, O>,
        inputs: &[PartialEvaluationValue<C::Value>],
        driver: &D,
    ) -> Result<Option<Vec<PartialEvaluationValue<C::Value>>>, ProgramError>
    where
        TracingContext<V, O>: StagingContext<Type = T, Constant = V, Operation = O>,
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
                    state_known[index]
                        && matches!(partition.outputs().get(index), Some(PartialEvaluationOutput::Known(_)))
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
        // iteration count and the split does not apply.
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
        // and its outputs are their known next values (the partition's trailing feeder-edge outputs are dropped, since
        // the residual loop recomputes them).
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
                PartialEvaluationOutput::Known(output) => {
                    known_program_outputs.get(*output).copied().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "`{WHILE_OPERATION_NAME}` body partition output {index} references missing known-program \
                             output {output}",
                        ))
                    })
                }
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
                "`{WHILE_OPERATION_NAME}` condition partition references missing known-program output \
                 {predicate_output}",
            ))
        })?;
        let known_condition = known_condition_builder.build::<Vec<V>, Vec<V>>(
            vec![predicate_atom],
            vec![Placeholder; known_count],
            vec![Placeholder; 1],
        )?;

        // Bind the known loop into the enclosing known-side context over the original known inits, and emit the
        // original loop unchanged into the residual program for the unknown state elements' finals.
        let known_while = WhileOperation::new().with_iteration_bound(self.iteration_bound())?;
        let known_inputs = known_state_indices.iter().map(|&index| inputs[index].clone()).collect::<Vec<_>>();
        let known_outputs = context.fold_or_residualize(
            O::from(known_while),
            vec![known_condition, known_body],
            known_inputs.as_slice(),
        )?;
        check_count!("output", known_outputs, known_count, ProgramError);
        let residual_outputs =
            context.residualize(O::from(*self), vec![condition.to_program(), body.to_program()], inputs)?;
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
        Self::validate_interfaces(region_interfaces)?;
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
        // is loop-invariant, so every iteration carries values of that extent. An identity that a first-class dimension
        // carry defines is defined by that carry's body input, and the body must return the state signature, so
        // identity validation rejects a body that returns a different dimension for the carry. The identity
        // nevertheless stays symbolic in the outputs, because the refinement only fixes identities that no state
        // element defines. Metadata such as shardings is not propagated, because the body may change it.
        let (_, body_interface) = Self::validate_interfaces(region_interfaces)?;
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

impl<
    T: WhileType + From<P::Referent> + From<ReferenceType<P::Referent>>,
    C: Context<Type = T, Operation: From<WhileOperation<T>> + From<ConditionOperation<T>>>,
    P: ReferenceDischargePolicy<C>,
> ReferenceDischargeableOperation<C, P> for WhileOperation<T>
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        // A `while`'s boundaries are all one state signature: its own inputs and outputs, its condition's inputs, and
        // its body's inputs and outputs carry the same carries at the same positions. Threading discharged state
        // therefore appends the same carries to every one of them, and the loop deliberately applies no read-only
        // pruning: an allocation the loop merely reads still occupies a carry position, because dropping it would leave
        // the body's boundary disagreeing with the condition's. The one asymmetry is forced by the operation's own
        // contract: the condition returns only a predicate, so it receives the entering state and, as long as it only
        // reads, publishes none.
        //
        // A condition that mutates a discharged (threaded) allocation has nowhere to publish its updates through that
        // boundary, so the rule rotates such a loop into do-while form: the original condition is discharged once into
        // the parent, producing the initial predicate and applying its effects exactly once before the loop; the
        // predicate becomes a trailing carry; the rebuilt body runs the original body and then the original condition,
        // returning the states the condition published in place of the body's and the fresh predicate; and the rebuilt
        // condition merely projects the predicate carry. The condition is thereby emitted twice, and the loop evaluates
        // it exactly as often and in the same order as the source did. Because rotation fixes the order to body then
        // condition, a root written by both regions is accepted as well. A bounded loop rotates with its bound lowered
        // by one, because the bound truncates the loop before the condition's next evaluation, whereas every rotated
        // iteration evaluates the condition after the body. The last body application then runs after the loop under
        // the final predicate, through a `condition` that otherwise forwards the state, and state that the body passes
        // through bypasses that `condition`. A condition that writes only *preserved* references needs no rotation,
        // because those writes remain explicit effects with no immutable successor state.
        //
        // A carry that partial reference discharge *preserved* keeps its declared position on every one of those
        // boundaries and widens nothing at all. A preserved allocation reached only through an inherited capture gains
        // a reference-typed carry so the rebuilt regions can bind that capture; in either case it enters as the
        // reference the caller already holds, its accesses replay as the operations the source performed, and it
        // publishes no state successor.
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

        // A read-only condition publishes nothing (it returns only its predicate), so its declared output allocations
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
                Some(initial[0].try_as_value(&format!("the `{WHILE_OPERATION_NAME}` condition predicate"))?.clone())
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
                    // The tail selects between the rebuilt body and the unchanged state under the final predicate. A
                    // position that the rebuilt body forwards from its own input (e.g., a first-class dimension carry)
                    // is the same on both sides, so it bypasses the `condition` and takes the loop's output directly,
                    // as the `Condition` frontend forwards unchanged outputs. Only the kept positions go through the
                    // branches, whose type rules are stricter than the loop's for forwarded state (e.g., a `condition`
                    // rejects a dimension output under a varying predicate, which a `while` carry accepts). With every
                    // position forwarded, the `condition` is bound only for the effects (e.g., a preserved-reference
                    // write) or deferred work of the body, which must still run under the final predicate.
                    let state_types = body_program.input_types();
                    let state_count = state_types.len();
                    let forwarded = passed_through_carries(body_program.entry_region_ref(), 0..state_count);
                    let kept = (0..state_count).filter(|&position| !forwarded[position]).collect::<Vec<_>>();
                    let body_effects = body_program.effects();
                    if !kept.is_empty() || !body_effects.classes().is_empty() || body_effects.has_deferred_work() {
                        let mut unchanged = ProgramBuilder::<C::Constant, C::Operation>::new();
                        let state =
                            state_types.into_iter().map(|r#type| unchanged.add_input(r#type)).collect::<Vec<_>>();
                        let unchanged = unchanged.build::<Vec<C::Constant>, Vec<C::Constant>>(
                            state,
                            vec![Placeholder; state_count],
                            vec![Placeholder; state_count],
                        )?;
                        let branches = match kept.len() == state_count {
                            true => vec![body_program, unchanged],
                            false => vec![body_program.with_outputs(&kept)?, unchanged.with_outputs(&kept)?],
                        };
                        let mut conditional_inputs = Vec::with_capacity(outputs.len() + 1);
                        conditional_inputs.push(predicate);
                        conditional_inputs.extend(outputs.iter().cloned());
                        let conditional_outputs = context.parent().bind(
                            ConditionOperation::<T>::new(),
                            branches,
                            conditional_inputs.as_slice(),
                        )?;
                        check_count!("output", conditional_outputs, kept.len(), ProgramError);
                        for (position, output) in kept.into_iter().zip(conditional_outputs) {
                            outputs[position] = output;
                        }
                    }
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

impl<C: Domain<Type: WhileType, Value: WhilePredicate>> InterpretableOperation<C> for WhileOperation<C::Type> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // The eager rule validates the application before the first condition (refer to `validate_application`), so
        // that a malformed application is reported as such even when no iteration runs. It then evaluates the condition
        // and masks each body update with its predicate. The `shard_map` emulator uses the same rule over one value per
        // device, whose `WhilePredicate` implementation requires device agreement and applies each device's own mask.
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

impl<C: Context<Type: WhileType, Operation: From<WhileOperation<C::Type>>>> PartiallyEvaluatableOperation<C>
    for WhileOperation<C::Type>
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // A while's inputs are the initial loop state and its outputs are the final loop state (the same arity).
        // Partial evaluation folds the known value of every *loop-invariant-known* state element into both nested
        // programs: a state element is loop-invariant-known iff the body passes it through unchanged (its next-state
        // output is its own state input), its init input is `PartialValue::Known`, and its type is not a reference and
        // mentions no type identity. Such an element holds its init value on every iteration, so binding it to that
        // constant inside the condition and body is sound and collapses every subcomputation that depended only on it.
        // Invariance is structural rather than decided by comparing the body's next-state values with the init values,
        // because value equality cannot distinguish values that the body changes without changing their equality class
        // (e.g., a body that negates a state element of `0.0` returns `-0.0`, which compares equal to `0.0`). A folded
        // element is rebuilt as a program constant, which excludes two kinds of state even when the body only passes
        // them through: a reference carry must keep its allocation flowing through the loop boundary, and a type that
        // mentions an identity cannot be rebuilt as a constant. A type that defines an identity (e.g., a first-class
        // dimension) is already defined by the state input that the rebuilt condition and body keep, so a constant of
        // that type would define the identity twice, and a type that refers to an identity (e.g., `f32[extent]`) would
        // be replaced by the concrete static type of the constant, which no longer matches the loop's state signature.
        //
        // Both the body and the condition are then partially evaluated with the invariant-known state knowledge through
        // the partial-evaluation driver's split requests. The condition reads the state too, so folding an invariant
        // element can shrink it as well.
        //
        // The residual while keeps the *same* state set and therefore the same output arity as the original operation.
        // A loop-invariant-known element is not dropped; instead its body next-state output is rebuilt as the constant
        // init value and its uses fold away inside both programs. Because the condition and body run on the same
        // loop-carried state each iteration, both residual programs are rebuilt over the loop's full state signature
        // (in state order): each surviving unknown state element feeds the matching state input, and every known
        // residual a program closed over is rebuilt as an inline residual-program constant (so the residual while needs
        // no captures). The `iteration_bound` is preserved. The rewrite is emitted over the guarded while inputs, which
        // an entry guard derives from the original inputs first (see below).
        //
        // If no state element is loop-invariant-known, or a probe fails or folds to a value that is not a constant, the
        // rule attempts the *closed-knownness split* before residualizing unchanged (provided that some input is
        // known): when a known state subset's next values and the predicate fold from known state alone, the loop
        // separates into a known loop bound on the known side and the residual loop kept whole (refer to
        // `WhileOperation::split_by_closed_knownness`). This is the split that makes `Program::linearize` total over
        // the fused doubled-state loops staged by the unbounded `while` forward-mode rule.

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
            return context.fold_or_residualize(*self, driver.region_programs(), inputs);
        }

        let state_types = body.input_types();
        let state_count = state_types.len();

        // The invariance probes below fold the condition and body through the *live* known-side context, and the
        // closed-knownness split's known loop re-runs the known part of every iteration. For an effectful loop the
        // probes would execute (eager) or stage (staging) the loop's effects once more and the split would run them
        // twice, so effectful loops skip both and residualize unchanged (see the effect placement contract on
        // `PartialEvaluationContext::fold_or_residualize`). Every reference operation is `OrderedState`, so a loop
        // touching references is never pure and no probe below can execute a reference operation or change the active
        // context's effect-ordering state. A reference carry that neither region accesses can still reach the pure
        // rewrites below: the invariance fold never folds it (it must keep its allocation flowing through the loop
        // boundary), while the closed-knownness split may carry a known one through the known loop, which only forwards
        // it. Deferred work skips both for the same reason as effects: every probe would residualize its obligation
        // again.
        if !condition.effects().classes().is_empty()
            || !body.effects().classes().is_empty()
            || condition.effects().has_deferred_work()
            || body.effects().has_deferred_work()
        {
            return context.fold_or_residualize(*self, driver.region_programs(), inputs);
        }

        // The rewrites below replace the loop's boundary: the closed-knownness split binds a known loop over the known
        // inputs alone and leaves the residual loop dead when only known finals are used, and the fold rebuilds both
        // regions. Neither would then check a relationship that the entry establishes between a known and an unknown
        // input (e.g., a known `f64[3]` next to an unknown `f64[n]`, both entering at `n`), so the relationships that
        // the input types leave unproven are checked by an entry guard first, which survives simplification. The fold,
        // the split, and every residualized loop use the guarded inputs, whose types prove them.
        let inputs = context.guard_inputs(&state_types, inputs.to_vec(), false)?;
        let inputs = inputs.as_slice();

        // Every pure fallback below first attempts the closed-knownness split (a known state subset whose next values
        // and predicate fold from known state alone separates into a known loop bound on the known side and the
        // residual loop kept whole) and only residualizes the loop unchanged when the split does not apply.
        let split_or_residualize = |context: &PartialEvaluationContext<C>| {
            if inputs.iter().any(PartialEvaluationValue::is_known)
                && let Some(outputs) = self.split_by_closed_knownness(context, condition, body, inputs, driver)?
            {
                return Ok(outputs);
            }
            context.fold_or_residualize(*self, driver.region_programs(), inputs)
        };

        // A state element can only fold if its type is not a reference and mentions no type identity (see the overview
        // above), the body passes it through unchanged, and its init input is known *and* resolves to a constant in the
        // known-side context: the folded value must be embeddable as a rebuilt-program constant, and skipping symbolic
        // knowns also keeps the probes from folding symbolic known work into a live staging context.
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

        // A probe failure falls back through `split_or_residualize`: the body may never run at runtime (the condition
        // can be false on entry), so a known-side fold that rejects its inputs must not fail partial evaluation,
        // because the branch's work, and its error if the loop is ever entered, stays behind the condition. Both
        // programs are pure here (the effects gate above), so partially completed probe folds are safe to discard, and
        // the split stays error-consistent: its partitions stage through fresh contexts without executing known work,
        // and its known loop replays the loop's exact runtime semantics (running nothing when the condition is false on
        // entry).
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
        // regions. Otherwise the residual while's inputs are exactly the guarded while inputs: each state element's
        // init value (now a known residual for the folded elements) in state order.
        if let Some(outputs) = while_operation.split_by_closed_knownness(
            context,
            residual_condition.entry_region_ref(),
            residual_body.entry_region_ref(),
            inputs,
            driver,
        )? {
            return Ok(outputs);
        }
        context.fold_or_residualize(while_operation, vec![residual_condition, residual_body], inputs)
    }
}

impl<C: Context<Type = ArrayType, Operation: From<WhileOperation<ArrayType>>>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for WhileOperation<ArrayType>
{
    #[inline]
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        context.batch_while(self, driver, inputs)
    }
}

impl<C: Context<Type = ArrayIrType, Operation: From<WhileOperation<ArrayIrType>>>>
    BatchableOperation<C, ArrayIrBatchingPolicy> for WhileOperation<ArrayIrType>
{
    #[inline]
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        context.batch_while(self, driver, inputs)
    }
}

impl<
    T: DifferentiableType + WhileType,
    C: Context<
            Type = T,
            Operation: ResidualZeroProvider<T, Operation = C::Operation>
                           + From<ConditionOperation<T>>
                           + From<WhileOperation<T>>
                           + From<ScanOperation<T>>
                           + WhileResidualStackOperation<T>,
        > + Zero<C::Value>,
> DifferentiableOperation<C> for WhileOperation<T>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = T, Constant = C::Constant, Operation = C::Operation>,
    C::Value: Concretizable<bool>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        // An eager context runs the loop directly at the concrete duals (refer to `jvp_while_eagerly`), so eager
        // forward mode is total over data-dependent `while` loops with no iteration bound. Staging contexts stage the
        // shared array-backed rule (refer to `WhileJvp::bind`), and the eager rule resumes with it from the current
        // duals once the predicate stops concretizing to one scalar Boolean (e.g., a batched predicate, which has no
        // single continuation decision): bounded loops stage the reverse-capable hybrid of an augmented primal `while`
        // and a masked tangent `scan`, after rewriting a batched predicate into its masked form, while unbounded loops
        // stage the forward-only fused doubled-state loop. Both type families share that rule; they differ only in
        // their array-backed storage, which `WhileResidualStackOperation` describes.

        // The rule requests all nested-computation work through its driver (region 0 is the condition and region 1 the
        // body), which keeps its bounds free of the operation family's own semantic traits.
        let input_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
        self.validate_application(&input_types, driver.region(0)?, driver.region(1)?)?;
        let condition = driver.region(0)?.to_program();
        let body = driver.region(1)?.to_program();
        if context.primal().is_eager() {
            return jvp_while_eagerly(self, &condition, &body, context, driver, inputs);
        }
        WhileJvp::new(driver, self, inputs)?.bind(&condition, &body, context, driver, inputs)
    }
}

impl<T: WhileType, V: Value<Type = T>, O: Operation<Type = T>> TransposableOperation<V, O> for WhileOperation<T>
where
    TracingContext<V, O>: StagingContext<Type = T, Constant = V, Operation = O>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        _accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        // This rule rejects the transposition of a staged linear `while` loop. The forward-mode rule stages one for an
        // unbounded loop (the doubled-state linear loop, which recomputes primal state *forward* through the
        // iterations, so transposing it would have to run that recomputation backwards, which a while loop cannot
        // express), and a linear program handed to `Program::transpose` can contain one directly, whether or not it is
        // bounded. Two paths avoid this rule: eager domains execute the loop directly over concrete duals and record a
        // straight-line pushforward during linearization, and the forward-mode rule never stages a linear `while` for a
        // bounded loop (`WhileOperation::with_iteration_bound`), because its tangent side is a masked linear scan whose
        // transpose is total.

        // The loop is validated as every rule validates it before the unsupported transposition is reported.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.validate_application(&input_types, driver.region(0)?, driver.region(1)?)?;
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{WHILE_OPERATION_NAME}` does not support transposition (reverse-mode differentiation through staged \
                 linear `{WHILE_OPERATION_NAME}` loops is not supported; eager differentiation executes concrete \
                 duals, and the forward-mode rule of loops built with `with_iteration_bound` stages a transposable \
                 masked scan)",
            ),
        }
        .into())
    }
}

impl From<WhileOperation<ArrayType>> for WhileOperation<ArrayIrType> {
    #[inline]
    fn from(operation: WhileOperation<ArrayType>) -> Self {
        Self { iteration_bound: operation.iteration_bound, marker: PhantomData }
    }
}

/// Operation-family constructors that the staged forward-mode rule of [`WhileOperation`] uses to store the
/// per-iteration residuals of a bounded loop in stacks and to make the per-item masking of a loop with a batched
/// predicate explicit.
///
/// The functions return operations rather than binding values, so the same contract works in the parent differentiation
/// context and in the nested tracing contexts used to build the augmented primal body and the masked tangent scan body.
/// The scalar and dimension gateway operations come from the [`ResidualStackOperation`] supertrait.
///
/// Custom backends implement this trait on their operation family to reuse the shared [`DifferentiableOperation`] rule
/// for [`WhileOperation`], whose bounds require it even though eager differentiation runs the loop directly and never
/// uses these constructors. Each constructor returns a region-free operation of the family with the semantics of the
/// named array operation over the family's array members, except that
/// [`residual_stack_vary`](Self::residual_stack_vary) may return an error for a family that cannot stage
/// manual-variation transitions.
///
/// Operation families over [`ArrayType`] that convert from the zero, one, addition, broadcast, dynamic-update-slice,
/// select, and reduction operations and provide [`ParallelVaryOperation`], and array IR families over [`ArrayIrType`]
/// whose array projection provides the same operations, already receive blanket implementations. An operation family
/// covered by a blanket implementation cannot also provide an overlapping implementation of its own.
pub trait WhileResidualStackOperation<T: Type>: ResidualStackOperation<T> {
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
    /// type `input_type`, so that it can be combined with stacks or state that vary over that axis.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when the operation family cannot stage manual-variation transitions or cannot
    /// provide the transition for `input_type`.
    fn residual_stack_vary(axis_name: &str, input_type: &ArrayType) -> Result<Self, ProgramError>;
}

impl<O: Operation<Type = ArrayType>> WhileResidualStackOperation<ArrayType> for O
where
    O: From<ZeroOperation<ArrayType>>
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

impl<O: ResidualStackOperation<ArrayIrType>> WhileResidualStackOperation<ArrayIrType> for O
where
    O: From<ZeroOperation<ArrayType>>
        + From<OneOperation<ArrayType>>
        + OperationProjection<
            ArrayType,
            Projected: From<AddOperation<ArrayType>>
                           + From<BroadcastOperation>
                           + From<DynamicUpdateSliceOperation>
                           + From<SelectOperation<ArrayType>>
                           + From<ReduceOperation>,
        >,
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
        let operation: O::Projected = BroadcastOperation::new(output_type, output_axes).into();
        Self::from(operation)
    }

    #[inline]
    fn residual_stack_update() -> Self {
        let operation: O::Projected = DynamicUpdateSliceOperation::new().into();
        Self::from(operation)
    }

    #[inline]
    fn residual_stack_add() -> Self {
        let operation: O::Projected = AddOperation::<ArrayType>::new().into();
        Self::from(operation)
    }

    #[inline]
    fn residual_stack_select() -> Self {
        let operation: O::Projected = SelectOperation::<ArrayType>::new().into();
        Self::from(operation)
    }

    #[inline]
    fn mask_reduce_any(axes: Vec<usize>) -> Self {
        let operation: O::Projected = ReduceOperation::new(axes, ReductionKind::Any).into();
        Self::from(operation)
    }

    #[inline]
    fn residual_stack_vary(axis_name: &str, input_type: &ArrayType) -> Result<Self, ProgramError> {
        Ok(Self::from(O::Projected::provide(ParallelVaryOperation::new(axis_name.to_string()), &[input_type])?))
    }
}

/// Configuration of [`While::r#while`](While::while) and [`r#while()`](fn@while). By default, the loop has no iteration
/// bound and runs for as long as its condition holds. No option carries a value, so, unlike
/// [`ScanOptions`](super::scan::ScanOptions), these options are not generic over a value type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhileOptions {
    /// Optional semantic iteration bound of the loop.
    iteration_bound: Option<usize>,
}

impl WhileOptions {
    /// Creates a new [`WhileOptions`] instance with no iteration bound.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the semantic iteration bound of the loop, replacing any earlier bound; [`None`] restores the default
    /// unbounded loop. A bounded loop runs at most `bound` iterations by definition, truncating even while its
    /// condition still holds, and supports staged reverse-mode differentiation. Refer to
    /// [`WhileOperation::with_iteration_bound`] for more information.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError::Type`] when `bound` is zero or exceeds the maximum supported extent.
    pub fn with_iteration_bound<B: Into<Option<usize>>>(mut self, bound: B) -> Result<Self, ProgramError> {
        // Reuse the operation's validation rather than define another bound contract for the frontend. The bound does
        // not depend on the type universe, so any universe validates it.
        self.iteration_bound = WhileOperation::<ArrayType>::new().with_iteration_bound(bound)?.iteration_bound();
        Ok(self)
    }

    /// Returns the semantic iteration bound of the loop, if any.
    #[inline]
    pub fn iteration_bound(&self) -> Option<usize> {
        self.iteration_bound
    }
}

impl Default for WhileOptions {
    #[inline]
    fn default() -> Self {
        Self { iteration_bound: None }
    }
}

/// Context construction capability for a structured [`WhileOperation`], the Ryft analogue of JAX's
/// [`lax.while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html). The loop is bound in this
/// explicit context, which need not be the domain of the state values; the global [`r#while()`](fn@while) function
/// instead recovers its context from the first state value.
///
/// The condition receives the loop state and returns one Boolean predicate; the body receives the same state and
/// returns the next state. The state may be a [`Parameterized`] structure with static fields; the body must return the
/// same structure and static fields, and its leaf types must remain valid loop state. A batched (prefix-shaped)
/// predicate is accepted and gives the masked semantics of [`WhileOperation`]: the loop continues while any item's
/// predicate holds, and items whose predicate is false keep their state. A reference leaf enters the loop whole, and
/// the body must return the reference it received.
///
/// Both functions are traced, not called once per runtime iteration. They may be retraced while the manual variation of
/// the state reaches a fixed point, so they must describe the same computation on every call. Inside a manual region
/// (e.g., the body of a [`ShardMapOperation`](crate::ShardMapOperation)), devices whose predicates differ run different
/// numbers of iterations, so every array leaf of the state is widened to vary over every manual axis that the predicate
/// varies over, in addition to every axis that the body makes it vary over. Reference leaves are not widened: the
/// referent of a reference leaf must already vary over those predicate axes, or the loop is rejected. The widening of
/// the initial state is applied before the loop, including when the loop runs no iterations, and its transpose
/// preserves the corresponding reductions.
/// Nonconstant values used by either function must be passed through the state; constants may be closed over.
pub trait While: Context {
    /// Traces `condition_fn` and `body_fn` and binds one [`WhileOperation`] in this context, returning its final state.
    ///
    /// # Parameters
    ///
    ///   - `condition_fn`: Traced function `state -> predicate`, returning one Boolean predicate.
    ///   - `body_fn`: Traced function `state -> next_state`.
    ///   - `state`: Initial loop-carried state.
    ///   - `options`: Iteration bound of the loop.
    ///
    /// # Errors
    ///
    /// Returns errors for invalid predicates, state structure/static-field changes, state signature changes, reference
    /// identities, or nonconvergent manual variation, and errors from tracing, alignment, or binding.
    fn r#while<
        State: Parameterized<Self::Value, Family: ParameterizedFamily<Tracer<NestedTracingContext<Self>>>>,
        ConditionFn: FnMut(
            State::To<Tracer<NestedTracingContext<Self>>>,
        ) -> Result<Tracer<NestedTracingContext<Self>>, ProgramError>,
        BodyFn: FnMut(
            State::To<Tracer<NestedTracingContext<Self>>>,
        ) -> Result<State::To<Tracer<NestedTracingContext<Self>>>, ProgramError>,
    >(
        &self,
        condition_fn: ConditionFn,
        body_fn: BodyFn,
        state: State,
        options: WhileOptions,
    ) -> Result<State, ProgramError>
    where
        State::ParameterStructure: PartialEq;
}

impl<C: Context<Type: WhileType, Operation: From<WhileOperation<C::Type>>>> While for C
where
    Tracer<NestedTracingContext<C>>: ManualVariationAlignment,
{
    fn r#while<
        State: Parameterized<Self::Value, Family: ParameterizedFamily<Tracer<NestedTracingContext<Self>>>>,
        ConditionFn: FnMut(
            State::To<Tracer<NestedTracingContext<Self>>>,
        ) -> Result<Tracer<NestedTracingContext<Self>>, ProgramError>,
        BodyFn: FnMut(
            State::To<Tracer<NestedTracingContext<Self>>>,
        ) -> Result<State::To<Tracer<NestedTracingContext<Self>>>, ProgramError>,
    >(
        &self,
        mut condition_fn: ConditionFn,
        mut body_fn: BodyFn,
        state: State,
        options: WhileOptions,
    ) -> Result<State, ProgramError>
    where
        State::ParameterStructure: PartialEq,
    {
        // Preserve the state structure and its static fields separately from the runtime leaves that become the loop
        // state. The condition and the body rebuild their structured arguments from the same structure.
        let state_structure = state.parameter_structure();
        let state_values = state.into_parameters().collect::<Vec<_>>();
        let state_count = state_values.len();
        let initial_state_types = state_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();

        // Trace both functions over the current state types until those types reach a manual-variation fixed point.
        // Both are traced on every pass so that their input signatures agree with the final state types. Variation
        // only grows, and each pass that does not converge adds at least one manual axis to one state element, which
        // bounds the number of passes.
        let mut state_types = initial_state_types.clone();
        let mut remaining_widenings = None;
        let (condition_program, body_program) = loop {
            let (_, condition_program) = NestedTracingContext::trace(
                self.clone(),
                |values| {
                    let state = State::To::<Tracer<NestedTracingContext<Self>>>::from_parameters(
                        state_structure.clone(),
                        values,
                    )?;
                    Ok(vec![condition_fn(state)?])
                },
                state_types.clone(),
            )?;

            // Devices whose predicates differ run different numbers of iterations, so every state element must vary
            // over each manual axis that the predicate varies over, as the operation requires.
            let condition_output_types = condition_program.output_types();
            let predicate_sharding = condition_output_types
                .first()
                .and_then(|r#type| r#type.as_array_type().ok())
                .and_then(ArrayType::sharding)
                .cloned();

            // Align every state element with its next value and with the predicate. The incoming value's aligned type
            // is the state type of the next pass, and the returned value's alignment is staged at the end of the body.
            let mut next_state_types = Vec::with_capacity(state_count);
            let (_, body_program) = NestedTracingContext::trace(
                self.clone(),
                |values| {
                    let state = State::To::<Tracer<NestedTracingContext<Self>>>::from_parameters(
                        state_structure.clone(),
                        values.clone(),
                    )?;
                    let next_state = body_fn(state)?;
                    if next_state.parameter_structure() != state_structure {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "`{WHILE_OPERATION_NAME}` body must preserve the state parameter structure and static \
                                 fields",
                            ),
                        });
                    }
                    let mut next_values = Vec::with_capacity(state_count);
                    for (incoming, returned) in values.into_iter().zip(next_state.into_parameters()) {
                        let mut pair = Tracer::align_manual_variation(&[incoming, returned])?;
                        if let Some(sharding) = &predicate_sharding {
                            pair = Tracer::align_manual_variation_to(&pair, sharding)?;
                        }
                        next_state_types.push(pair[0].r#type().into_owned());
                        next_values.push(pair.pop().unwrap());
                    }
                    Ok(next_values)
                },
                state_types.clone(),
            )?;
            let capacity = remaining_widenings.get_or_insert_with(|| {
                let axes = state_types
                    .iter()
                    .chain(condition_output_types.iter())
                    .chain(body_program.output_types().iter())
                    .filter_map(|r#type| r#type.as_array_type().ok().and_then(ArrayType::sharding))
                    .map(|sharding| sharding.mesh().axes().len())
                    .max()
                    .unwrap_or(0);
                state_count.saturating_mul(axes.max(1))
            });
            if next_state_types == state_types {
                break (condition_program, body_program);
            }
            if *capacity == 0 {
                return Err(ProgramError::InvalidArgument {
                    message: format!("`{WHILE_OPERATION_NAME}` state manual variation did not reach a fixed point"),
                });
            }
            *capacity -= 1;
            state_types = next_state_types;
        };

        // Which references the body returns is only visible in its program, which binding does not inspect when it
        // stages the loop, so validate the complete application here.
        let operation = WhileOperation::new().with_iteration_bound(options.iteration_bound())?;
        operation.validate_application(
            &state_types,
            condition_program.entry_region_ref(),
            body_program.entry_region_ref(),
        )?;

        // Replay entry alignment through this explicit context, not through the values' possibly different domains.
        // This emits the actual transitions only after convergence and leaves transpose reductions outside the loop.
        let mut inputs = state_values;
        if initial_state_types != state_types {
            let (_, alignment) = NestedTracingContext::trace(
                self.clone(),
                |values| {
                    values
                        .into_iter()
                        .zip(&state_types)
                        .map(|(value, r#type)| match r#type.as_array_type().ok().and_then(ArrayType::sharding) {
                            Some(sharding) => Ok(Tracer::align_manual_variation_to(&[value], sharding)?.remove(0)),
                            None => Ok(value),
                        })
                        .collect::<Result<Vec<_>, ProgramError>>()
                },
                initial_state_types,
            )?;
            inputs = alignment.interpret_in_context(self, inputs)?;
        }

        let outputs = self.bind(operation, vec![condition_program, body_program], &inputs)?;
        check_count!("output", outputs, state_count, ProgramError);
        Ok(State::from_parameters(state_structure, outputs)?)
    }
}

/// Runs [`While::r#while`](While::while) in the execution domain recovered from the first value of the initial `state`.
/// The condition can only compute its predicate from state values, so the state must contain at least one value
/// alongside any static fields.
///
/// ```
/// # use ryft_core::{Array, Compare, Context, ProgramError, WhileOptions, r#while};
/// # fn main() -> Result<(), ProgramError> {
/// let (count, value) = r#while(
///     |(count, _)| count.less_than(&count.context().lift(Array::scalar(3i64)?)?),
///     |(count, value)| {
///         let one = count.context().lift(Array::scalar(1i64)?)?;
///         let two = value.context().lift(Array::scalar(2f32)?)?;
///         Ok((count + one, value * two))
///     },
///     (Array::scalar(0i64)?, Array::scalar(1f32)?),
///     WhileOptions::new(),
/// )?;
/// assert_eq!(count, Array::scalar(3i64)?);
/// assert_eq!(value, Array::scalar(8f32)?);
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`ProgramError::InvalidArgument`] when the state contains no value that supplies a domain, and the errors of
/// [`While::r#while`](While::while).
#[inline]
pub fn r#while<
    V: Value<Domain: While>,
    State: Parameterized<V, Family: ParameterizedFamily<Tracer<NestedTracingContext<V::Domain>>>>,
    ConditionFn: FnMut(
        State::To<Tracer<NestedTracingContext<V::Domain>>>,
    ) -> Result<Tracer<NestedTracingContext<V::Domain>>, ProgramError>,
    BodyFn: FnMut(
        State::To<Tracer<NestedTracingContext<V::Domain>>>,
    ) -> Result<State::To<Tracer<NestedTracingContext<V::Domain>>>, ProgramError>,
>(
    condition_fn: ConditionFn,
    body_fn: BodyFn,
    state: State,
    options: WhileOptions,
) -> Result<State, ProgramError>
where
    State::ParameterStructure: PartialEq,
{
    let context = state.parameters().next().map(Value::domain).ok_or_else(|| ProgramError::InvalidArgument {
        message: format!("`{WHILE_OPERATION_NAME}` requires a state value to recover its context from"),
    })?;
    context.r#while(condition_fn, body_fn, state, options)
}

impl<C: Context<Type: WhileType, Operation: From<WhileOperation<C::Type>>>, P: BatchingPolicy<C>>
    BatchingContext<C, P>
{
    /// Batches an application of `operation` to `inputs`, binding one loop over the batched condition and body in the
    /// parent context, where it is interpreted eagerly under an eager parent and staged into the enclosing trace under
    /// a staging parent. Every [`BatchingPolicy`] uses this structural path, with its own carriers and boundary
    /// bookkeeping inputs. A predicate that becomes batched turns the loop into a per-item masked loop over widened
    /// array state.
    fn batch_while<D: BatchingDriver<C, P>>(
        &self,
        operation: &WhileOperation<C::Type>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<Vec<P::Batch>, BatchingError> {
        // The rule builds batched loop *structure* and binds it into the parent context, whose relaxed-predicate
        // interpretation owns the per-item masked semantics under an eager parent:
        //
        //   1. Every batched array state input is realigned to batch axis `0` in the parent context, and the body is
        //      batched at the state batch axes, iterating the axes to a fixed point (refer to `stabilize_carry_axes`):
        //      a while loop's state types are loop-invariant, so a replicated state element whose update depends on a
        //      batched element *becomes* batched. Each pass instantiates the body outputs at the current state axes
        //      (mirroring JAX's `instantiate=carry_bat`), so the converged body is already aligned to the
        //      loop-invariant state layout, and widened parent inputs gain their batch axis through the driver.
        //      First-class dimension state stays replicated, because array IR batching admits no mapped dimension
        //      values, and reference state keeps the batch axis that its referent fixes.
        //   2. The condition is batched at the stabilized axes. When its predicate output stays *replicated*, one
        //      `WhileOperation` over the batched condition and body is bound into the parent directly, preserving any
        //      semantic `iteration_bound` (so bounded loops stay reverse-capable under `batch`). A replicated predicate
        //      that already masks a prefix of the array state keeps that prefix ahead of the new mapped carry axis, so
        //      its per-item selection remains valid.
        //   3. When the predicate output is *batched* (per-item termination), every array state element is widened to a
        //      batched element, the condition's predicate output is instantiated at axis `0` (re-batching the condition
        //      only when its natural predicate axis is not already `0`), and one `WhileOperation` is bound directly
        //      with that batched predicate (mirroring JAX's `_while_loop_batching_rule`). The predicate's `[axis_size]`
        //      shape is a prefix of every widened array state shape, so the bound loop satisfies the relaxed predicate
        //      contract and its consumers own the masked semantics: eager interpretation continues while any per-item
        //      predicate is true and freezes finished items, and the XLA lowering reduces the predicate with `or` and
        //      masks carry updates with a broadcast select. Dimension state rides through the loop as loop-invariant
        //      state that masking never touches (the relaxed `WhileType` contract documents that invariance
        //      requirement), and the rebuilt loop's type contract decides whether reference state may ride under a
        //      batched predicate. The iteration bound is preserved (batch items share masked iterations, so capping the
        //      loop matches per-item truncation exactly).
        //
        // The policy's bookkeeping inputs (refer to `BatchingPolicy::boundary_inputs`) lead the state of the bound
        // loop. The condition consumes them as loop state but returns only its predicate, so their forwarded outputs
        // are projected out of its boundary.

        // The rule requests all nested-computation work through its region access (region 0 is the condition and region
        // 1 the body), which keeps its bounds free of the operation family's own semantic traits.
        let condition = driver.region(0)?;
        let body = driver.region(1)?;
        let input_types = inputs.iter().map(|input| P::unbatched_type(input).into_owned()).collect::<Vec<_>>();
        operation.validate_application(&input_types, condition, body)?;
        let state_count = inputs.len();
        let mut state = canonicalize_carry_axes(self, driver, inputs)?;
        let mut state_axes = state.iter().map(P::batch_axis).collect::<Vec<_>>();
        let batched_body =
            stabilize_carry_axes(WHILE_OPERATION_NAME, self, driver, body, &mut state_axes, 0..state_count)?
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
            driver.batch_program(self, condition, state_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)?;
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
                    self,
                    body,
                    state_axes.as_slice(),
                    ProgramBatchingOutputAxesPolicy::AlignEachTo(state_axes.clone()),
                )?
                .into_parts()
                .0;
            batched_condition = driver.batch_program(
                self,
                condition,
                state_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?;
        }
        // The condition batched above used these same state axes, so a predicate that naturally landed on axis 0 keeps
        // that program instead of being replayed for an alignment that would stage nothing.
        let batched_condition = match batch_varying {
            true => self.align_batched_program_outputs(
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
                *value = driver.align_batch_axis(self, value.clone(), axis)?;
            }
        }
        let bookkeeping_inputs = P::boundary_inputs(self.axis_extent());
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
            self.parent().bind(*operation, vec![batched_condition, batched_body], packed_inputs.as_slice())?;
        check_count!("output", outputs, bookkeeping_count + state_count, ProgramError);
        outputs.drain(..bookkeeping_count);
        Ok(outputs
            .into_iter()
            .zip(state_axes)
            .map(|(output, axis)| P::batch(output, axis))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

/// Represents a staged forward-mode application of a [`WhileOperation`] whose state enters as the instruction input
/// duals. The state elements keep their positions across the body boundary, so every state element is a loop carry and
/// the body's region input mask is the live state mask. A state element starts live when its tangent is live
/// (structural zeros are not), and carry liveness is the monotone fixed point of [`TangentActivity::from_loop_body`],
/// so inactive elements, such as a pass-through loop limit, keep structural-zero tangents instead of threading
/// materialized zeros through the loop. [`Self::bind`] stages the regime that the loop's iteration bound and predicate
/// select.
struct WhileJvp<C: Context> {
    /// [`WhileOperation`] being differentiated, whose iteration bound selects between the bounded and the fused
    /// regimes.
    operation: WhileOperation<C::Type>,

    /// Canonical body [`TangentActivity`] after the carry fixed point, including its live state indices, fused program,
    /// and output tangent slots. Every state element is a carry, so its output tangent liveness is also the converged
    /// tangent liveness of the state.
    activity: TangentActivity<C>,
}

impl<C: Context<Type: DifferentiableType + WhileType> + Zero<C::Value>> WhileJvp<C>
where
    TracingContext<C::Constant, C::Operation>:
        StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    C::Value: Concretizable<bool>,
    C::Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>
        + From<ConditionOperation<C::Type>>
        + From<WhileOperation<C::Type>>
        + From<ScanOperation<C::Type>>
        + WhileResidualStackOperation<C::Type>,
{
    /// Creates a new [`WhileJvp`] by analyzing the forward-mode activity of an application of `operation` to the input
    /// duals `inputs` and deriving its fused body through `driver`, whose region `1` is the body.
    ///
    /// # Errors
    ///
    /// Returns an error when `inputs` does not match the body's state signature in length, and the errors of analyzing
    /// the body's tangent activity through the differentiation driver.
    fn new<D: DifferentiationDriver<C>>(
        driver: &D,
        operation: &WhileOperation<C::Type>,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Self, DifferentiationError> {
        let body = driver.region(1)?;
        let state_count = body.input_ids().len();
        check_count!("input", inputs, state_count, ProgramError);
        let input_tangent_liveness = inputs.iter().map(|input| !input.tangent().is_zero()).collect();
        let (_, activity) =
            TangentActivity::from_loop_body(driver, body, input_tangent_liveness, 0..state_count, 0..state_count)?;
        Ok(Self { operation: *operation, activity })
    }

    /// Stages the forward-mode application of this loop with the condition and body programs `condition` and `body` to
    /// the input duals `inputs`, and returns the output duals. This rule serves staging contexts, and eager contexts
    /// whose loop predicate does not concretize to one scalar Boolean (e.g., a batched per-item predicate, which has no
    /// single continuation decision). The loop selects one of four regimes:
    ///
    ///   - A loop with no live state tangent after the carry fixed point is bound as it is, with structural-zero
    ///     tangents, because no residual stack, mask, or doubled tangent state would ever be read.
    ///   - A loop without a semantic [`iteration_bound`](WhileOperation::with_iteration_bound) has no statically shaped
    ///     residual stack, so it stages one fused loop over `[primal_state..., tangent_state...]` whose continuation
    ///     decision reads the primal half (refer to [`Self::bind_fused`]). That loop is forward-total but not
    ///     transposable: reverse mode through a staged unbounded loop reports the transposition error, as JAX's
    ///     `lax.while_loop` does.
    ///   - A bounded loop with a batched (per-item) predicate is first rewritten into its scalar-predicate masked form
    ///     over the same state (refer to [`masked_while_programs`]), whose forward mode is this same rule.
    ///   - A bounded loop with a scalar predicate stages the reverse-capable hybrid of an augmented primal `while` that
    ///     stores every per-iteration residual and a validity mask in `[bound, …]` stacks, and one masked tangent
    ///     `scan` of length `bound` (refer to [`Self::bind_bounded`]). The residuals and the mask are plain primal
    ///     input edges: they leave the augmented primal `while` as ordinary stacked outputs and re-enter the tangent
    ///     scan as ordinary stacked inputs, so the enclosing partial-evaluation split discovers the residual input
    ///     edges structurally, as it does for the scan and condition rules.
    fn bind<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        self,
        condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        body: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let state_count = body.input_count();

        // A loop without a live state tangent after the carry fixed point is bound as it is, with structural-zero
        // tangents: no residual stack, mask, or doubled tangent state would ever be read.
        if self.activity.input_indices().is_empty() {
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            let outputs = context.primal().bind(
                self.operation,
                vec![condition.clone(), body.clone()],
                primal_inputs.as_slice(),
            )?;
            check_count!("output", outputs, state_count, ProgramError);
            return outputs.into_iter().map(DifferentiationDual::new_with_zero_tangent).collect();
        }

        // An unbounded loop has no statically shaped residual stacks for the bounded hybrid rule below, so it stages
        // the fused doubled-state loop instead (see `WhileJvp::bind_fused`). The fused path needs no batched-predicate
        // rewrite: the fused state doubles every prefix-shaped state element, so a per-item predicate still satisfies
        // the relaxed predicate contract over the doubled state.
        let Some(bound) = self.operation.iteration_bound() else {
            return self.bind_fused(condition, context, driver, inputs);
        };

        // A batched (per-item) predicate cannot thread the bounded rule's augmented differentiation state through the
        // predicate-prefix contract (the scalar iteration counter and the `[bound, ...]` residual stacks are not
        // predicate-prefixed), so the loop is first rewritten into its scalar-predicate masked form over the same state
        // (see `masked_while_programs`) and differentiated recursively, since the masked loop's forward mode is this
        // same rule. The masked condition reduces the predicate of the state it receives and the masked body recomputes
        // that predicate to select its updates, so the masked loop carries no mask: no initial mask is evaluated here
        // and its outputs are exactly the state. A carried mask would have to be recomputed on the state after every
        // body application, including the last one that the bound admits, where no other path (interpretation, which
        // stops at the bound before evaluating the condition, or the lowering) evaluates the condition.
        if C::Type::is_batched_while_predicate(&condition.output_types()[0]) {
            let (masked_condition, masked_body) = masked_while_programs(condition, body)?;
            // The masked loop's condition and body are freshly built region programs, so the recursive `jvp` is
            // requested through the instruction-scoped driver over them.
            let masked_while = C::Operation::from(self.operation);
            let outputs =
                driver.bind_jvp_operation(context, &masked_while, vec![masked_condition, masked_body], inputs)?;
            check_count!("output", outputs, state_count, ProgramError);
            return Ok(outputs);
        }

        self.bind_bounded(condition, context, driver, inputs, bound)
    }

    /// Stages the reverse-capable forward-mode rule of a bounded loop with a scalar predicate and at least one live
    /// state tangent, given its condition program `condition`, its input duals `inputs`, and its iteration bound
    /// `bound`, and returns the output duals. It stages an augmented primal `while` and one masked tangent `scan` of
    /// length `bound`.
    ///
    /// For a bound `B`, the rule linearizes the body through its instruction-scoped driver, giving a primal body
    /// `[state] -> [next_state, residuals...]` and a tangent body `[state_tangent, residuals...] ->
    /// [next_state_tangent]` together with the residual count. It then:
    ///
    ///   1. Builds the augmented primal `while` over the state `[original_state..., counter (i64 scalar),
    ///      residual_stacks (one zero-initialized [B, ...] stack per residual), mask_stack (a false-initialized Boolean
    ///      [B] stack)]` with [`build_bounded_while_programs`] from the residual-extended primal body, keeping
    ///      `iteration_bound = B` so the per-iteration writes can never clamp, and stages it over the input primals
    ///      followed by the staged counter and stack zeros. Its outputs split into the original state outputs (the
    ///      primal outputs), the dropped counter, the stacked residual outputs, and the mask stack.
    ///   2. Stages a length-`B` tangent [`ScanOperation`] over the live tangent carries, residual storage stacks, and
    ///      one Boolean validity stack. Each iteration reads one scalar validity item and conditionally evaluates the
    ///      pushforward or forwards the carried tangent unchanged. Dimension residuals that vary across iterations
    ///      cross the scan boundary through checked scalar-array gateways; forwarded invariant dimensions are threaded
    ///      directly as known carries. The condition guards both derivative arithmetic and gateway restoration, so
    ///      placeholder storage after early exit is never evaluated as a residual.
    ///   3. Pairs each primal output tracer with its tangent output tracer into a [`DifferentiationDual`].
    ///
    /// Reverse mode is total with no while-specific transpose code: the staged tangent scan reads its residual and
    /// validity stacks as known stacked inputs and its invariant residuals as known carries that its body passes
    /// through unchanged, so the ordinary scan transposition flips the scan direction and transposes the body. The
    /// per-iteration `condition` operation that guards each tangent step forwards cotangents through inactive
    /// iterations without evaluating their placeholder residuals.
    fn bind_bounded<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
        bound: usize,
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let state_types = driver.region(1)?.input_types();
        let state_count = state_types.len();

        // Linearize the body once under the converged tangent liveness of the state (state elements keep positional
        // identity, so the body's region input mask is the live state mask). The nonlinear half returns the next state
        // followed by ordinary residuals, and the linear half consumes the live state tangents followed by those
        // residuals and returns the tangents of the live state elements alone.
        check_count!("input", inputs, state_count, ProgramError);
        let body = driver.region(1)?;
        let activity = &self.activity;
        let input_indices = activity.input_indices();
        let live_tangent_slots = activity.live_tangent_slots(activity.output_tangent_liveness());
        let (primal_program, tangent_program, residual_count) =
            driver.linearize_program(body, input_indices)?.into_parts();
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

        // Run the primal loop over its original state plus a counter, one stack per stored residual, and a validity
        // mask.
        let predicate_type = condition.output_types()[0].as_array_type()?.clone();
        let counter_type: C::Type = bounded_while_scalar_type(DataType::I64, &predicate_type)?.into();
        let boolean_scalar_type: C::Type = bounded_while_scalar_type(DataType::Boolean, &predicate_type)?.into();
        let mask_stack_type: C::Type =
            boolean_scalar_type.as_array_type()?.with_inserted_dimension(0, bound.into()).unwrap().into();
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
            WhileOperation::new().with_iteration_bound(bound)?,
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
            .map(C::Operation::residual_storage_type)
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
                let mut restored_residuals: Vec<(C::Type, Tracer<TracingContext<C::Constant, C::Operation>>)> =
                    Vec::new();
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
        let mut scan_body_input_types: Vec<C::Type> = vec![ArrayType::scalar(DataType::I64).into()];
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
                        ConditionOperation::new(),
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

        // The tangent scan consumes invariant dimension residuals, live tangent carries, stored residual stacks, and
        // one scalar validity stack.
        let tangent_inputs = context.dual_primal_to_tangent(inputs)?;
        let mut tangent_scan_inputs = invariant_residual_sources
            .iter()
            .flatten()
            .map(|&state_index| tangent_inputs[state_index].primal().clone())
            .collect::<Vec<_>>();
        // Each state element's own primal names every runtime quantity a reference-bearing tangent type omits, because
        // the tangent type derivation preserves shapes exactly.
        for &index in input_indices {
            tangent_scan_inputs.push(context.materialize_dual_tangent(&inputs[index])?);
        }
        tangent_scan_inputs.extend(
            residual_stacks
                .into_iter()
                .map(|value| context.primal_to_tangent(value))
                .collect::<Result<Vec<_>, _>>()?,
        );
        tangent_scan_inputs.push(context.primal_to_tangent(mask_stack)?);
        let tangent_scan = ScanOperation::<C::Type>::new(invariant_residual_count + tangent_state_count, bound);
        let tangent_outputs = context.tangent().bind(tangent_scan, vec![scan_body], tangent_scan_inputs.as_slice())?;
        check_count!("output", tangent_outputs, invariant_residual_count + tangent_state_count, ProgramError);
        let tangent_outputs = tangent_outputs.into_iter().skip(invariant_residual_count).collect();
        DifferentiationDual::from_live_tangents(primal_outputs, tangent_outputs, activity.output_tangent_liveness())
    }

    /// Stages **one fused** doubled-state forward-mode `while` for the condition program `condition` and the input
    /// duals `inputs`, and returns the output duals. Like [JAX's `jvp` of
    /// `lax.while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html), the fused loop runs the
    /// pushforward alongside the primal loop instead of storing per-iteration residuals. The fused state is
    /// `[primal_state..., tangent_state...]`: the body is the loop body's fused forward-mode program (built through the
    /// instruction-scoped driver) and the condition is the original condition extended with ignored tangent-state
    /// inputs, so the continuation decision reads the primal state alone and the fused loop runs exactly as long as the
    /// primal loop. The fused state is compact: state elements whose tangent type is a zero differential space carry no
    /// tangent boundary input or output, and their output duals are restored as structural zeros. Because no residuals
    /// are stored, the rule applies to loops with *no* [`WhileOperation::iteration_bound`].
    ///
    /// Shared construction binds the fused loop directly. Separate construction partitions the generated loop while
    /// requiring every primal state output to remain known; loops that require an unbounded history are rejected.
    fn bind_fused<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        condition: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let state_count = inputs.len();

        // Build the fused body over the compact state `[primal_state..., live(tangent_state)...]` through the
        // instruction-scoped driver (region 1 is the loop body). The fused body carries a tangent exactly for the state
        // elements whose tangent is live after the carry fixed point (refer to `WhileJvp::new`), so inactive elements,
        // such as a pass-through loop limit, keep structural-zero tangents instead of threading materialized zeros
        // through the loop.
        let body = driver.region(1)?;
        check_count!("input", inputs, body.input_types().len(), ProgramError);
        let activity = &self.activity;
        let live_tangent_slots = activity.live_tangent_slots(activity.output_tangent_liveness());
        let tangent_state_count = activity.input_indices().len();
        let fused_state_count = state_count + tangent_state_count;
        // The body is differentiated through its region's retained transform cache, so a body shared by several
        // programs is differentiated once and repeated attachments of the result intern by `Arc` identity. Its tangent
        // outputs are projected to the live state elements only when some state element is inactive.
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
        // live tangent-state inputs are ignored, so the fused loop's iteration count is driven by the primal state
        // alone.
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
        // differential state elements, and zip the output halves back into `DifferentiationDual`s in the original state
        // order, restoring structural zeros for the omitted zero-space state elements. The fused body takes each of
        // those tangents as a real program input, so their structural zeros are materialized against their own primal,
        // which supplies the runtime extents a reference-bearing tangent type omits.
        let fused_while = WhileOperation::new().with_iteration_bound(self.operation.iteration_bound())?;
        let mut fused_inputs = Vec::with_capacity(fused_state_count);
        fused_inputs.extend(inputs.iter().map(|input| input.primal().clone()));
        for &index in activity.input_indices() {
            fused_inputs.push(context.materialize_dual_tangent(&inputs[index])?);
        }
        let fused_regions = [Arc::new(fused_condition), fused_body];
        let outputs = if std::ptr::eq(context.primal(), context.tangent()) {
            context.primal().bind(fused_while, AttachedRegions::Callees(&fused_regions), &fused_inputs)?
        } else {
            let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
            let arguments =
                fused_inputs.iter().map(|value| builder.add_input(value.r#type().into_owned())).collect::<Vec<_>>();
            let regions =
                fused_regions.iter().map(|program| builder.import_region(program.entry_region_ref())).collect();
            let outputs = builder.add_instruction(fused_while, regions, arguments, None)?.to_vec();
            let program = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
                outputs,
                vec![Placeholder; fused_state_count],
                vec![Placeholder; fused_state_count],
            )?;
            let input_known = (0..fused_state_count).map(|index| index < state_count).collect::<Vec<_>>();
            let required_outputs = (0..state_count).collect::<Vec<_>>();
            let partition =
                driver.partition_jvp_program(program.entry_region_ref(), &input_known, &required_outputs)?;
            partition.interpret_in_context(context, &fused_inputs, state_count)?
        };
        check_count!("output", outputs, fused_state_count, ProgramError);
        let mut primal_outputs = outputs;
        let tangent_outputs = primal_outputs.split_off(state_count);
        DifferentiationDual::from_live_tangents(primal_outputs, tangent_outputs, activity.output_tangent_liveness())
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
            "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got `{predicate}`",
        )));
    }
    let predicate_dimensions = predicate.shape().dimensions();
    for state_type in state_types {
        if !state_type.shape().dimensions().starts_with(predicate_dimensions) {
            return Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition predicate shape must be a prefix of every array state shape, but \
                 predicate `{predicate}` is not a prefix of state `{state_type}`",
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

/// Rebuilds one partially evaluated region of a `while` loop (i.e., the condition or the body) over the full state
/// signature of the loop, for the loop-invariant fold of the [`PartiallyEvaluatableOperation`] implementation for
/// [`WhileOperation`].
///
/// The condition and the body both run on the *same* loop-carried state every iteration, so each is rebuilt over the
/// identical `state_atoms` input signature rather than over its own residual-input signature. Each residual input of
/// `evaluation` is fed either by the matching state atom (for an unknown state element that the region still reads) or
/// by an inline constant (for a known residual that the region closed over, such as a folded loop-invariant state
/// value). The residual program of the region is then spliced over those inputs and its outputs are reassembled: a
/// folded [`Known`](PartialEvaluationOutput::Known) output becomes an inline constant and an
/// [`Unknown`](PartialEvaluationOutput::Unknown) output reads the corresponding output of the spliced residual program.
///
/// # Parameters
///
///   - `context`: Partial evaluation context that converts known values into program constants.
///   - `builder`: Builder accumulating the rebuilt program.
///   - `state_atoms`: Input atoms holding the loop state, in state order, to which the residual inputs of the region
///     map back.
///   - `evaluation`: Partial evaluation of the region against the loop-invariant-known state knowledge.
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

/// Runs a `while` loop's forward-mode rule directly at concrete duals for an [eager](Context::is_eager) context and
/// returns the output duals, resuming with the staged rule (refer to [`WhileJvp::bind`]) from the current duals when
/// the loop's predicate stops concretizing to one scalar Boolean.
///
/// Each iteration evaluates the condition on the concrete primal carries and interprets the body once over the current
/// dual carries. Body instructions re-enter their JVP rules through the differentiation driver, so nested
/// data-dependent `while` operations recurse through this same eager path without a program-level unroll pre-pass.
/// Data-dependent iteration counts therefore need no iteration bound, as for [JAX's `jvp` through an eagerly executed
/// loop](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html), while a semantic
/// [`iteration_bound`](WhileOperation::with_iteration_bound) truncates the loop once it is reached, matching the
/// bounded-`while` truncation semantics. Each body effect executes exactly once per loop iteration.
///
/// A predicate may fail to concretize on any iteration, not only the first: a batched per-item predicate has no single
/// continuation decision, and an eager transform context (e.g., batching over an eager parent) can turn a replicated
/// predicate into a mapped one once a replicated state element mixes with a mapped one in the body. A while loop is a
/// fixed point of its state, so continuing from the current state with the bound lowered by the number of completed
/// iterations is exact. The staged rule then evaluates the condition on that state again, which is sound only for a
/// pure condition without deferred work; for any other condition the concretization error is returned instead, because
/// re-running the condition would repeat its effects.
///
/// This rule derives no program from the body region and therefore consults no per-region transform cache: it
/// interprets the body once per iteration at the concrete duals reached by the loop, so its work is a function of the
/// runtime values rather than of the region's contents alone.
fn jvp_while_eagerly<
    C: Context<Type: DifferentiableType + WhileType> + Zero<C::Value>,
    D: DifferentiationDriver<C>,
    P: DifferentiationPolicy<C>,
>(
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
    C::Value: Concretizable<bool>,
    C::Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>
        + From<ConditionOperation<C::Type>>
        + From<WhileOperation<C::Type>>
        + From<ScanOperation<C::Type>>
        + WhileResidualStackOperation<C::Type>,
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
            // The predicate does not concretize to one scalar Boolean (e.g., a batched per-item predicate, whose items
            // stop on different iterations, has no single continuation decision, or a predicate that an eager batching
            // context maps once the body mixes a replicated state element with a mapped one). The loop resumes from
            // the current duals under the staged rule with the remaining bound, which is at least one because the
            // loop exits above, before evaluating the condition, once the bound is reached. The staged rule evaluates
            // the condition on the current state once more, so only a pure condition without deferred work resumes.
            Err(error) => {
                let condition_effects = condition.effects();
                if !condition_effects.classes().is_empty() || condition_effects.has_deferred_work() {
                    return Err(error.into());
                }
                let remaining = WhileOperation::new()
                    .with_iteration_bound(operation.iteration_bound().map(|bound| bound - completed_iterations))?;
                let duals = primal_carries
                    .into_iter()
                    .zip(tangent_carries)
                    .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
                    .collect::<Result<Vec<_>, _>>()?;
                return WhileJvp::new(driver, &remaining, &duals)?.bind(condition, body, context, driver, &duals);
            }
        };
        if !predicate {
            break;
        }

        // Advance one iteration by replaying the body directly over dual values. This is deliberately not routed
        // through a program-level unroll pre-pass: such a rewrite interprets the body to discover concrete nested-loop
        // iteration counts, after which replaying its fused JVP would execute primal effects a second time. Direct dual
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

    Ok(primal_carries
        .into_iter()
        .zip(tangent_carries)
        .map(|(primal, tangent)| DifferentiationDual::new(primal, tangent))
        .collect::<Result<Vec<_>, _>>()?)
}

/// Rewrites the condition and body of a while loop with a batched (per-item) predicate into its scalar-predicate
/// *masked form* over the same state, as JAX's `_while_loop_batching_rule` does: the masked condition evaluates the
/// original condition on the state and reduces the per-item predicate with a Boolean `any` over every predicate axis,
/// and the masked body evaluates the original condition on the same state again, replays the original body for
/// candidate updates, and selects per array state element between the candidate and the carried state under the
/// (broadcast) predicate. A first-class dimension carry is loop-invariant under a batched predicate, so it carries no
/// selection and its candidate is forwarded directly (refer to the documentation of [`WhileType`]).
///
/// The masked form evaluates the condition on exactly the states that the source loop evaluates it on: the masked
/// condition runs on the entry state and on the state after every body application that the iteration bound admits,
/// and the masked body re-evaluates it only on the state that the masked condition has just accepted. Interpretation
/// likewise stops at the bound before evaluating the condition, and the XLA lowering threads the predicate so that
/// reaching the bound skips the condition, so a condition that errors, diverges, or differs on the state after the last
/// admitted body application behaves alike on every path. Both regions are pure under a batched predicate (refer to
/// [`WhileOperation::validate_interfaces`]), so the repeated evaluation is unobservable. The bounded while forward-mode
/// rule uses this form for batched-predicate loops, whose counter- and stack-augmented differentiation state is not
/// predicate-prefixed and therefore needs the loop's masking made explicit as program data hanging off a scalar
/// predicate.
fn masked_while_programs<V: Value<Type: WhileType>, O: WhileResidualStackOperation<V::Type>>(
    condition: &Program<V, O, Vec<V>, Vec<V>>,
    body: &Program<V, O, Vec<V>, Vec<V>>,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>), ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    let state_types = body.input_types();
    let state_count = state_types.len();
    let mask_axes = (0..condition.output_types()[0].as_array_type()?.rank()).collect::<Vec<_>>();

    // Masked condition: `any(condition(state))` over every predicate axis.
    let (_, masked_condition) = TracingContext::<V, O>::trace_with_context_and_named_axes(
        |trace_context, state: Vec<Tracer<TracingContext<V, O>>>| {
            let predicate = condition.interpret_in_context(trace_context, state)?;
            check_count!("output", predicate, 1, ProgramError);
            trace_context.bind(O::mask_reduce_any(mask_axes.clone()), Vec::new(), predicate.as_slice())
        },
        state_types.clone(),
        Vec::new(),
    )?;

    // Masked body: the per-item predicate of the state, candidate updates from the replayed body, and per-element
    // masked selection between the candidate update and the carried state.
    let (_, masked_body) = TracingContext::<V, O>::trace_with_context_and_named_axes(
        |trace_context, state: Vec<Tracer<TracingContext<V, O>>>| {
            let mut predicate = condition.interpret_in_context(trace_context, state.clone())?;
            check_count!("output", predicate, 1, ProgramError);
            let mask = predicate.remove(0);
            let candidates = body.interpret_in_context(trace_context, state.clone())?;
            check_count!("output", candidates, state_count, ProgramError);
            let mut next_state = Vec::with_capacity(state_count);
            for ((candidate, carried), state_type) in candidates.iter().zip(&state).zip(state_types.iter()) {
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
                let varied_mask = vary_while_index_or_mask(trace_context, mask.clone(), state_array_type)?;
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
            // The predicate is recomputed from the whole state on every iteration, exactly as interpretation, the
            // lowering, and the fused forward-mode rule recompute it: an item whose predicate reads its own state alone
            // stays finished, because its state is frozen, while an item whose predicate reads other items may rejoin
            // on every path alike.
            Ok(next_state)
        },
        state_types.clone(),
        Vec::new(),
    )?;
    Ok((masked_condition, masked_body))
}

/// Stages the transitions that make `value`, a loop index or Boolean mask inside a bounded or masked `while` program
/// that carries no tangent, vary over every manual mesh axis that `target` varies over, so that it can be combined with
/// stacks or state of type `target`. A value without a sharding is first placed, replicated, on the mesh of `target`.
/// The value is returned unchanged when it already varies over those axes, which keeps programs outside manual regions
/// unchanged. The transitions are bound as raw operations because the tracing contexts that build these programs bind
/// no named axes, and they are never transposed because their inputs are integer or Boolean values.
fn vary_while_index_or_mask<V: Value<Type: WhileType>, O: WhileResidualStackOperation<V::Type>>(
    context: &TracingContext<V, O>,
    value: Tracer<TracingContext<V, O>>,
    target: &ArrayType,
) -> Result<Tracer<TracingContext<V, O>>, ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
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

/// Builds the augmented condition and body programs of the bounded staged while loop (see [`WhileJvp::bind_bounded`]).
/// The body replays the original body through its tracing context before adding the augmented-state operations, while
/// the condition structurally relocates the original condition into a builder with the extended input boundary.
///
/// The augmented loop state is `[original_state..., counter (i64 scalar), residual_stacks..., mask_stack]`:
///
///   - The body runs the residual-extended primal body (which outputs `[next_state..., residuals...]`) on the
///     original state slots, then *stores* instead of returning each per-iteration residual: residual `k` is
///     broadcast to `[1, …]` and written into stack `k` at stack slot `counter` via `dynamic_update_slice`, a scalar
///     Boolean `one` (true) is written into the Boolean `[bound]` mask stack at stack slot `counter`, and the counter
///     advances by an i64 `one`. Because the enclosing while keeps `iteration_bound = bound`, the counter is always
///     strictly below `bound` whenever the body runs, so the writes can never clamp.
///   - The condition is the original loop condition extended with ignored extra-state inputs.
///
/// Returns the extended condition, the augmented body, and the `[bound, …]` residual stack types.
fn build_bounded_while_programs<V: Value<Type: WhileType>, O: WhileResidualStackOperation<V::Type>>(
    condition: &Program<V, O, Vec<V>, Vec<V>>,
    primal_body: &Program<V, O, Vec<V>, Vec<V>>,
    residual_types: &[V::Type],
    bound: usize,
) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>, Vec<V::Type>), ProgramError>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    let state_count = condition.input_types().len();
    let predicate_type = condition.output_types()[0].as_array_type()?.clone();
    let counter_type: V::Type = bounded_while_scalar_type(DataType::I64, &predicate_type)?.into();
    let boolean_scalar_type: V::Type = bounded_while_scalar_type(DataType::Boolean, &predicate_type)?.into();
    let mask_stack_type: V::Type =
        boolean_scalar_type.as_array_type()?.with_inserted_dimension(0, bound.into()).unwrap().into();
    let storage_types = residual_types.iter().map(O::residual_storage_type).collect::<Result<Vec<_>, _>>()?;
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
                .map(|storage_type| storage_type.with_inserted_dimension(0, bound.into()).unwrap().into())
        })
        .collect::<Result<Vec<V::Type>, _>>()?;

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
                    "bounded `{WHILE_OPERATION_NAME}` body adapter is missing the counter input",
                ))
            })?;
            let stack_inputs = extra_inputs.by_ref().take(stack_types.len()).cloned().collect::<Vec<_>>();
            check_count!("input", stack_inputs, stack_types.len(), ProgramError);
            let mask_input = extra_inputs.next().cloned().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "bounded `{WHILE_OPERATION_NAME}` body adapter is missing the mask input",
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayElement, ArrayOperation, ArrayReference, Dimension, DimensionBounds, DimensionError, DimensionType,
        DimensionValue, DimensionVariable, Layout, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
        ShardingDimension, StridedLayout,
    };
    use crate::axes::{NamedAxes, NamedAxis};
    use crate::batching::{Batch, batch};
    use crate::captures::ClosedProgram;
    use crate::contexts::{Domain, EagerContext, StagingContext, ValueResolution};
    use crate::differentiation::{
        ForwardModeDifferentiate, LinearizationTracer, ReverseModeDifferentiate, differentiate_at,
    };
    use crate::macros::{check_gradient, check_operation_batching};
    use crate::operations::arithmetic::{AddOperation, DivOperation, MulOperation, NegOperation, SubOperation};
    use crate::operations::collectives::parallel_reduce::ParallelReduceOperation;
    use crate::operations::comparisons::{COMPARE_OPERATION_NAME, Compare, CompareOperation, ComparisonDirection};
    use crate::operations::composites::CompositeTransposeOperation;
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
    use crate::operations::debugging::{PRINT_OPERATION_NAME, PrintOperation};
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::logical::{AndOperation, OrOperation};
    use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcastOperation};
    use crate::operations::manipulation::reshaping::DynamicReshapeOperation;
    use crate::operations::manipulation::reversing::ReverseOperation;
    use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
    use crate::operations::references::{
        REFERENCE_FREEZE_OPERATION_NAME, REFERENCE_READ_OPERATION_NAME, ReferenceAddUpdate,
        ReferenceAddUpdateOperation, ReferenceFreeze, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation,
        ReferenceRead, ReferenceReadOperation, ReferenceWriteOperation,
    };
    use crate::operations::sharding::shard_map::{ShardMap, ShardMapOperation};
    use crate::operations::trigonometric::SinOperation;
    use crate::parameters::Parameter;
    use crate::partial::{PartialTracer, ResidualInputSource};
    use crate::programs::{
        AttachedRegions, EffectClasses, InstructionId, Provenance, ProvenanceScope, ReferenceAnalysisError,
        ReferenceRoot, ReferenceSource, ReferenceType, RegionDataFlowRegionBoundary, RegionDriver,
    };
    use crate::tracing::{DomainTracingContext, Tracer, TracingContext};

    use super::*;

    /// Operation family of the homogeneous array `while` loops under test.
    type TestOperation = ArrayOperation<Array>;

    /// Homogeneous array program, used both for `while` regions and for the programs that apply `while` loops.
    type TestProgram = Program<Array, TestOperation, Vec<Array>, Vec<Array>>;

    /// Eager homogeneous array context.
    type TestEagerContext = EagerContext<Array, TestOperation>;

    /// Staging homogeneous array context.
    type TestTracingContext = TracingContext<Array, TestOperation>;

    /// Staged homogeneous array value of [`TestTracingContext`].
    type TestStagedTracer = Tracer<TestTracingContext>;

    /// Linearization tracer over [`TestEagerContext`].
    type TestTracer = LinearizationTracer<TestEagerContext>;

    /// Array IR program, used both for `while` regions and for the programs that apply `while` loops.
    type TestIrProgram = Program<TestIrValue, TestIrOperation, Vec<TestIrValue>, Vec<TestIrValue>>;

    /// Eager array IR context.
    type TestIrEagerContext = EagerContext<TestIrValue, TestIrOperation>;

    /// Staging array IR context.
    type TestIrTracingContext = TracingContext<TestIrValue, TestIrOperation>;

    /// Staged array IR value of [`TestIrTracingContext`].
    type TestIrStagedTracer = Tracer<TestIrTracingContext>;

    /// Eager test context that records the name and inputs of every operation bound through it before executing the
    /// operation in [`TestEagerContext`]. Clones share the record, so nested transformation contexts and region replay
    /// contribute to one observation.
    #[derive(Clone, Debug)]
    struct BindRecordingContext {
        /// Names and inputs of the operations bound through this context, in binding order.
        binds: Rc<RefCell<Vec<(&'static str, Vec<Array>)>>>,
    }

    impl BindRecordingContext {
        /// Creates a context that has recorded no binds.
        fn new() -> Self {
            Self { binds: Rc::new(RefCell::new(Vec::new())) }
        }

        /// Returns the inputs of each operation named `name` bound so far, in binding order.
        fn inputs_of(&self, name: &str) -> Vec<Vec<Array>> {
            let binds = self.binds.borrow();
            binds.iter().filter(|(bound, _)| *bound == name).map(|(_, inputs)| inputs.clone()).collect()
        }
    }

    impl Domain for BindRecordingContext {
        type Type = ArrayType;
        type Value = Array;
        type Constant = Array;
        type Operation = TestOperation;
    }

    impl Context for BindRecordingContext {
        fn lift(&self, constant: Array) -> Result<Array, ProgramError> {
            Ok(constant)
        }

        fn bind<'r, P: Into<Self::Operation>, R: Into<AttachedRegions<'r, Self::Constant, Self::Operation>>>(
            &self,
            operation: P,
            regions: R,
            inputs: &[Self::Value],
        ) -> Result<Vec<Self::Value>, ProgramError>
        where
            Self::Constant: 'r,
            Self::Operation: 'r,
        {
            let operation = operation.into();
            self.binds.borrow_mut().push((operation.name(), inputs.to_vec()));
            TestEagerContext::new().bind(operation, regions, inputs)
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

    impl Zero<Array> for BindRecordingContext {
        fn zero(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
            Zero::zero(&EagerContext::<Array>::new(), r#type)
        }
    }

    impl One<Array> for BindRecordingContext {
        fn one(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
            One::one(&EagerContext::<Array>::new(), r#type)
        }
    }

    impl<S: ArrayElement> Fill<S, Array> for BindRecordingContext {
        fn fill(&self, r#type: &ArrayType, value: S) -> Result<Array, ProgramError> {
            Fill::fill(&EagerContext::<Array>::new(), r#type, value)
        }
    }

    impl Iota<Array> for BindRecordingContext {
        fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<Array, ProgramError> {
            Iota::iota(&EagerContext::<Array>::new(), r#type, dimension)
        }
    }

    impl Constant<Array, Array> for BindRecordingContext {
        fn constant(&self, value: Array) -> Result<Array, ProgramError> {
            Ok(value)
        }
    }

    /// Builds a program that applies the `while` `operation` with its condition and body `regions` to inputs of the
    /// provided `input_types` and returns the final loop state.
    fn while_program<T: WhileType, V: Value<Type = T>, O: Clone + Operation<Type = T> + From<WhileOperation<T>>>(
        operation: WhileOperation<T>,
        regions: &[Program<V, O, Vec<V>, Vec<V>>],
        input_types: Vec<T>,
    ) -> Program<V, O, Vec<V>, Vec<V>> {
        let mut builder = ProgramBuilder::<V, O>::new();
        let regions = regions.iter().map(|region| builder.import_region(region.entry_region_ref())).collect();
        let state_count = input_types.len();
        let inputs = input_types.into_iter().map(|r#type| builder.add_input(r#type)).collect();
        let outputs = builder.add_instruction(operation, regions, inputs, None).unwrap().to_vec();
        builder.build(outputs, vec![Placeholder; state_count], vec![Placeholder; state_count]).unwrap()
    }

    /// Binds the `while` `operation` with its condition and body `regions` to `inputs` in the domain of the inputs and
    /// returns the final loop state.
    fn bind_while<V: Value<Type = ArrayType>>(
        operation: WhileOperation<ArrayType>,
        regions: Vec<TestProgram>,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError>
    where
        V::Domain: Context<Type = ArrayType, Constant = Array, Operation = TestOperation>,
    {
        inputs[0].domain().bind(TestOperation::While(operation), regions, inputs)
    }

    /// Traces the value and gradient of the final state of the `while` `operation` with its condition and body
    /// `regions` through [`TestTracingContext`] at an initial state of type `state_type`. The staged primal makes reverse
    /// mode take the staged (non-concretizing) paths of the `while` rules, and the returned program maps the initial
    /// state to the value and the gradient.
    fn trace_while_value_and_gradient(
        operation: WhileOperation<ArrayType>,
        regions: Vec<TestProgram>,
        state_type: ArrayType,
    ) -> Program<Array, TestOperation, Array, (Array, Array)> {
        let function = move |state: TestStagedTracer| {
            Ok(differentiate_at(state).value_and_gradient(move |state| {
                bind_while(operation, regions.clone(), &[state]).map(|mut outputs| outputs.remove(0))
            })?)
        };
        TestTracingContext::trace(function, state_type).unwrap().1
    }

    /// Traces reverse mode through the final state of the `while` `operation` with its condition and body `regions`
    /// through [`TestTracingContext`] at a scalar `f64` initial state, so that it takes the staged (non-concretizing)
    /// paths of the `while` rules. Returns the transposed pullback program, which maps the output cotangent followed by
    /// the residuals to the input cotangent, together with the traced program, which maps the initial state to the
    /// final state followed by those residuals.
    #[allow(clippy::type_complexity)]
    fn trace_while_vjp(
        operation: WhileOperation<ArrayType>,
        regions: Vec<TestProgram>,
    ) -> (Arc<TestProgram>, Program<Array, TestOperation, Array, (Array, Vec<Array>)>) {
        let mut pullback = None;
        let (_, program) = TestTracingContext::trace(
            |state: TestStagedTracer| {
                let (output, staged_pullback) = state.context().clone().vjp(
                    move |state, ()| Ok(bind_while(operation, regions.clone(), &[state])?.remove(0)),
                    state,
                    (),
                )?;
                let (staged_pullback, residuals) = staged_pullback.into_transposed_parts()?;
                pullback = Some(staged_pullback.into_program());
                Ok((output, residuals))
            },
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        (pullback.unwrap(), program)
    }

    /// Stages `while_operation` over one batched item (mapped at axis 0 with `batch_size` batch items) under tracing
    /// and returns the staged batched program for structural and numeric assertions.
    fn batch_while_under_tracing(
        while_operation: WhileOperation<ArrayType>,
        while_regions: Vec<TestProgram>,
        batch_size: usize,
    ) -> Program<Array, TestOperation, Array, Array> {
        let parent = DomainTracingContext::<TestEagerContext>::new();
        let builder = parent.builder().clone();
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(batch_size)]));
        let input_atom = builder.borrow_mut().add_input(input_type);
        let input_tracer = parent.tracer(input_atom, None);
        let output = batch(
            |item| {
                let mut outputs = item.context().bind(
                    TestOperation::While(while_operation),
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

    /// Builds the per-item countdown loop `while (x > 0) { x = x - 1 }` over a state of type `state_type`, returning
    /// its condition and body programs in `while` region order.
    fn countdown_while_regions(state_type: ArrayType) -> Vec<TestProgram> {
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = condition_builder.add_input(state_type.clone());
        let zero =
            condition_builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::GreaterThan),
                Vec::new(),
                vec![state, zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = body_builder.add_input(state_type);
        let one = body_builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let next_state =
            body_builder.add_instruction(SubOperation::new(), Vec::new(), vec![state, one], None).unwrap()[0];
        let body = body_builder.build(vec![next_state], vec![Placeholder], vec![Placeholder]).unwrap();
        vec![condition, body]
    }

    /// Builds a `while` condition and body over an `f64[extent]` state whose condition is always false and whose body
    /// returns a constant `f64[3]`. The body maps its state to itself only when `extent` is 3, so its declared
    /// signature is invalid even though an `f64[3]` input refines it to a valid one.
    fn refinement_only_while_regions() -> [TestProgram; 2] {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        builder.add_input(state_type.clone());
        let predicate = builder.add_constant(Array::scalar(false).unwrap());
        let condition = builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        builder.add_input(state_type);
        let output = builder.add_constant(Array::vector(vec![2.0f64; 3]).unwrap());
        let body = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        [condition, body]
    }

    /// Builds the `state < threshold` while condition program over one scalar state element.
    fn scalar_threshold_condition(threshold: f64) -> TestProgram {
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
    fn bounded_doubling_while_operation(threshold: f64, bound: usize) -> (WhileOperation<ArrayType>, Vec<TestProgram>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
    fn effectful_doubling_while_operation() -> (WhileOperation<ArrayType>, Vec<TestProgram>) {
        let condition = scalar_threshold_condition(8.0);
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let input = body_builder.add_input(ArrayType::scalar(DataType::F64));
        let printed =
            body_builder.add_instruction(PrintOperation::new("state"), Vec::new(), vec![input], None).unwrap()[0];
        let two = body_builder.add_constant(Array::scalar(2.0).unwrap());
        let output =
            body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![printed, two], None).unwrap()[0];
        let body = body_builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        (WhileOperation::new(), vec![condition, body])
    }

    /// Builds the `while (x < threshold) { x = x * x }` loop under the optional `iteration_bound`. Squaring captures
    /// the loop state itself as a loop-varying residual and makes the pushforward read the primal state every
    /// iteration, so differentiating the bounded loop exercises the per-iteration residual stacks of the bounded staged
    /// rule, staged forward mode through the unbounded loop exercises the primal/tangent coupling of the fused
    /// doubled-state rule, and linearizing the unbounded loop exercises the primal recomputation of the
    /// closed-knownness split inside the residual loop.
    fn squaring_while_operation(
        threshold: f64,
        iteration_bound: Option<usize>,
    ) -> (WhileOperation<ArrayType>, Vec<TestProgram>) {
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = builder.build(vec![squared], vec![Placeholder], vec![Placeholder]).unwrap();
        let operation = WhileOperation::new().with_iteration_bound(iteration_bound).unwrap();
        (operation, vec![scalar_threshold_condition(threshold), body])
    }

    /// Builds the `while (counter > 0) { (counter, value) = (counter - 1, value + value) }` loop whose predicate
    /// depends only on the counter state element.
    fn counter_doubling_while_operation() -> (WhileOperation<ArrayType>, Vec<TestProgram>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
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

    /// Builds the `while (i < 3) { (i, y, s) = (i + y, y, s + s) }` loop over the scalar `f64` state `[i, y, s]`. When
    /// `print_counter` is set, the condition prints `i` before comparing it, which makes the condition effectful.
    fn stepping_counter_while_operation(print_counter: bool) -> (WhileOperation<ArrayType>, Vec<TestProgram>) {
        let scalar_f64 = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
        let counter = condition_builder.add_input(scalar_f64.clone());
        condition_builder.add_input(scalar_f64.clone());
        condition_builder.add_input(scalar_f64.clone());
        let counter = match print_counter {
            true => condition_builder
                .add_instruction(PrintOperation::new("counter"), Vec::new(), vec![counter], None)
                .unwrap()[0],
            false => counter,
        };
        let limit = condition_builder.add_constant(Array::scalar(3.0).unwrap());
        let predicate = condition_builder
            .add_instruction(
                CompareOperation::new(ComparisonDirection::LessThan),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let counter = body_builder.add_input(scalar_f64.clone());
        let step = body_builder.add_input(scalar_f64.clone());
        let scale = body_builder.add_input(scalar_f64);
        let next_counter =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![counter, step], None).unwrap()[0];
        let next_scale =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![scale, scale], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![next_counter, step, next_scale],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        (WhileOperation::new(), vec![condition, body])
    }

    /// Builds a loop over `[counter, first, second, source]` whose body returns
    /// `[counter - 1, second, source, source]` while the counter is positive, under the optional `iteration_bound`.
    fn chained_carry_while_program(iteration_bound: Option<usize>) -> TestProgram {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let condition = condition_builder.build(vec![predicate], vec![Placeholder; 4], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let body_inputs = (0..4).map(|_| body_builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
        let one = body_builder.add_constant(Array::scalar(1.0).unwrap());
        let next_counter = body_builder
            .add_instruction(SubOperation::new(), Vec::new(), vec![body_inputs[0], one], None)
            .unwrap()[0];
        let body = body_builder
            .build(
                vec![next_counter, body_inputs[2], body_inputs[3], body_inputs[3]],
                vec![Placeholder; 4],
                vec![Placeholder; 4],
            )
            .unwrap();
        let operation = WhileOperation::new().with_iteration_bound(iteration_bound).unwrap();
        while_program(operation, &[condition, body], vec![scalar_type; 4])
    }

    /// Builds a program whose `while` loop runs over the states `[counter: f64[], accumulator: f64[]]` while `counter >
    /// 0`, computing `(counter - 1, accumulator + counter)` under the optional `iteration_bound`. The counter is a
    /// closed, but not loop-invariant, state element: its next value and the predicate depend on it alone.
    fn counter_accumulator_while_program(iteration_bound: Option<usize>) -> TestProgram {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let condition = condition_builder.build(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let counter = body_builder.add_input(scalar_type.clone());
        let accumulator = body_builder.add_input(scalar_type.clone());
        let one = body_builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
        let next_counter =
            body_builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
        let next_accumulator = body_builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![accumulator, counter], None)
            .unwrap()[0];
        let body = body_builder
            .build(vec![next_counter, next_accumulator], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let operation = WhileOperation::new().with_iteration_bound(iteration_bound).unwrap();
        while_program(operation, &[condition, body], vec![scalar_type; 2])
    }

    /// Builds a program whose `while` loop carries `[counter: f64[], x: f64[n], accumulator: f64[n]]`, runs while
    /// `counter > 0`, and computes `(counter - 1, x, accumulator + x)`, applied to inputs of the types `f64[]`,
    /// `f64[3]`, and `f64[n]`, so that its entry requires `n = 3`. Returns the program and the declared `f64[n]` state
    /// type.
    fn vector_accumulator_while_program() -> (TestProgram, ArrayType) {
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let vector_type = ArrayType::new(DataType::F64, Shape::new(vec![n.simplified_dimension()]));
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
        let counter = condition_builder.add_input(scalar_type.clone());
        condition_builder.add_input(vector_type.clone());
        condition_builder.add_input(vector_type.clone());
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
        let condition = condition_builder.build(vec![predicate], vec![Placeholder; 3], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let counter = body_builder.add_input(scalar_type.clone());
        let x = body_builder.add_input(vector_type.clone());
        let accumulator = body_builder.add_input(vector_type.clone());
        let one = body_builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
        let next_counter =
            body_builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
        let next_accumulator =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![accumulator, x], None).unwrap()[0];
        let body = body_builder
            .build(vec![next_counter, x, next_accumulator], vec![Placeholder; 3], vec![Placeholder; 3])
            .unwrap();
        let input_types = vec![scalar_type, ArrayType::new_static(DataType::F64, [3]), vector_type.clone()];
        (while_program(WhileOperation::new(), &[condition, body], input_types), vector_type)
    }

    /// Builds the array IR per-item countdown loop `while (x > 0) { x = x - 1 }` over one scalar `f64` array state
    /// element, returning its condition and body programs in `while` region order.
    fn array_ir_countdown_while_regions() -> Vec<TestIrProgram> {
        let scalar_f64 = ArrayIrType::Array(ArrayType::scalar(DataType::F64));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition_state = condition_builder.add_input(scalar_f64.clone());
        let zero = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::ZeroLike(ZeroLikeOperation::new())),
                Vec::new(),
                vec![condition_state],
                None,
            )
            .unwrap()[0];
        let predicate = condition_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::Compare(CompareOperation::new(
                    ComparisonDirection::GreaterThan,
                ))),
                Vec::new(),
                vec![condition_state, zero],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let body_state = body_builder.add_input(scalar_f64);
        let one = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::OneLike(OneLikeOperation::new())),
                Vec::new(),
                vec![body_state],
                None,
            )
            .unwrap()[0];
        let next = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::Sub(SubOperation::new())),
                Vec::new(),
                vec![body_state, one],
                None,
            )
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        vec![condition, body]
    }

    /// Builds the bounded loop body `x ↦ 1.5 x + 1` over an `f64[2]` state under the batched predicate `[x₀ < 1 ∨ 2 ≤
    /// x₁ < 3, x₁ < 3]`, whose item 0 reads item 1 and therefore rejoins the loop once item 1 reaches the window `[2,
    /// 3)`, returning its condition and body programs in `while` region order.
    fn rejoining_while_regions() -> Vec<TestProgram> {
        let state_type = ArrayType::new_static(DataType::F64, [2]);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        vec![condition, body]
    }

    /// Builds a program whose `while` loop runs over the states `[counter: f64[], vector: f64[rows]]` from
    /// unspecialized regions (`counter > 0`, and `[counter - 1, vector + vector]`) while the program feeds it a static
    /// `f64[3]` vector, so the loop's outputs are refined by its inputs rather than by re-typed regions.
    fn refined_vector_while_program(iteration_bound: Option<usize>) -> TestIrProgram {
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
        let condition = condition_builder.build(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap();

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
            .build(vec![next_counter, next_vector], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(iteration_bound).unwrap();
        let input_types = vec![counter_type, ArrayIrType::Array(ArrayType::new_static(DataType::F64, [3]))];
        while_program(operation, &[condition, body], input_types)
    }

    /// Builds a `while` condition and body over an `f32[]` carry whose condition is always false and whose body
    /// replaces the carry with the unsharded constant `1`. A body specialized at a sharded carry therefore no longer
    /// maps its carry to itself, so every rule that specializes the regions must fall back to the declared ones.
    fn sharding_dropping_while_regions() -> Vec<TestIrProgram> {
        let state_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(state_type.clone());
        let predicate = condition_builder.add_constant(array(Array::scalar(false).unwrap()));
        let condition = condition_builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        body_builder.add_input(state_type);
        let replacement = body_builder.add_constant(array(Array::scalar(1f32).unwrap()));
        let body = body_builder.build(vec![replacement], vec![Placeholder], vec![Placeholder]).unwrap();
        vec![condition, body]
    }

    /// Builds a `while` condition and body over two `ref<f32[]>` carries whose condition is always false and whose body
    /// returns each carry at the other's position, which gives the loop state no fixed point.
    fn swapped_reference_while_regions() -> Vec<TestIrProgram> {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(reference_type.clone());
        condition_builder.add_input(reference_type.clone());
        let predicate = condition_builder.add_constant(array(Array::scalar(false).unwrap()));
        let condition = condition_builder.build(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = body_builder.add_input(reference_type.clone());
        let second = body_builder.add_input(reference_type);
        let body = body_builder.build(vec![second, first], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        vec![condition, body]
    }

    /// Builds a while condition that increments reference state before comparing it with `limit`.
    fn incrementing_condition(limit: f32) -> TestIrProgram {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let update = builder.add_constant(array(Array::scalar::<f32>(1.0).unwrap()));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let counter =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let limit = builder.add_constant(array(Array::scalar::<f32>(limit).unwrap()));
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
    fn reference_while_program(condition: TestIrProgram, iteration_bound: Option<usize>) -> TestIrProgram {
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

    /// Returns the error that rejects a `while` body that does not return reference carry 0 as the reference that it
    /// entered with, which gives the loop state no fixed point.
    fn reference_carry_swap_error() -> ProgramError {
        ProgramError::Type(TypeError::invalid(format!(
            "`{WHILE_OPERATION_NAME}` body does not return carry 0 as the reference it entered with, so the loop state \
             has no fixed point",
        )))
    }

    /// Returns the error of the `while` transposition rule, which every staged linear `while` loop reports.
    fn while_transposition_error() -> ProgramError {
        ProgramError::UnsupportedOperation {
            message: format!(
                "`{WHILE_OPERATION_NAME}` does not support transposition (reverse-mode differentiation through staged \
                 linear `{WHILE_OPERATION_NAME}` loops is not supported; eager differentiation executes concrete \
                 duals, and the forward-mode rule of loops built with `with_iteration_bound` stages a transposable \
                 masked scan)",
            ),
        }
    }

    /// Builds a `while` body over one `ref<f32[]>` carry that doubles the referenced counter in place.
    fn doubling_reference_body() -> TestIrProgram {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let counter =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let doubled = builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, counter], None)
            .unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, doubled], None)
            .unwrap();
        builder.build(vec![reference], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds `f(x: f32[]) -> f32[]`, which allocates a local counter reference initialized to `x`, runs the
    /// [`doubling_reference_body`] loop under [`incrementing_condition`] with a limit of 10 and the optional
    /// `iteration_bound`, and freezes the counter.
    fn doubling_reference_while_program(iteration_bound: Option<usize>) -> TestIrProgram {
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![
            builder.import_program(incrementing_condition(10.0)),
            builder.import_program(doubling_reference_body()),
        ];
        let initial = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(iteration_bound).unwrap();
        let reference = builder.add_instruction(operation, regions, vec![reference], None).unwrap()[0];
        let value =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder.build(vec![value], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_while_type_validate_while_predicate() {
        // An array predicate must be Boolean, and a batched one must be a prefix of every state shape.
        let state_types = vec![ArrayType::new_static(DataType::F32, [2])];
        assert_eq!(ArrayType::validate_while_predicate(&ArrayType::scalar(DataType::Boolean), &state_types), Ok(()),);
        assert_eq!(
            ArrayType::validate_while_predicate(&ArrayType::new_static(DataType::Boolean, [2]), &state_types),
            Ok(()),
        );
        assert_eq!(
            ArrayType::validate_while_predicate(&ArrayType::scalar(DataType::F32), &state_types),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got `f32[]`",
            ))),
        );
        assert_eq!(
            ArrayType::validate_while_predicate(&ArrayType::new_static(DataType::Boolean, [3]), &state_types),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition predicate shape must be a prefix of every array state shape, but \
                 predicate `bool[3]` is not a prefix of state `f32[2]`",
            ))),
        );

        // An array IR predicate must be a Boolean array member, so a first-class dimension or a reference predicate is
        // rejected, and the prefix requirement applies to the array state members only.
        let dimension_type = ArrayIrType::Dimension(DimensionType::from(Dimension::Static(3)));
        let state_types = vec![dimension_type.clone(), ArrayType::new_static(DataType::F32, [2]).into()];
        assert_eq!(
            ArrayIrType::validate_while_predicate(&ArrayType::new_static(DataType::Boolean, [2]).into(), &state_types,),
            Ok(()),
        );
        assert_eq!(
            ArrayIrType::validate_while_predicate(&dimension_type, &state_types),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got `dimension<3>`",
            ))),
        );
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::Boolean)));
        assert_eq!(
            ArrayIrType::validate_while_predicate(&reference_type, &state_types),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got `ref<bool[]>`",
            ))),
        );
    }

    #[test]
    fn test_while_type_is_batched_while_predicate() {
        // Only a non-scalar Boolean array predicate is batched, so a scalar predicate, a first-class dimension, and a
        // reference to a non-scalar Boolean array are not.
        let batched_type = ArrayType::new_static(DataType::Boolean, [2]);
        assert!(!ArrayType::is_batched_while_predicate(&ArrayType::scalar(DataType::Boolean)));
        assert!(ArrayType::is_batched_while_predicate(&batched_type));
        assert!(!ArrayIrType::is_batched_while_predicate(&ArrayType::scalar(DataType::Boolean).into()));
        assert!(ArrayIrType::is_batched_while_predicate(&batched_type.clone().into()));
        assert!(!ArrayIrType::is_batched_while_predicate(&ArrayIrType::Dimension(DimensionType::from(
            Dimension::Static(2),
        ))));
        assert!(!ArrayIrType::is_batched_while_predicate(&ArrayIrType::Reference(ReferenceType::new(batched_type))));
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
                message: format!("cannot use a value of type `f64[1]` as a Boolean `{WHILE_OPERATION_NAME}` predicate"),
            }),
        );

        // Array IR predicates delegate to their array payload and reject dimension and reference predicates.
        assert_eq!(array(Array::vector(vec![false, true]).unwrap()).any_true(), Ok(true));
        assert_eq!(
            TestIrValue::Dimension(DimensionValue::constant(3).unwrap()).any_true(),
            Err(ProgramError::Concretization {
                message: format!("cannot use first-class dimension `3` as a `{WHILE_OPERATION_NAME}` predicate"),
            }),
        );
        assert_eq!(
            TestIrValue::Reference(ArrayReference::new(Array::scalar(true).unwrap())).any_true(),
            Err(ProgramError::Concretization {
                message: format!("cannot use reference `ref<bool[]>` as a `{WHILE_OPERATION_NAME}` predicate"),
            }),
        );

        // Symbolic values inherit the scalar default, which surfaces their concretization error.
        let trace = TestTracingContext::new();
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
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition predicate shape must be a prefix of every array state shape, but \
                 predicate `bool[2]` is not a prefix of state `f64[4]`",
            )))),
        );
        let matrix_predicate =
            Array::from_elements(ArrayType::new_static(DataType::Boolean, [2, 2]), &[false, true, true, false])
                .unwrap();
        assert_eq!(
            matrix_predicate.mask_select(&flattened, &flattened),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition predicate shape must be a prefix of every array state shape, but \
                 predicate `bool[2, 2]` is not a prefix of state `f64[4]`",
            )))),
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
        let scalar_true = array(Array::scalar(true).unwrap());
        let scalar_false = array(Array::scalar(false).unwrap());
        let batched = array(Array::vector(vec![false, true]).unwrap());
        assert_eq!(
            batched.mask_select(
                &array(Array::vector(vec![1.0f64, 2.0]).unwrap()),
                &array(Array::vector(vec![-1.0f64, -2.0]).unwrap()),
            ),
            Ok(array(Array::vector(vec![-1.0f64, 2.0]).unwrap())),
        );

        // Reference carries are selected wholesale by a scalar predicate: the handle itself is selected, never its
        // contents, so the selected carry aliases the chosen input. A batched predicate cannot mask a reference per
        // item, so it accepts only identical carries.
        let on_true = TestIrValue::Reference(ArrayReference::new(Array::scalar(1.0f32).unwrap()));
        let on_false = TestIrValue::Reference(ArrayReference::new(Array::scalar(2.0f32).unwrap()));
        assert_eq!(scalar_true.mask_select(&on_true, &on_false), Ok(on_true.clone()));
        assert_eq!(scalar_false.mask_select(&on_true, &on_false), Ok(on_false.clone()));
        assert_eq!(batched.mask_select(&on_true, &on_true), Ok(on_true.clone()));
        assert_eq!(
            batched.mask_select(&on_true, &on_false),
            Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "a batched `{WHILE_OPERATION_NAME}` predicate cannot select between distinct reference carries per \
                     batch item; reference state has one value per allocation",
                ),
            }),
        );

        // Dimension carries are selected wholesale by a scalar predicate. A batched predicate returns equal dimension
        // carries unchanged, because masking a loop-invariant carry is the identity, but one dimension value cannot
        // represent independently masked per-item extents, so distinct dimension carries need a scalar predicate.
        let three = TestIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let four = TestIrValue::Dimension(DimensionValue::constant(4).unwrap());
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
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` predicate cannot select between mismatched state types `dimension<3>` and \
                 `ref<f32[]>`",
            )))),
        );
        assert_eq!(
            three.mask_select(&on_true, &on_false),
            Err(ProgramError::Concretization {
                message: format!("cannot use first-class dimension `3` as a `{WHILE_OPERATION_NAME}` predicate"),
            }),
        );
        assert_eq!(
            on_true.mask_select(&three, &four),
            Err(ProgramError::Concretization {
                message: format!("cannot use reference `ref<f32[]>` as a `{WHILE_OPERATION_NAME}` predicate"),
            }),
        );

        // Symbolic values inherit the scalar default, which surfaces their concretization error.
        let trace = TestTracingContext::new();
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
            Err(TypeError::invalid(format!(
                "batched `{WHILE_OPERATION_NAME}` predicate `bool[2][sharding={{mesh<['m'=2:manual]>, [{{}}], \
                 varying_manual={{'m'}}}}]` varies over manual axes that state `bool[2]` does not vary over",
            ))
            .into()),
        );
    }

    #[test]
    fn test_while() {
        // A `while` payload carries only its optional iteration bound, so loops compare, hash, copy, and render by that
        // bound, and the default loop is the unbounded one.
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
        assert_eq!(operation.to_string(), "while");
        assert_eq!(bounded.to_string(), "while [iteration_bound=2]");
        assert_eq!(
            format!("{operation:?}"),
            "WhileOperation { iteration_bound: None, marker: PhantomData<fn() -> \
             ryft_core::arrays::types::arrays::ArrayType> }",
        );

        // Staging imports the condition and body programs as attached regions of the staged instruction instead of
        // driving the loop with a concrete predicate, and program rendering shows them with their declared slot names.
        let state_type = ArrayType::scalar(DataType::F64);
        let (output_type, program) = TestTracingContext::trace(
            |state: Tracer<TestTracingContext>| {
                let context = state.context().clone();
                Ok(context
                    .bind(bounded, countdown_while_regions(ArrayType::scalar(DataType::F64)), &[state])?
                    .remove(0))
            },
            state_type.clone(),
        )
        .unwrap();
        assert_eq!(output_type, state_type);
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
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` iteration bound must be at least 1",
            )))),
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
                    "`{WHILE_OPERATION_NAME}` iteration bound {bound} exceeds the maximum supported extent \
                     {MAX_DIMENSION_EXTENT}",
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
        let regions = countdown_while_regions(state_type.clone());
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
        let regions = swapped_reference_while_regions();
        assert_eq!(
            WhileOperation::<ArrayIrType>::new().validate_application(
                &[reference_type.clone(), reference_type],
                regions[0].entry_region_ref(),
                regions[1].entry_region_ref(),
            ),
            Err(reference_carry_swap_error()),
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
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` state 0 of type `dimension<2>` must be passed through unchanged by the body \
                 under a batched predicate, because it has no per-item value to mask",
            )))),
        );
    }

    #[test]
    fn test_while_validate_region_interfaces() {
        // Region interface validation checks the region count and the declared (unrefined) body signature, so a body
        // that maps `f64[extent]` to `f64[3]` is rejected, including through an operation family's derived dispatch
        // (the derive tests own the delegation of the other wrappers).
        let regions = refinement_only_while_regions();
        let interfaces = regions.each_ref().map(Program::interface);
        let expected = Err(ProgramError::Type(TypeError::invalid(format!(
            "`{WHILE_OPERATION_NAME}` body output type signature mismatch: expected [f64[extent]] but got [f64[3]]",
        ))));
        assert_eq!(WhileOperation::<ArrayType>::new().validate_region_interfaces(&interfaces), expected);
        assert_eq!(TestOperation::While(WhileOperation::new()).validate_region_interfaces(&interfaces), expected);
        assert_eq!(
            WhileOperation::<ArrayType>::new().validate_region_interfaces(&interfaces[..1]),
            Err(ProgramError::MalformedProgram(format!(
                "operation `{WHILE_OPERATION_NAME}` declares 2 region slots but 1 regions were attached",
            ))),
        );
    }

    #[test]
    fn test_while_validate_region_interfaces_rejects_bodies_that_only_fit_after_refinement() {
        // An `f64[3]` input refines the body's declared `f64[extent] -> f64[3]` signature to a valid one, but the
        // declared body does not map its state to itself. Builder construction, eager binding (even though the loop
        // never iterates), staging (which could otherwise specialize the body), partial evaluation, and forward-mode
        // differentiation all reject it.
        let regions = refinement_only_while_regions();
        let expected = ProgramError::Type(TypeError::invalid(format!(
            "`{WHILE_OPERATION_NAME}` body output type signature mismatch: expected [f64[extent]] but got [f64[3]]",
        )));
        let input_type = ArrayType::new_static(DataType::F64, [3]);
        let input = Array::vector(vec![1.0f64; 3]).unwrap();

        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let region_ids = regions.each_ref().map(|region| builder.import_region(region.entry_region_ref()));
        let state = builder.add_input(input_type.clone());
        assert_eq!(
            builder.add_instruction(WhileOperation::new(), region_ids.to_vec(), vec![state], None).map(|_| ()),
            Err(expected.clone()),
        );

        let context = TestEagerContext::new();
        assert_eq!(
            context.bind(WhileOperation::new(), regions.to_vec(), std::slice::from_ref(&input)),
            Err(expected.clone()),
        );

        let context = TestTracingContext::new();
        let state = context.input(input_type.clone());
        assert_eq!(context.bind(WhileOperation::new(), regions.to_vec(), &[state]).map(|_| ()), Err(expected.clone()));
        assert!(context.builder().borrow().instructions().is_empty());

        let context = PartialEvaluationContext::new(TestEagerContext::new());
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
        let interfaces = countdown_while_regions(state_type.clone()).iter().map(Program::interface).collect::<Vec<_>>();
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
        let regions = countdown_while_regions(state_type.clone());
        let condition = regions[0].interface();
        let body = regions[1].interface();
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
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` input 0 has type `f64[2]`, which does not refine its state type `f64[]`",
            ))),
        );

        // Inference rejects mismatched condition and body state signatures, non-Boolean condition outputs, conditions
        // with several outputs, and body outputs that do not match the state signature.
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = builder.add_input(ArrayType::new_static(DataType::F64, [2]));
        let zero = builder.add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let vector_body = builder.build::<Vec<Array>, Vec<Array>>(vec![zero], vec![Placeholder], vec![Placeholder]);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&state_type),
                &[condition.clone(), vector_body.unwrap().interface()],
            ),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition/body input type signature mismatch: expected [f64[2]] but got \
                 [f64[]]",
            ))),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&state_type), &[body.clone(), body.clone()]),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got `f64[]`",
            ))),
        );
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = builder.add_input(ArrayType::scalar(DataType::F64));
        let multi_output_condition = builder
            .build::<Vec<Array>, Vec<Array>>(vec![state, state], vec![Placeholder], vec![Placeholder, Placeholder])
            .unwrap();
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&state_type),
                &[multi_output_condition.interface(), body.clone()],
            ),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition must return exactly one predicate leaf but returned 2",
            ))),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&state_type), &[condition.clone(), condition]),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` body output type signature mismatch: expected [f64[]] but got [bool[]]",
            ))),
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
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` input 0 has type `f32[rows]`, which does not refine its state type \
                 `f32[extent]`",
            ))),
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
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition with a batched predicate cannot carry reference state \
                 `ref<f32[2]>` because effectful state cannot be masked per batch item; discharge the reference first",
            ))),
        );

        // The prefix requirement still holds for every array member under a batched predicate.
        let mismatched_condition_interface = RegionInterface::new(
            state_types.clone(),
            vec![ArrayIrType::Array(ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)])))],
            EffectClasses::NONE,
        );
        assert_eq!(
            operation.infer_output_types(state_types.as_slice(), &[mismatched_condition_interface, body_interface]),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition predicate shape must be a prefix of every array state shape, but \
                 predicate `bool[2]` is not a prefix of state `f32[extent]`",
            ))),
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
            operation.infer_output_types(std::slice::from_ref(&carry_type), &[condition_interface, shape_varying_body]),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` body output type signature mismatch: expected [f32[carry]] but got \
                 [f32[next]]",
            ))),
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
        let condition = countdown_while_regions(state_type.clone()).remove(0);
        let effectful_body = {
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` loop with a batched predicate must be pure because observable effects cannot \
                 be masked for finished batch items",
            ))),
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
        let error = Err(TypeError::invalid(format!(
            "`{WHILE_OPERATION_NAME}` loop with a batched predicate cannot carry deferred work because finished batch \
             items would repeat its transformation obligation",
        )));
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
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
            builder.add_input(state_type.clone());
            let predicate =
                builder.add_instruction(ZeroOperation::new(predicate_type), Vec::new(), vec![], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
            let state = builder.add_input(state_type.clone());
            builder.build::<Vec<Array>, Vec<Array>>(vec![state], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        assert_eq!(
            WhileOperation::new()
                .infer_output_types(std::slice::from_ref(&state_type), &[condition.interface(), body.interface()]),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition predicate shape must be a prefix of every array state shape, but \
                 predicate `bool[3]` is not a prefix of state `f64[2]`",
            ))),
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
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` state `f32[][sharding={{mesh<['m'=2:manual]>, []}}]` must vary over every \
                 manual axis that the predicate `bool[][sharding={{mesh<['m'=2:manual]>, [], \
                 varying_manual={{'m'}}}}]` varies over, because devices may run different numbers of iterations; \
                 insert `{PARALLEL_VARY_OPERATION_NAME}` on the initial state",
            ))),
        );
        assert_eq!(
            infer(varying_predicate.clone(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` state `f32[]` must vary over every manual axis that the predicate \
                 `bool[][sharding={{mesh<['m'=2:manual]>, [], varying_manual={{'m'}}}}]` varies over, because devices \
                 may run different numbers of iterations; insert `{PARALLEL_VARY_OPERATION_NAME}` on the initial state",
            ))),
        );
        let varying_reference = ArrayIrType::Reference(ReferenceType::new(varying(DataType::F32).unwrap()));
        assert_eq!(infer(varying_predicate.clone(), vec![varying_reference.clone()]), Ok(vec![varying_reference]));
        assert_eq!(
            infer(
                varying_predicate,
                vec![ArrayIrType::Reference(ReferenceType::new(invariant(DataType::F32).unwrap()))],
            ),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` state `f32[][sharding={{mesh<['m'=2:manual]>, []}}]` must vary over every \
                 manual axis that the predicate `bool[][sharding={{mesh<['m'=2:manual]>, [], \
                 varying_manual={{'m'}}}}]` varies over, because devices may run different numbers of iterations; \
                 insert `{PARALLEL_VARY_OPERATION_NAME}` on the initial state",
            ))),
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
        let interfaces = countdown_while_regions(state_type.clone()).iter().map(Program::interface).collect::<Vec<_>>();
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
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` input 1 has type `f32[9]`, which does not refine its state type \
                 `f32[extent]`",
            ))),
        );
        assert_eq!(
            WhileOperation::new().infer_output_types(&refined_input_types(DataType::F64, 5), interfaces.as_slice()),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` input 1 has type `f64[5]`, which does not refine its state type \
                 `f32[extent]`",
            ))),
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
    fn test_while_type_inference_falls_back_to_declared_regions_when_specialization_drops_a_refinement() {
        // The body replaces its carry with an unsharded constant, so the carry is unsharded after every iteration even
        // when it enters the loop with a sharding. The body specialized at the sharded carry no longer maps its carry
        // to itself, so staging falls back to the declared body, and the carry keeps its declared, unsharded type.
        let state_type = ArrayType::scalar(DataType::F32);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_type = state_type.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let regions = sharding_dropping_while_regions();
        let (output_type, program) = TestIrTracingContext::trace(
            |input: TestIrStagedTracer| {
                let context = input.context().clone();
                Ok(context.bind(WhileOperation::new(), regions.clone(), std::slice::from_ref(&input))?.remove(0))
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
    fn test_while_region_data_flow_joins_traced_carries_to_the_input_alias_family() {
        // A `while` declares no reference effects of its own; it states that its carry output denotes the same
        // reference as its carry input through `reference_output_identity_input` instead. Tracing honors that hook, so
        // the loop's result belongs to its input's alias family, and an access through it after the allocation is
        // frozen is reported at the access that performs it.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(scalar_type.clone()));
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(reference_type.clone());
        let predicate = condition_builder.add_constant(array(Array::scalar(false).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let carry = body_builder.add_input(reference_type);
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![carry], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let error = TestIrTracingContext::trace(
            |input: TestIrStagedTracer| {
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
            ProgramError::MalformedProgram(format!(
                "`{REFERENCE_READ_OPERATION_NAME}` reads a reference whose alias family \
                 `{REFERENCE_FREEZE_OPERATION_NAME}` already consumed",
            )),
        );
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
            Err(ProgramError::MalformedProgram(format!(
                "operation `{WHILE_OPERATION_NAME}` declares input 2 of region 1 with 2 inputs",
            ))),
        );

        // The rule itself rejects a boundary that does not share one state signature across the instruction and both
        // regions: a missing region, a condition or a body whose inputs do not match the instruction inputs.
        let malformed = Err(ProgramError::MalformedProgram(format!(
            "`{WHILE_OPERATION_NAME}` region data flow requires matching state boundaries and one condition output",
        )));
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
            Err(ProgramError::MalformedProgram(format!(
                "operation `{WHILE_OPERATION_NAME}` declares output 2 with 2 outputs",
            ))),
        );

        // The rule itself rejects an instruction whose outputs do not match its inputs and a body whose outputs do not
        // match the instruction outputs.
        let malformed = Err(ProgramError::MalformedProgram(format!(
            "`{WHILE_OPERATION_NAME}` region data flow requires matching state boundaries and one condition output",
        )));
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
                .execution_demands(&[false, false], boundary, &mut liveness),
            Ok(vec![Some(vec![true]), Some(vec![true, true])]),
        );
        assert_eq!(
            operation.region_data_flow().execution_demands(&operation, &[true], boundary, &mut liveness),
            Err(ProgramError::MalformedProgram(format!(
                "operation `{WHILE_OPERATION_NAME}` receives used outputs for 1 positions but has 2 outputs",
            ))),
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
            Err(ProgramError::MalformedProgram(format!(
                "`{WHILE_OPERATION_NAME}` region data flow requires matching state boundaries and one condition output",
            ))),
        );
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
        let step = body_builder.add_constant(array(Array::scalar(-1.0f32).unwrap()));
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
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), None);
        assert_eq!(
            discharged
                .program()
                .interpret(vec![array(Array::scalar(2.0f32).unwrap()), array(Array::scalar(5.0f32).unwrap())]),
            Ok(vec![array(Array::scalar(2.0f32).unwrap())]),
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
        let limit = condition_builder.add_constant(array(Array::scalar::<f32>(3.0).unwrap()));
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
        let inputs = vec![array(Array::scalar::<f32>(0.0).unwrap()), array(Array::scalar::<f32>(2.0).unwrap())];
        let expected = Ok(vec![array(Array::scalar::<f32>(4.0).unwrap()), array(Array::scalar::<f32>(2.0).unwrap())]);
        assert_eq!(source.interpret(inputs.clone()), expected);
        assert_eq!(discharged.program().interpret(inputs), expected);
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
        let condition = condition_builder.add_constant(array(Array::scalar(false).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![condition], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(array(Array::scalar::<f32>(1.0).unwrap()));
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
            discharged.program().interpret(vec![array(Array::scalar::<f32>(2.0).unwrap())]),
            Ok(vec![array(Array::scalar::<f32>(2.0).unwrap()), array(Array::scalar::<f32>(2.0).unwrap())]),
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
        let condition = condition_builder.add_constant(array(Array::scalar(true).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![condition], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first_reference = body_builder.add_input(reference_type.clone().into());
        let second_reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(array(Array::scalar::<f32>(1.0).unwrap()));
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
            discharged.program().interpret(vec![array(Array::scalar::<f32>(2.0).unwrap())]),
            Ok(vec![
                array(Array::scalar::<f32>(3.0).unwrap()),
                array(Array::scalar::<f32>(3.0).unwrap()),
                array(Array::scalar::<f32>(3.0).unwrap()),
            ]),
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
        let predicate = condition_builder.add_constant(array(Array::scalar(true).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(array(Array::scalar::<f32>(2.0).unwrap()));
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
            let expected = source.interpret(vec![array(Array::scalar::<f32>(initial).unwrap())]).unwrap();
            assert_eq!(
                discharged.program().interpret(vec![array(Array::scalar::<f32>(initial).unwrap())]),
                Ok(expected),
            );
        }
        assert_eq!(
            discharged.program().interpret(vec![array(Array::scalar::<f32>(1.0).unwrap())]),
            Ok(vec![array(Array::scalar::<f32>(7.0).unwrap())]),
        );
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
        let predicate = condition_builder.add_constant(array(Array::scalar(true).unwrap()));
        let loop_condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(array(Array::scalar::<f32>(1.0).unwrap()));
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
            discharged
                .program()
                .interpret(vec![array(Array::scalar(true).unwrap()), array(Array::scalar::<f32>(3.0).unwrap())]),
            Ok(vec![array(Array::scalar::<f32>(5.0).unwrap())]),
        );
        assert_eq!(
            discharged
                .program()
                .interpret(vec![array(Array::scalar(false).unwrap()), array(Array::scalar::<f32>(3.0).unwrap())]),
            Ok(vec![array(Array::scalar::<f32>(3.0).unwrap())]),
        );
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
            vec![TestIrValue::Reference(concrete_reference), array(Array::scalar(false).unwrap())],
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
            Ok(vec![array(Array::scalar(2f32).unwrap())]),
        );
        assert_eq!(initial.read().unwrap(), Array::scalar(2f32).unwrap());
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
        let predicate = condition_builder.add_constant(array(Array::scalar(true).unwrap()));
        let consuming_condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let source = reference_while_program(consuming_condition, None);
        let condition = source.entry_region_ref().instructions()[0].regions()[0];
        assert_eq!(
            source.discharge_references(0).unwrap_err(),
            ReferenceAnalysisError::ExternalReferenceConsumption {
                operation: REFERENCE_FREEZE_OPERATION_NAME,
                instruction: InstructionId::new(condition, 0),
                root: ReferenceRoot::RegionInput { region: condition, input_index: 0 },
                external_source: ReferenceSource::Input { index: 0 },
            }
            .into(),
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
        let update = body_builder.add_constant(array(Array::scalar::<f32>(10.0).unwrap()));
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
            discharged.program().interpret(vec![array(Array::scalar::<f32>(0.0).unwrap())]),
            Ok(vec![array(Array::scalar::<f32>(1.0).unwrap()), array(Array::scalar::<f32>(1.0).unwrap())]),
        );
        assert_eq!(
            discharged.program().interpret(vec![array(Array::scalar::<f32>(-5.0).unwrap())]),
            Ok(vec![array(Array::scalar::<f32>(7.0).unwrap()), array(Array::scalar::<f32>(7.0).unwrap())]),
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
        let step = oracle_body_builder.add_constant(array(Array::scalar::<f32>(1.0).unwrap()));
        let counter = oracle_body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![counter, step], None)
            .unwrap()[0];
        let limit = oracle_body_builder.add_constant(array(Array::scalar::<f32>(5.0).unwrap()));
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
        let step = oracle_builder.add_constant(array(Array::scalar::<f32>(1.0).unwrap()));
        let counter = oracle_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![initial, step], None)
            .unwrap()[0];
        let limit = oracle_builder.add_constant(array(Array::scalar::<f32>(5.0).unwrap()));
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
                .interpret(vec![
                    array(Array::scalar::<f32>(initial).unwrap()),
                    array(Array::scalar::<f32>(1.0).unwrap()),
                ])
                .unwrap();
            assert_eq!(
                discharged.program().interpret(vec![
                    array(Array::scalar::<f32>(initial).unwrap()),
                    array(Array::scalar::<f32>(1.0).unwrap()),
                ]),
                Ok(expected),
            );
        }
        assert_eq!(
            discharged
                .program()
                .interpret(vec![array(Array::scalar::<f32>(0.0).unwrap()), array(Array::scalar::<f32>(1.0).unwrap())]),
            Ok(vec![array(Array::scalar::<f32>(5.0).unwrap()), array(Array::scalar::<f32>(16.0).unwrap())]),
        );
        assert_eq!(
            discharged
                .program()
                .interpret(vec![array(Array::scalar::<f32>(7.0).unwrap()), array(Array::scalar::<f32>(1.0).unwrap())]),
            Ok(vec![array(Array::scalar::<f32>(8.0).unwrap()), array(Array::scalar::<f32>(1.0).unwrap())]),
        );
    }

    #[test]
    fn test_while_reference_discharge_rotates_a_root_written_by_both_condition_and_body() {
        // The body doubles the carried counter and the condition increments it, so the rotation's fixed body-then-
        // condition order is observable in the final value. The oracle is the hand-rotated reference-free loop.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let condition = builder.import_program(incrementing_condition(10.0));
        let body = builder.import_program(doubling_reference_body());
        let reference = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
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
        let step = oracle_body_builder.add_constant(array(Array::scalar::<f32>(1.0).unwrap()));
        let counter = oracle_body_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![doubled, step], None)
            .unwrap()[0];
        let limit = oracle_body_builder.add_constant(array(Array::scalar::<f32>(10.0).unwrap()));
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
        let step = oracle_builder.add_constant(array(Array::scalar::<f32>(1.0).unwrap()));
        let counter = oracle_builder
            .add_instruction(AddOperation::<ArrayType>::new(), Vec::new(), vec![initial, step], None)
            .unwrap()[0];
        let limit = oracle_builder.add_constant(array(Array::scalar::<f32>(10.0).unwrap()));
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
            let expected = oracle.interpret(vec![array(Array::scalar::<f32>(initial).unwrap())]).unwrap();
            assert_eq!(
                discharged.program().interpret(vec![array(Array::scalar::<f32>(initial).unwrap())]),
                Ok(expected),
            );
        }
        assert_eq!(
            discharged.program().interpret(vec![array(Array::scalar::<f32>(0.0).unwrap())]),
            Ok(vec![array(Array::scalar::<f32>(15.0).unwrap()), array(Array::scalar::<f32>(15.0).unwrap())]),
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
                TestIrValue::Reference(ArrayReference::new(Array::scalar(4.0f32).unwrap())),
                array(Array::scalar::<f32>(1.0).unwrap()),
                array(Array::scalar(false).unwrap()),
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
        let update = condition_builder.add_constant(array(Array::scalar::<f32>(1.0).unwrap()));
        condition_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![first_reference, update], None)
            .unwrap();
        let counter = condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second_reference], None)
            .unwrap()[0];
        let limit = condition_builder.add_constant(array(Array::scalar::<f32>(3.0).unwrap()));
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
            discharged.program().interpret(vec![array(Array::scalar::<f32>(0.0).unwrap())]),
            Ok(vec![
                array(Array::scalar::<f32>(3.0).unwrap()),
                array(Array::scalar::<f32>(3.0).unwrap()),
                array(Array::scalar::<f32>(3.0).unwrap()),
            ]),
        );
        assert_eq!(
            discharged.program().interpret(vec![array(Array::scalar::<f32>(7.0).unwrap())]),
            Ok(vec![
                array(Array::scalar::<f32>(8.0).unwrap()),
                array(Array::scalar::<f32>(8.0).unwrap()),
                array(Array::scalar::<f32>(8.0).unwrap()),
            ]),
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
    fn test_while_reference_discharge_rotates_a_bounded_mutating_condition() {
        // A bounded loop evaluates its condition exactly `bound` times when the bound truncates it, whereas every
        // rotated iteration evaluates the condition after the body. The rotated loop therefore runs one iteration
        // fewer and the last body application runs after the loop under the final predicate; a bound of one leaves
        // no loop at all. The body doubles the counter that the condition increments before testing it against 10,
        // so from 0 the source tests 1, 3, 7, and 15 and doubles to 2, 6, and 14: a bound of 1 truncates it at 2, a
        // bound of 2 truncates it at 6, and a bound of 5 lets it exit at 15. A trailing step that skipped or repeated
        // the last body application would change each of these values.
        let source = |bound: usize| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let regions = vec![
                builder.import_program(incrementing_condition(10.0)),
                builder.import_program(doubling_reference_body()),
            ];
            let reference = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
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
                Ok(vec![array(Array::scalar(expected).unwrap())]),
            );
            assert_eq!(initial.read().unwrap(), Array::scalar(expected).unwrap());
            let discharged = source.discharge_references(0).unwrap();
            assert_eq!(
                discharged.program().interpret(vec![array(Array::scalar::<f32>(0.0).unwrap())]),
                Ok(vec![array(Array::scalar(expected).unwrap()), array(Array::scalar(expected).unwrap())]),
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
    fn test_while_reference_discharge_forwards_passed_through_state_around_the_bounded_rotation_tail() {
        // Inside a manual region, the loop of the preceding test carries a first-class dimension next to its counter
        // reference, and its predicate varies over the manual axis because the counter does. A `while` carry accepts a
        // dimension under a varying predicate, but a `condition` output does not, so the trailing body application of
        // the bounded rotation forwards the state that the body passes through (the dimension) around its `condition`
        // and selects only the discharged counter state. From 0 the source tests 1 and 3 and doubles to 2 and 6 before
        // the bound of two truncates it, and the discharged program computes the same values.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let value_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let reference_type: ArrayIrType = ReferenceType::new(value_type.clone()).into();
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone());
        condition_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let one = condition_builder.add_constant(array(Array::from_elements(value_type.clone(), &[1f32]).unwrap()));
        condition_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, one], None)
            .unwrap();
        let counter = condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let limit = condition_builder.add_constant(array(Array::from_elements(value_type.clone(), &[10f32]).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::from(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![counter, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.clone());
        let extent = body_builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
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
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![reference, extent],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![builder.import_program(condition), builder.import_program(body)];
        let reference = builder.add_input(reference_type);
        let extent = builder.add_input(ArrayIrType::Dimension(extent_type.clone()));
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
        let outputs = builder.add_instruction(operation, regions, vec![reference, extent], None).unwrap().to_vec();
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[0]], None).unwrap()[0];
        let source = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![value, outputs[1]],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let initial = ArrayReference::new(Array::from_elements(value_type.clone(), &[0f32]).unwrap());
        let expected = array(Array::from_elements(value_type.clone(), &[6f32]).unwrap());
        assert_eq!(
            source.interpret(vec![TestIrValue::Reference(initial.clone()), dimension(&extent_type, 3)]),
            Ok(vec![expected.clone(), dimension(&extent_type, 3)]),
        );
        let discharged = source.discharge_references(0).unwrap();
        let varying_scalar = "f32[][sharding={mesh<['m'=2:manual]>, [], varying_manual={'m'}}]";
        let varying_predicate = "bool[][sharding={mesh<['m'=2:manual]>, [], varying_manual={'m'}}]";
        assert_eq!(
            discharged.program().to_string(),
            formatdoc! {"
                lambda %0:{varying_scalar}, %1:dimension<extent ∈ [1, 8)> .
                let %2:{varying_scalar} = const 1.0
                    %3:{varying_scalar} = const 10.0
                    %4:{varying_scalar} = add %0 %2
                    %5:{varying_predicate} = compare [direction=LessThan] %4 %3
                    %6:{varying_scalar}, %7:dimension<extent ∈ [1, 8)>, %8:{varying_predicate} = while \
                    [iteration_bound=1] %4 %1 %5 [
                        condition={{
                            lambda %0:{varying_scalar}, %1:dimension<extent ∈ [1, 8)>, %2:{varying_predicate} .
                            in (%2)
                        }},
                        body={{
                            lambda %0:{varying_scalar}, %1:dimension<extent ∈ [1, 8)>, %2:{varying_predicate} .
                            let %3:{varying_scalar} = add %0 %0
                                %4:{varying_scalar} = const 1.0
                                %5:{varying_scalar} = const 10.0
                                %6:{varying_scalar} = add %3 %4
                                %7:{varying_predicate} = compare [direction=LessThan] %6 %5
                            in (%6, %1, %7)
                        }},
                    ]
                    %9:{varying_scalar} = condition %8 %6 %7 [
                        true={{
                            lambda %0:{varying_scalar}, %1:dimension<extent ∈ [1, 8)> .
                            let %2:{varying_scalar} = add %0 %0
                            in (%2)
                        }},
                        false={{
                            lambda %0:{varying_scalar}, %1:dimension<extent ∈ [1, 8)> .
                            in (%0)
                        }},
                    ]
                in (%9, %7, %9)
            "}
            .trim_end(),
        );
        let inputs = vec![array(Array::from_elements(value_type, &[0f32]).unwrap()), dimension(&extent_type, 3)];
        assert_eq!(
            discharged.program().interpret(inputs),
            Ok(vec![expected.clone(), dimension(&extent_type, 3), expected]),
        );
    }

    #[test]
    fn test_while_reference_discharge_omits_the_bounded_rotation_tail_when_the_body_forwards_every_position() {
        // The condition increments the counter before testing it against 3, and the body passes the counter reference
        // through unchanged. The rebuilt body therefore forwards every state position and has no effects, so the
        // trailing body application of the bounded rotation would select the same state on both sides and no
        // `condition` is bound after the loop. From 0 the source tests 1, 2, and 3: a bound of 1 truncates it at 1, a
        // bound of 2 truncates it at 2, and a bound of 5 lets it exit at 3.
        for (bound, expected) in [(1, 1f32), (2, 2.0), (5, 3.0)] {
            let source = reference_while_program(incrementing_condition(3.0), Some(bound));
            let initial = ArrayReference::new(Array::scalar(0f32).unwrap());
            assert_eq!(
                source.interpret(vec![TestIrValue::Reference(initial.clone())]),
                Ok(vec![array(Array::scalar(expected).unwrap())]),
            );
            assert_eq!(initial.read().unwrap(), Array::scalar(expected).unwrap());
            let discharged = source.discharge_references(0).unwrap();
            assert_eq!(
                discharged.program().interpret(vec![array(Array::scalar(0f32).unwrap())]),
                Ok(vec![array(Array::scalar(expected).unwrap()), array(Array::scalar(expected).unwrap())]),
            );
        }
        assert_eq!(
            reference_while_program(incrementing_condition(3.0), Some(2))
                .discharge_references(0)
                .unwrap()
                .program()
                .to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = const 1.0
                    %2:f32[] = const 3.0
                    %3:f32[] = add %0 %1
                    %4:bool[] = compare [direction=LessThan] %3 %2
                    %5:f32[], %6:bool[] = while [iteration_bound=1] %3 %4 [
                        condition={
                            lambda %0:f32[], %1:bool[] .
                            in (%1)
                        },
                        body={
                            lambda %0:f32[], %1:bool[] .
                            let %2:f32[] = const 1.0
                                %3:f32[] = const 3.0
                                %4:f32[] = add %0 %2
                                %5:bool[] = compare [direction=LessThan] %4 %3
                            in (%4, %5)
                        },
                    ]
                in (%5, %5)"},
        );
    }

    #[test]
    fn test_while_reference_discharge_binds_an_effect_only_bounded_rotation_tail_for_preserved_writes() {
        // The condition increments a discharged counter before testing it against 3, and the body only increments a
        // preserved reference, passing both carries through. The rebuilt body therefore forwards every state position,
        // but its preserved write must still run once more after the rotated loop under the final predicate, so the
        // tail `condition` is bound with no outputs for that effect alone. From counters `[0, 10]` the source tests 1
        // and 2 and increments the preserved reference twice before a bound of 2 truncates it, whereas a bound of 1
        // truncates it after one increment, and a bound of 5 exits when it tests 3, after two increments, so the tail
        // takes its `false` branch and writes nothing.
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = condition_builder.add_input(reference_type.clone());
        condition_builder.add_input(reference_type.clone());
        let one = condition_builder.add_constant(array(Array::scalar(1.0f32).unwrap()));
        condition_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![counter, one], None)
            .unwrap();
        let value = condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![counter], None)
            .unwrap()[0];
        let limit = condition_builder.add_constant(array(Array::scalar(3.0f32).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![value, limit],
                None,
            )
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = body_builder.add_input(reference_type.clone());
        let preserved = body_builder.add_input(reference_type);
        let one = body_builder.add_constant(array(Array::scalar(1.0f32).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![preserved, one], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![counter, preserved],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let source = |bound: usize| {
            let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
            let regions = vec![builder.import_program(condition.clone()), builder.import_program(body.clone())];
            let initial =
                (0..2).map(|_| builder.add_input(ArrayType::scalar(DataType::F32).into())).collect::<Vec<_>>();
            let references = initial
                .into_iter()
                .map(|initial| {
                    builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0]
                })
                .collect::<Vec<_>>();
            let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(bound).unwrap();
            let outputs = builder.add_instruction(operation, regions, references, None).unwrap().to_vec();
            let values = outputs
                .into_iter()
                .map(|output| {
                    builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![output], None).unwrap()[0]
                })
                .collect::<Vec<_>>();
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(values, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let partially_discharge = |source: &TestIrProgram| {
            let targets = source.reference_discharge_targets(0).unwrap();
            source.clone().partially_discharge_references(0, &targets[..1]).unwrap()
        };

        let inputs = vec![array(Array::scalar(0f32).unwrap()), array(Array::scalar(10f32).unwrap())];
        for (bound, counter, preserved) in [(1, 1f32, 11f32), (2, 2.0, 12.0), (5, 3.0, 12.0)] {
            let source = source(bound);
            let expected = Ok(vec![array(Array::scalar(counter).unwrap()), array(Array::scalar(preserved).unwrap())]);
            assert_eq!(source.interpret(inputs.clone()), expected);
            let discharged = partially_discharge(&source);
            assert_eq!(discharged.external_reference_bindings(), &[]);
            assert_eq!(discharged.program().interpret(inputs.clone()), expected);
        }
        assert_eq!(
            partially_discharge(&source(1)).program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:ref<f32[]> = reference_new %1
                    %3:f32[] = const 1.0
                    %4:f32[] = const 3.0
                    %5:f32[] = add %0 %3
                    %6:bool[] = compare [direction=LessThan] %5 %4
                    () = condition %6 %5 %2 [
                        true={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = const 1.0
                                () = reference_add_update %1 %2
                            in ()
                        },
                        false={
                            lambda %0:f32[], %1:ref<f32[]> .
                            in ()
                        },
                    ]
                    %7:f32[] = reference_freeze %2
                in (%5, %7)"},
        );
        assert_eq!(
            partially_discharge(&source(2)).program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:ref<f32[]> = reference_new %1
                    %3:f32[] = const 1.0
                    %4:f32[] = const 3.0
                    %5:f32[] = add %0 %3
                    %6:bool[] = compare [direction=LessThan] %5 %4
                    %7:f32[], %8:ref<f32[]>, %9:bool[] = while [iteration_bound=1] %5 %2 %6 [
                        condition={
                            lambda %0:f32[], %1:ref<f32[]>, %2:bool[] .
                            in (%2)
                        },
                        body={
                            lambda %0:f32[], %1:ref<f32[]>, %2:bool[] .
                            let %3:f32[] = const 1.0
                                %4:f32[] = const 1.0
                                %5:f32[] = const 3.0
                                () = reference_add_update %1 %3
                                %6:f32[] = add %0 %4
                                %7:bool[] = compare [direction=LessThan] %6 %5
                            in (%6, %1, %7)
                        },
                    ]
                    () = condition %9 %7 %8 [
                        true={
                            lambda %0:f32[], %1:ref<f32[]> .
                            let %2:f32[] = const 1.0
                                () = reference_add_update %1 %2
                            in ()
                        },
                        false={
                            lambda %0:f32[], %1:ref<f32[]> .
                            in ()
                        },
                    ]
                    %10:f32[] = reference_freeze %2
                in (%7, %10)"},
        );
    }

    #[test]
    fn test_while_reference_discharge_rotated_loops_support_jvp_and_batching() {
        // Rotation threads the predicate as a trailing Boolean carry, which both transformations of the discharged loop
        // keep: it has no tangent in forward mode, and it becomes a batched predicate under batching. The body doubles
        // a local counter that the condition increments before testing it against 10, so the loop computes
        // `f(x) = 8 x + 15` from `x = 0.5` (testing 1.5, 4, 9, and 19) and `f(x) = 2 x + 3` from `x = 5` (testing 6
        // and 13).
        let discharged = doubling_reference_while_program(None)
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
    fn test_while_reference_discharge_rotated_bounded_loops_support_jvp_and_batching() {
        // A bounded rotated loop ends with the trailing body application under the final predicate, which both
        // transformations of the discharged loop must keep. The body doubles a local counter that the condition
        // increments before testing it against 10, under a bound of 2. From `x = 0.5` the source tests 1.5 and 4 and
        // the bound truncates it after its second doubling, so it computes `f(x) = 4 x + 6 = 8`. From `x = 5` it tests
        // 6 and 13 and exits after one doubling, so it computes `f(x) = 2 x + 3 = 13`.
        let source = doubling_reference_while_program(Some(2));
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

        // The staged forward mode of the bounded rotated loop stores the predicate as residual state and differentiates
        // the trailing step under the final predicate.
        let jvp = discharged.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = const 1.0
                    %3:f32[] = const 10.0
                    %4:f32[] = add %0 %2
                    %5:bool[] = compare [direction=LessThan] %4 %3
                    %6:i64[] = zero [type=i64[]]
                    %7:bool[1] = zero [type=bool[1]]
                    %8:f32[], %9:bool[], %10:i64[], %11:bool[1] = while [iteration_bound=1] %4 %5 %6 %7 [
                        condition={
                            lambda %0:f32[], %1:bool[], %2:i64[], %3:bool[1] .
                            in (%1)
                        },
                        body={
                            lambda %0:f32[], %1:bool[], %2:i64[], %3:bool[1] .
                            let %4:f32[] = const 1.0
                                %5:f32[] = const 10.0
                                %6:f32[] = add %0 %0
                                %7:f32[] = add %6 %4
                                %8:bool[] = compare [direction=LessThan] %7 %5
                                %9:bool[] = one [type=bool[]]
                                %10:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %9
                                %11:bool[1] = dynamic_update_slice %3 %10 %2
                                %12:i64[] = one [type=i64[]]
                                %13:i64[] = add %2 %12
                            in (%7, %8, %13, %11)
                        },
                    ]
                    %12:f32[] = scan [carry_count=1, iteration_count=1, reverse=false] %1 %11 [
                        body={
                            lambda %0:i64[], %1:f32[], %2:bool[] .
                            let %3:f32[] = condition %2 %1 [
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
                            in (%3)
                        },
                    ]
                    %13:f32[], %14:f32[] = condition %9 %8 %12 [
                        true={
                            lambda %0:f32[], %1:f32[] .
                            let %2:f32[] = add %0 %0
                                %3:f32[] = add %1 %1
                            in (%2, %3)
                        },
                        false={
                            lambda %0:f32[], %1:f32[] .
                            in (%0, %1)
                        },
                    ]
                in (%13, %14)"},
        );
        for (x, expected_primal, expected_tangent) in [(0.5f32, 8.0f32, 4.0f32), (5.0, 13.0, 2.0)] {
            assert_eq!(
                jvp.interpret(vec![array(Array::scalar(x).unwrap()), array(Array::scalar(1.0f32).unwrap())]),
                Ok(vec![
                    array(Array::scalar(expected_primal).unwrap()),
                    array(Array::scalar(expected_tangent).unwrap()),
                ]),
            );
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
    fn test_while_interpretation() {
        // Eager interpretation applies the body while the condition holds, and returns the initial state unchanged when
        // the condition is false on entry.
        let context = TestEagerContext::new();
        let regions = countdown_while_regions(ArrayType::scalar(DataType::F64));
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
        let regions = countdown_while_regions(ArrayType::new_static(DataType::F64, [3]));
        let operation = WhileOperation::new();
        let context = TestEagerContext::new();
        assert_eq!(
            context.bind(operation, regions.clone(), &[Array::vector(vec![3.0, 1.0, 2.0]).unwrap()]),
            Ok(vec![Array::vector(vec![0.0, 0.0, 0.0]).unwrap()]),
        );

        // The semantic iteration bound truncates the shared masked iterations: item 0 stops at 1.0 after two body
        // applications, item 1 finishes on its own predicate after one, and item 2 reaches 0 exactly at the bound.
        let bounded = operation.with_iteration_bound(2).unwrap();
        assert_eq!(
            context.bind(bounded, regions, &[Array::vector(vec![3.0, 1.0, 2.0]).unwrap()]),
            Ok(vec![Array::vector(vec![1.0, 0.0, 0.0]).unwrap()]),
        );
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
        let context = TestIrEagerContext::new();
        let inputs = [dimension(&dimension_type, 2), array(Array::vector(vec![0f32, 2.0]).unwrap())];
        assert_eq!(
            context.bind(WhileOperation::<ArrayIrType>::new(), vec![condition.clone(), body(false)], &inputs),
            Ok(vec![dimension(&dimension_type, 2), array(Array::vector(vec![3f32, 3.0]).unwrap())]),
        );

        // A body that replaces the dimension carry, even by an equal constant, does not pass it through, so the
        // application is rejected before the first iteration (refer to `WhileOperation::validate_application`).
        assert_eq!(
            context.bind(WhileOperation::<ArrayIrType>::new(), vec![condition, body(true)], &inputs),
            Err(ProgramError::from(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` state 0 of type `dimension<2>` must be passed through unchanged by the body \
                 under a batched predicate, because it has no per-item value to mask",
            )))),
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
            let one = builder.add_constant(array(Array::vector(vec![1.0f32, 1.0]).unwrap()));
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![one], None).unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, state], None)
                .unwrap();
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let done = builder.add_constant(array(Array::scalar(false).unwrap()));
            builder
                .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                    vec![state, second, third, done],
                    vec![Placeholder; 4],
                    vec![Placeholder; 4],
                )
                .unwrap()
        };
        let program = while_program(
            WhileOperation::new(),
            &[condition, body],
            vec![array_type.into(), boolean_type.clone().into(), boolean_type.clone().into(), boolean_type.into()],
        );
        let predicate = |value| array(Array::scalar(value).unwrap());
        assert_eq!(
            program.interpret(vec![
                array(Array::vector(vec![1.0f32, 2.0]).unwrap()),
                predicate(true),
                predicate(true),
                predicate(true),
            ]),
            Ok(vec![
                array(Array::vector(vec![4.0f32, 5.0]).unwrap()),
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
        let regions = swapped_reference_while_regions();
        let context = TestIrEagerContext::new();
        let first = TestIrValue::Reference(ArrayReference::new(Array::scalar(1f32).unwrap()));
        let second = TestIrValue::Reference(ArrayReference::new(Array::scalar(2f32).unwrap()));
        assert_eq!(
            context.bind(WhileOperation::<ArrayIrType>::new(), regions.clone(), &[first.clone(), second]),
            Err(reference_carry_swap_error()),
        );
        assert_eq!(
            context.bind(WhileOperation::<ArrayIrType>::new(), regions.clone(), &[first.clone(), first]),
            Err(reference_carry_swap_error()),
        );

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = regions.into_iter().map(|region| builder.import_program(region)).collect::<Vec<_>>();
        let first = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let second = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let first = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        let second = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![second], None).unwrap()[0];
        let outputs = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), regions, vec![first, second], None)
            .unwrap()
            .to_vec();
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[0]], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(program.discharge_references(0).map(|_| ()), Err(reference_carry_swap_error()));
    }

    #[test]
    fn test_while_partial_evaluation() {
        // A loop-invariant known state element folds into the condition and the body: the residual while keeps the same
        // state set (so its output arity is preserved), and every subcomputation that depended only on the known
        // element collapses to a constant. The loop carries `[counter, accumulator, k]`, runs while `counter > 0`, and
        // its body returns `[counter - 1, accumulator + k * k, k]`. With `k` known (`3`) and `counter` and
        // `accumulator` unknown, the body passes `k` through unchanged, so `k * k` folds to `9`, the final `k` is
        // rebuilt as the constant `3`, and the bound carries over to the residual while.
        let scalar = || ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
            let counter = builder.add_input(scalar());
            let accumulator = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
            let next_counter =
                builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
            let k_squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![k, k], None).unwrap()[0];
            let next_accumulator = builder
                .add_instruction(AddOperation::new(), Vec::new(), vec![accumulator, k_squared], None)
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(
                    vec![next_counter, next_accumulator, k],
                    vec![Placeholder; 3],
                    vec![Placeholder; 3],
                )
                .unwrap()
        };
        let operation = WhileOperation::new().with_iteration_bound(8).unwrap();
        let program = while_program(operation, &[condition, body], vec![scalar(); 3]);

        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(scalar()),
                PartialValue::Unknown(scalar()),
                PartialValue::Known(Array::scalar(3.0).unwrap()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
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

        // The loop runs four times (counter `4 -> 0`), so `accumulator` threads `1 -> 10 -> 19 -> 28 -> 37` and the
        // loop-invariant `k` stays `3`, exactly as the source program computes.
        let expected = vec![Array::scalar(0.0).unwrap(), Array::scalar(37.0).unwrap(), Array::scalar(3.0).unwrap()];
        let inputs = vec![Array::scalar(4.0).unwrap(), Array::scalar(1.0).unwrap(), Array::scalar(3.0).unwrap()];
        assert_eq!(program.interpret(inputs), Ok(expected.clone()));
        assert_eq!(
            evaluation.interpret(&TestEagerContext::new(), &[Array::scalar(4.0).unwrap(), Array::scalar(1.0).unwrap()]),
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
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
            let counter = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let one = builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
            let next_counter =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next_counter, k], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let program = while_program(WhileOperation::new(), &[condition, body], vec![scalar(); 2]);

        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(scalar()), PartialValue::Known(Array::scalar(3.0).unwrap())])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
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
        assert_eq!(evaluation.interpret(&TestEagerContext::new(), &[Array::scalar(0.0).unwrap()]), Ok(expected));
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

        let program = while_program(WhileOperation::new(), &[condition, body], state_types.clone());

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
            evaluation.program().to_string(),
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
                &TestIrEagerContext::new(),
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

        let program = while_program(WhileOperation::new(), &[condition, body], state_types.clone());

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
            evaluation.program().to_string(),
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
            evaluation.interpret(&TestIrEagerContext::new(), &[array(Array::scalar(3.0).unwrap())]),
            Ok(expected),
        );
    }

    #[test]
    fn test_while_partial_evaluation_preserves_signed_zero_state() {
        // Body `[counter, value] -> [counter - 1, -value]` over an unknown counter and a known value of `0.0`. Negating
        // `0.0` yields `-0.0`, which compares equal to `0.0`, so a value-based invariance check would wrongly fold the
        // value to its initial `0.0`. After an odd number of iterations, the final value must be `-0.0`.
        let scalar = || ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let program = while_program(WhileOperation::new(), &[condition, body], vec![scalar(), scalar()]);

        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(scalar()), PartialValue::Known(Array::scalar(0.0f64).unwrap())])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
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
        let outputs = evaluation.interpret(&TestEagerContext::new(), &[Array::scalar(3.0f64).unwrap()]).unwrap();
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
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
            let x = builder.add_input(state_type.clone());
            let k = builder.add_input(state_type.clone());
            let one = builder.add_constant(Array::from_elements::<i32>(state_type.clone(), &[1]).unwrap());
            let inverse = builder.add_instruction(DivOperation::new(), Vec::new(), vec![one, k], None).unwrap()[0];
            let next_x = builder.add_instruction(AddOperation::new(), Vec::new(), vec![x, inverse], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next_x, k], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let program =
            while_program(WhileOperation::new(), &[condition, body], vec![state_type.clone(), state_type.clone()]);

        let knowledge = vec![
            PartialValue::Unknown(state_type.clone()),
            PartialValue::Known(Array::from_elements::<i32>(state_type, &[0]).unwrap()),
        ];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
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
        let context = TestEagerContext::new();
        let scalar = |value: i32| Array::from_elements::<i32>(ArrayType::scalar(DataType::I32), &[value]).unwrap();
        assert_eq!(program.interpret(vec![scalar(-1), scalar(0)]), Ok(vec![scalar(-1), scalar(0)]));
        assert_eq!(evaluation.interpret(&context, &[scalar(-1)]), Ok(vec![scalar(-1), scalar(0)]));
        assert_eq!(program.interpret(vec![scalar(2), scalar(0)]), Ok(vec![scalar(0), scalar(0)]));
        assert_eq!(evaluation.interpret(&context, &[scalar(2)]), Ok(vec![scalar(0), scalar(0)]));
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
        let context = TestIrEagerContext::new();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(counter.clone()), PartialValue::Unknown(static_vector_type)])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
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
            evaluation.program().to_string(),
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
    fn test_while_partial_evaluation_residualizes_reference_carry_whole() {
        // `f(n, r) = { while n > 0 { add_update(r, 1); n -= 1 }; read(r) }` with the reference carried through the
        // loop.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let counter = condition_builder.add_input(scalar_type.clone().into());
        condition_builder.add_input(reference_type.clone().into());
        let zero = condition_builder.add_constant(array(Array::scalar(0.0f32).unwrap()));
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
        let one = body_builder.add_constant(array(Array::scalar(1.0f32).unwrap()));
        let step = body_builder.add_constant(array(Array::scalar(-1.0f32).unwrap()));
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
            evaluation.interpret(&TestIrEagerContext::new(), &[array(Array::scalar(3.0f32).unwrap())]),
            Ok(vec![array(Array::scalar(5.0f32).unwrap())]),
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
        let program =
            while_program(WhileOperation::new(), &[condition, body], vec![scalar_type.clone(), reference_type]);
        let live = ArrayReference::new(Array::vector(vec![1f32, 2.0, 3.0]).unwrap());
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
        assert_eq!(live.read(), Ok(Array::vector(vec![1f32, 2.0, 3.0]).unwrap()));
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(
            evaluation.interpret(&TestIrEagerContext::new(), &[array(Array::scalar(2f32).unwrap())]),
            Ok(vec![array(Array::scalar(0f32).unwrap()), TestIrValue::Reference(live.clone())]),
        );
        assert_eq!(live.read(), Ok(Array::vector(vec![4f32, 8.0, 12.0]).unwrap()));
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

        let program = while_program(WhileOperation::new(), &[condition, body], state_types.clone());

        let reference = TestIrValue::Reference(ArrayReference::new(Array::scalar(5.0f32).unwrap()));
        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(scalar_type.into()), PartialValue::Known(reference.clone())])
            .unwrap();

        // The residual regions are the source regions: the reference still flows from its state input to its state
        // output.
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
                            %3:f32[] = sub %0 %2
                        in (%3, %1)
                    },
                ]
                in (%2, %3)"},
        );
    }

    #[test]
    fn test_while_partial_evaluation_residualizes_effectful_loops_unchanged() {
        // The invariance probes fold the regions through the live known-side context, so for an effectful loop they
        // would run its effects once more, and the closed-knownness split would run the known part of every iteration
        // a second time. One effects gate therefore skips both, and the loop residualizes unchanged with every output
        // unknown even though `k` is loop-invariant known: the loop carries `[counter, k]`, runs while `counter > 0`,
        // and its body prints `k` and returns `[counter - k * k, k]`, so a probe would print the known `k`, whereas
        // partial evaluation prints nothing and the residual body still squares `k`. Partial evaluation over a tracing
        // known-side context makes that observable, because every print that a probe or the split binds on the known
        // side stages an instruction into the known-side trace.
        let scalar = || ArrayType::scalar(DataType::F64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
            let counter = builder.add_input(scalar());
            let k = builder.add_input(scalar());
            let printed = builder.add_instruction(PrintOperation::new("k"), Vec::new(), vec![k], None).unwrap()[0];
            let k_squared =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![printed, printed], None).unwrap()[0];
            let next_counter =
                builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, k_squared], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next_counter, k], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let program = while_program(WhileOperation::new(), &[condition, body], vec![scalar(); 2]);

        let context = TestTracingContext::new();
        let k = context.lift(Array::scalar(1.0).unwrap()).unwrap();
        let evaluation = program
            .partially_evaluate_in_context(&context, &[PartialValue::Unknown(scalar()), PartialValue::Known(k)])
            .unwrap();
        assert!(context.builder().borrow().instructions().is_empty());
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        let expected_program = indoc! {"
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
            in (%2, %3)"};
        assert_eq!(evaluation.program().to_string(), expected_program);

        // Eager partial evaluation residualizes the same loop, which interprets to the final state.
        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(scalar()), PartialValue::Known(Array::scalar(1.0).unwrap())])
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(evaluation.program().to_string(), expected_program);
        assert_eq!(
            evaluation.interpret(&TestEagerContext::new(), &[Array::scalar(3.0).unwrap()]),
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
        let k_squared = body_builder
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
                vec![body_inputs[0], k_squared],
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
        let program = while_program(WhileOperation::<ArrayIrType>::new(), &[condition, body], state_types.clone());

        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(state_types[0].clone()),
                PartialValue::Known(array(Array::scalar(1.0f64).unwrap())),
                PartialValue::Unknown(state_types[2].clone()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
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
        // The loop carries `[counter, accumulator]` and runs while `counter > 0`; its body computes `next_counter =
        // counter - 1` and `next_accumulator = accumulator + counter`. The `counter` element is *time-varying* known
        // (its value changes every iteration, so the loop-invariant rewrite cannot fold it) but *closed*: its next
        // value and the predicate fold from it alone. With `counter` known (`3`) and `accumulator` unknown, the
        // closed-knownness split runs the known counter loop on the known side, folding the final counter to the known
        // value `0` (3 -> 2 -> 1 -> 0), and keeps the whole loop residual for `accumulator`, recomputing the counter
        // chain inside it (there is no statically shaped residual stream to feed `accumulator + counter` with).
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

        // From `accumulator = 10`, the residual loop adds 3, 2, and 1.
        let context = TestEagerContext::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(10.0).unwrap()]),
            Ok(vec![Array::scalar(0.0).unwrap(), Array::scalar(16.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_partial_evaluation_split_truncates_both_loops_at_the_iteration_bound() {
        // A bound of two iterations, below the three iterations of the unbounded loop, truncates the known counter loop
        // at `3 - 2 = 1` and the residual loop after adding 3 and 2, because both loops keep the bound of the source
        // loop.
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
        let context = TestEagerContext::new();
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
        let context = TestEagerContext::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(10.0).unwrap()]),
            Ok(vec![Array::scalar(0.0).unwrap(), Array::scalar(10.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_partial_evaluation_split_declines_when_the_predicate_reads_unknown_state() {
        // The counter of `while (x < 10) { (counter, x) = (counter - 1, x + x) }` is closed known, but the predicate
        // reads the unknown `x`, so a known counter loop could not reproduce the number of iterations. The rule
        // therefore keeps every output unknown and residualizes the loop unchanged.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
        condition_builder.add_input(scalar_type.clone());
        let x = condition_builder.add_input(scalar_type.clone());
        let limit = condition_builder.add_constant(Array::scalar(10.0).unwrap());
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![x, limit], None)
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let counter = body_builder.add_input(scalar_type.clone());
        let x = body_builder.add_input(scalar_type.clone());
        let one = body_builder.add_instruction(OneLikeOperation::new(), Vec::new(), vec![counter], None).unwrap()[0];
        let next_counter =
            body_builder.add_instruction(SubOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
        let doubled = body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![x, x], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_counter, doubled], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let program = while_program(WhileOperation::new(), &[condition, body], vec![scalar_type.clone(); 2]);

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
        let context = TestEagerContext::new();
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
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
        let i = condition_builder.add_input(scalar_type.clone());
        let n = condition_builder.add_input(scalar_type.clone());
        condition_builder.add_input(scalar_type.clone());
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![i, n], None)
            .unwrap()[0];
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let program = while_program(WhileOperation::new(), &[condition, body], vec![scalar_type.clone(); 3]);

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
        let context = TestEagerContext::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(1.5).unwrap()]),
            Ok(vec![Array::scalar(5.0).unwrap(), Array::scalar(5.0).unwrap(), Array::scalar(48.0).unwrap()]),
        );
    }

    #[test]
    fn test_while_partial_evaluation_preserves_entry_relationships_across_the_closed_knownness_split() {
        // A known `x: f64[3]` next to an unknown `accumulator: f64[n]` requires `n = 3` at the entry of the loop that
        // `vector_accumulator_while_program` builds. The rule guards that relationship before the split runs the closed
        // known `[counter, x]` loop on the known side, and the residual program must still reject an `accumulator` of
        // any other length, like the source program does.
        let (program, vector_type) = vector_accumulator_while_program();
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
                lambda %0:f64[n], %1:f64[3], %2:f64[] .
                let %3:f64[3], %4:f64[3] = guard [boundary=\"partial evaluation\", declared_types=[f64[n], f64[n]]] %1 \
                %0
                    %5:f64[], %6:f64[3], %7:f64[3] = while %2 %1 %4 [
                        condition={
                            lambda %0:f64[], %1:f64[3], %2:f64[3] .
                            let %3:f64[] = zero_like %0
                                %4:bool[] = compare [direction=GreaterThan] %0 %3
                            in (%4)
                        },
                        body={
                            lambda %0:f64[], %1:f64[3], %2:f64[3] .
                            let %3:f64[] = one_like %0
                                %4:f64[] = sub %0 %3
                                %5:f64[3] = add %2 %1
                            in (%4, %1, %5)
                        },
                    ]
                in (%7)"},
        );
        // From `accumulator = [5, 5, 5]`, both programs add `x` twice, and both reject an `accumulator` of length 4.
        let context = TestEagerContext::new();
        let conforming = Array::vector(vec![5.0f64; 3]).unwrap();
        let expected = vec![Array::scalar(0.0f64).unwrap(), known.clone(), Array::vector(vec![7.0f64; 3]).unwrap()];
        assert_eq!(program.interpret(vec![counter.clone(), known.clone(), conforming.clone()]), Ok(expected.clone()));
        assert_eq!(evaluation.interpret(&context, std::slice::from_ref(&conforming)), Ok(expected));
        let nonconforming = Array::vector(vec![5.0f64; 4]).unwrap();
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 3,
            actual: 4,
        })));
        assert_eq!(program.interpret(vec![counter, known, nonconforming.clone()]), mismatch);
        assert_eq!(evaluation.interpret(&context, std::slice::from_ref(&nonconforming)), mismatch);
    }

    #[test]
    fn test_while_partial_evaluation_guards_the_entry_when_only_known_finals_are_used() {
        // The loop of the preceding test with only its known finals `[counter, x]` used. The split leaves the residual
        // loop dead, so only the entry guard checks that the unknown `accumulator` has the extent `3` of the known `x`:
        // the partially evaluated program rejects an `accumulator` of length 4 exactly like the source program.
        let (program, vector_type) = vector_accumulator_while_program();
        let program = program.with_outputs(&[0, 1]).unwrap();
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
                lambda %0:f64[n], %1:f64[3], %2:f64[] .
                let %3:f64[3], %4:f64[3] = guard [boundary=\"partial evaluation\", declared_types=[f64[n], f64[n]]] %1 \
                %0
                in ()"},
        );
        let context = TestEagerContext::new();
        let conforming = Array::vector(vec![5.0f64; 3]).unwrap();
        assert_eq!(
            evaluation.interpret(&context, std::slice::from_ref(&conforming)),
            Ok(vec![Array::scalar(0.0f64).unwrap(), known.clone()]),
        );
        let nonconforming = Array::vector(vec![5.0f64; 4]).unwrap();
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 3,
            actual: 4,
        })));
        assert_eq!(program.interpret(vec![counter, known, nonconforming.clone()]), mismatch);
        assert_eq!(evaluation.interpret(&context, std::slice::from_ref(&nonconforming)), mismatch);
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
    fn test_while_partial_evaluation_partition_propagates_unknown_carries_across_multiple_rounds() {
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
        let attached_regions = AttachedRegions::from(regions.clone());
        check_operation_batching!(
            @exact,
            context = TestEagerContext::new(),
            driver = &CountingBatchingDriver::new(&attached_regions),
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
    fn test_while_batching_stages_batched_predicate_loops_under_tracing() {
        // Batching the per-item countdown loop under tracing makes the predicate `x > 0` per batch item, so the rule
        // stages exactly one `while`, without unrolling it, whose condition returns the batched `bool[3]` predicate
        // directly and without building any masking program data: interpretation and lowering own the masked
        // semantics. Batch items [3, 1, 2] terminate after 3, 1, and 2 iterations, and finished batch items keep their
        // final state, matching the eager path batch item for batch item. Staging needs only one structural batching
        // pass per region.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let program = batch_while_under_tracing(WhileOperation::new(), countdown_while_regions(scalar_type.clone()), 3);
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

        // When the batched predicate is false for every batch item at entry, the staged loop runs no iteration and
        // returns the state of every batch item unchanged, exactly like the source loop applied to each batch item.
        assert_eq!(
            program.interpret(Array::vector(vec![0.0, -1.0, -2.5]).unwrap()),
            Ok(Array::vector(vec![0.0, -1.0, -2.5]).unwrap()),
        );

        // The rule batches the condition with natural output axes to detect per-item termination and then instantiates
        // its predicate at axis 0. `AlignEachTo` stages axis movement only where a natural axis differs from a mapped
        // target, so a predicate that already landed on axis 0 over unchanged state axes keeps its discovery program:
        // binding the rule directly performs one structural pass per region instead of re-batching the condition, and
        // stages the same loop.
        let parent = DomainTracingContext::<TestEagerContext>::new();
        let builder = parent.builder().clone();
        let packed_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let state_atom = builder.borrow_mut().add_input(packed_type.clone());
        let state = parent.tracer(state_atom, None);
        let context = BatchingContext::new(parent, 3);
        let inputs = vec![ArrayBatch::new(state, BatchAxis::new(0)).unwrap()];
        let countdown_regions = AttachedRegions::from(countdown_while_regions(scalar_type.clone()));
        let driver = CountingBatchingDriver::new(&countdown_regions);
        let outputs = WhileOperation::new().batch(&context, &driver, inputs.as_slice()).unwrap();
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
        let bounded = WhileOperation::new().with_iteration_bound(2).unwrap();
        let program = batch_while_under_tracing(bounded, countdown_while_regions(scalar_type), 3);
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
    fn test_while_batching_preserves_a_replicated_predicate_prefix() {
        // A replicated `[true, false]` predicate already masks the leading state axis, so batching inserts the mapped
        // axis after that prefix: the first state item doubles twice for every batch item and the second never does.
        let state_type = ArrayType::new_static(DataType::F64, [2]);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
        condition_builder.add_input(state_type.clone());
        let predicate = condition_builder.add_constant(Array::vector(vec![true, false]).unwrap());
        let condition = condition_builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = body_builder.add_input(state_type.clone());
        let doubled =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![doubled], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let operation = WhileOperation::new().with_iteration_bound(2).unwrap();
        let source = while_program(operation, &[condition, body], vec![state_type]);
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
        let doubled_vectors = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [2, 3]),
            &[4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
        )
        .unwrap();
        assert_eq!(
            batched.interpret(vec![
                TestIrValue::Dimension(extent.clone()),
                array(Array::scalar(2.0).unwrap()),
                array(vectors),
            ]),
            Ok(vec![TestIrValue::Dimension(extent), array(Array::scalar(0.0).unwrap()), array(doubled_vectors)]),
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
        let step = body_builder.add_constant(array(Array::scalar(-1.0f32).unwrap()));
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

        // Batching the discharged program carries the per-item referent as an ordinary array carry.
        let (discharged, _) = discharged.into_parts();
        assert_eq!(
            discharged.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[], %2:f32[2] .
                let %3:dimension<2>, %4:f32[], %5:f32[2] = while %0 %1 %2 [
                    condition={
                        lambda %0:dimension<2>, %1:f32[], %2:f32[2] .
                        let %3:f32[] = zero_like %1
                            %4:bool[] = compare [direction=GreaterThan] %1 %3
                        in (%4)
                    },
                    body={
                        lambda %0:dimension<2>, %1:f32[], %2:f32[2] .
                        let %3:f32[] = const -1.0
                            %4:f32[] = add %1 %3
                            %5:f32[2] = add %2 %2
                        in (%0, %4, %5)
                    },
                ]
                in (%0, %5)"},
        );

        // Both batched programs run the loop to the same values: three iterations double each item's referent.
        let inputs = vec![
            TestIrValue::Dimension(axis_extent.clone()),
            array(Array::scalar(3.0f32).unwrap()),
            array(Array::vector(vec![1.0f32, 10.0]).unwrap()),
        ];
        let expected = vec![TestIrValue::Dimension(axis_extent), array(Array::vector(vec![8.0f32, 80.0]).unwrap())];
        assert_eq!(direct.interpret(inputs.clone()), Ok(expected.clone()));
        assert_eq!(discharged.interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_while_batching_reuses_naturally_aligned_batch_varying_array_ir_programs() {
        // The array IR rule batches the condition with natural output axes to detect per-item termination and rebuilds
        // both regions only when that widening changes the invariant boundary: with every array carry already mapped at
        // axis 0 and a predicate that naturally landed on axis 0, the rebuilds would replay the exact same programs, so
        // the discovery body and condition are kept and batching the per-item countdown loop performs one structural
        // pass per region.
        let countdown_regions = AttachedRegions::from(array_ir_countdown_while_regions());

        // Batching batch items [3, 1, 2] under an eager parent performs exactly one structural pass per region (the
        // fixed-point discovery body and the natural condition) and interprets the loop with its masked per-item
        // semantics: every batch item counts down to 0 even though the items terminate after different numbers of
        // iterations.
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(
            TestIrEagerContext::new(),
            TestIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        );
        let inputs =
            vec![ArrayIrBatch::new(array(Array::vector(vec![3.0, 1.0, 2.0]).unwrap()), BatchAxis::new(0)).unwrap()];
        let driver = CountingBatchingDriver::new(&countdown_regions);
        let countdown_operation = WhileOperation::<ArrayIrType>::new();
        let outputs = countdown_operation.batch(&context, &driver, inputs.as_slice()).unwrap();
        assert_eq!(driver.batch_program_calls(), 2);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value(), &array(Array::vector(vec![0.0, 0.0, 0.0]).unwrap()));

        // The skipped alignment rebuilds are byte-identical to the discovery programs the rule kept: replaying each
        // region with `AlignEachTo` at the already-aligned axes (each region returns one output, the predicate or the
        // state, whose target is axis 0) renders exactly the natural discovery program.
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
        for (region, expected) in countdown_regions.regions().zip(expected) {
            let natural = driver
                .batch_program(&context, region, state_axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)
                .unwrap();
            let aligned = driver
                .batch_program(
                    &context,
                    region,
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
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let batch_type = DimensionType::from(batch.clone());
        let trace = TestIrTracingContext::new();
        let batch_extent = trace.input(batch_type.clone().into());
        let state =
            trace.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(batch.clone())])).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let outputs = context
            .bind(
                TestIrOperation::While(WhileOperation::new()),
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
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![outputs[0].batch().value().atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
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
                TestIrValue::Dimension(DimensionValue::new(batch_type, 3).unwrap()),
                array(Array::vector(vec![3.0, 1.0, 2.0]).unwrap()),
            ]),
            Ok(vec![array(Array::vector(vec![0.0, 0.0, 0.0]).unwrap())]),
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
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
        let source = while_program(operation, &[condition, body], vec![state_type]);
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
        let regions = AttachedRegions::from(vec![condition, body]);
        let driver = CountingBatchingDriver::new(&regions);

        // Both carries are mapped under a staging parent, whose trace must stay empty after the rejection.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9)).unwrap());
        let trace = TestIrTracingContext::new();
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
        let (operation, regions) = squaring_while_operation(100.0, Some(4));
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
        let function = |state: TestTracer| {
            let mut outputs = state.context().bind(
                WhileOperation::new(),
                countdown_while_regions(ArrayType::scalar(DataType::F64)),
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
                let regions = countdown_while_regions(ArrayType::scalar(DataType::F64));
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
        let context = BindRecordingContext::new();
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

        assert_eq!(primal, Array::scalar(8.0).unwrap());
        assert_eq!(tangent, Array::scalar(8.0).unwrap());
        assert_eq!(observed_context.inputs_of(PRINT_OPERATION_NAME).len(), 3);
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
        let limit = condition_builder.add_constant(array(Array::scalar(4.0f64).unwrap()));
        let predicate = condition_builder
            .add_instruction(
                TestOperation::from(CompareOperation::new(ComparisonDirection::LessThan)),
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
        let zero = body_builder.add_constant(array(Array::scalar(0.0f64).unwrap()));
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

        let regions = vec![condition, body];
        assert_eq!(
            differentiate_at(array(Array::scalar(1.0f64).unwrap())).jvp(
                array(Array::scalar(3.0f64).unwrap()),
                |input| {
                    let context = input.context().clone();
                    Ok(context.bind(WhileOperation::new(), regions.clone(), &[input])?.remove(0))
                },
            ),
            Ok((array(Array::scalar(4.0f64).unwrap()), array(Array::scalar(12.0f64).unwrap()))),
        );
    }

    #[test]
    fn test_while_differentiation_eager_jvp_resumes_with_the_staged_rule_when_the_predicate_stops_concretizing() {
        // Under `batch` over an eager domain, the eager rule concretizes the predicate `i < 3` while it is replicated.
        // The body `(i + y, y, s + s)` maps `i` once it adds the mapped step `y = [1, 2]`, so the predicate stops
        // concretizing on the second iteration and the rule resumes from the current duals under the staged rule.
        // Item 0 runs three iterations (0 → 1 → 2 → 3) and item 1 two (0 → 2 → 4), so `s = 1.5` with tangent `2`
        // scales to `[12, 6]` with tangent `[16, 8]`, and the primal agrees with plain batching of the loop.
        let (operation, regions) = stepping_counter_while_operation(false);
        let inputs = (
            Array::scalar(0.0).unwrap(),
            Array::vector(vec![1.0, 2.0]).unwrap(),
            Array::scalar(1.5).unwrap(),
            Array::scalar(2.0).unwrap(),
        );
        let input_axes = (BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::replicated());
        assert_eq!(
            batch(
                |(counter, step, scale, scale_tangent)| {
                    Ok(differentiate_at(scale).with_captures((counter, step)).jvp(
                        scale_tangent,
                        |scale, (counter, step)| {
                            Ok(bind_while(operation, regions.clone(), &[counter, step, scale])?.remove(2))
                        },
                    )?)
                },
                inputs.clone(),
                input_axes,
                BatchAxis::new(0),
                None,
            ),
            Ok((Array::vector(vec![12.0, 6.0]).unwrap(), Array::vector(vec![16.0, 8.0]).unwrap())),
        );
        assert_eq!(
            batch(
                |(counter, step, scale, _)| Ok(
                    bind_while(operation, regions.clone(), &[counter, step, scale])?.remove(2),
                ),
                inputs,
                input_axes,
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::vector(vec![12.0, 6.0]).unwrap()),
        );
    }

    #[test]
    fn test_while_differentiation_eager_jvp_resumes_bounded_loops_with_the_remaining_bound() {
        // The loop of the preceding test with an iteration bound of 2. The eager rule runs the first iteration, and the
        // predicate stops concretizing on the second, so the staged rule resumes with the remaining bound of 1. Both
        // items run that one remaining iteration, giving `s = [6, 6]` with tangent `[8, 8]`, which agrees with plain
        // batching of the bounded loop. A remaining bound of 2 would instead give `([12, 6], [16, 8])`.
        let (operation, regions) = stepping_counter_while_operation(false);
        let operation = operation.with_iteration_bound(Some(2)).unwrap();
        let inputs = (
            Array::scalar(0.0).unwrap(),
            Array::vector(vec![1.0, 2.0]).unwrap(),
            Array::scalar(1.5).unwrap(),
            Array::scalar(2.0).unwrap(),
        );
        let input_axes = (BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::replicated());
        assert_eq!(
            batch(
                |(counter, step, scale, scale_tangent)| {
                    Ok(differentiate_at(scale).with_captures((counter, step)).jvp(
                        scale_tangent,
                        |scale, (counter, step)| {
                            Ok(bind_while(operation, regions.clone(), &[counter, step, scale])?.remove(2))
                        },
                    )?)
                },
                inputs.clone(),
                input_axes,
                BatchAxis::new(0),
                None,
            ),
            Ok((Array::vector(vec![6.0, 6.0]).unwrap(), Array::vector(vec![8.0, 8.0]).unwrap())),
        );
        assert_eq!(
            batch(
                |(counter, step, scale, _)| Ok(
                    bind_while(operation, regions.clone(), &[counter, step, scale])?.remove(2),
                ),
                inputs,
                input_axes,
                BatchAxis::new(0),
                None,
            ),
            Ok(Array::vector(vec![6.0, 6.0]).unwrap()),
        );
    }

    #[test]
    fn test_while_differentiation_eager_jvp_rejects_resuming_effectful_conditions() {
        // The unbounded loop of the preceding tests with a condition that prints `i`. Its predicate stops concretizing
        // on the second iteration as well, but resuming under the staged rule would evaluate the condition, and
        // therefore its print, on the current state a second time, so the eager rule reports the concretization error
        // instead.
        let (operation, regions) = stepping_counter_while_operation(true);
        assert_eq!(
            batch(
                |(counter, step, scale, scale_tangent)| {
                    Ok(differentiate_at(scale).with_captures((counter, step)).jvp(
                        scale_tangent,
                        |scale, (counter, step)| {
                            Ok(bind_while(operation, regions.clone(), &[counter, step, scale])?.remove(2))
                        },
                    )?)
                },
                (
                    Array::scalar(0.0).unwrap(),
                    Array::vector(vec![1.0, 2.0]).unwrap(),
                    Array::scalar(1.5).unwrap(),
                    Array::scalar(2.0).unwrap(),
                ),
                (BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::replicated()),
                BatchAxis::new(0),
                None,
            ),
            Err(BatchingError::Program(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from a value batched along axis 0".to_string(),
            })),
        );
    }

    #[test]
    fn test_while_differentiation_binds_the_original_loop_without_live_state_tangents() {
        // Forward mode with respect to `x` alone leaves every state tangent of a loop over the reference `r` a
        // structural zero, and the carry fixed point keeps it that way, so the staged rule binds the original loop with
        // structural-zero tangents instead of staging residual stacks, a mask, or a fused doubled-state loop. The loop
        // carries only a reference, so it reaches the rule instead of the all-zero fast path of the forward-mode
        // transform. The condition increments the referenced counter before comparing it against 3, so from 0 it
        // observes 1, 2, and 3 and the final value is 3.
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.clone());
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let regions = vec![builder.import_program(incrementing_condition(3.0)), builder.import_program(body)];
        let reference = builder.add_input(reference_type);
        let x = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference = builder
            .add_instruction(WhileOperation::<ArrayIrType>::new(), regions, vec![reference], None)
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![value, x], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let jvp = program.jvp_with_respect_to(&[1]).unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[], %2:f32[] .
                let %3:ref<f32[]> = while %0 [
                    condition={
                        lambda %0:ref<f32[]> .
                        let %1:f32[] = const 1.0
                            () = reference_add_update %0 %1
                            %2:f32[] = reference_read %0
                            %3:f32[] = const 3.0
                            %4:bool[] = compare [direction=LessThan] %2 %3
                        in (%4)
                    },
                    body={
                        lambda %0:ref<f32[]> .
                        in (%0)
                    },
                ]
                    %4:f32[] = reference_read %3
                    %5:f32[] = zero [type=f32[]]
                in (%4, %1, %5, %2)"},
        );
        assert_eq!(
            jvp.interpret(vec![
                TestIrValue::Reference(ArrayReference::new(Array::scalar(0f32).unwrap())),
                array(Array::scalar(5f32).unwrap()),
                array(Array::scalar(2f32).unwrap()),
            ]),
            Ok(vec![
                array(Array::scalar(3f32).unwrap()),
                array(Array::scalar(5f32).unwrap()),
                array(Array::scalar(0f32).unwrap()),
                array(Array::scalar(2f32).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_while_differentiation_keeps_tangents_of_inactive_state_structural() {
        // `limit` passes through the loop without a tangent, so both forward-mode rules keep its tangent a structural
        // zero instead of threading a materialized zero through the loop, and linearization succeeds. At `x = 2` with
        // limit `10`, the loop squares twice, so `f(x) = x⁴` with value `16` and derivative `4 x³ = 32`.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
        let condition_inputs = (0..2).map(|_| condition_builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), condition_inputs, None)
            .unwrap()[0];
        let condition = condition_builder.build(vec![predicate], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let body_inputs = (0..2).map(|_| body_builder.add_input(scalar_type.clone())).collect::<Vec<_>>();
        let square = body_builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![body_inputs[0], body_inputs[0]], None)
            .unwrap()[0];
        let body = body_builder
            .build(vec![square, body_inputs[1]], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let regions = [condition, body];
        let check = |iteration_bound: Option<usize>, expected_jvp: &str| {
            let operation = WhileOperation::new().with_iteration_bound(iteration_bound).unwrap();
            let program = while_program(operation, &regions, vec![scalar_type.clone(); 2]);
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
                    %11:f64[] = scan [carry_count=1, iteration_count=3, reverse=false] %2 %9 %10 [
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
                    %15:f64[], %16:f64[], %17:f64[] = scan [carry_count=3, iteration_count=4, reverse=false] %13 %14 \
                    %4 %12 [
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
        // Reverse mode runs end to end through a staged bounded while loop.
        // `f(x) = while (x < 8, iteration_bound = 5) { x = 2 * x }` at `x = 1` runs three iterations (`x` visits 1,
        // 2, 4), so the actual iteration count 3 is strictly below the bound 5 and the two trailing stack slots matter:
        // their mask entries are false, so they must pass tangents through unchanged in the forward scan and cotangents
        // through unchanged in the transposed scan. Locally `f(x) = 8 x`: value 8, gradient 8.
        let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
        let (pullback, program) = trace_while_vjp(while_operation, while_regions);
        let (output, residuals) = program.interpret(Array::scalar(1.0).unwrap()).unwrap();
        assert_eq!(output, Array::scalar(8.0).unwrap());

        // The pullback contains the transposed (reversed) linear scan and no while loop, and every cotangent seed
        // scales the hand-computed gradient 8. The direct-transpose pullback consumes `[cotangent ++ residuals]`.
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[5] .
                let %2:f64[] = scan [carry_count=1, iteration_count=5, reverse=true] %0 %1 [
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
            pullback.interpret(pullback_inputs(Array::scalar(1.0).unwrap())),
            Ok(vec![Array::scalar(8.0).unwrap()]),
        );
        assert_eq!(
            pullback.interpret(pullback_inputs(Array::scalar(2.0).unwrap())),
            Ok(vec![Array::scalar(16.0).unwrap()]),
        );

        // `value_and_gradient` composes the same machinery end to end.
        let (while_operation, while_regions) = bounded_doubling_while_operation(8.0, 5);
        let program = trace_while_value_and_gradient(while_operation, while_regions, ArrayType::scalar(DataType::F64));
        assert_eq!(
            program.interpret(Array::scalar(1.0).unwrap()),
            Ok((Array::scalar(8.0).unwrap(), Array::scalar(8.0).unwrap())),
        );
    }

    #[test]
    fn test_while_differentiation_stores_loop_varying_residual_stacks() {
        // The bounded rule stores loop-varying residuals instead of recomputing them:
        // `while (x < 100, iteration_bound = 4) { x = x * x }` at `x = 2` squares three times (`x` visits 2, 4, 16 →
        // 256, three iterations below the bound 4), and the product rule references the per-iteration state as a
        // loop-varying residual, so the gradient depends on the stored stack slots `[2, 4, 16, 0]`, including the zero
        // slot beyond the last iteration, which the mask must keep inert in both directions. Locally `f(x) = x⁸`: value
        // 256 and gradient `8 x⁷ = 1024`.
        let (while_operation, while_regions) = squaring_while_operation(100.0, Some(4));
        let (pullback, program) = trace_while_vjp(while_operation, while_regions);
        let (output, residuals) = program.interpret(Array::scalar(2.0).unwrap()).unwrap();
        assert_eq!(output, Array::scalar(256.0).unwrap());
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:bool[4], %2:f64[4] .
                let %3:f64[] = scan [carry_count=1, iteration_count=4, reverse=true] %0 %1 %2 [
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
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![Array::scalar(1024.0).unwrap()]));

        // The eager-domain reverse-mode entry point produces the same value and gradient numbers.
        let (while_operation, while_regions) = squaring_while_operation(100.0, Some(4));
        let (value, gradient) = differentiate_at(Array::scalar(2.0).unwrap())
            .value_and_gradient(move |x| {
                let mut outputs = x
                    .context()
                    .bind(TestOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                    .unwrap();
                outputs.remove(0)
            })
            .unwrap();
        assert_eq!(value, Array::scalar(256.0).unwrap());
        assert_eq!(gradient, Array::scalar(1024.0).unwrap());
        let (while_operation, while_regions) = squaring_while_operation(100.0, Some(4));
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
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
            let program = trace_while_value_and_gradient(operation, regions, ArrayType::scalar(DataType::F64));
            assert_eq!(
                program.interpret(Array::scalar(initial).unwrap()),
                Ok((Array::scalar(expected_value).unwrap(), Array::scalar(expected_gradient).unwrap())),
            );
        }
    }

    #[test]
    fn test_while_differentiation_supports_vector_state() {
        // The bounded staged path differentiates loops over vector state, whose per-iteration residuals are vectors
        // rather than scalars. The loop `while (sum(x) < 20, iteration_bound = 4) { x = x * x }` at `x = [1.5, 2]`
        // squares twice (sums visit 3.5 and 6.25 before reaching 21.0625), so `f(x) = sum(x⁴)` locally: value
        // `1.5⁴ + 2⁴ = 21.0625` and gradient `4 x³ = [13.5, 32]`, with two iterations strictly below the bound 4.
        let vector_f64 = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
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

        let (_, program) = TestTracingContext::trace(
            move |x: TestStagedTracer| {
                Ok(differentiate_at(x).value_and_gradient(move |x| {
                    let mut outputs = x
                        .context()
                        .bind(TestOperation::While(while_operation), while_regions.clone(), &[x.clone()])
                        .unwrap();
                    let state = outputs.remove(0);
                    let mut outputs = state
                        .context()
                        .bind(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), &[state.clone()])
                        .unwrap();
                    outputs.remove(0)
                })?)
            },
            vector_f64,
        )
        .unwrap();
        assert_eq!(
            program.interpret(Array::vector(vec![1.5, 2.0]).unwrap()),
            Ok((Array::scalar(21.0625).unwrap(), Array::vector(vec![13.5, 32.0]).unwrap())),
        );
    }

    #[test]
    fn test_while_differentiation_never_reads_the_placeholder_residuals_of_loops_without_iterations() {
        // At `x = 200`, the condition of `while (x < 100, iteration_bound = 4) { x = x * x }` is false at entry, so the
        // staged loop runs no iteration: every mask entry stays false and every residual stack slot keeps its zero
        // placeholder, which the guarded tangent steps never read. Locally `f(x) = x`: value 200 and derivative 1.
        let (operation, regions) = squaring_while_operation(100.0, Some(4));
        let program = while_program(operation, &regions, vec![ArrayType::scalar(DataType::F64)]);
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![Array::scalar(200.0).unwrap()]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![Array::scalar(200.0).unwrap()]);
        assert_eq!(residuals, vec![Array::vector(vec![false; 4]).unwrap(), Array::vector(vec![0.0f64; 4]).unwrap()]);
        let mut tangent_inputs = vec![Array::scalar(3.0).unwrap()];
        tangent_inputs.extend(residuals.clone());
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![Array::scalar(3.0).unwrap()]));
        let mut pullback_inputs = vec![Array::scalar(1.0).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![Array::scalar(1.0).unwrap()]),
        );

        // Staged reverse mode composes the same masked scan end to end, and finite differences confirm the identity.
        let program = trace_while_value_and_gradient(operation, regions.clone(), ArrayType::scalar(DataType::F64));
        assert_eq!(
            program.interpret(Array::scalar(200.0).unwrap()),
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
        // entry point and staged reverse mode (whose validity mask is true for every iteration) both compute the value
        // 16 and the gradient 8.
        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        assert_eq!(
            TestEagerContext::new().bind(while_operation, while_regions, &[Array::scalar(2.0).unwrap()]),
            Ok(vec![Array::scalar(16.0).unwrap()]),
        );

        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let (value, gradient) = differentiate_at(Array::scalar(2.0).unwrap())
            .value_and_gradient(move |x| {
                let mut outputs = x.context().bind(while_operation, while_regions.clone(), &[x.clone()]).unwrap();
                outputs.remove(0)
            })
            .unwrap();
        assert_eq!(value, Array::scalar(16.0).unwrap());
        assert_eq!(gradient, Array::scalar(8.0).unwrap());

        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        let program = trace_while_value_and_gradient(while_operation, while_regions, ArrayType::scalar(DataType::F64));
        assert_eq!(
            program.interpret(Array::scalar(2.0).unwrap()),
            Ok((Array::scalar(16.0).unwrap(), Array::scalar(8.0).unwrap())),
        );
        let (while_operation, while_regions) = bounded_doubling_while_operation(f64::INFINITY, 3);
        check_gradient!(
            |x| bind_while(while_operation, while_regions.clone(), &[x]).map(|mut outputs| outputs.remove(0)),
            at = Array::scalar(2.0).unwrap(),
            step = 1e-4,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_while_differentiation_rejects_dynamically_shaped_residual_storage() {
        // The bounded rule stores one residual per iteration in a `[bound, …]` stack, which needs a statically shaped
        // storage type, but squaring an `f64[n]` state stores the dynamically shaped state itself.
        let n = DimensionVariable::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let state_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(n)]));
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = condition_builder.add_input(state_type.clone());
        let sum = condition_builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![state], None)
            .unwrap()[0];
        let limit = condition_builder.add_constant(Array::scalar(100.0).unwrap());
        let predicate = condition_builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![sum, limit], None)
            .unwrap()[0];
        let condition = condition_builder.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = body_builder.add_input(state_type.clone());
        let squared =
            body_builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = body_builder.build(vec![squared], vec![Placeholder], vec![Placeholder]).unwrap();
        let operation = WhileOperation::new().with_iteration_bound(3).unwrap();
        let program = while_program(operation, &[condition, body], vec![state_type]);
        assert_eq!(
            program.jvp().map(|_| ()),
            Err(DifferentiationError::from(ProgramError::from(TypeError::invalid(format!(
                "`jvp` of a bounded `{WHILE_OPERATION_NAME}` loop requires statically shaped residual storage but got \
                 `f64[n]`",
            ))))),
        );
    }

    #[test]
    fn test_while_differentiation_truncates_masked_loops_at_the_iteration_bound() {
        // Under the batched predicate `x < [100, 3]`, `while (iteration_bound = 2) { x = x * x }` at `x = [2, 2]` stops
        // item 1 after one iteration (2 -> 4) and truncates item 0 after two (2 -> 4 -> 16) while its predicate is
        // still true. Locally `f(x) = [x₀⁴, x₁²]`: value `[16, 4]` and per-item derivatives `[4 x₀³, 2 x₁] = [32, 4]`.
        let state_type = ArrayType::new_static(DataType::F64, [2]);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
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
    fn test_while_differentiation_evaluates_masked_conditions_only_on_the_states_of_the_source() {
        // The masked form of a bounded loop with a batched predicate must evaluate the condition on exactly the states
        // that the source loop evaluates it on, so that a condition that errors, diverges, or differs past the bound
        // behaves alike on every path. Under the batched predicate `x < [2, 20]`, `x ↦ 2 x + 1` from `[0, 0]` stops
        // item 0 after two iterations and item 1 after five. The eager rule cannot concretize the batched predicate,
        // so it differentiates the masked form, whose eager forward mode evaluates the condition through this
        // context. The context records the state that every comparison receives (the condition holds the only one), and
        // the recorded states must be those on which interpretation evaluates the condition: with the bound 1 or 2,
        // never the state after the last admitted body application, and with the bound 10, every state up to the one
        // where both items have stopped.
        let state_type = ArrayType::new_static(DataType::F64, [2]);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = condition_builder.add_input(state_type.clone());
        let limits = condition_builder.add_constant(Array::vector(vec![2.0, 20.0]).unwrap());
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
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = body_builder.add_input(state_type.clone());
        let doubled =
            body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let one = body_builder.add_constant(Array::vector(vec![1.0, 1.0]).unwrap());
        let next = body_builder.add_instruction(AddOperation::new(), Vec::new(), vec![doubled, one], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![next], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let regions = vec![condition, body];
        let input = Array::vector(vec![0.0, 0.0]).unwrap();
        let cases = [
            (1, vec![vec![0.0, 0.0]], vec![1.0, 1.0], vec![2.0, 2.0]),
            (2, vec![vec![0.0, 0.0], vec![1.0, 1.0]], vec![3.0, 3.0], vec![4.0, 4.0]),
            (
                10,
                vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![3.0, 3.0], vec![3.0, 7.0], vec![3.0, 15.0], vec![3.0, 31.0]],
                vec![3.0, 31.0],
                vec![4.0, 32.0],
            ),
        ];
        for (bound, evaluated_states, primal, tangent) in cases {
            let operation = WhileOperation::new().with_iteration_bound(bound).unwrap();
            let program = while_program(operation, &regions, vec![state_type.clone()]);
            let primal = Array::vector(primal).unwrap();
            assert_eq!(program.interpret(vec![input.clone()]), Ok(vec![primal.clone()]));
            let context = BindRecordingContext::new();
            let regions = regions.clone();
            assert_eq!(
                context.jvp(
                    move |x, ()| Ok(x.context().bind(operation, regions.clone(), &[x.clone()])?.remove(0)),
                    input.clone(),
                    Array::vector(vec![1.0, 1.0]).unwrap(),
                    (),
                ),
                Ok((primal, Array::vector(tangent).unwrap())),
            );
            let mut compared = context
                .inputs_of(COMPARE_OPERATION_NAME)
                .into_iter()
                .map(|mut inputs| inputs.remove(0))
                .collect::<Vec<_>>();
            compared.dedup();
            assert_eq!(
                compared,
                evaluated_states.into_iter().map(|state| Array::vector(state).unwrap()).collect::<Vec<_>>(),
            );
        }
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
        let operation = WhileOperation::new().with_iteration_bound(5).unwrap();
        let regions = rejoining_while_regions();
        let program = while_program(operation, &regions, vec![state_type]);
        let input = Array::vector(vec![0.0, 0.0]).unwrap();
        let expected = Array::vector(vec![2.5, 4.75]).unwrap();
        assert_eq!(program.interpret(vec![input.clone()]), Ok(vec![expected.clone()]));

        // The staged bounded forward mode differentiates the masked form (refer to `test_masked_while_programs`),
        // which lets item 0 rejoin as well.
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
    fn test_while_differentiation_pulls_back_array_ir_loops_with_batched_predicates() {
        // Reverse mode runs through a bounded array IR loop with a batched predicate: `while (x < [2, 4, 8],
        // iteration_bound = 4) { x = x + x }` from `[1, 1, 1]` doubles item `i` exactly `i + 1` times before its own
        // predicate turns false, so the primal is `[2, 4, 8]` and the pullback scales a unit cotangent by the matching
        // per-item factors `[2, 4, 8]`. The augmented primal loop is the bounded rule over the masked form, whose
        // condition and body each evaluate the per-item predicate of the current state instead of carrying a mask.
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

        let operation = WhileOperation::new().with_iteration_bound(4).unwrap();
        let program = while_program(operation, &[condition, body], vec![ArrayIrType::Array(vector_type)]);

        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:i64[] = zero [type=i64[]]
                    %2:bool[4, 3] = zero [type=bool[4, 3]]
                    %3:bool[4] = zero [type=bool[4]]
                    %4:f64[3], %5:i64[], %6:bool[4, 3], %7:bool[4] = while [iteration_bound=4] %0 %1 %2 %3 [
                        condition={
                            lambda %0:f64[3], %1:i64[], %2:bool[4, 3], %3:bool[4] .
                            let %4:f64[3] = const [2.0, 4.0, 8.0]
                                %5:bool[3] = compare [direction=LessThan] %0 %4
                                %6:bool[] = reduce [kind=any, axes=[0]] %5
                            in (%6)
                        },
                        body={
                            lambda %0:f64[3], %1:i64[], %2:bool[4, 3], %3:bool[4] .
                            let %4:f64[3] = const [2.0, 4.0, 8.0]
                                %5:bool[3] = compare [direction=LessThan] %0 %4
                                %6:f64[3] = add %0 %0
                                %7:f64[3] = select %5 %6 %0
                                %8:i64[] = one [type=i64[]]
                                %9:i64[] = add %1 %8
                                %10:bool[1, 3] = broadcast [output_type=bool[1, 3], output_axes=[1]] %5
                                %11:i64[] = zero [type=i64[]]
                                %12:bool[4, 3] = dynamic_update_slice %2 %10 %1 %11
                                %13:bool[] = one [type=bool[]]
                                %14:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %13
                                %15:bool[4] = dynamic_update_slice %3 %14 %1
                            in (%7, %9, %12, %15)
                        },
                    ]
                in (%4, %7, %6)"},
        );
        assert_eq!(
            linearization.pullback().unwrap().program().to_string(),
            indoc! {"
                lambda %0:f64[3], %1:bool[4], %2:bool[4, 3] .
                let %3:f64[3] = scan [carry_count=1, iteration_count=4, reverse=true] %0 %1 %2 [
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
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap())]);
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
        // The staged loop is the bounded rule over the masked form, whose condition and body each evaluate the
        // per-item predicate of the current state instead of carrying a mask, and which forwards the dimension carry.
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

        let operation = WhileOperation::new().with_iteration_bound(4).unwrap();
        let input_types = vec![ArrayIrType::Dimension(extent_type.clone()), ArrayIrType::Array(vector_type)];
        let program = while_program(operation, &[condition, body], input_types);
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[3], %2:f64[3] .
                let %3:i64[] = zero [type=i64[]]
                    %4:bool[4, 3] = zero [type=bool[4, 3]]
                    %5:bool[4] = zero [type=bool[4]]
                    %6:dimension<extent ∈ [1, 8)>, %7:f64[3], %8:i64[], %9:bool[4, 3], %10:bool[4] = while \
                    [iteration_bound=4] %0 %1 %3 %4 %5 [
                        condition={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[3], %2:i64[], %3:bool[4, 3], %4:bool[4] .
                            let %5:f64[3] = const [2.0, 4.0, 8.0]
                                %6:bool[3] = compare [direction=LessThan] %1 %5
                                %7:bool[] = reduce [kind=any, axes=[0]] %6
                            in (%7)
                        },
                        body={
                            lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[3], %2:i64[], %3:bool[4, 3], %4:bool[4] .
                            let %5:f64[3] = const [2.0, 4.0, 8.0]
                                %6:bool[3] = compare [direction=LessThan] %1 %5
                                %7:f64[3] = add %1 %1
                                %8:f64[3] = select %6 %7 %1
                                %9:i64[] = zero [type=i64[]]
                                %10:bool[1, 3] = broadcast [output_type=bool[1, 3], output_axes=[1]] %6
                                %11:bool[4, 3] = dynamic_update_slice %3 %10 %2 %9
                                %12:bool[] = one [type=bool[]]
                                %13:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %12
                                %14:bool[4] = dynamic_update_slice %4 %13 %2
                                %15:i64[] = one [type=i64[]]
                                %16:i64[] = add %2 %15
                            in (%0, %8, %16, %11, %14)
                        },
                    ]
                    %11:f64[3] = scan [carry_count=1, iteration_count=4, reverse=false] %2 %9 %10 [
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
                in (%6, %7, %11)"},
        );
        assert_eq!(
            jvp.interpret(vec![
                dimension(&extent_type, 4),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
                array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap()),
            ]),
            Ok(vec![
                dimension(&extent_type, 4),
                array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap()),
                array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap()),
            ]),
        );

        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization
            .primal()
            .interpret(vec![dimension(&extent_type, 4), array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())])
            .unwrap();
        let residuals = primal_outputs.split_off(2);
        assert_eq!(
            primal_outputs,
            vec![dimension(&extent_type, 4), array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap())],
        );
        let mut pullback_inputs = vec![array(Array::vector(vec![1.0, 1.0, 1.0]).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(
            linearization.pullback().unwrap().program().interpret(pullback_inputs),
            Ok(vec![array(Array::vector(vec![2.0, 4.0, 8.0]).unwrap())]),
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
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let counter = builder.add_input(counter_type.clone());
        let state = builder.add_input(state_type.clone());
        let one = builder.add_constant(Array::scalar(1i64).unwrap());
        let next_counter =
            builder.add_instruction(AddOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
        let next_state = builder.add_instruction(SinOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_counter, next_state], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let (_, program) = TestTracingContext::trace_with_named_axes(
            |inputs: Vec<Tracer<TestTracingContext>>| {
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
        let varying_vector = "f32[4][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        let varying_slice = "f32[1, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}]";
        let varying_stack = "f32[3, 4][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}]";
        let invariant_index = "i64[][sharding={mesh<['x'=2:manual]>, []}]";
        let varying_index = "i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]";
        assert_eq!(
            jvp.to_string(),
            formatdoc! {"
                lambda %0:i64[], %1:{varying_vector}, %2:{varying_vector} .
                let %3:i64[] = zero [type=i64[]]
                    %4:{varying_stack} = zero [type={varying_stack}]
                    %5:bool[3] = zero [type=bool[3]]
                    %6:i64[], %7:{varying_vector}, %8:i64[], %9:{varying_stack}, %10:bool[3] = while \
                    [iteration_bound=3] %0 %1 %3 %4 %5 [
                        condition={{
                            lambda %0:i64[], %1:{varying_vector}, %2:i64[], %3:{varying_stack}, %4:bool[3] .
                            let %5:i64[] = const 100
                                %6:bool[] = compare [direction=LessThan] %0 %5
                            in (%6)
                        }},
                        body={{
                            lambda %0:i64[], %1:{varying_vector}, %2:i64[], %3:{varying_stack}, %4:bool[3] .
                            let %5:i64[] = const 1
                                %6:i64[] = add %0 %5
                                %7:{varying_vector} = sin %1
                                %8:{varying_vector} = cos %1
                                %9:{varying_slice} = broadcast [
                                    output_type={varying_slice},
                                    output_axes=[1],
                                ] %8
                                %10:{invariant_index} = broadcast [output_type={invariant_index}, output_axes=[]] %2
                                %11:{varying_index} = parallel_vary [axis_name=\"x\"] %10
                                %12:{varying_index} = zero [type={varying_index}]
                                %13:{varying_stack} = dynamic_update_slice %3 %9 %11 %12
                                %14:bool[] = one [type=bool[]]
                                %15:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %14
                                %16:bool[3] = dynamic_update_slice %4 %15 %2
                                %17:i64[] = one [type=i64[]]
                                %18:i64[] = add %2 %17
                            in (%6, %7, %18, %13, %16)
                        }},
                    ]
                    %11:{varying_vector} = scan [carry_count=1, iteration_count=3, reverse=false] %2 %9 %10 [
                        body={{
                            lambda %0:i64[], %1:{varying_vector}, %2:{varying_vector}, %3:bool[] .
                            let %4:{varying_vector} = condition %3 %1 %2 [
                                true={{
                                    lambda %0:{varying_vector}, %1:{varying_vector} .
                                    let %2:{varying_vector} = mul %1 %0
                                    in (%2)
                                }},
                                false={{
                                    lambda %0:{varying_vector}, %1:{varying_vector} .
                                    in (%0)
                                }},
                            ]
                            in (%4)
                        }},
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
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = builder.add_input(state_type.clone());
        let limit = builder.add_constant(Array::from_elements(state_type.clone(), &[10f32]).unwrap());
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![state, limit], None)
            .unwrap()[0];
        let condition = builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = builder.add_input(state_type.clone());
        let next_state = builder.add_instruction(MulOperation::new(), Vec::new(), vec![state, state], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_state], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let (_, program) = TestTracingContext::trace_with_named_axes(
            |inputs: Vec<Tracer<TestTracingContext>>| {
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
        let varying_scalar = "f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]";
        let varying_slice = "f32[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        let varying_stack = "f32[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        let varying_counter = "i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]";
        let varying_predicate = "bool[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]";
        let varying_mask_slice = "bool[1][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        let varying_mask = "bool[3][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        assert_eq!(
            jvp.to_string(),
            formatdoc! {"
                lambda %0:{varying_scalar}, %1:{varying_scalar} .
                let %2:{varying_counter} = zero [type={varying_counter}]
                    %3:{varying_stack} = zero [type={varying_stack}]
                    %4:{varying_mask} = zero [type={varying_mask}]
                    %5:{varying_scalar}, %6:{varying_counter}, %7:{varying_stack}, %8:{varying_mask} = while \
                    [iteration_bound=3] %0 %2 %3 %4 [
                        condition={{
                            lambda %0:{varying_scalar}, %1:{varying_counter}, %2:{varying_stack}, %3:{varying_mask} .
                            let %4:{varying_scalar} = const 10.0
                                %5:{varying_predicate} = compare [direction=LessThan] %0 %4
                            in (%5)
                        }},
                        body={{
                            lambda %0:{varying_scalar}, %1:{varying_counter}, %2:{varying_stack}, %3:{varying_mask} .
                            let %4:{varying_scalar} = mul %0 %0
                                %5:{varying_slice} = broadcast [
                                    output_type={varying_slice},
                                    output_axes=[],
                                ] %0
                                %6:{varying_stack} = dynamic_update_slice %2 %5 %1
                                %7:{varying_predicate} = one [type={varying_predicate}]
                                %8:{varying_mask_slice} = broadcast [
                                    output_type={varying_mask_slice},
                                    output_axes=[],
                                ] %7
                                %9:{varying_mask} = dynamic_update_slice %3 %8 %1
                                %10:{varying_counter} = one [type={varying_counter}]
                                %11:{varying_counter} = add %1 %10
                            in (%4, %11, %6, %9)
                        }},
                    ]
                    %9:{varying_scalar} = scan [carry_count=1, iteration_count=3, reverse=false] %1 %7 %8 [
                        body={{
                            lambda %0:i64[], %1:{varying_scalar}, %2:{varying_scalar}, %3:{varying_predicate} .
                            let %4:{varying_scalar} = condition %3 %1 %2 [
                                true={{
                                    lambda %0:{varying_scalar}, %1:{varying_scalar} .
                                    let %2:{varying_scalar} = mul %1 %0
                                        %3:{varying_scalar} = mul %1 %0
                                        %4:{varying_scalar} = add %2 %3
                                    in (%4)
                                }},
                                false={{
                                    lambda %0:{varying_scalar}, %1:{varying_scalar} .
                                    in (%0)
                                }},
                            ]
                            in (%4)
                        }},
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
            formatdoc! {"
                lambda %0:{varying_scalar}, %1:{varying_mask}, %2:{varying_stack} .
                let %3:{varying_scalar} = scan [carry_count=1, iteration_count=3, reverse=true] %0 %1 %2 [
                    body={{
                        lambda %0:i64[], %1:{varying_scalar}, %2:{varying_predicate}, %3:{varying_scalar} .
                        let %4:{varying_scalar} = condition %2 %1 %3 [
                            true={{
                                lambda %0:{varying_scalar}, %1:{varying_scalar} .
                                let %2:{varying_scalar} = mul %1 %0
                                    %3:{varying_scalar} = mul %1 %0
                                    %4:{varying_scalar} = add %2 %3
                                in (%4)
                            }},
                            false={{
                                lambda %0:{varying_scalar}, %1:{varying_scalar} .
                                in (%0)
                            }},
                        ]
                        in (%4)
                    }},
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
        // The masked normal form of a batched-predicate loop selects every array state element under the predicate of
        // the current state, which its condition reduces and its body recomputes, so the rendering pins one predicate
        // evaluation in each region and no mask carry. Inside a manual region, the Boolean mask, which carries no
        // tangent, first takes the variation of the state element that it selects through `parallel_vary`, and its
        // broadcast keeps the resulting sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let counter_type = ArrayType::new_static(DataType::I64, [2]);
        let state_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_sharding(Sharding::replicated(mesh.clone(), 2).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let counter = builder.add_input(counter_type.clone());
        let state = builder.add_input(state_type.clone());
        let one = builder.add_constant(Array::vector(vec![1i64, 1]).unwrap());
        let next_counter =
            builder.add_instruction(AddOperation::new(), Vec::new(), vec![counter, one], None).unwrap()[0];
        let next_state = builder.add_instruction(SinOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![next_counter, next_state], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let (_, program) = TestTracingContext::trace_with_named_axes(
            |inputs: Vec<Tracer<TestTracingContext>>| {
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
        let varying_state = "f32[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}]";
        let varying_slice = "f32[1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}]";
        let varying_stack = "f32[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}]";
        let invariant_predicate = "bool[2][sharding={mesh<['x'=2:manual]>, [{}]}]";
        let varying_predicate = "bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]";
        let varying_mask = "bool[2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}], varying_manual={'x'}}]";
        let varying_mask_slice = "bool[1, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}]";
        let varying_mask_stack = "bool[3, 2, 3][sharding={mesh<['x'=2:manual]>, [{}, {}, {}], varying_manual={'x'}}]";
        let invariant_index = "i64[][sharding={mesh<['x'=2:manual]>, []}]";
        let varying_index = "i64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]";
        assert_eq!(
            jvp.to_string(),
            formatdoc! {"
                lambda %0:i64[2], %1:{varying_state}, %2:{varying_state} .
                let %3:i64[] = zero [type=i64[]]
                    %4:{varying_stack} = zero [
                        type={varying_stack},
                    ]
                    %5:{varying_mask_stack} = zero [
                        type={varying_mask_stack},
                    ]
                    %6:bool[3] = zero [type=bool[3]]
                    %7:i64[2], %8:{varying_state}, %9:i64[], %10:{varying_stack}, %11:{varying_mask_stack}, \
                    %12:bool[3] = while [iteration_bound=3] %0 %1 %3 %4 %5 %6 [
                        condition={{
                            lambda %0:i64[2], %1:{varying_state}, %2:i64[], %3:{varying_stack}, \
                            %4:{varying_mask_stack}, %5:bool[3] .
                            let %6:i64[2] = const [1, 2]
                                %7:bool[2] = compare [direction=LessThan] %0 %6
                                %8:bool[] = reduce [kind=any, axes=[0]] %7
                            in (%8)
                        }},
                        body={{
                            lambda %0:i64[2], %1:{varying_state}, %2:i64[], %3:{varying_stack}, \
                            %4:{varying_mask_stack}, %5:bool[3] .
                            let %6:i64[2] = const [1, 2]
                                %7:i64[2] = const [1, 1]
                                %8:bool[2] = compare [direction=LessThan] %0 %6
                                %9:i64[2] = add %0 %7
                                %10:i64[2] = select %8 %9 %0
                                %11:{invariant_predicate} = broadcast [output_type={invariant_predicate}, \
                                output_axes=[0]] %8
                                %12:{varying_predicate} = parallel_vary [axis_name=\"x\"] %11
                                %13:{varying_mask} = broadcast [
                                    output_type={varying_mask},
                                    output_axes=[0],
                                ] %12
                                %14:{varying_state} = sin %1
                                %15:{varying_state} = select %13 %14 %1
                                %16:{varying_state} = cos %1
                                %17:{varying_slice} = broadcast [
                                    output_type={varying_slice},
                                    output_axes=[1, 2],
                                ] %16
                                %18:{invariant_index} = broadcast [output_type={invariant_index}, output_axes=[]] %2
                                %19:{varying_index} = parallel_vary [axis_name=\"x\"] %18
                                %20:{varying_index} = zero [type={varying_index}]
                                %21:{varying_stack} = dynamic_update_slice %3 %17 %19 %20 %20
                                %22:{varying_mask_slice} = broadcast [
                                    output_type={varying_mask_slice},
                                    output_axes=[1, 2],
                                ] %13
                                %23:{varying_mask_stack} = dynamic_update_slice %4 %22 %19 %20 %20
                                %24:bool[] = one [type=bool[]]
                                %25:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %24
                                %26:bool[3] = dynamic_update_slice %5 %25 %2
                                %27:i64[] = one [type=i64[]]
                                %28:i64[] = add %2 %27
                            in (%10, %15, %28, %21, %23, %26)
                        }},
                    ]
                    %13:{varying_state} = scan [carry_count=1, iteration_count=3, reverse=false] %2 %10 %11 %12 [
                        body={{
                            lambda %0:i64[], %1:{varying_state}, %2:{varying_state}, %3:{varying_mask}, %4:bool[] .
                            let %5:{varying_state} = condition %4 %1 %2 %3 [
                                true={{
                                    lambda %0:{varying_state}, %1:{varying_state}, %2:{varying_mask} .
                                    let %3:{varying_state} = mul %1 %0
                                        %4:{varying_state} = select %2 %3 %0
                                    in (%4)
                                }},
                                false={{
                                    lambda %0:{varying_state}, %1:{varying_state}, %2:{varying_mask} .
                                    in (%0)
                                }},
                            ]
                            in (%5)
                        }},
                    ]
                in (%7, %8, %13)
            "}
            .trim_end(),
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
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
        let state = builder.add_input(state_type.clone());
        let limit = builder.add_constant(Array::from_elements(state_type.clone(), &[10f32]).unwrap());
        let predicate = builder
            .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![state, limit], None)
            .unwrap()[0];
        let condition = builder
            .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let (_, program) = TestTracingContext::trace_with_named_axes(
            |inputs: Vec<Tracer<TestTracingContext>>| {
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
            Err(DifferentiationError::from(ProgramError::from(TypeError::invalid(format!(
                "`jvp` of a bounded `{WHILE_OPERATION_NAME}` loop requires every residual to vary over the manual axes \
                 that the predicate `bool[][sharding={{mesh<['x'=2:manual]>, [], varying_manual={{'x'}}}}]` varies \
                 over, but got residual storage `f32[][sharding={{mesh<['x'=2:manual]>, []}}]`",
            ))))),
        );
    }

    #[test]
    fn test_while_differentiation_runs_bounded_loops_in_shard_map_bodies() {
        // Forward and reverse mode differentiate bounded loops inside `shard_map` bodies, whose devices may run
        // different numbers of iterations.
        /// Builds a program that applies `shard_map` to inputs of the `global_types`, with a body that runs the
        /// bounded `while` loop over the condition and body `regions` on the local values of those inputs.
        fn mapped_while_program(
            shard_map: ShardMap,
            global_types: Vec<ArrayType>,
            regions: Vec<TestIrProgram>,
        ) -> TestIrProgram {
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
    fn test_while_differentiation_omits_the_dimension_state_tangent() {
        // The forwarded dimension carry has an empty tangent space, so it has neither a tangent input nor a tangent
        // output. From `x = 1`, the loop doubles `x` three times, so its value is 8 and its derivative is 8.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let state_types =
            vec![ArrayIrType::Dimension(extent_type.clone()), ArrayIrType::Array(ArrayType::scalar(DataType::F64))];
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        condition_builder.add_input(state_types[0].clone());
        let state = condition_builder.add_input(state_types[1].clone());
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
        let extent = body_builder.add_input(state_types[0].clone());
        let state = body_builder.add_input(state_types[1].clone());
        let doubled = body_builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::from(AddOperation::new())),
                Vec::new(),
                vec![state, state],
                None,
            )
            .unwrap()[0];
        let body = body_builder.build(vec![extent, doubled], vec![Placeholder; 2], vec![Placeholder; 2]).unwrap();
        let operation = WhileOperation::new().with_iteration_bound(4).unwrap();
        let program = while_program(operation, &[condition, body], state_types);

        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:dimension<extent ∈ [1, 8)>, %1:f64[], %2:f64[] .
                let %3:i64[] = zero [type=i64[]]
                    %4:bool[4] = zero [type=bool[4]]
                    %5:dimension<extent ∈ [1, 8)>, %6:f64[], %7:i64[], %8:bool[4] = while [iteration_bound=4] %0 %1 %3 \
                    %4 [
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
                    %9:f64[] = scan [carry_count=1, iteration_count=4, reverse=false] %2 %8 [
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

        let program = while_program(
            WhileOperation::new().with_iteration_bound(4).unwrap(),
            &[condition, body],
            vec![
                ArrayIrType::Dimension(extent_type.clone()),
                ArrayIrType::Array(vector_type),
                ArrayIrType::Array(ArrayType::scalar(DataType::I64)),
            ],
        );
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
                let %3:dimension<extent ∈ [1, 8)>, %4:f64[extent] = scan [carry_count=2, iteration_count=4, \
                reverse=false] %1 %0 %2 [
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

        let program = while_program(
            WhileOperation::new().with_iteration_bound(4).unwrap(),
            &[condition, body],
            vec![ArrayIrType::Array(scalar_f64), ArrayIrType::Array(scalar_u64)],
        );
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
                let %4:f64[] = scan [carry_count=1, iteration_count=4, reverse=false] %0 %1 %2 %3 [
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
            Ok(vec![array(Array::scalar(2.0).unwrap())]),
        );
    }

    #[test]
    fn test_while_differentiation_stages_one_fused_doubled_state_loop() {
        // Forward mode through a staged unbounded while loop stages one fused doubled-state loop whose condition reads
        // the primal half. For `f(x) = while (x < 16) { x = x * x }` at `x = 2` the loop runs twice (2 -> 4 -> 16), so
        // locally `f(x) = x⁴`: primal 16, tangent `4 x³ = 32`.
        let (while_operation, while_regions) = squaring_while_operation(16.0, None);
        let staged_regions = while_regions.clone();
        let (_, program) = TestTracingContext::trace(
            move |(x, tangent): (TestStagedTracer, TestStagedTracer)| {
                Ok(x.context().clone().jvp(
                    move |x, ()| Ok(bind_while(while_operation, staged_regions.clone(), &[x])?.remove(0)),
                    x,
                    tangent,
                    (),
                )?)
            },
            (ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)),
        )
        .unwrap();
        assert_eq!(
            program.interpret((Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap())),
            Ok((Array::scalar(16.0).unwrap(), Array::scalar(32.0).unwrap())),
        );

        // The fused forward-mode program contains exactly one while loop over the doubled state `[primal, tangent]`,
        // with no unrolling and no residual stacks, and its number of iterations stays data-dependent: interpreting the
        // same fused program at `x = 16` never enters the loop, so the state passes through unchanged.
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
        // A staged unbounded loop whose state mixes a differentiable carry with an element whose tangent space is
        // empty, here an `i64` iteration counter. The compact fused-JVP contract omits the counter's tangent entirely,
        // so the fused loop state is `[x, i, live(ẋ)]` and the counter's output dual is restored as a structural zero.
        // `f(x) = while (i < 3) { x = x * x; i = i + 1 }` at `x = 2` squares three times: primal `x⁸ = 256` with
        // tangent `8 x⁷ = 1024`.
        let float_type = ArrayType::scalar(DataType::F64);
        let counter_type = ArrayType::scalar(DataType::I64);
        let condition = {
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
            let mut builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let program = while_program(WhileOperation::new(), &[condition, body], vec![float_type, counter_type.clone()]);

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
    fn test_while_differentiation_fuses_unbounded_loops_with_batched_predicates() {
        // The eager rule cannot concretize the batched predicate `x < [16, 4, 2]` of an unbounded loop, so it falls
        // back to the fused doubled-state loop, whose doubled state still satisfies the predicate-prefix contract.
        // Squaring `x = [2, 2, 2]` runs 2, 1, and 0 iterations, so the value is `[16, 4, 2]` and the per-item
        // derivatives are `[4 x³, 2 x, 1] = [32, 4, 1]`.
        let state_type = ArrayType::new_static(DataType::F64, [3]);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        // One staged linearization pushes both tangents `1` and `2` forward.
        let (while_operation, while_regions) = squaring_while_operation(16.0, None);
        let scalar = ArrayType::scalar(DataType::F64);
        let (_, program) = TestTracingContext::trace(
            move |(x, first, second): (TestStagedTracer, TestStagedTracer, TestStagedTracer)| {
                let (output, pushforward) = x.context().clone().linearize(
                    move |x, ()| Ok(bind_while(while_operation, while_regions.clone(), &[x])?.remove(0)),
                    x,
                    (),
                )?;
                Ok((output, pushforward.apply(first)?, pushforward.apply(second)?))
            },
            (scalar.clone(), scalar.clone(), scalar),
        )
        .unwrap();
        assert_eq!(
            program.interpret((Array::scalar(2.0).unwrap(), Array::scalar(1.0).unwrap(), Array::scalar(2.0).unwrap())),
            Ok((Array::scalar(16.0).unwrap(), Array::scalar(32.0).unwrap(), Array::scalar(64.0).unwrap())),
        );

        // The staged linearization pins that structure: the primal program binds the primal loop alone and forwards
        // its initial state as the residual, and the tangent program keeps the fused loop whole.
        let (operation, regions) = squaring_while_operation(16.0, None);
        let linearization =
            while_program(operation, &regions, vec![ArrayType::scalar(DataType::F64)]).linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = while %0 [
                    condition={
                        lambda %0:f64[] .
                        let %1:f64[] = const 16.0
                            %2:bool[] = compare [direction=LessThan] %0 %1
                        in (%2)
                    },
                    body={
                        lambda %0:f64[] .
                        let %1:f64[] = mul %0 %0
                        in (%1)
                    },
                ]
                in (%1, %0)"},
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[], %3:f64[] = while %1 %0 [
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
                in (%3)"},
        );
    }

    #[test]
    fn test_while_differentiation_linearizes_unbounded_loops_with_batched_predicates() {
        // An unbounded loop with the batched predicate `x < [10, 3, 1]` squares `x = [2, 2, 2]` 2, 1, and 0 times, so
        // its value is `[16, 4, 2]` and its per-item derivatives are `[4 x³, 2 x, 1] = [32, 4, 1]`.
        let state_type = ArrayType::new_static(DataType::F64, [3]);
        let mut condition_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        let mut body_builder = ProgramBuilder::<Array, TestOperation>::new();
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
        assert_eq!(linearization.pullback().map(|_| ()), Err(DifferentiationError::from(while_transposition_error())));
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
            let bound = builder.add_constant(array(Array::scalar(2i64).unwrap()));
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
            let one = builder.add_constant(array(Array::scalar(1i64).unwrap()));
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
        let counter = builder.add_constant(array(Array::scalar(0i64).unwrap()));
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

        let inputs = vec![array(Array::scalar(1.5f32).unwrap()), array(Array::scalar(1.0f32).unwrap())];
        let expected = program
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .jvp()
            .unwrap()
            .interpret(inputs.clone())
            .unwrap();
        assert_eq!(expected, vec![array(Array::scalar(6.0f32).unwrap()), array(Array::scalar(4.0f32).unwrap())]);
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
        let bound = condition_builder.add_constant(array(Array::scalar(2i64).unwrap()));
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
        let one = body_builder.add_constant(array(Array::scalar(1i64).unwrap()));
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
        let counter = builder.add_constant(array(Array::scalar(0i64).unwrap()));
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
        let jvp = program.jvp_with_respect_to(&[1]).unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[], %2:f32[] .
                let %3:i64[] = const 0
                    %4:ref<f32[]>, %5:i64[], %6:f32[], %7:f32[] = while %0 %3 %1 %2 [
                        condition={
                            lambda %0:ref<f32[]>, %1:i64[], %2:f32[], %3:f32[] .
                            let %4:i64[] = const 2
                                %5:bool[] = compare [direction=LessThan] %1 %4
                            in (%5)
                        },
                        body={
                            lambda %0:ref<f32[]>, %1:i64[], %2:f32[], %3:f32[] .
                            let %4:i64[] = const 1
                                %5:f32[] = reference_read %0
                                %6:f32[] = add %2 %5
                                %7:i64[] = add %1 %4
                            in (%0, %7, %6, %3)
                        },
                    ]
                in (%6, %7)"},
        );
        assert_eq!(
            jvp.interpret(vec![
                TestIrValue::Reference(ArrayReference::new(Array::scalar(5.0f32).unwrap())),
                array(Array::scalar(3.0f32).unwrap()),
                array(Array::scalar(2.0f32).unwrap()),
            ]),
            Ok(vec![array(Array::scalar(13.0f32).unwrap()), array(Array::scalar(2.0f32).unwrap())]),
        );
    }

    #[test]
    fn test_while_differentiation_after_local_reference_discharge() {
        // Discharging a local allocation turns the loop's mutated state into an ordinary carry, so the derived forward,
        // linearized, and pullback programs are pure and reference-free, and reverse mode remains available through
        // the bounded loop.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut condition_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let predicate = condition_builder.add_constant(array(Array::scalar(true).unwrap()));
        let condition = condition_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(array(Array::scalar(1.0f32).unwrap()));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
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
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let discharged = source.discharge_references(0).unwrap().into_program_without_external_references().unwrap();
        let jvp = discharged.jvp().unwrap();
        let linearization = discharged.linearize().unwrap();
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
                    %7:f32[] = scan [carry_count=1, iteration_count=2, reverse=false] %1 %6 [
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
                let %2:f32[] = scan [carry_count=1, iteration_count=2, reverse=false] %0 %1 [
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
                let %2:f32[] = scan [carry_count=1, iteration_count=2, reverse=true] %0 %1 [
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
            jvp.interpret(vec![array(Array::scalar(3.0f32).unwrap()), array(Array::scalar(2.0f32).unwrap())]),
            Ok(vec![array(Array::scalar(5.0f32).unwrap()), array(Array::scalar(2.0f32).unwrap())]),
        );
        let mut primal_outputs = linearization.primal().interpret(vec![array(Array::scalar(3.0f32).unwrap())]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![array(Array::scalar(5.0f32).unwrap())]);
        let mut pullback_inputs = vec![array(Array::scalar(4.0f32).unwrap())];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![array(Array::scalar(4.0f32).unwrap())]));
    }

    #[test]
    fn test_while_differentiation_falls_back_to_declared_regions_when_specialization_drops_a_refinement() {
        // Forward-mode differentiation specializes attached regions before applying the `while` rule. A body that
        // replaces its sharded carry with an unsharded constant cannot be specialized usefully, so differentiation uses
        // the declared regions, exactly like staging does.
        let state_type = ArrayType::scalar(DataType::F32);
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_type = state_type.clone().with_sharding(Sharding::replicated(mesh, 0)).unwrap();
        let regions = sharding_dropping_while_regions();
        let (output_types, program) = TestIrTracingContext::trace(
            |input: TestIrStagedTracer| {
                let (primal, tangent) = differentiate_at(input.clone()).jvp(input, |state| {
                    let context = state.context().clone();
                    Ok(context.bind(WhileOperation::new(), regions.clone(), &[state])?.remove(0))
                })?;
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
        let counter = array(Array::scalar(2.0).unwrap());
        let vector = array(Array::vector(vec![1.0, 2.0, 3.0]).unwrap());

        let jvp = refined_vector_while_program(None).jvp().unwrap();
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
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3] .
                let %2:i64[] = zero [type=i64[]]
                    %3:bool[4] = zero [type=bool[4]]
                    %4:f64[], %5:f64[3], %6:i64[], %7:bool[4] = while [iteration_bound=4] %0 %1 %2 %3 [
                        condition={
                            lambda %0:f64[], %1:f64[3], %2:i64[], %3:bool[4] .
                            let %4:f64[] = const 0.0
                                %5:bool[] = compare [direction=GreaterThan] %0 %4
                            in (%5)
                        },
                        body={
                            lambda %0:f64[], %1:f64[3], %2:i64[], %3:bool[4] .
                            let %4:f64[] = const 1.0
                                %5:f64[] = sub %0 %4
                                %6:f64[3] = add %1 %1
                                %7:i64[] = one [type=i64[]]
                                %8:i64[] = add %2 %7
                                %9:bool[] = one [type=bool[]]
                                %10:bool[1] = broadcast [output_type=bool[1], output_axes=[]] %9
                                %11:bool[4] = dynamic_update_slice %3 %10 %2
                            in (%5, %6, %8, %11)
                        },
                    ]
                in (%4, %5, %7)"},
        );
        let mut primal_outputs = linearization.primal().interpret(vec![counter, vector]).unwrap();
        let residuals = primal_outputs.split_off(2);
        assert_eq!(
            primal_outputs,
            vec![array(Array::scalar(0.0).unwrap()), array(Array::vector(vec![4.0, 8.0, 12.0]).unwrap())],
        );
        let pullback = linearization.pullback().unwrap().into_program();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[3], %2:bool[4] .
                let %3:f64[], %4:f64[3] = scan [carry_count=2, iteration_count=4, reverse=true] %0 %1 %2 [
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
    fn test_while_differentiation_composes_with_earlier_batching() {
        // Forward mode through a batched bounded loop under the non-concretizing tracing context: batching stages one
        // bounded loop with the per-item predicate `x < 8` and the iteration bound 5, so the JVP rule takes the
        // bounded staged path over its masked normal form, storing residual stacks and staging a masked linear scan.
        // Batch items [1, 5, 9] double 3, 1, and 0 times, so the primal is [8, 10, 9] and the per-item tangent scale
        // is two to the power of the number of iterations, [8, 2, 1].
        fn batched_bounded_while<V>(x: V) -> Result<V, ProgramError>
        where
            V: Value<Type = ArrayType> + Broadcast + Transpose,
            V::Domain: Context<Type = ArrayType, Value = V, Constant = Array, Operation = TestOperation>,
            TestOperation: BatchableOperation<V::Domain, ArrayBatchingPolicy>
                + From<TransposeOperation>
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
        let vector_f64 = ArrayType::new_static(DataType::F64, [3]);
        let input = Array::vector(vec![1.0, 5.0, 9.0]).unwrap();
        let ones = Array::vector(vec![1.0, 1.0, 1.0]).unwrap();
        let (_, program) = TestTracingContext::trace(
            |(x, tangent): (TestStagedTracer, TestStagedTracer)| {
                Ok(x.context().clone().jvp(|input, ()| batched_bounded_while(input), x, tangent, ())?)
            },
            (vector_f64.clone(), vector_f64.clone()),
        )
        .unwrap();
        assert_eq!(
            program.interpret((input.clone(), ones.clone())),
            Ok((Array::vector(vec![8.0, 10.0, 9.0]).unwrap(), Array::vector(vec![8.0, 2.0, 1.0]).unwrap())),
        );

        // The plain eager domain produces the same numbers.
        let (primal, tangent) = differentiate_at(Array::vector(vec![1.0, 5.0, 9.0]).unwrap())
            .jvp(Array::vector(vec![1.0, 1.0, 1.0]).unwrap(), batched_bounded_while)
            .unwrap();
        assert_eq!(primal, Array::vector(vec![8.0, 10.0, 9.0]).unwrap());
        assert_eq!(tangent, Array::vector(vec![8.0, 2.0, 1.0]).unwrap());

        // Reverse mode composes through the masked linear scan: the pullback transposes the scan without any `while`
        // loop, and the per-item gradients match the tangent scales.
        let mut transposed = None;
        let (_, program) = TestTracingContext::trace(
            |(x, cotangent): (TestStagedTracer, TestStagedTracer)| {
                let (output, pullback) = x.context().clone().vjp(|input, ()| batched_bounded_while(input), x, ())?;
                transposed = Some(
                    pullback
                        .linear_program()
                        .transpose_with_trailing_residuals_shared(pullback.residuals().len(), &[])?
                        .program()
                        .to_string(),
                );
                Ok((output, pullback.apply_materialized(cotangent)?))
            },
            (vector_f64.clone(), vector_f64),
        )
        .unwrap();
        assert_eq!(
            program.interpret((input.clone(), ones.clone())),
            Ok((Array::vector(vec![8.0, 10.0, 9.0]).unwrap(), Array::vector(vec![8.0, 2.0, 1.0]).unwrap())),
        );
        assert_eq!(
            transposed.unwrap(),
            indoc! {"
                lambda %0:f64[3], %1:bool[5], %2:f64[5, 3], %3:bool[5, 3] .
                let %4:f64[3] = scan [carry_count=1, iteration_count=5, reverse=true] %0 %1 %2 %3 [
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

        let (operation, regions) = squaring_while_operation(10.0, Some(4));
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
                    %14:f64[3] = scan [carry_count=1, iteration_count=4, reverse=false] %1 %12 %13 [
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

        let (operation, regions) = squaring_while_operation(10.0, None);
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
    fn test_while_differentiation_composes_staged_linearizations_with_later_batching() {
        // Batching the tangent and pullback programs of a staged bounded linearization over tangents and cotangents
        // that share one primal point keeps their residual stacks replicated. `while (x < 10) { x = x * x }` at
        // `x = 2` squares twice to 16, so locally `f(x) = x⁴` and both programs scale by `4 x³ = 32`.
        let (operation, regions) = squaring_while_operation(10.0, Some(4));
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

        // The batched tangent and pullback programs are the same masked scan, in opposite iteration orders.
        let scan = |reverse: bool| {
            formatdoc! {"
                lambda %0:f64[3], %1:bool[4], %2:f64[4] .
                let %3:f64[3] = scan [carry_count=1, iteration_count=4, reverse={reverse}] %0 %1 %2 [
                    body={{
                        lambda %0:i64[], %1:f64[3], %2:bool[], %3:f64[] .
                        let %4:f64[3] = condition %2 %1 %3 [
                            true={{
                                lambda %0:f64[3], %1:f64[] .
                                let %2:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                                    %3:f64[3] = mul %2 %0
                                    %4:f64[3] = broadcast [output_type=f64[3], output_axes=[]] %1
                                    %5:f64[3] = mul %4 %0
                                    %6:f64[3] = add %3 %5
                                in (%6)
                            }},
                            false={{
                                lambda %0:f64[3], %1:f64[] .
                                in (%0)
                            }},
                        ]
                        in (%4)
                    }},
                ]
                in (%3)"}
        };

        let tangent = linearization
            .tangent()
            .batched(3, ShardingDimension::Replicated, axes.as_slice(), ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap();
        assert_eq!(tangent.output_axes(), &[BatchAxis::new(0)]);
        let tangent = tangent.into_parts().0;
        assert_eq!(tangent.to_string(), scan(false));
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
        assert_eq!(pullback.to_string(), scan(true));
        let mut pullback_inputs = vec![Array::vector(vec![1.0, 2.0, -1.0]).unwrap()];
        pullback_inputs.extend(residuals);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![Array::vector(vec![32.0, 64.0, -32.0]).unwrap()]));
    }

    #[test]
    fn test_while_transposition() {
        // A staged unbounded loop has no transposable form: its tangent program keeps the fused loop, which recomputes
        // primal state forward through the iterations, so transposing that program, or any other linear `while`,
        // reports the error of the `while` transposition rule.
        let error = DifferentiationError::from(while_transposition_error());
        let (operation, regions) = squaring_while_operation(16.0, None);
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
        let (while_operation, while_regions) = squaring_while_operation(16.0, None);
        let scalar = ArrayType::scalar(DataType::F64);
        let traced = TestTracingContext::trace(
            move |(x, cotangent): (TestStagedTracer, TestStagedTracer)| {
                let (_, pullback) = x
                    .context()
                    .clone()
                    .vjp(move |x, ()| Ok(bind_while(while_operation, while_regions.clone(), &[x])?.remove(0)), x, ())
                    .unwrap();
                pullback.apply(cotangent).map(|_| ())
            },
            (scalar.clone(), scalar),
        );
        assert_eq!(traced.map(|_| ()), Err(while_transposition_error()));
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
        let predicate = condition.add_constant(array(Array::scalar(false).unwrap()));
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

        let error = reference_carry_swap_error();
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
        let predicate = condition.add_constant(array(Array::scalar(true).unwrap()));
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
        assert_eq!(known.interpret(&TestIrEagerContext::new(), &[]), Ok(expected.clone()));
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
        assert_eq!(mixed.interpret(&TestIrEagerContext::new(), &[scalar(10.0)]), Ok(expected.clone()));

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
                    %8:ref<f32[]>, %9:ref<f32[]>, %10:f32[], %11:i64[], %12:bool[2] = while [iteration_bound=2] %4 %4 \
                    %1 %6 %7 [
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
                    %13:ref<f32[]>, %14:ref<f32[]>, %15:f32[] = scan [carry_count=3, iteration_count=2, reverse=false] \
                    %5 %5 %3 %12 [
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

        assert_eq!(program.transpose().map(|_| ()), Err(DifferentiationError::from(while_transposition_error())));
    }

    #[test]
    fn test_while_residual_stack_operation() {
        // The array family constructs each operation directly.
        let state_type = ArrayType::new_static(DataType::F32, [2]);
        let mask_type = ArrayType::new_static(DataType::Boolean, [2, 3]);
        let scalar_type = ArrayType::scalar(DataType::I64);
        assert_eq!(
            TestOperation::residual_stack_zero(state_type.clone()),
            TestOperation::Zero(ZeroOperation::new(state_type.clone())),
        );
        assert_eq!(
            TestOperation::residual_stack_one(scalar_type.clone()),
            TestOperation::One(OneOperation::new(scalar_type.clone())),
        );
        assert_eq!(
            TestOperation::residual_stack_broadcast(mask_type.clone(), vec![0]),
            TestOperation::Broadcast(BroadcastOperation::new(mask_type.clone(), vec![0])),
        );
        assert_eq!(
            TestOperation::residual_stack_update(),
            TestOperation::DynamicUpdateSlice(DynamicUpdateSliceOperation::new()),
        );
        assert_eq!(TestOperation::residual_stack_add(), TestOperation::Add(AddOperation::new()));
        assert_eq!(TestOperation::residual_stack_select(), TestOperation::Select(SelectOperation::new()));
        assert_eq!(
            TestOperation::mask_reduce_any(vec![0, 1]),
            TestOperation::Reduce(ReduceOperation::new(vec![0, 1], ReductionKind::Any)),
        );
        assert_eq!(
            TestOperation::residual_stack_vary("x", &scalar_type),
            Ok(TestOperation::ParallelVary(ParallelVaryOperation::new("x".to_string()))),
        );

        // The array IR family wraps each operation of its array projection, including the statically shaped zeros and
        // ones that it lifts from array-typed state.
        assert_eq!(
            TestIrOperation::residual_stack_zero(state_type.clone().into()),
            TestIrOperation::Array(TestOperation::Zero(ZeroOperation::new(state_type))),
        );
        assert_eq!(
            TestIrOperation::residual_stack_one(scalar_type.clone().into()),
            TestIrOperation::Array(TestOperation::One(OneOperation::new(scalar_type.clone()))),
        );
        assert_eq!(
            TestIrOperation::residual_stack_broadcast(mask_type.clone(), vec![0]),
            TestIrOperation::Array(TestOperation::Broadcast(BroadcastOperation::new(mask_type, vec![0]))),
        );
        assert_eq!(
            TestIrOperation::residual_stack_update(),
            TestIrOperation::Array(TestOperation::DynamicUpdateSlice(DynamicUpdateSliceOperation::new())),
        );
        assert_eq!(
            TestIrOperation::residual_stack_add(),
            TestIrOperation::Array(TestOperation::Add(AddOperation::new())),
        );
        assert_eq!(
            TestIrOperation::residual_stack_select(),
            TestIrOperation::Array(TestOperation::Select(SelectOperation::new())),
        );
        assert_eq!(
            TestIrOperation::mask_reduce_any(vec![0, 1]),
            TestIrOperation::Array(TestOperation::Reduce(ReduceOperation::new(vec![0, 1], ReductionKind::Any))),
        );
        assert_eq!(
            TestIrOperation::residual_stack_vary("x", &scalar_type),
            Ok(TestIrOperation::Array(TestOperation::ParallelVary(ParallelVaryOperation::new("x".to_string())))),
        );
    }

    #[test]
    fn test_while_options_new() {
        // New options describe the default unbounded loop.
        let options = WhileOptions::new();
        assert_eq!(options.iteration_bound(), None);
        assert_eq!(options, WhileOptions::default());
    }

    #[test]
    fn test_while_options_with_iteration_bound() {
        // The bound is validated like the bound of the operation, and `None` restores the default unbounded loop.
        let options = WhileOptions::new().with_iteration_bound(3).unwrap();
        assert_eq!(options.iteration_bound(), Some(3));
        assert_eq!(options.with_iteration_bound(None).map(|options| options.iteration_bound()), Ok(None));
        assert_eq!(
            WhileOptions::new()
                .with_iteration_bound(MAX_DIMENSION_EXTENT)
                .map(|options| options.iteration_bound()),
            Ok(Some(MAX_DIMENSION_EXTENT)),
        );
        assert_eq!(
            WhileOptions::new().with_iteration_bound(0),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` iteration bound must be at least 1",
            )))),
        );
        // On hosts whose `usize` cannot exceed the backend maximum, no bound is rejected for being too large.
        if let Some(bound) = MAX_DIMENSION_EXTENT.checked_add(1) {
            assert_eq!(
                WhileOptions::new().with_iteration_bound(bound),
                Err(ProgramError::Type(TypeError::invalid(format!(
                    "`{WHILE_OPERATION_NAME}` iteration bound {bound} exceeds the maximum supported extent \
                     {MAX_DIMENSION_EXTENT}",
                )))),
            );
        }
    }

    #[test]
    fn test_while_while() {
        // The loop doubles `value` while `count < 3`, and an iteration bound truncates it after that many iterations.
        assert_eq!(
            r#while(
                |(count, _)| count.less_than(&StagingContext::constant(count.context(), Array::scalar(3i64).unwrap())),
                |(count, value)| {
                    let one = StagingContext::constant(count.context(), Array::scalar(1i64).unwrap());
                    let two = StagingContext::constant(value.context(), Array::scalar(2f32).unwrap());
                    Ok((count + one, value * two))
                },
                (Array::scalar(0i64).unwrap(), Array::scalar(1f32).unwrap()),
                WhileOptions::new(),
            ),
            Ok((Array::scalar(3i64).unwrap(), Array::scalar(8f32).unwrap())),
        );
        assert_eq!(
            r#while(
                |(count, _)| count.less_than(&StagingContext::constant(count.context(), Array::scalar(3i64).unwrap())),
                |(count, value)| {
                    let one = StagingContext::constant(count.context(), Array::scalar(1i64).unwrap());
                    let two = StagingContext::constant(value.context(), Array::scalar(2f32).unwrap());
                    Ok((count + one, value * two))
                },
                (Array::scalar(0i64).unwrap(), Array::scalar(1f32).unwrap()),
                WhileOptions::new().with_iteration_bound(2).unwrap(),
            ),
            Ok((Array::scalar(2i64).unwrap(), Array::scalar(4f32).unwrap())),
        );
    }

    #[test]
    fn test_while_while_empty_state() {
        // Without state values, the global function has no domain to recover. An explicit context traces the condition
        // over the empty state, but the condition has no value to compute a predicate from, so its own error surfaces.
        let error = ProgramError::InvalidArgument { message: "the empty state has no predicate".to_string() };
        assert_eq!(
            TestEagerContext::new().r#while(|()| Err(error.clone()), |()| Ok(()), (), WhileOptions::new()),
            Err(error.clone()),
        );
        assert_eq!(
            r#while::<Array, (), _, _>(|()| Err(error.clone()), |()| Ok(()), (), WhileOptions::new()),
            Err(ProgramError::InvalidArgument {
                message: format!("`{WHILE_OPERATION_NAME}` requires a state value to recover its context from"),
            }),
        );
    }

    #[test]
    fn test_while_while_staging() {
        // Under tracing, the frontend stages one `while` whose regions are the traced condition and body, with the
        // iteration bound of its options.
        let (_, program) = TestTracingContext::trace(
            |(count, value)| {
                r#while(
                    |(count, _)| {
                        count.less_than(&StagingContext::constant(count.context(), Array::scalar(3i64).unwrap()))
                    },
                    |(count, value)| {
                        let one = StagingContext::constant(count.context(), Array::scalar(1i64).unwrap());
                        let two = StagingContext::constant(value.context(), Array::scalar(2f32).unwrap());
                        Ok((count + one, value * two))
                    },
                    (count, value),
                    WhileOptions::new().with_iteration_bound(4)?,
                )
            },
            (ArrayType::scalar(DataType::I64), ArrayType::scalar(DataType::F32)),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:i64[], %1:f32[] .
                let %2:i64[], %3:f32[] = while [iteration_bound=4] %0 %1 [
                    condition={
                        lambda %0:i64[], %1:f32[] .
                        let %2:i64[] = const 3
                            %3:bool[] = compare [direction=LessThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:i64[], %1:f32[] .
                        let %2:i64[] = const 1
                            %3:f32[] = const 2.0
                            %4:i64[] = add %0 %2
                            %5:f32[] = mul %1 %3
                        in (%4, %5)
                    },
                ]
                in (%2, %3)"},
        );
        assert_eq!(
            program.interpret((Array::scalar(0i64).unwrap(), Array::scalar(1f32).unwrap())),
            Ok((Array::scalar(3i64).unwrap(), Array::scalar(8f32).unwrap())),
        );
    }

    #[test]
    fn test_while_while_structure() {
        /// A state whose static tag must survive tracing, without requiring `Clone` on runtime structures.
        #[derive(ryft_macros::Parameterized)]
        struct State<P: Parameter> {
            /// Runtime state leaf.
            value: P,

            /// Static state tag.
            tag: &'static str,
        }

        impl Clone for State<Placeholder> {
            fn clone(&self) -> Self {
                Self { value: self.value, tag: self.tag }
            }
        }

        impl PartialEq for State<Placeholder> {
            fn eq(&self, other: &Self) -> bool {
                self.tag == other.tag
            }
        }

        // Tuples and custom structures nest, and the static tag reaches both functions and the final state.
        let (count, state) = r#while(
            |(count, state)| {
                assert_eq!(state.tag, "state");
                count.less_than(&StagingContext::constant(count.context(), Array::scalar(2i64).unwrap()))
            },
            |(count, state)| {
                let one = StagingContext::constant(count.context(), Array::scalar(1i64).unwrap());
                Ok((count + one, State { value: state.value.clone() + state.value, tag: state.tag }))
            },
            (Array::scalar(0i64).unwrap(), State { value: Array::scalar(1f32).unwrap(), tag: "state" }),
            WhileOptions::new(),
        )
        .unwrap();
        assert_eq!(count, Array::scalar(2i64).unwrap());
        assert_eq!(state.value, Array::scalar(4f32).unwrap());
        assert_eq!(state.tag, "state");

        // The body must return the static fields that it received.
        let result = r#while(
            |state| state.value.less_than(&state.value),
            |state| Ok(State { value: state.value, tag: "changed" }),
            State { value: Array::scalar(0f32).unwrap(), tag: "state" },
            WhileOptions::new(),
        );
        assert!(matches!(
            result,
            Err(ProgramError::InvalidArgument { message })
                if message == format!(
                    "`{WHILE_OPERATION_NAME}` body must preserve the state parameter structure and static fields",
                ),
        ));
    }

    #[test]
    fn test_while_while_validation() {
        // The traced regions are validated as a whole before binding: the predicate must be Boolean, and the body must
        // return every reference carry as the reference that it received.
        assert_eq!(
            r#while(|value| Ok(value), |value| Ok(value), Array::scalar(0f32).unwrap(), WhileOptions::new()),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` condition output type must be a Boolean array, but got `f32[]`",
            )))),
        );
        let first = TestIrValue::Reference(ArrayReference::new(Array::scalar(1f32).unwrap()));
        let second = TestIrValue::Reference(ArrayReference::new(Array::scalar(2f32).unwrap()));
        assert_eq!(
            r#while(
                |(_, _, flag)| Ok(flag),
                |(first, second, flag)| Ok((second, first, flag)),
                (first, second, array(Array::scalar(false).unwrap())),
                WhileOptions::new(),
            ),
            Err(reference_carry_swap_error()),
        );
    }

    #[test]
    fn test_while_while_batched_predicate() {
        // A vector predicate over vector state masks the state per item: each item stops being updated once its own
        // predicate is false, while the loop runs as long as any item continues, up to the iteration bound.
        let limits = Array::vector(vec![2f32, 4.0, 1.0]).unwrap();
        assert_eq!(
            r#while(
                |x| x.less_than(&StagingContext::constant(x.context(), limits.clone())),
                |x| Ok(x.clone() + StagingContext::constant(x.context(), Array::scalar(1f32).unwrap())),
                Array::vector(vec![0f32, 0.0, 5.0]).unwrap(),
                WhileOptions::new(),
            ),
            Ok(Array::vector(vec![2f32, 4.0, 5.0]).unwrap()),
        );
        assert_eq!(
            r#while(
                |x| x.less_than(&StagingContext::constant(x.context(), limits.clone())),
                |x| Ok(x.clone() + StagingContext::constant(x.context(), Array::scalar(1f32).unwrap())),
                Array::vector(vec![0f32, 0.0, 5.0]).unwrap(),
                WhileOptions::new().with_iteration_bound(3).unwrap(),
            ),
            Ok(Array::vector(vec![2f32, 3.0, 5.0]).unwrap()),
        );
    }

    #[test]
    fn test_while_while_references() {
        // A reference leaf enters the loop whole: the condition increments the counter that it holds before comparing
        // it with 3, and the body returns the reference that it received.
        let reference = ArrayReference::new(Array::scalar(0f32).unwrap());
        assert_eq!(
            r#while(
                |counter| {
                    counter.add_update(&StagingContext::constant(
                        counter.context(),
                        array(Array::scalar(1f32).unwrap()),
                    ))?;
                    let limit = StagingContext::constant(counter.context(), array(Array::scalar(3f32).unwrap()));
                    counter.read()?.less_than(&limit)
                },
                |counter| Ok(counter),
                TestIrValue::Reference(reference.clone()),
                WhileOptions::new(),
            ),
            Ok(TestIrValue::Reference(reference.clone())),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(3f32).unwrap()));

        // Staged, the frontend produces the loop that `reference_while_program` builds by hand, except that its
        // constants precede the other instructions of the condition, and both programs leave the same counter behind.
        let (_, program) = TestIrTracingContext::trace(
            |reference: Tracer<_>| {
                let reference = r#while(
                    |counter| {
                        let one = StagingContext::constant(counter.context(), array(Array::scalar(1f32).unwrap()));
                        counter.add_update(&one)?;
                        let limit = StagingContext::constant(counter.context(), array(Array::scalar(3f32).unwrap()));
                        counter.read()?.less_than(&limit)
                    },
                    |counter| Ok(counter),
                    reference,
                    WhileOptions::new(),
                )?;
                reference.read()
            },
            ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32))),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:ref<f32[]> .
                let %1:ref<f32[]> = while %0 [
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
                    %2:f32[] = reference_read %1
                in (%2)"},
        );
        assert_eq!(
            program.interpret(TestIrValue::Reference(ArrayReference::new(Array::scalar(0f32).unwrap()))),
            Ok(array(Array::scalar(3f32).unwrap())),
        );
        assert_eq!(
            reference_while_program(incrementing_condition(3.0), None)
                .interpret(vec![TestIrValue::Reference(ArrayReference::new(Array::scalar(0f32).unwrap()))]),
            Ok(vec![array(Array::scalar(3f32).unwrap())]),
        );
    }

    #[test]
    fn test_while_while_propagates_manual_variation_across_state() {
        // The manual variation of one state element propagates to the elements that the body rotates it into, so the
        // frontend widens the initial state to the fixed point of that propagation before it binds the loop.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"])]).unwrap();
        let invariant_type = ArrayType::new_static(DataType::F32, [1]).with_sharding(replicated.clone()).unwrap();
        let varying_type = ArrayType::new_static(DataType::F32, [1])
            .with_sharding(replicated.clone().with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let invariant = "f32[1][sharding={mesh<['m'=2:manual]>, [{}]}]";
        let varying = "f32[1][sharding={mesh<['m'=2:manual]>, [{}], varying_manual={'m'}}]";

        // Variation reaches the second state element first, then the first one on the next trace. Entry transitions
        // belong outside the loop even when it runs no iteration, where the initial values still acquire the
        // fixed-point types. A constant predicate varies over no axis, so it requires no further widening.
        for (continues, first, second) in [(false, [1f32, 1.0], [2f32, 2.0]), (true, [3f32, 4.0], [3f32, 4.0])] {
            let (_, program) = TestIrTracingContext::trace_with_named_axes(
                |inputs: Vec<Tracer<_>>| {
                    let (first, second, source) = r#while(
                        |(first, _, _)| {
                            Ok(StagingContext::constant(first.context(), array(Array::scalar(continues).unwrap())))
                        },
                        |(_, second, source)| Ok((second, source.clone(), source)),
                        (inputs[0].clone(), inputs[1].clone(), inputs[2].clone()),
                        WhileOptions::new().with_iteration_bound(2)?,
                    )?;
                    Ok(vec![first, second, source])
                },
                vec![
                    ArrayIrType::Array(invariant_type.clone()),
                    ArrayIrType::Array(invariant_type.clone()),
                    ArrayIrType::Array(varying_type.clone()),
                ],
                vec![("m".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
            assert_eq!(
                program.to_string(),
                formatdoc! {"
                    lambda %0:{invariant}, %1:{invariant}, %2:{varying} .
                    let %3:{varying} = parallel_vary [axis_name=\"m\"] %0
                        %4:{varying} = parallel_vary [axis_name=\"m\"] %1
                        %5:{varying}, %6:{varying}, %7:{varying} = while [iteration_bound=2] %3 %4 %2 [
                            condition={{
                                lambda %0:{varying}, %1:{varying}, %2:{varying} .
                                let %3:bool[] = const {continues}
                                in (%3)
                            }},
                            body={{
                                lambda %0:{varying}, %1:{varying}, %2:{varying} .
                                in (%1, %2, %2)
                            }},
                        ]
                    in (%5, %6, %7)"},
            );

            // Under `shard_map`, every device rotates its own copy of the state, so the outputs are varying.
            let map = ShardMap::new(
                mesh.clone(),
                vec![replicated.clone(), replicated.clone(), sharded.clone()],
                vec![sharded.clone(); 3],
                Vec::new(),
            )
            .unwrap();
            let global_types = vec![
                ArrayType::new_static(DataType::F32, [1]).into(),
                ArrayType::new_static(DataType::F32, [1]).into(),
                ArrayType::new_static(DataType::F32, [2]).into(),
            ];
            let operation = ShardMapOperation::from_program(&program, global_types, map).unwrap();
            let output_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharded.clone()).unwrap();
            assert_eq!(
                TestIrEagerContext::new().bind(
                    operation,
                    vec![program],
                    &[
                        array(Array::vector(vec![1f32]).unwrap()),
                        array(Array::vector(vec![2f32]).unwrap()),
                        array(Array::vector(vec![3f32, 4.0]).unwrap()),
                    ],
                ),
                Ok(vec![
                    array(Array::from_elements(output_type.clone(), &first).unwrap()),
                    array(Array::from_elements(output_type.clone(), &second).unwrap()),
                    array(Array::from_elements(output_type, &[3f32, 4.0]).unwrap()),
                ]),
            );
        }
    }

    #[test]
    fn test_while_while_widens_state_to_a_varying_predicate() {
        // The counter varies over `m`, and so does the predicate computed from it. Devices whose counters differ run
        // different numbers of iterations, so the invariant value is widened to vary over `m` before the loop, even
        // though the body alone would keep it invariant.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"])]).unwrap();
        let invariant_type = ArrayType::new_static(DataType::F32, [1]).with_sharding(replicated.clone()).unwrap();
        let varying_type = ArrayType::new_static(DataType::F32, [1])
            .with_sharding(replicated.clone().with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let invariant = "f32[1][sharding={mesh<['m'=2:manual]>, [{}]}]";
        let varying = "f32[1][sharding={mesh<['m'=2:manual]>, [{}], varying_manual={'m'}}]";
        let predicate = "bool[1][sharding={mesh<['m'=2:manual]>, [{}], varying_manual={'m'}}]";
        let (_, program) = TestIrTracingContext::trace_with_named_axes(
            |inputs: Vec<Tracer<_>>| {
                let (counter, value) = r#while(
                    |(counter, _)| {
                        let limit = Array::from_elements(varying_type.clone(), &[3f32]).unwrap();
                        counter.less_than(&StagingContext::constant(counter.context(), array(limit)))
                    },
                    |(counter, value)| {
                        let one = Array::from_elements(varying_type.clone(), &[1f32]).unwrap();
                        Ok((
                            counter.clone() + StagingContext::constant(counter.context(), array(one)),
                            value.clone() + value,
                        ))
                    },
                    (inputs[0].clone(), inputs[1].clone()),
                    WhileOptions::new(),
                )?;
                Ok(vec![counter, value])
            },
            vec![ArrayIrType::Array(varying_type.clone()), ArrayIrType::Array(invariant_type.clone())],
            vec![("m".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{varying}, %1:{invariant} .
                let %2:{varying} = parallel_vary [axis_name=\"m\"] %1
                    %3:{varying}, %4:{varying} = while %0 %2 [
                        condition={{
                            lambda %0:{varying}, %1:{varying} .
                            let %2:{varying} = const [3.0]
                                %3:{predicate} = compare [direction=LessThan] %0 %2
                            in (%3)
                        }},
                        body={{
                            lambda %0:{varying}, %1:{varying} .
                            let %2:{varying} = const [1.0]
                                %3:{varying} = add %0 %2
                                %4:{varying} = add %1 %1
                            in (%3, %4)
                        }},
                    ]
                in (%3, %4)"},
        );

        // Device 0 counts from 0 and doubles the value three times, while device 1 counts from 2 and doubles it once.
        let map = ShardMap::new(mesh, vec![sharded.clone(), replicated], vec![sharded.clone(); 2], Vec::new()).unwrap();
        let global_types =
            vec![ArrayType::new_static(DataType::F32, [2]).into(), ArrayType::new_static(DataType::F32, [1]).into()];
        let operation = ShardMapOperation::from_program(&program, global_types, map).unwrap();
        let output_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharded).unwrap();
        assert_eq!(
            TestIrEagerContext::new().bind(
                operation,
                vec![program],
                &[array(Array::vector(vec![0f32, 2.0]).unwrap()), array(Array::vector(vec![1f32]).unwrap())],
            ),
            Ok(vec![
                array(Array::from_elements(output_type.clone(), &[3f32, 3.0]).unwrap()),
                array(Array::from_elements(output_type, &[8f32, 2.0]).unwrap()),
            ]),
        );
    }

    #[test]
    fn test_while_while_replays_entry_alignment_in_the_explicit_context() {
        /// Executes arrays through an explicit domain distinct from `Array::Domain`, recording every parent bind.
        #[derive(Clone)]
        struct RecordingContext {
            /// Parent operations observed by this explicit context.
            binds: Rc<RefCell<Vec<&'static str>>>,
        }

        impl Domain for RecordingContext {
            type Type = ArrayType;
            type Value = Array;
            type Constant = Array;
            type Operation = TestOperation;
        }

        impl Context for RecordingContext {
            fn lift(&self, constant: Array) -> Result<Array, ProgramError> {
                Ok(constant)
            }

            fn bind<'r, O: Into<Self::Operation>, R: Into<AttachedRegions<'r, Array, Self::Operation>>>(
                &self,
                operation: O,
                regions: R,
                inputs: &[Array],
            ) -> Result<Vec<Array>, ProgramError>
            where
                Array: 'r,
                Self::Operation: 'r,
            {
                let operation = operation.into();
                let regions = regions.into();
                self.binds.borrow_mut().push(operation.name());
                if matches!(operation, ArrayOperation::ParallelVary(_)) {
                    let output_types = operation.infer_output_types(
                        &inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
                        &regions.region_interfaces(),
                    )?;
                    return Ok(vec![Array::from_logical_bytes(output_types[0].clone(), &inputs[0].logical_bytes())?]);
                }
                TestEagerContext::new().bind(operation, regions, inputs)
            }

            fn is_eager(&self) -> bool {
                true
            }

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

        impl NamedAxes for RecordingContext {
            fn named_axis(&self, name: &str) -> Option<NamedAxis> {
                (name == "m").then(|| NamedAxis::Mesh {
                    axis: 0,
                    size: 2,
                    mesh: LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
                })
            }

            fn named_axes(&self) -> Vec<(String, NamedAxis)> {
                vec![("m".to_string(), self.named_axis("m").unwrap())]
            }
        }

        // The varying counter makes the predicate vary, so the invariant value is widened once, before the loop,
        // through the explicit context rather than through the domain of the values.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant_type =
            ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let varying_type = ArrayType::scalar(DataType::F32)
            .with_sharding(Sharding::replicated(mesh, 0).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        let context = RecordingContext { binds: Rc::new(RefCell::new(Vec::new())) };
        assert_eq!(
            context.r#while(
                |(counter, _)| {
                    let limit = Array::from_elements(varying_type.clone(), &[3f32]).unwrap();
                    counter.less_than(&StagingContext::constant(counter.context(), limit))
                },
                |(counter, value)| {
                    let one = Array::from_elements(varying_type.clone(), &[1f32]).unwrap();
                    Ok((counter.clone() + StagingContext::constant(counter.context(), one), value.clone() + value))
                },
                (
                    Array::from_elements(varying_type.clone(), &[0f32]).unwrap(),
                    Array::from_elements(invariant_type, &[1f32]).unwrap(),
                ),
                WhileOptions::new(),
            ),
            Ok((
                Array::from_elements(varying_type.clone(), &[3f32]).unwrap(),
                Array::from_elements(varying_type, &[8f32]).unwrap(),
            )),
        );
        assert_eq!(*context.binds.borrow(), vec![PARALLEL_VARY_OPERATION_NAME, WHILE_OPERATION_NAME]);
    }

    #[test]
    fn test_while_while_differentiation_and_batching() {
        // `while (x < 100, iteration_bound = 4) { x = x * x }` through the frontend agrees with the primitive loop that
        // `squaring_while_operation` builds by hand: at `x = 2` it squares three times, so locally `f(x) = x⁸`
        // with value 256 and derivative `8 x⁷ = 1024`.
        let (operation, regions) = squaring_while_operation(100.0, Some(4));
        let expected = Ok((Array::scalar(256.0).unwrap(), Array::scalar(1024.0).unwrap()));
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                r#while(
                    |x| x.less_than(&StagingContext::constant(x.context(), Array::scalar(100.0).unwrap())),
                    |x| Ok(x.clone() * x),
                    x,
                    WhileOptions::new().with_iteration_bound(4)?,
                )
            }),
            expected,
        );
        assert_eq!(
            differentiate_at(Array::scalar(2.0).unwrap()).jvp(Array::scalar(1.0).unwrap(), |x| {
                bind_while(operation, regions.clone(), &[x]).map(|mut outputs| outputs.remove(0))
            }),
            expected,
        );
        let (output, pullback) = differentiate_at(Array::scalar(2.0).unwrap())
            .vjp(|x| {
                r#while(
                    |x| x.less_than(&StagingContext::constant(x.context(), Array::scalar(100.0).unwrap())),
                    |x| Ok(x.clone() * x),
                    x,
                    WhileOptions::new().with_iteration_bound(4)?,
                )
            })
            .unwrap();
        assert_eq!(output, Array::scalar(256.0).unwrap());
        assert_eq!(pullback.apply(Array::scalar(1.0).unwrap()), Ok(MaybeZero::Value(Array::scalar(1024.0).unwrap())));
        check_gradient!(
            |x| r#while(
                |x| x.less_than(&StagingContext::constant(x.context(), Array::scalar(100.0).unwrap())),
                |x| Ok(x.clone() * x),
                x,
                WhileOptions::new().with_iteration_bound(4)?,
            ),
            at = Array::scalar(2.0).unwrap(),
            step = 1e-4,
            tolerance = 1e-3,
        );

        // Batching makes the predicate batched, so each item stops once its own predicate is false: `11` squares once.
        let inputs = Array::vector(vec![2.0, 11.0]).unwrap();
        let expected = Ok(Array::vector(vec![256.0, 121.0]).unwrap());
        assert_eq!(
            batch(
                |x| {
                    r#while(
                        |x| x.less_than(&StagingContext::constant(x.context(), Array::scalar(100.0).unwrap())),
                        |x| Ok(x.clone() * x),
                        x,
                        WhileOptions::new().with_iteration_bound(4)?,
                    )
                },
                inputs.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            ),
            expected,
        );
        assert_eq!(
            batch(
                |x| bind_while(operation, regions.clone(), &[x]).map(|mut outputs| outputs.remove(0)),
                inputs,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            ),
            expected,
        );
    }

    #[test]
    fn test_masked_while_programs() {
        // The masked normal form carries no mask: its condition reduces the predicate of the current state and its body
        // recomputes that predicate to select the updates, instead of narrowing a carried mask, so a finished item
        // rejoins whenever its predicate turns true again, and the condition is never evaluated on the state after
        // the last body application that the bound admits.
        let regions = rejoining_while_regions();
        let (masked_condition, masked_body) = masked_while_programs(&regions[0], &regions[1]).unwrap();
        assert_eq!(
            masked_condition.to_string(),
            indoc! {"
                lambda %0:f64[2] .
                let %1:f64[2] = const [1.0, 3.0]
                    %2:f64[2] = const [2.0, 10.0]
                    %3:f64[2] = const [3.0, 3.0]
                    %4:f64[2] = reverse [axes=[0]] %0
                    %5:bool[2] = compare [direction=LessThan] %0 %1
                    %6:bool[2] = compare [direction=GreaterThanOrEqual] %4 %2
                    %7:bool[2] = compare [direction=LessThan] %4 %3
                    %8:bool[2] = and %6 %7
                    %9:bool[2] = or %5 %8
                    %10:bool[] = reduce [kind=any, axes=[0]] %9
                in (%10)"},
        );
        assert_eq!(
            masked_body.to_string(),
            indoc! {"
                lambda %0:f64[2] .
                let %1:f64[2] = const [1.0, 3.0]
                    %2:f64[2] = const [2.0, 10.0]
                    %3:f64[2] = const [3.0, 3.0]
                    %4:f64[2] = reverse [axes=[0]] %0
                    %5:bool[2] = compare [direction=LessThan] %0 %1
                    %6:bool[2] = compare [direction=GreaterThanOrEqual] %4 %2
                    %7:bool[2] = compare [direction=LessThan] %4 %3
                    %8:bool[2] = and %6 %7
                    %9:bool[2] = or %5 %8
                    %10:f64[2] = const [1.5, 1.5]
                    %11:f64[2] = const [1.0, 1.0]
                    %12:f64[2] = mul %0 %10
                    %13:f64[2] = add %12 %11
                    %14:f64[2] = select %9 %13 %0
                in (%14)"},
        );
    }
}
