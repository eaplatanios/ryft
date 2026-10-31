use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;

use crate::arrays::{ArrayType, DataType, ShardingDimension};
use crate::axes::NamedAxis;
use crate::batching::{
    BatchAxis, BatchableType, BatchedProgram, BatchingError, BatchingLevel, BatchingPolicy, DiagonalBatchingPolicy,
    ProgramBatchingOutputAxesPolicy, RecursiveBatchingPolicy,
};
use crate::contexts::StagingContext;
use crate::differentiation::{
    CotangentAccumulator, CotangentBatchingPolicy, CotangentDestinationKind, DifferentiableOperation,
    DifferentiableType, DifferentiationError, ResidualZeroProvider, TransposableOperation, TransposedProgramCotangent,
    TranspositionContext,
};
use crate::operations::arithmetic::{Add, AddOperation};
use crate::operations::composites::operations::COMPOSITE_OPERATION_NAME;
use crate::operations::composites::rules::specializations::{CustomRuleDerivation, transpose_backward_source_program};
use crate::operations::composites::rules::{
    CustomRuleBatchedOutputs, CustomRuleDerivedDefinitionKey, CustomRuleSpecializationRequest, CustomRuleTracer,
};
use crate::operations::control_flow::scan::{ScanOperation, ScanType};
use crate::operations::references::{ReferenceAddUpdateOperation, ReferenceNewOperation};
use crate::parameters::Parameter;
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationProvider, Program, ProgramError, ReferenceAccessOperation, ReferenceDischargePolicy,
    ReferenceDischargeableOperation, ReferenceDischargeableType, ReferenceMemberType, ReferenceTransform,
    ReferenceType, Type, Typed, Value,
};
use crate::tracing::TracingContext;

#[cfg(doc)]
use crate::operations::composites::rules::registrations::{CustomRuleReference, CustomRuleRegistration};

#[cfg(doc)]
use crate::operations::composites::rules::specializations::CustomRuleSpecializer;

#[cfg(doc)]
use crate::operations::composites::operations::CompositeTransposeOperation;

/// Next process-unique identity assigned to a [`CustomRuleDefinition`].
static NEXT_CUSTOM_RULE_DEFINITION_ID: AtomicU64 = AtomicU64::new(0);

// The retained rules are stored behind the following object-safe traits rather than behind `dyn Fn` types because their
// signatures mention tracing contexts, which require `O: Operation`. Keeping that requirement on the functions leaves
// `CustomRuleDefinition` and the operations free of an `O: Operation` bound, which would otherwise form a cycle with
// the payload bounds of any operation family that contains them.

/// Retained custom Jacobian-Vector Product (JVP) rule `(p, x, ẋ) ↦ (y, ẏ)`. It receives every primal input (i.e., the
/// non-differentiated inputs `p` followed by the differentiated inputs `x`) and the tangents of the differentiated
/// inputs only, and returns the primal outputs together with their tangents.
pub trait CustomJvpRule<V, O>: Send + Sync {
    /// Applies this rule to the provided primal inputs and differentiated input tangents.
    fn apply(
        &self,
        primals: &[CustomRuleTracer<V, O>],
        tangents: &[CustomRuleTracer<V, O>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomJvpRule<V, O> for F
where
    F: Fn(
            &[CustomRuleTracer<V, O>],
            &[CustomRuleTracer<V, O>],
        ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
        + Send
        + Sync,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn apply(
        &self,
        primals: &[CustomRuleTracer<V, O>],
        tangents: &[CustomRuleTracer<V, O>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError> {
        self(primals, tangents)
    }
}

/// Retained custom Jacobian-Vector Product (JVP) rule `(p, x, ẋ) ↦ (y, ẏ)` that receives the structural zeros of its
/// input tangents as [`MaybeZero::Zero`] leaves instead of materialized zeros, so that it can skip the work that they
/// would otherwise require, and that returns its output tangents as [`MaybeZero`]s, so that it can declare an output
/// tangent to be a structural zero without computing it. It is otherwise identical to [`CustomJvpRule`].
pub trait CustomSymbolicZeroJvpRule<V, O>: Send + Sync {
    /// Applies this rule to the provided primal inputs and differentiated input tangents.
    fn apply(
        &self,
        primals: &[CustomRuleTracer<V, O>],
        tangents: &[MaybeZero<CustomRuleTracer<V, O>>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<MaybeZero<CustomRuleTracer<V, O>>>), ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomSymbolicZeroJvpRule<V, O> for F
where
    F: Fn(
            &[CustomRuleTracer<V, O>],
            &[MaybeZero<CustomRuleTracer<V, O>>],
        ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<MaybeZero<CustomRuleTracer<V, O>>>), ProgramError>
        + Send
        + Sync,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn apply(
        &self,
        primals: &[CustomRuleTracer<V, O>],
        tangents: &[MaybeZero<CustomRuleTracer<V, O>>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<MaybeZero<CustomRuleTracer<V, O>>>), ProgramError> {
        self(primals, tangents)
    }
}

/// Retained custom Vector-Jacobian Product (VJP) forward rule `(p, x) ↦ (y, r)`, which returns the primal outputs
/// together with the residuals that the backward rule consumes.
pub trait CustomVjpForwardRule<V, O>: Send + Sync {
    /// Applies this rule to the provided primal inputs.
    fn apply(
        &self,
        primals: &[CustomRuleTracer<V, O>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomVjpForwardRule<V, O> for F
where
    F: Fn(
            &[CustomRuleTracer<V, O>],
        ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
        + Send
        + Sync,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn apply(
        &self,
        primals: &[CustomRuleTracer<V, O>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError> {
        self(primals)
    }
}

/// Retained custom Vector-Jacobian Product (VJP) backward rule `(p, r, ȳ) ↦ x̄`. It receives the known leading inputs
/// (i.e., the non-differentiated inputs `p` followed by the residuals `r`) and one materialized cotangent seed per
/// output, and it returns one cotangent per differentiated input. This is the primary reverse mode form: it is a pure
/// function of its inputs, like any other traced program, and transposition adds its results to whatever destination
/// each differentiated input's cotangent has. The explicit tracing context lets structured adapters replay a rule
/// even when its boundary has no leading inputs or seeds.
pub trait CustomVjpBackwardRule<V, O>: Send + Sync {
    /// Applies this rule to the provided leading inputs and output cotangent seeds.
    fn apply(
        &self,
        context: &TracingContext<V, O>,
        leading_inputs: &[CustomRuleTracer<V, O>],
        seeds: &[CustomRuleTracer<V, O>],
    ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomVjpBackwardRule<V, O> for F
where
    F: Fn(
            &TracingContext<V, O>,
            &[CustomRuleTracer<V, O>],
            &[CustomRuleTracer<V, O>],
        ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError>
        + Send
        + Sync,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn apply(
        &self,
        context: &TracingContext<V, O>,
        leading_inputs: &[CustomRuleTracer<V, O>],
        seeds: &[CustomRuleTracer<V, O>],
    ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError> {
        self(context, leading_inputs, seeds)
    }
}

/// Retained custom Vector-Jacobian Product (VJP) backward rule `(p, r, ȳ) ↦ x̄` that receives its structural-zero
/// output cotangent seeds as [`MaybeZero::Zero`] leaves instead of materialized zeros, so that it can skip the work
/// that they would otherwise require, and that returns its input cotangents as [`MaybeZero`]s, so that it can declare
/// an input cotangent to be a structural zero without computing it. It is otherwise identical to
/// [`CustomVjpBackwardRule`].
pub trait CustomSymbolicZeroVjpBackwardRule<V, O>: Send + Sync {
    /// Applies this rule to the provided leading inputs and output cotangent seeds.
    fn apply(
        &self,
        context: &TracingContext<V, O>,
        leading_inputs: &[CustomRuleTracer<V, O>],
        seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
    ) -> Result<Vec<MaybeZero<CustomRuleTracer<V, O>>>, ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomSymbolicZeroVjpBackwardRule<V, O> for F
where
    F: Fn(
            &TracingContext<V, O>,
            &[CustomRuleTracer<V, O>],
            &[MaybeZero<CustomRuleTracer<V, O>>],
        ) -> Result<Vec<MaybeZero<CustomRuleTracer<V, O>>>, ProgramError>
        + Send
        + Sync,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn apply(
        &self,
        context: &TracingContext<V, O>,
        leading_inputs: &[CustomRuleTracer<V, O>],
        seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
    ) -> Result<Vec<MaybeZero<CustomRuleTracer<V, O>>>, ProgramError> {
        self(context, leading_inputs, seeds)
    }
}

/// Retained accumulating custom Vector-Jacobian Product (VJP) backward rule. It is invoked while transposing a
/// [`CompositeTransposeOperation`] with the known leading inputs (i.e., `p` followed by the residuals `r`), the
/// output cotangent seeds, and one [`CotangentAccumulator`] per carrier input, and it submits the contributions
/// of the differentiated inputs to their accumulators. A single rule therefore serves returned, caller-buffer,
/// and ignored destinations, and can, for example, update only the affected entries of a caller-provided buffer
/// instead of returning a full-sized cotangent.
pub trait CustomVjpAccumulatingBackwardRule<V, O>: Send + Sync {
    /// Applies this rule in the provided transposition context.
    fn apply(
        &self,
        context: &mut TranspositionContext<V, O>,
        inputs: &[PartialValue<CustomRuleTracer<V, O>>],
        seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomVjpAccumulatingBackwardRule<V, O> for F
where
    F: Fn(
            &mut TranspositionContext<V, O>,
            &[PartialValue<CustomRuleTracer<V, O>>],
            &[MaybeZero<CustomRuleTracer<V, O>>],
            &[CotangentAccumulator],
        ) -> Result<(), ProgramError>
        + Send
        + Sync,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn apply(
        &self,
        context: &mut TranspositionContext<V, O>,
        inputs: &[PartialValue<CustomRuleTracer<V, O>>],
        seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), ProgramError> {
        self(context, inputs, seeds, accumulators)
    }
}

/// Retained custom batching rule, which batches a call of a [`CustomRuleDefinition`] instead of structurally batching
/// its primal region. It receives the call's batched inputs, their batch axes, the batching level, and the level's
/// boundary inputs (e.g., a first-class batch extent), and it returns the batched outputs together with their batch
/// axes.
pub trait CustomBatchingRule<V: Typed, O>: Send + Sync {
    /// Applies this rule to the provided batched inputs.
    fn apply(
        &self,
        inputs: &[CustomRuleTracer<V, O>],
        input_axes: &[BatchAxis],
        level: &BatchingLevel<V::Type>,
        boundary_inputs: &[CustomRuleTracer<V, O>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomBatchingRule<V, O> for F
where
    F: Fn(
            &[CustomRuleTracer<V, O>],
            &[BatchAxis],
            &BatchingLevel<V::Type>,
            &[CustomRuleTracer<V, O>],
        ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError>
        + Send
        + Sync,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn apply(
        &self,
        inputs: &[CustomRuleTracer<V, O>],
        input_axes: &[BatchAxis],
        level: &BatchingLevel<V::Type>,
        boundary_inputs: &[CustomRuleTracer<V, O>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError> {
        self(inputs, input_axes, level, boundary_inputs)
    }
}

/// Operations with which the derived batching rules of a broadcast sensitive custom batching rule recover exact
/// per-item tangents (refer to [`CustomRuleDefinition::with_broadcast_sensitive_batching_rule`]).
pub(super) trait CustomRuleBroadcastSensitivity<V, O>: Send + Sync {
    /// Returns the elements of `value` whose indices along two of its batch axes of equal extent agree (refer to
    /// [`DiagonalBatchingPolicy::diagonal`]).
    fn diagonal(
        &self,
        value: &CustomRuleTracer<V, O>,
        kept_axis: usize,
        removed_axis: usize,
    ) -> Result<CustomRuleTracer<V, O>, ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;

    /// Returns the sum of two values of the same type, after aligning their manual variation (refer to
    /// [`ManualVariationAlignment`](crate::ManualVariationAlignment)).
    fn add(
        &self,
        left: &CustomRuleTracer<V, O>,
        right: &CustomRuleTracer<V, O>,
    ) -> Result<CustomRuleTracer<V, O>, ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>;
}

/// [`CustomRuleBroadcastSensitivity`] that takes diagonals with the canonical batching policy of the family's type
/// universe and adds values with the [`Add`] capability. The two contributions that it adds are tangents produced by
/// different rule programs, so the capability aligns their manual variation before binding the addition.
struct PolicyBroadcastSensitivity<V, O>(PhantomData<fn() -> (V, O)>);

impl<V, O> CustomRuleBroadcastSensitivity<V, O> for PolicyBroadcastSensitivity<V, O>
where
    V: Value<Type: BatchableType<Policy: DiagonalBatchingPolicy<TracingContext<V, O>>>>,
    O: Operation<Type = V::Type>,
    CustomRuleTracer<V, O>: Add<V::Type>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    #[inline]
    fn diagonal(
        &self,
        value: &CustomRuleTracer<V, O>,
        kept_axis: usize,
        removed_axis: usize,
    ) -> Result<CustomRuleTracer<V, O>, ProgramError> {
        Ok(<<V::Type as BatchableType>::Policy as DiagonalBatchingPolicy<TracingContext<V, O>>>::diagonal(
            value.context(),
            value,
            kept_axis,
            removed_axis,
        )?)
    }

    #[inline]
    fn add(
        &self,
        left: &CustomRuleTracer<V, O>,
        right: &CustomRuleTracer<V, O>,
    ) -> Result<CustomRuleTracer<V, O>, ProgramError> {
        <CustomRuleTracer<V, O> as Add<V::Type>>::add(left, right)
    }
}

/// Forward mode differentiation rule of a [`CustomRuleDefinition`].
pub enum CustomRuleJvp<V, O> {
    /// Retained custom Jacobian-Vector Product (JVP) rule that receives materialized input tangents.
    Rule(Arc<dyn CustomJvpRule<V, O>>),

    /// Retained custom Jacobian-Vector Product (JVP) rule that receives structural-zero input tangents.
    SymbolicZeroRule(Arc<dyn CustomSymbolicZeroJvpRule<V, O>>),

    /// Jacobian-Vector Product (JVP) derived from the primal region of each call, which is an explicit registration
    /// choice (i.e., never a fallback for a missing rule).
    Primal,
}

/// Borrowed retained Jacobian-Vector Product (JVP) rule of a [`CustomRuleDefinition`],
/// as applied by its forward mode differentiation specializations.
pub(super) enum CustomRuleJvpRule<'r, V, O> {
    /// Rule that receives materialized input tangents.
    Materialized(&'r Arc<dyn CustomJvpRule<V, O>>),

    /// Rule that receives structural-zero input tangents.
    SymbolicZeros(&'r Arc<dyn CustomSymbolicZeroJvpRule<V, O>>),
}

/// Reverse mode differentiation rules of a [`CustomRuleDefinition`].
pub struct CustomRuleVjp<V: Typed + Parameter, O> {
    /// Retained forward rule `(p, x) ↦ (y, r)`.
    forward: Arc<dyn CustomVjpForwardRule<V, O>>,

    /// Retained backward rule.
    backward: CustomRuleVjpBackward<V, O>,

    /// Transposition of a backward-rule specialization source program with respect to the provided inputs and
    /// destination kinds, returning the transposed program and the representation of each selected input's cotangent
    /// (refer to [`TransposedProgram`](crate::TransposedProgram)), so that the cotangents that a backward rule
    /// declares to be structural zeros stay structural zeros instead of being materialized. The reverse mode rule
    /// builders capture it where the operation family's transposition bounds are known, because requiring them on the
    /// carrier's own transposition rule would form a cycle with the family's payload bounds.
    #[allow(clippy::type_complexity)]
    pub(super) transposer: Arc<
        dyn Fn(
                &Program<V, O, Vec<V>, Vec<V>>,
                &[usize],
                &[CotangentDestinationKind],
            )
                -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<TransposedProgramCotangent>), DifferentiationError>
            + Send
            + Sync,
    >,
}

impl<V: Typed + Parameter, O> CustomRuleVjp<V, O> {
    /// Returns the retained forward rule, which computes the primal outputs and residuals from the primal inputs.
    #[inline]
    pub fn forward(&self) -> &(dyn 'static + CustomVjpForwardRule<V, O>) {
        self.forward.as_ref()
    }

    /// Returns the retained backward rule, including whether it receives materialized or structural zero seeds or
    /// accumulates its contributions directly into cotangent destinations.
    #[inline]
    pub fn backward(&self) -> &CustomRuleVjpBackward<V, O> {
        &self.backward
    }
}

/// Retained custom Vector-Jacobian Product (VJP) backward rule of a [`CustomRuleDefinition`].
pub enum CustomRuleVjpBackward<V, O> {
    /// Pure backward differentiation rule that receives materialized seeds.
    Pure(Arc<dyn CustomVjpBackwardRule<V, O>>),

    /// Pure backward differentiation rule that receives structural-zero seeds.
    PureWithSymbolicZeros(Arc<dyn CustomSymbolicZeroVjpBackwardRule<V, O>>),

    /// Accumulating backward differentiation rule.
    Accumulating(Arc<dyn CustomVjpAccumulatingBackwardRule<V, O>>),
}

/// Batching rule of a [`CustomRuleBatching`].
pub(super) enum CustomRuleBatchingRule<V: Typed + Parameter, O> {
    /// Retained custom batching rule.
    Custom(Arc<dyn CustomBatchingRule<V, O>>),

    /// Derivation of the definition from a source definition, whose batching rule the definition's calls apply in
    /// derived form.
    Derived(CustomRuleDerivation<V, O>),
}

/// Custom batching rule of a [`CustomRuleDefinition`], together with what the batching rules of the definitions
/// derived from it need.
pub struct CustomRuleBatching<V: Typed + Parameter, O> {
    /// Rule that batches calls instead of structurally batching their primal regions.
    pub(super) rule: CustomRuleBatchingRule<V, O>,

    /// Forward mode differentiation of a traced rule program with respect to the provided inputs, returning its fused
    /// Jacobian-Vector Product (JVP) program (refer to [`Program::jvp_with_respect_to`]). The batching rules of derived
    /// definitions (refer to [`CustomRuleDerivation`]) differentiate the batched programs of their source's batching
    /// rule with it. [`CustomRuleDefinition::with_batching_rule`] captures it where the operation family's
    /// differentiation bounds are known, because requiring them on the operations' own rules would form a cycle with
    /// the family's payload bounds.
    #[allow(clippy::type_complexity)]
    pub(super) differentiator: Arc<
        dyn Fn(
                // Program to differentiate.
                &Program<V, O, Vec<V>, Vec<V>>,
                // Indices of the inputs with respect to which the program is differentiated.
                &[usize],
            ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
            + Send
            + Sync,
    >,

    /// Operations with which the batching rules of the definitions derived from this definition recover exact per-item
    /// tangents, present when [`Self::rule`] depends on its input batch axes (refer to
    /// [`CustomRuleDefinition::with_broadcast_sensitive_batching_rule`]).
    pub(super) broadcast_sensitivity: Option<Arc<dyn CustomRuleBroadcastSensitivity<V, O>>>,
}

impl<V: Typed + Parameter, O> CustomRuleBatching<V, O> {
    /// Returns the retained custom batching callback when this is a source definition's rule. A derived definition
    /// instead applies a derivative of its source's rule and returns [`None`] here (refer to [`Self::is_derived`]).
    #[inline]
    pub fn custom_rule(&self) -> Option<&(dyn 'static + CustomBatchingRule<V, O>)> {
        match &self.rule {
            CustomRuleBatchingRule::Custom(rule) => Some(rule.as_ref()),
            CustomRuleBatchingRule::Derived(_) => None,
        }
    }

    /// Returns whether this batching rule is derived from a source definition's rule. Such a rule differentiates the
    /// source rule instead of retaining a separate custom callback.
    #[inline]
    pub fn is_derived(&self) -> bool {
        matches!(self.rule, CustomRuleBatchingRule::Derived(_))
    }
}

/// Operation family `Self` over values `V` whose programs the reverse mode rules of a [`CustomRuleDefinition`] (refer
/// to [`CustomRuleDefinition::with_vjp`]) can serve: the family transposes the programs that apply the rules, and it
/// provides the zeros, reference allocations, reference updates, and additions with which transposition delivers
/// cotangents to their destinations (including caller-provided buffers). It names those requirements once for every
/// reverse mode rule builder, and it is implemented for every family that satisfies them.
pub trait CustomVjpOperation<V: Value<Type: ReferenceMemberType>>:
    Operation<Type = V::Type>
    + TransposableOperation<V, Self>
    + ResidualZeroProvider<V::Type, Operation = Self>
    + ReferenceAccessOperation<Transform: ReferenceTransform<Referent = <V::Type as ReferenceMemberType>::Referent>>
    + OperationProvider<
        V::Type,
        ReferenceNewOperation<<V::Type as ReferenceMemberType>::Referent, V::Type>,
        Operation = Self,
    > + OperationProvider<
        V::Type,
        ReferenceAddUpdateOperation<
            <V::Type as ReferenceMemberType>::Referent,
            V::Type,
            <Self as ReferenceAccessOperation>::Transform,
        >,
        Operation = Self,
    > + OperationProvider<V::Type, AddOperation<V::Type>, Operation = Self>
where
    TracingContext<V, Self>: StagingContext<Type = V::Type, Constant = V, Operation = Self>,
{
}

impl<V: Value<Type: ReferenceMemberType>, O> CustomVjpOperation<V> for O
where
    O: Operation<Type = V::Type>
        + TransposableOperation<V, O>
        + ResidualZeroProvider<V::Type, Operation = O>
        + ReferenceAccessOperation<Transform: ReferenceTransform<Referent = <V::Type as ReferenceMemberType>::Referent>>
        + OperationProvider<
            V::Type,
            ReferenceNewOperation<<V::Type as ReferenceMemberType>::Referent, V::Type>,
            Operation = O,
        > + OperationProvider<
            V::Type,
            ReferenceAddUpdateOperation<
                <V::Type as ReferenceMemberType>::Referent,
                V::Type,
                <O as ReferenceAccessOperation>::Transform,
            >,
            Operation = O,
        > + OperationProvider<V::Type, AddOperation<V::Type>, Operation = O>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
}

/// Definition of a custom rule set whose derivative rules are retained callbacks that are traced lazily, on the first
/// request of each specialization. A definition is registered once (refer to [`CustomRuleRegistration`]) and shared by
/// every call that uses it, and two definitions are the same rule set exactly when they are the same definition (i.e.,
/// identity is a process-unique identifier assigned at construction rather than structural). [`Self::name`] is a
/// human-readable label for rendering and diagnostics only; distinct definitions may share it.
///
/// Specialization is deterministic and repeatable but not exactly-once: eviction, retries after a failed
/// specialization, concurrent cold requests, and requests made after the registration handle was dropped may invoke
/// a rule again, so rules must not depend on mutable state outside their key. Host-side side effects of a rule (e.g.,
/// counting its invocations) are not program effects.
///
/// # Ownership
///
/// The specialized programs are not stored in the definition, because a program that calls the definition (e.g., a
/// rule that calls its own definition, directly or through other definitions) would then keep the definition alive
/// forever. They are stored in caches owned by the [`CustomRuleRegistration`] handle, while calls and rules refer to
/// the definition through a [`CustomRuleReference`], which holds the definition strongly and the caches only weakly.
/// The strong references therefore always point from programs to definitions and from the handle to the caches, and
/// never back from a definition to a program. Refer to [`CustomRuleRegistration`] for how rules must refer to
/// definitions.
pub struct CustomRuleDefinition<V: Typed + Parameter, O> {
    /// Process-unique identity of this definition.
    id: u64,

    /// Human-readable label used for rendering and diagnostics.
    name: Cow<'static, str>,

    /// Named axes with which every rule is traced.
    named_axes: Vec<(String, NamedAxis)>,

    /// Optional forward mode differentiation rule.
    jvp: Option<CustomRuleJvp<V, O>>,

    /// Optional reverse mode differentiation rules.
    vjp: Option<CustomRuleVjp<V, O>>,

    /// Whether reverse mode differentiation defers the forward rule of [`Self::vjp`] to the backward pass,
    /// which has no effect without reverse mode rules.
    deferred_forward: bool,

    /// Optional structural batching of a traced rule program at one recorded level with the provided input batch
    /// axes. Each output is aligned to its required batch axis, or keeps its natural axis when the requirement is
    /// [`None`]. The result is the batched program, which consumes the level's boundary inputs before the source
    /// program's inputs, together with its output batch axes. Without it, batched calls can be executed but not
    /// differentiated. [`Self::with_batching_support`] captures it where the operation family's batching bounds are
    /// known, because requiring them on the operations' own rules would form a cycle with the family's payload bounds.
    #[allow(clippy::type_complexity)]
    structural_batcher: Option<
        Arc<
            dyn Fn(
                    &Program<V, O, Vec<V>, Vec<V>>,
                    &[BatchAxis],
                    &[Option<BatchAxis>],
                    &BatchingLevel<<V as Typed>::Type>,
                    CustomRuleBatchedOutputs,
                ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError>
                + Send
                + Sync,
        >,
    >,

    /// Optional construction of the scan with which calls are batched sequentially when there is no custom batching
    /// rule. It receives the batch extent (i.e., the scan's trip count), the number of carries (i.e., replicated
    /// inputs), the batch axis placement, and the number of stacked outputs, and returns the scan operation and the
    /// type of the slice index that the scan body receives first. [`Self::with_sequential_batching`] captures it where
    /// the family's scan bounds are known, because requiring them on the operations' own rules would restrict every
    /// family that contains the operations to scannable types.
    #[allow(clippy::type_complexity)]
    sequential_batching_scan: Option<
        Arc<
            dyn Fn(usize, usize, &ShardingDimension, usize) -> Result<(O, <V as Typed>::Type), ProgramError>
                + Send
                + Sync,
        >,
    >,

    /// Optional batch axes of the outputs of batched calls, which default to mapping every output at axis 0.
    batched_output_axes: Option<Vec<BatchAxis>>,

    /// Optional custom batching rule, which is a derivation for the definitions derived from a definition that has one.
    batching_rule: Option<CustomRuleBatching<V, O>>,

    /// Optional reference discharge of a traced rule program whose call was discharged, replacing its local reference
    /// state with explicitly threaded values. Without it, discharged calls can be differentiated only when their rule
    /// programs contain no reference state. [`Self::with_reference_discharge`] captures it where the operation family's
    /// discharge bounds are known, because requiring them on the operations' own rules would form a cycle with the
    /// family's payload bounds and would exclude families without references.
    #[allow(clippy::type_complexity)]
    discharger: Option<
        Arc<dyn Fn(Program<V, O, Vec<V>, Vec<V>>) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> + Send + Sync>,
    >,

    /// Specializations that each thread is producing without a cache (i.e., after the registration handle was
    /// dropped), which reject recursive production exactly as a [`SpecializationCache`] does. The markers contain
    /// static keys only, so they never retain a program.
    uncached_specializations: Mutex<HashSet<(ThreadId, CustomRuleSpecializationRequest<V::Type>)>>,

    /// Definitions derived from this definition (refer to [`CustomRuleSpecializer::derived`]), held weakly so that a
    /// derived definition that some call still holds keeps its identity whether or not its registration is alive,
    /// while a derived definition, which holds this definition strongly, never forms a cycle with it.
    derived_definitions: Mutex<HashMap<CustomRuleDerivedDefinitionKey, Weak<CustomRuleDefinition<V, O>>>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleDefinition<V, O> {
    /// Creates a new [`CustomRuleDefinition`] with the provided label and no derivative rules.
    #[inline]
    pub fn new<N: Into<Cow<'static, str>>>(name: N) -> Self {
        Self {
            id: NEXT_CUSTOM_RULE_DEFINITION_ID.fetch_add(1, Ordering::Relaxed),
            name: name.into(),
            named_axes: Vec::new(),
            jvp: None,
            vjp: None,
            deferred_forward: false,
            structural_batcher: None,
            sequential_batching_scan: None,
            batched_output_axes: None,
            batching_rule: None,
            discharger: None,
            uncached_specializations: Mutex::new(HashSet::new()),
            derived_definitions: Mutex::new(HashMap::new()),
        }
    }

    /// Creates a definition derived from this definition, inheriting its named axes and transform support. The derived
    /// definition uses its primal region for forward mode and the provided derived batching rule, with no reverse
    /// mode rules or explicit output batch axes of its own.
    pub(super) fn new_derived(&self, name: String, batching_rule: Option<CustomRuleBatching<V, O>>) -> Self {
        Self {
            named_axes: self.named_axes.clone(),
            jvp: Some(CustomRuleJvp::Primal),
            structural_batcher: self.structural_batcher.clone(),
            sequential_batching_scan: self.sequential_batching_scan.clone(),
            batching_rule,
            discharger: self.discharger.clone(),
            ..Self::new(name)
        }
    }

    /// Returns this [`CustomRuleDefinition`] with the provided named axes, with which every rule is traced, so that
    /// the rules resolve the names bound where the calls that apply them are made (e.g., an `axis_index` over the axis
    /// of an enclosing batching level). The rules are traced lazily and in fresh traces, which see no enclosing
    /// bindings otherwise. The definitions that are derived from this definition trace their rules with the same axes.
    #[inline]
    pub fn with_named_axes(mut self, named_axes: Vec<(String, NamedAxis)>) -> Self {
        self.named_axes = named_axes;
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided forward mode differentiation rule, replacing any
    /// previously configured forward mode rule (including a JVP derived from the primal). The rule implements
    /// `(p, x, ẋ) ↦ (y, ẏ)`: it receives every primal input (i.e., the non-differentiated inputs `p` followed by the
    /// differentiated inputs `x`) and one tangent per differentiated input, in which structural zeros are materialized,
    /// and it returns the primal outputs followed by one tangent per output.
    #[inline]
    pub fn with_jvp<R>(mut self, rule: R) -> Self
    where
        V: Value,
        O: Operation<Type = V::Type>,
        R: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
                &[CustomRuleTracer<V, O>],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
            + Send
            + Sync,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        self.jvp = Some(CustomRuleJvp::Rule(Arc::new(rule)));
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided forward mode rule, replacing any previously configured
    /// forward mode rule. The rule is used as the rule of [`Self::with_jvp`], except that it receives each structurally
    /// zero input tangent as a [`MaybeZero::Zero`] rather than as a materialized zero, which lets it skip the work that
    /// the zero would otherwise require, and that it returns each output tangent as a [`MaybeZero`], so that it can
    /// declare an output tangent to be a structural zero (e.g., that of an output that does not depend on the
    /// differentiated inputs). Each pattern of structurally zero input tangents is a separate specialization of the
    /// rule. A structural-zero output tangent is materialized at the boundary of the specialization program, from the
    /// primal outputs and inputs, as a zero-producing operation, which forward mode differentiation recognizes as a
    /// structural zero again when it replays the program.
    #[inline]
    pub fn with_symbolic_zero_jvp<R>(mut self, rule: R) -> Self
    where
        V: Value,
        O: Operation<Type = V::Type>,
        R: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
                &[MaybeZero<CustomRuleTracer<V, O>>],
            )
                -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<MaybeZero<CustomRuleTracer<V, O>>>), ProgramError>
            + Send
            + Sync,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        self.jvp = Some(CustomRuleJvp::SymbolicZeroRule(Arc::new(rule)));
        self
    }

    /// Returns this [`CustomRuleDefinition`] with a forward mode rule derived from the primal region of each call,
    /// replacing any previously configured forward mode rule. This is how a rule set that only customizes reverse
    /// mode (or batching) keeps ordinary forward mode differentiation, which a missing rule never provides implicitly:
    /// forward mode then differentiates the primal region, reusable linearization partitions that derivative once, and
    /// reverse mode transposes it unless reverse mode rules take precedence.
    #[inline]
    pub fn with_jvp_from_primal(mut self) -> Self {
        self.jvp = Some(CustomRuleJvp::Primal);
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided reverse mode forward rule `(p, x) ↦ (y, r)` and pure
    /// backward rule `(p, r, ȳ) ↦ x̄`. The forward rule receives every primal input and returns the primal outputs
    /// followed by the residuals `r`. The backward rule receives the known leading inputs (i.e., the non-differentiated
    /// inputs `p` followed by the residuals) and one cotangent seed per output, in which structural zeros are
    /// materialized, and it returns one cotangent per differentiated input, which transposition adds to whatever
    /// destination that input's cotangent has.
    #[inline]
    pub fn with_vjp<F, B>(self, forward: F, backward: B) -> Self
    where
        V: 'static + Value<Type: DifferentiableType + ReferenceMemberType>,
        O: 'static + CustomVjpOperation<V>,
        F: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
            + Send
            + Sync,
        B: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
                &[CustomRuleTracer<V, O>],
            ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError>
            + Send
            + Sync,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        self.with_vjp_in_context(forward, move |_, leading_inputs, seeds| backward(leading_inputs, seeds))
    }

    /// Installs reverse mode differentiation rules whose backward adapter receives materialized seeds and the trace in
    /// which it must replay. Structured adapters need that trace when neither leading inputs nor seeds can supply it.
    #[inline]
    pub(in crate::operations::composites) fn with_vjp_in_context<F, B>(mut self, forward: F, backward: B) -> Self
    where
        V: 'static + Value<Type: DifferentiableType + ReferenceMemberType>,
        O: 'static + CustomVjpOperation<V>,
        F: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
            + Send
            + Sync,
        B: 'static
            + Fn(
                &TracingContext<V, O>,
                &[CustomRuleTracer<V, O>],
                &[CustomRuleTracer<V, O>],
            ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError>
            + Send
            + Sync,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        self.vjp = Some(CustomRuleVjp {
            forward: Arc::new(forward),
            backward: CustomRuleVjpBackward::Pure(Arc::new(backward)),
            transposer: Arc::new(transpose_backward_source_program::<V, O>),
        });
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided reverse mode forward rule `(p, x) ↦ (y, r)` and pure
    /// backward rule `(p, r, ȳ) ↦ x̄`. The rules are used as the rules of [`Self::with_vjp`], except that the backward
    /// rule receives each structural-zero seed as a [`MaybeZero::Zero`] rather than as a materialized zero, and that
    /// it returns each input cotangent as a [`MaybeZero`], so that it can declare an input cotangent to be a structural
    /// zero (e.g., that of an input that it treats as a parameter). Transposition contributes nothing to the
    /// destination of a structural-zero cotangent, so the cotangent stays a structural zero through the transposed
    /// program (refer to [`TransposedProgramCotangent::Zero`]) unless its consumer materializes it.
    #[inline]
    pub fn with_symbolic_zero_vjp<F, B>(self, forward: F, backward: B) -> Self
    where
        V: 'static + Value<Type: DifferentiableType + ReferenceMemberType>,
        O: 'static + CustomVjpOperation<V>,
        F: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
            + Send
            + Sync,
        B: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
                &[MaybeZero<CustomRuleTracer<V, O>>],
            ) -> Result<Vec<MaybeZero<CustomRuleTracer<V, O>>>, ProgramError>
            + Send
            + Sync,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        self.with_symbolic_zero_vjp_in_context(forward, move |_, leading_inputs, seeds| backward(leading_inputs, seeds))
    }

    /// Installs reverse mode differentiation rules whose backward adapter receives symbolic seeds and the trace
    /// in which it must replay, including when every seed is structural zero and there are no leading inputs.
    #[inline]
    pub(in crate::operations::composites) fn with_symbolic_zero_vjp_in_context<F, B>(
        mut self,
        forward: F,
        backward: B,
    ) -> Self
    where
        V: 'static + Value<Type: DifferentiableType + ReferenceMemberType>,
        O: 'static + CustomVjpOperation<V>,
        F: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
            + Send
            + Sync,
        B: 'static
            + Fn(
                &TracingContext<V, O>,
                &[CustomRuleTracer<V, O>],
                &[MaybeZero<CustomRuleTracer<V, O>>],
            ) -> Result<Vec<MaybeZero<CustomRuleTracer<V, O>>>, ProgramError>
            + Send
            + Sync,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        self.vjp = Some(CustomRuleVjp {
            forward: Arc::new(forward),
            backward: CustomRuleVjpBackward::PureWithSymbolicZeros(Arc::new(backward)),
            transposer: Arc::new(transpose_backward_source_program::<V, O>),
        });
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided reverse mode forward rule `(p, x) ↦ (y, r)` and
    /// accumulating backward rule. The forward rule is used as the forward rule of [`Self::with_vjp`]. The backward
    /// rule is invoked while its carrier is transposed, with the [`TranspositionContext`], the carrier's inputs (i.e.,
    /// the known leading inputs followed by the differentiated inputs' tangents), the output cotangent seeds, and one
    /// [`CotangentAccumulator`] per carrier input, and it submits the contributions of the differentiated inputs to
    /// their accumulators. A single rule therefore serves returned, caller-buffer, and ignored destinations, and it
    /// can, for example, update only the affected entries of a caller-provided buffer instead of returning a
    /// full-sized cotangent.
    #[inline]
    pub fn with_accumulating_vjp<F, B>(mut self, forward: F, backward: B) -> Self
    where
        V: 'static + Value<Type: DifferentiableType + ReferenceMemberType>,
        O: 'static + CustomVjpOperation<V>,
        F: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
            + Send
            + Sync,
        B: 'static
            + Fn(
                &mut TranspositionContext<V, O>,
                &[PartialValue<CustomRuleTracer<V, O>>],
                &[MaybeZero<CustomRuleTracer<V, O>>],
                &[CotangentAccumulator],
            ) -> Result<(), ProgramError>
            + Send
            + Sync,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        self.vjp = Some(CustomRuleVjp {
            forward: Arc::new(forward),
            backward: CustomRuleVjpBackward::Accumulating(Arc::new(backward)),
            transposer: Arc::new(transpose_backward_source_program::<V, O>),
        });
        self
    }

    /// Returns this [`CustomRuleDefinition`] with its reverse mode differentiation forward rule deferred to the
    /// backward pass (i.e., the analogue of JAX's `custom_vjp(..., optimize_remat=True)`). Reverse mode then computes
    /// the outputs of a call with its primal region, which is cheaper than the forward rule whenever the residuals cost
    /// extra work, and stages a [`CompositeTransposeOperation`] carrier whose leading inputs are the call's own inputs
    /// rather than the residuals. Transposing that carrier replays the forward rule on those inputs to recover the
    /// residuals and then applies the backward rule. The call's inputs are therefore the only values that its
    /// derivative keeps alive between the two passes, which residual policies (e.g., that of a
    /// [`RematerializeOperation`](crate::RematerializeOperation)) save or recompute like any
    /// other value, and the forward rule runs only once, in the backward pass.
    ///
    /// The swap applies whenever reverse mode partitions a call, and it has no effect on a definition without reverse
    /// mode rules (refer to [`Self::with_vjp`]). A call whose unused outputs were pruned (refer to
    /// [`Program::into_pruned`]) is differentiated without the swap, because only the forward rule computes the pruned
    /// outputs, whose cotangent seeds the backward rule receives.
    ///
    /// # Errors
    ///
    /// The forward rule runs in the backward pass, where its effects would execute out of order, so reverse mode
    /// rejects a call whose traced forward rule has effects (e.g., a `print` or a reference update) with a
    /// [`ProgramError::UnsupportedOperation`] that names this function. Batching a carrier with a deferred
    /// forward rule (i.e., batching a derivative after it is taken but before it is transposed) is rejected with
    /// a [`BatchingError::UnsupportedOperation`], while batching a call before it is differentiated is supported.
    #[inline]
    pub fn with_deferred_forward_rule(mut self) -> Self {
        self.deferred_forward = true;
        self
    }

    /// Returns this [`CustomRuleDefinition`] with support for differentiating batched calls. Batching a call records
    /// its batching level without tracing any rule, and a later derivative request traces the rule at the unbatched
    /// types (sharing that specialization with unbatched calls) and structurally batches the traced program once per
    /// recorded level. Without this, batched calls still execute their primal but reject differentiation, and the
    /// derivatives of a custom batching rule cannot be batched (refer to [`Self::with_batching_rule`]).
    pub fn with_batching_support(mut self) -> Self
    where
        V: Value<Type: DifferentiableType + BatchableType>,
        <V::Type as BatchableType>::Policy:
            RecursiveBatchingPolicy<TracingContext<V, O>> + CotangentBatchingPolicy<TracingContext<V, O>>,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        self.structural_batcher = Some(Arc::new(|program, input_axes, output_axes, level, outputs| {
            // Align every required output first (a replicated alignment request keeps the natural axis), then adapt
            // the policy boundary, collapsing any output that is still mapped while its requirement is replicated.
            let batched = <<V::Type as BatchableType>::Policy as RecursiveBatchingPolicy<TracingContext<V, O>>>::batch_program_at_level(
                level,
                program.entry_region_ref(),
                input_axes,
                ProgramBatchingOutputAxesPolicy::AlignEachTo(
                    output_axes.iter().map(|axis| axis.unwrap_or_default()).collect(),
                ),
            )?;
            let required_output_axes = output_axes
                .iter()
                .zip(batched.output_axes())
                .map(|(required, natural)| required.unwrap_or(*natural))
                .collect::<Vec<_>>();
            let adapted =
                <<V::Type as BatchableType>::Policy as BatchingPolicy<TracingContext<V, O>>>::adapt_batched_program(
                    batched,
                    Some(required_output_axes.as_slice()),
                    |context, output, axis| {
                        match outputs {
                            CustomRuleBatchedOutputs::Values => Err(BatchingError::InvalidBatchMetadata {
                                message: format!(
                                    "a custom rule output is mapped along axis {axis} but its batched layout requires \
                                     it to be replicated",
                                ),
                            }),
                            CustomRuleBatchedOutputs::Cotangents => {
                                <<V::Type as BatchableType>::Policy as CotangentBatchingPolicy<TracingContext<V, O>>>::sum_mapped_cotangents(
                                    context, output, axis,
                                )
                            }
                        }
                    },
                )?;
            Ok(adapted.into_parts())
        }));
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided batch axes for the outputs of batched calls, one per
    /// output. They default to mapping every output at axis 0, which is always valid but may broadcast outputs that are
    /// replicated across the batch. Declared axes apply at every batching level: the primal is aligned to them when a
    /// call is batched, and each derivative rule is validated against them when it is first traced at that level.
    #[inline]
    pub fn with_batched_output_axes(mut self, output_axes: Vec<BatchAxis>) -> Self {
        self.batched_output_axes = Some(output_axes);
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided custom batching rule, which batches its calls instead
    /// of structurally batching their primal regions (i.e., the analogue of JAX's `custom_vmap`). The rule receives the
    /// call's batched inputs, their batch axes, the batching level, and the level's boundary inputs (e.g., a first-class
    /// batch extent), and it returns the batched outputs together with their batch axes. It is traced when a call is
    /// batched, once per batching level and input signature, and its program becomes the batched call's primal region,
    /// so it also determines the batch axes of the call's outputs (instead of [`Self::with_batched_output_axes`]). Each
    /// output, unbatched along the batch axis that the rule declares for it, must have the type of the corresponding
    /// output of the call's primal region, which is validated when the call is batched.
    ///
    /// Batching a call with this rule preserves the call and its derivative rules: derivative requests of a
    /// batched call still trace the derivative rules at the unbatched types and batch them structurally (refer to
    /// [`Self::with_batching_support`]), aligning their outputs to the batch axes that the rule declared, while a
    /// forward mode rule derived from the primal differentiates the rule's batched program. The rule is responsible
    /// for computing the batched primal consistently with those derivative rules.
    ///
    /// When the forward mode rule is derived from the primal, differentiating an unbatched call stages a derived call
    /// whose batching applies the derivative of this rule, which may apply the rule with every input replicated. This
    /// function therefore also captures the family's differentiation of traced programs, which is why it requires the
    /// family's differentiation bounds. Batching those derivatives also requires [`Self::with_batching_support`],
    /// which this function cannot configure without the family's batching bounds. Without it, the first request
    /// that batches such a derivative is rejected with a diagnostic that names it.
    #[inline]
    pub fn with_batching_rule<R>(mut self, rule: R) -> Self
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>
            + PartiallyEvaluatableOperation<TracingContext<V, O>>
            + DifferentiableOperation<TracingContext<V, O>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
            + ResidualZeroProvider<V::Type, Operation = O>,
        R: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
                &[BatchAxis],
                &BatchingLevel<V::Type>,
                &[CustomRuleTracer<V, O>],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError>
            + Send
            + Sync,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        self.batching_rule = Some(CustomRuleBatching {
            rule: CustomRuleBatchingRule::Custom(Arc::new(rule)),
            differentiator: Arc::new(|program, input_indices| program.jvp_with_respect_to(input_indices)),
            broadcast_sensitivity: None,
        });
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided custom batching rule, which is used as the rule of
    /// [`Self::with_batching_rule`] except that its per-item numerical behavior may change when a replicated input is
    /// broadcast (e.g., a rule that applies a different approximation when an input is replicated). A batching rule
    /// that only computes the batched primal gives the same values whether a replicated input is broadcast or not,
    /// so the derived batching rules of a call whose forward mode rule is derived from its primal apply it with a
    /// replicated primal broadcast when that primal's tangent is mapped. For a broadcast sensitive rule, they instead
    /// apply it once, at the primal inputs' own batch axes, and recover the per-item tangents of such inputs from the
    /// outer product of their mapped tangents with the rule's batch (i.e., by batching the rule's derivative with
    /// respect to them over a second level of the same extent and taking the diagonal of the two batch axes; refer to
    /// [`DiagonalBatchingPolicy`]). This costs as many applications of that derivative as the batch has items, which
    /// is why it is not the default (refer to [`Self::with_batching_rule`]).
    #[inline]
    pub fn with_broadcast_sensitive_batching_rule<R>(self, rule: R) -> Self
    where
        V: 'static
            + Value<Type: DifferentiableType + BatchableType<Policy: DiagonalBatchingPolicy<TracingContext<V, O>>>>,
        O: 'static
            + Operation<Type = V::Type>
            + PartiallyEvaluatableOperation<TracingContext<V, O>>
            + DifferentiableOperation<TracingContext<V, O>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
            + ResidualZeroProvider<V::Type, Operation = O>,
        R: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
                &[BatchAxis],
                &BatchingLevel<V::Type>,
                &[CustomRuleTracer<V, O>],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError>
            + Send
            + Sync,
        CustomRuleTracer<V, O>: Add<V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        let mut definition = self.with_batching_rule(rule);
        if let Some(batching_rule) = &mut definition.batching_rule {
            batching_rule.broadcast_sensitivity = Some(Arc::new(PolicyBroadcastSensitivity(PhantomData)));
        }
        definition
    }

    /// Returns this [`CustomRuleDefinition`] with sequential batching, which batches each call by looping over its
    /// batch items with a [`ScanOperation`] whose body applies the unbatched call to one item, instead of structurally
    /// batching its primal region (i.e., the Ryft analogue of JAX's
    /// [`sequential_vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.custom_batching.sequential_vmap.html)).
    /// The mapped inputs are moved to axis `0` and become the scan's stacked inputs, the replicated inputs become
    /// invariant carries, and every output is mapped at axis `0`. Batching therefore replaces the call with the scan,
    /// rather than preserving the call around a batched primal region, and the derivatives of the batched call are
    /// those of the scan: its body differentiates the unbatched call with its own rules, one batch item at a time.
    /// A forward mode rule derived from the primal stays sequential when its derivative is batched after it is taken,
    /// because differentiating an unbatched call stages a derived call that is itself batched sequentially. A custom
    /// batching rule (refer to [`Self::with_batching_rule`]) takes precedence over sequential batching. The batch
    /// extent must be statically known, including for a first-class dimension, and the batch axis must be unnamed
    /// because the scan does not bind named collectives across its items for the primal or derivative rules.
    #[inline]
    pub fn with_sequential_batching(mut self) -> Self
    where
        V: Value<Type: ScanType>,
        O: Operation<Type = V::Type> + From<ScanOperation<V::Type>>,
    {
        self.sequential_batching_scan = Some(Arc::new(|length, carry_count, axis_sharding, stacked_output_count| {
            let scan = ScanOperation::<V::Type>::new(carry_count, length)
                .with_stacked_axis_shardings(vec![axis_sharding.clone(); stacked_output_count]);
            Ok((O::from(scan), ArrayType::scalar(DataType::I64).into()))
        }));
        self
    }

    /// Returns this [`CustomRuleDefinition`] with support for differentiating calls whose reference state was
    /// discharged (refer to [`Program::discharge_references`]). Discharging a call discharges the local reference state
    /// of its primal region immediately, and each rule program traced for that call later is discharged before it is
    /// cached or replayed, so that the call's derivatives contain no reference state either. Without this support, a
    /// discharged call whose traced rule programs contain reference state rejects differentiation.
    #[inline]
    pub fn with_reference_discharge<P>(mut self) -> Self
    where
        V: Value<Type: ReferenceDischargeableType<Policy = P> + From<P::Referent>>,
        V::Type: From<ReferenceType<P::Referent>>,
        O: Operation<Type = V::Type> + ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        P: ReferenceDischargePolicy<TracingContext<V, O>>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        self.discharger = Some(Arc::new(|program: Program<V, O, Vec<V>, Vec<V>>| {
            program.discharge_references::<P>(0)?.into_program_without_external_references()
        }));
        self
    }

    /// Returns the process-unique identity of this [`CustomRuleDefinition`]. Two definitions may share a name, but
    /// their identities differ even when their rules are otherwise identical.
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the human-readable label of this [`CustomRuleDefinition`], borrowed without cloning it.
    #[inline]
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// Returns the named axes with which the rules of this [`CustomRuleDefinition`] are traced.
    #[inline]
    pub fn named_axes(&self) -> &[(String, NamedAxis)] {
        &self.named_axes
    }

    /// Returns the configured forward mode rule, or [`None`] when no forward mode rule is configured. An explicit
    /// choice to derive the rule from the primal is represented by [`CustomRuleJvp::Primal`].
    #[inline]
    pub fn jvp(&self) -> Option<&CustomRuleJvp<V, O>> {
        self.jvp.as_ref()
    }

    /// Returns the configured reverse mode forward and backward rules, or [`None`] when they are not configured.
    #[inline]
    pub fn vjp(&self) -> Option<&CustomRuleVjp<V, O>> {
        self.vjp.as_ref()
    }

    /// Returns whether reverse mode is configured to defer the forward rule to the backward pass. This flag has no
    /// effect when no reverse mode rules are configured.
    #[inline]
    pub fn deferred_forward(&self) -> bool {
        self.deferred_forward
    }

    /// Returns the retained callback for structurally batching a rule program, or [`None`] when structural batching
    /// support is not configured. The callback aligns outputs to the requested axes and applies the selected
    /// [`CustomRuleBatchedOutputs`] policy to replicated outputs.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn structural_batcher(
        &self,
    ) -> Option<
        &(
             dyn 'static
                 + Fn(
            &Program<V, O, Vec<V>, Vec<V>>,
            &[BatchAxis],
            &[Option<BatchAxis>],
            &BatchingLevel<V::Type>,
            CustomRuleBatchedOutputs,
        ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError>
                 + Send
                 + Sync
         ),
    > {
        self.structural_batcher.as_deref()
    }

    /// Returns the retained scan constructor for sequential batching, or [`None`] when sequential batching is not
    /// configured. The constructor receives the batch extent, carry count, batch axis sharding, and stacked output
    /// count, and returns the scan operation and its slice index type.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn sequential_batching_scan(
        &self,
    ) -> Option<
        &(
             dyn 'static
                 + Fn(usize, usize, &ShardingDimension, usize) -> Result<(O, V::Type), ProgramError>
                 + Send
                 + Sync
         ),
    > {
        self.sequential_batching_scan.as_deref()
    }

    /// Returns the declared batch axes of batched call outputs, or [`None`] when every output defaults to axis 0.
    #[inline]
    pub fn batched_output_axes(&self) -> Option<&[BatchAxis]> {
        self.batched_output_axes.as_deref()
    }

    /// Returns the configured custom batching rule and its support for derived definitions, or [`None`] when there
    /// is no custom batching rule.
    #[inline]
    pub fn batching_rule(&self) -> Option<&CustomRuleBatching<V, O>> {
        self.batching_rule.as_ref()
    }

    /// Returns the retained callback for discharging a rule program's reference state, or [`None`] when reference
    /// discharge support is not configured. The callback replaces local reference state with explicitly threaded
    /// values and rejects external references.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn discharger(
        &self,
    ) -> Option<
        &(
             dyn 'static
                 + Fn(Program<V, O, Vec<V>, Vec<V>>) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
                 + Send
                 + Sync
         ),
    > {
        self.discharger.as_deref()
    }

    /// Returns `true` if this [`CustomRuleDefinition`] has a forward mode rule or reverse mode rules.
    #[inline]
    pub fn has_derivative_rules(&self) -> bool {
        self.jvp.is_some() || self.vjp.is_some()
    }

    /// Returns whether any thread is producing an uncached specialization of this definition, which tests typically
    /// use to observe that a failed or reentrant request leaves no production behind.
    #[inline]
    pub fn has_uncached_specializations(&self) -> bool {
        !self.uncached_specializations.lock().expect("custom rule in-flight mutex is poisoned").is_empty()
    }

    /// Returns the internal recursion markers for uncached specialization. Callers must release the lock before
    /// producing a specialization and remove their marker on completion, failure, or unwinding.
    pub(super) fn uncached_specializations(
        &self,
    ) -> &Mutex<HashSet<(ThreadId, CustomRuleSpecializationRequest<V::Type>)>> {
        &self.uncached_specializations
    }

    /// Returns the weak registry of derived definitions. Callers reuse live definitions by key and must not retain a
    /// derived definition strongly in this registry, because each derived definition retains its source.
    pub(super) fn derived_definitions(
        &self,
    ) -> &Mutex<HashMap<CustomRuleDerivedDefinitionKey, Weak<CustomRuleDefinition<V, O>>>> {
        &self.derived_definitions
    }

    /// Returns `program` discharged when `discharged` is set. A program without reference state needs no discharge,
    /// which also makes definitions without discharge support usable for discharged calls whose rules keep no reference
    /// state.
    pub(super) fn discharge_if(
        &self,
        discharged: bool,
        program: Program<V, O, Vec<V>, Vec<V>>,
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
        TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
    {
        let has_reference_state =
            program.regions().iter().flat_map(|region| region.atoms()).any(|atom| atom.r#type().is_reference());
        if !discharged || !has_reference_state {
            return Ok(program);
        }
        let Some(discharger) = &self.discharger else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "{} cannot differentiate a call whose reference state was discharged, because its rule programs \
                     contain reference state and its definition has no reference discharge support",
                    {
                        let name = self.name.as_ref();
                        if name == COMPOSITE_OPERATION_NAME {
                            format!("`{COMPOSITE_OPERATION_NAME}`")
                        } else {
                            format!("`{COMPOSITE_OPERATION_NAME}` `{name}`")
                        }
                    },
                ),
            }
            .into());
        };
        Ok(discharger(program)?)
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for CustomRuleDefinition<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CustomRuleDefinition")
            .field("name", &self.name)
            .field("has_jvp", &matches!(self.jvp, Some(CustomRuleJvp::Rule(_) | CustomRuleJvp::SymbolicZeroRule(_))))
            .field("derives_jvp_from_primal", &matches!(self.jvp, Some(CustomRuleJvp::Primal)))
            .field("has_vjp", &self.vjp.is_some())
            .finish_non_exhaustive()
    }
}

// TODO(eaplatanios): Review from here onwards.

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, ShardingDimension,
    };
    use crate::batching::BatchingLevelExtent;
    use crate::differentiation::{CotangentDestinationKind, TransposedProgramCotangent};
    use crate::operations::composites::rules::registrations::CustomRuleRegistration;
    use crate::operations::composites::rules::specializations::{
        CustomRuleSource, CustomRuleSpecializer, LiftedCustomRules, UnavailableCustomRules,
    };
    use crate::operations::composites::rules::{
        CustomRuleBackwardSpecializationKey, CustomRuleBatchingLevel, CustomRuleBatchingSpecializationKey,
        CustomRuleDerivationKind, CustomRuleForwardSpecializationKey, CustomRuleSpecialization,
    };
    use crate::operations::composites::tests::{
        MemberDefinition, RuleCounters, TestDefinition, call_jvp, cube_definition,
    };
    use crate::operations::references::{ReferenceAddUpdate, ReferenceNew, ReferenceRead};
    use crate::programs::TypeError;

    use super::*;

    /// Returns the specialization key of an unbatched and undischarged call with the provided input and output types,
    /// whose differentiated inputs (i.e., all of them) have the provided tangent activity.
    fn unbatched_key<T>(
        input_types: Vec<T>,
        output_types: Vec<T>,
        tangent_activity: Vec<bool>,
    ) -> CustomRuleForwardSpecializationKey<T> {
        CustomRuleForwardSpecializationKey {
            input_types,
            output_types,
            kept_outputs: None,
            non_differentiated_count: 0,
            tangent_activity,
            levels: Vec::new(),
            discharged: false,
        }
    }

    /// Returns the specialization key of the unbatched backward rule of a call without non-differentiated inputs or
    /// seed extents, whose output seeds have the provided types and whose input cotangents are all returned.
    fn backward_key(
        leading_input_types: Vec<ArrayType>,
        input_tangent_types: Vec<ArrayType>,
        seed_types: Vec<Option<ArrayType>>,
    ) -> CustomRuleBackwardSpecializationKey<ArrayType> {
        CustomRuleBackwardSpecializationKey {
            leading_input_types,
            seed_extent_count: 0,
            destination_kinds: vec![CotangentDestinationKind::Return; input_tangent_types.len()],
            input_tangent_types,
            output_tangent_types: vec![ArrayType::scalar(DataType::F64)],
            seed_types,
            levels: Vec::new(),
            discharged: false,
        }
    }

    #[test]
    fn test_custom_rule_definition_new() {
        // A new definition has no rules, and its identity is its own: an equally named definition is another rule set.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let reference = registration.reference();
        assert_eq!(reference.definition().name(), "cube");
        assert!(reference.definition().named_axes().is_empty());
        assert!(!reference.definition().has_derivative_rules());
        assert!(!reference.has_jvp());
        assert!(!reference.derives_jvp_from_primal());
        assert!(!reference.has_vjp());
        assert_eq!(reference.batched_output_axes(), None);
        assert!(!reference.has_batching_rule());
        assert!(!reference.has_sequential_batching());
        assert_eq!(
            format!("{:?}", reference.definition()),
            "CustomRuleDefinition { name: \"cube\", has_jvp: false, derives_jvp_from_primal: false, has_vjp: false, \
             .. }",
        );
        assert_ne!(CustomRuleRegistration::new(TestDefinition::new("cube")).reference(), reference);

        // Every request for a rule that the definition lacks is rejected.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let key = unbatched_key(vec![scalar_type.clone()], vec![scalar_type.clone()], vec![true]);
        assert!(matches!(
            reference.jvp_specialization(key.clone()),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` has no forward-mode (i.e., JVP) rule",
        ));
        assert!(matches!(
            reference.forward_specialization(key),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` has no reverse-mode forward rule",
        ));
        assert!(matches!(
            reference.backward_specialization(backward_key(
                vec![scalar_type.clone()],
                vec![scalar_type.clone()],
                vec![Some(scalar_type.clone())],
            )),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite_transpose` `cube` has no backward rule",
        ));
        assert!(matches!(
            reference.batching_rule_specialization(CustomRuleBatchingSpecializationKey {
                level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                boundary_input_types: Vec::new(),
                input_types: vec![ArrayType::new_static(DataType::F64, [3])],
                unbatched_input_types: vec![scalar_type],
                input_axes: vec![BatchAxis::new(0)],
                discharged: false,
            }),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` has no custom batching rule",
        ));
        assert!(matches!(
            reference.sequential_batching_scan(3, 0, &ShardingDimension::Replicated, 1),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` does not batch its calls sequentially",
        ));
    }

    #[test]
    fn test_custom_rule_definition_with_named_axes() {
        // Every rule is traced with the named axes, which the definitions derived from the definition inherit.
        let named_axes = vec![("batch".to_string(), NamedAxis::Batched { size: Some(3) })];
        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_named_axes(named_axes.clone())
                .with_jvp_from_primal()
                .with_batching_support()
                .with_batching_rule(|inputs, input_axes, _, _| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        assert_eq!(registration.reference().definition().named_axes(), named_axes.as_slice());
        let key = CustomRuleDerivedDefinitionKey {
            kind: CustomRuleDerivationKind::Jvp,
            input_count: 1,
            active_input_indices: vec![0],
            output_tangent_mask: vec![true],
        };
        let derived = registration.reference().derived(key).unwrap();
        assert_eq!(derived.definition().named_axes(), named_axes.as_slice());
    }

    #[test]
    fn test_custom_rule_definition_with_jvp() {
        // The rule receives every primal input and one tangent per differentiated input, in which an inactive tangent
        // is a materialized zero, and its specialization returns the primal outputs followed by their tangents.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let reference = registration.reference();
        assert!(reference.has_jvp());
        assert!(!reference.derives_jvp_from_primal());
        assert!(reference.definition().has_derivative_rules());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let key =
            |tangent_activity| unbatched_key(vec![scalar_type.clone()], vec![scalar_type.clone()], tangent_activity);
        let active = reference.jvp_specialization(key(vec![true])).unwrap();
        assert!(active.output_axes.is_empty());
        assert_eq!(
            active.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %0 %0
                    %3:f64[] = mul %2 %0
                    %4:f64[] = mul %2 %1
                in (%3, %4)
            "}
            .trim_end(),
        );
        let inactive = reference.jvp_specialization(key(vec![false])).unwrap();
        assert_eq!(
            inactive.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = zero [type=f64[]]
                    %2:f64[] = mul %0 %0
                    %3:f64[] = mul %2 %0
                    %4:f64[] = mul %2 %1
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(counters.counts(), (2, 0, 0));

        // The rule replaces any previously configured forward mode rule, including a JVP derived from the primal.
        let replaced = CustomRuleRegistration::new(
            TestDefinition::new("identity")
                .with_jvp_from_primal()
                .with_jvp(|primals, tangents| Ok((primals.to_vec(), tangents.to_vec()))),
        );
        assert!(replaced.reference().has_jvp());
        assert!(!replaced.reference().derives_jvp_from_primal());
    }

    #[test]
    fn test_custom_rule_definition_with_symbolic_zero_jvp() {
        // The rule of `(x, y) ↦ xy` receives the inactive tangents as structural zeros, which it skips, and declares
        // the output tangent a structural zero when every input tangent is one. The specialization materializes that
        // zero at its boundary.
        let zero_tangents = Arc::new(AtomicUsize::new(0));
        let registration = CustomRuleRegistration::new(TestDefinition::new("product").with_symbolic_zero_jvp({
            let zero_tangents = zero_tangents.clone();
            move |primals, tangents| {
                let (x, y) = (&primals[0], &primals[1]);
                let tangent = [(&tangents[0], y), (&tangents[1], x)]
                    .into_iter()
                    .filter_map(|(tangent, factor)| match tangent {
                        MaybeZero::Value(tangent) => Some(tangent.clone() * factor.clone()),
                        MaybeZero::Zero(_) => {
                            zero_tangents.fetch_add(1, Ordering::SeqCst);
                            None
                        }
                    })
                    .reduce(|left, right| left + right);
                let tangent = match tangent {
                    Some(tangent) => MaybeZero::Value(tangent),
                    None => MaybeZero::Zero(x.r#type().into_owned()),
                };
                Ok((vec![x.clone() * y.clone()], vec![tangent]))
            }
        }));
        assert!(registration.reference().has_jvp());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let key = |tangent_activity| {
            unbatched_key(vec![scalar_type.clone(), scalar_type.clone()], vec![scalar_type.clone()], tangent_activity)
        };
        let specialization = registration.reference().jvp_specialization(key(vec![true, false])).unwrap();
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = mul %2 %1
                    %4:f64[] = mul %0 %1
                in (%4, %3)
            "}
            .trim_end(),
        );
        assert_eq!(zero_tangents.load(Ordering::SeqCst), 1);
        let specialization = registration.reference().jvp_specialization(key(vec![false, false])).unwrap();
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %0 %1
                    %3:f64[] = zero [type=f64[]]
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert_eq!(zero_tangents.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_custom_rule_definition_with_jvp_from_primal() {
        // A forward mode rule derived from the primal is not traced by the definition: forward mode differentiates the
        // primal region of each call instead, which computes the true derivative `3x²` of the cube.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(cube_definition(&counters, true, false).with_jvp_from_primal());
        let reference = registration.reference();
        assert!(reference.derives_jvp_from_primal());
        assert!(!reference.has_jvp());
        assert!(reference.definition().has_derivative_rules());
        let scalar_type = ArrayType::scalar(DataType::F64);
        assert!(matches!(
            reference.jvp_specialization(unbatched_key(vec![scalar_type.clone()], vec![scalar_type], vec![true])),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` has no forward-mode (i.e., JVP) rule",
        ));
        assert_eq!(
            call_jvp(&registration, Array::scalar(2f64).unwrap(), Array::scalar(1f64).unwrap()),
            Ok((vec![Array::scalar(8f64).unwrap()], vec![Array::scalar(12f64).unwrap()])),
        );
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_custom_rule_definition_with_vjp() {
        // The forward rule's specialization returns the primal outputs followed by the residuals, and the backward
        // rule's specialization consumes the live seeds followed by the leading inputs (here, the residual `x²`) and
        // returns the cotangent `2x²ȳ` of the differentiated input.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(cube_definition(&counters, false, true));
        let reference = registration.reference();
        assert!(reference.has_vjp());
        assert!(!reference.has_jvp());
        assert!(reference.definition().has_derivative_rules());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let forward = reference
            .forward_specialization(unbatched_key(vec![scalar_type.clone()], vec![scalar_type.clone()], Vec::new()))
            .unwrap();
        assert_eq!(
            forward.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = mul %1 %0
                in (%2, %1)
            "}
            .trim_end(),
        );
        let backward = reference
            .backward_specialization(backward_key(
                vec![scalar_type.clone()],
                vec![scalar_type.clone()],
                vec![Some(scalar_type.clone())],
            ))
            .unwrap();
        assert_eq!(backward.cotangents, vec![TransposedProgramCotangent::Output]);
        assert_eq!(
            backward.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                    %3:f64[] = add %2 %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(counters.counts(), (0, 1, 1));

        // A structural-zero seed reaches the backward rule materialized, so its specialization has no seed input.
        let backward = reference
            .backward_specialization(backward_key(vec![scalar_type.clone()], vec![scalar_type], vec![None]))
            .unwrap();
        assert_eq!(
            backward.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = zero [type=f64[]]
                    %2:f64[] = mul %0 %1
                    %3:f64[] = add %2 %2
                in (%3)
            "}
            .trim_end(),
        );
        assert_eq!(counters.counts(), (0, 1, 2));
    }

    #[test]
    fn test_custom_rule_definition_with_symbolic_zero_vjp() {
        // The backward rule of `(x, y) ↦ xy` treats `y` as a parameter, declaring its cotangent a structural zero, and
        // returns structural zeros for a structural-zero seed. Structural-zero cotangents consume no program output.
        let registration = CustomRuleRegistration::new(TestDefinition::new("product").with_symbolic_zero_vjp(
            |primals| Ok((vec![primals[0].clone() * primals[1].clone()], vec![primals[1].clone()])),
            |leading_inputs, seeds| {
                let zero = MaybeZero::Zero(leading_inputs[0].r#type().into_owned());
                match &seeds[0] {
                    MaybeZero::Value(seed) => {
                        Ok(vec![MaybeZero::Value(seed.clone() * leading_inputs[0].clone()), zero])
                    }
                    MaybeZero::Zero(_) => Ok(vec![zero.clone(), zero]),
                }
            },
        ));
        assert!(registration.reference().has_vjp());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let key = |seed_type| {
            backward_key(vec![scalar_type.clone()], vec![scalar_type.clone(), scalar_type.clone()], vec![seed_type])
        };
        let backward = registration.reference().backward_specialization(key(Some(scalar_type.clone()))).unwrap();
        assert_eq!(backward.cotangents, vec![TransposedProgramCotangent::Output, TransposedProgramCotangent::Zero]);
        assert_eq!(
            backward.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %0 %1
                in (%2)
            "}
            .trim_end(),
        );
        let backward = registration.reference().backward_specialization(key(None)).unwrap();
        assert_eq!(backward.cotangents, vec![TransposedProgramCotangent::Zero, TransposedProgramCotangent::Zero]);
        assert_eq!(
            backward.program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                in ()
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_custom_rule_definition_with_accumulating_vjp() {
        // The backward rule of the cube submits the contribution `x²ȳ` of its differentiated input to that input's
        // accumulator, which follows the accumulators of the leading inputs (here, the residual `x²`).
        let invocations = Arc::new(AtomicUsize::new(0));
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube").with_accumulating_vjp(
            |primals| {
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square]))
            },
            {
                let invocations = invocations.clone();
                move |context, inputs, seeds, accumulators| {
                    invocations.fetch_add(1, Ordering::SeqCst);
                    let (Some(residual), MaybeZero::Value(seed)) = (inputs[0].as_known(), &seeds[0]) else {
                        return Ok(());
                    };
                    accumulators[1].accumulate(context, MaybeZero::Value(residual.clone() * seed.clone()))?;
                    Ok(())
                }
            },
        ));
        assert!(registration.reference().has_vjp());
        let scalar_type = ArrayType::scalar(DataType::F64);
        let backward = registration
            .reference()
            .backward_specialization(backward_key(
                vec![scalar_type.clone()],
                vec![scalar_type.clone()],
                vec![Some(scalar_type)],
            ))
            .unwrap();
        assert_eq!(backward.cotangents, vec![TransposedProgramCotangent::Output]);
        assert_eq!(
            backward.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_custom_rule_definition_with_deferred_forward_rule() {
        // The deferral applies only to a definition with reverse mode rules, configuring it traces no rule, and the
        // sources of converted calls report the deferral of their member definition, even when unavailable.
        fn defers_forward_in_ir<S: CustomRuleSource<ArrayIrValue<Array>, ArrayIrOperation<Array>>>(rules: &S) -> bool {
            rules.defers_forward()
        }
        let counters = Arc::new(RuleCounters::default());
        let deferred =
            CustomRuleRegistration::new(cube_definition(&counters, false, true).with_deferred_forward_rule());
        assert!(deferred.reference().defers_forward());
        assert!(!CustomRuleRegistration::new(cube_definition(&counters, false, true)).reference().defers_forward());
        assert!(
            !CustomRuleRegistration::new(cube_definition(&counters, true, false).with_deferred_forward_rule())
                .reference()
                .defers_forward(),
        );
        assert!(defers_forward_in_ir(&LiftedCustomRules::Member(deferred.reference())));
        assert!(defers_forward_in_ir(&LiftedCustomRules::<Array, ArrayOperation<Array>>::Unavailable(
            UnavailableCustomRules::from_source(&deferred.reference()),
        )));
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_custom_rule_definition_with_batching_support() {
        // A batched specialization structurally batches the specialization traced at the unbatched types, which it
        // shares with unbatched calls, and aligns each output and tangent to the output axes of its level.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let scalar_type = ArrayType::scalar(DataType::F64);
        let key = CustomRuleForwardSpecializationKey {
            levels: vec![CustomRuleBatchingLevel {
                level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                boundary_input_count: 0,
                input_axes: vec![BatchAxis::new(0)],
                output_axes: vec![BatchAxis::new(0)],
            }],
            ..unbatched_key(vec![scalar_type.clone()], vec![scalar_type], vec![true])
        };
        let specialization = registration.reference().jvp_specialization(key.clone()).unwrap();
        assert_eq!(specialization.output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[3], %1:f64[3] .
                let %2:f64[3] = mul %0 %0
                    %3:f64[3] = mul %2 %0
                    %4:f64[3] = mul %2 %1
                in (%3, %4)
            "}
            .trim_end(),
        );
        assert_eq!(registration.caches().jvp_specializations.len(), 2);
        assert_eq!(counters.counts(), (1, 0, 0));

        // Without batching support, batched calls cannot be differentiated.
        let unsupported = CustomRuleRegistration::new(
            TestDefinition::new("cube").with_jvp(|primals, tangents| Ok((primals.to_vec(), tangents.to_vec()))),
        );
        assert!(matches!(
            unsupported.reference().jvp_specialization(key),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `cube` cannot differentiate a batched call because its definition has no \
                               batching support (refer to `CustomRuleDefinition::with_batching_support`)",
        ));
    }

    #[test]
    fn test_custom_rule_definition_with_batched_output_axes() {
        // Batched calls map every output at axis 0 unless the definition declares their batch axes, one per output.
        let default = CustomRuleRegistration::new(TestDefinition::new("pair"));
        assert_eq!(default.reference().batched_output_axes(), None);
        assert_eq!(default.reference().batched_call_output_axes(2), Ok(vec![BatchAxis::new(0); 2]));
        let output_axes = vec![BatchAxis::new(0), BatchAxis::replicated()];
        let declared =
            CustomRuleRegistration::new(TestDefinition::new("pair").with_batched_output_axes(output_axes.clone()));
        assert_eq!(declared.reference().batched_output_axes(), Some(output_axes.as_slice()));
        assert_eq!(declared.reference().batched_call_output_axes(2), Ok(output_axes));
        assert_eq!(
            declared.reference().batched_call_output_axes(1),
            Err(TypeError::invalid("expected 1 batched output axis but got 2")),
        );
    }

    #[test]
    fn test_custom_rule_definition_with_batching_rule() {
        // The rule's program is the batched primal of the call, with the batch axes that the rule declares.
        let registration = CustomRuleRegistration::new(MemberDefinition::new("square").with_batching_rule(
            |inputs, input_axes, _, _| Ok((vec![inputs[0].clone() * inputs[0].clone()], input_axes.to_vec())),
        ));
        assert!(registration.reference().has_batching_rule());
        let key = CustomRuleBatchingSpecializationKey {
            level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
            boundary_input_types: Vec::new(),
            input_types: vec![ArrayType::new_static(DataType::F64, [3])],
            unbatched_input_types: vec![ArrayType::scalar(DataType::F64)],
            input_axes: vec![BatchAxis::new(0)],
            discharged: false,
        };
        let specialization = registration.reference().batching_rule_specialization(key.clone()).unwrap();
        assert_eq!(specialization.output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:f64[3] .
                let %1:f64[3] = mul %0 %0
                in (%1)
            "}
            .trim_end(),
        );

        // The rule must declare one batch axis per output.
        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("square").with_batching_rule(|inputs, _, _, _| Ok((inputs.to_vec(), Vec::new()))),
        );
        assert!(matches!(
            registration.reference().batching_rule_specialization(key),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "`composite` `square` batching rule returned 1 outputs but 0 output batch axes",
        ));
    }

    #[test]
    fn test_custom_rule_definition_with_broadcast_sensitive_batching_rule() {
        // The rule batches calls exactly as a plain batching rule does, and it records the operations with which the
        // derived definitions recover exact per-item tangents (refer to the `CustomRuleDerivation` tests).
        let rule = |inputs: &[CustomRuleTracer<Array, ArrayOperation<Array>>],
                    axes: &[BatchAxis],
                    _: &BatchingLevel<ArrayType>,
                    _: &[CustomRuleTracer<Array, ArrayOperation<Array>>]| {
            Ok::<_, ProgramError>((vec![inputs[0].clone() * inputs[0].clone()], axes.to_vec()))
        };
        let plain = CustomRuleRegistration::new(MemberDefinition::new("square").with_batching_rule(rule));
        let broadcast_sensitive =
            CustomRuleRegistration::new(MemberDefinition::new("square").with_broadcast_sensitive_batching_rule(rule));
        assert!(broadcast_sensitive.reference().has_batching_rule());
        assert!(broadcast_sensitive.definition().batching_rule.as_ref().unwrap().broadcast_sensitivity.is_some());
        assert!(plain.definition().batching_rule.as_ref().unwrap().broadcast_sensitivity.is_none());
        let key = CustomRuleBatchingSpecializationKey {
            level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
            boundary_input_types: Vec::new(),
            input_types: vec![ArrayType::new_static(DataType::F64, [3])],
            unbatched_input_types: vec![ArrayType::scalar(DataType::F64)],
            input_axes: vec![BatchAxis::new(0)],
            discharged: false,
        };
        let specialization = broadcast_sensitive.reference().batching_rule_specialization(key.clone()).unwrap();
        let plain_specialization = plain.reference().batching_rule_specialization(key).unwrap();
        assert_eq!(specialization.output_axes, plain_specialization.output_axes);
        assert_eq!(specialization.program.to_string(), plain_specialization.program.to_string());
    }

    #[test]
    fn test_custom_rule_definition_with_sequential_batching() {
        // Sequential batching loops over the batch items with a scan that carries the replicated inputs and stacks the
        // outputs, whose body receives an `i64[]` slice index first.
        let registration = CustomRuleRegistration::new(MemberDefinition::new("square").with_sequential_batching());
        assert!(registration.reference().has_sequential_batching());
        assert!(!registration.reference().has_batching_rule());
        let (operation, index_type) =
            registration.reference().sequential_batching_scan(3, 1, &ShardingDimension::Replicated, 2).unwrap();
        assert_eq!(operation.to_string(), "scan [carry_count=1, iteration_count=3, reverse=false]");
        assert_eq!(index_type, ArrayType::scalar(DataType::I64));
    }

    #[test]
    fn test_custom_rule_definition_with_reference_discharge() {
        // The JVP rule `(x, ẋ) ↦ (2x, 2ẋ)` keeps local reference state, which the specializations of discharged calls
        // discharge when the definition supports it and reject otherwise.
        type IrTracer = CustomRuleTracer<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let definition = || {
            CustomRuleDefinition::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new("doubled").with_jvp(
                |primals: &[IrTracer], tangents: &[IrTracer]| {
                    let output = primals[0].reference_new()?;
                    output.add_update(&primals[0])?;
                    let tangent = tangents[0].reference_new()?;
                    tangent.add_update(&tangents[0])?;
                    Ok((vec![output.read()?], vec![tangent.read()?]))
                },
            )
        };
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let key = |discharged| CustomRuleForwardSpecializationKey {
            discharged,
            ..unbatched_key(vec![scalar_type.clone()], vec![scalar_type.clone()], vec![true])
        };
        let has_reference_state =
            |specialization: &Arc<CustomRuleSpecialization<ArrayIrValue<Array>, ArrayIrOperation<Array>>>| {
                let regions = specialization.program.regions();
                regions.iter().flat_map(|region| region.atoms()).any(|atom| atom.r#type().is_reference())
            };
        let supported = CustomRuleRegistration::new(definition().with_reference_discharge());
        let discharged = supported.reference().jvp_specialization(key(true)).unwrap();
        assert!(!has_reference_state(&discharged));
        assert_eq!(
            discharged.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = add %0 %0
                    %3:f32[] = add %1 %1
                in (%2, %3)
            "}
            .trim_end(),
        );
        assert!(has_reference_state(&supported.reference().jvp_specialization(key(false)).unwrap()));

        // Without discharge support, the specializations of discharged calls cannot keep the reference state.
        let unsupported = CustomRuleRegistration::new(definition());
        assert!(matches!(
            unsupported.reference().jvp_specialization(key(true)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`composite` `doubled` cannot differentiate a call whose reference state was \
                               discharged, because its rule programs contain reference state and its definition has \
                               no reference discharge support",
        ));
    }
}
