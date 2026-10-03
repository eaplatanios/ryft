//! Retained custom rule sets, whose derivative rules are callbacks rather than programs. A [`CustomRuleDefinition`]
//! holds a human-readable name, an optional forward-mode (JVP) rule, optional reverse-mode (VJP) forward and backward
//! rules, and optional batching support. A [`CustomFunctionOperation`](crate::CustomFunctionOperation) with
//! retained rules calls a definition over a traced primal program: ordinary execution uses only the primal, forward
//! mode traces and replays the JVP rule, and reverse mode traces the VJP forward rule and stages a
//! [`CustomFunctionTransposeOperation`] whose transposition specializes the backward rule against the actual
//! cotangent destinations. Every rule is traced lazily, on the first request of each static specialization, and cached
//! by the definition's [`CustomRuleRegistration`]. Batching a call or carrier likewise traces no rule: it records the
//! batching level, and a later derivative request structurally batches the rule program traced at the unbatched types
//! (refer to the Batching sections of [`CustomFunctionOperation`](crate::CustomFunctionOperation) and
//! [`CustomFunctionTransposeOperation`]).
//!
//! This module is the flat, family-level representation behind the public
//! [`custom_function`](fn@crate::custom_function) builder, which adapts structured closures to the flat rule interfaces
//! of this module. Rules may receive materialized zeros (e.g., [`CustomRuleDefinition::with_jvp`]) or structural zeros
//! (e.g., [`CustomRuleDefinition::with_symbolic_zero_jvp`]).

// TODO(eaplatanios): Review this module.

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;

use crate::arrays::ShardingDimension;
use crate::axes::{Axis, NamedAxis};
use crate::batching::{
    BatchAxis, BatchableType, BatchedProgram, BatchingError, BatchingLevel, BatchingLevelExtent, BatchingPolicy,
    DiagonalBatchingPolicy, ProgramBatchingOutputAxesPolicy, RecursiveBatchingPolicy,
};
use crate::contexts::Context;
use crate::differentiation::{
    CotangentAccumulator, CotangentBatchingPolicy, CotangentDestinationKind, DifferentiableOperation,
    DifferentiableType, DifferentiationError, ResidualZeroProvider, TransposableOperation, TranspositionContext,
};
use crate::macros::{check_count, check_types};
use crate::operations::arithmetic::AddOperation;
use crate::operations::custom_functions::operations::{
    CUSTOM_FUNCTION_OPERATION_NAME, CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME, CustomFunctionTransposeOperation,
};
use crate::operations::references::{ReferenceAddUpdateOperation, ReferenceNewOperation};
use crate::parameters::{Parameter, Placeholder};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationProvider, Program, ProgramBuilder, ProgramError, ReferenceAccessOperation,
    ReferenceDischargePolicy, ReferenceDischargeableOperation, ReferenceDischargeableType, ReferenceMemberType,
    ReferenceTransform, ReferenceType, Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::specialization::{ReentrantSpecializationError, SpecializationCache, SpecializationCacheError};
use crate::tracing::{Tracer, TracingContext};

/// Maximum number of specializations that each rule of a [`CustomRuleDefinition`] retains.
pub(super) const CUSTOM_RULE_SPECIALIZATION_CAPACITY: usize = 16;

/// Next process-unique identity assigned to a [`CustomRuleDefinition`].
static NEXT_CUSTOM_RULE_DEFINITION_ID: AtomicU64 = AtomicU64::new(0);

/// Tracer over which retained custom rules are written. It is typed by the registration family's staged constant and
/// operation types only, so a rule can be traced lazily under whichever transform later requests it.
pub type CustomRuleTracer<V, O> = Tracer<TracingContext<V, O>>;

// The retained rules are stored behind these object-safe traits rather than behind `dyn Fn` types because their
// signatures mention tracing contexts, which require `O: Operation`. Keeping that requirement on the methods leaves
// `CustomRuleDefinition` and the operations free of an `O: Operation` bound, which would otherwise form a cycle with
// the payload bounds of any operation family that contains them.

/// Retained custom Jacobian-Vector Product (JVP) rule `(p, x, ẋ) ↦ (y, ẏ)`. It receives every primal input (i.e., the
/// non-differentiated inputs `p` followed by the differentiated inputs `x`) and the tangents of the differentiated
/// inputs only, and returns the primal outputs together with their tangents.
trait CustomJvpRule<V, O>: Send + Sync {
    /// Applies this rule to the provided primal inputs and differentiated input tangents.
    fn apply(
        &self,
        primals: &[CustomRuleTracer<V, O>],
        tangents: &[CustomRuleTracer<V, O>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomJvpRule<V, O> for F
where
    F: Fn(
            &[CustomRuleTracer<V, O>],
            &[CustomRuleTracer<V, O>],
        ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
        + Send
        + Sync,
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
/// would otherwise require. It is otherwise identical to [`CustomJvpRule`].
trait CustomSymbolicZeroJvpRule<V, O>: Send + Sync {
    /// Applies this rule to the provided primal inputs and differentiated input tangents.
    fn apply(
        &self,
        primals: &[CustomRuleTracer<V, O>],
        tangents: &[MaybeZero<CustomRuleTracer<V, O>>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomSymbolicZeroJvpRule<V, O> for F
where
    F: Fn(
            &[CustomRuleTracer<V, O>],
            &[MaybeZero<CustomRuleTracer<V, O>>],
        ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
        + Send
        + Sync,
{
    #[inline]
    fn apply(
        &self,
        primals: &[CustomRuleTracer<V, O>],
        tangents: &[MaybeZero<CustomRuleTracer<V, O>>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError> {
        self(primals, tangents)
    }
}

/// Retained custom Vector-Jacobian Product (VJP) forward rule `(p, x) ↦ (y, r)`, which returns the primal outputs
/// together with the residuals that the backward rule consumes.
trait CustomVjpForwardRule<V, O>: Send + Sync {
    /// Applies this rule to the provided primal inputs.
    fn apply(
        &self,
        primals: &[CustomRuleTracer<V, O>],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomVjpForwardRule<V, O> for F
where
    F: Fn(
            &[CustomRuleTracer<V, O>],
        ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
        + Send
        + Sync,
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
/// output, and it returns one cotangent per differentiated input. This is the primary reverse-mode form: it is a pure
/// function of its inputs, like any other traced program, and transposition adds its results to whatever destination
/// each differentiated input's cotangent has.
pub(super) trait CustomVjpBackwardRule<V, O>: Send + Sync {
    /// Applies this rule to the provided leading inputs and output cotangent seeds.
    fn apply(
        &self,
        leading_inputs: &[CustomRuleTracer<V, O>],
        seeds: &[CustomRuleTracer<V, O>],
    ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomVjpBackwardRule<V, O> for F
where
    F: Fn(&[CustomRuleTracer<V, O>], &[CustomRuleTracer<V, O>]) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError>
        + Send
        + Sync,
{
    #[inline]
    fn apply(
        &self,
        leading_inputs: &[CustomRuleTracer<V, O>],
        seeds: &[CustomRuleTracer<V, O>],
    ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError> {
        self(leading_inputs, seeds)
    }
}

/// Retained custom Vector-Jacobian Product (VJP) backward rule `(p, r, ȳ) ↦ x̄` that receives its structural-zero
/// output cotangent seeds as [`MaybeZero::Zero`] leaves instead of materialized zeros, so that it can skip the work
/// that they would otherwise require. It is otherwise identical to [`CustomVjpBackwardRule`].
pub(super) trait CustomSymbolicZeroVjpBackwardRule<V, O>: Send + Sync {
    /// Applies this rule to the provided leading inputs and output cotangent seeds.
    fn apply(
        &self,
        leading_inputs: &[CustomRuleTracer<V, O>],
        seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
    ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomSymbolicZeroVjpBackwardRule<V, O> for F
where
    F: Fn(
            &[CustomRuleTracer<V, O>],
            &[MaybeZero<CustomRuleTracer<V, O>>],
        ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError>
        + Send
        + Sync,
{
    #[inline]
    fn apply(
        &self,
        leading_inputs: &[CustomRuleTracer<V, O>],
        seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
    ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError> {
        self(leading_inputs, seeds)
    }
}

/// Retained accumulating custom Vector-Jacobian Product (VJP) backward rule. It is invoked while transposing a
/// [`CustomFunctionTransposeOperation`] with the known leading inputs (i.e., `p` followed by the residuals `r`), the
/// output cotangent seeds, and one [`CotangentAccumulator`] per carrier input, and it submits the contributions of the
/// differentiated inputs to their accumulators. A single rule therefore serves returned, caller-buffer, and ignored
/// destinations, and can, for example, update only the affected entries of a caller-provided buffer instead of
/// returning a full-sized cotangent.
pub(super) trait CustomVjpAccumulatingBackwardRule<V, O>: Send + Sync {
    /// Applies this rule in the provided transposition context.
    fn apply(
        &self,
        context: &mut TranspositionContext<V, O>,
        inputs: &[PartialValue<CustomRuleTracer<V, O>>],
        seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError>
    where
        V: Value,
        O: Operation<Type = V::Type>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomVjpAccumulatingBackwardRule<V, O> for F
where
    F: Fn(
            &mut TranspositionContext<V, O>,
            &[PartialValue<CustomRuleTracer<V, O>>],
            &[MaybeZero<CustomRuleTracer<V, O>>],
            &[CotangentAccumulator],
        ) -> Result<(), DifferentiationError>
        + Send
        + Sync,
{
    #[inline]
    fn apply(
        &self,
        context: &mut TranspositionContext<V, O>,
        inputs: &[PartialValue<CustomRuleTracer<V, O>>],
        seeds: &[MaybeZero<CustomRuleTracer<V, O>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        self(context, inputs, seeds, accumulators)
    }
}

/// Retained custom batching rule, which batches a call of a [`CustomRuleDefinition`] instead of structurally batching
/// its primal region. It receives the batching level, the level's boundary inputs (e.g., a first-class batch extent),
/// the call's batched inputs, and their batch axes, and it returns the batched outputs together with their batch axes.
trait CustomBatchingRule<V: Typed, O>: Send + Sync {
    /// Applies this rule to the provided batched inputs.
    fn apply(
        &self,
        level: &BatchingLevel<V::Type>,
        boundary_inputs: &[CustomRuleTracer<V, O>],
        inputs: &[CustomRuleTracer<V, O>],
        input_axes: &[BatchAxis],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>;
}

impl<V: Value, O: Operation<Type = V::Type>, F> CustomBatchingRule<V, O> for F
where
    F: Fn(
            &BatchingLevel<V::Type>,
            &[CustomRuleTracer<V, O>],
            &[CustomRuleTracer<V, O>],
            &[BatchAxis],
        ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError>
        + Send
        + Sync,
{
    #[inline]
    fn apply(
        &self,
        level: &BatchingLevel<V::Type>,
        boundary_inputs: &[CustomRuleTracer<V, O>],
        inputs: &[CustomRuleTracer<V, O>],
        input_axes: &[BatchAxis],
    ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError> {
        self(level, boundary_inputs, inputs, input_axes)
    }
}

/// Operations with which the derived batching rules of an axis-dependent custom batching rule recover exact per-item
/// tangents (refer to [`CustomRuleDefinition::with_axis_dependent_batching_rule`]).
trait CustomRuleAxisDependence<V, O>: Send + Sync {
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
        O: Operation<Type = V::Type>;

    /// Returns the sum of two values of the same type.
    fn add(
        &self,
        left: &CustomRuleTracer<V, O>,
        right: &CustomRuleTracer<V, O>,
    ) -> Result<CustomRuleTracer<V, O>, ProgramError>
    where
        V: Value,
        O: Operation<Type = V::Type>;
}

/// [`CustomRuleAxisDependence`] that takes diagonals with the canonical batching policy of the family's type universe
/// and adds values with [`AddOperation`].
struct PolicyAxisDependence<V, O>(PhantomData<fn() -> (V, O)>);

impl<V, O> CustomRuleAxisDependence<V, O> for PolicyAxisDependence<V, O>
where
    V: Value<Type: BatchableType<Policy: DiagonalBatchingPolicy<TracingContext<V, O>>>>,
    O: Operation<Type = V::Type> + From<AddOperation<V::Type>>,
{
    fn diagonal(
        &self,
        value: &CustomRuleTracer<V, O>,
        kept_axis: usize,
        removed_axis: usize,
    ) -> Result<CustomRuleTracer<V, O>, ProgramError> {
        type Policy<T> = <T as BatchableType>::Policy;
        Ok(<Policy<V::Type> as DiagonalBatchingPolicy<TracingContext<V, O>>>::diagonal(
            value.context(),
            value,
            kept_axis,
            removed_axis,
        )?)
    }

    fn add(
        &self,
        left: &CustomRuleTracer<V, O>,
        right: &CustomRuleTracer<V, O>,
    ) -> Result<CustomRuleTracer<V, O>, ProgramError> {
        let mut outputs = left.context().bind(AddOperation::new(), Vec::new(), &[left.clone(), right.clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Forward-mode rule of a [`CustomRuleDefinition`].
enum CustomRuleJvp<V, O> {
    /// Retained custom JVP rule that receives materialized input tangents (refer to [`CustomJvpRule`]).
    Rule(Arc<dyn CustomJvpRule<V, O>>),

    /// Retained custom JVP rule that receives structural-zero input tangents (refer to [`CustomSymbolicZeroJvpRule`]).
    SymbolicZeroRule(Arc<dyn CustomSymbolicZeroJvpRule<V, O>>),

    /// JVP derived from the primal region of each call, which is an explicit registration choice (i.e., never a
    /// fallback for a missing rule).
    Primal,
}

/// Borrowed retained JVP rule of a [`CustomRuleDefinition`], as applied by its forward-mode specializations.
enum CustomRuleJvpRule<'r, V, O> {
    /// Rule that receives materialized input tangents.
    Materialized(&'r Arc<dyn CustomJvpRule<V, O>>),

    /// Rule that receives structural-zero input tangents.
    SymbolicZeros(&'r Arc<dyn CustomSymbolicZeroJvpRule<V, O>>),
}

/// Retained custom Vector-Jacobian Product (VJP) backward rule of a [`CustomRuleDefinition`].
pub(super) enum CustomVjpBackward<V, O> {
    /// Pure backward rule that receives materialized seeds (refer to [`CustomVjpBackwardRule`]).
    Pure(Arc<dyn CustomVjpBackwardRule<V, O>>),

    /// Pure backward rule that receives structural-zero seeds (refer to [`CustomSymbolicZeroVjpBackwardRule`]).
    PureWithSymbolicZeros(Arc<dyn CustomSymbolicZeroVjpBackwardRule<V, O>>),

    /// Accumulating backward rule (refer to [`CustomVjpAccumulatingBackwardRule`]).
    Accumulating(Arc<dyn CustomVjpAccumulatingBackwardRule<V, O>>),
}

/// Transposition of a backward-rule specialization source program with respect to the provided inputs and destination
/// kinds. [`CustomRuleDefinition::with_vjp`] captures it where the operation family's transposition bounds are known,
/// because requiring them on the carrier's own transposition rule would form a cycle with the family's payload bounds.
type CustomRuleTransposer<V, O> = dyn Fn(
        &Program<V, O, Vec<V>, Vec<V>>,
        &[usize],
        &[CotangentDestinationKind],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
    + Send
    + Sync;

/// Reference discharge of a traced rule program whose call was discharged, which replaces the program's local reference
/// state with explicitly threaded values. [`CustomRuleDefinition::with_reference_discharge`] captures it where the
/// operation family's discharge bounds are known, because requiring them on the operations' own rules would form a
/// cycle with the family's payload bounds (and would exclude families without references).
type CustomRuleDischarger<V, O> =
    dyn Fn(Program<V, O, Vec<V>, Vec<V>>) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> + Send + Sync;

/// Kind of values that a batched rule program returns, which determines how an output that is mapped while its
/// required batch axis is replicated is reconciled.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CustomRuleBatchedOutputs {
    /// Primal or tangent values, for which such an output violates the required layout and is rejected.
    Values,

    /// Cotangents of replicated inputs, for which such an output is summed along its mapped axis (i.e., the transpose
    /// of broadcasting the input across the batch).
    Cotangents,
}

/// Structural batching of a traced rule program at one recorded [`BatchingLevel`] with the provided input batch axes.
/// Each output is aligned to its required batch axis, or keeps its natural batch axis when the requirement is [`None`].
/// The result is the batched program, which consumes the level's boundary inputs before the source program's inputs,
/// together with its output batch axes. [`CustomRuleDefinition::with_batching`] captures it where the operation
/// family's batching bounds are known, because requiring them on the operations' own rules would form a cycle with the
/// family's payload bounds.
type CustomRuleBatcher<V, O> = dyn Fn(
        &BatchingLevel<<V as Typed>::Type>,
        &Program<V, O, Vec<V>, Vec<V>>,
        &[BatchAxis],
        &[Option<BatchAxis>],
        CustomRuleBatchedOutputs,
    ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<BatchAxis>), BatchingError>
    + Send
    + Sync;

/// Forward-mode differentiation of a traced rule program with respect to the provided inputs, which returns its fused
/// Jacobian-Vector Product (JVP) program (refer to [`Program::jvp_with_respect_to`]). The batching rules of derived
/// definitions (refer to [`CustomRuleDerivation`]) differentiate the batched programs of their source's custom
/// batching rule with it. [`CustomRuleDefinition::with_batching_rule`] captures it where the operation family's
/// differentiation bounds are known, because requiring them on the operations' own rules would form a cycle with the
/// family's payload bounds.
type CustomRuleDifferentiator<V, O> = dyn Fn(&Program<V, O, Vec<V>, Vec<V>>, &[usize]) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
    + Send
    + Sync;

/// One batching level applied to a [`CustomFunctionOperation`](crate::CustomFunctionOperation) or
/// [`CustomFunctionTransposeOperation`] after it was staged. Batching such an operation batches no derivative rule.
/// Instead, it records this level so that a later derivative request can batch the rule program that it traces at the
/// unbatched types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomRuleBatchingLevel<T> {
    /// Context-neutral description of the level.
    pub(super) level: BatchingLevel<T>,

    /// Number of policy boundary inputs (e.g., a first-class batch extent) that batching at this level prepended to the
    /// operation's inputs (refer to the documentation of [`BatchingPolicy::boundary_inputs`]).
    pub(super) boundary_input_count: usize,

    /// Batch axes of the operation's inputs at this level, excluding the boundary inputs prepended at this level.
    pub(super) input_axes: Vec<BatchAxis>,

    /// Batch axes of the operation's outputs at this level.
    pub(super) output_axes: Vec<BatchAxis>,
}

/// Batching applied to a [`CustomFunctionOperation`](crate::CustomFunctionOperation) or
/// [`CustomFunctionTransposeOperation`] after it was staged, which determines how its lazily traced rule programs are
/// batched.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct CustomRuleBatching<T> {
    /// Unbatched types of the operation's inputs, excluding all boundary inputs.
    pub(super) input_types: Vec<T>,

    /// Unbatched types of the operation's outputs.
    pub(super) output_types: Vec<T>,

    /// Batching levels, from the innermost (i.e., the first one applied) to the outermost.
    pub(super) levels: Vec<CustomRuleBatchingLevel<T>>,
}

impl<T: Type> CustomRuleBatching<T> {
    /// Returns the number of boundary inputs that all levels together prepended to the operation's inputs.
    pub(super) fn boundary_input_count(&self) -> usize {
        boundary_input_count(&self.levels)
    }

    /// Returns this [`CustomRuleBatching`] with its types converted into the type family `U`, as when its operation is
    /// converted into a family that contains the operation's family.
    pub(super) fn map_types<U: Type + From<T>>(self) -> CustomRuleBatching<U> {
        let map_types = |types: Vec<T>| types.into_iter().map(U::from).collect::<Vec<_>>();
        CustomRuleBatching {
            input_types: map_types(self.input_types),
            output_types: map_types(self.output_types),
            levels: self
                .levels
                .into_iter()
                .map(|level| CustomRuleBatchingLevel {
                    level: BatchingLevel::new(
                        match level.level.extent() {
                            BatchingLevelExtent::Static(extent) => BatchingLevelExtent::Static(*extent),
                            BatchingLevelExtent::Dynamic(r#type) => {
                                BatchingLevelExtent::Dynamic(U::from(r#type.clone()))
                            }
                        },
                        level.level.axis_name().map(str::to_owned),
                        level.level.axis_sharding().clone(),
                    ),
                    boundary_input_count: level.boundary_input_count,
                    input_axes: level.input_axes,
                    output_axes: level.output_axes,
                })
                .collect(),
        }
    }

    /// Returns this [`CustomRuleBatching`] after renaming the type identities in its types and levels.
    pub(super) fn rename_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        let rename =
            |types: &[T]| types.iter().map(|r#type| r#type.rename_identities(renaming)).collect::<Result<Vec<_>, _>>();
        Ok(Self {
            input_types: rename(&self.input_types)?,
            output_types: rename(&self.output_types)?,
            levels: self
                .levels
                .iter()
                .map(|level| {
                    Ok(CustomRuleBatchingLevel { level: level.level.rename_identities(renaming)?, ..level.clone() })
                })
                .collect::<Result<Vec<_>, TypeError>>()?,
        })
    }
}

impl<T: Display> Display for CustomRuleBatching<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let axes = |axes: &[BatchAxis]| axes.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        write!(formatter, "[")?;
        for (index, level) in self.levels.iter().enumerate() {
            if index > 0 {
                write!(formatter, ", ")?;
            }
            write!(formatter, "(extent=")?;
            match level.level.extent() {
                BatchingLevelExtent::Static(extent) => write!(formatter, "{extent}")?,
                BatchingLevelExtent::Dynamic(r#type) => write!(formatter, "{type}")?,
            }
            if let Some(axis_name) = level.level.axis_name() {
                write!(formatter, ", axis_name={axis_name:?}")?;
            }
            write!(
                formatter,
                ", input_axes=[{}], output_axes=[{}])",
                axes(&level.input_axes),
                axes(&level.output_axes)
            )?;
        }
        write!(formatter, "]")
    }
}

/// Specialization key of the forward-mode and reverse-mode primal rules of a [`CustomRuleDefinition`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomRuleSpecializationKey<T> {
    /// Unbatched types of every input of the call, excluding boundary inputs.
    pub(super) input_types: Vec<T>,

    /// Unbatched types of every output of the call. The key must include them, even though rules are traced from the
    /// inputs alone, because a specialization is validated against them and its results are partitioned by them: two
    /// calls with the same inputs but different primal outputs must never share a specialization.
    pub(super) output_types: Vec<T>,

    /// Number of leading inputs that parameterize the rule without being differentiated, excluding boundary inputs.
    pub(super) non_differentiated_count: usize,

    /// Whether each differentiated input's tangent is active (i.e., not a structural zero) in a forward-mode request,
    /// which is empty for reverse-mode forward requests, whose rules receive no tangents. A forward-mode specialization
    /// receives only the active tangents, and it materializes the structural zeros itself (or hands them to a rule that
    /// receives structural zeros), so that the zeros that the rule propagates remain recognizable as zeros.
    pub(super) tangent_activity: Vec<bool>,

    /// Batching levels of the call, from the innermost to the outermost, which are empty for an unbatched call.
    pub(super) levels: Vec<CustomRuleBatchingLevel<T>>,

    /// Whether the call's reference state was discharged, in which case the specialization is discharged as well.
    pub(super) discharged: bool,
}

impl<T: Type + Eq + Hash> CustomRuleSpecializationKey<T> {
    /// Returns the key of the specialization from which this key's specialization is batched (i.e., without its
    /// outermost level), together with that level, or [`None`] when this key is unbatched.
    fn split_outermost_level(&self) -> Option<(Self, &CustomRuleBatchingLevel<T>)> {
        let (outermost, inner) = self.levels.split_last()?;
        Some((Self { levels: inner.to_vec(), ..self.clone() }, outermost))
    }
}

/// Specialization key of the backward rule of a [`CustomRuleDefinition`]. It contains only static information, and
/// never a runtime reference identity or the contents of a value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomRuleBackwardSpecializationKey<T> {
    /// Unbatched types of the known leading inputs (i.e., the non-differentiated inputs, the residuals, and the seed
    /// geometry), excluding boundary inputs.
    pub(super) leading_input_types: Vec<T>,

    /// Number of trailing leading inputs that carry seed geometry (refer to [`CustomFunctionTransposeOperation`]).
    pub(super) seed_geometry_count: usize,

    /// Unbatched tangent types of the differentiated inputs.
    pub(super) input_tangent_types: Vec<T>,

    /// Unbatched tangent types of the outputs.
    pub(super) output_tangent_types: Vec<T>,

    /// Unbatched type of each output cotangent seed, or [`None`] for a structural-zero seed.
    pub(super) seed_types: Vec<Option<T>>,

    /// Destination kind of each differentiated input's cotangent.
    pub(super) destination_kinds: Vec<CotangentDestinationKind>,

    /// Batching levels of the carrier, from the innermost to the outermost, which are empty for an unbatched carrier.
    pub(super) levels: Vec<CustomRuleBatchingLevel<T>>,

    /// Whether the carrier's reference state was discharged, in which case the specialization is discharged as well.
    pub(super) discharged: bool,
}

impl<T: Type + Eq + Hash> CustomRuleBackwardSpecializationKey<T> {
    /// Returns the key of the specialization from which this key's specialization is batched (i.e., without its
    /// outermost level), together with that level, or [`None`] when this key is unbatched.
    fn split_outermost_level(&self) -> Option<(Self, &CustomRuleBatchingLevel<T>)> {
        let (outermost, inner) = self.levels.split_last()?;
        Some((Self { levels: inner.to_vec(), ..self.clone() }, outermost))
    }
}

/// Specialization key of the custom batching rule of a [`CustomRuleDefinition`] (refer to
/// [`CustomRuleDefinition::with_batching_rule`]). It contains only static information: the level and the types and
/// batch axes of the batched call's inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomRuleBatchingRuleKey<T> {
    /// Batching level at which the call is batched.
    pub(super) level: BatchingLevel<T>,

    /// Types of the level's boundary inputs (e.g., a first-class batch extent).
    pub(super) boundary_input_types: Vec<T>,

    /// Types of the call's batched inputs, which include any boundary inputs of earlier levels.
    pub(super) input_types: Vec<T>,

    /// Per-item types of the call's batched inputs at this level (i.e., the input types of the call's primal region
    /// before this level batched it), which derived batching rules need to batch programs structurally (refer to
    /// [`CustomRuleDerivation`]).
    pub(super) unbatched_input_types: Vec<T>,

    /// Batch axis of each batched input.
    pub(super) input_axes: Vec<BatchAxis>,
}

/// Kind of call that a derived definition (refer to [`CustomRuleDerivation`]) stages for the forward-mode derivative
/// of a call whose forward-mode rule is derived from its primal.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum CustomRuleDerivationKind {
    /// Call that maps the source call's inputs and the active tangents of its differentiated inputs to its outputs and
    /// their live tangents, which fused differentiation contexts stage.
    Jvp,

    /// Call that maps the source call's inputs and the active tangents of its differentiated inputs to the live
    /// tangents of its outputs only, which partitioned differentiation contexts (e.g., linearization) stage on their
    /// tangent side.
    Pushforward,
}

/// Key of a definition derived from a source [`CustomRuleDefinition`] (refer to [`CustomRuleSpecializer::derived`]). It
/// contains only static information about the derived calls, so that calls whose primal regions differ share one
/// derived definition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CustomRuleDerivationKey {
    /// Kind of the derived calls.
    pub(super) kind: CustomRuleDerivationKind,

    /// Number of inputs of the source call.
    pub(super) input_count: usize,

    /// Indices of the source call's inputs whose tangents are active, in increasing order, which is the order of the
    /// derived calls' tangent inputs.
    pub(super) active_input_indices: Vec<usize>,

    /// Whether each output of the source call has a live tangent, in output order, which determines the derived calls'
    /// tangent outputs.
    pub(super) output_tangent_mask: Vec<bool>,
}

/// Derivation of a definition from a source definition, which is stored in the derived definition in place of a
/// batching rule (refer to [`CustomRuleSpecializer::derived`]).
struct CustomRuleDerivation<V: Typed + Parameter, O> {
    /// Rules of the source definition.
    source: CustomRuleReference<V, O>,

    /// Key of this derivation.
    key: CustomRuleDerivationKey,
}

/// Request for one specialization of a retained rule of a [`CustomRuleDefinition`], which identifies both the cache
/// that retains it and, when no cache is alive, the in-flight marker that rejects its recursive production.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum CustomRuleSpecializationRequest<T> {
    /// Specialization of the forward-mode rule.
    Jvp(CustomRuleSpecializationKey<T>),

    /// Specialization of the reverse-mode forward rule.
    Forward(CustomRuleSpecializationKey<T>),

    /// Destination-specialized transpose of the reverse-mode backward rule.
    Backward(CustomRuleBackwardSpecializationKey<T>),

    /// Batched primal program produced by the custom batching rule.
    BatchingRule(CustomRuleBatchingRuleKey<T>),
}

/// Program produced by specializing a retained custom rule, together with the batch axes of its outputs at the
/// outermost batching level of its key (which are empty for an unbatched specialization).
pub struct CustomRuleSpecialization<V: Typed + Parameter, O> {
    /// Specialized program.
    pub(super) program: Program<V, O, Vec<V>, Vec<V>>,

    /// Batch axes of the specialized program's outputs.
    pub(super) output_axes: Vec<BatchAxis>,
}

/// Shared [`CustomRuleSpecialization`], as retained by the specialization caches of a [`CustomRuleDefinition`].
pub type CustomRuleProgram<V, O> = Arc<CustomRuleSpecialization<V, O>>;

/// Immutable definition of a custom rule set whose derivative rules are retained callbacks that are traced lazily, on
/// the first request of each specialization. A definition is registered once (refer to [`CustomRuleRegistration`]) and
/// shared by every call that uses it, and two definitions are the same rule set exactly when they are the same
/// definition (i.e., identity is a process-unique identifier assigned at construction rather than structural).
/// [`Self::name`] is a human-readable label for rendering and diagnostics only; distinct definitions may share it.
///
/// Specialization is deterministic and repeatable but not exactly-once: eviction, retries after a failed
/// specialization, concurrent cold requests, and requests made after the registration handle was dropped may invoke a
/// rule again, so rules must not depend on mutable state outside their key. Host-side side effects of a rule (e.g.,
/// counting its invocations) are not program effects.
///
/// # Ownership
///
/// The specialized programs are not stored in the definition, because a program that calls the definition (e.g., a
/// rule that calls its own definition, directly or through other definitions) would then keep the definition alive
/// forever. They are stored in caches owned by the [`CustomRuleRegistration`] handle, while calls and
/// rules refer to the definition through a [`CustomRuleReference`], which holds the definition strongly and the caches
/// only weakly. The strong references therefore always point from programs to definitions and from the handle to the
/// caches, and never back from a definition to a program. Refer to [`CustomRuleRegistration`] for how rules must refer
/// to definitions.
pub struct CustomRuleDefinition<V: Typed + Parameter, O> {
    /// Process-unique identity of this definition.
    id: u64,

    /// Human-readable label used for rendering and diagnostics.
    name: Cow<'static, str>,

    /// Named axes with which every rule is traced (refer to [`Self::with_named_axes`]).
    named_axes: Vec<(String, NamedAxis)>,

    /// Optional forward-mode rule.
    jvp: Option<CustomRuleJvp<V, O>>,

    /// Optional retained reverse-mode forward rule, present exactly when [`Self::backward`] is present.
    forward: Option<Arc<dyn CustomVjpForwardRule<V, O>>>,

    /// Optional retained reverse-mode backward rule, present exactly when [`Self::forward`] is present.
    pub(super) backward: Option<CustomVjpBackward<V, O>>,

    /// Transposer of backward-rule specialization source programs, present exactly when [`Self::backward`] is present.
    transposer: Option<Arc<CustomRuleTransposer<V, O>>>,

    /// Optional batcher of traced rule programs, without which batched calls can be executed but not differentiated.
    batcher: Option<Arc<CustomRuleBatcher<V, O>>>,

    /// Optional batch axes of the outputs of batched calls, which default to mapping every output at axis 0.
    batched_output_axes: Option<Vec<BatchAxis>>,

    /// Optional custom batching rule, which batches calls instead of structurally batching their primal regions.
    batching_rule: Option<Arc<dyn CustomBatchingRule<V, O>>>,

    /// Differentiator of traced rule programs, present when [`Self::batching_rule`] is present, with which the batching
    /// rules of the definitions derived from this definition differentiate its batching rule's programs.
    differentiator: Option<Arc<CustomRuleDifferentiator<V, O>>>,

    /// Operations with which the batching rules of the definitions derived from this definition recover exact per-item
    /// tangents, present when [`Self::batching_rule`] depends on its input batch axes (refer to
    /// [`Self::with_axis_dependent_batching_rule`]).
    axis_dependence: Option<Arc<dyn CustomRuleAxisDependence<V, O>>>,

    /// Derivation of this definition from a source definition, whose batching rule this definition's calls apply
    /// instead of [`Self::batching_rule`] (refer to [`CustomRuleDerivation`]).
    derivation: Option<CustomRuleDerivation<V, O>>,

    /// Optional reference discharge of traced rule programs, without which discharged calls can be differentiated
    /// only when their rule programs contain no reference state.
    discharger: Option<Arc<CustomRuleDischarger<V, O>>>,

    /// Specializations that each thread is producing without a cache (i.e., after the registration handle was
    /// dropped), which reject recursive production exactly as a [`SpecializationCache`] does. The markers contain
    /// static keys only, so they never retain a program.
    uncached_in_flight: Mutex<HashSet<(ThreadId, CustomRuleSpecializationRequest<V::Type>)>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleDefinition<V, O> {
    /// Creates a new [`CustomRuleDefinition`] with the provided label and no derivative rules.
    pub fn new<N: Into<Cow<'static, str>>>(name: N) -> Self {
        Self {
            id: NEXT_CUSTOM_RULE_DEFINITION_ID.fetch_add(1, Ordering::Relaxed),
            name: name.into(),
            named_axes: Vec::new(),
            jvp: None,
            forward: None,
            backward: None,
            transposer: None,
            batcher: None,
            batched_output_axes: None,
            batching_rule: None,
            differentiator: None,
            axis_dependence: None,
            derivation: None,
            discharger: None,
            uncached_in_flight: Mutex::new(HashSet::new()),
        }
    }

    /// Returns this [`CustomRuleDefinition`] with the provided named axes, with which every rule is traced, so that
    /// the rules resolve the names bound where the calls that apply them are made (e.g., an `axis_index` over the axis
    /// of an enclosing batching level). The rules are traced lazily and in fresh traces, which see no enclosing
    /// bindings otherwise. The definitions that are derived from this definition trace their rules with the same axes.
    pub fn with_named_axes(mut self, named_axes: Vec<(String, NamedAxis)>) -> Self {
        self.named_axes = named_axes;
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided forward-mode rule, replacing any previously configured
    /// forward-mode rule (including a JVP derived from the primal). The rule implements `(p, x, ẋ) ↦ (y, ẏ)`: it
    /// receives every primal input (i.e., the non-differentiated inputs `p` followed by the differentiated inputs `x`)
    /// and one tangent per differentiated input, in which structural zeros are materialized, and it returns the primal
    /// outputs followed by one tangent per output.
    pub fn with_jvp<R>(mut self, rule: R) -> Self
    where
        V: Value,
        O: 'static + Operation<Type = V::Type>,
        R: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
                &[CustomRuleTracer<V, O>],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
            + Send
            + Sync,
    {
        self.jvp = Some(CustomRuleJvp::Rule(Arc::new(rule)));
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided forward-mode rule, replacing any previously configured
    /// forward-mode rule. The rule is used as the rule of [`Self::with_jvp`], except that it receives each structurally
    /// zero input tangent as a [`MaybeZero::Zero`] rather than as a materialized zero, which lets it skip the work that
    /// the zero would otherwise require. Each pattern of structurally zero input tangents is a separate specialization
    /// of the rule.
    pub fn with_symbolic_zero_jvp<R>(mut self, rule: R) -> Self
    where
        V: Value,
        O: 'static + Operation<Type = V::Type>,
        R: 'static
            + Fn(
                &[CustomRuleTracer<V, O>],
                &[MaybeZero<CustomRuleTracer<V, O>>],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>), ProgramError>
            + Send
            + Sync,
    {
        self.jvp = Some(CustomRuleJvp::SymbolicZeroRule(Arc::new(rule)));
        self
    }

    /// Returns this [`CustomRuleDefinition`] with a forward-mode rule derived from the primal region of each call,
    /// replacing any previously configured forward-mode rule. This is how a rule set that only customizes reverse mode
    /// (or batching) keeps ordinary forward-mode differentiation, which a missing rule never provides implicitly:
    /// forward mode then differentiates the primal region, reusable linearization partitions that derivative once, and
    /// reverse mode transposes it unless reverse-mode rules take precedence.
    pub fn with_jvp_from_primal(mut self) -> Self {
        self.jvp = Some(CustomRuleJvp::Primal);
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided reverse-mode forward rule `(p, x) ↦ (y, r)` and pure
    /// backward rule `(p, r, ȳ) ↦ x̄`. The forward rule receives every primal input and returns the primal outputs
    /// followed by the residuals `r`. The backward rule receives the known leading inputs (i.e., the non-differentiated
    /// inputs `p` followed by the residuals) and one cotangent seed per output, in which structural zeros are
    /// materialized, and it returns one cotangent per differentiated input, which transposition adds to whatever
    /// destination that input's cotangent has.
    pub fn with_vjp<F, B>(mut self, forward: F, backward: B) -> Self
    where
        V: 'static + Value<Type: DifferentiableType + ReferenceMemberType>,
        O: 'static
            + TransposableOperation<V, O>
            + ResidualZeroProvider<V::Type, Operation = O>
            + ReferenceAccessOperation<
                Transform: ReferenceTransform<Referent = <V::Type as ReferenceMemberType>::Referent>,
            >
            + OperationProvider<
                V::Type,
                ReferenceNewOperation<<V::Type as ReferenceMemberType>::Referent, V::Type>,
                Operation = O,
            >
            + OperationProvider<
                V::Type,
                ReferenceAddUpdateOperation<
                    <V::Type as ReferenceMemberType>::Referent,
                    V::Type,
                    <O as ReferenceAccessOperation>::Transform,
                >,
                Operation = O,
            >
            + From<AddOperation<V::Type>>,
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
    {
        self.forward = Some(Arc::new(forward));
        self.backward = Some(CustomVjpBackward::Pure(Arc::new(backward)));
        self.transposer = Some(Arc::new(|program, input_indices, destination_kinds| {
            program.transpose_with_respect_to(input_indices, destination_kinds)
        }));
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided reverse-mode forward rule `(p, x) ↦ (y, r)` and pure
    /// backward rule `(p, r, ȳ) ↦ x̄`. The rules are used as the rules of [`Self::with_vjp`], except that the backward
    /// rule receives each structural-zero seed as a [`MaybeZero::Zero`] rather than as a materialized zero.
    pub fn with_symbolic_zero_vjp<F, B>(mut self, forward: F, backward: B) -> Self
    where
        V: 'static + Value<Type: DifferentiableType + ReferenceMemberType>,
        O: 'static
            + TransposableOperation<V, O>
            + ResidualZeroProvider<V::Type, Operation = O>
            + ReferenceAccessOperation<
                Transform: ReferenceTransform<Referent = <V::Type as ReferenceMemberType>::Referent>,
            >
            + OperationProvider<
                V::Type,
                ReferenceNewOperation<<V::Type as ReferenceMemberType>::Referent, V::Type>,
                Operation = O,
            >
            + OperationProvider<
                V::Type,
                ReferenceAddUpdateOperation<
                    <V::Type as ReferenceMemberType>::Referent,
                    V::Type,
                    <O as ReferenceAccessOperation>::Transform,
                >,
                Operation = O,
            >
            + From<AddOperation<V::Type>>,
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
            ) -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError>
            + Send
            + Sync,
    {
        self.forward = Some(Arc::new(forward));
        self.backward = Some(CustomVjpBackward::PureWithSymbolicZeros(Arc::new(backward)));
        self.transposer = Some(Arc::new(|program, input_indices, destination_kinds| {
            program.transpose_with_respect_to(input_indices, destination_kinds)
        }));
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided reverse-mode forward rule `(p, x) ↦ (y, r)` and
    /// accumulating backward rule. The forward rule is used as the forward rule of [`Self::with_vjp`]. The backward
    /// rule is invoked while its carrier is transposed, with the [`TranspositionContext`], the carrier's inputs (i.e.,
    /// the known leading inputs followed by the differentiated inputs' tangents), the output cotangent seeds, and one
    /// [`CotangentAccumulator`] per carrier input, and it submits the contributions of the differentiated inputs to
    /// their accumulators. A single rule therefore serves returned, caller-buffer, and ignored destinations, and it
    /// can, for example, update only the affected entries of a caller-provided buffer instead of returning a full-sized
    /// cotangent.
    pub fn with_accumulating_vjp<F, B>(mut self, forward: F, backward: B) -> Self
    where
        V: 'static + Value<Type: DifferentiableType + ReferenceMemberType>,
        O: 'static
            + TransposableOperation<V, O>
            + ResidualZeroProvider<V::Type, Operation = O>
            + ReferenceAccessOperation<
                Transform: ReferenceTransform<Referent = <V::Type as ReferenceMemberType>::Referent>,
            >
            + OperationProvider<
                V::Type,
                ReferenceNewOperation<<V::Type as ReferenceMemberType>::Referent, V::Type>,
                Operation = O,
            >
            + OperationProvider<
                V::Type,
                ReferenceAddUpdateOperation<
                    <V::Type as ReferenceMemberType>::Referent,
                    V::Type,
                    <O as ReferenceAccessOperation>::Transform,
                >,
                Operation = O,
            >
            + From<AddOperation<V::Type>>,
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
            ) -> Result<(), DifferentiationError>
            + Send
            + Sync,
    {
        self.forward = Some(Arc::new(forward));
        self.backward = Some(CustomVjpBackward::Accumulating(Arc::new(backward)));
        self.transposer = Some(Arc::new(|program, input_indices, destination_kinds| {
            program.transpose_with_respect_to(input_indices, destination_kinds)
        }));
        self
    }

    /// Returns this [`CustomRuleDefinition`] with support for differentiating batched calls. Batching a call records
    /// its batching level without tracing any rule, and a later derivative request traces the rule at the unbatched
    /// types (sharing that specialization with unbatched calls) and structurally batches the traced program once per
    /// recorded level. Without this, batched calls still execute their primal but reject differentiation.
    pub fn with_batching(mut self) -> Self
    where
        V: 'static + Value<Type: DifferentiableType + BatchableType>,
        O: 'static + Operation<Type = V::Type>,
        <V::Type as BatchableType>::Policy:
            RecursiveBatchingPolicy<TracingContext<V, O>> + CotangentBatchingPolicy<TracingContext<V, O>>,
    {
        self.batcher = Some(Arc::new(|level, program, input_axes, output_axes, outputs| {
            type Policy<T> = <T as BatchableType>::Policy;

            // Align every required output first (a replicated alignment request keeps the natural axis), then adapt
            // the policy boundary, collapsing any output that is still mapped while its requirement is replicated.
            let batched = <Policy<V::Type> as RecursiveBatchingPolicy<TracingContext<V, O>>>::batch_program_at_level(
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
            let adapted = <Policy<V::Type> as BatchingPolicy<TracingContext<V, O>>>::adapt_batched_program(
                batched,
                Some(required_output_axes.as_slice()),
                |context, output, axis| match outputs {
                    CustomRuleBatchedOutputs::Values => Err(BatchingError::InvalidBatchMetadata {
                        message: format!(
                            "a custom rule output is mapped along axis {axis} but its batched layout requires it to \
                             be replicated",
                        ),
                    }),
                    CustomRuleBatchedOutputs::Cotangents => {
                        <Policy<V::Type> as CotangentBatchingPolicy<TracingContext<V, O>>>::sum_mapped_cotangents(
                            context, output, axis,
                        )
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
    pub fn with_batched_output_axes(mut self, output_axes: Vec<BatchAxis>) -> Self {
        self.batched_output_axes = Some(output_axes);
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided custom batching rule, which batches its calls instead of
    /// structurally batching their primal regions (i.e., the analogue of JAX's `custom_vmap`). The rule receives the
    /// batching level, the level's boundary inputs (e.g., a first-class batch extent), the call's batched inputs, and
    /// their batch axes, and it returns the batched outputs together with their batch axes. It is traced when a call is
    /// batched, once per batching level and input signature, and its program becomes the batched call's primal region,
    /// so it also determines the batch axes of the call's outputs (instead of [`Self::with_batched_output_axes`]).
    ///
    /// Batching a call with this rule preserves the call and its derivative rules: derivative requests of a batched
    /// call still trace the derivative rules at the unbatched types and batch them structurally (refer to
    /// [`Self::with_batching`]), aligning their outputs to the batch axes that the rule declared, while a forward-mode
    /// rule derived from the primal differentiates the rule's batched program. The rule is responsible for computing
    /// the batched primal consistently with those derivative rules.
    ///
    /// When the forward-mode rule is derived from the primal, differentiating an unbatched call stages a derived call
    /// whose batching applies the derivative of this rule (refer to [`CustomRuleSpecializer::derived`]), which may
    /// apply the rule with every input replicated. This function therefore also captures the family's differentiation
    /// of traced programs, which is why it requires the family's differentiation bounds.
    pub fn with_batching_rule<R>(mut self, rule: R) -> Self
    where
        V: 'static + Value<Type: DifferentiableType>,
        O: 'static
            + Operation<Type = V::Type>
            + PartiallyEvaluatableOperation<TracingContext<V, O>>
            + DifferentiableOperation<TracingContext<V, O>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
            + ResidualZeroProvider<V::Type, Operation = O>,
        R: 'static
            + Fn(
                &BatchingLevel<V::Type>,
                &[CustomRuleTracer<V, O>],
                &[CustomRuleTracer<V, O>],
                &[BatchAxis],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError>
            + Send
            + Sync,
    {
        self.batching_rule = Some(Arc::new(rule));
        self.differentiator = Some(Arc::new(|program, input_indices| program.jvp_with_respect_to(input_indices)));
        self
    }

    /// Returns this [`CustomRuleDefinition`] with the provided custom batching rule, which is used as the rule of
    /// [`Self::with_batching_rule`] except that its result may depend on its input batch axes beyond computing the
    /// batched primal for them (e.g., a rule that applies a different approximation when an input is replicated). A
    /// batching rule that only computes the batched primal gives the same values whether a replicated input is
    /// broadcast or not, so the derived batching rules of a call whose forward-mode rule is derived from its primal
    /// apply it with a replicated primal broadcast when that primal's tangent is mapped. For an axis-dependent rule,
    /// they instead apply it once, at the primal inputs' own batch axes, and recover the per-item tangents of such
    /// inputs from the outer product of their mapped tangents with the rule's batch (i.e., by batching the rule's
    /// derivative with respect to them over a second level of the same extent and taking the diagonal of the two batch
    /// axes; refer to [`DiagonalBatchingPolicy`]). This costs as many applications of that derivative as the batch has
    /// items, which is why it is not the default (refer to [`CustomRuleSpecializer::derived`]).
    pub fn with_axis_dependent_batching_rule<R>(self, rule: R) -> Self
    where
        V: 'static
            + Value<Type: DifferentiableType + BatchableType<Policy: DiagonalBatchingPolicy<TracingContext<V, O>>>>,
        O: 'static
            + Operation<Type = V::Type>
            + PartiallyEvaluatableOperation<TracingContext<V, O>>
            + DifferentiableOperation<TracingContext<V, O>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
            + ResidualZeroProvider<V::Type, Operation = O>
            + From<AddOperation<V::Type>>,
        R: 'static
            + Fn(
                &BatchingLevel<V::Type>,
                &[CustomRuleTracer<V, O>],
                &[CustomRuleTracer<V, O>],
                &[BatchAxis],
            ) -> Result<(Vec<CustomRuleTracer<V, O>>, Vec<BatchAxis>), ProgramError>
            + Send
            + Sync,
    {
        let mut definition = self.with_batching_rule(rule);
        definition.axis_dependence = Some(Arc::new(PolicyAxisDependence(PhantomData)));
        definition
    }

    /// Returns this [`CustomRuleDefinition`] with support for differentiating calls whose reference state was
    /// discharged (refer to [`Program::discharge_references`]). Discharging a call discharges the local reference state
    /// of its primal region immediately, and each rule program traced for that call later is discharged before it is
    /// cached or replayed, so that the call's derivatives contain no reference state either. Without this support, a
    /// discharged call whose traced rule programs contain reference state rejects differentiation.
    pub fn with_reference_discharge<P>(mut self) -> Self
    where
        V: 'static
            + Value<Type: ReferenceDischargeableType<Policy = P> + From<P::Referent> + From<ReferenceType<P::Referent>>>,
        O: 'static + Operation<Type = V::Type> + ReferenceDischargeableOperation<TracingContext<V, O>, P>,
        P: ReferenceDischargePolicy<TracingContext<V, O>>,
        for<'t> &'t ReferenceType<P::Referent>: TryFrom<&'t V::Type>,
    {
        self.discharger = Some(Arc::new(|program: Program<V, O, Vec<V>, Vec<V>>| {
            program.discharge_references::<P>(0)?.into_program_without_external_references()
        }));
        self
    }

    /// Returns the human-readable label of this [`CustomRuleDefinition`].
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the named axes with which the rules of this [`CustomRuleDefinition`] are traced.
    #[inline]
    pub fn named_axes(&self) -> &[(String, NamedAxis)] {
        &self.named_axes
    }

    /// Returns `true` if this [`CustomRuleDefinition`] has a forward-mode rule or reverse-mode rules.
    #[inline]
    pub(super) fn has_derivative_rules(&self) -> bool {
        self.jvp.is_some() || self.forward.is_some()
    }

    /// Returns `program` discharged when `discharged` is set (refer to [`Self::with_reference_discharge`]). A program
    /// without reference state needs no discharge, which also makes definitions without discharge support usable for
    /// discharged calls whose rules keep no reference state.
    fn discharge_if(
        &self,
        discharged: bool,
        program: Program<V, O, Vec<V>, Vec<V>>,
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
    where
        V: Value,
        O: Operation<Type = V::Type>,
    {
        let has_reference_state =
            program.regions().iter().flat_map(|region| region.atoms()).any(|atom| atom.r#type().is_reference());
        if !discharged || !has_reference_state {
            return Ok(program);
        }
        let Some(discharger) = &self.discharger else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` cannot differentiate a call whose reference state was \
                     discharged, because its rule programs contain reference state and its definition has no \
                     reference discharge support",
                    self.name(),
                ),
            }
            .into());
        };
        Ok(discharger(program)?)
    }

    /// Produces the specialization for `request` with `produce_fn` without caching it, which is how specializations are
    /// produced once the registration handle, and with it the caches, was dropped. A recursive request for the same
    /// specialization on the same thread is rejected with the [`ReentrantSpecializationError`] that a cache reports.
    fn produce_uncached<F: FnOnce() -> Result<CustomRuleProgram<V, O>, DifferentiationError>>(
        &self,
        request: CustomRuleSpecializationRequest<V::Type>,
        produce_fn: F,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError> {
        /// Removes an in-flight marker when production finishes, fails, or unwinds.
        struct InFlightMarker<'d, T: Type + Eq + Hash> {
            /// Set from which the marker is removed.
            in_flight: &'d Mutex<HashSet<(ThreadId, CustomRuleSpecializationRequest<T>)>>,

            /// Marker to remove.
            marker: (ThreadId, CustomRuleSpecializationRequest<T>),
        }

        impl<T: Type + Eq + Hash> Drop for InFlightMarker<'_, T> {
            fn drop(&mut self) {
                self.in_flight.lock().expect("custom rule in-flight mutex is poisoned").remove(&self.marker);
            }
        }

        let marker = (std::thread::current().id(), request);
        if !self
            .uncached_in_flight
            .lock()
            .expect("custom rule in-flight mutex is poisoned")
            .insert(marker.clone())
        {
            return Err(specialization_error(SpecializationCacheError::<DifferentiationError>::Reentrant(
                ReentrantSpecializationError,
            )));
        }
        let _marker = InFlightMarker { in_flight: &self.uncached_in_flight, marker };
        produce_fn()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for CustomRuleDefinition<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CustomRuleDefinition")
            .field("name", &self.name)
            .field("has_jvp", &matches!(self.jvp, Some(CustomRuleJvp::Rule(_) | CustomRuleJvp::SymbolicZeroRule(_))))
            .field("derives_jvp_from_primal", &matches!(self.jvp, Some(CustomRuleJvp::Primal)))
            .field("has_vjp", &self.backward.is_some())
            .finish_non_exhaustive()
    }
}

/// Bounded specialization caches of one registered [`CustomRuleDefinition`], keyed by static information only and
/// owned by its [`CustomRuleRegistration`] handle (refer to the Ownership section of [`CustomRuleDefinition`]).
pub(super) struct CustomRuleCaches<V: Typed + Parameter, O> {
    /// Specializations of the forward-mode rule.
    pub(super) jvp_specializations: SpecializationCache<CustomRuleSpecializationKey<V::Type>, CustomRuleProgram<V, O>>,

    /// Specializations of the reverse-mode forward rule.
    pub(super) forward_specializations:
        SpecializationCache<CustomRuleSpecializationKey<V::Type>, CustomRuleProgram<V, O>>,

    /// Destination-specialized transposes of the reverse-mode backward rule.
    pub(super) backward_specializations:
        SpecializationCache<CustomRuleBackwardSpecializationKey<V::Type>, CustomRuleProgram<V, O>>,

    /// Batched primal programs produced by the custom batching rule.
    pub(super) batching_rule_specializations:
        SpecializationCache<CustomRuleBatchingRuleKey<V::Type>, CustomRuleProgram<V, O>>,

    /// Registrations of the definitions derived from this definition (refer to [`CustomRuleDerivation`]), which these
    /// caches own so that they live exactly as long as the specializations of this definition.
    pub(super) derived_registrations: Mutex<HashMap<CustomRuleDerivationKey, CustomRuleRegistration<V, O>>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleCaches<V, O> {
    /// Creates empty [`CustomRuleCaches`] that retain up to [`CUSTOM_RULE_SPECIALIZATION_CAPACITY`] specializations of
    /// each rule.
    fn new() -> Self {
        Self {
            jvp_specializations: SpecializationCache::new(CUSTOM_RULE_SPECIALIZATION_CAPACITY),
            forward_specializations: SpecializationCache::new(CUSTOM_RULE_SPECIALIZATION_CAPACITY),
            backward_specializations: SpecializationCache::new(CUSTOM_RULE_SPECIALIZATION_CAPACITY),
            batching_rule_specializations: SpecializationCache::new(CUSTOM_RULE_SPECIALIZATION_CAPACITY),
            derived_registrations: Mutex::new(HashMap::new()),
        }
    }
}

/// Handle to a registered [`CustomRuleDefinition`], which owns the definition's specialization caches. Calls are staged
/// with a [`CustomRuleReference`] obtained from [`Self::reference`], and the specializations that their derivative
/// requests produce are cached for as long as some handle to the registration is alive. Once every handle is dropped,
/// the caches are freed, and later derivative requests of the remaining calls trace their rules again on every request
/// (enclosing region transform caches still apply).
///
/// # Referring to Definitions from Rules
///
/// A rule that calls a definition must capture a [`CustomRuleReference`] (or, for its own definition, the
/// [`WeakCustomRuleRegistration`] passed to [`Self::new_cyclic`]), never a handle, because a rule that owned the caches
/// of a definition whose programs reach the rule would form a reference cycle. Handles are deliberately not [`Sync`],
/// so the compiler rejects a rule, which must be [`Send`] and [`Sync`], that captures one. References to other
/// definitions cannot form a cycle either, because a definition can only capture definitions that were registered
/// before it, except for itself through [`Self::new_cyclic`]. For example, two mutually recursive definitions are
/// registered by building the second one inside the [`Self::new_cyclic`] closure of the first one, capturing the weak
/// handle of the first one in the rules of the second one.
pub struct CustomRuleRegistration<V: Typed + Parameter, O> {
    /// Registered definition.
    definition: Arc<CustomRuleDefinition<V, O>>,

    /// Specialization caches of the definition, which only handles hold strongly.
    caches: Arc<CustomRuleCaches<V, O>>,

    /// Marker that keeps this handle from being [`Sync`] (refer to the documentation of [`CustomRuleRegistration`]).
    marker: PhantomData<Cell<()>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleRegistration<V, O> {
    /// Registers the provided definition with empty specialization caches.
    pub fn new(definition: CustomRuleDefinition<V, O>) -> Self {
        Self { definition: Arc::new(definition), caches: Arc::new(CustomRuleCaches::new()), marker: PhantomData }
    }

    /// Registers the definition returned by `build_fn`, which receives a weak handle to that same registration so that
    /// the definition's rules can call it (i.e., recursively) without keeping it alive.
    pub fn new_cyclic<F: FnOnce(&WeakCustomRuleRegistration<V, O>) -> CustomRuleDefinition<V, O>>(build_fn: F) -> Self {
        let caches = Arc::new(CustomRuleCaches::new());
        let definition = Arc::new_cyclic(|definition| {
            build_fn(&WeakCustomRuleRegistration { definition: definition.clone(), caches: Arc::downgrade(&caches) })
        });
        Self { definition, caches, marker: PhantomData }
    }

    /// Returns a [`CustomRuleReference`] to this registration, with which calls are staged and rules refer to it.
    #[inline]
    pub fn reference(&self) -> CustomRuleReference<V, O> {
        CustomRuleReference { definition: self.definition.clone(), caches: Arc::downgrade(&self.caches) }
    }
}

#[cfg(test)]
impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleDefinition<V, O> {
    /// Returns whether any thread is producing an uncached specialization of this definition, which tests use to
    /// observe that a failed or reentrant request leaves no production behind.
    pub(super) fn has_uncached_in_flight(&self) -> bool {
        !self.uncached_in_flight.lock().expect("custom rule in-flight mutex is poisoned").is_empty()
    }
}

#[cfg(test)]
impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleRegistration<V, O> {
    /// Returns the registered definition, which tests use to observe its ownership.
    pub(super) fn definition(&self) -> &Arc<CustomRuleDefinition<V, O>> {
        &self.definition
    }

    /// Returns the specialization caches, which tests use to observe lazy tracing and cache ownership.
    pub(super) fn caches(&self) -> &Arc<CustomRuleCaches<V, O>> {
        &self.caches
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Clone for CustomRuleRegistration<V, O> {
    fn clone(&self) -> Self {
        Self { definition: self.definition.clone(), caches: self.caches.clone(), marker: PhantomData }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for CustomRuleRegistration<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("CustomRuleRegistration").field(&self.definition).finish()
    }
}

/// Reference to a registered [`CustomRuleDefinition`], with which calls are staged and rules refer to definitions. It
/// keeps the definition alive and uses the registration's specialization caches while some
/// [`CustomRuleRegistration`] handle keeps them alive, producing uncached specializations otherwise. Equality and
/// hashing use the identity of the definition.
pub struct CustomRuleReference<V: Typed + Parameter, O> {
    /// Referenced definition.
    definition: Arc<CustomRuleDefinition<V, O>>,

    /// Specialization caches of the definition, which are alive exactly while some handle to it is alive.
    caches: Weak<CustomRuleCaches<V, O>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleReference<V, O> {
    /// Returns the referenced definition.
    #[inline]
    pub fn definition(&self) -> &CustomRuleDefinition<V, O> {
        &self.definition
    }

    /// Returns the specialization for `request`, produced by `produce_fn` on a miss. The specialization is retained in
    /// the registration's caches while they are alive, and produced without caching otherwise.
    fn specialize<F: FnOnce() -> Result<CustomRuleProgram<V, O>, DifferentiationError>>(
        &self,
        request: CustomRuleSpecializationRequest<V::Type>,
        produce_fn: F,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError> {
        let Some(caches) = self.caches.upgrade() else {
            return self.definition.produce_uncached(request, produce_fn);
        };
        match request {
            CustomRuleSpecializationRequest::Jvp(key) => {
                caches.jvp_specializations.get_or_try_insert_with(key, produce_fn)
            }
            CustomRuleSpecializationRequest::Forward(key) => {
                caches.forward_specializations.get_or_try_insert_with(key, produce_fn)
            }
            CustomRuleSpecializationRequest::Backward(key) => {
                caches.backward_specializations.get_or_try_insert_with(key, produce_fn)
            }
            CustomRuleSpecializationRequest::BatchingRule(key) => {
                caches.batching_rule_specializations.get_or_try_insert_with(key, produce_fn)
            }
        }
        .map_err(specialization_error)
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Clone for CustomRuleReference<V, O> {
    fn clone(&self) -> Self {
        Self { definition: self.definition.clone(), caches: self.caches.clone() }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for CustomRuleReference<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("CustomRuleReference").field(&self.definition).finish()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> PartialEq for CustomRuleReference<V, O> {
    fn eq(&self, other: &Self) -> bool {
        self.definition.id == other.definition.id
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Eq for CustomRuleReference<V, O> {}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Hash for CustomRuleReference<V, O> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.definition.id.hash(state);
    }
}

/// Weak handle to a registered [`CustomRuleDefinition`], which keeps neither the definition nor its caches alive. It is
/// how a definition's rules refer to that same definition (refer to [`CustomRuleRegistration::new_cyclic`]).
pub struct WeakCustomRuleRegistration<V: Typed + Parameter, O> {
    /// Registered definition.
    definition: Weak<CustomRuleDefinition<V, O>>,

    /// Specialization caches of the definition.
    caches: Weak<CustomRuleCaches<V, O>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> WeakCustomRuleRegistration<V, O> {
    /// Returns a [`CustomRuleReference`] to the registration, with which the rules of that same definition stage its
    /// calls. A rule is only invoked while its definition is alive, so this fails only when the reference is used
    /// outside a rule after the definition was dropped.
    pub fn reference(&self) -> Result<CustomRuleReference<V, O>, ProgramError> {
        let definition = self.definition.upgrade().ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("the `{CUSTOM_FUNCTION_OPERATION_NAME}` definition was dropped"),
        })?;
        Ok(CustomRuleReference { definition, caches: self.caches.clone() })
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Clone for WeakCustomRuleRegistration<V, O> {
    fn clone(&self) -> Self {
        Self { definition: self.definition.clone(), caches: self.caches.clone() }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for WeakCustomRuleRegistration<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("WeakCustomRuleRegistration").finish_non_exhaustive()
    }
}

/// Seals [`CustomRuleSource`] and [`CustomRuleSpecializer`] (refer to the documentation of [`CustomRuleSource`]).
mod sealed {
    /// Supertrait of [`CustomRuleSource`](super::CustomRuleSource) that only this module can implement.
    pub trait Sealed {}
}

impl<V: Typed + Parameter, O> sealed::Sealed for CustomRuleReference<V, O> {}

impl<Vm: Typed + Parameter, Om> sealed::Sealed for LiftedCustomRules<Vm, Om> {}

/// Source of the retained rules that a [`CustomFunctionOperation`](crate::CustomFunctionOperation) and its
/// [`CustomFunctionTransposeOperation`] apply, which provides their specializations as programs of the operations'
/// own family `(V, O)`. [`CustomRuleReference`] is the source of a definition registered in that family, and
/// [`LiftedCustomRules`] is the source of a call that was converted into that family from a member family (e.g., from
/// array programs into array IR programs), whose specializations the member definition produces and caches before they
/// are converted. Equality and hashing use the identity of the source's definition.
///
/// This trait describes the rules only, which is all that every transform except differentiation needs, and
/// [`CustomRuleSpecializer`] adds the production of their specializations. Keeping the two apart keeps the heavier
/// bounds of converted sources (e.g., on the member family's operations) out of the [`Operation`] implementation of the
/// operations, where they would form a cycle with the payload bounds of the families that contain them.
///
/// This trait and [`CustomRuleSpecializer`] are sealed: their only implementations are [`CustomRuleReference`] and
/// [`LiftedCustomRules`]. Their specialization requests and results are internal to this module (their keys have no
/// public constructors or accessors), so a source outside of it could only delegate to one of these two. New kinds of
/// rules are registered through [`CustomRuleDefinition`] instead.
pub trait CustomRuleSource<V: Typed<Type: Eq + Hash> + Parameter, O>:
    sealed::Sealed + Clone + Debug + Eq + Hash
{
    /// Returns the process-unique identity of the source's definition.
    fn id(&self) -> u64;

    /// Returns the human-readable label of the source's definition.
    fn name(&self) -> &str;

    /// Returns whether the source has a retained forward-mode (i.e., JVP) rule.
    fn has_jvp(&self) -> bool;

    /// Returns whether the source derives its forward-mode rule from the primal region of each call (refer to
    /// [`CustomRuleDefinition::with_jvp_from_primal`]).
    fn derives_jvp_from_primal(&self) -> bool;

    /// Returns whether the source has reverse-mode (i.e., VJP) forward and backward rules.
    fn has_vjp(&self) -> bool;

    /// Returns the declared batch axes of the outputs of batched calls (refer to
    /// [`CustomRuleDefinition::with_batched_output_axes`]), or [`None`] when they are not declared.
    fn batched_output_axes(&self) -> Option<&[BatchAxis]>;

    /// Returns whether the source has a custom batching rule (refer to [`CustomRuleDefinition::with_batching_rule`]),
    /// which takes precedence over structurally batching the primal region of its calls.
    fn has_batching_rule(&self) -> bool;

    /// Returns the definition whose rules the source applies directly, or [`None`] when the source obtains its
    /// specializations elsewhere (e.g., from a member definition).
    fn native_definition(&self) -> Option<&CustomRuleDefinition<V, O>>;

    /// Returns whether backward specializations of the source can write cotangents to caller-provided buffers. A
    /// source that cannot returns those cotangents instead, and the carrier's accumulators add them to the buffers.
    fn supports_reference_destinations(&self) -> bool;

    /// Returns the batch axes of the outputs of batched calls that have `output_count` outputs, which map every output
    /// at axis 0 unless the source declares them.
    fn batched_call_output_axes(&self, output_count: usize) -> Result<Vec<BatchAxis>, TypeError> {
        let Some(output_axes) = self.batched_output_axes() else {
            return Ok(vec![BatchAxis::new(0); output_count]);
        };
        check_count!("batched output axis", output_axes, output_count, TypeError);
        Ok(output_axes.to_vec())
    }
}

/// [`CustomRuleSource`] that produces the specializations of its rules, as programs of the operations' own family
/// `(V, O)`. The functions that produce specializations carry their bounds themselves rather than on the trait, like
/// the rule traits, which keeps the operations free of bounds on `O` that would form a cycle with the payload bounds of
/// the families that contain them. It is sealed (refer to the documentation of [`CustomRuleSource`]).
pub trait CustomRuleSpecializer<V: Typed<Type: Eq + Hash> + Parameter, O>: CustomRuleSource<V, O> {
    /// Returns the forward-mode rule specialization for `key`, whose program takes the call's inputs followed by the
    /// active tangents of its differentiated inputs (refer to the tangent activity of the key) and returns the call's
    /// outputs followed by their tangents.
    fn jvp_specialization(
        &self,
        key: CustomRuleSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type, Operation = O>;

    /// Returns the reverse-mode forward rule specialization for `key`, whose program takes the call's inputs and
    /// returns the call's outputs followed by the residuals.
    fn forward_specialization(
        &self,
        key: CustomRuleSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>;

    /// Returns the destination-specialized backward rule for `key`, whose program consumes the boundary inputs, then
    /// the live seeds, then the caller buffers, then the remaining known leading inputs, and returns the cotangents of
    /// the differentiated inputs with returned destinations.
    fn backward_specialization(
        &self,
        key: CustomRuleBackwardSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + From<CustomFunctionTransposeOperation<V, O>>;

    /// Returns the batched primal program that the custom batching rule produces for `key`, whose program takes the
    /// level's boundary inputs followed by the call's batched inputs and returns the batched outputs, together with the
    /// batch axes of those outputs.
    fn batching_rule_specialization(
        &self,
        key: CustomRuleBatchingRuleKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>;

    /// Returns the rules of the definition derived from this source for `key`, with which forward-mode differentiation
    /// stages the derived call of a call whose forward-mode rule is derived from its primal and whose source has a
    /// custom batching rule. This keeps that rule on the path of forward-mode derivatives that are batched after they
    /// are taken (i.e., the analogue of JAX's `custom_vmap_jvp`): the derived call's primal region is the derivative
    /// of the call's primal region (i.e., its Jacobian-Vector Product (JVP), or the tangent half of it), and the
    /// derived definition holds no program. It derives its forward-mode rule from each call's primal region, so that
    /// higher orders recurse through each call's own region, and its batching rule applies the JVP of the source's
    /// batching rule. Calls whose primal regions differ therefore share one derived definition.
    ///
    /// The derived batching rule always computes the derived call's primal outputs with the source rule applied at the
    /// batch axes of the primal inputs, which is exactly how batching the source call computes them, so a rule whose
    /// result depends on its input axes gives a derivative whose primal outputs are those of the batched call. Its
    /// tangents are computed in one of two modes:
    ///
    ///   - **Structural:** when every primal input is replicated, and so is every output of the rule applied to them
    ///     (e.g., the replicated primals and mapped tangents of a forward-mode Jacobian), the JVP of the rule's program
    ///     is batched structurally over the given batch axes.
    ///   - **Rule:** otherwise, each primal and tangent pair with a mapped side is aligned to one axis (i.e., its
    ///     replicated side is broadcast, and a tangent mapped along another axis than its primal is moved), and the JVP
    ///     of the source rule applied at the aligned batch axes computes the tangents. Inputs without a tangent keep
    ///     their batch axes. Alignment changes the rule's input axes only for a replicated primal whose tangent is
    ///     mapped, because the rule cannot apply per-item perturbations of a shared input otherwise; only then is the
    ///     rule applied twice (i.e., once at the primal axes for the primal outputs and once at the aligned axes for
    ///     the tangents). When the source's rule depends on its input batch axes (refer to
    ///     [`CustomRuleDefinition::with_axis_dependent_batching_rule`]), the rule is instead applied once, at the
    ///     primal axes, and the tangents of such primals are recovered exactly from an outer product (refer to
    ///     [`DiagonalBatchingPolicy`]).
    ///
    /// Derived definitions are owned by the source's registration, like its specializations: every request for the
    /// same key returns the same derived definition while the registration's caches are alive, and a fresh one
    /// otherwise.
    fn derived(&self, key: CustomRuleDerivationKey) -> Result<Self, DifferentiationError>
    where
        Self: Sized;
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleSource<V, O> for CustomRuleReference<V, O> {
    #[inline]
    fn id(&self) -> u64 {
        self.definition.id
    }

    #[inline]
    fn name(&self) -> &str {
        self.definition.name()
    }

    #[inline]
    fn has_jvp(&self) -> bool {
        matches!(self.definition.jvp, Some(CustomRuleJvp::Rule(_) | CustomRuleJvp::SymbolicZeroRule(_)))
    }

    #[inline]
    fn derives_jvp_from_primal(&self) -> bool {
        matches!(self.definition.jvp, Some(CustomRuleJvp::Primal))
    }

    #[inline]
    fn has_vjp(&self) -> bool {
        self.definition.backward.is_some()
    }

    #[inline]
    fn batched_output_axes(&self) -> Option<&[BatchAxis]> {
        self.definition.batched_output_axes.as_deref()
    }

    #[inline]
    fn has_batching_rule(&self) -> bool {
        self.definition.batching_rule.is_some() || self.definition.derivation.is_some()
    }

    #[inline]
    fn native_definition(&self) -> Option<&CustomRuleDefinition<V, O>> {
        Some(&self.definition)
    }

    #[inline]
    fn supports_reference_destinations(&self) -> bool {
        true
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleSpecializer<V, O> for CustomRuleReference<V, O> {
    fn jvp_specialization(
        &self,
        key: CustomRuleSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type, Operation = O>,
    {
        // Returns the JVP rule specialization for `key`, whose program takes the call's inputs followed by the active
        // tangents of its differentiated inputs and returns the call's outputs followed by their tangents. An unbatched
        // key traces the rule on its first request, while a batched key batches the specialization of its inner key at
        // its outermost level, so that every batching level shares one trace of the rule.
        //
        // The traced rule is validated against the unbatched output types of `key`.
        let rule = match &self.definition.jvp {
            Some(CustomRuleJvp::Rule(rule)) => CustomRuleJvpRule::Materialized(rule),
            Some(CustomRuleJvp::SymbolicZeroRule(rule)) => CustomRuleJvpRule::SymbolicZeros(rule),
            _ => {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` has no forward-mode (i.e., JVP) rule",
                        self.definition.name()
                    ),
                }
                .into());
            }
        };
        let inner = match key.split_outermost_level() {
            Some((inner_key, level)) => Some((self.jvp_specialization(inner_key)?, level.clone())),
            None => None,
        };
        self.specialize(CustomRuleSpecializationRequest::Jvp(key.clone()), || {
            let Some((inner, level)) = inner else {
                // The traced program takes every primal input followed by the active tangents of the differentiated
                // inputs and returns the primal outputs followed by their tangents, which is exactly the rule program
                // that `replay_custom_jvp_rule` replays. The structural zeros are materialized inside the program, from
                // their own primals (which name every runtime quantity that their tangent types omit), so that the
                // zeros that the rule propagates remain recognizable as zeros when the program is replayed.
                let differentiated_input_types = &key.input_types[key.non_differentiated_count..];
                if key.tangent_activity.len() != differentiated_input_types.len() {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` JVP specialization key has {} tangent activity \
                         flags for {} differentiated inputs",
                        self.definition.name(),
                        key.tangent_activity.len(),
                        differentiated_input_types.len(),
                    ))
                    .into());
                }
                let mut rule_input_types = key.input_types.clone();
                for (r#type, active) in differentiated_input_types.iter().zip(&key.tangent_activity) {
                    if *active {
                        rule_input_types.push(r#type.tangent()?);
                    }
                }
                // The rule's two result partitions are validated separately before they are flattened, because
                // the flattened types alone cannot tell a missing output from a misplaced tangent.
                let output_count = key.output_types.len();
                let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
                    |values: Vec<CustomRuleTracer<V, O>>| {
                        let (primals, active_tangents) = values.split_at(key.input_types.len());
                        let mut active_tangents = active_tangents.iter().cloned();
                        let tangents = primals[key.non_differentiated_count..]
                            .iter()
                            .zip(&key.tangent_activity)
                            .map(|(primal, active)| match active {
                                true => Ok(MaybeZero::Value(active_tangents.next().unwrap())),
                                false => Ok(MaybeZero::Zero(primal.r#type().tangent()?)),
                            })
                            .collect::<Result<Vec<_>, ProgramError>>()?;
                        let (mut outputs, output_tangents) = match rule {
                            CustomRuleJvpRule::Materialized(rule) => {
                                let tangents = primals[key.non_differentiated_count..]
                                    .iter()
                                    .zip(tangents)
                                    .map(|(primal, tangent)| {
                                        O::materialize_zero_from_residual_sources(
                                            primal.context(),
                                            tangent,
                                            std::iter::once(primal),
                                        )
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                rule.apply(primals, &tangents)?
                            }
                            CustomRuleJvpRule::SymbolicZeros(rule) => rule.apply(primals, &tangents)?,
                        };
                        if outputs.len() != output_count || output_tangents.len() != output_count {
                            return Err(TypeError::invalid(format!(
                                "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` JVP rule returned {} outputs and {} output \
                                     tangents but the primal has {output_count} outputs",
                                self.definition.name(),
                                outputs.len(),
                                output_tangents.len(),
                            ))
                            .into());
                        }
                        outputs.extend(output_tangents);
                        Ok(outputs)
                    },
                    rule_input_types,
                    self.definition.named_axes.clone(),
                )?;
                let mut expected_output_types = key.output_types.clone();
                for r#type in &key.output_types {
                    expected_output_types.push(r#type.tangent()?);
                }
                let program =
                    conform_rule_output_types(self.definition.name(), "JVP", program, &expected_output_types)?;
                let program = self.definition.discharge_if(key.discharged, program)?;
                return Ok(Arc::new(CustomRuleSpecialization { program, output_axes: Vec::new() }));
            };

            // Each active tangent follows the batch axis of its primal, and each output tangent follows its output.
            let differentiated_input_start =
                key.non_differentiated_count + boundary_input_count(&key.levels[..key.levels.len() - 1]);
            let differentiated_input_axes = level.input_axes.get(differentiated_input_start..).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` batching level has too few input axes",
                    self.definition.name(),
                ))
            })?;
            let mut input_axes = level.input_axes.clone();
            input_axes.extend(
                differentiated_input_axes
                    .iter()
                    .zip(&key.tangent_activity)
                    .filter_map(|(axis, active)| active.then_some(*axis)),
            );
            let output_axes = level.output_axes.iter().chain(&level.output_axes).copied().map(Some).collect::<Vec<_>>();
            self.definition.batched_specialization(
                &level,
                &inner.program,
                &input_axes,
                &output_axes,
                CustomRuleBatchedOutputs::Values,
            )
        })
    }

    fn forward_specialization(
        &self,
        key: CustomRuleSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>,
    {
        // Returns the reverse-mode forward rule specialization for `key`, whose program takes the call's inputs and
        // returns the call's outputs followed by the residuals. Specializations are traced and batched as for
        // [`Self::jvp_specialization`], except that residuals keep the batch axes that batching naturally produces,
        // which the specialization records.
        //
        // The traced rule is validated against the unbatched output types of `key`.
        let Some(forward) = &self.definition.forward else {
            return Err(ProgramError::MalformedProgram(format!(
                "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` has no reverse-mode forward rule",
                self.definition.name(),
            ))
            .into());
        };
        let inner = match key.split_outermost_level() {
            Some((inner_key, level)) => Some((self.forward_specialization(inner_key)?, level.clone())),
            None => None,
        };
        let output_count = key.output_types.len();
        self.specialize(CustomRuleSpecializationRequest::Forward(key.clone()), || {
            let Some((inner, level)) = inner else {
                // The primal outputs are validated as their own partition before the residuals are appended,
                // because a residual whose type matches a missing output would otherwise pass as that output.
                let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
                    |values: Vec<CustomRuleTracer<V, O>>| {
                        let (mut outputs, residuals) = forward.apply(&values)?;
                        if outputs.len() != output_count {
                            return Err(TypeError::invalid(format!(
                                "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` forward rule returned {} outputs but the \
                                     primal has {output_count}",
                                self.definition.name(),
                                outputs.len(),
                            ))
                            .into());
                        }
                        outputs.extend(residuals);
                        Ok(outputs)
                    },
                    key.input_types.clone(),
                    self.definition.named_axes.clone(),
                )?;
                let program = conform_rule_output_types(self.definition.name(), "forward", program, &key.output_types)?;
                // Reference residuals preserve handles, not snapshots, so each one must forward a distinct leading
                // non-differentiated input by identity, which the backward rule then receives as plumbing.
                let leading_inputs = &program.input_ids()[..key.non_differentiated_count];
                let mut forwarded_inputs = HashSet::new();
                for (index, residual) in program.output_ids().iter().skip(output_count).enumerate() {
                    let r#type = program.atoms()[residual.index()].r#type();
                    if !r#type.is_reference() {
                        continue;
                    }
                    if !leading_inputs.contains(residual) {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` forward rule returns residual {index} of \
                             reference type `{type}` that is not a leading non-differentiated input forwarded by \
                             identity",
                            self.definition.name(),
                        ))
                        .into());
                    }
                    if !forwarded_inputs.insert(*residual) {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` forward rule returns residual {index} of \
                             reference type `{type}` from an input already forwarded by an earlier residual",
                            self.definition.name(),
                        ))
                        .into());
                    }
                }
                let program = self.definition.discharge_if(key.discharged, program)?;
                return Ok(Arc::new(CustomRuleSpecialization { program, output_axes: Vec::new() }));
            };
            let residual_count = inner.program.output_types().len() - output_count;
            let output_axes = level
                .output_axes
                .iter()
                .copied()
                .map(Some)
                .chain(std::iter::repeat_n(None, residual_count))
                .collect::<Vec<_>>();
            self.definition.batched_specialization(
                &level,
                &inner.program,
                &level.input_axes,
                &output_axes,
                CustomRuleBatchedOutputs::Values,
            )
        })
    }

    fn backward_specialization(
        &self,
        key: CustomRuleBackwardSpecializationKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type> + From<CustomFunctionTransposeOperation<V, O>>,
    {
        // Returns the backward rule specialization for `key`. An unbatched key transposes a source program that applies
        // a directly invoking [`CustomFunctionTransposeOperation`] and returns only its live-seed outputs, so
        // transposition hands the rule a structural zero for every dropped output. A batched key batches the
        // specialization of its inner key at its outermost level.
        let Some(transposer) = &self.definition.transposer else {
            return Err(ProgramError::MalformedProgram(format!(
                "`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` `{}` has no backward rule",
                self.definition.name(),
            ))
            .into());
        };
        let inner = match key.split_outermost_level() {
            Some((inner_key, level)) => Some((self.backward_specialization(inner_key)?, level.clone())),
            None => None,
        };
        self.specialize(CustomRuleSpecializationRequest::Backward(key.clone()), || {
            let leading_input_count = key.leading_input_types.len();
            let input_count = leading_input_count + key.input_tangent_types.len();
            let Some((inner, level)) = inner else {
                let source_carrier = CustomFunctionTransposeOperation::direct(
                    self.clone(),
                    leading_input_count,
                    key.seed_geometry_count,
                    key.input_tangent_types.clone(),
                    key.output_tangent_types.clone(),
                    key.discharged,
                );
                let mut builder = ProgramBuilder::<V, O>::new();
                let source_inputs = key
                    .leading_input_types
                    .iter()
                    .chain(&key.input_tangent_types)
                    .map(|r#type| builder.add_input(r#type.clone()))
                    .collect::<Vec<_>>();
                let source_outputs = builder.add_instruction(source_carrier, Vec::new(), source_inputs, None)?;
                let live_outputs = source_outputs
                    .iter()
                    .zip(&key.seed_types)
                    .filter_map(|(output, seed_type)| seed_type.as_ref().map(|_| *output))
                    .collect::<Vec<_>>();
                let live_output_count = live_outputs.len();
                let source = builder.build::<Vec<V>, Vec<V>>(
                    live_outputs,
                    vec![Placeholder; input_count],
                    vec![Placeholder; live_output_count],
                )?;
                let linear_inputs = (leading_input_count..input_count).collect::<Vec<_>>();
                let program = transposer(&source, &linear_inputs, &key.destination_kinds)?;
                let program = self.definition.discharge_if(key.discharged, program)?;
                return Ok(Arc::new(CustomRuleSpecialization { program, output_axes: Vec::new() }));
            };

            // The inner program consumes its inner boundary inputs, then the live seeds, then the caller buffers, then
            // the remaining leading inputs. Each seed follows its output, each caller buffer follows its input, and
            // each leading input keeps its recorded axis. Each returned cotangent is aligned to its input.
            let inner_boundary_input_count = boundary_input_count(&key.levels[..key.levels.len() - 1]);
            let (leading_axes, tangent_axes) =
                level.input_axes.split_at(inner_boundary_input_count + leading_input_count);
            let mut input_axes = leading_axes[..inner_boundary_input_count].to_vec();
            input_axes.extend(
                level
                    .output_axes
                    .iter()
                    .zip(&key.seed_types)
                    .filter_map(|(axis, seed_type)| seed_type.as_ref().map(|_| *axis)),
            );
            input_axes.extend(
                tangent_axes
                    .iter()
                    .zip(&key.destination_kinds)
                    .filter_map(|(axis, kind)| (*kind == CotangentDestinationKind::Reference).then_some(*axis)),
            );
            input_axes.extend_from_slice(&leading_axes[inner_boundary_input_count..]);
            let output_axes = tangent_axes
                .iter()
                .zip(&key.destination_kinds)
                .filter_map(|(axis, kind)| (*kind == CotangentDestinationKind::Return).then_some(Some(*axis)))
                .collect::<Vec<_>>();
            self.definition.batched_specialization(
                &level,
                &inner.program,
                &input_axes,
                &output_axes,
                CustomRuleBatchedOutputs::Cotangents,
            )
        })
    }
    fn batching_rule_specialization(
        &self,
        key: CustomRuleBatchingRuleKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        O: Operation<Type = V::Type>,
    {
        // Traces the custom batching rule over the level's boundary inputs followed by the batched inputs, recording
        // the batch axes that it declares for its outputs. A derived definition applies the JVP of its source's rule
        // instead.
        let Some(rule) = &self.definition.batching_rule else {
            let Some(derivation) = &self.definition.derivation else {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` has no custom batching rule",
                    self.definition.name(),
                ))
                .into());
            };
            return self.specialize(CustomRuleSpecializationRequest::BatchingRule(key.clone()), || {
                derivation.batching_rule_specialization(self.definition.name(), &key)
            });
        };
        self.specialize(CustomRuleSpecializationRequest::BatchingRule(key.clone()), || {
            let boundary_input_count = key.boundary_input_types.len();
            let mut output_axes = Vec::new();
            let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
                |values: Vec<CustomRuleTracer<V, O>>| {
                    let (boundary_inputs, inputs) = values.split_at(boundary_input_count);
                    let (outputs, axes) = rule.apply(&key.level, boundary_inputs, inputs, &key.input_axes)?;
                    if axes.len() != outputs.len() {
                        return Err(TypeError::invalid(format!(
                            "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` batching rule returned {} outputs but {} output \
                             batch axes",
                            self.definition.name(),
                            outputs.len(),
                            axes.len(),
                        ))
                        .into());
                    }
                    output_axes = axes;
                    Ok(outputs)
                },
                key.boundary_input_types.iter().chain(&key.input_types).cloned().collect::<Vec<_>>(),
                self.definition.named_axes.clone(),
            )?;
            Ok(Arc::new(CustomRuleSpecialization { program, output_axes }))
        })
    }

    fn derived(&self, key: CustomRuleDerivationKey) -> Result<Self, DifferentiationError> {
        // A derived definition copies the capabilities that its calls need from the source definition and holds no
        // program, so calls whose primal regions differ share it (refer to `CustomRuleSpecializer::derived`).
        if !self.has_batching_rule() {
            return Err(ProgramError::MalformedProgram(format!(
                "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` has no custom batching rule from which to derive rules",
                self.definition.name(),
            ))
            .into());
        }
        let derive = |key: CustomRuleDerivationKey| {
            let prefix = match key.kind {
                CustomRuleDerivationKind::Jvp => "jvp",
                CustomRuleDerivationKind::Pushforward => "pushforward",
            };
            CustomRuleDefinition {
                name: Cow::Owned(format!("{prefix}({})", self.definition.name())),
                named_axes: self.definition.named_axes.clone(),
                jvp: Some(CustomRuleJvp::Primal),
                batcher: self.definition.batcher.clone(),
                differentiator: self.definition.differentiator.clone(),
                axis_dependence: self.definition.axis_dependence.clone(),
                derivation: Some(CustomRuleDerivation { source: self.clone(), key }),
                discharger: self.definition.discharger.clone(),
                ..CustomRuleDefinition::new("")
            }
        };
        let Some(caches) = self.caches.upgrade() else {
            return Ok(CustomRuleRegistration::new(derive(key)).reference());
        };
        let mut registrations = caches.derived_registrations.lock().expect("custom rule derivation mutex is poisoned");
        Ok(registrations
            .entry(key.clone())
            .or_insert_with(|| CustomRuleRegistration::new(derive(key)))
            .reference())
    }
}

/// Rules of a definition that cannot be converted into a family (i.e., a definition registered in a family whose
/// values that family cannot represent, such as array IR programs converted into a backend whose constants are only
/// captures). Its calls still execute their primal, but every derivative request fails with an exact error.
#[derive(Clone, Debug)]
pub struct UnavailableCustomRules {
    /// Process-unique identity of the original definition.
    id: u64,

    /// Human-readable label of the original definition.
    name: Cow<'static, str>,

    /// Whether the original definition has a retained forward-mode (i.e., JVP) rule.
    has_jvp: bool,

    /// Whether the original definition derives its forward-mode rule from the primal.
    derives_jvp_from_primal: bool,

    /// Whether the original definition has reverse-mode (i.e., VJP) rules.
    has_vjp: bool,

    /// Declared batch axes of the outputs of batched calls of the original definition.
    batched_output_axes: Option<Vec<BatchAxis>>,

    /// Whether the original definition has a custom batching rule.
    has_batching_rule: bool,
}

impl UnavailableCustomRules {
    /// Creates [`UnavailableCustomRules`] that keep the identity and description of the provided source.
    pub fn from_source<V: Typed<Type: Eq + Hash> + Parameter, O, S: CustomRuleSource<V, O>>(rules: &S) -> Self {
        Self {
            id: rules.id(),
            name: Cow::Owned(rules.name().to_owned()),
            has_jvp: rules.has_jvp(),
            derives_jvp_from_primal: rules.derives_jvp_from_primal(),
            has_vjp: rules.has_vjp(),
            batched_output_axes: rules.batched_output_axes().map(<[BatchAxis]>::to_vec),
            has_batching_rule: rules.has_batching_rule(),
        }
    }

    /// Returns the error that every derivative request of these rules reports.
    fn error(&self) -> DifferentiationError {
        ProgramError::UnsupportedOperation {
            message: format!(
                "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` was converted from a family whose values its current family \
                 cannot represent, so its derivative rules are unavailable",
                self.name,
            ),
        }
        .into()
    }
}

/// Source of the rules of a call that was converted from a member family `(Vm, Om)` into a family that contains it
/// (e.g., from array programs into array IR programs, or into a backend family). Each specialization request projects
/// its key into the member family, obtains the specialization from the member definition (so that the member
/// definition's caches, batcher, and transposer serve every family that its calls reach), and converts the program
/// into the requesting family with [`Program::into_unprojected`]. Nested member calls inside that program are converted
/// in turn, by the same conversion that converted the call.
///
/// A member family cannot represent every value of the families that contain it, so some requests of converted calls
/// fail with exact errors at their first derivative request: rule inputs whose types have no member representation
/// (e.g., references or first-class dimensions of array IR programs), and batching levels whose dynamic extents have no
/// member representation. A backward rule of a converted call returns cotangents whose destinations are caller-provided
/// buffers, and the carrier's accumulators add them to those buffers.
pub enum LiftedCustomRules<Vm: Typed + Parameter, Om> {
    /// Rules of a definition registered in the member family.
    Member(CustomRuleReference<Vm, Om>),

    /// Rules of a definition that cannot be converted into the requesting family.
    Unavailable(UnavailableCustomRules),
}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> LiftedCustomRules<Vm, Om> {
    /// Returns the rules of the member definition, or the error that a derivative request of unavailable rules reports.
    fn member(&self) -> Result<&CustomRuleReference<Vm, Om>, DifferentiationError> {
        match self {
            Self::Member(rules) => Ok(rules),
            Self::Unavailable(rules) => Err(rules.error()),
        }
    }

    /// Projects the provided types of a converted call into the member family, rejecting types that have no member
    /// representation.
    fn project_types<Tc>(&self, types: &[Tc]) -> Result<Vec<Vm::Type>, DifferentiationError>
    where
        Tc: Type,
        for<'t> &'t Vm::Type: TryFrom<&'t Tc, Error = TypeError>,
    {
        types
            .iter()
            .map(|r#type| {
                <&Vm::Type>::try_from(r#type).cloned().map_err(|_| {
                    ProgramError::UnsupportedOperation {
                        message: format!(
                            "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` cannot be differentiated with respect to type \
                             `{type}`, which the family in which its rules were registered cannot represent",
                            self.name_or_unavailable(),
                        ),
                    }
                    .into()
                })
            })
            .collect()
    }

    /// Projects the provided batching levels of a converted call into the member family, rejecting dynamic extents
    /// whose types have no member representation.
    fn project_levels<Tc>(
        &self,
        levels: &[CustomRuleBatchingLevel<Tc>],
    ) -> Result<Vec<CustomRuleBatchingLevel<Vm::Type>>, DifferentiationError>
    where
        Tc: Type,
        for<'t> &'t Vm::Type: TryFrom<&'t Tc, Error = TypeError>,
    {
        levels
            .iter()
            .map(|level| {
                let extent =
                    match level.level.extent() {
                        BatchingLevelExtent::Static(extent) => BatchingLevelExtent::Static(*extent),
                        BatchingLevelExtent::Dynamic(r#type) => BatchingLevelExtent::Dynamic(
                            <&Vm::Type>::try_from(r#type).cloned().map_err(|_| ProgramError::UnsupportedOperation {
                                message: format!(
                                    "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` cannot differentiate a batched call whose \
                                     batch extent has type `{type}`, which the family in which its rules were \
                                     registered cannot represent; differentiate the call before batching it instead",
                                    self.name_or_unavailable(),
                                ),
                            })?,
                        ),
                    };
                Ok(CustomRuleBatchingLevel {
                    level: BatchingLevel::new(
                        extent,
                        level.level.axis_name().map(str::to_owned),
                        level.level.axis_sharding().clone(),
                    ),
                    boundary_input_count: level.boundary_input_count,
                    input_axes: level.input_axes.clone(),
                    output_axes: level.output_axes.clone(),
                })
            })
            .collect()
    }

    /// Returns the name of the member definition or of the unavailable definition.
    fn name_or_unavailable(&self) -> &str {
        match self {
            Self::Member(rules) => rules.definition.name(),
            Self::Unavailable(rules) => &rules.name,
        }
    }
}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> Clone for LiftedCustomRules<Vm, Om> {
    fn clone(&self) -> Self {
        match self {
            Self::Member(rules) => Self::Member(rules.clone()),
            Self::Unavailable(rules) => Self::Unavailable(rules.clone()),
        }
    }
}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> Debug for LiftedCustomRules<Vm, Om> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Member(rules) => formatter.debug_tuple("Member").field(rules).finish(),
            Self::Unavailable(rules) => formatter.debug_tuple("Unavailable").field(rules).finish(),
        }
    }
}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> PartialEq for LiftedCustomRules<Vm, Om> {
    fn eq(&self, other: &Self) -> bool {
        let id = |rules: &Self| match rules {
            Self::Member(rules) => rules.definition.id,
            Self::Unavailable(rules) => rules.id,
        };
        id(self) == id(other)
    }
}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> Eq for LiftedCustomRules<Vm, Om> {}

impl<Vm: Typed<Type: Eq + Hash> + Parameter, Om> Hash for LiftedCustomRules<Vm, Om> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Member(rules) => rules.definition.id.hash(state),
            Self::Unavailable(rules) => rules.id.hash(state),
        }
    }
}

impl<Vm, Om, Vc, Oc> CustomRuleSource<Vc, Oc> for LiftedCustomRules<Vm, Om>
where
    Vm: Typed<Type: Eq + Hash> + Parameter,
    Vc: Typed<Type: Eq + Hash> + Parameter,
{
    fn id(&self) -> u64 {
        match self {
            Self::Member(rules) => rules.definition.id,
            Self::Unavailable(rules) => rules.id,
        }
    }

    #[inline]
    fn name(&self) -> &str {
        self.name_or_unavailable()
    }

    fn has_jvp(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_jvp(),
            Self::Unavailable(rules) => rules.has_jvp,
        }
    }

    fn derives_jvp_from_primal(&self) -> bool {
        match self {
            Self::Member(rules) => rules.derives_jvp_from_primal(),
            Self::Unavailable(rules) => rules.derives_jvp_from_primal,
        }
    }

    fn has_vjp(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_vjp(),
            Self::Unavailable(rules) => rules.has_vjp,
        }
    }

    fn batched_output_axes(&self) -> Option<&[BatchAxis]> {
        match self {
            Self::Member(rules) => rules.batched_output_axes(),
            Self::Unavailable(rules) => rules.batched_output_axes.as_deref(),
        }
    }

    fn has_batching_rule(&self) -> bool {
        match self {
            Self::Member(rules) => rules.has_batching_rule(),
            Self::Unavailable(rules) => rules.has_batching_rule,
        }
    }

    #[inline]
    fn native_definition(&self) -> Option<&CustomRuleDefinition<Vc, Oc>> {
        None
    }

    #[inline]
    fn supports_reference_destinations(&self) -> bool {
        false
    }
}

impl<Vm, Om, Vc, Oc> CustomRuleSpecializer<Vc, Oc> for LiftedCustomRules<Vm, Om>
where
    Vm: Value<Type: DifferentiableType + Eq + Hash>,
    Om: Operation<Type = Vm::Type>
        + From<CustomFunctionTransposeOperation<Vm, Om>>
        + ResidualZeroProvider<Vm::Type, Operation = Om>,
    Vc: ValueProjection<Vm::Type, Projected = Vm, Type: Eq + Hash + From<Vm::Type>>,
    Oc: From<Om>,
    for<'t> &'t Vm::Type: TryFrom<&'t Vc::Type, Error = TypeError>,
{
    fn jvp_specialization(
        &self,
        key: CustomRuleSpecializationKey<Vc::Type>,
    ) -> Result<CustomRuleProgram<Vc, Oc>, DifferentiationError>
    where
        Vc: Value<Type: DifferentiableType>,
        Oc: Operation<Type = Vc::Type> + ResidualZeroProvider<Vc::Type, Operation = Oc>,
    {
        let member = self.member()?;
        let specialization = member.jvp_specialization(CustomRuleSpecializationKey {
            input_types: self.project_types(&key.input_types)?,
            output_types: self.project_types(&key.output_types)?,
            non_differentiated_count: key.non_differentiated_count,
            tangent_activity: key.tangent_activity.clone(),
            levels: self.project_levels(&key.levels)?,
            discharged: key.discharged,
        })?;
        Ok(Arc::new(CustomRuleSpecialization {
            program: specialization.program.clone().into_unprojected::<Vc, Oc>()?,
            output_axes: specialization.output_axes.clone(),
        }))
    }

    fn forward_specialization(
        &self,
        key: CustomRuleSpecializationKey<Vc::Type>,
    ) -> Result<CustomRuleProgram<Vc, Oc>, DifferentiationError>
    where
        Vc: Value<Type: DifferentiableType>,
        Oc: Operation<Type = Vc::Type>,
    {
        let member = self.member()?;
        let specialization = member.forward_specialization(CustomRuleSpecializationKey {
            input_types: self.project_types(&key.input_types)?,
            output_types: self.project_types(&key.output_types)?,
            non_differentiated_count: key.non_differentiated_count,
            tangent_activity: key.tangent_activity.clone(),
            levels: self.project_levels(&key.levels)?,
            discharged: key.discharged,
        })?;
        Ok(Arc::new(CustomRuleSpecialization {
            program: specialization.program.clone().into_unprojected::<Vc, Oc>()?,
            output_axes: specialization.output_axes.clone(),
        }))
    }

    fn backward_specialization(
        &self,
        key: CustomRuleBackwardSpecializationKey<Vc::Type>,
    ) -> Result<CustomRuleProgram<Vc, Oc>, DifferentiationError>
    where
        Vc: Value<Type: DifferentiableType>,
        Oc: Operation<Type = Vc::Type> + From<CustomFunctionTransposeOperation<Vc, Oc>>,
    {
        // Carriers of converted calls request returned cotangents for caller buffers (refer to
        // `supports_reference_destinations`), so a buffer destination here means that a carrier skipped that step.
        let member = self.member()?;
        if key.destination_kinds.contains(&CotangentDestinationKind::Reference) {
            return Err(ProgramError::MalformedProgram(format!(
                "`{CUSTOM_FUNCTION_TRANSPOSE_OPERATION_NAME}` `{}` requested a caller-buffer destination from rules \
                 that cannot write caller buffers",
                self.name_or_unavailable(),
            ))
            .into());
        }
        let seed_types = key
            .seed_types
            .iter()
            .map(|r#type| {
                r#type
                    .as_ref()
                    .map(|r#type| Ok(self.project_types(std::slice::from_ref(r#type))?.remove(0)))
                    .transpose()
            })
            .collect::<Result<Vec<_>, DifferentiationError>>()?;
        let specialization = member.backward_specialization(CustomRuleBackwardSpecializationKey {
            leading_input_types: self.project_types(&key.leading_input_types)?,
            seed_geometry_count: key.seed_geometry_count,
            input_tangent_types: self.project_types(&key.input_tangent_types)?,
            output_tangent_types: self.project_types(&key.output_tangent_types)?,
            seed_types,
            destination_kinds: key.destination_kinds.clone(),
            levels: self.project_levels(&key.levels)?,
            discharged: key.discharged,
        })?;
        Ok(Arc::new(CustomRuleSpecialization {
            program: specialization.program.clone().into_unprojected::<Vc, Oc>()?,
            output_axes: specialization.output_axes.clone(),
        }))
    }
    fn batching_rule_specialization(
        &self,
        key: CustomRuleBatchingRuleKey<Vc::Type>,
    ) -> Result<CustomRuleProgram<Vc, Oc>, DifferentiationError>
    where
        Vc: Value<Type: DifferentiableType>,
        Oc: Operation<Type = Vc::Type>,
    {
        // The member rule batches the call at the projected input types, and its program is converted back.
        let member = self.member()?;
        let project = |r#type: &Vc::Type| {
            <&Vm::Type>::try_from(r#type).cloned().map_err(|_| {
                DifferentiationError::from(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` cannot batch a call over type `{type}`, which the \
                         family in which its rules were registered cannot represent",
                        self.name_or_unavailable(),
                    ),
                })
            })
        };
        let extent = match key.level.extent() {
            BatchingLevelExtent::Static(extent) => BatchingLevelExtent::Static(*extent),
            BatchingLevelExtent::Dynamic(r#type) => BatchingLevelExtent::Dynamic(project(r#type)?),
        };
        let specialization = member.batching_rule_specialization(CustomRuleBatchingRuleKey {
            level: BatchingLevel::new(
                extent,
                key.level.axis_name().map(str::to_owned),
                key.level.axis_sharding().clone(),
            ),
            boundary_input_types: key.boundary_input_types.iter().map(project).collect::<Result<_, _>>()?,
            input_types: key.input_types.iter().map(project).collect::<Result<_, _>>()?,
            unbatched_input_types: key.unbatched_input_types.iter().map(project).collect::<Result<_, _>>()?,
            input_axes: key.input_axes.clone(),
        })?;
        Ok(Arc::new(CustomRuleSpecialization {
            program: specialization.program.clone().into_unprojected::<Vc, Oc>()?,
            output_axes: specialization.output_axes.clone(),
        }))
    }

    fn derived(&self, key: CustomRuleDerivationKey) -> Result<Self, DifferentiationError> {
        // The member definition derives in its own family, and the derived rules follow the call like the member's
        // rules do. Unavailable rules stay unavailable, so the derived call reports the same error on its first
        // request.
        match self {
            Self::Member(rules) => Ok(Self::Member(rules.derived(key)?)),
            Self::Unavailable(rules) => Ok(Self::Unavailable(rules.clone())),
        }
    }
}

impl<V: Value<Type: DifferentiableType + Eq + Hash>, O: Operation<Type = V::Type>> CustomRuleDerivation<V, O> {
    /// Returns the batched program of a derived call for `key` (refer to [`CustomRuleDerivation`] for its two modes),
    /// which consumes the level's boundary inputs followed by the derived call's batched inputs (i.e., its primal
    /// inputs, including the boundary inputs of earlier levels, followed by its active tangents) and returns its
    /// batched outputs, together with their batch axes.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `key`: Specialization key of the batched derived call.
    fn batching_rule_specialization(
        &self,
        name: &str,
        key: &CustomRuleBatchingRuleKey<V::Type>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError> {
        let source = &self.source.definition;
        let (Some(differentiator), Some(batcher)) = (&source.differentiator, &source.batcher) else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{name}` cannot batch a derivative because its source \
                     definition has no batching support",
                ),
            }
            .into());
        };
        let input_count = key.input_types.len();
        let tangent_count = self.key.active_input_indices.len();
        let primal_count = input_count
            .checked_sub(tangent_count)
            .filter(|count| *count >= self.key.input_count)
            .filter(|_| key.unbatched_input_types.len() == input_count && key.input_axes.len() == input_count)
            .ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "batched `{CUSTOM_FUNCTION_OPERATION_NAME}` `{name}` has {input_count} inputs, which do not cover \
                     its {} primal inputs and {tangent_count} tangents",
                    self.key.input_count,
                ))
            })?;

        // The primal inputs of a derived call that was already batched start with the boundary inputs of the earlier
        // levels, which have no tangents.
        let earlier_boundary_input_count = primal_count - self.key.input_count;
        let tangent_primal_indices = self
            .key
            .active_input_indices
            .iter()
            .map(|index| earlier_boundary_input_count + index)
            .collect::<Vec<_>>();
        let (primal_axes, tangent_axes) = key.input_axes.split_at(primal_count);
        let mut primal_tangents = vec![None; primal_count];
        for (tangent, &primal) in tangent_primal_indices.iter().enumerate() {
            primal_tangents[primal] = Some(tangent);
        }
        let boundary_input_count = key.boundary_input_types.len();
        let jvp_input_indices =
            tangent_primal_indices.iter().map(|index| boundary_input_count + index).collect::<Vec<_>>();
        let rule_key = |input_types: Vec<V::Type>, input_axes: Vec<BatchAxis>| CustomRuleBatchingRuleKey {
            level: key.level.clone(),
            boundary_input_types: key.boundary_input_types.clone(),
            input_types,
            unbatched_input_types: key.unbatched_input_types[..primal_count].to_vec(),
            input_axes,
        };
        let program_types = key.boundary_input_types.iter().chain(&key.input_types).cloned().collect::<Vec<_>>();
        let tangents_only = self.key.kind == CustomRuleDerivationKind::Pushforward;

        // The source rule applied at the primal inputs' own batch axes is the batched primal of the source call, so it
        // computes the derived call's primal outputs in every mode.
        let primal_rule = self
            .source
            .batching_rule_specialization(rule_key(key.input_types[..primal_count].to_vec(), primal_axes.to_vec()))?;

        // Structural mode: every primal input is replicated, and so is every output of the rule applied to them.
        if primal_axes.iter().chain(&primal_rule.output_axes).all(BatchAxis::is_replicated) {
            let jvp = self.jvp_program(
                name,
                differentiator.as_ref(),
                &primal_rule.program,
                &jvp_input_indices,
                tangents_only,
            )?;
            let mut input_axes = vec![BatchAxis::replicated(); boundary_input_count];
            input_axes.extend_from_slice(&key.input_axes);
            let (batched, output_axes) = batcher(
                &key.level,
                &jvp,
                &input_axes,
                &vec![None; jvp.output_types().len()],
                CustomRuleBatchedOutputs::Values,
            )
            .map_err(ProgramError::from)?;

            // The batched program consumes the level's boundary inputs before the JVP program's own copy of them.
            let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
                |values: Vec<CustomRuleTracer<V, O>>| {
                    let context = values[0].context().clone();
                    let mut inputs = values[..boundary_input_count].to_vec();
                    inputs.extend(values);
                    Ok(batched.interpret_in_context(&context, inputs)?)
                },
                program_types,
                self.source.definition.named_axes.clone(),
            )?;
            return Ok(Arc::new(CustomRuleSpecialization { program, output_axes }));
        }

        // Rule mode: align each primal and tangent pair with a mapped side to one axis, preferring the primal's axis. Only
        // a replicated primal whose tangent is mapped is broadcast, in which case the rule applied at the aligned axes
        // computes the tangents, while the primal outputs still come from the rule applied at the primal axes.
        let mut aligned_axes = key.input_axes.clone();
        for (tangent, &primal) in tangent_primal_indices.iter().enumerate() {
            let axis = if primal_axes[primal].is_replicated() { tangent_axes[tangent] } else { primal_axes[primal] };
            aligned_axes[primal] = axis;
            aligned_axes[primal_count + tangent] = axis;
        }
        let broadcasts_primals = aligned_axes[..primal_count] != *primal_axes;
        if broadcasts_primals && let Some(axis_dependence) = &source.axis_dependence {
            return self.axis_dependent_batching_rule_specialization(
                name,
                key,
                &primal_rule,
                &tangent_primal_indices,
                axis_dependence.as_ref(),
                differentiator.as_ref(),
                batcher.as_ref(),
            );
        }
        let aligned_inputs =
            (0..input_count).filter(|&index| aligned_axes[index] != key.input_axes[index]).collect::<Vec<_>>();
        let mut aligned_types = key.input_types.clone();
        let alignment = match aligned_inputs.is_empty() {
            true => None,
            false => {
                let alignment = Self::alignment_program(
                    name,
                    batcher.as_ref(),
                    &key.level,
                    &aligned_inputs.iter().map(|&index| key.unbatched_input_types[index].clone()).collect::<Vec<_>>(),
                    &aligned_inputs.iter().map(|&index| key.input_axes[index]).collect::<Vec<_>>(),
                    &aligned_inputs.iter().map(|&index| aligned_axes[index]).collect::<Vec<_>>(),
                )?;
                for (&index, r#type) in aligned_inputs.iter().zip(alignment.output_types()) {
                    aligned_types[index] = r#type;
                }
                Some(alignment)
            }
        };
        let rule = match broadcasts_primals {
            true => self.source.batching_rule_specialization(rule_key(
                aligned_types[..primal_count].to_vec(),
                aligned_axes[..primal_count].to_vec(),
            ))?,
            false => primal_rule.clone(),
        };
        let separate_primal = broadcasts_primals && !tangents_only;
        let jvp = self.jvp_program(
            name,
            differentiator.as_ref(),
            &rule.program,
            &jvp_input_indices,
            tangents_only || separate_primal,
        )?;
        let tangent_output_axes = rule
            .output_axes
            .iter()
            .zip(&self.key.output_tangent_mask)
            .filter_map(|(axis, live)| live.then_some(*axis));
        let output_axes = match tangents_only {
            true => tangent_output_axes.collect(),
            false => primal_rule.output_axes.iter().copied().chain(tangent_output_axes).collect(),
        };
        let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
            |values: Vec<CustomRuleTracer<V, O>>| {
                let context = values[0].context().clone();
                let (boundary_inputs, inputs) = values.split_at(boundary_input_count);
                let mut inputs = inputs.to_vec();
                if let Some(alignment) = &alignment {
                    let mut alignment_inputs = boundary_inputs.to_vec();
                    alignment_inputs.extend(aligned_inputs.iter().map(|&index| inputs[index].clone()));
                    let aligned = alignment.interpret_in_context(&context, alignment_inputs)?;
                    for (&index, value) in aligned_inputs.iter().zip(aligned) {
                        inputs[index] = value;
                    }
                }
                let mut outputs = Vec::new();
                if separate_primal {
                    let primal_inputs = boundary_inputs.iter().chain(&values[boundary_input_count..][..primal_count]);
                    outputs = primal_rule.program.interpret_in_context(&context, primal_inputs.cloned().collect())?;
                }
                let mut jvp_inputs = boundary_inputs.to_vec();
                jvp_inputs.extend(inputs);
                outputs.extend(jvp.interpret_in_context(&context, jvp_inputs)?);
                Ok(outputs)
            },
            program_types,
            self.source.definition.named_axes.clone(),
        )?;
        Ok(Arc::new(CustomRuleSpecialization { program, output_axes }))
    }

    /// Returns the batched program of a derived call for `key` when its source batching rule depends on its input batch
    /// axes and some replicated primal input has a mapped tangent (refer to
    /// [`CustomRuleDefinition::with_axis_dependent_batching_rule`]). The rule is applied once, at the primal inputs'
    /// own batch axes, as `P`, which computes the primal outputs. The tangents are the sum of two contributions:
    ///
    ///   - **Following:** the JVP of `P` with respect to the tangents that can follow their primal's batch axis, after
    ///     aligning them to it (i.e., broadcasting or moving them).
    ///   - **Tangent-only:** the JVP of `P` with respect to the mapped tangents of replicated primals. Those tangents
    ///     differ per batch item while their primals are shared, so the JVP is batched structurally over them at a
    ///     second level of the same extent, which yields the outer product of their batch with the batch of `P`. Item
    ///     `i` of the result is item `i` of that outer product along both axes, which is its diagonal.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `key`: Specialization key of the batched derived call.
    ///   - `primal_rule`: Source rule applied at the primal inputs' batch axes (i.e., `P`).
    ///   - `tangent_primal_indices`: Index of the primal input of each active tangent, among the derived call's primal
    ///     inputs.
    ///   - `axis_dependence`: Operations of the source definition that take diagonals and add contributions.
    ///   - `differentiator`: Differentiator of the source definition.
    ///   - `batcher`: Batcher of the source definition.
    fn axis_dependent_batching_rule_specialization(
        &self,
        name: &str,
        key: &CustomRuleBatchingRuleKey<V::Type>,
        primal_rule: &CustomRuleProgram<V, O>,
        tangent_primal_indices: &[usize],
        axis_dependence: &dyn CustomRuleAxisDependence<V, O>,
        differentiator: &CustomRuleDifferentiator<V, O>,
        batcher: &CustomRuleBatcher<V, O>,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError> {
        let tangents_only = self.key.kind == CustomRuleDerivationKind::Pushforward;
        let boundary_input_count = key.boundary_input_types.len();
        let primal_count = key.input_types.len() - tangent_primal_indices.len();
        let (primal_axes, tangent_axes) = key.input_axes.split_at(primal_count);

        // Batching policies report normalized (i.e., non-negative) batch axis positions.
        let position = |axis: Axis| {
            usize::try_from(axis.value()).map_err(|_| {
                ProgramError::MalformedProgram(format!(
                    "batched `{CUSTOM_FUNCTION_OPERATION_NAME}` `{name}` has the unnormalized batch axis {axis}",
                ))
            })
        };
        let (following, tangent_only): (Vec<usize>, Vec<usize>) =
            (0..tangent_primal_indices.len()).partition(|&tangent| {
                !primal_axes[tangent_primal_indices[tangent]].is_replicated() || tangent_axes[tangent].is_replicated()
            });
        let jvp_input_indices = |tangents: &[usize]| {
            tangents
                .iter()
                .map(|&tangent| boundary_input_count + tangent_primal_indices[tangent])
                .collect::<Vec<_>>()
        };

        // The following contribution takes the JVP of `P` after moving or broadcasting each tangent to its primal's
        // batch axis.
        let (following_jvp, following_mask) = match following.is_empty() {
            true => (None, vec![false; self.key.output_tangent_mask.len()]),
            false => {
                let (jvp, mask) = Self::partial_jvp_program(
                    differentiator,
                    &primal_rule.program,
                    &jvp_input_indices(&following),
                    true,
                )?;
                (Some(jvp), mask)
            }
        };
        let misaligned_following = following
            .iter()
            .copied()
            .filter(|&tangent| tangent_axes[tangent] != primal_axes[tangent_primal_indices[tangent]])
            .collect::<Vec<_>>();
        let following_alignment = match misaligned_following.is_empty() {
            true => None,
            false => Some(Self::alignment_program(
                name,
                batcher,
                &key.level,
                &misaligned_following
                    .iter()
                    .map(|&tangent| key.unbatched_input_types[primal_count + tangent].clone())
                    .collect::<Vec<_>>(),
                &misaligned_following.iter().map(|&tangent| tangent_axes[tangent]).collect::<Vec<_>>(),
                &misaligned_following
                    .iter()
                    .map(|&tangent| primal_axes[tangent_primal_indices[tangent]])
                    .collect::<Vec<_>>(),
            )?),
        };

        // The tangent-only contribution batches the JVP of `P` over the mapped tangents at a second, anonymous and
        // replicated level of the same extent, which consumes the same boundary inputs as the call's level.
        let (tangent_only_jvp, tangent_only_mask) =
            Self::partial_jvp_program(differentiator, &primal_rule.program, &jvp_input_indices(&tangent_only), true)?;
        let outer_level = BatchingLevel::new(key.level.extent().clone(), None, ShardingDimension::Replicated);
        let mut outer_input_axes = vec![BatchAxis::replicated(); boundary_input_count + primal_count];
        outer_input_axes.extend(tangent_only.iter().map(|&tangent| tangent_axes[tangent]));
        let (outer_product, outer_axes) = batcher(
            &outer_level,
            &tangent_only_jvp,
            &outer_input_axes,
            &vec![None; tangent_only_jvp.output_types().len()],
            CustomRuleBatchedOutputs::Values,
        )
        .map_err(ProgramError::from)?;
        let combined_mask = following_mask
            .iter()
            .zip(&tangent_only_mask)
            .map(|(following, tangent_only)| *following || *tangent_only);
        self.validate_output_tangent_mask(name, &combined_mask.collect::<Vec<_>>())?;

        // Each live output tangent is the sum of the contributions that it has. The batch axis of `P`'s output is
        // kept when it is mapped. Otherwise, the tangent-only contribution varies only along the second level's
        // axis, which becomes the output's batch axis, and the following contribution is broadcast along it.
        let mut following_index = 0;
        let mut tangent_only_index = 0;
        let mut tangents = Vec::new();
        for (output, axis) in primal_rule.output_axes.iter().enumerate() {
            let following = following_mask[output].then(|| {
                following_index += 1;
                following_index - 1
            });
            let tangent_only = tangent_only_mask[output].then(|| {
                tangent_only_index += 1;
                tangent_only_index - 1
            });
            if following.is_none() && tangent_only.is_none() {
                continue;
            }
            let outer_axis = match tangent_only.and_then(|index| outer_axes[index].axis()) {
                None => None,
                Some(outer_axis) => Some(position(outer_axis)?),
            };
            tangents.push((following, tangent_only, outer_axis, *axis));
        }
        let output_axes = tangents
            .iter()
            .map(|(_, _, outer_axis, axis)| match (axis.is_replicated(), outer_axis) {
                (true, Some(outer_axis)) => BatchAxis::new(*outer_axis),
                _ => *axis,
            })
            .collect::<Vec<_>>();
        let broadcast_following = tangents
            .iter()
            .zip(&output_axes)
            .filter_map(|((following, _, _, axis), output_axis)| {
                following
                    .filter(|_| axis.is_replicated() && !output_axis.is_replicated())
                    .map(|index| (index, *output_axis))
            })
            .collect::<Vec<_>>();
        let following_broadcast = match broadcast_following.is_empty() {
            true => None,
            false => {
                let following_types = following_jvp.as_ref().unwrap().output_types();
                Some(Self::alignment_program(
                    name,
                    batcher,
                    &key.level,
                    &broadcast_following.iter().map(|(index, _)| following_types[*index].clone()).collect::<Vec<_>>(),
                    &vec![BatchAxis::replicated(); broadcast_following.len()],
                    &broadcast_following.iter().map(|(_, axis)| *axis).collect::<Vec<_>>(),
                )?)
            }
        };
        let mut all_output_axes = match tangents_only {
            true => Vec::new(),
            false => primal_rule.output_axes.clone(),
        };
        all_output_axes.extend(output_axes.iter().copied());

        let (_, program) = TracingContext::<V, O>::trace_with_named_axes(
            |values: Vec<CustomRuleTracer<V, O>>| {
                let context = values[0].context().clone();
                let (boundary_inputs, inputs) = values.split_at(boundary_input_count);
                let (primals, input_tangents) = inputs.split_at(primal_count);
                let leading_inputs = boundary_inputs.iter().chain(primals).cloned().collect::<Vec<_>>();
                let mut outputs = Vec::new();
                if !tangents_only {
                    outputs = primal_rule.program.interpret_in_context(&context, leading_inputs.clone())?;
                }

                // Following contribution.
                let mut following_tangents =
                    following.iter().map(|&tangent| input_tangents[tangent].clone()).collect::<Vec<_>>();
                if let Some(alignment) = &following_alignment {
                    let mut alignment_inputs = boundary_inputs.to_vec();
                    alignment_inputs
                        .extend(misaligned_following.iter().map(|&tangent| input_tangents[tangent].clone()));
                    let aligned = alignment.interpret_in_context(&context, alignment_inputs)?;
                    for (&tangent, value) in misaligned_following.iter().zip(aligned) {
                        let position = following.iter().position(|candidate| *candidate == tangent).unwrap();
                        following_tangents[position] = value;
                    }
                }
                let mut following_outputs = match &following_jvp {
                    None => Vec::new(),
                    Some(jvp) => {
                        let mut jvp_inputs = leading_inputs.clone();
                        jvp_inputs.extend(following_tangents);
                        jvp.interpret_in_context(&context, jvp_inputs)?
                    }
                };
                if let Some(broadcast) = &following_broadcast {
                    let mut broadcast_inputs = boundary_inputs.to_vec();
                    broadcast_inputs
                        .extend(broadcast_following.iter().map(|(index, _)| following_outputs[*index].clone()));
                    let broadcast = broadcast.interpret_in_context(&context, broadcast_inputs)?;
                    for ((index, _), value) in broadcast_following.iter().zip(broadcast) {
                        following_outputs[*index] = value;
                    }
                }

                // Tangent-only contribution.
                let mut outer_inputs = boundary_inputs.to_vec();
                outer_inputs.extend(leading_inputs.iter().cloned());
                outer_inputs.extend(tangent_only.iter().map(|&tangent| input_tangents[tangent].clone()));
                let outer_outputs = outer_product.interpret_in_context(&context, outer_inputs)?;

                for (following, tangent_only, outer_axis, axis) in &tangents {
                    let tangent_only = match (tangent_only, axis.axis()) {
                        (None, _) => None,
                        (Some(index), None) => Some(outer_outputs[*index].clone()),
                        (Some(index), Some(axis)) => {
                            // The second level's axis precedes `P`'s batch axis in the outer product exactly when its
                            // position does not exceed that axis's position in `P`'s output.
                            let outer_axis = outer_axis.unwrap();
                            let axis = position(axis)?;
                            let kept_axis = if outer_axis <= axis { axis + 1 } else { axis };
                            Some(axis_dependence.diagonal(&outer_outputs[*index], kept_axis, outer_axis)?)
                        }
                    };
                    let following = following.map(|index| following_outputs[index].clone());
                    // Every recorded output has at least one contribution.
                    outputs.push(match (following, tangent_only) {
                        (Some(following), Some(tangent_only)) => axis_dependence.add(&following, &tangent_only)?,
                        (Some(tangent), None) => tangent,
                        (None, tangent) => tangent.unwrap(),
                    });
                }
                Ok(outputs)
            },
            key.boundary_input_types.iter().chain(&key.input_types).cloned().collect::<Vec<_>>(),
            self.source.definition.named_axes.clone(),
        )?;
        Ok(Arc::new(CustomRuleSpecialization { program, output_axes: all_output_axes }))
    }

    /// Returns the Jacobian-Vector Product (JVP) program of the source rule's batched `program` with respect to
    /// `input_indices`, restricted to its tangent outputs when `tangents_only` is set, after validating that its live
    /// tangent outputs are those of the derived calls. The derivatives with respect to subsets of the active inputs
    /// (refer to [`Self::axis_dependent_batching_rule_specialization`]) use [`Self::partial_jvp_program`] instead.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `differentiator`: Differentiator of the source definition.
    ///   - `program`: Batched program of the source rule.
    ///   - `input_indices`: Indices of the program inputs whose tangents are active.
    ///   - `tangents_only`: Whether the returned program returns only the tangent outputs (e.g., for pushforward
    ///     derivations, or when another program computes the primal outputs).
    fn jvp_program(
        &self,
        name: &str,
        differentiator: &CustomRuleDifferentiator<V, O>,
        program: &Program<V, O, Vec<V>, Vec<V>>,
        input_indices: &[usize],
        tangents_only: bool,
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        let (jvp, output_tangent_mask) =
            Self::partial_jvp_program(differentiator, program, input_indices, tangents_only)?;
        self.validate_output_tangent_mask(name, &output_tangent_mask)?;
        Ok(jvp)
    }

    /// Returns the Jacobian-Vector Product (JVP) program of the source rule's batched `program` with respect to
    /// `input_indices`, restricted to its tangent outputs when `tangents_only` is set, together with whether each
    /// output has a live tangent with respect to those inputs (i.e., which outputs the JVP program's tangent outputs
    /// belong to).
    fn partial_jvp_program(
        differentiator: &CustomRuleDifferentiator<V, O>,
        program: &Program<V, O, Vec<V>, Vec<V>>,
        input_indices: &[usize],
        tangents_only: bool,
    ) -> Result<(Program<V, O, Vec<V>, Vec<V>>, Vec<bool>), DifferentiationError> {
        let output_tangent_mask = program.entry_region_ref().tangent_output_mask(input_indices)?;
        let jvp = differentiator(program, input_indices)?;
        if !tangents_only {
            return Ok((jvp, output_tangent_mask));
        }
        let inputs = jvp.input_ids();
        let tangent_outputs = &jvp.output_ids()[program.output_types().len()..];
        Ok((jvp.filtered(&inputs, tangent_outputs, &inputs)?.0, output_tangent_mask))
    }

    /// Validates that the source rule's batched program has live tangents for exactly the outputs whose tangents the
    /// derivative of the derived calls' primal has.
    fn validate_output_tangent_mask(
        &self,
        name: &str,
        output_tangent_mask: &[bool],
    ) -> Result<(), DifferentiationError> {
        if output_tangent_mask != self.key.output_tangent_mask {
            return Err(TypeError::invalid(format!(
                "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{name}` batching rule program has live tangents for the outputs \
                 {output_tangent_mask:?}, but the derivative of its primal has them for the outputs {:?}",
                self.key.output_tangent_mask,
            ))
            .into());
        }
        Ok(())
    }

    /// Returns the program that aligns values of the provided per-item `types` from `input_axes` to `output_axes` at
    /// `level` (i.e., that broadcasts a replicated value or moves a mapped axis), which consumes the level's boundary
    /// inputs followed by the values and returns the aligned values.
    ///
    /// # Parameters
    ///
    ///   - `name`: Name of the derived definition, used in diagnostics.
    ///   - `batcher`: Batcher of the source definition.
    ///   - `level`: Batching level of the values.
    ///   - `types`: Per-item types of the values.
    ///   - `input_axes`: Batch axes of the values.
    ///   - `output_axes`: Batch axes of the aligned values.
    fn alignment_program(
        name: &str,
        batcher: &CustomRuleBatcher<V, O>,
        level: &BatchingLevel<V::Type>,
        types: &[V::Type],
        input_axes: &[BatchAxis],
        output_axes: &[BatchAxis],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        let mut builder = ProgramBuilder::<V, O>::new();
        let inputs = types.iter().map(|r#type| builder.add_input(r#type.clone())).collect::<Vec<_>>();
        let identity =
            builder.build::<Vec<V>, Vec<V>>(inputs, vec![Placeholder; types.len()], vec![Placeholder; types.len()])?;
        let required_axes = output_axes.iter().copied().map(Some).collect::<Vec<_>>();
        let (alignment, aligned_axes) =
            batcher(level, &identity, input_axes, &required_axes, CustomRuleBatchedOutputs::Values)
                .map_err(ProgramError::from)?;
        if aligned_axes != output_axes {
            return Err(ProgramError::MalformedProgram(format!(
                "batched `{CUSTOM_FUNCTION_OPERATION_NAME}` `{name}` could not align values from the batch axes \
                 {input_axes:?} to {output_axes:?}",
            ))
            .into());
        }
        Ok(alignment)
    }
}

impl<V: Value<Type: DifferentiableType + Eq + Hash>, O: Operation<Type = V::Type>> CustomRuleDefinition<V, O> {
    /// Batches `program` at the provided level with the retained batcher, rejecting the request when this definition
    /// has no batching support.
    fn batched_specialization(
        &self,
        level: &CustomRuleBatchingLevel<V::Type>,
        program: &Program<V, O, Vec<V>, Vec<V>>,
        input_axes: &[BatchAxis],
        output_axes: &[Option<BatchAxis>],
        outputs: CustomRuleBatchedOutputs,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError> {
        let Some(batcher) = &self.batcher else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{}` cannot differentiate a batched call because its \
                     definition has no batching support",
                    self.name(),
                ),
            }
            .into());
        };
        let (program, output_axes) =
            batcher(&level.level, program, input_axes, output_axes, outputs).map_err(ProgramError::from)?;
        Ok(Arc::new(CustomRuleSpecialization { program, output_axes }))
    }
}

/// Returns the number of boundary inputs that the provided batching levels together prepended to an operation's inputs.
pub(super) fn boundary_input_count<T>(levels: &[CustomRuleBatchingLevel<T>]) -> usize {
    levels.iter().map(|level| level.boundary_input_count).sum()
}

/// Validates that a retained rule of the custom rule set named `name` returned the `expected` output types.
pub(super) fn validate_rule_output_types<T: Type>(
    name: &str,
    rule: &str,
    expected: &[T],
    actual: &[T],
) -> Result<(), TypeError> {
    check_types!(@same, format!("`{CUSTOM_FUNCTION_OPERATION_NAME}` `{name}` {rule} rule output"), [
        expected,
        actual,
    ]);
    Ok(())
}

/// Validates that `program`, a retained rule of the custom rule set named `name` traced in its own trace, returns the
/// `expected` types as its leading outputs, and returns it with its type identities renamed onto those of `expected`.
/// The trace establishes fresh identities for the extents that the rule computes (e.g., a recomputed `n · n` extent of
/// the primal outputs), so its outputs can agree with `expected` only up to a renaming of those identities. The
/// renaming is derived across the complete boundary, so that the identities of the program inputs map to themselves,
/// and it must be bijective, so that it preserves which outputs share an identity. It is applied to the whole program,
/// including any trailing outputs (e.g., residuals) that name the renamed identities.
fn conform_rule_output_types<V: Value, O: Operation<Type = V::Type>>(
    name: &str,
    rule: &str,
    program: Program<V, O, Vec<V>, Vec<V>>,
    expected: &[V::Type],
) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
    let input_types = program.input_types();
    let output_types = program.output_types();
    let actual = output_types.get(..expected.len()).unwrap_or(&output_types);
    let boundary = |outputs: &[V::Type]| input_types.iter().chain(outputs).cloned().collect::<Vec<_>>();
    let Some(renaming) = derive_bijective_identity_renaming(&boundary(actual), &boundary(expected)) else {
        // Signatures that render equally differ only in their identities (e.g., when the rule computes two extents
        // separately that the expected outputs share), which the rendering alone would not show.
        let render = |types: &[V::Type]| types.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        let (expected, actual) = (render(expected), render(actual));
        let reason = if expected == actual { " with different type identities" } else { "" };
        return Err(TypeError::invalid(format!(
            "`{CUSTOM_FUNCTION_OPERATION_NAME}` `{name}` {rule} rule output type signature mismatch: expected \
             [{expected}] but got [{actual}]{reason}",
        ))
        .into());
    };
    if renaming.is_identity() {
        return Ok(program);
    }
    program.rename_type_identities(&renaming)
}

/// Returns the renaming of the type identities of `source` onto those of `target` when the two type signatures are
/// equal up to a bijective renaming of their identities, which holds exactly when each renames into the other.
pub(super) fn derive_bijective_identity_renaming<T: Type>(
    source: &[T],
    target: &[T],
) -> Option<TypeIdentityRenaming<T::Identity>> {
    let renames_into = |declared: &[T], actual: &[T]| {
        let renaming = T::derive_identity_renaming(declared, actual).ok()?;
        declared
            .iter()
            .zip(actual)
            .all(|(declared, actual)| declared.rename_identities(&renaming).is_ok_and(|renamed| &renamed == actual))
            .then_some(renaming)
    };
    if source.len() != target.len() {
        return None;
    }
    renames_into(target, source)?;
    renames_into(source, target)
}

/// Converts a failed or reentrant specialization of a retained custom rule into a [`DifferentiationError`], keeping
/// the rule's own error when it produced one.
fn specialization_error<E: Debug + Display + Into<DifferentiationError>>(
    error: SpecializationCacheError<E>,
) -> DifferentiationError {
    match error {
        SpecializationCacheError::Production(error) => error.into(),
        error @ SpecializationCacheError::Reentrant(_) => {
            ProgramError::UnsupportedOperation { message: error.to_string() }.into()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionBounds,
        DimensionType, DimensionVariable, ShardingDimension,
    };
    use crate::contexts::Context;
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::custom_functions::operations::CustomFunctionOperation;
    use crate::operations::custom_functions::tests::{
        MemberDefinition, RuleCounters, TestDefinition, call_jvp, cube_definition, cube_program, custom_rule_program,
        member_cube_definition,
    };
    use crate::programs::ReferenceType;

    use super::*;

    #[test]
    fn test_lifted_custom_rules() {
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(member_cube_definition(&counters));
        let lifted = LiftedCustomRules::Member(definition.reference());
        let source = |rules: &LiftedCustomRules<Array, ArrayOperation<Array>>| {
            <LiftedCustomRules<_, _> as CustomRuleSource<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::name(rules)
                .to_owned()
        };
        assert_eq!(source(&lifted), "cube");
        let key = |input_type: ArrayIrType, levels| CustomRuleSpecializationKey {
            input_types: vec![input_type.clone()],
            output_types: vec![input_type],
            non_differentiated_count: 0,
            tangent_activity: vec![true],
            levels,
            discharged: false,
        };
        let specialize = |rules: &LiftedCustomRules<Array, ArrayOperation<Array>>, key| {
            <LiftedCustomRules<_, _> as CustomRuleSpecializer<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::jvp_specialization(
                rules, key,
            )
            .map(|_| ())
            .unwrap_err()
            .to_string()
        };

        // Types without a member representation are rejected at the first derivative request.
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F64)));
        assert_eq!(
            specialize(&lifted, key(reference_type, Vec::new())),
            "`custom_function` `cube` cannot be differentiated with respect to type `ref<f64[]>`, which the family \
             in which its rules were registered cannot represent",
        );

        // So are first-class dimension batch extents, which array IR batching produces.
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let level = CustomRuleBatchingLevel {
            level: BatchingLevel::new(
                BatchingLevelExtent::Dynamic(DimensionType::from(items).into()),
                None,
                ShardingDimension::Replicated,
            ),
            boundary_input_count: 1,
            input_axes: vec![BatchAxis::new(0)],
            output_axes: vec![BatchAxis::new(0)],
        };
        assert_eq!(
            specialize(&lifted, key(ArrayType::scalar(DataType::F64).into(), vec![level])),
            "`custom_function` `cube` cannot differentiate a batched call whose batch extent has type \
             `dimension<items ∈ [1, 9)>`, which the family in which its rules were registered cannot represent; \
             differentiate the call before batching it instead",
        );

        // Unavailable rules keep the identity and description of their definition but reject every derivative request.
        let unavailable = LiftedCustomRules::<Array, ArrayOperation<Array>>::Unavailable(
            UnavailableCustomRules::from_source(&definition.reference()),
        );
        assert_eq!(unavailable, lifted);
        assert_eq!(source(&unavailable), "cube");
        assert_eq!(
            specialize(&unavailable, key(ArrayType::scalar(DataType::F64).into(), Vec::new())),
            "`custom_function` `cube` was converted from a family whose values its current family cannot represent, \
             so its derivative rules are unavailable",
        );
        assert_eq!(counters.counts(), (0, 0, 0));
    }

    #[test]
    fn test_lifted_custom_rules_derived() {
        // A lifted source derives in the family that owns its definition, and unavailable rules stay unavailable.
        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_jvp_from_primal()
                .with_batching()
                .with_batching_rule(|_, _, inputs, input_axes| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        let key = CustomRuleDerivationKey {
            kind: CustomRuleDerivationKind::Jvp,
            input_count: 1,
            active_input_indices: vec![0],
            output_tangent_mask: vec![true],
        };
        let derive = |rules: &LiftedCustomRules<Array, ArrayOperation<Array>>| {
            <LiftedCustomRules<_, _> as CustomRuleSpecializer<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::derived(
                rules,
                key.clone(),
            )
            .unwrap()
        };
        let lifted = LiftedCustomRules::Member(registration.reference());
        assert_eq!(derive(&lifted), LiftedCustomRules::Member(registration.reference().derived(key.clone()).unwrap()));
        let unavailable = LiftedCustomRules::<Array, ArrayOperation<Array>>::Unavailable(
            UnavailableCustomRules::from_source(&registration.reference()),
        );
        assert!(matches!(derive(&unavailable), LiftedCustomRules::Unavailable(_)));
        assert_eq!(derive(&unavailable), unavailable);
    }

    #[test]
    fn test_custom_rule_definition_batched_specialization_dynamic_extent() {
        // A first-class batch extent reaches batched specializations as a leading boundary input, so the traced rule is
        // batched at a level that records only the extent's type.
        let invocations = Arc::new(AtomicUsize::new(0));
        let definition = CustomRuleDefinition::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new("cube")
            .with_jvp({
                let invocations = invocations.clone();
                move |primals, tangents| {
                    invocations.fetch_add(1, Ordering::SeqCst);
                    let multiply = |left: &CustomRuleTracer<_, _>, right: &CustomRuleTracer<_, _>| {
                        let operation = ArrayIrOperation::Array(ArrayOperation::Mul(MulOperation::new()));
                        Ok::<_, ProgramError>(
                            left.context().bind(operation, Vec::new(), &[left.clone(), right.clone()])?.remove(0),
                        )
                    };
                    let square = multiply(&primals[0], &primals[0])?;
                    Ok((vec![multiply(&square, &primals[0])?], vec![multiply(&square, &tangents[0])?]))
                }
            })
            .with_batching();
        let item_type: ArrayIrType = ArrayType::new_static(DataType::F32, [3]).into();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let level = CustomRuleBatchingLevel {
            level: BatchingLevel::new(
                BatchingLevelExtent::Dynamic(DimensionType::from(items).into()),
                None,
                ShardingDimension::Replicated,
            ),
            boundary_input_count: 1,
            input_axes: vec![BatchAxis::new(0)],
            output_axes: vec![BatchAxis::new(0)],
        };
        let key = CustomRuleSpecializationKey {
            input_types: vec![item_type.clone()],
            output_types: vec![item_type],
            non_differentiated_count: 0,
            tangent_activity: vec![true],
            levels: vec![level],
            discharged: false,
        };
        let specialization = CustomRuleRegistration::new(definition).reference().jvp_specialization(key).unwrap();
        assert_eq!(specialization.output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            specialization.program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3], %2:f32[items, 3] .
                let %3:f32[items, 3] = mul %1 %1
                    %4:f32[items, 3] = mul %3 %1
                    %5:f32[items, 3] = mul %3 %2
                in (%4, %5)
            "}
            .trim_end(),
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_custom_rule_registration_ownership() {
        // The outer JVP rule calls the inner definition through a reference, so the outer rule's cached specialization
        // retains the inner definition but not its caches, which only the inner handle owns.
        let counters = Arc::new(RuleCounters::default());
        let inner = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let (retained_inner, retained_inner_caches) =
            (Arc::downgrade(&inner.definition()), Arc::downgrade(&inner.caches()));
        let inner_reference = inner.reference();
        let outer = CustomRuleRegistration::new(TestDefinition::new("outer").with_jvp(move |primals, tangents| {
            let r#type = primals[0].r#type().into_owned();
            let operation = CustomFunctionOperation::new(inner_reference.clone());
            let outputs = primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
            Ok((outputs, vec![tangents[0].clone()]))
        }));
        let (retained_outer, retained_outer_caches) =
            (Arc::downgrade(&outer.definition()), Arc::downgrade(&outer.caches()));
        let program = custom_rule_program(&outer, ArrayType::scalar(DataType::F64));
        let linearization = program.linearize().unwrap();
        assert_eq!(counters.counts(), (0, 0, 0));
        assert_eq!(outer.caches().jvp_specializations.len(), 1);

        // Dropping the handles frees their caches, including the outer specialization that called the inner definition,
        // while the staged programs keep both definitions alive until they are dropped as well.
        drop((inner, outer));
        assert!(retained_outer_caches.upgrade().is_none());
        assert!(retained_inner_caches.upgrade().is_none());
        assert!(retained_outer.upgrade().is_some());
        drop((program, linearization));
        assert!(retained_outer.upgrade().is_none());
        assert!(retained_inner.upgrade().is_none());
    }

    #[test]
    fn test_custom_rule_registration_ownership_self_recursive() {
        // The JVP rule calls its own definition through the weak handle of `new_cyclic`, so its cached specialization
        // retains the definition. Only the handle owns the caches, so dropping it frees that specialization, and the
        // definition itself lives exactly as long as the staged programs that call it.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new_cyclic(|this| {
            let (this, counters) = (this.clone(), counters.clone());
            TestDefinition::new("recursive").with_jvp(move |primals, tangents| {
                counters.jvp.fetch_add(1, Ordering::SeqCst);
                let r#type = primals[0].r#type().into_owned();
                let operation = CustomFunctionOperation::new(this.reference()?);
                let outputs =
                    primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
                Ok((outputs, vec![primals[0].clone() * primals[0].clone() * tangents[0].clone()]))
            })
        });
        let (retained_definition, retained_caches) =
            (Arc::downgrade(&definition.definition()), Arc::downgrade(&definition.caches()));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let linearization = program.linearize().unwrap();
        assert_eq!(counters.counts(), (1, 0, 0));
        assert_eq!(definition.caches().jvp_specializations.len(), 1);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = custom_function [name=\"recursive\"] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = mul %0 %0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                    %2:f64[] = mul %0 %0
                in (%1, %2)
            "}
            .trim_end(),
        );
        drop(definition);
        assert!(retained_caches.upgrade().is_none());
        assert!(retained_definition.upgrade().is_some());
        drop((program, linearization));
        assert!(retained_definition.upgrade().is_none());
    }

    #[test]
    fn test_custom_rule_registration_ownership_mutually_recursive() {
        // The rules of `a` call `b` through a reference and the rules of `b` call `a` through its weak handle, which is
        // how mutually recursive definitions are registered without a cycle: `b` is registered inside the `new_cyclic`
        // closure of `a`. Filling both caches (the second-order linearization traces the rule of `b`) and then
        // dropping both handles and every program must free every definition and cache.
        let mut b = None;
        let a = CustomRuleRegistration::new_cyclic(|a| {
            let a = a.clone();
            let registration =
                CustomRuleRegistration::new(TestDefinition::new("b").with_jvp(move |primals, tangents| {
                    let r#type = primals[0].r#type().into_owned();
                    let operation = CustomFunctionOperation::new(a.reference()?);
                    let outputs =
                        primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
                    Ok((outputs, vec![tangents[0].clone()]))
                }));
            let reference = registration.reference();
            b = Some(registration);
            TestDefinition::new("a").with_jvp(move |primals, tangents| {
                let r#type = primals[0].r#type().into_owned();
                let operation = CustomFunctionOperation::new(reference.clone());
                let outputs =
                    primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
                Ok((outputs, vec![tangents[0].clone()]))
            })
        });
        let b = b.unwrap();
        let probes = [
            (Arc::downgrade(&a.definition()), Arc::downgrade(&a.caches())),
            (Arc::downgrade(&b.definition()), Arc::downgrade(&b.caches())),
        ];
        let program = custom_rule_program(&a, ArrayType::scalar(DataType::F64));
        let first = program.linearize().unwrap();
        let second = first.primal().linearize().unwrap();
        assert_eq!(a.caches().jvp_specializations.len(), 1);
        assert_eq!(b.caches().jvp_specializations.len(), 1);
        drop((a, b));
        assert!(probes.iter().all(|(_, caches)| caches.upgrade().is_none()));
        drop((program, first, second));
        assert!(probes.iter().all(|(definition, _)| definition.upgrade().is_none()));
    }

    #[test]
    fn test_custom_rule_registration_drop() {
        // Once the handle is dropped, the remaining calls trace their rules again on every derivative request, with
        // results identical to the cached ones.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let cached = program.linearize().unwrap().tangent().to_string();
        drop(definition);
        for expected_invocations in [2, 3] {
            assert_eq!(program.linearize().unwrap().tangent().to_string(), cached);
            assert_eq!(counters.counts(), (expected_invocations, 0, 0));
        }

        // A rule that requests its own specialization while it is being traced is rejected, with and without a live
        // handle, and a rejected request leaves nothing in flight.
        let definition = CustomRuleRegistration::new_cyclic(|this| {
            let this = this.clone();
            TestDefinition::new("reentrant").with_jvp(move |primals, tangents| {
                let r#type = primals[0].r#type().into_owned();
                let key = CustomRuleSpecializationKey {
                    input_types: vec![r#type.clone()],
                    output_types: vec![r#type],
                    non_differentiated_count: 0,
                    tangent_activity: vec![true],
                    levels: Vec::new(),
                    discharged: false,
                };
                this.reference()?
                    .jvp_specialization(key)
                    .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                Ok((primals.to_vec(), tangents.to_vec()))
            })
        });
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let reentrant = "recursive request for a specialization that is already being produced on this thread";
        assert_eq!(program.jvp().unwrap_err().to_string(), reentrant);
        let retained = Arc::downgrade(&definition.definition());
        drop(definition);
        for _ in 0..2 {
            assert_eq!(program.jvp().unwrap_err().to_string(), reentrant);
            assert!(!retained.upgrade().unwrap().has_uncached_in_flight());
        }
    }

    #[test]
    fn test_custom_rule_reference_specialization_eviction() {
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let jvp = |extent: usize| {
            let x = Array::vector(vec![2f64; extent]).unwrap();
            let tangent = Array::vector(vec![1f64; extent]).unwrap();
            assert_eq!(
                call_jvp(&definition, x, tangent),
                Ok((
                    vec![Array::vector(vec![8f64; extent]).unwrap()],
                    vec![Array::vector(vec![4f64; extent]).unwrap()]
                )),
            );
        };
        let specialization = |extent: usize| {
            let r#type = ArrayType::new_static(DataType::F64, [extent]);
            let key = CustomRuleSpecializationKey {
                input_types: vec![r#type.clone()],
                output_types: vec![r#type],
                non_differentiated_count: 0,
                tangent_activity: vec![true],
                levels: Vec::new(),
                discharged: false,
            };
            Arc::downgrade(&definition.reference().jvp_specialization(key).unwrap())
        };
        let baseline = Arc::strong_count(&definition.definition());
        let mut probes = Vec::new();
        for extent in 1..=CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1 {
            jvp(extent);
            probes.push(specialization(extent));
        }
        assert_eq!(definition.caches().jvp_specializations.len(), CUSTOM_RULE_SPECIALIZATION_CAPACITY);
        assert_eq!(counters.counts(), (CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1, 0, 0));

        // Bounded caches bound the retained memory: the evicted specialization is freed, and no cached program retains
        // the definition.
        assert!(probes[0].upgrade().is_none());
        assert!(probes[1..].iter().all(|probe| probe.upgrade().is_some()));
        assert_eq!(Arc::strong_count(&definition.definition()), baseline);

        // The evicted oldest specialization is traced again, while a resident one is reused.
        jvp(1);
        jvp(CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1);
        assert_eq!(counters.counts(), (CUSTOM_RULE_SPECIALIZATION_CAPACITY + 2, 0, 0));
    }

    #[test]
    fn test_custom_rule_reference_specialization_retry_after_failure() {
        // A failed specialization is not retained, so the next request invokes the rule again.
        let attempts = Arc::new(AtomicUsize::new(0));
        let definition = CustomRuleRegistration::new(TestDefinition::new("flaky").with_jvp({
            let attempts = attempts.clone();
            move |primals, tangents| {
                if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(ProgramError::InvalidArgument { message: "transient failure".to_string() });
                }
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
            }
        }));
        let x = Array::scalar(2f64).unwrap();
        let one = Array::scalar(1f64).unwrap();
        assert_eq!(call_jvp(&definition, x.clone(), one.clone()).unwrap_err().to_string(), "transient failure");
        assert!(definition.caches().jvp_specializations.is_empty());
        for _ in 0..2 {
            assert_eq!(
                call_jvp(&definition, x.clone(), one.clone()),
                Ok((vec![Array::scalar(8f64).unwrap()], vec![Array::scalar(4f64).unwrap()])),
            );
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_custom_rule_reference_specialization_concurrent_cold_requests() {
        // Both threads enter the rule before either finishes, so both derive the same cold specialization. The
        // duplicate derivations produce one resident result, which later requests reuse.
        let barrier = Arc::new(Barrier::new(2));
        let invocations = Arc::new(AtomicUsize::new(0));
        let definition = CustomRuleRegistration::new(TestDefinition::new("concurrent").with_jvp({
            let barrier = barrier.clone();
            let invocations = invocations.clone();
            move |primals, tangents| {
                if invocations.fetch_add(1, Ordering::SeqCst) < 2 {
                    barrier.wait();
                }
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
            }
        }));
        let x = Array::scalar(2f64).unwrap();
        let one = Array::scalar(1f64).unwrap();
        let expected = (vec![Array::scalar(8f64).unwrap()], vec![Array::scalar(4f64).unwrap()]);
        std::thread::scope(|scope| {
            let threads = (0..2)
                .map(|_| {
                    let (definition, x, one) = (definition.clone(), x.clone(), one.clone());
                    scope.spawn(move || call_jvp(&definition, x, one))
                })
                .collect::<Vec<_>>();
            for thread in threads {
                assert_eq!(thread.join().unwrap(), Ok(expected.clone()));
            }
        });
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
        assert_eq!(definition.caches().jvp_specializations.len(), 1);
        assert_eq!(call_jvp(&definition, x, one), Ok(expected));
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_custom_rule_reference_specialization_debug_recheck() {
        // A region-transform cache hit re-derives the linearization in debug builds and compares renderings. The
        // re-derivation reuses the retained specialization, so it neither invokes the rule nor changes the rendering.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let first = program.linearize().unwrap();
        let second = program.linearize().unwrap();
        assert_eq!(second.tangent().to_string(), first.tangent().to_string());
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_custom_rule_reference_derived() {
        // Derivations of one key share a definition that the source registration owns, together with its caches, and
        // the definition keeps only its source definition alive (i.e., there is no strong cycle back to the caches).
        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_jvp_from_primal()
                .with_batching()
                .with_batching_rule(|_, _, inputs, input_axes| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        let key = |kind| CustomRuleDerivationKey {
            kind,
            input_count: 1,
            active_input_indices: vec![0],
            output_tangent_mask: vec![true],
        };
        let derived = registration.reference().derived(key(CustomRuleDerivationKind::Jvp)).unwrap();
        assert_eq!(derived.name(), "jvp(sine)");
        assert!(derived.derives_jvp_from_primal());
        assert!(derived.has_batching_rule());
        assert!(!derived.has_jvp());
        assert!(!derived.has_vjp());
        assert_eq!(registration.reference().derived(key(CustomRuleDerivationKind::Jvp)).unwrap(), derived);
        let pushforward = registration.reference().derived(key(CustomRuleDerivationKind::Pushforward)).unwrap();
        assert_eq!(pushforward.name(), "pushforward(sine)");
        assert_ne!(pushforward, derived);
        assert_eq!(registration.caches().derived_registrations.lock().unwrap().len(), 2);
        assert!(derived.caches.upgrade().is_some());

        // Dropping the source registration frees the derived registrations and their caches, and later derivations
        // produce fresh definitions without caches, exactly like specializations.
        let source = registration.reference();
        drop(registration);
        assert!(derived.caches.upgrade().is_none());
        let fresh = source.derived(key(CustomRuleDerivationKind::Jvp)).unwrap();
        assert_ne!(fresh, derived);
        assert!(fresh.caches.upgrade().is_none());
        let derived_definition = Arc::downgrade(&derived.definition);
        drop((derived, pushforward, fresh));
        assert!(derived_definition.upgrade().is_none());

        // A source without a custom batching rule has nothing to derive from.
        assert_eq!(
            CustomRuleRegistration::new(MemberDefinition::new("sine").with_jvp_from_primal())
                .reference()
                .derived(key(CustomRuleDerivationKind::Jvp))
                .unwrap_err()
                .to_string(),
            "encountered malformed program: `custom_function` `sine` has no custom batching rule from which to derive \
             rules",
        );
    }
}
