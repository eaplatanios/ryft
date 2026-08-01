//! Contains the named-axis [`CollectiveOperation`], which reduces a value across a named axis (`psum` / `pmean` /
//! `pmax`), together with its interpretation, partial-evaluation, batching, forward-mode differentiation, and
//! transposition rules. These are the analogues of
//! [JAX's parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators) `jax.lax.psum`,
//! `jax.lax.pmean`, and `jax.lax.pmax`.
//!
//! Collectives reference an enclosing named-axis binder by name, validated against the active
//! [`NamedAxes`] environment at staging time. A name bound by an enclosing `batch` level is
//! resolved at trace time by the operations' batching rules, which collapse or materialize the mapped batch axis at
//! the binding level, while a name bound to a device mesh axis by a `shard_map` manual region stays in the staged
//! body and lowers to cross-device collectives over that mesh axis.

use std::fmt::{Debug, Display};
use std::ops::Mul;

use crate::axes::{AxisError, NamedAxes, NamedAxis};
use crate::backends::dimensions::DimensionValue;
use crate::backends::scalars::Scalar;
use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver,
    BatchingError,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::constants::{Fill, ZeroLike};
use crate::operations::dimensions::{DimensionDivFloor, DimensionMul, DimensionRequirement, DimensionSize};
use crate::operations::manipulation::slicing::resized_output_sharding;
use crate::operations::manipulation::{Concatenate, LegacyBroadcast, Reshape, Slice, Transpose};
use crate::operations::math::{Reduce, ReductionKind};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::{ProjectedValue, ValueProjection};
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::sharding::ShardingDimension;
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayProgramType, ArrayType, DataType, Dimension, DimensionType, Shape};

// TODO(eaplatanios): Review this module.

/// Kind of collective performed by a [`CollectiveOperation`].
///
/// Collectives operate on a named axis, resolved against the active [`NamedAxes`] environment. When an enclosing
/// `batch` level binds the name (a [`NamedAxis::Batched`](crate::axes::NamedAxis) axis), the matching
/// [`BatchingContext`] consumes the mapped batch axis at trace time.
/// When a `shard_map` manual region binds the name to a device mesh axis (a
/// [`NamedAxis::Mesh`](crate::axes::NamedAxis) axis), the collective stays in the staged body and lowers to a
/// cross-device `all_reduce` over that mesh axis. The operations described here mirror JAX's
/// `jax.lax.{psum, pmean, pmax}` family.
///
/// `PSum`/`PMean`/`PMax` reduce the named axis away, producing a result that is identical across all batch items or
/// device shards (replicated). Shape-changing collectives use their dedicated operation payloads below because their
/// result types also depend on a ranked array axis and on whether the named axis is materialized or tiled.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CollectiveKind {
    /// Sum reduction across the named axis (`jax.lax.psum`).
    PSum,

    /// Mean reduction across the named axis (`jax.lax.pmean`).
    PMean,

    /// Maximum reduction across the named axis (`jax.lax.pmax`).
    PMax,
}

/// Shape semantics used by collectives that can either materialize a named axis or tile an existing array axis.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CollectiveMode {
    /// Materializes the named axis as a new ranked array dimension, or consumes one ranked dimension when scattering.
    #[default]
    Untiled,

    /// Preserves array rank by multiplying or dividing an existing ranked array dimension.
    Tiled,
}

/// Named-axis variance carried by an all-gather result.
///
/// This is an operation option rather than parallel type metadata. Type inference maps it onto the canonical
/// [`Sharding::varying_manual_axes`](crate::Sharding::varying_manual_axes) and
/// [`Sharding::reduced_axes`](crate::Sharding::reduced_axes) sets.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum AllGatherOutputVariance {
    /// The result continues to vary across the gathered manual mesh axis.
    #[default]
    Varying,

    /// The result is invariant across the gathered manual mesh axis.
    Invariant,

    /// The result records the gathered manual mesh axis as reduced.
    Reduced,
}

/// Shared shape and grouping options for all-gather, sum-scatter, and all-to-all.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct CollectiveOptions {
    /// Rank-changing or rank-preserving shape semantics.
    mode: CollectiveMode,

    /// Optional ordered partition of logical participant indices.
    axis_index_groups: Option<Vec<Vec<usize>>>,
}

impl Debug for CollectiveOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.axis_index_groups {
            None => Debug::fmt(&self.mode, formatter),
            Some(axis_index_groups) => formatter
                .debug_struct("CollectiveOptions")
                .field("mode", &self.mode)
                .field("axis_index_groups", axis_index_groups)
                .finish(),
        }
    }
}

impl CollectiveOptions {
    /// Creates collective options for `mode` with no participant subgroups.
    #[inline]
    pub fn new(mode: CollectiveMode) -> Self {
        Self { mode, axis_index_groups: None }
    }

    /// Creates rank-preserving tiled collective options with no participant subgroups.
    #[inline]
    pub fn tiled() -> Self {
        Self::new(CollectiveMode::Tiled)
    }

    /// Returns these options with the provided ordered participant groups.
    #[inline]
    pub fn with_axis_index_groups(mut self, axis_index_groups: Vec<Vec<usize>>) -> Self {
        self.axis_index_groups = Some(axis_index_groups);
        self
    }

    /// Returns the selected shape mode.
    #[inline]
    pub fn mode(&self) -> CollectiveMode {
        self.mode
    }

    /// Returns the ordered participant groups, if any.
    #[inline]
    pub fn axis_index_groups(&self) -> Option<&[Vec<usize>]> {
        self.axis_index_groups.as_deref()
    }

    /// Validates these options against the full named-axis size and returns the effective group size used for shape
    /// arithmetic.
    fn effective_axis_size(&self, operation_name: &str, axis_size: usize) -> Result<usize, TypeError> {
        effective_collective_axis_size(operation_name, axis_size, self.axis_index_groups())
    }
}

/// Validates an optional ordered participant partition and returns its effective group size without copying it.
fn effective_collective_axis_size(
    operation_name: &str,
    axis_size: usize,
    groups: Option<&[Vec<usize>]>,
) -> Result<usize, TypeError> {
    validate_collective_axis_size(operation_name, axis_size)?;
    let Some(groups) = groups else {
        return Ok(axis_size);
    };
    let Some(first_group) = groups.first() else {
        return Err(TypeError::invalid(format!("'{operation_name}' axis index groups must not be empty")));
    };
    if first_group.is_empty() {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' axis index groups must contain at least one participant",
        )));
    }
    let group_size = first_group.len();
    let mut seen = vec![false; axis_size];
    for (group_index, group) in groups.iter().enumerate() {
        if group.len() != group_size {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' axis index group {group_index} has size {} but every group must have size \
                     {group_size}",
                group.len(),
            )));
        }
        for &participant in group {
            let Some(participant_seen) = seen.get_mut(participant) else {
                return Err(TypeError::invalid(format!(
                    "'{operation_name}' axis index {participant} is out of bounds for axis size {axis_size}",
                )));
            };
            if *participant_seen {
                return Err(TypeError::invalid(format!(
                    "'{operation_name}' axis index groups contain participant {participant} more than once",
                )));
            }
            *participant_seen = true;
        }
    }
    if let Some(missing) = seen.iter().position(|seen| !seen) {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' axis index groups do not contain participant {missing}",
        )));
    }
    Ok(group_size)
}

impl CollectiveKind {
    /// Returns the canonical operation name suffix for this kind.
    pub fn name(self) -> &'static str {
        match self {
            Self::PSum => "psum",
            Self::PMean => "pmean",
            Self::PMax => "pmax",
        }
    }

    /// Returns the [`ReductionKind`] used to collapse the named axis.
    pub fn reduction_kind(self) -> ReductionKind {
        match self {
            Self::PSum | Self::PMean => ReductionKind::Sum,
            Self::PMax => ReductionKind::Max,
        }
    }
}

impl Display for CollectiveKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Primitive representing one named-axis collective operation.
///
/// [`CollectiveOperation`] is identity at the per-item level (the named axis does not exist in
/// per-item semantics) and collapses the mapped axis when invoked inside a
/// [`BatchingContext`] whose
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches this collective's axis name. Under
/// nested `batch` levels, the batching rule below owns that decision: a matching level consumes the mapped
/// batch axis, while a non-matching level forwards the collective untouched to its parent context via
/// [`forward_collective_to_parent`], where the next level repeats the same name resolution.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CollectiveOperation {
    /// Axis name referenced by this collective. Matches the `axis_name` argument of an enclosing
    /// [`BatchingContext::new`](crate::batching::BatchingContext::new) call.
    axis_name: String,

    /// Kind of collective.
    kind: CollectiveKind,

    /// Full size of the named axis when participant subgroups are present.
    axis_size: Option<usize>,

    /// Optional ordered partition of logical participant indices.
    axis_index_groups: Option<Vec<Vec<usize>>>,
}

impl CollectiveOperation {
    /// Creates a new [`CollectiveOperation`] with the supplied axis name and kind.
    #[inline]
    pub fn new(axis_name: String, kind: CollectiveKind) -> Self {
        Self { axis_name, kind, axis_size: None, axis_index_groups: None }
    }

    /// Creates a grouped collective after validating that `axis_index_groups` is an equal-sized exact partition of
    /// `0..axis_size`.
    pub fn grouped(
        axis_name: String,
        kind: CollectiveKind,
        axis_size: usize,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, TypeError> {
        effective_collective_axis_size(kind.name(), axis_size, Some(axis_index_groups.as_slice()))?;
        Ok(Self { axis_name, kind, axis_size: Some(axis_size), axis_index_groups: Some(axis_index_groups) })
    }

    /// Returns the axis name referenced by this collective.
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }

    /// Returns the kind of collective.
    #[inline]
    pub fn kind(&self) -> CollectiveKind {
        self.kind
    }

    /// Returns the full named-axis size recorded for a grouped collective.
    #[inline]
    pub fn axis_size(&self) -> Option<usize> {
        self.axis_size
    }

    /// Returns the ordered participant groups, if any.
    #[inline]
    pub fn axis_index_groups(&self) -> Option<&[Vec<usize>]> {
        self.axis_index_groups.as_deref()
    }

    /// Returns the number of participants in each group, or `None` for an ungrouped collective whose size is supplied
    /// by the enclosing binder.
    pub fn group_size(&self) -> Option<usize> {
        self.axis_index_groups.as_ref().and_then(|groups| groups.first()).map(Vec::len)
    }
}

impl Display for CollectiveOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for CollectiveOperation {
    #[inline]
    fn name(&self) -> &'static str {
        match self.kind {
            CollectiveKind::PSum => "psum",
            CollectiveKind::PMean => "pmean",
            CollectiveKind::PMax => "pmax",
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match (&self.axis_index_groups, self.axis_size) {
            (Some(axis_index_groups), Some(axis_size)) => {
                effective_collective_axis_size(self.name(), axis_size, Some(axis_index_groups.as_slice()))?;
            }
            (None, None) => {}
            _ => {
                return Err(TypeError::invalid(format!(
                    "'{}' must store both the full axis size and axis index groups, or neither",
                    self.name(),
                )));
            }
        }
        // The per-item operation is identity; the named axis only exists physically inside an
        // enclosing `BatchingContext` where the batching rule will collapse it.
        Ok(vec![input_types[0].clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
            if let Some(axis_size) = self.axis_size {
                operation.field("axis_size", axis_size)?;
            }
            if let Some(axis_index_groups) = &self.axis_index_groups {
                operation.field("axis_index_groups", format_args!("{axis_index_groups:?}"))?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for CollectiveOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // Per-item interpretation is identity: the named axis does not exist in per-item semantics, so reducing across
        // it is a no-op. Staging a collective over an unbound axis is now rejected up front (see `Collective`), so an
        // enclosing binder always collapses the mapped axis through the batching rule before this per-item fallback
        // matters; it remains defined for a program interpreted directly, where a collective simply passes its operand
        // through per item.
        Ok(vec![inputs[0].clone()])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for CollectiveOperation where
    C::Operation: From<CollectiveOperation>
{
}

/// Batching rule for [`CollectiveOperation`]. This rule owns named-axis resolution: when the active context's
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches this collective's axis name, the mapped batch
/// axis is consumed; otherwise the collective targets an outer `batch` level (or a device mesh axis) and is forwarded
/// untouched to the parent context via [`forward_collective_to_parent`].
///
/// The consuming arm collapses the mapped axis through `collective_reduce_batch` and binds a `PMean`'s `1 / N`
/// rank-0 fill into the parent context — interpreted eagerly under an eager parent and staged into the enclosing
/// trace under a staging parent — so one rule serves eager and staged batching alike.
impl<C, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for CollectiveOperation
where
    C: Context<Type = ArrayType> + Fill<Scalar, C::Value>,
    C::Operation: From<CollectiveOperation>,
    <C as Domain>::Value: Reduce + Mul<Output = <C as Domain>::Value>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            let parent_operation = C::Operation::from(self.clone());
            return forward_collective_to_parent(context, parent_operation, inputs);
        }
        if self.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "'{}' axis index groups are not supported when a batch transform binds the collective axis",
                    self.name(),
                ),
            });
        }
        collective_reduce_batch(self.kind, inputs, |factor_type, inverse_axis_size| {
            // The `1 / N` rank-0 factor binds into the batching context's parent — interpreted eagerly under an eager
            // parent, staged into the enclosing trace under a staging parent.
            context.parent().fill(&factor_type, Scalar::from(inverse_axis_size))
        })
    }
}

/// Re-stages a collective that targets a different (outer) named axis into the batching context's parent.
///
/// Under nested `batch` levels, a collective is consumed by the level whose
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches its axis name and must pass through
/// every inner level untouched: each inner batch item participates in the outer collective independently, so the
/// operands' mapped axes are preserved as-is on the forwarded outputs. The parent may itself be another
/// [`BatchingContext`] — whose own rule dispatch repeats this name
/// resolution at the next level — or an ordinary tracing context. Batching rules for custom collective-like
/// operations should use this helper for their "not my axis" arm.
pub fn forward_collective_to_parent<C, P: ArrayBatchingPolicy<C>>(
    context: &BatchingContext<C, ArrayBatching<P>>,
    parent_operation: C::Operation,
    inputs: &[ArrayBatch<<C as Domain>::Value>],
) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
{
    let parent_input_values: Vec<<C as Domain>::Value> = inputs.iter().map(|batch| batch.value().clone()).collect();
    let parent_outputs = context.parent().bind(parent_operation, Vec::new(), &parent_input_values)?;
    check_count!("output", parent_outputs, inputs.len(), ProgramError);
    parent_outputs
        .into_iter()
        .zip(inputs.iter())
        .map(|(parent_value, input_batch)| {
            let physical_type = parent_value.r#type().into_owned();
            ArrayBatch::new(physical_type, parent_value, input_batch.batch_axis())
        })
        .collect()
}

/// Shared reduce-and-optionally-mean skeleton for [`CollectiveOperation`] batching. It collapses the mapped batch
/// axis with the kind's [`ReductionKind`] and, for `PMean`, scales the replicated result by `1 / N` using a
/// `make_pmean_factor`-produced rank-0 factor (relying on implicit rank-0 broadcasting in the multiplication).
/// Outside a matching batching context (no mapped axis), it is an identity pass-through.
fn collective_reduce_batch<V, MakePMeanFactor>(
    kind: CollectiveKind,
    inputs: &[ArrayBatch<V>],
    make_pmean_factor: MakePMeanFactor,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    V: Value<Type = ArrayType> + Reduce + Mul<Output = V>,
    MakePMeanFactor: FnOnce(ArrayType, f64) -> Result<V, ProgramError>,
{
    check_count!("input", inputs, 1, ProgramError);
    let input = &inputs[0];
    let Some(batch_axis) = input.batch_axis_position() else {
        // Outside any matching batching context: identity pass-through.
        return Ok(vec![input.clone()]);
    };
    // Reduce along the mapped batch axis with the corresponding reduction kind. The output is replicated: every
    // batch item sees the same reduced value, matching JAX's `psum`/`pmean`/`pmax` broadcast semantics.
    let mut output_value = input.value().clone().reduce(&[batch_axis], kind.reduction_kind());
    if matches!(kind, CollectiveKind::PMean) {
        // PMean divides the summed value by the batch size, which must be statically known to scale by `1 / N`.
        let inverse_axis_size = 1.0 / pmean_batch_size(input)? as f64;
        let factor_type = pmean_factor_type(output_value.r#type().data_type());
        output_value = make_pmean_factor(factor_type, inverse_axis_size)? * output_value;
    }
    let output_type = output_value.r#type().into_owned();
    Ok(vec![ArrayBatch::new(output_type, output_value, BatchAxis::replicated())?])
}

/// Returns the static batch size for a `PMean` over the mapped batch axis of `input`, erroring when
/// the batch size is dynamic (a mean cannot be scaled by `1 / N` without a static `N`).
fn pmean_batch_size<V: Value<Type = ArrayType>>(input: &ArrayBatch<V>) -> Result<usize, BatchingError> {
    input.batch_size()?.ok_or_else(|| BatchingError::UnsupportedOperation {
        message: "pmean requires a static batch size; the staged batch axis is dynamic".to_string(),
    })
}

/// Builds the rank-0 [`ArrayType`] of `data_type` used to hold a `PMean`'s `1 / N` factor.
fn pmean_factor_type(data_type: DataType) -> ArrayType {
    ArrayType::new(data_type, Shape::scalar())
}

impl_differentiable_operation! {
    CollectiveOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<CollectiveOperation>,
    {
        |operation, context, _driver, inputs| {
            // Forward-mode (JVP) rule for [`CollectiveOperation`]. `PSum`/`PMean` are linear and self-adjoint, so the
            // tangent is the same collective applied to the operand tangent: `tangent_out =
            // collective(input.tangent())`. A structural-zero operand tangent is preserved as-is rather than staging a
            // collective on a zero, keeping `collective(zero)` out of the tangent program. `PMax` is non-linear and
            // reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
            check_count!("input", inputs, 1, ProgramError);
            if matches!(operation.kind, CollectiveKind::PMax) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "pmax differentiation is not yet supported".to_string(),
                }
                .into());
            }
            let primal = stage_collective(context, operation, inputs[0].primal())?;
            // A collective of a structural zero stays a structural zero, keeping `collective(zero)` out of the
            // tangent program.
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
                MaybeZero::Value(tangent) => MaybeZero::Value(stage_collective(context, operation, tangent)?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<ArrayType> + From<CollectiveOperation>,
    {
        |operation, context, _driver, inputs, outputs| {
            // Transpose rule for [`CollectiveOperation`]. `psum`/`pmean` are self-adjoint, so the operand cotangent is
            // the same collective applied to the output cotangent. The single operand is linear (its [`PartialValue`]
            // is [`Unknown`](PartialValue::Unknown)); a known operand contributes no cotangent and so receives a
            // structural zero. `PMax` reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            if matches!(operation.kind, CollectiveKind::PMax) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "pmax transpose is not yet supported".to_string(),
                }
                .into());
            }
            // A known (non-linear) operand contributes no cotangent.
            if inputs[0].is_known() {
                return Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]);
            }
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let contribution = stage_collective(context, operation, cotangent)?;
                    Ok(vec![MaybeZero::Value(contribution)])
                }
                MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
            }
        }
    },
}

/// Re-stages this collective of the same axis name and kind on a single tracer operand, returning its single output.
///
/// Both the forward-mode (`jvp`) and the transpose rules below re-stage the collective on a tracer (the primal, the
/// tangent, or the output cotangent), which is exactly one operation with one input and one output.
fn stage_collective<C>(
    context: &C,
    operation: &CollectiveOperation,
    operand: &C::Value,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayType>,
    C::Operation: From<CollectiveOperation>,
{
    let mut outputs = context.bind(operation.clone(), Vec::new(), std::slice::from_ref(operand))?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Value-level entry point for staging a collective operation.
///
/// The staged operation references an enclosing named-axis binder by name and the name is validated against the active
/// [`NamedAxes`] environment at staging time: an unbound name fails fast rather than silently acting as identity. A
/// name bound by an enclosing `batch` level is collapsed at trace time by
/// [`BatchableOperation::batch`] (which reduces the mapped batch axis),
/// while a name bound to a device mesh axis by a `shard_map` manual region stays in the staged body program and lowers
/// to a cross-device `all_reduce` over that mesh axis.
pub trait Collective: Sized {
    /// Stages a collective of the given kind referencing axis `axis_name`, validating that the name is bound by an
    /// enclosing transform. Returns [`AxisError::UnboundAxisName`] (surfaced as
    /// [`BatchingError::Axis`] riding a [`ProgramError::Custom`] payload) when no
    /// enclosing binder binds `axis_name`.
    fn collective(&self, axis_name: &str, kind: CollectiveKind) -> Result<Self, ProgramError>;

    /// Stages a grouped collective after validating that the groups cover the named axis exactly once.
    fn collective_with_axis_index_groups(
        &self,
        axis_name: &str,
        kind: CollectiveKind,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError>;
}

/// Any context-carrying value applies a collective by validating the axis name against the active [`NamedAxes`]
/// environment and binding a [`CollectiveOperation`] through its own context: a staged tracer records the operation,
/// a batching tracer resolves the named axis against the batching context stack, and a JVP dual forwards to the
/// primal-side resolution. An unbound name fails fast with [`AxisError::UnboundAxisName`] rather than silently acting
/// as identity.
impl<V: Value> Collective for V
where
    V::DispatchDomain: Context + NamedAxes,
    <V::DispatchDomain as Domain>::Operation: From<CollectiveOperation>,
{
    fn collective(&self, axis_name: &str, kind: CollectiveKind) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        if context.named_axis(axis_name).is_none() {
            return Err(BatchingError::Axis(AxisError::UnboundAxisName { name: axis_name.to_string() }).into());
        }
        let mut outputs =
            context.bind(CollectiveOperation::new(axis_name.to_string(), kind), Vec::new(), &[self.clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    fn collective_with_axis_index_groups(
        &self,
        axis_name: &str,
        kind: CollectiveKind,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let operation = CollectiveOperation::grouped(axis_name.to_string(), kind, axis_size, axis_index_groups)?;
        let mut outputs = context.bind(operation, Vec::new(), &[self.clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Resolves the size of the named axis bound by the active [`NamedAxes`] environment, failing fast with
/// [`AxisError::UnboundAxisName`] when no enclosing binder binds `axis_name`. The shape-changing collective
/// capabilities bake the resolved size into their operation payloads at staging time, because their output shapes
/// depend on it while [`Operation::infer_output_types`] only sees input types.
fn resolve_named_axis_size<C: NamedAxes>(context: &C, axis_name: &str) -> Result<usize, ProgramError> {
    match context.named_axis(axis_name) {
        Some(NamedAxis::Batched { size: Some(size) } | NamedAxis::Mesh { size, .. }) if size > 0 => Ok(size),
        Some(NamedAxis::Batched { size: Some(_) } | NamedAxis::Mesh { .. }) => {
            Err(TypeError::invalid(format!("collective axis '{axis_name}' must contain at least one participant",))
                .into())
        }
        Some(NamedAxis::Batched { size: None }) => Err(BatchingError::UnsupportedOperation {
            message: format!(
                "collective axis '{axis_name}' has a dynamic extent that must remain a first-class operand"
            ),
        }
        .into()),
        None => Err(BatchingError::Axis(AxisError::UnboundAxisName { name: axis_name.to_string() }).into()),
    }
}

/// Rejects an invalid zero-participant collective before any multiplication, division, or remainder operation.
pub(crate) fn validate_collective_axis_size(operation_name: &str, axis_size: usize) -> Result<(), TypeError> {
    if axis_size == 0 {
        Err(TypeError::invalid(format!("'{operation_name}' axis size must be greater than zero")))
    } else {
        Ok(())
    }
}

/// Validates the shared operand contract of the shape-changing collectives (exactly one operand with no unreduced
/// axes) and returns the operand's static dimensions.
fn shape_changing_collective_dimensions(
    operation_name: &str,
    input_types: &[ArrayType],
) -> Result<Vec<usize>, TypeError> {
    check_count!("input", input_types, 1, TypeError);
    if !input_types[0].unreduced_axes().is_empty() {
        return Err(TypeError::invalid(format!("'{operation_name}' does not support unreduced operands")));
    }
    let Some(shape) = input_types[0].static_shape() else {
        return Err(TypeError::invalid(format!("'{operation_name}' does not support dynamically shaped operands")));
    };
    Ok(shape.dimensions().to_vec())
}

/// Builds a shape-changing collective's output type from its operand and resized dimensions, carrying the operand
/// sharding through with the same per-dimension placement (the dimension count never changes).
fn shape_changing_collective_output_type(
    operation_name: &'static str,
    input_type: &ArrayType,
    output_dimensions: Vec<usize>,
) -> Result<ArrayType, TypeError> {
    let output_sizes = output_dimensions.into_iter().map(Dimension::Static).collect::<Vec<_>>();
    let sharding = resized_output_sharding(input_type, output_sizes.as_slice(), operation_name)?;
    let mut output_type =
        ArrayType::new(input_type.data_type(), Shape::new(output_sizes)).with_memory(input_type.memory());
    output_type.sharding = sharding;
    Ok(output_type)
}

/// Applies an all-gather's named-axis variance transition to the canonical sharding metadata.
fn all_gather_output_type(
    input_type: &ArrayType,
    mut output_type: ArrayType,
    operation: &AllGatherOperation,
) -> Result<ArrayType, TypeError> {
    let Some(input_sharding) = input_type.sharding() else {
        if operation.output_variance == AllGatherOutputVariance::Reduced {
            return Err(TypeError::invalid(
                "'all_gather' with reduced output variance requires sharding metadata".to_string(),
            ));
        }
        return Ok(output_type);
    };
    if input_type.unreduced_axes().contains(operation.axis_name()) {
        return Err(TypeError::invalid(format!(
            "'all_gather' does not support an operand that is unreduced over axis '{}'",
            operation.axis_name(),
        )));
    }

    let mut varying_axes = input_sharding.varying_manual_axes().clone();
    let mut reduced_axes = input_sharding.reduced_axes().clone();
    match operation.output_variance {
        AllGatherOutputVariance::Varying => {
            if reduced_axes.contains(operation.axis_name()) {
                return Err(TypeError::invalid(format!(
                    "'all_gather' cannot make axis '{}' varying because the operand records it as reduced",
                    operation.axis_name(),
                )));
            }
            varying_axes.insert(operation.axis_name.clone());
        }
        AllGatherOutputVariance::Invariant => {
            if reduced_axes.contains(operation.axis_name()) {
                return Err(TypeError::invalid(format!(
                    "'all_gather' cannot make axis '{}' invariant because the operand records it as reduced",
                    operation.axis_name(),
                )));
            }
            varying_axes.remove(operation.axis_name());
        }
        AllGatherOutputVariance::Reduced => {
            if !varying_axes.remove(operation.axis_name()) {
                return Err(TypeError::invalid(format!(
                    "'all_gather' with reduced output variance requires an operand varying over axis '{}'",
                    operation.axis_name(),
                )));
            }
            if !reduced_axes.insert(operation.axis_name.clone()) {
                return Err(TypeError::invalid(format!(
                    "'all_gather' operand is already reduced over axis '{}'",
                    operation.axis_name(),
                )));
            }
        }
    }
    let output_sharding = output_type.sharding().expect("shape projection preserves sharding").clone();
    let output_sharding = output_sharding
        .with_varying_manual_axes(varying_axes)
        .and_then(|sharding| sharding.with_reduced_axes(reduced_axes))
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    output_type.sharding = Some(output_sharding);
    Ok(output_type)
}

/// Infers one canonical mixed collective result from an array operand followed by one explicit extent per output
/// axis.
fn infer_explicit_shape_changing_collective_output_type(
    operation_name: &'static str,
    input_types: &[ArrayProgramType],
    base_output_type: ArrayType,
    unchanged_input_axes: &[Option<usize>],
    validate_exact_extents: impl FnOnce(&ArrayType, &[Dimension]) -> Result<(), TypeError>,
) -> Result<Vec<ArrayProgramType>, TypeError> {
    let expected = 1 + base_output_type.rank();
    check_count!("input", input_types, expected, TypeError);
    let input_type = <&ArrayType>::try_from(&input_types[0])?;
    if !input_type.unreduced_axes().is_empty() {
        return Err(TypeError::invalid(format!("'{operation_name}' does not support unreduced operands")));
    }
    let output_extents = input_types[1..]
        .iter()
        .map(|r#type| <&DimensionType>::try_from(r#type).map(DimensionType::to_dimension))
        .collect::<Result<Vec<_>, _>>()?;
    if unchanged_input_axes.len() != output_extents.len() {
        return Err(TypeError::invalid(format!(
            "'{operation_name}' internal output-axis mapping has length {} but the result rank is {}",
            unchanged_input_axes.len(),
            output_extents.len(),
        )));
    }
    for (output_axis, (&input_axis, output_extent)) in unchanged_input_axes.iter().zip(&output_extents).enumerate() {
        let Some(input_axis) = input_axis else { continue };
        let input_extent = input_type.shape().dimensions().get(input_axis).ok_or_else(|| {
            TypeError::invalid(format!(
                "'{operation_name}' unchanged output axis {output_axis} references input axis {input_axis}, which is \
                 out of bounds for rank {}",
                input_type.rank(),
            ))
        })?;
        if output_extent != input_extent {
            return Err(TypeError::invalid(format!(
                "'{operation_name}' output axis {output_axis} extent {output_extent} must equal unchanged input axis \
                 {input_axis} extent {input_extent}",
            )));
        }
    }
    validate_exact_extents(input_type, output_extents.as_slice())?;
    Ok(vec![base_output_type.with_shape(Shape::new(output_extents)).into()])
}

/// Infers the composite all-gather contract.
pub(crate) fn infer_explicit_all_gather_output_types(
    operation: &AllGatherOperation,
    input_types: &[ArrayProgramType],
) -> Result<Vec<ArrayProgramType>, TypeError> {
    let effective_axis_size = operation.effective_axis_size()?;
    let Some(input_type) = input_types.first() else {
        return Err(TypeError::invalid("'all_gather' expects an array followed by its output extents"));
    };
    let input_type = <&ArrayType>::try_from(input_type)?;
    let (base_output_type, unchanged_input_axes) = match operation.options.mode {
        CollectiveMode::Untiled => (
            input_type.with_inserted_dimension(operation.concat_axis, Dimension::Static(effective_axis_size))?,
            (0..=input_type.rank())
                .map(|axis| {
                    if axis == operation.concat_axis {
                        None
                    } else if axis < operation.concat_axis {
                        Some(axis)
                    } else {
                        Some(axis - 1)
                    }
                })
                .collect::<Vec<_>>(),
        ),
        CollectiveMode::Tiled => {
            if operation.concat_axis >= input_type.rank() {
                return Err(TypeError::invalid(format!(
                    "'all_gather' concat axis {} is out of bounds for rank {}",
                    operation.concat_axis,
                    input_type.rank(),
                )));
            }
            let mut dimensions = input_type.shape().dimensions().to_vec();
            dimensions[operation.concat_axis] = Dimension::Static(0);
            let sharding = resized_output_sharding(input_type, dimensions.as_slice(), ALL_GATHER_OPERATION_NAME)?;
            let mut output_type =
                ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
            output_type.sharding = sharding;
            (output_type, (0..input_type.rank()).map(|axis| (axis != operation.concat_axis).then_some(axis)).collect())
        }
    };
    let mut output_types = infer_explicit_shape_changing_collective_output_type(
        ALL_GATHER_OPERATION_NAME,
        input_types,
        base_output_type,
        unchanged_input_axes.as_slice(),
        |input_type, output_extents| {
            match operation.options.mode {
                CollectiveMode::Untiled => {
                    let output_extent = &output_extents[operation.concat_axis];
                    if output_extent != &Dimension::Static(effective_axis_size) {
                        return Err(TypeError::invalid(format!(
                            "'all_gather' inserted output axis {} extent must equal axis group size \
                             {effective_axis_size} but got {output_extent}",
                            operation.concat_axis,
                        )));
                    }
                }
                CollectiveMode::Tiled => {
                    let input_extent = &input_type.shape().dimensions()[operation.concat_axis];
                    let output_extent = &output_extents[operation.concat_axis];
                    if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                        (input_extent, output_extent)
                    {
                        let expected = input_extent.checked_mul(effective_axis_size).ok_or_else(|| {
                            TypeError::invalid("'all_gather' result extent does not fit in usize".to_string())
                        })?;
                        if *output_extent != expected {
                            return Err(TypeError::invalid(format!(
                                "'all_gather' result extent must equal input axis {} extent {input_extent} multiplied \
                                 by axis group size {effective_axis_size}; expected {expected} but got {output_extent}",
                                operation.concat_axis,
                            )));
                        }
                    }
                }
            }
            Ok(())
        },
    )?;
    let input_type = <&ArrayType>::try_from(&input_types[0])?;
    let output_type = <&ArrayType>::try_from(&output_types.remove(0))?.clone();
    Ok(vec![all_gather_output_type(input_type, output_type, operation)?.into()])
}

/// Infers the composite sum-scatter contract.
pub(crate) fn infer_explicit_psum_scatter_output_types(
    operation: &PSumScatterOperation,
    input_types: &[ArrayProgramType],
) -> Result<Vec<ArrayProgramType>, TypeError> {
    let effective_axis_size = operation.effective_axis_size()?;
    let Some(input_type) = input_types.first() else {
        return Err(TypeError::invalid("'psum_scatter' expects an array followed by its output extents"));
    };
    let input_type = <&ArrayType>::try_from(input_type)?;
    if operation.options.mode == CollectiveMode::Untiled {
        let Some(input_extent) = input_type.shape().dimensions().get(operation.scatter_axis) else {
            return Err(TypeError::invalid(format!(
                "'psum_scatter' scatter axis {} is out of bounds for rank {}",
                operation.scatter_axis,
                input_type.rank(),
            )));
        };
        if let Dimension::Static(input_extent) = input_extent
            && *input_extent != effective_axis_size
        {
            return Err(TypeError::invalid(format!(
                "'psum_scatter' untiled scatter axis {} size {input_extent} must equal group size \
                 {effective_axis_size}",
                operation.scatter_axis,
            )));
        }
        let base_output_type = input_type.without_dimension(operation.scatter_axis)?.0;
        let unchanged_input_axes = (0..base_output_type.rank())
            .map(|axis| if axis < operation.scatter_axis { Some(axis) } else { Some(axis + 1) })
            .collect::<Vec<_>>();
        return infer_explicit_shape_changing_collective_output_type(
            PSUM_SCATTER_OPERATION_NAME,
            input_types,
            base_output_type,
            unchanged_input_axes.as_slice(),
            |_, _| Ok(()),
        );
    }
    if operation.scatter_axis >= input_type.rank() {
        return Err(TypeError::invalid(format!(
            "'psum_scatter' scatter axis {} is out of bounds for rank {}",
            operation.scatter_axis,
            input_type.rank(),
        )));
    }
    let mut dimensions = input_type.shape().dimensions().to_vec();
    dimensions[operation.scatter_axis] = Dimension::Static(0);
    let sharding = resized_output_sharding(input_type, dimensions.as_slice(), PSUM_SCATTER_OPERATION_NAME)?;
    let mut base_output_type =
        ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
    base_output_type.sharding = sharding;
    let unchanged_input_axes = (0..input_type.rank())
        .map(|axis| (axis != operation.scatter_axis).then_some(axis))
        .collect::<Vec<_>>();
    infer_explicit_shape_changing_collective_output_type(
        PSUM_SCATTER_OPERATION_NAME,
        input_types,
        base_output_type,
        unchanged_input_axes.as_slice(),
        |input_type, output_extents| {
            let rank = input_type.rank();
            let Some(input_extent) = input_type.shape().dimensions().get(operation.scatter_axis) else {
                return Err(TypeError::invalid(format!(
                    "'psum_scatter' scatter axis {} is out of bounds for rank {rank}",
                    operation.scatter_axis,
                )));
            };
            if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                (input_extent, &output_extents[operation.scatter_axis])
            {
                if *input_extent % effective_axis_size != 0 {
                    return Err(TypeError::invalid(format!(
                        "'psum_scatter' scatter axis {} size {input_extent} is not divisible by group size \
                         {effective_axis_size}",
                        operation.scatter_axis,
                    )));
                }
                let expected = *input_extent / effective_axis_size;
                if *output_extent != expected {
                    return Err(TypeError::invalid(format!(
                        "'psum_scatter' result extent must equal input axis {} extent {input_extent} divided by axis \
                         group size {effective_axis_size}; expected {expected} but got {output_extent}",
                        operation.scatter_axis,
                    )));
                }
            }
            Ok(())
        },
    )
}

/// Infers the composite all-to-all contract.
pub(crate) fn infer_explicit_all_to_all_output_types(
    operation: &AllToAllOperation,
    input_types: &[ArrayProgramType],
) -> Result<Vec<ArrayProgramType>, TypeError> {
    let effective_axis_size = operation.effective_axis_size()?;
    let Some(input_type) = input_types.first() else {
        return Err(TypeError::invalid("'all_to_all' expects an array followed by its output extents"));
    };
    let input_type = <&ArrayType>::try_from(input_type)?;
    if operation.options.mode == CollectiveMode::Untiled {
        let Some(input_extent) = input_type.shape().dimensions().get(operation.split_axis) else {
            return Err(TypeError::invalid(format!(
                "'all_to_all' split axis {} is out of bounds for rank {}",
                operation.split_axis,
                input_type.rank(),
            )));
        };
        if let Dimension::Static(input_extent) = input_extent
            && *input_extent != effective_axis_size
        {
            return Err(TypeError::invalid(format!(
                "'all_to_all' untiled split axis {} size {input_extent} must equal group size {effective_axis_size}",
                operation.split_axis,
            )));
        }
        let output_type = input_type
            .without_dimension(operation.split_axis)?
            .0
            .with_inserted_dimension(operation.concat_axis, Dimension::Static(effective_axis_size))?;
        let mut unchanged_input_axes =
            (0..input_type.rank()).filter(|axis| *axis != operation.split_axis).map(Some).collect::<Vec<_>>();
        unchanged_input_axes.insert(operation.concat_axis, None);
        return infer_explicit_shape_changing_collective_output_type(
            ALL_TO_ALL_OPERATION_NAME,
            input_types,
            output_type,
            unchanged_input_axes.as_slice(),
            |_, output_extents| {
                let output_extent = &output_extents[operation.concat_axis];
                if output_extent != &Dimension::Static(effective_axis_size) {
                    return Err(TypeError::invalid(format!(
                        "'all_to_all' inserted output axis {} extent must equal axis group size \
                         {effective_axis_size} but got {output_extent}",
                        operation.concat_axis,
                    )));
                }
                Ok(())
            },
        );
    }
    if operation.split_axis == operation.concat_axis {
        let Some(input_extent) = input_type.shape().dimensions().get(operation.split_axis) else {
            return Err(TypeError::invalid(format!(
                "'all_to_all' split axis {} is out of bounds for rank {}",
                operation.split_axis,
                input_type.rank(),
            )));
        };
        if let Dimension::Static(input_extent) = input_extent
            && *input_extent % effective_axis_size != 0
        {
            return Err(TypeError::invalid(format!(
                "'all_to_all' split axis {} size {input_extent} is not divisible by group size \
                 {effective_axis_size}",
                operation.split_axis,
            )));
        }
        return infer_explicit_shape_changing_collective_output_type(
            ALL_TO_ALL_OPERATION_NAME,
            input_types,
            input_type.clone(),
            &(0..input_type.rank()).map(Some).collect::<Vec<_>>(),
            |_, _| Ok(()),
        );
    }
    if operation.split_axis >= input_type.rank() || operation.concat_axis >= input_type.rank() {
        return Err(TypeError::invalid(format!(
            "'all_to_all' split axis {} or concat axis {} is out of bounds for rank {}",
            operation.split_axis,
            operation.concat_axis,
            input_type.rank(),
        )));
    }
    let mut dimensions = input_type.shape().dimensions().to_vec();
    dimensions[operation.split_axis] = Dimension::Static(0);
    dimensions[operation.concat_axis] = Dimension::Static(0);
    let sharding = resized_output_sharding(input_type, dimensions.as_slice(), ALL_TO_ALL_OPERATION_NAME)?;
    let mut base_output_type =
        ArrayType::new(input_type.data_type(), Shape::new(dimensions)).with_memory(input_type.memory());
    base_output_type.sharding = sharding;
    let unchanged_input_axes = (0..input_type.rank())
        .map(|axis| (axis != operation.split_axis && axis != operation.concat_axis).then_some(axis))
        .collect::<Vec<_>>();
    infer_explicit_shape_changing_collective_output_type(
        ALL_TO_ALL_OPERATION_NAME,
        input_types,
        base_output_type,
        unchanged_input_axes.as_slice(),
        |input_type, output_extents| {
            if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                (&input_type.shape().dimensions()[operation.split_axis], &output_extents[operation.split_axis])
            {
                if *input_extent % effective_axis_size != 0 {
                    return Err(TypeError::invalid(format!(
                        "'all_to_all' split axis {} size {input_extent} is not divisible by group size \
                         {effective_axis_size}",
                        operation.split_axis,
                    )));
                }
                let expected = *input_extent / effective_axis_size;
                if *output_extent != expected {
                    return Err(TypeError::invalid(format!(
                        "'all_to_all' split result extent must equal input axis {} extent {input_extent} divided by \
                         group size {effective_axis_size}; expected {expected} but got {output_extent}",
                        operation.split_axis,
                    )));
                }
            }
            if let (Dimension::Static(input_extent), Dimension::Static(output_extent)) =
                (&input_type.shape().dimensions()[operation.concat_axis], &output_extents[operation.concat_axis])
            {
                let expected = input_extent.checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid("'all_to_all' concatenation result extent does not fit in usize".to_string())
                })?;
                if *output_extent != expected {
                    return Err(TypeError::invalid(format!(
                        "'all_to_all' concat result extent must equal input axis {} extent {input_extent} multiplied \
                         by group size {effective_axis_size}; expected {expected} but got {output_extent}",
                        operation.concat_axis,
                    )));
                }
            }
            Ok(())
        },
    )
}

/// Interprets a shape-changing collective outside any binder: only the degenerate single-participant axis
/// (`axis_size == 1`) has defined per-item semantics (the identity), and any larger axis reports an error because
/// the other participants do not exist per item.
fn interpret_degenerate_collective<V: Clone>(
    operation_name: &str,
    axis_name: &str,
    axis_size: usize,
    inputs: &[V],
) -> Result<Vec<V>, ProgramError> {
    check_count!("input", inputs, 1, ProgramError);
    if axis_size != 1 {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "cannot interpret '{operation_name}' over axis '{axis_name}' of size {axis_size} without an \
                 enclosing binder",
            ),
        });
    }
    Ok(vec![inputs[0].clone()])
}

/// Shared operand preparation for the shape-changing collective batching rules at a matching `batch` level. It
/// validates the single-operand contract, realigns the mapped batch axis to the leading physical axis, and returns
/// the packed physical value together with its static dimensions (whose leading entry is the batch size).
///
/// A replicated operand is first materialized as `axis_size` identical batch items via [`ArrayBatch::broadcast`],
/// which yields the degenerate collective-of-a-replicated-value semantics for free: an `all_gather` of a replicated
/// value concatenates `axis_size` copies, a `psum_scatter` scatters the `axis_size`-fold sum, and so on. A mapped
/// batch axis whose size disagrees with the operation's resolved `axis_size` reports an error; the staging
/// capabilities resolve both from the same binder, so a mismatch indicates a hand-constructed operation.
fn shape_changing_collective_batch_operand<V>(
    operation_name: &str,
    axis_name: &str,
    axis_size: usize,
    inputs: &[ArrayBatch<V>],
) -> Result<(V, Vec<usize>), BatchingError>
where
    V: Value<Type = ArrayType> + LegacyBroadcast + Transpose,
{
    check_count!("input", inputs, 1, ProgramError);
    let input = if inputs[0].batch_axis().is_replicated() {
        inputs[0].broadcast(0, axis_size, ShardingDimension::Replicated)?
    } else {
        inputs[0].move_axis(0)?
    };
    // The operand is mapped at axis 0 by construction (a replicated operand was just broadcast onto a mapped axis),
    // so a missing batch size is impossible here; a dynamic batch axis errors through `batch_size` itself.
    let batch_size = input.batch_size()?.unwrap();
    if batch_size != axis_size {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "'{operation_name}' over axis '{axis_name}' resolved axis size {axis_size} but the mapped batch \
                 axis has size {batch_size}",
            ),
        });
    }
    let Some(shape) = input.value().r#type().static_shape() else {
        return Err(BatchingError::UnsupportedOperation {
            message: format!("'{operation_name}' batching requires statically shaped operands"),
        });
    };
    let dimensions = shape.dimensions().to_vec();
    Ok((input.into_value(), dimensions))
}

/// Implements the shared structure of the shape-changing collectives: the operation struct with its accessors, the
/// `Display`/`Operation` implementations (with payload-dependent output-shape inference provided as a closure over
/// the operand dimensions), degenerate interpretation, default partial evaluation, the linear forward-mode rule
/// (the tangent rides the same collective), and the value-level staging capability that resolves the named axis size
/// from the active [`NamedAxes`] environment. The batching rules are hand-written below the macro invocations
/// because each collective materializes the mapped batch axis differently.
macro_rules! shape_changing_collective {
    (
        $(#[$operation_documentation:meta])*
        operation = $operation:ident,
        name = $operation_name:ident = $name_literal:literal,
        $(#[$capability_documentation:meta])*
        capability = $capability:ident::$method:ident,
        fields = { $($(#[$field_documentation:meta])* $field:ident: $field_type:ty),* $(,)? },
        infer = |$infer_self:ident, $input_type:ident, $dimensions:ident| $infer:block $(,)?
    ) => {
        /// Canonical operation name for the operation.
        pub const $operation_name: &str = $name_literal;

        $(#[$operation_documentation])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $operation {
            /// Axis name referenced by this collective.
            axis_name: String,

            /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment when
            /// the operation is staged.
            axis_size: usize,

            $($(#[$field_documentation])* $field: $field_type,)*
        }

        impl $operation {
            /// Creates a new operation over the named axis with the provided resolved axis size.
            #[inline]
            pub fn new(axis_name: String, axis_size: usize, $($field: $field_type),*) -> Self {
                Self { axis_name, axis_size, $($field),* }
            }

            /// Returns the axis name referenced by this collective.
            #[inline]
            pub fn axis_name(&self) -> &str {
                &self.axis_name
            }

            /// Returns the number of participants along the named axis.
            #[inline]
            pub fn axis_size(&self) -> usize {
                self.axis_size
            }
        }

        impl Display for $operation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                Operation::<ArrayType>::render(self, formatter, 0)
            }
        }

        impl Operation<ArrayType> for $operation {
            #[inline]
            fn name(&self) -> &'static str {
                $operation_name
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                check_count!("region", region_interfaces, 0, TypeError);
                validate_collective_axis_size($name_literal, self.axis_size)?;
                let $dimensions = shape_changing_collective_dimensions($name_literal, input_types)?;
                let $infer_self = self;
                let $input_type = &input_types[0];
                Ok(vec![$infer?])
            }

            fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
                OperationFormatter::new(formatter, indentation, $operation_name)?.bracketed(|operation| {
                    operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
                    operation.field("axis_size", &self.axis_size)?;
                    $(operation.field(stringify!($field), format_args!("{:?}", &self.$field))?;)*
                    Ok(())
                })
            }
        }

        impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for $operation {
            fn interpret<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                interpret_degenerate_collective($name_literal, &self.axis_name, self.axis_size, inputs)
            }
        }

        /// Partial evaluation defers to the default fold-or-residualize behavior of
        /// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
        impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for $operation where
            C::Operation: From<$operation>
        {
        }

        /// Forward-mode rule: the collective is linear, so the tangent rides the same collective. Structural-zero
        /// tangents stay symbolic, retyped to the output tangent type (the collective changes shapes).
        impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for $operation
        where
            C::Operation: From<$operation>,
        {
            fn jvp<D: DifferentiationDriver<C>>(
                &self,
                context: &C,
                _driver: &D,
                inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                check_count!("input", inputs, 1, ProgramError);
                let mut primal_outputs =
                    context.bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].primal()))?;
                check_count!("output", primal_outputs, 1, ProgramError);
                let primal = primal_outputs.remove(0);
                let tangent = match inputs[0].tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                    MaybeZero::Value(tangent) => {
                        let mut tangent_outputs =
                            context.bind(self.clone(), Vec::new(), std::slice::from_ref(tangent))?;
                        check_count!("output", tangent_outputs, 1, ProgramError);
                        MaybeZero::Value(tangent_outputs.remove(0))
                    }
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
        }

        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            /// Stages this collective over axis `axis_name`, resolving the axis size from the active [`NamedAxes`]
            /// environment and returning an [`AxisError::UnboundAxisName`] error when no enclosing binder binds the
            /// name.
            fn $method(&self, axis_name: &str, $($field: $field_type),*) -> Result<Self, ProgramError>;
        }

        impl<V: Value<Type = ArrayType>> $capability for V
        where
            V::DispatchDomain: Context + NamedAxes,
            <V::DispatchDomain as Domain>::Operation: From<$operation>,
        {
            fn $method(&self, axis_name: &str, $($field: $field_type),*) -> Result<Self, ProgramError> {
                let context = self.dispatch_domain();
                let axis_size = resolve_named_axis_size(&context, axis_name)?;
                let mut outputs = context.bind(
                    $operation::new(axis_name.to_string(), axis_size, $($field),*),
                    Vec::new(),
                    std::slice::from_ref(self),
                )?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(outputs.remove(0))
            }
        }
    };
}

shape_changing_collective! {
    /// [`Operation`] that concatenates every participant's operand along `concat_axis` across the named axis, so
    /// every participant receives the full concatenation — the analogue of
    /// [JAX's `all_gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_gather.html) with `tiled = True`
    /// and [StableHLO's `all_gather`](https://openxla.org/stablehlo/spec#all_gather). The output extends
    /// `concat_axis` by the axis size; all other dimensions are unchanged. The collective is linear and its
    /// transpose is [`PSumScatterOperation`] over the same axis and dimension. A matching `batch` level consumes the
    /// mapped batch axis by merging it item-major into `concat_axis`, replicating the gathered value across the
    /// batch items.
    operation = AllGatherOperation,
    name = ALL_GATHER_OPERATION_NAME = "all_gather",
    #[doc(hidden)]
    /// Frozen homogeneous capability retained for array-only Phase 4–9 consumers.
    capability = LegacyAllGather::all_gather,
    fields = {
        /// Axis of the operand along which the participants' values are concatenated.
        concat_axis: usize,

        /// Shared rank and participant-group semantics.
        options: CollectiveOptions,

        /// Named-axis variance of the result.
        output_variance: AllGatherOutputVariance,
    },
    infer = |operation, input_type, dimensions| {
        let effective_axis_size = operation.effective_axis_size()?;
        let output_type = match operation.options.mode {
            CollectiveMode::Untiled => input_type
                .with_inserted_dimension(operation.concat_axis, Dimension::Static(effective_axis_size))?,
            CollectiveMode::Tiled => {
                let mut output_dimensions = dimensions;
                let Some(dimension) = output_dimensions.get_mut(operation.concat_axis) else {
                    return Err(TypeError::invalid(format!(
                        "'all_gather' concat axis {} is out of bounds for rank {}",
                        operation.concat_axis,
                        output_dimensions.len(),
                    )));
                };
                *dimension = dimension.checked_mul(effective_axis_size).ok_or_else(|| {
                    TypeError::invalid("'all_gather' result extent does not fit in usize".to_string())
                })?;
                shape_changing_collective_output_type(ALL_GATHER_OPERATION_NAME, input_type, output_dimensions)?
            }
        };
        all_gather_output_type(input_type, output_type, operation)
    },
}

impl AllGatherOperation {
    /// Returns the axis of the operand along which the participants' values are concatenated.
    #[inline]
    pub fn concat_axis(&self) -> usize {
        self.concat_axis
    }

    /// Returns the shared rank and participant-group semantics.
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the named-axis variance of the result.
    #[inline]
    pub fn output_variance(&self) -> AllGatherOutputVariance {
        self.output_variance
    }

    /// Returns the participant count used for result-shape arithmetic.
    #[inline]
    pub fn effective_axis_size(&self) -> Result<usize, TypeError> {
        if self.output_variance != AllGatherOutputVariance::Varying && self.options.axis_index_groups.is_some() {
            return Err(TypeError::invalid(
                "'all_gather' axis index groups are not supported with invariant or reduced output variance"
                    .to_string(),
            ));
        }
        self.options.effective_axis_size(ALL_GATHER_OPERATION_NAME, self.axis_size)
    }
}

shape_changing_collective! {
    /// [`Operation`] that sums every participant's operand across the named axis and scatters the result: each
    /// participant receives its own chunk of the sum along `scatter_axis` — the analogue of
    /// [JAX's `psum_scatter`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.psum_scatter.html) with
    /// `tiled = True` and [StableHLO's `reduce_scatter`](https://openxla.org/stablehlo/spec#reduce_scatter) with a
    /// sum reduction. The output shrinks `scatter_axis` by the axis size (the dimension must be divisible by it).
    /// The collective is linear and its transpose is [`AllGatherOperation`] over the same axis and dimension. A
    /// matching `batch` level consumes the mapped batch axis by summing over it and re-mapping the chunks of
    /// `scatter_axis` onto it, so batch item `i` receives chunk `i` of the sum.
    operation = PSumScatterOperation,
    name = PSUM_SCATTER_OPERATION_NAME = "psum_scatter",
    #[doc(hidden)]
    /// Frozen homogeneous capability retained for array-only Phase 4–9 consumers.
    capability = LegacyPSumScatter::psum_scatter,
    fields = {
        /// Axis of the operand along which the summed result is scattered across the participants.
        scatter_axis: usize,

        /// Shared rank and participant-group semantics.
        options: CollectiveOptions,
    },
    infer = |operation, input_type, dimensions| {
        let effective_axis_size = operation.effective_axis_size()?;
        match operation.options.mode {
            CollectiveMode::Untiled => {
                let Some(dimension) = dimensions.get(operation.scatter_axis) else {
                    return Err(TypeError::invalid(format!(
                        "'psum_scatter' scatter axis {} is out of bounds for rank {}",
                        operation.scatter_axis,
                        dimensions.len(),
                    )));
                };
                if *dimension != effective_axis_size {
                    return Err(TypeError::invalid(format!(
                        "'psum_scatter' untiled scatter axis {} size {dimension} must equal group size \
                         {effective_axis_size}",
                        operation.scatter_axis,
                    )));
                }
                Ok::<_, TypeError>(input_type.without_dimension(operation.scatter_axis)?.0)
            }
            CollectiveMode::Tiled => {
                let mut output_dimensions = dimensions;
                let Some(dimension) = output_dimensions.get_mut(operation.scatter_axis) else {
                    return Err(TypeError::invalid(format!(
                        "'psum_scatter' scatter axis {} is out of bounds for rank {}",
                        operation.scatter_axis,
                        output_dimensions.len(),
                    )));
                };
                if *dimension % effective_axis_size != 0 {
                    return Err(TypeError::invalid(format!(
                        "'psum_scatter' scatter axis {} size {} is not divisible by group size {}",
                        operation.scatter_axis,
                        *dimension,
                        effective_axis_size,
                    )));
                }
                *dimension /= effective_axis_size;
                shape_changing_collective_output_type(
                    PSUM_SCATTER_OPERATION_NAME,
                    input_type,
                    output_dimensions,
                )
            }
        }
    },
}

impl PSumScatterOperation {
    /// Returns the axis of the operand along which the summed result is scattered across the participants.
    #[inline]
    pub fn scatter_axis(&self) -> usize {
        self.scatter_axis
    }

    /// Returns the shared rank and participant-group semantics.
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the participant count used for result-shape arithmetic.
    #[inline]
    pub fn effective_axis_size(&self) -> Result<usize, TypeError> {
        self.options.effective_axis_size(PSUM_SCATTER_OPERATION_NAME, self.axis_size)
    }
}

shape_changing_collective! {
    /// [`Operation`] that sends every participant's operand to another participant along the named axis according
    /// to explicit `(source, target)` pairs — the analogue of
    /// [JAX's `ppermute`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html) and
    /// [StableHLO's `collective_permute`](https://openxla.org/stablehlo/spec#collective_permute). Participants that
    /// no pair targets receive zeros. The output shape is unchanged. The collective is linear and its transpose is
    /// the permutation with every pair inverted. A matching `batch` level consumes the mapped batch axis by
    /// reassembling it in target order from per-item slices, with zero slices at untargeted positions.
    operation = PpermuteOperation,
    name = PPERMUTE_OPERATION_NAME = "ppermute",
    /// Value-level entry point for staging a [`PpermuteOperation`]. Refer to its documentation for the semantics
    /// and transform rules.
    capability = Ppermute::ppermute,
    fields = {
        /// Pairs of `(source, target)` positions along the named axis: the value of participant `source` is sent to
        /// participant `target`.
        source_target_pairs: Vec<(usize, usize)>,
    },
    infer = |operation, input_type, dimensions| {
        let mut seen_sources = std::collections::BTreeSet::new();
        let mut seen_targets = std::collections::BTreeSet::new();
        for (source, target) in &operation.source_target_pairs {
            if *source >= operation.axis_size || *target >= operation.axis_size {
                return Err(TypeError::invalid(format!(
                        "'ppermute' pair ({source}, {target}) is out of bounds for axis size {}",
                        operation.axis_size,
                    )));
            }
            if !seen_sources.insert(*source) || !seen_targets.insert(*target) {
                return Err(TypeError::invalid(format!(
                        "'ppermute' pairs must have unique sources and targets but ({source}, {target}) repeats one",
                    )));
            }
        }
        shape_changing_collective_output_type(PPERMUTE_OPERATION_NAME, input_type, dimensions)
    },
}

impl PpermuteOperation {
    /// Returns the `(source, target)` pairs of participant positions along the named axis.
    #[inline]
    pub fn source_target_pairs(&self) -> &[(usize, usize)] {
        self.source_target_pairs.as_slice()
    }
}

shape_changing_collective! {
    /// [`Operation`] that exchanges chunks between the participants along the named axis: every participant splits
    /// its operand into `axis_size` chunks along `split_axis` and receives the participants' chunks concatenated
    /// along `concat_axis` — the analogue of
    /// [JAX's `all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.all_to_all.html) and
    /// [StableHLO's `all_to_all`](https://openxla.org/stablehlo/spec#all_to_all). The output shrinks `split_axis`
    /// by the axis size (the dimension must be divisible by it) and extends `concat_axis` by it. The collective is
    /// linear and its transpose is the exchange with the split and concatenation axes swapped. A matching `batch`
    /// level consumes the mapped batch axis with a reshape/transpose block exchange: batch item `i` receives every
    /// item's chunk `i` of `split_axis`, concatenated item-major along `concat_axis`.
    operation = AllToAllOperation,
    name = ALL_TO_ALL_OPERATION_NAME = "all_to_all",
    #[doc(hidden)]
    /// Frozen homogeneous capability retained for array-only Phase 4–9 consumers.
    capability = LegacyAllToAll::all_to_all,
    fields = {
        /// Axis of the operand that is split into one chunk per participant.
        split_axis: usize,

        /// Axis of the output along which the received chunks are concatenated.
        concat_axis: usize,

        /// Shared rank and participant-group semantics.
        options: CollectiveOptions,
    },
    infer = |operation, input_type, dimensions| {
        let effective_axis_size = operation.effective_axis_size()?;
        let mut output_dimensions = dimensions;
        let rank = output_dimensions.len();
        if operation.split_axis >= rank || operation.concat_axis >= rank {
            return Err(TypeError::invalid(format!(
                    "'all_to_all' split axis {} or concat axis {} is out of bounds for rank {rank}",
                    operation.split_axis,
                    operation.concat_axis,
                )));
        }
        if operation.options.mode == CollectiveMode::Untiled {
            if output_dimensions[operation.split_axis] != effective_axis_size {
                return Err(TypeError::invalid(format!(
                    "'all_to_all' untiled split axis {} size {} must equal group size {}",
                    operation.split_axis,
                    output_dimensions[operation.split_axis],
                    effective_axis_size,
                )));
            }
            input_type
                .without_dimension(operation.split_axis)?
                .0
                .with_inserted_dimension(operation.concat_axis, Dimension::Static(effective_axis_size))
        } else {
            if output_dimensions[operation.split_axis] % effective_axis_size != 0 {
                return Err(TypeError::invalid(format!(
                    "'all_to_all' split axis {} size {} is not divisible by group size {}",
                    operation.split_axis,
                    output_dimensions[operation.split_axis],
                    effective_axis_size,
                )));
            }
            output_dimensions[operation.split_axis] /= effective_axis_size;
            output_dimensions[operation.concat_axis] = output_dimensions[operation.concat_axis]
                .checked_mul(effective_axis_size)
                .ok_or_else(|| {
                    TypeError::invalid("'all_to_all' concatenation result extent does not fit in usize".to_string())
                })?;
            shape_changing_collective_output_type(ALL_TO_ALL_OPERATION_NAME, input_type, output_dimensions)
        }
    },
}

impl AllToAllOperation {
    /// Returns the axis of the operand that is split into one chunk per participant.
    #[inline]
    pub fn split_axis(&self) -> usize {
        self.split_axis
    }

    /// Returns the axis of the output along which the received chunks are concatenated.
    #[inline]
    pub fn concat_axis(&self) -> usize {
        self.concat_axis
    }

    /// Returns the shared rank and participant-group semantics.
    #[inline]
    pub fn options(&self) -> &CollectiveOptions {
        &self.options
    }

    /// Returns the participant count used for result-shape arithmetic.
    #[inline]
    pub fn effective_axis_size(&self) -> Result<usize, TypeError> {
        self.options.effective_axis_size(ALL_TO_ALL_OPERATION_NAME, self.axis_size)
    }
}

/// Returns an exact first-class collective extent constant.
fn collective_extent_constant<V>(context: &V::DispatchDomain, extent: usize) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayProgramType>,
    V::DispatchDomain: Context<Type = ArrayProgramType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    context.lift(DimensionValue::constant(extent)?.into())
}

/// Returns one first-class dimension for every input array axis, using exact constants for static axes and explicit
/// [`DimensionSize`] gateways for dynamic axes.
fn collective_input_extents<V>(context: &V::DispatchDomain, value: &V) -> Result<Vec<V>, ProgramError>
where
    V: Value<Type = ArrayProgramType> + DimensionSize<V>,
    V::DispatchDomain: Context<Type = ArrayProgramType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
{
    let r#type = value.r#type();
    let input_type = <&ArrayType>::try_from(r#type.as_ref())?;
    input_type
        .shape()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(axis, dimension)| match dimension {
            Dimension::Static(extent) => collective_extent_constant(context, *extent),
            Dimension::Dynamic(_) => value.dimension_size(axis),
        })
        .collect()
}

/// Computes one tiled collective result extent by multiplying an input-axis extent by the effective participant count.
fn multiplied_collective_extent<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayProgramType> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayProgramType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionMul,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    Ok(<V as ValueProjection<DimensionType>>::from_projected(input_extent.dimension_mul(&effective_axis_size)?))
}

/// Computes one tiled collective result extent by requiring exact divisibility and dividing an input-axis extent by
/// the effective participant count.
fn divided_collective_extent<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayProgramType> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayProgramType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionDivFloor + DimensionRequirement,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    input_extent.require_divisible_by(&effective_axis_size)?;
    Ok(<V as ValueProjection<DimensionType>>::from_projected(input_extent.dimension_div_floor(&effective_axis_size)?))
}

/// Requires an input axis extent to equal the effective participant count used by an untiled collective.
fn require_collective_axis_extent<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayProgramType> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayProgramType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionRequirement,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    input_extent.require_equal(&effective_axis_size)
}

/// Requires an input axis extent to be exactly divisible by the effective participant count.
fn require_collective_axis_divisible<V>(
    context: &V::DispatchDomain,
    input_extent: &V,
    effective_axis_size: usize,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayProgramType> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayProgramType>,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionRequirement,
{
    let input_extent = <V as ValueProjection<DimensionType>>::into_projected(input_extent.clone())?;
    let effective_axis_size = collective_extent_constant(context, effective_axis_size)?;
    let effective_axis_size = <V as ValueProjection<DimensionType>>::into_projected(effective_axis_size)?;
    input_extent.require_divisible_by(&effective_axis_size)
}

/// Stages an all-gather with first-class dynamic tiled extents and rank-changing untiled semantics.
pub trait AllGather: Sized {
    /// Stacks participants along a new axis at `concat_axis`, producing an output that varies across `axis_name`.
    #[inline]
    fn all_gather(&self, axis_name: &str, concat_axis: usize) -> Result<Self, ProgramError> {
        self.all_gather_with_options(
            axis_name,
            concat_axis,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        )
    }

    /// Concatenates participants into the existing `concat_axis`, producing an output that varies across
    /// `axis_name`.
    #[inline]
    fn all_gather_tiled(&self, axis_name: &str, concat_axis: usize) -> Result<Self, ProgramError> {
        self.all_gather_with_options(
            axis_name,
            concat_axis,
            CollectiveOptions::new(CollectiveMode::Tiled),
            AllGatherOutputVariance::Varying,
        )
    }

    /// Gathers participants using explicit shape, grouping, and output-variance semantics.
    fn all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: AllGatherOutputVariance,
    ) -> Result<Self, ProgramError>;
}

impl<V> AllGather for V
where
    V: Value<Type = ArrayProgramType> + DimensionSize<V> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayProgramType> + NamedAxes,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V::DispatchDomain as Domain>::Operation: From<AllGatherOperation>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionMul,
{
    fn all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: AllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(ALL_GATHER_OPERATION_NAME, axis_size)?;
        if output_variance != AllGatherOutputVariance::Varying && options.axis_index_groups.is_some() {
            return Err(TypeError::invalid(
                "'all_gather' axis index groups are not supported with invariant or reduced output variance"
                    .to_string(),
            )
            .into());
        }
        let operation =
            AllGatherOperation::new(axis_name.to_string(), axis_size, concat_axis, options.clone(), output_variance);
        let mut output_extents = collective_input_extents(&context, self)?;
        match options.mode {
            CollectiveMode::Untiled => {
                if concat_axis > output_extents.len() {
                    return Err(TypeError::invalid(format!(
                        "'all_gather' concat axis {concat_axis} is out of bounds for rank {}",
                        output_extents.len(),
                    ))
                    .into());
                }
                output_extents.insert(concat_axis, collective_extent_constant(&context, effective_axis_size)?);
            }
            CollectiveMode::Tiled => {
                let rank = output_extents.len();
                let Some(output_extent) = output_extents.get_mut(concat_axis) else {
                    return Err(TypeError::invalid(format!(
                        "'all_gather' concat axis {concat_axis} is out of bounds for rank {rank}",
                    ))
                    .into());
                };
                *output_extent = multiplied_collective_extent(&context, output_extent, effective_axis_size)?;
            }
        };
        let inputs = std::iter::once(self.clone()).chain(output_extents).collect::<Vec<_>>();
        Ok(context.bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
    }
}

impl<V> AllGather for ProjectedValue<ArrayType, V>
where
    V: AllGather + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
{
    fn all_gather_with_options(
        &self,
        axis_name: &str,
        concat_axis: usize,
        options: CollectiveOptions,
        output_variance: AllGatherOutputVariance,
    ) -> Result<Self, ProgramError> {
        self.value()
            .all_gather_with_options(axis_name, concat_axis, options, output_variance)?
            .into_projected()
            .map_err(Into::into)
    }
}

/// Stages a sum-scatter with first-class dynamic tiled extents and rank-changing untiled semantics.
pub trait PSumScatter: Sized {
    /// Sums participants and consumes `scatter_axis`, whose extent must equal the effective participant count.
    #[inline]
    fn psum_scatter(&self, axis_name: &str, scatter_axis: usize) -> Result<Self, ProgramError> {
        self.psum_scatter_with_options(axis_name, scatter_axis, CollectiveOptions::default())
    }

    /// Sums participants and scatters equal chunks along the existing `scatter_axis`.
    #[inline]
    fn psum_scatter_tiled(&self, axis_name: &str, scatter_axis: usize) -> Result<Self, ProgramError> {
        self.psum_scatter_with_options(axis_name, scatter_axis, CollectiveOptions::new(CollectiveMode::Tiled))
    }

    /// Sums and scatters participants using explicit shape and grouping semantics.
    fn psum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError>;
}

impl<V> PSumScatter for V
where
    V: Value<Type = ArrayProgramType> + DimensionSize<V> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayProgramType> + NamedAxes,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V::DispatchDomain as Domain>::Operation: From<PSumScatterOperation>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionDivFloor + DimensionRequirement,
{
    fn psum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(PSUM_SCATTER_OPERATION_NAME, axis_size)?;
        let operation = PSumScatterOperation::new(axis_name.to_string(), axis_size, scatter_axis, options.clone());
        let mut output_extents = collective_input_extents(&context, self)?;
        if scatter_axis >= output_extents.len() {
            return Err(TypeError::invalid(format!(
                "'psum_scatter' scatter axis {scatter_axis} is out of bounds for rank {}",
                output_extents.len(),
            ))
            .into());
        }
        match options.mode {
            CollectiveMode::Untiled => {
                require_collective_axis_extent(&context, &output_extents[scatter_axis], effective_axis_size)?;
                output_extents.remove(scatter_axis);
            }
            CollectiveMode::Tiled => {
                output_extents[scatter_axis] =
                    divided_collective_extent(&context, &output_extents[scatter_axis], effective_axis_size)?;
            }
        };
        let inputs = std::iter::once(self.clone()).chain(output_extents).collect::<Vec<_>>();
        Ok(context.bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
    }
}

impl<V> PSumScatter for ProjectedValue<ArrayType, V>
where
    V: PSumScatter + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
{
    fn psum_scatter_with_options(
        &self,
        axis_name: &str,
        scatter_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        self.value()
            .psum_scatter_with_options(axis_name, scatter_axis, options)?
            .into_projected()
            .map_err(Into::into)
    }
}

/// Stages an all-to-all with first-class dynamic tiled extents and rank-changing untiled semantics.
pub trait AllToAll: Sized {
    /// Maps `split_axis` onto the named axis and materializes that named axis at `concat_axis`.
    #[inline]
    fn all_to_all(&self, axis_name: &str, split_axis: usize, concat_axis: usize) -> Result<Self, ProgramError> {
        self.all_to_all_with_options(axis_name, split_axis, concat_axis, CollectiveOptions::default())
    }

    /// Exchanges equal chunks while preserving rank.
    #[inline]
    fn all_to_all_tiled(&self, axis_name: &str, split_axis: usize, concat_axis: usize) -> Result<Self, ProgramError> {
        self.all_to_all_with_options(axis_name, split_axis, concat_axis, CollectiveOptions::new(CollectiveMode::Tiled))
    }

    /// Exchanges participants using explicit shape and grouping semantics.
    fn all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError>;
}

impl<V> AllToAll for V
where
    V: Value<Type = ArrayProgramType> + DimensionSize<V> + ValueProjection<DimensionType>,
    V::DispatchDomain: Context<Type = ArrayProgramType> + NamedAxes,
    <V::DispatchDomain as Domain>::Constant: From<DimensionValue>,
    <V::DispatchDomain as Domain>::Operation: From<AllToAllOperation>,
    <V as ValueProjection<DimensionType>>::Projected: DimensionDivFloor + DimensionMul + DimensionRequirement,
{
    fn all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let effective_axis_size = options.effective_axis_size(ALL_TO_ALL_OPERATION_NAME, axis_size)?;
        let operation =
            AllToAllOperation::new(axis_name.to_string(), axis_size, split_axis, concat_axis, options.clone());
        let mut output_extents = collective_input_extents(&context, self)?;
        let rank = output_extents.len();
        if split_axis >= rank || concat_axis >= rank {
            return Err(TypeError::invalid(format!(
                "'all_to_all' split axis {split_axis} or concat axis {concat_axis} is out of bounds for rank {rank}",
            ))
            .into());
        }
        match options.mode {
            CollectiveMode::Untiled => {
                require_collective_axis_extent(&context, &output_extents[split_axis], effective_axis_size)?;
                output_extents.remove(split_axis);
                output_extents.insert(concat_axis, collective_extent_constant(&context, effective_axis_size)?);
            }
            CollectiveMode::Tiled if split_axis == concat_axis => {
                require_collective_axis_divisible(&context, &output_extents[split_axis], effective_axis_size)?;
            }
            CollectiveMode::Tiled => {
                let split_extent =
                    divided_collective_extent(&context, &output_extents[split_axis], effective_axis_size)?;
                let concat_extent =
                    multiplied_collective_extent(&context, &output_extents[concat_axis], effective_axis_size)?;
                output_extents[split_axis] = split_extent;
                output_extents[concat_axis] = concat_extent;
            }
        };
        let inputs = std::iter::once(self.clone()).chain(output_extents).collect::<Vec<_>>();
        Ok(context.bind(operation, Vec::new(), inputs.as_slice())?.remove(0))
    }
}

impl<V> AllToAll for ProjectedValue<ArrayType, V>
where
    V: AllToAll + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
{
    fn all_to_all_with_options(
        &self,
        axis_name: &str,
        split_axis: usize,
        concat_axis: usize,
        options: CollectiveOptions,
    ) -> Result<Self, ProgramError> {
        self.value()
            .all_to_all_with_options(axis_name, split_axis, concat_axis, options)?
            .into_projected()
            .map_err(Into::into)
    }
}

/// Convenience permutation encoded as the source participant selected for each output participant.
pub trait Pshuffle: Sized {
    /// Permutes a named axis using `permutation[output] = input` encoding.
    fn pshuffle(&self, axis_name: &str, permutation: &[usize]) -> Result<Self, ProgramError>;
}

impl<V> Pshuffle for V
where
    V: Value<Type = ArrayProgramType>,
    V::DispatchDomain: Context<Type = ArrayProgramType> + NamedAxes,
    <V::DispatchDomain as Domain>::Operation: From<PpermuteOperation>,
{
    fn pshuffle(&self, axis_name: &str, permutation: &[usize]) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        if permutation.len() != axis_size {
            return Err(TypeError::invalid(format!(
                "'pshuffle' permutation length {} must equal axis size {axis_size}",
                permutation.len(),
            ))
            .into());
        }
        let mut seen = vec![false; axis_size];
        for &source in permutation {
            let Some(source_seen) = seen.get_mut(source) else {
                return Err(TypeError::invalid(format!(
                    "'pshuffle' source index {source} is out of bounds for axis size {axis_size}",
                ))
                .into());
            };
            if *source_seen {
                return Err(TypeError::invalid(format!(
                    "'pshuffle' permutation contains source index {source} more than once",
                ))
                .into());
            }
            *source_seen = true;
        }
        Ok(context
            .bind(
                PpermuteOperation::new(
                    axis_name.to_string(),
                    axis_size,
                    permutation.iter().copied().zip(0..axis_size).collect(),
                ),
                Vec::new(),
                std::slice::from_ref(self),
            )?
            .remove(0))
    }
}

/// Convenience untiled all-to-all that exchanges one ranked array axis with a named axis.
pub trait PSwapAxes: AllToAll {
    /// Swaps `axis` with `axis_name` over the full named axis.
    #[inline]
    fn pswapaxes(&self, axis_name: &str, axis: usize) -> Result<Self, ProgramError> {
        self.all_to_all(axis_name, axis, axis)
    }

    /// Swaps `axis` with `axis_name` within the provided ordered participant groups.
    #[inline]
    fn pswapaxes_with_axis_index_groups(
        &self,
        axis_name: &str,
        axis: usize,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        self.all_to_all_with_options(
            axis_name,
            axis,
            axis,
            CollectiveOptions::default().with_axis_index_groups(axis_index_groups),
        )
    }
}

impl<V: AllToAll> PSwapAxes for V {}

/// Batching rule for [`AllGatherOperation`]. A matching `batch` level consumes the mapped batch axis by
/// materializing the gather: the batch axis is transposed to sit immediately before the per-item `concat_axis` and
/// merged into it, laying the gathered chunks out item-major (item 0's chunk first), which matches the tiled
/// StableHLO `all_gather` ordering. Every batch item sees the same gathered value, so the output is replicated. A
/// non-matching level forwards the collective untouched to the parent context via [`forward_collective_to_parent`].
impl<C, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for AllGatherOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<AllGatherOperation>,
    <C as Domain>::Value: LegacyBroadcast + Reshape + Transpose,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs);
        }
        if self.options.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: "'all_gather' axis index groups are not supported when a batch transform binds the \
                          collective axis"
                    .to_string(),
            });
        }
        if self.output_variance == AllGatherOutputVariance::Reduced {
            return Err(BatchingError::UnsupportedOperation {
                message: "'all_gather' with reduced output variance is not supported when a batch transform binds \
                          the collective axis"
                    .to_string(),
            });
        }
        let (value, dimensions) = shape_changing_collective_batch_operand(
            ALL_GATHER_OPERATION_NAME,
            &self.axis_name,
            self.axis_size,
            inputs,
        )?;
        let per_item_rank = dimensions.len() - 1;
        let axis_is_out_of_bounds = match self.options.mode {
            CollectiveMode::Untiled => self.concat_axis > per_item_rank,
            CollectiveMode::Tiled => self.concat_axis >= per_item_rank,
        };
        if axis_is_out_of_bounds {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "'all_gather' concat axis {} is out of bounds for rank {per_item_rank}",
                    self.concat_axis,
                ),
            });
        }
        let moved = value.move_axis(0, self.concat_axis)?;
        let gathered = match self.options.mode {
            CollectiveMode::Untiled => moved,
            CollectiveMode::Tiled => {
                // The physical layout is `[b, d_0, ..., d_{r-1}]`. Moving the leading batch axis to position
                // `concat_axis` places it immediately before the per-item `concat_axis` dimension, so the row-major
                // merge of `(b, d_c)` into `b * d_c` concatenates the batch items item-major.
                let mut output_dimensions = dimensions[1..].to_vec();
                output_dimensions[self.concat_axis] *= dimensions[0];
                moved.reshape(Shape::new(output_dimensions.into_iter().map(Dimension::Static).collect()))?
            }
        };
        Ok(vec![ArrayBatch::replicated(gathered)])
    }
}

/// Batching rule for [`PSumScatterOperation`]. A matching `batch` level consumes the mapped batch axis by summing
/// over it and re-mapping the chunks of the per-item `scatter_axis` onto it: the sum's `scatter_axis` is split into
/// `(b, d_s / b)` chunks and the new chunk axis becomes the output batch axis, so batch item `i` receives chunk `i`
/// of the sum. A non-matching level forwards the collective untouched to the parent context via
/// [`forward_collective_to_parent`].
impl<C, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for PSumScatterOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<PSumScatterOperation>,
    <C as Domain>::Value: LegacyBroadcast + Reduce + Reshape + Transpose,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs);
        }
        if self.options.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: "'psum_scatter' axis index groups are not supported when a batch transform binds the \
                          collective axis"
                    .to_string(),
            });
        }
        let (value, dimensions) = shape_changing_collective_batch_operand(
            PSUM_SCATTER_OPERATION_NAME,
            &self.axis_name,
            self.axis_size,
            inputs,
        )?;
        let batch_size = dimensions[0];
        let per_item_rank = dimensions.len() - 1;
        if self.scatter_axis >= per_item_rank {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "'psum_scatter' scatter axis {} is out of bounds for rank {per_item_rank}",
                    self.scatter_axis,
                ),
            });
        }
        let scatter_dimension = dimensions[self.scatter_axis + 1];
        let summed = value.reduce(&[0], ReductionKind::Sum);
        let scattered = match self.options.mode {
            CollectiveMode::Untiled => {
                if scatter_dimension != batch_size {
                    return Err(BatchingError::UnsupportedOperation {
                        message: format!(
                            "'psum_scatter' untiled scatter axis {} size {scatter_dimension} must equal axis size \
                             {batch_size}",
                            self.scatter_axis,
                        ),
                    });
                }
                summed.move_axis(self.scatter_axis, 0)?
            }
            CollectiveMode::Tiled => {
                if scatter_dimension % batch_size != 0 {
                    return Err(BatchingError::UnsupportedOperation {
                        message: format!(
                            "'psum_scatter' scatter axis {} size {scatter_dimension} is not divisible by axis size \
                             {batch_size}",
                            self.scatter_axis,
                        ),
                    });
                }
                // Split the per-item `scatter_axis` into `(b, d_s / b)` chunks and map the chunk axis at the front.
                let mut split_dimensions = dimensions[1..].to_vec();
                split_dimensions[self.scatter_axis] = batch_size;
                split_dimensions.insert(self.scatter_axis + 1, scatter_dimension / batch_size);
                let split =
                    summed.reshape(Shape::new(split_dimensions.into_iter().map(Dimension::Static).collect()))?;
                split.move_axis(self.scatter_axis, 0)?
            }
        };
        let physical_type = scattered.r#type().into_owned();
        Ok(vec![ArrayBatch::new(physical_type, scattered, Some(0))?])
    }
}

/// Batching rule for [`PpermuteOperation`]. A matching `batch` level consumes the mapped batch axis by reassembling
/// it in target order: for each position `t` along the batch axis, the output receives the slice of the source item
/// that sends to `t`, or a zero slice when no pair targets `t`. A non-matching level forwards the collective
/// untouched to the parent context via [`forward_collective_to_parent`].
impl<C, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for PpermuteOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<PpermuteOperation>,
    <C as Domain>::Value: LegacyBroadcast + Concatenate + Slice + Transpose + ZeroLike,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs);
        }
        let (value, dimensions) =
            shape_changing_collective_batch_operand(PPERMUTE_OPERATION_NAME, &self.axis_name, self.axis_size, inputs)?;
        let batch_size = dimensions[0];
        // Map each target position along the batch axis to the source item that sends to it; positions that no pair
        // targets receive zeros. Pair uniqueness is enforced by output type inference, so it is not revalidated here.
        let mut sources = vec![None; batch_size];
        for (source, target) in &self.source_target_pairs {
            if *source >= batch_size || *target >= batch_size {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "'ppermute' pair ({source}, {target}) is out of bounds for axis size {batch_size}",
                    ),
                });
            }
            sources[*target] = Some(*source);
        }
        // Slice each item `[i, i + 1)` from the leading batch axis and concatenate the slices back in target order.
        let rank = dimensions.len();
        let strides = vec![1; rank];
        let slice_item = |item: usize| -> Result<<C as Domain>::Value, ProgramError> {
            let mut start_indices = vec![0; rank];
            let mut limit_indices = dimensions.clone();
            start_indices[0] = item;
            limit_indices[0] = item + 1;
            value.slice(&start_indices, &limit_indices, &strides)
        };
        let mut zero_item = None;
        let mut items = Vec::with_capacity(batch_size);
        for source in sources {
            match source {
                Some(source) => items.push(slice_item(source)?),
                None => {
                    if zero_item.is_none() {
                        zero_item = Some(slice_item(0)?.zero_like());
                    }
                    // The zero slice was materialized right above when absent.
                    items.push(zero_item.clone().unwrap());
                }
            }
        }
        let permuted = Concatenate::concatenate(&items, 0)?;
        let physical_type = permuted.r#type().into_owned();
        Ok(vec![ArrayBatch::new(physical_type, permuted, Some(0))?])
    }
}

/// Batching rule for [`AllToAllOperation`]. A matching `batch` level consumes the mapped batch axis with a
/// reshape/transpose block exchange: the per-item `split_axis` is split into `(b, d_p / b)` chunks, the chunk axis
/// is swapped with the leading batch axis (so the batch axis indexes the *receiving* item), and the sender axis is
/// then merged item-major into the per-item `concat_axis` — batch item `i` receives every item's chunk `i`,
/// concatenated along `concat_axis`. A non-matching level forwards the collective untouched to the parent context
/// via [`forward_collective_to_parent`].
impl<C, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for AllToAllOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<AllToAllOperation>,
    <C as Domain>::Value: LegacyBroadcast + Reshape + Transpose,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            return forward_collective_to_parent(context, C::Operation::from(self.clone()), inputs);
        }
        if self.options.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: "'all_to_all' axis index groups are not supported when a batch transform binds the \
                          collective axis"
                    .to_string(),
            });
        }
        let (value, dimensions) = shape_changing_collective_batch_operand(
            ALL_TO_ALL_OPERATION_NAME,
            &self.axis_name,
            self.axis_size,
            inputs,
        )?;
        let batch_size = dimensions[0];
        let per_item_rank = dimensions.len() - 1;
        if self.split_axis >= per_item_rank || self.concat_axis >= per_item_rank {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "'all_to_all' split axis {} or concat axis {} is out of bounds for rank {per_item_rank}",
                    self.split_axis, self.concat_axis,
                ),
            });
        }
        let split_dimension = dimensions[self.split_axis + 1];
        match self.options.mode {
            CollectiveMode::Untiled if split_dimension != batch_size => {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "'all_to_all' untiled split axis {} size {split_dimension} must equal axis size {batch_size}",
                        self.split_axis,
                    ),
                });
            }
            CollectiveMode::Tiled if split_dimension % batch_size != 0 => {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "'all_to_all' split axis {} size {split_dimension} is not divisible by axis size {batch_size}",
                        self.split_axis,
                    ),
                });
            }
            CollectiveMode::Untiled | CollectiveMode::Tiled => {}
        }
        // Split the per-item `split_axis` (physical position `split_axis + 1`) into `(b, d_p / b)` so its leading
        // factor indexes the chunks, then swap the chunk axis with the leading sender axis: afterwards the leading
        // axis indexes the *receiving* item and the axis at `split_axis + 1` indexes the sender.
        let mut split_dimensions = dimensions.clone();
        split_dimensions[self.split_axis + 1] = batch_size;
        split_dimensions.insert(self.split_axis + 2, split_dimension / batch_size);
        let split = value.reshape(Shape::new(split_dimensions.into_iter().map(Dimension::Static).collect()))?;
        let exchanged = split.swap_axes(0, self.split_axis + 1)?;
        let received = match self.options.mode {
            CollectiveMode::Untiled => {
                // The chunk size is one. Remove that singleton, then move the sender axis from the deleted split
                // position to the requested materialized named-axis position.
                let mut squeezed_dimensions = dimensions;
                squeezed_dimensions[self.split_axis + 1] = batch_size;
                let squeezed =
                    exchanged.reshape(Shape::new(squeezed_dimensions.into_iter().map(Dimension::Static).collect()))?;
                squeezed.move_axis(self.split_axis + 1, self.concat_axis + 1)?
            }
            CollectiveMode::Tiled => {
                // Move the sender axis before `concat_axis` and merge it with that existing dimension.
                let moved = exchanged.move_axis(self.split_axis + 1, self.concat_axis + 1)?;
                let mut output_dimensions = dimensions;
                output_dimensions[self.split_axis + 1] /= batch_size;
                output_dimensions[self.concat_axis + 1] *= batch_size;
                moved.reshape(Shape::new(output_dimensions.into_iter().map(Dimension::Static).collect()))?
            }
        };
        let physical_type = received.r#type().into_owned();
        Ok(vec![ArrayBatch::new(physical_type, received, Some(0))?])
    }
}

/// Transpose rule for [`AllGatherOperation`]. A varying all-gather is the adjoint of a sum-scatter with the same
/// mode, axis, and participant groups, so the operand cotangent is a [`PSumScatterOperation`] of the output
/// cotangent. Invariant and reduced variance require the Phase 6 residual-aware adjoints.
impl<V, O> TransposableOperation<V, O> for AllGatherOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<PSumScatterOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if self.output_variance != AllGatherOutputVariance::Varying {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "transposing 'all_gather' with {:?} output variance requires the Phase 6 variance-specific \
                     adjoint",
                    self.output_variance,
                ),
            }
            .into());
        }
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            PSumScatterOperation::new(self.axis_name.clone(), self.axis_size, self.concat_axis, self.options.clone()),
        )
    }
}

/// Transpose rule for [`PSumScatterOperation`]. A sum-scatter is the adjoint of a varying all-gather with the same
/// mode, axis, and participant groups, so the operand cotangent is an [`AllGatherOperation`] of the output
/// cotangent.
impl<V, O> TransposableOperation<V, O> for PSumScatterOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<AllGatherOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            AllGatherOperation::new(
                self.axis_name.clone(),
                self.axis_size,
                self.scatter_axis,
                self.options.clone(),
                AllGatherOutputVariance::Varying,
            ),
        )
    }
}

/// Transpose rule for [`PpermuteOperation`]: sending along `(source, target)` pulls cotangents back along
/// `(target, source)`, so the operand cotangent is the permutation with every pair inverted.
impl<V, O> TransposableOperation<V, O> for PpermuteOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<PpermuteOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        let inverted_pairs =
            self.source_target_pairs.iter().map(|(source, target)| (*target, *source)).collect::<Vec<_>>();
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            PpermuteOperation::new(self.axis_name.clone(), self.axis_size, inverted_pairs),
        )
    }
}

/// Transpose rule for [`AllToAllOperation`]: the chunk exchange is its own adjoint with the split and concatenation
/// axes swapped.
impl<V, O> TransposableOperation<V, O> for AllToAllOperation
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<AllToAllOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        transpose_shape_changing_collective(
            context,
            inputs,
            outputs,
            AllToAllOperation::new(
                self.axis_name.clone(),
                self.axis_size,
                self.concat_axis,
                self.split_axis,
                self.options.clone(),
            ),
        )
    }
}

/// Stages the adjoint collective of a linear shape-changing collective on the output cotangent: a known operand
/// receives a structural zero, a zero output cotangent stays symbolic, and a live cotangent rides the provided
/// adjoint operation.
fn transpose_shape_changing_collective<V, O, A>(
    context: &mut TracingContext<V, O>,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    adjoint: A,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError>
where
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<A>,
    A: Operation<ArrayType>,
{
    check_count!("input", inputs, 1, ProgramError);
    check_count!("output", outputs, 1, ProgramError);
    if inputs[0].is_known() {
        return Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]);
    }
    match &outputs[0] {
        MaybeZero::Value(cotangent) => {
            let mut contributions = context.bind(O::from(adjoint), Vec::new(), std::slice::from_ref(cotangent))?;
            check_count!("output", contributions, 1, ProgramError);
            Ok(vec![MaybeZero::Value(contributions.remove(0))])
        }
        MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::array_programs::batching::{ArrayProgramBatch, ArrayProgramBatching};
    use crate::backends::array_programs::{ArrayProgramOperation, ArrayProgramValue};
    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::dimensions::DimensionValue;
    use crate::batching::{
        ArrayBatch, BatchAxis, BatchAxisSpecification, BatchableOperation, BatchingContext, BatchingError,
        BatchingTracer, batch,
    };
    use crate::contexts::EagerContext;
    use crate::differentiation::value_and_gradient;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use crate::types::{ArrayProgramType, Dimension, DimensionBounds, DimensionType, DimensionVariable, Shape};

    use super::*;

    /// Creates an active batching frame binding the named axis `"i"` over an eager parent whose operation family
    /// contains every operation the collective batching rule may bind (notably constants and broadcasts for `PMean`).
    fn batching_context(
        axis_size: usize,
    ) -> BatchingContext<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching> {
        BatchingContext::new(EagerContext::new(), axis_size).with_axis_name("i".to_string())
    }

    #[test]
    fn test_collective_psum_reduces_along_the_batch_axis() {
        // Mapped input shape [3] at axis 0: per-item scalar. PSum collapses the batch axis to a
        // replicated scalar holding the total.
        let input = {
            let value = Array::vector(vec![1.0, 2.0, 3.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = CollectiveOperation::new("i".to_string(), CollectiveKind::PSum)
            .batch(&batching_context(3), &crate::EmptyRegionDriver, &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[6.0]);
    }

    #[test]
    fn test_collective_pmax_reduces_along_the_batch_axis() {
        let input = {
            let value = Array::vector(vec![1.0, 4.0, 2.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = CollectiveOperation::new("i".to_string(), CollectiveKind::PMax)
            .batch(&batching_context(3), &crate::EmptyRegionDriver, &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[4.0]);
    }

    #[test]
    fn test_collective_pmean_divides_by_batch_size() {
        // Per-item scalar input of shape [3] mapped at axis 0. PMean returns the mean of the three batch items as a
        // replicated scalar, exercising the `1 / N` factor that distinguishes it from PSum. The batching frame binds
        // the axis name `"data"` to show the rule matches on the collective's own axis name rather than a fixture
        // default.
        let input = {
            let value = Array::vector(vec![2.0, 4.0, 6.0]);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3)
            .with_axis_name("data".to_string());
        let outputs = CollectiveOperation::new("data".to_string(), CollectiveKind::PMean)
            .batch(&context, &crate::EmptyRegionDriver, &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        let values = outputs[0].value().to_f64s();
        assert_eq!(values.len(), 1);
        let delta = (values[0] - 4.0).abs();
        assert!(delta < 1e-9, "expected pmean = 4.0, got {}", values[0]);
    }

    #[test]
    fn test_collective_passes_through_replicated_input() {
        let input = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]));
        let outputs = CollectiveOperation::new("i".to_string(), CollectiveKind::PSum)
            .batch(&batching_context(3), &crate::EmptyRegionDriver, &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_collective_over_unbound_axis_is_rejected() {
        use crate::axes::AxisError;
        use crate::backends::arrays::{Array, ArrayOperation};
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer};
        use crate::contexts::EagerContext;

        // The batch binds the axis `"i"`, but the collective names `"j"`, which no enclosing transform binds. Rather
        // than silently acting as identity (the pre-validation behavior), staging the collective fails fast with
        // `AxisError::UnboundAxisName`, matching JAX's error for a collective over an unbound axis name. The error
        // rides the `ProgramError::Custom` channel as a `BatchingError::Axis` and is re-typed at the public `batch`
        // boundary, so the surfaced error is exactly that variant.
        let result: Result<Array, BatchingError> = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.collective("j", CollectiveKind::PSum)
            },
            Array::vector(vec![1.0, 2.0, 3.0]),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "j".to_string() }));
    }

    #[test]
    fn test_collective_psum_value_and_grad_through_vmap_re_sums_the_cotangent() {
        use crate::backends::arrays::{Array, ArrayOperation};
        use crate::batching::BatchAxisSpecification;
        use crate::contexts::EagerContext;
        use crate::differentiation::LinearizationTracer;

        // `g(x) = psum_i(x)`: the vmapped `psum` over the mapped axis `"i"` consumes that axis, producing the
        // replicated total `S = Σ_j x_j`. Reverse mode pulls the scalar ones cotangent back through the
        // self-adjoint `psum`, which re-broadcasts the cotangent across the batch items, giving `∂g/∂x_i = 1`
        // for every input. With `x = [1, 2, 3]` the value is `6` and the gradient is `[1, 1, 1]`.
        let (value, gradient) = value_and_gradient(
            |x: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let total: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = batch(
                    |item| item.collective("i", CollectiveKind::PSum),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                total
            },
            Array::vector(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(gradient.to_f64s(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_collective_pmean_value_and_grad_through_vmap_carries_the_inverse_batch_size() {
        use crate::backends::arrays::{Array, ArrayOperation};
        use crate::batching::BatchAxisSpecification;
        use crate::contexts::EagerContext;
        use crate::differentiation::LinearizationTracer;

        // `g(x) = pmean_i(x)`: the vmapped `pmean` over the mapped axis `"i"` consumes that axis, producing the
        // replicated mean `M = (1/N)·Σ_j x_j`. Reverse mode pulls the scalar ones cotangent back through the
        // self-adjoint `pmean`, which carries the `1/N` factor, so `∂g/∂x_i = 1/N` for every input. With `x =
        // [1, 2, 3]` (so `N = 3`) the value is `2` and the gradient is `[1/3, 1/3, 1/3]`, witnessing the `1/N`
        // scaling that distinguishes `pmean` from `psum`.
        let (value, gradient) = value_and_gradient(
            |x: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>>| {
                let mean: LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>> = batch(
                    |item| item.collective("i", CollectiveKind::PMean),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                mean
            },
            Array::vector(vec![1.0, 2.0, 3.0]),
        )
        .unwrap();
        assert_eq!(value.to_f64s(), vec![2.0]);
        assert_eq!(gradient.to_f64s(), vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn test_nested_batch_named_axes_route_collectives_to_matching_level() {
        // The inner `psum` targets the *outer* named axis, so each inner batch item must reduce over the
        // outer batch items: column sums of [[1, 2], [3, 4]].
        let x = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let output: Array = batch(
            |row| {
                Ok(batch(
                    |scalar| scalar.collective("outer", CollectiveKind::PSum),
                    row,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("inner"),
                )?)
            },
            x,
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("outer"),
        )
        .unwrap();

        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])),);
        assert_eq!(output.to_f64s(), vec![4.0, 6.0]);
    }

    /// Returns the static `f32` vector type of the provided length used by the shape-changing collective tests.
    fn f32_vector(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(length)]))
    }

    #[test]
    fn test_collective_options_validate_axis_index_groups() {
        let options = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        assert_eq!(options.mode(), CollectiveMode::Tiled);
        assert_eq!(options.axis_index_groups(), Some([vec![0, 2], vec![3, 1]].as_slice()));
        assert_eq!(options.effective_axis_size("all_gather", 4), Ok(2));

        assert_eq!(
            CollectiveOptions::default().with_axis_index_groups(Vec::new()).effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("'all_gather' axis index groups must not be empty")),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![2]])
                .effective_axis_size("all_gather", 3),
            Err(TypeError::invalid("'all_gather' axis index group 1 has size 1 but every group must have size 2",)),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![1, 2]])
                .effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("'all_gather' axis index groups contain participant 1 more than once",)),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![2, 4]])
                .effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("'all_gather' axis index 4 is out of bounds for axis size 4")),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1]])
                .effective_axis_size("all_gather", 3),
            Err(TypeError::invalid("'all_gather' axis index groups do not contain participant 2")),
        );
    }

    #[test]
    fn test_grouped_reduction_collective_validates_and_renders_partition() {
        let operation =
            CollectiveOperation::grouped("x".to_string(), CollectiveKind::PMean, 4, vec![vec![0, 2], vec![3, 1]])
                .unwrap();
        assert_eq!(operation.axis_size(), Some(4));
        assert_eq!(operation.group_size(), Some(2));
        assert_eq!(operation.axis_index_groups(), Some([vec![0, 2], vec![3, 1]].as_slice()));
        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::F32)], &[]).unwrap(),
            vec![ArrayType::scalar(DataType::F32)],
        );
        assert_eq!(operation.to_string(), "pmean [axis_name=\"x\", axis_size=4, axis_index_groups=[[0, 2], [3, 1]]]",);

        assert!(matches!(
            CollectiveOperation::grouped("x".to_string(), CollectiveKind::PSum, 4, vec![vec![0, 1], vec![1, 2]]),
            Err(TypeError::Invalid { .. }),
        ));
    }

    #[test]
    fn test_pshuffle_composes_ppermute_in_the_composite_domain() {
        type Parent = EagerContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let context = BatchingContext::<_, ArrayProgramBatching>::new(
            Parent::new(),
            ArrayProgramValue::Dimension(DimensionValue::constant(3).unwrap()),
        )
        .with_axis_name("x".to_string());
        let input = ArrayProgramValue::Array(Array::matrix(3, 2, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let input = ArrayProgramBatch::new(input, BatchAxis::new(0)).unwrap();
        let input = BatchingTracer::new(context, input);
        let output = input.pshuffle("x", &[2, 0, 1]).unwrap().into_batch();

        assert_eq!(output.batch_axis(), BatchAxis::new(0));
        let ArrayProgramValue::Array(output) = output.into_value() else {
            panic!("pshuffle must preserve the array member kind");
        };
        assert_eq!(output.to_f64s(), vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pswapaxes_composes_untiled_all_to_all_in_the_composite_domain() {
        type TestContext = TracingContext<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>;

        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let (_, program) = TestContext::trace_with_named_axes(
            |input| input.pswapaxes("x", 0),
            ArrayProgramType::Array(input_type),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2 })],
        )
        .unwrap();

        let all_to_all = program.instructions().last().unwrap();
        let ArrayProgramOperation::AllToAll(operation) = all_to_all.operation() else {
            panic!("pswapaxes must compose the canonical all-to-all operation");
        };
        assert_eq!(operation.split_axis(), 0);
        assert_eq!(operation.concat_axis(), 0);
        assert_eq!(operation.options(), &CollectiveOptions::default());
        assert_eq!(all_to_all.inputs().len(), 3);
        assert_eq!(all_to_all.inputs()[0], program.input_ids()[0]);
    }

    #[test]
    fn test_untiled_collective_type_inference() {
        let shape = |dimensions| ArrayType::new(DataType::F32, Shape::new(dimensions));

        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    4,
                    1,
                    CollectiveOptions::default(),
                    AllGatherOutputVariance::Varying,
                ),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(3)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().clone().into(),
                    DimensionValue::constant(4).unwrap().r#type().clone().into(),
                    DimensionValue::constant(3).unwrap().r#type().clone().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            infer_explicit_psum_scatter_output_types(
                &PSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().clone().into(),
                    DimensionValue::constant(3).unwrap().r#type().clone().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 4, 1, 0, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(4).unwrap().r#type().clone().into(),
                    DimensionValue::constant(2).unwrap().r#type().clone().into(),
                    DimensionValue::constant(3).unwrap().r#type().clone().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(4), Dimension::Static(2), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 4, 1, 1, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().clone().into(),
                    DimensionValue::constant(4).unwrap().r#type().clone().into(),
                    DimensionValue::constant(3).unwrap().r#type().clone().into(),
                ],
            ),
            Ok(vec![shape(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Static(3)]).into()]),
        );
        assert_eq!(
            infer_explicit_psum_scatter_output_types(
                &PSumScatterOperation::new("x".to_string(), 4, 1, CollectiveOptions::default()),
                &[
                    shape(vec![Dimension::Static(2), Dimension::Static(5)]).into(),
                    DimensionValue::constant(2).unwrap().r#type().clone().into(),
                ],
            ),
            Err(TypeError::invalid("'psum_scatter' untiled scatter axis 1 size 5 must equal group size 4",)),
        );
    }

    #[test]
    fn test_grouped_collective_shape_arithmetic_uses_group_size() {
        let grouped = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        let result_extent = DimensionValue::constant(6).unwrap().r#type().clone();
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new("x".to_string(), 4, 0, grouped.clone(), AllGatherOutputVariance::Varying,),
                &[f32_vector(3).into(), result_extent.into(),],
            ),
            Ok(vec![f32_vector(6).into()]),
        );
        assert_eq!(
            infer_explicit_psum_scatter_output_types(
                &PSumScatterOperation::new("x".to_string(), 4, 0, grouped),
                &[f32_vector(6).into(), DimensionValue::constant(3).unwrap().r#type().clone().into()],
            ),
            Ok(vec![f32_vector(3).into()]),
        );
    }

    #[test]
    fn test_all_gather_output_variance_updates_canonical_sharding_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = Sharding::replicated(mesh, 1).with_varying_manual_axes(["x"]).unwrap();
        let input = f32_vector(3).with_sharding(varying_sharding).unwrap();

        let infer = |output_variance| {
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new("x".to_string(), 2, 0, CollectiveOptions::default(), output_variance),
                &[
                    ArrayProgramType::Array(input.clone()),
                    DimensionValue::constant(2).unwrap().r#type().clone().into(),
                    DimensionValue::constant(3).unwrap().r#type().clone().into(),
                ],
            )
        };
        let varying = infer(AllGatherOutputVariance::Varying).unwrap();
        let varying = <&ArrayType>::try_from(&varying[0]).unwrap();
        assert_eq!(varying.sharding().unwrap().varying_manual_axes(), &["x".to_string()].into_iter().collect());
        assert!(varying.sharding().unwrap().reduced_axes().is_empty());

        let invariant = infer(AllGatherOutputVariance::Invariant).unwrap();
        let invariant = <&ArrayType>::try_from(&invariant[0]).unwrap();
        assert!(invariant.sharding().unwrap().varying_manual_axes().is_empty());
        assert!(invariant.sharding().unwrap().reduced_axes().is_empty());

        let reduced = infer(AllGatherOutputVariance::Reduced).unwrap();
        let reduced = <&ArrayType>::try_from(&reduced[0]).unwrap();
        assert!(reduced.sharding().unwrap().varying_manual_axes().is_empty());
        assert_eq!(reduced.sharding().unwrap().reduced_axes(), &["x".to_string()].into_iter().collect());
    }

    #[test]
    fn test_explicit_shape_changing_collective_type_inference() {
        let input_axis = DimensionVariable::new("input", DimensionBounds::new(1, Some(17)).unwrap());
        let split_result = DimensionVariable::new("split", DimensionBounds::new(1, Some(9)).unwrap());
        let concat_result = DimensionVariable::new("concat", DimensionBounds::new(2, Some(33)).unwrap());
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(input_axis.clone()), Dimension::Static(3)]),
        );

        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                &[
                    input_type.clone().into(),
                    ArrayProgramType::Dimension(DimensionType::new(concat_result.clone())),
                    DimensionValue::constant(3).unwrap().r#type().clone().into(),
                ],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(concat_result.clone()), Dimension::Static(3)]),
                )
                .into()
            ]),
        );
        assert_eq!(
            infer_explicit_psum_scatter_output_types(
                &PSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::tiled()),
                &[
                    input_type.clone().into(),
                    ArrayProgramType::Dimension(DimensionType::new(split_result.clone())),
                    DimensionValue::constant(3).unwrap().r#type().clone().into(),
                ],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(split_result.clone()), Dimension::Static(3)]),
                )
                .into()
            ]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 2, 0, 1, CollectiveOptions::tiled()),
                &[
                    input_type.clone().into(),
                    ArrayProgramType::Dimension(DimensionType::new(split_result.clone())),
                    ArrayProgramType::Dimension(DimensionType::new(concat_result.clone())),
                ],
            ),
            Ok(vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(split_result), Dimension::Dynamic(concat_result),]),
                )
                .into()
            ]),
        );
        assert_eq!(
            infer_explicit_all_to_all_output_types(
                &AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::tiled()),
                &[
                    ArrayProgramType::Array(input_type.clone()),
                    ArrayProgramType::Dimension(DimensionType::new(input_axis)),
                    DimensionValue::constant(3).unwrap().r#type().clone().into(),
                ],
            ),
            Ok(vec![input_type.into()]),
        );

        let exact_six = DimensionValue::constant(6).unwrap().r#type().clone();
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                &[f32_vector(3).into(), exact_six.into()],
            ),
            Ok(vec![f32_vector(6).into()]),
        );
        let exact_five = DimensionValue::constant(5).unwrap().r#type().clone();
        assert_eq!(
            infer_explicit_all_gather_output_types(
                &AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying
                ),
                &[f32_vector(3).into(), exact_five.into()],
            ),
            Err(TypeError::invalid(
                "'all_gather' result extent must equal input axis 0 extent 3 multiplied by axis group size 2; \
                 expected 6 \
                 but got 5"
                    .to_string(),
            )),
        );
        assert_eq!(
            infer_explicit_psum_scatter_output_types(
                &PSumScatterOperation::new("empty".to_string(), 0, 0, CollectiveOptions::tiled()),
                &[f32_vector(3).into(), DimensionValue::constant(3).unwrap().r#type().clone().into()],
            ),
            Err(TypeError::invalid("'psum_scatter' axis size must be greater than zero")),
        );
    }

    #[test]
    fn test_all_gather_type_inference() {
        use crate::macros::check_operation_type_inference;

        let operation = AllGatherOperation::new(
            "x".to_string(),
            4,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        );
        assert_eq!(operation.axis_name(), "x");
        assert_eq!(operation.axis_size(), 4);
        assert_eq!(operation.concat_axis(), 0);
        assert_eq!(operation.name(), ALL_GATHER_OPERATION_NAME);
        assert_eq!(
            operation.to_string(),
            indoc::indoc! {r#"
                all_gather [
                    axis_name="x",
                    axis_size=4,
                    concat_axis=0,
                    options=Tiled,
                    output_variance=Varying,
                ]
            "#}
            .trim_end(),
        );
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    input_types = [f32_vector(2)],
                    output_types = [f32_vector(8)],
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "'all_gather' concat axis 0 is out of bounds for rank 0",
                },
                {
                    input_types = [ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]))],
                    error = "'all_gather' does not support dynamically shaped operands",
                },
            ],
        );
        check_operation_type_inference!(
            @reject @unreduced,
            operation = AllGatherOperation::new("x".to_string(), 4, 0, CollectiveOptions::tiled(), AllGatherOutputVariance::Varying),
            input_types = [f32_vector(2)],
        );
    }

    #[test]
    fn test_psum_scatter_type_inference() {
        use crate::macros::check_operation_type_inference;

        check_operation_type_inference!(
            operation = PSumScatterOperation::new("x".to_string(), 4, 0, CollectiveOptions::tiled()),
            cases = [
                {
                    input_types = [f32_vector(8)],
                    output_types = [f32_vector(2)],
                },
                {
                    input_types = [f32_vector(6)],
                    error = "'psum_scatter' scatter axis 0 size 6 is not divisible by group size 4",
                },
                {
                    input_types = [ArrayType::scalar(DataType::F32)],
                    error = "'psum_scatter' scatter axis 0 is out of bounds for rank 0",
                },
            ],
        );
    }

    #[test]
    fn test_ppermute_type_inference() {
        use crate::macros::check_operation_type_inference;

        check_operation_type_inference!(
            operation = PpermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]),
            cases = [{
                input_types = [f32_vector(3)],
                output_types = [f32_vector(3)],
            }],
        );
        check_operation_type_inference!(
            operation = PpermuteOperation::new("x".to_string(), 2, vec![(0, 2)]),
            cases = [{
                input_types = [f32_vector(3)],
                error = "'ppermute' pair (0, 2) is out of bounds for axis size 2",
            }],
        );
        check_operation_type_inference!(
            operation = PpermuteOperation::new("x".to_string(), 2, vec![(0, 1), (0, 0)]),
            cases = [{
                input_types = [f32_vector(3)],
                error = "'ppermute' pairs must have unique sources and targets but (0, 0) repeats one",
            }],
        );
    }

    #[test]
    fn test_all_to_all_type_inference() {
        use crate::macros::check_operation_type_inference;

        let matrix = |rows: usize, columns: usize| {
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(rows), Dimension::Static(columns)]))
        };
        check_operation_type_inference!(
            operation = AllToAllOperation::new("x".to_string(), 4, 0, 1, CollectiveOptions::tiled()),
            cases = [
                {
                    input_types = [matrix(8, 3)],
                    output_types = [matrix(2, 12)],
                },
                {
                    input_types = [matrix(6, 3)],
                    error = "'all_to_all' split axis 0 size 6 is not divisible by group size 4",
                },
            ],
        );
    }

    #[test]
    fn test_all_gather_interpretation_requires_an_enclosing_binder() {
        use crate::interpretation::InterpretableOperation;

        // A single-participant axis is degenerate: the gather concatenates exactly one operand, so interpretation is
        // the identity.
        let outputs = AllGatherOperation::new(
            "x".to_string(),
            1,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        )
        .interpret(
            &EagerContext::<Array, ArrayOperation<Array>>::new(),
            &crate::EmptyRegionDriver,
            &[Array::vector(vec![1.0, 2.0])],
        )
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].values(), &[1.0, 2.0]);

        // Any larger axis has no per-item semantics: the other participants do not exist outside an enclosing binder.
        let error = AllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        )
        .interpret(
            &EagerContext::<Array, ArrayOperation<Array>>::new(),
            &crate::EmptyRegionDriver,
            &[Array::vector(vec![1.0, 2.0])],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ProgramError::UnsupportedOperation { message }
                if message == "cannot interpret 'all_gather' over axis 'x' of size 2 without an enclosing binder",
        ));
    }

    #[test]
    fn test_all_gather_over_unbound_axis_is_rejected() {
        use crate::batching::BatchingTracer;

        // The batch binds only the axis `"i"`, but the `all_gather` names `"x"`, which no enclosing transform binds.
        // Axis-size resolution fails fast at staging time with `AxisError::UnboundAxisName` rather than silently
        // acting as identity.
        let result: Result<Array, BatchingError> = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.all_gather("x", 0, CollectiveOptions::tiled(), AllGatherOutputVariance::Varying)
            },
            Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "x".to_string() }));
    }

    #[test]
    fn test_all_gather_over_batched_axis_materializes_the_gather() {
        use crate::batching::BatchingTracer;

        // The batch binds the axis `"x"` that the `all_gather` names, so the matching batching rule consumes the
        // mapped axis: every item receives the item-major concatenation of all items along `concat_axis`,
        // replicated across the batch. With items `[1, 2]` and `[3, 4]` the gathered value is `[1, 2, 3, 4]`,
        // matching the verified cross-device `shard_map` execution semantics of the tiled StableHLO `all_gather`.
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.all_gather("x", 0, CollectiveOptions::tiled(), AllGatherOutputVariance::Varying)
            },
            Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(4)])));
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_all_gather_of_replicated_input_concatenates_copies() {
        // A replicated operand at a matching level is first materialized as `axis_size` identical batch items, so
        // the gather degenerates to the item-major concatenation of that many copies of the shared value.
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let outputs = AllGatherOperation::new(
            "x".to_string(),
            2,
            0,
            CollectiveOptions::tiled(),
            AllGatherOutputVariance::Varying,
        )
        .batch(&context, &crate::EmptyRegionDriver, &[ArrayBatch::replicated(Array::vector(vec![1.0, 2.0]))])
        .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().values(), &[1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_untiled_collectives_over_batched_axis_materialize_rank_changes() {
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let mapped_matrix = || {
            ArrayBatch::new(
                Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]).r#type().into_owned(),
                Array::matrix(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0]),
                Some(0),
            )
            .unwrap()
        };

        let gathered = AllGatherOperation::new(
            "x".to_string(),
            2,
            1,
            CollectiveOptions::default(),
            AllGatherOutputVariance::Varying,
        )
        .batch(&context, &crate::EmptyRegionDriver, &[mapped_matrix()])
        .unwrap();
        assert_eq!(gathered[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(gathered[0].value(), &Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0]),);

        let scattered = PSumScatterOperation::new("x".to_string(), 2, 0, CollectiveOptions::default())
            .batch(&context, &crate::EmptyRegionDriver, &[mapped_matrix()])
            .unwrap();
        assert_eq!(scattered[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(scattered[0].value(), &Array::vector(vec![4.0_f32, 6.0]));
        assert_eq!(scattered[0].unbatched_type(), ArrayType::scalar(DataType::F32));

        let exchanged = AllToAllOperation::new("x".to_string(), 2, 0, 0, CollectiveOptions::default())
            .batch(&context, &crate::EmptyRegionDriver, &[mapped_matrix()])
            .unwrap();
        assert_eq!(exchanged[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(exchanged[0].value(), &Array::matrix(2, 2, vec![1.0_f32, 3.0, 2.0, 4.0]),);
    }

    #[test]
    fn test_psum_scatter_over_batched_axis_sums_and_scatters() {
        use crate::batching::BatchingTracer;

        // The batch binds the axis `"x"` that the `psum_scatter` names, so the matching batching rule sums over the
        // mapped axis and re-maps the chunks of `scatter_axis` onto it. With items `[1, 2, 3, 4]` and
        // `[10, 20, 30, 40]` the sum is `[11, 22, 33, 44]`, so item 0 receives `[11, 22]` and item 1 receives
        // `[33, 44]`, matching the verified cross-device `shard_map` execution semantics of StableHLO's
        // `reduce_scatter`.
        let x = Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.psum_scatter("x", 0, CollectiveOptions::tiled())
            },
            x,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(output.to_f64s(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_ppermute_over_batched_axis_permutes_the_items() {
        use crate::batching::BatchingTracer;

        // The rotation `[(0, 1), (1, 0)]` swaps the two batch items: item 0 receives item 1's `[3, 4]` and item 1
        // receives item 0's `[1, 2]`.
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.ppermute("x", vec![(0, 1), (1, 0)])
            },
            Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(output.to_f64s(), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_ppermute_over_batched_axis_zeros_untargeted_items() {
        use crate::batching::BatchingTracer;

        // With the single pair `(0, 1)`, item 1 receives item 0's `[1, 2]` while no pair targets item 0, so it
        // receives zeros, matching JAX's `ppermute` semantics for untargeted participants.
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.ppermute("x", vec![(0, 1)])
            },
            Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(output.to_f64s(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_all_to_all_over_batched_axis_exchanges_chunks() {
        use crate::batching::BatchingTracer;

        // Block exchange with `split_axis == concat_axis == 0`: each item splits its vector into two chunks and
        // receives its own chunk index from every item, concatenated item-major. With items `[1, 2, 3, 4]` and
        // `[5, 6, 7, 8]`, item 0 receives `[1, 2, 5, 6]` and item 1 receives `[3, 4, 7, 8]`, matching the verified
        // cross-device `shard_map` execution semantics of StableHLO's `all_to_all`.
        let x = Array::matrix(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.all_to_all("x", 0, 0, CollectiveOptions::tiled())
            },
            x,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)])),
        );
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_all_to_all_over_batched_axis_with_distinct_axes_exchanges_chunks() {
        use crate::batching::BatchingTracer;

        // Distinct split and concatenation axes over per-item `[2, 2]` matrices: each item splits its rows across
        // the items and receives its own row index from every item, concatenated item-major along the columns. With
        // item 0 = `[[1, 2], [3, 4]]` and item 1 = `[[5, 6], [7, 8]]`, item 0 receives `[[1, 2, 5, 6]]` and item 1
        // receives `[[3, 4, 7, 8]]` (per-item shape `[1, 4]`).
        let x = Array::from_f64s(
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
            ),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let output: Array = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.all_to_all("x", 0, 1, CollectiveOptions::tiled())
            },
            x,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(1), Dimension::Static(4)])
            ),
        );
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_all_gather_transposes_to_psum_scatter() {
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // A tiled all-gather is the adjoint of a sum-scatter over the same axis and dimension, so the pullback stages
        // a `psum_scatter` on the output cotangent with the gather's concat axis as its scatter axis.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(f32_vector(2));
        let output = builder
            .add_instruction(
                AllGatherOperation::new(
                    "x".to_string(),
                    2,
                    0,
                    CollectiveOptions::tiled(),
                    AllGatherOutputVariance::Varying,
                ),
                Vec::new(),
                vec![input],
            )
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc::indoc! {r#"
                lambda %0:f32[4] .
                let %1:f32[2] = psum_scatter [axis_name="x", axis_size=2, scatter_axis=0, options=Tiled] %0
                in (%1)
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_ppermute_transposes_to_inverted_pairs() {
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;

        // Sending along `(source, target)` pulls cotangents back along `(target, source)`, so the pullback stages the
        // permutation with every pair inverted.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(f32_vector(2));
        let output = builder
            .add_instruction(PpermuteOperation::new("x".to_string(), 2, vec![(0, 1), (1, 0)]), Vec::new(), vec![input])
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc::indoc! {r#"
                lambda %0:f32[2] .
                let %1:f32[2] = ppermute [axis_name="x", axis_size=2, source_target_pairs=[(1, 0), (0, 1)]] %0
                in (%1)
            "#}
            .trim_end(),
        );
    }
}
