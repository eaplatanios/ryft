//! Contains the named-axis [`ParallelReduceOperation`], which reduces a value across a named axis (`parallel_sum` /
//! `parallel_mean` / `parallel_max`), together with its interpretation, partial-evaluation, batching, forward-mode
//! differentiation, and transposition rules. These are the analogues of
//! [JAX's parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators) `jax.lax.psum`,
//! `jax.lax.pmean`, and `jax.lax.pmax`.

// TODO(eaplatanios): Review this module.

use std::fmt::Display;
use std::ops::Mul as StdMul;

use crate::arrays::{ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayType, DataType, Shape};
use crate::axes::{AxisError, NamedAxes};
use crate::batching::{BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::constants::fill::Fill;
use crate::operations::math::reduce::{Reduce, ReductionKind};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
};

use super::{
    effective_collective_axis_size, forward_collective_to_parent, reject_ragged_collective_inputs,
    resolve_named_axis_size,
};

/// Kind of collective performed by a [`ParallelReduceOperation`].
///
/// Collectives operate on a named axis, resolved against the active [`NamedAxes`] environment. When an enclosing
/// `batch` level binds the name (a [`NamedAxis::Batched`](crate::axes::NamedAxis) axis), the matching
/// [`BatchingContext`] consumes the mapped batch axis at trace time. When a `shard_map` manual region binds the name
/// to a device mesh axis (a [`NamedAxis::Mesh`](crate::axes::NamedAxis) axis), the collective stays in the staged body
/// and lowers to a cross-device `all_reduce` over that mesh axis. The operations described here mirror JAX's
/// `jax.lax.{psum, pmean, pmax}` family.
///
/// `Sum`/`Mean`/`Max` reduce the named axis away, producing a result that is identical across all batch items or
/// device shards (replicated). Shape-changing collectives use their dedicated operation payloads below because their
/// result types also depend on a ranked array axis and on whether the named axis is materialized or tiled.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParallelReductionKind {
    /// Sum reduction across the named axis (`jax.lax.psum`).
    Sum,

    /// Mean reduction across the named axis (`jax.lax.pmean`).
    Mean,

    /// Maximum reduction across the named axis (`jax.lax.pmax`).
    Max,
}

impl ParallelReductionKind {
    /// Returns the canonical operation name suffix for this kind.
    pub fn name(self) -> &'static str {
        match self {
            Self::Sum => "parallel_sum",
            Self::Mean => "parallel_mean",
            Self::Max => "parallel_max",
        }
    }

    /// Returns the [`ReductionKind`] used to collapse the named axis.
    pub fn reduction_kind(self) -> ReductionKind {
        match self {
            Self::Sum | Self::Mean => ReductionKind::Sum,
            Self::Max => ReductionKind::Max,
        }
    }
}

impl Display for ParallelReductionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

/// Primitive representing one named-axis collective operation.
///
/// [`ParallelReduceOperation`] is identity at the per-item level (the named axis does not exist in per-item semantics)
/// and collapses the mapped axis when invoked inside a [`BatchingContext`] whose
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches this collective's axis name. Under nested
/// `batch` levels, the batching rule below owns that decision: a matching level consumes the mapped batch axis, while
/// a non-matching level forwards the collective untouched to its parent context via
/// [`forward_collective_to_parent`], where the next level repeats the same name resolution.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParallelReduceOperation {
    /// Axis name referenced by this collective. Matches the `axis_name` argument of an enclosing
    /// [`BatchingContext::new`](crate::batching::BatchingContext::new) call.
    axis_name: String,

    /// Kind of collective.
    kind: ParallelReductionKind,

    /// Full size of the named axis when participant subgroups are present.
    axis_size: Option<usize>,

    /// Optional ordered partition of logical participant indices.
    axis_index_groups: Option<Vec<Vec<usize>>>,
}

impl ParallelReduceOperation {
    /// Creates a new [`ParallelReduceOperation`] with the supplied axis name and kind.
    #[inline]
    pub fn new(axis_name: String, kind: ParallelReductionKind) -> Self {
        Self { axis_name, kind, axis_size: None, axis_index_groups: None }
    }

    /// Creates a grouped collective after validating that `axis_index_groups` is an equal-sized exact partition of
    /// `0..axis_size`.
    pub fn grouped(
        axis_name: String,
        kind: ParallelReductionKind,
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
    pub fn kind(&self) -> ParallelReductionKind {
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

impl Display for ParallelReduceOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ParallelReduceOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        match self.kind {
            ParallelReductionKind::Sum => "parallel_sum",
            ParallelReductionKind::Mean => "parallel_mean",
            ParallelReductionKind::Max => "parallel_max",
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
                    "`{}` must store both the full axis size and axis index groups, or neither",
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

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for ParallelReduceOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // Per-item interpretation is identity: the named axis does not exist in per-item semantics, so reducing across
        // it is a no-op. Staging a collective over an unbound axis is now rejected up front (see `ParallelReduce`), so
        // an enclosing binder always collapses the mapped axis through the batching rule before this per-item fallback
        // matters; it remains defined for a program interpreted directly, where a collective simply passes its operand
        // through per item.
        Ok(vec![inputs[0].clone()])
    }
}

// Partial evaluation defers to the default fold-or-residualize behavior of
// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ParallelReduceOperation where
    C::Operation: From<ParallelReduceOperation>
{
}

// Batching rule for [`ParallelReduceOperation`]. This rule owns named-axis resolution: when the active context's
// [`axis_name`](crate::batching::BatchingContext::axis_name) matches this collective's axis name, the mapped batch
// axis is consumed; otherwise the collective targets an outer `batch` level (or a device mesh axis) and is forwarded
// untouched to the parent context via [`forward_collective_to_parent`].
//
// The consuming arm collapses the mapped axis through `collective_reduce_batch` and binds a `Mean`'s `1 / N`
// rank-0 fill into the parent context — interpreted eagerly under an eager parent and staged into the enclosing
// trace under a staging parent — so one rule serves eager and staged batching alike.
impl<C, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for ParallelReduceOperation
where
    C: Context<Type = ArrayType> + Fill<f64, C::Value>,
    C::Operation: From<ParallelReduceOperation>,
    <C as Domain>::Value: Reduce + StdMul<Output = <C as Domain>::Value>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        reject_ragged_collective_inputs(self.name(), inputs)?;
        if context.axis_name() != Some(self.axis_name.as_str()) {
            let parent_operation = C::Operation::from(self.clone());
            return Ok(forward_collective_to_parent(context, parent_operation, inputs)?.into());
        }
        if self.axis_index_groups.is_some() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{}` axis index groups are not supported when a batch transform binds the collective axis",
                    self.name(),
                ),
            });
        }
        Ok(collective_reduce_batch(self.kind, inputs, |factor_type, inverse_axis_size| {
            // The `1 / N` rank-0 factor binds into the batching context's parent — interpreted eagerly under an eager
            // parent, staged into the enclosing trace under a staging parent.
            context.parent().fill(&factor_type, inverse_axis_size)
        })?
        .into())
    }
}

/// Shared reduce-and-optionally-mean skeleton for [`ParallelReduceOperation`] batching. It collapses the mapped batch
/// axis with the kind's [`ReductionKind`] and, for `Mean`, scales the replicated result by `1 / N` using a
/// `make_parallel_mean_factor`-produced rank-0 factor (relying on implicit rank-0 broadcasting in the multiplication).
/// Outside a matching batching context (no mapped axis), it is an identity pass-through.
fn collective_reduce_batch<V, MakeParallelMeanFactor>(
    kind: ParallelReductionKind,
    inputs: &[ArrayBatch<V>],
    make_parallel_mean_factor: MakeParallelMeanFactor,
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    V: Value<Type = ArrayType> + Reduce + StdMul<Output = V>,
    MakeParallelMeanFactor: FnOnce(ArrayType, f64) -> Result<V, ProgramError>,
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
    if matches!(kind, ParallelReductionKind::Mean) {
        // Mean divides the summed value by the batch size, which must be statically known to scale by `1 / N`.
        let inverse_axis_size = 1.0 / parallel_mean_batch_size(input)? as f64;
        let factor_type = parallel_mean_factor_type(output_value.r#type().data_type());
        output_value = make_parallel_mean_factor(factor_type, inverse_axis_size)? * output_value;
    }
    Ok(vec![ArrayBatch::new(output_value, BatchAxis::replicated())?])
}

/// Returns the static batch size for a `Mean` over the mapped batch axis of `input`, erroring when
/// the batch size is dynamic (a mean cannot be scaled by `1 / N` without a static `N`).
fn parallel_mean_batch_size<V: Value<Type = ArrayType>>(input: &ArrayBatch<V>) -> Result<usize, BatchingError> {
    input.batch_size()?.ok_or_else(|| BatchingError::UnsupportedOperation {
        message: "`parallel_mean` requires a static batch size; the staged batch axis is dynamic".to_string(),
    })
}

/// Builds the rank-0 [`ArrayType`] of `data_type` used to hold a `Mean`'s `1 / N` factor.
fn parallel_mean_factor_type(data_type: DataType) -> ArrayType {
    ArrayType::new(data_type, Shape::scalar())
}

impl_differentiable_operation! {
    ParallelReduceOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ParallelReduceOperation>,
    {
        |operation, context, _driver, inputs| {
            // Forward-mode (JVP) rule for [`ParallelReduceOperation`]. `Sum`/`Mean` are linear and self-adjoint, so the
            // tangent is the same collective applied to the operand tangent: `tangent_out =
            // collective(input.tangent())`. A structural-zero operand tangent is preserved as-is rather than staging a
            // collective on a zero, keeping `collective(zero)` out of the tangent program. `Max` is non-linear and
            // reports an [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
            check_count!("input", inputs, 1, ProgramError);
            if matches!(operation.kind, ParallelReductionKind::Max) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "`parallel_max` differentiation is not yet supported".to_string(),
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
        O: Operation<Type = ArrayType> + From<ParallelReduceOperation>,
    {
        |operation, context, _driver, inputs, outputs| {
            // Transpose rule for [`ParallelReduceOperation`]. `parallel_sum`/`parallel_mean` are self-adjoint, so the
            // operand cotangent is the same collective applied to the output cotangent. The single operand is linear
            // (its [`PartialValue`] is [`Unknown`](PartialValue::Unknown)); a known operand contributes no cotangent
            // and so receives a structural zero. `Max` reports an
            // [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            if matches!(operation.kind, ParallelReductionKind::Max) {
                return Err(ProgramError::UnsupportedOperation {
                    message: "`parallel_max` transpose is not yet supported".to_string(),
                }
                .into());
            }
            // A known (non-linear) operand contributes no cotangent.
            if inputs[0].is_known() {
                return Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent()?)]);
            }
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let contribution = stage_collective(context, operation, cotangent)?;
                    Ok(vec![MaybeZero::Value(contribution)])
                }
                MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent()?)]),
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
    operation: &ParallelReduceOperation,
    operand: &C::Value,
) -> Result<C::Value, ProgramError>
where
    C: Context<Type = ArrayType>,
    C::Operation: From<ParallelReduceOperation>,
{
    let mut outputs = context.bind(operation.clone(), Vec::new(), std::slice::from_ref(operand))?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Value-level entry point for staging a collective operation.
///
/// The staged operation references an enclosing named-axis binder by name and the name is validated against the active
/// [`NamedAxes`] environment at staging time: an unbound name fails fast rather than silently acting as identity. A
/// name bound by an enclosing `batch` level is collapsed at trace time by [`BatchableOperation::batch`] (which reduces
/// the mapped batch axis), while a name bound to a device mesh axis by a `shard_map` manual region stays in the staged
/// body program and lowers to a cross-device `all_reduce` over that mesh axis.
pub trait ParallelReduce: Sized {
    /// Stages a collective of the given kind referencing axis `axis_name`, validating that the name is bound by an
    /// enclosing transform. Returns [`AxisError::UnboundAxisName`] (surfaced as [`BatchingError::Axis`] riding a
    /// [`ProgramError::Custom`] payload) when no enclosing binder binds `axis_name`.
    fn parallel_reduce(&self, axis_name: &str, kind: ParallelReductionKind) -> Result<Self, ProgramError>;

    /// Stages a grouped collective after validating that the groups cover the named axis exactly once.
    fn parallel_reduce_with_axis_index_groups(
        &self,
        axis_name: &str,
        kind: ParallelReductionKind,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError>;
}

// Any context-carrying value applies a collective by validating the axis name against the active [`NamedAxes`]
// environment and binding a [`ParallelReduceOperation`] through its own context: a staged tracer records the
// operation, a batching tracer resolves the named axis against the batching context stack, and a JVP dual forwards to
// the primal-side resolution. An unbound name fails fast with [`AxisError::UnboundAxisName`] rather than silently
// acting as identity.
impl<V: Value> ParallelReduce for V
where
    V::DispatchDomain: Context + NamedAxes,
    <V::DispatchDomain as Domain>::Operation: From<ParallelReduceOperation>,
{
    fn parallel_reduce(&self, axis_name: &str, kind: ParallelReductionKind) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        if context.named_axis(axis_name).is_none() {
            return Err(BatchingError::Axis(AxisError::UnboundAxisName { name: axis_name.to_string() }).into());
        }
        let mut outputs =
            context.bind(ParallelReduceOperation::new(axis_name.to_string(), kind), Vec::new(), &[self.clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    fn parallel_reduce_with_axis_index_groups(
        &self,
        axis_name: &str,
        kind: ParallelReductionKind,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let operation = ParallelReduceOperation::grouped(axis_name.to_string(), kind, axis_size, axis_index_groups)?;
        let mut outputs = context.bind(operation, Vec::new(), &[self.clone()])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayOperation, DataType, Dimension, DimensionBounds, DimensionVariable, RaggedAxis, Shape,
    };
    use crate::batching::{BatchAxisSpecification, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;

    use super::*;

    /// Creates an active batching frame binding the named axis `"i"` over an eager parent whose operation family
    /// contains every operation the collective batching rule may bind (notably constants and broadcasts for `Mean`).
    fn batching_context(
        axis_size: usize,
    ) -> BatchingContext<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching> {
        BatchingContext::new(EagerContext::new(), axis_size).with_axis_name("i".to_string())
    }

    #[test]
    fn test_parallel_reduce_sum_reduces_along_the_batch_axis() {
        // Mapped input shape [3] at axis 0: per-item scalar. A `Sum` reduction collapses the batch axis to a
        // replicated scalar holding the total.
        let input = {
            let value = Array::vector(vec![1.0, 2.0, 3.0]);
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let outputs = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum)
            .batch(&batching_context(3), &crate::EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().to_f64s(), vec![6.0]);
    }

    #[test]
    fn test_parallel_reduce_max_reduces_along_the_batch_axis() {
        let input = {
            let value = Array::vector(vec![1.0, 4.0, 2.0]);
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let outputs = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Max)
            .batch(&batching_context(3), &crate::EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().to_f64s(), vec![4.0]);
    }

    #[test]
    fn test_parallel_reduce_mean_divides_by_batch_size() {
        // Per-item scalar input of shape [3] mapped at axis 0. A `Mean` reduction returns the mean of the three batch
        // items as a replicated scalar, exercising the `1 / N` factor that distinguishes it from `Sum`. The batching
        // frame binds the axis name `"data"` to show the rule matches on the collective's own axis name rather than a
        // fixture default.
        let input = {
            let value = Array::vector(vec![2.0, 4.0, 6.0]);
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 3)
            .with_axis_name("data".to_string());
        let outputs = ParallelReduceOperation::new("data".to_string(), ParallelReductionKind::Mean)
            .batch(&context, &crate::EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        let values = outputs[0].value().to_f64s();
        assert_eq!(values.len(), 1);
        let delta = (values[0] - 4.0).abs();
        assert!(delta < 1e-9, "expected parallel_mean = 4.0, got {}", values[0]);
    }

    #[test]
    fn test_parallel_reduce_batching_rejects_ragged_operands_before_binding() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0_f32; 6]), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]), variable, vec![0])])
            .unwrap();

        assert_eq!(
            ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum).batch(
                &batching_context(2),
                &crate::EmptyRegionDriver,
                &[input]
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`parallel_sum` does not support bounded ragged dimension `length` on operand 0".to_string(),
            }),
        );
    }

    #[test]
    fn test_parallel_reduce_passes_through_replicated_input() {
        let input = ArrayBatch::replicated(Array::vector(vec![1.0, 2.0, 3.0]));
        let outputs = ParallelReduceOperation::new("i".to_string(), ParallelReductionKind::Sum)
            .batch(&batching_context(3), &crate::EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parallel_reduce_over_unbound_axis_is_rejected() {
        use crate::arrays::{Array, ArrayOperation};
        use crate::axes::AxisError;
        use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer};
        use crate::contexts::EagerContext;

        // The batch binds the axis `"i"`, but the collective names `"j"`, which no enclosing transform binds. Rather
        // than silently acting as identity (the pre-validation behavior), staging the collective fails fast with
        // `AxisError::UnboundAxisName`, matching JAX's error for a collective over an unbound axis name. The error
        // rides the `ProgramError::Custom` channel as a `BatchingError::Axis` and is re-typed at the public `batch`
        // boundary, so the surfaced error is exactly that variant.
        let result: Result<Array, BatchingError> = batch(
            |item: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching>| {
                item.parallel_reduce("j", ParallelReductionKind::Sum)
            },
            Array::vector(vec![1.0, 2.0, 3.0]),
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "j".to_string() }));
    }

    #[test]
    fn test_parallel_reduce_sum_value_and_grad_through_vmap_re_sums_the_cotangent() {
        use crate::arrays::Array;
        use crate::batching::BatchAxisSpecification;

        // `g(x) = parallel_sum_i(x)`: the vmapped `parallel_sum` over the mapped axis `"i"` consumes that axis,
        // producing the replicated total `S = Σ_j x_j`. Reverse mode pulls the scalar ones cotangent back through
        // the self-adjoint `parallel_sum`, which re-broadcasts the cotangent across the batch items, giving
        // `∂g/∂x_i = 1` for every input. With `x = [1, 2, 3]` the value is `6` and the gradient is `[1, 1, 1]`.
        let (value, gradient) = differentiate_at(Array::vector(vec![1.0, 2.0, 3.0]))
            .value_and_gradient(|x| {
                let total = batch(
                    |item| item.parallel_reduce("i", ParallelReductionKind::Sum),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                total
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(gradient.to_f64s(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_parallel_reduce_mean_value_and_grad_through_vmap_carries_the_inverse_batch_size() {
        use crate::arrays::Array;
        use crate::batching::BatchAxisSpecification;

        // `g(x) = parallel_mean_i(x)`: the vmapped `parallel_mean` over the mapped axis `"i"` consumes that axis,
        // producing the replicated mean `M = (1/N)·Σ_j x_j`. Reverse mode pulls the scalar ones cotangent back
        // through the self-adjoint `parallel_mean`, which carries the `1/N` factor, so `∂g/∂x_i = 1/N` for every
        // input. With `x = [1, 2, 3]` (so `N = 3`) the value is `2` and the gradient is `[1/3, 1/3, 1/3]`,
        // witnessing the `1/N` scaling that distinguishes `parallel_mean` from `parallel_sum`.
        let (value, gradient) = differentiate_at(Array::vector(vec![1.0, 2.0, 3.0]))
            .value_and_gradient(|x| {
                let mean = batch(
                    |item| item.parallel_reduce("i", ParallelReductionKind::Mean),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::replicated(),
                    BatchAxisSpecification::named("i"),
                )
                .unwrap();
                mean
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![2.0]);
        assert_eq!(gradient.to_f64s(), vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn test_nested_batch_named_axes_route_collectives_to_matching_level() {
        // The inner `parallel_sum` targets the *outer* named axis, so each inner batch item must reduce over the
        // outer batch items: column sums of [[1, 2], [3, 4]].
        let x = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let output: Array = batch(
            |row| {
                Ok(batch(
                    |scalar| scalar.parallel_reduce("outer", ParallelReductionKind::Sum),
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

        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)])));
        assert_eq!(output.to_f64s(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_grouped_reduction_collective_validates_and_renders_partition() {
        let operation = ParallelReduceOperation::grouped(
            "x".to_string(),
            ParallelReductionKind::Mean,
            4,
            vec![vec![0, 2], vec![3, 1]],
        )
        .unwrap();
        assert_eq!(operation.axis_size(), Some(4));
        assert_eq!(operation.group_size(), Some(2));
        assert_eq!(operation.axis_index_groups(), Some([vec![0, 2], vec![3, 1]].as_slice()));
        assert_eq!(
            operation.infer_output_types(&[ArrayType::scalar(DataType::F32)], &[]).unwrap(),
            vec![ArrayType::scalar(DataType::F32)],
        );
        assert_eq!(
            operation.to_string(),
            "parallel_mean [axis_name=\"x\", axis_size=4, axis_index_groups=[[0, 2], [3, 1]]]",
        );

        assert!(matches!(
            ParallelReduceOperation::grouped(
                "x".to_string(),
                ParallelReductionKind::Sum,
                4,
                vec![vec![0, 1], vec![1, 2]]
            ),
            Err(TypeError::Invalid { .. }),
        ));
    }
}
