use std::fmt::Display;
use std::ops::Mul;

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::operations::constants::{Fill, SupportsFill};
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::tracing::Tracer;
use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Kind of collective performed by a [`CollectiveOperation`].
///
/// Collectives operate on a named batched axis: when the surrounding
/// [`BatchingContext`](crate::tracing_v2::batching::BatchingContext) maps an axis with the matching
/// name, the collective consumes that axis. The operations described here mirror JAX's
/// `jax.lax.{psum, pmean, pmax}` family.
///
/// `PSum`/`PMean`/`PMax` reduce the mapped axis away, producing a result that is identical
/// across all lanes (lane-uniform). `AllGather`-style gather variants are deferred until the
/// machinery for shape-extending collectives lands.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CollectiveKind {
    /// Sum reduction across the named axis (`jax.lax.psum`).
    PSum,

    /// Mean reduction across the named axis (`jax.lax.pmean`).
    PMean,

    /// Maximum reduction across the named axis (`jax.lax.pmax`).
    PMax,
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

/// Trait for operation types that include or can wrap [`CollectiveOperation`].
/// Backend-owned closed operation enums (such as
/// [`ArrayOperation`](super::ArrayOperation), for example) implement this trait so that generic
/// transform code can stage [`CollectiveOperation`] without knowing the concrete operation enum.
#[doc(hidden)]
pub trait SupportsCollective<T: Type> {
    /// Constructs the backend-specific representation of the collective [`Operation`] with the
    /// provided axis name and kind.
    fn collective_operation(axis_name: String, kind: CollectiveKind) -> Self;
}

/// Value-level entry point for staging a collective operation.
///
/// The staged operation references the surrounding [`BatchingContext`](crate::tracing_v2::batching::BatchingContext)
/// by name; outside of any matching context it lowers to an identity pass-through (the operand carries no mapped axis
/// to reduce). Inside a matching context,
/// [`BatchableOperation::batch`](crate::tracing_v2::batching::BatchableOperation::batch)
/// collapses the mapped axis.
pub trait Collective: Sized {
    /// Stages a collective of the given kind referencing axis `axis_name`.
    fn collective(self, axis_name: &str, kind: CollectiveKind) -> Self;
}

impl<C> Collective for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: SupportsCollective<ArrayType>,
{
    #[inline]
    fn collective(self, axis_name: &str, kind: CollectiveKind) -> Self {
        self.unary(C::Operation::collective_operation(axis_name.to_string(), kind))
    }
}

/// Primitive representing one named-axis collective operation.
///
/// [`CollectiveOperation`] is identity at the per-lane level (the named axis does not exist in
/// per-lane semantics) and collapses the mapped axis when invoked inside a
/// [`BatchingContext`](crate::tracing_v2::batching::BatchingContext) whose
/// [`axis_name`](crate::tracing_v2::batching::BatchingContext::axis_name) matches this collective's axis name. Under
/// nested `batch` levels, the traced batching rule below owns that decision: a matching level consumes the mapped
/// lane axis, while a non-matching level forwards the collective untouched to its parent context via
/// [`forward_collective_to_parent`], where the next level repeats the same name resolution. The value-level rule has
/// no level metadata to match against and always reduces the mapped axis, which corresponds to eager batching with a
/// single level.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CollectiveOperation {
    /// Axis name referenced by this collective. Matches the `axis_name` field of an enclosing
    /// [`BatchingContext::with_axis_name`](crate::tracing_v2::batching::BatchingContext::with_axis_name) call.
    axis_name: String,

    /// Kind of collective.
    kind: CollectiveKind,
}

impl CollectiveOperation {
    /// Creates a new [`CollectiveOperation`] with the supplied axis name and kind.
    #[inline]
    pub fn new(axis_name: String, kind: CollectiveKind) -> Self {
        Self { axis_name, kind }
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

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        // The per-lane operation is identity; the named axis only exists physically inside an
        // enclosing `BatchingContext` where the batching rule will collapse it.
        Ok(vec![input_types[0].clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("axis_name", format_args!("{:?}", self.axis_name)))
    }
}

impl<V: Value<ArrayType>> InterpretableOperation<ArrayType, V> for CollectiveOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // Outside a batching domain the collective is identity: per-lane semantics says the
        // named axis does not exist, so reducing across it is a no-op. JAX errors in this case,
        // but our staged programs can also encounter this when the operation is interpreted
        // directly on a fully-eager value.
        Ok(vec![inputs[0].clone()])
    }
}

/// Value-level batching rule for eager backends, where the reduced value already carries its concrete data and a
/// `PMean`'s `1 / N` factor can be synthesized directly through [`Fill`].
///
/// Both this and the traced [`BatchingContext`](crate::tracing_v2::batching::BatchingContext) rule below share
/// [`collective_reduce_batch`]; they differ only in how the `PMean` factor is produced. A [`Tracer`] cannot satisfy
/// the [`Type`]-driven [`Fill`] used here because it has no ambient context, so the traced rule stages the fill
/// instead.
impl<V: Value<ArrayType> + Reduce + Fill<ArrayType, f64> + Mul<Output = V>>
    crate::tracing_v2::batching::BatchableOperation<V, ()> for CollectiveOperation
where
    CollectiveOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &(),
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        collective_reduce_batch(self.kind, inputs, |factor_type, inverse_axis_size| {
            V::fill(&factor_type, inverse_axis_size)
        })
    }
}

/// Traced batching rule for [`Tracer`] values inside a [`BatchingContext`](
/// crate::tracing_v2::batching::BatchingContext). This rule owns named-axis resolution: when the context's
/// [`axis_name`](crate::tracing_v2::batching::BatchingContext::axis_name) matches this collective's axis name, the
/// mapped lane axis is consumed; otherwise the collective targets an outer `batch` level and is forwarded untouched
/// to the parent context via [`forward_collective_to_parent`].
///
/// The consuming arm shares [`collective_reduce_batch`] with the eager rule above but stages a `PMean`'s `1 / N`
/// rank-0 fill into the reduced value's own parent context (via [`StagingContext::stage_operation`]) instead of
/// synthesizing it through the [`Type`]-driven [`Fill`], which a [`Tracer`] cannot implement.
impl<C> crate::tracing_v2::batching::BatchableOperation<Tracer<C>, crate::tracing_v2::batching::BatchingContext<C>>
    for CollectiveOperation
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: SupportsCollective<ArrayType> + SupportsFill<ArrayType, f64>,
    Tracer<C>: Reduce + Mul<Output = Tracer<C>>,
    CollectiveOperation: InterpretableOperation<ArrayType, Tracer<C>>,
{
    fn batch(
        &self,
        context: &crate::tracing_v2::batching::BatchingContext<C>,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<Tracer<C>>>, ProgramError> {
        if context.axis_name() != Some(self.axis_name.as_str()) {
            let parent_operation = C::Operation::collective_operation(self.axis_name.clone(), self.kind);
            return forward_collective_to_parent(context, parent_operation, inputs);
        }
        collective_reduce_batch(self.kind, inputs, |factor_type, inverse_axis_size| {
            inputs[0]
                .value()
                .context()
                .stage_operation::<&Tracer<C>>(C::Operation::fill_operation(factor_type, inverse_axis_size), &[])?
                .into_iter()
                .next()
                .ok_or(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }.into())
        })
    }
}

/// Re-stages a collective that targets a different (outer) named axis into the batching context's parent.
///
/// Under nested `batch` levels, a collective is consumed by the level whose
/// [`axis_name`](crate::tracing_v2::batching::BatchingContext::axis_name) matches its axis name and must pass through
/// every inner level untouched: each inner lane participates in the outer collective independently, so the operands'
/// mapped axes are preserved as-is on the forwarded outputs. The parent may itself be another
/// [`BatchingContext`](crate::tracing_v2::batching::BatchingContext) — whose own rule dispatch repeats this name
/// resolution at the next level — or an ordinary tracing context. Batching rules for custom collective-like
/// operations should use this helper for their "not my axis" arm.
pub fn forward_collective_to_parent<C: StagingContext<Type = ArrayType>>(
    context: &crate::tracing_v2::batching::BatchingContext<C>,
    parent_operation: C::Operation,
    inputs: &[crate::tracing_v2::batching::ArrayBatch<Tracer<C>>],
) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<Tracer<C>>>, ProgramError> {
    let parent_input_tracers: Vec<&Tracer<C>> = inputs.iter().map(|batch| batch.value()).collect();
    let parent_outputs = context.parent_context().stage_operation(parent_operation, parent_input_tracers.as_slice())?;
    check_count!("output", parent_outputs, inputs.len(), ProgramError);
    parent_outputs
        .into_iter()
        .zip(inputs.iter())
        .map(|(parent_tracer, input_batch)| {
            let physical_type = parent_tracer.r#type().into_owned();
            crate::tracing_v2::batching::ArrayBatch::new(physical_type, parent_tracer, input_batch.batch_axis())
        })
        .collect()
}

/// Shared reduce-and-optionally-mean skeleton for [`CollectiveOperation`] batching, used by both the eager and traced
/// rules above. It collapses the mapped lane axis with the kind's [`ReductionKind`] and, for `PMean`, scales the
/// lane-uniform result by `1 / N` using a `make_pmean_factor`-produced rank-0 factor (relying on implicit rank-0
/// broadcasting in the multiplication). Outside a matching batching context (no mapped axis), it is an identity
/// pass-through. The two callers differ only in `make_pmean_factor`: eager backends synthesize the factor directly,
/// while traced contexts stage it into their owning program.
fn collective_reduce_batch<V, MakePMeanFactor>(
    kind: CollectiveKind,
    inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    make_pmean_factor: MakePMeanFactor,
) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError>
where
    V: Value<ArrayType> + Reduce + Mul<Output = V>,
    MakePMeanFactor: FnOnce(ArrayType, f64) -> Result<V, ProgramError>,
{
    check_count!("input", inputs, 1, ProgramError);
    let input = &inputs[0];
    let Some(batch_axis) = input.batch_axis() else {
        // Outside any matching batching context: identity pass-through.
        return Ok(vec![input.clone()]);
    };
    // Reduce along the mapped lane axis with the corresponding reduction kind. The output is lane-uniform: every lane
    // sees the same reduced value, matching JAX's `psum`/`pmean`/`pmax` broadcast semantics.
    let mut output_value = input.value().clone().reduce(&[batch_axis], kind.reduction_kind());
    if matches!(kind, CollectiveKind::PMean) {
        // PMean divides the summed value by the lane count, which must be statically known to scale by `1 / N`.
        let inverse_axis_size = 1.0 / pmean_lane_count(input)? as f64;
        let factor_type = pmean_factor_type(output_value.r#type().data_type());
        output_value = make_pmean_factor(factor_type, inverse_axis_size)? * output_value;
    }
    let output_type = output_value.r#type().into_owned();
    Ok(vec![crate::tracing_v2::batching::ArrayBatch::new(output_type, output_value, None)?])
}

/// Returns the static lane count for a `PMean` over the mapped batch axis of `input`, erroring when
/// the lane size is dynamic (a mean cannot be scaled by `1 / N` without a static `N`).
fn pmean_lane_count<V: Value<ArrayType>>(
    input: &crate::tracing_v2::batching::ArrayBatch<V>,
) -> Result<usize, ProgramError> {
    input.axis_size()?.ok_or_else(|| {
        crate::batching::BatchingError::UnsupportedOperation {
            message: "pmean requires a static lane size; the staged batch axis is dynamic".to_string(),
        }
        .into()
    })
}

/// Builds the rank-0 [`ArrayType`] of `data_type` used to hold a `PMean`'s `1 / N` factor.
fn pmean_factor_type(data_type: DataType) -> ArrayType {
    ArrayType::new(data_type, crate::types::Shape::scalar())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tests::TestArray;
    use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};

    use super::*;

    #[test]
    fn test_collective_psum_reduces_along_batched_lane_axis() {
        // Mapped input shape [3] at axis 0: per-lane scalar. PSum collapses the lane axis to a
        // lane-uniform scalar holding the total.
        let input = ArrayBatch::mapped(TestArray::vector(vec![1.0, 2.0, 3.0]), 0).unwrap();
        let outputs = CollectiveOperation::new("i".to_string(), CollectiveKind::PSum).batch(&(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].value().values(), &[6.0]);
    }

    #[test]
    fn test_collective_pmax_reduces_along_batched_lane_axis() {
        let input = ArrayBatch::mapped(TestArray::vector(vec![1.0, 4.0, 2.0]), 0).unwrap();
        let outputs = CollectiveOperation::new("i".to_string(), CollectiveKind::PMax).batch(&(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].value().values(), &[4.0]);
    }

    #[test]
    fn test_collective_passes_through_lane_uniform_input() {
        let input = ArrayBatch::unbatched(TestArray::vector(vec![1.0, 2.0, 3.0]));
        let outputs = CollectiveOperation::new("i".to_string(), CollectiveKind::PSum).batch(&(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), None);
        assert_eq!(outputs[0].value().values(), &[1.0, 2.0, 3.0]);
    }
}
