use std::fmt::Display;
use std::ops::Mul;

use crate::macros::check_count;
use crate::operations::constants::ConstantLike;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::{Context, Traceable, Tracer, TracingError};
use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};
use crate::types::{ArrayType, Type, TypeError};

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
pub trait SupportsCollective<T: Type, V: Traceable<T>> {
    /// Constructs the backend-specific representation of the collective [`Operation`] with the
    /// provided axis name and kind.
    fn collective_operation(axis_name: String, kind: CollectiveKind) -> Self;
}

/// Operation-level introspection that lets generic transforms recognize a collective operation
/// without knowing the concrete operation enum.
///
/// [`Context::stage_operation`](crate::tracing::contexts::Context::stage_operation) on
/// [`BatchingContext`](crate::tracing_v2::batching::BatchingContext) uses this to intercept collectives whose
/// `axis_name` matches the enclosing batching context's named axis, lowering them to the corresponding reduction over
/// the mapped lane axis before the operation enum's context-aware batching rule fires.
/// Operation enums without a collective variant should return `None`.
pub trait MaybeCollective {
    /// Returns the collective's axis name and kind when this operation is a collective; `None`
    /// otherwise.
    fn as_collective(&self) -> Option<(&str, CollectiveKind)>;
}

/// Value-level entry point for staging a collective operation.
///
/// The staged operation references the surrounding [`BatchingContext`] by name; outside of any
/// matching context it lowers to an identity pass-through (the operand carries no mapped axis to
/// reduce). Inside a matching context,
/// [`BatchableOperation::batch`](crate::tracing_v2::batching::BatchableOperation::batch)
/// collapses the mapped axis.
pub trait Collective: Sized {
    /// Stages a collective of the given kind referencing axis `axis_name`.
    fn collective(self, axis_name: &str, kind: CollectiveKind) -> Self;
}

impl<C> Collective for Tracer<C>
where
    C: Context<Type = ArrayType>,
    C::Operation: SupportsCollective<ArrayType, C::Value>,
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
/// [`BatchingContext`](crate::tracing_v2::batching::BatchingContext). The current implementation
/// reduces along whichever physical axis the input carries the batch annotation, which matches
/// the named axis when there is a single enclosing `vmap` level. Multi-level / nested
/// name-resolution is a future extension.
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
        write!(formatter, "{}({:?})", self.kind, self.axis_name)
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

impl<V: Traceable<ArrayType>> InterpretableOperation<ArrayType, V> for CollectiveOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        // Outside a batching domain the collective is identity: per-lane semantics says the
        // named axis does not exist, so reducing across it is a no-op. JAX errors in this case,
        // but our staged programs can also encounter this when the operation is interpreted
        // directly on a fully-eager value.
        Ok(vec![inputs[0].clone()])
    }
}

impl<V: Traceable<ArrayType> + Reduce + ConstantLike<f64> + Mul<Output = V>, RuleContext>
    crate::tracing_v2::batching::BatchableOperation<V, RuleContext> for CollectiveOperation
where
    CollectiveOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        _context: &RuleContext,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let Some(batch_axis) = input.batch_axis() else {
            // Outside any matching batching context: identity pass-through.
            return Ok(vec![input.clone()]);
        };
        // Reduce along the mapped lane axis with the corresponding reduction kind. The output is
        // lane-uniform: every lane sees the same reduced value, matching JAX's `psum`/`pmean`/
        // `pmax` broadcast semantics.
        let mut output_value = input.value().clone().reduce(&[batch_axis], self.kind.reduction_kind());
        if matches!(self.kind, CollectiveKind::PMean) {
            // PMean divides the summed value by the lane count. The lane size must be statically
            // known to scale by `1 / N`.
            let axis_size =
                input.axis_size()?.ok_or_else(|| crate::tracing_v2::batching::BatchingError::MissingBatchingRule {
                    operation: "pmean requires a static lane size; the staged batch axis is dynamic".to_string(),
                })?;
            let inverse_axis_size = 1.0 / axis_size as f64;
            let factor = output_value.constant_like(inverse_axis_size);
            output_value = factor * output_value;
        }
        let output_type = output_value.r#type().into_owned();
        Ok(vec![crate::tracing_v2::batching::ArrayBatch::new(output_type, output_value, None)?])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};
    use crate::tracing_v2::test_util::TestArray;

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
    fn test_maybe_collective_recognizes_collective_variants_on_operation() {
        use crate::tracing_v2::operations::collective::MaybeCollective;
        use crate::tracing_v2::operations::primitive::ArrayOperation;
        use crate::types::ArrayType;
        let operation: ArrayOperation<TestArray, ArrayType> =
            ArrayOperation::Collective { axis_name: "data".to_string(), kind: CollectiveKind::PSum };
        assert_eq!(operation.as_collective(), Some(("data", CollectiveKind::PSum)));
        let non_collective: ArrayOperation<TestArray, ArrayType> = ArrayOperation::Add;
        assert_eq!(non_collective.as_collective(), None);
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
