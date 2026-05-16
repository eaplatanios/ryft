use std::fmt::Display;

use half::{bf16, f16};

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::tracing::domains::{Tracer, TracingDomain};
use crate::tracing::{Traceable, TracingError};
use crate::types::{ArrayType, Type, TypeError};

/// Trait that represents [`Operation`] carrier types that support/include [`SelectOperation`].
/// Backend-owned closed [`Operation`] carrier types (such as
/// [`ArrayOperation`](super::ArrayOperation), for example) implement this trait so that generic
/// transform code can stage [`SelectOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsSelect<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the per-element select [`Operation`].
    fn select_operation() -> Self;
}

/// Value-level per-element select capability.
///
/// `Self::select(predicate, on_true, on_false)` returns a value whose `i`-th element equals
/// `on_true`'s `i`-th element when the corresponding `i`-th element of `predicate` is logically
/// true, and `on_false`'s otherwise. All three inputs must share the same shape; for batched
/// applications they share the same lane configuration as well, so the predicate selects
/// per-lane between the two operands.
///
/// For numeric value types whose data type is not strictly Boolean, the convention is that
/// `0.0` (or zero element) is interpreted as false and any non-zero element as true. Real
/// Boolean values (when the data type is [`DataType::Boolean`](crate::types::DataType::Boolean))
/// use `true` / `false` directly.
pub trait Select: Sized {
    /// Per-element select between `on_true` and `on_false` driven by `predicate`.
    fn select(predicate: Self, on_true: Self, on_false: Self) -> Result<Self, TracingError>;
}

impl<'domain, D> Select for Tracer<'domain, D>
where
    D: TracingDomain<Type = ArrayType>,
    D::OperationCarrier: SupportsSelect<ArrayType, D::Value>,
{
    fn select(predicate: Self, on_true: Self, on_false: Self) -> Result<Self, TracingError> {
        let context = predicate.context().clone();
        Ok(context
            .stage(D::OperationCarrier::select_operation(), &[&predicate, &on_true, &on_false])?
            .into_iter()
            .next()
            .expect("select should produce one traced output"))
    }
}

macro_rules! impl_select_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Select for $ty {
                #[inline]
                fn select(predicate: Self, on_true: Self, on_false: Self) -> Result<Self, TracingError> {
                    Ok(if predicate != <$ty>::from_f32(0.0) { on_true } else { on_false })
                }
            }
        )*
    };
}

impl_select_for_scalar!(bf16, f16);

impl Select for f32 {
    #[inline]
    fn select(predicate: Self, on_true: Self, on_false: Self) -> Result<Self, TracingError> {
        Ok(if predicate != 0.0 { on_true } else { on_false })
    }
}

impl Select for f64 {
    #[inline]
    fn select(predicate: Self, on_true: Self, on_false: Self) -> Result<Self, TracingError> {
        Ok(if predicate != 0.0 { on_true } else { on_false })
    }
}

/// Primitive representing one per-element select between two values driven by a predicate.
///
/// All three inputs must share the same shape; the output has the same type as `on_true` (which
/// equals `on_false`).
#[derive(Clone, Copy, Debug)]
pub struct SelectOperation;

impl Display for SelectOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.name())
    }
}

impl Operation<ArrayType> for SelectOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "select"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        if input_types[0].shape() != input_types[1].shape() {
            return Err(TypeError {
                message: (format!(
                    "select predicate shape {} differs from on_true shape {}",
                    input_types[0].shape(),
                    input_types[1].shape(),
                ))
                .into(),
            });
        }
        if input_types[1].shape() != input_types[2].shape() {
            return Err(TypeError {
                message: (format!(
                    "select on_true shape {} differs from on_false shape {}",
                    input_types[1].shape(),
                    input_types[2].shape(),
                ))
                .into(),
            });
        }
        if input_types[1].data_type() != input_types[2].data_type() {
            return Err(TypeError {
                message: (format!(
                    "select on_true data type {} differs from on_false data type {}",
                    input_types[1].data_type(),
                    input_types[2].data_type(),
                ))
                .into(),
            });
        }
        Ok(vec![input_types[1].clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name()).map(|_| ())
    }
}

impl<V: Traceable<ArrayType> + Select> InterpretableOperation<ArrayType, V> for SelectOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 3, TracingError);
        Ok(vec![V::select(inputs[0].clone(), inputs[1].clone(), inputs[2].clone())?])
    }
}

impl<V> crate::tracing_v2::batching::BatchableOperation<V> for SelectOperation
where
    V: Traceable<ArrayType>
        + crate::tracing_v2::operations::broadcast::BroadcastInDim
        + crate::tracing_v2::operations::transpose::Transpose,
    SelectOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, TracingError> {
        crate::tracing_v2::batching::apply_elementwise_batch(self, inputs)
    }
}
