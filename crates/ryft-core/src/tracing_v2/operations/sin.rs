use std::fmt::Display;

use half::{bf16, f16};

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableOperation, LinearizableEngine};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

use super::SupportsScale;
use super::cos::Cos;

/// Trait that represents [`Operation`] carrier types that support/include [`SinOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`SinOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsSin<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the sine [`Operation`].
    fn sin_operation() -> Self;
}

/// Elementwise sine capability.
///
/// This trait is the value-level entry point that lets generic user code write `x.sin()` whether
/// `x` is a concrete scalar, a traced leaf, or a JVP leaf.
pub trait Sin: Sized {
    /// Computes the elementwise sine.
    fn sin(self) -> Self;
}

impl Sin for f32 {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
}

impl Sin for f64 {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
}

impl Sin for bf16 {
    #[inline]
    fn sin(self) -> Self {
        Self::from_f32(self.to_f32().sin())
    }
}

impl Sin for f16 {
    #[inline]
    fn sin(self) -> Self {
        Self::from_f32(self.to_f32().sin())
    }
}

/// Elementwise sine primitive.
///
/// [`SinOperation`] is the staged-program representation of the sine primitive. Ordinary traced programs
/// store this op (or a backend-specific carrier that wraps it), while JVP rules delegate through
/// its semantic implementation.
#[derive(Clone, Debug, Default)]
pub struct SinOperation;

impl Display for SinOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(<Self as Operation<ArrayType>>::name(self))
    }
}

impl<T: Type> Operation<T> for SinOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "sin"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<V: Clone + Typed<ArrayType> + Sin> InterpretableOperation<ArrayType, V> for SinOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().sin()])
    }
}

impl<V: Clone + Typed<DataType> + Sin> InterpretableOperation<DataType, V> for SinOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().sin()])
    }
}

impl<E> DifferentiableOperation<E> for SinOperation
where
    E: LinearizableEngine,
    SinOperation: Operation<E::Type>,
    E::Value: Sin + Cos + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsScale<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let input = &inputs[0];
        let tangent_outputs = context.stage(
            <E::LinearOperationCarrier as SupportsScale<E::Type, E::Value>>::scale_operation(
                input.primal.clone().cos(),
            ),
            &[input.tangent],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal: input.primal.clone().sin(), tangent: tangent_outputs[0] }])
    }
}

impl<'engine, E> Sin for Tracer<'engine, E>
where
    E: TracingEngine,
    E::Value: Sin,
    E::OperationCarrier: SupportsSin<E::Type, E::Value>,
{
    #[inline]
    fn sin(self) -> Self {
        self.unary(E::OperationCarrier::sin_operation())
    }
}
