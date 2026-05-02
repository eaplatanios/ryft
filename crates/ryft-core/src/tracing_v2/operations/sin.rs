use std::fmt::{Debug, Display};

use half::{bf16, f16};

use crate::macros::check_input_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableOperation, LinearEngine};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

use super::SupportsScale;
use super::cos::Cos;

/// Hidden carrier capability for staging the sine primitive.
#[doc(hidden)]
pub trait SupportsSin<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the sine primitive.
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
#[derive(Clone, Default)]
pub struct SinOperation;

impl Debug for SinOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Sin")
    }
}

impl Display for SinOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "sin")
    }
}

impl<T: Type> Operation<T> for SinOperation {
    fn name(&self) -> &'static str {
        "sin"
    }

    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        check_input_count!(input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl<V: Typed<ArrayType> + Clone + Sin> InterpretableOperation<ArrayType, V> for SinOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().sin()])
    }
}

impl<V: Typed<DataType> + Clone + Sin> InterpretableOperation<DataType, V> for SinOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().sin()])
    }
}

impl<E> DifferentiableOperation<E> for SinOperation
where
    E: LinearEngine + ?Sized,
    SinOperation: Operation<E::Type>,
    E::Value: Sin + Cos + Differentiable<E::Type, Tangent = E::Value>,
    E::LinearOperation: SupportsScale<E::Type, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        let input = &inputs[0];
        let tangent = context
            .apply_operation(
                &[input.tangent],
                <E::LinearOperation as SupportsScale<E::Type, E::Value>>::scale_operation(input.primal.clone().cos()),
                1,
            )?
            .into_iter()
            .next()
            .expect("sin jvp should produce one tangent");
        Ok(vec![JvpTracer { primal: input.primal.clone().sin(), tangent }])
    }
}

impl<'engine, E> Sin for Tracer<'engine, E>
where
    E: TracingEngine + ?Sized,
    E::Value: Sin,
    E::Operation: SupportsSin<E::Type, E::Value>,
{
    #[inline]
    fn sin(self) -> Self {
        self.unary(E::Operation::sin_operation())
    }
}
