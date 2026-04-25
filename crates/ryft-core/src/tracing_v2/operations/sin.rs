use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing::{Traceable, TracingError};
use crate::tracing_v2::{
    engines::{DifferentiableEngine, Engine},
    forward::{Differentiable, EngineTangent, JvpTracer, TangentSpace},
    jit::Tracer,
};
use crate::types::{ArrayType, Type, TypeError};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearAddOperation, LinearNegOperation, LinearScaleOperation,
    Operation, cos::Cos, unary_abstract,
};

/// Hidden staging trait for the sine primitive.
#[doc(hidden)]
pub trait SinTracingOperation<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the sine primitive.
    fn sin_op() -> Self;
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

impl Operation<ArrayType> for SinOperation {
    fn name(&self) -> &'static str {
        "sin"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(vec![unary_abstract(input_types)?])
    }
}

impl<V: Traceable<ArrayType> + Sin> InterpretableOperation<ArrayType, V> for SinOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![inputs[0].clone().sin()])
    }
}

impl<E> DifferentiableOperation<E> for SinOperation
where
    E: DifferentiableEngine<Type = ArrayType> + ?Sized,
    E::Value: Traceable<ArrayType> + Sin + Cos + Differentiable<ArrayType>,
    E::LinearOperation: LinearAddOperation<ArrayType, E::Value>
        + LinearNegOperation<ArrayType, E::Value>
        + LinearScaleOperation<ArrayType, E::Value>,
{
    fn jvp(
        &self,
        _engine: &E,
        inputs: &[JvpTracer<E::Value, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<E::Value, EngineTangent<E>>>, TracingError> {
        check_input_count!(inputs, 1);
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().sin(),
            tangent: EngineTangent::<E>::scale(input.primal.clone().cos(), input.tangent.clone()),
        }])
    }
}

impl<V: Traceable<ArrayType> + Sin + Cos, T: TangentSpace<ArrayType, V>> Sin for JvpTracer<V, T> {
    #[inline]
    fn sin(self) -> Self {
        Self { primal: self.primal.clone().sin(), tangent: T::scale(self.primal.cos(), self.tangent) }
    }
}

impl<'engine, V: Traceable<ArrayType> + Sin, E, O> Sin for Tracer<'engine, E, O>
where
    E: Engine<Type = ArrayType, Value = V> + ?Sized,
    O: Clone + Operation<ArrayType> + SinTracingOperation<ArrayType, V>,
{
    #[inline]
    fn sin(self) -> Self {
        self.unary(O::sin_op())
    }
}
