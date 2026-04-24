use std::{
    fmt::{Debug, Display},
    ops::Neg,
};

use crate::macros::check_input_count;
use crate::tracing::{Traceable, TracingError};
use crate::tracing_v2::{
    engine::Engine,
    forward::{Differentiable, EngineTangent, JvpTracer, TangentSpace},
    jit::Tracer,
};
use crate::types::{ArrayType, Type, TypeError};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearAddOperation, LinearNegOperation, LinearScaleOperation,
    Operation, sin::Sin, unary_abstract,
};

/// Hidden staging trait for the cosine primitive.
#[doc(hidden)]
pub trait CosTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the cosine primitive.
    fn cos_op() -> Self;
}

/// Elementwise cosine capability.
///
/// This trait is implemented by concrete values and the transform-local wrappers that can represent
/// cosine symbolically.
pub trait Cos: Sized {
    /// Computes the elementwise cosine.
    fn cos(self) -> Self;
}

impl Cos for f32 {
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl Cos for f64 {
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

/// Elementwise cosine primitive.
///
/// [`CosOperation`] is stored in staged programs whenever cosine is traced explicitly, and its JVP rule is
/// reused by both forward-mode evaluation and higher-order traced transforms.
#[derive(Clone, Default)]
pub struct CosOperation;

impl Debug for CosOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Cos")
    }
}

impl Display for CosOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "cos")
    }
}

impl Operation<ArrayType> for CosOperation {
    fn name(&self) -> &'static str {
        "cos"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        Ok(vec![unary_abstract(input_types)?])
    }
}

impl<V: Traceable<ArrayType> + Cos> InterpretableOperation<ArrayType, V> for CosOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![inputs[0].clone().cos()])
    }
}

impl<E> DifferentiableOperation<E> for CosOperation
where
    E: Engine<Type = ArrayType> + ?Sized,
    E::Value: Traceable<ArrayType> + Cos + Sin + Neg<Output = E::Value> + Differentiable<ArrayType>,
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
            primal: input.primal.clone().cos(),
            tangent: EngineTangent::<E>::neg(EngineTangent::<E>::scale(
                input.primal.clone().sin(),
                input.tangent.clone(),
            )),
        }])
    }
}

impl<V: Traceable<ArrayType> + Cos + Sin + Neg<Output = V>, T: TangentSpace<ArrayType, V>> Cos for JvpTracer<V, T> {
    #[inline]
    fn cos(self) -> Self {
        Self { primal: self.primal.clone().cos(), tangent: T::neg(T::scale(self.primal.sin(), self.tangent)) }
    }
}

impl<'engine, V: Traceable<ArrayType> + Cos, E: Engine<Type = ArrayType, Value = V> + ?Sized> Cos for Tracer<'engine, E>
where
    E::TracingOperation: CosTracingOperation<ArrayType, V>,
{
    #[inline]
    fn cos(self) -> Self {
        self.unary(E::TracingOperation::cos_op())
    }
}
