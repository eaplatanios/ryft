//! Cosine primitive for [`crate::tracing_v2`].
//!
//! This module is the cosine-side companion to [`super::sin`]. It exposes the value-level cosine
//! capability used in generic user code and the staged primitive used by tracing, batching, and
//! derivative construction.

use std::{
    fmt::{Debug, Display},
    ops::Neg,
};

use crate::tracing_v2::{
    Traceable, TracingError,
    batch::Batch,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    jit::Tracer,
};
use crate::types::{ArrayType, Type};

use super::{
    DifferentiableOperation, InterpretableOperation, Operation, VectorizableOperation, expect_input_count, sin::Sin,
    unary_abstract,
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

impl Operation for CosOperation {
    fn name(&self) -> &'static str {
        "cos"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<V: Traceable<ArrayType> + Cos> InterpretableOperation<ArrayType, V> for CosOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().cos()])
    }
}

impl<V: Traceable<ArrayType> + Cos + Sin + Neg<Output = V>, T: TangentSpace<ArrayType, V>, O: Clone, L: Clone>
    DifferentiableOperation<ArrayType, V, T, O, L> for CosOperation
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TracingError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().cos(),
            tangent: T::neg(T::scale(input.primal.clone().sin(), input.tangent.clone())),
        }])
    }
}

impl<V: Traceable<ArrayType> + Cos> VectorizableOperation<ArrayType, V> for CosOperation {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TracingError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| lane.cos()).collect())])
    }
}

impl<V: Traceable<ArrayType> + Cos + Sin + Neg<Output = V>, T: TangentSpace<ArrayType, V>> Cos for JvpTracer<V, T> {
    #[inline]
    fn cos(self) -> Self {
        Self { primal: self.primal.clone().cos(), tangent: T::neg(T::scale(self.primal.sin(), self.tangent)) }
    }
}

impl<
    'engine,
    V: Traceable<ArrayType> + Cos,
    O: CosTracingOperation<ArrayType, V>,
    L: Clone,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> Cos for Tracer<'engine, E>
where
    O: InterpretableOperation<ArrayType, V> + Operation<ArrayType>,
{
    #[inline]
    fn cos(self) -> Self {
        self.unary(O::cos_op())
    }
}

impl<V: Traceable<ArrayType> + Cos> Cos for Batch<V> {
    #[inline]
    fn cos(self) -> Self {
        let outputs = CosOperation.batch(&[self]).expect("cos batching rule should succeed");
        debug_assert_eq!(outputs.len(), 1, "cos should produce one batched output");
        outputs.into_iter().next().expect("cos batching should return one output")
    }
}
