//! Cosine primitive for [`crate::tracing_v2`].

use std::{
    fmt::{Debug, Display},
    ops::Neg,
};

use crate::tracing_v2::{
    TraceError, Traceable,
    batch::Batch,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    jit::JitTracer,
};
use crate::types::{ArrayType, Type};

use super::{DifferentiableOp, InterpretableOp, Op, VectorizableOp, expect_input_count, sin::Sin, unary_abstract};

/// Hidden staging trait for the cosine primitive.
#[doc(hidden)]
pub trait CosTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the cosine primitive.
    fn cos_op() -> Self;
}

/// Elementwise cosine capability.
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
#[derive(Clone, Default)]
pub struct CosOp;

impl Debug for CosOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Cos")
    }
}

impl Display for CosOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "cos")
    }
}

impl Op for CosOp {
    fn name(&self) -> &'static str {
        "cos"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<V: Traceable<ArrayType> + Cos> InterpretableOp<ArrayType, V> for CosOp {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().cos()])
    }
}

impl<V: Traceable<ArrayType> + Cos + Sin + Neg<Output = V>, T: TangentSpace<ArrayType, V>, O: Clone, L: Clone>
    DifferentiableOp<ArrayType, V, T, O, L> for CosOp
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().cos(),
            tangent: T::neg(T::scale(input.primal.clone().sin(), input.tangent.clone())),
        }])
    }
}

impl<V: Traceable<ArrayType> + Cos> VectorizableOp<ArrayType, V> for CosOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
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

impl<V: Traceable<ArrayType> + Cos, O: CosTracingOperation<ArrayType, V>, L: Clone> Cos
    for JitTracer<ArrayType, V, O, L>
where
    O: Op<ArrayType>,
{
    #[inline]
    fn cos(self) -> Self {
        self.unary(O::cos_op())
    }
}

impl<V: Traceable<ArrayType> + Cos> Cos for Batch<V> {
    #[inline]
    fn cos(self) -> Self {
        let outputs = CosOp.batch(&[self]).expect("cos batching rule should succeed");
        debug_assert_eq!(outputs.len(), 1, "cos should produce one batched output");
        outputs.into_iter().next().expect("cos batching should return one output")
    }
}
