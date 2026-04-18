//! Sine primitive for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    TraceError, Traceable,
    batch::Batch,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    jit::JitTracer,
};
use crate::types::{ArrayType, Type};

use super::{DifferentiableOp, InterpretableOp, Op, VectorizableOp, cos::Cos, expect_input_count, unary_abstract};

/// Hidden staging trait for the sine primitive.
#[doc(hidden)]
pub trait SinTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the sine primitive.
    fn sin_op() -> Self;
}

/// Elementwise sine capability.
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
#[derive(Clone, Default)]
pub struct SinOp;

impl Debug for SinOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Sin")
    }
}

impl Display for SinOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "sin")
    }
}

impl Op for SinOp {
    fn name(&self) -> &'static str {
        "sin"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<V: Traceable<ArrayType> + Sin> InterpretableOp<ArrayType, V> for SinOp {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().sin()])
    }
}

impl<V: Traceable<ArrayType> + Sin + Cos, T: TangentSpace<ArrayType, V>, O: Clone, L: Clone>
    DifferentiableOp<ArrayType, V, T, O, L> for SinOp
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().sin(),
            tangent: T::scale(input.primal.clone().cos(), input.tangent.clone()),
        }])
    }
}

impl<V: Traceable<ArrayType> + Sin> VectorizableOp<ArrayType, V> for SinOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| lane.sin()).collect())])
    }
}

impl<V: Traceable<ArrayType> + Sin + Cos, T: TangentSpace<ArrayType, V>> Sin for JvpTracer<V, T> {
    #[inline]
    fn sin(self) -> Self {
        Self { primal: self.primal.clone().sin(), tangent: T::scale(self.primal.cos(), self.tangent) }
    }
}

impl<V: Traceable<ArrayType> + Sin, O: SinTracingOperation<ArrayType, V>, L: Clone> Sin
    for JitTracer<ArrayType, V, O, L>
where
    O: Op<ArrayType>,
{
    #[inline]
    fn sin(self) -> Self {
        self.unary(O::sin_op())
    }
}

impl<V: Traceable<ArrayType> + Sin> Sin for Batch<V> {
    #[inline]
    fn sin(self) -> Self {
        let outputs = SinOp.batch(&[self]).expect("sin batching rule should succeed");
        debug_assert_eq!(outputs.len(), 1, "sin should produce one batched output");
        outputs.into_iter().next().expect("sin batching should return one output")
    }
}
