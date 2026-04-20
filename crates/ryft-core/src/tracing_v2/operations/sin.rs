//! Sine primitive for [`crate::tracing_v2`].
//!
//! This module shows the pattern used for nonlinear elementwise primitives in `tracing_v2`: a
//! small value-level capability trait for concrete leaves, a semantic primitive type for staged
//! programs, and glue impls that teach [`JvpTracer`](crate::tracing_v2::JvpTracer),
//! [`Tracer`](crate::tracing_v2::Tracer), and [`Batch`](crate::tracing_v2::Batch) how to reuse the
//! same operation in forward-mode, symbolic tracing, and batching.

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    Traceable, TracingError,
    batch::Batch,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    jit::Tracer,
};
use crate::types::{ArrayType, Type};

use super::{
    DifferentiableOperation, InterpretableOperation, Operation, VectorizableOperation, cos::Cos, expect_input_count,
    unary_abstract,
};

/// Hidden staging trait for the sine primitive.
#[doc(hidden)]
pub trait SinTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the sine primitive.
    fn sin_op() -> Self;
}

/// Elementwise sine capability.
///
/// This trait is the value-level entry point that lets generic user code write `x.sin()` whether
/// `x` is a concrete scalar, a traced leaf, a JVP leaf, or a batched leaf.
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
/// store this op (or a backend-specific carrier that wraps it), while JVP and batching rules
/// delegate through its semantic implementation.
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

impl Operation for SinOperation {
    fn name(&self) -> &'static str {
        "sin"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<V: Traceable<ArrayType> + Sin> InterpretableOperation<ArrayType, V> for SinOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().sin()])
    }
}

impl<V: Traceable<ArrayType> + Sin + Cos, T: TangentSpace<ArrayType, V>, O: Clone, L: Clone>
    DifferentiableOperation<ArrayType, V, T, O, L> for SinOperation
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TracingError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().sin(),
            tangent: T::scale(input.primal.clone().cos(), input.tangent.clone()),
        }])
    }
}

impl<V: Traceable<ArrayType> + Sin> VectorizableOperation<ArrayType, V> for SinOperation {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TracingError> {
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

impl<
    'engine,
    V: Traceable<ArrayType> + Sin,
    O: SinTracingOperation<ArrayType, V>,
    L: Clone,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
> Sin for Tracer<'engine, E>
where
    O: InterpretableOperation<ArrayType, V> + Operation<ArrayType>,
{
    #[inline]
    fn sin(self) -> Self {
        self.unary(O::sin_op())
    }
}

impl<V: Traceable<ArrayType> + Sin> Sin for Batch<V> {
    #[inline]
    fn sin(self) -> Self {
        let outputs = SinOperation.batch(&[self]).expect("sin batching rule should succeed");
        debug_assert_eq!(outputs.len(), 1, "sin should produce one batched output");
        outputs.into_iter().next().expect("sin batching should return one output")
    }
}
