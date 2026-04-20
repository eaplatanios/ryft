//! Matrix multiplication primitive for [`crate::tracing_v2`].
//!
//! This module defines the staged matrix-multiplication primitive itself. The supporting matrix
//! capability traits live in [`super::matrix`]; this file is the concrete semantic op that traced
//! programs store once those capabilities are available.

use std::fmt::{Debug, Display};

use crate::macros::{check_batch_sizes, check_input_count};
use crate::tracing_v2::{Traceable, TracingError, batch::Batch as BatchedValue, engine::Engine, forward::JvpTracer};
use crate::types::{ArrayType, Type};

use super::{
    DifferentiableOperation, InterpretableOperation, Operation, VectorizableOperation,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
};

/// Hidden staging trait for the matrix multiplication primitive.
#[doc(hidden)]
pub trait MatMulTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the matrix multiplication primitive.
    fn matmul_op() -> Self;
}

/// Primitive representing matrix multiplication.
///
/// [`MatMulOperation`] is the matrix-valued analogue of the core scalar arithmetic primitives. Its JVP
/// rule delegates to the matrix tangent-space helpers so the same op can be reused for concrete
/// execution, traced execution, and batching.
#[derive(Clone, Default)]
pub struct MatMulOperation;

impl Debug for MatMulOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MatMul")
    }
}

impl Display for MatMulOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "matmul")
    }
}

impl Operation for MatMulOperation {
    fn name(&self) -> &'static str {
        "matmul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        check_input_count!(inputs, 2);
        Ok(vec![matmul_abstract(&inputs[0], &inputs[1], "matmul")?])
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for MatMulOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 2);
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }
}

impl<V: MatrixValue, T: super::matrix::MatrixTangentSpace<V>, O: Clone, L: Clone>
    DifferentiableOperation<ArrayType, V, T, O, L> for MatMulOperation
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TracingError> {
        check_input_count!(inputs, 2);
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }
}

impl<V: MatrixValue> VectorizableOperation<ArrayType, V> for MatMulOperation {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TracingError> {
        check_input_count!(inputs, 2);
        check_batch_sizes!(inputs);
        Ok(vec![BatchedValue::new(
            inputs[0]
                .lanes()
                .iter()
                .cloned()
                .zip(inputs[1].lanes().iter().cloned())
                .map(|(left, right)| left.matmul(right))
                .collect(),
        )])
    }
}
