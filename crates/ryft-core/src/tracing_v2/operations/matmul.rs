//! Matrix multiplication primitive for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    TraceError,
    batch::Batch as BatchedValue,
    engine::Engine,
    forward::JvpTracer,
    ops::{DifferentiableOp, InterpretableOp, Op, VectorizableOp},
};
use crate::types::ArrayType;

use super::{
    expect_batch_sizes_match, expect_input_count,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
};

/// Primitive representing matrix multiplication.
#[derive(Clone, Default)]
pub struct MatMulOp;

impl Debug for MatMulOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MatMul")
    }
}

impl Display for MatMulOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "matmul")
    }
}

impl Op for MatMulOp {
    fn name(&self) -> &'static str {
        "matmul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![matmul_abstract(&inputs[0], &inputs[1], "matmul")?])
    }
}

impl<V: MatrixValue> InterpretableOp<ArrayType, V> for MatMulOp {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }
}

impl<V: MatrixValue, T: super::matrix::MatrixTangentSpace<V>, O: Clone, L: Clone>
    DifferentiableOp<ArrayType, V, T, O, L> for MatMulOp
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }
}

impl<V: MatrixValue> VectorizableOp<ArrayType, V> for MatMulOp {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        expect_batch_sizes_match(&inputs[0], &inputs[1])?;
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
