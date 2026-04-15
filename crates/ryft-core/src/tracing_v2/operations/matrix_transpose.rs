//! Matrix transpose primitive for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    FloatExt, TraceError, ZeroLike,
    batch::Batch as BatchedValue,
    forward::JvpTracer,
    linear::LinearTerm,
    ops::{DifferentiableOp, InterpretableOp, LinearOp, Op, PrimitiveOp, VectorizableOp},
};
use crate::types::ArrayType;

use super::{
    expect_input_count,
    matrix::{MatrixOps, MatrixValue, transpose_abstract},
};

/// Primitive representing matrix transposition.
#[derive(Clone, Default)]
pub struct MatrixTransposeOp;

impl Debug for MatrixTransposeOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MatrixTranspose")
    }
}

impl Display for MatrixTransposeOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "matrix_transpose")
    }
}

impl Op for MatrixTransposeOp {
    fn name(&self) -> &'static str {
        "matrix_transpose"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![transpose_abstract(&inputs[0], "matrix_transpose")?])
    }
}

impl<V: MatrixValue> InterpretableOp<V> for MatrixTransposeOp {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V: MatrixValue + FloatExt + ZeroLike> LinearOp<V> for MatrixTransposeOp {
    fn transpose(
        &self,
        inputs: &[V],
        outputs: &[V],
        output_cotangents: &[LinearTerm<V>],
    ) -> Result<Vec<Option<LinearTerm<V>>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(
            LinearTerm::apply_staged_op(
                std::slice::from_ref(&output_cotangents[0]),
                PrimitiveOp::LinearMatrixTranspose,
                1,
            )?
            .into_iter()
            .next()
            .expect("matrix transpose should produce one cotangent contribution"),
        )])
    }
}

impl<V: MatrixValue + FloatExt + ZeroLike, T: super::matrix::MatrixTangentSpace<V>> DifferentiableOp<V, T>
    for MatrixTransposeOp
{
    fn jvp(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V: MatrixValue> VectorizableOp<V> for MatrixTransposeOp {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(inputs[0].lanes().iter().cloned().map(MatrixOps::transpose_matrix).collect())])
    }
}
