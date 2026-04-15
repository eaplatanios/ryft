//! Linear matrix-transpose primitive for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    TraceError,
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

/// Linear transpose primitive used inside matrix-valued pushforwards and pullbacks.
#[derive(Clone, Default)]
pub struct LinearMatrixTransposeOp;

impl Debug for LinearMatrixTransposeOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearMatrixTranspose")
    }
}

impl Display for LinearMatrixTransposeOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "linear_matrix_transpose")
    }
}

impl Op for LinearMatrixTransposeOp {
    fn name(&self) -> &'static str {
        "linear_matrix_transpose"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![transpose_abstract(&inputs[0], "linear_matrix_transpose")?])
    }
}

impl<V: MatrixValue> InterpretableOp<V> for LinearMatrixTransposeOp {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V: MatrixValue> LinearOp<V> for LinearMatrixTransposeOp {
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
            .expect("linear matrix transpose should produce one cotangent contribution"),
        )])
    }
}

impl<V: MatrixValue, T: super::matrix::MatrixTangentSpace<V>> DifferentiableOp<V, T> for LinearMatrixTransposeOp {
    fn jvp(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V: MatrixValue> VectorizableOp<V> for LinearMatrixTransposeOp {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(inputs[0].lanes().iter().cloned().map(MatrixOps::transpose_matrix).collect())])
    }
}
