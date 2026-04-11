//! Matrix transpose primitive for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    FloatExt, TraceError, ZeroLike,
    batch::Batch as BatchedValue,
    forward::JvpTracer,
    graph::AtomId,
    ops::{BatchOp, DifferentiableOp, Eval, LinearOp, Op},
    program::ProgramBuilder,
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "matrix_transpose"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![transpose_abstract(&inputs[0], "matrix_transpose")?])
    }
}

impl<V: MatrixValue> Eval<V> for MatrixTransposeOp {
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V: MatrixValue + FloatExt + ZeroLike> LinearOp<V> for MatrixTransposeOp {
    fn transpose(
        &self,
        _builder: &mut ProgramBuilder<V>,
        _inputs: &[AtomId],
        _outputs: &[AtomId],
        _output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        Err(TraceError::HigherOrderOpFailure {
            op: "transpose_linear_program",
            message: format!("transpose rule for staged op '{}' is not implemented", self.name()),
        })
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

impl<V: MatrixValue> BatchOp<V> for MatrixTransposeOp {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(inputs[0].lanes().iter().cloned().map(MatrixOps::transpose_matrix).collect())])
    }
}
