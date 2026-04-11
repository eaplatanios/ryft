//! Matrix transpose primitive for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    FloatExt, OneLike, TraceError, TransformLeaf, ZeroLike,
    batch::Batch as BatchedValue,
    forward::JvpTracer,
    jit::JitTracer,
    linear::LinearTerm,
    ops::{BatchOp, DifferentiableOp, Eval, Op},
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

impl<V: MatrixValue> DifferentiableOp<V> for MatrixTransposeOp {
    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }

    fn apply_program_jvp_rule(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + OneLike + MatrixOps + super::reshape::ReshapeOps,
    {
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
