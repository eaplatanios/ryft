//! Left matrix-multiplication primitive for [`crate::tracing_v2`].

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::tracing_v2::{
    FloatExt, TraceError, TransformLeaf, ZeroLike,
    batch::Batch as BatchedValue,
    forward::{JvpTracer, TangentSpace},
    graph::AtomId,
    jit::JitTracer,
    linear::LinearTerm,
    ops::{BatchOp, Op},
    program::ProgramBuilder,
};
use crate::types::{ArrayType, Typed};

use super::{
    expect_input_count, lift_jit_constant,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
};

/// Linear map `tangent -> factor @ tangent`.
#[derive(Clone)]
pub struct LeftMatMulOp<V: MatrixValue> {
    factor: V,
}

impl<V: MatrixValue> LeftMatMulOp<V> {
    /// Creates one left multiplication op capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor }
    }

    /// Returns the captured matrix factor.
    #[inline]
    pub fn factor(&self) -> &V {
        &self.factor
    }
}

impl<V: MatrixValue> Debug for LeftMatMulOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LeftMatMul")
    }
}

impl<V: MatrixValue> Display for LeftMatMulOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "left_matmul")
    }
}

impl<V: MatrixValue> Op<V> for LeftMatMulOp<V> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "left_matmul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![matmul_abstract(&<V as Typed<ArrayType>>::tpe(&self.factor), &inputs[0], "left_matmul")?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor.clone().matmul(inputs[0].clone())])
    }

    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 1)?;
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        let factor = JvpTracer { primal: factor.clone(), tangent: LinearTerm::zero_like(&factor, &inputs[0].tangent) };
        Ok(vec![factor.matmul(inputs[0].clone())])
    }

    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution = builder.add_equation(
            Arc::new(LeftMatMulOp::new(self.factor.clone().transpose_matrix())),
            vec![output_cotangents[0]],
        )?[0];
        Ok(vec![Some(contribution)])
    }
}

impl<V: MatrixValue> BatchOp<V> for LeftMatMulOp<V> {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(
            inputs[0].lanes().iter().cloned().map(|lane| self.factor.clone().matmul(lane)).collect(),
        )])
    }
}
