//! Right matrix-multiplication primitive for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    FloatExt, OneLike, TraceError, Traceable, ZeroLike,
    batch::Batch as BatchedValue,
    forward::{JvpTracer, TangentSpace},
    jit::JitTracer,
    linear::LinearTerm,
    ops::{DifferentiableOp, InterpretableOp, LinearOp, LinearPrimitiveOp, Op, VectorizableOp},
};
use crate::types::{ArrayType, Typed};

use super::{
    expect_input_count, lift_jit_constant,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
};

/// Linear map `tangent -> tangent @ factor`.
#[derive(Clone)]
pub struct RightMatMulOp<V: MatrixValue> {
    factor: V,
}

impl<V: MatrixValue> RightMatMulOp<V> {
    /// Creates one right multiplication op capturing the provided factor.
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

/// Validates abstract inputs using the factor's abstract type without needing a concrete instance.
pub fn right_matmul_abstract_eval(factor_type: &ArrayType, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
    expect_input_count(inputs.len(), 1)?;
    Ok(vec![matmul_abstract(&inputs[0], factor_type, "right_matmul")?])
}

impl<V: MatrixValue> Debug for RightMatMulOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "RightMatMul")
    }
}

impl<V: MatrixValue> Display for RightMatMulOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "right_matmul")
    }
}

impl<V: MatrixValue> Op for RightMatMulOp<V> {
    fn name(&self) -> &'static str {
        "right_matmul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        right_matmul_abstract_eval(&<V as Typed<ArrayType>>::tpe(&self.factor), inputs)
    }

    fn try_simplify(
        &self,
        inputs: &[usize],
        _is_zero_constant: &dyn Fn(usize) -> bool,
        _is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        if crate::tracing_v2::graph::is_identity_one(&self.factor) { Some(inputs.to_vec()) } else { None }
    }
}

impl<V: MatrixValue> InterpretableOp<V> for RightMatMulOp<V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().matmul(self.factor.clone())])
    }
}

impl<V: MatrixValue + FloatExt + ZeroLike + OneLike + crate::tracing_v2::operations::reshape::ReshapeOps> LinearOp<V>
    for RightMatMulOp<V>
{
    fn transpose(&self, output_cotangents: &[LinearTerm<V>]) -> Result<Vec<Option<LinearTerm<V>>>, TraceError> {
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(
            LinearTerm::apply_staged_op(
                std::slice::from_ref(&output_cotangents[0]),
                LinearPrimitiveOp::RightMatMul { factor: self.factor.clone().transpose_matrix() },
                1,
            )?
            .into_iter()
            .next()
            .expect("right matmul should produce one cotangent contribution"),
        )])
    }
}

impl<V: Traceable<ArrayType> + FloatExt + ZeroLike + OneLike + MatrixOps>
    InterpretableOp<crate::tracing_v2::linear::Linearized<JitTracer<V>>> for RightMatMulOp<V>
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<JitTracer<V>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<JitTracer<V>>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        let factor = JvpTracer { primal: factor.clone(), tangent: LinearTerm::zero_like(&factor, &inputs[0].tangent) };
        Ok(vec![inputs[0].clone().matmul(factor)])
    }
}

impl<V: MatrixValue + FloatExt + ZeroLike + OneLike + crate::tracing_v2::operations::reshape::ReshapeOps>
    DifferentiableOp<V, LinearTerm<V>> for RightMatMulOp<V>
{
    fn jvp(&self, inputs: &[JvpTracer<V, LinearTerm<V>>]) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let factor = JvpTracer {
            primal: self.factor().clone(),
            tangent: TangentSpace::zero_like(&self.factor, &inputs[0].tangent),
        };
        Ok(vec![inputs[0].clone().matmul(factor)])
    }
}

impl<V: MatrixValue> VectorizableOp<V> for RightMatMulOp<V> {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(
            inputs[0].lanes().iter().cloned().map(|lane| lane.matmul(self.factor.clone())).collect(),
        )])
    }
}
