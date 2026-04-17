//! Left matrix-multiplication primitive for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    TraceError, Traceable, ZeroLike,
    batch::Batch as BatchedValue,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    jit::JitTracer,
    linear::LinearTerm,
    ops::{
        DifferentiableOp, InterpretableOp, LinearOperation, LinearPrimitiveOp, Op, OperationSet, SupportsMatMul,
        SupportsMatrixTranspose, VectorizableOp,
    },
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

/// Validates abstract inputs using the factor's abstract type without needing a concrete instance.
pub fn left_matmul_abstract_eval(factor_type: &ArrayType, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
    expect_input_count(inputs.len(), 1)?;
    Ok(vec![matmul_abstract(factor_type, &inputs[0], "left_matmul")?])
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

impl<V: MatrixValue> Op for LeftMatMulOp<V> {
    fn name(&self) -> &'static str {
        "left_matmul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        left_matmul_abstract_eval(&<V as Typed<ArrayType>>::tpe(&self.factor), inputs)
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

impl<V: MatrixValue> InterpretableOp<ArrayType, V> for LeftMatMulOp<V> {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor.clone().matmul(inputs[0].clone())])
    }
}

impl<V: MatrixValue> LinearOperation<ArrayType, V> for LeftMatMulOp<V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TraceError> {
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(
            LinearTerm::apply_staged_op(
                std::slice::from_ref(&output_cotangents[0]),
                LinearPrimitiveOp::LeftMatMul { factor: self.factor.clone().transpose_matrix() },
                1,
            )?
            .into_iter()
            .next()
            .expect("left matmul should produce one cotangent contribution"),
        )])
    }
}

impl<
    V: Traceable<ArrayType> + MatrixOps + ZeroLike,
    S: OperationSet<ArrayType, V> + SupportsMatMul<ArrayType, V> + SupportsMatrixTranspose<ArrayType, V>,
> InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>> for LeftMatMulOp<V>
where
    S::TracingOperation: Op<ArrayType>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V, S>>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        let factor = JvpTracer { primal: factor.clone(), tangent: LinearTerm::zero_like(&factor, &inputs[0].tangent) };
        Ok(vec![factor.matmul(inputs[0].clone())])
    }
}

impl<V: MatrixValue + ZeroLike, S: OperationSet<ArrayType, V>>
    DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V>, S> for LeftMatMulOp<V>
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, OperationSet = S>,
        inputs: &[JvpTracer<V, LinearTerm<ArrayType, V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<ArrayType, V>>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let factor = JvpTracer {
            primal: self.factor().clone(),
            tangent: TangentSpace::zero_like(&self.factor, &inputs[0].tangent),
        };
        Ok(vec![factor.matmul(inputs[0].clone())])
    }
}

impl<V: MatrixValue> VectorizableOp<ArrayType, V> for LeftMatMulOp<V> {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(
            inputs[0].lanes().iter().cloned().map(|lane| self.factor.clone().matmul(lane)).collect(),
        )])
    }
}
