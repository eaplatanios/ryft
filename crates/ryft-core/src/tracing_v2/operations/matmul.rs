use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::DifferentiableEngine;
use crate::tracing_v2::forward::{Differentiable, JvpContext, JvpTracer};
use crate::types::{ArrayType, Type, TypeError};

use super::left_matmul::SupportsLeftMatMul;
use super::matrix::{MatrixOps, MatrixValue, matmul_abstract};
use super::right_matmul::SupportsRightMatMul;
use super::{DifferentiableOperation, InterpretableOperation, Operation, SupportsAdd};

/// Hidden carrier capability for staging the matrix multiplication primitive.
#[doc(hidden)]
pub trait SupportsMatMul<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the matrix multiplication primitive.
    fn matmul_operation() -> Self;
}

/// Primitive representing matrix multiplication.
///
/// [`MatMulOperation`] is the matrix-valued analogue of the core scalar arithmetic primitives. Its JVP
/// rule delegates to the matrix tangent-space helpers so the same op can be reused for concrete
/// execution and traced execution.
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

impl Operation<ArrayType> for MatMulOperation {
    fn name(&self) -> &'static str {
        "matmul"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != 2 {
            return Err(TypeError { message: format!("matmul expected 2 input types but got {}", input_types.len()) });
        }
        Ok(vec![matmul_abstract(&input_types[0], &input_types[1], "matmul")?])
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for MatMulOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }
}

impl<E> DifferentiableOperation<E> for MatMulOperation
where
    E: DifferentiableEngine<Type = ArrayType> + ?Sized,
    E::Value: MatrixValue + Differentiable<ArrayType, Tangent = E::Value>,
    E::LinearOperation: SupportsAdd<ArrayType, E::Value>
        + SupportsLeftMatMul<ArrayType, E::Value>
        + SupportsRightMatMul<ArrayType, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 2, TracingError);
        let left = &inputs[0];
        let right = &inputs[1];
        let primal = left.primal.clone().matmul(right.primal.clone());
        let left_term = context
            .apply_operation(
                &[left.tangent],
                <E::LinearOperation as SupportsRightMatMul<ArrayType, E::Value>>::right_matmul_operation(
                    right.primal.clone(),
                ),
                1,
            )?
            .into_iter()
            .next()
            .expect("matmul jvp right matmul should produce one tangent");
        let right_term = context
            .apply_operation(
                &[right.tangent],
                <E::LinearOperation as SupportsLeftMatMul<ArrayType, E::Value>>::left_matmul_operation(
                    left.primal.clone(),
                ),
                1,
            )?
            .into_iter()
            .next()
            .expect("matmul jvp left matmul should produce one tangent");
        let tangent = context
            .apply_operation(
                &[left_term, right_term],
                <E::LinearOperation as SupportsAdd<ArrayType, E::Value>>::add_operation(),
                1,
            )?
            .into_iter()
            .next()
            .expect("matmul jvp add should produce one tangent");
        Ok(vec![JvpTracer { primal, tangent }])
    }
}
