use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing::TracingError;
use crate::types::{ArrayType, Type, TypeError};
use crate::{
    tracing::Traceable,
    tracing_v2::{
        engines::DifferentiableEngine,
        forward::{Differentiable, EngineTangent, JvpTracer},
    },
};

use super::{
    DifferentiableOperation, InterpretableOperation, Operation, SupportsAdd, SupportsNeg, SupportsScale,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
};

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
        check_input_count!(inputs, 2);
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }
}

impl<E> DifferentiableOperation<E> for MatMulOperation
where
    E: DifferentiableEngine<Type = ArrayType> + ?Sized,
    E::Value: MatrixValue + Differentiable<ArrayType>,
    E::LinearOperation:
        SupportsAdd<ArrayType, E::Value> + SupportsNeg<ArrayType, E::Value> + SupportsScale<ArrayType, E::Value>,
    EngineTangent<E>: super::matrix::MatrixTangentSpace<E::Value>,
{
    fn jvp(
        &self,
        _engine: &E,
        inputs: &[JvpTracer<E::Value, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<E::Value, EngineTangent<E>>>, TracingError> {
        check_input_count!(inputs, 2);
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }
}
