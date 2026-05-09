use std::fmt::Display;

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
use crate::types::{ArrayType, Type, TypeError};

use super::left_matmul::SupportsLeftMatMul;
use super::matrix::{MatrixOps, MatrixValue, matmul_abstract};
use super::right_matmul::SupportsRightMatMul;
use crate::operations::arithmetic::SupportsAdd;

/// Trait that represents [`Operation`] carrier types that support/include [`MatMulOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`MatMulOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsMatMul<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the matrix multiplication [`Operation`].
    fn matmul_operation() -> Self;
}

/// Primitive representing matrix multiplication.
///
/// [`MatMulOperation`] is the matrix-valued analogue of the core scalar arithmetic primitives. Its JVP
/// rule delegates to the matrix tangent-space helpers so the same op can be reused for concrete
/// execution and traced execution.
#[derive(Clone, Debug, Default)]
pub struct MatMulOperation;

impl Display for MatMulOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Operation<ArrayType> for MatMulOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "matmul"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        Ok(vec![matmul_abstract(&input_types[0], &input_types[1], "matmul")?])
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for MatMulOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 2, TracingError);
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }
}

impl<E> DifferentiableOperation<E> for MatMulOperation
where
    E: DifferentiableEngine<Type = ArrayType>,
    E::Value: MatrixValue + Differentiable<ArrayType>,
    E::LinearOperationCarrier:
        SupportsLeftMatMul<ArrayType, E::Tangent, E::Value> + SupportsRightMatMul<ArrayType, E::Tangent, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 2, TracingError);
        let left = &inputs[0];
        let right = &inputs[1];
        let primal = left.primal.clone().matmul(right.primal.clone());
        let left_term_outputs =
            context.stage(E::LinearOperationCarrier::right_matmul_operation(right.primal.clone()), &[left.tangent])?;
        check_count!("output", left_term_outputs, 1, TracingError);
        let right_term_outputs =
            context.stage(E::LinearOperationCarrier::left_matmul_operation(left.primal.clone()), &[right.tangent])?;
        check_count!("output", right_term_outputs, 1, TracingError);
        let tangent_outputs = context
            .stage(E::LinearOperationCarrier::add_operation(), &[left_term_outputs[0], right_term_outputs[0]])?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal, tangent: tangent_outputs[0] }])
    }
}
