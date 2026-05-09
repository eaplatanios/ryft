use std::fmt::Display;

use half::{bf16, f16};

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::engines::{Tracer, TracingEngine};
use crate::tracing::{Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
use crate::types::{ArrayType, Type, TypeError};

use super::left_matmul::{LeftMatMul, SupportsLeftMatMul};
use super::matrix::{MatrixValue, matmul_abstract};
use super::right_matmul::{RightMatMul, SupportsRightMatMul};

/// Trait that represents [`Operation`] carrier types that support/include [`MatMulOperation`]. Backend-owned closed
/// [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this trait
/// so that generic transform code can stage [`MatMulOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsMatMul<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the matrix multiplication [`Operation`].
    fn matmul_operation() -> Self;
}

/// Value-level matrix multiplication capability.
///
/// [`MatMul`] fills the same role for [`MatMulOperation`] that [`std::ops::Add`] fills for elementwise addition, but
/// keeps the matrix-specific operation out of the standard operator namespace.
pub trait MatMul<Rhs = Self>: Sized {
    /// Computes `self @ rhs`.
    fn matmul(self, rhs: Rhs) -> Self;
}

macro_rules! impl_matmul_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl MatMul for $ty {
                #[inline]
                fn matmul(self, rhs: Self) -> Self {
                    self * rhs
                }
            }
        )*
    };
}

impl_matmul_for_scalar!(bf16, f16, f32, f64);

impl<'engine, V: Traceable<ArrayType>, E> MatMul for Tracer<'engine, E>
where
    E: TracingEngine<Type = ArrayType, Value = V>,
    E::OperationCarrier: SupportsMatMul<ArrayType, V>,
{
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        self.binary(rhs, E::OperationCarrier::matmul_operation())
    }
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
    fn jvp<'jvp>(
        &self,
        _context: &mut JvpContext<'jvp, E>,
        inputs: &[JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>],
    ) -> Result<Vec<JvpTracer<E::Value, Tracer<'jvp, E::LinearEngine>>>, TracingError> {
        check_count!("input", inputs, 2, TracingError);
        let left = &inputs[0];
        let right = &inputs[1];
        let primal = left.primal.clone().matmul(right.primal.clone());
        let tangent = left.tangent.clone().right_matmul(right.primal.clone())
            + right.tangent.clone().left_matmul(left.primal.clone());
        Ok(vec![JvpTracer { primal, tangent }])
    }
}
