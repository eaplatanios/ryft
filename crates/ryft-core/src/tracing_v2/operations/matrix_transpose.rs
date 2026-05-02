use std::fmt::Display;

use crate::macros::check_input_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableOperation, LinearizableEngine};
use crate::types::{ArrayType, Type, TypeError};

use super::LinearArrayOperation;
use super::matrix::{MatrixOps, MatrixValue, transpose_abstract};

/// Trait that represents [`Operation`] carrier types that support/include [`MatrixTransposeOperation`]. Backend-owned
/// closed [`Operation`] carrier types (such as [`ArrayOperation`](super::ArrayOperation), for example) implement this
/// trait so that generic transform code can stage [`MatrixTransposeOperation`] without knowing which carrier is in use.
#[doc(hidden)]
pub trait SupportsMatrixTranspose<T: Type, V: Traceable<T>> {
    /// Constructs the carrier-specific representation of the matrix transposition [`Operation`].
    fn matrix_transpose_operation() -> Self;
}

/// Primitive representing matrix transposition.
///
/// [`MatrixTransposeOperation`] is stored directly in traced programs whenever a matrix leaf is
/// transposed symbolically.
#[derive(Clone, Debug, Default)]
pub struct MatrixTransposeOperation;

impl Display for MatrixTransposeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Operation<ArrayType> for MatrixTransposeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "matrix_transpose"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_input_count!(input_types, 1, TypeError);
        Ok(vec![transpose_abstract(&input_types[0], "matrix_transpose")?])
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for MatrixTransposeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V: MatrixValue> LinearOperation<ArrayType, V, LinearArrayOperation<V>> for MatrixTransposeOperation {
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<ArrayType, V, LinearArrayOperation<V>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => Ok(vec![Some(
                context
                    .stage(LinearArrayOperation::Transpose, &[atom])?
                    .into_iter()
                    .next()
                    .expect("matrix transpose should produce one cotangent contribution"),
            )]),
            None => Ok(vec![None]),
        }
    }
}

impl<E> DifferentiableOperation<E> for MatrixTransposeOperation
where
    E: LinearizableEngine<Type = ArrayType> + ?Sized,
    E::Value: MatrixValue + Differentiable<ArrayType, Tangent = E::Value>,
    E::LinearOperationCarrier: SupportsMatrixTranspose<ArrayType, E::Value>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 1, TracingError);
        let primal = inputs[0].primal.clone().transpose_matrix();
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperationCarrier as SupportsMatrixTranspose<ArrayType, E::Value>>::matrix_transpose_operation(
                ),
                1,
            )?
            .into_iter()
            .next()
            .expect("matrix transpose jvp should produce one tangent");
        Ok(vec![JvpTracer { primal, tangent }])
    }
}
