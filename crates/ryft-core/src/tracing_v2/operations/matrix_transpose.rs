use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::DifferentiableEngine;
use crate::tracing_v2::forward::{Differentiable, JvpContext, JvpTracer};
use crate::types::{ArrayType, Type, TypeError};

use super::matrix::{MatrixOps, MatrixValue, transpose_abstract};
use super::{DifferentiableOperation, InterpretableOperation, LinearArrayOperation, LinearOperation, Operation};

/// Hidden carrier capability for staging the matrix transposition primitive.
#[doc(hidden)]
pub trait SupportsMatrixTranspose<T: Type, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the matrix transposition primitive.
    fn matrix_transpose_operation() -> Self;
}

/// Primitive representing matrix transposition.
///
/// [`MatrixTransposeOperation`] is stored directly in traced programs whenever a matrix leaf is
/// transposed symbolically.
#[derive(Clone, Default)]
pub struct MatrixTransposeOperation;

impl Debug for MatrixTransposeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MatrixTranspose")
    }
}

impl Display for MatrixTransposeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "matrix_transpose")
    }
}

impl Operation<ArrayType> for MatrixTransposeOperation {
    fn name(&self) -> &'static str {
        "matrix_transpose"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != 1 {
            return Err(TypeError {
                message: format!("matrix_transpose expected 1 input type but got {}", input_types.len()),
            });
        }
        Ok(vec![transpose_abstract(&input_types[0], "matrix_transpose")?])
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for MatrixTransposeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V: MatrixValue> LinearOperation<ArrayType, V, LinearArrayOperation<V>> for MatrixTransposeOperation {
    fn transpose(
        &self,
        context: &mut crate::tracing_v2::operations::TranspositionContext<'_, ArrayType, V, LinearArrayOperation<V>>,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_input_count!(output_cotangents, 1);
        match output_cotangents[0] {
            Some(atom) => Ok(vec![Some(
                context
                    .apply_operation(&[atom], LinearArrayOperation::Transpose, 1)?
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
    E: DifferentiableEngine<Type = ArrayType> + ?Sized,
    E::Value: MatrixValue + Differentiable<ArrayType, Tangent = E::Value>,
    E::LinearOperation: SupportsMatrixTranspose<ArrayType, E::Value>,
{
    fn jvp(
        &self,
        _engine: &E,
        context: &mut JvpContext<'_, E::Value, E::LinearOperation>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_input_count!(inputs, 1);
        let primal = inputs[0].primal.clone().transpose_matrix();
        let tangent = context
            .apply_operation(
                &[inputs[0].tangent],
                <E::LinearOperation as SupportsMatrixTranspose<ArrayType, E::Value>>::matrix_transpose_operation(),
                1,
            )?
            .into_iter()
            .next()
            .expect("matrix transpose jvp should produce one tangent");
        Ok(vec![JvpTracer { primal, tangent }])
    }
}
