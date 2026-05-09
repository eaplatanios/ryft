use std::fmt::Display;

use crate::macros::check_count;
use crate::operations::{InterpretableOperation, Operation};
use crate::tracing::transposition::LinearOperation;
use crate::tracing::{AtomId, Traceable, TracingError};
use crate::tracing_v2::differentiation::{Differentiable, JvpContext, JvpTracer};
use crate::tracing_v2::{DifferentiableEngine, DifferentiableOperation};
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
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![transpose_abstract(&input_types[0], "matrix_transpose")?])
    }
}

impl<V: MatrixValue> InterpretableOperation<ArrayType, V> for MatrixTransposeOperation {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}

impl<V: MatrixValue> LinearOperation<ArrayType, V, LinearArrayOperation<V, ArrayType>> for MatrixTransposeOperation {
    fn transpose(
        &self,
        context: &mut crate::tracing::transposition::TranspositionContext<
            ArrayType,
            V,
            LinearArrayOperation<V, ArrayType>,
        >,
        output_cotangents: &[Option<crate::tracing::AtomId>],
    ) -> Result<Vec<Option<crate::tracing::AtomId>>, TracingError> {
        check_count!("output", output_cotangents, 1, TracingError);
        match output_cotangents[0] {
            Some(atom) => {
                let cotangent_outputs = context.stage(LinearArrayOperation::Transpose, &[atom])?;
                check_count!("output", cotangent_outputs, 1, TracingError);
                Ok(vec![Some(cotangent_outputs[0])])
            }
            None => Ok(vec![None]),
        }
    }
}

impl<E> DifferentiableOperation<E> for MatrixTransposeOperation
where
    E: DifferentiableEngine<Type = ArrayType>,
    E::Value: MatrixValue + Differentiable<ArrayType>,
    <E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier:
        SupportsMatrixTranspose<ArrayType, E::Tangent>,
{
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError> {
        check_count!("input", inputs, 1, TracingError);
        let primal = inputs[0].primal.clone().transpose_matrix();
        let tangent_outputs = context.stage(
            <<E::LinearEngine as crate::tracing::engines::TracingEngine>::OperationCarrier as SupportsMatrixTranspose<
                ArrayType,
                E::Tangent,
            >>::matrix_transpose_operation(),
            &[inputs[0].tangent],
        )?;
        check_count!("output", tangent_outputs, 1, TracingError);
        Ok(vec![JvpTracer { primal, tangent: tangent_outputs[0] }])
    }
}
