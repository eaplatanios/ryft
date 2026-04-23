use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing::TracingError;
use crate::types::{ArrayType, Type, TypeError};
use crate::{
    tracing::Traceable,
    tracing_v2::{engine::Engine, forward::JvpTracer, linear::LinearTerm},
};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearOperation, LinearPrimitiveOperation, Operation,
    matrix::{MatrixOps, MatrixValue, transpose_abstract},
};

/// Hidden staging trait for the matrix transposition primitive.
#[doc(hidden)]
pub trait MatrixTransposeTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the matrix transposition primitive.
    fn matrix_transpose_op() -> Self;
}

/// Hidden staging trait for the matrix transposition primitive in linear programs.
#[doc(hidden)]
pub trait LinearMatrixTransposeOperation<T: Type + Display, V: Traceable<T>>: Clone {
    /// Constructs the carrier-specific representation of the linear matrix transposition primitive.
    fn linear_matrix_transpose_op() -> Self;
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

impl Operation for MatrixTransposeOperation {
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

impl<V: MatrixValue> LinearOperation<ArrayType, V> for MatrixTransposeOperation {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TracingError> {
        check_input_count!(output_cotangents, 1);
        Ok(vec![Some(
            LinearTerm::apply_staged_op(
                output_cotangents[0].builder.clone(),
                std::slice::from_ref(&output_cotangents[0]),
                LinearPrimitiveOperation::MatrixTranspose,
                1,
            )?
            .into_iter()
            .next()
            .expect("matrix transpose should produce one cotangent contribution"),
        )])
    }
}

impl<V: MatrixValue, T: super::matrix::MatrixTangentSpace<V>, O: Clone, L: Clone>
    DifferentiableOperation<ArrayType, V, T, O, L> for MatrixTransposeOperation
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }
}
