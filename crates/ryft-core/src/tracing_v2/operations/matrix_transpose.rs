//! Matrix transpose primitive for [`crate::tracing_v2`].
//!
//! Matrix transpose is one of the structural matrix primitives that many higher-order rules depend
//! on. This module provides its staged semantic op so matrix-aware transforms can reuse a single
//! abstract-eval, replay, batching, and transpose implementation.

use std::fmt::{Debug, Display};

use crate::macros::check_input_count;
use crate::tracing_v2::{
    Traceable, TracingError, batch::Batch as BatchedValue, engine::Engine, forward::JvpTracer, linear::LinearTerm,
};
use crate::types::{ArrayType, Type};

use super::{
    DifferentiableOperation, InterpretableOperation, LinearOperation, LinearPrimitiveOperation, Operation,
    VectorizableOperation,
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

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![transpose_abstract(&inputs[0], "matrix_transpose")?])
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

impl<V: MatrixValue> VectorizableOperation<ArrayType, V> for MatrixTransposeOperation {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TracingError> {
        check_input_count!(inputs, 1);
        Ok(vec![BatchedValue::new(inputs[0].lanes().iter().cloned().map(MatrixOps::transpose_matrix).collect())])
    }
}
