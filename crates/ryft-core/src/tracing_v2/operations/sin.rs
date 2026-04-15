//! Sine primitive for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

use crate::tracing_v2::{
    FloatExt, TraceError, TraceValue, ZeroLike,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    linear::LinearTerm,
    ops::{DifferentiableOp, InterpretableOp, LinearOp, Op, VectorizableOp},
};
use crate::types::ArrayType;

use super::{expect_input_count, unary_abstract};

/// Elementwise sine primitive.
#[derive(Clone, Default)]
pub struct SinOp;

impl Debug for SinOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Sin")
    }
}

impl Display for SinOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "sin")
    }
}

impl Op for SinOp {
    fn name(&self) -> &'static str {
        "sin"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<V: TraceValue + FloatExt> InterpretableOp<V> for SinOp {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().sin()])
    }
}

impl<V: TraceValue + FloatExt + ZeroLike> LinearOp<V> for SinOp {
    fn transpose(
        &self,
        _inputs: &[V],
        _outputs: &[V],
        _output_cotangents: &[LinearTerm<V>],
    ) -> Result<Vec<Option<LinearTerm<V>>>, TraceError> {
        Err(TraceError::HigherOrderOpFailure {
            op: "transpose_linear_program",
            message: format!("transpose rule for staged op '{}' is not implemented", self.name()),
        })
    }
}

impl<V: TraceValue + FloatExt, T: TangentSpace<V>> DifferentiableOp<V, T> for SinOp {
    fn jvp(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().sin(),
            tangent: T::scale(input.primal.clone().cos(), input.tangent.clone()),
        }])
    }
}

impl<V: TraceValue + FloatExt> VectorizableOp<V> for SinOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| lane.sin()).collect())])
    }
}
