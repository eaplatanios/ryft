//! Cosine primitive for [`crate::tracing_v2`].

use std::{
    any::Any,
    fmt::{Debug, Display},
};

use crate::tracing_v2::{
    FloatExt, TraceError, TraceValue, ZeroLike,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    ops::{BatchOp, DifferentiableOp, Eval, LinearOp, Op},
};
use crate::types::ArrayType;

use super::{expect_input_count, unary_abstract};

/// Elementwise cosine primitive.
#[derive(Clone, Default)]
pub struct CosOp;

impl Debug for CosOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Cos")
    }
}

impl Display for CosOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "cos")
    }
}

impl Op for CosOp {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "cos"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<V: TraceValue + FloatExt> Eval<V> for CosOp {
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().cos()])
    }
}

impl<V: TraceValue + FloatExt + ZeroLike> LinearOp<V> for CosOp {}

impl<V: TraceValue + FloatExt, T: TangentSpace<V>> DifferentiableOp<V, T> for CosOp {
    fn jvp(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().cos(),
            tangent: T::neg(T::scale(input.primal.clone().sin(), input.tangent.clone())),
        }])
    }
}

impl<V: TraceValue + FloatExt> BatchOp<V> for CosOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| lane.cos()).collect())])
    }
}
