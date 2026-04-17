//! Negation primitive for [`crate::tracing_v2`].

use std::{
    fmt::{Debug, Display},
    ops::Neg,
};

use crate::tracing_v2::{
    TraceError, Traceable, ZeroLike,
    batch::Batch,
    engine::Engine,
    forward::{JvpTracer, TangentSpace},
    linear::LinearTerm,
    ops::{DifferentiableOp, InterpretableOp, LinearOperation, Op, OperationSet, VectorizableOp},
};
use crate::types::ArrayType;

use super::{expect_input_count, unary_abstract};

/// Elementwise negation primitive.
#[derive(Clone, Default)]
pub struct NegOp;

impl Debug for NegOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Neg")
    }
}

impl Display for NegOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "neg")
    }
}

impl Op for NegOp {
    fn name(&self) -> &'static str {
        "neg"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }

    fn try_simplify(
        &self,
        inputs: &[usize],
        is_zero_constant: &dyn Fn(usize) -> bool,
        _is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        if inputs.len() == 1 && is_zero_constant(inputs[0]) { Some(vec![inputs[0]]) } else { None }
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V>> InterpretableOp<ArrayType, V> for NegOp {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![-inputs[0].clone()])
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V> + ZeroLike> LinearOperation<ArrayType, V> for NegOp {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TraceError> {
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(output_cotangents[0].clone().neg())])
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V>, T: TangentSpace<ArrayType, V>, S: OperationSet<ArrayType, V>>
    DifferentiableOp<ArrayType, V, T, S> for NegOp
{
    fn jvp(
        &self,
        _engine: &dyn Engine<Type = ArrayType, Value = V, OperationSet = S>,
        inputs: &[JvpTracer<V, T>],
    ) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![JvpTracer { primal: -inputs[0].primal.clone(), tangent: T::neg(inputs[0].tangent.clone()) }])
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V>> VectorizableOp<ArrayType, V> for NegOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| -lane).collect())])
    }
}
