//! Negation primitive for [`crate::tracing_v2`].

use std::{
    fmt::{Debug, Display},
    ops::Neg,
};

use crate::tracing_v2::{
    TraceError, TraceValue, ZeroLike,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    graph::AtomId,
    ops::{BatchOp, DifferentiableOp, InterpretableOp, LinearOp, Op, PrimitiveOp},
    program::ProgramBuilder,
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

impl<V: TraceValue + Neg<Output = V>> InterpretableOp<V> for NegOp {
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![-inputs[0].clone()])
    }
}

impl<V: TraceValue + Neg<Output = V> + ZeroLike> LinearOp<V> for NegOp {
    fn transpose(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let abstract_value = builder
            .atom(output_cotangents[0])
            .expect("output cotangent atom should exist")
            .abstract_value
            .clone();
        let example_value = builder
            .atom(output_cotangents[0])
            .expect("output cotangent atom should exist")
            .example_value
            .clone();
        let contribution = builder.add_equation_prevalidated(
            PrimitiveOp::Neg,
            vec![output_cotangents[0]],
            vec![abstract_value],
            vec![example_value],
        )[0];
        Ok(vec![Some(contribution)])
    }
}

impl<V: TraceValue + Neg<Output = V>, T: TangentSpace<V>> DifferentiableOp<V, T> for NegOp {
    fn jvp(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![JvpTracer { primal: -inputs[0].primal.clone(), tangent: T::neg(inputs[0].tangent.clone()) }])
    }
}

impl<V: TraceValue + Neg<Output = V>> BatchOp<V> for NegOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| -lane).collect())])
    }
}
