//! Negation primitive for [`crate::tracing_v2`].

use std::{
    any::Any,
    fmt::{Debug, Display},
    ops::Neg,
};

use crate::tracing_v2::{
    TraceError, TraceValue, TransformLeaf, ZeroLike,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    graph::AtomId,
    jit::JitTracer,
    linear::LinearTerm,
    ops::{BatchOp, DifferentiableOp, Eval, LinearOp, Op, PrimitiveOp},
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
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "neg"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<V: TraceValue + Neg<Output = V>> Eval<V> for NegOp {
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![-inputs[0].clone()])
    }
}

impl<V: TraceValue + Neg<Output = V> + ZeroLike> LinearOp<V> for NegOp {
    fn transpose_program_op(
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

    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![JvpTracer { primal: -inputs[0].primal.clone(), tangent: inputs[0].tangent.clone().neg() }])
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
