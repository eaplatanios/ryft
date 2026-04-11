//! Scaling primitive for [`crate::tracing_v2`].

use std::{
    any::Any,
    fmt::{Debug, Display},
    ops::Mul,
};

#[cfg(test)]
use indoc::indoc;

use crate::tracing_v2::{
    TraceError, TraceValue, TransformLeaf, ZeroLike,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    graph::AtomId,
    jit::JitTracer,
    linear::LinearTerm,
    ops::{BatchOp, DifferentiableOp, Eval, Op, PrimitiveOp},
    program::ProgramBuilder,
};
use crate::types::ArrayType;

use super::{expect_input_count, lift_jit_constant, unary_abstract};

/// Unary linear operation that multiplies its input by a captured factor.
#[derive(Clone)]
pub struct ScaleOp<V: TraceValue> {
    factor: V,
}

impl<V: TraceValue> ScaleOp<V> {
    /// Creates a new scale operation capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor }
    }

    /// Returns the captured scale factor.
    #[inline]
    pub fn factor(&self) -> &V {
        &self.factor
    }

    /// Validates abstract inputs without needing a concrete instance.
    pub fn abstract_eval_static(inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }
}

impl<V: TraceValue> Debug for ScaleOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Scale")
    }
}

impl<V: TraceValue> Display for ScaleOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scale")
    }
}

impl<V: TraceValue> Op for ScaleOp<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "scale"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Self::abstract_eval_static(inputs)
    }
}

impl<V: TraceValue + Mul<Output = V>> Eval<V> for ScaleOp<V> {
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor().clone() * inputs[0].clone()])
    }
}

impl<V: TraceValue + Mul<Output = V> + ZeroLike> DifferentiableOp<V> for ScaleOp<V> {
    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 1)?;
        let factor = lift_jit_constant(self.factor(), &inputs[0].primal);
        Ok(vec![JvpTracer {
            primal: factor.clone() * inputs[0].primal.clone(),
            tangent: inputs[0].tangent.clone().scale(factor),
        }])
    }

    fn apply_program_jvp_rule(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError> {
        self.jvp(inputs)
    }

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
        let abstract_value = builder.atom(output_cotangents[0]).expect("output cotangent atom should exist").abstract_value.clone();
        let example_value = builder.atom(output_cotangents[0]).expect("output cotangent atom should exist").example_value.clone();
        let contribution = builder.add_equation_prevalidated(
            PrimitiveOp::Scale { factor: self.factor().clone() },
            vec![output_cotangents[0]],
            vec![abstract_value],
            vec![example_value],
        )[0];
        Ok(vec![Some(contribution)])
    }

    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: self.factor().clone() * input.primal.clone(),
            tangent: T::scale(self.factor().clone(), input.tangent.clone()),
        }])
    }
}

impl<V: TraceValue + Mul<Output = V>> BatchOp<V> for ScaleOp<V> {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| self.factor().clone() * lane).collect())])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{parameters::Placeholder, tracing_v2::ProgramBuilder};

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_scale_transpose_scales_output_cotangents() {
        let mut forward_builder = ProgramBuilder::<f64>::new();
        let input = forward_builder.add_input(&1.0f64);
        let output = forward_builder.add_equation(PrimitiveOp::Scale { factor: 3.0f64 }, vec![input]).unwrap()[0];

        let mut transpose_builder = ProgramBuilder::<f64>::new();
        let output_cotangent = transpose_builder.add_input(&1.0f64);
        let contribution = ScaleOp::new(3.0f64)
            .transpose_program_op(&mut transpose_builder, &[input], &[output], &[output_cotangent])
            .unwrap()[0]
            .unwrap();

        let transpose_graph = transpose_builder.build::<f64, f64>(vec![contribution], Placeholder, Placeholder);
        approx_eq(transpose_graph.call(2.0f64).unwrap(), 6.0);
        assert_eq!(
            transpose_graph.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = scale %0
                in (%1)
            "}
            .trim_end(),
        );
    }
}
