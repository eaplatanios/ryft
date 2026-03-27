//! Scaling primitive for [`crate::tracing_v2`].

use std::{
    any::Any,
    fmt::{Debug, Display},
    ops::Mul,
    sync::Arc,
};

#[cfg(test)]
use indoc::indoc;
#[cfg(feature = "xla")]
use ryft_mlir::dialects::stable_hlo;
#[cfg(feature = "xla")]
use ryft_mlir::{Block, Operation, Value, ValueRef};

use crate::tracing_v2::{
    FloatExt, MatrixOps, TraceError, TraceValue, TransformLeaf, ZeroLike,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    graph::AtomId,
    jit::JitTracer,
    linear::LinearTerm,
    ops::{BatchOp, JvpOp, Op},
    program::ProgramBuilder,
};
use crate::types::ArrayType;
#[cfg(feature = "xla")]
use crate::xla::lowering::{LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode};

use super::{expect_input_count, lift_jit_constant, unary_abstract};

/// Unary linear operation that multiplies its input by a captured factor.
#[derive(Clone)]
pub(crate) struct ScaleOp<V>
where
    V: TraceValue,
{
    factor: V,
}

impl<V> ScaleOp<V>
where
    V: TraceValue,
{
    /// Creates a new scale operation capturing the provided factor.
    #[inline]
    pub fn new(factor: V) -> Self {
        Self { factor }
    }

    /// Returns the captured scale factor.
    #[inline]
    pub(crate) fn factor(&self) -> &V {
        &self.factor
    }
}

impl<V> Debug for ScaleOp<V>
where
    V: TraceValue,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Scale")
    }
}

impl<V> Display for ScaleOp<V>
where
    V: TraceValue,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scale")
    }
}

impl<V> Op<V> for ScaleOp<V>
where
    V: TraceValue + Mul<Output = V>,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "scale"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor().clone() * inputs[0].clone()])
    }

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
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        self.jvp(inputs)
    }

    fn transpose_program_op(
        &self,
        builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        expect_input_count(inputs.len(), 1)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        let contribution =
            builder.add_equation(Arc::new(ScaleOp::new(self.factor().clone())), vec![output_cotangents[0]])?[0];
        Ok(vec![Some(contribution)])
    }

    #[cfg(feature = "xla")]
    fn lower_plain_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        let factor = match mode {
            PlainMlirLoweringMode::Unpacked => lowerer.lower_literal_value(self.factor())?,
            PlainMlirLoweringMode::Packed { .. } => {
                lowerer.lower_packed_literal_value(self.factor(), &output_types[0])?
            }
        };
        let operation = lowerer.block.append_operation(stable_hlo::multiply(factor, input_values[0], lowerer.location));
        Ok(vec![operation.result(0).expect("stablehlo.multiply should return one result").as_ref()])
    }
}

impl<V> JvpOp<V> for ScaleOp<V>
where
    V: TraceValue + Mul<Output = V>,
{
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

impl<V> BatchOp<V> for ScaleOp<V>
where
    V: TraceValue + Mul<Output = V>,
{
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
        let output = forward_builder.add_equation(Arc::new(ScaleOp::new(3.0f64)), vec![input]).unwrap()[0];

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
