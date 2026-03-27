//! Multiplication primitive for [`crate::tracing_v2`].

use std::{
    any::Any,
    fmt::{Debug, Display},
    ops::Mul,
};

#[cfg(feature = "xla")]
use ryft_mlir::dialects::stable_hlo;
#[cfg(feature = "xla")]
use ryft_mlir::{Block, Operation, Value, ValueRef};

use crate::tracing_v2::{
    FloatExt, MatrixOps, TraceError, TraceValue, TransformLeaf, ZeroLike,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    jit::JitTracer,
    linear::LinearTerm,
    ops::{BatchOp, JvpOp, Op},
};
use crate::types::ArrayType;
#[cfg(feature = "xla")]
use crate::xla::lowering::{
    LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode, ShardMapMlirLowerer,
};

use super::{binary_same_abstract, expect_batch_sizes_match, expect_input_count};

/// Elementwise multiplication primitive.
#[derive(Clone, Default)]
pub(crate) struct MulOp;

impl Debug for MulOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Mul")
    }
}

impl Display for MulOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "mul")
    }
}

impl<V> Op<V> for MulOp
where
    V: TraceValue + Mul<Output = V>,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "mul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![binary_same_abstract("mul", inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone() * inputs[1].clone()])
    }

    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 2)?;
        let left = &inputs[0];
        let right = &inputs[1];
        Ok(vec![JvpTracer {
            primal: left.primal.clone() * right.primal.clone(),
            tangent: LinearTerm::add(
                LinearTerm::scale(left.tangent.clone(), right.primal.clone()),
                LinearTerm::scale(right.tangent.clone(), left.primal.clone()),
            ),
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

    #[cfg(feature = "xla")]
    fn lower_plain_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        let operation =
            lowerer
                .block
                .append_operation(stable_hlo::multiply(input_values[0], input_values[1], lowerer.location));
        Ok(vec![operation.result(0).expect("stablehlo.multiply should return one result").as_ref()])
    }

    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't, 'm>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't, 'm>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let operation =
            lowerer
                .block
                .append_operation(stable_hlo::multiply(input_values[0], input_values[1], lowerer.location));
        Ok(vec![operation.result(0).expect("stablehlo.multiply should return one result").as_ref()])
    }
}

impl<V> JvpOp<V> for MulOp
where
    V: TraceValue + Mul<Output = V>,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 2)?;
        let left = &inputs[0];
        let right = &inputs[1];
        Ok(vec![JvpTracer {
            primal: left.primal.clone() * right.primal.clone(),
            tangent: T::add(
                T::scale(right.primal.clone(), left.tangent.clone()),
                T::scale(left.primal.clone(), right.tangent.clone()),
            ),
        }])
    }
}

impl<V> BatchOp<V> for MulOp
where
    V: TraceValue + Mul<Output = V>,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        expect_batch_sizes_match(&inputs[0], &inputs[1])?;
        Ok(vec![Batch::new(
            inputs[0]
                .lanes()
                .iter()
                .cloned()
                .zip(inputs[1].lanes().iter().cloned())
                .map(|(left, right)| left * right)
                .collect(),
        )])
    }
}

#[cfg(test)]
mod tests {
    use crate::tracing_v2::test_support;

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn test_mul_jvp_matches_the_product_rule() {
        let output = MulOp
            .jvp::<f64>(&[
                JvpTracer { primal: 2.0f64, tangent: 3.0f64 },
                JvpTracer { primal: 5.0f64, tangent: -1.0f64 },
            ])
            .unwrap()
            .pop()
            .unwrap();

        approx_eq(output.primal, 10.0);
        approx_eq(output.tangent, 13.0);
        test_support::assert_bilinear_pushforward_rendering();
    }
}
