//! Matrix multiplication primitive for [`crate::tracing_v2`].

use std::fmt::{Debug, Display};

#[cfg(feature = "xla")]
use ryft_mlir::dialects::stable_hlo;
#[cfg(feature = "xla")]
use ryft_mlir::{Block, Operation, Value, ValueRef};

use crate::tracing_v2::{
    FloatExt, TraceError, TransformLeaf, ZeroLike,
    batch::Batch as BatchedValue,
    forward::JvpTracer,
    jit::JitTracer,
    linear::LinearTerm,
    ops::{BatchOp, Op},
};
use crate::types::ArrayType;
#[cfg(feature = "xla")]
use crate::xla::lowering::{
    LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode, ShardMapMlirLowerer,
};

use super::{
    expect_batch_sizes_match, expect_input_count,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
};

/// Primitive representing matrix multiplication.
#[derive(Clone, Default)]
pub(crate) struct MatMulOp;

impl Debug for MatMulOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MatMul")
    }
}

impl Display for MatMulOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "matmul")
    }
}

impl<V: MatrixValue> Op<V> for MatMulOp {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "matmul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![matmul_abstract(&inputs[0], &inputs[1], "matmul")?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }

    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
    }

    fn apply_program_jvp_rule(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone().matmul(inputs[1].clone())])
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
        let output_type = lowerer.lower_tensor_type(&output_types[0])?;
        let dot_dimensions = match mode {
            PlainMlirLoweringMode::Unpacked => lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]),
            PlainMlirLoweringMode::Packed { .. } => match output_types[0].shape.dimensions.len() {
                2 => lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]),
                3 => lowerer.context.stable_hlo_dot_dimensions(&[0], &[0], &[2], &[1]),
                _ => return Err(LoweringError::UnsupportedOp { op: <Self as Op<V>>::name(self).to_string() }),
            },
        };
        let operation = lowerer.block.append_operation(stable_hlo::dot_general(
            input_values[0],
            input_values[1],
            dot_dimensions,
            Some((stable_hlo::Precision::Default, stable_hlo::Precision::Default)),
            None,
            output_type,
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }

    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let output_type = lowerer.lower_tensor_type(&output_types[0])?;
        let operation = lowerer.block.append_operation(stable_hlo::dot_general(
            input_values[0],
            input_values[1],
            lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]),
            Some((stable_hlo::Precision::Default, stable_hlo::Precision::Default)),
            None,
            output_type,
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }
}

impl<V: MatrixValue> BatchOp<V> for MatMulOp {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        expect_batch_sizes_match(&inputs[0], &inputs[1])?;
        Ok(vec![BatchedValue::new(
            inputs[0]
                .lanes()
                .iter()
                .cloned()
                .zip(inputs[1].lanes().iter().cloned())
                .map(|(left, right)| left.matmul(right))
                .collect(),
        )])
    }
}
