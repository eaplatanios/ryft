//! Matrix transpose primitive for [`crate::tracing_v2`].

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
    expect_input_count,
    matrix::{MatrixOps, MatrixValue, transpose_abstract},
};

/// Primitive representing matrix transposition.
#[derive(Clone, Default)]
pub(crate) struct MatrixTransposeOp;

impl Debug for MatrixTransposeOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "MatrixTranspose")
    }
}

impl Display for MatrixTransposeOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "matrix_transpose")
    }
}

impl<V> Op<V> for MatrixTransposeOp
where
    V: MatrixValue,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "matrix_transpose"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![transpose_abstract(&inputs[0], "matrix_transpose")?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }

    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
    }

    fn apply_program_jvp_rule(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().transpose_matrix()])
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
        let permutation = match mode {
            PlainMlirLoweringMode::Unpacked => vec![1, 0],
            PlainMlirLoweringMode::Packed { .. } => match output_types[0].shape.dimensions.len() {
                2 => vec![1, 0],
                3 => vec![0, 2, 1],
                _ => return Err(LoweringError::UnsupportedOp { op: <Self as Op<V>>::name(self).to_string() }),
            },
        };
        let operation = lowerer.block.append_operation(stable_hlo::transpose(
            input_values[0],
            permutation.as_slice(),
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.transpose should return one result").as_ref()])
    }

    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't, 'm>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't, 'm>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let operation =
            lowerer.block.append_operation(stable_hlo::transpose(input_values[0], &[1, 0], lowerer.location));
        Ok(vec![operation.result(0).expect("stablehlo.transpose should return one result").as_ref()])
    }
}

impl<V> BatchOp<V> for MatrixTransposeOp
where
    V: MatrixValue,
{
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(inputs[0].lanes().iter().cloned().map(MatrixOps::transpose_matrix).collect())])
    }
}
