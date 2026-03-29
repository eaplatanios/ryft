//! Sine primitive for [`crate::tracing_v2`].

use std::{
    any::Any,
    fmt::{Debug, Display},
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

use super::{expect_input_count, unary_abstract};

/// Elementwise sine primitive.
#[derive(Clone, Default)]
pub(crate) struct SinOp;

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

impl<V: TraceValue + FloatExt> Op<V> for SinOp {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "sin"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().sin()])
    }

    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().sin(),
            tangent: LinearTerm::scale(input.tangent.clone(), input.primal.clone().cos()),
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
        let operation = lowerer.block.append_operation(stable_hlo::sine(
            input_values[0],
            stable_hlo::Accuracy::Default,
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.sine should return one result").as_ref()])
    }

    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let operation = lowerer.block.append_operation(stable_hlo::sine(
            input_values[0],
            stable_hlo::Accuracy::Default,
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.sine should return one result").as_ref()])
    }
}

impl<V: TraceValue + FloatExt> JvpOp<V> for SinOp {
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().sin(),
            tangent: T::scale(input.primal.clone().cos(), input.tangent.clone()),
        }])
    }
}

impl<V: TraceValue + FloatExt> BatchOp<V> for SinOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| lane.sin()).collect())])
    }
}
