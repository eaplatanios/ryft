//! Addition primitive for [`crate::tracing_v2`].

use std::{
    any::Any,
    fmt::{Debug, Display},
    ops::Add,
};

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
use crate::xla::lowering::{
    LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode, ShardMapMlirLowerer,
};

use super::{binary_same_abstract, expect_batch_sizes_match, expect_input_count};

/// Elementwise addition primitive.
#[derive(Clone, Default)]
pub(crate) struct AddOp;

impl Debug for AddOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Add")
    }
}

impl Display for AddOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "add")
    }
}

impl<V: TraceValue + Add<Output = V>> Op<V> for AddOp {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "add"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![binary_same_abstract("add", inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![inputs[0].clone() + inputs[1].clone()])
    }

    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![JvpTracer {
            primal: inputs[0].primal.clone() + inputs[1].primal.clone(),
            tangent: inputs[0].tangent.clone().add(inputs[1].tangent.clone()),
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
        _builder: &mut ProgramBuilder<V>,
        inputs: &[AtomId],
        outputs: &[AtomId],
        output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        expect_input_count(inputs.len(), 2)?;
        expect_input_count(outputs.len(), 1)?;
        expect_input_count(output_cotangents.len(), 1)?;
        Ok(vec![Some(output_cotangents[0]), Some(output_cotangents[0])])
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
            lowerer.block.append_operation(stable_hlo::add(input_values[0], input_values[1], lowerer.location));
        Ok(vec![operation.result(0).expect("stablehlo.add should return one result").as_ref()])
    }

    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let operation =
            lowerer.block.append_operation(stable_hlo::add(input_values[0], input_values[1], lowerer.location));
        Ok(vec![operation.result(0).expect("stablehlo.add should return one result").as_ref()])
    }
}

impl<V: TraceValue + Add<Output = V>> JvpOp<V> for AddOp {
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 2)?;
        Ok(vec![JvpTracer {
            primal: inputs[0].primal.clone() + inputs[1].primal.clone(),
            tangent: T::add(inputs[0].tangent.clone(), inputs[1].tangent.clone()),
        }])
    }
}

impl<V: TraceValue + Add<Output = V>> BatchOp<V> for AddOp {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 2)?;
        expect_batch_sizes_match(&inputs[0], &inputs[1])?;
        Ok(vec![Batch::new(
            inputs[0]
                .lanes()
                .iter()
                .cloned()
                .zip(inputs[1].lanes().iter().cloned())
                .map(|(left, right)| left + right)
                .collect(),
        )])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{
        tracing_v2::test_support,
        types::{DataType, Layout, Shape, StridedLayout},
    };

    use super::*;

    #[test]
    fn test_add_abstract_eval_rejects_incompatible_inputs() {
        let error = <AddOp as Op<f64>>::abstract_eval(
            &AddOp,
            &[ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F64)],
        )
        .unwrap_err();

        assert_eq!(error, TraceError::IncompatibleAbstractValues { op: "add" });
        test_support::assert_reference_graph_rendering();
    }

    #[test]
    fn test_add_abstract_eval_drops_layout_when_inputs_disagree() {
        let output = <AddOp as Op<f64>>::abstract_eval(
            &AddOp,
            &[
                ArrayType::new(DataType::F32, Shape::scalar(), Some(Layout::Strided(StridedLayout::new(vec![]))), None),
                ArrayType::scalar(DataType::F32),
            ],
        )
        .unwrap();

        assert_eq!(output, vec![ArrayType::scalar(DataType::F32)]);
    }

    #[test]
    fn test_add_batch_requires_matching_lane_counts() {
        let error = AddOp.batch(&[Batch::new(vec![1.0f64, 2.0f64]), Batch::new(vec![3.0f64])]).unwrap_err();

        assert_eq!(error, TraceError::MismatchedBatchSize);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }
}
