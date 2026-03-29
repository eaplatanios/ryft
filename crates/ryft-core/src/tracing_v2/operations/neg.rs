//! Negation primitive for [`crate::tracing_v2`].

use std::{
    any::Any,
    fmt::{Debug, Display},
    ops::Neg,
    sync::Arc,
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

use super::{expect_input_count, unary_abstract};

/// Elementwise negation primitive.
#[derive(Clone, Default)]
pub(crate) struct NegOp;

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

impl<V: TraceValue + Neg<Output = V>> Op<V> for NegOp {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "neg"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract(inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![-inputs[0].clone()])
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
        let contribution = builder.add_equation(Arc::new(NegOp), vec![output_cotangents[0]])?[0];
        Ok(vec![Some(contribution)])
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
        let operation = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location));
        Ok(vec![operation.result(0).expect("stablehlo.negate should return one result").as_ref()])
    }

    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let operation = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location));
        Ok(vec![operation.result(0).expect("stablehlo.negate should return one result").as_ref()])
    }
}

impl<V: TraceValue + Neg<Output = V>> JvpOp<V> for NegOp {
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
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
