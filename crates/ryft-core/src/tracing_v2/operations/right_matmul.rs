//! Right matrix-multiplication primitive for [`crate::tracing_v2`].

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

#[cfg(feature = "xla")]
use ryft_mlir::ValueRef;
#[cfg(all(feature = "xla", feature = "ndarray"))]
use ryft_mlir::dialects::stable_hlo;
#[cfg(all(feature = "xla", feature = "ndarray"))]
use ryft_mlir::{Block, Operation, Value};

use crate::tracing_v2::{
    FloatExt, TraceError, TransformLeaf, ZeroLike,
    batch::Batch as BatchedValue,
    forward::{JvpTracer, TangentSpace},
    graph::AtomId,
    jit::JitTracer,
    linear::LinearTerm,
    ops::{BatchOp, Op},
    program::ProgramBuilder,
};
use crate::types::{ArrayType, Typed};
#[cfg(feature = "xla")]
use crate::xla::lowering::{LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode};

use super::{
    expect_input_count, lift_jit_constant,
    matrix::{MatrixOps, MatrixValue, matmul_abstract},
};

/// Linear map `tangent -> tangent @ factor`.
#[derive(Clone)]
pub(crate) struct RightMatMulOp<V: MatrixValue> {
    factor: V,
}

impl<V: MatrixValue> RightMatMulOp<V> {
    /// Creates one right multiplication op capturing the provided factor.
    #[inline]
    pub(crate) fn new(factor: V) -> Self {
        Self { factor }
    }

    /// Returns the captured matrix factor.
    #[inline]
    pub(crate) fn factor(&self) -> &V {
        &self.factor
    }
}

impl<V: MatrixValue> Debug for RightMatMulOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "RightMatMul")
    }
}

impl<V: MatrixValue> Display for RightMatMulOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "right_matmul")
    }
}

impl<V: MatrixValue> Op<V> for RightMatMulOp<V> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "right_matmul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![matmul_abstract(&inputs[0], &<V as Typed<ArrayType>>::tpe(&self.factor), "right_matmul")?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().matmul(self.factor.clone())])
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
        let factor = JvpTracer { primal: factor.clone(), tangent: LinearTerm::zero_like(&factor, &inputs[0].tangent) };
        Ok(vec![inputs[0].clone().matmul(factor)])
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
        let contribution = builder.add_equation(
            Arc::new(RightMatMulOp::new(self.factor.clone().transpose_matrix())),
            vec![output_cotangents[0]],
        )?[0];
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
        if !matches!(mode, PlainMlirLoweringMode::Unpacked) {
            return Err(LoweringError::UnsupportedOp { op: self.name().to_string() });
        }
        #[cfg(feature = "ndarray")]
        {
            let output_type = lowerer.lower_tensor_type(&output_types[0])?;
            let factor = lowerer.lower_literal_value(&self.factor.clone().transpose_matrix())?;
            let operation = lowerer.block.append_operation(stable_hlo::dot_general(
                input_values[0],
                factor,
                lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[1]),
                Some((stable_hlo::Precision::Default, stable_hlo::Precision::Default)),
                None,
                output_type,
                lowerer.location,
            ));
            Ok(vec![operation.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
        }
        #[cfg(not(feature = "ndarray"))]
        {
            let _ = output_types;
            let _ = input_values;
            let _ = lowerer;
            Err(LoweringError::UnsupportedOp { op: self.name().to_string() })
        }
    }
}

impl<V: MatrixValue> BatchOp<V> for RightMatMulOp<V> {
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(
            inputs[0].lanes().iter().cloned().map(|lane| lane.matmul(self.factor.clone())).collect(),
        )])
    }
}
