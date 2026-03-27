//! Matrix-specific tracing extensions built on top of the core `tracing_v2` primitives.
//!
//! This module keeps the non-commutative details of matrix multiplication localized by introducing explicit left and
//! right linear actions for transposition, while reusing the same JVP, batching, and JIT infrastructure as scalar
//! values.

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

#[cfg(feature = "xla")]
use ryft_mlir::dialects::stable_hlo;
#[cfg(feature = "xla")]
use ryft_mlir::{Block, Operation, Value, ValueRef};

#[cfg(feature = "xla")]
use crate::xla::lowering::{
    LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode, ShardMapMlirLowerer,
};
use crate::{
    tracing_v2::{
        FloatExt, TraceError, TraceValue, TransformLeaf, ZeroLike,
        batch::Batch as BatchedValue,
        forward::{JvpTracer, TangentSpace},
        graph::AtomId,
        jit::JitTracer,
        linear::LinearTerm,
        ops::{BatchOp, Op},
        program::ProgramBuilder,
    },
    types::{ArrayType, DataType, Shape, Size, Typed},
};

/// Matrix operations required by the tracing prototype.
pub trait MatrixOps: Sized {
    /// Matrix multiplication.
    fn matmul(self, rhs: Self) -> Self;

    /// Matrix transpose.
    fn transpose_matrix(self) -> Self;
}

/// Convenience trait for traceable matrix leaves.
///
/// Matrix values use [`ArrayType`] as their staged descriptor. The matrix-specific primitives in this module expect
/// those array types to describe rank-2 matrices with static dimensions and floating-point element types.
pub trait MatrixValue: TraceValue + MatrixOps {}

impl<T> MatrixValue for T where T: TraceValue + MatrixOps {}

impl MatrixOps for f32 {
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        self
    }
}

impl MatrixOps for f64 {
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        self
    }
}

/// Tangent representation for matrix-valued primals.
pub trait MatrixTangentSpace<V>: TangentSpace<V>
where
    V: MatrixValue,
{
    /// Applies the linear map `tangent -> factor @ tangent`.
    fn matmul_left(factor: V, tangent: Self) -> Self;

    /// Applies the linear map `tangent -> tangent @ factor`.
    fn matmul_right(tangent: Self, factor: V) -> Self;

    /// Transposes a tangent value.
    fn transpose_matrix(value: Self) -> Self;
}

impl<V> MatrixTangentSpace<V> for V
where
    V: MatrixValue + FloatExt + ZeroLike,
{
    #[inline]
    fn matmul_left(factor: V, tangent: Self) -> Self {
        factor.matmul(tangent)
    }

    #[inline]
    fn matmul_right(tangent: Self, factor: V) -> Self {
        tangent.matmul(factor)
    }

    #[inline]
    fn transpose_matrix(value: Self) -> Self {
        value.transpose_matrix()
    }
}

fn expect_input_count(inputs: usize, expected: usize) -> Result<(), TraceError> {
    if inputs == expected { Ok(()) } else { Err(TraceError::InvalidInputCount { expected, got: inputs }) }
}

fn expect_batch_sizes_match<V>(left: &BatchedValue<V>, right: &BatchedValue<V>) -> Result<(), TraceError> {
    if left.len() == right.len() { Ok(()) } else { Err(TraceError::MismatchedBatchSize) }
}

fn matrix_array_type(data_type: DataType, rows: usize, cols: usize) -> ArrayType {
    ArrayType::new(data_type, Shape::new(vec![Size::Static(rows), Size::Static(cols)]), None)
}

fn matrix_parts(r#type: &ArrayType, op: &'static str) -> Result<(DataType, usize, usize), TraceError> {
    if !matches!(r#type.data_type, DataType::F32 | DataType::F64) || r#type.rank() != 2 {
        return Err(TraceError::IncompatibleAbstractValues { op });
    }

    let Size::Static(rows) = r#type.dimension(0) else {
        return Err(TraceError::IncompatibleAbstractValues { op });
    };
    let Size::Static(cols) = r#type.dimension(1) else {
        return Err(TraceError::IncompatibleAbstractValues { op });
    };
    Ok((r#type.data_type, rows, cols))
}

fn matmul_abstract(lhs: &ArrayType, rhs: &ArrayType, op: &'static str) -> Result<ArrayType, TraceError> {
    let (lhs_data_type, lhs_rows, lhs_cols) = matrix_parts(lhs, op)?;
    let (rhs_data_type, rhs_rows, rhs_cols) = matrix_parts(rhs, op)?;
    if lhs_data_type != rhs_data_type || lhs_cols != rhs_rows {
        return Err(TraceError::IncompatibleAbstractValues { op });
    }
    Ok(matrix_array_type(lhs_data_type, lhs_rows, rhs_cols))
}

fn transpose_abstract(input: &ArrayType, op: &'static str) -> Result<ArrayType, TraceError> {
    let (data_type, rows, cols) = matrix_parts(input, op)?;
    Ok(matrix_array_type(data_type, cols, rows))
}

fn matrix_transpose_is_identity_type(r#type: &ArrayType) -> bool {
    matches!(r#type.shape.dimensions.as_slice(), [Size::Static(1), Size::Static(1)])
}

fn single_batch_output<V>(mut outputs: Vec<BatchedValue<V>>, op: &'static str) -> BatchedValue<V>
where
    V: MatrixValue,
{
    debug_assert_eq!(outputs.len(), 1, "{op} should produce a single batched output");
    outputs.pop().expect("single-output matrix primitive should return one batched output")
}

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

impl<V> Op<V> for MatMulOp
where
    V: MatrixValue,
{
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
            None,
            None,
            output_type,
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }

    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't, 'm>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't, 'm>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let output_type = lowerer.lower_tensor_type(&output_types[0])?;
        let operation = lowerer.block.append_operation(stable_hlo::dot_general(
            input_values[0],
            input_values[1],
            lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[1], &[0]),
            None,
            None,
            output_type,
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.dot_general should return one result").as_ref()])
    }
}

impl<V> BatchOp<V> for MatMulOp
where
    V: MatrixValue,
{
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

/// Linear map `tangent -> factor @ tangent`.
#[derive(Clone)]
pub(crate) struct LeftMatMulOp<V>
where
    V: MatrixValue,
{
    factor: V,
}

impl<V> LeftMatMulOp<V>
where
    V: MatrixValue,
{
    #[inline]
    pub(crate) fn new(factor: V) -> Self {
        Self { factor }
    }

    #[inline]
    pub(crate) fn factor(&self) -> &V {
        &self.factor
    }
}

impl<V> Debug for LeftMatMulOp<V>
where
    V: MatrixValue,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LeftMatMul")
    }
}

impl<V> Display for LeftMatMulOp<V>
where
    V: MatrixValue,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "left_matmul")
    }
}

impl<V> Op<V> for LeftMatMulOp<V>
where
    V: MatrixValue,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "left_matmul"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![matmul_abstract(&<V as Typed<ArrayType>>::tpe(&self.factor), &inputs[0], "left_matmul")?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![self.factor.clone().matmul(inputs[0].clone())])
    }

    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        expect_input_count(inputs.len(), 1)?;
        let factor = super::ops::lift_jit_constant(self.factor(), &inputs[0].primal);
        let factor = JvpTracer { primal: factor.clone(), tangent: LinearTerm::zero_like(&factor, &inputs[0].tangent) };
        Ok(vec![factor.matmul(inputs[0].clone())])
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
            Arc::new(LeftMatMulOp::new(self.factor.clone().transpose_matrix())),
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
            let output_abstract = &output_types[0];
            let transposed_output_abstract = match output_abstract.shape.dimensions.as_slice() {
                [first, second] => ArrayType::new(output_abstract.data_type, Shape::new(vec![*second, *first]), None),
                _ => return Err(LoweringError::UnsupportedOp { op: self.name().to_string() }),
            };
            let transposed_output_type = lowerer.lower_tensor_type(&transposed_output_abstract)?;
            let factor = lowerer.lower_literal_value(&self.factor.clone().transpose_matrix())?;
            let dot = lowerer.block.append_operation(stable_hlo::dot_general(
                input_values[0],
                factor,
                lowerer.context.stable_hlo_dot_dimensions(&[], &[], &[0], &[0]),
                None,
                None,
                transposed_output_type,
                lowerer.location,
            ));
            let operation = lowerer.block.append_operation(stable_hlo::transpose(
                dot.result(0).expect("stablehlo.dot_general should return one result").as_ref(),
                &[1, 0],
                lowerer.location,
            ));
            Ok(vec![operation.result(0).expect("stablehlo.transpose should return one result").as_ref()])
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

impl<V> BatchOp<V> for LeftMatMulOp<V>
where
    V: MatrixValue,
{
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(
            inputs[0].lanes().iter().cloned().map(|lane| self.factor.clone().matmul(lane)).collect(),
        )])
    }
}

/// Linear map `tangent -> tangent @ factor`.
#[derive(Clone)]
pub(crate) struct RightMatMulOp<V>
where
    V: MatrixValue,
{
    factor: V,
}

impl<V> RightMatMulOp<V>
where
    V: MatrixValue,
{
    #[inline]
    pub(crate) fn new(factor: V) -> Self {
        Self { factor }
    }

    #[inline]
    pub(crate) fn factor(&self) -> &V {
        &self.factor
    }
}

impl<V> Debug for RightMatMulOp<V>
where
    V: MatrixValue,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "RightMatMul")
    }
}

impl<V> Display for RightMatMulOp<V>
where
    V: MatrixValue,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "right_matmul")
    }
}

impl<V> Op<V> for RightMatMulOp<V>
where
    V: MatrixValue,
{
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
        let factor = super::ops::lift_jit_constant(self.factor(), &inputs[0].primal);
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
                None,
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

impl<V> BatchOp<V> for RightMatMulOp<V>
where
    V: MatrixValue,
{
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(
            inputs[0].lanes().iter().cloned().map(|lane| lane.matmul(self.factor.clone())).collect(),
        )])
    }
}

/// Linear transpose primitive used inside matrix-valued pushforwards and pullbacks.
#[derive(Clone, Default)]
pub(crate) struct LinearMatrixTransposeOp;

impl Debug for LinearMatrixTransposeOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "LinearMatrixTranspose")
    }
}

impl Display for LinearMatrixTransposeOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "linear_matrix_transpose")
    }
}

impl<V> Op<V> for LinearMatrixTransposeOp
where
    V: MatrixValue,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &'static str {
        "linear_matrix_transpose"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![transpose_abstract(&inputs[0], "linear_matrix_transpose")?])
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
        let contribution = builder.add_equation(Arc::new(LinearMatrixTransposeOp), vec![output_cotangents[0]])?[0];
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
        <MatrixTransposeOp as Op<V>>::lower_plain_mlir(&MatrixTransposeOp, input_values, output_types, mode, lowerer)
    }
}

impl<V> BatchOp<V> for LinearMatrixTransposeOp
where
    V: MatrixValue,
{
    fn batch(&self, inputs: &[BatchedValue<V>]) -> Result<Vec<BatchedValue<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![BatchedValue::new(inputs[0].lanes().iter().cloned().map(MatrixOps::transpose_matrix).collect())])
    }
}

impl<V, T> MatrixOps for JvpTracer<V, T>
where
    V: MatrixValue,
    T: MatrixTangentSpace<V>,
{
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        JvpTracer {
            primal: self.primal.clone().matmul(rhs.primal.clone()),
            tangent: T::add(T::matmul_right(self.tangent, rhs.primal), T::matmul_left(self.primal, rhs.tangent)),
        }
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        if matrix_transpose_is_identity_type(&self.primal.tpe()) {
            return self;
        }
        JvpTracer { primal: self.primal.transpose_matrix(), tangent: T::transpose_matrix(self.tangent) }
    }
}

impl<V> MatrixOps for JitTracer<V>
where
    V: TransformLeaf,
{
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        self.binary(rhs, Arc::new(MatMulOp), MatrixOps::matmul)
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        if matrix_transpose_is_identity_type(&self.tpe()) {
            return self;
        }
        self.unary(Arc::new(MatrixTransposeOp), MatrixOps::transpose_matrix)
    }
}

impl<V> MatrixOps for BatchedValue<V>
where
    V: MatrixValue,
{
    #[inline]
    fn matmul(self, rhs: Self) -> Self {
        single_batch_output(MatMulOp.batch(&[self, rhs]).expect("batched matmul rule should succeed"), "matmul")
    }

    #[inline]
    fn transpose_matrix(self) -> Self {
        if self.lanes().first().map(|lane| matrix_transpose_is_identity_type(&lane.tpe())).unwrap_or(false) {
            return self;
        }
        single_batch_output(
            MatrixTransposeOp.batch(&[self]).expect("batched transpose rule should succeed"),
            "matrix_transpose",
        )
    }
}

impl<V> MatrixTangentSpace<V> for LinearTerm<V>
where
    V: MatrixValue + FloatExt + ZeroLike,
{
    #[inline]
    fn matmul_left(factor: V, tangent: Self) -> Self {
        tangent.apply_linear_op(Arc::new(LeftMatMulOp::new(factor)))
    }

    #[inline]
    fn matmul_right(tangent: Self, factor: V) -> Self {
        tangent.apply_linear_op(Arc::new(RightMatMulOp::new(factor)))
    }

    #[inline]
    fn transpose_matrix(value: Self) -> Self {
        value.apply_linear_op(Arc::new(LinearMatrixTransposeOp))
    }
}

#[cfg(any(feature = "ndarray", test))]
mod ndarray_support {
    use ndarray::Array2;

    use super::{MatrixOps, matrix_array_type};
    use crate::{
        parameters::Parameter,
        tracing_v2::{CoordinateValue, FloatExt, OneLike, TraceValue, ZeroLike},
        types::{ArrayType, DataType, Typed},
    };

    impl Parameter for Array2<f32> {}
    impl Parameter for Array2<f64> {}

    impl FloatExt for Array2<f32> {
        #[inline]
        fn sin(self) -> Self {
            self.mapv(f32::sin)
        }

        #[inline]
        fn cos(self) -> Self {
            self.mapv(f32::cos)
        }
    }

    impl FloatExt for Array2<f64> {
        #[inline]
        fn sin(self) -> Self {
            self.mapv(f64::sin)
        }

        #[inline]
        fn cos(self) -> Self {
            self.mapv(f64::cos)
        }
    }

    impl Typed<ArrayType> for Array2<f32> {
        #[inline]
        fn tpe(&self) -> ArrayType {
            matrix_array_type(DataType::F32, self.nrows(), self.ncols())
        }
    }

    impl TraceValue for Array2<f32> {}

    impl Typed<ArrayType> for Array2<f64> {
        #[inline]
        fn tpe(&self) -> ArrayType {
            matrix_array_type(DataType::F64, self.nrows(), self.ncols())
        }
    }

    impl TraceValue for Array2<f64> {}

    impl ZeroLike for Array2<f32> {
        #[inline]
        fn zero_like(&self) -> Self {
            Array2::from_elem(self.raw_dim(), 0.0)
        }
    }

    impl ZeroLike for Array2<f64> {
        #[inline]
        fn zero_like(&self) -> Self {
            Array2::from_elem(self.raw_dim(), 0.0)
        }
    }

    impl OneLike for Array2<f32> {
        #[inline]
        fn one_like(&self) -> Self {
            Array2::from_elem(self.raw_dim(), 1.0)
        }
    }

    impl OneLike for Array2<f64> {
        #[inline]
        fn one_like(&self) -> Self {
            Array2::from_elem(self.raw_dim(), 1.0)
        }
    }

    impl CoordinateValue for Array2<f32> {
        type Coordinate = f32;

        #[inline]
        fn coordinate_count(&self) -> usize {
            self.len()
        }

        fn coordinate_basis(&self) -> Vec<Self> {
            let mut basis = Vec::with_capacity(self.len());
            for row in 0..self.nrows() {
                for col in 0..self.ncols() {
                    let mut tangent = Array2::from_elem(self.raw_dim(), 0.0);
                    tangent[(row, col)] = 1.0;
                    basis.push(tangent);
                }
            }
            basis
        }

        #[inline]
        fn coordinates(&self) -> Vec<Self::Coordinate> {
            self.iter().copied().collect::<Vec<_>>()
        }
    }

    impl CoordinateValue for Array2<f64> {
        type Coordinate = f64;

        #[inline]
        fn coordinate_count(&self) -> usize {
            self.len()
        }

        fn coordinate_basis(&self) -> Vec<Self> {
            let mut basis = Vec::with_capacity(self.len());
            for row in 0..self.nrows() {
                for col in 0..self.ncols() {
                    let mut tangent = Array2::from_elem(self.raw_dim(), 0.0);
                    tangent[(row, col)] = 1.0;
                    basis.push(tangent);
                }
            }
            basis
        }

        #[inline]
        fn coordinates(&self) -> Vec<Self::Coordinate> {
            self.iter().copied().collect::<Vec<_>>()
        }
    }

    impl MatrixOps for Array2<f32> {
        #[inline]
        fn matmul(self, rhs: Self) -> Self {
            self.dot(&rhs)
        }

        #[inline]
        fn transpose_matrix(self) -> Self {
            if self.nrows() == 1 && self.ncols() == 1 {
                return self;
            }
            self.reversed_axes()
        }
    }

    impl MatrixOps for Array2<f64> {
        #[inline]
        fn matmul(self, rhs: Self) -> Self {
            self.dot(&rhs)
        }

        #[inline]
        fn transpose_matrix(self) -> Self {
            if self.nrows() == 1 && self.ncols() == 1 {
                return self;
            }
            self.reversed_axes()
        }
    }
}
