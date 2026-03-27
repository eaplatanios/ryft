//! Primitive operation traits and scalar primitive implementations for `tracing_v2`.
//!
//! The op set is intentionally open: each primitive is represented by its own type implementing one or more
//! transform-specific traits. This keeps graph representations extensible without requiring central enums.

use std::{
    any::Any,
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg},
    sync::Arc,
};

#[cfg(feature = "xla")]
use ryft_mlir::dialects::stable_hlo;
#[cfg(feature = "xla")]
use ryft_mlir::{Block, Operation, Value, ValueRef};

use crate::tracing_v2::{
    FloatExt, TraceError, TraceValue, TransformLeaf, ZeroLike,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    graph::AtomId,
    jit::JitTracer,
    linear::LinearTerm,
    matmul::MatrixOps,
    program::ProgramBuilder,
};
use crate::types::ArrayType;
#[cfg(feature = "xla")]
use crate::xla::lowering::{
    LoweringError, MlirLowerableValue, PlainMlirLowerer, PlainMlirLoweringMode, ShardMapMlirLowerer,
};

/// Core primitive operation interface understood by staged graphs.
pub(crate) trait Op<V>: Debug + Display
where
    V: TraceValue,
{
    /// Returns this operation as [`Any`] for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns the stable primitive name used in diagnostics and pretty-printing.
    fn name(&self) -> &'static str;

    /// Computes abstract outputs from abstract inputs without executing the operation.
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError>;

    /// Executes the operation on concrete values.
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>;

    /// Replays this staged op while tracing a linearized JIT program.
    fn replay_linearized_jit(
        &self,
        _inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        Err(TraceError::HigherOrderOpFailure {
            op: "replay_program_graph",
            message: format!("replaying linearized values through staged op '{}' is not implemented", self.name()),
        })
    }

    /// Applies this op's program-level JVP rule while linearizing a staged program.
    fn apply_program_jvp_rule(
        &self,
        _inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        Err(TraceError::HigherOrderOpFailure {
            op: "linearize_program",
            message: format!("JVP rule for staged op '{}' is not implemented", self.name()),
        })
    }

    /// Applies this op's transpose rule while transposing a linearized staged program.
    fn transpose_program_op(
        &self,
        _builder: &mut ProgramBuilder<V>,
        _inputs: &[AtomId],
        _outputs: &[AtomId],
        _output_cotangents: &[AtomId],
    ) -> Result<Vec<Option<AtomId>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        Err(TraceError::HigherOrderOpFailure {
            op: "transpose_linear_program",
            message: format!("transpose rule for staged op '{}' is not implemented", self.name()),
        })
    }

    /// Lowers this op inside a plain StableHLO MLIR graph.
    #[cfg(feature = "xla")]
    fn lower_plain_mlir<'b, 'c, 't>(
        &self,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _mode: PlainMlirLoweringMode,
        _lowerer: &mut PlainMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        Err(LoweringError::UnsupportedOp { op: self.name().to_string() })
    }

    /// Lowers this op inside a Shardy/StableHLO MLIR graph for traced XLA programs.
    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't, 'm>(
        &self,
        _input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        _lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't, 'm>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        Err(LoweringError::UnsupportedOp { op: self.name().to_string() })
    }
}

/// Shared reference to a dynamically dispatched staged operation.
pub(crate) type StagedOpRef<V> = Arc<dyn Op<V>>;

/// Primitive operation with a forward-mode differentiation rule.
pub(crate) trait JvpOp<V>: Op<V>
where
    V: TraceValue,
{
    /// Applies the primitive's forward-mode rule to traced inputs.
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>;
}

/// Primitive operation with a batching rule used by `vmap`.
pub(crate) trait BatchOp<V>: Op<V>
where
    V: TraceValue,
{
    /// Applies the primitive's batching rule to batched inputs.
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError>;
}

impl<T, V> Op<V> for Arc<T>
where
    T: Op<V> + ?Sized,
    V: TraceValue,
{
    #[inline]
    fn as_any(&self) -> &dyn Any {
        (**self).as_any()
    }

    #[inline]
    fn name(&self) -> &'static str {
        (**self).name()
    }

    #[inline]
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        (**self).abstract_eval(inputs)
    }

    #[inline]
    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        (**self).eval(inputs)
    }

    #[inline]
    fn replay_linearized_jit(
        &self,
        inputs: Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>,
    ) -> Result<Vec<JvpTracer<JitTracer<V>, LinearTerm<JitTracer<V>>>>, TraceError>
    where
        V: TransformLeaf,
    {
        (**self).replay_linearized_jit(inputs)
    }

    #[inline]
    fn apply_program_jvp_rule(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
    {
        (**self).apply_program_jvp_rule(inputs)
    }

    #[inline]
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
        (**self).transpose_program_op(builder, inputs, outputs, output_cotangents)
    }

    #[cfg(feature = "xla")]
    #[inline]
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
        (**self).lower_plain_mlir(input_values, output_types, mode, lowerer)
    }

    #[cfg(feature = "xla")]
    #[inline]
    fn lower_shard_map_mlir<'b, 'c, 't, 'm>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't, 'm>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError>
    where
        V: MlirLowerableValue,
    {
        (**self).lower_shard_map_mlir(input_values, output_types, lowerer)
    }
}

impl<T, V> JvpOp<V> for Arc<T>
where
    T: JvpOp<V> + ?Sized,
    V: TraceValue,
{
    #[inline]
    fn jvp<U>(&self, inputs: &[JvpTracer<V, U>]) -> Result<Vec<JvpTracer<V, U>>, TraceError>
    where
        U: TangentSpace<V>,
    {
        (**self).jvp(inputs)
    }
}

impl<T, V> BatchOp<V> for Arc<T>
where
    T: BatchOp<V> + ?Sized,
    V: TraceValue,
{
    #[inline]
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        (**self).batch(inputs)
    }
}
fn expect_input_count(inputs: usize, expected: usize) -> Result<(), TraceError> {
    if inputs == expected { Ok(()) } else { Err(TraceError::InvalidInputCount { expected, got: inputs }) }
}

fn expect_batch_sizes_match<V>(left: &Batch<V>, right: &Batch<V>) -> Result<(), TraceError> {
    if left.len() == right.len() { Ok(()) } else { Err(TraceError::MismatchedBatchSize) }
}

pub(crate) fn lift_jit_constant<V>(constant: &V, exemplar: &JitTracer<V>) -> JitTracer<V>
where
    V: TraceValue,
{
    let builder = exemplar.builder_handle();
    let atom = builder.borrow_mut().add_constant(constant.clone());
    JitTracer::from_staged_parts(constant.clone(), atom, builder, exemplar.staging_error_handle())
}

fn unary_abstract<V>(_op: &'static str, inputs: &[ArrayType]) -> Result<ArrayType, TraceError>
where
    V: TraceValue,
{
    expect_input_count(inputs.len(), 1)?;
    Ok(inputs[0].clone())
}

fn binary_same_abstract<V>(op: &'static str, inputs: &[ArrayType]) -> Result<ArrayType, TraceError>
where
    V: TraceValue,
{
    expect_input_count(inputs.len(), 2)?;
    if inputs[0].data_type != inputs[1].data_type || inputs[0].shape != inputs[1].shape {
        Err(TraceError::IncompatibleAbstractValues { op })
    } else {
        Ok(ArrayType {
            data_type: inputs[0].data_type,
            shape: inputs[0].shape.clone(),
            layout: if inputs[0].layout == inputs[1].layout { inputs[0].layout.clone() } else { None },
        })
    }
}

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

impl<V> Op<V> for AddOp
where
    V: TraceValue + Add<Output = V>,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "add"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![binary_same_abstract::<V>("add", inputs)?])
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
    fn lower_shard_map_mlir<'b, 'c, 't, 'm>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't, 'm>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let operation =
            lowerer.block.append_operation(stable_hlo::add(input_values[0], input_values[1], lowerer.location));
        Ok(vec![operation.result(0).expect("stablehlo.add should return one result").as_ref()])
    }
}

impl<V> JvpOp<V> for AddOp
where
    V: TraceValue + Add<Output = V>,
{
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

impl<V> BatchOp<V> for AddOp
where
    V: TraceValue + Add<Output = V>,
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
                .map(|(left, right)| left + right)
                .collect(),
        )])
    }
}

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
        Ok(vec![binary_same_abstract::<V>("mul", inputs)?])
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

impl<V> Op<V> for NegOp
where
    V: TraceValue + Neg<Output = V>,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "neg"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract::<V>("neg", inputs)?])
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
    fn lower_shard_map_mlir<'b, 'c, 't, 'm>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't, 'm>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let operation = lowerer.block.append_operation(stable_hlo::negate(input_values[0], lowerer.location));
        Ok(vec![operation.result(0).expect("stablehlo.negate should return one result").as_ref()])
    }
}

impl<V> JvpOp<V> for NegOp
where
    V: TraceValue + Neg<Output = V>,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![JvpTracer { primal: -inputs[0].primal.clone(), tangent: T::neg(inputs[0].tangent.clone()) }])
    }
}

impl<V> BatchOp<V> for NegOp
where
    V: TraceValue + Neg<Output = V>,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| -lane).collect())])
    }
}

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

impl<V> Op<V> for SinOp
where
    V: TraceValue + FloatExt,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "sin"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract::<V>("sin", inputs)?])
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
    fn lower_shard_map_mlir<'b, 'c, 't, 'm>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't, 'm>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let operation = lowerer.block.append_operation(stable_hlo::sine(
            input_values[0],
            stable_hlo::Accuracy::Default,
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.sine should return one result").as_ref()])
    }
}

impl<V> JvpOp<V> for SinOp
where
    V: TraceValue + FloatExt,
{
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

impl<V> BatchOp<V> for SinOp
where
    V: TraceValue + FloatExt,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| lane.sin()).collect())])
    }
}

/// Elementwise cosine primitive.
#[derive(Clone, Default)]
pub(crate) struct CosOp;

impl Debug for CosOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Cos")
    }
}

impl Display for CosOp {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "cos")
    }
}

impl<V> Op<V> for CosOp
where
    V: TraceValue + FloatExt,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "cos"
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        Ok(vec![unary_abstract::<V>("cos", inputs)?])
    }

    fn eval(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![inputs[0].clone().cos()])
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
            primal: input.primal.clone().cos(),
            tangent: LinearTerm::neg(LinearTerm::scale(input.tangent.clone(), input.primal.clone().sin())),
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
        let operation = lowerer.block.append_operation(stable_hlo::cosine(
            input_values[0],
            stable_hlo::Accuracy::Default,
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.cosine should return one result").as_ref()])
    }

    #[cfg(feature = "xla")]
    fn lower_shard_map_mlir<'b, 'c, 't, 'm>(
        &self,
        input_values: &[ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't, 'm>,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, LoweringError> {
        let operation = lowerer.block.append_operation(stable_hlo::cosine(
            input_values[0],
            stable_hlo::Accuracy::Default,
            lowerer.location,
        ));
        Ok(vec![operation.result(0).expect("stablehlo.cosine should return one result").as_ref()])
    }
}

impl<V> JvpOp<V> for CosOp
where
    V: TraceValue + FloatExt,
{
    fn jvp<T>(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>
    where
        T: TangentSpace<V>,
    {
        expect_input_count(inputs.len(), 1)?;
        let input = &inputs[0];
        Ok(vec![JvpTracer {
            primal: input.primal.clone().cos(),
            tangent: T::neg(T::scale(input.primal.clone().sin(), input.tangent.clone())),
        }])
    }
}

impl<V> BatchOp<V> for CosOp
where
    V: TraceValue + FloatExt,
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        expect_input_count(inputs.len(), 1)?;
        Ok(vec![Batch::new(inputs[0].lanes().iter().cloned().map(|lane| lane.cos()).collect())])
    }
}

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
        Ok(vec![unary_abstract::<V>("scale", inputs)?])
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
    use std::sync::Arc;

    use indoc::indoc;

    use crate::{
        parameters::Placeholder,
        tracing_v2::{TraceError, test_support},
        types::{ArrayType, DataType, Layout, Shape, StridedLayout},
    };

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    #[test]
    fn add_abstract_eval_rejects_incompatible_inputs() {
        let error = <AddOp as Op<f64>>::abstract_eval(
            &AddOp,
            &[ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F64)],
        )
        .unwrap_err();
        assert_eq!(error, TraceError::IncompatibleAbstractValues { op: "add" });
        test_support::assert_reference_graph_rendering();
    }

    #[test]
    fn add_abstract_eval_drops_layout_when_inputs_disagree() {
        let output = <AddOp as Op<f64>>::abstract_eval(
            &AddOp,
            &[
                ArrayType::new(DataType::F32, Shape::scalar(), Some(Layout::Strided(StridedLayout::new(vec![])))),
                ArrayType::scalar(DataType::F32),
            ],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::scalar(DataType::F32)]);
    }

    #[test]
    fn mul_jvp_matches_the_product_rule() {
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

    #[test]
    fn add_batch_requires_matching_lane_counts() {
        let error = AddOp.batch(&[Batch::new(vec![1.0f64, 2.0f64]), Batch::new(vec![3.0f64])]).unwrap_err();
        assert_eq!(error, TraceError::MismatchedBatchSize);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn scale_transpose_scales_output_cotangents() {
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
