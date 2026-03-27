//! Primitive operation traits for `tracing_v2`.
//!
//! The staged op set is intentionally open: each primitive is represented by its own concrete type implementing one
//! or more transform-specific traits. This module keeps only the operation-neutral dispatch interfaces.

use std::{
    any::Any,
    fmt::{Debug, Display},
    sync::Arc,
};

#[cfg(feature = "xla")]
use ryft_mlir::ValueRef;

use crate::tracing_v2::{
    FloatExt, MatrixOps, TraceError, TraceValue, TransformLeaf, ZeroLike,
    batch::Batch,
    forward::{JvpTracer, TangentSpace},
    graph::AtomId,
    jit::JitTracer,
    linear::LinearTerm,
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
