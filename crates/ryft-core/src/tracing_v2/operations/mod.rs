//! Concrete staged operations for [`crate::tracing_v2`].

use crate::tracing_v2::{TraceError, TraceValue, batch::Batch, jit::JitTracer};
use crate::types::ArrayType;
use crate::xla::sharding::Sharding;

fn is_replicated_sharding(sharding: &Sharding) -> bool {
    sharding
        .dimensions
        .iter()
        .all(|dimension| matches!(dimension, crate::sharding::ShardingDimension::Replicated))
}

fn binary_output_sharding(inputs: &[ArrayType]) -> Option<Sharding> {
    match (&inputs[0].sharding, &inputs[1].sharding) {
        (Some(left), Some(right)) if left == right => Some(left.clone()),
        (Some(left), Some(right)) if is_replicated_sharding(left) => Some(right.clone()),
        (Some(left), Some(right)) if is_replicated_sharding(right) => Some(left.clone()),
        (Some(left), None) => Some(left.clone()),
        (None, Some(right)) => Some(right.clone()),
        _ => None,
    }
}

/// Elementwise addition.
pub(crate) mod add;

/// Elementwise cosine.
pub(crate) mod cos;

/// Linear matrix transposition.
pub(crate) mod linear_matrix_transpose;

/// Linear left matrix multiplication.
pub(crate) mod left_matmul;

/// Matrix capability layer shared by matrix staged operations.
pub(crate) mod matrix;

/// Matrix multiplication.
pub(crate) mod matmul;

/// Matrix transposition.
pub(crate) mod matrix_transpose;

/// Elementwise multiplication.
pub(crate) mod mul;

/// Elementwise negation.
pub(crate) mod neg;

/// Linear right matrix multiplication.
pub(crate) mod right_matmul;

/// Scalar and tensor scaling.
pub(crate) mod scale;

/// Elementwise sine.
pub(crate) mod sin;

/// Traced `vmap` operations.
pub(crate) mod vmap;

#[cfg(feature = "xla")]
/// Traced XLA sharding-constraint primitive.
pub(crate) mod with_sharding_constraint;

#[cfg(feature = "xla")]
/// Traced `shard_map` operations.
pub(crate) mod shard_map;

pub(crate) use add::AddOp;
pub(crate) use cos::CosOp;
pub(crate) use left_matmul::LeftMatMulOp;
pub(crate) use linear_matrix_transpose::LinearMatrixTransposeOp;
pub(crate) use matmul::MatMulOp;
pub(crate) use matrix_transpose::MatrixTransposeOp;
pub(crate) use mul::MulOp;
pub(crate) use neg::NegOp;
pub(crate) use right_matmul::RightMatMulOp;
pub(crate) use scale::ScaleOp;
#[cfg(feature = "xla")]
pub(crate) use shard_map::{LinearShardMapEvalMode, ShardMapOp};
pub(crate) use sin::SinOp;
pub(crate) use vmap::{FlatTracedVMap, VMapOp};
#[cfg(feature = "xla")]
pub(crate) use with_sharding_constraint::WithShardingConstraintOp;

/// Returns an input-count error when one staged op receives the wrong arity.
pub(crate) fn expect_input_count(inputs: usize, expected: usize) -> Result<(), TraceError> {
    if inputs == expected { Ok(()) } else { Err(TraceError::InvalidInputCount { expected, got: inputs }) }
}

/// Returns a batch-size error when two batched inputs disagree on their lane count.
pub(crate) fn expect_batch_sizes_match<V>(left: &Batch<V>, right: &Batch<V>) -> Result<(), TraceError> {
    if left.len() == right.len() { Ok(()) } else { Err(TraceError::MismatchedBatchSize) }
}

/// Lifts one concrete value into the staged graph owned by a JIT tracer.
pub(crate) fn lift_jit_constant<V>(constant: &V, exemplar: &JitTracer<V>) -> JitTracer<V>
where
    V: TraceValue,
{
    let builder = exemplar.builder_handle();
    let atom = builder.borrow_mut().add_constant(constant.clone());
    JitTracer::from_staged_parts(constant.clone(), atom, builder, exemplar.staging_error_handle())
}

/// Propagates one unary input type through a shape-preserving staged op.
pub(crate) fn unary_abstract(inputs: &[ArrayType]) -> Result<ArrayType, TraceError> {
    expect_input_count(inputs.len(), 1)?;
    Ok(inputs[0].clone())
}

/// Propagates one binary input type through a shape-preserving staged op.
pub(crate) fn binary_same_abstract(op: &'static str, inputs: &[ArrayType]) -> Result<ArrayType, TraceError> {
    expect_input_count(inputs.len(), 2)?;
    if inputs[0].data_type != inputs[1].data_type || inputs[0].shape != inputs[1].shape {
        Err(TraceError::IncompatibleAbstractValues { op })
    } else {
        Ok(ArrayType {
            data_type: inputs[0].data_type,
            shape: inputs[0].shape.clone(),
            layout: if inputs[0].layout == inputs[1].layout { inputs[0].layout.clone() } else { None },
            sharding: binary_output_sharding(inputs),
        })
    }
}
