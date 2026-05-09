use std::collections::BTreeSet;

use crate::sharding::{Sharding, ShardingDimension};
use crate::tracing::Traceable;
use crate::types::{ArrayType, DataType, Shape, Size, TypeError};

use super::matmul::MatMul;
use super::matrix_transpose::MatrixTranspose;

/// Matrix operations required by the tracing prototype.
///
/// This convenience trait groups the matrix value-level operations used by generic user code and primitive replay.
pub trait MatrixOps: MatMul<Self> + MatrixTranspose {}

impl<T: MatMul<Self> + MatrixTranspose> MatrixOps for T {}

/// Convenience trait for traceable matrix leaves.
///
/// Matrix values use [`ArrayType`] as their staged descriptor. The matrix-specific primitives in
/// this module expect those array types to describe rank-2 matrices with static dimensions and
/// floating-point element types.
pub trait MatrixValue: Traceable<ArrayType> + MatrixOps {}

impl<T: Traceable<ArrayType> + MatrixOps> MatrixValue for T {}

fn matrix_array_type(data_type: DataType, rows: usize, cols: usize, sharding: Option<Sharding>) -> ArrayType {
    ArrayType::new(data_type, Shape::new(vec![Size::Static(rows), Size::Static(cols)]), None, sharding)
        .expect("matrix abstract evaluation should preserve rank-2 sharding")
}

fn matrix_parts(r#type: &ArrayType, op: &'static str) -> Result<(DataType, usize, usize), TypeError> {
    let is_supported_data_type =
        matches!(r#type.data_type, DataType::BF16 | DataType::F16 | DataType::F32 | DataType::F64);
    if !is_supported_data_type || r#type.rank() != 2 {
        return Err(TypeError { message: format!("{op} expects rank-2 floating-point matrix inputs") });
    }

    let Size::Static(rows) = r#type.dimension(0) else {
        return Err(TypeError { message: format!("{op} requires statically shaped matrix inputs") });
    };
    let Size::Static(cols) = r#type.dimension(1) else {
        return Err(TypeError { message: format!("{op} requires statically shaped matrix inputs") });
    };
    Ok((r#type.data_type, rows, cols))
}

fn is_replicated_sharding(sharding: &Sharding) -> bool {
    sharding.dimensions.iter().all(|dimension| matches!(dimension, ShardingDimension::Replicated))
}

fn merge_unique_axes(left: &BTreeSet<String>, right: &BTreeSet<String>) -> BTreeSet<String> {
    left.union(right).cloned().collect()
}

fn transpose_array_sharding(input: &ArrayType) -> Option<Sharding> {
    let sharding = input.sharding.clone()?;
    if sharding.rank() != 2 {
        return None;
    }
    Sharding::with_manual_axes(
        sharding.mesh.clone(),
        vec![sharding.dimensions[1].clone(), sharding.dimensions[0].clone()],
        sharding.unreduced_axes.clone(),
        sharding.reduced_manual_axes.clone(),
        sharding.varying_manual_axes.clone(),
    )
    .map(|sharding| sharding.without_auto_axes())
    .ok()
}

fn matmul_array_sharding(lhs: &ArrayType, rhs: &ArrayType) -> Option<Sharding> {
    let left = lhs.sharding.clone()?;
    let right = rhs.sharding.clone()?;
    if left == right && is_replicated_sharding(&left) {
        return Some(left);
    }
    if left.rank() != 2 || right.rank() != 2 {
        return None;
    }
    if !matches!(left.dimensions[1], ShardingDimension::Replicated)
        || !matches!(right.dimensions[0], ShardingDimension::Replicated)
    {
        return None;
    }
    Sharding::with_manual_axes(
        left.mesh.clone(),
        vec![left.dimensions[0].clone(), right.dimensions[1].clone()],
        merge_unique_axes(&left.unreduced_axes, &right.unreduced_axes),
        merge_unique_axes(&left.reduced_manual_axes, &right.reduced_manual_axes),
        merge_unique_axes(&left.varying_manual_axes, &right.varying_manual_axes),
    )
    .map(|sharding| sharding.without_auto_axes())
    .ok()
}

/// Computes the abstract output type of one matrix multiplication.
///
/// This is the shared shape-and-sharding rule used by matrix multiplication across tracing,
/// transposition, and backend wrappers.
pub fn matmul_abstract(lhs: &ArrayType, rhs: &ArrayType, op: &'static str) -> Result<ArrayType, TypeError> {
    let (lhs_data_type, lhs_rows, lhs_cols) = matrix_parts(lhs, op)?;
    let (rhs_data_type, rhs_rows, rhs_cols) = matrix_parts(rhs, op)?;
    if lhs_data_type != rhs_data_type || lhs_cols != rhs_rows {
        return Err(TypeError { message: format!("{op} input matrix dimensions or element types are incompatible") });
    }
    let sharding = matmul_array_sharding(lhs, rhs);
    Ok(matrix_array_type(lhs_data_type, lhs_rows, rhs_cols, sharding))
}

/// Computes the abstract output type of one matrix transpose.
///
/// This centralizes the matrix-transpose metadata rule so both the core primitive and any backend
/// wrappers agree on how shapes and sharding should propagate.
pub fn transpose_abstract(input: &ArrayType, op: &'static str) -> Result<ArrayType, TypeError> {
    let (data_type, rows, cols) = matrix_parts(input, op)?;
    let sharding = transpose_array_sharding(input);
    Ok(matrix_array_type(data_type, cols, rows, sharding))
}
