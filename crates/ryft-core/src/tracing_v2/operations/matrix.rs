use std::collections::BTreeSet;

use crate::sharding::{Sharding, ShardingDimension};
use crate::types::{ArrayType, DataType, Shape, Size, TypeError};

use super::dot::{Dot, DotDimensionNumbers};
use super::transpose::Transpose;

/// Generalized N-D dot and transpose capability.
///
/// This convenience trait groups the value-level [`Dot`] and [`Transpose`] operations used by the unified
/// [`DotOperation`](super::dot::DotOperation) and [`TransposeOperation`](super::transpose::TransposeOperation)
/// primitives.
pub trait DotOps: Dot + Transpose {}

impl<T: Dot + Transpose> DotOps for T {}

fn matrix_array_type(data_type: DataType, rows: usize, cols: usize, sharding: Option<Sharding>) -> ArrayType {
    ArrayType::new(data_type, Shape::new(vec![Size::Static(rows), Size::Static(cols)]))
        .with_sharding(sharding)
        .expect("matrix abstract evaluation should preserve rank-2 sharding")
}

fn matrix_parts(r#type: &ArrayType, op: &'static str) -> Result<(DataType, usize, usize), TypeError> {
    let is_supported_data_type =
        matches!(r#type.data_type(), DataType::BF16 | DataType::F16 | DataType::F32 | DataType::F64);
    if !is_supported_data_type || r#type.rank() != 2 {
        return Err(TypeError { message: format!("{op} expects rank-2 floating-point matrix inputs") });
    }

    let Size::Static(rows) = r#type.dimension(0) else {
        return Err(TypeError { message: format!("{op} requires statically shaped matrix inputs") });
    };
    let Size::Static(cols) = r#type.dimension(1) else {
        return Err(TypeError { message: format!("{op} requires statically shaped matrix inputs") });
    };
    Ok((r#type.data_type(), rows, cols))
}

fn is_replicated_sharding(sharding: &Sharding) -> bool {
    sharding.dimensions().iter().all(|dimension| matches!(dimension, ShardingDimension::Replicated))
}

fn merge_unique_axes(left: &BTreeSet<String>, right: &BTreeSet<String>) -> BTreeSet<String> {
    left.union(right).cloned().collect()
}

fn transpose_array_sharding(input: &ArrayType) -> Option<Sharding> {
    let sharding = input.sharding()?.clone();
    if sharding.rank() != 2 {
        return None;
    }
    Sharding::with_manual_axes(
        sharding.mesh().clone(),
        vec![sharding.dimensions()[1].clone(), sharding.dimensions()[0].clone()],
        sharding.unreduced_axes().clone(),
        sharding.reduced_manual_axes().clone(),
        sharding.varying_manual_axes().clone(),
    )
    .map(|sharding| sharding.without_auto_axes())
    .ok()
}

fn matmul_array_sharding(lhs: &ArrayType, rhs: &ArrayType) -> Option<Sharding> {
    let left = lhs.sharding()?.clone();
    let right = rhs.sharding()?.clone();
    if left == right && is_replicated_sharding(&left) {
        return Some(left);
    }
    if left.rank() != 2 || right.rank() != 2 {
        return None;
    }
    if !matches!(left.dimensions()[1], ShardingDimension::Replicated)
        || !matches!(right.dimensions()[0], ShardingDimension::Replicated)
    {
        return None;
    }
    Sharding::with_manual_axes(
        left.mesh().clone(),
        vec![left.dimensions()[0].clone(), right.dimensions()[1].clone()],
        merge_unique_axes(left.unreduced_axes(), right.unreduced_axes()),
        merge_unique_axes(left.reduced_manual_axes(), right.reduced_manual_axes()),
        merge_unique_axes(left.varying_manual_axes(), right.varying_manual_axes()),
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

/// Computes the abstract output type of one generalized dot product.
///
/// The result shape is `[batching..., lhs_result..., rhs_result...]`, where the result
/// dimensions are the operand axes that are neither batching nor contracting, in their original
/// order. The output element type is the LHS element type (after a compatibility check with the
/// RHS element type). Sharding metadata is dropped when the input ranks differ from 2 because
/// the legacy 2-D sharding rule does not generalize to arbitrary contractions; the result keeps
/// the legacy rank-2 sharding only when both operands match the 2-D matmul case.
pub fn dot_abstract(
    lhs: &ArrayType,
    rhs: &ArrayType,
    dimensions: &DotDimensionNumbers,
    op: &'static str,
) -> Result<ArrayType, TypeError> {
    if lhs.data_type() != rhs.data_type() {
        return Err(TypeError { message: format!("{op} input element types are incompatible") });
    }
    let lhs_rank = lhs.rank();
    let rhs_rank = rhs.rank();
    let lhs_batching = dimensions.lhs_batching_dimensions();
    let rhs_batching = dimensions.rhs_batching_dimensions();
    let lhs_contracting = dimensions.lhs_contracting_dimensions();
    let rhs_contracting = dimensions.rhs_contracting_dimensions();

    if lhs_batching.len() != rhs_batching.len() {
        return Err(TypeError {
            message: format!("{op} batching dimensions have different lengths on the two operands"),
        });
    }
    if lhs_contracting.len() != rhs_contracting.len() {
        return Err(TypeError {
            message: format!("{op} contracting dimensions have different lengths on the two operands"),
        });
    }
    if lhs_batching.iter().any(|axis| *axis >= lhs_rank) || lhs_contracting.iter().any(|axis| *axis >= lhs_rank) {
        return Err(TypeError { message: format!("{op} LHS dimension index out of bounds") });
    }
    if rhs_batching.iter().any(|axis| *axis >= rhs_rank) || rhs_contracting.iter().any(|axis| *axis >= rhs_rank) {
        return Err(TypeError { message: format!("{op} RHS dimension index out of bounds") });
    }

    for (lhs_axis, rhs_axis) in lhs_batching.iter().zip(rhs_batching.iter()) {
        if lhs.dimension(*lhs_axis as isize) != rhs.dimension(*rhs_axis as isize) {
            return Err(TypeError {
                message: format!(
                    "{op} batching dimension sizes do not match (LHS axis {lhs_axis}, RHS axis {rhs_axis})"
                ),
            });
        }
    }
    for (lhs_axis, rhs_axis) in lhs_contracting.iter().zip(rhs_contracting.iter()) {
        if lhs.dimension(*lhs_axis as isize) != rhs.dimension(*rhs_axis as isize) {
            return Err(TypeError {
                message: format!(
                    "{op} contracting dimension sizes do not match (LHS axis {lhs_axis}, RHS axis {rhs_axis})"
                ),
            });
        }
    }

    let lhs_result: Vec<usize> = (0..lhs_rank)
        .filter(|axis| !lhs_batching.contains(axis) && !lhs_contracting.contains(axis))
        .collect();
    let rhs_result: Vec<usize> = (0..rhs_rank)
        .filter(|axis| !rhs_batching.contains(axis) && !rhs_contracting.contains(axis))
        .collect();

    let output_dimensions: Vec<Size> = lhs_batching
        .iter()
        .map(|axis| lhs.dimension(*axis as isize))
        .chain(lhs_result.iter().map(|axis| lhs.dimension(*axis as isize)))
        .chain(rhs_result.iter().map(|axis| rhs.dimension(*axis as isize)))
        .collect();

    let sharding =
        if is_legacy_matmul_layout(lhs_batching, rhs_batching, lhs_contracting, rhs_contracting, lhs_rank, rhs_rank) {
            matmul_array_sharding(lhs, rhs)
        } else {
            None
        };

    ArrayType::new(lhs.data_type(), Shape::new(output_dimensions))
        .with_sharding(sharding)
        .map_err(|error| TypeError { message: error.to_string() })
}

fn is_legacy_matmul_layout(
    lhs_batching: &[usize],
    rhs_batching: &[usize],
    lhs_contracting: &[usize],
    rhs_contracting: &[usize],
    lhs_rank: usize,
    rhs_rank: usize,
) -> bool {
    lhs_batching.is_empty()
        && rhs_batching.is_empty()
        && lhs_rank == 2
        && rhs_rank == 2
        && lhs_contracting == [1]
        && rhs_contracting == [0]
}
