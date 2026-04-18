//! Concrete staged operations for [`crate::tracing_v2`].

use std::collections::BTreeSet;

use crate::sharding::Sharding;
use crate::tracing_v2::{TraceError, Traceable, batch::Batch, jit::JitTracer};
use crate::types::ArrayType;

fn is_replicated_sharding(sharding: &Sharding) -> bool {
    sharding
        .dimensions
        .iter()
        .all(|dimension| matches!(dimension, crate::sharding::ShardingDimension::Replicated))
}

fn merge_unique_axes(left: &BTreeSet<String>, right: &BTreeSet<String>) -> BTreeSet<String> {
    left.union(right).cloned().collect()
}

fn merge_sharding_state(base: &Sharding, other: &Sharding) -> Sharding {
    let mut sharding = base.clone();
    sharding.unreduced_axes = merge_unique_axes(&base.unreduced_axes, &other.unreduced_axes);
    sharding.reduced_manual_axes = merge_unique_axes(&base.reduced_manual_axes, &other.reduced_manual_axes);
    sharding.varying_manual_axes = merge_unique_axes(&base.varying_manual_axes, &other.varying_manual_axes);
    sharding
}

fn binary_output_sharding(inputs: &[ArrayType]) -> Option<Sharding> {
    match (&inputs[0].sharding, &inputs[1].sharding) {
        (Some(left), Some(right))
            if left.mesh == right.mesh
                && left.dimensions == right.dimensions
                && left.unreduced_axes == right.unreduced_axes
                && left.reduced_manual_axes == right.reduced_manual_axes =>
        {
            Some(merge_sharding_state(left, right))
        }
        (Some(left), Some(right)) if is_replicated_sharding(left) => Some(merge_sharding_state(right, left)),
        (Some(left), Some(right)) if is_replicated_sharding(right) => Some(merge_sharding_state(left, right)),
        (Some(left), None) => Some(left.clone()),
        (None, Some(right)) => Some(right.clone()),
        _ => None,
    }
}

/// Elementwise addition.
pub mod add;

/// Elementwise cosine.
pub mod cos;

/// Linear left matrix multiplication.
pub mod left_matmul;

/// Matrix capability layer shared by matrix staged operations.
pub mod matrix;

/// Matrix multiplication.
pub mod matmul;

/// Matrix transposition.
pub mod matrix_transpose;

/// Elementwise multiplication.
pub mod mul;

/// Elementwise negation.
pub mod neg;

/// Reshaping primitive.
pub mod reshape;

/// Linear right matrix multiplication.
pub mod right_matmul;

/// Scalar and tensor scaling.
pub mod scale;

/// Elementwise sine.
pub mod sin;

/// Traced rematerialization boundary.
pub mod rematerialize;

/// Traced `vmap` operations.
pub mod vmap;

pub use add::AddOp;
pub use cos::{Cos, CosOp};
pub use left_matmul::LeftMatMulOp;
pub use matmul::MatMulOp;
pub use matrix_transpose::MatrixTransposeOp;
pub use mul::MulOp;
pub use neg::NegOp;
pub use rematerialize::{FlatTracedRematerialize, LinearRematerializeOp, RematerializeOp};
pub use reshape::ReshapeOp;
pub use right_matmul::RightMatMulOp;
pub use scale::ScaleOp;
pub use sin::{Sin, SinOp};
pub use vmap::{FlatTracedVMap, LinearVMapOp, VMapOp};

/// Returns an input-count error when one staged op receives the wrong arity.
pub fn expect_input_count(inputs: usize, expected: usize) -> Result<(), TraceError> {
    if inputs == expected { Ok(()) } else { Err(TraceError::InvalidInputCount { expected, got: inputs }) }
}

/// Returns a batch-size error when two batched inputs disagree on their lane count.
pub fn expect_batch_sizes_match<V>(left: &Batch<V>, right: &Batch<V>) -> Result<(), TraceError> {
    if left.len() == right.len() { Ok(()) } else { Err(TraceError::MismatchedBatchSize) }
}

/// Lifts one concrete value into the staged graph owned by a JIT tracer.
pub fn lift_jit_constant<V: Traceable<ArrayType>, O: Clone + 'static, L: Clone + 'static>(
    constant: &V,
    exemplar: &JitTracer<ArrayType, V, O, L>,
) -> JitTracer<ArrayType, V, O, L> {
    let builder = exemplar.builder_handle();
    let atom = builder.borrow_mut().add_constant(constant.clone());
    JitTracer::from_staged_parts(atom, builder, exemplar.staging_error_handle(), exemplar.engine())
}

/// Propagates one unary input type through a shape-preserving staged op.
pub fn unary_abstract(inputs: &[ArrayType]) -> Result<ArrayType, TraceError> {
    expect_input_count(inputs.len(), 1)?;
    Ok(inputs[0].clone())
}

/// Propagates one binary input type through a shape-preserving staged op.
pub fn binary_same_abstract(op: &'static str, inputs: &[ArrayType]) -> Result<ArrayType, TraceError> {
    expect_input_count(inputs.len(), 2)?;
    if inputs[0].data_type != inputs[1].data_type || inputs[0].shape != inputs[1].shape {
        Err(TraceError::IncompatibleAbstractValues { op })
    } else {
        let sharding = binary_output_sharding(inputs);
        ArrayType::new(
            inputs[0].data_type,
            inputs[0].shape.clone(),
            if inputs[0].layout == inputs[1].layout { inputs[0].layout.clone() } else { None },
            sharding,
        )
        .map_err(|_| TraceError::InternalInvariantViolation("binary output sharding should match operand rank"))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{DataType, Shape, Size};

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn test_binary_same_abstract_unions_varying_axes() {
        let mesh = test_mesh();
        let left = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let right = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["y"],
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            binary_same_abstract("add", &[left, right]).map(|output| output.sharding.unwrap().varying_manual_axes),
            Ok(BTreeSet::from(["x".to_string(), "y".to_string()]))
        );
    }

    #[test]
    fn test_binary_same_abstract_preserves_reduced_axes_from_replicated_input() {
        let mesh = test_mesh();
        let left = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()),
        )
        .unwrap();
        let right = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::replicated()],
                    Vec::<&str>::new(),
                    ["y"],
                    Vec::<&str>::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            binary_same_abstract("add", &[left, right]).map(|output| output.sharding.unwrap().reduced_manual_axes),
            Ok(BTreeSet::from(["y".to_string()]))
        );
    }
}
