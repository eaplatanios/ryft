//! Differentiation and batching rules for the sharding-control operations
//! ([`ReshardOperation`] and [`ShardingConstraintOperation`]). Both operations are linear, so their JVP applies the
//! same operation to the primal and tangent. They differ under transposition and batching, mirroring the
//! type-level contrast documented on [`crate::operations::sharding`]:
//!
//! - [`ReshardOperation`] is a tracked sharding transition: its transpose reshards the cotangent to the *cotangent
//!   dual* of the input's sharding (so the produced input cotangent is distributed like the input), and batching
//!   inserts the input-derived mapped-axis sharding at the new batch dimension.
//! - [`ShardingConstraintOperation`] is an untracked Auto-axes hint: its transpose is *self-adjoint* (it applies the
//!   same hint to the cotangent), and batching inserts a [`ShardingDimension::Unconstrained`] entry at the new batch
//!   dimension (the hinted axes are the ones the type system does not track, so the new dimension is left open for
//!   the backend to fill), matching JAX's `with_sharding_constraint` batcher.

use crate::batching::BatchingError;
use crate::differentiation::{Cotangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::sharding::{
    ConstrainSharding, Reshard, ReshardOperation, ShardingConstraintOperation, SupportsReshard,
    SupportsShardingConstraint,
};
use crate::operations::{InterpretableOperation, Operation};
use crate::programs::{ProgramError, Value};
use crate::sharding::ShardingDimension;
use crate::tracing::AbstractTracingContext;
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, apply_with_axes, batch_dimension_sharding};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
use crate::types::ArrayType;

/// Transpose rule for [`ReshardOperation`]: the cotangent of a reshard is itself a reshard of the output cotangent
/// to the cotangent dual of the *input*'s sharding (swapping its unreduced and reduced axes), so the produced input
/// cotangent is distributed like the input. An input that carries no sharding receives its cotangent unconstrained.
impl<V: Value<ArrayType> + Reshard, O> TransposableOperation<ArrayType, V, O> for ReshardOperation
where
    O: Operation<ArrayType> + SupportsReshard,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                let contribution = match input_types[0].sharding() {
                    Some(input_sharding) => cotangent.clone().reshard(&input_sharding.cotangent_dual()),
                    None => cotangent.clone(),
                };
                Ok(vec![Cotangent::Staged(contribution)])
            }
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

/// JVP rule for [`ReshardOperation`]. Resharding is linear, so the pushforward reshards both the primal and the
/// tangent to the same target sharding.
impl<D> DifferentiableOperation<D> for ReshardOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Reshard,
    D::Tangent: Reshard,
    LinearOperationOf<D>: SupportsReshard,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().clone().reshard(self.sharding());
        let tangent = inputs[0].tangent().clone().reshard(self.sharding());
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// Batching rule for [`ReshardOperation`]. The lifted reshard's target sharding gains the mapped axis's sharding
/// (derived from the batched inputs via [`batch_dimension_sharding`]) at the new batch dimension.
impl<V: Value<ArrayType>, C> BatchableOperation<V, C> for ReshardOperation
where
    ReshardOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, _) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let (lifted_sharding, output_axis) = match input_axes[0] {
            Some(batch_axis) => {
                let batch_dimension = batch_dimension_sharding(inputs)?;
                let lifted = self
                    .sharding()
                    .inserting_dimension(batch_axis, batch_dimension)
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
                (lifted, Some(batch_axis))
            }
            None => (self.sharding().clone(), None),
        };
        let lifted_op = ReshardOperation::new(lifted_sharding);
        apply_with_axes(&lifted_op, inputs, &[output_axis])
    }
}

/// Transpose rule for [`ShardingConstraintOperation`]: the operation is self-adjoint, so the cotangent of the output
/// is constrained by the *same* hint (mirroring JAX registering `with_sharding_constraint` with `ad.deflinear2`).
/// Unlike [`ReshardOperation`], the input's sharding is not consulted — the hint is the operation's own.
impl<V: Value<ArrayType> + ConstrainSharding, O> TransposableOperation<ArrayType, V, O> for ShardingConstraintOperation
where
    O: Operation<ArrayType> + SupportsShardingConstraint,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                Ok(vec![Cotangent::Staged(cotangent.clone().constrain_sharding(self.sharding()))])
            }
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

/// JVP rule for [`ShardingConstraintOperation`]. The hint is linear, so the pushforward applies the same hint to the
/// primal and the tangent.
impl<D> DifferentiableOperation<D> for ShardingConstraintOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: ConstrainSharding,
    D::Tangent: ConstrainSharding,
    LinearOperationOf<D>: SupportsShardingConstraint,
{
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().clone().constrain_sharding(self.sharding());
        let tangent = inputs[0].tangent().clone().constrain_sharding(self.sharding());
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// Batching rule for [`ShardingConstraintOperation`]. The lifted hint gains a [`ShardingDimension::Unconstrained`]
/// entry at the new batch dimension: the hint governs only the compiler-propagated auto axes, so the new dimension
/// is left open for the backend to fill rather than pinned to a derived or replicated entry (matching JAX's
/// `with_sharding_constraint` batcher, which inserts `PartitionSpec.UNCONSTRAINED`).
impl<V: Value<ArrayType>, C> BatchableOperation<V, C> for ShardingConstraintOperation
where
    ShardingConstraintOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(&self, _context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, _) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let (lifted_sharding, output_axis) = match input_axes[0] {
            Some(batch_axis) => {
                let lifted = self
                    .sharding()
                    .inserting_dimension(batch_axis, ShardingDimension::Unconstrained)
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
                (lifted, Some(batch_axis))
            }
            None => (self.sharding().clone(), None),
        };
        let lifted_op = ShardingConstraintOperation::new(lifted_sharding);
        apply_with_axes(&lifted_op, inputs, &[output_axis])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use crate::tests::{TestArray, TestArrayDomain};
    use crate::tracing::trace;
    use crate::tracing_v2::batching::BatchContext;
    use crate::tracing_v2::{ArrayOperation, LinearArrayOperation, LinearizationTracer};
    use crate::types::{DataType, Shape, Size};

    use super::*;

    fn mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    fn vector_type(size: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(size)]))
    }

    fn matrix_type(rows: usize, columns: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(rows), Size::Static(columns)]))
    }

    #[test]
    fn test_reshard_transposition_reshards_the_cotangent_to_the_input_dual() {
        let mesh = mesh();
        // The input is unreduced along the manual axis `m`, so its cotangent must be distributed like the input: the
        // dual sharding (reduced along `m`).
        let input_sharding =
            Sharding::with_unreduced_axes(mesh.clone(), vec![ShardingDimension::replicated()], ["m"]).unwrap();
        let input = TestArray::new(vector_type(8).with_sharding(input_sharding.clone()).unwrap(), vec![1.0; 8]);
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let (_output, pullback) = TestArrayDomain
            .vjp(
                {
                    let target = target.clone();
                    move |x: LinearizationTracer<'_, TestArrayDomain>| Ok(x.reshard(&target))
                },
                input,
            )
            .unwrap();

        let staged = pullback
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                LinearArrayOperation::Reshard { sharding } => Some(sharding.clone()),
                _ => None,
            })
            .expect("the pullback should stage a reshard transposition");
        assert_eq!(staged, input_sharding.cotangent_dual());
    }

    #[test]
    fn test_reshard_transposition_passes_through_an_unsharded_input() {
        let mesh = mesh();
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        // The input carries no sharding, so the reshard's cotangent flows back unconstrained — the pullback stages no
        // reshard transposition at all.
        let (_output, pullback) = TestArrayDomain
            .vjp(
                {
                    let target = target.clone();
                    move |x: LinearizationTracer<'_, TestArrayDomain>| Ok(x.reshard(&target))
                },
                TestArray::vector(vec![1.0; 8]),
            )
            .unwrap();
        assert!(
            !pullback
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), LinearArrayOperation::Reshard { .. })),
            "an unsharded input should not stage a reshard transposition",
        );
    }

    #[test]
    fn test_reshard_jvp_reshards_the_primal_and_the_tangent() {
        let mesh = mesh();
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        // Resharding is linear, so the pushforward stages a reshard for the tangent to the same target sharding.
        let (output, pushforward) = TestArrayDomain
            .linearize(
                {
                    let target = target.clone();
                    move |x: LinearizationTracer<'_, TestArrayDomain>| Ok(x.reshard(&target))
                },
                TestArray::vector(vec![2.0; 8]),
            )
            .unwrap();
        assert_eq!(output.values, vec![2.0; 8]);
        let staged = pushforward
            .program()
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                LinearArrayOperation::Reshard { sharding } => Some(sharding.clone()),
                _ => None,
            })
            .expect("the pushforward should stage a reshard for the tangent");
        assert_eq!(staged, target);
    }

    #[test]
    fn test_reshard_batching_lifts_the_target_sharding() {
        let mesh = mesh();
        // The lane reshards to a rank-1 sharding; batching over an unsharded input inserts a replicated entry at the
        // new lane axis, so the lifted reshard targets a rank-2 sharding.
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let expected_lifted = target.inserting_dimension(0, ShardingDimension::Replicated).unwrap();
        let (_output_type, program) = trace(
            &TestArrayDomain,
            |x| {
                let context = x.context().clone();
                let target = target.clone();
                Ok(BatchContext::batch(&context, move |lane| Ok(lane.reshard(&target)), x, Some(0), Some(0), None)
                    .unwrap())
            },
            matrix_type(2, 3),
        )
        .unwrap();
        let ArrayOperation::Reshard { sharding } = program.instructions()[0].operation() else {
            panic!("expected the batched program to stage a reshard operation");
        };
        assert_eq!(sharding, &expected_lifted);
    }

    #[test]
    fn test_sharding_constraint_transposition_is_self_adjoint() {
        let mesh = mesh();
        // The hint targets the auto axis `a`. The constraint is self-adjoint, so its transpose re-applies the same
        // hint to the cotangent rather than dualizing it.
        let hint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let (_output, pullback) = TestArrayDomain
            .vjp(
                {
                    let hint = hint.clone();
                    move |x: LinearizationTracer<'_, TestArrayDomain>| Ok(x.constrain_sharding(&hint))
                },
                TestArray::vector(vec![1.0; 8]),
            )
            .unwrap();
        let staged = pullback
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                LinearArrayOperation::ShardingConstraint { sharding } => Some(sharding.clone()),
                _ => None,
            })
            .expect("the pullback should stage a self-adjoint sharding-constraint transposition");
        assert_eq!(staged, hint);
    }

    #[test]
    fn test_sharding_constraint_batching_inserts_an_unconstrained_dimension() {
        let mesh = mesh();
        // The hint governs only the compiler-propagated auto axes, so batching leaves the new lane axis unconstrained
        // for the backend to fill rather than pinning it to a derived or replicated entry.
        let hint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let expected_lifted = hint.inserting_dimension(0, ShardingDimension::Unconstrained).unwrap();
        let (_output_type, program) = trace(
            &TestArrayDomain,
            |x| {
                let context = x.context().clone();
                let hint = hint.clone();
                Ok(BatchContext::batch(
                    &context,
                    move |lane| Ok(lane.constrain_sharding(&hint)),
                    x,
                    Some(0),
                    Some(0),
                    None,
                )
                .unwrap())
            },
            matrix_type(2, 3),
        )
        .unwrap();
        let ArrayOperation::ShardingConstraint { sharding } = program.instructions()[0].operation() else {
            panic!("expected the batched program to stage a sharding_constraint operation");
        };
        assert_eq!(sharding, &expected_lifted);
    }
}
