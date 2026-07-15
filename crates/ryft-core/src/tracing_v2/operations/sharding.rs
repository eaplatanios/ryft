//! Differentiation and batching rules for the sharding-control operations
//! ([`ReshardOperation`](crate::operations::sharding::ReshardOperation) and
//! [`ShardingConstraintOperation`](crate::operations::sharding::ShardingConstraintOperation)). Both operations are
//! linear, so their JVP applies the same operation to the primal and tangent. They differ under transposition and
//! batching, mirroring the type-level contrast documented on [`crate::operations::sharding`]:
//!
//! - [`ReshardOperation`](crate::operations::sharding::ReshardOperation) is a tracked sharding transition: its
//!   transpose reshards the cotangent to the *cotangent dual* of the input's sharding (so the produced input cotangent
//!   is distributed like the input), and batching inserts the input-derived mapped-axis sharding at the new batch
//!   dimension.
//! - [`ShardingConstraintOperation`](crate::operations::sharding::ShardingConstraintOperation) is an untracked
//!   Auto-axes hint: its transpose is *self-adjoint* (it applies the same hint to the cotangent), and batching inserts
//!   a [`ShardingDimension::Unconstrained`](crate::sharding::ShardingDimension::Unconstrained) entry at the new batch
//!   dimension (the hinted axes are the ones the type system does not track, so the new dimension is left open for
//!   the backend to fill), matching JAX's `with_sharding_constraint` batcher.

use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingError, InterpretableBatchableOperation};
use crate::contexts::Context;
use crate::differentiation::{DifferentiableOperation, DifferentiationError, TransposableOperation};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::sharding::{ConstrainSharding, Reshard, ReshardOperation, ShardingConstraintOperation};
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, Value};
use crate::sharding::ShardingDimension;
use crate::tracing::{Tracer, TracingContext};

use crate::batching::{BatchingContext, BatchingDriver};
use crate::differentiation::{DifferentiationDriver, DifferentiationDual, TranspositionDriver};
use crate::types::{ArrayType, Typed};

/// Transpose rule for [`ReshardOperation`]: the cotangent of a reshard is itself a reshard of the output cotangent
/// to the cotangent dual of the *input*'s sharding (swapping its unreduced and reduced axes), so the produced input
/// cotangent is distributed like the input. An input that carries no sharding receives its cotangent unconstrained.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for ReshardOperation
where
    O: Operation<ArrayType> + From<ReshardOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Value(cotangent) => {
                let contribution = match inputs[0].r#type().sharding() {
                    Some(input_sharding) => cotangent.reshard(&input_sharding.cotangent()),
                    None => cotangent.clone(),
                };
                Ok(vec![MaybeZero::Value(contribution)])
            }
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().into_owned())]),
        }
    }
}

/// Forward-mode rule for [`ReshardOperation`]: `reshard` is structural-linear, so the tangent is resharded by the
/// same target sharding as the primal. The shared all-zero fast path handles a zero operand tangent before this rule is
/// consulted, so the operand tangent reaching here is always live.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for ReshardOperation
where
    C::Operation: From<ReshardOperation>,
    C::Value: Reshard,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().reshard(self.sharding());
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.reshard(self.sharding())),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Batching rule for [`ReshardOperation`]. The lifted reshard's target sharding gains the mapped axis's sharding
/// (derived from the batched inputs via [`ArrayBatch::sharding_for_inputs`]) at the new batch dimension.
impl<C: Context<Type = ArrayType>> BatchableOperation<C> for ReshardOperation
where
    ReshardOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        // Validates that a mapped batch axis has a static size before lifting.
        ArrayBatch::common_batch_size(inputs)?;
        let (lifted_sharding, output_axis) = match inputs[0].batch_axis_position() {
            Some(batch_axis) => {
                let batch_dimension = ArrayBatch::sharding_for_inputs(inputs)?;
                let lifted = self
                    .sharding()
                    .with_inserted_dimension(batch_axis, batch_dimension)
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
                (lifted, Some(batch_axis))
            }
            None => (self.sharding().clone(), None),
        };
        let lifted_op = ReshardOperation::new(lifted_sharding);
        lifted_op.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_optional_position(output_axis)])
    }
}

/// Transpose rule for [`ShardingConstraintOperation`]: the operation is self-adjoint, so the cotangent of the output
/// is constrained by the *same* hint (mirroring JAX registering `with_sharding_constraint` with `ad.deflinear2`).
/// Unlike [`ReshardOperation`], the input's sharding is not consulted — the hint is the operation's own.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for ShardingConstraintOperation
where
    O: Operation<ArrayType> + From<ShardingConstraintOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match &outputs[0] {
            MaybeZero::Value(cotangent) => Ok(vec![MaybeZero::Value(cotangent.constrain_sharding(self.sharding()))]),
            MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().into_owned())]),
        }
    }
}

/// Forward-mode rule for [`ShardingConstraintOperation`]: the sharding hint is linear, so the same hint applies
/// to the operand tangent. The shared all-zero fast path handles a zero operand tangent before this rule is consulted,
/// so the operand tangent reaching here is always live.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for ShardingConstraintOperation
where
    C::Operation: From<ShardingConstraintOperation>,
    C::Value: ConstrainSharding,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().constrain_sharding(self.sharding());
        let tangent = match inputs[0].tangent() {
            MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().into_owned()),
            MaybeZero::Value(tangent) => MaybeZero::Value(tangent.constrain_sharding(self.sharding())),
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)])
    }
}

/// Batching rule for [`ShardingConstraintOperation`]. The lifted hint gains a [`ShardingDimension::Unconstrained`]
/// entry at the new batch dimension: the hint governs only the compiler-propagated auto axes, so the new dimension
/// is left open for the backend to fill rather than pinned to a derived or replicated entry (matching JAX's
/// `with_sharding_constraint` batcher, which inserts `PartitionSpec.UNCONSTRAINED`).
impl<C: Context<Type = ArrayType>> BatchableOperation<C> for ShardingConstraintOperation
where
    ShardingConstraintOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        // Validates that a mapped batch axis has a static size before lifting.
        ArrayBatch::common_batch_size(inputs)?;
        let (lifted_sharding, output_axis) = match inputs[0].batch_axis_position() {
            Some(batch_axis) => {
                let lifted = self
                    .sharding()
                    .with_inserted_dimension(batch_axis, ShardingDimension::Unconstrained)
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
                (lifted, Some(batch_axis))
            }
            None => (self.sharding().clone(), None),
        };
        let lifted_op = ShardingConstraintOperation::new(lifted_sharding);
        lifted_op.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_optional_position(output_axis)])
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::batching::{Batch, BatchAxis};
    use crate::contexts::EagerContext;
    use crate::differentiation::{LinearizationTracer, ReverseModeDifferentiate};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{DataType, Shape, Size};

    use crate::tracing::Trace;

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

        let (_output, pullback) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                {
                    let target = target.clone();
                    move |x: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                        Ok(x.reshard(&target))
                    }
                },
                input,
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();

        let staged = pullback
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                ArrayOperation::Reshard(operation) => Some(operation.sharding().clone()),
                _ => None,
            })
            .expect("the pullback should stage a reshard transposition");
        assert_eq!(staged, input_sharding.cotangent());
    }

    #[test]
    fn test_reshard_transposition_passes_through_an_unsharded_input() {
        let mesh = mesh();
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        // The input carries no sharding, so the reshard's cotangent flows back unconstrained — the pullback stages no
        // reshard transposition at all.
        let (_output, pullback) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                {
                    let target = target.clone();
                    move |x: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                        Ok(x.reshard(&target))
                    }
                },
                TestArray::vector(vec![1.0; 8]),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        assert!(
            !pullback
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayOperation::Reshard(_))),
            "an unsharded input should not stage a reshard transposition",
        );
    }

    #[test]
    fn test_reshard_batching_lifts_the_target_sharding() {
        let mesh = mesh();
        // The batch item reshards to a rank-1 sharding; batching over an unsharded input inserts a replicated entry at
        // the new batch axis, so the lifted reshard targets a rank-2 sharding.
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let expected_lifted = target.with_inserted_dimension(0, ShardingDimension::Replicated).unwrap();
        let (_output_type, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |x| {
                let context = x.context().clone();
                let target = target.clone();
                Ok(Batch::batch(
                    &context,
                    move |item| Ok(item.reshard(&target)),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap())
            },
            matrix_type(2, 3),
        )
        .unwrap();
        let ArrayOperation::Reshard(operation) = program.instructions()[0].operation() else {
            panic!("expected the batched program to stage a reshard operation");
        };
        assert_eq!(operation.sharding(), &expected_lifted);
    }

    #[test]
    fn test_sharding_constraint_transposition_is_self_adjoint() {
        let mesh = mesh();
        // The hint targets the auto axis `a`. The constraint is self-adjoint, so its transpose re-applies the same
        // hint to the cotangent rather than dualizing it.
        let hint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let (_output, pullback) = EagerContext::<TestArray, ArrayOperation<TestArray>>::new()
            .vjp(
                {
                    let hint = hint.clone();
                    move |x: LinearizationTracer<EagerContext<TestArray, ArrayOperation<TestArray>>>| {
                        Ok(x.constrain_sharding(&hint))
                    }
                },
                TestArray::vector(vec![1.0; 8]),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_parts();
        let staged = pullback
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                ArrayOperation::ShardingConstraint(operation) => Some(operation.sharding().clone()),
                _ => None,
            })
            .expect("the pullback should stage a self-adjoint sharding-constraint transposition");
        assert_eq!(staged, hint);
    }

    #[test]
    fn test_sharding_constraint_batching_inserts_an_unconstrained_dimension() {
        let mesh = mesh();
        // The hint governs only the compiler-propagated auto axes, so batching leaves the new batch axis unconstrained
        // for the backend to fill rather than pinning it to a derived or replicated entry.
        let hint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let expected_lifted = hint.with_inserted_dimension(0, ShardingDimension::Unconstrained).unwrap();
        let (_output_type, program) = EagerContext::<TestArray, ArrayOperation<TestArray>>::trace(
            |x| {
                let context = x.context().clone();
                let hint = hint.clone();
                Ok(Batch::batch(
                    &context,
                    move |item| Ok(item.constrain_sharding(&hint)),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )
                .unwrap())
            },
            matrix_type(2, 3),
        )
        .unwrap();
        let ArrayOperation::ShardingConstraint(operation) = program.instructions()[0].operation() else {
            panic!("expected the batched program to stage a sharding_constraint operation");
        };
        assert_eq!(operation.sharding(), &expected_lifted);
    }
}
