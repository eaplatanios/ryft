//! Sharding-control operations: [`ReshardOperation`] and [`ShardingConstraintOperation`].
//!
//! Both operations are unary, leave the array value and its shape/data type untouched, and carry a target
//! [`Sharding`](crate::arrays::Sharding). They differ in *how that sharding relates to the type system* and *which mesh
//! axis types they govern* — a distinction mirroring JAX's split between
//! [`jax.sharding.reshard`](https://docs.jax.dev/en/latest/jax.sharding.html) and
//! [`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html):
//!
//! - [`ReshardOperation`] performs a **type-level sharding transition** over
//!   [`Explicit`](crate::arrays::MeshAxisType::Explicit) and [`Manual`](crate::arrays::MeshAxisType::Manual)
//!   mesh axes. Type inference *replaces* the output's [`Sharding`](crate::arrays::Sharding) with the requested one, so
//!   the new sharding is tracked by the type system, dualized under transposition (the cotangent is resharded to the
//!   cotangent dual of the input's sharding), and validated against the operand. It rejects requests that name
//!   [`Auto`](crate::arrays::MeshAxisType::Auto) axes (those are the compiler's to place — use a
//!   [`ShardingConstraintOperation`] instead).
//!
//! - [`ShardingConstraintOperation`] is an **untracked propagation hint** over
//!   [`Auto`](crate::arrays::MeshAxisType::Auto) mesh axes. Type inference is the *identity* (the output type — its
//!   sharding included — equals the input type), so the hint never becomes type-level state; it is self-adjoint under
//!   transposition (the same hint is applied to the cotangent) and is materialized only at lowering, where it steers
//!   the compiler's (e.g. [GSPMD](https://arxiv.org/abs/2105.04663) / [Shardy](https://openxla.org/shardy))
//!   sharding propagation. It rejects requests whose [`Sharded`](crate::arrays::ShardingDimension::Sharded) entries
//!   name non-[`Auto`](crate::arrays::MeshAxisType::Auto) axes (use a [`ReshardOperation`] for those).
//!
//! Both lower to the same backend sharding-constraint operation (the [`Shardy`](https://openxla.org/shardy)
//! `sdy.sharding_constraint` in the XLA backend); the only difference at the boundary is whether the type system
//! tracked the result. The operations themselves are backend-agnostic — they carry a
//! [`Sharding`](crate::arrays::Sharding) and have purely type-level and autodiff semantics — and each backend decides
//! how to lower them.

// TODO(eaplatanios): Review this module.

use std::fmt::Display;

use crate::arrays::{ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayType, Sharding, ShardingDimension};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
};

/// Canonical operation name for [`ReshardOperation`].
pub const RESHARD_OPERATION_NAME: &str = "reshard";

/// Canonical operation name for [`ShardingConstraintOperation`].
pub const SHARDING_CONSTRAINT_OPERATION_NAME: &str = "sharding_constraint";

/// Returns the mesh-axis names referenced by `sharding` — those that shard a ranked dimension plus those in its
/// unreduced and reduced sets. Used by both operations to validate the requested sharding against the mesh-axis
/// types they govern.
fn referenced_axes(sharding: &Sharding) -> impl Iterator<Item = &str> {
    sharding
        .dimensions()
        .iter()
        .filter_map(|dimension| match dimension {
            ShardingDimension::Sharded(axis_names) => Some(axis_names.iter()),
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => None,
        })
        .flatten()
        .chain(sharding.unreduced_axes())
        .chain(sharding.reduced_axes())
        .map(String::as_str)
}

/// Unary [`Operation`] that performs a type-level sharding transition, the analogue of JAX's
/// [`jax.sharding.reshard`](https://docs.jax.dev/en/latest/jax.sharding.html). It leaves the array value, shape, and
/// data type unchanged and *replaces* the output's [`Sharding`] with the requested one.
///
/// # Differs from [`ShardingConstraintOperation`]
///
/// [`ReshardOperation`] is a *tracked* sharding transition over [`Explicit`](crate::arrays::MeshAxisType::Explicit)
/// and [`Manual`](crate::arrays::MeshAxisType::Manual) axes: the requested sharding becomes the output type's
/// sharding and is dualized under transposition. [`ShardingConstraintOperation`] is instead an *untracked* hint over
/// [`Auto`](crate::arrays::MeshAxisType::Auto) axes whose type inference is the identity. Refer to the
/// [module documentation](self) for the full contrast.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshardOperation {
    /// Target [`Sharding`](crate::arrays::Sharding) the input is resharded to.
    sharding: Sharding,
}

impl ReshardOperation {
    /// Creates a new [`ReshardOperation`] resharding its input to `sharding`.
    #[inline]
    pub fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    /// Returns the target [`Sharding`].
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        &self.sharding
    }
}

impl Display for ReshardOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReshardOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        RESHARD_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let input = &input_types[0];
        if input.rank() != self.sharding.rank() {
            return Err(TypeError::invalid(format!(
                "{RESHARD_OPERATION_NAME} target sharding rank ({}) does not match the input rank ({})",
                self.sharding.rank(),
                input.rank(),
            )));
        }
        if referenced_axes(&self.sharding)
            .any(|axis| self.sharding.mesh().axis_type(axis) == Some(crate::arrays::MeshAxisType::Auto))
        {
            return Err(TypeError::invalid(format!(
                "{RESHARD_OPERATION_NAME} cannot target auto mesh axes; use a sharding constraint to hint \
                     propagation over auto axes"
            )));
        }
        // The resharded value still varies across whatever manual axes the input varied across; that fact is
        // orthogonal to the requested placement, so it is carried over rather than taken from the target.
        let varying_manual_axes =
            input.sharding().map(|sharding| sharding.varying_manual_axes().clone()).unwrap_or_default();
        let sharding = self
            .sharding
            .clone()
            .with_varying_manual_axes(varying_manual_axes)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        Ok(vec![
            ArrayType::new(input.data_type(), input.shape().clone())
                .with_layout(input.layout().cloned())
                .with_memory(input.memory())
                .with_sharding(sharding)
                .map_err(|error| TypeError::invalid(error.to_string()))?,
        ])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("sharding", &self.sharding))
    }
}

/// Value-level resharding capability, the receiver-style entry point for staging or executing [`ReshardOperation`].
///
/// The provided default returns the value unchanged, which is correct for concrete (single-device) values, for which
/// a sharding only describes distribution metadata. Staging values override it to stage the operation, which keeps
/// transforms that apply operations through interpretation (e.g. program batching and re-tracing) from silently
/// dropping the resharding.
pub trait Reshard: Clone {
    /// Reshards `self` to `sharding`.
    fn reshard(&self, sharding: &Sharding) -> Self {
        let _ = sharding;
        self.clone()
    }
}

/// Any context-carrying value reshards by binding a [`ReshardOperation`] through its own context. The
/// `From<ReshardOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Reshard for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ReshardOperation>,
{
    fn reshard(&self, sharding: &Sharding) -> Self {
        self.dispatch_domain()
            .bind(ReshardOperation::new(sharding.clone()), Vec::new(), std::slice::from_ref(self))
            .expect("`reshard` operation failed")
            .remove(0)
    }
}

impl<C: Domain<Type = ArrayType, Value: Reshard>> InterpretableOperation<C> for ReshardOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // The resharding flows through the capability so interpretation over staging values (program batching,
        // re-tracing) preserves it; concrete values pass through unchanged.
        Ok(vec![inputs[0].reshard(&self.sharding)])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for ReshardOperation where C::Operation: From<ReshardOperation> {}

impl_differentiable_operation! {
    ReshardOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ReshardOperation>,
        C::Value: Reshard,
    {
        |operation, _context, _driver, inputs| {
            // Forward-mode rule for [`ReshardOperation`]: `reshard` is structural-linear, so the tangent is resharded by
            // the same target sharding as the primal. The shared all-zero fast path handles a zero operand tangent
            // before this rule is consulted, so the operand tangent reaching here is always live.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().reshard(operation.sharding());
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.reshard(operation.sharding())),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<BroadcastOperation> + From<ReshardOperation>,
    {
        |_operation, _context, _driver, inputs, outputs| {
            // Transpose rule for [`ReshardOperation`]: the cotangent of a reshard is itself a reshard of the output
            // cotangent to the cotangent dual of the *input*'s sharding (swapping its unreduced and reduced axes), so
            // the produced input cotangent is distributed like the input. An input that carries no sharding receives
            // an exactly unsharded cotangent through an identity-axis broadcast.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            let input_cotangent_type = inputs[0].r#type().cotangent();
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let contribution = match input_cotangent_type.sharding() {
                        Some(input_cotangent_sharding) => cotangent.reshard(input_cotangent_sharding),
                        None => cotangent.broadcast(
                            input_cotangent_type.clone(),
                            &(0..input_cotangent_type.shape().rank()).collect::<Vec<_>>(),
                        )?,
                    };
                    Ok(vec![MaybeZero::Value(contribution)])
                }
                MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(input_cotangent_type)]),
            }
        }
    },
}

/// Batching rule for [`ReshardOperation`]. The lifted reshard's target sharding gains the mapped axis's sharding
/// (derived from the batched inputs via [`ArrayBatch::sharding_for_inputs`]) at the new batch dimension.
impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for ReshardOperation
where
    ReshardOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        // Validates that a mapped batch axis has a static size before lifting.
        ArrayBatch::common_batch_size(inputs)?;
        let (lifted_sharding, output_axis) = match inputs[0].batch_axis_position() {
            Some(batch_axis) => {
                let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
                let lifted = self
                    .sharding()
                    .with_inserted_dimension(batch_axis, axis_sharding)
                    .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() })?;
                (lifted, Some(batch_axis))
            }
            None => (self.sharding().clone(), None),
        };
        let lifted_op = ReshardOperation::new(lifted_sharding);
        Ok(lifted_op
            .interpret_with_batch_axes(context, inputs, &[BatchAxis::from_optional_position(output_axis)])?
            .into())
    }
}

/// Unary [`Operation`] that records a sharding-propagation hint, the analogue of JAX's
/// [`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html).
/// It leaves the array value, type, and sharding untouched at the type level and only steers the backend compiler's
/// sharding propagation over [`Auto`](crate::arrays::MeshAxisType::Auto) mesh axes at lowering time.
///
/// # Differs from [`ReshardOperation`]
///
/// [`ShardingConstraintOperation`]'s type inference is the *identity* (the output type, sharding included, equals the
/// input type), so the hint is never tracked as type-level state; it is self-adjoint under transposition (the same
/// hint applies to the cotangent). [`ReshardOperation`] instead performs a *tracked* sharding transition over
/// Explicit/Manual axes. Refer to the [module documentation](self) for the full contrast.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShardingConstraintOperation {
    /// Sharding hint to record for the backend's propagation over auto mesh axes.
    sharding: Sharding,
}

impl ShardingConstraintOperation {
    /// Creates a new [`ShardingConstraintOperation`] hinting `sharding`.
    #[inline]
    pub fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    /// Returns the sharding hint.
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        &self.sharding
    }
}

impl Display for ShardingConstraintOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ShardingConstraintOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        SHARDING_CONSTRAINT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let input = &input_types[0];
        if input.rank() != self.sharding.rank() {
            return Err(TypeError::invalid(format!(
                "{SHARDING_CONSTRAINT_OPERATION_NAME} hint rank ({}) does not match the input rank ({})",
                self.sharding.rank(),
                input.rank(),
            )));
        }
        // The hint may only place ranked dimensions over auto axes (the axes the compiler propagates); naming an
        // explicit or manual axis is the inverse error of `reshard`'s auto-axis rejection.
        for dimension in self.sharding.dimensions() {
            if let ShardingDimension::Sharded(axis_names) = dimension {
                for axis_name in axis_names {
                    if self.sharding.mesh().axis_type(axis_name) != Some(crate::arrays::MeshAxisType::Auto) {
                        return Err(TypeError::invalid(format!(
                            "{SHARDING_CONSTRAINT_OPERATION_NAME} can only hint placement over auto mesh axes, \
                                 but `{axis_name}` is not auto; use reshard for explicit or manual axes"
                        )));
                    }
                }
            }
        }
        // The hint is untracked: the output type (sharding included) is identical to the input. The requested
        // sharding only takes effect when the backend lowers the operation.
        Ok(vec![input.clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("sharding", &self.sharding))
    }
}

/// Value-level sharding-constraint capability, the receiver-style entry point for staging or executing
/// [`ShardingConstraintOperation`]. The provided default returns the value unchanged (the hint is type-level
/// metadata, meaningful only at lowering); staging values override it to stage the operation so interpretation-driven
/// transforms do not drop the hint.
pub trait ConstrainSharding: Clone {
    /// Records `sharding` as a propagation hint on `self`.
    fn constrain_sharding(&self, sharding: &Sharding) -> Self {
        let _ = sharding;
        self.clone()
    }
}

/// Any context-carrying value constrains its sharding by binding a [`ShardingConstraintOperation`] through its own
/// context. The `From<ShardingConstraintOperation>` bound makes this disjoint from the eager value types (whose
/// context operation is `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete
/// implementations.
impl<V: Value<Type = ArrayType>> ConstrainSharding for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ShardingConstraintOperation>,
{
    fn constrain_sharding(&self, sharding: &Sharding) -> Self {
        self.dispatch_domain()
            .bind(ShardingConstraintOperation::new(sharding.clone()), Vec::new(), std::slice::from_ref(self))
            .expect("`constrain_sharding` operation failed")
            .remove(0)
    }
}

impl<C: Domain<Type = ArrayType, Value: ConstrainSharding>> InterpretableOperation<C> for ShardingConstraintOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // The hint flows through the capability so interpretation over staging values preserves it; concrete values
        // pass through unchanged.
        Ok(vec![inputs[0].constrain_sharding(&self.sharding)])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for ShardingConstraintOperation where
    C::Operation: From<ShardingConstraintOperation>
{
}

impl_differentiable_operation! {
    ShardingConstraintOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ShardingConstraintOperation>,
        C::Value: ConstrainSharding,
    {
        |operation, _context, _driver, inputs| {
            // Forward-mode rule for [`ShardingConstraintOperation`]: the sharding hint is linear, so the same hint
            // applies to the operand tangent. The shared all-zero fast path handles a zero operand tangent before this
            // rule is consulted, so the operand tangent reaching here is always live.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().constrain_sharding(operation.sharding());
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.constrain_sharding(operation.sharding())),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<ShardingConstraintOperation>,
    {
        |operation, _context, _driver, inputs, outputs| {
            // Transpose rule for [`ShardingConstraintOperation`]: the operation is self-adjoint, so the cotangent of
            // the output is constrained by the *same* hint (mirroring JAX registering `with_sharding_constraint` with
            // `ad.deflinear2`). Unlike [`ReshardOperation`], the input's sharding is not consulted — the hint is the
            // operation's own.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    Ok(vec![MaybeZero::Value(cotangent.constrain_sharding(operation.sharding()))])
                }
                MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
            }
        }
    },
}

/// Batching rule for [`ShardingConstraintOperation`]. The lifted hint gains a [`ShardingDimension::Unconstrained`]
/// entry at the new batch dimension: the hint governs only the compiler-propagated auto axes, so the new dimension
/// is left open for the backend to fill rather than pinned to a derived or replicated entry (matching JAX's
/// `with_sharding_constraint` batcher, which inserts `PartitionSpec.UNCONSTRAINED`).
impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for ShardingConstraintOperation
where
    ShardingConstraintOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
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
        Ok(lifted_op
            .interpret_with_batch_axes(context, inputs, &[BatchAxis::from_optional_position(output_axis)])?
            .into())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayOperation, DataType, Dimension, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
        ShardingDimension,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::tracing::Trace;

    use super::*;

    fn explicit_manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    fn vector_type(size: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(size)]))
    }

    #[test]
    fn test_reshard_replaces_the_output_sharding() {
        let mesh = explicit_manual_mesh();
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let operation = ReshardOperation::new(target.clone());

        assert_eq!(operation.name(), RESHARD_OPERATION_NAME);
        assert_eq!(operation.to_string(), format!("reshard [sharding={target}]"));

        // The output keeps the value/shape but adopts the target sharding; an input carrying varying-manual axes
        // carries them over.
        let input = vector_type(8)
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes(["m"])
                    .unwrap(),
            )
            .unwrap();
        let expected = vector_type(8)
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["m"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input), &[]), Ok(vec![expected]));
    }

    #[test]
    fn test_reshard_rejects_auto_axes_and_rank_mismatch() {
        let mesh = explicit_manual_mesh();
        let auto_target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        assert_eq!(
            ReshardOperation::new(auto_target).infer_output_types(&[vector_type(8)], &[]),
            Err(TypeError::invalid(
                "reshard cannot target auto mesh axes; use a sharding constraint to hint propagation over \
                          auto axes"
                    .to_string()
            )),
        );

        let target = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            ReshardOperation::new(target).infer_output_types(
                &[ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(2)]),)],
                &[]
            ),
            Err(TypeError::invalid("reshard target sharding rank (1) does not match the input rank (2)".to_string())),
        );
    }

    #[test]
    fn test_sharding_constraint_is_type_level_identity() {
        let mesh = explicit_manual_mesh();
        let hint = Sharding::new(mesh, vec![ShardingDimension::sharded(["a"])]).unwrap();
        let operation = ShardingConstraintOperation::new(hint.clone());

        assert_eq!(operation.name(), SHARDING_CONSTRAINT_OPERATION_NAME);
        assert_eq!(operation.to_string(), format!("sharding_constraint [sharding={hint}]"));

        // Inference is the identity: the input type passes through untouched, hint included, even though the input
        // carries no sharding of its own.
        let input = vector_type(8);
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input), &[]), Ok(vec![input]));
    }

    #[test]
    fn test_sharding_constraint_rejects_non_auto_axes() {
        let mesh = explicit_manual_mesh();
        let explicit_hint = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            ShardingConstraintOperation::new(explicit_hint).infer_output_types(&[vector_type(8)], &[]),
            Err(TypeError::invalid(
                "sharding_constraint can only hint placement over auto mesh axes, but `x` is not auto; use \
                          reshard for explicit or manual axes"
                    .to_string()
            )),
        );
    }

    fn mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    fn vector_f64_type(size: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(size)]))
    }

    fn matrix_type(rows: usize, columns: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(rows), Dimension::Static(columns)]))
    }

    #[test]
    fn test_reshard_transposition_reshards_the_cotangent_to_the_input_dual() {
        let mesh = mesh();
        // The input is unreduced along the manual axis `m`, so its cotangent must be distributed like the input: the
        // dual sharding (reduced along `m`).
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["m"])
            .unwrap();
        let input = Array::from_f64s(vector_f64_type(8).with_sharding(input_sharding.clone()).unwrap(), vec![1.0; 8]);
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let (_output, pullback) = differentiate_at(input)
            .vjp({
                let target = target.clone();
                move |x| Ok(x.reshard(&target))
            })
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
    fn test_reshard_transposition_restores_an_unsharded_input_cotangent() {
        let mesh = mesh();
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(8)]));
        let input = Array::from_f64s(input_type.clone(), vec![1.0; 8]);
        let (output, pullback) = differentiate_at(input.clone())
            .vjp({
                let target = target.clone();
                move |x| Ok(x.reshard(&target))
            })
            .unwrap();
        let cotangent = pullback.apply(Array::from_f64s(output.r#type().cotangent(), vec![1.0; 8])).unwrap();
        assert_eq!(cotangent.r#type().as_ref(), &input_type.cotangent());
        assert_eq!(cotangent.to_f64s(), vec![1.0; 8]);

        let jacobian = differentiate_at(input)
            .jacobian_reverse({
                let target = target.clone();
                move |x| Ok(x.reshard(&target))
            })
            .unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.input_type(), &input_type);
        assert_eq!(block.value().r#type().data_type(), DataType::F32);
        assert_eq!(block.value().r#type().static_shape().unwrap().as_slice(), &[8, 8]);
        assert_eq!(block.value().r#type().sharding(), None);
        assert_eq!(
            block.value().to_f64s(),
            (0..64).map(|index| if index / 8 == index % 8 { 1.0 } else { 0.0 }).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_reshard_batching_lifts_the_target_sharding() {
        let mesh = mesh();
        // The batch item reshards to a rank-1 sharding; batching over an unsharded input inserts a replicated entry at
        // the new batch axis, so the lifted reshard targets a rank-2 sharding.
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let expected_lifted = target.with_inserted_dimension(0, ShardingDimension::Replicated).unwrap();
        let (_output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let target = target.clone();
                Ok(batch(move |item| Ok(item.reshard(&target)), x, BatchAxis::new(0), BatchAxis::new(0), None).unwrap())
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
        let (_output, pullback) = differentiate_at(Array::vector(vec![1.0; 8]))
            .vjp({
                let hint = hint.clone();
                move |x| Ok(x.constrain_sharding(&hint))
            })
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
        let (_output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let hint = hint.clone();
                Ok(batch(move |item| Ok(item.constrain_sharding(&hint)), x, BatchAxis::new(0), BatchAxis::new(0), None)
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
