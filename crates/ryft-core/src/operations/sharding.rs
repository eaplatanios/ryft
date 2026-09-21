//! Operations that control how an array is distributed over a device mesh without changing its elements. Both are
//! identity functions on their data and carry a target [`Sharding`]. They only differ in whether the type system tracks
//! that sharding, which mirrors the split between JAX's [`reshard`](https://docs.jax.dev/en/latest/jax.sharding.html)
//! and [`with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html)
//! This module provides the following:
//!
//!   - The [`Reshard`] value capability and the [`ReshardOperation`] it stages, which perform a tracked sharding
//!     transition over [`Explicit`](MeshAxisType::Explicit) and [`Manual`](MeshAxisType::Manual) mesh axes. Type
//!     inference replaces the output's sharding with the requested one, so the transition is visible to every later
//!     type check, and transposition reshards the cotangent to the dual of the input's sharding so that the input
//!     cotangent is distributed like the input. Requests that name [`Auto`](MeshAxisType::Auto) axes are rejected,
//!     because placement over those axes belongs to the compiler.
//!   - The [`ConstrainSharding`] value capability and the [`ShardingConstraintOperation`] it stages, which record an
//!     untracked propagation hint over [`Auto`](MeshAxisType::Auto) mesh axes. Type inference is the identity, so the
//!     hint never becomes type-level state, transposition applies the same hint to the cotangent, and the hint only
//!     takes effect when a backend lowers the program and its compiler propagates shardings. Requests that shard a
//!     dimension over a non-auto axis are rejected in favor of a reshard.
//!
//! Batching lifts both operations by inserting an entry for the new batch axis into the target sharding: a reshard
//! takes the mapped axis's own sharding, whereas a constraint leaves the new axis unconstrained for the compiler to
//! fill. Backends lower both to the same sharding-constraint construct (e.g., `sdy.sharding_constraint` in the XLA
//! backend); the only difference at that boundary is whether the type system already tracked the result.
//!
//! # Example
//!
//! A reshard changes the traced value's type, whereas a sharding constraint leaves it unchanged and only records the
//! hint on the staged instruction:
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, ConstrainSharding, DataType, LogicalMesh, MeshAxis, MeshAxisType, Operation,
//! #     Reshard, Sharding, ShardingDimension, TracingContext,
//! # };
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mesh = LogicalMesh::new(vec![
//!     MeshAxis::new("x", 2, MeshAxisType::Explicit)?,
//!     MeshAxis::new("a", 2, MeshAxisType::Auto)?,
//! ])?;
//! let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])?;
//! let hint = Sharding::new(mesh, vec![ShardingDimension::sharded(["a"])])?;
//! let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
//!     |input| input.reshard(&target)?.constrain_sharding(&hint),
//!     ArrayType::new_static(DataType::F32, [4]),
//! )?;
//! assert_eq!(output_type.sharding(), Some(&target));
//! let operations = program.instructions().iter().map(|instruction| instruction.operation().name());
//! assert_eq!(operations.collect::<Vec<_>>(), ["reshard", "sharding_constraint"]);
//! # Ok(())
//! # }
//! ```

use std::fmt::Display;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayType, MeshAxisType, Sharding,
    ShardingDimension,
};
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

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`ReshardOperation`].
pub const RESHARD_OPERATION_NAME: &str = "reshard";

/// [`Operation`] that reshards its input to a target [`Sharding`], the analogue of JAX's
/// [`jax.sharding.reshard`](https://docs.jax.dev/en/latest/jax.sharding.html). The array's elements, shape, and data
/// type are unchanged, and the output type carries the target sharding in place of the input's, extended with the
/// manual axes the input varied over, since that variation is orthogonal to placement. The target may only name
/// [`Explicit`](MeshAxisType::Explicit) and [`Manual`](MeshAxisType::Manual) mesh axes; placement over
/// [`Auto`](MeshAxisType::Auto) axes is hinted with a [`ShardingConstraintOperation`] instead. Refer to the
/// documentation of [`Reshard`] for more information.
///
/// Interpretation passes the value through and records the target on its type. Batching inserts the mapped axis's
/// sharding into the target at the new batch axis. Differentiation reshards the tangent to the same target, and
/// transposition reshards the cotangent to the dual of the input's sharding, so that the input cotangent is
/// distributed like the input.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshardOperation {
    /// Refer to the documentation of [`sharding`](Self::sharding) for more information.
    sharding: Sharding,
}

impl ReshardOperation {
    /// Creates a new [`ReshardOperation`] that reshards its input to `sharding`.
    #[inline]
    pub fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    /// Returns the target [`Sharding`] that the input is resharded to.
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
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let input = &input_types[0];
        if input.rank() != self.sharding.rank() {
            return Err(TypeError::invalid(format!(
                "`{RESHARD_OPERATION_NAME}` target sharding rank ({}) does not match the input rank ({})",
                self.sharding.rank(),
                input.rank(),
            )));
        }
        // Every mesh axis the target references, whether it shards a dimension or carries reduction state, must be
        // one the type system governs.
        let sharded_axes = self.sharding.dimensions().iter().flat_map(|dimension| match dimension {
            ShardingDimension::Sharded(axis_names) => axis_names.as_slice(),
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => &[],
        });
        let reduction_axes = self.sharding.unreduced_axes().iter().chain(self.sharding.reduced_axes());
        if sharded_axes
            .chain(reduction_axes)
            .any(|axis| self.sharding.mesh().axis_type(axis) == Some(MeshAxisType::Auto))
        {
            return Err(TypeError::invalid(format!(
                "`{RESHARD_OPERATION_NAME}` cannot target auto mesh axes; use `{SHARDING_CONSTRAINT_OPERATION_NAME}` \
                 to hint propagation over them"
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
        Ok(vec![input.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()))?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("sharding", &self.sharding))
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
        Ok(vec![inputs[0].reshard(&self.sharding)?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for ReshardOperation where C::Operation: From<ReshardOperation> {}

// Batching rule for [`ReshardOperation`]. The lifted reshard's target sharding gains the mapped axis's sharding
// (derived from the batched inputs via [`ArrayBatch::sharding_for_inputs`]) at the new batch dimension.
impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for ReshardOperation
where
    ReshardOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        // Validates that a mapped batch axis has a static size before lifting.
        ArrayBatch::common_batch_size(inputs)?;
        let (lifted_sharding, output_axis) = match inputs[0].batch_axis_position() {
            Some(batch_axis) => {
                let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
                let lifted = self.sharding().batched(batch_axis, axis_sharding)?;
                (lifted, Some(batch_axis))
            }
            None => (self.sharding().clone(), None),
        };
        Ok(ReshardOperation::new(lifted_sharding)
            .interpret_with_batch_axes(context, inputs, &[BatchAxis::from_optional_position(output_axis)])?
            .into())
    }
}

impl_differentiable_operation! {
    ReshardOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ReshardOperation>,
        C::Value: Reshard,
    {
        |operation, _context, _driver, inputs| {
            // Resharding is linear, so the tangent is resharded to the same target sharding as the primal.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().reshard(operation.sharding())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.reshard(operation.sharding())?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<BroadcastOperation> + From<ReshardOperation>,
    {
        |_operation, context, _driver, inputs, outputs, accumulators| {
            // Transpose rule for [`ReshardOperation`]: the cotangent of a reshard is itself a reshard of the output
            // cotangent to the cotangent dual of the *input*'s sharding (swapping its unreduced and reduced axes), so
            // the produced input cotangent is distributed like the input. An input that carries no sharding receives
            // an exactly unsharded cotangent through an identity-axis broadcast.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            let input_cotangent_type = inputs[0].r#type().cotangent()?;
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let contribution = match input_cotangent_type.sharding() {
                        Some(input_cotangent_sharding) => cotangent.reshard(input_cotangent_sharding)?,
                        None => cotangent.broadcast(
                            input_cotangent_type.clone(),
                            &(0..input_cotangent_type.shape().rank()).collect::<Vec<_>>(),
                        )?,
                    };
                    accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;
                    Ok(())
                }
                MaybeZero::Zero(_) => Ok(()),
            }
        }
    },
}

/// Represents the ability to reshard a value to a target [`Sharding`]. [`Reshard`] stages a [`ReshardOperation`],
/// which is an identity function on the array's elements whose output type carries the target sharding. The provided
/// default returns the value unchanged, which is correct for concrete single-device values, for which a sharding only
/// describes distribution metadata; context-carrying values stage the operation instead, so that transforms that apply
/// operations through interpretation (e.g., program batching and re-tracing) preserve the resharding.
pub trait Reshard: Clone {
    /// Reshards `self` to `sharding`, and returns a [`ProgramError`] if `sharding` is not a valid target for `self` or
    /// the resharding cannot be recorded in the value's context.
    fn reshard(&self, sharding: &Sharding) -> Result<Self, ProgramError> {
        let _ = sharding;
        Ok(self.clone())
    }
}

// Any context-carrying value reshards by binding a [`ReshardOperation`] through its own context. The
// `From<ReshardOperation>` bound makes this disjoint from the eager value types (whose context operation is
// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Reshard for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ReshardOperation>,
{
    fn reshard(&self, sharding: &Sharding) -> Result<Self, ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            ReshardOperation::new(sharding.clone()),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

// An `Array` is a concrete single-device value, so resharding is a no-op on its payload. Its type still records the
// requested distribution metadata — mirroring the `ReshardOperation` type-inference rule, which carries the input's
// varying manual axes over to the target sharding — so interpreted programs preserve their declared boundaries
// exactly. An invalid target sharding is reported as an error, which the type-level validation performed before
// interpretation rules out for staged programs.
impl Reshard for Array {
    fn reshard(&self, sharding: &Sharding) -> Result<Self, ProgramError> {
        let varying_manual_axes =
            self.r#type().sharding().map(|sharding| sharding.varying_manual_axes().clone()).unwrap_or_default();
        let sharding = sharding
            .clone()
            .with_varying_manual_axes(varying_manual_axes)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let r#type = self
            .r#type()
            .into_owned()
            .with_sharding(sharding)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        Ok(Self::new_unchecked(r#type, self.shared_storage().clone()))
    }
}

/// Canonical operation name for [`ShardingConstraintOperation`].
pub const SHARDING_CONSTRAINT_OPERATION_NAME: &str = "sharding_constraint";

/// [`Operation`] that records a sharding-propagation hint on its input, the analogue of JAX's
/// [`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html).
/// Type inference is the identity, so the output type, sharding included, equals the input type and the hint never
/// becomes type-level state; it only steers the backend compiler's sharding propagation over
/// [`Auto`](MeshAxisType::Auto) mesh axes when the program is lowered. A hint that shards a dimension over a
/// non-auto axis is rejected, because such placements are tracked transitions that belong to a [`ReshardOperation`].
/// Refer to the documentation of [`ConstrainSharding`] for more information.
///
/// Interpretation passes the value through unchanged. Batching leaves the new batch axis unconstrained in the lifted
/// hint, so that the compiler remains free to place it. Differentiation applies the same hint to the tangent, and the
/// operation is self-adjoint under transposition, so the same hint applies to the cotangent as well.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShardingConstraintOperation {
    /// Refer to the documentation of [`sharding`](Self::sharding) for more information.
    sharding: Sharding,
}

impl ShardingConstraintOperation {
    /// Creates a new [`ShardingConstraintOperation`] that records `sharding` as a hint on its input.
    #[inline]
    pub fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    /// Returns the [`Sharding`] hint recorded for the backend's propagation over auto mesh axes.
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
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let input = &input_types[0];
        if input.rank() != self.sharding.rank() {
            return Err(TypeError::invalid(format!(
                "`{SHARDING_CONSTRAINT_OPERATION_NAME}` hint rank ({}) does not match the input rank ({})",
                self.sharding.rank(),
                input.rank(),
            )));
        }
        // The hint may only place dimensions over auto axes, which are the axes the compiler propagates; naming an
        // explicit or manual axis is the mirror image of the auto-axis rejection of `reshard`.
        for dimension in self.sharding.dimensions() {
            let ShardingDimension::Sharded(axis_names) = dimension else {
                continue;
            };
            if let Some(axis_name) = axis_names
                .iter()
                .find(|axis_name| self.sharding.mesh().axis_type(axis_name) != Some(MeshAxisType::Auto))
            {
                return Err(TypeError::invalid(format!(
                    "`{SHARDING_CONSTRAINT_OPERATION_NAME}` can only hint placement over auto mesh axes but \
                     `{axis_name}` is not one; use `{RESHARD_OPERATION_NAME}` for explicit or manual axes"
                )));
            }
        }
        // The hint is untracked: the output type, sharding included, is identical to the input, and the requested
        // sharding only takes effect when the backend lowers the operation.
        Ok(vec![input.clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("sharding", &self.sharding))
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
        Ok(vec![inputs[0].constrain_sharding(&self.sharding)?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for ShardingConstraintOperation where
    C::Operation: From<ShardingConstraintOperation>
{
}

// Batching rule for [`ShardingConstraintOperation`]. The lifted hint gains a [`ShardingDimension::Unconstrained`]
// entry at the new batch dimension: the hint governs only the compiler-propagated auto axes, so the new dimension
// is left open for the backend to fill rather than pinned to a derived or replicated entry (matching JAX's
// `with_sharding_constraint` batcher, which inserts `PartitionSpec.UNCONSTRAINED`).
impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for ShardingConstraintOperation
where
    ShardingConstraintOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        // Validates that a mapped batch axis has a static size before lifting.
        ArrayBatch::common_batch_size(inputs)?;
        let (lifted_sharding, output_axis) = match inputs[0].batch_axis_position() {
            Some(batch_axis) => {
                let lifted = self.sharding().batched(batch_axis, ShardingDimension::Unconstrained)?;
                (lifted, Some(batch_axis))
            }
            None => (self.sharding().clone(), None),
        };
        Ok(ShardingConstraintOperation::new(lifted_sharding)
            .interpret_with_batch_axes(context, inputs, &[BatchAxis::from_optional_position(output_axis)])?
            .into())
    }
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
            let primal = inputs[0].primal().constrain_sharding(operation.sharding())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.constrain_sharding(operation.sharding())?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<ShardingConstraintOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Transpose rule for [`ShardingConstraintOperation`]: the operation is self-adjoint, so the cotangent of
            // the output is constrained by the *same* hint (mirroring JAX registering `with_sharding_constraint` with
            // `ad.deflinear2`). Unlike [`ReshardOperation`], the input's sharding is not consulted — the hint is the
            // operation's own.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let contribution = MaybeZero::Value(cotangent.constrain_sharding(operation.sharding())?);
                    accumulators[0].accumulate(context, contribution)?;
                    Ok(())
                }
                MaybeZero::Zero(_) => Ok(()),
            }
        }
    },
}

/// Represents the ability to record a sharding-propagation hint on a value. [`ConstrainSharding`] stages a
/// [`ShardingConstraintOperation`], which is an identity function whose hint only takes effect when a backend lowers
/// the program. The provided default returns the value unchanged, which is correct for concrete single-device values;
/// context-carrying values stage the operation instead, so that transforms that apply operations through
/// interpretation preserve the hint.
pub trait ConstrainSharding: Clone {
    /// Records `sharding` as a propagation hint on `self`, and returns a [`ProgramError`] if the hint cannot be
    /// recorded in the value's context.
    fn constrain_sharding(&self, sharding: &Sharding) -> Result<Self, ProgramError> {
        let _ = sharding;
        Ok(self.clone())
    }
}

// Any context-carrying value constrains its sharding by binding a [`ShardingConstraintOperation`] through its own
// context. The `From<ShardingConstraintOperation>` bound makes this disjoint from the eager value types (whose
// context operation is `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete
// implementations.
impl<V: Value<Type = ArrayType>> ConstrainSharding for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ShardingConstraintOperation>,
{
    fn constrain_sharding(&self, sharding: &Sharding) -> Result<Self, ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            ShardingConstraintOperation::new(sharding.clone()),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

// The sharding-constraint hint is untracked: the output type (sharding included) is identical to the input, so the
// identity default is exactly the `ShardingConstraintOperation` interpretation contract for a concrete value.
impl ConstrainSharding for Array {}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, DataType, LogicalMesh, MeshAxis, f8e8m0fnu};
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::programs::{EffectClasses, EmptyRegionDriver};
    use crate::tracing::TracingContext;

    use super::*;

    /// Returns a mesh with one axis of each type: `x` is explicit, `m` is manual, and `a` is auto.
    fn mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn test_reshard() {
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let operation = ReshardOperation::new(target.clone());

        // Operation identity, accessors, and rendering.
        assert_eq!(operation.name(), RESHARD_OPERATION_NAME);
        assert_eq!(operation.sharding(), &target);
        assert_eq!(operation.to_string(), format!("reshard [sharding={target}]"));
    }

    #[test]
    fn test_reshard_type_inference() {
        let mesh = mesh();
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [8]);
        // The output keeps the input's shape and data type and adopts the target sharding; an input varying over
        // manual axes carries that variation over to the target.
        let varying_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = ReshardOperation::new(target.clone()),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_sharding(target.clone()).unwrap()],
            }, {
                input_types = [varying_input_type],
                output_types = [input_type
                    .clone()
                    .with_sharding(target.clone().with_varying_manual_axes(["m"]).unwrap())
                    .unwrap()],
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [8, 2])],
                error = "`reshard` target sharding rank (1) does not match the input rank (2)",
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }],
        );

        // Placement over auto axes belongs to the compiler, whether the axis shards a dimension or carries reduction
        // state, and the operation cannot own nested regions.
        let auto_target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let auto_error = format!(
            "`{RESHARD_OPERATION_NAME}` cannot target auto mesh axes; use `{SHARDING_CONSTRAINT_OPERATION_NAME}` to \
             hint propagation over them"
        );
        check_operation_type_inference!(
            operation = ReshardOperation::new(auto_target),
            cases = [{
                input_types = [input_type.clone()],
                error = auto_error.clone(),
            }],
        );
        let auto_unreduced_target = Sharding::replicated(mesh, 1).with_unreduced_axes(["a"]).unwrap();
        check_operation_type_inference!(
            operation = ReshardOperation::new(auto_unreduced_target),
            cases = [{
                input_types = [input_type.clone()],
                error = auto_error,
            }],
        );
        assert_eq!(
            ReshardOperation::new(target).infer_output_types(
                std::slice::from_ref(&input_type),
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_reshard_interpretation() {
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let operation = ReshardOperation::new(target.clone());
        let input = Array::vector(vec![1.0_f32, 2.0]).unwrap();
        // Interpretation passes the elements through and records the target on the output type.
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input)),
            Ok(vec![
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [2]).with_sharding(target).unwrap(),
                    &[1.0_f32, 2.0]
                )
                .unwrap(),
            ]),
        );
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_reshard_partial_evaluation() {
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        check_operation_partial_evaluation!(
            operation = ReshardOperation::new(target.clone()),
            inputs = [Array::vector(vec![1.0_f32, 2.0]).unwrap()],
            expected = Array::from_elements(
                ArrayType::new_static(DataType::F32, [2]).with_sharding(target).unwrap(),
                &[1.0_f32, 2.0],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_reshard_batching() {
        // The batch item reshards to a rank-1 sharding; batching over an unsharded input inserts a replicated entry at
        // the new batch axis, so the lifted reshard targets a rank-2 sharding.
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let expected_lifted = target.with_inserted_dimension(0, ShardingDimension::Replicated).unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let target = target.clone();
                Ok(batch(move |item| item.reshard(&target), x, BatchAxis::new(0), BatchAxis::new(0), None)?)
            },
            ArrayType::new_static(DataType::F64, [2, 3]),
        )
        .unwrap();
        let ArrayOperation::Reshard(operation) = program.instructions()[0].operation() else {
            panic!("expected the batched program to stage a reshard operation");
        };
        assert_eq!(operation.sharding(), &expected_lifted);
    }

    #[test]
    fn test_reshard_differentiation() {
        // Resharding is linear, so the tangent is resharded to the same target as the primal.
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (primal, tangent) = differentiate_at(Array::vector(vec![1.0_f32, 2.0]).unwrap())
            .jvp(Array::vector(vec![3.0_f32, 4.0]).unwrap(), |x| x.reshard(&target))
            .unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(target).unwrap();
        assert_eq!(primal, Array::from_elements(expected_type.clone(), &[1.0_f32, 2.0]).unwrap());
        assert_eq!(tangent, Array::from_elements(expected_type, &[3.0_f32, 4.0]).unwrap());
    }

    #[test]
    fn test_reshard_transposition() {
        let mesh = mesh();
        // The input is unreduced along the manual axis `m`, so its cotangent must be distributed like the input: the
        // dual sharding, which is reduced along `m`.
        let input_sharding = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["m"]).unwrap();
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [8]).with_sharding(input_sharding.clone()).unwrap(),
            &[1.0; 8],
        )
        .unwrap();
        let target = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.reshard(&target)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        let staged = pullback.instructions().iter().find_map(|instruction| match instruction.operation() {
            ArrayOperation::Reshard(operation) => Some(operation.sharding().clone()),
            _ => None,
        });
        assert_eq!(staged, Some(input_sharding.cotangent()));
    }

    #[test]
    fn test_reshard_transposition_unsharded_input() {
        // An input without a sharding receives an exactly unsharded cotangent through an identity broadcast rather
        // than a reshard, including for element formats whose cotangent space widens to `f32`.
        let target = Sharding::new(mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F8E8M0FNU, [8]);
        let input = Array::from_elements::<f8e8m0fnu>(
            input_type.clone(),
            &[1.0; 8].map(|value| f8e8m0fnu::from_f64(value).unwrap()),
        )
        .unwrap();
        let (output, pullback) = differentiate_at(input.clone()).vjp(|x| x.reshard(&target)).unwrap();
        let cotangent = pullback
            .apply(Array::from_elements::<f32>(output.r#type().cotangent().unwrap(), &[1.0; 8]).unwrap())
            .unwrap();
        assert_eq!(cotangent.r#type().as_ref(), &input_type.cotangent().unwrap());
        assert_eq!(cotangent.to_f64s(), vec![1.0; 8]);

        let jacobian = differentiate_at(input).jacobian_reverse(|x| x.reshard(&target)).unwrap();
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
    fn test_array_reshard() {
        let mesh = mesh();
        // Resharding records the target on the type, carrying the input's varying manual axes over exactly like the
        // `ReshardOperation` type-inference rule, and leaves the payload untouched.
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [2])
                .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap())
                .unwrap(),
            &[1.0, 2.0],
        )
        .unwrap();
        let target = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = input.reshard(&target).unwrap();
        assert_eq!(resharded.r#type().sharding(), Some(&target.clone().with_varying_manual_axes(["m"]).unwrap()));
        assert_eq!(resharded.storage_bytes(), input.storage_bytes());

        // A target on a mesh without the input's varying manual axes is not a valid destination.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_target = Sharding::new(other_mesh, vec![ShardingDimension::sharded(["y"])]).unwrap();
        assert!(matches!(input.reshard(&other_target), Err(ProgramError::Type(TypeError::Invalid { .. }))));
    }

    #[test]
    fn test_sharding_constraint() {
        let hint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let operation = ShardingConstraintOperation::new(hint.clone());

        // Operation identity, accessors, and rendering.
        assert_eq!(operation.name(), SHARDING_CONSTRAINT_OPERATION_NAME);
        assert_eq!(operation.sharding(), &hint);
        assert_eq!(operation.to_string(), format!("sharding_constraint [sharding={hint}]"));
    }

    #[test]
    fn test_sharding_constraint_type_inference() {
        let mesh = mesh();
        let hint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [8]);
        let sharded_input_type = input_type
            .clone()
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        // Inference is the identity: the input type passes through untouched, whether or not it carries a sharding of
        // its own, and the hint never appears on it.
        check_operation_type_inference!(
            operation = ShardingConstraintOperation::new(hint.clone()),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone()],
            }, {
                input_types = [sharded_input_type.clone()],
                output_types = [sharded_input_type],
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [8, 2])],
                error = "`sharding_constraint` hint rank (1) does not match the input rank (2)",
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }],
        );

        // Placement over explicit or manual axes is a tracked transition, and the operation cannot own nested regions.
        let explicit_hint = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        check_operation_type_inference!(
            operation = ShardingConstraintOperation::new(explicit_hint),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{SHARDING_CONSTRAINT_OPERATION_NAME}` can only hint placement over auto mesh axes but `x` is \
                     not one; use `{RESHARD_OPERATION_NAME}` for explicit or manual axes"
                ),
            }],
        );
        assert_eq!(
            ShardingConstraintOperation::new(hint).infer_output_types(
                std::slice::from_ref(&input_type),
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_sharding_constraint_interpretation() {
        let mesh = mesh();
        let hint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let operation = ShardingConstraintOperation::new(hint);
        // An untracked hint preserves the payload and all existing type metadata, including manual-axis variation.
        let input = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [2])
                .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["m"]).unwrap())
                .unwrap(),
            &[1.0, 2.0],
        )
        .unwrap();
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input)),
            Ok(vec![input]),
        );
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_sharding_constraint_partial_evaluation() {
        let hint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        check_operation_partial_evaluation!(
            operation = ShardingConstraintOperation::new(hint),
            inputs = [Array::vector(vec![1.0_f32, 2.0]).unwrap()],
            expected = Array::vector(vec![1.0_f32, 2.0]).unwrap(),
        );
    }

    #[test]
    fn test_sharding_constraint_batching() {
        // The hint governs only the compiler-propagated auto axes, so batching leaves the new batch axis unconstrained
        // for the backend to fill rather than pinning it to a derived or replicated entry.
        let hint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let expected_lifted = hint.with_inserted_dimension(0, ShardingDimension::Unconstrained).unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let hint = hint.clone();
                Ok(batch(move |item| item.constrain_sharding(&hint), x, BatchAxis::new(0), BatchAxis::new(0), None)?)
            },
            ArrayType::new_static(DataType::F64, [2, 3]),
        )
        .unwrap();
        let ArrayOperation::ShardingConstraint(operation) = program.instructions()[0].operation() else {
            panic!("expected the batched program to stage a sharding_constraint operation");
        };
        assert_eq!(operation.sharding(), &expected_lifted);
    }

    #[test]
    fn test_sharding_constraint_differentiation() {
        // The hint is linear, so the JVP applies the same hint to the primal and to the tangent.
        let hint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| x.constrain_sharding(&hint),
            ArrayType::new_static(DataType::F32, [4]),
        )
        .unwrap();
        let jvp = program.to_flat_program().jvp().unwrap();
        let hints = jvp
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::ShardingConstraint(operation) => operation.sharding().clone(),
                operation => panic!("expected only sharding constraints but got `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(hints, vec![hint.clone(), hint]);
    }

    #[test]
    fn test_sharding_constraint_transposition() {
        // The constraint is self-adjoint, so its transpose re-applies the same hint to the cotangent rather than
        // dualizing it.
        let hint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let (_, pullback) =
            differentiate_at(Array::vector(vec![1.0; 8]).unwrap()).vjp(|x| x.constrain_sharding(&hint)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        let staged = pullback.instructions().iter().find_map(|instruction| match instruction.operation() {
            ArrayOperation::ShardingConstraint(operation) => Some(operation.sharding().clone()),
            _ => None,
        });
        assert_eq!(staged, Some(hint));
    }

    #[test]
    fn test_array_constrain_sharding() {
        // The hint is metadata for lowering only, so a concrete array is returned unchanged.
        let hint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let input = Array::vector(vec![1.0_f32, 2.0]).unwrap();
        assert_eq!(input.constrain_sharding(&hint), Ok(input));
    }
}
