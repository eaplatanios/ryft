//! Operations that control how an array is distributed over a device mesh without changing its elements. Both are
//! identity functions on their data and carry a target [`Sharding`]. They only differ in whether the type system tracks
//! that sharding, which mirrors the split between JAX's [`reshard`](https://docs.jax.dev/en/latest/jax.sharding.html)
//! and [`with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html).
//! This module provides the following:
//!
//!   - The [`Reshard`] value capability and the [`ReshardOperation`] it stages, which perform a tracked sharding
//!     transition over [`Explicit`](MeshAxisType::Explicit) mesh axes. Type inference replaces the output's sharding
//!     with the requested one, so the transition is visible to every later type check, and transposition reshards the
//!     cotangent to the dual of the input's sharding so that the input cotangent is distributed like the input.
//!     Requests that name [`Auto`](MeshAxisType::Auto) axes are rejected, because placement over those axes belongs
//!     to a backend-specific compiler.
//!   - The [`ConstrainSharding`] value capability and the [`ConstrainShardingOperation`] it stages, which record a
//!     sharding constraint over [`Auto`](MeshAxisType::Auto) mesh axes that the type system does not track. Type
//!     inference is the identity, so the constraint never becomes type-level state, transposition applies the same
//!     constraint to the cotangent, and the constraint is enforced when a backend lowers the program (i.e., the backend
//!     compiler must place the value as requested over the auto axes, and remains free everywhere the constraint is
//!     unconstrained). Requests that shard a dimension over a non-auto axis are rejected in favor of a reshard.
//!
//! Batching lifts both operations by inserting an entry for the new batch axis into the target sharding (i.e., a
//! reshard takes the mapped axis's own sharding, whereas a constraint leaves the new axis unconstrained for the
//! compiler to fill). Backends lower both to the same sharding-constraint construct (e.g., `sdy.sharding_constraint`
//! in the XLA backend)> A reshard lowers its target as is, since that target is the tracked output sharding, whereas
//! a constraint lowers the merge of the input's tracked placement with its own auto-axis placement (refer to the
//! documentation of [`ConstrainShardingOperation::lowered_sharding`] for more information), so that the emitted
//! constraint never contradicts the type the program was checked against.
//!
//! # Example
//!
//! A reshard changes the traced value's type, whereas a sharding constraint leaves it unchanged and only records the
//! constraint on the staged instruction:
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
//! let constraint = Sharding::new(mesh, vec![ShardingDimension::sharded(["a"])])?;
//! let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
//!     |input| input.reshard(&target)?.constrain_sharding(&constraint),
//!     ArrayType::new_static(DataType::F32, [4]),
//! )?;
//! assert_eq!(output_type.sharding(), Some(&target));
//! let operations = program.instructions().iter().map(|instruction| instruction.operation().name());
//! assert_eq!(operations.collect::<Vec<_>>(), ["reshard", "constrain_sharding"]);
//! # Ok(())
//! # }
//! ```
//!
//! Here, `reshard(&target)` partitions the array's only dimension over the two positions of the explicit mesh axis `x`
//! and records that placement in the traced value's type. The following `constrain_sharding(&constraint)` requires the
//! backend compiler to also partition that dimension over the two positions of the auto mesh axis `a`, while preserving
//! the placement over `x`. The combined placement partitions the four elements across all four mesh positions, but the
//! output type still records only `target` as placement over `a` is enforced during lowering rather than being tracked
//! by the type system. Neither operation changes the array's logical shape or elements.

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

/// Canonical operation name for [`ReshardOperation`].
pub const RESHARD_OPERATION_NAME: &str = "reshard";

/// [`Operation`] that reshards its input to a target [`Sharding`]. This operation is the Ryft analogue of JAX's
/// [`jax.sharding.reshard`](https://docs.jax.dev/en/latest/jax.sharding.html). The array's elements, shape, and data
/// type are unchanged, and the output type carries the target sharding in place of the input's, extended with the
/// manual variation and reduction facts of its input. The target may only name [`MeshAxisType::Explicit`] mesh axes.
/// Placement over [`MeshAxisType::Auto`] axes belongs to backend compilers and can be constrained using
/// [`ConstrainShardingOperation`]s while transitions over [`MeshAxisType::Manual`] axes require their corresponding
/// collectives. Refer to the documentation of [`Reshard`] for more information.
///
/// Interpretation passes the value through and records the target on its type. Batching inserts the mapped axis's
/// sharding into the target at the new batch axis. Differentiation reshards the tangent to the same target, and
/// transposition reshards the cotangent to the dual of the input's sharding, so that the input cotangent is
/// distributed like the input.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshardOperation {
    /// Target [`Sharding`] that the input is resharded to.
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
    #[inline]
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
                "`{}` target sharding rank ({}) does not match the input rank ({})",
                RESHARD_OPERATION_NAME,
                self.sharding.rank(),
                input.rank(),
            )));
        }

        // Every mesh axis the target references, whether it shards a dimension or carries reduction state,
        // must be one the type system governs.
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
                "`{RESHARD_OPERATION_NAME}` cannot target auto mesh axes; use `{CONSTRAIN_SHARDING_OPERATION_NAME}` \
                 to constrain placement over them",
            )));
        }

        if self
            .sharding
            .dimensions()
            .iter()
            .flat_map(|dimension| match dimension {
                ShardingDimension::Sharded(axes) => axes.as_slice(),
                _ => &[],
            })
            .chain(self.sharding.unreduced_axes())
            .chain(self.sharding.reduced_axes())
            .any(|axis| self.sharding.mesh().axis_type(axis) == Some(MeshAxisType::Manual))
        {
            return Err(TypeError::invalid(format!(
                "`{RESHARD_OPERATION_NAME}` cannot target manual mesh axes; use manual collectives \
                     for transitions over them",
            )));
        }

        // Explicit redistribution preserves manual variation and reduction obligations. Their transitions belong
        // to manual collectives, including when a cotangent dual swaps reduced and unreduced state.
        let input_sharding = input.sharding();
        let manual_unreduced = input_sharding
            .into_iter()
            .flat_map(|sharding| {
                sharding
                    .unreduced_axes()
                    .iter()
                    .filter(|axis| sharding.mesh().axis_type(axis) == Some(MeshAxisType::Manual))
            })
            .cloned();
        let manual_reduced = input_sharding
            .into_iter()
            .flat_map(|sharding| {
                sharding
                    .reduced_axes()
                    .iter()
                    .filter(|axis| sharding.mesh().axis_type(axis) == Some(MeshAxisType::Manual))
            })
            .cloned();
        let sharding = self
            .sharding
            .clone()
            .with_unreduced_axes(self.sharding.unreduced_axes().iter().cloned().chain(manual_unreduced))
            .and_then(|sharding| {
                sharding.with_reduced_axes(self.sharding.reduced_axes().iter().cloned().chain(manual_reduced))
            })
            .and_then(|sharding| {
                sharding.with_varying_manual_axes(
                    input_sharding.into_iter().flat_map(|sharding| sharding.varying_manual_axes()).cloned(),
                )
            })
            .map_err(|error| TypeError::invalid(error.to_string()))?;

        Ok(vec![input.clone().with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()))?])
    }

    #[inline]
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
        // The resharding flows through the capability so interpretation over staging values (e.g., program batching,
        // re-tracing, etc.) preserves it. Concrete values pass through unchanged.
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reshard(&self.sharding)?])
    }
}

impl<C: Context<Operation: From<ReshardOperation>>> PartiallyEvaluatableOperation<C> for ReshardOperation {}

// TODO(eaplatanios): Review from here onwards.

// Batching rule for [`ReshardOperation`]. The lifted reshard's target sharding gains the mapped axis's sharding
// (derived from the batched inputs via [`ArrayBatch::sharding_for_inputs`]) at the new batch dimension. Lifting needs
// only the batch axis's position and placement, never its extent, so mapped axes with dynamic extents lift as well.
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
        let lifted_sharding = match inputs[0].batch_axis_position() {
            Some(batch_axis) => self.sharding().batched(batch_axis, ArrayBatch::sharding_for_inputs(inputs)?)?,
            None => self.sharding().clone(),
        };
        rebatch_geometry_preserving_output(context, &ReshardOperation::new(lifted_sharding), inputs)
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
                        Some(input_cotangent_sharding) => {
                            // The operation carries only explicit placement; the input cotangent already carries
                            // the preserved manual reduction state, so it must not be requested as a transition.
                            cotangent.reshard(&input_cotangent_sharding.without_manual_reduction_axes())?
                        },
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
/// which is an identity function on the array's elements whose output type carries the target sharding. Concrete
/// single-device values record the target on their type and leave their payload untouched, and context-carrying values
/// stage the operation instead, so that transforms that apply operations through interpretation (e.g., program
/// batching and re-tracing) preserve the resharding. Every implementation validates the target exactly like
/// [`ReshardOperation`] type inference does, so that eager and staged evaluation accept the same programs.
pub trait Reshard: Clone {
    /// Reshards `self` to `sharding`, and returns a [`ProgramError`] if `sharding` is not a valid target for `self` or
    /// the resharding cannot be recorded in the value's context.
    fn reshard(&self, sharding: &Sharding) -> Result<Self, ProgramError>;
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
// target, and the output type comes from the `ReshardOperation` type-inference rule itself, so that eager evaluation
// validates the target and carries the input's varying manual axes over exactly like staged programs do.
impl Reshard for Array {
    fn reshard(&self, sharding: &Sharding) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        let mut output_types = ReshardOperation::new(sharding.clone()).infer_output_types(&[input_type], &[])?;
        check_count!("output", output_types, 1, ProgramError);
        Ok(Self::new_unchecked(output_types.remove(0), self.shared_storage().clone()))
    }
}

/// Canonical operation name for [`ConstrainShardingOperation`].
pub const CONSTRAIN_SHARDING_OPERATION_NAME: &str = "constrain_sharding";

/// [`Operation`] that constrains the placement of its input over [`Auto`](MeshAxisType::Auto) mesh axes, the analogue
/// of JAX's
/// [`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html).
/// Type inference is the identity, so the output type, sharding included, equals the input type and the constraint
/// never becomes type-level state. The constraint is nevertheless binding: when the program is lowered, the backend
/// compiler must place the value as requested over the auto axes and only remains free where the constraint is
/// unconstrained. A constraint that shards a dimension over a non-auto axis is rejected, because such placements are
/// tracked transitions that belong to a [`ReshardOperation`], and a constraint on a different mesh than the input's
/// tracked sharding is rejected as well. Refer to the documentation of [`ConstrainSharding`] for more information.
///
/// Interpretation passes the value through unchanged. Batching leaves the new batch axis unconstrained in the lifted
/// constraint, so that the compiler remains free to place it. Differentiation applies the same constraint to the
/// tangent, and the operation is self-adjoint under transposition, so the same constraint applies to the cotangent as
/// well. Backends lower [`lowered_sharding`](Self::lowered_sharding) rather than the constraint itself, so that the
/// emitted constraint also carries the input's tracked placement.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstrainShardingOperation {
    /// Refer to the documentation of [`sharding`](Self::sharding) for more information.
    sharding: Sharding,
}

impl ConstrainShardingOperation {
    /// Creates a new [`ConstrainShardingOperation`] that constrains the placement of its input to `sharding`.
    #[inline]
    pub fn new(sharding: Sharding) -> Self {
        Self { sharding }
    }

    /// Returns the [`Sharding`] constraint over auto mesh axes.
    #[inline]
    pub fn sharding(&self) -> &Sharding {
        &self.sharding
    }

    /// Returns the [`Sharding`] a backend must constrain an input of type `input_type` to. An input without a tracked
    /// sharding is constrained to [`sharding`](Self::sharding) as is. Otherwise, the tracked placement is merged into
    /// the constraint dimension by dimension, so that the emitted constraint never contradicts the type the program
    /// was checked against: a tracked sharded dimension keeps its axes and gains the constraint's auto axes after them
    /// (mirroring JAX's `with_sharding_constraint` lowering), or stays as is where the constraint is replicated or
    /// unconstrained, whereas a tracked replicated dimension takes the constraint's entry, since a tracked type is
    /// only replicated over the axes the type system governs and leaves the auto axes to the compiler. The tracked
    /// unreduced, reduced, and varying manual axes are carried over as well.
    ///
    /// # Parameters
    ///
    ///   - `input_type`: Type of the input this operation is applied to, whose rank and mesh (if it carries a
    ///     sharding) must match those of this constraint.
    pub fn lowered_sharding(&self, input_type: &ArrayType) -> Result<Sharding, TypeError> {
        let Some(tracked) = input_type.sharding() else {
            return Ok(self.sharding.clone());
        };
        // A tracked type may still name auto axes (e.g., the actual placement of a concrete input), but placement
        // over those axes is the compiler's to decide and exactly what this constraint overrides, so only the
        // placement the type system governs is carried over.
        let tracked = tracked.without_auto_axes();
        if tracked.rank() != self.sharding.rank() {
            return Err(TypeError::invalid(format!(
                "`{CONSTRAIN_SHARDING_OPERATION_NAME}` rank ({}) does not match the input rank ({})",
                self.sharding.rank(),
                tracked.rank(),
            )));
        }
        if tracked.mesh() != self.sharding.mesh() {
            return Err(TypeError::invalid(format!(
                "`{CONSTRAIN_SHARDING_OPERATION_NAME}` sharding {} is on a different mesh than the input sharding {}",
                self.sharding, tracked,
            )));
        }
        let dimensions = tracked
            .dimensions()
            .iter()
            .zip(self.sharding.dimensions())
            .map(|(tracked, constrained)| match (tracked, constrained) {
                (ShardingDimension::Sharded(tracked_axes), ShardingDimension::Sharded(constrained_axes)) => {
                    ShardingDimension::Sharded(tracked_axes.iter().chain(constrained_axes).cloned().collect())
                }
                (ShardingDimension::Sharded(_), ShardingDimension::Replicated | ShardingDimension::Unconstrained) => {
                    tracked.clone()
                }
                (ShardingDimension::Replicated | ShardingDimension::Unconstrained, constrained) => constrained.clone(),
            })
            .collect();
        self.sharding
            .with_dimensions(dimensions)
            .and_then(|sharding| {
                sharding.with_unreduced_axes(tracked.unreduced_axes().union(self.sharding.unreduced_axes()).cloned())
            })
            .and_then(|sharding| {
                sharding.with_reduced_axes(tracked.reduced_axes().union(self.sharding.reduced_axes()).cloned())
            })
            .and_then(|sharding| {
                sharding.with_varying_manual_axes(
                    tracked.varying_manual_axes().union(self.sharding.varying_manual_axes()).cloned(),
                )
            })
            .map_err(|error| TypeError::invalid(error.to_string()))
    }
}

impl Display for ConstrainShardingOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ConstrainShardingOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        CONSTRAIN_SHARDING_OPERATION_NAME
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
                "`{CONSTRAIN_SHARDING_OPERATION_NAME}` rank ({}) does not match the input rank ({})",
                self.sharding.rank(),
                input.rank(),
            )));
        }
        if let Some(tracked) = input.sharding()
            && tracked.mesh() != self.sharding.mesh()
        {
            return Err(TypeError::invalid(format!(
                "`{CONSTRAIN_SHARDING_OPERATION_NAME}` sharding {} is on a different mesh than the input sharding {}",
                self.sharding, tracked,
            )));
        }
        // The constraint may only place dimensions over auto axes, which are the axes the compiler propagates; naming
        // an explicit or manual axis is the mirror image of the auto-axis rejection of `reshard`.
        for dimension in self.sharding.dimensions() {
            let ShardingDimension::Sharded(axis_names) = dimension else {
                continue;
            };
            if let Some(axis_name) = axis_names
                .iter()
                .find(|axis_name| self.sharding.mesh().axis_type(axis_name) != Some(MeshAxisType::Auto))
            {
                return Err(TypeError::invalid(format!(
                    "`{CONSTRAIN_SHARDING_OPERATION_NAME}` can only constrain placement over auto mesh axes but \
                     `{axis_name}` is not one; use `{RESHARD_OPERATION_NAME}` for explicit axes"
                )));
            }
        }
        // The constraint is untracked: the output type, sharding included, is identical to the input, and the
        // constraint is enforced only when the backend lowers the operation.
        Ok(vec![input.clone()])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("sharding", &self.sharding))
    }
}

impl<C: Domain<Type = ArrayType, Value: ConstrainSharding>> InterpretableOperation<C> for ConstrainShardingOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // The constraint flows through the capability so interpretation over staging values preserves it; concrete
        // values pass through unchanged.
        Ok(vec![inputs[0].constrain_sharding(&self.sharding)?])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for ConstrainShardingOperation where
    C::Operation: From<ConstrainShardingOperation>
{
}

// Batching rule for [`ConstrainShardingOperation`]. The lifted constraint gains a
// [`ShardingDimension::Unconstrained`] entry at the new batch dimension: the constraint governs only the
// compiler-propagated auto axes, so the new dimension is left open for the backend to fill rather than pinned to a
// derived or replicated entry (matching JAX's `with_sharding_constraint` batcher, which inserts
// `PartitionSpec.UNCONSTRAINED`). Like the reshard rule, lifting never needs the batch axis's extent.
impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for ConstrainShardingOperation
where
    ConstrainShardingOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let lifted_sharding = match inputs[0].batch_axis_position() {
            Some(batch_axis) => self.sharding().batched(batch_axis, ShardingDimension::Unconstrained)?,
            None => self.sharding().clone(),
        };
        rebatch_geometry_preserving_output(context, &ConstrainShardingOperation::new(lifted_sharding), inputs)
    }
}

impl_differentiable_operation! {
    ConstrainShardingOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ConstrainShardingOperation>,
        C::Value: ConstrainSharding,
    {
        |operation, _context, _driver, inputs| {
            // The constraint is linear, so the same constraint applies to the tangent as to the primal.
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
        O: Operation<Type = ArrayType> + From<ConstrainShardingOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // The operation is self-adjoint, so the output cotangent is constrained by the same constraint (mirroring
            // JAX registering `with_sharding_constraint` with `ad.deflinear2`). Unlike [`ReshardOperation`], the
            // input's sharding is not consulted, because the constraint is the operation's own.
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

/// Represents the ability to constrain the placement of a value over auto mesh axes. [`ConstrainSharding`] stages a
/// [`ConstrainShardingOperation`], which is an identity function whose constraint is enforced when a backend lowers
/// the program. Concrete single-device values are returned unchanged, and context-carrying values stage the operation
/// instead, so that transforms that apply operations through interpretation preserve the constraint. Every
/// implementation validates the constraint exactly like [`ConstrainShardingOperation`] type inference does, so that
/// eager and staged evaluation accept the same programs.
pub trait ConstrainSharding: Clone {
    /// Constrains the placement of `self` to `sharding`, and returns a [`ProgramError`] if `sharding` is not a valid
    /// constraint for `self` or the constraint cannot be recorded in the value's context.
    fn constrain_sharding(&self, sharding: &Sharding) -> Result<Self, ProgramError>;
}

// Any context-carrying value constrains its sharding by binding a [`ConstrainShardingOperation`] through its own
// context. The `From<ConstrainShardingOperation>` bound makes this disjoint from the eager value types (whose
// context operation is `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete
// implementations.
impl<V: Value<Type = ArrayType>> ConstrainSharding for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ConstrainShardingOperation>,
{
    fn constrain_sharding(&self, sharding: &Sharding) -> Result<Self, ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            ConstrainShardingOperation::new(sharding.clone()),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

// The constraint is untracked, so the output of a concrete single-device value is the value itself once the
// `ConstrainShardingOperation` type-inference rule has accepted the constraint for its type.
impl ConstrainSharding for Array {
    fn constrain_sharding(&self, sharding: &Sharding) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        ConstrainShardingOperation::new(sharding.clone()).infer_output_types(&[input_type], &[])?;
        Ok(self.clone())
    }
}

/// Interprets the lifted geometry-preserving `operation` on the packed value of the single batch in `inputs` and
/// repackages its output with that batch's axis and bounded ragged axes. Both sharding-control operations leave the
/// packed geometry untouched, so every piece of batch metadata carries over as is.
fn rebatch_geometry_preserving_output<C, P, O>(
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    operation: &O,
    inputs: &[ArrayBatch<C::Value>],
) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError>
where
    C: Context<Type = ArrayType>,
    P: ArrayExtentBatchingPolicy<C>,
    O: InterpretableOperation<C>,
{
    check_count!("input", inputs, 1, ProgramError);
    let batch_axis = BatchAxis::from_optional_position(inputs[0].batch_axis_position());
    let mut outputs = operation.interpret_with_batch_axes(context, inputs, &[batch_axis])?;
    check_count!("output", outputs, 1, ProgramError);
    let output = ArrayBatch::new(outputs.remove(0).into_value(), batch_axis)?
        .with_ragged_axes(inputs[0].ragged_axes().to_vec())?;
    Ok(vec![output].into())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayExtentBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, Dimension, DimensionBounds, DimensionType,
        DimensionVariable, LogicalMesh, MeshAxis, RaggedAxis, Shape, f8e8m0fnu,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::programs::{EffectClasses, EmptyRegionDriver, ValueProjection};
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
        // state.
        let auto_target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let auto_error = format!(
            "`{RESHARD_OPERATION_NAME}` cannot target auto mesh axes; use `{CONSTRAIN_SHARDING_OPERATION_NAME}` to \
             constrain placement over them"
        );
        check_operation_type_inference!(
            operation = ReshardOperation::new(auto_target),
            cases = [{
                input_types = [input_type.clone()],
                error = auto_error.clone(),
            }],
        );
        let auto_unreduced_target = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["a"]).unwrap();
        check_operation_type_inference!(
            operation = ReshardOperation::new(auto_unreduced_target),
            cases = [{
                input_types = [input_type.clone()],
                error = auto_error,
            }],
        );

        // Explicit redistribution preserves manual reduction obligations; targets cannot create or discharge them.
        let unreduced_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["m"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = ReshardOperation::new(target.clone()),
            cases = [{
                input_types = [unreduced_input_type],
                output_types = [input_type.clone().with_sharding(target.clone().with_unreduced_axes(["m"]).unwrap()).unwrap()],
            }],
        );
        check_operation_type_inference!(
            operation = ReshardOperation::new(Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["m"]).unwrap()),
            cases = [{
                input_types = [input_type.clone()],
                error = "`reshard` cannot target manual mesh axes; use manual collectives for transitions over them",
            }],
        );
        check_operation_type_inference!(
            operation = ReshardOperation::new(Sharding::new(mesh, vec![ShardingDimension::sharded(["m"])]).unwrap()),
            cases = [{
                input_types = [input_type.clone()],
                error = "`reshard` cannot target manual mesh axes; use manual collectives for transitions over them",
            }],
        );

        // The operation cannot own nested regions.
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

        // Resharding preserves the packed geometry, so a bounded ragged axis on the input carries over to the output.
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let array =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1_f32, 2., 3., 4., 5., 6.]).unwrap();
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1_i32, 3]).unwrap();
        let input = ArrayBatch::new(array, BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, variable, vec![0])])
            .unwrap();
        let ragged_axes = input.ragged_axes().to_vec();
        let (outputs, _) = ReshardOperation::new(target.clone())
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(outputs[0].ragged_axes(), ragged_axes.as_slice());
        assert_eq!(outputs[0].r#type().sharding(), Some(&expected_lifted));

        // Lifting needs only the batch axis's position and placement, so a mapped axis with a dynamic extent lifts.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let extent = trace.input(DimensionType::from(items.clone()).into());
        let shape = Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(3)]);
        let input = trace.input(ArrayType::new(DataType::F32, shape.clone()).into());
        let input = <_ as ValueProjection<ArrayType>>::into_projected(input).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace),
            extent,
        );
        let input = ArrayBatch::new(input, BatchAxis::new(0)).unwrap();
        let (outputs, _) =
            ReshardOperation::new(target).batch(&context, &EmptyRegionDriver, &[input]).unwrap().into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().shape(), &shape);
        assert_eq!(outputs[0].r#type().sharding(), Some(&expected_lifted));
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
        let input_cotangent_type = input.r#type().cotangent().unwrap();
        let (output, pullback) = differentiate_at(input).vjp(|x| x.reshard(&target)).unwrap();
        let cotangent = pullback
            .apply(Array::from_elements::<f64>(output.r#type().cotangent().unwrap(), &[1.0; 8]).unwrap())
            .unwrap();
        assert_eq!(cotangent.r#type().as_ref(), &input_cotangent_type);
        assert_eq!(cotangent.to_f64s(), vec![1.0; 8]);
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        let staged = pullback.instructions().iter().find_map(|instruction| match instruction.operation() {
            ArrayOperation::Reshard(operation) => Some(operation.sharding().clone()),
            _ => None,
        });
        assert_eq!(staged, Some(Sharding::replicated(input_sharding.mesh().clone(), 1)));
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
        let target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let resharded = input.reshard(&target).unwrap();
        assert_eq!(resharded.r#type().sharding(), Some(&target.clone().with_varying_manual_axes(["m"]).unwrap()));
        assert_eq!(resharded.storage_bytes(), input.storage_bytes());

        // Eager evaluation validates the target exactly like staged programs do.
        assert!(matches!(
            input.reshard(&Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!(
                    "`{RESHARD_OPERATION_NAME}` cannot target auto mesh axes; use \
                     `{CONSTRAIN_SHARDING_OPERATION_NAME}` to constrain placement over them"
                ),
        ));
        assert!(matches!(
            input.reshard(&Sharding::replicated(mesh, 2)),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`reshard` target sharding rank (2) does not match the input rank (1)",
        ));
    }

    #[test]
    fn test_constrain_sharding() {
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let operation = ConstrainShardingOperation::new(constraint.clone());

        // Operation identity, accessors, and rendering.
        assert_eq!(operation.name(), CONSTRAIN_SHARDING_OPERATION_NAME);
        assert_eq!(operation.sharding(), &constraint);
        assert_eq!(operation.to_string(), format!("constrain_sharding [sharding={constraint}]"));
    }

    #[test]
    fn test_constrain_sharding_type_inference() {
        let mesh = mesh();
        let constraint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [8]);
        let sharded_input_type = input_type
            .clone()
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        // Inference is the identity: the input type passes through untouched, whether or not it carries a sharding of
        // its own, and the constraint never appears on it.
        check_operation_type_inference!(
            operation = ConstrainShardingOperation::new(constraint.clone()),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone()],
            }, {
                input_types = [sharded_input_type.clone()],
                output_types = [sharded_input_type],
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [8, 2])],
                error = "`constrain_sharding` rank (1) does not match the input rank (2)",
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }],
        );

        // A tracked sharding on another mesh cannot be merged with the constraint, placement over explicit or manual
        // axes is a tracked transition, and the operation cannot own nested regions.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_input_type = input_type.clone().with_sharding(Sharding::replicated(other_mesh, 1)).unwrap();
        check_operation_type_inference!(
            operation = ConstrainShardingOperation::new(constraint.clone()),
            cases = [{
                input_types = [other_input_type],
                error = "`constrain_sharding` sharding {mesh<['x'=2:explicit, 'm'=2:manual, 'a'=2:auto]>, [{'a'}]} \
                         is on a different mesh than the input sharding {mesh<['y'=2:explicit]>, [{}]}",
            }],
        );
        let explicit_constraint = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        check_operation_type_inference!(
            operation = ConstrainShardingOperation::new(explicit_constraint),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{CONSTRAIN_SHARDING_OPERATION_NAME}` can only constrain placement over auto mesh axes but `x` \
                     is not one; use `{RESHARD_OPERATION_NAME}` for explicit axes"
                ),
            }],
        );
        assert_eq!(
            ConstrainShardingOperation::new(constraint).infer_output_types(
                std::slice::from_ref(&input_type),
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_constrain_sharding_lowered_sharding() {
        let mesh = mesh();
        let constraint = Sharding::new(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["a"]),
                ShardingDimension::replicated(),
                ShardingDimension::unconstrained(),
            ],
        )
        .unwrap();
        let operation = ConstrainShardingOperation::new(constraint.clone());
        let input_type = ArrayType::new_static(DataType::F32, [8, 4, 2]);

        // An input without a tracked sharding is constrained to the constraint as is.
        assert_eq!(operation.lowered_sharding(&input_type), Ok(constraint));

        // A tracked sharded dimension keeps its axes and gains the constraint's auto axes after them, or stays as is
        // where the constraint is replicated or unconstrained, whereas a tracked replicated dimension takes the
        // constraint's entry. Tracked variation and reduction state carries over.
        let tracked = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
        )
        .unwrap()
        .with_varying_manual_axes(["m"])
        .unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(tracked).unwrap()),
            Ok(Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["x", "a"]),
                    ShardingDimension::sharded(["m"]),
                    ShardingDimension::unconstrained(),
                ],
            )
            .unwrap()
            .with_varying_manual_axes(["m"])
            .unwrap()),
        );
        let tracked = Sharding::replicated(mesh.clone(), 3).with_unreduced_axes(["x"]).unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(tracked).unwrap()),
            Ok(Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["a"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::unconstrained(),
                ],
            )
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap()),
        );
        let tracked = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(tracked).unwrap()),
            Ok(Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["a"]),
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::unconstrained(),
                ],
            )
            .unwrap()),
        );

        // Tracked placement over auto axes is the compiler's to decide and is overridden by the constraint.
        let tracked = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["a"]), ShardingDimension::replicated()],
        )
        .unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(tracked).unwrap()),
            Ok(Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["a"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::unconstrained(),
                ],
            )
            .unwrap()),
        );

        // A tracked sharding on another mesh or of another rank cannot be merged.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        assert_eq!(
            operation.lowered_sharding(&input_type.clone().with_sharding(Sharding::replicated(other_mesh, 3)).unwrap()),
            Err(TypeError::invalid(
                "`constrain_sharding` sharding {mesh<['x'=2:explicit, 'm'=2:manual, 'a'=2:auto]>, [{'a'}, {}, {?}]} \
                 is on a different mesh than the input sharding {mesh<['y'=2:explicit]>, [{}, {}, {}]}"
            )),
        );
        assert_eq!(
            operation.lowered_sharding(
                &ArrayType::new_static(DataType::F32, [8]).with_sharding(Sharding::replicated(mesh, 1)).unwrap()
            ),
            Err(TypeError::invalid("`constrain_sharding` rank (3) does not match the input rank (1)")),
        );
    }

    #[test]
    fn test_constrain_sharding_interpretation() {
        let mesh = mesh();
        let constraint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let operation = ConstrainShardingOperation::new(constraint);
        // An untracked constraint preserves the payload and all existing type metadata, including manual-axis
        // variation.
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
    fn test_constrain_sharding_partial_evaluation() {
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        check_operation_partial_evaluation!(
            operation = ConstrainShardingOperation::new(constraint),
            inputs = [Array::vector(vec![1.0_f32, 2.0]).unwrap()],
            expected = Array::vector(vec![1.0_f32, 2.0]).unwrap(),
        );
    }

    #[test]
    fn test_constrain_sharding_batching() {
        // The constraint governs only the compiler-propagated auto axes, so batching leaves the new batch axis
        // unconstrained for the backend to fill rather than pinning it to a derived or replicated entry.
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let expected_lifted = constraint.with_inserted_dimension(0, ShardingDimension::Unconstrained).unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let constraint = constraint.clone();
                Ok(batch(
                    move |item| item.constrain_sharding(&constraint),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            ArrayType::new_static(DataType::F64, [2, 3]),
        )
        .unwrap();
        let ArrayOperation::ConstrainSharding(operation) = program.instructions()[0].operation() else {
            panic!("expected the batched program to stage a constrain_sharding operation");
        };
        assert_eq!(operation.sharding(), &expected_lifted);

        // The constraint preserves the packed geometry, so a bounded ragged axis on the input carries over to the
        // output.
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let array =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[1_f32, 2., 3., 4., 5., 6.]).unwrap();
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let extents = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1_i32, 3]).unwrap();
        let input = ArrayBatch::new(array.clone(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents, variable, vec![0])])
            .unwrap();
        let ragged_axes = input.ragged_axes().to_vec();
        let (outputs, _) = ConstrainShardingOperation::new(constraint.clone())
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(outputs[0].ragged_axes(), ragged_axes.as_slice());
        assert_eq!(outputs[0].value(), &array);

        // Lifting needs only the batch axis's position, so a mapped axis with a dynamic extent lifts.
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let extent = trace.input(DimensionType::from(items.clone()).into());
        let shape = Shape::new(vec![Dimension::Dynamic(items), Dimension::Static(3)]);
        let input = trace.input(ArrayType::new(DataType::F32, shape.clone()).into());
        let input = <_ as ValueProjection<ArrayType>>::into_projected(input).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace),
            extent,
        );
        let input = ArrayBatch::new(input, BatchAxis::new(0)).unwrap();
        let (outputs, _) = ConstrainShardingOperation::new(constraint)
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().shape(), &shape);
        assert_eq!(outputs[0].r#type().sharding(), None);
    }

    #[test]
    fn test_constrain_sharding_differentiation() {
        // The constraint is linear, so the JVP applies the same constraint to the primal and to the tangent.
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| x.constrain_sharding(&constraint),
            ArrayType::new_static(DataType::F32, [4]),
        )
        .unwrap();
        let jvp = program.to_flat_program().jvp().unwrap();
        let constraints = jvp
            .instructions()
            .iter()
            .map(|instruction| match instruction.operation() {
                ArrayOperation::ConstrainSharding(operation) => operation.sharding().clone(),
                operation => panic!("expected only sharding constraints but got `{operation}`"),
            })
            .collect::<Vec<_>>();
        assert_eq!(constraints, vec![constraint.clone(), constraint]);
    }

    #[test]
    fn test_constrain_sharding_transposition() {
        // The constraint is self-adjoint, so its transpose re-applies the same constraint to the cotangent rather than
        // dualizing it.
        let constraint = Sharding::new(mesh(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let (_, pullback) = differentiate_at(Array::vector(vec![1.0; 8]).unwrap())
            .vjp(|x| x.constrain_sharding(&constraint))
            .unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        let staged = pullback.instructions().iter().find_map(|instruction| match instruction.operation() {
            ArrayOperation::ConstrainSharding(operation) => Some(operation.sharding().clone()),
            _ => None,
        });
        assert_eq!(staged, Some(constraint));
    }

    #[test]
    fn test_array_constrain_sharding() {
        // The constraint is metadata for lowering only, so a concrete array is returned unchanged.
        let mesh = mesh();
        let constraint = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        let input = Array::vector(vec![1.0_f32, 2.0]).unwrap();
        assert_eq!(input.constrain_sharding(&constraint), Ok(input.clone()));

        // Eager evaluation validates the constraint exactly like staged programs do.
        assert!(matches!(
            input.constrain_sharding(&Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!(
                    "`{CONSTRAIN_SHARDING_OPERATION_NAME}` can only constrain placement over auto mesh axes but `x` \
                     is not one; use `{RESHARD_OPERATION_NAME}` for explicit axes"
                ),
        ));
        assert!(matches!(
            input.constrain_sharding(&Sharding::replicated(mesh, 2)),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`constrain_sharding` rank (2) does not match the input rank (1)",
        ));
    }
}
