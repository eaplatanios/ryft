//! Operations that control how arrays are distributed over a device mesh. A reshard and a sharding constraint place an
//! array without changing its elements: both are identity functions on their data and carry a target [`Sharding`], and
//! they only differ in whether the type system tracks that sharding, which mirrors the split between JAX's
//! [`reshard`](https://docs.jax.dev/en/latest/jax.sharding.html) and
//! [`with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html).
//! A [`shard_map`](fn@shard_map), in contrast, does not redistribute a value but runs a local computation once per
//! device of a manual SPMD computation over a mesh, which mirrors JAX's
//! [`shard_map`](https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html). This module provides the following:
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
//!   - The [`shard_map`](fn@shard_map), [`shard_map_with_options`], and [`shard_map_in_context`] closure entry points,
//!     which trace a local body and bind it on input values, the [`trace_shard_map`], [`trace_shard_map_with_options`],
//!     and [`trace_shard_map_with_named_axes`] closure entry points, which trace a local body for global input types
//!     and return it unbound as a [`TracedShardMap`], and the [`ShardMapOperation`] they stage, which runs its local
//!     body once per device coordinate along the active manual axes of a [`ShardMap`]. The body receives the local
//!     shards that the input shardings assign to the executing device, and the global outputs are assembled from its
//!     local outputs through the output shardings. The closures of [`shard_map_in_context`] and
//!     [`trace_shard_map_with_named_axes`] also receive the [`ShardMapContext`] of the body, through
//!     which bodies without inputs can create values (e.g., device coordinates along manual axes).
//!     Refer to the [`shard_map`](mod@shard_map) module for its complete semantics, including the ownership contract
//!     for references that cross its boundary.
//!
//! Batching lifts the reshard and the sharding constraint by inserting an entry for the new batch axis into the
//! target sharding (i.e., a reshard takes the mapped axis's own sharding, whereas a constraint leaves the new axis
//! unconstrained for the compiler to fill). Backends lower both to the same sharding-constraint construct (e.g.,
//! `sdy.sharding_constraint` in the XLA backend). A reshard lowers its target as is, since that target is the tracked
//! output sharding, whereas a constraint lowers the merge of the input's tracked placement with its own auto-axis
//! placement (refer to the documentation of [`ConstrainShardingOperation::lowered_sharding`] for more information),
//! so that the emitted constraint never contradicts the type the program was checked against.
//!
//! The [`Reshard`] and [`ConstrainSharding`] capabilities apply resharding and sharding constraints to every leaf
//! of a structured value through primitive value dispatch, preserving its structure and static fields.
//!
//! # Example
//!
//! A reshard changes the traced value's type, whereas a sharding constraint leaves it unchanged and only records the
//! constraint on the staged instruction:
//!
//! ```rust
//! # use indoc::indoc;
//! # use pretty_assertions::assert_eq;
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, ConstrainSharding, DataType, LogicalMesh, MeshAxis, MeshAxisType,
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
//! assert_eq!(
//!     program.to_string(),
//!     indoc! {"
//!         lambda %0:f32[4] .
//!         let %1:f32[4][sharding={mesh<['x'=2:explicit, 'a'=2:auto]>, [{'x'}]}] = reshard \
//!                 [sharding={mesh<['x'=2:explicit, 'a'=2:auto]>, [{'x'}]}] %0
//!             %2:f32[4][sharding={mesh<['x'=2:explicit, 'a'=2:auto]>, [{'x'}]}] = constrain_sharding \
//!                 [sharding={mesh<['x'=2:explicit, 'a'=2:auto]>, [{'a'}]}] %1
//!         in (%2)
//!     "}
//!     .trim_end(),
//! );
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

use crate::operations::Capability;
use crate::parameters::Parameter;

#[cfg(doc)]
use crate::arrays::{ArrayIrType, ArrayType, MeshAxisType, Sharding};

pub mod constrain_sharding;
pub mod reshard;
pub mod shard_map;

pub use constrain_sharding::{
    CONSTRAIN_SHARDING_OPERATION_NAME, ConstrainSharding, ConstrainShardingDispatch, ConstrainShardingOperation,
};
pub use reshard::{RESHARD_OPERATION_NAME, Reshard, ReshardDispatch, ReshardOperation};
pub use shard_map::{
    SHARD_MAP_OPERATION_NAME, ShardMap, ShardMapContext, ShardMapError, ShardMapOperation, ShardMapTracer,
    TracedShardMap, shard_map, shard_map_in_context, shard_map_with_options, trace_shard_map,
    trace_shard_map_with_named_axes, trace_shard_map_with_options,
};

/// Group of the sharding-control capabilities [`Reshard`] and [`ConstrainSharding`]. It is implemented automatically
/// for every receiver that implements both capabilities. `P` is the receiver's leaf parameter and defaults to `Self`
/// for a single value; structured receivers use the same bundle with their nested leaf type.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// leaves use [`ArrayType`] and composite array IR leaves use [`ArrayIrType`], the two universes that implement this
/// bundle (for a structured receiver, the universe of its leaf `P`). Both members execute in that universe, and generic
/// operation bundles forward it explicitly rather than discarding their execution-family constraint.
pub trait ShardingOperations<P: Parameter + Capability = Self, T = <P as Capability>::Universe>:
    Reshard<P, T> + ConstrainSharding<P, T>
{
}

impl<T, P: Parameter + Capability, S: Reshard<P, T> + ConstrainSharding<P, T>> ShardingOperations<P, T> for S {}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayOperations, ArrayType, DataType,
        LogicalMesh, MeshAxis, MeshAxisType, Sharding,
    };
    use crate::differentiation::DifferentiableType;
    use crate::programs::{ProgramError, Value};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_sharding_operations_universes() {
        /// Asserts that one receiver supports sharding its leaves in the requested universe.
        fn assert_bundle<T, P: Parameter + Capability, S: ShardingOperations<P, T>>() {}

        /// Exercises both members through the array bundle while retaining an independent value-type bound.
        fn apply<T, V: ArrayOperations<T>>(
            input: &V,
            target: &Sharding,
            constraint: &Sharding,
        ) -> Result<(V, V::Type), ProgramError>
        where
            V::Type: DifferentiableType,
        {
            let tangent_type = input.r#type().tangent()?;
            Ok((input.reshard(target)?.constrain_sharding(constraint)?, tangent_type))
        }

        /// Exercises the default universe through a normal generic bundle bound.
        fn apply_default<V: Value + ShardingOperations>(input: &V, target: &Sharding) -> Result<V, ProgramError> {
            input.reshard(target)
        }

        assert_bundle::<ArrayType, Array, Array>();
        assert_bundle::<ArrayIrType, ArrayIrValue<Array>, ArrayIrValue<Array>>();
        assert_bundle::<ArrayType, Array, Vec<Array>>();
        assert_bundle::<ArrayIrType, ArrayIrValue<Array>, (ArrayIrValue<Array>, Vec<ArrayIrValue<Array>>)>();

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let target = Sharding::replicated(mesh.clone(), 0);
        let constraint = Sharding::replicated(mesh, 0);
        let input = Array::scalar(3.0f32).unwrap();
        let expected = input.reshard(&target).unwrap();
        assert_eq!(
            apply::<ArrayType, _>(&input, &target, &constraint),
            Ok((expected.clone(), ArrayType::scalar(DataType::F32))),
        );
        assert_eq!(apply_default(&input, &target), Ok(expected.clone()));

        let input = ArrayIrValue::Array(input);
        let expected = ArrayIrValue::Array(expected);
        assert_eq!(
            apply::<ArrayIrType, _>(&input, &target, &constraint),
            Ok((expected.clone(), ArrayIrType::Array(ArrayType::scalar(DataType::F32)))),
        );
        assert_eq!(apply_default(&input, &target), Ok(expected));

        // Explicit bundle universes also remain usable when the leaf is a context-carrying tracer.
        let (output_type, _) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| apply::<ArrayType, _>(&input, &target, &constraint).map(|(output, _)| output),
            ArrayType::scalar(DataType::F32),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::scalar(DataType::F32).with_sharding(target.clone()).unwrap());
        let (output_type, _) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| apply::<ArrayIrType, _>(&input, &target, &constraint).map(|(output, _)| output),
            ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32).with_sharding(target).unwrap()));
    }
}
