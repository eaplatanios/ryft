//! Operations that control how arrays are distributed over a device mesh. A reshard and a sharding constraint
//! place an array without changing its elements: both are identity functions on their data and carry a target
//! [`Sharding`](crate::Sharding), and they only differ in whether the type system tracks that sharding, which
//! mirrors the split between JAX's [`reshard`](https://docs.jax.dev/en/latest/jax.sharding.html) and
//! [`with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html).
//! A [`shard_map`](fn@shard_map), in contrast, does not redistribute a value but runs a local computation once per
//! device of a manual SPMD computation over a mesh, which mirrors JAX's
//! [`shard_map`](https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html). This module provides the following:
//!
//!   - The [`Reshard`] value capability and the [`ReshardOperation`] it stages, which perform a tracked sharding
//!     transition over [`Explicit`](crate::MeshAxisType::Explicit) mesh axes. Type inference replaces the output's
//!     sharding with the requested one, so the transition is visible to every later type check, and transposition
//!     reshards the cotangent to the dual of the input's sharding so that the input cotangent is distributed like the
//!     input. Requests that name [`Auto`](crate::MeshAxisType::Auto) axes are rejected, because placement over those
//!     axes belongs to a backend-specific compiler.
//!   - The [`ConstrainSharding`] value capability and the [`ConstrainShardingOperation`] it stages, which record a
//!     sharding constraint over [`Auto`](crate::MeshAxisType::Auto) mesh axes that the type system does not track.
//!     Type inference is the identity, so the constraint never becomes type-level state, transposition applies the same
//!     constraint to the cotangent, and the constraint is enforced when a backend lowers the program (i.e., the backend
//!     compiler must place the value as requested over the auto axes, and remains free everywhere the constraint is
//!     unconstrained). Requests that shard a dimension over a non-auto axis are rejected in favor of a reshard.
//!   - The [`Place`] value capability and the [`PlaceOperation`] it stages, which move a value to a concrete
//!     [`Placement`](crate::Placement) (i.e., a sharding over the devices of a [`DeviceMesh`](crate::DeviceMesh),
//!     optionally with a memory and a layout), possibly on another mesh, mirroring JAX's
//!     [`device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html). Type inference records the
//!     target sharding without its [`Auto`](crate::MeshAxisType::Auto) axes, and transposition moves the cotangent back
//!     to the source placement of the value when it is known. A value reports the placement that it concretely lives on
//!     through the [`Placed`] trait.
//!   - The [`ShardMap`] context capability and the global [`shard_map()`](fn@shard_map) function, which trace a local
//!     body and bind it on input values, and the [`ShardMapOperation`] they stage, which runs its local body once per
//!     device coordinate along the active manual axes of a [`ShardMapBoundary`]. Both take [`ShardMapOptions`], which
//!     select these axes. The body receives the local shards that the input shardings assign to the executing device,
//!     and the global outputs are assembled from its local outputs through the output shardings. The body function of
//!     [`ShardMap`] also receives the body's array view of its tracing context (refer to
//!     [`ProjectedContext`](crate::ProjectedContext)), through which bodies without inputs can create values
//!     (e.g., device coordinates along manual axes). The global [`shard_map_with_inferred_input_shardings()`]
//!     function takes the input shardings from the input types, and the global [`shard_map_along()`] function maps
//!     over a single manual axis, placing selected dimensions of its inputs and outputs along it. Like every other
//!     operation, a `shard_map` is traced without values by invoking it on tracers (e.g., through
//!     [`trace`](crate::trace)). Refer to the [`shard_map`](mod@shard_map) module for its complete
//!     semantics, including the ownership contract for references that cross its boundary.
//!   - The global [`auto_axes()`](fn@auto_axes) and [`explicit_axes()`](fn@explicit_axes) functions, which switch
//!     selected mesh axes to [`Auto`](crate::MeshAxisType::Auto) or [`Explicit`](crate::MeshAxisType::Explicit) for the
//!     duration of a function, mirroring JAX's `jax.sharding.auto_axes` and `jax.sharding.explicit_axes`. They retype
//!     the inputs and outputs of the function through reshards and run the function inline, so they add no operation
//!     of their own. Refer to the [`axis_types`](mod@axis_types) module for their conceptual model.
//!
//! Batching lifts the reshard and the sharding constraint by inserting an entry for the new batch axis into the target
//! sharding (i.e., a reshard takes the mapped axis's own sharding without the axes that are auto in its target mesh,
//! whereas a constraint leaves the new axis unconstrained for the compiler to fill). Backends lower both to the same
//! sharding-constraint construct (e.g., `sdy.sharding_constraint` in the XLA backend). A reshard lowers its target,
//! since that target is the tracked output sharding, whereas a constraint lowers the merge of the input's tracked
//! placement with its own auto-axis placement (refer to the documentation of
//! [`ConstrainShardingOperation::lowered_sharding`] for more information), so that the emitted constraint never
//! contradicts the type the program was checked against.
//!
//! [`Reshard`], [`ConstrainSharding`], and [`Place`] apply to individual values and to the leaves of structured
//! values, preserving their structure and static fields. [`ShardingOperations`] groups the two capabilities that
//! take logical shardings. [`Place`] takes concrete placements and infers an omitted source for each leaf.
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

use crate::arrays::LogicalMesh;
use crate::operations::Capability;
use crate::parameters::Parameter;
use crate::programs::TypeError;

pub mod axis_types;
pub mod constrain_sharding;
pub mod place;
pub mod reshard;
pub mod shard_map;

pub use axis_types::{AxisTypesError, auto_axes, explicit_axes};
pub use constrain_sharding::{
    CONSTRAIN_SHARDING_OPERATION_NAME, ConstrainSharding, ConstrainShardingDispatch, ConstrainShardingOperation,
};
pub use place::{
    IntoPlaceTarget, PLACE_OPERATION_NAME, Place, PlaceDispatch, PlaceOperation, PlaceTarget, Placed, PlacementSources,
};
pub use reshard::{RESHARD_OPERATION_NAME, Reshard, ReshardDispatch, ReshardOperation};
pub use shard_map::{
    LockstepDeviceGrid, LockstepDomain, LockstepOperation, LockstepValue, SHARD_MAP_OPERATION_NAME, ShardMap,
    ShardMapBoundary, ShardMapError, ShardMapOperation, ShardMapOptions, ShardMapValueKind, shard_map, shard_map_along,
    shard_map_with_inferred_input_shardings,
};

/// Group of the sharding-control capabilities [`Reshard`] and [`ConstrainSharding`]. It is implemented automatically
/// for every receiver that implements both capabilities. `P` is the receiver's leaf parameter and defaults to `Self`
/// for a single value; structured receivers use the same bundle with their nested leaf type.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// leaves use [`ArrayType`](crate::ArrayType) and array IR leaves use [`ArrayIrType`](crate::ArrayIrType), the two
/// universes that implement this bundle (for a structured receiver, the universe of its leaf `P`). Both members execute
/// in that universe, and generic operation bundles forward it explicitly rather than discarding their execution-family
/// constraint.
pub trait ShardingOperations<P: Parameter + Capability = Self, T = <P as Capability>::Universe>:
    Reshard<P, T> + ConstrainSharding<P, T>
{
}

impl<T, P: Parameter + Capability, S: Reshard<P, T> + ConstrainSharding<P, T>> ShardingOperations<P, T> for S {}

/// Validates that `target_mesh` defines every mesh axis of `input_mesh` that `preserved_axes` names identically (i.e.,
/// with the same name, size, and type). Operations that redistribute or move a value preserve its manual variation and
/// reduction state by axis name, while the size and type of a mesh axis are semantic. Each Ryft type carries its own
/// mesh, so without this check a target could silently redefine such an axis (e.g., turn an explicit unreduced axis
/// into an auto axis, over which array types record no reduction state). Names that `input_mesh` does not define are
/// skipped, because sharding validation already rejects them.
///
/// # Parameters
///
///   - `operation_name`: Name of the validating operation, used in error messages.
///   - `input_mesh`: [`LogicalMesh`] over which the input's state is recorded.
///   - `preserved_axes`: Names of the mesh axes of `input_mesh` that carry the preserved state.
///   - `target_mesh`: [`LogicalMesh`] that must define each preserved axis identically.
fn validate_preserved_mesh_axes<'o, I: IntoIterator<Item = &'o String>>(
    operation_name: &str,
    input_mesh: &LogicalMesh,
    preserved_axes: I,
    target_mesh: &LogicalMesh,
) -> Result<(), TypeError> {
    let preserved_axes = preserved_axes
        .into_iter()
        .filter_map(|axis_name| input_mesh.axis_index(axis_name).map(|index| &input_mesh.axes()[index]));
    for input_axis in preserved_axes {
        let target_axis = target_mesh.axis_index(input_axis.name()).map(|index| &target_mesh.axes()[index]);
        if target_axis != Some(input_axis) {
            let target_axis = target_axis.map_or_else(|| "an undefined axis".to_owned(), |axis| format!("`{axis}`"));
            return Err(TypeError::invalid(format!(
                "`{}` cannot redefine preserved {} mesh axis `{}` as {}",
                operation_name,
                input_axis.r#type(),
                input_axis,
                target_axis,
            )));
        }
    }
    Ok(())
}

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

    #[test]
    fn test_validate_preserved_mesh_axes() {
        let x = MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap();
        let y = MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap();
        let free = MeshAxis::new("free", 4, MeshAxisType::Auto).unwrap();
        let input = LogicalMesh::new(vec![x.clone(), free, y.clone()]).unwrap();
        let preserved = ["unknown".to_owned(), "x".to_owned(), "y".to_owned()];

        // The target may reorder the preserved axes and omit the unselected axis. Unknown input names are skipped.
        let reordered = LogicalMesh::new(vec![y, x.clone()]).unwrap();
        assert_eq!(validate_preserved_mesh_axes(RESHARD_OPERATION_NAME, &input, &preserved, &reordered), Ok(()));

        // A matching first axis still leaves the second axis to validate.
        for (replacement, expected) in [
            (None, "an undefined axis"),
            (Some(MeshAxis::new("y", 4, MeshAxisType::Manual).unwrap()), "`'y'=4:manual`"),
            (Some(MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap()), "`'y'=2:auto`"),
        ] {
            let target = LogicalMesh::new(std::iter::once(x.clone()).chain(replacement).collect()).unwrap();
            assert_eq!(
                validate_preserved_mesh_axes(RESHARD_OPERATION_NAME, &input, &preserved, &target),
                Err(TypeError::invalid(format!(
                    "`{RESHARD_OPERATION_NAME}` cannot redefine preserved manual mesh axis \
                     `'y'=2:manual` as {expected}",
                ))),
            );
        }
    }
}
