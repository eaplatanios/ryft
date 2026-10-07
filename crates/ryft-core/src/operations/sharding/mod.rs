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

#[cfg(doc)]
use crate::arrays::{ArrayIrType, ArrayType, MeshAxisType, Sharding};

pub mod constrain_sharding;
pub mod reshard;

pub use constrain_sharding::{CONSTRAIN_SHARDING_OPERATION_NAME, ConstrainSharding, ConstrainShardingOperation};
pub use reshard::{RESHARD_OPERATION_NAME, Reshard, ReshardOperation};

/// Group of the sharding-control capabilities [`Reshard`] and [`ConstrainSharding`]. It is implemented automatically
/// for every type that implements all of its members.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor and is passed to every member,
/// so that homogeneous array values implement this bundle for [`ArrayType`] and composite array IR values implement it
/// for [`ArrayIrType`].
pub trait ShardingOperations<T = <Self as Capability>::Universe>:
    Capability + Reshard<T> + ConstrainSharding<T>
{
}

impl<T, V: Reshard<T> + ConstrainSharding<T>> ShardingOperations<T> for V {}
