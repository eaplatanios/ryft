// TODO(eaplatanios): Review this module.

//! Sharding-control operations: [`ReshardOperation`] and [`ShardingConstraintOperation`].
//!
//! Both operations are unary, leave the array value and its shape/data type untouched, and carry a target
//! [`Sharding`]. They differ in *how that sharding relates to the type system* and *which mesh axis types they
//! govern* — a distinction mirroring JAX's split between
//! [`jax.sharding.reshard`](https://docs.jax.dev/en/latest/jax.sharding.html) and
//! [`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html):
//!
//! - [`ReshardOperation`] performs a **type-level sharding transition** over
//!   [`Explicit`](crate::sharding::MeshAxisType::Explicit) and [`Manual`](crate::sharding::MeshAxisType::Manual)
//!   mesh axes. Type inference *replaces* the output's [`Sharding`] with the requested one, so the new sharding is
//!   tracked by the type system, dualized under transposition (the cotangent is resharded to the cotangent dual of
//!   the input's sharding), and validated against the operand. It rejects requests that name
//!   [`Auto`](crate::sharding::MeshAxisType::Auto) axes (those are the compiler's to place — use a
//!   [`ShardingConstraintOperation`] instead).
//!
//! - [`ShardingConstraintOperation`] is an **untracked propagation hint** over
//!   [`Auto`](crate::sharding::MeshAxisType::Auto) mesh axes. Type inference is the *identity* (the output type — its
//!   sharding included — equals the input type), so the hint never becomes type-level state; it is self-adjoint under
//!   transposition (the same hint is applied to the cotangent) and is materialized only at lowering, where it steers
//!   the compiler's (e.g. [GSPMD](https://arxiv.org/abs/2105.04663) / [Shardy](https://openxla.org/shardy))
//!   sharding propagation. It rejects requests whose [`Sharded`](crate::sharding::ShardingDimension::Sharded) entries
//!   name non-[`Auto`](MeshAxisType::Auto) axes (use a [`ReshardOperation`] for those).
//!
//! Both lower to the same backend sharding-constraint operation (the [`Shardy`](https://openxla.org/shardy)
//! `sdy.sharding_constraint` in the XLA backend); the only difference at the boundary is whether the type system
//! tracked the result. The operations themselves are backend-agnostic — they carry a [`Sharding`] and have purely
//! type-level and autodiff semantics — and each backend decides how to lower them.

use std::fmt::Display;

use crate::contexts::Context;
use crate::contexts::Domain;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::{Operation, OperationFormatter};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::sharding::{MeshAxisType, Sharding, ShardingDimension};
use crate::types::{ArrayType, TypeError};

/// Canonical operation name for [`ReshardOperation`].
pub const RESHARD_OPERATION_NAME: &'static str = "reshard";

/// Canonical operation name for [`ShardingConstraintOperation`].
pub const SHARDING_CONSTRAINT_OPERATION_NAME: &'static str = "sharding_constraint";

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
/// [`ReshardOperation`] is a *tracked* sharding transition over [`Explicit`](crate::sharding::MeshAxisType::Explicit)
/// and [`Manual`](crate::sharding::MeshAxisType::Manual) axes: the requested sharding becomes the output type's
/// sharding and is dualized under transposition. [`ShardingConstraintOperation`] is instead an *untracked* hint over
/// [`Auto`](crate::sharding::MeshAxisType::Auto) axes whose type inference is the identity. Refer to the
/// [module documentation](self) for the full contrast.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshardOperation {
    /// Target [`Sharding`] the input is resharded to.
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

impl Operation<ArrayType> for ReshardOperation {
    #[inline]
    fn name(&self) -> &'static str {
        RESHARD_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let input = &input_types[0];
        if input.rank() != self.sharding.rank() {
            return Err(TypeError {
                message: format!(
                    "{RESHARD_OPERATION_NAME} target sharding rank ({}) does not match the input rank ({})",
                    self.sharding.rank(),
                    input.rank(),
                ),
            });
        }
        if referenced_axes(&self.sharding).any(|axis| self.sharding.mesh().axis_type(axis) == Some(MeshAxisType::Auto))
        {
            return Err(TypeError {
                message: format!(
                    "{RESHARD_OPERATION_NAME} cannot target auto mesh axes; use a sharding constraint to hint \
                     propagation over auto axes"
                ),
            });
        }
        // The resharded value still varies across whatever manual axes the input varied across; that fact is
        // orthogonal to the requested placement, so it is carried over rather than taken from the target.
        let varying_manual_axes =
            input.sharding().map(|sharding| sharding.varying_manual_axes().clone()).unwrap_or_default();
        let sharding = Sharding::with_manual_axes(
            self.sharding.mesh().clone(),
            self.sharding.dimensions().to_vec(),
            self.sharding.unreduced_axes().clone(),
            self.sharding.reduced_axes().clone(),
            varying_manual_axes,
        )
        .map_err(|error| TypeError { message: error.to_string() })?;
        Ok(vec![
            ArrayType::new(input.data_type(), input.shape().clone())
                .with_layout(input.layout().cloned())
                .with_memory(input.memory())
                .with_sharding(sharding)
                .map_err(|error| TypeError { message: error.to_string() })?,
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
            .bind(ReshardOperation::new(sharding.clone()), &[], &[], &[self.clone()])
            .expect("`reshard` operation failed")
            .remove(0)
    }
}

impl<V: Value<Type = ArrayType> + Reshard, C> InterpretableOperation<V, C> for ReshardOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // The resharding flows through the capability so interpretation over staging values (program batching,
        // re-tracing) preserves it; concrete values pass through unchanged.
        Ok(vec![inputs[0].reshard(&self.sharding)])
    }
}

impl<C: Context> PartiallyEvaluatableOperation<C> for ReshardOperation where C::Operation: From<ReshardOperation> {}

/// Unary [`Operation`] that records a sharding-propagation hint, the analogue of JAX's
/// [`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html).
/// It leaves the array value, type, and sharding untouched at the type level and only steers the backend compiler's
/// sharding propagation over [`Auto`](crate::sharding::MeshAxisType::Auto) mesh axes at lowering time.
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

impl Operation<ArrayType> for ShardingConstraintOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SHARDING_CONSTRAINT_OPERATION_NAME
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let input = &input_types[0];
        if input.rank() != self.sharding.rank() {
            return Err(TypeError {
                message: format!(
                    "{SHARDING_CONSTRAINT_OPERATION_NAME} hint rank ({}) does not match the input rank ({})",
                    self.sharding.rank(),
                    input.rank(),
                ),
            });
        }
        // The hint may only place ranked dimensions over auto axes (the axes the compiler propagates); naming an
        // explicit or manual axis is the inverse error of `reshard`'s auto-axis rejection.
        for dimension in self.sharding.dimensions() {
            if let ShardingDimension::Sharded(axis_names) = dimension {
                for axis_name in axis_names {
                    if self.sharding.mesh().axis_type(axis_name) != Some(MeshAxisType::Auto) {
                        return Err(TypeError {
                            message: format!(
                                "{SHARDING_CONSTRAINT_OPERATION_NAME} can only hint placement over auto mesh axes, \
                                 but '{axis_name}' is not auto; use reshard for explicit or manual axes"
                            ),
                        });
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
            .bind(ShardingConstraintOperation::new(sharding.clone()), &[], &[], &[self.clone()])
            .expect("`constrain_sharding` operation failed")
            .remove(0)
    }
}

impl<V: Value<Type = ArrayType> + ConstrainSharding, C> InterpretableOperation<V, C> for ShardingConstraintOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{DataType, Shape, Size};

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
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(size)]))
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
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::replicated()],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["m"],
                )
                .unwrap(),
            )
            .unwrap();
        let expected = vector_type(8)
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["m"],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input)), Ok(vec![expected]));
    }

    #[test]
    fn test_reshard_rejects_auto_axes_and_rank_mismatch() {
        let mesh = explicit_manual_mesh();
        let auto_target = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["a"])]).unwrap();
        assert_eq!(
            ReshardOperation::new(auto_target).infer_output_types(&[vector_type(8)]),
            Err(TypeError {
                message: "reshard cannot target auto mesh axes; use a sharding constraint to hint propagation over \
                          auto axes"
                    .to_string(),
            }),
        );

        let target = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            ReshardOperation::new(target).infer_output_types(&[ArrayType::new(
                DataType::F32,
                Shape::new(vec![Size::Static(8), Size::Static(2)]),
            )]),
            Err(TypeError {
                message: "reshard target sharding rank (1) does not match the input rank (2)".to_string(),
            }),
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
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input)), Ok(vec![input]));
    }

    #[test]
    fn test_sharding_constraint_rejects_non_auto_axes() {
        let mesh = explicit_manual_mesh();
        let explicit_hint = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            ShardingConstraintOperation::new(explicit_hint).infer_output_types(&[vector_type(8)]),
            Err(TypeError {
                message: "sharding_constraint can only hint placement over auto mesh axes, but 'x' is not auto; use \
                          reshard for explicit or manual axes"
                    .to_string(),
            }),
        );
    }
}
