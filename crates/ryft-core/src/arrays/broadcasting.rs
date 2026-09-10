use std::borrow::Borrow;
use std::collections::BTreeSet;

use thiserror::Error;

use crate::arrays::sharding::ShardingError;
use crate::arrays::sharding::shardings::{Sharding, ShardingDimension};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::{DataType, DataTypeError};
use crate::arrays::types::dimensions::{Dimension, Shape};
use crate::arrays::types::memories::Memory;
use crate::parameters::{ParameterError, Parameterized};

/// Represents broadcasting-related errors.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BroadcastingError {
    #[error("cannot broadcast an empty collection of types")]
    EmptyBroadcastingInput,

    #[error("failed to broadcast due to incompatible data types; {0}")]
    IncompatibleDataTypes(#[from] DataTypeError),

    #[error("failed to broadcast shape `{lhs}` to shape `{rhs}`")]
    IncompatibleShapes { lhs: Shape, rhs: Shape },

    #[error("failed to broadcast due to incompatible shardings; lhs={lhs:?}, rhs={rhs:?}")]
    IncompatibleShardings { lhs: Option<Box<Sharding>>, rhs: Option<Box<Sharding>> },

    #[error(
        "failed to broadcast memory space `{lhs}` to memory space `{rhs}`; broadcasting never moves values, so \
        operands must reside in the same memory space and combining them requires staging an explicit transfer first"
    )]
    IncompatibleMemories { lhs: Memory, rhs: Memory },

    #[error("failed to reconstruct the parameterized structure after broadcasting; {0}")]
    ParameterError(#[from] ParameterError),

    #[error("failed to broadcast sharding information; {0}")]
    ShardingError(#[from] ShardingError),
}

/// Represents [`Type`](crate::Type)s or values that can be broadcast together.
///
/// Broadcasting in Ryft has two orthogonal components:
///
///   - **Parameter Broadcasting:** Each concrete implementer defines what it means to combine two
///     [`Parameter`](crate::Parameter)s. For example, [`DataType`] uses data-type promotion, [`Shape`]
///     follows the standard [NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html),
///     and [`ArrayType`] combines both by broadcasting its [`DataType`] and [`Shape`] and by using these rules:
///       - [`Memory`] spaces must match. Combining values in different spaces requires an explicit transfer beforehand.
///       - Symmetric broadcasting preserves an explicit [`Layout`](crate::Layout) only when both leaves agree on it.
///         Directional broadcasting adopts the target's layout.
///       - [`Sharding`]s are aligned with the shapes, padding missing leading axes with replicated dimensions.
///         Replication is neutral; compatible non-replicated assignments are preserved. Two specified meshes must
///         agree, a mesh axis may not appear in multiple output dimensions, and reduction-state and varying-manual-axis
///         sets must agree when both leaves specify sharding. A non-replicated singleton dimension retains its sharding
///         constraints.
///   - **Structural Broadcasting:** For [`Parameterized`] values whose leaves are [`ArrayType`]s, Ryft first aligns
///     the left-hand side to the target parameter structure using [`Parameterized::broadcast_to_parameter_structure`].
///     That alignment uses path-prefix broadcasting on named parameters, so a value with a smaller compatible parameter
///     structure can be broadcast into a larger one. Once the structures are aligned, leaf broadcasting is applied
///     pairwise and the final structured value is reconstructed.
///
/// # NumPy-style Broadcasting Semantics
///
/// For [`Shape`]s, Ryft follows the same broadcasting rules that NumPy uses:
///
///   - Dimensions are compared from right to left.
///   - Two aligned dimensions are compatible when they are equal or when one of them is `1`.
///   - If the operands have different ranks, missing leading dimensions are treated as if they had size `1`.
///
/// Conceptually, dimensions of size `1` are stretched to match the other operand. As NumPy notes, that stretching is
/// a semantic model for compatibility and result-shape inference; it does not imply that an implementation must
/// materialize expanded copies of the underlying data.
///
/// These rules imply that, for example:
///
///   - a scalar broadcasts to any shape,
///   - `(3,)` broadcasts with `(4, 3)` to `(4, 3)`,
///   - `(4,)` does not broadcast with `(4, 3)` because the trailing dimensions `4` and `3` are incompatible, and
///   - `(4, 1)` broadcasts with `(3,)` to `(4, 3)`.
///
/// The [`Broadcastable::broadcast`] operation is symmetric and returns the least common result that both operands can
/// broadcast to. The [`Broadcastable::broadcast_to`] operation is directional and requires the left-hand side to
/// broadcast exactly to the right-hand side's target. The default [`Broadcastable::broadcasted`] helper folds the
/// symmetric operation over multiple values from left to right.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Broadcastable, BroadcastingError, Shape};
/// #
/// # fn main() -> Result<(), BroadcastingError> {
/// let column = Shape::new(vec![4.into(), 1.into()]);
/// let row = Shape::new(vec![3.into()]);
/// let matrix = Shape::new(vec![4.into(), 3.into()]);
/// assert_eq!(column.broadcast(&row)?, matrix);
/// assert_eq!(row.broadcast_to(&matrix)?, matrix);
/// assert!(!matrix.is_broadcastable_to(&row));
/// # Ok(())
/// # }
/// ```
///
/// [`ArrayType`] broadcasting combines [`Shape`] broadcasting and [`DataType`] promotion independently at each leaf:
///
/// ```rust
/// # use ryft_core::{ArrayType, Broadcastable, BroadcastingError, DataType, Shape};
/// #
/// # fn main() -> Result<(), BroadcastingError> {
/// let source = (
///     ArrayType::scalar(DataType::Boolean),
///     ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 3.into()])),
/// );
/// let target = (
///     ArrayType::new(DataType::F32, Shape::new(vec![2.into(), 3.into()])),
///     ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 1.into()])),
/// );
/// assert_eq!(
///     source.broadcast(&target)?,
///     (
///         ArrayType::new(DataType::F32, Shape::new(vec![2.into(), 3.into()])),
///         ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()])),
///     ),
/// );
/// # Ok(())
/// # }
/// ```
pub trait Broadcastable: Sized {
    /// Broadcasts this value with `other` and returns the least common result that both values can broadcast to.
    /// This operation is _symmetric_. For example, with [`Shape`] values this returns the smallest shape that both
    /// operands can broadcast to, and with [`ArrayType`] values it combines [`DataType`] promotion with [`Shape`]
    /// broadcasting. Shapes may expand on either side, and element types are promoted to a common type. Incompatible
    /// dimensions, types, parameter structures, memory spaces, or shardings return the corresponding
    /// [`BroadcastingError`]. Neither operand is modified.
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError>;

    /// Broadcasts this value to the provided `other` value. Unlike [`Broadcastable::broadcast`], this operation
    /// is _not symmetric_ (i.e., `x.broadcast_to(y)` and `y.broadcast_to(x)` may differ and one of them may even fail).
    /// The source cannot lose axes, shrink non-singleton dimensions, or require a data-type promotion beyond the target
    /// type. For array types, the target layout is adopted and compatible sharding information from both operands is
    /// combined, so the result may carry more sharding information than `other`. Incompatibility returns the
    /// corresponding [`BroadcastingError`]. Neither operand is modified.
    fn broadcast_to(&self, other: &Self) -> Result<Self, BroadcastingError>;

    /// Broadcasts the provided values into a single value by folding over [`Broadcastable::broadcast`]
    /// from left to right. A singleton collection returns a clone of its sole value. An empty collection returns
    /// [`BroadcastingError::EmptyBroadcastingInput`]; otherwise the first incompatible pair stops the fold.
    fn broadcasted<I: Borrow<Self>>(values: &[I]) -> Result<Self, BroadcastingError>
    where
        Self: Clone,
    {
        let (head, tail) = values.split_first().ok_or(BroadcastingError::EmptyBroadcastingInput)?;
        tail.iter().try_fold(head.borrow().clone(), |accum, value| accum.broadcast(value.borrow()))
    }

    /// Returns whether [`Broadcastable::broadcast_to`] would succeed, without returning the resulting metadata or
    /// incompatibility error. This check is directional and includes element types, shapes, parameter structure,
    /// memory spaces, and sharding constraints for array types.
    ///
    /// # Parameters
    ///
    ///   - `other`: Target metadata against which `self` is checked.
    fn is_broadcastable_to(&self, other: &Self) -> bool {
        self.broadcast_to(other).is_ok()
    }
}

impl Broadcastable for DataType {
    #[inline]
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError> {
        Ok(DataType::promoted(&[*self, *other])?)
    }

    #[inline]
    fn broadcast_to(&self, other: &Self) -> Result<Self, BroadcastingError> {
        Ok(self.promote_to(*other)?)
    }

    #[inline]
    fn is_broadcastable_to(&self, other: &Self) -> bool {
        self.is_promotable_to(*other)
    }
}

impl Broadcastable for Shape {
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError> {
        // Handle differing array ranks by (conceptually) padding the shorter shape with ones on the left
        // (i.e., as a prefix), up to the rank of the longer shape.
        let broadcasted_rank = self.rank().max(other.rank());
        let self_offset = broadcasted_rank - self.rank();
        let other_offset = broadcasted_rank - other.rank();
        let mut broadcasted_dimensions = Vec::with_capacity(broadcasted_rank);
        for i in 0..broadcasted_rank {
            let self_size =
                if i < self_offset { Dimension::Static(1) } else { self.dimensions()[i - self_offset].clone() };
            let other_size =
                if i < other_offset { Dimension::Static(1) } else { other.dimensions()[i - other_offset].clone() };
            let broadcasted_size = match (&self_size, &other_size) {
                (_, Dimension::Static(1)) => self_size,
                (Dimension::Static(1), _) => other_size,
                (Dimension::Static(x), Dimension::Static(y)) if x == y => Dimension::Static(*x),
                (Dimension::Dynamic(x), Dimension::Dynamic(y)) if x == y => Dimension::Dynamic(x.clone()),
                _ => {
                    return Err(BroadcastingError::IncompatibleShapes { lhs: self.clone(), rhs: other.clone() });
                }
            };
            broadcasted_dimensions.push(broadcasted_size);
        }

        Ok(Shape::new(broadcasted_dimensions))
    }

    fn broadcast_to(&self, other: &Self) -> Result<Self, BroadcastingError> {
        if self.rank() > other.rank() {
            return Err(BroadcastingError::IncompatibleShapes { lhs: self.clone(), rhs: other.clone() });
        }

        // Handle differing array ranks by (conceptually) padding the dimension sizes of the left shape with
        // ones on the left (i.e., as a prefix), up to the rank of the right shape.
        let broadcasted_rank = other.rank();
        let offset = broadcasted_rank - self.rank();
        let mut broadcasted_shape = Vec::with_capacity(broadcasted_rank);
        for i in 0..broadcasted_rank {
            let self_size = if i < offset { Dimension::Static(1) } else { self.dimensions()[i - offset].clone() };
            let other_size = other.dimensions()[i].clone();
            let broadcasted_size = match (&self_size, &other_size) {
                (Dimension::Static(1), _) => other_size,
                (Dimension::Static(x), Dimension::Static(y)) if x == y => Dimension::Static(*y),
                (Dimension::Dynamic(x), Dimension::Dynamic(y)) if x == y => Dimension::Dynamic(y.clone()),
                _ => {
                    return Err(BroadcastingError::IncompatibleShapes { lhs: self.clone(), rhs: other.clone() });
                }
            };
            broadcasted_shape.push(broadcasted_size);
        }

        Ok(Shape::new(broadcasted_shape))
    }

    fn is_broadcastable_to(&self, other: &Self) -> bool {
        if self.rank() > other.rank() {
            return false;
        }

        let broadcasted_rank = other.rank();
        let offset = broadcasted_rank - self.rank();
        for i in 0..broadcasted_rank {
            let self_size = if i < offset { Dimension::Static(1) } else { self.dimensions()[i - offset].clone() };
            let other_size = other.dimensions()[i].clone();
            match (&self_size, &other_size) {
                (Dimension::Static(1), _) => continue,
                (Dimension::Static(x), Dimension::Static(y)) if x == y => continue,
                (Dimension::Dynamic(x), Dimension::Dynamic(y)) if x == y => continue,
                _ => return false,
            };
        }

        true
    }
}

impl<T: Parameterized<ArrayType>> Broadcastable for T {
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError> {
        let broadcast_to = |lhs: &Self, rhs: &Self| -> Result<Self, BroadcastingError> {
            let structure = rhs.parameter_structure();
            let broadcasted_array_types = lhs
                .broadcast_to_parameter_structure::<T>(structure.clone())?
                .parameters()
                .zip(rhs.parameters())
                .map(|(lhs, rhs)| {
                    if lhs.memory() != rhs.memory() {
                        return Err(BroadcastingError::IncompatibleMemories { lhs: lhs.memory(), rhs: rhs.memory() });
                    }
                    let broadcasted_data_type = lhs.data_type().broadcast(&rhs.data_type())?;
                    let broadcasted_shape = lhs.shape().broadcast(rhs.shape())?;
                    let broadcasted_layout = (lhs.layout() == rhs.layout()).then(|| lhs.layout().cloned()).flatten();
                    let broadcasted_sharding = broadcast_sharding(
                        lhs.shape(),
                        lhs.sharding(),
                        rhs.shape(),
                        rhs.sharding(),
                        &broadcasted_shape,
                    )?;
                    Ok(ArrayType::new(broadcasted_data_type, broadcasted_shape)
                        .with_layout(broadcasted_layout)
                        .with_sharding(broadcasted_sharding)?
                        .with_memory(lhs.memory()))
                })
                .collect::<Result<Vec<_>, BroadcastingError>>()?;
            Ok(Self::from_parameters(structure, broadcasted_array_types)?)
        };

        match broadcast_to(self, other) {
            Ok(broadcasted) => Ok(broadcasted),
            Err(_) => broadcast_to(other, self),
        }
    }

    fn broadcast_to(&self, other: &Self) -> Result<Self, BroadcastingError> {
        let structure = other.parameter_structure();
        let broadcasted_array_types = self
            .broadcast_to_parameter_structure::<T>(structure.clone())?
            .parameters()
            .zip(other.parameters())
            .map(|(lhs, rhs)| {
                if lhs.memory() != rhs.memory() {
                    return Err(BroadcastingError::IncompatibleMemories { lhs: lhs.memory(), rhs: rhs.memory() });
                }
                let broadcasted_data_type = lhs.data_type().broadcast_to(&rhs.data_type())?;
                let broadcasted_shape = lhs.shape().broadcast_to(rhs.shape())?;
                let broadcasted_sharding =
                    broadcast_sharding(lhs.shape(), lhs.sharding(), rhs.shape(), rhs.sharding(), &broadcasted_shape)?;
                Ok(ArrayType::new(broadcasted_data_type, broadcasted_shape)
                    .with_layout(rhs.layout().cloned())
                    .with_sharding(broadcasted_sharding)?
                    .with_memory(lhs.memory()))
            })
            .collect::<Result<Vec<_>, BroadcastingError>>()?;
        Ok(Self::from_parameters(structure, broadcasted_array_types)?)
    }

    fn is_broadcastable_to(&self, other: &Self) -> bool {
        let Ok(broadcasted_self) = self.broadcast_to_parameter_structure::<T>(other.parameter_structure()) else {
            return false;
        };
        broadcasted_self.parameters().zip(other.parameters()).all(|(lhs, rhs)| {
            lhs.memory() == rhs.memory()
                && lhs.data_type().is_broadcastable_to(&rhs.data_type())
                && lhs.shape().is_broadcastable_to(rhs.shape())
                && is_sharding_broadcastable_to(lhs.shape(), lhs.sharding(), rhs.shape(), rhs.sharding())
        })
    }
}

/// Broadcasts an optional [`Sharding`] paired with a [`Shape`] to another optional [`Sharding`] paired with a [`Shape`]
/// and returns the resulting [`Sharding`], using the following broadcasting rules:
///
///   - If neither operand carries sharding information, then this function returns no sharding information.
///   - Any provided [`Sharding`] must already have the same rank as its source [`Shape`], as guaranteed by the
///     containing [`ArrayType`].
///   - When the operands have different ranks, the lower-rank sharding is left-padded with replicated dimensions so
///     that sharding alignment follows the same leading-rank promotion rules as [`Shape`] broadcasting.
///   - On an aligned axis, a singleton dimension is only treated as broadcast-trivial when its [`ShardingDimension`] is
///     already [`ShardingDimension::Replicated`]. Non-replicated singleton-axis shardings are preserved and must still
///     be compatible with the other operand.
///   - If neither aligned axis is a singleton, identical sharding dimensions remain unchanged. A replicated dimension
///     is neutral and yields to the other operand's [`Sharding`].
///   - If both aligned non-singleton axes carry different non-replicated shardings, the operands are considered
///     incompatible and this function will return a [`BroadcastingError`].
///   - Rank promotion and outer-product style broadcasts preserve the contributing operand's [`Sharding`] on axes that
///     only one operand meaningfully contributes to.
///   - Both operands must use the same [`LogicalMesh`](crate::LogicalMesh) when they are both sharded. After the
///     per-axis dimensions are combined, reusing the same mesh axis across multiple result dimensions is treated as an
///     incompatible broadcast, and for those cases, this function will return a [`BroadcastingError`].
///   - The [`Sharding::unreduced_axes`], [`Sharding::reduced_axes`], and [`Sharding::varying_manual_axes`] sets are
///     only preserved when both inputs already agree on them, or when only one operand carries sharding information.
///     Generic [`ArrayType`] broadcasting does not attempt primitive-specific manual-axis merges.
///
/// # Parameters
///
///   - `lhs_shape`: [`Shape`] of the left-hand operand before broadcasting.
///   - `lhs_sharding`: Optional [`Sharding`] for the left-hand operand.
///   - `rhs_shape`: [`Shape`] of the right-hand operand before broadcasting.
///   - `rhs_sharding`: Optional [`Sharding`] for the right-hand operand.
///   - `broadcasted_shape`: Already validated result shape of the symmetric or directional broadcast.
fn broadcast_sharding(
    lhs_shape: &Shape,
    lhs_sharding: Option<&Sharding>,
    rhs_shape: &Shape,
    rhs_sharding: Option<&Sharding>,
    broadcasted_shape: &Shape,
) -> Result<Option<Sharding>, BroadcastingError> {
    let mesh = match (lhs_sharding, rhs_sharding) {
        (None, None) => {
            return Ok(None);
        }
        (Some(left), None) => left.mesh().clone(),
        (None, Some(right)) => right.mesh().clone(),
        (Some(left), Some(right)) if left.mesh() == right.mesh() => left.mesh().clone(),
        (Some(left), Some(right)) => {
            return Err(BroadcastingError::IncompatibleShardings {
                lhs: Some(Box::new(left.clone())),
                rhs: Some(Box::new(right.clone())),
            });
        }
    };

    let result_rank = broadcasted_shape.rank();
    let lhs_offset = result_rank - lhs_shape.rank();
    let rhs_offset = result_rank - rhs_shape.rank();

    let mut used_axes = BTreeSet::new();
    let mut broadcasted_dimensions = Vec::with_capacity(result_rank);
    for index in 0..result_rank {
        let lhs_size =
            if index < lhs_offset { Dimension::Static(1) } else { lhs_shape.dimensions()[index - lhs_offset].clone() };
        let rhs_size =
            if index < rhs_offset { Dimension::Static(1) } else { rhs_shape.dimensions()[index - rhs_offset].clone() };
        let lhs_dimension = padded_sharding_dimension(lhs_sharding, lhs_offset, index);
        let rhs_dimension = padded_sharding_dimension(rhs_sharding, rhs_offset, index);
        let Some(dimension) = broadcast_sharding_dimension(lhs_size, lhs_dimension, rhs_size, rhs_dimension) else {
            return Err(BroadcastingError::IncompatibleShardings {
                lhs: lhs_sharding.cloned().map(Box::new),
                rhs: rhs_sharding.cloned().map(Box::new),
            });
        };
        if let ShardingDimension::Sharded(axis_names) = dimension {
            for axis_name in axis_names {
                if !used_axes.insert(axis_name.clone()) {
                    return Err(BroadcastingError::IncompatibleShardings {
                        lhs: lhs_sharding.cloned().map(Box::new),
                        rhs: rhs_sharding.cloned().map(Box::new),
                    });
                }
            }
        }
        broadcasted_dimensions.push(dimension.clone());
    }

    let unreduced_axes = match (lhs_sharding, rhs_sharding) {
        (None, None) => BTreeSet::new(),
        (Some(left), None) => left.unreduced_axes().clone(),
        (None, Some(right)) => right.unreduced_axes().clone(),
        (Some(left), Some(right)) if left.unreduced_axes() == right.unreduced_axes() => left.unreduced_axes().clone(),
        (Some(_), Some(_)) => {
            return Err(BroadcastingError::IncompatibleShardings {
                lhs: lhs_sharding.cloned().map(Box::new),
                rhs: rhs_sharding.cloned().map(Box::new),
            });
        }
    };

    let reduced_axes = match (lhs_sharding, rhs_sharding) {
        (None, None) => BTreeSet::new(),
        (Some(left), None) => left.reduced_axes().clone(),
        (None, Some(right)) => right.reduced_axes().clone(),
        (Some(left), Some(right)) if left.reduced_axes() == right.reduced_axes() => left.reduced_axes().clone(),
        (Some(_), Some(_)) => {
            return Err(BroadcastingError::IncompatibleShardings {
                lhs: lhs_sharding.cloned().map(Box::new),
                rhs: rhs_sharding.cloned().map(Box::new),
            });
        }
    };

    let varying_manual_axes = match (lhs_sharding, rhs_sharding) {
        (None, None) => BTreeSet::new(),
        (Some(left), None) => left.varying_manual_axes().clone(),
        (None, Some(right)) => right.varying_manual_axes().clone(),
        (Some(left), Some(right)) if left.varying_manual_axes() == right.varying_manual_axes() => {
            left.varying_manual_axes().clone()
        }
        (Some(_), Some(_)) => {
            return Err(BroadcastingError::IncompatibleShardings {
                lhs: lhs_sharding.cloned().map(Box::new),
                rhs: rhs_sharding.cloned().map(Box::new),
            });
        }
    };

    Ok(Some(
        Sharding::new(mesh, broadcasted_dimensions)?
            .with_unreduced_axes(unreduced_axes)?
            .with_reduced_axes(reduced_axes)?
            .with_varying_manual_axes(varying_manual_axes)?,
    ))
}

/// Returns `true` if the provided [`Sharding`]s are broadcastable, according to the rules of [`broadcast_sharding`].
fn is_sharding_broadcastable_to(
    lhs_shape: &Shape,
    lhs_sharding: Option<&Sharding>,
    rhs_shape: &Shape,
    rhs_sharding: Option<&Sharding>,
) -> bool {
    match (lhs_sharding, rhs_sharding) {
        (None, None) => return true,
        (Some(left), Some(right)) if left.mesh() != right.mesh() => return false,
        _ => {}
    }

    let result_rank = rhs_shape.rank();
    let lhs_offset = result_rank - lhs_shape.rank();
    let rhs_offset = result_rank - rhs_shape.rank();
    let mut used_axes = BTreeSet::new();
    for index in 0..result_rank {
        let lhs_size =
            if index < lhs_offset { Dimension::Static(1) } else { lhs_shape.dimensions()[index - lhs_offset].clone() };
        let rhs_size =
            if index < rhs_offset { Dimension::Static(1) } else { rhs_shape.dimensions()[index - rhs_offset].clone() };
        let lhs_dimension = padded_sharding_dimension(lhs_sharding, lhs_offset, index);
        let rhs_dimension = padded_sharding_dimension(rhs_sharding, rhs_offset, index);
        let Some(dimension) = broadcast_sharding_dimension(lhs_size, lhs_dimension, rhs_size, rhs_dimension) else {
            return false;
        };
        if let ShardingDimension::Sharded(axis_names) = dimension {
            for axis_name in axis_names {
                if !used_axes.insert(axis_name.as_str()) {
                    return false;
                }
            }
        }
    }

    match (lhs_sharding, rhs_sharding) {
        (Some(left), Some(right)) => {
            left.unreduced_axes() == right.unreduced_axes()
                && left.reduced_axes() == right.reduced_axes()
                && left.varying_manual_axes() == right.varying_manual_axes()
        }
        _ => true,
    }
}

/// Returns the [`ShardingDimension`] visible at `index` after left-padding lower-rank shardings
/// with [`ShardingDimension::Replicated`] axes.
fn padded_sharding_dimension(sharding: Option<&Sharding>, offset: usize, index: usize) -> &ShardingDimension {
    (index < offset)
        .then_some(&ShardingDimension::Replicated)
        .or_else(|| sharding.map(|sharding| &sharding.dimensions()[index - offset]))
        .unwrap_or(&ShardingDimension::Replicated)
}

/// Combines two aligned [`ShardingDimension`]s using the rules described in [`broadcast_sharding`].
fn broadcast_sharding_dimension<'d>(
    lhs_dimension: Dimension,
    lhs_sharding: &'d ShardingDimension,
    rhs_dimension: Dimension,
    rhs_sharding: &'d ShardingDimension,
) -> Option<&'d ShardingDimension> {
    match (lhs_sharding, rhs_sharding) {
        (lhs_sharding, rhs_sharding) if lhs_sharding == rhs_sharding => Some(lhs_sharding),
        (ShardingDimension::Replicated, rhs_sharding) if matches!(lhs_dimension, Dimension::Static(1)) => {
            Some(rhs_sharding)
        }
        (lhs_sharding, ShardingDimension::Replicated) if matches!(rhs_dimension, Dimension::Static(1)) => {
            Some(lhs_sharding)
        }
        (ShardingDimension::Replicated, rhs_sharding) => Some(rhs_sharding),
        (lhs_sharding, ShardingDimension::Replicated) => Some(lhs_sharding),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameterized;

    use crate::arrays::DataType::*;
    use crate::arrays::sharding::meshes::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::arrays::types::layouts::{Layout, StridedLayout, Tile, TileDimension, TiledLayout};
    use crate::parameters::{Parameter, ParameterError};

    use super::*;

    #[derive(Parameterized, Clone, Debug, PartialEq, Eq)]
    enum BroadcastParameters<P: Parameter> {
        Wrapped { inner: P },
        Pair { left: P, right: P },
    }

    #[test]
    fn test_data_type_broadcast() {
        assert_eq!(Boolean.broadcast(&U16), Ok(U16));
        assert_eq!(U16.broadcast(&Boolean), Ok(U16));
        assert!(
            matches!(F8E3M4.broadcast(&F32), Err(BroadcastingError::IncompatibleDataTypes(DataTypeError::InvalidPromotion { message, .. })) if message == "cannot promote types `f8e3m4` and `f32` to a common type"),
        );
    }

    #[test]
    fn test_data_type_broadcast_to() {
        assert_eq!(Boolean.broadcast_to(&U16), Ok(U16));
        assert!(
            matches!(F64.broadcast_to(&I32), Err(BroadcastingError::IncompatibleDataTypes(DataTypeError::InvalidPromotion { message, .. })) if message == "cannot promote type `f64` to type `i32`"),
        );
    }

    #[test]
    fn test_data_type_broadcasted() {
        assert_eq!(DataType::broadcasted(&[&Boolean]), Ok(Boolean));
        assert_eq!(DataType::broadcasted(&[&Boolean, &U16]), Ok(U16));
        assert!(matches!(DataType::broadcasted::<DataType>(&[]), Err(BroadcastingError::EmptyBroadcastingInput)));
        assert!(
            matches!(DataType::broadcasted(&[&F8E3M4, &F32]), Err(BroadcastingError::IncompatibleDataTypes(DataTypeError::InvalidPromotion { message, .. })) if message == "cannot promote types `f8e3m4` and `f32` to a common type"),
        );
    }

    #[test]
    fn test_data_type_is_broadcastable_to() {
        assert!(Boolean.is_broadcastable_to(&U16));
        assert!(!F64.is_broadcastable_to(&I32));
    }

    #[test]
    fn test_shape_broadcast() {
        let matrix = Shape::new(vec![42.into(), 4.into()]);
        let row = Shape::new(vec![1.into(), 4.into()]);
        let scalar = Shape::scalar();
        let incompatible_matrix = Shape::new(vec![5.into(), 3.into()]);

        // Leading rank promotion and singleton expansion combine independently.
        assert_eq!(row.broadcast(&Shape::new(vec![42.into(), 1.into()])), Ok(matrix.clone()));
        assert_eq!(row.broadcast(&scalar), Ok(row.clone()));
        assert_eq!(scalar.broadcast(&row), Ok(row.clone()));
        assert_eq!(
            matrix.broadcast(&incompatible_matrix),
            Err(BroadcastingError::IncompatibleShapes { lhs: matrix.clone(), rhs: incompatible_matrix.clone() }),
        );
    }

    #[test]
    fn test_shape_broadcast_zero_and_dynamic_dimensions() {
        let singleton = Shape::new(vec![1.into()]);
        let empty = Shape::new(vec![0.into()]);
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let dynamic = Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("extent", bounds))]);
        let other_dynamic = Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("other_extent", bounds))]);

        // Singleton expansion accepts zero extents and symbols, but bounds do not establish symbol identity.
        assert_eq!(singleton.broadcast(&empty), Ok(empty.clone()));
        assert_eq!(empty.broadcast(&singleton), Ok(empty.clone()));
        assert_eq!(singleton.broadcast(&dynamic), Ok(dynamic.clone()));
        assert_eq!(dynamic.broadcast(&dynamic), Ok(dynamic.clone()));
        assert_eq!(
            dynamic.broadcast(&other_dynamic),
            Err(BroadcastingError::IncompatibleShapes { lhs: dynamic.clone(), rhs: other_dynamic }),
        );
        assert_eq!(dynamic.broadcast(&empty), Err(BroadcastingError::IncompatibleShapes { lhs: dynamic, rhs: empty }));
    }

    #[test]
    fn test_shape_broadcast_to() {
        let row = Shape::new(vec![1.into(), 4.into()]);
        let scalar = Shape::scalar();

        // Directional broadcasting adopts target shape and layout.
        assert_eq!(scalar.broadcast_to(&row), Ok(row.clone()));
        assert_eq!(
            row.broadcast_to(&scalar),
            Err(BroadcastingError::IncompatibleShapes { lhs: row.clone(), rhs: scalar.clone() }),
        );
    }

    #[test]
    fn test_shape_broadcast_to_zero_and_dynamic_dimensions() {
        let singleton = Shape::new(vec![1.into()]);
        let empty = Shape::new(vec![0.into()]);
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let dynamic = Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("extent", bounds))]);
        let other_dynamic = Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("other_extent", bounds))]);

        // Singleton expansion accepts zero extents and symbols, but bounds do not establish symbol identity.
        assert_eq!(singleton.broadcast_to(&empty), Ok(empty.clone()));
        assert_eq!(singleton.broadcast_to(&dynamic), Ok(dynamic.clone()));
        assert_eq!(dynamic.broadcast_to(&dynamic), Ok(dynamic.clone()));
        assert_eq!(
            empty.broadcast_to(&singleton),
            Err(BroadcastingError::IncompatibleShapes { lhs: empty, rhs: singleton.clone() }),
        );
        assert_eq!(
            dynamic.broadcast_to(&singleton),
            Err(BroadcastingError::IncompatibleShapes { lhs: dynamic.clone(), rhs: singleton }),
        );
        assert_eq!(
            dynamic.broadcast_to(&other_dynamic),
            Err(BroadcastingError::IncompatibleShapes { lhs: dynamic, rhs: other_dynamic }),
        );
    }

    #[test]
    fn test_shape_broadcasted() {
        let matrix = Shape::new(vec![42.into(), 4.into()]);
        let row = Shape::new(vec![1.into(), 4.into()]);
        let scalar = Shape::scalar();
        let incompatible_matrix = Shape::new(vec![5.into(), 3.into()]);

        assert_eq!(Shape::broadcasted(&[&matrix]), Ok(matrix.clone()));
        assert_eq!(Shape::broadcasted(&[&row, &scalar]), Ok(row.clone()));
        assert_eq!(Shape::broadcasted(&[&scalar, &row]), Ok(row.clone()));
        assert!(matches!(Shape::broadcasted::<Shape>(&[]), Err(BroadcastingError::EmptyBroadcastingInput)));
        assert_eq!(
            Shape::broadcasted(&[&matrix, &incompatible_matrix]),
            Err(BroadcastingError::IncompatibleShapes { lhs: matrix.clone(), rhs: incompatible_matrix.clone() }),
        );
    }

    #[test]
    fn test_shape_is_broadcastable_to() {
        let matrix = Shape::new(vec![42.into(), 4.into()]);
        let row = Shape::new(vec![1.into(), 4.into()]);
        let scalar = Shape::scalar();
        let incompatible_matrix = Shape::new(vec![5.into(), 3.into()]);

        assert!(scalar.is_broadcastable_to(&row));
        assert!(!matrix.is_broadcastable_to(&incompatible_matrix));
    }

    #[test]
    fn test_shape_is_broadcastable_to_zero_and_dynamic_dimensions() {
        let singleton = Shape::new(vec![1.into()]);
        let empty = Shape::new(vec![0.into()]);
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let dynamic = Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("extent", bounds))]);
        let other_dynamic = Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("other_extent", bounds))]);

        // Singleton expansion accepts zero extents and symbols, but bounds do not establish symbol identity.
        assert!(singleton.is_broadcastable_to(&empty));
        assert!(singleton.is_broadcastable_to(&dynamic));
        assert!(dynamic.is_broadcastable_to(&dynamic));
        assert!(!empty.is_broadcastable_to(&singleton));
        assert!(!dynamic.is_broadcastable_to(&singleton));
        assert!(!dynamic.is_broadcastable_to(&other_dynamic));
    }

    #[test]
    fn test_array_type_broadcast() {
        let matrix = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let row = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()]));
        let scalar = ArrayType::scalar(Boolean);
        let incompatible_matrix = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));

        assert_eq!(row.broadcast(&scalar), Ok(row.clone()));
        assert_eq!(scalar.broadcast(&row), Ok(row.clone()));
        assert_eq!(
            matrix.broadcast(&incompatible_matrix),
            Err(BroadcastingError::IncompatibleShapes {
                lhs: incompatible_matrix.shape().clone(),
                rhs: matrix.shape().clone()
            }),
        );
    }

    #[test]
    fn test_array_type_broadcast_layout() {
        let tiled_layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![Tile::new(vec![TileDimension::Sized(4)])]));
        let strided_layout = Layout::Strided(StridedLayout::new(vec![16, 4]));
        let matrix = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let tiled_matrix = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()])).with_layout(tiled_layout.clone());
        let same_tiled_matrix =
            ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()])).with_layout(tiled_layout.clone());
        let strided_matrix = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()])).with_layout(strided_layout);
        let tiled_row = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()])).with_layout(tiled_layout);

        // Agreeing layouts survive; disagreement or a missing layout leaves the result unspecified.
        assert_eq!(tiled_matrix.broadcast(&same_tiled_matrix), Ok(tiled_matrix.clone()));
        assert_eq!(tiled_matrix.broadcast(&strided_matrix), Ok(matrix.clone()));
        assert_eq!(tiled_row.broadcast(&matrix), Ok(matrix.clone()));
    }

    #[test]
    fn test_array_type_broadcast_memory() {
        let device = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let pinned_host = device.clone().with_memory(Memory::Host { pinned: true });
        let pinned_host_scalar = ArrayType::scalar(F32).with_memory(Memory::Host { pinned: true });

        // Broadcasting preserves placement; its reversed retry determines the error operand order.
        assert_eq!(pinned_host.broadcast(&pinned_host_scalar), Ok(pinned_host.clone()));
        assert_eq!(
            device.broadcast(&pinned_host),
            Err(BroadcastingError::IncompatibleMemories { lhs: Memory::Host { pinned: true }, rhs: Memory::Device }),
        );
    }

    #[test]
    fn test_array_type_broadcast_sharding() {
        let manual_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let single_axis_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let two_axis_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let manual_sharding = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_varying_manual_axes(["x"])
            .unwrap();
        let same_manual_sharding = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_varying_manual_axes(["x"])
            .unwrap();
        let different_manual_sharding = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_varying_manual_axes(["y"])
            .unwrap();
        let vector_sharding = Sharding::new(single_axis_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let column_sharding = Sharding::new(
            single_axis_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let row_sharding =
            Sharding::new(single_axis_mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let outer_column_sharding = Sharding::new(
            two_axis_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let outer_row_sharding = Sharding::new(
            two_axis_mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])],
        )
        .unwrap();
        let singleton_sharding = Sharding::new(two_axis_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let conflicting_vector_sharding =
            Sharding::new(two_axis_mesh, vec![ShardingDimension::sharded(["y"])]).unwrap();
        let manual_vector = ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(manual_sharding).unwrap();
        let same_manual_vector =
            ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(same_manual_sharding).unwrap();
        let different_manual_vector =
            ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(different_manual_sharding).unwrap();
        let sharded_vector = ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(vector_sharding).unwrap();
        let unsharded_matrix = ArrayType::new(F32, Shape::new(vec![4.into(), 8.into()]));
        let sharded_column =
            ArrayType::new(F32, Shape::new(vec![4.into(), 1.into()])).with_sharding(column_sharding).unwrap();
        let sharded_row =
            ArrayType::new(F32, Shape::new(vec![1.into(), 8.into()])).with_sharding(row_sharding).unwrap();
        let outer_column = ArrayType::new(F32, Shape::new(vec![4.into(), 1.into()]))
            .with_sharding(outer_column_sharding)
            .unwrap();
        let outer_row =
            ArrayType::new(F32, Shape::new(vec![1.into(), 8.into()])).with_sharding(outer_row_sharding).unwrap();
        let sharded_singleton =
            ArrayType::new(F32, Shape::new(vec![1.into()])).with_sharding(singleton_sharding).unwrap();
        let conflicting_vector =
            ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(conflicting_vector_sharding).unwrap();
        let unsharded_vector = ArrayType::new(F32, Shape::new(vec![8.into()]));

        // Manual-axis variation must agree when both operands carry sharding.
        assert_eq!(
            manual_vector.broadcast(&same_manual_vector).map(|output| output
                .sharding()
                .unwrap()
                .varying_manual_axes()
                .clone()),
            Ok(BTreeSet::from(["x".to_string()])),
        );
        assert_eq!(
            manual_vector.broadcast(&different_manual_vector),
            Err(BroadcastingError::IncompatibleShardings {
                lhs: different_manual_vector.sharding().cloned().map(Box::new),
                rhs: manual_vector.sharding().cloned().map(Box::new)
            }),
        );
        // Rank promotion pads shardings with replication; a mesh axis cannot serve two output dimensions.
        assert_eq!(
            sharded_vector
                .broadcast(&unsharded_matrix)
                .map(|output| output.sharding().unwrap().dimensions().to_vec()),
            Ok(vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]),
        );
        assert_eq!(
            sharded_column.broadcast(&sharded_row),
            Err(BroadcastingError::IncompatibleShardings {
                lhs: sharded_row.sharding().cloned().map(Box::new),
                rhs: sharded_column.sharding().cloned().map(Box::new)
            }),
        );
        // Independent mesh axes combine, while non-replicated singleton axes retain their constraints.
        assert_eq!(
            outer_column.broadcast(&outer_row).map(|output| output.sharding().unwrap().dimensions().to_vec()),
            Ok(vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])]),
        );
        assert_eq!(
            sharded_singleton.broadcast(&unsharded_vector).map(|output| output
                .sharding()
                .unwrap()
                .dimensions()
                .to_vec()),
            Ok(vec![ShardingDimension::sharded(["x"])]),
        );
        assert_eq!(
            sharded_singleton.broadcast(&conflicting_vector),
            Err(BroadcastingError::IncompatibleShardings {
                lhs: conflicting_vector.sharding().cloned().map(Box::new),
                rhs: sharded_singleton.sharding().cloned().map(Box::new)
            }),
        );
    }

    #[test]
    fn test_array_type_broadcast_to() {
        let tiled_layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![Tile::new(vec![TileDimension::Sized(4)])]));
        let manual_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let single_axis_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let reduced_replicated_sharding =
            Sharding::new(manual_mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_reduced_axes(["y"])
                .unwrap();
        let reduced_sharding = Sharding::new(
            manual_mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap()
        .with_reduced_axes(["y"])
        .unwrap();
        let unreduced_sharding = Sharding::new(
            manual_mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let vector_manual_sharding = Sharding::new(manual_mesh, vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_varying_manual_axes(["x"])
            .unwrap();
        let vector_sharding = Sharding::new(single_axis_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let matrix = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let row = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()]));
        let scalar = ArrayType::scalar(Boolean);
        let incompatible_matrix = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));
        let tiled_matrix = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()])).with_layout(tiled_layout.clone());
        let reduced_row = ArrayType::new(F32, Shape::new(vec![1.into(), 8.into()]))
            .with_sharding(reduced_replicated_sharding)
            .unwrap();
        let reduced_matrix =
            ArrayType::new(F32, Shape::new(vec![2.into(), 8.into()])).with_sharding(reduced_sharding).unwrap();
        let unreduced_matrix =
            ArrayType::new(F32, Shape::new(vec![2.into(), 8.into()])).with_sharding(unreduced_sharding).unwrap();
        let manual_target =
            ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(vector_manual_sharding).unwrap();
        let sharded_vector = ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(vector_sharding).unwrap();
        let unsharded_matrix = ArrayType::new(F32, Shape::new(vec![4.into(), 8.into()]));

        // Directional broadcasting adopts target shape and layout.
        assert_eq!(scalar.broadcast_to(&row), Ok(row.clone()));
        assert_eq!(scalar.broadcast_to(&tiled_matrix), Ok(tiled_matrix.clone()));
        assert_eq!(
            matrix.broadcast_to(&incompatible_matrix),
            Err(BroadcastingError::IncompatibleShapes {
                lhs: matrix.shape().clone(),
                rhs: incompatible_matrix.shape().clone()
            }),
        );
        // Reduction metadata must agree, and compatible sharding survives rank promotion.
        assert_eq!(
            reduced_row
                .broadcast_to(&reduced_matrix)
                .map(|output| output.sharding().unwrap().reduced_axes().clone()),
            Ok(BTreeSet::from(["y".to_string()])),
        );
        assert_eq!(
            reduced_row.broadcast_to(&unreduced_matrix),
            Err(BroadcastingError::IncompatibleShardings {
                lhs: reduced_row.sharding().cloned().map(Box::new),
                rhs: unreduced_matrix.sharding().cloned().map(Box::new)
            }),
        );
        assert_eq!(
            scalar.broadcast_to(&manual_target).map(|output| output.sharding().unwrap().dimensions().to_vec()),
            Ok(vec![ShardingDimension::sharded(["x"])]),
        );
        assert_eq!(
            sharded_vector.broadcast_to(&unsharded_matrix).map(|output| output
                .sharding()
                .unwrap()
                .dimensions()
                .to_vec()),
            Ok(vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]),
        );
    }

    #[test]
    fn test_array_type_broadcast_to_memory() {
        let device = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let pinned_host = device.clone().with_memory(Memory::Host { pinned: true });
        let pinned_host_scalar = ArrayType::scalar(F32).with_memory(Memory::Host { pinned: true });

        assert_eq!(pinned_host_scalar.broadcast_to(&pinned_host), Ok(pinned_host.clone()));
        assert_eq!(
            device.broadcast_to(&pinned_host),
            Err(BroadcastingError::IncompatibleMemories { lhs: Memory::Device, rhs: Memory::Host { pinned: true } }),
        );
    }

    #[test]
    fn test_array_type_broadcasted() {
        let matrix = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let row = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()]));
        let scalar = ArrayType::scalar(Boolean);
        let incompatible_matrix = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));

        assert_eq!(ArrayType::broadcasted(&[&matrix]), Ok(matrix.clone()));
        assert_eq!(ArrayType::broadcasted(&[&row, &scalar]), Ok(row.clone()));
        assert_eq!(ArrayType::broadcasted(&[&scalar, &row]), Ok(row.clone()));
        assert!(matches!(ArrayType::broadcasted::<ArrayType>(&[]), Err(BroadcastingError::EmptyBroadcastingInput)));
        assert_eq!(
            ArrayType::broadcasted(&[&matrix, &incompatible_matrix]),
            Err(BroadcastingError::IncompatibleShapes {
                lhs: incompatible_matrix.shape().clone(),
                rhs: matrix.shape().clone()
            }),
        );
    }

    #[test]
    fn test_array_type_is_broadcastable_to() {
        let manual_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let single_axis_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let two_axis_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let manual_sharding = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_varying_manual_axes(["x"])
            .unwrap();
        let different_manual_sharding = Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_varying_manual_axes(["y"])
            .unwrap();
        let reduced_replicated_sharding =
            Sharding::new(manual_mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_reduced_axes(["y"])
                .unwrap();
        let reduced_sharding = Sharding::new(
            manual_mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap()
        .with_reduced_axes(["y"])
        .unwrap();
        let unreduced_sharding = Sharding::new(
            manual_mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let vector_manual_sharding = Sharding::new(manual_mesh, vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_varying_manual_axes(["x"])
            .unwrap();
        let vector_sharding = Sharding::new(single_axis_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let column_sharding = Sharding::new(
            single_axis_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let row_sharding =
            Sharding::new(single_axis_mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let singleton_sharding = Sharding::new(two_axis_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let matrix = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let row = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()]));
        let scalar = ArrayType::scalar(Boolean);
        let incompatible_matrix = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));
        let manual_vector = ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(manual_sharding).unwrap();
        let different_manual_vector =
            ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(different_manual_sharding).unwrap();
        let reduced_row = ArrayType::new(F32, Shape::new(vec![1.into(), 8.into()]))
            .with_sharding(reduced_replicated_sharding)
            .unwrap();
        let reduced_matrix =
            ArrayType::new(F32, Shape::new(vec![2.into(), 8.into()])).with_sharding(reduced_sharding).unwrap();
        let unreduced_matrix =
            ArrayType::new(F32, Shape::new(vec![2.into(), 8.into()])).with_sharding(unreduced_sharding).unwrap();
        let manual_target =
            ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(vector_manual_sharding).unwrap();
        let sharded_vector = ArrayType::new(F32, Shape::new(vec![8.into()])).with_sharding(vector_sharding).unwrap();
        let unsharded_matrix = ArrayType::new(F32, Shape::new(vec![4.into(), 8.into()]));
        let sharded_column =
            ArrayType::new(F32, Shape::new(vec![4.into(), 1.into()])).with_sharding(column_sharding).unwrap();
        let sharded_row =
            ArrayType::new(F32, Shape::new(vec![1.into(), 8.into()])).with_sharding(row_sharding).unwrap();
        let sharded_singleton =
            ArrayType::new(F32, Shape::new(vec![1.into()])).with_sharding(singleton_sharding).unwrap();
        let unsharded_vector = ArrayType::new(F32, Shape::new(vec![8.into()]));

        assert!(scalar.is_broadcastable_to(&row));
        assert!(!matrix.is_broadcastable_to(&incompatible_matrix));
        assert!(scalar.is_broadcastable_to(&manual_target));
        assert!(reduced_row.is_broadcastable_to(&reduced_matrix));
        assert!(sharded_vector.is_broadcastable_to(&unsharded_matrix));
        assert!(sharded_singleton.is_broadcastable_to(&unsharded_vector));
        assert!(!manual_vector.is_broadcastable_to(&different_manual_vector));
        assert!(!reduced_row.is_broadcastable_to(&unreduced_matrix));
        assert!(!sharded_column.is_broadcastable_to(&sharded_row));
    }

    #[test]
    fn test_array_type_is_broadcastable_to_memory() {
        let device = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let pinned_host = device.clone().with_memory(Memory::Host { pinned: true });
        let pinned_host_scalar = ArrayType::scalar(F32).with_memory(Memory::Host { pinned: true });

        assert!(pinned_host_scalar.is_broadcastable_to(&pinned_host));
        assert!(!device.is_broadcastable_to(&pinned_host));
        assert!(!pinned_host.is_broadcastable_to(&device));
    }

    #[test]
    fn test_parameterized_array_type_broadcast() {
        let source = BroadcastParameters::Pair {
            left: ArrayType::scalar(F32),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()])),
        };
        let target = BroadcastParameters::Pair {
            left: ArrayType::new(F64, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F64, Shape::new(vec![3.into(), 4.into()])),
        };
        let incompatible_target = BroadcastParameters::Pair {
            left: ArrayType::new(F32, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 3.into()])),
        };
        let wrapped = BroadcastParameters::Wrapped { inner: ArrayType::scalar(F32) };

        assert_eq!(source.broadcast(&target), Ok(target.clone()));
        assert!(matches!(
            wrapped.broadcast(&incompatible_target),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));
    }

    #[test]
    fn test_parameterized_array_type_broadcast_to() {
        let source = BroadcastParameters::Pair {
            left: ArrayType::scalar(F32),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()])),
        };
        let target = BroadcastParameters::Pair {
            left: ArrayType::new(F64, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F64, Shape::new(vec![3.into(), 4.into()])),
        };
        let incompatible_target = BroadcastParameters::Pair {
            left: ArrayType::new(F32, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 3.into()])),
        };
        let wrapped = BroadcastParameters::Wrapped { inner: ArrayType::scalar(F32) };

        assert_eq!(source.broadcast_to(&target), Ok(target.clone()));
        assert!(matches!(
            wrapped.broadcast_to(&incompatible_target),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));
    }

    #[test]
    fn test_parameterized_array_type_broadcasted() {
        let source = BroadcastParameters::Pair {
            left: ArrayType::scalar(F32),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()])),
        };
        let target = BroadcastParameters::Pair {
            left: ArrayType::new(F64, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F64, Shape::new(vec![3.into(), 4.into()])),
        };
        let incompatible_target = BroadcastParameters::Pair {
            left: ArrayType::new(F32, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 3.into()])),
        };
        let wrapped = BroadcastParameters::Wrapped { inner: ArrayType::scalar(F32) };

        assert_eq!(BroadcastParameters::broadcasted(&[&source]), Ok(source.clone()));
        assert_eq!(
            BroadcastParameters::broadcasted(&[&source, &target]),
            Ok(BroadcastParameters::Pair {
                left: ArrayType::new(F64, Shape::new(vec![2.into(), 1.into()])),
                right: ArrayType::new(F64, Shape::new(vec![3.into(), 4.into()])),
            }),
        );
        assert!(matches!(
            BroadcastParameters::broadcasted(&[&wrapped, &incompatible_target]),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));
    }

    #[test]
    fn test_parameterized_array_type_is_broadcastable_to() {
        let source = BroadcastParameters::Pair {
            left: ArrayType::scalar(F32),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()])),
        };
        let target = BroadcastParameters::Pair {
            left: ArrayType::new(F64, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F64, Shape::new(vec![3.into(), 4.into()])),
        };
        let incompatible_target = BroadcastParameters::Pair {
            left: ArrayType::new(F32, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 3.into()])),
        };
        let wrapped = BroadcastParameters::Wrapped { inner: ArrayType::scalar(F32) };

        assert!(source.is_broadcastable_to(&target));
        assert!(!wrapped.is_broadcastable_to(&incompatible_target));
    }
}
