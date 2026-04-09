use std::collections::BTreeSet;

use thiserror::Error;

use crate::{
    parameters::{ParameterError, Parameterized},
    sharding::{Sharding, ShardingDimension, ShardingError},
    types::data_types::DataTypeError,
    types::{ArrayType, DataType, Shape, Size},
};

/// Represents broadcasting-related errors.
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BroadcastingError {
    #[error("cannot broadcast an empty collection of types")]
    EmptyBroadcastingInput,

    #[error("failed to broadcast due to incompatible data types; {0}")]
    IncompatibleDataTypes(#[from] DataTypeError),

    #[error("failed to broadcast shape `{lhs}` to shape `{rhs}`")]
    IncompatibleShapes { lhs: Shape, rhs: Shape },

    #[error("failed to broadcast due to incompatible shardings; lhs={lhs:?}, rhs={rhs:?}")]
    IncompatibleShardings { lhs: Option<Sharding>, rhs: Option<Sharding> },

    #[error("failed to reconstruct the parameterized structure after broadcasting; {0}")]
    ParameterError(#[from] ParameterError),

    #[error("failed to broadcast sharding information; {0}")]
    ShardingError(#[from] ShardingError),
}

/// Represents [`Type`](crate::types::Type)s or values that can be broadcast together.
///
/// Broadcasting in Ryft has two orthogonal components:
///
///   - **Parameter Broadcasting:** Each concrete implementer defines what it means to combine two
///     [`Parameter`](crate::parameters::Parameter)s. For example, [`DataType`] uses data-type promotion, [`Shape`]
///     follows the standard [NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html),
///     and [`ArrayType`] combines both by broadcasting its [`DataType`] and [`Shape`].
///     [`Layout`](crate::types::layouts::Layout) metadata is only preserved when both operands already agree on the
///     same unchanged layout; otherwise the result leaves its layout unspecified.
///   - **Structural Broadcasting:** For [`Parameterized`] values whose leaves are [`ArrayType`]s, Ryft first aligns
///     the left-hand side to the target parameter structure using [`Parameterized::broadcast_to_parameter_structure`].
///     That alignment uses path-prefix broadcasting on named parameters, so a value with a smaller compatible parameter
///     structure can be broadcast into a larger one. Once the structures are aligned, leaf broadcasting is applied
///     pairwise and the final structured value is reconstructed.
///
/// ## NumPy-style Broadcasting Semantics
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
/// ## Examples
///
/// ```rust
/// # use ryft_core::broadcasting::{Broadcastable, BroadcastingError};
/// # use ryft_core::types::data_types::DataType::{Boolean, F32, F64};
/// # use ryft_core::types::{ArrayType, Shape};
/// let x = Shape::new(vec![4.into(), 3.into()]);
/// let y = Shape::new(vec![3.into()]);
/// let z = Shape::new(vec![4.into(), 1.into()]);
/// let w = Shape::new(vec![4.into()]);
///
/// assert_eq!(x.broadcast(&y)?, x);
/// assert_eq!(y.broadcast_to(&x)?, x);
/// assert_eq!(z.broadcast(&y)?, Shape::new(vec![4.into(), 3.into()]));
/// assert!(w.broadcast(&x).is_err());
///
/// let lhs = (ArrayType::scalar(Boolean), ArrayType::new(F32, Shape::new(vec![1.into(), 3.into()]), None, None)?);
/// let rhs = (
///     ArrayType::new(F32, Shape::new(vec![2.into(), 3.into()]), None, None)?,
///     ArrayType::new(F64, Shape::new(vec![2.into(), 1.into()]), None, None)?,
/// );
///
/// assert_eq!(
///     lhs.broadcast(&rhs)?,
///     (
///         ArrayType::new(F32, Shape::new(vec![2.into(), 3.into()]), None, None)?,
///         ArrayType::new(F64, Shape::new(vec![2.into(), 3.into()]), None, None)?,
///     ),
/// );
/// # Ok::<(), BroadcastingError>(())
/// ```
pub trait Broadcastable: Sized {
    /// Broadcasts this value with `other` and returns the least common result that both values can broadcast to.
    /// This operation is _symmetric_. For example, with [`Shape`] values this returns the smallest shape that both
    /// operands can broadcast to, and with [`ArrayType`] values it combines [`DataType`] promotion with [`Shape`]
    /// broadcasting.
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError>;

    /// Broadcasts this value to the provided `other` value. Unlike [`Broadcastable::broadcast`], this operation
    /// is _not symmetric_: `x.broadcast_to(y)` and `y.broadcast_to(x)` may differ and one of them may even fail.
    fn broadcast_to(&self, other: &Self) -> Result<Self, BroadcastingError>;

    /// Broadcasts the provided values into a single value by folding over [`Broadcastable::broadcast`] from left to
    /// right. Returns [`BroadcastingError::EmptyBroadcastingInput`] if no values are provided.
    fn broadcasted(values: &[&Self]) -> Result<Self, BroadcastingError>
    where
        Self: Clone,
    {
        let (head, tail) = values.split_first().ok_or(BroadcastingError::EmptyBroadcastingInput)?;
        tail.iter().try_fold((*head).clone(), |accumulator, value| accumulator.broadcast(*value))
    }

    /// Returns `true` if this value can be broadcast to `other`, and `false` otherwise.
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
            let self_size = if i < self_offset { Size::Static(1) } else { self.dimensions[i - self_offset] };
            let other_size = if i < other_offset { Size::Static(1) } else { other.dimensions[i - other_offset] };
            let broadcasted_size = match (&self_size, &other_size) {
                (_, Size::Static(1)) => self_size,
                (Size::Static(1), _) => other_size,
                (Size::Static(x), Size::Static(y)) if x == y => Size::Static(*x),
                (Size::Dynamic(x), Size::Dynamic(y)) if x == y => Size::Dynamic(*x),
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
            let self_size = if i < offset { Size::Static(1) } else { self.dimensions[i - offset] };
            let other_size = other.dimensions[i];
            let broadcasted_size = match (&self_size, &other_size) {
                (Size::Static(1), _) => other_size,
                (Size::Static(x), Size::Static(y)) if x == y => Size::Static(*y),
                (Size::Dynamic(x), Size::Dynamic(y)) if x == y => Size::Dynamic(*y),
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
            let self_size = if i < offset { Size::Static(1) } else { self.dimensions[i - offset] };
            let other_size = other.dimensions[i];
            match (&self_size, &other_size) {
                (Size::Static(1), _) => continue,
                (Size::Static(x), Size::Static(y)) if x == y => continue,
                (Size::Dynamic(x), Size::Dynamic(y)) if x == y => continue,
                _ => return false,
            };
        }

        true
    }
}

impl<T: Parameterized<ArrayType, ParameterStructure: Clone>> Broadcastable for T {
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError> {
        let broadcast_to = |lhs: &Self, rhs: &Self| -> Result<Self, BroadcastingError> {
            let structure = rhs.parameter_structure();
            let broadcasted_array_types = lhs
                .broadcast_to_parameter_structure::<T>(structure.clone())?
                .parameters()
                .zip(rhs.parameters())
                .map(|(lhs, rhs)| {
                    let broadcasted_data_type = lhs.data_type.broadcast(&rhs.data_type)?;
                    let broadcasted_shape = lhs.shape.broadcast(&rhs.shape)?;
                    let broadcasted_layout = (lhs.layout == rhs.layout).then(|| lhs.layout.clone()).flatten();
                    let broadcasted_sharding = broadcast_sharding(
                        &lhs.shape,
                        lhs.sharding.as_ref(),
                        &rhs.shape,
                        rhs.sharding.as_ref(),
                        &broadcasted_shape,
                    )?;
                    Ok(ArrayType::new(
                        broadcasted_data_type,
                        broadcasted_shape,
                        broadcasted_layout,
                        broadcasted_sharding,
                    )?)
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
                let broadcasted_data_type = lhs.data_type.broadcast_to(&rhs.data_type)?;
                let broadcasted_shape = lhs.shape.broadcast_to(&rhs.shape)?;
                let broadcasted_sharding = broadcast_sharding(
                    &lhs.shape,
                    lhs.sharding.as_ref(),
                    &rhs.shape,
                    rhs.sharding.as_ref(),
                    &broadcasted_shape,
                )?;
                Ok(ArrayType::new(broadcasted_data_type, broadcasted_shape, rhs.layout.clone(), broadcasted_sharding)?)
            })
            .collect::<Result<Vec<_>, BroadcastingError>>()?;
        Ok(Self::from_parameters(structure, broadcasted_array_types)?)
    }

    fn is_broadcastable_to(&self, other: &Self) -> bool {
        let Ok(broadcasted_self) = self.broadcast_to_parameter_structure::<T>(other.parameter_structure()) else {
            return false;
        };
        broadcasted_self.parameters().zip(other.parameters()).all(|(lhs, rhs)| {
            lhs.data_type.is_broadcastable_to(&rhs.data_type)
                && lhs.shape.is_broadcastable_to(&rhs.shape)
                && is_sharding_broadcastable_to(&lhs.shape, lhs.sharding.as_ref(), &rhs.shape, rhs.sharding.as_ref())
        })
    }
}

/// Broadcasts an optional [`Sharding`] paired with a [`Shape`] to another optional [`Sharding`] paired with a [`Shape`]
/// and returns the resulting [`Sharding`], using the following broadcasting rules:
///
///   - If neither operand carries sharding information, then this function returns no sharding information.
///   - Any provided [`Sharding`] must already have the same rank as its source [`Shape`]. Mismatched shardings will be
///     rejected before any broadcast-specific logic is applied.
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
///   - Both operands must use the same [`LogicalMesh`](crate::sharding::LogicalMesh) when they are both sharded. After
///     the per-axis dimensions are combined, reusing the same mesh axis across multiple result dimensions is treated as
///     an incompatible broadcast, and for those cases, this function will return a [`BroadcastingError`].
///   - The [`Sharding::unreduced_axes`], [`Sharding::reduced_manual_axes`], and [`Sharding::varying_manual_axes`] sets
///     are only preserved when both inputs already agree on them, or when only one operand carries sharding
///     information. Generic [`ArrayType`] broadcasting does not attempt primitive-specific manual-axis merges.
///
/// # Parameters
///
///   - `lhs_shape`: [`Shape`] of the left-hand operand before broadcasting.
///   - `lhs_sharding`: Optional [`Sharding`] for the left-hand operand.
///   - `rhs_shape`: [`Shape`] of the right-hand operand before broadcasting.
///   - `rhs_sharding`: Optional [`Sharding`] for the right-hand operand.
///   - `broadcasted_shape`: Result of broadcasting `lhs_shape` to `rhs_shape`.
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
        (Some(left), None) => left.mesh.clone(),
        (None, Some(right)) => right.mesh.clone(),
        (Some(left), Some(right)) if left.mesh == right.mesh => left.mesh.clone(),
        (Some(left), Some(right)) => {
            return Err(BroadcastingError::IncompatibleShardings { lhs: Some(left.clone()), rhs: Some(right.clone()) });
        }
    };

    let result_rank = broadcasted_shape.rank();
    let lhs_offset = result_rank - lhs_shape.rank();
    let rhs_offset = result_rank - rhs_shape.rank();

    let mut used_axes = BTreeSet::new();
    let mut broadcasted_dimensions = Vec::with_capacity(result_rank);
    for index in 0..result_rank {
        let lhs_size = if index < lhs_offset { Size::Static(1) } else { lhs_shape.dimensions[index - lhs_offset] };
        let rhs_size = if index < rhs_offset { Size::Static(1) } else { rhs_shape.dimensions[index - rhs_offset] };
        let lhs_dimension = padded_sharding_dimension(lhs_sharding, lhs_offset, index);
        let rhs_dimension = padded_sharding_dimension(rhs_sharding, rhs_offset, index);
        let Some(dimension) = broadcast_sharding_dimension(lhs_size, lhs_dimension, rhs_size, rhs_dimension) else {
            return Err(BroadcastingError::IncompatibleShardings {
                lhs: lhs_sharding.cloned(),
                rhs: rhs_sharding.cloned(),
            });
        };
        if let ShardingDimension::Sharded(axis_names) = dimension {
            for axis_name in axis_names {
                if !used_axes.insert(axis_name.clone()) {
                    return Err(BroadcastingError::IncompatibleShardings {
                        lhs: lhs_sharding.cloned(),
                        rhs: rhs_sharding.cloned(),
                    });
                }
            }
        }
        broadcasted_dimensions.push(dimension.clone());
    }

    let unreduced_axes = match (lhs_sharding, rhs_sharding) {
        (None, None) => BTreeSet::new(),
        (Some(left), None) => left.unreduced_axes.clone(),
        (None, Some(right)) => right.unreduced_axes.clone(),
        (Some(left), Some(right)) if left.unreduced_axes == right.unreduced_axes => left.unreduced_axes.clone(),
        (Some(_), Some(_)) => {
            return Err(BroadcastingError::IncompatibleShardings {
                lhs: lhs_sharding.cloned(),
                rhs: rhs_sharding.cloned(),
            });
        }
    };

    let reduced_manual_axes = match (lhs_sharding, rhs_sharding) {
        (None, None) => BTreeSet::new(),
        (Some(left), None) => left.reduced_manual_axes.clone(),
        (None, Some(right)) => right.reduced_manual_axes.clone(),
        (Some(left), Some(right)) if left.reduced_manual_axes == right.reduced_manual_axes => {
            left.reduced_manual_axes.clone()
        }
        (Some(_), Some(_)) => {
            return Err(BroadcastingError::IncompatibleShardings {
                lhs: lhs_sharding.cloned(),
                rhs: rhs_sharding.cloned(),
            });
        }
    };

    let varying_manual_axes = match (lhs_sharding, rhs_sharding) {
        (None, None) => BTreeSet::new(),
        (Some(left), None) => left.varying_manual_axes.clone(),
        (None, Some(right)) => right.varying_manual_axes.clone(),
        (Some(left), Some(right)) if left.varying_manual_axes == right.varying_manual_axes => {
            left.varying_manual_axes.clone()
        }
        (Some(_), Some(_)) => {
            return Err(BroadcastingError::IncompatibleShardings {
                lhs: lhs_sharding.cloned(),
                rhs: rhs_sharding.cloned(),
            });
        }
    };

    Ok(Some(Sharding::with_manual_axes(
        mesh,
        broadcasted_dimensions,
        unreduced_axes,
        reduced_manual_axes,
        varying_manual_axes,
    )?))
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
        (Some(left), Some(right)) if left.mesh != right.mesh => return false,
        _ => {}
    }

    let result_rank = rhs_shape.rank();
    let lhs_offset = result_rank - lhs_shape.rank();
    let rhs_offset = result_rank - rhs_shape.rank();
    let mut used_axes = BTreeSet::new();
    for index in 0..result_rank {
        let lhs_size = if index < lhs_offset { Size::Static(1) } else { lhs_shape.dimensions[index - lhs_offset] };
        let rhs_size = if index < rhs_offset { Size::Static(1) } else { rhs_shape.dimensions[index - rhs_offset] };
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
            left.unreduced_axes == right.unreduced_axes
                && left.reduced_manual_axes == right.reduced_manual_axes
                && left.varying_manual_axes == right.varying_manual_axes
        }
        _ => true,
    }
}

/// Returns the [`ShardingDimension`] visible at `index` after left-padding lower-rank shardings
/// with [`ShardingDimension::Replicated`] axes.
#[inline]
fn padded_sharding_dimension(sharding: Option<&Sharding>, offset: usize, index: usize) -> &ShardingDimension {
    (index < offset)
        .then_some(&ShardingDimension::Replicated)
        .or_else(|| sharding.map(|sharding| &sharding.dimensions[index - offset]))
        .unwrap_or(&ShardingDimension::Replicated)
}

/// Combines two aligned [`ShardingDimension`]s using the rules described in [`broadcast_sharding`].
#[inline]
fn broadcast_sharding_dimension<'d>(
    lhs_size: Size,
    lhs_dimension: &'d ShardingDimension,
    rhs_size: Size,
    rhs_dimension: &'d ShardingDimension,
) -> Option<&'d ShardingDimension> {
    match (lhs_dimension, rhs_dimension) {
        (lhs_dimension, rhs_dimension) if lhs_dimension == rhs_dimension => Some(lhs_dimension),
        (ShardingDimension::Replicated, rhs_dimension) if matches!(lhs_size, Size::Static(1)) => Some(rhs_dimension),
        (lhs_dimension, ShardingDimension::Replicated) if matches!(rhs_size, Size::Static(1)) => Some(lhs_dimension),
        (ShardingDimension::Replicated, rhs_dimension) => Some(rhs_dimension),
        (lhs_dimension, ShardingDimension::Replicated) => Some(lhs_dimension),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameterized;

    use crate::parameters::{Parameter, ParameterError};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::types::data_types::DataType::*;
    use crate::types::{Layout, Shape, StridedLayout, Tile, TileDimension, TiledLayout};

    use super::*;

    #[test]
    fn test_data_type_broadcasting() {
        assert_eq!(Boolean.broadcast(&U16), Ok(U16));
        assert_eq!(U16.broadcast(&Boolean), Ok(U16));
        assert!(matches!(F8E3M4.broadcast(&F32), Err(BroadcastingError::IncompatibleDataTypes(_))));

        assert_eq!(Boolean.broadcast_to(&U16), Ok(U16));
        assert!(matches!(F64.broadcast_to(&I32), Err(BroadcastingError::IncompatibleDataTypes(_))));

        assert!(Boolean.is_broadcastable_to(&U16));
        assert!(!F64.is_broadcastable_to(&I32));

        assert_eq!(DataType::broadcasted(&[&Boolean]), Ok(Boolean));
        assert_eq!(DataType::broadcasted(&[&Boolean, &U16]), Ok(U16));
        assert!(matches!(DataType::broadcasted(&[]), Err(BroadcastingError::EmptyBroadcastingInput)));
        assert!(matches!(DataType::broadcasted(&[&F8E3M4, &F32]), Err(BroadcastingError::IncompatibleDataTypes(_))));
    }

    #[test]
    fn test_shape_broadcasting() {
        let s0 = Shape::new(vec![42.into(), 4.into()]);
        let s1 = Shape::new(vec![1.into(), 4.into()]);
        let s2 = Shape::scalar();
        let s3 = Shape::new(vec![5.into(), 3.into()]);

        assert_eq!(s1.broadcast(&s2), Ok(s1.clone()));
        assert_eq!(s2.broadcast(&s1), Ok(s1.clone()));
        assert!(matches!(s0.broadcast(&s3), Err(BroadcastingError::IncompatibleShapes { .. })));

        assert_eq!(s2.broadcast_to(&s1), Ok(s1.clone()));
        assert!(matches!(s1.broadcast_to(&s2), Err(BroadcastingError::IncompatibleShapes { .. })));

        assert_eq!(Shape::broadcasted(&[&s0]), Ok(s0.clone()));
        assert_eq!(Shape::broadcasted(&[&s1, &s2]), Ok(s1.clone()));
        assert_eq!(Shape::broadcasted(&[&s2, &s1]), Ok(s1.clone()));
        assert!(matches!(Shape::broadcasted(&[]), Err(BroadcastingError::EmptyBroadcastingInput)));
        assert!(matches!(Shape::broadcasted(&[&s0, &s3]), Err(BroadcastingError::IncompatibleShapes { .. })));

        assert!(s2.is_broadcastable_to(&s1));
        assert!(!s0.is_broadcastable_to(&s3));
    }

    #[test]
    fn test_array_type_broadcasting() {
        let l0 = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![Tile::new(vec![TileDimension::Sized(4)])]));
        let l1 = Layout::Strided(StridedLayout::new(vec![16, 4]));

        let m0 = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let m1 = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let m2 = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();

        let s0 = Sharding::with_manual_axes(
            m0.clone(),
            vec![ShardingDimension::sharded(["x"])],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
            ["x"],
        )
        .unwrap();
        let s1 = Sharding::with_manual_axes(
            m0.clone(),
            vec![ShardingDimension::sharded(["x"])],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
            ["x"],
        )
        .unwrap();
        let s2 = Sharding::with_manual_axes(
            m0.clone(),
            vec![ShardingDimension::sharded(["x"])],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
            ["y"],
        )
        .unwrap();
        let s3 = Sharding::with_manual_axes(
            m0.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
            Vec::<&str>::new(),
            ["y"],
            Vec::<&str>::new(),
        )
        .unwrap();
        let s4 = Sharding::with_manual_axes(
            m0.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])],
            Vec::<&str>::new(),
            ["y"],
            Vec::<&str>::new(),
        )
        .unwrap();
        let s5 = Sharding::new(m0.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
            .unwrap();
        let s6 = Sharding::with_manual_axes(
            m0,
            vec![ShardingDimension::sharded(["x"])],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
            ["x"],
        )
        .unwrap();
        let s7 = Sharding::new(m1.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let s8 = Sharding::new(m1.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
            .unwrap();
        let s9 = Sharding::new(m1, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap();
        let s10 = Sharding::new(m2.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
            .unwrap();
        let s11 = Sharding::new(m2.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["y"])])
            .unwrap();
        let s12 = Sharding::new(m2.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let s13 = Sharding::new(m2, vec![ShardingDimension::sharded(["y"])]).unwrap();

        let t0 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]), None, None).unwrap();
        let t1 = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()]), None, None).unwrap();
        let t2 = ArrayType::scalar(Boolean);
        let t3 = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]), None, None).unwrap();
        let t4 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]), Some(l0.clone()), None).unwrap();
        let t5 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]), Some(l0.clone()), None).unwrap();
        let t6 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]), Some(l1), None).unwrap();
        let t7 = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()]), Some(l0.clone()), None).unwrap();
        let t8 = ArrayType::new(F32, Shape::new(vec![8.into()]), None, Some(s0)).unwrap();
        let t9 = ArrayType::new(F32, Shape::new(vec![8.into()]), None, Some(s1)).unwrap();
        let t10 = ArrayType::new(F32, Shape::new(vec![8.into()]), None, Some(s2)).unwrap();
        let t11 = ArrayType::new(F32, Shape::new(vec![1.into(), 8.into()]), None, Some(s3)).unwrap();
        let t12 = ArrayType::new(F32, Shape::new(vec![2.into(), 8.into()]), None, Some(s4)).unwrap();
        let t13 = ArrayType::new(F32, Shape::new(vec![2.into(), 8.into()]), None, Some(s5)).unwrap();
        let t14 = ArrayType::new(F32, Shape::new(vec![8.into()]), None, Some(s6)).unwrap();
        let t15 = ArrayType::new(F32, Shape::new(vec![8.into()]), None, Some(s7.clone())).unwrap();
        let t16 = ArrayType::new(F32, Shape::new(vec![4.into(), 8.into()]), None, None).unwrap();
        let t17 = ArrayType::new(F32, Shape::new(vec![4.into(), 1.into()]), None, Some(s8)).unwrap();
        let t18 = ArrayType::new(F32, Shape::new(vec![1.into(), 8.into()]), None, Some(s9)).unwrap();
        let t19 = ArrayType::new(F32, Shape::new(vec![4.into(), 1.into()]), None, Some(s10)).unwrap();
        let t20 = ArrayType::new(F32, Shape::new(vec![1.into(), 8.into()]), None, Some(s11)).unwrap();
        let t21 = ArrayType::new(F32, Shape::new(vec![1.into()]), None, Some(s12)).unwrap();
        let t22 = ArrayType::new(F32, Shape::new(vec![8.into()]), None, Some(s13)).unwrap();
        let t23 = ArrayType::new(F32, Shape::new(vec![8.into()]), None, None).unwrap();

        assert_eq!(t1.broadcast(&t2), Ok(t1.clone()));
        assert_eq!(t2.broadcast(&t1), Ok(t1.clone()));
        assert!(matches!(t0.broadcast(&t3), Err(BroadcastingError::IncompatibleShapes { .. })));
        assert_eq!(t4.broadcast(&t5), Ok(t4.clone()));
        assert_eq!(t4.broadcast(&t6), Ok(t0.clone()));
        assert_eq!(t7.broadcast(&t0), Ok(t0.clone()));
        assert_eq!(
            t8.broadcast(&t9).map(|output| output.sharding.unwrap().varying_manual_axes),
            Ok(BTreeSet::from(["x".to_string()]))
        );
        assert!(matches!(t8.broadcast(&t10), Err(BroadcastingError::IncompatibleShardings { .. })));
        assert_eq!(
            t15.broadcast(&t16).map(|output| output.sharding.unwrap().dimensions),
            Ok(vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
        );
        assert!(matches!(t17.broadcast(&t18), Err(BroadcastingError::IncompatibleShardings { .. })));
        assert_eq!(
            t19.broadcast(&t20).map(|output| output.sharding.unwrap().dimensions),
            Ok(vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
        );
        assert_eq!(
            t21.broadcast(&t23).map(|output| output.sharding.unwrap().dimensions),
            Ok(vec![ShardingDimension::sharded(["x"])])
        );
        assert!(matches!(t21.broadcast(&t22), Err(BroadcastingError::IncompatibleShardings { .. })));

        assert_eq!(t2.broadcast_to(&t1), Ok(t1.clone()));
        assert_eq!(t2.broadcast_to(&t4), Ok(t4.clone()));
        assert!(matches!(t0.broadcast_to(&t3), Err(BroadcastingError::IncompatibleShapes { .. })));
        assert_eq!(
            t11.broadcast_to(&t12).map(|output| output.sharding.unwrap().reduced_manual_axes),
            Ok(BTreeSet::from(["y".to_string()]))
        );
        assert!(matches!(t11.broadcast_to(&t13), Err(BroadcastingError::IncompatibleShardings { .. })));
        assert_eq!(
            t2.broadcast_to(&t14).map(|output| output.sharding.unwrap().dimensions),
            Ok(vec![ShardingDimension::sharded(["x"])])
        );
        assert_eq!(
            t15.broadcast_to(&t16).map(|output| output.sharding.unwrap().dimensions),
            Ok(vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
        );

        assert_eq!(ArrayType::broadcasted(&[&t0]), Ok(t0.clone()));
        assert_eq!(ArrayType::broadcasted(&[&t1, &t2]), Ok(t1.clone()));
        assert_eq!(ArrayType::broadcasted(&[&t2, &t1]), Ok(t1.clone()));
        assert!(matches!(ArrayType::broadcasted(&[]), Err(BroadcastingError::EmptyBroadcastingInput)));
        assert!(matches!(ArrayType::broadcasted(&[&t0, &t3]), Err(BroadcastingError::IncompatibleShapes { .. })));

        assert!(t2.is_broadcastable_to(&t1));
        assert!(!t0.is_broadcastable_to(&t3));
        assert!(t2.is_broadcastable_to(&t14));
        assert!(t11.is_broadcastable_to(&t12));
        assert!(t15.is_broadcastable_to(&t16));
        assert!(t21.is_broadcastable_to(&t23));
        assert!(!t8.is_broadcastable_to(&t10));
        assert!(!t11.is_broadcastable_to(&t13));
        assert!(!t17.is_broadcastable_to(&t18));
    }

    #[test]
    fn test_parameterized_array_type_broadcastable() {
        #[derive(Parameterized, Clone, Debug, Eq, PartialEq)]
        #[ryft(crate = "crate::parameters")]
        enum TestEnum<P: Parameter> {
            Wrapped { inner: P },
            Pair { left: P, right: P },
        }

        let t0 = TestEnum::Pair {
            left: ArrayType::scalar(F32),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()]), None, None).unwrap(),
        };

        let t1 = TestEnum::Pair {
            left: ArrayType::new(F64, Shape::new(vec![2.into(), 1.into()]), None, None).unwrap(),
            right: ArrayType::new(F64, Shape::new(vec![3.into(), 4.into()]), None, None).unwrap(),
        };

        let t2 = TestEnum::Pair {
            left: ArrayType::new(F32, Shape::new(vec![2.into(), 1.into()]), None, None).unwrap(),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 3.into()]), None, None).unwrap(),
        };

        let t3 = TestEnum::Wrapped { inner: ArrayType::scalar(F32) };

        assert_eq!(t0.broadcast(&t1), Ok(t1.clone()));
        assert!(matches!(
            t3.broadcast(&t2),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));

        assert_eq!(t0.broadcast_to(&t1), Ok(t1.clone()));
        assert!(matches!(
            t3.broadcast_to(&t2),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));

        assert_eq!(TestEnum::broadcasted(&[&t0]), Ok(t0.clone()));
        assert_eq!(
            TestEnum::broadcasted(&[&t0, &t1]),
            Ok(TestEnum::Pair {
                left: ArrayType::new(F64, Shape::new(vec![2.into(), 1.into()]), None, None).unwrap(),
                right: ArrayType::new(F64, Shape::new(vec![3.into(), 4.into()]), None, None).unwrap(),
            }),
        );
        assert!(matches!(
            TestEnum::broadcasted(&[&t3, &t2]),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));

        assert!(t0.is_broadcastable_to(&t1));
        assert!(!t3.is_broadcastable_to(&t2));
    }
}
