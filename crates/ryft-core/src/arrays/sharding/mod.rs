use thiserror::Error;

use crate::programs::{Type, TypeError};

pub mod meshes;
pub mod shardings;
pub mod visualizations;

pub use meshes::{Device, DeviceId, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, ProcessIndex};
pub use shardings::{Sharding, ShardingDimension};
pub use visualizations::ShardingVisualization;

/// Represents sharding-related errors.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShardingError {
    #[error("mesh axis names must not be empty")]
    EmptyMeshAxisName,

    #[error("unknown mesh axis name: `{name}`")]
    UnknownMeshAxisName { name: String },

    #[error("mesh axis name `{name}` appears more than once")]
    DuplicateMeshAxisName { name: String },

    #[error("mesh axis `{name}` must have size > 0")]
    EmptyMeshAxis { name: String },

    #[error("mesh axis `{name}` must have type manual")]
    ExpectedManualMeshAxis { name: String },

    #[error("cannot remove dimension {dimension} because it is sharded over the non-manual mesh axis `{name}`")]
    NonManualShardedDimensionRemoval { dimension: usize, name: String },

    #[error("manual axis `{name}` cannot be both varying and unreduced")]
    ConflictingVaryingAndUnreducedMeshAxis { name: String },

    #[error("manual axis `{name}` cannot be both varying and reduced")]
    ConflictingVaryingAndReducedMeshAxis { name: String },

    #[error("device ID `{id}` appears more than once")]
    DuplicateDeviceId { id: DeviceId },

    #[error("mesh has {actual} device(s), but its axis sizes imply {expected} device(s)")]
    DeviceCountMismatch { expected: usize, actual: usize },

    #[error("mesh mismatch; expected `{expected:?}` but got `{actual:?}`")]
    MeshMismatch { expected: LogicalMesh, actual: LogicalMesh },

    #[error("sharding dimension #{dimension} has no axes")]
    EmptySharding { dimension: usize },

    #[error("sharding rank ({sharding_rank}) does not match array rank ({array_rank})")]
    ShardingRankMismatch { sharding_rank: usize, array_rank: usize },

    #[error("dimension index {dimension} is out of bounds for a sharding of rank {rank}")]
    DimensionOutOfBounds { dimension: usize, rank: usize },

    #[error("broadcast axis mapping has length {actual}, but the sharding has rank {expected}")]
    BroadcastAxisCountMismatch { expected: usize, actual: usize },

    #[error("broadcast output dimension {dimension} is out of bounds for rank {rank}")]
    BroadcastDimensionOutOfBounds { dimension: usize, rank: usize },

    #[error("broadcast axis mapping contains output dimension {dimension} more than once")]
    DuplicateBroadcastDimension { dimension: usize },

    #[error("sharding visualization only supports rank-1 and rank-2 shapes, but got rank {rank}")]
    UnsupportedVisualizationRank { rank: usize },
}

impl From<ShardingError> for TypeError {
    #[inline]
    fn from(error: ShardingError) -> Self {
        Self::custom(error)
    }
}

/// Read-only access to the optional [`Sharding`] of a type family. Arrays expose their placement, references expose
/// their referent's placement, and scalar data types and first-class dimensions carry no sharding. Array IR types
/// follow the corresponding member's contract. Operations separately validate which members they accept and which
/// placement or variation transitions they permit.
pub trait ShardableType: Type {
    /// Returns the sharding metadata of this type, or of its referent for a reference, when present.
    fn sharding(&self) -> Option<&Sharding>;

    /// Returns whether this type records variation across at least one manual mesh axis through
    /// [`Sharding::varying_manual_axes`]. This describes permitted differences between devices;
    /// actual values may still be equal. Types without sharding metadata return `false`.
    #[inline]
    fn is_varying(&self) -> bool {
        self.sharding().is_some_and(|sharding| !sharding.varying_manual_axes().is_empty())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionType};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::programs::ReferenceType;

    use super::*;

    #[test]
    fn test_sharding_error_to_type_error() {
        let error = ShardingError::ShardingRankMismatch { sharding_rank: 1, array_rank: 2 };
        let type_error = TypeError::from(error.clone());
        assert_eq!(type_error.downcast_custom::<ShardingError>(), Some(&error));
        assert_eq!(type_error.clone().downcast_custom::<ShardingError>(), Some(&error));
        assert_eq!(type_error.to_string(), error.to_string());
    }

    #[test]
    fn test_shardable_type_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1).with_varying_manual_axes(["devices"]).unwrap();
        let array = ArrayType::new_static(DataType::F32, [3]).with_sharding(sharding.clone()).unwrap();
        let reference = ReferenceType::new(array.clone());
        let dimension = DimensionType::from(Dimension::Static(3));

        // Disambiguate the trait query from the existing concrete array accessor.
        assert_eq!(ShardableType::sharding(&array), Some(&sharding));
        assert_eq!(reference.sharding(), Some(&sharding));
        assert_eq!(ArrayIrType::Array(array).sharding(), Some(&sharding));
        assert_eq!(ArrayIrType::Reference(reference).sharding(), Some(&sharding));
        assert_eq!(ArrayType::scalar(DataType::Boolean).sharding(), None);
        assert_eq!(DataType::F32.sharding(), None);
        assert_eq!(dimension.sharding(), None);
        assert_eq!(ArrayIrType::Dimension(dimension).sharding(), None);
    }

    #[test]
    fn test_shardable_type_is_varying() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("devices", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(mesh.clone(), 0)).unwrap();
        let varying = invariant
            .clone()
            .with_sharding(invariant.sharding().unwrap().clone().with_varying_manual_axes(["devices"]).unwrap())
            .unwrap();
        let tiled = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["devices"])]).unwrap())
            .unwrap();
        assert!(!ArrayType::scalar(DataType::Boolean).is_varying());
        assert!(!invariant.is_varying());
        assert!(!tiled.is_varying());
        assert!(varying.is_varying());
        assert!(!ReferenceType::new(invariant.clone()).is_varying());
        assert!(ReferenceType::new(varying.clone()).is_varying());
        assert!(!ArrayIrType::Array(invariant).is_varying());
        assert!(ArrayIrType::Array(varying.clone()).is_varying());
        assert!(ArrayIrType::Reference(ReferenceType::new(varying)).is_varying());
        assert!(!DataType::F32.is_varying());
        let dimension = DimensionType::from(Dimension::Static(3));
        assert!(!dimension.is_varying());
        assert!(!ArrayIrType::Dimension(dimension).is_varying());
    }
}
