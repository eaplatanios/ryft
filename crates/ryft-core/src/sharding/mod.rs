pub mod devices;
pub mod meshes;
pub mod shardings;
pub mod visualizations;

pub use devices::{Device, DeviceId, DeviceMesh, ProcessIndex};
pub use meshes::{LogicalMesh, MeshAxis, MeshAxisType};
pub use shardings::{Sharding, ShardingDimension};
pub use visualizations::ShardingVisualization;

use thiserror::Error;

/// Represents sharding-related errors.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShardingError {
    #[error("mesh axis names must not be empty")]
    EmptyMeshAxisName,

    #[error("unknown mesh axis name: '{name}'")]
    UnknownMeshAxisName { name: String },

    #[error("mesh axis name '{name}' appears more than once")]
    DuplicateMeshAxisName { name: String },

    #[error("mesh axis '{name}' must have size > 0")]
    EmptyMeshAxis { name: String },

    #[error("mesh axis '{name}' must have type manual")]
    ExpectedManualMeshAxis { name: String },

    #[error("cannot remove dimension {dimension} because it is sharded over the non-manual mesh axis '{name}'")]
    NonManualShardedDimensionRemoval { dimension: usize, name: String },

    #[error("manual axis '{name}' cannot be both varying and unreduced")]
    ConflictingVaryingAndUnreducedMeshAxis { name: String },

    #[error("manual axis '{name}' cannot be both varying and reduced")]
    ConflictingVaryingAndReducedMeshAxis { name: String },

    #[error("device ID '{id}' appears more than once")]
    DuplicateDeviceId { id: DeviceId },

    #[error("mesh has {actual} device(s), but its axis sizes imply {expected} device(s)")]
    DeviceCountMismatch { expected: usize, actual: usize },

    #[error("mesh mismatch; expected '{expected:?}' but got '{actual:?}'")]
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
