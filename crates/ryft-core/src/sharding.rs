use thiserror::Error;

/// Type alias used to represent [`Device`] IDs, which are unique among devices of the same type (e.g., CPUs, GPUs)
/// and, on multi-host environments, are also unique across all devices and all hosts.
pub type DeviceId = usize;

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

    #[error("mesh device ID '{id}' appears more than once")]
    DuplicateMeshDeviceId { id: DeviceId },

    #[error("mesh has {actual_count} device(s), but its axis sizes imply {expected_count} device(s)")]
    MeshDeviceCountMismatch { expected_count: usize, actual_count: usize },

    #[error("partition specification dimension #{dimension} has no axes")]
    EmptyPartitionSpecification { dimension: usize },

    #[error("partition specification rank ({partition_rank}) does not match array rank ({array_rank})")]
    PartitionSpecificationRankMismatch { partition_rank: usize, array_rank: usize },
}

/// [`MeshAxis`] type which controls sharding constraint propagation. Each axis in a [`LogicalMesh`] can be tagged with
/// a [`MeshAxisType`] that tells the compiler (e.g., Shardy or [GSPMD](https://arxiv.org/abs/2105.04663)) how to treat
/// shardings along that axis during sharding constraint propagation. This type corresponds to
/// [`jax.sharding.AxisType`](https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.AxisType).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum MeshAxisType {
    /// Used for mesh axes whose sharding information is inferred automatically by the compiler
    /// (for example, by Shardy or [GSPMD](https://arxiv.org/abs/2105.04663)).
    #[default]
    Auto,

    /// Used for mesh axes whose sharding information is represented explicitly as part of the
    /// type-level sharding metadata and propagated before compilation.
    Explicit,

    // TODO(eaplatanios): Link to the `shard_map` operation once we have it.
    /// Used for mesh axes for which the user manages all device communication explicitly
    /// (e.g., using an operation like `shard_map` which is analogous to
    /// [JAX's `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html)).
    Manual,
}

/// Named axis in a [`LogicalMesh`]. Each axis represents one dimension of the device grid with a human-readable name, a
/// size (i.e., the number of devices along that dimension), and a [`MeshAxisType`] that controls sharding propagation
/// behavior for that axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshAxis {
    /// Name of this [`MeshAxis`].
    pub name: String,

    /// Number of devices along this [`MeshAxis`].
    pub size: usize,

    /// Type of this [`MeshAxis`], controlling sharding propagation behavior.
    pub r#type: MeshAxisType,
}

impl MeshAxis {
    /// Creates a new [`MeshAxis`].
    pub fn new<N: Into<String>>(name: N, size: usize, r#type: MeshAxisType) -> Result<Self, ShardingError> {
        let name = name.into();
        if name.is_empty() {
            Err(ShardingError::EmptyMeshAxisName)
        } else if size == 0 {
            Err(ShardingError::EmptyMeshAxis { name })
        } else {
            Ok(Self { name, size, r#type })
        }
    }
}
