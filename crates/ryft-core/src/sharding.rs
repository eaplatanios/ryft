use std::collections::HashMap;

use thiserror::Error;

#[cfg(feature = "xla")]
use ryft_mlir::{Location, dialects::shardy};

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
    DuplicateMeshDeviceId { id: MeshDeviceId },

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

    /// Used for mesh axes for which the user manages all device communication explicitly
    /// (e.g., using an operation like [`shard_map`] which is analogous to
    /// [JAX's `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html)).
    Manual,
}

/// Named axis in a [`LogicalMesh`]. Each axis represents one dimension of the device grid with a human-readable name,
/// a size (i.e., the number of devices along that dimension), and a [`MeshAxisType`] that controls sharding propagation
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

/// Logical mesh that represent a device topology that is to be used for sharding. A [`LogicalMesh`] captures the mesh
/// axis names, sizes, and types of a device mesh without binding to physical devices. This is the compilation-time view
/// of a mesh: it provides enough information to validate partition specifications and generate sharding-related code
/// (e.g., [Shardy](https://openxla.org/shardy) MLIR attributes), but it does not carry any device-specific information.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalMesh {
    /// Named and sized axes that define this logical mesh topology.
    pub axes: Vec<MeshAxis>,

    /// Mapping from [`MeshAxis`] names to their indices/positions in [`Self::axes`].
    pub axis_indices: HashMap<String, usize>,
}

impl LogicalMesh {
    /// Creates a new [`LogicalMesh`].
    pub fn new(axes: Vec<MeshAxis>) -> Result<Self, ShardingError> {
        let mut axis_indices = HashMap::with_capacity(axes.len());
        for (axis_index, axis) in axes.iter().enumerate() {
            if axis.name.is_empty() {
                return Err(ShardingError::EmptyMeshAxisName);
            }
            if axis.size == 0 {
                return Err(ShardingError::EmptyMeshAxis { name: axis.name.clone() });
            }
            if axis_indices.insert(axis.name.clone(), axis_index).is_some() {
                return Err(ShardingError::DuplicateMeshAxisName { name: axis.name.clone() });
            }
        }
        Ok(Self { axes, axis_indices })
    }

    /// Returns the rank (i.e., number of axes) of this [`LogicalMesh`].
    #[inline]
    pub fn rank(&self) -> usize {
        self.axes.len()
    }

    /// Returns the size of the [`MeshAxis`] in this [`LogicalMesh`] with the provided name, if such an axis exists.
    #[inline]
    pub fn axis_size<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.axis_indices.get(axis_name.as_ref()).map(|axis_index| self.axes[*axis_index].size)
    }

    /// Returns the type of the [`MeshAxis`] in this [`LogicalMesh`] with the provided name, if such an axis exists.
    #[inline]
    pub fn axis_type<S: AsRef<str>>(&self, axis_name: S) -> Option<MeshAxisType> {
        self.axis_indices.get(axis_name.as_ref()).map(|axis_index| self.axes[*axis_index].r#type)
    }

    /// Returns the total number of devices that the topology defined by this [`LogicalMesh`] contains.
    #[inline]
    pub fn device_count(&self) -> usize {
        self.axes.iter().fold(1usize, |count, axis| count * axis.size)
    }
}

/// Canonical symbol name used for emitted Shardy [`LogicalMesh`] declarations and references.
#[cfg(feature = "xla")]
pub(crate) const SHARDY_MESH_SYMBOL_NAME: &str = "mesh";

#[cfg(feature = "xla")]
impl LogicalMesh {
    /// Constructs a new [`shardy::DetachedMeshOperation`] that corresponds to this [`LogicalMesh`].
    /// The mesh in the returned operation will be named `"mesh"`.
    pub fn to_shardy_mesh<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> shardy::DetachedMeshOperation<'c, 't> {
        let context = location.context();
        let axes = self
            .axes
            .iter()
            .map(|axis| context.shardy_mesh_axis(axis.name.as_str(), axis.size))
            .collect::<Vec<_>>();
        let attribute = context.shardy_mesh(axes.as_slice(), &[]);
        shardy::mesh(SHARDY_MESH_SYMBOL_NAME, attribute, location)
    }
}

/// Type alias used to represent [`MeshDevice`] IDs, which are unique among devices of the same type (e.g., CPUs, GPUs)
/// and, on multi-host environments, are also unique across all devices and all hosts.
pub type MeshDeviceId = usize;

/// Type alias used to represent process indices in multi-process/multi-host environments.
pub type MeshProcessIndex = usize;

/// Device that belongs to a mesh topology. This type separates global device identity that is described by a
/// [`MeshDeviceId`], from host/process ownership, that is described by a [`ProcessIndex`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshDevice {
    /// Globally (i.e., across all hosts/processes) unique device ID.
    pub id: MeshDeviceId,

    /// Index of the process that owns this device. In single-host setups, this will always be set to `0`. In multi-host
    /// setups it determines _addressability_. That is, a _shard_ of an array that is located on some device `d` is
    /// _addressable_ from a process with index `p` if and only if `d.process_index == p`.
    pub process_index: MeshProcessIndex,
}

impl MeshDevice {
    /// Creates a new [`MeshDevice`].
    pub fn new(id: MeshDeviceId, process_index: MeshProcessIndex) -> Self {
        Self { id, process_index }
    }
}
