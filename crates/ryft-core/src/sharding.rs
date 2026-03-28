use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock, Weak};

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
    #[inline]
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

/// Key used to intern [`LogicalMesh`] instances.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LogicalMeshKey {
    axes: Vec<MeshAxis>,
}

/// Interned immutable data for a [`LogicalMesh`].
#[doc(hidden)]
#[derive(Debug, PartialEq, Eq)]
pub struct LogicalMeshData {
    /// Named and sized axes that define this logical mesh topology.
    pub axes: Vec<MeshAxis>,

    /// Mapping from [`MeshAxis`] names to their indices/positions in [`Self::axes`].
    pub axis_indices: HashMap<String, usize>,
}

/// Returns all interned [`LogicalMesh`] instances.
#[inline]
fn interned_logical_meshes() -> &'static Mutex<HashMap<LogicalMeshKey, Weak<LogicalMeshData>>> {
    static INTERNER: OnceLock<Mutex<HashMap<LogicalMeshKey, Weak<LogicalMeshData>>>> = OnceLock::new();
    INTERNER.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Logical mesh that represent a device topology that is to be used for sharding. A [`LogicalMesh`] captures the mesh
/// axis names, sizes, and types of a device mesh without binding to physical devices. This is the compilation-time view
/// of a mesh: it provides enough information to validate partition specifications and generate sharding-related code
/// (e.g., [Shardy](https://openxla.org/shardy) MLIR attributes), but it does not carry any device-specific information.
/// Note that equivalent meshes are interned within the process and so repeated constructions share immutable storage.
#[derive(Clone, PartialEq, Eq)]
pub struct LogicalMesh(Arc<LogicalMeshData>);

impl Debug for LogicalMesh {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LogicalMesh")
            .field("axes", &self.axes)
            .field("axis_indices", &self.axis_indices)
            .finish()
    }
}

impl Deref for LogicalMesh {
    type Target = LogicalMeshData;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl LogicalMesh {
    /// Creates a new [`LogicalMesh`].
    #[inline]
    pub fn new(axes: Vec<MeshAxis>) -> Result<Self, ShardingError> {
        let mut axis_indices = HashMap::with_capacity(axes.len());
        for (axis_index, axis) in axes.iter().enumerate() {
            if axis_indices.insert(axis.name.clone(), axis_index).is_some() {
                return Err(ShardingError::DuplicateMeshAxisName { name: axis.name.clone() });
            }
        }
        let mut interner = interned_logical_meshes().lock().expect("poisoned logical mesh interner mutex");
        match interner.entry(LogicalMeshKey { axes }) {
            Entry::Occupied(mut occupied) => {
                if let Some(mesh) = occupied.get().upgrade() {
                    Ok(Self(mesh))
                } else {
                    let mesh = Arc::new(LogicalMeshData { axes: occupied.key().axes.clone(), axis_indices });
                    occupied.insert(Arc::downgrade(&mesh));
                    Ok(Self(mesh))
                }
            }
            Entry::Vacant(vacant) => {
                let mesh = Arc::new(LogicalMeshData { axes: vacant.key().axes.clone(), axis_indices });
                vacant.insert(Arc::downgrade(&mesh));
                Ok(Self(mesh))
            }
        }
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
    /// Creates a new [`shardy::DetachedMeshOperation`] that corresponds to this [`LogicalMesh`].
    /// The mesh in the returned operation will be named `"mesh"`.
    #[inline]
    pub fn to_shardy_mesh<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> shardy::DetachedMeshOperation<'c, 't> {
        let context = location.context();
        let attribute = context
            .shardy_mesh(self.axes.iter().map(|axis| context.shardy_mesh_axis(axis.name.as_str(), axis.size)), &[]);
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
    #[inline]
    pub fn new(id: MeshDeviceId, process_index: MeshProcessIndex) -> Self {
        Self { id, process_index }
    }
}

/// Mesh of devices used by sharding layouts. A [`DeviceMesh`] organizes a set of [`MeshDevice`]s into a
/// [`LogicalMesh`]. Devices are stored in **row-major order** with respect to the [`MeshAxis`] list (e.g.,
/// for a two-dimensional mesh with axes `("data"=4, "model"=2)`, the device at mesh coordinate `(i, j)` has
/// linear index `i * 2 + j`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceMesh {
    /// Logical mesh topology that defines the names, sizes, and types of the mesh axes.
    pub logical_mesh: LogicalMesh,

    /// Physical devices laid out in row-major order with respect to [`Self::logical_mesh`].
    pub devices: Vec<MeshDevice>,
}

impl DeviceMesh {
    /// Creates a new [`DeviceMesh`].
    #[inline]
    pub fn new(logical_mesh: LogicalMesh, devices: Vec<MeshDevice>) -> Result<Self, ShardingError> {
        let expected_device_count = logical_mesh.device_count();
        if devices.len() != expected_device_count {
            return Err(ShardingError::MeshDeviceCountMismatch {
                expected_count: expected_device_count,
                actual_count: devices.len(),
            });
        }

        let mut seen_device_ids = HashSet::with_capacity(devices.len());
        for device in &devices {
            if !seen_device_ids.insert(device.id) {
                return Err(ShardingError::DuplicateMeshDeviceId { id: device.id });
            }
        }

        Ok(Self { logical_mesh, devices })
    }

    /// Returns the rank (i.e., number of axes) of this [`DeviceMesh`].
    #[inline]
    pub fn rank(&self) -> usize {
        self.logical_mesh.rank()
    }

    /// Returns the size of the [`MeshAxis`] in this [`DeviceMesh`] with the provided name, if such an axis exists.
    #[inline]
    pub fn axis_size<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.logical_mesh.axis_size(axis_name)
    }

    /// Returns the type of the [`MeshAxis`] in this [`DeviceMesh`] with the provided name, if such an axis exists.
    #[inline]
    pub fn axis_type<S: AsRef<str>>(&self, axis_name: S) -> Option<MeshAxisType> {
        self.logical_mesh.axis_type(axis_name)
    }

    /// Returns the total number of devices that the topology defined by this [`DeviceMesh`] contains.
    #[inline]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Returns the mesh coordinates of the [`MeshDevice`] at the provided index, if valid.
    #[inline]
    pub fn device_coordinates(&self, device_index: usize) -> Option<Vec<usize>> {
        (device_index < self.devices.len()).then(|| {
            let axis_sizes = self.logical_mesh.axes.iter().map(|axis| axis.size).collect::<Vec<_>>();
            let mut coordinates = vec![0usize; axis_sizes.len()];
            let mut index = device_index;
            for (axis_index, axis_size) in axis_sizes.iter().enumerate().rev() {
                coordinates[axis_index] = index % axis_size;
                index /= axis_size;
            }
            coordinates
        })
    }
}

#[cfg(feature = "xla")]
impl DeviceMesh {
    /// Creates a new [`shardy::DetachedMeshOperation`] that corresponds to this [`DeviceMesh`].
    /// The mesh in the returned operation will be named `"mesh"`.
    #[inline]
    pub fn to_shardy_mesh<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> shardy::DetachedMeshOperation<'c, 't> {
        self.logical_mesh.to_shardy_mesh(location)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    #[cfg(feature = "xla")]
    use ryft_mlir::{Block, Context as MlirContext};

    use super::*;

    #[test]
    fn test_mesh_axis_type() {
        assert_eq!(MeshAxisType::default(), MeshAxisType::Auto);
    }

    #[test]
    fn test_mesh_axis() {
        let axis = MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap();
        assert_eq!(axis.name, "x");
        assert_eq!(axis.size, 2);
        assert_eq!(axis.r#type, MeshAxisType::Auto);

        let axis = MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap();
        assert_eq!(axis.name, "y");
        assert_eq!(axis.size, 3);
        assert_eq!(axis.r#type, MeshAxisType::Manual);

        assert!(matches!(MeshAxis::new("", 4, MeshAxisType::Auto), Err(ShardingError::EmptyMeshAxisName)));
        assert!(matches!(
            MeshAxis::new("x", 0, MeshAxisType::Auto),
            Err(ShardingError::EmptyMeshAxis { name }) if name == "x",
        ));
    }

    #[test]
    fn test_logical_mesh() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 1, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        assert_eq!(mesh.axes.iter().map(|axis| axis.name.as_str()).collect::<Vec<_>>(), vec!["x", "y", "z"]);
        assert_eq!(mesh.axes.iter().map(|axis| axis.size).collect::<Vec<_>>(), vec![2, 3, 1]);
        assert_eq!(
            mesh.axes.iter().map(|axis| axis.r#type).collect::<Vec<_>>(),
            vec![MeshAxisType::Auto, MeshAxisType::Manual, MeshAxisType::Explicit]
        );
        assert_eq!(mesh.axis_indices.get("x"), Some(&0));
        assert_eq!(mesh.axis_indices.get("y"), Some(&1));
        assert_eq!(mesh.axis_indices.get("z"), Some(&2));
        assert_eq!(mesh.axis_indices.get("w"), None);
        assert_eq!(mesh.rank(), 3);
        assert_eq!(mesh.axis_size("x"), Some(2));
        assert_eq!(mesh.axis_size("y"), Some(3));
        assert_eq!(mesh.axis_size("z"), Some(1));
        assert_eq!(mesh.axis_size("w"), None);
        assert_eq!(mesh.axis_type("x"), Some(MeshAxisType::Auto));
        assert_eq!(mesh.axis_type("y"), Some(MeshAxisType::Manual));
        assert_eq!(mesh.axis_type("z"), Some(MeshAxisType::Explicit));
        assert_eq!(mesh.axis_type("w"), None);
        assert_eq!(mesh.device_count(), 6);

        assert!(matches!(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("x", 3, MeshAxisType::Auto).unwrap(),
            ]),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));

        let mesh_0 = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let mesh_1 = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let mesh_2 = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        assert!(Arc::ptr_eq(&mesh_0.0, &mesh_1.0));
        assert!(!Arc::ptr_eq(&mesh_0.0, &mesh_2.0));
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_logical_mesh_to_shardy_mesh() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 1, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let context = MlirContext::new();
        let module = context.module(context.unknown_location());
        assert_eq!(
            module.body().append_operation(mesh.to_shardy_mesh(context.unknown_location())).to_string(),
            format!("sdy.mesh @{SHARDY_MESH_SYMBOL_NAME} = <[\"x\"=2, \"y\"=3, \"z\"=1]>"),
        );
    }

    #[test]
    fn test_device_mesh() {
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        assert_eq!(&mesh.logical_mesh, &logical_mesh);
        assert_eq!(&mesh.devices, &devices);
        assert_eq!(mesh.rank(), 2);
        assert_eq!(mesh.axis_size("x"), Some(2));
        assert_eq!(mesh.axis_size("y"), Some(2));
        assert_eq!(mesh.axis_size("z"), None);
        assert_eq!(mesh.axis_type("x"), Some(MeshAxisType::Auto));
        assert_eq!(mesh.axis_type("y"), Some(MeshAxisType::Manual));
        assert_eq!(mesh.axis_type("z"), None);
        assert_eq!(mesh.device_count(), 4);
        assert_eq!(mesh.device_coordinates(0), Some(vec![0, 0]));
        assert_eq!(mesh.device_coordinates(1), Some(vec![0, 1]));
        assert_eq!(mesh.device_coordinates(2), Some(vec![1, 0]));
        assert_eq!(mesh.device_coordinates(3), Some(vec![1, 1]));
        assert_eq!(mesh.device_coordinates(4), None);

        assert!(matches!(
            DeviceMesh::new(
                logical_mesh.clone(),
                vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1)],
            ),
            Err(ShardingError::MeshDeviceCountMismatch { expected_count: 4, actual_count: 3 }),
        ));
        assert!(matches!(
            DeviceMesh::new(
                logical_mesh.clone(),
                vec![MeshDevice::new(0, 0), MeshDevice::new(0, 0), MeshDevice::new(1, 1), MeshDevice::new(2, 1)],
            ),
            Err(ShardingError::DuplicateMeshDeviceId { id }) if id == 0,
        ));
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_device_mesh_to_shardy_mesh() {
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        let mesh = DeviceMesh::new(logical_mesh.clone(), devices.clone()).unwrap();
        let context = MlirContext::new();
        let module = context.module(context.unknown_location());
        assert_eq!(
            module.body().append_operation(mesh.to_shardy_mesh(context.unknown_location())).to_string(),
            format!("sdy.mesh @{SHARDY_MESH_SYMBOL_NAME} = <[\"x\"=2, \"y\"=2]>"),
        );
    }
}
