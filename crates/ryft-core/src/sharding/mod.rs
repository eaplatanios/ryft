pub mod visualizations;

pub use visualizations::ShardingVisualization;

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use thiserror::Error;

use ryft_macros::Parameter;

use crate::parameters::Parameter;

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

    #[error("sharding visualization only supports rank-1 and rank-2 shapes, but got rank {rank}")]
    UnsupportedVisualizationRank { rank: usize },
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
    /// (e.g., using an operation like `shard_map` which is analogous to
    /// [JAX's `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html)).
    Manual,
}

/// Named axis in a [`LogicalMesh`]. Each axis represents one dimension of the device grid with a human-readable name,
/// a size (i.e., the number of devices along that dimension), and a [`MeshAxisType`] that controls sharding propagation
/// behavior for that axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshAxis {
    /// Name of this [`MeshAxis`].
    name: String,

    /// Number of devices along this [`MeshAxis`].
    size: usize,

    /// Type of this [`MeshAxis`], controlling sharding propagation behavior.
    r#type: MeshAxisType,
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

    /// Returns the name of this [`MeshAxis`].
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the number of devices along this [`MeshAxis`].
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the type of this [`MeshAxis`], controlling sharding propagation behavior.
    #[inline]
    pub fn r#type(&self) -> MeshAxisType {
        self.r#type
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
    axes: Vec<MeshAxis>,

    /// Mapping from [`MeshAxis`] names to their indices/positions in [`Self::axes`].
    axis_indices: HashMap<String, usize>,
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

    /// Returns the named and sized axes that define this logical mesh topology.
    #[inline]
    pub fn axes(&self) -> &[MeshAxis] {
        &self.axes
    }

    /// Returns the index of the [`MeshAxis`] with the provided name, if such an axis exists.
    #[inline]
    pub fn axis_index<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.axis_indices.get(axis_name.as_ref()).copied()
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

impl Debug for LogicalMesh {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LogicalMesh")
            .field("axes", &self.axes)
            .field("axis_indices", &self.axis_indices)
            .finish()
    }
}

impl Hash for LogicalMesh {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.axes.hash(state);
    }
}

impl Deref for LogicalMesh {
    type Target = LogicalMeshData;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

/// Type alias used to represent [`Device`] IDs, which are unique among devices of the same type (e.g., CPUs, GPUs)
/// and, on multi-host environments, are also unique across all devices and all hosts.
pub type DeviceId = usize;

/// Type alias used to represent process indices in multi-process/multi-host environments.
pub type ProcessIndex = usize;

/// Device that belongs to a mesh topology. This type separates global device identity that is described by a
/// [`DeviceId`], from host/process ownership, that is described by a [`ProcessIndex`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Device {
    /// Globally (i.e., across all hosts/processes) unique [`DeviceId`].
    id: DeviceId,

    /// Index of the process that owns this device. In single-host setups, this will always be set to `0`. In multi-host
    /// setups it determines _addressability_. That is, a _shard_ of an array that is located on some device `d` is
    /// _addressable_ from a process with index `p` if and only if `d.process_index == p`.
    process_index: ProcessIndex,
}

impl Device {
    /// Creates a new [`Device`].
    #[inline]
    pub fn new(id: DeviceId, process_index: ProcessIndex) -> Self {
        Self { id, process_index }
    }

    /// Returns globally (i.e., across all hosts/processes) unique [`DeviceId`] of this [`Device`].
    #[inline]
    pub fn id(&self) -> DeviceId {
        self.id
    }

    /// Returns the index of the process that owns this device. In single-host setups, this will always be set to `0`.
    /// In multi-host setups it determines _addressability_. That is, a _shard_ of an array that is located on some
    /// device `d` is _addressable_ from a process with index `p` if and only if `d.process_index == p`.
    #[inline]
    pub fn process_index(&self) -> ProcessIndex {
        self.process_index
    }
}

/// Mesh of devices used by sharding layouts. A [`DeviceMesh`] organizes a set of [`Device`]s into a [`LogicalMesh`].
/// Devices are stored in **row-major order** with respect to the [`MeshAxis`] list (e.g., for a two-dimensional mesh
/// with axes `("data"=4, "model"=2)`, the device at mesh coordinate `(i, j)` has linear index `i * 2 + j`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceMesh {
    /// Logical mesh topology that defines the names, sizes, and types of the mesh axes.
    pub(crate) logical_mesh: LogicalMesh,

    /// Physical devices laid out in row-major order with respect to [`Self::logical_mesh`].
    pub(crate) devices: Vec<Device>,
}

impl DeviceMesh {
    /// Creates a new [`DeviceMesh`].
    #[inline]
    pub fn new(logical_mesh: LogicalMesh, devices: Vec<Device>) -> Result<Self, ShardingError> {
        let expected_device_count = logical_mesh.device_count();
        if devices.len() != expected_device_count {
            return Err(ShardingError::DeviceCountMismatch { expected: expected_device_count, actual: devices.len() });
        }

        let mut seen_device_ids = HashSet::with_capacity(devices.len());
        for device in &devices {
            if !seen_device_ids.insert(device.id) {
                return Err(ShardingError::DuplicateDeviceId { id: device.id });
            }
        }

        Ok(Self { logical_mesh, devices })
    }

    /// Returns the logical mesh topology that defines the names, sizes, and types of the mesh axes.
    #[inline]
    pub fn logical_mesh(&self) -> &LogicalMesh {
        &self.logical_mesh
    }

    /// Returns the physical devices laid out in row-major order with respect to [`Self::logical_mesh`].
    #[inline]
    pub fn devices(&self) -> &[Device] {
        &self.devices
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

    /// Returns the mesh coordinates of the [`Device`] at the provided index, if valid.
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

/// Describes how a single dimension of an array/tensor is distributed across [`LogicalMesh`] axes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShardingDimension {
    /// Dimension that is replicated across the devices in a mesh instead of being sharded/partitioned.
    Replicated,

    /// Dimension that is sharded/partitioned by the mesh axes with the specified names. The dimension is sharded along
    /// the product of the specified axes, in major to minor order. For example, with a `4x2` mesh with `"data"` and
    /// `"model"` axes and `Sharded(["data", "model"])`, a dimension of size `24` is split into `4 * 2 = 8` partitions.
    Sharded(Vec<String>),

    /// Dimension that is unconstrained when it comes to sharding, meaning that the compiler is free to decide
    /// if and how to shard it.
    Unconstrained,
}

impl ShardingDimension {
    /// Creates a new [`Self::Replicated`].
    #[inline]
    pub fn replicated() -> Self {
        Self::Replicated
    }

    /// Creates a new [`Self::Sharded`].
    #[inline]
    pub fn sharded<N: Into<String>, I: IntoIterator<Item = N>>(axis_names: I) -> Self {
        Self::Sharded(axis_names.into_iter().map(Into::into).collect())
    }

    /// Creates a new [`Self::Unconstrained`].
    #[inline]
    pub fn unconstrained() -> Self {
        Self::Unconstrained
    }
}

impl Display for ShardingDimension {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Replicated => write!(formatter, "{{}}"),
            Self::Unconstrained => write!(formatter, "{{?}}"),
            Self::Sharded(axis_names) => {
                write!(formatter, "{{")?;
                if let Some((first_axis_name, remaining_axis_names)) = axis_names.split_first() {
                    write!(formatter, "'{}'", first_axis_name.replace('\'', "\\'"))?;
                    for axis_name in remaining_axis_names {
                        write!(formatter, ", '{}'", axis_name.replace('\'', "\\'"))?;
                    }
                }
                write!(formatter, "}}")
            }
        }
    }
}

/// [`LogicalMesh`]-bound sharding for a logical array value. This is the primary user-facing sharding type for
/// compilation-time annotations. It owns the [`LogicalMesh`] together with the per-dimension [`ShardingDimension`]
/// assignments and any additional state needed to model partial reductions and [`MeshAxisType::Manual`] mesh axes.
///
/// # Example
///
/// Consider the following [`Sharding`]:
///
/// ```ignore
/// Sharding {
///     mesh,
///     dimensions: vec![
///         ShardingDimension::sharded(["data"]),
///         ShardingDimension::replicated(),
///     ],
///     unreduced_axes: std::collections::BTreeSet::from(["model".to_string()]),
///     reduced_axes: std::collections::BTreeSet::new(),
///     varying_manual_axes: std::collections::BTreeSet::new(),
/// };
/// ```
///
/// In this case, the `"data"` [`MeshAxis`] shards array dimension `0`, while `"model"` does not shard any ranked
/// dimension and instead marks the value as still unreduced along the mesh axis `"model"`. Without `unreduced_axes`,
/// that unused mesh axis would be indistinguishable from a truly replicated axis.
///
/// # References
///
/// For more information on the approach Ryft takes to sharding, you can refer to the relevant JAX documentation that
/// inspired it. The following pages are particularly relevant:
///
/// - [Distributed Arrays and Automatic Parallelization](
///   https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
/// - [Explicit Sharding](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)
/// - [Manual Parallelism with `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html#so-let-s-see-a-shard-map).
/// - [Memories and Host Offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct Sharding {
    /// Refer to the documentation of [`Self::mesh`] for information on this field.
    pub(crate) mesh: LogicalMesh,

    /// Refer to the documentation of [`Self::dimensions`] for information on this field.
    pub(crate) dimensions: Vec<ShardingDimension>,

    /// Refer to the documentation of [`Self::unreduced_axes`] for information on this field.
    pub(crate) unreduced_axes: BTreeSet<String>,

    /// Refer to the documentation of [`Self::reduced_axes`] for information on this field.
    pub(crate) reduced_axes: BTreeSet<String>,

    /// Refer to the documentation of [`Self::varying_manual_axes`] for information on this field.
    pub(crate) varying_manual_axes: BTreeSet<String>,
}

impl Sharding {
    /// Creates a new [`Sharding`] from a [`LogicalMesh`] and a per-dimension list of [`ShardingDimension`]s.
    /// Use [`Self::with_unreduced_axes`] or [`Self::with_manual_axes`] when you also need to specify unreduced,
    /// reduced, or varying manual axes.
    pub fn new(mesh: LogicalMesh, dimensions: Vec<ShardingDimension>) -> Result<Self, ShardingError> {
        Self::with_manual_axes::<&str, _, &str, _, &str, _>(mesh, dimensions, [], [], [])
    }

    /// Creates a new [`Sharding`] with explicit unreduced axes. Use this when the sharding carries partial results
    /// along certain mesh axes that still need cross-device reduction.
    pub fn with_unreduced_axes<U: Into<String>, UI: IntoIterator<Item = U>>(
        mesh: LogicalMesh,
        dimensions: Vec<ShardingDimension>,
        unreduced_axes: UI,
    ) -> Result<Self, ShardingError> {
        Self::with_manual_axes::<_, _, &str, _, &str, _>(mesh, dimensions, unreduced_axes, [], [])
    }

    /// Creates a new [`Sharding`] with full control over unreduced, reduced, and varying manual axes. Prefer
    /// [`Self::new`] or [`Self::with_unreduced_axes`] when the reduction-state and manual-axis fields are not needed.
    pub fn with_manual_axes<
        U: Into<String>,
        UI: IntoIterator<Item = U>,
        R: Into<String>,
        RI: IntoIterator<Item = R>,
        V: Into<String>,
        VI: IntoIterator<Item = V>,
    >(
        mesh: LogicalMesh,
        dimensions: Vec<ShardingDimension>,
        unreduced_axes: UI,
        reduced_axes: RI,
        varying_manual_axes: VI,
    ) -> Result<Self, ShardingError> {
        let unreduced_axes = unreduced_axes.into_iter().map(Into::into).collect();
        let reduced_axes = reduced_axes.into_iter().map(Into::into).collect();
        let varying_manual_axes = varying_manual_axes.into_iter().map(Into::into).collect();
        let sharding = Self { mesh, dimensions, unreduced_axes, reduced_axes, varying_manual_axes };

        let mut used_axis_names = HashSet::new();
        for (dimension, partition_dimension) in sharding.dimensions.iter().enumerate() {
            if let ShardingDimension::Sharded(axis_names) = partition_dimension {
                if axis_names.is_empty() {
                    return Err(ShardingError::EmptySharding { dimension });
                }

                let mut seen_axis_names = HashSet::new();
                for axis_name in axis_names {
                    if !sharding.mesh.axis_indices.contains_key(axis_name) {
                        return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
                    }

                    if !seen_axis_names.insert(axis_name.clone()) || !used_axis_names.insert(axis_name.clone()) {
                        return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
                    }
                }
            }
        }

        for axis_name in &sharding.unreduced_axes {
            if !sharding.mesh.axis_indices.contains_key(axis_name) {
                return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
            }

            if used_axis_names.contains(axis_name) {
                return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
            }

            used_axis_names.insert(axis_name.clone());
        }

        for axis_name in &sharding.reduced_axes {
            if !sharding.mesh.axis_indices.contains_key(axis_name) {
                return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
            }

            if used_axis_names.contains(axis_name) {
                return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
            }

            used_axis_names.insert(axis_name.clone());
        }

        for axis_name in &sharding.varying_manual_axes {
            if !sharding.mesh.axis_indices.contains_key(axis_name) {
                return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
            }

            if sharding.mesh.axis_type(axis_name) != Some(MeshAxisType::Manual) {
                return Err(ShardingError::ExpectedManualMeshAxis { name: axis_name.clone() });
            }

            if sharding.unreduced_axes.contains(axis_name) {
                return Err(ShardingError::ConflictingVaryingAndUnreducedMeshAxis { name: axis_name.clone() });
            }

            if sharding.reduced_axes.contains(axis_name) {
                return Err(ShardingError::ConflictingVaryingAndReducedMeshAxis { name: axis_name.clone() });
            }
        }

        Ok(sharding)
    }

    /// Creates a new _fully-replicated_ [`Sharding`] for an array with rank `rank`. All dimensions in the resulting
    /// sharding are going to be [`ShardingDimension::Replicated`], meaning that a copy of the full array will be
    /// present on every device.
    #[inline]
    pub fn replicated(mesh: LogicalMesh, rank: usize) -> Self {
        Self {
            mesh,
            dimensions: vec![ShardingDimension::Replicated; rank],
            unreduced_axes: BTreeSet::new(),
            reduced_axes: BTreeSet::new(),
            varying_manual_axes: BTreeSet::new(),
        }
    }

    /// Returns the [`LogicalMesh`] that describes the device topology underlying this [`Sharding`] and gives meaning
    /// to every [`MeshAxis`] name stored in it. This is effectively the coordinate system for the rest of this struct.
    /// Every axis name mentioned in [`Self::dimensions`], [`Self::unreduced_axes`], [`Self::reduced_axes`], and
    /// [`Self::varying_manual_axes`] is resolved against this mesh.
    #[inline]
    pub fn mesh(&self) -> &LogicalMesh {
        &self.mesh
    }

    /// Returns the ranked per-array dimension [`Sharding`] partition assignments. This is the array-rank-indexed part
    /// of this sharding: `dimensions[i]` describes how the logical array dimension `i` is partitioned across the mesh.
    /// For example, on a mesh with axes `("data", "model")`, the [`dimensions`](Self::dimensions) assignment
    /// `[ShardingDimension::sharded(["data"]), ShardingDimension::replicated()]` means that the first array dimension
    /// is split across `"data"` while the second array dimension is replicated on every device. This field
    /// intentionally does not try to encode every mesh-related fact about the value. Mesh axes that matter
    /// semantically but do not correspond to a ranked array dimension are stored separately in
    /// [`Self::unreduced_axes`], [`Self::reduced_axes`], and [`Self::varying_manual_axes`].
    #[inline]
    pub fn dimensions(&self) -> &[ShardingDimension] {
        &self.dimensions
    }

    /// Returns the mesh axes along which values carry per-device partial results. This is the "a cross-device reduction
    /// still needs to happen" marker. An axis can disappear from [`Self::dimensions`] after a local computation
    /// reduces over the corresponding array dimension, but the value may still not be truly replicated; each shard can
    /// still hold a different partial result that must later be combined across that mesh axis. Concretely, imagine a
    /// mesh with axes `("data", "model")` and a value whose first tensor dimension is sharded by `"data"`. If a local
    /// computation then sums over a `"model"`-partitioned feature dimension, the resulting value may have no ranked
    /// dimension left that mentions `"model"`, yet each `"model"` shard still owns a different partial sum. Setting
    /// `unreduced_axes` to `["model"]` preserves that fact. This is why this field is needed even though the mesh axis
    /// no longer appears in [`Self::dimensions`]; without it, an axis that is absent from ranked dimensions would be
    /// indistinguishable from ordinary replication.
    #[inline]
    pub fn unreduced_axes(&self) -> &BTreeSet<String> {
        &self.unreduced_axes
    }

    /// Returns the mesh axes across which values are known to have already been reduced. This is the dual of
    /// [`Self::unreduced_axes`]. A reduced axis is computationally indistinguishable from a replicated one (every
    /// device holds the same data along it), and the marker records that the value was produced by a reduction across
    /// that axis, even though that fact no longer has a direct ranked-dimension representation. The two sets are also
    /// duals under transposition (i.e., the cotangent of a value that is unreduced along an axis is reduced along that
    /// axis, and vice versa), mirroring how [JAX's `PartitionSpec`](
    /// https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec) pairs `unreduced` with `reduced`.
    /// For [`MeshAxisType::Manual`] axes, the marker records that a manual mesh axis has already been consumed by a
    /// reduction inside a `shard_map` body: a concrete example is an output that is replicated in [`Self::dimensions`]
    /// but was produced by first summing across the active manual axis `"data"` inside the mapped computation, where
    /// `reduced_axes` being set to `["data"]` distinguishes "this value is already reduced across `data`" from both
    /// "this value is still unreduced across `data`" and "this axis was never relevant to the value".
    #[inline]
    pub fn reduced_axes(&self) -> &BTreeSet<String> {
        &self.reduced_axes
    }

    /// Returns the [`MeshAxisType::Manual`] mesh axes for which `shard_map` values are known to vary along. Unlike
    /// [`Self::dimensions`], this is not a placement description. It answers a typing question used while tracing
    /// `shard_map`: if we compared two otherwise identical devices that differ only along one of these axes, could this
    /// local value still be different? A concrete nested-`shard_map` example is an outer map that is manual over `"y"`
    /// and an inner map whose input sharding specifications additionally shard the value over manual axis `"x"`. Inside
    /// the inner body, the local array can still have the same rank and local shape as before, but it now semantically
    /// varies across both manual axes, and so the trace has `varying_manual_axes` set to `["y", "x"]`. This is needed
    /// because neither ranked sharding nor reduction-state fields can say whether a local value is uniform across the
    /// active manual shards. For example, constants created inside `shard_map` preserve [`Self::unreduced_axes`] and
    /// [`Self::reduced_axes`] but clear [`Self::varying_manual_axes`], because a constant does not vary from shard to
    /// shard even when it is traced under manual axes.
    #[inline]
    pub fn varying_manual_axes(&self) -> &BTreeSet<String> {
        &self.varying_manual_axes
    }

    /// Returns the rank (i.e., number of dimensions) of this [`Sharding`].
    #[inline]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the names of the mesh axes that are implicitly or explicitly replicated by this [`Sharding`].
    pub fn replicated_axes(&self) -> Vec<&str> {
        let mut used_axes = HashSet::new();
        for dimension in &self.dimensions {
            if let ShardingDimension::Sharded(axis_names) = dimension {
                used_axes.extend(axis_names.iter().map(String::as_str));
            }
        }
        used_axes.extend(self.unreduced_axes.iter().map(String::as_str));
        used_axes.extend(self.reduced_axes.iter().map(String::as_str));
        self.mesh
            .axes
            .iter()
            .filter_map(|axis| {
                let axis_name = axis.name.as_str();
                (matches!(self.mesh.axis_type(axis_name), Some(MeshAxisType::Explicit | MeshAxisType::Manual))
                    && !used_axes.contains(axis_name))
                .then_some(axis_name)
            })
            .collect()
    }

    /// Returns the partition index for the provided array dimension that is owned by the device at the provided
    /// mesh coordinates. Each dimension of a sharded array is partitioned independently; a device's full shard is the
    /// intersection of its per-dimension partitions. For example, with sharding `[Sharded(["x"]), Sharded(["y"])]` on
    /// a `2×2` mesh, the device at `(x=1, y=0)` owns partition `1` of dimension `0` (i.e., the second row-band) and
    /// partition `0` of dimension `1` (i.e., the first column-band). Together these identify the rectangular tile that
    /// device holds.
    ///
    /// The returned index is computed as follows:
    ///   - [`ShardingDimension::Replicated`] and [`ShardingDimension::Unconstrained`] always have partition index `0`,
    ///     since every device holds the full extent of that dimension.
    ///   - [`ShardingDimension::Sharded`] results in the row-major linearization of the device's mesh coordinates along
    ///     the sharding axes. For example, given `Sharded(["data", "model"])` where `data` has size `4` and `model` has
    ///     size `2`, a device at mesh coordinates `(data=2, model=1)` maps to partition index `2 * 2 + 1 = 5`.
    pub fn partition_index(&self, dimension: usize, device_mesh_coordinates: &[usize]) -> Result<usize, ShardingError> {
        let sharding_dimension = self
            .dimensions
            .get(dimension)
            .ok_or(ShardingError::DimensionOutOfBounds { dimension, rank: self.rank() })?;
        match sharding_dimension {
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => Ok(0),
            ShardingDimension::Sharded(axis_names) => axis_names.iter().try_fold(0usize, |index, axis_name| {
                let axis_index = self
                    .mesh
                    .axis_indices
                    .get(axis_name.as_str())
                    .copied()
                    .ok_or_else(|| ShardingError::UnknownMeshAxisName { name: axis_name.clone() })?;
                Ok(index * self.mesh.axes[axis_index].size + device_mesh_coordinates[axis_index])
            }),
        }
    }

    /// Returns a copy of this [`Sharding`] with all of its [`MeshAxisType::Auto`] mesh axes removed.
    pub fn without_auto_axes(&self) -> Self {
        let dimensions = self
            .dimensions
            .iter()
            .map(|dimension| match dimension {
                ShardingDimension::Replicated => ShardingDimension::Replicated,
                ShardingDimension::Unconstrained => ShardingDimension::Unconstrained,
                ShardingDimension::Sharded(axis_names) => {
                    let axis_names = axis_names
                        .iter()
                        .filter(|name| {
                            matches!(self.mesh.axis_type(name), Some(MeshAxisType::Explicit | MeshAxisType::Manual))
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    if axis_names.is_empty() {
                        ShardingDimension::Replicated
                    } else {
                        ShardingDimension::Sharded(axis_names)
                    }
                }
            })
            .collect();
        let unreduced_axes = self
            .unreduced_axes
            .iter()
            .filter(|name| matches!(self.mesh.axis_type(name), Some(MeshAxisType::Explicit | MeshAxisType::Manual)))
            .cloned()
            .collect();
        let reduced_axes = self
            .reduced_axes
            .iter()
            .filter(|name| matches!(self.mesh.axis_type(name), Some(MeshAxisType::Explicit | MeshAxisType::Manual)))
            .cloned()
            .collect();
        Self { dimensions, unreduced_axes, reduced_axes, ..self.clone() }
    }

    // TODO(eaplatanios): Review this function. Also no tests.
    /// Returns whether this [`Sharding`] and `other` (which must share this sharding's mesh) disagree on any state
    /// that involves an [`Explicit`](MeshAxisType::Explicit) mesh axis: a per-dimension placement entry that differs
    /// while either side shards that dimension over an explicit axis, or an [`unreduced`](Self::unreduced_axes) /
    /// [`reduced`](Self::reduced_axes) axis set whose symmetric difference contains an explicit axis. Differences
    /// confined to [`Manual`](MeshAxisType::Manual) / [`Auto`](MeshAxisType::Auto) axes — and any
    /// [`varying_manual_axes`](Self::varying_manual_axes) difference — are ignored, mirroring how the dot, transpose,
    /// and reduce sharding rules gate their hard errors to explicit axes so that `shard_map` (manual) and
    /// compiler-propagated (auto) shardings pass through. Used by the operations that require their operands to be
    /// "sharded identically" (for example, concatenate and dynamic-update-slice) to decide whether a disagreement is
    /// actually an error.
    pub fn conflicts_on_explicit_axes_with(&self, other: &Sharding) -> bool {
        let dimension_has_explicit_axis = |dimension: &ShardingDimension| {
            matches!(dimension, ShardingDimension::Sharded(axis_names)
                if axis_names.iter().any(|name| self.mesh.axis_type(name) == Some(MeshAxisType::Explicit)))
        };
        if self.dimensions.len() != other.dimensions.len() {
            return true;
        }
        for (left, right) in self.dimensions.iter().zip(&other.dimensions) {
            if left != right && (dimension_has_explicit_axis(left) || dimension_has_explicit_axis(right)) {
                return true;
            }
        }
        let explicit_in_symmetric_difference = |left: &BTreeSet<String>, right: &BTreeSet<String>| {
            left.symmetric_difference(right)
                .any(|name| self.mesh.axis_type(name) == Some(MeshAxisType::Explicit))
        };
        explicit_in_symmetric_difference(&self.unreduced_axes, &other.unreduced_axes)
            || explicit_in_symmetric_difference(&self.reduced_axes, &other.reduced_axes)
    }


    // TODO(eaplatanios): Review this function. Also no tests.
    /// Returns a copy of this [`Sharding`] with the provided [`ShardingDimension`] inserted at dimension `index`,
    /// shifting all subsequent dimensions one position to the right. Batching rules use this to extend an explicit
    /// output sharding with an entry for a newly introduced batch dimension. The resulting sharding is revalidated,
    /// so inserting a [`ShardingDimension::Sharded`] entry that references unknown or already-used mesh axes fails.
    pub fn inserting_dimension(&self, index: usize, dimension: ShardingDimension) -> Result<Self, ShardingError> {
        if index > self.dimensions.len() {
            return Err(ShardingError::DimensionOutOfBounds { dimension: index, rank: self.rank() });
        }
        let mut dimensions = self.dimensions.clone();
        dimensions.insert(index, dimension);
        Self::with_manual_axes(
            self.mesh.clone(),
            dimensions,
            self.unreduced_axes.clone(),
            self.reduced_axes.clone(),
            self.varying_manual_axes.clone(),
        )
    }

    // TODO(eaplatanios): Review this function. Also no tests.
    /// Returns a copy of this [`Sharding`] with the dimension entry at `axis` removed, shifting subsequent dimensions
    /// one position to the left. This is the sharding-level analogue of removing an array dimension. The
    /// reduction-state sets are unchanged, but the removed entry's placement is reconciled with the manual-axis
    /// model: a dimension sharded over [`MeshAxisType::Manual`] axes moves those axes into the varying set (the value
    /// now varies across them rather than being placed along a ranked dimension), while a dimension sharded over a
    /// non-manual (e.g. [`MeshAxisType::Explicit`]) axis cannot be dropped structurally — that would silently discard
    /// an explicit placement that only a reduction or collective can remove — and yields a
    /// [`ShardingError::NonManualShardedDimensionRemoval`]. [`ShardingDimension::Replicated`] and
    /// [`ShardingDimension::Unconstrained`] entries are dropped without further effect.
    pub fn without_dimension(&self, axis: usize) -> Result<Self, ShardingError> {
        if axis >= self.dimensions.len() {
            return Err(ShardingError::DimensionOutOfBounds { dimension: axis, rank: self.rank() });
        }
        let mut dimensions = self.dimensions.clone();
        let removed_dimension = dimensions.remove(axis);
        let mut varying_manual_axes = self.varying_manual_axes.clone();
        if let ShardingDimension::Sharded(axis_names) = removed_dimension {
            for axis_name in axis_names {
                if self.mesh.axis_type(&axis_name) != Some(MeshAxisType::Manual) {
                    return Err(ShardingError::NonManualShardedDimensionRemoval { dimension: axis, name: axis_name });
                }
                varying_manual_axes.insert(axis_name);
            }
        }
        Self::with_manual_axes(
            self.mesh.clone(),
            dimensions,
            self.unreduced_axes.clone(),
            self.reduced_axes.clone(),
            varying_manual_axes,
        )
    }

    // TODO(eaplatanios): Review this function. Also no tests.
    /// Returns a copy of this [`Sharding`] whose dimension entries are reordered by `permutation`, so that output
    /// dimension `i` carries the entry of input dimension `permutation[i]`. This is the sharding-level analogue of an
    /// array axis permutation (transpose): each [`ShardingDimension`] follows its array dimension, while the
    /// reduction-state and manual-axis sets are unchanged. `permutation` must be a permutation of `0..rank` matching
    /// the rank of this [`Sharding`]; otherwise a [`ShardingError::DimensionOutOfBounds`] is returned.
    pub fn permuting_dimensions(&self, permutation: &[usize]) -> Result<Self, ShardingError> {
        if permutation.len() != self.dimensions.len() {
            return Err(ShardingError::DimensionOutOfBounds { dimension: permutation.len(), rank: self.rank() });
        }
        let mut dimensions = Vec::with_capacity(self.dimensions.len());
        for axis in permutation {
            let dimension = self
                .dimensions
                .get(*axis)
                .ok_or(ShardingError::DimensionOutOfBounds { dimension: *axis, rank: self.rank() })?;
            dimensions.push(dimension.clone());
        }
        Self::with_manual_axes(
            self.mesh.clone(),
            dimensions,
            self.unreduced_axes.clone(),
            self.reduced_axes.clone(),
            self.varying_manual_axes.clone(),
        )
    }
}

// TODO(eaplatanios): Review this function. Also no tests.
/// Returns the union of two mesh-axis-name sets. This is the shared helper used by sharding rules that combine the
/// reduction-state and manual-axis sets of two operands (for example, the dot product output sharding rule and the
/// `shard_map` output validation in `ryft-xla`).
pub fn merge_axis_sets(left: &BTreeSet<String>, right: &BTreeSet<String>) -> BTreeSet<String> {
    left.union(right).cloned().collect()
}

impl Display for Sharding {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        fn write_names<I, S>(formatter: &mut Formatter<'_>, names: I) -> std::fmt::Result
        where
            I: IntoIterator<Item = S>,
            S: AsRef<str>,
        {
            write!(formatter, "{{")?;
            write!(
                formatter,
                "{}",
                names
                    .into_iter()
                    .map(|name| format!("'{}'", name.as_ref().replace('\'', "\\'")))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
            write!(formatter, "}}")
        }

        write!(formatter, "{{mesh<[")?;
        write!(
            formatter,
            "{}",
            self.mesh
                .axes
                .iter()
                .map(|axis| format!("'{}'={}", axis.name.replace('\'', "\\'"), axis.size))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        write!(formatter, "]>")?;

        write!(formatter, ", [")?;
        write!(formatter, "{}", self.dimensions.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))?;
        write!(formatter, "]")?;

        if !self.unreduced_axes.is_empty() {
            write!(formatter, ", unreduced=")?;
            write_names(formatter, self.unreduced_axes.iter())?;
        }

        if !self.reduced_axes.is_empty() {
            write!(formatter, ", reduced=")?;
            write_names(formatter, self.reduced_axes.iter())?;
        }

        if !self.varying_manual_axes.is_empty() {
            write!(formatter, ", varying_manual=")?;
            write_names(formatter, self.varying_manual_axes.iter())?;
        }

        write!(formatter, "}}")
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

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

    #[test]
    fn test_device_mesh() {
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let devices = vec![Device::new(0, 0), Device::new(1, 0), Device::new(2, 1), Device::new(3, 1)];
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
            DeviceMesh::new(logical_mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0), Device::new(2, 1)],),
            Err(ShardingError::DeviceCountMismatch { expected: 4, actual: 3 }),
        ));
        assert!(matches!(
            DeviceMesh::new(
                logical_mesh.clone(),
                vec![Device::new(0, 0), Device::new(0, 0), Device::new(1, 1), Device::new(2, 1)],
            ),
            Err(ShardingError::DuplicateDeviceId { id }) if id == 0,
        ));
    }

    #[test]
    fn test_sharding_dimension() {
        assert_eq!(ShardingDimension::replicated().to_string(), "{}");
        assert_eq!(ShardingDimension::unconstrained().to_string(), "{?}");
        assert_eq!(ShardingDimension::sharded(["x"]).to_string(), "{'x'}");
        assert_eq!(ShardingDimension::sharded(["x", "y"]).to_string(), "{'x', 'y'}");
        assert_eq!(ShardingDimension::sharded([r"path\to", "x'y"]).to_string(), "{'path\\to', 'x\\'y'}");
    }

    #[test]
    fn test_sharding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();

        let sharding = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()],
            Vec::<&str>::new(),
            ["manual"],
            Vec::<&str>::new(),
        )
        .unwrap();
        assert_eq!(sharding.mesh, mesh.clone());
        assert_eq!(sharding.dimensions, vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()]);
        assert_eq!(sharding.unreduced_axes, BTreeSet::new());
        assert_eq!(sharding.reduced_axes, BTreeSet::from(["manual".to_string()]));
        assert_eq!(sharding.varying_manual_axes, BTreeSet::new());
        assert_eq!(sharding.rank(), 2);
        assert_eq!(sharding.partition_index(0, &[0, 0]), Ok(0));
        assert_eq!(sharding.partition_index(0, &[2, 1]), Ok(2));
        assert_eq!(sharding.partition_index(0, &[3, 0]), Ok(3));
        assert_eq!(sharding.partition_index(1, &[0, 0]), Ok(0));
        assert_eq!(sharding.partition_index(1, &[3, 1]), Ok(0));
        assert_eq!(
            sharding.partition_index(2, &[0, 0]),
            Err(ShardingError::DimensionOutOfBounds { dimension: 2, rank: 2 })
        );
        assert_eq!(sharding.replicated_axes(), Vec::<&str>::new());
        assert_eq!(sharding.to_string(), "{mesh<['data'=4, 'manual'=2]>, [{'data'}, {}], reduced={'manual'}}",);

        let replicated = Sharding::replicated(mesh.clone(), 3);
        assert_eq!(replicated.mesh, mesh);
        assert_eq!(
            replicated.dimensions,
            vec![ShardingDimension::replicated(), ShardingDimension::replicated(), ShardingDimension::replicated(),]
        );
        assert_eq!(replicated.unreduced_axes, BTreeSet::new());
        assert_eq!(replicated.reduced_axes, BTreeSet::new());
        assert_eq!(replicated.varying_manual_axes, BTreeSet::new());
        assert_eq!(replicated.rank(), 3);
        assert_eq!(replicated.partition_index(0, &[0, 0]), Ok(0));
        assert_eq!(replicated.partition_index(1, &[3, 1]), Ok(0));
        assert_eq!(replicated.partition_index(2, &[2, 0]), Ok(0));
        assert_eq!(
            replicated.partition_index(3, &[0, 0]),
            Err(ShardingError::DimensionOutOfBounds { dimension: 3, rank: 3 })
        );
        assert_eq!(replicated.replicated_axes(), Vec::from(["data", "manual"]));
        assert_eq!(replicated.to_string(), "{mesh<['data'=4, 'manual'=2]>, [{}, {}, {}]}");

        assert!(matches!(
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["z"])]),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));
        assert!(matches!(
            Sharding::new(mesh.clone(), vec![ShardingDimension::Sharded(Vec::new())]),
            Err(ShardingError::EmptySharding { dimension }) if dimension == 0,
        ));
        assert!(matches!(
            Sharding::with_manual_axes(
                mesh.clone(),
                vec![ShardingDimension::replicated()],
                ["manual"],
                Vec::<&str>::new(),
                ["manual"],
            ),
            Err(ShardingError::ConflictingVaryingAndUnreducedMeshAxis { name }) if name == "manual",
        ));
        assert!(matches!(
            Sharding::with_manual_axes(
                mesh,
                vec![ShardingDimension::replicated()],
                Vec::<&str>::new(),
                ["manual"],
                ["manual"],
            ),
            Err(ShardingError::ConflictingVaryingAndReducedMeshAxis { name }) if name == "manual",
        ));

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();

        let reduced = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::replicated()],
            Vec::<&str>::new(),
            ["y"],
            Vec::<&str>::new(),
        )
        .unwrap();
        assert_eq!(reduced.reduced_axes, BTreeSet::from(["y".to_string()]));

        assert!(matches!(
            Sharding::with_manual_axes(
                mesh.clone(),
                vec![ShardingDimension::replicated()],
                Vec::<&str>::new(),
                Vec::<&str>::new(),
                ["y"],
            ),
            Err(ShardingError::ExpectedManualMeshAxis { name }) if name == "y",
        ));

        assert!(matches!(
            Sharding::with_manual_axes(
                mesh,
                vec![ShardingDimension::replicated()],
                ["z"],
                ["z"],
                Vec::<&str>::new(),
            ),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "z",
        ));
    }

    #[test]
    fn test_sharding_without_auto_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("batch", 8, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("hidden", 16, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("reduction", 16, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("carry", 32, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::with_unreduced_axes(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["data", "model", "batch"]),
                ShardingDimension::sharded(["hidden"]),
                ShardingDimension::replicated(),
            ],
            ["reduction", "carry"],
        )
        .unwrap();
        assert_eq!(
            sharding.without_auto_axes(),
            Sharding::with_unreduced_axes(
                mesh,
                vec![
                    ShardingDimension::sharded(["data", "batch"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
                ["carry"],
            )
            .unwrap(),
        );

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("w", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::with_unreduced_axes(mesh.clone(), vec![ShardingDimension::sharded(["x", "y", "z"])], ["w"])
                .unwrap()
                .without_auto_axes();
        assert_eq!(sharding, Sharding::new(mesh, vec![ShardingDimension::sharded(["x", "z"])]).unwrap(),);
        assert!(sharding.replicated_axes().is_empty());
        assert!(sharding.unreduced_axes.is_empty());

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::replicated()],
            Vec::<&str>::new(),
            BTreeSet::from(["y".to_string(), "z".to_string()]),
            BTreeSet::from(["x".to_string()]),
        )
        .unwrap();
        assert_eq!(
            sharding.without_auto_axes(),
            Sharding::with_manual_axes(mesh, vec![ShardingDimension::replicated()], Vec::<&str>::new(), ["z"], ["x"])
                .unwrap(),
        );
    }

    // TODO(eaplatanios): Review this function.
    #[test]
    fn test_sharding_inserting_dimension() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("model", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data"])]).unwrap();

        assert_eq!(
            sharding.inserting_dimension(0, ShardingDimension::replicated()),
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["data"])],),
        );
        assert_eq!(
            sharding.inserting_dimension(1, ShardingDimension::sharded(["model"])),
            Sharding::new(
                mesh.clone(),
                vec![ShardingDimension::sharded(["data"]), ShardingDimension::sharded(["model"])],
            ),
        );
        assert!(matches!(
            sharding.inserting_dimension(2, ShardingDimension::replicated()),
            Err(ShardingError::DimensionOutOfBounds { dimension: 2, rank: 1 }),
        ));
        // The resulting sharding is revalidated, so reusing an already-used axis fails.
        assert!(matches!(
            sharding.inserting_dimension(0, ShardingDimension::sharded(["data"])),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "data",
        ));
    }
}
