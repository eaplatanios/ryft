use std::collections::{BTreeSet, HashMap, HashSet, hash_map::Entry};
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use thiserror::Error;

use ryft_macros::Parameter;
#[cfg(feature = "xla")]
use ryft_mlir::{Location, dialects::shardy};

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

    #[error("mesh device ID '{id}' appears more than once")]
    DuplicateMeshDeviceId { id: MeshDeviceId },

    #[error("mesh has {actual} device(s), but its axis sizes imply {expected} device(s)")]
    MeshDeviceCountMismatch { expected: usize, actual: usize },

    #[error("mesh mismatch; expected '{expected:?}' but got '{actual:?}'")]
    MeshMismatch { expected: LogicalMesh, actual: LogicalMesh },

    #[error("sharding dimension #{dimension} has no axes")]
    EmptySharding { dimension: usize },

    #[error("sharding rank ({sharding_rank}) does not match array rank ({array_rank})")]
    ShardingRankMismatch { sharding_rank: usize, array_rank: usize },

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

/// Canonical symbol name used for emitted Shardy [`LogicalMesh`] declarations and references.
#[cfg(feature = "xla")]
pub(crate) const SHARDY_MESH_SYMBOL_NAME: &str = "mesh";

#[cfg(feature = "xla")]
impl LogicalMesh {
    /// Creates a new [`shardy::DetachedMeshOperation`] that corresponds to this [`LogicalMesh`].
    /// The mesh in the returned operation will be named `"mesh"`.
    #[inline]
    pub fn to_shardy<'c, 't: 'c, L: Location<'c, 't>>(&self, location: L) -> shardy::DetachedMeshOperation<'c, 't> {
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
                expected: expected_device_count,
                actual: devices.len(),
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
    pub fn to_shardy<'c, 't: 'c, L: Location<'c, 't>>(&self, location: L) -> shardy::DetachedMeshOperation<'c, 't> {
        self.logical_mesh.to_shardy(location)
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
///     reduced_manual_axes: std::collections::BTreeSet::new(),
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
    /// [`LogicalMesh`] that describes the device topology underlying this [`Sharding`] and gives meaning to every
    /// [`MeshAxis`] name stored in it. This is effectively the coordinate system for the rest of this struct. Every
    /// axis name mentioned in [`Self::dimensions`], [`Self::unreduced_axes`], [`Self::reduced_manual_axes`], and
    /// [`Self::varying_manual_axes`] is resolved against this mesh.
    pub mesh: LogicalMesh,

    /// Ranked per-array dimension [`Sharding`] partition assignments. This is the array-rank-indexed part of this
    /// sharding: `dimensions[i]` describes how the logical array dimension `i` is partitioned across the mesh.
    /// For example, on a mesh with axes `("data", "model")`, the [`dimensions`](Self::dimensions) assignment
    /// `[ShardingDimension::sharded(["data"]), ShardingDimension::replicated()]` means that the first array
    /// dimension is split across `"data"` while the second array dimension is replicated on every device. This field
    /// intentionally does not try to encode every mesh-related fact about the value. Mesh axes that matter semantically
    /// but do not correspond to a ranked array dimension are stored separately in [`Self::unreduced_axes`],
    /// [`Self::reduced_manual_axes`], and [`Self::varying_manual_axes`].
    pub dimensions: Vec<ShardingDimension>,

    /// Mesh axes along which values carry per-device partial results. This is the "a cross-device reduction still
    /// needs to happen" marker. An axis can disappear from [`Self::dimensions`] after a local computation reduces over
    /// the corresponding array dimension, but the value may still not be truly replicated; each shard can still hold a
    /// different partial result that must later be combined across that mesh axis. Concretely, imagine a mesh with axes
    /// `("data", "model")` and a value whose first tensor dimension is sharded by `"data"`. If a local computation then
    /// sums over a `"model"`-partitioned feature dimension, the resulting value may have no ranked dimension left that
    /// mentions `"model"`, yet each `"model"` shard still owns a different partial sum. Setting `unreduced_axes` to
    /// `["model"]` preserves that fact. This is why this field is needed even though the mesh axis no longer appears in
    /// [`Self::dimensions`]; without it, an axis that is absent from ranked dimensions would be indistinguishable from
    /// ordinary replication.
    pub unreduced_axes: BTreeSet<String>,

    /// [`MeshAxisType::Manual`] mesh axes across which values are known to have already been reduce. This is the dual
    /// of [`Self::unreduced_axes`] though specific to manual axes because this property is implicit for other axis
    /// types. It records that a manual mesh axis has already been consumed by a reduction, even though that fact no
    /// longer has a direct ranked-dimension representation. A concrete `shard_map` example is an output that is
    /// replicated in [`Self::dimensions`] but was produced by first summing across the active manual axis `"data"`
    /// inside the mapped computation. In that case `reduced_manual_axes` being set to `["data"]` distinguishes "this
    /// value is already reduced across `data`" from both "this value is still unreduced across `data`" and "this axis
    /// was never relevant to the value". This field exists primarily for type-level `shard_map` reasoning and
    /// validation.
    pub reduced_manual_axes: BTreeSet<String>,

    /// [`MeshAxisType::Manual`] mesh axes for which `shard_map` values are known to vary along. Unlike
    /// [`Self::dimensions`], this is not a placement description. It answers a typing question used while tracing
    /// `shard_map`: if we compared two otherwise identical devices that differ only along one of these axes, could this
    /// local value still be different? A concrete nested-`shard_map` example is an outer map that is manual over `"y"`
    /// and an inner map whose input sharding specifications additionally shard the value over manual axis `"x"`. Inside
    /// the inner body, the local array can still have the same rank and local shape as before, but it now semantically
    /// varies across both manual axes, and so the trace has `varying_manual_axes` set to `["y", "x"]`. This is needed
    /// because neither ranked sharding nor reduction-state fields can say whether a local value is uniform across the
    /// active manual shards. For example, constants created inside `shard_map` preserve [`Self::unreduced_axes`] and
    /// [`Self::reduced_manual_axes`] but clear [`Self::varying_manual_axes`], because a constant does not vary from
    /// shard to shard even when it is traced under manual axes.
    pub varying_manual_axes: BTreeSet<String>,
}

impl Sharding {
    /// Creates a new [`Sharding`].
    pub fn new<
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
        reduced_manual_axes: RI,
        varying_manual_axes: VI,
    ) -> Result<Self, ShardingError> {
        let unreduced_axes = unreduced_axes.into_iter().map(Into::into).collect();
        let reduced_manual_axes = reduced_manual_axes.into_iter().map(Into::into).collect();
        let varying_manual_axes = varying_manual_axes.into_iter().map(Into::into).collect();
        let sharding = Self { mesh, dimensions, unreduced_axes, reduced_manual_axes, varying_manual_axes };

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

        for axis_name in &sharding.reduced_manual_axes {
            if !sharding.mesh.axis_indices.contains_key(axis_name) {
                return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
            }

            if sharding.mesh.axis_type(axis_name) != Some(MeshAxisType::Manual) {
                return Err(ShardingError::ExpectedManualMeshAxis { name: axis_name.clone() });
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
            reduced_manual_axes: BTreeSet::new(),
            varying_manual_axes: BTreeSet::new(),
        }
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
        used_axes.extend(self.reduced_manual_axes.iter().map(String::as_str));
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

    /// Returns a copy of this [`Sharding`] with all of its [`MeshAxisType::Auto`] mesh axes removed.
    pub(crate) fn without_auto_axes(&self) -> Self {
        let dimensions = self
            .dimensions
            .iter()
            .map(|dimension| match dimension {
                ShardingDimension::Replicated => ShardingDimension::Replicated,
                ShardingDimension::Unconstrained => ShardingDimension::Unconstrained,
                ShardingDimension::Sharded(axis_names) => {
                    let axis_names = axis_names
                        .iter()
                        .filter(|axis_name| match self.mesh.axis_type(axis_name) {
                            Some(MeshAxisType::Explicit | MeshAxisType::Manual) => true,
                            _ => false,
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
            .filter(|axis_name| match self.mesh.axis_type(axis_name) {
                Some(MeshAxisType::Explicit | MeshAxisType::Manual) => true,
                _ => false,
            })
            .cloned()
            .collect();
        Self { dimensions, unreduced_axes, ..self.clone() }
    }
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

        if !self.reduced_manual_axes.is_empty() {
            write!(formatter, ", reduced_manual=")?;
            write_names(formatter, self.reduced_manual_axes.iter())?;
        }

        if !self.varying_manual_axes.is_empty() {
            write!(formatter, ", varying_manual=")?;
            write_names(formatter, self.varying_manual_axes.iter())?;
        }

        write!(formatter, "}}")
    }
}

#[cfg(feature = "xla")]
impl Sharding {
    /// Creates a new [`shardy::TensorShardingAttributeRef`] that corresponds to this [`Sharding`].
    /// The returned attribute uses the canonical `@mesh` symbol name in the MLIR context associated with `location`.
    pub fn to_shardy<'c, 't: 'c, L: Location<'c, 't>>(
        &self,
        location: L,
    ) -> shardy::TensorShardingAttributeRef<'c, 't> {
        let context = location.context();
        let mesh_symbol_ref = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
        let dimensions = self
            .dimensions
            .iter()
            .map(|dimension| match dimension {
                ShardingDimension::Replicated => context.shardy_dimension_sharding([], true, None),
                ShardingDimension::Sharded(axis_names) => context.shardy_dimension_sharding(
                    axis_names.iter().map(|axis_name| context.shardy_axis_ref(axis_name, None)),
                    true,
                    None,
                ),
                ShardingDimension::Unconstrained => context.shardy_dimension_sharding([], false, None),
            })
            .collect::<Vec<_>>();
        let replicated_axes = self
            .replicated_axes()
            .iter()
            .map(|axis_name| context.shardy_axis_ref(axis_name, None))
            .collect::<Vec<_>>();
        let unreduced_axes = self
            .unreduced_axes
            .iter()
            .map(|axis_name| context.shardy_axis_ref(axis_name, None))
            .collect::<Vec<_>>();
        context.shardy_tensor_sharding(
            mesh_symbol_ref,
            dimensions.as_slice(),
            replicated_axes.as_slice(),
            unreduced_axes.as_slice(),
        )
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
    fn test_logical_mesh_to_shardy() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 1, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let context = MlirContext::new();
        let module = context.module(context.unknown_location());
        assert_eq!(
            module.body().append_operation(mesh.to_shardy(context.unknown_location())).to_string(),
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
            Err(ShardingError::MeshDeviceCountMismatch { expected: 4, actual: 3 }),
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
    fn test_device_mesh_to_shardy() {
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
            module.body().append_operation(mesh.to_shardy(context.unknown_location())).to_string(),
            format!("sdy.mesh @{SHARDY_MESH_SYMBOL_NAME} = <[\"x\"=2, \"y\"=2]>"),
        );
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

        let sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()],
            Vec::<&str>::new(),
            ["manual"],
            ["manual"],
        )
        .unwrap();
        assert_eq!(sharding.mesh, mesh.clone());
        assert_eq!(sharding.dimensions, vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()]);
        assert_eq!(sharding.unreduced_axes, BTreeSet::new());
        assert_eq!(sharding.reduced_manual_axes, BTreeSet::from(["manual".to_string()]));
        assert_eq!(sharding.varying_manual_axes, BTreeSet::from(["manual".to_string()]));
        assert_eq!(sharding.rank(), 2);
        assert_eq!(sharding.replicated_axes(), Vec::<&str>::new());
        assert_eq!(
            sharding.to_string(),
            "{mesh<['data'=4, 'manual'=2]>, [{'data'}, {}], reduced_manual={'manual'}, varying_manual={'manual'}}",
        );

        let replicated = Sharding::replicated(mesh.clone(), 3);
        assert_eq!(replicated.mesh, mesh);
        assert_eq!(
            replicated.dimensions,
            vec![ShardingDimension::replicated(), ShardingDimension::replicated(), ShardingDimension::replicated(),]
        );
        assert_eq!(replicated.unreduced_axes, BTreeSet::new());
        assert_eq!(replicated.reduced_manual_axes, BTreeSet::new());
        assert_eq!(replicated.varying_manual_axes, BTreeSet::new());
        assert_eq!(replicated.rank(), 3);
        assert_eq!(replicated.replicated_axes(), Vec::from(["data", "manual"]));
        assert_eq!(replicated.to_string(), "{mesh<['data'=4, 'manual'=2]>, [{}, {}, {}]}");
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
        let sharding = Sharding::new(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["data", "model", "batch"]),
                ShardingDimension::sharded(["hidden"]),
                ShardingDimension::replicated(),
            ],
            ["reduction", "carry"],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
        )
        .unwrap();
        assert_eq!(
            sharding.without_auto_axes(),
            Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["data", "batch"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
                ["carry"],
                Vec::<&str>::new(),
                Vec::<&str>::new(),
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
        let sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x", "y", "z"])],
            ["w"],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
        )
        .unwrap()
        .without_auto_axes();
        assert_eq!(
            sharding,
            Sharding::new(
                mesh,
                vec![ShardingDimension::sharded(["x", "z"])],
                Vec::<&str>::new(),
                Vec::<&str>::new(),
                Vec::<&str>::new(),
            )
            .unwrap(),
        );
        assert!(sharding.replicated_axes().is_empty());
        assert!(sharding.unreduced_axes.is_empty());

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::replicated()],
            Vec::<&str>::new(),
            BTreeSet::from(["x".to_string(), "z".to_string()]),
            BTreeSet::from(["x".to_string()]),
        )
        .unwrap();
        assert_eq!(
            sharding.without_auto_axes(),
            Sharding::new(mesh, vec![ShardingDimension::replicated()], Vec::<&str>::new(), ["x", "z"], ["x"],).unwrap(),
        );
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_sharding_to_shardy() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 6, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(
            mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            ["y"],
            Vec::<&str>::new(),
            Vec::<&str>::new(),
        )
        .unwrap();
        let context = MlirContext::new();
        assert_eq!(
            sharding.to_shardy(context.unknown_location()).to_string(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced={\"y\"}>",
        );
    }
}
