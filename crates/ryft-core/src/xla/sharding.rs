//! This module provides the core data structures for representing how arrays are _sharded_ (or _partitioned_) across
//! devices in a multi-device or multi-host environment. The design mirrors [JAX's sharding model][jax-sharding] and
//! supports conversion to [Shardy][shardy] MLIR dialect attributes for annotating StableHLO programs.
//!
//! [jax-sharding]: https://docs.jax.dev/en/latest/jax.sharding.html
//! [shardy]: https://openxla.org/shardy/overview
//!
//! # Relationship to JAX and Shardy
//!
//! The types in this module correspond directly to their JAX and Shardy (OpenXLA) counterparts:
//!
//! | Ryft type | JAX equivalent | Shardy MLIR representation |
//! |---|---|---|
//! | [`AbstractMesh`] | Topology of [`jax.sharding.Mesh`][jax-mesh] (axes only) | `sdy.mesh @name = <["axis"=size, ...]>` |
//! | [`Mesh`] | [`jax.sharding.Mesh`][jax-mesh] (axes + devices) | `sdy.mesh @name = <["axis"=size, ...]>` |
//! | [`MeshAxis`] | One entry in `Mesh.shape` | `MeshAxisAttr` (name + size pair) |
//! | [`MeshDevice`] | One element in `Mesh.devices` | Device ID in `MeshAttr.device_ids` |
//! | [`PartitionSpec`] | [`jax.sharding.PartitionSpec`][jax-pspec] | Array of `DimensionShardingAttr` |
//! | [`PartitionDimension`] | One element of a `PartitionSpec` | `DimensionShardingAttr` |
//! | [`NamedSharding`] | [`jax.sharding.NamedSharding`][jax-ns] | `#sdy.sharding<@mesh, [dim_shardings...]>` |
//! | [`ShardingLayout`] | Computed internally by `jax.Array` | — (runtime metadata, no MLIR form) |
//! | [`ShardDescriptor`] | `jax.Shard` (from `array.global_shards`) | — (runtime metadata) |
//!
//! [jax-mesh]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh
//! [jax-pspec]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec
//! [jax-ns]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.NamedSharding
//!
//! # Abstract mesh vs concrete mesh
//!
//! [`AbstractMesh`] captures only the logical topology (axis names and sizes) and is used
//! wherever device identity is irrelevant — principally in [`NamedSharding`] and for
//! rendering Shardy MLIR attributes at compilation time.
//!
//! [`Mesh`] wraps an [`AbstractMesh`] and adds a concrete device list, which is needed at
//! runtime for computing per-device shard metadata in [`ShardingLayout`].
//!
//! # Sharding context
//!
//! [`ShardingContext`] controls how unsharded and unconstrained dimensions are rendered in
//! Shardy MLIR. `ExplicitSharding` produces closed dimensions (`{}`), while
//! `ShardingConstraint` leaves unsharded dimensions open (`{?}`), allowing the Shardy
//! propagator to fill them in.
//!
//! # Practical usage
//!
//! The typical workflow for sharded computation mirrors JAX's model:
//!
//! 1. **Create a device mesh** that organizes available devices into a named logical grid:
//!
//!    ```ignore
//!    // 1D mesh for data parallelism across 8 devices.
//!    // JAX equivalent: Mesh(devices, ('batch',))
//!    let mesh = Mesh::new(
//!        vec![MeshAxis::new("batch", 8)?],
//!        mesh_devices,
//!    )?;
//!
//!    // 2D mesh for data + model parallelism.
//!    // JAX equivalent: Mesh(np.array(devices).reshape(4, 2), ('data', 'model'))
//!    let mesh = Mesh::new(
//!        vec![MeshAxis::new("data", 4)?, MeshAxis::new("model", 2)?],
//!        mesh_devices,
//!    )?;
//!    ```
//!
//! 2. **Create partition specifications** that describe how each array dimension maps to mesh axes:
//!
//!    ```ignore
//!    // Shard dim 0 along "data", replicate dim 1.
//!    // JAX equivalent: PartitionSpec('data', None)
//!    let spec = PartitionSpec::new(vec![
//!        PartitionDimension::sharded("data"),
//!        PartitionDimension::unsharded(),
//!    ]);
//!
//!    // Shard dim 0 along both "data" and "model" axes.
//!    // JAX equivalent: PartitionSpec(('data', 'model'),)
//!    let spec = PartitionSpec::new(vec![
//!        PartitionDimension::sharded_by(["data", "model"]),
//!    ]);
//!
//!    // Fully replicated across all devices.
//!    // JAX equivalent: PartitionSpec()
//!    let spec = PartitionSpec::replicated(2);
//!
//!    // Unconstrained dimension (let the propagator decide).
//!    // JAX equivalent: PartitionSpec.UNCONSTRAINED
//!    let spec = PartitionSpec::new(vec![
//!        PartitionDimension::unconstrained(),
//!    ]);
//!    ```
//!
//! 3. **Combine mesh and partition spec** into a named sharding:
//!
//!    ```ignore
//!    // JAX equivalent: NamedSharding(mesh, spec)
//!    let sharding = NamedSharding::new(mesh.abstract_mesh().clone(), spec)?;
//!    ```
//!
//! 4. **Compute shard metadata** to determine per-device array slices and identify addressable
//!    shards:
//!
//!    ```ignore
//!    let layout = ShardingLayout::new(vec![32, 128], mesh, partition_spec)?;
//!
//!    // Inspect all global shards (like `array.global_shards` in JAX):
//!    for shard in layout.shards() {
//!        println!(
//!            "shard {} on device {:?}: shape {:?}, slices {:?}",
//!            shard.shard_index(), shard.device(), shard.shape(), shard.slices(),
//!        );
//!    }
//!
//!    // Find shards local to the current host process (like `array.addressable_shards` in JAX):
//!    let local_shard_indices = layout.shard_indices_for_process(0);
//!    ```
//!
//! 5. **Convert to Shardy MLIR attributes** for StableHLO program annotation:
//!
//!    ```ignore
//!    // Generates: sdy.mesh @mesh = <["data"=4, "model"=2]>
//!    let mesh_op = mesh.abstract_mesh().to_shardy_mesh_operation("mesh")?;
//!
//!    // Generates: #sdy.sharding<@mesh, [{"data"}, {}]>
//!    let attr = sharding.to_shardy_tensor_sharding_attribute("mesh")?;
//!    ```
//!
//! # Multi-host and addressability
//!
//! In multi-host (multi-process) execution, each host owns a disjoint subset of devices.
//! [`MeshDevice`] records both the global device ID and the owning process index, mirroring
//! JAX's distinction between [`jax.devices()`][jax-devices] (all global devices) and
//! [`jax.local_devices()`][jax-local-devices] (devices addressable by the current process).
//!
//! [jax-devices]: https://docs.jax.dev/en/latest/jax.html#jax.devices
//! [jax-local-devices]: https://docs.jax.dev/en/latest/jax.html#jax.local_devices
//!
//! [`ShardingLayout`] computes metadata for *all* global shards—including those on remote
//! hosts—so that the full sharding picture is available for compilation. To identify which
//! shards are locally addressable, use [`ShardingLayout::shard_indices_for_process`] with
//! the current process index. Only addressable shards can be backed by actual PJRT buffers;
//! non-addressable shards exist only as metadata describing remote device placements.
//!
//! This mirrors JAX's `array.addressable_shards` (local) vs `array.global_shards` (all),
//! where accessing `.data` on a non-addressable shard raises an error.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use ryft_pjrt::DeviceId;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Error type for mesh/sharding definitions and shard layout construction.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShardingError {
    /// Error returned when a mesh axis name is empty.
    #[error("mesh axis names must be non-empty")]
    EmptyMeshAxisName,

    /// Error returned when a mesh axis has size `0`.
    #[error("mesh axis '{axis_name}' must have size > 0")]
    InvalidMeshAxisSize { axis_name: String },

    /// Error returned when mesh axis names are not unique.
    #[error("mesh axis '{axis_name}' appears more than once")]
    DuplicateMeshAxisName { axis_name: String },

    /// Error returned when device IDs in a mesh are not unique.
    #[error("mesh device id {device_id} appears more than once")]
    DuplicateMeshDeviceId { device_id: DeviceId },

    /// Error returned when the number of mesh devices does not match the product of axis sizes.
    #[error("mesh has {actual_device_count} device(s), but axis sizes imply {expected_device_count} device(s)")]
    MeshDeviceCountMismatch { expected_device_count: usize, actual_device_count: usize },

    /// Error returned when partitioning references a mesh axis that does not exist.
    #[error("partitioning references unknown mesh axis '{axis_name}'")]
    UnknownMeshAxis { axis_name: String },

    /// Error returned when a partitioned dimension references no mesh axes.
    #[error("partition specification dimension #{dimension} has an empty mesh-axis list")]
    EmptyPartitionAxisList { dimension: usize },

    /// Error returned when a mesh axis appears more than once in the partition specification.
    #[error("mesh axis '{axis_name}' is used multiple times in the partition specification")]
    DuplicatePartitionAxis { axis_name: String },

    /// Error returned when a mesh symbol name used to render Shardy attributes is empty.
    #[error("mesh symbol names used in Shardy attributes must be non-empty")]
    EmptyMeshSymbolName,

    /// Error returned when a mesh symbol name used to render Shardy attributes is invalid.
    #[error("invalid mesh symbol name '{mesh_symbol_name}' used in Shardy attributes")]
    InvalidMeshSymbolName { mesh_symbol_name: String },

    /// Error returned when a partition specification rank does not match the array rank.
    #[error("partition specification rank {partition_rank} does not match array rank {array_rank}")]
    RankMismatch { partition_rank: usize, array_rank: usize },

    /// Error returned when computing partition slices with an invalid partition count.
    #[error("partition count must be > 0")]
    InvalidPartitionCount,

    /// Error returned when a partition index is out of range.
    #[error("partition index {partition_index} is out of range for partition count {partition_count}")]
    InvalidPartitionIndex { partition_index: usize, partition_count: usize },

    /// Error returned when an invalid slice range is constructed.
    #[error("invalid shard slice range [{start}, {end})")]
    InvalidShardSlice { start: usize, end: usize },

    /// Error returned when arithmetic overflows while building shard metadata.
    #[error("overflow while {context}")]
    Overflow { context: String },

    /// Error returned when an `Unconstrained` dimension appears in a [`ShardingLayout`].
    ///
    /// `Unconstrained` dimensions are valid in [`PartitionSpec`] and [`NamedSharding`] (they
    /// render as open `{?}` in Shardy), but cannot be used to compute concrete shard slices.
    #[error("partition specification dimension #{dimension} is unconstrained and cannot be used in a sharding layout")]
    UnconstrainedInLayout { dimension: usize },

    /// Error returned when the number of axis types does not match the number of axes.
    #[error("expected {expected} axis type(s), but got {actual}")]
    AxisTypeCountMismatch { expected: usize, actual: usize },

    /// Error returned when a replicated/unreduced axis in a [`NamedSharding`] does not exist in the mesh.
    #[error("replicated/unreduced axis '{axis_name}' does not exist in the mesh")]
    UnknownExtraAxis { axis_name: String },

    /// Error returned when a replicated/unreduced axis in a [`NamedSharding`] is already used
    /// in the partition specification.
    #[error("replicated/unreduced axis '{axis_name}' is already used in the partition specification")]
    ExtraAxisConflictsWithPartition { axis_name: String },

    /// Error returned when the same axis appears in both the replicated and unreduced sets.
    #[error("axis '{axis_name}' appears in both replicated and unreduced sets")]
    AxisInBothReplicatedAndUnreduced { axis_name: String },
}

// ---------------------------------------------------------------------------
// Axis type
// ---------------------------------------------------------------------------

/// Per-axis property that controls sharding propagation behavior.
///
/// Each axis in a mesh can be tagged with an `AxisType` that tells the compiler (Shardy/GSPMD)
/// how to treat shardings along that axis during propagation.
///
/// # JAX equivalent
///
/// Corresponds to [`jax.sharding.AxisType`][jax-axis-type]:
///
/// | Variant | Meaning |
/// |---|---|
/// | `Auto` | Compiler decides sharding automatically (default) |
/// | `Explicit` | Sharding is part of the type system, propagated at trace time |
/// | `Manual` | User manages all device communication (used with `shard_map`) |
///
/// [jax-axis-type]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.AxisType
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum AxisType {
    /// Compiler (Shardy/GSPMD) decides sharding automatically.
    #[default]
    Auto,
    /// Sharding is part of the type system, propagated at trace time.
    Explicit,
    /// User manages all device communication (used with `shard_map`).
    Manual,
}

// ---------------------------------------------------------------------------
// Sharding context
// ---------------------------------------------------------------------------

/// Controls how partition dimensions are rendered in Shardy MLIR attributes.
///
/// In Shardy's sharding representation, each dimension can be *closed* (fixed set of axes,
/// propagator will not change it) or *open* (ends with `?`, propagator may add axes).
///
/// | Variant | `Unsharded` renders as | `Sharded(["x"])` renders as | `Unconstrained` renders as |
/// |---|---|---|---|
/// | `ExplicitSharding` | `{}` (closed) | `{"x"}` (closed) | `{?}` (open) |
/// | `ShardingConstraint` | `{?}` (open) | `{"x"}` (closed) | `{?}` (open) |
///
/// Use `ExplicitSharding` when the sharding is fully determined (e.g., from a runtime array
/// in a JIT flow). Use `ShardingConstraint` when annotating an intermediate value where
/// unsharded dimensions should remain open for the propagator.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShardingContext {
    /// Fully determined sharding: unsharded dimensions are closed (`{}`).
    ExplicitSharding,
    /// Constraint hint: unsharded dimensions are open (`{?}`), allowing the propagator to
    /// assign axes.
    ShardingConstraint,
}

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

/// A named axis in a logical device mesh.
///
/// Each axis represents one dimension of the device grid with a human-readable name and a
/// size (the number of devices along that dimension).
///
/// # JAX equivalent
///
/// In JAX, mesh axes are defined implicitly by the shape and `axis_names` arguments to
/// [`jax.sharding.Mesh`][jax-mesh]. For example, in
/// `Mesh(np.array(devices).reshape(4, 2), ('data', 'model'))`, the `'data'` axis has size 4
/// and the `'model'` axis has size 2. Here, each axis is an explicit first-class value:
///
/// ```ignore
/// let data_axis = MeshAxis::new("data", 4)?;   // JAX: axis_names[0]='data', shape=(4,...)
/// let model_axis = MeshAxis::new("model", 2)?;  // JAX: axis_names[1]='model', shape=(...,2)
/// ```
///
/// [jax-mesh]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh
///
/// # Shardy representation
///
/// Each `MeshAxis` corresponds to one [`MeshAxisAttr`][sdy-mesh] entry in a Shardy
/// `sdy.mesh` operation:
///
/// ```mlir
/// sdy.mesh @mesh = <["data"=4, "model"=2]>
/// //                 ^^^^^^^^^  ^^^^^^^^^^
/// //                 MeshAxis   MeshAxis
/// ```
///
/// [sdy-mesh]: https://openxla.org/shardy/sdy_dialect#sdymesh_sdymeshop
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshAxis {
    name: String,
    size: usize,
}

impl MeshAxis {
    /// Creates a mesh axis.
    pub fn new<N: Into<String>>(name: N, size: usize) -> Result<Self, ShardingError> {
        let name = name.into();
        if name.is_empty() {
            return Err(ShardingError::EmptyMeshAxisName);
        }
        if size == 0 {
            return Err(ShardingError::InvalidMeshAxisSize { axis_name: name });
        }
        Ok(Self { name, size })
    }

    /// Name of this axis.
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Size of this axis.
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Device entry in a logical mesh.
///
/// Separates global device identity ([`DeviceId`]) from host/process ownership
/// (`process_index`), enabling the same sharding metadata to describe both local and remote
/// shards. This mirrors [`jax.Device`][jax-device], where each device has:
///
/// - A globally unique `id` (from `device.id`).
/// - A `process_index` indicating which host owns it (from `device.process_index`).
///
/// [jax-device]: https://docs.jax.dev/en/latest/jax.html#jax.Device
///
/// In a single-host setup all devices share `process_index = 0`. In multi-host setups,
/// `process_index` determines addressability: a shard on a device is *addressable* from
/// process `p` if and only if `device.process_index == p`. This is the same rule JAX uses
/// to distinguish [`jax.local_devices()`][jax-local] from [`jax.devices()`][jax-all].
///
/// [jax-local]: https://docs.jax.dev/en/latest/jax.html#jax.local_devices
/// [jax-all]: https://docs.jax.dev/en/latest/jax.html#jax.devices
///
/// # Shardy representation
///
/// In Shardy's `MeshAttr`, devices are represented by an optional `device_ids` list. When
/// omitted, devices follow implicit iota ordering `[0, 1, 2, ...]`, which matches the
/// row-major storage used by [`Mesh`]. Explicit device IDs are only needed when the
/// physical-to-logical mapping is non-trivial.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshDevice {
    id: DeviceId,
    process_index: usize,
}

impl MeshDevice {
    /// Creates a mesh-device entry.
    pub fn new(id: DeviceId, process_index: usize) -> Self {
        Self { id, process_index }
    }

    /// Global PJRT device ID.
    pub fn id(&self) -> DeviceId {
        self.id
    }

    /// Process index of the host owning this device.
    pub fn process_index(&self) -> usize {
        self.process_index
    }
}

/// Abstract logical mesh topology (axes only, no devices).
///
/// An `AbstractMesh` captures the axis names and sizes of a device mesh without binding to
/// specific physical devices. This is the compilation-time view of a mesh: it provides enough
/// information to generate Shardy MLIR attributes and validate partition specifications, but
/// does not carry device identity or process ownership.
///
/// Use [`AbstractMesh`] wherever device placement is irrelevant — principally in
/// [`NamedSharding`] and for rendering Shardy attributes.
///
/// # JAX equivalent
///
/// This corresponds to the topology portion of [`jax.sharding.Mesh`][jax-mesh] — the
/// `axis_names` and `shape` without the `devices` array.
///
/// [jax-mesh]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh
///
/// # Shardy representation
///
/// Rendered as an `sdy.mesh` operation via
/// [`to_shardy_mesh_operation`][AbstractMesh::to_shardy_mesh_operation]:
///
/// ```mlir
/// sdy.mesh @mesh = <["data"=4, "model"=2]>
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbstractMesh {
    axes: Vec<MeshAxis>,
    axis_types: Vec<AxisType>,
    axis_index_by_name: HashMap<String, usize>,
}

impl AbstractMesh {
    /// Creates an abstract mesh from named axes.
    ///
    /// All axis types default to [`AxisType::Auto`], matching JAX's `Mesh` constructor behavior.
    ///
    /// Validates that all axis names are non-empty, all sizes are positive, and names are unique.
    pub fn new(axes: Vec<MeshAxis>) -> Result<Self, ShardingError> {
        let axis_types = vec![AxisType::Auto; axes.len()];
        Self::build(axes, axis_types)
    }

    /// Creates an abstract mesh from named axes with explicit per-axis types.
    ///
    /// Returns [`ShardingError::AxisTypeCountMismatch`] if `axis_types.len() != axes.len()`.
    pub fn with_axis_types(axes: Vec<MeshAxis>, axis_types: Vec<AxisType>) -> Result<Self, ShardingError> {
        if axis_types.len() != axes.len() {
            return Err(ShardingError::AxisTypeCountMismatch { expected: axes.len(), actual: axis_types.len() });
        }
        Self::build(axes, axis_types)
    }

    fn build(axes: Vec<MeshAxis>, axis_types: Vec<AxisType>) -> Result<Self, ShardingError> {
        let mut axis_index_by_name = HashMap::with_capacity(axes.len());
        for (axis_index, axis) in axes.iter().enumerate() {
            if axis.name.is_empty() {
                return Err(ShardingError::EmptyMeshAxisName);
            }
            if axis.size == 0 {
                return Err(ShardingError::InvalidMeshAxisSize { axis_name: axis.name.clone() });
            }
            if axis_index_by_name.insert(axis.name.clone(), axis_index).is_some() {
                return Err(ShardingError::DuplicateMeshAxisName { axis_name: axis.name.clone() });
            }
        }
        Ok(Self { axes, axis_types, axis_index_by_name })
    }

    /// Returns the axes of this mesh.
    pub fn axes(&self) -> &[MeshAxis] {
        self.axes.as_slice()
    }

    /// Returns the per-axis types.
    pub fn axis_types(&self) -> &[AxisType] {
        self.axis_types.as_slice()
    }

    /// Returns axis names as a convenience accessor.
    pub fn axis_names(&self) -> Vec<&str> {
        self.axes.iter().map(|a| a.name()).collect()
    }

    /// Returns axis sizes as a convenience accessor.
    pub fn axis_sizes(&self) -> Vec<usize> {
        self.axes.iter().map(|a| a.size()).collect()
    }

    /// Returns the total number of devices implied by axis sizes.
    pub fn device_count(&self) -> Result<usize, ShardingError> {
        self.axes.iter().try_fold(1usize, |count, axis| {
            count.checked_mul(axis.size).ok_or_else(|| ShardingError::Overflow {
                context: "computing mesh device count from axis sizes".to_string(),
            })
        })
    }

    /// Returns the index of `axis_name` in this mesh, if present.
    pub fn axis_index<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.axis_index_by_name.get(axis_name.as_ref()).copied()
    }

    /// Returns the size of `axis_name` in this mesh, if present.
    pub fn axis_size<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.axis_index(axis_name).map(|axis_index| self.axes[axis_index].size)
    }

    /// Returns `true` if all axes have type [`AxisType::Auto`].
    pub fn are_all_axes_auto(&self) -> bool {
        self.axis_types.iter().all(|t| *t == AxisType::Auto)
    }

    /// Returns `true` if all axes have type [`AxisType::Explicit`].
    pub fn are_all_axes_explicit(&self) -> bool {
        self.axis_types.iter().all(|t| *t == AxisType::Explicit)
    }

    /// Returns `true` if all axes have type [`AxisType::Manual`].
    pub fn are_all_axes_manual(&self) -> bool {
        self.axis_types.iter().all(|t| *t == AxisType::Manual)
    }

    /// Returns the names of axes with type [`AxisType::Auto`].
    pub fn auto_axes(&self) -> Vec<&str> {
        self.axes_with_type(AxisType::Auto)
    }

    /// Returns the names of axes with type [`AxisType::Explicit`].
    pub fn explicit_axes(&self) -> Vec<&str> {
        self.axes_with_type(AxisType::Explicit)
    }

    /// Returns the names of axes with type [`AxisType::Manual`].
    pub fn manual_axes(&self) -> Vec<&str> {
        self.axes_with_type(AxisType::Manual)
    }

    fn axes_with_type(&self, axis_type: AxisType) -> Vec<&str> {
        self.axes
            .iter()
            .zip(self.axis_types.iter())
            .filter_map(|(axis, t)| (*t == axis_type).then_some(axis.name()))
            .collect()
    }

    /// Returns a new `AbstractMesh` with selected axes' types changed.
    ///
    /// Unknown names in `name_to_type` are silently ignored, matching JAX behavior.
    pub fn with_updated_axis_types(&self, name_to_type: &HashMap<String, AxisType>) -> Self {
        let axis_types = self
            .axes
            .iter()
            .zip(self.axis_types.iter())
            .map(|(axis, current_type)| name_to_type.get(axis.name()).copied().unwrap_or(*current_type))
            .collect();
        Self { axes: self.axes.clone(), axis_types, axis_index_by_name: self.axis_index_by_name.clone() }
    }

    /// Renders this mesh as the right-hand side of a Shardy `sdy.mesh` declaration.
    ///
    /// Example output for axes `("x", 8)` and `("y", 2)`:
    /// `<["x"=8, "y"=2]>`.
    pub fn to_shardy_mesh_literal(&self) -> String {
        let mut literal = String::from("<[");
        for (axis_index, axis) in self.axes.iter().enumerate() {
            if axis_index > 0 {
                literal.push_str(", ");
            }
            literal.push('"');
            literal.push_str(escape_shardy_string(axis.name()).as_str());
            literal.push_str("\"=");
            literal.push_str(axis.size().to_string().as_str());
        }
        literal.push_str("]>");
        literal
    }

    /// Renders a complete Shardy `sdy.mesh` operation declaration.
    ///
    /// The output is a valid MLIR operation that can be placed at module scope in a StableHLO
    /// program. For example:
    ///
    /// ```text
    /// sdy.mesh @mesh = <["data"=4, "model"=2]>
    /// ```
    ///
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used in MLIR (without or with leading `'@'`).
    pub fn to_shardy_mesh_operation<S: AsRef<str>>(&self, mesh_symbol_name: S) -> Result<String, ShardingError> {
        let mesh_symbol_name = normalize_mesh_symbol_name(mesh_symbol_name)?;
        Ok(format!("sdy.mesh @{mesh_symbol_name} = {}", self.to_shardy_mesh_literal()))
    }
}

/// Logical mesh of devices used by sharding layouts.
///
/// A mesh organizes physical devices into a multi-dimensional grid where each dimension has
/// a human-readable name. Devices are stored in **row-major order** with respect to the axis
/// list: for a 2D mesh with axes `("data"=4, "model"=2)`, the device at mesh coordinate
/// `(i, j)` has linear index `i * 2 + j`. This matches NumPy's default C-order and JAX's
/// `mesh.devices.flat`.
///
/// A `Mesh` wraps an [`AbstractMesh`] (the logical topology) and adds a concrete device
/// list. Use [`abstract_mesh()`][Mesh::abstract_mesh] to access the topology-only view,
/// which is needed for [`NamedSharding`] and Shardy attribute rendering.
///
/// # JAX equivalent
///
/// This corresponds directly to [`jax.sharding.Mesh`][jax-mesh]:
///
/// | JAX | Ryft |
/// |---|---|
/// | `Mesh(np.array(devs).reshape(4, 2), ('data', 'model'))` | `Mesh::new(vec![MeshAxis("data", 4), MeshAxis("model", 2)], devs)` |
/// | `mesh.shape` | `mesh.abstract_mesh().axes()` |
/// | `mesh.devices` (ndarray) | `mesh.devices()` (flat row-major slice) |
/// | `mesh.size` | `mesh.device_count()` |
/// | `mesh.local_devices` | filter `mesh.devices()` by `process_index` |
///
/// [jax-mesh]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh
///
/// # Shardy representation
///
/// Rendered via the inner [`AbstractMesh`]:
///
/// ```mlir
/// sdy.mesh @mesh = <["data"=4, "model"=2]>
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mesh {
    abstract_mesh: AbstractMesh,
    devices: Vec<MeshDevice>,
    device_index_by_id: HashMap<DeviceId, usize>,
}

impl Mesh {
    /// Creates a mesh from named axes and row-major devices.
    ///
    /// All axis types default to [`AxisType::Auto`].
    /// The expected number of `devices` is the product of all `axes` sizes. For an empty axis list, the
    /// expected device count is `1`.
    pub fn new(axes: Vec<MeshAxis>, devices: Vec<MeshDevice>) -> Result<Self, ShardingError> {
        let abstract_mesh = AbstractMesh::new(axes)?;
        Self::from_abstract(abstract_mesh, devices)
    }

    /// Creates a mesh from named axes, explicit per-axis types, and row-major devices.
    pub fn with_axis_types(
        axes: Vec<MeshAxis>,
        axis_types: Vec<AxisType>,
        devices: Vec<MeshDevice>,
    ) -> Result<Self, ShardingError> {
        let abstract_mesh = AbstractMesh::with_axis_types(axes, axis_types)?;
        Self::from_abstract(abstract_mesh, devices)
    }

    /// Creates a mesh from a pre-validated [`AbstractMesh`] and row-major devices.
    pub fn from_abstract(abstract_mesh: AbstractMesh, devices: Vec<MeshDevice>) -> Result<Self, ShardingError> {
        let expected_device_count = abstract_mesh.device_count()?;
        if devices.len() != expected_device_count {
            return Err(ShardingError::MeshDeviceCountMismatch {
                expected_device_count,
                actual_device_count: devices.len(),
            });
        }

        let mut device_index_by_id = HashMap::with_capacity(devices.len());
        for (device_index, device) in devices.iter().enumerate() {
            if device_index_by_id.insert(device.id, device_index).is_some() {
                return Err(ShardingError::DuplicateMeshDeviceId { device_id: device.id });
            }
        }

        Ok(Self { abstract_mesh, devices, device_index_by_id })
    }

    /// Returns the abstract mesh (topology only).
    pub fn abstract_mesh(&self) -> &AbstractMesh {
        &self.abstract_mesh
    }

    /// Returns the axes of this mesh.
    pub fn axes(&self) -> &[MeshAxis] {
        self.abstract_mesh.axes()
    }

    /// Returns the per-axis types.
    pub fn axis_types(&self) -> &[AxisType] {
        self.abstract_mesh.axis_types()
    }

    /// Returns axis names as a convenience accessor.
    pub fn axis_names(&self) -> Vec<&str> {
        self.abstract_mesh.axis_names()
    }

    /// Returns axis sizes as a convenience accessor.
    pub fn axis_sizes(&self) -> Vec<usize> {
        self.abstract_mesh.axis_sizes()
    }

    /// Returns mesh devices in row-major order.
    pub fn devices(&self) -> &[MeshDevice] {
        self.devices.as_slice()
    }

    /// Returns the number of devices in this mesh.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Returns the index of `axis_name` in this mesh, if present.
    pub fn axis_index<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.abstract_mesh.axis_index(axis_name)
    }

    /// Returns the size of `axis_name` in this mesh, if present.
    pub fn axis_size<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.abstract_mesh.axis_size(axis_name)
    }

    /// Returns `true` if all axes have type [`AxisType::Auto`].
    pub fn are_all_axes_auto(&self) -> bool {
        self.abstract_mesh.are_all_axes_auto()
    }

    /// Returns `true` if all axes have type [`AxisType::Explicit`].
    pub fn are_all_axes_explicit(&self) -> bool {
        self.abstract_mesh.are_all_axes_explicit()
    }

    /// Returns `true` if all axes have type [`AxisType::Manual`].
    pub fn are_all_axes_manual(&self) -> bool {
        self.abstract_mesh.are_all_axes_manual()
    }

    /// Returns the names of axes with type [`AxisType::Auto`].
    pub fn auto_axes(&self) -> Vec<&str> {
        self.abstract_mesh.auto_axes()
    }

    /// Returns the names of axes with type [`AxisType::Explicit`].
    pub fn explicit_axes(&self) -> Vec<&str> {
        self.abstract_mesh.explicit_axes()
    }

    /// Returns the names of axes with type [`AxisType::Manual`].
    pub fn manual_axes(&self) -> Vec<&str> {
        self.abstract_mesh.manual_axes()
    }

    /// Returns the row-major mesh index of `device_id`, if present.
    pub fn device_index(&self, device_id: DeviceId) -> Option<usize> {
        self.device_index_by_id.get(&device_id).copied()
    }

    /// Returns just the device IDs.
    pub fn device_ids(&self) -> Vec<DeviceId> {
        self.devices.iter().map(|d| d.id()).collect()
    }

    /// Returns `true` if any two devices belong to different processes.
    pub fn is_multi_process(&self) -> bool {
        let mut seen = None;
        for device in &self.devices {
            match seen {
                None => seen = Some(device.process_index()),
                Some(p) if p != device.process_index() => return true,
                _ => {}
            }
        }
        false
    }

    /// Returns devices belonging to the given `process_index`.
    pub fn local_devices(&self, process_index: usize) -> Vec<&MeshDevice> {
        self.devices.iter().filter(|d| d.process_index() == process_index).collect()
    }

    /// Returns the mesh coordinate of the device at `device_index`, if valid.
    pub fn coordinate_for_device_index(&self, device_index: usize) -> Option<Vec<usize>> {
        (device_index < self.devices.len()).then(|| {
            let axis_sizes = self.abstract_mesh.axes.iter().map(MeshAxis::size).collect::<Vec<_>>();
            coordinate_for_linear_index(device_index, axis_sizes.as_slice())
        })
    }

    /// Returns the mesh coordinate of `device_id`, if present.
    pub fn coordinate_for_device(&self, device_id: DeviceId) -> Option<Vec<usize>> {
        self.device_index(device_id).and_then(|device_index| self.coordinate_for_device_index(device_index))
    }
}

// ---------------------------------------------------------------------------
// Partition specification
// ---------------------------------------------------------------------------

/// Partitioning assignment for one logical array dimension.
///
/// Each element of a [`PartitionSpec`] is a `PartitionDimension` describing how the
/// corresponding tensor dimension is distributed across mesh axes.
///
/// # JAX equivalent
///
/// This is equivalent to one entry in JAX's [`PartitionSpec`][jax-pspec]:
///
/// | JAX `PartitionSpec` element | `PartitionDimension` |
/// |---|---|
/// | `None` | [`Unsharded`][PartitionDimension::Unsharded] |
/// | `'axis_name'` | [`sharded("axis_name")`][PartitionDimension::sharded] |
/// | `('axis_a', 'axis_b')` | [`sharded_by(["axis_a", "axis_b"])`][PartitionDimension::sharded_by] |
/// | `PartitionSpec.UNCONSTRAINED` | [`Unconstrained`][PartitionDimension::Unconstrained] |
///
/// [jax-pspec]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec
///
/// When multiple mesh axes are listed (the `Sharded` variant), the dimension is sharded
/// along their product, major to minor. For example, with a 4x2 mesh and
/// `Sharded(["data", "model"])`, a dimension of size 24 is split into `4 * 2 = 8`
/// partitions.
///
/// # Shardy representation
///
/// Each `PartitionDimension` maps to a [`DimensionShardingAttr`][sdy-dim] in Shardy MLIR.
/// The rendering depends on the [`ShardingContext`]:
///
/// | `PartitionDimension` | `ExplicitSharding` | `ShardingConstraint` |
/// |---|---|---|
/// | `Unsharded` | `{}` (closed) | `{?}` (open) |
/// | `Sharded(["x"])` | `{"x"}` (closed) | `{"x"}` (closed) |
/// | `Sharded(["x", "y"])` | `{"x", "y"}` (closed) | `{"x", "y"}` (closed) |
/// | `Unconstrained` | `{?}` (open) | `{?}` (open) |
///
/// [sdy-dim]: https://openxla.org/shardy/sharding_representation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PartitionDimension {
    /// Dimension is replicated / unpartitioned.
    ///
    /// Equivalent to `None` in JAX's `PartitionSpec`. The entire extent of this dimension is
    /// present on every device (with respect to this dimension; other dimensions may still
    /// shard the tensor).
    Unsharded,

    /// Dimension is partitioned by the provided mesh axis names from major to minor.
    ///
    /// Equivalent to a single axis name `'x'` (when `len == 1`) or a tuple of axis names
    /// `('x', 'y')` (when `len > 1`) in JAX's `PartitionSpec`. The total number of
    /// partitions along this dimension equals the product of the referenced axis sizes.
    Sharded(Vec<String>),

    /// Dimension is unconstrained — the Shardy propagator is free to decide how to shard it.
    ///
    /// Equivalent to `PartitionSpec.UNCONSTRAINED` in JAX. This is only meaningful in
    /// constraint annotations; it cannot be used in a [`ShardingLayout`] because concrete
    /// shard slices cannot be computed without knowing the partitioning.
    Unconstrained,
}

impl PartitionDimension {
    /// Creates an unsharded partition dimension.
    ///
    /// Equivalent to `None` in JAX's `PartitionSpec`.
    pub fn unsharded() -> Self {
        Self::Unsharded
    }

    /// Creates a partitioned dimension using exactly one mesh axis.
    ///
    /// Equivalent to `'axis_name'` in JAX's `PartitionSpec`.
    pub fn sharded<N: Into<String>>(axis_name: N) -> Self {
        Self::Sharded(vec![axis_name.into()])
    }

    /// Creates a partitioned dimension using multiple mesh axes (major to minor).
    ///
    /// Equivalent to `('axis_a', 'axis_b')` in JAX's `PartitionSpec`. The dimension is split
    /// along the product of the referenced axis sizes.
    pub fn sharded_by<I, N>(axis_names: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<String>,
    {
        Self::Sharded(axis_names.into_iter().map(Into::into).collect())
    }

    /// Creates an unconstrained partition dimension.
    ///
    /// Equivalent to `PartitionSpec.UNCONSTRAINED` in JAX.
    pub fn unconstrained() -> Self {
        Self::Unconstrained
    }

    /// Returns mesh axes used for partitioning this dimension, if it is sharded.
    pub fn mesh_axes(&self) -> Option<&[String]> {
        match self {
            Self::Sharded(axis_names) => Some(axis_names.as_slice()),
            _ => None,
        }
    }
}

/// Partition specification for all logical array dimensions.
///
/// A sequence of per-dimension [`PartitionDimension`] entries describing how a tensor is
/// distributed across a mesh.
///
/// # JAX equivalent
///
/// This mirrors [`jax.sharding.PartitionSpec`][jax-pspec] (commonly aliased as `P`):
///
/// | JAX | Ryft |
/// |---|---|
/// | `P('data', None)` | `new(vec![sharded("data"), unsharded()])` |
/// | `P('data', 'model')` | `new(vec![sharded("data"), sharded("model")])` |
/// | `P(('data', 'model'),)` | `new(vec![sharded_by(["data", "model"])])` |
/// | `P()` (replicated, rank 2) | `replicated(2)` |
/// | `P(UNCONSTRAINED)` | `new(vec![unconstrained()])` |
///
/// [jax-pspec]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec
///
/// Each mesh axis may appear **at most once** across all dimensions. This constraint is
/// validated when creating a [`NamedSharding`], not at `PartitionSpec` construction
/// time, because validation requires knowing the mesh.
///
/// # Shardy representation
///
/// Rendered as a bracket-enclosed list of dimension shardings via
/// [`to_shardy_dimension_shardings_literal`][PartitionSpec::to_shardy_dimension_shardings_literal]:
///
/// ```text
/// [{"data"}, {}]         <- P('data', None) with ExplicitSharding
/// [{"data"}, {?}]        <- P('data', None) with ShardingConstraint
/// [{"data"}, {"model"}]  <- P('data', 'model')
/// [{}, {}]               <- P() with rank 2, ExplicitSharding
/// [{?}]                  <- P(UNCONSTRAINED)
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PartitionSpec {
    dimensions: Vec<PartitionDimension>,
}

impl PartitionSpec {
    /// Creates a partition specification from per-dimension assignments.
    pub fn new(dimensions: Vec<PartitionDimension>) -> Self {
        Self { dimensions }
    }

    /// Creates a fully replicated specification for an array with rank `rank`.
    ///
    /// All dimensions are [`Unsharded`][PartitionDimension::Unsharded], meaning the full
    /// tensor is present on every device. Equivalent to `PartitionSpec()` in JAX (padded
    /// to the tensor rank with `None`).
    pub fn replicated(rank: usize) -> Self {
        Self { dimensions: vec![PartitionDimension::Unsharded; rank] }
    }

    /// Returns per-dimension partition assignments.
    pub fn dimensions(&self) -> &[PartitionDimension] {
        self.dimensions.as_slice()
    }

    /// Rank represented by this partition specification.
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Renders this partition specification as a Shardy dimension list literal.
    ///
    /// The output is a bracket-enclosed, comma-separated list of dimension shardings suitable
    /// for embedding in a `#sdy.sharding<...>` attribute.
    ///
    /// The `context` parameter controls whether unsharded dimensions are rendered as closed
    /// (`{}`) or open (`{?}`). See [`ShardingContext`] for details.
    pub fn to_shardy_dimension_shardings_literal(&self, context: ShardingContext) -> String {
        let mut literal = String::from("[");
        for (dimension_index, dimension) in self.dimensions.iter().enumerate() {
            if dimension_index > 0 {
                literal.push_str(", ");
            }
            match dimension {
                PartitionDimension::Unsharded => match context {
                    ShardingContext::ExplicitSharding => literal.push_str("{}"),
                    ShardingContext::ShardingConstraint => literal.push_str("{?}"),
                },
                PartitionDimension::Sharded(axis_names) => {
                    literal.push('{');
                    for (axis_index, axis_name) in axis_names.iter().enumerate() {
                        if axis_index > 0 {
                            literal.push_str(", ");
                        }
                        literal.push('"');
                        literal.push_str(escape_shardy_string(axis_name).as_str());
                        literal.push('"');
                    }
                    literal.push('}');
                }
                PartitionDimension::Unconstrained => literal.push_str("{?}"),
            }
        }
        literal.push(']');
        literal
    }
}

// ---------------------------------------------------------------------------
// Named sharding
// ---------------------------------------------------------------------------

/// Named sharding defined by an [`AbstractMesh`] and a [`PartitionSpec`].
///
/// This is the primary user-facing sharding type for compilation-time annotations,
/// fully describing how a tensor is distributed across a mesh topology. It uses
/// [`AbstractMesh`] rather than [`Mesh`] because device identity is not needed for
/// generating Shardy MLIR attributes.
///
/// # JAX equivalent
///
/// Corresponds to [`jax.sharding.NamedSharding(mesh, spec)`][jax-ns]:
///
/// ```ignore
/// // JAX:   NamedSharding(mesh, PartitionSpec('data', None))
/// // Ryft:
/// let sharding = NamedSharding::new(abstract_mesh, PartitionSpec::new(vec![
///     PartitionDimension::sharded("data"),
///     PartitionDimension::unsharded(),
/// ]))?;
/// ```
///
/// [jax-ns]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.NamedSharding
///
/// # Replicated and unreduced axes
///
/// In Shardy's [`TensorShardingAttr`][sdy-tensor], axes not used in any dimension sharding
/// can be explicitly listed as *replicated* or *unreduced*:
///
/// - **Replicated axes** indicate that the tensor is fully replicated along those mesh axes.
/// - **Unreduced axes** indicate a partial reduction: the tensor has been partitioned but the
///   all-reduce has not yet been applied.
///
/// These are rendered as trailing `replicated={"y"}` and `unreduced={"z"}` in the Shardy
/// attribute:
///
/// ```text
/// #sdy.sharding<@mesh, [{"data"}, {}], replicated={"y"}>
/// #sdy.sharding<@mesh, [{"data"}, {}], unreduced={"z"}>
/// ```
///
/// [sdy-tensor]: https://openxla.org/shardy/sharding_representation
///
/// # Validation
///
/// The constructor validates that:
///
/// - Every mesh axis referenced in the partition specification exists in the mesh.
/// - No mesh axis is used more than once across all dimensions of the partition specification.
/// - Replicated and unreduced axes exist in the mesh, are not used in the partition
///   specification, and do not overlap with each other.
///
/// # Shardy representation
///
/// Rendered as a [`TensorShardingAttr`][sdy-tensor] (`#sdy.sharding<...>`) via
/// [`to_shardy_tensor_sharding_attribute`][NamedSharding::to_shardy_tensor_sharding_attribute]:
///
/// ```text
/// #sdy.sharding<@mesh, [{"data"}, {}]>
/// #sdy.sharding<@mesh, [{"data"}, {}], replicated={"y"}>
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedSharding {
    mesh: AbstractMesh,
    partition_spec: PartitionSpec,
    replicated_axes: Vec<String>,
    unreduced_axes: Vec<String>,
}

impl NamedSharding {
    /// Creates a named sharding with no extra replicated or unreduced axes.
    pub fn new(mesh: AbstractMesh, partition_spec: PartitionSpec) -> Result<Self, ShardingError> {
        Self::with_extra_axes(mesh, partition_spec, Vec::new(), Vec::new())
    }

    /// Creates a named sharding with explicit replicated and/or unreduced axes.
    pub fn with_extra_axes(
        mesh: AbstractMesh,
        partition_spec: PartitionSpec,
        replicated_axes: Vec<String>,
        unreduced_axes: Vec<String>,
    ) -> Result<Self, ShardingError> {
        let partition_used = validate_partition_spec(&mesh, &partition_spec)?;

        // Validate replicated/unreduced axes.
        let mut extra_set = HashSet::new();
        for axis_name in replicated_axes.iter().chain(unreduced_axes.iter()) {
            if mesh.axis_index(axis_name).is_none() {
                return Err(ShardingError::UnknownExtraAxis { axis_name: axis_name.clone() });
            }
            if partition_used.contains(axis_name.as_str()) {
                return Err(ShardingError::ExtraAxisConflictsWithPartition { axis_name: axis_name.clone() });
            }
        }
        for axis_name in &replicated_axes {
            extra_set.insert(axis_name.clone());
        }
        for axis_name in &unreduced_axes {
            if extra_set.contains(axis_name) {
                return Err(ShardingError::AxisInBothReplicatedAndUnreduced { axis_name: axis_name.clone() });
            }
        }

        Ok(Self { mesh, partition_spec, replicated_axes, unreduced_axes })
    }

    /// Returns the abstract mesh of this sharding.
    pub fn mesh(&self) -> &AbstractMesh {
        &self.mesh
    }

    /// Returns the partition specification of this sharding.
    pub fn partition_spec(&self) -> &PartitionSpec {
        &self.partition_spec
    }

    /// Returns explicitly replicated axes.
    pub fn replicated_axes(&self) -> &[String] {
        self.replicated_axes.as_slice()
    }

    /// Returns explicitly unreduced axes.
    pub fn unreduced_axes(&self) -> &[String] {
        self.unreduced_axes.as_slice()
    }

    /// Renders this sharding as a Shardy tensor sharding attribute.
    ///
    /// The output is a valid Shardy `#sdy.sharding<...>` attribute that can be attached to
    /// tensor types in a StableHLO program. For example:
    ///
    /// ```text
    /// #sdy.sharding<@mesh, [{"x"}, {}]>
    /// #sdy.sharding<@mesh, [{"x"}, {}], replicated={"y"}>
    /// ```
    ///
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used by the corresponding `sdy.mesh` op (without or with leading `'@'`).
    ///   - `context`: Controls open/closed rendering for unsharded dimensions.
    pub fn to_shardy_tensor_sharding_attribute<S: AsRef<str>>(
        &self,
        mesh_symbol_name: S,
        context: ShardingContext,
    ) -> Result<String, ShardingError> {
        let mesh_symbol_name = normalize_mesh_symbol_name(mesh_symbol_name)?;
        let dim_shardings = self.partition_spec.to_shardy_dimension_shardings_literal(context);
        let mut result = format!("#sdy.sharding<@{mesh_symbol_name}, {dim_shardings}");

        if !self.replicated_axes.is_empty() {
            result.push_str(", replicated={");
            for (i, axis_name) in self.replicated_axes.iter().enumerate() {
                if i > 0 {
                    result.push_str(", ");
                }
                result.push('"');
                result.push_str(escape_shardy_string(axis_name).as_str());
                result.push('"');
            }
            result.push('}');
        }

        if !self.unreduced_axes.is_empty() {
            result.push_str(", unreduced={");
            for (i, axis_name) in self.unreduced_axes.iter().enumerate() {
                if i > 0 {
                    result.push_str(", ");
                }
                result.push('"');
                result.push_str(escape_shardy_string(axis_name).as_str());
                result.push('"');
            }
            result.push('}');
        }

        result.push('>');
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Shard metadata
// ---------------------------------------------------------------------------

/// Half-open slice `[start, end)` for one logical array dimension in a shard.
///
/// Describes which contiguous range of elements along a single dimension a particular shard
/// holds. This is analogous to one element of the index tuple in JAX's `Shard.index`, which
/// uses Python `slice` objects (e.g., `slice(0, 4)`).
///
/// For an unsharded dimension, the slice spans the full extent `[0, dim_size)`. For a sharded
/// dimension, the slice covers the partition assigned to a specific device based on its mesh
/// coordinate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShardSlice {
    start: usize,
    end: usize,
}

impl ShardSlice {
    /// Creates a new shard slice.
    pub fn new(start: usize, end: usize) -> Result<Self, ShardingError> {
        if start > end {
            return Err(ShardingError::InvalidShardSlice { start, end });
        }
        Ok(Self { start, end })
    }

    /// Inclusive start index.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Exclusive end index.
    pub fn end(&self) -> usize {
        self.end
    }

    /// Length of this slice.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` iff this slice is empty.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Metadata for one global shard of a distributed array.
///
/// Each shard corresponds to one device in the mesh and describes the portion of the global
/// array that device holds. This is pure metadata — it does not contain actual buffer data.
///
/// # JAX equivalent
///
/// Analogous to one entry in JAX's [`array.global_shards`][jax-global-shards], which returns
/// a list of `Shard` objects:
///
/// | JAX `Shard` field | `ShardDescriptor` method |
/// |---|---|
/// | `shard.device` | [`device()`][ShardDescriptor::device] |
/// | `shard.index` (tuple of slices) | [`slices()`][ShardDescriptor::slices] |
/// | `shard.data.shape` | [`shape()`][ShardDescriptor::shape] |
/// | `shard.replica_id` | (derivable from mesh coordinate) |
///
/// [jax-global-shards]: https://docs.jax.dev/en/latest/jax.html#jax.Array.global_shards
///
/// Unlike JAX's `Shard.data`, which provides access to the actual tensor data (only on
/// addressable shards), a `ShardDescriptor` never holds buffer data. Buffer data is stored
/// separately in [`AddressableShard`][super::arrays::AddressableShard] for local shards
/// backed by PJRT buffers, and remains inaccessible for remote shards.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShardDescriptor {
    shard_index: usize,
    device: MeshDevice,
    mesh_coordinate: Vec<usize>,
    slices: Vec<ShardSlice>,
    shape: Vec<usize>,
}

impl ShardDescriptor {
    /// Global shard index in row-major mesh order.
    pub fn shard_index(&self) -> usize {
        self.shard_index
    }

    /// Device that owns this shard.
    pub fn device(&self) -> MeshDevice {
        self.device
    }

    /// Row-major mesh coordinate of this shard.
    pub fn mesh_coordinate(&self) -> &[usize] {
        self.mesh_coordinate.as_slice()
    }

    /// Per-dimension logical slices for this shard.
    pub fn slices(&self) -> &[ShardSlice] {
        self.slices.as_slice()
    }

    /// Logical shape of this shard.
    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }
}

// ---------------------------------------------------------------------------
// Sharding layout
// ---------------------------------------------------------------------------

/// Precomputed global shard metadata for a logical array.
///
/// Given a global array shape, a [`Mesh`], and a [`PartitionSpec`], this structure computes
/// the [`ShardDescriptor`] for every device in the mesh. It provides the information needed
/// to:
///
/// - Determine the per-device shard shape and index range.
/// - Identify which shards are local to a given process (host).
/// - Map device IDs to shard indices for buffer lookup.
///
/// # JAX equivalent
///
/// This corresponds to the internal bookkeeping that JAX performs when constructing a
/// `jax.Array`: materializing [`array.global_shards`][jax-global] and computing the
/// [`devices_indices_map(global_shape)`][jax-indices-map] that maps each device to its
/// array slice.
///
/// [jax-global]: https://docs.jax.dev/en/latest/jax.html#jax.Array.global_shards
/// [jax-indices-map]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Sharding.devices_indices_map
///
/// # Addressable vs non-addressable shards
///
/// All shards are computed, including those on remote hosts. Use
/// [`shard_indices_for_process`][ShardingLayout::shard_indices_for_process] to identify
/// which shards are *addressable* from a given process. In JAX terms:
///
/// - `layout.shards()` ~ `array.global_shards`
/// - `layout.shard_indices_for_process(p)` ~ indices of shards in
///   `array.addressable_shards` when running on process `p`
///
/// Only addressable shards can be backed by actual PJRT buffers. Non-addressable shard
/// descriptors are useful for understanding the full distribution and for generating
/// compiler-level sharding annotations.
///
/// # Unconstrained dimensions
///
/// [`PartitionDimension::Unconstrained`] cannot appear in a `ShardingLayout` — construction
/// will return [`ShardingError::UnconstrainedInLayout`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardingLayout {
    global_shape: Vec<usize>,
    mesh: Mesh,
    partition_spec: PartitionSpec,
    shards: Vec<ShardDescriptor>,
    shard_index_by_device: HashMap<DeviceId, usize>,
}

impl ShardingLayout {
    /// Constructs shard metadata for all devices in the mesh.
    ///
    /// Returns [`ShardingError::UnconstrainedInLayout`] if the partition spec contains an
    /// `Unconstrained` dimension.
    pub fn new(global_shape: Vec<usize>, mesh: Mesh, partition_spec: PartitionSpec) -> Result<Self, ShardingError> {
        validate_partition_spec(mesh.abstract_mesh(), &partition_spec)?;

        let partition_rank = partition_spec.rank();
        let array_rank = global_shape.len();
        if partition_rank != array_rank {
            return Err(ShardingError::RankMismatch { partition_rank, array_rank });
        }

        // Reject Unconstrained dimensions.
        for (dimension, partition_dimension) in partition_spec.dimensions().iter().enumerate() {
            if matches!(partition_dimension, PartitionDimension::Unconstrained) {
                return Err(ShardingError::UnconstrainedInLayout { dimension });
            }
        }

        let mut shards = Vec::with_capacity(mesh.device_count());
        let mut shard_index_by_device = HashMap::with_capacity(mesh.device_count());
        for (shard_index, mesh_device) in mesh.devices().iter().copied().enumerate() {
            let mesh_coordinate = mesh
                .coordinate_for_device_index(shard_index)
                .expect("mesh coordinate should exist for valid mesh device index");

            let mut slices = Vec::with_capacity(global_shape.len());
            let mut shape = Vec::with_capacity(global_shape.len());
            for (dimension, dimension_size) in global_shape.iter().copied().enumerate() {
                let slice = match &partition_spec.dimensions()[dimension] {
                    PartitionDimension::Unsharded => ShardSlice::new(0, dimension_size)?,
                    PartitionDimension::Sharded(axis_names) => {
                        let (partition_index, partition_count) = partition_index_for_axes(
                            mesh.abstract_mesh(),
                            mesh_coordinate.as_slice(),
                            axis_names.as_slice(),
                        )?;
                        partition_slice(dimension_size, partition_count, partition_index)?
                    }
                    PartitionDimension::Unconstrained => {
                        // Already rejected above; this is unreachable.
                        unreachable!()
                    }
                };
                shape.push(slice.len());
                slices.push(slice);
            }

            shard_index_by_device.insert(mesh_device.id(), shard_index);
            shards.push(ShardDescriptor { shard_index, device: mesh_device, mesh_coordinate, slices, shape });
        }

        Ok(Self { global_shape, mesh, partition_spec, shards, shard_index_by_device })
    }

    /// Global array shape.
    pub fn global_shape(&self) -> &[usize] {
        self.global_shape.as_slice()
    }

    /// The mesh used to build this layout.
    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    /// The partition spec used to build this layout.
    pub fn partition_spec(&self) -> &PartitionSpec {
        &self.partition_spec
    }

    /// Shard descriptors for all mesh devices.
    pub fn shards(&self) -> &[ShardDescriptor] {
        self.shards.as_slice()
    }

    /// Returns the descriptor for `shard_index`, if it exists.
    pub fn shard(&self, shard_index: usize) -> Option<&ShardDescriptor> {
        self.shards.get(shard_index)
    }

    /// Returns the shard index for `device_id`, if the device is in the mesh.
    pub fn shard_index_for_device(&self, device_id: DeviceId) -> Option<usize> {
        self.shard_index_by_device.get(&device_id).copied()
    }

    /// Returns the shard descriptor for `device_id`, if the device is in the mesh.
    pub fn shard_for_device(&self, device_id: DeviceId) -> Option<&ShardDescriptor> {
        self.shard_index_for_device(device_id).and_then(|index| self.shard(index))
    }

    /// Returns shard indices that belong to `process_index`.
    ///
    /// These are the shards that are *addressable* (backed by local PJRT buffers) when
    /// executing on the host identified by `process_index`. This corresponds to filtering
    /// JAX's `array.global_shards` down to `array.addressable_shards`.
    pub fn shard_indices_for_process(&self, process_index: usize) -> Vec<usize> {
        self.shards
            .iter()
            .filter_map(|descriptor| {
                (descriptor.device.process_index() == process_index).then_some(descriptor.shard_index())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn normalize_mesh_symbol_name<S: AsRef<str>>(mesh_symbol_name: S) -> Result<String, ShardingError> {
    let mesh_symbol_name = mesh_symbol_name.as_ref().trim();
    if mesh_symbol_name.is_empty() {
        return Err(ShardingError::EmptyMeshSymbolName);
    }

    let mesh_symbol_name = mesh_symbol_name.strip_prefix('@').unwrap_or(mesh_symbol_name);
    if mesh_symbol_name.is_empty() || mesh_symbol_name.chars().any(char::is_whitespace) {
        return Err(ShardingError::InvalidMeshSymbolName { mesh_symbol_name: mesh_symbol_name.to_string() });
    }

    Ok(mesh_symbol_name.to_string())
}

fn escape_shardy_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Validates a partition spec against an abstract mesh. Returns the set of axis names used.
fn validate_partition_spec(
    mesh: &AbstractMesh,
    partition_spec: &PartitionSpec,
) -> Result<HashSet<String>, ShardingError> {
    let mut used_axes = HashSet::new();
    for (dimension, partition_dimension) in partition_spec.dimensions().iter().enumerate() {
        if let PartitionDimension::Sharded(axis_names) = partition_dimension {
            if axis_names.is_empty() {
                return Err(ShardingError::EmptyPartitionAxisList { dimension });
            }

            let mut axes_in_dimension = HashSet::new();
            for axis_name in axis_names {
                if mesh.axis_index(axis_name).is_none() {
                    return Err(ShardingError::UnknownMeshAxis { axis_name: axis_name.clone() });
                }
                if !axes_in_dimension.insert(axis_name.clone()) || !used_axes.insert(axis_name.clone()) {
                    return Err(ShardingError::DuplicatePartitionAxis { axis_name: axis_name.clone() });
                }
            }
        }
    }
    Ok(used_axes)
}

fn coordinate_for_linear_index(mut index: usize, axis_sizes: &[usize]) -> Vec<usize> {
    if axis_sizes.is_empty() {
        return Vec::new();
    }

    let mut coordinate = vec![0usize; axis_sizes.len()];
    for axis in (0..axis_sizes.len()).rev() {
        let axis_size = axis_sizes[axis];
        coordinate[axis] = index % axis_size;
        index /= axis_size;
    }
    coordinate
}

fn partition_index_for_axes(
    mesh: &AbstractMesh,
    mesh_coordinate: &[usize],
    axis_names: &[String],
) -> Result<(usize, usize), ShardingError> {
    let mut partition_index = 0usize;
    let mut partition_count = 1usize;

    for axis_name in axis_names {
        let axis_index = mesh
            .axis_index(axis_name)
            .ok_or_else(|| ShardingError::UnknownMeshAxis { axis_name: axis_name.clone() })?;
        let axis_size = mesh.axes()[axis_index].size();
        let axis_coordinate = mesh_coordinate[axis_index];

        partition_index = partition_index
            .checked_mul(axis_size)
            .and_then(|value| value.checked_add(axis_coordinate))
            .ok_or_else(|| ShardingError::Overflow {
            context: format!("computing partition index for axis '{axis_name}'"),
        })?;
        partition_count = partition_count.checked_mul(axis_size).ok_or_else(|| ShardingError::Overflow {
            context: format!("computing partition count for axis '{axis_name}'"),
        })?;
    }

    Ok((partition_index, partition_count))
}

fn partition_slice(
    dimension_size: usize,
    partition_count: usize,
    partition_index: usize,
) -> Result<ShardSlice, ShardingError> {
    if partition_count == 0 {
        return Err(ShardingError::InvalidPartitionCount);
    }
    if partition_index >= partition_count {
        return Err(ShardingError::InvalidPartitionIndex { partition_index, partition_count });
    }

    let base_size = dimension_size / partition_count;
    let remainder = dimension_size % partition_count;
    let extra_before = partition_index.min(remainder);

    let start = partition_index
        .checked_mul(base_size)
        .and_then(|value| value.checked_add(extra_before))
        .ok_or_else(|| ShardingError::Overflow { context: "computing shard-slice start index".to_string() })?;
    let size = base_size + usize::from(partition_index < remainder);
    let end = start
        .checked_add(size)
        .ok_or_else(|| ShardingError::Overflow { context: "computing shard-slice end index".to_string() })?;

    ShardSlice::new(start, end)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_abstract_mesh_2x2() -> AbstractMesh {
        AbstractMesh::new(vec![MeshAxis::new("x", 2).unwrap(), MeshAxis::new("y", 2).unwrap()]).unwrap()
    }

    fn test_mesh_2x2() -> Mesh {
        let axes = vec![MeshAxis::new("x", 2).unwrap(), MeshAxis::new("y", 2).unwrap()];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        Mesh::new(axes, devices).unwrap()
    }

    // -----------------------------------------------------------------------
    // AbstractMesh tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_abstract_mesh_construction_and_lookups() {
        let mesh = test_abstract_mesh_2x2();
        assert_eq!(mesh.axes().len(), 2);
        assert_eq!(mesh.axis_index("x"), Some(0));
        assert_eq!(mesh.axis_index("y"), Some(1));
        assert_eq!(mesh.axis_index("z"), None);
        assert_eq!(mesh.axis_size("x"), Some(2));
        assert_eq!(mesh.axis_size("y"), Some(2));
        assert_eq!(mesh.device_count().unwrap(), 4);
    }

    #[test]
    fn test_abstract_mesh_validation() {
        let axes = vec![MeshAxis::new("x", 2).unwrap(), MeshAxis::new("x", 3).unwrap()];
        assert!(matches!(
            AbstractMesh::new(axes),
            Err(ShardingError::DuplicateMeshAxisName { axis_name }) if axis_name == "x",
        ));
    }

    #[test]
    fn test_abstract_mesh_shardy_rendering() {
        let mesh = test_abstract_mesh_2x2();
        assert_eq!(mesh.to_shardy_mesh_literal(), "<[\"x\"=2, \"y\"=2]>");
        assert_eq!(mesh.to_shardy_mesh_operation("mesh").unwrap(), "sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>");
        assert_eq!(mesh.to_shardy_mesh_operation("@mesh").unwrap(), "sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>");
        assert!(matches!(mesh.to_shardy_mesh_operation(" "), Err(ShardingError::EmptyMeshSymbolName)));
        assert!(matches!(mesh.to_shardy_mesh_operation("my mesh"), Err(ShardingError::InvalidMeshSymbolName { .. })));
    }

    // -----------------------------------------------------------------------
    // Mesh tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mesh_from_abstract() {
        let abstract_mesh = test_abstract_mesh_2x2();
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        let mesh = Mesh::from_abstract(abstract_mesh.clone(), devices).unwrap();
        assert_eq!(mesh.abstract_mesh(), &abstract_mesh);
        assert_eq!(mesh.axes(), abstract_mesh.axes());
        assert_eq!(mesh.device_count(), 4);
    }

    #[test]
    fn test_mesh_coordinate_mapping() {
        let mesh = test_mesh_2x2();
        assert_eq!(mesh.coordinate_for_device(0), Some(vec![0, 0]));
        assert_eq!(mesh.coordinate_for_device(1), Some(vec![0, 1]));
        assert_eq!(mesh.coordinate_for_device(2), Some(vec![1, 0]));
        assert_eq!(mesh.coordinate_for_device(3), Some(vec![1, 1]));
        assert_eq!(mesh.coordinate_for_device(99), None);
    }

    #[test]
    fn test_mesh_validation() {
        assert!(matches!(MeshAxis::new("", 4), Err(ShardingError::EmptyMeshAxisName)));
        assert!(matches!(
            MeshAxis::new("x", 0),
            Err(ShardingError::InvalidMeshAxisSize { axis_name }) if axis_name == "x",
        ));

        let axes = vec![MeshAxis::new("x", 2).unwrap(), MeshAxis::new("x", 2).unwrap()];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 0), MeshDevice::new(3, 0)];
        assert!(matches!(
            Mesh::new(axes, devices),
            Err(ShardingError::DuplicateMeshAxisName { axis_name }) if axis_name == "x",
        ));

        let axes = vec![MeshAxis::new("x", 2).unwrap()];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(0, 0)];
        assert!(matches!(
            Mesh::new(axes, devices),
            Err(ShardingError::DuplicateMeshDeviceId { device_id }) if device_id == 0,
        ));
    }

    // -----------------------------------------------------------------------
    // PartitionDimension / PartitionSpec tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_partition_dimension_unconstrained() {
        let dim = PartitionDimension::unconstrained();
        assert!(matches!(dim, PartitionDimension::Unconstrained));
        assert_eq!(dim.mesh_axes(), None);
    }

    #[test]
    fn test_partition_spec_shardy_rendering_explicit() {
        let spec = PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        assert_eq!(spec.to_shardy_dimension_shardings_literal(ShardingContext::ExplicitSharding), "[{\"x\"}, {}]");
    }

    #[test]
    fn test_partition_spec_shardy_rendering_constraint() {
        let spec = PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        assert_eq!(spec.to_shardy_dimension_shardings_literal(ShardingContext::ShardingConstraint), "[{\"x\"}, {?}]");
    }

    #[test]
    fn test_partition_spec_shardy_rendering_unconstrained() {
        let spec = PartitionSpec::new(vec![PartitionDimension::unconstrained()]);
        // Unconstrained is always open, regardless of context.
        assert_eq!(spec.to_shardy_dimension_shardings_literal(ShardingContext::ExplicitSharding), "[{?}]");
        assert_eq!(spec.to_shardy_dimension_shardings_literal(ShardingContext::ShardingConstraint), "[{?}]");
    }

    // -----------------------------------------------------------------------
    // NamedSharding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_named_sharding_validation() {
        let mesh = test_abstract_mesh_2x2();

        let unknown_axis_partition = PartitionSpec::new(vec![PartitionDimension::sharded("z")]);
        assert!(matches!(
            NamedSharding::new(mesh.clone(), unknown_axis_partition),
            Err(ShardingError::UnknownMeshAxis { axis_name }) if axis_name == "z",
        ));

        let duplicate_axis_partition =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("x")]);
        assert!(matches!(
            NamedSharding::new(mesh.clone(), duplicate_axis_partition),
            Err(ShardingError::DuplicatePartitionAxis { axis_name }) if axis_name == "x",
        ));

        let empty_axis_partition = PartitionSpec::new(vec![PartitionDimension::Sharded(Vec::new())]);
        assert!(matches!(
            NamedSharding::new(mesh, empty_axis_partition),
            Err(ShardingError::EmptyPartitionAxisList { dimension }) if dimension == 0,
        ));
    }

    #[test]
    fn test_named_sharding_shardy_rendering() {
        let mesh = test_abstract_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let sharding = NamedSharding::new(mesh, partition_spec).unwrap();
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute("mesh", ShardingContext::ExplicitSharding).unwrap(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}]>"
        );
    }

    #[test]
    fn test_named_sharding_replicated_axes() {
        let mesh = test_abstract_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let sharding = NamedSharding::with_extra_axes(mesh, partition_spec, vec!["y".to_string()], Vec::new()).unwrap();
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute("mesh", ShardingContext::ExplicitSharding).unwrap(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], replicated={\"y\"}>"
        );
    }

    #[test]
    fn test_named_sharding_unreduced_axes() {
        let mesh = test_abstract_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let sharding = NamedSharding::with_extra_axes(mesh, partition_spec, Vec::new(), vec!["y".to_string()]).unwrap();
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute("mesh", ShardingContext::ExplicitSharding).unwrap(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced={\"y\"}>"
        );
    }

    #[test]
    fn test_named_sharding_replicated_and_unreduced_axes() {
        let mesh = AbstractMesh::new(vec![
            MeshAxis::new("x", 2).unwrap(),
            MeshAxis::new("y", 2).unwrap(),
            MeshAxis::new("z", 2).unwrap(),
        ])
        .unwrap();
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);
        let sharding =
            NamedSharding::with_extra_axes(mesh, partition_spec, vec!["y".to_string()], vec!["z".to_string()]).unwrap();
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute("mesh", ShardingContext::ExplicitSharding).unwrap(),
            "#sdy.sharding<@mesh, [{\"x\"}], replicated={\"y\"}, unreduced={\"z\"}>"
        );
    }

    #[test]
    fn test_named_sharding_extra_axis_validation() {
        let mesh = test_abstract_mesh_2x2();
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);

        // Unknown extra axis.
        assert!(matches!(
            NamedSharding::with_extra_axes(mesh.clone(), partition_spec.clone(), vec!["z".to_string()], Vec::new()),
            Err(ShardingError::UnknownExtraAxis { axis_name }) if axis_name == "z",
        ));

        // Conflicts with partition spec.
        assert!(matches!(
            NamedSharding::with_extra_axes(mesh.clone(), partition_spec.clone(), vec!["x".to_string()], Vec::new()),
            Err(ShardingError::ExtraAxisConflictsWithPartition { axis_name }) if axis_name == "x",
        ));

        // Same axis in both replicated and unreduced.
        assert!(matches!(
            NamedSharding::with_extra_axes(mesh, partition_spec, vec!["y".to_string()], vec!["y".to_string()]),
            Err(ShardingError::AxisInBothReplicatedAndUnreduced { axis_name }) if axis_name == "y",
        ));
    }

    // -----------------------------------------------------------------------
    // ShardingLayout tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sharding_layout_rank_mismatch() {
        let mesh = test_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("y")]);
        assert!(matches!(
            ShardingLayout::new(vec![8usize], mesh, partition_spec),
            Err(ShardingError::RankMismatch { partition_rank: 2, array_rank: 1 }),
        ));
    }

    #[test]
    fn test_sharding_layout_unconstrained_rejected() {
        let mesh = test_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unconstrained()]);
        assert!(matches!(
            ShardingLayout::new(vec![8, 6], mesh, partition_spec),
            Err(ShardingError::UnconstrainedInLayout { dimension: 1 }),
        ));
    }

    #[test]
    fn test_sharding_layout_even_2d_partitioning() {
        let mesh = test_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("y")]);
        let layout = ShardingLayout::new(vec![8, 6], mesh, partition_spec).unwrap();

        let shard0 = layout.shard_for_device(0).unwrap();
        assert_eq!(shard0.shape(), &[4, 3]);
        assert_eq!(shard0.slices()[0], ShardSlice::new(0, 4).unwrap());
        assert_eq!(shard0.slices()[1], ShardSlice::new(0, 3).unwrap());

        let shard3 = layout.shard_for_device(3).unwrap();
        assert_eq!(shard3.shape(), &[4, 3]);
        assert_eq!(shard3.slices()[0], ShardSlice::new(4, 8).unwrap());
        assert_eq!(shard3.slices()[1], ShardSlice::new(3, 6).unwrap());
    }

    #[test]
    fn test_sharding_layout_uneven_partitioning() {
        let axes = vec![MeshAxis::new("x", 2).unwrap()];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0)];
        let mesh = Mesh::new(axes, devices).unwrap();
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);
        let layout = ShardingLayout::new(vec![5], mesh, partition_spec).unwrap();

        let shard0 = layout.shard_for_device(0).unwrap();
        assert_eq!(shard0.shape(), &[3]);
        assert_eq!(shard0.slices()[0], ShardSlice::new(0, 3).unwrap());

        let shard1 = layout.shard_for_device(1).unwrap();
        assert_eq!(shard1.shape(), &[2]);
        assert_eq!(shard1.slices()[0], ShardSlice::new(3, 5).unwrap());
    }

    #[test]
    fn test_sharding_layout_multi_axis_single_dimension_partitioning() {
        let mesh = test_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded_by(["x".to_string(), "y".to_string()])]);
        let layout = ShardingLayout::new(vec![10], mesh, partition_spec).unwrap();

        assert_eq!(layout.shard_for_device(0).unwrap().slices()[0], ShardSlice::new(0, 3).unwrap());
        assert_eq!(layout.shard_for_device(1).unwrap().slices()[0], ShardSlice::new(3, 6).unwrap());
        assert_eq!(layout.shard_for_device(2).unwrap().slices()[0], ShardSlice::new(6, 8).unwrap());
        assert_eq!(layout.shard_for_device(3).unwrap().slices()[0], ShardSlice::new(8, 10).unwrap());
    }

    #[test]
    fn test_sharding_layout_process_filtering() {
        let mesh = test_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("y")]);
        let layout = ShardingLayout::new(vec![8, 6], mesh, partition_spec).unwrap();

        assert_eq!(layout.shard_indices_for_process(0), vec![0, 1]);
        assert_eq!(layout.shard_indices_for_process(1), vec![2, 3]);
        assert_eq!(layout.shard_indices_for_process(42), Vec::<usize>::new());
    }

    #[test]
    fn test_sharding_layout_mesh_and_partition_spec_accessors() {
        let mesh = test_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let layout = ShardingLayout::new(vec![8, 6], mesh.clone(), partition_spec.clone()).unwrap();

        assert_eq!(layout.mesh(), &mesh);
        assert_eq!(layout.partition_spec(), &partition_spec);
    }

    // -----------------------------------------------------------------------
    // AxisType tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_axis_type_default() {
        assert_eq!(AxisType::default(), AxisType::Auto);
    }

    #[test]
    fn test_abstract_mesh_default_axis_types() {
        let mesh = test_abstract_mesh_2x2();
        assert_eq!(mesh.axis_types(), &[AxisType::Auto, AxisType::Auto]);
    }

    #[test]
    fn test_abstract_mesh_with_axis_types() {
        let axes = vec![MeshAxis::new("x", 2).unwrap(), MeshAxis::new("y", 2).unwrap()];
        let types = vec![AxisType::Manual, AxisType::Explicit];
        let mesh = AbstractMesh::with_axis_types(axes, types).unwrap();
        assert_eq!(mesh.axis_types(), &[AxisType::Manual, AxisType::Explicit]);
    }

    #[test]
    fn test_abstract_mesh_axis_type_count_mismatch() {
        let axes = vec![MeshAxis::new("x", 2).unwrap(), MeshAxis::new("y", 2).unwrap()];
        let types = vec![AxisType::Auto];
        assert!(matches!(
            AbstractMesh::with_axis_types(axes, types),
            Err(ShardingError::AxisTypeCountMismatch { expected: 2, actual: 1 }),
        ));
    }

    #[test]
    fn test_abstract_mesh_axis_type_queries() {
        // All auto.
        let mesh = test_abstract_mesh_2x2();
        assert!(mesh.are_all_axes_auto());
        assert!(!mesh.are_all_axes_explicit());
        assert!(!mesh.are_all_axes_manual());
        assert_eq!(mesh.auto_axes(), vec!["x", "y"]);
        assert!(mesh.explicit_axes().is_empty());
        assert!(mesh.manual_axes().is_empty());

        // Mixed types.
        let axes = vec![MeshAxis::new("a", 2).unwrap(), MeshAxis::new("b", 2).unwrap(), MeshAxis::new("c", 2).unwrap()];
        let types = vec![AxisType::Auto, AxisType::Explicit, AxisType::Manual];
        let mesh = AbstractMesh::with_axis_types(axes, types).unwrap();
        assert!(!mesh.are_all_axes_auto());
        assert!(!mesh.are_all_axes_explicit());
        assert!(!mesh.are_all_axes_manual());
        assert_eq!(mesh.auto_axes(), vec!["a"]);
        assert_eq!(mesh.explicit_axes(), vec!["b"]);
        assert_eq!(mesh.manual_axes(), vec!["c"]);
    }

    #[test]
    fn test_abstract_mesh_with_updated_axis_types() {
        let mesh = test_abstract_mesh_2x2();
        let updates = HashMap::from([("y".to_string(), AxisType::Manual)]);
        let updated = mesh.with_updated_axis_types(&updates);
        assert_eq!(updated.axis_types(), &[AxisType::Auto, AxisType::Manual]);

        // Unknown names are silently ignored.
        let updates = HashMap::from([("z".to_string(), AxisType::Explicit)]);
        let updated = mesh.with_updated_axis_types(&updates);
        assert_eq!(updated.axis_types(), &[AxisType::Auto, AxisType::Auto]);
    }

    #[test]
    fn test_abstract_mesh_axis_names_and_sizes() {
        let mesh = test_abstract_mesh_2x2();
        assert_eq!(mesh.axis_names(), vec!["x", "y"]);
        assert_eq!(mesh.axis_sizes(), vec![2, 2]);
    }

    #[test]
    fn test_mesh_with_axis_types() {
        let axes = vec![MeshAxis::new("x", 2).unwrap(), MeshAxis::new("y", 2).unwrap()];
        let types = vec![AxisType::Explicit, AxisType::Manual];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 0), MeshDevice::new(3, 0)];
        let mesh = Mesh::with_axis_types(axes, types, devices).unwrap();
        assert_eq!(mesh.axis_types(), &[AxisType::Explicit, AxisType::Manual]);
    }

    #[test]
    fn test_mesh_axis_type_delegation() {
        let mesh = test_mesh_2x2();
        assert!(mesh.are_all_axes_auto());
        assert!(!mesh.are_all_axes_explicit());
        assert!(!mesh.are_all_axes_manual());
        assert_eq!(mesh.auto_axes(), vec!["x", "y"]);
        assert!(mesh.explicit_axes().is_empty());
        assert!(mesh.manual_axes().is_empty());
        assert_eq!(mesh.axis_names(), vec!["x", "y"]);
        assert_eq!(mesh.axis_sizes(), vec![2, 2]);
    }

    #[test]
    fn test_mesh_is_multi_process() {
        // Single process.
        let axes = vec![MeshAxis::new("x", 2).unwrap()];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0)];
        let mesh = Mesh::new(axes, devices).unwrap();
        assert!(!mesh.is_multi_process());

        // Multi process.
        let mesh = test_mesh_2x2(); // devices 0,1 on process 0; devices 2,3 on process 1.
        assert!(mesh.is_multi_process());
    }

    #[test]
    fn test_mesh_local_devices() {
        let mesh = test_mesh_2x2();
        let local_0: Vec<DeviceId> = mesh.local_devices(0).iter().map(|d| d.id()).collect();
        assert_eq!(local_0, vec![0, 1]);
        let local_1: Vec<DeviceId> = mesh.local_devices(1).iter().map(|d| d.id()).collect();
        assert_eq!(local_1, vec![2, 3]);
        assert!(mesh.local_devices(42).is_empty());
    }

    #[test]
    fn test_mesh_device_ids() {
        let mesh = test_mesh_2x2();
        assert_eq!(mesh.device_ids(), vec![0, 1, 2, 3]);
    }
}
