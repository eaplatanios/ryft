//! Runtime sharded-array data structures for XLA execution.
//!
//! These types intentionally mirror the layering used by JAX and IFRT:
//!
//! - [`Mesh`] corresponds to `jax.sharding.Mesh`: a named logical grid over global devices.
//! - [`PartitionSpecification`] corresponds to `jax.sharding.PartitionSpec`: per-array-dimension axis assignment.
//! - [`NamedSharding`] corresponds to `jax.sharding.NamedSharding`: mesh + partition specification.
//! - [`Array`] corresponds to `jax.Array` / IFRT `Array`: global array metadata plus local addressable device buffers.
//!
//! In JAX JIT execution, user-level shardings are lowered into compiler-level sharding annotations. For StableHLO
//! programs that use Shardy attributes, this module provides conversion helpers from runtime array shardings to
//! textual Shardy attribute representations (`sdy.mesh` and `#sdy.sharding<...>`). This enables a JIT flow where
//! input arrays determine compilation-time sharding attributes.
//!
//! Note that this module focuses on runtime bookkeeping and argument marshalling for PJRT execution. It does not
//! implement the full IFRT API surface.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use ryft_pjrt::{Buffer, BufferType, DeviceId, Error as PjrtError, ExecutionDeviceInputs, ExecutionInput};

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
}

/// Error type for [`Array`] construction and execution-input preparation.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ArrayError {
    /// Underlying error returned by PJRT.
    #[error("{0}")]
    PjrtError(#[from] PjrtError),

    /// Underlying sharding/layout error.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Error returned when an addressable buffer is placed on a device not present in the array mesh.
    #[error("addressable buffer is placed on device {device_id}, but that device is not in the mesh")]
    AddressableBufferDeviceNotInMesh { device_id: DeviceId },

    /// Error returned when more than one addressable buffer is provided for the same device.
    #[error("got multiple addressable buffers for device {device_id}")]
    DuplicateAddressableBufferDevice { device_id: DeviceId },

    /// Error returned when a buffer element type does not match the array element type.
    #[error("buffer on device {device_id} has element type {actual}, but array element type is {expected}")]
    BufferElementTypeMismatch { device_id: DeviceId, expected: BufferType, actual: BufferType },

    /// Error returned when a buffer shape dimension cannot be represented as `usize`.
    #[error("buffer on device {device_id} has shape dimension #{dimension}={size}, which does not fit in usize")]
    BufferShapeDimensionTooLarge { device_id: DeviceId, dimension: usize, size: u64 },

    /// Error returned when a buffer shape does not match the expected shard shape.
    #[error(
        "buffer on device {device_id} has shape {actual_shape:?}, but shard #{shard_index} expects {expected_shape:?}"
    )]
    BufferShapeMismatch {
        device_id: DeviceId,
        shard_index: usize,
        expected_shape: Vec<usize>,
        actual_shape: Vec<usize>,
    },

    /// Error returned when a buffer process index does not match the process index encoded in the mesh.
    #[error(
        "buffer on device {device_id} reports process index {actual_process_index}, but the mesh expects {expected_process_index}"
    )]
    BufferProcessIndexMismatch { device_id: DeviceId, expected_process_index: usize, actual_process_index: usize },

    /// Error returned when the number of donation flags does not match the number of arrays.
    #[error("got {actual_count} donation flag(s), but expected {expected_count}")]
    DonationFlagCountMismatch { expected_count: usize, actual_count: usize },

    /// Error returned when the device list for execution contains duplicate IDs.
    #[error("device {device_id} appears multiple times in the execution device order")]
    DuplicateExecutionDeviceId { device_id: DeviceId },

    /// Error returned when an array does not have an addressable shard for a required device.
    #[error("input array #{array_index} has no addressable shard for device {device_id}")]
    MissingArrayShardForDevice { array_index: usize, device_id: DeviceId },

    /// Error returned when an array has an addressable shard for a device that is not in the execution device order.
    #[error("input array #{array_index} has an unexpected addressable shard for device {device_id}")]
    UnexpectedArrayShardDevice { array_index: usize, device_id: DeviceId },
}

/// A named axis in a logical device mesh.
///
/// Equivalent conceptually to one axis entry of JAX's `Mesh` shape (e.g., axis `"data"` with size `8`).
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
/// This separates global device identity (`id`) from host/process ownership (`process_index`) so the same
/// sharding metadata can represent both local and remote shards.
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

/// Logical mesh of devices used by named shardings.
///
/// This corresponds to JAX's `jax.sharding.Mesh`. Devices are stored in row-major order with respect to `axes`.
/// The row-major index is also used as the global shard index ordering in [`ShardingLayout`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mesh {
    axes: Vec<MeshAxis>,
    devices: Vec<MeshDevice>,
    axis_index_by_name: HashMap<String, usize>,
    device_index_by_id: HashMap<DeviceId, usize>,
}

impl Mesh {
    /// Creates a mesh from named axes and row-major devices.
    ///
    /// The expected number of `devices` is the product of all `axes` sizes. For an empty axis list, the
    /// expected device count is `1`.
    pub fn new(axes: Vec<MeshAxis>, devices: Vec<MeshDevice>) -> Result<Self, ShardingError> {
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

        let expected_device_count = axes.iter().try_fold(1usize, |count, axis| {
            count.checked_mul(axis.size).ok_or_else(|| ShardingError::Overflow {
                context: "computing mesh device count from axis sizes".to_string(),
            })
        })?;
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

        Ok(Self { axes, devices, axis_index_by_name, device_index_by_id })
    }

    /// Returns the axes of this mesh.
    pub fn axes(&self) -> &[MeshAxis] {
        self.axes.as_slice()
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
        self.axis_index_by_name.get(axis_name.as_ref()).copied()
    }

    /// Returns the size of `axis_name` in this mesh, if present.
    pub fn axis_size<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.axis_index(axis_name).map(|axis_index| self.axes[axis_index].size)
    }

    /// Returns the row-major mesh index of `device_id`, if present.
    pub fn device_index(&self, device_id: DeviceId) -> Option<usize> {
        self.device_index_by_id.get(&device_id).copied()
    }

    /// Returns the mesh coordinate of the device at `device_index`, if valid.
    pub fn coordinate_for_device_index(&self, device_index: usize) -> Option<Vec<usize>> {
        (device_index < self.devices.len()).then(|| {
            let axis_sizes = self.axes.iter().map(MeshAxis::size).collect::<Vec<_>>();
            coordinate_for_linear_index(device_index, axis_sizes.as_slice())
        })
    }

    /// Returns the mesh coordinate of `device_id`, if present.
    pub fn coordinate_for_device(&self, device_id: DeviceId) -> Option<Vec<usize>> {
        self.device_index(device_id).and_then(|device_index| self.coordinate_for_device_index(device_index))
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
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used in MLIR (without or with leading `'@'`).
    pub fn to_shardy_mesh_operation<S: AsRef<str>>(&self, mesh_symbol_name: S) -> Result<String, ShardingError> {
        let mesh_symbol_name = normalize_mesh_symbol_name(mesh_symbol_name)?;
        Ok(format!("sdy.mesh @{mesh_symbol_name} = {}", self.to_shardy_mesh_literal()))
    }
}

/// Partitioning assignment for one logical array dimension.
///
/// Equivalent to one entry in JAX's `PartitionSpec`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PartitionDimension {
    /// Dimension is replicated / unpartitioned.
    Unsharded,

    /// Dimension is partitioned by the provided mesh axis names from major to minor.
    MeshAxes(Vec<String>),
}

impl PartitionDimension {
    /// Creates an unsharded partition dimension.
    pub fn unsharded() -> Self {
        Self::Unsharded
    }

    /// Creates a partitioned dimension using exactly one mesh axis.
    pub fn sharded<N: Into<String>>(axis_name: N) -> Self {
        Self::MeshAxes(vec![axis_name.into()])
    }

    /// Creates a partitioned dimension using multiple mesh axes (major to minor).
    pub fn sharded_by<I, N>(axis_names: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<String>,
    {
        Self::MeshAxes(axis_names.into_iter().map(Into::into).collect())
    }

    /// Returns mesh axes used for partitioning this dimension, if it is partitioned.
    pub fn mesh_axes(&self) -> Option<&[String]> {
        match self {
            Self::Unsharded => None,
            Self::MeshAxes(axis_names) => Some(axis_names.as_slice()),
        }
    }
}

/// Partition specification for all logical array dimensions.
///
/// This mirrors JAX's `PartitionSpec` semantics at the level needed to compute shard indices and shapes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PartitionSpecification {
    dimensions: Vec<PartitionDimension>,
}

impl PartitionSpecification {
    /// Creates a partition specification from per-dimension assignments.
    pub fn new(dimensions: Vec<PartitionDimension>) -> Self {
        Self { dimensions }
    }

    /// Creates a fully replicated specification for an array with rank `rank`.
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
    /// For example, a 2D partition spec `("x", None)` is rendered as:
    /// `[{"x"}, {}]`.
    pub fn to_shardy_dimension_shardings_literal(&self) -> String {
        let mut literal = String::from("[");
        for (dimension_index, dimension) in self.dimensions.iter().enumerate() {
            if dimension_index > 0 {
                literal.push_str(", ");
            }
            match dimension {
                PartitionDimension::Unsharded => literal.push_str("{}"),
                PartitionDimension::MeshAxes(axis_names) => {
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
            }
        }
        literal.push(']');
        literal
    }
}

/// Named sharding defined by a [`Mesh`] and [`PartitionSpecification`].
///
/// This corresponds to JAX's `NamedSharding(mesh, spec)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedSharding {
    mesh: Mesh,
    partition_spec: PartitionSpecification,
}

impl NamedSharding {
    /// Creates a named sharding.
    pub fn new(mesh: Mesh, partition_spec: PartitionSpecification) -> Result<Self, ShardingError> {
        validate_partition_spec(&mesh, &partition_spec)?;
        Ok(Self { mesh, partition_spec })
    }

    /// Returns the mesh of this sharding.
    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    /// Returns the partition specification of this sharding.
    pub fn partition_spec(&self) -> &PartitionSpecification {
        &self.partition_spec
    }

    /// Renders this sharding as a Shardy tensor sharding attribute.
    ///
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used by the corresponding `sdy.mesh` op (without or with leading `'@'`).
    ///
    /// Example output:
    /// `#sdy.sharding<@mesh, [{"x"}, {}]>`.
    pub fn to_shardy_tensor_sharding_attribute<S: AsRef<str>>(
        &self,
        mesh_symbol_name: S,
    ) -> Result<String, ShardingError> {
        let mesh_symbol_name = normalize_mesh_symbol_name(mesh_symbol_name)?;
        Ok(format!(
            "#sdy.sharding<@{mesh_symbol_name}, {}>",
            self.partition_spec.to_shardy_dimension_shardings_literal(),
        ))
    }
}

/// Half-open slice `[start, end)` for one logical array dimension in a shard.
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShardDescriptor {
    shard_index: usize,
    device: MeshDevice,
    mesh_coordinate: Vec<usize>,
    slices: Vec<ShardSlice>,
    shape: Vec<usize>,
    element_type: BufferType,
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

    /// Element type of this shard.
    pub fn element_type(&self) -> BufferType {
        self.element_type
    }
}

/// Precomputed global shard metadata for a logical array.
///
/// This contains all shard descriptors for the full mesh, including non-addressable shards on remote hosts.
/// In JAX terms, this is similar to materializing the per-device shard map that is usually inspected through
/// `array.global_shards` and `array.addressable_shards`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardingLayout {
    global_shape: Vec<usize>,
    element_type: BufferType,
    sharding: NamedSharding,
    shards: Vec<ShardDescriptor>,
    shard_index_by_device: HashMap<DeviceId, usize>,
}

impl ShardingLayout {
    /// Constructs shard metadata for all devices in the sharding mesh.
    pub fn new(
        global_shape: Vec<usize>,
        element_type: BufferType,
        sharding: NamedSharding,
    ) -> Result<Self, ShardingError> {
        let partition_rank = sharding.partition_spec().rank();
        let array_rank = global_shape.len();
        if partition_rank != array_rank {
            return Err(ShardingError::RankMismatch { partition_rank, array_rank });
        }

        let mut shards = Vec::with_capacity(sharding.mesh().device_count());
        let mut shard_index_by_device = HashMap::with_capacity(sharding.mesh().device_count());
        for (shard_index, mesh_device) in sharding.mesh().devices().iter().copied().enumerate() {
            let mesh_coordinate = sharding
                .mesh()
                .coordinate_for_device_index(shard_index)
                .expect("mesh coordinate should exist for valid mesh device index");

            let mut slices = Vec::with_capacity(global_shape.len());
            let mut shape = Vec::with_capacity(global_shape.len());
            for (dimension, dimension_size) in global_shape.iter().copied().enumerate() {
                let slice = match &sharding.partition_spec().dimensions()[dimension] {
                    PartitionDimension::Unsharded => ShardSlice::new(0, dimension_size)?,
                    PartitionDimension::MeshAxes(axis_names) => {
                        let (partition_index, partition_count) = partition_index_for_axes(
                            sharding.mesh(),
                            mesh_coordinate.as_slice(),
                            axis_names.as_slice(),
                        )?;
                        partition_slice(dimension_size, partition_count, partition_index)?
                    }
                };
                shape.push(slice.len());
                slices.push(slice);
            }

            shard_index_by_device.insert(mesh_device.id(), shard_index);
            shards.push(ShardDescriptor {
                shard_index,
                device: mesh_device,
                mesh_coordinate,
                slices,
                shape,
                element_type,
            });
        }

        Ok(Self { global_shape, element_type, sharding, shards, shard_index_by_device })
    }

    /// Global array shape.
    pub fn global_shape(&self) -> &[usize] {
        self.global_shape.as_slice()
    }

    /// Global array element type.
    pub fn element_type(&self) -> BufferType {
        self.element_type
    }

    /// Sharding metadata used to build this layout.
    pub fn sharding(&self) -> &NamedSharding {
        &self.sharding
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
    pub fn shard_indices_for_process(&self, process_index: usize) -> Vec<usize> {
        self.shards
            .iter()
            .filter_map(|descriptor| {
                (descriptor.device.process_index() == process_index).then_some(descriptor.shard_index())
            })
            .collect()
    }
}

/// Addressable shard on the current host.
///
/// Each entry ties one local [`Buffer`] to one global shard index.
/// This corresponds to one entry in JAX's `array.addressable_shards`.
pub struct AddressableShard<'o> {
    shard_index: usize,
    device_id: DeviceId,
    process_index: usize,
    buffer: Buffer<'o>,
}

impl<'o> AddressableShard<'o> {
    /// Global shard index for this buffer.
    pub fn shard_index(&self) -> usize {
        self.shard_index
    }

    /// Device ID on which this buffer is placed.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Process index owning the device on which this buffer is placed.
    pub fn process_index(&self) -> usize {
        self.process_index
    }

    /// Addressable shard buffer.
    pub fn buffer(&self) -> &Buffer<'o> {
        &self.buffer
    }
}

impl std::fmt::Debug for AddressableShard<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AddressableShard")
            .field("shard_index", &self.shard_index)
            .field("device_id", &self.device_id)
            .field("process_index", &self.process_index)
            .finish()
    }
}

/// Distributed array backed by local addressable PJRT buffers and global sharding metadata.
///
/// This is conceptually aligned with JAX/IFRT arrays:
/// - `layout` describes all global shards across the full mesh.
/// - `addressable_shards` contains only shards local to the current host process.
/// - each addressable buffer is mapped to its global shard index.
///
/// In JAX terminology, this is the runtime pairing of:
/// - sharding metadata (`NamedSharding`), and
/// - addressable device buffers (the local portion of an IFRT array).
pub struct Array<'o> {
    layout: ShardingLayout,
    addressable_shards: Vec<AddressableShard<'o>>,
    addressable_shard_index_by_device: HashMap<DeviceId, usize>,
}

impl<'o> Array<'o> {
    /// Creates an [`Array`] from precomputed sharding metadata and local addressable buffers.
    ///
    /// Each buffer is mapped to a shard using its device ID. Buffer shape and element type are validated against
    /// shard metadata.
    pub fn new(layout: ShardingLayout, addressable_buffers: Vec<Buffer<'o>>) -> Result<Self, ArrayError> {
        let mut seen_devices = HashSet::with_capacity(addressable_buffers.len());
        let mut addressable_shards = Vec::with_capacity(addressable_buffers.len());

        for buffer in addressable_buffers {
            let device = buffer.device()?;
            let device_id = device.id()?;
            if !seen_devices.insert(device_id) {
                return Err(ArrayError::DuplicateAddressableBufferDevice { device_id });
            }

            let shard_index = layout
                .shard_index_for_device(device_id)
                .ok_or(ArrayError::AddressableBufferDeviceNotInMesh { device_id })?;
            let shard = layout
                .shard(shard_index)
                .expect("layout shard index should exist for valid layout device-to-shard mapping");

            let process_index = device.process_index()?;
            if process_index != shard.device().process_index() {
                return Err(ArrayError::BufferProcessIndexMismatch {
                    device_id,
                    expected_process_index: shard.device().process_index(),
                    actual_process_index: process_index,
                });
            }

            let actual_element_type = buffer.element_type()?;
            if actual_element_type != layout.element_type() {
                return Err(ArrayError::BufferElementTypeMismatch {
                    device_id,
                    expected: layout.element_type(),
                    actual: actual_element_type,
                });
            }

            let actual_shape = buffer
                .dimensions()?
                .iter()
                .enumerate()
                .map(|(dimension, size)| {
                    usize::try_from(*size).map_err(|_| ArrayError::BufferShapeDimensionTooLarge {
                        device_id,
                        dimension,
                        size: *size,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if actual_shape != shard.shape() {
                return Err(ArrayError::BufferShapeMismatch {
                    device_id,
                    shard_index,
                    expected_shape: shard.shape().to_vec(),
                    actual_shape,
                });
            }

            addressable_shards.push(AddressableShard { shard_index, device_id, process_index, buffer });
        }

        addressable_shards.sort_by_key(AddressableShard::shard_index);
        let addressable_shard_index_by_device = addressable_shards
            .iter()
            .enumerate()
            .map(|(addressable_shard_index, shard)| (shard.device_id(), addressable_shard_index))
            .collect::<HashMap<_, _>>();

        Ok(Self { layout, addressable_shards, addressable_shard_index_by_device })
    }

    /// Creates an [`Array`] from shape/type/sharding metadata and local addressable buffers.
    pub fn from_sharding(
        global_shape: Vec<usize>,
        element_type: BufferType,
        sharding: NamedSharding,
        addressable_buffers: Vec<Buffer<'o>>,
    ) -> Result<Self, ArrayError> {
        let layout = ShardingLayout::new(global_shape, element_type, sharding)?;
        Self::new(layout, addressable_buffers)
    }

    /// Returns global sharding layout metadata.
    pub fn layout(&self) -> &ShardingLayout {
        &self.layout
    }

    /// Returns this array's named sharding.
    pub fn named_sharding(&self) -> &NamedSharding {
        self.layout.sharding()
    }

    /// Returns the global array shape.
    pub fn global_shape(&self) -> &[usize] {
        self.layout.global_shape()
    }

    /// Returns the global array element type.
    pub fn element_type(&self) -> BufferType {
        self.layout.element_type()
    }

    /// Returns metadata for all global shards.
    pub fn shards(&self) -> &[ShardDescriptor] {
        self.layout.shards()
    }

    /// Returns addressable local shards.
    pub fn addressable_shards(&self) -> &[AddressableShard<'o>] {
        self.addressable_shards.as_slice()
    }

    /// Returns the addressable shard for `device_id`, if local.
    pub fn addressable_shard_for_device(&self, device_id: DeviceId) -> Option<&AddressableShard<'o>> {
        self.addressable_shard_index_by_device
            .get(&device_id)
            .and_then(|index| self.addressable_shards.get(*index))
    }

    /// Returns global shard metadata for `device_id`, if it exists in the mesh.
    pub fn shard_for_device(&self, device_id: DeviceId) -> Option<&ShardDescriptor> {
        self.layout.shard_for_device(device_id)
    }

    /// Returns global shard metadata for a local addressable shard index.
    pub fn shard_for_addressable_index(&self, addressable_shard_index: usize) -> Option<&ShardDescriptor> {
        self.addressable_shards
            .get(addressable_shard_index)
            .and_then(|addressable_shard| self.layout.shard(addressable_shard.shard_index()))
    }

    /// Renders the Shardy mesh declaration (`sdy.mesh`) implied by this array's sharding.
    ///
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used in MLIR (without or with leading `'@'`).
    pub fn to_shardy_mesh_operation<S: AsRef<str>>(&self, mesh_symbol_name: S) -> Result<String, ShardingError> {
        self.named_sharding().mesh().to_shardy_mesh_operation(mesh_symbol_name)
    }

    /// Renders the Shardy tensor sharding attribute (`#sdy.sharding<...>`) implied by this array.
    ///
    /// This is the key conversion used in JIT flows: runtime input arrays determine compile-time sharding
    /// annotations for StableHLO function parameters/results.
    ///
    /// # Parameters
    ///
    ///   - `mesh_symbol_name`: Symbol name used by the corresponding `sdy.mesh` op (without or with leading `'@'`).
    pub fn to_shardy_tensor_sharding_attribute<S: AsRef<str>>(
        &self,
        mesh_symbol_name: S,
    ) -> Result<String, ShardingError> {
        self.named_sharding().to_shardy_tensor_sharding_attribute(mesh_symbol_name)
    }

    /// Converts distributed arrays to per-device execution arguments for [`ryft_pjrt::LoadedExecutable::execute`].
    ///
    /// Inputs are generated in `addressable_device_ids` order. The resulting [`ExecuteArguments`] can be converted
    /// to `Vec<ExecutionDeviceInputs>` via [`ExecuteArguments::as_execution_device_inputs`].
    pub fn into_execute_arguments(
        arrays: Vec<Self>,
        addressable_device_ids: &[DeviceId],
    ) -> Result<ExecuteArguments<'o>, ArrayError> {
        let donation_flags = vec![false; arrays.len()];
        ExecuteArguments::from_arrays_with_donation(arrays, addressable_device_ids, donation_flags.as_slice())
    }

    /// Same as [`Array::into_execute_arguments`] but with explicit per-input donation flags.
    pub fn into_execute_arguments_with_donation(
        arrays: Vec<Self>,
        addressable_device_ids: &[DeviceId],
        donation_flags: &[bool],
    ) -> Result<ExecuteArguments<'o>, ArrayError> {
        ExecuteArguments::from_arrays_with_donation(arrays, addressable_device_ids, donation_flags)
    }

    fn into_addressable_buffers_by_device(self) -> HashMap<DeviceId, Buffer<'o>> {
        self.addressable_shards
            .into_iter()
            .map(|addressable_shard| (addressable_shard.device_id(), addressable_shard.buffer))
            .collect()
    }
}

impl std::fmt::Debug for Array<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Array")
            .field("global_shape", &self.global_shape())
            .field("element_type", &self.element_type())
            .field("global_shard_count", &self.shards().len())
            .field("addressable_shard_count", &self.addressable_shards.len())
            .finish()
    }
}

/// Prepared execution inputs for calling [`ryft_pjrt::LoadedExecutable::execute`].
///
/// This stores one `Vec<ExecutionInput>` per addressable device (in caller-provided device order).
pub struct ExecuteArguments<'o> {
    addressable_device_ids: Vec<DeviceId>,
    inputs_by_device: Vec<Vec<ExecutionInput<'o>>>,
}

impl<'o> ExecuteArguments<'o> {
    /// Returns addressable device IDs corresponding to [`Self::inputs_by_device`].
    pub fn addressable_device_ids(&self) -> &[DeviceId] {
        self.addressable_device_ids.as_slice()
    }

    /// Returns execution inputs grouped by device.
    pub fn inputs_by_device(&self) -> &[Vec<ExecutionInput<'o>>] {
        self.inputs_by_device.as_slice()
    }

    /// Creates PJRT `ExecutionDeviceInputs` in the same device order as [`Self::addressable_device_ids`].
    pub fn as_execution_device_inputs<'l>(&'l self) -> Vec<ExecutionDeviceInputs<'o, 'l>> {
        self.inputs_by_device.iter().map(|inputs| ExecutionDeviceInputs::from(inputs.as_slice())).collect()
    }

    fn from_arrays_with_donation(
        arrays: Vec<Array<'o>>,
        addressable_device_ids: &[DeviceId],
        donation_flags: &[bool],
    ) -> Result<Self, ArrayError> {
        if donation_flags.len() != arrays.len() {
            return Err(ArrayError::DonationFlagCountMismatch {
                expected_count: arrays.len(),
                actual_count: donation_flags.len(),
            });
        }

        let mut seen_devices = HashSet::with_capacity(addressable_device_ids.len());
        for &device_id in addressable_device_ids {
            if !seen_devices.insert(device_id) {
                return Err(ArrayError::DuplicateExecutionDeviceId { device_id });
            }
        }

        let mut buffers_by_array =
            arrays.into_iter().map(Array::into_addressable_buffers_by_device).collect::<Vec<_>>();

        let mut inputs_by_device = Vec::with_capacity(addressable_device_ids.len());
        for &device_id in addressable_device_ids {
            let mut device_inputs = Vec::with_capacity(buffers_by_array.len());
            for (array_index, array_buffers_by_device) in buffers_by_array.iter_mut().enumerate() {
                let buffer = array_buffers_by_device
                    .remove(&device_id)
                    .ok_or(ArrayError::MissingArrayShardForDevice { array_index, device_id })?;
                device_inputs.push(ExecutionInput { buffer, donatable: donation_flags[array_index] });
            }
            inputs_by_device.push(device_inputs);
        }

        for (array_index, array_buffers_by_device) in buffers_by_array.iter().enumerate() {
            if let Some(device_id) = array_buffers_by_device.keys().next().copied() {
                return Err(ArrayError::UnexpectedArrayShardDevice { array_index, device_id });
            }
        }

        Ok(Self { addressable_device_ids: addressable_device_ids.to_vec(), inputs_by_device })
    }
}

impl std::fmt::Debug for ExecuteArguments<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_counts = self.inputs_by_device.iter().map(Vec::len).collect::<Vec<_>>();
        formatter
            .debug_struct("ExecuteArguments")
            .field("addressable_device_ids", &self.addressable_device_ids)
            .field("input_counts_per_device", &input_counts)
            .finish()
    }
}

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

fn validate_partition_spec(mesh: &Mesh, partition_spec: &PartitionSpecification) -> Result<(), ShardingError> {
    let mut used_axes = HashSet::new();
    for (dimension, partition_dimension) in partition_spec.dimensions().iter().enumerate() {
        if let PartitionDimension::MeshAxes(axis_names) = partition_dimension {
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
    Ok(())
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
    mesh: &Mesh,
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};

    use super::*;

    fn test_spmd_compilation_options(partition_count: usize) -> CompilationOptions {
        CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: partition_count as i64,
                use_spmd_partitioning: true,
                use_shardy_partitioner: true,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        }
    }

    fn f32_values_to_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn two_f32s_from_bytes(bytes: &[u8]) -> [f32; 2] {
        assert_eq!(bytes.len(), 2 * size_of::<f32>());
        let first = f32::from_ne_bytes(bytes[..size_of::<f32>()].try_into().unwrap());
        let second = f32::from_ne_bytes(bytes[size_of::<f32>()..].try_into().unwrap());
        [first, second]
    }

    fn test_mesh_2x2() -> Mesh {
        let axes = vec![MeshAxis::new("x", 2).unwrap(), MeshAxis::new("y", 2).unwrap()];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        Mesh::new(axes, devices).unwrap()
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

    #[test]
    fn test_named_sharding_validation() {
        let mesh = test_mesh_2x2();

        let unknown_axis_partition = PartitionSpecification::new(vec![PartitionDimension::sharded("z")]);
        assert!(matches!(
            NamedSharding::new(mesh.clone(), unknown_axis_partition),
            Err(ShardingError::UnknownMeshAxis { axis_name }) if axis_name == "z",
        ));

        let duplicate_axis_partition =
            PartitionSpecification::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("x")]);
        assert!(matches!(
            NamedSharding::new(mesh.clone(), duplicate_axis_partition),
            Err(ShardingError::DuplicatePartitionAxis { axis_name }) if axis_name == "x",
        ));

        let empty_axis_partition = PartitionSpecification::new(vec![PartitionDimension::MeshAxes(Vec::new())]);
        assert!(matches!(
            NamedSharding::new(mesh, empty_axis_partition),
            Err(ShardingError::EmptyPartitionAxisList { dimension }) if dimension == 0,
        ));
    }

    #[test]
    fn test_shardy_rendering_from_named_sharding() {
        let mesh = test_mesh_2x2();
        let partition_spec =
            PartitionSpecification::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let sharding = NamedSharding::new(mesh.clone(), partition_spec.clone()).unwrap();

        assert_eq!(mesh.to_shardy_mesh_literal(), "<[\"x\"=2, \"y\"=2]>");
        assert_eq!(mesh.to_shardy_mesh_operation("mesh").unwrap(), "sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>");
        assert_eq!(mesh.to_shardy_mesh_operation("@mesh").unwrap(), "sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>");
        assert!(matches!(mesh.to_shardy_mesh_operation(" "), Err(ShardingError::EmptyMeshSymbolName),));
        assert!(matches!(mesh.to_shardy_mesh_operation("my mesh"), Err(ShardingError::InvalidMeshSymbolName { .. }),));

        assert_eq!(partition_spec.to_shardy_dimension_shardings_literal(), "[{\"x\"}, {}]");
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute("mesh").unwrap(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}]>"
        );
    }

    #[test]
    fn test_sharding_layout_rank_mismatch() {
        let mesh = test_mesh_2x2();
        let partition_spec =
            PartitionSpecification::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("y")]);
        let sharding = NamedSharding::new(mesh, partition_spec).unwrap();
        assert!(matches!(
            ShardingLayout::new(vec![8usize], BufferType::F32, sharding),
            Err(ShardingError::RankMismatch { partition_rank: 2, array_rank: 1 }),
        ));
    }

    #[test]
    fn test_sharding_layout_even_2d_partitioning() {
        let mesh = test_mesh_2x2();
        let partition_spec =
            PartitionSpecification::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("y")]);
        let sharding = NamedSharding::new(mesh, partition_spec).unwrap();
        let layout = ShardingLayout::new(vec![8, 6], BufferType::F32, sharding).unwrap();

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
        let partition_spec = PartitionSpecification::new(vec![PartitionDimension::sharded("x")]);
        let sharding = NamedSharding::new(mesh, partition_spec).unwrap();
        let layout = ShardingLayout::new(vec![5], BufferType::F32, sharding).unwrap();

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
            PartitionSpecification::new(vec![PartitionDimension::sharded_by(["x".to_string(), "y".to_string()])]);
        let sharding = NamedSharding::new(mesh, partition_spec).unwrap();
        let layout = ShardingLayout::new(vec![10], BufferType::F32, sharding).unwrap();

        assert_eq!(layout.shard_for_device(0).unwrap().slices()[0], ShardSlice::new(0, 3).unwrap());
        assert_eq!(layout.shard_for_device(1).unwrap().slices()[0], ShardSlice::new(3, 6).unwrap());
        assert_eq!(layout.shard_for_device(2).unwrap().slices()[0], ShardSlice::new(6, 8).unwrap());
        assert_eq!(layout.shard_for_device(3).unwrap().slices()[0], ShardSlice::new(8, 10).unwrap());
    }

    #[test]
    fn test_sharding_layout_process_filtering() {
        let mesh = test_mesh_2x2();
        let partition_spec =
            PartitionSpecification::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("y")]);
        let sharding = NamedSharding::new(mesh, partition_spec).unwrap();
        let layout = ShardingLayout::new(vec![8, 6], BufferType::F32, sharding).unwrap();

        assert_eq!(layout.shard_indices_for_process(0), vec![0, 1]);
        assert_eq!(layout.shard_indices_for_process(1), vec![2, 3]);
        assert_eq!(layout.shard_indices_for_process(42), Vec::<usize>::new());
    }

    #[test]
    fn test_array_driven_shardy_jit_sharded_matmul_on_cpu() {
        // Use the same 8-device CPU setup as `ryft_pjrt` tests.
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(8) }))
            .expect("failed to create 8-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 8);

        // Build mesh + shardings used for runtime arrays. In a JIT setting, we derive StableHLO Shardy
        // annotations directly from these arrays.
        let mesh_devices = client_devices
            .iter()
            .map(|device| MeshDevice::new(device.id().unwrap(), device.process_index().unwrap()))
            .collect::<Vec<_>>();
        let mesh = Mesh::new(vec![MeshAxis::new("x", 8).unwrap()], mesh_devices).unwrap();

        let lhs_sharding = NamedSharding::new(
            mesh.clone(),
            PartitionSpecification::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]),
        )
        .unwrap();
        let rhs_sharding = NamedSharding::new(mesh.clone(), PartitionSpecification::replicated(2)).unwrap();

        // Global lhs matrix is 8x4, split by rows across 8 devices (each shard is 1x4).
        // Row i is [i, i+1, i+2, i+3].
        let lhs_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(row_index, device)| {
                let row = row_index as f32;
                client
                    .buffer(
                        f32_values_to_bytes(&[row, row + 1.0, row + 2.0, row + 3.0]).as_slice(),
                        BufferType::F32,
                        [1u64, 4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        // Global rhs matrix is replicated on each device.
        // [[1, 2], [0, 1], [1, 0], [2, 1]]
        let rhs_values = [1.0f32, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let rhs_buffers = client_devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        f32_values_to_bytes(rhs_values.as_slice()).as_slice(),
                        BufferType::F32,
                        [4u64, 2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let lhs_array = Array::from_sharding(vec![8, 4], BufferType::F32, lhs_sharding.clone(), lhs_buffers).unwrap();
        let rhs_array = Array::from_sharding(vec![4, 2], BufferType::F32, rhs_sharding, rhs_buffers).unwrap();

        // Derive Shardy attributes from runtime arrays (JIT-style).
        let mesh_operation = lhs_array.to_shardy_mesh_operation("mesh").unwrap();
        let lhs_sharding_attribute = lhs_array.to_shardy_tensor_sharding_attribute("mesh").unwrap();
        let rhs_sharding_attribute = rhs_array.to_shardy_tensor_sharding_attribute("mesh").unwrap();
        let output_sharding_attribute = lhs_array.to_shardy_tensor_sharding_attribute("mesh").unwrap();

        assert_eq!(mesh_operation, "sdy.mesh @mesh = <[\"x\"=8]>");
        assert_eq!(lhs_sharding_attribute, "#sdy.sharding<@mesh, [{\"x\"}, {}]>");
        assert_eq!(rhs_sharding_attribute, "#sdy.sharding<@mesh, [{}, {}]>");

        let mlir_program = format!(
            r#"
                module {{
                    {mesh_operation}
                    func.func @main(
                        %arg0: tensor<8x4xf32> {{sdy.sharding = {lhs_sharding_attribute}}},
                        %arg1: tensor<4x2xf32> {{sdy.sharding = {rhs_sharding_attribute}}}
                    ) -> (tensor<8x2xf32> {{sdy.sharding = {output_sharding_attribute}}}) {{
                        %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [] x [], contracting_dims = [1] x [0]
                            : (tensor<8x4xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>
                        return %0 : tensor<8x2xf32>
                    }}
                }}
            "#
        );
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(8)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 8);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let row_start_by_device = execution_device_ids
            .iter()
            .map(|device_id| {
                let row_start = lhs_array.shard_for_device(*device_id).unwrap().slices()[0].start();
                (*device_id, row_start)
            })
            .collect::<HashMap<_, _>>();

        let execute_arguments =
            Array::into_execute_arguments(vec![lhs_array, rhs_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), 0, None, Some(file!()), None, None)
            .unwrap();

        // Validate each output shard: row r should be [4r + 8, 4r + 4].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            output.done.r#await().unwrap();
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values = two_f32s_from_bytes(output_bytes.as_slice());
            let row = *row_start_by_device.get(&device_id).unwrap() as f32;
            assert_eq!(values[0], 4.0 * row + 8.0);
            assert_eq!(values[1], 4.0 * row + 4.0);
        }
    }
}
