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
//! | Ryft type              | JAX equivalent                         | Shardy MLIR representation                 |
//! | ---------------------- | -------------------------------------- | ------------------------------------------ |
//! | [`LogicalMesh`]        | `Mesh` axes + shape only               | `sdy.mesh @name = <["axis"=size, ...]>`    |
//! | [`DeviceMesh`]         | [`jax.sharding.Mesh`][jax-mesh]        | `sdy.mesh @name = <["axis"=size, ...]>`    |
//! | [`MeshAxis`]           | One entry in `Mesh.shape`              | `MeshAxisAttr` (name + size pair)          |
//! | [`MeshDevice`]         | One element in `Mesh.devices`          | Device ID in `MeshAttr.device_ids`         |
//! | [`Sharding`] | [`jax.sharding.NamedSharding`][jax-ns] | `#sdy.sharding<@mesh, [dim_shardings...]>` |
//! | [`ShardingDimension`]     | One element of a `PartitionSpec`-like payload | `DimensionShardingAttr`                    |
//! | [`ShardingLayout`]     | Computed internally by `jax.Array`     | runtime metadata only                      |
//! | [`ShardDescriptor`]    | `jax.Shard` from `array.global_shards` | runtime metadata only                      |
//!
//! [jax-mesh]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh
//! [jax-ns]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.NamedSharding
//!
//! # Logical mesh vs concrete mesh
//!
//! [`LogicalMesh`] captures only the logical topology (axis names and sizes) and is used
//! wherever device identity is irrelevant - principally in [`Sharding`] and for
//! rendering Shardy MLIR attributes at compilation time.
//!
//! [`DeviceMesh`] wraps a [`LogicalMesh`] and adds a concrete device list, which is needed at
//! runtime for computing per-device shard metadata in [`ShardingLayout`].
//!
//! # Generic vs specialized Shardy lowering
//!
//! The generic [`Sharding`] renderer emits fully explicit Shardy shardings, where
//! replicated dimensions are closed (`{}`). Specialized lowering sites such as `shard_map` can
//! still compute open dimensions (`{?}`) when that is required by Shardy's manual-computation
//! semantics.
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
//!    let mesh = DeviceMesh::new(
//!        LogicalMesh::new(vec![MeshAxis::new("batch", 8, MeshAxisType::Auto)?])?,
//!        mesh_devices,
//!    )?;
//!
//!    // 2D mesh for data + model parallelism.
//!    // JAX equivalent: Mesh(np.array(devices).reshape(4, 2), ('data', 'model'))
//!    let mesh = DeviceMesh::new(
//!        LogicalMesh::new(vec![
//!            MeshAxis::new("data", 4, MeshAxisType::Auto)?,
//!            MeshAxis::new("model", 2, MeshAxisType::Auto)?,
//!        ])?,
//!        mesh_devices,
//!    )?;
//!    ```
//!
//! 2. **Create shardings** that describe how each array dimension maps to mesh axes:
//!
//!    ```ignore
//!    // Shard dim 0 along "data", replicate dim 1.
//!    // JAX equivalent: NamedSharding(mesh, PartitionSpec('data', None))
//!    let sharding = Sharding::new(
//!        mesh.logical_mesh.clone(),
//!        vec![
//!            ShardingDimension::sharded(["data"]),
//!            ShardingDimension::replicated(),
//!        ],
//!        vec![],
//!    )?;
//!
//!    // Shard dim 0 along both "data" and "model" axes.
//!    // JAX equivalent: NamedSharding(mesh, PartitionSpec(('data', 'model'),))
//!    let sharding = Sharding::new(
//!        mesh.logical_mesh.clone(),
//!        vec![ShardingDimension::sharded(["data", "model"])],
//!        vec![],
//!    )?;
//!
//!    // Fully replicated across all devices.
//!    // JAX equivalent: NamedSharding(mesh, PartitionSpec())
//!    let sharding = Sharding::replicated(mesh.logical_mesh.clone(), 2);
//!
//!    // Unconstrained dimension (let the propagator decide).
//!    // JAX equivalent: NamedSharding(mesh, PartitionSpec(UNCONSTRAINED))
//!    let sharding = Sharding::new(
//!        mesh.logical_mesh.clone(),
//!        vec![ShardingDimension::unconstrained()],
//!        vec![],
//!    )?;
//!    ```
//!
//! 3. **Compute shard metadata** to determine per-device array slices and identify addressable
//!    shards:
//!
//!    ```ignore
//!    let layout = ShardingLayout::new(vec![32, 128], mesh, sharding)?;
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
//! 4. **Convert to Shardy MLIR attributes** for StableHLO program annotation:
//!
//!    ```ignore
//!    // Generates: sdy.mesh @mesh = <["data"=4, "model"=2]>
//!    let context = MlirContext::new();
//!    let mesh_module = context.module(context.unknown_location());
//!    let mesh_op = mesh.logical_mesh.to_shardy_mesh(context.unknown_location());
//!    let mesh_op = mesh_module.body().append_operation(mesh_op);
//!
//!    // Generates: #sdy.sharding<@mesh, [{"data"}, {}]>
//!    let attr = sharding.to_shardy_tensor_sharding_attribute();
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
//! [`ShardingLayout`] computes metadata for *all* global shards - including those on remote
//! hosts - so that the full sharding picture is available for compilation. To identify which
//! shards are locally addressable, use [`ShardingLayout::shard_indices_for_process`] with
//! the current process index. Only addressable shards can be backed by actual PJRT buffers;
//! non-addressable shards exist only as metadata describing remote device placements.
//!
//! This mirrors JAX's `array.addressable_shards` (local) vs `array.global_shards` (all),
//! where accessing `.data` on a non-addressable shard raises an error.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use ryft_mlir::Context as MlirContext;
use ryft_mlir::dialects::shardy::{DimensionShardingAttributeRef, TensorShardingAttributeRef};

use crate::parameters::Parameter;
use crate::sharding::{
    DeviceMesh, LogicalMesh, MeshAxisType, MeshDevice, MeshDeviceId, SHARDY_MESH_SYMBOL_NAME, ShardingDimension,
    ShardingError,
};

// ---------------------------------------------------------------------------
// Sharding
// ---------------------------------------------------------------------------

/// Mesh-bound sharding for one logical array value.
///
/// This is the primary user-facing sharding type for compilation-time annotations. It owns the
/// [`LogicalMesh`] together with the per-dimension [`ShardingDimension`] assignments and any
/// unreduced mesh axes needed to model partial reductions.
///
/// # JAX equivalent
///
/// This corresponds to [`jax.sharding.NamedSharding(mesh, spec)`][jax-ns], while the
/// [`dimensions`](Self::dimensions) field carries the semantics of the nested
/// [`jax.sharding.PartitionSpec`][jax-pspec]:
///
/// ```ignore
/// let sharding = Sharding::new(
///     logical_mesh,
///     vec![
///         ShardingDimension::sharded(["data"]),
///         ShardingDimension::replicated(),
///     ],
///     vec![],
/// )?;
/// ```
///
/// [jax-ns]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.NamedSharding
/// [jax-pspec]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec
///
/// # Ranked dimensions vs unreduced axes
///
/// `dimensions` is indexed by tensor rank, while `unreduced_axes` is not. The former says how
/// each tensor dimension is partitioned; the latter records mesh axes that still carry
/// partial-reduction state even though they do not correspond to any tensor dimension.
///
/// ```ignore
/// let sharding = Sharding::new(
///     logical_mesh,
///     vec![
///         ShardingDimension::sharded(["data"]),
///         ShardingDimension::replicated(),
///     ],
///     vec!["model".into()],
/// )?;
///
/// assert_eq!(sharding.dimensions.len(), 2);
/// assert_eq!(sharding.unreduced_axes, vec!["model"]);
/// ```
///
/// In this example, `"data"` partitions tensor dimension `0`, while `"model"` does not shard any
/// ranked dimension and instead marks the value as still unreduced along the mesh axis `"model"`.
/// Without `unreduced_axes`, that unused mesh axis would be indistinguishable from a truly
/// replicated axis.
///
/// # Validation
///
/// The constructor validates that:
///
/// - Every referenced mesh axis exists in the mesh.
/// - No mesh axis is used more than once across all ranked dimensions.
/// - Every unreduced axis exists in the mesh and is not already used by the ranked dimensions.
///
/// # Shardy representation
///
/// Rendered as a [`TensorShardingAttr`][sdy-tensor] (`#sdy.sharding<...>`) via
/// [`to_shardy_tensor_sharding_attribute`][Sharding::to_shardy_tensor_sharding_attribute]:
///
/// ```text
/// #sdy.sharding<@mesh, [{"data"}, {}]>
/// #sdy.sharding<@mesh, [{"data"}, {}], replicated={"y"}>
/// #sdy.sharding<@mesh, [{"data"}, {}], unreduced={"z"}>
/// ```
///
/// [sdy-tensor]: https://openxla.org/shardy/sharding_representation
#[derive(Clone, Debug, PartialEq, Eq, ryft_macros::Parameter)]
pub struct Sharding {
    mesh: LogicalMesh,

    /// Ranked per-dimension partition assignments.
    pub dimensions: Vec<ShardingDimension>,

    /// Unreduced mesh axes attached to this sharding.
    pub unreduced_axes: Vec<String>,
}

impl Sharding {
    /// Creates a sharding from a mesh and per-dimension assignments.
    pub fn new(
        mesh: LogicalMesh,
        dimensions: Vec<ShardingDimension>,
        unreduced_axes: Vec<String>,
    ) -> Result<Self, ShardingError> {
        let mut seen = HashSet::new();
        let unreduced_axes = unreduced_axes.into_iter().filter(|axis_name| seen.insert(axis_name.clone())).collect();
        let sharding = Self { mesh, dimensions, unreduced_axes };
        validate_sharding(&sharding)?;
        Ok(sharding)
    }

    /// Creates a fully replicated sharding for an array with rank `rank`.
    ///
    /// All dimensions are [`Replicated`][ShardingDimension::Replicated], meaning the full
    /// tensor is present on every device. Equivalent to `NamedSharding(mesh, PartitionSpec())`
    /// in JAX, padded to the tensor rank with `None`.
    pub fn replicated(mesh: LogicalMesh, rank: usize) -> Self {
        Self { mesh, dimensions: vec![ShardingDimension::Replicated; rank], unreduced_axes: Vec::new() }
    }

    /// Returns the logical mesh of this sharding.
    pub fn mesh(&self) -> &LogicalMesh {
        &self.mesh
    }

    /// Rank represented by this sharding.
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the visible mesh axes that are implicitly replicated by this sharding.
    pub fn replicated_axes(&self) -> Vec<&str> {
        let used_axes = used_axes_in_sharding(self);
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

    /// Projects this sharding into traced/type-level semantics by hiding `Auto` mesh axes.
    ///
    /// This mirrors JAX's distinction between concrete shardings and type-specified shardings:
    /// auto axes may exist in the concrete mesh and runtime placement, but they are omitted from
    /// the traced sharding view carried by types.
    pub(crate) fn project_for_traced_sharding(&self) -> Self {
        let dimensions = self
            .dimensions
            .iter()
            .map(|dimension| match dimension {
                ShardingDimension::Replicated => ShardingDimension::Replicated,
                ShardingDimension::Unconstrained => ShardingDimension::Unconstrained,
                ShardingDimension::Sharded(axis_names) => {
                    let visible_axis_names = axis_names
                        .iter()
                        .filter(|axis_name| {
                            matches!(
                                self.mesh.axis_type(axis_name),
                                Some(MeshAxisType::Explicit | MeshAxisType::Manual)
                            )
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    if visible_axis_names.is_empty() {
                        ShardingDimension::Replicated
                    } else {
                        ShardingDimension::Sharded(visible_axis_names)
                    }
                }
            })
            .collect();
        let unreduced_axes = self
            .unreduced_axes
            .iter()
            .filter(|axis_name| {
                matches!(self.mesh.axis_type(axis_name), Some(MeshAxisType::Explicit | MeshAxisType::Manual))
            })
            .cloned()
            .collect();
        Self { mesh: self.mesh.clone(), dimensions, unreduced_axes }
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
    /// Uses the canonical `@mesh` symbol name.
    pub fn to_shardy_tensor_sharding_attribute(&self) -> String {
        let context = MlirContext::new();
        self.to_shardy_tensor_sharding(&context).to_string()
    }

    /// Builds this sharding as typed Shardy dimension shardings.
    pub(crate) fn to_shardy_dimension_shardings<'c, 't>(
        &self,
        context: &'c MlirContext<'t>,
    ) -> Vec<DimensionShardingAttributeRef<'c, 't>> {
        self.dimensions
            .iter()
            .map(|dimension| match dimension {
                ShardingDimension::Replicated => context.shardy_dimension_sharding(&[], true, None),
                ShardingDimension::Sharded(axis_names) => {
                    let axes =
                        axis_names.iter().map(|axis_name| context.shardy_axis_ref(axis_name, None)).collect::<Vec<_>>();
                    context.shardy_dimension_sharding(axes.as_slice(), true, None)
                }
                ShardingDimension::Unconstrained => context.shardy_dimension_sharding(&[], false, None),
            })
            .collect()
    }

    /// Builds this sharding as a typed Shardy tensor-sharding attribute.
    pub(crate) fn to_shardy_tensor_sharding<'c, 't>(
        &self,
        context: &'c MlirContext<'t>,
    ) -> TensorShardingAttributeRef<'c, 't> {
        let mesh_symbol_ref = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
        let dim_shardings = self.to_shardy_dimension_shardings(context);
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
            dim_shardings.as_slice(),
            replicated_axes.as_slice(),
            unreduced_axes.as_slice(),
        )
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
/// For a replicated dimension, the slice spans the full extent `[0, dim_size)`. For a sharded
/// dimension, the slice covers the partition assigned to a specific device based on its mesh
/// coordinate.
pub type ShardSlice = Range<usize>;

/// Metadata for one global shard of a distributed array.
///
/// Each shard corresponds to one device in the mesh and describes the portion of the global
/// array that device holds. This is pure metadata - it does not contain actual buffer data.
///
/// # JAX equivalent
///
/// Analogous to one entry in JAX's [`array.global_shards`][jax-global-shards], which returns
/// a list of `Shard` objects:
///
/// | JAX `Shard` field           | `ShardDescriptor` method              |
/// | --------------------------- | ------------------------------------- |
/// | `shard.device`              | [`device()`][ShardDescriptor::device] |
/// | `shard.index` (slice tuple) | [`slices()`][ShardDescriptor::slices] |
/// | `shard.data.shape`          | [`shape()`][ShardDescriptor::shape]   |
/// | `shard.replica_id`          | derivable from mesh coordinate        |
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
/// Given a global array shape, a [`DeviceMesh`], and a [`Sharding`], this structure computes
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardingLayout {
    global_shape: Vec<usize>,
    mesh: DeviceMesh,
    sharding: Sharding,
    shards: Vec<ShardDescriptor>,
    shard_index_by_device: HashMap<MeshDeviceId, usize>,
}

impl ShardingLayout {
    /// Constructs shard metadata for all devices in the mesh.
    pub fn new(global_shape: Vec<usize>, mesh: DeviceMesh, sharding: Sharding) -> Result<Self, ShardingError> {
        if mesh.logical_mesh != sharding.mesh().clone() {
            return Err(ShardingError::MeshMismatch {
                expected: mesh.logical_mesh.clone(),
                actual: sharding.mesh().clone(),
            });
        }
        validate_sharding(&sharding)?;

        let partition_rank = sharding.rank();
        let array_rank = global_shape.len();
        if partition_rank != array_rank {
            return Err(ShardingError::ShardingRankMismatch { sharding_rank: partition_rank, array_rank });
        }

        let mut shards = Vec::with_capacity(mesh.device_count());
        let mut shard_index_by_device = HashMap::with_capacity(mesh.device_count());
        for (shard_index, mesh_device) in mesh.devices.iter().copied().enumerate() {
            let mesh_coordinate = mesh
                .device_coordinates(shard_index)
                .expect("mesh coordinate should exist for valid mesh device index");

            let mut slices = Vec::with_capacity(global_shape.len());
            let mut shape = Vec::with_capacity(global_shape.len());
            for (dimension, dimension_size) in global_shape.iter().copied().enumerate() {
                let slice = match &sharding.dimensions[dimension] {
                    ShardingDimension::Replicated => 0..dimension_size,
                    ShardingDimension::Sharded(axis_names) => {
                        let mut partition_index = 0usize;
                        let mut partition_count = 1usize;
                        for axis_name in axis_names {
                            let axis_index = mesh
                                .logical_mesh
                                .axis_indices
                                .get(axis_name.as_str())
                                .copied()
                                .expect("sharding mesh axes should be validated before building shard slices");
                            let axis_size = mesh.logical_mesh.axes[axis_index].size;
                            let axis_coordinate = mesh_coordinate[axis_index];

                            partition_index = partition_index * axis_size + axis_coordinate;
                            partition_count *= axis_size;
                        }

                        let base_size = dimension_size / partition_count;
                        let remainder = dimension_size % partition_count;
                        let extra_before = partition_index.min(remainder);

                        let start = partition_index * base_size + extra_before;
                        let size = base_size + usize::from(partition_index < remainder);
                        start..start + size
                    }
                    ShardingDimension::Unconstrained => 0..dimension_size,
                };
                shape.push(slice.len());
                slices.push(slice);
            }

            shard_index_by_device.insert(mesh_device.id, shard_index);
            shards.push(ShardDescriptor { shard_index, device: mesh_device, mesh_coordinate, slices, shape });
        }

        Ok(Self { global_shape, mesh, sharding, shards, shard_index_by_device })
    }

    /// Global array shape.
    pub fn global_shape(&self) -> &[usize] {
        self.global_shape.as_slice()
    }

    /// The mesh used to build this layout.
    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    /// The sharding used to build this layout.
    pub fn sharding(&self) -> &Sharding {
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
    pub fn shard_index_for_device(&self, device_id: MeshDeviceId) -> Option<usize> {
        self.shard_index_by_device.get(&device_id).copied()
    }

    /// Returns the shard descriptor for `device_id`, if the device is in the mesh.
    pub fn shard_for_device(&self, device_id: MeshDeviceId) -> Option<&ShardDescriptor> {
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
                (descriptor.device.process_index == process_index).then_some(descriptor.shard_index())
            })
            .collect()
    }
}

/// Validates a sharding against its logical mesh.
fn validate_sharding(sharding: &Sharding) -> Result<(), ShardingError> {
    let mut used_axes = HashSet::new();
    for (dimension, partition_dimension) in sharding.dimensions.iter().enumerate() {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
            if axis_names.is_empty() {
                return Err(ShardingError::EmptySharding { dimension });
            }

            let mut axes_in_dimension = HashSet::new();
            for axis_name in axis_names {
                if !sharding.mesh.axis_indices.contains_key(axis_name) {
                    return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
                }
                if !axes_in_dimension.insert(axis_name.clone()) || !used_axes.insert(axis_name.clone()) {
                    return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
                }
            }
        }
    }

    for axis_name in &sharding.unreduced_axes {
        if !sharding.mesh.axis_indices.contains_key(axis_name) {
            return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
        }
        if used_axes.contains(axis_name) {
            return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
        }
        used_axes.insert(axis_name.clone());
    }
    Ok(())
}

fn used_axes_in_sharding(sharding: &Sharding) -> HashSet<&str> {
    let mut used_axes = HashSet::new();
    for partition_dimension in &sharding.dimensions {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
            for axis_name in axis_names {
                used_axes.insert(axis_name.as_str());
            }
        }
    }
    for axis_name in &sharding.unreduced_axes {
        used_axes.insert(axis_name.as_str());
    }
    used_axes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::sharding::MeshAxis;

    use super::*;

    fn test_logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    fn test_device_mesh_2x2() -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        DeviceMesh::new(logical_mesh, devices).unwrap()
    }

    // -----------------------------------------------------------------------
    // Sharding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sharding_project_for_traced_sharding_hides_auto_axes() {
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
            vec!["reduction".into(), "carry".into()],
        )
        .unwrap();

        assert_eq!(
            sharding.project_for_traced_sharding(),
            Sharding::new(
                mesh,
                vec![
                    ShardingDimension::sharded(["data", "batch"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
                vec!["carry".into()],
            )
            .unwrap()
        );
    }

    #[test]
    fn test_sharding_validation() {
        let mesh = test_logical_mesh_2x2();

        assert!(matches!(
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["z"])], vec![]),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));

        assert!(matches!(
            Sharding::new(
                mesh.clone(),
                vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["x"])],
                vec![],
            ),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));

        assert!(matches!(
            Sharding::new(mesh, vec![ShardingDimension::Sharded(Vec::new())], vec![]),
            Err(ShardingError::EmptySharding { dimension }) if dimension == 0,
        ));
    }

    #[test]
    fn test_sharding_shardy_rendering() {
        let mesh = test_logical_mesh_2x2();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()], vec![])
                .unwrap();
        assert_eq!(sharding.to_shardy_tensor_sharding_attribute(), "#sdy.sharding<@mesh, [{\"x\"}, {}]>");
    }

    #[test]
    fn test_sharding_replicated_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()], vec![])
                .unwrap();

        assert_eq!(sharding.replicated_axes(), vec!["y"]);
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], replicated={\"y\"}>"
        );
    }

    #[test]
    fn test_sharding_unreduced_axes() {
        let mesh = test_logical_mesh_2x2();
        let sharding = Sharding::new(
            mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            vec!["y".into()],
        )
        .unwrap();
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute(),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced={\"y\"}>"
        );
    }

    #[test]
    fn test_sharding_replicated_and_unreduced_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])], vec!["z".into()]).unwrap();

        assert_eq!(sharding.replicated_axes(), vec!["y"]);
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute(),
            "#sdy.sharding<@mesh, [{\"x\"}], replicated={\"y\"}, unreduced={\"z\"}>"
        );
    }

    #[test]
    fn test_sharding_shardy_rendering_escapes_axis_names() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x\"y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new(r"path\to", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z\"w", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x\"y"])], vec!["z\"w".into()]).unwrap();

        assert_eq!(sharding.replicated_axes(), vec![r"path\to"]);
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute(),
            r#"#sdy.sharding<@mesh, [{"x\22y"}], replicated={"path\\to"}, unreduced={"z\22w"}>"#
        );
    }

    #[test]
    fn test_sharding_project_for_traced_sharding_filters_auto_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("w", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y", "z"])], vec!["w".into()]).unwrap();
        let projected = sharding.project_for_traced_sharding();

        assert_eq!(projected, Sharding::new(mesh, vec![ShardingDimension::sharded(["x", "z"])], vec![]).unwrap());
        assert!(projected.replicated_axes().is_empty());
        assert!(projected.unreduced_axes.is_empty());
    }

    #[test]
    fn test_sharding_unreduced_axis_validation() {
        let mesh = test_logical_mesh_2x2();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])], vec![]).unwrap();

        assert!(matches!(
            Sharding::new(mesh.clone(), sharding.dimensions.clone(), vec!["z".into()]),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));

        assert!(matches!(
            Sharding::new(mesh, sharding.dimensions, vec!["x".into()]),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));
    }

    // -----------------------------------------------------------------------
    // ShardingLayout tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sharding_layout_rank_mismatch() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
            vec![],
        )
        .unwrap();
        assert!(matches!(
            ShardingLayout::new(vec![8usize], mesh, sharding),
            Err(ShardingError::ShardingRankMismatch { sharding_rank: 2, array_rank: 1 }),
        ));
    }

    #[test]
    fn test_sharding_layout_unconstrained_is_ignored() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()],
            vec![],
        )
        .unwrap();
        let layout = ShardingLayout::new(vec![8, 6], mesh, sharding).unwrap();

        let shard0 = layout.shard_for_device(0).unwrap();
        let shard3 = layout.shard_for_device(3).unwrap();
        assert_eq!(shard0.slices()[0], 0..4);
        assert_eq!(shard0.slices()[1], 0..6);
        assert_eq!(shard3.slices()[0], 4..8);
        assert_eq!(shard3.slices()[1], 0..6);
        assert_eq!(shard0.shape(), &[4, 6]);
        assert_eq!(shard3.shape(), &[4, 6]);
    }

    #[test]
    fn test_sharding_layout_even_2d_partitioning() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
            vec![],
        )
        .unwrap();
        let layout = ShardingLayout::new(vec![8, 6], mesh, sharding).unwrap();

        let shard0 = layout.shard_for_device(0).unwrap();
        assert_eq!(shard0.shape(), &[4, 3]);
        assert_eq!(shard0.slices()[0], 0..4);
        assert_eq!(shard0.slices()[1], 0..3);

        let shard3 = layout.shard_for_device(3).unwrap();
        assert_eq!(shard3.shape(), &[4, 3]);
        assert_eq!(shard3.slices()[0], 4..8);
        assert_eq!(shard3.slices()[1], 3..6);
    }

    #[test]
    fn test_sharding_layout_uneven_partitioning() {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0)];
        let mesh = DeviceMesh::new(logical_mesh, devices).unwrap();
        let sharding =
            Sharding::new(mesh.logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])], vec![]).unwrap();
        let layout = ShardingLayout::new(vec![5], mesh, sharding).unwrap();

        let shard0 = layout.shard_for_device(0).unwrap();
        assert_eq!(shard0.shape(), &[3]);
        assert_eq!(shard0.slices()[0], 0..3);

        let shard1 = layout.shard_for_device(1).unwrap();
        assert_eq!(shard1.shape(), &[2]);
        assert_eq!(shard1.slices()[0], 3..5);
    }

    #[test]
    fn test_sharding_layout_multi_axis_single_dimension_partitioning() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x".to_string(), "y".to_string()])],
            vec![],
        )
        .unwrap();
        let layout = ShardingLayout::new(vec![10], mesh, sharding).unwrap();

        assert_eq!(layout.shard_for_device(0).unwrap().slices()[0], 0..3);
        assert_eq!(layout.shard_for_device(1).unwrap().slices()[0], 3..6);
        assert_eq!(layout.shard_for_device(2).unwrap().slices()[0], 6..8);
        assert_eq!(layout.shard_for_device(3).unwrap().slices()[0], 8..10);
    }

    #[test]
    fn test_sharding_layout_process_filtering() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
            vec![],
        )
        .unwrap();
        let layout = ShardingLayout::new(vec![8, 6], mesh, sharding).unwrap();

        assert_eq!(layout.shard_indices_for_process(0), vec![0, 1]);
        assert_eq!(layout.shard_indices_for_process(1), vec![2, 3]);
        assert_eq!(layout.shard_indices_for_process(42), Vec::<usize>::new());
    }

    #[test]
    fn test_sharding_layout_mesh_and_sharding_accessors() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            vec![],
        )
        .unwrap();
        let layout = ShardingLayout::new(vec![8, 6], mesh.clone(), sharding.clone()).unwrap();

        assert_eq!(layout.mesh(), &mesh);
        assert_eq!(layout.sharding(), &sharding);
    }

    #[test]
    fn test_sharding_layout_mesh_mismatch_reports_expected_and_actual_meshes() {
        let mesh = test_device_mesh_2x2();
        let actual = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::new(actual.clone(), vec![ShardingDimension::sharded(["z"])], vec![]).unwrap();

        assert_eq!(
            ShardingLayout::new(vec![8], mesh.clone(), sharding),
            Err(ShardingError::MeshMismatch { expected: mesh.logical_mesh, actual })
        );
    }
}
