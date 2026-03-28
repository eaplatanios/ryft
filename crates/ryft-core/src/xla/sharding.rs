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
//! | [`PartitionSpec`]      | [`jax.sharding.PartitionSpec`][jax-pspec] | Array of `DimensionShardingAttr`        |
//! | [`PartitionDimension`] | One element of a `PartitionSpec`       | `DimensionShardingAttr`                    |
//! | [`NamedSharding`]      | [`jax.sharding.NamedSharding`][jax-ns] | `#sdy.sharding<@mesh, [dim_shardings...]>` |
//! | [`ShardingLayout`]     | Computed internally by `jax.Array`     | runtime metadata only                      |
//! | [`ShardDescriptor`]    | `jax.Shard` from `array.global_shards` | runtime metadata only                      |
//!
//! [jax-mesh]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh
//! [jax-pspec]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.PartitionSpec
//! [jax-ns]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.NamedSharding
//!
//! # Logical mesh vs concrete mesh
//!
//! [`LogicalMesh`] captures only the logical topology (axis names and sizes) and is used
//! wherever device identity is irrelevant — principally in [`NamedSharding`] and for
//! rendering Shardy MLIR attributes at compilation time.
//!
//! [`DeviceMesh`] wraps a [`LogicalMesh`] and adds a concrete device list, which is needed at
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
//!    let mesh = DeviceMesh::new(
//!        vec![MeshAxis::new("batch", 8, MeshAxisType::Auto)?],
//!        mesh_devices,
//!    )?;
//!
//!    // 2D mesh for data + model parallelism.
//!    // JAX equivalent: Mesh(np.array(devices).reshape(4, 2), ('data', 'model'))
//!    let mesh = DeviceMesh::new(
//!        vec![
//!            MeshAxis::new("data", 4, MeshAxisType::Auto)?,
//!            MeshAxis::new("model", 2, MeshAxisType::Auto)?,
//!        ],
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
//!    let sharding = NamedSharding::new(mesh.logical_mesh().clone(), spec)?;
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
//!    let context = MlirContext::new();
//!    let mesh_module = context.module(context.unknown_location());
//!    let mesh_op = mesh
//!        .logical_mesh()
//!        .to_shardy_mesh_operation(context.unknown_location());
//!    let mesh_op = mesh_module.body().append_operation(mesh_op);
//!
//!    // Generates: #sdy.sharding<@mesh, [{"data"}, {}]>
//!    let attr = sharding.to_shardy_tensor_sharding_attribute(ShardingContext::ExplicitSharding);
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
use std::ops::Range;

#[cfg(test)]
use ryft_mlir::Block;
use ryft_mlir::Context as MlirContext;
use ryft_mlir::dialects::shardy::{DimensionShardingAttributeRef, TensorShardingAttributeRef};

use crate::parameters::Parameter;
use crate::sharding::{DeviceId, LogicalMesh, MeshAxis, MeshAxisType, SHARDY_MESH_SYMBOL_NAME, ShardingError};

// ---------------------------------------------------------------------------
// Sharding context
// ---------------------------------------------------------------------------

/// Controls how partition dimensions are rendered in Shardy MLIR attributes.
///
/// In Shardy's sharding representation, each dimension can be *closed* (fixed set of axes,
/// propagator will not change it) or *open* (ends with `?`, propagator may add axes).
///
/// | Variant              | `Unsharded` renders as | `Sharded(["x"])` renders as | `Unconstrained` renders as |
/// | -------------------- | ----------------------- | --------------------------- | -------------------------- |
/// | `ExplicitSharding`   | `{}` (closed)           | `{"x"}` (closed)            | `{?}` (open)               |
/// | `ShardingConstraint` | `{?}` (open)            | `{"x"}` (closed)            | `{?}` (open)               |
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
// Mesh-related types
// ---------------------------------------------------------------------------

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
/// row-major storage used by [`DeviceMesh`]. Explicit device IDs are only needed when the
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

/// Logical mesh of devices used by sharding layouts.
///
/// A mesh organizes physical devices into a multi-dimensional grid where each dimension has
/// a human-readable name. Devices are stored in **row-major order** with respect to the axis
/// list: for a 2D mesh with axes `("data"=4, "model"=2)`, the device at mesh coordinate
/// `(i, j)` has linear index `i * 2 + j`. This matches NumPy's default C-order and JAX's
/// `mesh.devices.flat`.
///
/// A `DeviceMesh` wraps a [`LogicalMesh`] (the logical topology) and adds a concrete device
/// list. Use [`logical_mesh()`][DeviceMesh::logical_mesh] to access the topology-only view,
/// which is needed for [`NamedSharding`] and Shardy attribute rendering.
///
/// # JAX equivalent
///
/// This corresponds directly to [`jax.sharding.Mesh`][jax-mesh]:
///
/// | JAX                                 | Ryft                                             |
/// | ----------------------------------- | ------------------------------------------------ |
/// | `Mesh(np.array(devs).reshape(...))` | `DeviceMesh::new(vec![...], devs)`               |
/// | `mesh.shape`                        | `mesh.logical_mesh().axes`                       |
/// | `mesh.devices` (ndarray)            | `mesh.devices()` (flat row-major slice)          |
/// | `mesh.size`                         | `mesh.device_count()`                            |
/// | `mesh.local_devices`                | `mesh.devices()` filtered by `process_index`     |
///
/// [jax-mesh]: https://docs.jax.dev/en/latest/jax.sharding.html#jax.sharding.Mesh
///
/// # Shardy representation
///
/// Rendered via the inner [`LogicalMesh`]:
///
/// ```mlir
/// sdy.mesh @mesh = <["data"=4, "model"=2]>
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceMesh {
    logical_mesh: LogicalMesh,
    devices: Vec<MeshDevice>,
}

impl DeviceMesh {
    /// Creates a mesh from named axes and row-major devices.
    ///
    /// Preserves the type of each provided axis.
    /// The expected number of `devices` is the product of all `axes` sizes. For an empty axis list, the
    /// expected device count is `1`.
    pub fn new(axes: Vec<MeshAxis>, devices: Vec<MeshDevice>) -> Result<Self, ShardingError> {
        let logical_mesh = LogicalMesh::new(axes)?;
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

    /// Returns the logical mesh topology (axes only).
    pub fn logical_mesh(&self) -> &LogicalMesh {
        &self.logical_mesh
    }

    /// Returns mesh devices in row-major order.
    pub fn devices(&self) -> &[MeshDevice] {
        self.devices.as_slice()
    }

    /// Returns the number of devices in this mesh.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Returns the size of `axis_name` in this mesh, if present.
    pub fn axis_size<S: AsRef<str>>(&self, axis_name: S) -> Option<usize> {
        self.logical_mesh.axis_size(axis_name)
    }

    /// Returns the names of axes with type [`MeshAxisType::Manual`].
    pub fn manual_axes(&self) -> Vec<&str> {
        self.logical_mesh
            .axes
            .iter()
            .filter_map(|axis| (axis.r#type == MeshAxisType::Manual).then_some(axis.name.as_str()))
            .collect()
    }

    /// Returns the mesh coordinate of the device at `device_index`, if valid.
    pub fn coordinate_for_device_index(&self, device_index: usize) -> Option<Vec<usize>> {
        (device_index < self.devices.len()).then(|| {
            let axis_sizes = self.logical_mesh.axes.iter().map(|axis| axis.size).collect::<Vec<_>>();
            coordinate_for_linear_index(device_index, axis_sizes.as_slice())
        })
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
/// | JAX `PartitionSpec` element   | `PartitionDimension`                                   |
/// | ----------------------------- | ------------------------------------------------------ |
/// | `None`                        | [`Unsharded`][PartitionDimension::Unsharded]           |
/// | `'axis_name'`                 | [`sharded("axis_name")`][PartitionDimension::sharded]  |
/// | `('axis_a', 'axis_b')`        | [`sharded_by(["axis_a", "axis_b"])`][PartitionDimension::sharded_by] |
/// | `PartitionSpec.UNCONSTRAINED` | [`Unconstrained`][PartitionDimension::Unconstrained]   |
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
/// | `PartitionDimension`  | `ExplicitSharding`    | `ShardingConstraint` |
/// | --------------------- | --------------------- | -------------------- |
/// | `Unsharded`           | `{}` (closed)         | `{?}` (open)         |
/// | `Sharded(["x"])`      | `{"x"}` (closed)      | `{"x"}` (closed)     |
/// | `Sharded(["x", "y"])` | `{"x", "y"}` (closed) | `{"x", "y"}` (closed) |
/// | `Unconstrained`       | `{?}` (open)          | `{?}` (open)         |
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
    /// Equivalent to `PartitionSpec.UNCONSTRAINED` in JAX. This is primarily meaningful in
    /// constraint annotations. When used in a [`ShardingLayout`], it is ignored for concrete
    /// slice computation and therefore behaves like a full-range dimension for shard metadata.
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
}

/// Escapes a string for inclusion in quoted Shardy text syntax.
#[cfg(feature = "xla")]
pub(crate) fn escape_shardy_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Partition specification for all logical array dimensions.
///
/// A sequence of per-dimension [`PartitionDimension`] entries describing how a tensor is
/// distributed across a mesh, plus any unreduced mesh axes needed to model partial reductions.
///
/// # JAX equivalent
///
/// This mirrors [`jax.sharding.PartitionSpec`][jax-pspec] (commonly aliased as `P`):
///
/// | JAX                     | Ryft                                           |
/// | ----------------------- | ---------------------------------------------- |
/// | `P('data', None)`       | `new(vec![sharded("data"), unsharded()])`      |
/// | `P('data', 'model')`    | `new(vec![sharded("data"), sharded("model")])` |
/// | `P(('data', 'model'),)` | `new(vec![sharded_by(["data", "model"])])`     |
/// | `P()` (replicated)      | `replicated(2)`                                |
/// | `P(UNCONSTRAINED)`      | `new(vec![unconstrained()])`                   |
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, ryft_macros::Parameter)]
pub struct PartitionSpec {
    dimensions: Vec<PartitionDimension>,
    unreduced_axes: Vec<String>,
}

impl PartitionSpec {
    /// Creates a partition specification from per-dimension assignments.
    pub fn new(dimensions: Vec<PartitionDimension>) -> Self {
        Self { dimensions, unreduced_axes: Vec::new() }
    }

    /// Creates a fully replicated specification for an array with rank `rank`.
    ///
    /// All dimensions are [`Unsharded`][PartitionDimension::Unsharded], meaning the full
    /// tensor is present on every device. Equivalent to `PartitionSpec()` in JAX (padded
    /// to the tensor rank with `None`).
    pub fn replicated(rank: usize) -> Self {
        Self { dimensions: vec![PartitionDimension::Unsharded; rank], unreduced_axes: Vec::new() }
    }

    /// Returns a copy of this partition specification with explicit unreduced axes attached.
    ///
    /// These axes are carried separately from the ranked dimension list because they represent
    /// partial-reduction state rather than per-dimension partitioning.
    pub fn with_unreduced_axes<I, N>(mut self, axis_names: I) -> Self
    where
        I: IntoIterator<Item = N>,
        N: Into<String>,
    {
        let mut seen = HashSet::new();
        self.unreduced_axes =
            axis_names.into_iter().map(Into::into).filter(|axis_name| seen.insert(axis_name.clone())).collect();
        self
    }

    /// Returns per-dimension partition assignments.
    pub fn dimensions(&self) -> &[PartitionDimension] {
        self.dimensions.as_slice()
    }

    /// Returns the unreduced mesh axes attached to this partition specification.
    pub fn unreduced_axes(&self) -> &[String] {
        self.unreduced_axes.as_slice()
    }

    /// Rank represented by this partition specification.
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Projects this partition specification into the traced/type-level sharding view of `mesh`.
    ///
    /// `Auto` mesh axes are hidden from traced shardings, so any dimension sharded only by auto
    /// axes becomes [`Unsharded`](PartitionDimension::Unsharded) in the projected view.
    fn project_for_traced_sharding(&self, mesh: &LogicalMesh) -> Self {
        let dimensions = self
            .dimensions
            .iter()
            .map(|dimension| match dimension {
                PartitionDimension::Unsharded => PartitionDimension::Unsharded,
                PartitionDimension::Unconstrained => PartitionDimension::Unconstrained,
                PartitionDimension::Sharded(axis_names) => {
                    let visible_axis_names = axis_names
                        .iter()
                        .filter(|axis_name| {
                            matches!(mesh.axis_type(axis_name), Some(MeshAxisType::Explicit | MeshAxisType::Manual))
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    if visible_axis_names.is_empty() {
                        PartitionDimension::Unsharded
                    } else {
                        PartitionDimension::Sharded(visible_axis_names)
                    }
                }
            })
            .collect();
        let unreduced_axes = self
            .unreduced_axes
            .iter()
            .filter(|axis_name| {
                matches!(mesh.axis_type(axis_name), Some(MeshAxisType::Explicit | MeshAxisType::Manual))
            })
            .cloned()
            .collect();
        Self { dimensions, unreduced_axes }
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

    /// Builds this partition specification as typed Shardy dimension shardings.
    pub(crate) fn to_shardy_dimension_shardings<'c, 't>(
        &self,
        context: &'c MlirContext<'t>,
        sharding_context: ShardingContext,
    ) -> Vec<DimensionShardingAttributeRef<'c, 't>> {
        self.dimensions
            .iter()
            .map(|dimension| match dimension {
                PartitionDimension::Unsharded => context.shardy_dimension_sharding(
                    &[],
                    matches!(sharding_context, ShardingContext::ExplicitSharding),
                    None,
                ),
                PartitionDimension::Sharded(axis_names) => {
                    let axes =
                        axis_names.iter().map(|axis_name| context.shardy_axis_ref(axis_name, None)).collect::<Vec<_>>();
                    context.shardy_dimension_sharding(axes.as_slice(), true, None)
                }
                PartitionDimension::Unconstrained => context.shardy_dimension_sharding(&[], false, None),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Named sharding
// ---------------------------------------------------------------------------

/// Named sharding defined by a [`LogicalMesh`] and a [`PartitionSpec`].
///
/// This is the primary user-facing sharding type for compilation-time annotations,
/// fully describing how a tensor is distributed across a mesh topology. It uses
/// [`LogicalMesh`] rather than [`DeviceMesh`] because device identity is not needed for
/// generating Shardy MLIR attributes.
///
/// # JAX equivalent
///
/// Corresponds to [`jax.sharding.NamedSharding(mesh, spec)`][jax-ns]:
///
/// ```ignore
/// // JAX:   NamedSharding(mesh, PartitionSpec('data', None))
/// // Ryft:
/// let sharding = NamedSharding::new(logical_mesh, PartitionSpec::new(vec![
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
/// - **Replicated axes** are derived from visible mesh axes not used by the partition spec.
/// - **Unreduced axes** are stored on the [`PartitionSpec`] and indicate a partial reduction:
///   the tensor has been partitioned but the all-reduce has not yet been applied.
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
/// - Every unreduced axis exists in the mesh and is not already used in the partition
///   specification.
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
    mesh: LogicalMesh,
    partition_spec: PartitionSpec,
}

impl NamedSharding {
    /// Creates a named sharding from a logical mesh and partition specification.
    pub fn new(mesh: LogicalMesh, partition_spec: PartitionSpec) -> Result<Self, ShardingError> {
        validate_partition_spec(&mesh, &partition_spec)?;
        Ok(Self { mesh, partition_spec })
    }

    /// Returns the logical mesh of this sharding.
    pub fn mesh(&self) -> &LogicalMesh {
        &self.mesh
    }

    /// Returns the partition specification of this sharding.
    pub fn partition_spec(&self) -> &PartitionSpec {
        &self.partition_spec
    }

    /// Returns the visible mesh axes that are implicitly replicated by this sharding.
    pub fn replicated_axes(&self) -> Vec<&str> {
        let used_axes = used_axes_in_partition_spec(&self.partition_spec);
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
        let partition_spec = self.partition_spec.project_for_traced_sharding(&self.mesh);
        Self { mesh: self.mesh.clone(), partition_spec }
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
    ///   - `context`: Controls open/closed rendering for unsharded dimensions.
    ///
    /// Uses the canonical `@mesh` symbol name.
    pub fn to_shardy_tensor_sharding_attribute(&self, context: ShardingContext) -> String {
        let dim_shardings = self.partition_spec.to_shardy_dimension_shardings_literal(context);
        let mut result = format!("#sdy.sharding<@{SHARDY_MESH_SYMBOL_NAME}, {dim_shardings}");

        let replicated_axes = self.replicated_axes();
        if !replicated_axes.is_empty() {
            result.push_str(", replicated={");
            for (i, axis_name) in replicated_axes.iter().enumerate() {
                if i > 0 {
                    result.push_str(", ");
                }
                result.push('"');
                result.push_str(escape_shardy_string(axis_name).as_str());
                result.push('"');
            }
            result.push('}');
        }

        if !self.partition_spec.unreduced_axes.is_empty() {
            result.push_str(", unreduced={");
            for (i, axis_name) in self.partition_spec.unreduced_axes.iter().enumerate() {
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
        result
    }

    /// Builds this sharding as a typed Shardy tensor-sharding attribute.
    pub(crate) fn to_shardy_tensor_sharding<'c, 't>(
        &self,
        context: &'c MlirContext<'t>,
        sharding_context: ShardingContext,
    ) -> TensorShardingAttributeRef<'c, 't> {
        let mesh_symbol_ref = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
        let dim_shardings = self.partition_spec.to_shardy_dimension_shardings(context, sharding_context);
        let replicated_axes = self
            .replicated_axes()
            .iter()
            .map(|axis_name| context.shardy_axis_ref(axis_name, None))
            .collect::<Vec<_>>();
        let unreduced_axes = self
            .partition_spec
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
/// For an unsharded dimension, the slice spans the full extent `[0, dim_size)`. For a sharded
/// dimension, the slice covers the partition assigned to a specific device based on its mesh
/// coordinate.
pub type ShardSlice = Range<usize>;

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
/// Given a global array shape, a [`DeviceMesh`], and a [`PartitionSpec`], this structure computes
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
    partition_spec: PartitionSpec,
    shards: Vec<ShardDescriptor>,
    shard_index_by_device: HashMap<DeviceId, usize>,
}

impl ShardingLayout {
    /// Constructs shard metadata for all devices in the mesh.
    pub fn new(
        global_shape: Vec<usize>,
        mesh: DeviceMesh,
        partition_spec: PartitionSpec,
    ) -> Result<Self, ShardingError> {
        validate_partition_spec(mesh.logical_mesh(), &partition_spec)?;

        let partition_rank = partition_spec.rank();
        let array_rank = global_shape.len();
        if partition_rank != array_rank {
            return Err(ShardingError::PartitionSpecificationRankMismatch { partition_rank, array_rank });
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
                    PartitionDimension::Unsharded => 0..dimension_size,
                    PartitionDimension::Sharded(axis_names) => {
                        let mut partition_index = 0usize;
                        let mut partition_count = 1usize;
                        for axis_name in axis_names {
                            let axis_index =
                                mesh.logical_mesh().axis_indices.get(axis_name.as_str()).copied().expect(
                                    "partition spec mesh axes should be validated before building shard slices",
                                );
                            let axis_size = mesh.logical_mesh().axes[axis_index].size;
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
                    PartitionDimension::Unconstrained => 0..dimension_size,
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
    pub fn mesh(&self) -> &DeviceMesh {
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

/// Validates a partition spec against a logical mesh.
fn validate_partition_spec(mesh: &LogicalMesh, partition_spec: &PartitionSpec) -> Result<(), ShardingError> {
    let mut used_axes = HashSet::new();
    for (dimension, partition_dimension) in partition_spec.dimensions().iter().enumerate() {
        if let PartitionDimension::Sharded(axis_names) = partition_dimension {
            if axis_names.is_empty() {
                return Err(ShardingError::EmptyPartitionSpecification { dimension });
            }

            let mut axes_in_dimension = HashSet::new();
            for axis_name in axis_names {
                if !mesh.axis_indices.contains_key(axis_name) {
                    return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
                }
                if !axes_in_dimension.insert(axis_name.clone()) || !used_axes.insert(axis_name.clone()) {
                    return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
                }
            }
        }
    }

    for axis_name in partition_spec.unreduced_axes() {
        if !mesh.axis_indices.contains_key(axis_name) {
            return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
        }
        if used_axes.contains(axis_name) {
            return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
        }
        used_axes.insert(axis_name.clone());
    }
    Ok(())
}

fn used_axes_in_partition_spec(partition_spec: &PartitionSpec) -> HashSet<&str> {
    let mut used_axes = HashSet::new();
    for partition_dimension in partition_spec.dimensions() {
        if let PartitionDimension::Sharded(axis_names) = partition_dimension {
            for axis_name in axis_names {
                used_axes.insert(axis_name.as_str());
            }
        }
    }
    for axis_name in partition_spec.unreduced_axes() {
        used_axes.insert(axis_name.as_str());
    }
    used_axes
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    fn test_device_mesh_2x2() -> DeviceMesh {
        let axes = vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        DeviceMesh::new(axes, devices).unwrap()
    }

    // -----------------------------------------------------------------------
    // LogicalMesh tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_logical_mesh_construction_and_lookups() {
        let mesh = test_logical_mesh_2x2();
        assert_eq!(mesh.axes.len(), 2);
        assert_eq!(mesh.axis_indices.get("x"), Some(&0));
        assert_eq!(mesh.axis_indices.get("y"), Some(&1));
        assert_eq!(mesh.axis_indices.get("z"), None);
        assert_eq!(mesh.axis_size("x"), Some(2));
        assert_eq!(mesh.axis_size("y"), Some(2));
        assert_eq!(mesh.device_count(), 4);
    }

    #[test]
    fn test_logical_mesh_validation() {
        let axes = vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("x", 3, MeshAxisType::Auto).unwrap(),
        ];
        assert!(matches!(
            LogicalMesh::new(axes),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));
    }

    #[test]
    fn test_logical_mesh_shardy_rendering() {
        let mesh = test_logical_mesh_2x2();
        let context = MlirContext::new();
        let module = context.module(context.unknown_location());
        assert_eq!(
            module.body().append_operation(mesh.to_shardy_mesh(context.unknown_location())).to_string(),
            "sdy.mesh @mesh = <[\"x\"=2, \"y\"=2]>"
        );
    }

    // -----------------------------------------------------------------------
    // DeviceMesh tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_device_mesh_construction_preserves_logical_mesh_data() {
        let axes = vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 1), MeshDevice::new(3, 1)];
        let mesh = DeviceMesh::new(axes, devices).unwrap();
        assert_eq!(mesh.logical_mesh().axes.iter().map(|axis| axis.name.as_str()).collect::<Vec<_>>(), vec!["x", "y"]);
        assert_eq!(mesh.device_count(), 4);
    }

    #[test]
    fn test_device_mesh_coordinate_mapping_by_index() {
        let mesh = test_device_mesh_2x2();
        assert_eq!(mesh.coordinate_for_device_index(0), Some(vec![0, 0]));
        assert_eq!(mesh.coordinate_for_device_index(1), Some(vec![0, 1]));
        assert_eq!(mesh.coordinate_for_device_index(2), Some(vec![1, 0]));
        assert_eq!(mesh.coordinate_for_device_index(3), Some(vec![1, 1]));
        assert_eq!(mesh.coordinate_for_device_index(99), None);
    }

    #[test]
    fn test_device_mesh_validation() {
        assert!(matches!(MeshAxis::new("", 4, MeshAxisType::Auto), Err(ShardingError::EmptyMeshAxisName)));
        assert!(matches!(
            MeshAxis::new("x", 0, MeshAxisType::Auto),
            Err(ShardingError::EmptyMeshAxis { name }) if name == "x",
        ));

        let axes = vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
        ];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 0), MeshDevice::new(3, 0)];
        assert!(matches!(
            DeviceMesh::new(axes, devices),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));

        let axes = vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(0, 0)];
        assert!(matches!(
            DeviceMesh::new(axes, devices),
            Err(ShardingError::DuplicateMeshDeviceId { id }) if id == 0,
        ));
    }

    // -----------------------------------------------------------------------
    // PartitionDimension / PartitionSpec tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_partition_dimension_unconstrained() {
        let dim = PartitionDimension::unconstrained();
        assert!(matches!(dim, PartitionDimension::Unconstrained));
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

    #[test]
    fn test_partition_spec_project_for_traced_sharding_hides_auto_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("batch", 8, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let spec = PartitionSpec::new(vec![
            PartitionDimension::sharded_by(["data", "model", "batch"]),
            PartitionDimension::sharded("model"),
            PartitionDimension::unsharded(),
        ])
        .with_unreduced_axes(["model", "batch"]);

        assert_eq!(
            spec.project_for_traced_sharding(&mesh),
            PartitionSpec::new(vec![
                PartitionDimension::sharded_by(["data", "batch"]),
                PartitionDimension::unsharded(),
                PartitionDimension::unsharded(),
            ])
            .with_unreduced_axes(["batch"])
        );
    }

    // -----------------------------------------------------------------------
    // NamedSharding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_named_sharding_validation() {
        let mesh = test_logical_mesh_2x2();

        let unknown_axis_partition = PartitionSpec::new(vec![PartitionDimension::sharded("z")]);
        assert!(matches!(
            NamedSharding::new(mesh.clone(), unknown_axis_partition),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));

        let duplicate_axis_partition =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("x")]);
        assert!(matches!(
            NamedSharding::new(mesh.clone(), duplicate_axis_partition),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));

        let empty_axis_partition = PartitionSpec::new(vec![PartitionDimension::Sharded(Vec::new())]);
        assert!(matches!(
            NamedSharding::new(mesh, empty_axis_partition),
            Err(ShardingError::EmptyPartitionSpecification { dimension }) if dimension == 0,
        ));
    }

    #[test]
    fn test_named_sharding_shardy_rendering() {
        let mesh = test_logical_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let sharding = NamedSharding::new(mesh, partition_spec).unwrap();
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute(ShardingContext::ExplicitSharding),
            "#sdy.sharding<@mesh, [{\"x\"}, {}]>"
        );
    }

    #[test]
    fn test_named_sharding_replicated_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let sharding = NamedSharding::new(mesh, partition_spec).unwrap();

        assert_eq!(sharding.replicated_axes(), vec!["y"]);
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute(ShardingContext::ExplicitSharding),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], replicated={\"y\"}>"
        );
    }

    #[test]
    fn test_named_sharding_unreduced_axes() {
        let mesh = test_logical_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()])
                .with_unreduced_axes(["y"]);
        let sharding = NamedSharding::new(mesh, partition_spec).unwrap();
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute(ShardingContext::ExplicitSharding),
            "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced={\"y\"}>"
        );
    }

    #[test]
    fn test_named_sharding_replicated_and_unreduced_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]).with_unreduced_axes(["z"]);
        let sharding = NamedSharding::new(mesh, partition_spec).unwrap();

        assert_eq!(sharding.replicated_axes(), vec!["y"]);
        assert_eq!(
            sharding.to_shardy_tensor_sharding_attribute(ShardingContext::ExplicitSharding),
            "#sdy.sharding<@mesh, [{\"x\"}], replicated={\"y\"}, unreduced={\"z\"}>"
        );
    }

    #[test]
    fn test_named_sharding_project_for_traced_sharding_filters_auto_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("w", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let sharding = NamedSharding::new(
            mesh,
            PartitionSpec::new(vec![PartitionDimension::sharded_by(["x", "y", "z"])]).with_unreduced_axes(["w"]),
        )
        .unwrap();
        let projected = sharding.project_for_traced_sharding();

        assert_eq!(projected.partition_spec(), &PartitionSpec::new(vec![PartitionDimension::sharded_by(["x", "z"])]));
        assert!(projected.replicated_axes().is_empty());
        assert!(projected.partition_spec().unreduced_axes().is_empty());
    }

    #[test]
    fn test_named_sharding_unreduced_axis_validation() {
        let mesh = test_logical_mesh_2x2();
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);

        assert!(matches!(
            NamedSharding::new(mesh.clone(), partition_spec.clone().with_unreduced_axes(["z"])),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));

        assert!(matches!(
            NamedSharding::new(mesh, partition_spec.with_unreduced_axes(["x"])),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));
    }

    // -----------------------------------------------------------------------
    // ShardingLayout tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sharding_layout_rank_mismatch() {
        let mesh = test_device_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("y")]);
        assert!(matches!(
            ShardingLayout::new(vec![8usize], mesh, partition_spec),
            Err(ShardingError::PartitionSpecificationRankMismatch { partition_rank: 2, array_rank: 1 }),
        ));
    }

    #[test]
    fn test_sharding_layout_unconstrained_is_ignored() {
        let mesh = test_device_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unconstrained()]);
        let layout = ShardingLayout::new(vec![8, 6], mesh, partition_spec).unwrap();

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
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("y")]);
        let layout = ShardingLayout::new(vec![8, 6], mesh, partition_spec).unwrap();

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
        let axes = vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0)];
        let mesh = DeviceMesh::new(axes, devices).unwrap();
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded("x")]);
        let layout = ShardingLayout::new(vec![5], mesh, partition_spec).unwrap();

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
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded_by(["x".to_string(), "y".to_string()])]);
        let layout = ShardingLayout::new(vec![10], mesh, partition_spec).unwrap();

        assert_eq!(layout.shard_for_device(0).unwrap().slices()[0], 0..3);
        assert_eq!(layout.shard_for_device(1).unwrap().slices()[0], 3..6);
        assert_eq!(layout.shard_for_device(2).unwrap().slices()[0], 6..8);
        assert_eq!(layout.shard_for_device(3).unwrap().slices()[0], 8..10);
    }

    #[test]
    fn test_sharding_layout_process_filtering() {
        let mesh = test_device_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::sharded("y")]);
        let layout = ShardingLayout::new(vec![8, 6], mesh, partition_spec).unwrap();

        assert_eq!(layout.shard_indices_for_process(0), vec![0, 1]);
        assert_eq!(layout.shard_indices_for_process(1), vec![2, 3]);
        assert_eq!(layout.shard_indices_for_process(42), Vec::<usize>::new());
    }

    #[test]
    fn test_sharding_layout_mesh_and_partition_spec_accessors() {
        let mesh = test_device_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded("x"), PartitionDimension::unsharded()]);
        let layout = ShardingLayout::new(vec![8, 6], mesh.clone(), partition_spec.clone()).unwrap();

        assert_eq!(layout.mesh(), &mesh);
        assert_eq!(layout.partition_spec(), &partition_spec);
    }

    // -----------------------------------------------------------------------
    // MeshAxisType tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mesh_axis_type_default() {
        assert_eq!(MeshAxisType::default(), MeshAxisType::Auto);
    }

    #[test]
    fn test_mesh_axis_default_type() {
        let axis = MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap();
        assert_eq!(axis.r#type, MeshAxisType::Auto);
    }

    #[test]
    fn test_mesh_axis_with_explicit_type() {
        let axis = MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap();
        assert_eq!(axis.r#type, MeshAxisType::Manual);
    }

    #[test]
    fn test_logical_mesh_default_axis_types() {
        let mesh = test_logical_mesh_2x2();
        assert_eq!(
            mesh.axes.iter().map(|axis| axis.r#type).collect::<Vec<_>>(),
            vec![MeshAxisType::Auto, MeshAxisType::Auto]
        );
    }

    #[test]
    fn test_logical_mesh_preserves_axis_types_from_axes() {
        let axes = vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ];
        let mesh = LogicalMesh::new(axes).unwrap();
        assert_eq!(
            mesh.axes.iter().map(|axis| axis.r#type).collect::<Vec<_>>(),
            vec![MeshAxisType::Manual, MeshAxisType::Explicit]
        );
    }

    #[test]
    fn test_logical_mesh_axis_type_queries() {
        // All auto.
        let mesh = test_logical_mesh_2x2();
        assert!(mesh.axes.iter().all(|axis| axis.r#type == MeshAxisType::Auto));
        assert!(mesh.axes.iter().all(|axis| axis.r#type != MeshAxisType::Explicit));
        assert!(mesh.axes.iter().all(|axis| axis.r#type != MeshAxisType::Manual));
        assert_eq!(
            mesh.axes
                .iter()
                .filter_map(|axis| (axis.r#type == MeshAxisType::Auto).then_some(axis.name.as_str()))
                .collect::<Vec<_>>(),
            vec!["x", "y"]
        );
        assert!(
            mesh.axes
                .iter()
                .filter_map(|axis| (axis.r#type == MeshAxisType::Explicit).then_some(axis.name.as_str()))
                .collect::<Vec<_>>()
                .is_empty()
        );
        assert!(
            mesh.axes
                .iter()
                .filter_map(|axis| (axis.r#type == MeshAxisType::Manual).then_some(axis.name.as_str()))
                .collect::<Vec<_>>()
                .is_empty()
        );

        // Mixed types.
        let axes = vec![
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("b", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("c", 2, MeshAxisType::Manual).unwrap(),
        ];
        let mesh = LogicalMesh::new(axes).unwrap();
        assert!(!mesh.axes.iter().all(|axis| axis.r#type == MeshAxisType::Auto));
        assert!(!mesh.axes.iter().all(|axis| axis.r#type == MeshAxisType::Explicit));
        assert!(!mesh.axes.iter().all(|axis| axis.r#type == MeshAxisType::Manual));
        assert_eq!(
            mesh.axes
                .iter()
                .filter_map(|axis| (axis.r#type == MeshAxisType::Auto).then_some(axis.name.as_str()))
                .collect::<Vec<_>>(),
            vec!["a"]
        );
        assert_eq!(
            mesh.axes
                .iter()
                .filter_map(|axis| (axis.r#type == MeshAxisType::Explicit).then_some(axis.name.as_str()))
                .collect::<Vec<_>>(),
            vec!["b"]
        );
        assert_eq!(
            mesh.axes
                .iter()
                .filter_map(|axis| (axis.r#type == MeshAxisType::Manual).then_some(axis.name.as_str()))
                .collect::<Vec<_>>(),
            vec!["c"]
        );
    }

    #[test]
    fn test_logical_mesh_axis_names_and_sizes() {
        let mesh = test_logical_mesh_2x2();
        assert_eq!(mesh.axes.iter().map(|axis| axis.name.as_str()).collect::<Vec<_>>(), vec!["x", "y"]);
        assert_eq!(mesh.axes.iter().map(|axis| axis.size).collect::<Vec<_>>(), vec![2, 2]);
    }

    #[test]
    fn test_device_mesh_preserves_axis_types_from_axes() {
        let axes = vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ];
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0), MeshDevice::new(2, 0), MeshDevice::new(3, 0)];
        let mesh = DeviceMesh::new(axes, devices).unwrap();
        assert_eq!(
            mesh.logical_mesh().axes.iter().map(|axis| axis.r#type).collect::<Vec<_>>(),
            vec![MeshAxisType::Explicit, MeshAxisType::Manual]
        );
    }

    #[test]
    fn test_device_mesh_exposes_logical_mesh_axis_metadata() {
        let mesh = test_device_mesh_2x2();
        assert!(mesh.logical_mesh().axes.iter().all(|axis| axis.r#type == MeshAxisType::Auto));
        assert!(mesh.manual_axes().is_empty());
        assert_eq!(mesh.logical_mesh().axes.iter().map(|axis| axis.name.as_str()).collect::<Vec<_>>(), vec!["x", "y"]);
        assert_eq!(mesh.logical_mesh().axes.iter().map(|axis| axis.size).collect::<Vec<_>>(), vec![2, 2]);
    }
}
