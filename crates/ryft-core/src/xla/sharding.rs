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
//! 2. **Create partition specifications** that describe how each array dimension maps to mesh axes:
//!
//!    ```ignore
//!    // Shard dim 0 along "data", replicate dim 1.
//!    // JAX equivalent: PartitionSpec('data', None)
//!    let spec = PartitionSpec::new(vec![
//!        PartitionDimension::sharded(["data"]),
//!        PartitionDimension::unsharded(),
//!    ]);
//!
//!    // Shard dim 0 along both "data" and "model" axes.
//!    // JAX equivalent: PartitionSpec(('data', 'model'),)
//!    let spec = PartitionSpec::new(vec![
//!        PartitionDimension::sharded(["data", "model"]),
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
//!    let sharding = NamedSharding::new(mesh.logical_mesh.clone(), spec)?;
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
//!    let mesh_op = mesh.logical_mesh.to_shardy_mesh(context.unknown_location());
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

use ryft_mlir::Context as MlirContext;
use ryft_mlir::dialects::shardy::{DimensionShardingAttributeRef, TensorShardingAttributeRef};

use crate::parameters::Parameter;
use crate::sharding::{
    DeviceMesh, LogicalMesh, MeshAxisType, MeshDevice, MeshDeviceId, SHARDY_MESH_SYMBOL_NAME, ShardingError,
};

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
/// | `'axis_name'`                 | [`sharded(["axis_name"])`][PartitionDimension::sharded] |
/// | `('axis_a', 'axis_b')`        | [`sharded(["axis_a", "axis_b"])`][PartitionDimension::sharded] |
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

    /// Creates a partitioned dimension using one or more mesh axes, from major to minor.
    ///
    /// Equivalent to `'axis_name'` in JAX's `PartitionSpec` when exactly one axis name is
    /// provided, or to `('axis_a', 'axis_b')` when multiple axis names are provided. The
    /// dimension is split along the product of the referenced axis sizes.
    pub fn sharded<I, N>(axis_names: I) -> Self
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
/// | `P('data', None)`       | `new(vec![sharded(["data"]), unsharded()])`      |
/// | `P('data', 'model')`    | `new(vec![sharded(["data"]), sharded(["model"])])` |
/// | `P(('data', 'model'),)` | `new(vec![sharded(["data", "model"])])`          |
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
///     PartitionDimension::sharded(["data"]),
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
    shard_index_by_device: HashMap<MeshDeviceId, usize>,
}

impl ShardingLayout {
    /// Constructs shard metadata for all devices in the mesh.
    pub fn new(
        global_shape: Vec<usize>,
        mesh: DeviceMesh,
        partition_spec: PartitionSpec,
    ) -> Result<Self, ShardingError> {
        validate_partition_spec(&mesh.logical_mesh, &partition_spec)?;

        let partition_rank = partition_spec.rank();
        let array_rank = global_shape.len();
        if partition_rank != array_rank {
            return Err(ShardingError::PartitionSpecificationRankMismatch { partition_rank, array_rank });
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
                let slice = match &partition_spec.dimensions()[dimension] {
                    PartitionDimension::Unsharded => 0..dimension_size,
                    PartitionDimension::Sharded(axis_names) => {
                        let mut partition_index = 0usize;
                        let mut partition_count = 1usize;
                        for axis_name in axis_names {
                            let axis_index =
                                mesh.logical_mesh.axis_indices.get(axis_name.as_str()).copied().expect(
                                    "partition spec mesh axes should be validated before building shard slices",
                                );
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
                    PartitionDimension::Unconstrained => 0..dimension_size,
                };
                shape.push(slice.len());
                slices.push(slice);
            }

            shard_index_by_device.insert(mesh_device.id, shard_index);
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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
    // PartitionDimension / PartitionSpec tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_partition_dimension_unconstrained() {
        let dim = PartitionDimension::unconstrained();
        assert!(matches!(dim, PartitionDimension::Unconstrained));
    }

    #[test]
    fn test_partition_spec_shardy_rendering_explicit() {
        let spec = PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::unsharded()]);
        assert_eq!(spec.to_shardy_dimension_shardings_literal(ShardingContext::ExplicitSharding), "[{\"x\"}, {}]");
    }

    #[test]
    fn test_partition_spec_shardy_rendering_constraint() {
        let spec = PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::unsharded()]);
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
            PartitionDimension::sharded(["data", "model", "batch"]),
            PartitionDimension::sharded(["model"]),
            PartitionDimension::unsharded(),
        ])
        .with_unreduced_axes(["model", "batch"]);

        assert_eq!(
            spec.project_for_traced_sharding(&mesh),
            PartitionSpec::new(vec![
                PartitionDimension::sharded(["data", "batch"]),
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

        let unknown_axis_partition = PartitionSpec::new(vec![PartitionDimension::sharded(["z"])]);
        assert!(matches!(
            NamedSharding::new(mesh.clone(), unknown_axis_partition),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));

        let duplicate_axis_partition =
            PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::sharded(["x"])]);
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
            PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::unsharded()]);
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
            PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::unsharded()]);
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
            PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::unsharded()])
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
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded(["x"])]).with_unreduced_axes(["z"]);
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
            PartitionSpec::new(vec![PartitionDimension::sharded(["x", "y", "z"])]).with_unreduced_axes(["w"]),
        )
        .unwrap();
        let projected = sharding.project_for_traced_sharding();

        assert_eq!(projected.partition_spec(), &PartitionSpec::new(vec![PartitionDimension::sharded(["x", "z"])]));
        assert!(projected.replicated_axes().is_empty());
        assert!(projected.partition_spec().unreduced_axes().is_empty());
    }

    #[test]
    fn test_named_sharding_unreduced_axis_validation() {
        let mesh = test_logical_mesh_2x2();
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded(["x"])]);

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
            PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::sharded(["y"])]);
        assert!(matches!(
            ShardingLayout::new(vec![8usize], mesh, partition_spec),
            Err(ShardingError::PartitionSpecificationRankMismatch { partition_rank: 2, array_rank: 1 }),
        ));
    }

    #[test]
    fn test_sharding_layout_unconstrained_is_ignored() {
        let mesh = test_device_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::unconstrained()]);
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
            PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::sharded(["y"])]);
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
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0)];
        let mesh = DeviceMesh::new(logical_mesh, devices).unwrap();
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded(["x"])]);
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
        let partition_spec = PartitionSpec::new(vec![PartitionDimension::sharded(["x".to_string(), "y".to_string()])]);
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
            PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::sharded(["y"])]);
        let layout = ShardingLayout::new(vec![8, 6], mesh, partition_spec).unwrap();

        assert_eq!(layout.shard_indices_for_process(0), vec![0, 1]);
        assert_eq!(layout.shard_indices_for_process(1), vec![2, 3]);
        assert_eq!(layout.shard_indices_for_process(42), Vec::<usize>::new());
    }

    #[test]
    fn test_sharding_layout_mesh_and_partition_spec_accessors() {
        let mesh = test_device_mesh_2x2();
        let partition_spec =
            PartitionSpec::new(vec![PartitionDimension::sharded(["x"]), PartitionDimension::unsharded()]);
        let layout = ShardingLayout::new(vec![8, 6], mesh.clone(), partition_spec.clone()).unwrap();

        assert_eq!(layout.mesh(), &mesh);
        assert_eq!(layout.partition_spec(), &partition_spec);
    }
}
