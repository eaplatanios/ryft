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
//!        Vec::<&str>::new(),
//!        Vec::<&str>::new(),
//!        Vec::<&str>::new(),
//!    )?;
//!
//!    // Shard dim 0 along both "data" and "model" axes.
//!    // JAX equivalent: NamedSharding(mesh, PartitionSpec(('data', 'model'),))
//!    let sharding = Sharding::new(
//!        mesh.logical_mesh.clone(),
//!        vec![ShardingDimension::sharded(["data", "model"])],
//!        Vec::<&str>::new(),
//!        Vec::<&str>::new(),
//!        Vec::<&str>::new(),
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
//!        Vec::<&str>::new(),
//!        Vec::<&str>::new(),
//!        Vec::<&str>::new(),
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

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::ops::Range;

#[cfg(feature = "xla")]
use ryft_mlir::Context as MlirContext;
#[cfg(feature = "xla")]
use ryft_mlir::dialects::shardy::{DimensionShardingAttributeRef, TensorShardingAttributeRef};

use ryft_macros::Parameter;

use crate::parameters::Parameter;

#[cfg(feature = "xla")]
use crate::sharding::SHARDY_MESH_SYMBOL_NAME;
use crate::sharding::{
    DeviceMesh, LogicalMesh, MeshAxisType, MeshDevice, MeshDeviceId, ShardingDimension, ShardingError,
};

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
    /// Creates a sharding from a mesh and per-dimension assignments.
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

        for axis_name in &sharding.reduced_manual_axes {
            if !sharding.mesh.axis_indices.contains_key(axis_name) {
                return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
            }
            if sharding.mesh.axis_type(axis_name) != Some(MeshAxisType::Manual) {
                return Err(ShardingError::ExpectedManualMeshAxis { name: axis_name.clone() });
            }
            if used_axes.contains(axis_name) {
                return Err(ShardingError::DuplicateMeshAxisName { name: axis_name.clone() });
            }
            used_axes.insert(axis_name.clone());
        }

        for axis_name in &sharding.varying_manual_axes {
            if !sharding.mesh.axis_indices.contains_key(axis_name) {
                return Err(ShardingError::UnknownMeshAxisName { name: axis_name.clone() });
            }
        }

        Ok(sharding)
    }

    /// Creates a fully replicated sharding for an array with rank `rank`.
    ///
    /// All dimensions are [`Replicated`][ShardingDimension::Replicated], meaning the full
    /// tensor is present on every device. Equivalent to `NamedSharding(mesh, PartitionSpec())`
    /// in JAX, padded to the tensor rank with `None`.
    pub fn replicated(mesh: LogicalMesh, rank: usize) -> Self {
        Self {
            mesh,
            dimensions: vec![ShardingDimension::Replicated; rank],
            unreduced_axes: BTreeSet::new(),
            reduced_manual_axes: BTreeSet::new(),
            varying_manual_axes: BTreeSet::new(),
        }
    }

    /// Rank represented by this sharding.
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Renders a visualization of this sharding over a concrete device mesh.
    ///
    /// This ports the core behavior of JAX's
    /// [`jax.debug.visualize_array_sharding`](https://github.com/jax-ml/jax/blob/main/jax/_src/debugging.py):
    /// it materializes the per-device shard layout for a rank-1 or rank-2 array, groups devices
    /// that own the same logical slice, and renders the resulting chunks as uniform cells.
    ///
    /// Unlike JAX's implementation, this method returns a string instead of printing directly,
    /// making it suitable for logs, tests, and snapshots. When `colored` is `true`, the returned
    /// string contains ANSI color escape sequences that approximate JAX's colorized `rich` table.
    ///
    /// # Parameters
    ///
    ///   - `global_shape`: Global logical array shape to visualize. Only rank-1 and rank-2
    ///     shapes are supported.
    ///   - `mesh`: Concrete device mesh whose device IDs should populate the rendered chunks.
    ///   - `colored`: Whether to return ANSI colorized cell backgrounds in a rich-table-like
    ///     rendering. When `false`, the output is stable plain text without escape sequences.
    pub fn visualize(&self, global_shape: &[usize], mesh: &DeviceMesh, colored: bool) -> Result<String, ShardingError> {
        if !matches!(global_shape.len(), 1 | 2) {
            return Err(ShardingError::UnsupportedVisualizationRank { rank: global_shape.len() });
        }

        let layout = ShardingLayout::new(global_shape.to_vec(), mesh.clone(), self.clone())?;
        let row_segments = collect_visualization_segments(
            layout.shards().iter().map(
                |shard| {
                    if global_shape.len() == 2 { slice_bounds(&shard.slices()[0]) } else { (0, 1) }
                },
            ),
        );
        let column_segments = collect_visualization_segments(layout.shards().iter().map(|shard| {
            if global_shape.len() == 2 { slice_bounds(&shard.slices()[1]) } else { slice_bounds(&shard.slices()[0]) }
        }));
        let row_indices = row_segments
            .iter()
            .copied()
            .enumerate()
            .map(|(row_index, bounds)| (bounds, row_index))
            .collect::<HashMap<_, _>>();
        let column_indices = column_segments
            .iter()
            .copied()
            .enumerate()
            .map(|(column_index, bounds)| (bounds, column_index))
            .collect::<HashMap<_, _>>();

        let mut devices_by_cell = HashMap::<(usize, usize), Vec<MeshDeviceId>>::new();
        for shard in layout.shards() {
            let row_bounds = if global_shape.len() == 2 { slice_bounds(&shard.slices()[0]) } else { (0, 1) };
            let column_bounds = if global_shape.len() == 2 {
                slice_bounds(&shard.slices()[1])
            } else {
                slice_bounds(&shard.slices()[0])
            };
            let row_index = row_indices
                .get(&row_bounds)
                .copied()
                .expect("visualization row bounds should have been collected from shard slices");
            let column_index = column_indices
                .get(&column_bounds)
                .copied()
                .expect("visualization column bounds should have been collected from shard slices");
            devices_by_cell.entry((row_index, column_index)).or_default().push(shard.device().id);
        }

        let row_count = row_segments.len();
        let column_count = column_segments.len();
        let mut cells = vec![vec![VisualizationCell { label: String::new(), style: None }; column_count]; row_count];
        let mut cell_width = VISUALIZATION_MIN_CELL_WIDTH;
        let mut cell_styles =
            if colored { Some(make_visualization_styles(row_count, column_count).into_iter()) } else { None };
        for row_index in 0..row_count {
            for column_index in 0..column_count {
                let label = devices_by_cell
                    .get_mut(&(row_index, column_index))
                    .map(|device_ids| {
                        device_ids.sort_unstable();
                        device_ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
                    })
                    .unwrap_or_default();
                cell_width = cell_width.max(label.chars().count() + 2);
                cells[row_index][column_index] =
                    VisualizationCell { label, style: cell_styles.as_mut().and_then(Iterator::next) };
            }
        }
        let cell_height =
            if global_shape.len() == 1 { VISUALIZATION_1D_CELL_HEIGHT } else { VISUALIZATION_2D_CELL_HEIGHT };
        Ok(render_visualization(cells.as_slice(), cell_width, cell_height, colored))
    }

    /// Returns the visible mesh axes that are implicitly replicated by this sharding.
    pub fn replicated_axes(&self) -> Vec<&str> {
        let mut used_axes = HashSet::new();
        for partition_dimension in &self.dimensions {
            if let ShardingDimension::Sharded(axis_names) = partition_dimension {
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
        let reduced_manual_axes = self
            .reduced_manual_axes
            .iter()
            .filter(|axis_name| matches!(self.mesh.axis_type(axis_name), Some(MeshAxisType::Manual)))
            .cloned()
            .collect();
        Self {
            mesh: self.mesh.clone(),
            dimensions,
            unreduced_axes,
            reduced_manual_axes,
            varying_manual_axes: self.varying_manual_axes.clone(),
        }
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
    #[cfg(feature = "xla")]
    pub fn to_shardy_tensor_sharding_attribute(&self) -> String {
        let context = MlirContext::new();
        self.to_shardy_tensor_sharding(&context).to_string()
    }

    /// Builds this sharding as typed Shardy dimension shardings.
    #[cfg(feature = "xla")]
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
    #[cfg(feature = "xla")]
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

impl Display for Sharding {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        fn write_axis_set(formatter: &mut Formatter<'_>, axis_names: &BTreeSet<String>) -> std::fmt::Result {
            write!(formatter, "{{")?;
            for (axis_index, axis_name) in axis_names.iter().enumerate() {
                if axis_index > 0 {
                    write!(formatter, ", ")?;
                }
                write!(formatter, "'{}'", axis_name.replace('\'', "\\'"))?;
            }
            write!(formatter, "}}")
        }

        write!(formatter, "{{mesh<[")?;
        for (axis_index, axis) in self.mesh.axes.iter().enumerate() {
            if axis_index > 0 {
                write!(formatter, ", ")?;
            }
            write!(formatter, "'{}'", axis.name.replace('\'', "\\'"))?;
            write!(formatter, "={}", axis.size)?;
        }
        write!(formatter, "]>, [")?;
        for (dimension_index, dimension) in self.dimensions.iter().enumerate() {
            if dimension_index > 0 {
                write!(formatter, ", ")?;
            }
            write!(formatter, "{dimension}")?;
        }
        write!(formatter, "]")?;
        let replicated_axes = self.replicated_axes();
        if !replicated_axes.is_empty() {
            write!(formatter, ", replicated={{")?;
            for (axis_index, axis_name) in replicated_axes.iter().enumerate() {
                if axis_index > 0 {
                    write!(formatter, ", ")?;
                }
                write!(formatter, "'{}'", axis_name.replace('\'', "\\'"))?;
            }
            write!(formatter, "}}")?;
        }
        if !self.unreduced_axes.is_empty() {
            write!(formatter, ", unreduced=")?;
            write_axis_set(formatter, &self.unreduced_axes)?;
        }
        if !self.reduced_manual_axes.is_empty() {
            write!(formatter, ", reduced=")?;
            write_axis_set(formatter, &self.reduced_manual_axes)?;
        }
        if !self.varying_manual_axes.is_empty() {
            write!(formatter, ", varying=")?;
            write_axis_set(formatter, &self.varying_manual_axes)?;
        }
        write!(formatter, "}}")
    }
}

const VISUALIZATION_MIN_CELL_WIDTH: usize = 5;
const VISUALIZATION_1D_CELL_HEIGHT: usize = 1;
const VISUALIZATION_2D_CELL_HEIGHT: usize = 3;
const VISUALIZATION_COLOR_PALETTE: &[(u8, u8, u8)] = &[
    (57, 59, 121),
    (82, 84, 163),
    (107, 110, 207),
    (156, 158, 222),
    (99, 121, 57),
    (140, 162, 82),
    (181, 207, 107),
    (206, 219, 156),
    (140, 109, 49),
    (189, 158, 57),
    (231, 186, 82),
    (231, 203, 148),
    (132, 60, 57),
    (173, 73, 74),
    (214, 97, 107),
    (231, 150, 156),
    (123, 65, 115),
    (165, 81, 148),
    (206, 109, 189),
    (222, 158, 214),
];

type VisualizationBounds = (usize, usize);

#[derive(Clone, Debug, PartialEq, Eq)]
struct VisualizationCell {
    label: String,
    style: Option<VisualizationStyle>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct VisualizationStyle {
    foreground: (u8, u8, u8),
    background: (u8, u8, u8),
}

impl VisualizationStyle {
    fn render(&self, text: &str) -> String {
        let (foreground_red, foreground_green, foreground_blue) = self.foreground;
        let (background_red, background_green, background_blue) = self.background;
        format!(
            "\u{1b}[38;2;{foreground_red};{foreground_green};{foreground_blue}m\
\u{1b}[48;2;{background_red};{background_green};{background_blue}m{text}\u{1b}[0m"
        )
    }
}

fn slice_bounds(slice: &ShardSlice) -> VisualizationBounds {
    (slice.start, slice.end)
}

fn collect_visualization_segments<I: IntoIterator<Item = VisualizationBounds>>(bounds: I) -> Vec<VisualizationBounds> {
    let mut segments = bounds.into_iter().collect::<Vec<_>>();
    segments.sort_unstable();
    segments.dedup();
    segments
}

fn make_visualization_styles(row_count: usize, column_count: usize) -> Vec<VisualizationStyle> {
    let cell_count = row_count * column_count;
    if cell_count == 0 {
        return Vec::new();
    }

    assign_visualization_palette_indices(row_count, column_count)
        .into_iter()
        .map(|palette_index| {
            let background = VISUALIZATION_COLOR_PALETTE[palette_index];
            VisualizationStyle { foreground: contrasting_text_color(background), background }
        })
        .collect()
}

fn assign_visualization_palette_indices(row_count: usize, column_count: usize) -> Vec<usize> {
    let cell_count = row_count * column_count;
    let palette_count = VISUALIZATION_COLOR_PALETTE.len();
    let unique_prefix_length = cell_count.min(palette_count);
    let mut palette_indices = (0..unique_prefix_length).collect::<Vec<_>>();
    let mut next_palette_index = 0usize;

    for cell_index in unique_prefix_length..cell_count {
        let row_index = cell_index / column_count;
        let column_index = cell_index % column_count;
        let left_neighbor = (column_index > 0).then_some(palette_indices[cell_index - 1]);
        let upper_neighbor = (row_index > 0).then_some(palette_indices[cell_index - column_count]);

        let mut assigned_palette_index = None;
        for offset in 0..palette_count {
            let candidate_palette_index = (next_palette_index + offset) % palette_count;
            if Some(candidate_palette_index) != left_neighbor && Some(candidate_palette_index) != upper_neighbor {
                assigned_palette_index = Some(candidate_palette_index);
                next_palette_index = (candidate_palette_index + 1) % palette_count;
                break;
            }
        }
        palette_indices.push(
            assigned_palette_index
                .expect("the visualization palette should be large enough to avoid orthogonal collisions in a grid"),
        );
    }
    palette_indices
}

fn contrasting_text_color(background: (u8, u8, u8)) -> (u8, u8, u8) {
    let (red, green, blue) = background;
    let luminance = f32::from(red) * 0.299 + f32::from(green) * 0.587 + f32::from(blue) * 0.114;
    if luminance > 186.0 { (0, 0, 0) } else { (255, 255, 255) }
}

fn render_visualization(
    cells: &[Vec<VisualizationCell>],
    cell_width: usize,
    cell_height: usize,
    colored: bool,
) -> String {
    if colored {
        return render_colored_visualization(cells, cell_width, cell_height);
    }

    render_plain_visualization(cells, cell_width, cell_height)
}

fn render_plain_visualization(cells: &[Vec<VisualizationCell>], cell_width: usize, cell_height: usize) -> String {
    let top_border = render_horizontal_border('┌', '┬', '┐', cells[0].len(), cell_width);
    let middle_border = render_horizontal_border('├', '┼', '┤', cells[0].len(), cell_width);
    let bottom_border = render_horizontal_border('└', '┴', '┘', cells[0].len(), cell_width);
    let mut lines = Vec::new();
    lines.push(top_border);

    for (row_index, row_cells) in cells.iter().enumerate() {
        let label_line = cell_height / 2;
        for line_index in 0..cell_height {
            let mut line = String::from("│");
            for cell in row_cells {
                let contents = if line_index == label_line {
                    center_text(cell.label.as_str(), cell_width)
                } else {
                    " ".repeat(cell_width)
                };
                line.push_str(contents.as_str());
                line.push('│');
            }
            lines.push(line);
        }
        if row_index + 1 == cells.len() {
            lines.push(bottom_border.clone());
        } else {
            lines.push(middle_border.clone());
        }
    }

    lines.join("\n")
}

fn render_colored_visualization(cells: &[Vec<VisualizationCell>], cell_width: usize, cell_height: usize) -> String {
    let label_line = cell_height / 2;
    let mut lines = Vec::new();
    for row_cells in cells {
        for line_index in 0..cell_height {
            let mut line = String::new();
            for cell in row_cells {
                let contents = if line_index == label_line {
                    center_text(cell.label.as_str(), cell_width)
                } else {
                    " ".repeat(cell_width)
                };
                if let Some(style) = cell.style {
                    line.push_str(style.render(contents.as_str()).as_str());
                } else {
                    line.push_str(contents.as_str());
                }
            }
            lines.push(line);
        }
    }

    lines.join("\n")
}

fn render_horizontal_border(
    left_corner: char,
    intersection: char,
    right_corner: char,
    column_count: usize,
    cell_width: usize,
) -> String {
    let mut line = String::new();
    line.push(left_corner);
    if column_count > 0 {
        line.push_str("─".repeat(cell_width).as_str());
        for _ in 1..column_count {
            line.push(intersection);
            line.push_str("─".repeat(cell_width).as_str());
        }
    }
    line.push(right_corner);
    line
}

fn center_text(text: &str, width: usize) -> String {
    let text_width = text.chars().count();
    if text_width >= width {
        return text.chars().take(width).collect();
    }

    let left_padding = (width - text_width) / 2;
    let right_padding = width - text_width - left_padding;
    format!("{}{}{}", " ".repeat(left_padding), text, " ".repeat(right_padding))
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
        if mesh.logical_mesh != sharding.mesh.clone() {
            return Err(ShardingError::MeshMismatch {
                expected: mesh.logical_mesh.clone(),
                actual: sharding.mesh.clone(),
            });
        }

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use indoc::indoc;
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

    fn test_device_mesh_1x2() -> DeviceMesh {
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let devices = vec![MeshDevice::new(0, 0), MeshDevice::new(1, 0)];
        DeviceMesh::new(logical_mesh, devices).unwrap()
    }

    fn strip_ansi_codes(value: &str) -> String {
        let mut stripped = String::new();
        let mut characters = value.chars();
        while let Some(character) = characters.next() {
            if character == '\u{1b}' {
                if characters.next() == Some('[') {
                    for character in characters.by_ref() {
                        if character == 'm' {
                            break;
                        }
                    }
                }
            } else {
                stripped.push(character);
            }
        }
        stripped
    }

    fn empty_axes() -> Vec<&'static str> {
        Vec::new()
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
            ["reduction", "carry"],
            empty_axes(),
            empty_axes(),
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
                ["carry"],
                empty_axes(),
                empty_axes(),
            )
            .unwrap()
        );
    }

    #[test]
    fn test_sharding_validation() {
        let mesh = test_logical_mesh_2x2();

        assert!(matches!(
            Sharding::new(
                mesh.clone(),
                vec![ShardingDimension::sharded(["z"])],
                empty_axes(),
                empty_axes(),
                empty_axes(),
            ),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));

        assert!(matches!(
            Sharding::new(
                mesh.clone(),
                vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["x"])],
                empty_axes(),
                empty_axes(),
                empty_axes(),
            ),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));

        assert!(matches!(
            Sharding::new(
                mesh,
                vec![ShardingDimension::Sharded(Vec::new())],
                empty_axes(),
                empty_axes(),
                empty_axes(),
            ),
            Err(ShardingError::EmptySharding { dimension }) if dimension == 0,
        ));
    }

    #[test]
    fn test_sharding_shardy_rendering() {
        let mesh = test_logical_mesh_2x2();
        let sharding = Sharding::new(
            mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
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
        let sharding = Sharding::new(
            mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
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
            ["y"],
            empty_axes(),
            empty_axes(),
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
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])], ["z"], empty_axes(), empty_axes()).unwrap();

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
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x\"y"])], ["z\"w"], empty_axes(), empty_axes())
                .unwrap();

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
        let sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x", "y", "z"])],
            ["w"],
            empty_axes(),
            empty_axes(),
        )
        .unwrap();
        let projected = sharding.project_for_traced_sharding();

        assert_eq!(
            projected,
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x", "z"])], empty_axes(), empty_axes(), empty_axes())
                .unwrap()
        );
        assert!(projected.replicated_axes().is_empty());
        assert!(projected.unreduced_axes.is_empty());
    }

    #[test]
    fn test_sharding_unreduced_axis_validation() {
        let mesh = test_logical_mesh_2x2();
        let sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();

        assert!(matches!(
            Sharding::new(mesh.clone(), sharding.dimensions.clone(), ["z"], empty_axes(), empty_axes()),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "z",
        ));

        assert!(matches!(
            Sharding::new(mesh, sharding.dimensions, ["x"], empty_axes(), empty_axes()),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "x",
        ));
    }

    #[test]
    fn test_sharding_reduced_axis_validation() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();

        assert!(matches!(
            Sharding::new(
                mesh.clone(),
                vec![ShardingDimension::replicated()],
                empty_axes(),
                ["y"],
                empty_axes(),
            ),
            Err(ShardingError::ExpectedManualMeshAxis { name }) if name == "y",
        ));

        assert!(matches!(
            Sharding::new(
                mesh,
                vec![ShardingDimension::replicated()],
                ["z"],
                ["z"],
                empty_axes(),
            ),
            Err(ShardingError::DuplicateMeshAxisName { name }) if name == "z",
        ));
    }

    #[test]
    fn test_sharding_varying_manual_axes_validation() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();

        assert_eq!(
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()], empty_axes(), empty_axes(), ["z"]),
            Ok(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()], empty_axes(), empty_axes(), ["z"])
                .unwrap())
        );

        assert_eq!(
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()], empty_axes(), empty_axes(), ["x", "x"]),
            Ok(Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()], empty_axes(), empty_axes(), ["x"])
                .unwrap())
        );

        assert!(matches!(
            Sharding::new(mesh, vec![ShardingDimension::replicated()], empty_axes(), empty_axes(), ["unknown"]),
            Err(ShardingError::UnknownMeshAxisName { name }) if name == "unknown",
        ));
    }

    #[test]
    fn test_sharding_project_for_traced_sharding_filters_reduced_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding {
            mesh: mesh.clone(),
            dimensions: vec![ShardingDimension::replicated()],
            unreduced_axes: BTreeSet::new(),
            reduced_manual_axes: BTreeSet::from(["x".to_string(), "y".to_string(), "z".to_string()]),
            varying_manual_axes: BTreeSet::from(["y".to_string()]),
        };

        assert_eq!(
            sharding.project_for_traced_sharding(),
            Sharding::new(mesh, vec![ShardingDimension::replicated()], empty_axes(), ["x", "z"], ["y"],).unwrap()
        );
    }

    #[test]
    fn test_sharding_preserves_varying_manual_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();

        assert_eq!(
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()], empty_axes(), empty_axes(), ["x"])
                .map(|sharding| sharding.varying_manual_axes),
            Ok(BTreeSet::from(["x".to_string()]))
        );
    }

    #[test]
    fn test_sharding_display() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("carry", 8, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(
            mesh,
            vec![
                ShardingDimension::sharded(["data"]),
                ShardingDimension::replicated(),
                ShardingDimension::unconstrained(),
            ],
            ["carry"],
            ["manual"],
            ["manual"],
        )
        .unwrap();

        assert_eq!(
            sharding.to_string(),
            "{mesh<['data'=4, 'manual'=2, 'carry'=8]>, [{'data'}, {}, {?}], unreduced={'carry'}, reduced={'manual'}, \
             varying={'manual'}}"
        );
    }

    #[test]
    fn test_sharding_replicated_axes_ignore_reduced_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::replicated()], empty_axes(), ["x"], empty_axes()).unwrap();

        assert_eq!(sharding.replicated_axes(), vec!["y"]);
    }

    #[test]
    fn test_sharding_visualize_groups_replicated_devices() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();

        assert_eq!(
            sharding.visualize(&[8], &mesh, false),
            Ok(indoc! {"
                ┌─────┬─────┐
                │ 0,1 │ 2,3 │
                └─────┴─────┘
            "}
            .trim_end()
            .to_string())
        );
    }

    #[test]
    fn test_sharding_visualize_uneven_1d_partitioning() {
        let mesh = test_device_mesh_1x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();

        assert_eq!(
            sharding.visualize(&[5], &mesh, false),
            Ok(indoc! {"
                ┌─────┬─────┐
                │  0  │  1  │
                └─────┴─────┘
            "}
            .trim_end()
            .to_string())
        );
    }

    #[test]
    fn test_sharding_visualize_2d_partitioning() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();

        assert_eq!(
            sharding.visualize(&[8, 6], &mesh, false),
            Ok(indoc! {"
                ┌─────┬─────┐
                │     │     │
                │  0  │  1  │
                │     │     │
                ├─────┼─────┤
                │     │     │
                │  2  │  3  │
                │     │     │
                └─────┴─────┘
            "}
            .trim_end()
            .to_string())
        );
    }

    #[test]
    fn test_sharding_visualize_colorizes_cells() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();

        let colored = sharding.visualize(&[8], &mesh, true).unwrap();

        assert!(colored.contains("\u{1b}[38;2;"));
        assert!(colored.contains("\u{1b}[48;2;"));
        assert!(!colored.contains('┌'));
        assert!(!colored.contains('│'));
        assert!(!colored.contains('└'));
        assert_eq!(strip_ansi_codes(colored.as_str()), " 0,1  2,3 ".to_string());
    }

    #[test]
    fn test_visualization_palette_uses_unique_prefix_and_avoids_neighbor_collisions() {
        let row_count = 5;
        let column_count = 5;
        let styles = make_visualization_styles(row_count, column_count);
        let backgrounds = styles.iter().map(|style| style.background).collect::<Vec<_>>();
        let unique_prefix = backgrounds.iter().take(VISUALIZATION_COLOR_PALETTE.len()).copied().collect::<HashSet<_>>();

        assert_eq!(unique_prefix.len(), VISUALIZATION_COLOR_PALETTE.len());
        for row_index in 0..row_count {
            for column_index in 0..column_count {
                let cell_index = row_index * column_count + column_index;
                if column_index + 1 < column_count {
                    assert_ne!(backgrounds[cell_index], backgrounds[cell_index + 1]);
                }
                if row_index + 1 < row_count {
                    assert_ne!(backgrounds[cell_index], backgrounds[cell_index + column_count]);
                }
            }
        }
    }

    #[test]
    fn test_sharding_visualize_rejects_unsupported_rank() {
        let mesh = test_device_mesh_1x2();
        let sharding = Sharding::replicated(mesh.logical_mesh.clone(), 3);

        assert_eq!(
            sharding.visualize(&[2, 3, 4], &mesh, false),
            Err(ShardingError::UnsupportedVisualizationRank { rank: 3 })
        );
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
            empty_axes(),
            empty_axes(),
            empty_axes(),
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
            empty_axes(),
            empty_axes(),
            empty_axes(),
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
            empty_axes(),
            empty_axes(),
            empty_axes(),
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
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();
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
            empty_axes(),
            empty_axes(),
            empty_axes(),
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
            empty_axes(),
            empty_axes(),
            empty_axes(),
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
            empty_axes(),
            empty_axes(),
            empty_axes(),
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
        let sharding = Sharding::new(
            actual.clone(),
            vec![ShardingDimension::sharded(["z"])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();

        assert_eq!(
            ShardingLayout::new(vec![8], mesh.clone(), sharding),
            Err(ShardingError::MeshMismatch { expected: mesh.logical_mesh, actual })
        );
    }
}
