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
//! | [`Shard`]              | `jax.Shard` from `array.global_shards` | runtime metadata only                      |
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
//! runtime for computing per-device [`Shard`] metadata.
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
//! 3. **Use the sharding metadata at runtime** when pairing a logical array type with a concrete
//!    [`DeviceMesh`]. Runtime XLA arrays cache one [`Shard`] per mesh device so they can
//!    validate local PJRT buffers, identify addressable shards, and determine the per-device slices
//!    implied by the global sharding.
//!
//! 4. **Convert to Shardy MLIR attributes** for StableHLO program annotation:
//!
//!    ```ignore
//!    // Generates: sdy.mesh @mesh = <["data"=4, "model"=2]>
//!    let context = MlirContext::new();
//!    let mesh_module = context.module(context.unknown_location());
//!    let mesh_op = mesh.logical_mesh.to_shardy(context.unknown_location());
//!    let mesh_op = mesh_module.body().append_operation(mesh_op);
//!
//!    // Generates: #sdy.sharding<@mesh, [{"data"}, {}]>
//!    let attr = sharding.to_shardy(context.unknown_location());
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
//! Runtime shard-metadata computation covers *all* global shards - including those on remote
//! hosts - so that the full sharding picture is available for compilation and execution. Only
//! addressable shards can be backed by actual PJRT buffers; non-addressable shard descriptors
//! exist only as metadata describing remote device placements.
//!
//! This mirrors JAX's `array.addressable_shards` (local) vs `array.global_shards` (all),
//! where accessing `.data` on a non-addressable shard raises an error.

use std::collections::HashMap;
use std::ops::Range;

#[cfg(test)]
use ryft_mlir::Context as MlirContext;

use crate::sharding::{DeviceMesh, MeshDevice, MeshDeviceId, Sharding, ShardingDimension, ShardingError};
use crate::utilities::colors::Color;

#[cfg(test)]
use crate::sharding::{LogicalMesh, MeshAxisType};

impl Sharding {
    /// Builds a visualization of this sharding over the [`LogicalMesh`] stored in [`Self::mesh`].
    ///
    /// This ports the core behavior of JAX's
    /// [`jax.debug.visualize_array_sharding`](https://github.com/jax-ml/jax/blob/main/jax/_src/debugging.py):
    /// it groups devices that own the same logical partition and arranges them into a rank-1
    /// (single row) or rank-2 (row x column) grid. Devices are labeled with sequential indices
    /// `0..device_count` based on their row-major position in the logical mesh.
    ///
    /// The returned [`ShardingVisualization`] can be rendered to a string via
    /// [`ShardingVisualization::render`].
    pub fn visualize(&self) -> Result<ShardingVisualization, ShardingError> {
        let rank = self.rank();
        if !matches!(rank, 1 | 2) {
            return Err(ShardingError::UnsupportedVisualizationRank { rank });
        }

        let axis_sizes = self.mesh.axes.iter().map(|axis| axis.size).collect::<Vec<_>>();
        let device_count = self.mesh.device_count();

        // Compute per-device partition coordinates and group devices into grid cells.
        let mut devices_by_cell = HashMap::<(usize, usize), Vec<usize>>::new();
        for device_index in 0..device_count {
            // Decompose the linear device index into row-major mesh coordinates.
            let mut remaining = device_index;
            let mut mesh_coordinate = vec![0usize; axis_sizes.len()];
            for (axis_index, axis_size) in axis_sizes.iter().enumerate().rev() {
                mesh_coordinate[axis_index] = remaining % axis_size;
                remaining /= axis_size;
            }
            let cell = self.visualization_cell(&mesh_coordinate);
            devices_by_cell.entry(cell).or_default().push(device_index);
        }

        // Collect the distinct row and column partition indices.
        let row_count = devices_by_cell.keys().map(|(row, _)| *row).max().map_or(0, |max| max + 1);
        let column_count = devices_by_cell.keys().map(|(_, column)| *column).max().map_or(0, |max| max + 1);

        // Build the visualization cell grid.
        let mut cells = vec![vec![String::new(); column_count]; row_count];
        let mut cell_width = VISUALIZATION_MIN_CELL_WIDTH;
        for row_index in 0..row_count {
            for column_index in 0..column_count {
                let label = devices_by_cell
                    .get_mut(&(row_index, column_index))
                    .map(|device_indices| {
                        device_indices.sort_unstable();
                        device_indices.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
                    })
                    .unwrap_or_default();
                cell_width = cell_width.max(label.chars().count() + 2);
                cells[row_index][column_index] = label;
            }
        }
        let cell_height = if rank == 1 { VISUALIZATION_1D_CELL_HEIGHT } else { VISUALIZATION_2D_CELL_HEIGHT };
        Ok(ShardingVisualization { cells, cell_width, cell_height })
    }

    /// Returns the `(row, column)` grid cell for a device at the given mesh coordinate.
    ///
    /// For rank-1 shardings the row is always `0` and the column is the partition index for
    /// dimension 0. For rank-2 shardings row and column map to the partition indices of the
    /// first and second dimensions respectively.
    fn visualization_cell(&self, mesh_coordinate: &[usize]) -> (usize, usize) {
        if self.dimensions.len() == 1 {
            (0, self.dimension_partition_index(0, mesh_coordinate))
        } else {
            (self.dimension_partition_index(0, mesh_coordinate), self.dimension_partition_index(1, mesh_coordinate))
        }
    }

    /// Returns the partition index of a single sharding dimension for a device at the given mesh
    /// coordinate. Replicated and unconstrained dimensions always return `0` (single partition).
    fn dimension_partition_index(&self, dimension: usize, mesh_coordinate: &[usize]) -> usize {
        match &self.dimensions[dimension] {
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => 0,
            ShardingDimension::Sharded(axis_names) => {
                let mut partition_index = 0usize;
                for axis_name in axis_names {
                    let axis_index = self
                        .mesh
                        .axis_indices
                        .get(axis_name.as_str())
                        .copied()
                        .expect("sharding mesh axes should be validated at construction");
                    let axis_size = self.mesh.axes[axis_index].size;
                    partition_index = partition_index * axis_size + mesh_coordinate[axis_index];
                }
                partition_index
            }
        }
    }
}

/// Grid-based visualization of a [`Sharding`] produced by [`Sharding::visualize`].
///
/// Each cell in the grid holds a label listing the device indices that share the corresponding
/// partition. Call [`Self::render`] to produce a displayable string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardingVisualization {
    /// Row-major grid of device-index labels (e.g., `"0,1"`).
    cells: Vec<Vec<String>>,

    /// Character width of every cell in the rendered output.
    cell_width: usize,

    /// Line height of every cell in the rendered output.
    cell_height: usize,
}

impl ShardingVisualization {
    /// Renders this visualization to a string.
    ///
    /// When `colored` is `true` the output contains ANSI 24-bit color escape sequences that
    /// approximate JAX's colorized `rich` table. When `false` the output is stable plain text
    /// with box-drawing borders and no escape sequences.
    pub fn render(&self, colored: bool) -> String {
        if colored {
            render_colored_visualization(&self.cells, self.cell_width, self.cell_height)
        } else {
            render_plain_visualization(&self.cells, self.cell_width, self.cell_height)
        }
    }
}

/// Minimum width in characters for each cell in the rendered visualization grid. Cells expand
/// beyond this minimum when device labels (e.g., `"0,1,2"`) plus padding exceed it.
const VISUALIZATION_MIN_CELL_WIDTH: usize = 5;

/// Height in lines of each cell when visualizing a rank-1 sharding (single-row grid).
const VISUALIZATION_1D_CELL_HEIGHT: usize = 1;

/// Height in lines of each cell when visualizing a rank-2 sharding (row x column grid). The extra
/// height gives the label a blank line above and below for readability.
const VISUALIZATION_2D_CELL_HEIGHT: usize = 3;

/// RGB color palette used for ANSI-colored visualization output. Colors are assigned to grid cells
/// via a greedy graph-coloring scheme that avoids giving the same color to horizontally or
/// vertically adjacent cells. The palette is adapted from the Tableau 20 categorical color scheme.
const VISUALIZATION_COLOR_PALETTE: &[Color] = &[
    Color::new(57, 59, 121),
    Color::new(82, 84, 163),
    Color::new(107, 110, 207),
    Color::new(156, 158, 222),
    Color::new(99, 121, 57),
    Color::new(140, 162, 82),
    Color::new(181, 207, 107),
    Color::new(206, 219, 156),
    Color::new(140, 109, 49),
    Color::new(189, 158, 57),
    Color::new(231, 186, 82),
    Color::new(231, 203, 148),
    Color::new(132, 60, 57),
    Color::new(173, 73, 74),
    Color::new(214, 97, 107),
    Color::new(231, 150, 156),
    Color::new(123, 65, 115),
    Color::new(165, 81, 148),
    Color::new(206, 109, 189),
    Color::new(222, 158, 214),
];

fn render_plain_visualization(cells: &[Vec<String>], cell_width: usize, cell_height: usize) -> String {
    let top_border = render_horizontal_border('┌', '┬', '┐', cells[0].len(), cell_width);
    let middle_border = render_horizontal_border('├', '┼', '┤', cells[0].len(), cell_width);
    let bottom_border = render_horizontal_border('└', '┴', '┘', cells[0].len(), cell_width);
    let mut lines = Vec::new();
    lines.push(top_border);

    for (row_index, row_cells) in cells.iter().enumerate() {
        let label_line = cell_height / 2;
        for line_index in 0..cell_height {
            let mut line = String::from("│");
            for label in row_cells {
                let contents =
                    if line_index == label_line { center_text(label.as_str(), cell_width) } else { " ".repeat(cell_width) };
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

fn render_colored_visualization(cells: &[Vec<String>], cell_width: usize, cell_height: usize) -> String {
    let row_count = cells.len();
    let column_count = cells.first().map_or(0, Vec::len);
    let background_colors = assign_visualization_background_colors(row_count, column_count);
    let label_line = cell_height / 2;
    let mut lines = Vec::new();
    for (row_index, row_cells) in cells.iter().enumerate() {
        for line_index in 0..cell_height {
            let mut line = String::new();
            for (column_index, label) in row_cells.iter().enumerate() {
                let contents =
                    if line_index == label_line { center_text(label.as_str(), cell_width) } else { " ".repeat(cell_width) };
                let background = background_colors[row_index * column_count + column_index];
                line.push_str(Color::colored_text(contents.as_str(), background.foreground_color(), background).as_str());
            }
            lines.push(line);
        }
    }

    lines.join("\n")
}

/// Assigns one background [`Color`] per grid cell using a greedy graph-coloring approach that
/// avoids giving the same color to horizontally or vertically adjacent cells.
fn assign_visualization_background_colors(row_count: usize, column_count: usize) -> Vec<Color> {
    let cell_count = row_count * column_count;
    if cell_count == 0 {
        return Vec::new();
    }

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

    palette_indices.into_iter().map(|index| VISUALIZATION_COLOR_PALETTE[index]).collect()
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
/// | JAX `Shard` field           | `Shard` method                        |
/// | --------------------------- | ------------------------------------- |
/// | `shard.device`              | [`device()`][Shard::device]           |
/// | `shard.index` (slice tuple) | [`slices()`][Shard::slices]           |
/// | `shard.data.shape`          | [`shape()`][Shard::shape]             |
/// | `shard.replica_id`          | derivable from mesh coordinate        |
///
/// [jax-global-shards]: https://docs.jax.dev/en/latest/jax.html#jax.Array.global_shards
///
/// Unlike JAX's `Shard.data`, which provides access to the actual tensor data (only on
/// addressable shards), a `Shard` never holds buffer data. Buffer data is stored
/// alongside descriptors in [`ArrayShard`][super::arrays::ArrayShard] for local shards backed
/// by PJRT buffers, and remains inaccessible for remote shards.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shard {
    shard_index: usize,
    device: MeshDevice,
    mesh_coordinate: Vec<usize>,
    slices: Vec<ShardSlice>,
    shape: Vec<usize>,
}

impl Shard {
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
// Shard-metadata helpers
// ---------------------------------------------------------------------------

/// Computes one [`Shard`] per mesh device for the provided global shape and [`Sharding`].
pub(crate) fn compute_shard_descriptors(
    global_shape: &[usize],
    mesh: &DeviceMesh,
    sharding: &Sharding,
) -> Result<(Vec<Shard>, HashMap<MeshDeviceId, usize>), ShardingError> {
    if mesh.logical_mesh != sharding.mesh {
        return Err(ShardingError::MeshMismatch { expected: mesh.logical_mesh.clone(), actual: sharding.mesh.clone() });
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
        shards.push(Shard { shard_index, device: mesh_device, mesh_coordinate, slices, shape });
    }

    Ok((shards, shard_index_by_device))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

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

    #[cfg(feature = "xla")]
    fn to_shardy_string(sharding: &Sharding) -> String {
        let context = MlirContext::new();
        sharding.to_shardy(context.unknown_location()).to_string()
    }

    // -----------------------------------------------------------------------
    // Sharding tests
    // -----------------------------------------------------------------------

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
        assert_eq!(to_shardy_string(&sharding), "#sdy.sharding<@mesh, [{\"x\"}, {}]>");
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
        assert_eq!(to_shardy_string(&sharding), "#sdy.sharding<@mesh, [{\"x\"}, {}], replicated={\"y\"}>");
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
        assert_eq!(to_shardy_string(&sharding), "#sdy.sharding<@mesh, [{\"x\"}, {}], unreduced={\"y\"}>");
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
            to_shardy_string(&sharding),
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
            to_shardy_string(&sharding),
            r#"#sdy.sharding<@mesh, [{"x\22y"}], replicated={"path\\to"}, unreduced={"z\22w"}>"#
        );
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
    fn test_sharding_display() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("varying", 8, MeshAxisType::Manual).unwrap(),
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
            ["varying"],
        )
        .unwrap();

        assert_eq!(
            sharding.to_string(),
            "{mesh<['data'=4, 'manual'=2, 'varying'=8, 'carry'=8]>, [{'data'}, {}, {?}], unreduced={'carry'}, \
            reduced_manual={'manual'}, varying_manual={'varying'}}",
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
        let mesh = test_logical_mesh_2x2();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])], empty_axes(), empty_axes(), empty_axes())
                .unwrap();

        assert_eq!(
            sharding.visualize().unwrap().render(false),
            indoc! {"
                ┌─────┬─────┐
                │ 0,1 │ 2,3 │
                └─────┴─────┘
            "}
            .trim_end()
            .to_string()
        );
    }

    #[test]
    fn test_sharding_visualize_uneven_1d_partitioning() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])], empty_axes(), empty_axes(), empty_axes())
                .unwrap();

        assert_eq!(
            sharding.visualize().unwrap().render(false),
            indoc! {"
                ┌─────┬─────┐
                │  0  │  1  │
                └─────┴─────┘
            "}
            .trim_end()
            .to_string()
        );
    }

    #[test]
    fn test_sharding_visualize_2d_partitioning() {
        let mesh = test_logical_mesh_2x2();
        let sharding = Sharding::new(
            mesh,
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();

        assert_eq!(
            sharding.visualize().unwrap().render(false),
            indoc! {"
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
            .to_string()
        );
    }

    #[test]
    fn test_sharding_visualize_colorizes_cells() {
        let mesh = test_logical_mesh_2x2();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])], empty_axes(), empty_axes(), empty_axes())
                .unwrap();

        let colored = sharding.visualize().unwrap().render(true);

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
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 3);

        assert_eq!(sharding.visualize(), Err(ShardingError::UnsupportedVisualizationRank { rank: 3 }));
    }

    fn shard_for_device<'a>(
        shards: &'a [Shard],
        shard_index_by_device: &HashMap<MeshDeviceId, usize>,
        device_id: MeshDeviceId,
    ) -> &'a Shard {
        let shard_index =
            shard_index_by_device.get(&device_id).copied().expect("device should have a shard descriptor");
        &shards[shard_index]
    }

    fn shard_indices_for_process(shards: &[Shard], process_index: usize) -> Vec<usize> {
        shards
            .iter()
            .filter_map(|descriptor| {
                (descriptor.device().process_index == process_index).then_some(descriptor.shard_index())
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Shard metadata tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shard_metadata_rank_mismatch() {
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
            compute_shard_descriptors(&[8usize], &mesh, &sharding),
            Err(ShardingError::ShardingRankMismatch { sharding_rank: 2, array_rank: 1 }),
        ));
    }

    #[test]
    fn test_shard_metadata_unconstrained_is_ignored() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::unconstrained()],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();
        let (shards, shard_index_by_device) = compute_shard_descriptors(&[8, 6], &mesh, &sharding).unwrap();

        let shard0 = shard_for_device(&shards, &shard_index_by_device, 0);
        let shard3 = shard_for_device(&shards, &shard_index_by_device, 3);
        assert_eq!(shard0.slices()[0], 0..4);
        assert_eq!(shard0.slices()[1], 0..6);
        assert_eq!(shard3.slices()[0], 4..8);
        assert_eq!(shard3.slices()[1], 0..6);
        assert_eq!(shard0.shape(), &[4, 6]);
        assert_eq!(shard3.shape(), &[4, 6]);
    }

    #[test]
    fn test_shard_metadata_even_2d_partitioning() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();
        let (shards, shard_index_by_device) = compute_shard_descriptors(&[8, 6], &mesh, &sharding).unwrap();

        let shard0 = shard_for_device(&shards, &shard_index_by_device, 0);
        assert_eq!(shard0.shape(), &[4, 3]);
        assert_eq!(shard0.slices()[0], 0..4);
        assert_eq!(shard0.slices()[1], 0..3);

        let shard3 = shard_for_device(&shards, &shard_index_by_device, 3);
        assert_eq!(shard3.shape(), &[4, 3]);
        assert_eq!(shard3.slices()[0], 4..8);
        assert_eq!(shard3.slices()[1], 3..6);
    }

    #[test]
    fn test_shard_metadata_uneven_partitioning() {
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
        let (shards, shard_index_by_device) = compute_shard_descriptors(&[5], &mesh, &sharding).unwrap();

        let shard0 = shard_for_device(&shards, &shard_index_by_device, 0);
        assert_eq!(shard0.shape(), &[3]);
        assert_eq!(shard0.slices()[0], 0..3);

        let shard1 = shard_for_device(&shards, &shard_index_by_device, 1);
        assert_eq!(shard1.shape(), &[2]);
        assert_eq!(shard1.slices()[0], 3..5);
    }

    #[test]
    fn test_shard_metadata_multi_axis_single_dimension_partitioning() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x".to_string(), "y".to_string()])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();
        let (shards, shard_index_by_device) = compute_shard_descriptors(&[10], &mesh, &sharding).unwrap();

        assert_eq!(shard_for_device(&shards, &shard_index_by_device, 0).slices()[0], 0..3);
        assert_eq!(shard_for_device(&shards, &shard_index_by_device, 1).slices()[0], 3..6);
        assert_eq!(shard_for_device(&shards, &shard_index_by_device, 2).slices()[0], 6..8);
        assert_eq!(shard_for_device(&shards, &shard_index_by_device, 3).slices()[0], 8..10);
    }

    #[test]
    fn test_shard_metadata_process_filtering() {
        let mesh = test_device_mesh_2x2();
        let sharding = Sharding::new(
            mesh.logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
            empty_axes(),
            empty_axes(),
            empty_axes(),
        )
        .unwrap();
        let (shards, _) = compute_shard_descriptors(&[8, 6], &mesh, &sharding).unwrap();

        assert_eq!(shard_indices_for_process(&shards, 0), vec![0, 1]);
        assert_eq!(shard_indices_for_process(&shards, 1), vec![2, 3]);
        assert_eq!(shard_indices_for_process(&shards, 42), Vec::<usize>::new());
    }

    #[test]
    fn test_shard_metadata_mesh_mismatch_reports_expected_and_actual_meshes() {
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
            compute_shard_descriptors(&[8], &mesh, &sharding),
            Err(ShardingError::MeshMismatch { expected: mesh.logical_mesh, actual })
        );
    }
}
