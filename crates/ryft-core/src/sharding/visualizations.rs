use std::collections::HashMap;

use crate::utilities::colors::Color;

use super::{MeshDeviceId, Sharding, ShardingError};

/// Minimum width in characters for each cell in a rendered [`Sharding`] visualization grid. Cells expand beyond this
/// minimum when device labels (e.g., `"0,1,2"`) plus padding exceed it.
const VISUALIZATION_MIN_CELL_WIDTH: usize = 5;

/// Height in lines of each cell when visualizing a rank-1 [`Sharding`].
const VISUALIZATION_1D_CELL_HEIGHT: usize = 1;

/// Height in lines of each cell when visualizing a rank-2 [`Sharding`].
/// The extra height gives the label a blank line above and below for readability.
const VISUALIZATION_2D_CELL_HEIGHT: usize = 3;

/// [`Color`] palette used for ANSI-colored [`Sharding`] visualization renderings. Colors are assigned to grid cells
/// via a greedy graph-coloring scheme that avoids giving the same color to horizontally or vertically adjacent cells.
/// The palette matches Matplotlib's [`tab20b`](https://matplotlib.org/stable/gallery/color/colormap_reference.html)
/// qualitative colormap, which is the same palette that JAX uses for its sharding visualizations.
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

impl Sharding {
    /// Builds a [`ShardingVisualization`] of this sharding that can later be rendered to text using
    /// [`ShardingVisualization::render`]. This function groups devices that own the same logical partition and arranges
    /// them into a one-dimensional or two-dimensional grid (higher rank [`Sharding`]s cannot be visualized and will
    /// result in a [`ShardingError::UnsupportedVisualizationRank`] error instead). Devices are labeled with sequential
    /// indices (i.e., `0..device_count`) based on their row-major position in the [`LogicalMesh`].
    ///
    /// This function is heavily inspired by [JAX's `jax.debug.visualize_array_sharding`](
    /// https://docs.jax.dev/en/latest/_autosummary/jax.debug.visualize_array_sharding.html).
    ///
    /// # Examples
    ///
    /// Below are some example [`Sharding`]s along with their visualizations:
    ///
    /// ```
    /// # use indoc::indoc;
    /// # use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    ///
    /// // A rank-1 sharding over a `2×2` mesh with dimension `0` sharded along axis `"x"` produces a single-row
    /// // grid where devices sharing the same partition are grouped together (devices `0` and `1` share one `"x"`
    /// // coordinate, and devices `2` and `3` share the other):
    /// let mesh = LogicalMesh::new(vec![
    ///     MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
    ///     MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
    /// ]).unwrap();
    /// let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
    /// assert_eq!(
    ///     sharding.visualize().unwrap().render(false),
    ///     indoc! {"
    ///         ┌─────┬─────┐
    ///         │ 0,1 │ 2,3 │
    ///         └─────┴─────┘
    ///     "}
    ///     .trim_end()
    /// );
    ///
    /// // A rank-2 sharding over the same mesh with each dimension sharded along a different axis produces
    /// // a two-dimensional grid:
    /// let sharding = Sharding::new(
    ///     mesh,
    ///     vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])],
    /// ).unwrap();
    /// assert_eq!(
    ///     sharding.visualize().unwrap().render(false),
    ///     indoc! {"
    ///         ┌─────┬─────┐
    ///         │     │     │
    ///         │  0  │  1  │
    ///         │     │     │
    ///         ├─────┼─────┤
    ///         │     │     │
    ///         │  2  │  3  │
    ///         │     │     │
    ///         └─────┴─────┘
    ///     "}
    ///     .trim_end()
    /// );
    /// ```
    pub fn visualize(&self) -> Result<ShardingVisualization, ShardingError> {
        let rank = self.rank();
        if !matches!(rank, 1 | 2) {
            return Err(ShardingError::UnsupportedVisualizationRank { rank });
        }

        // Compute per-device partition coordinates and group devices into grid cells.
        let mut devices_by_cell = HashMap::<(usize, usize), Vec<MeshDeviceId>>::new();
        for device_index in 0..self.mesh.device_count() {
            // Decompose the linear device index into row-major mesh coordinates.
            let mut remaining = device_index;
            let mut mesh_coordinates = vec![0usize; self.mesh.axes.len()];
            for (axis_index, axis) in self.mesh.axes.iter().enumerate().rev() {
                mesh_coordinates[axis_index] = remaining % axis.size;
                remaining /= axis.size;
            }

            // Map the mesh coordinates to a `(row, column)` grid cell. For rank-1 shardings the row is always `0`
            // and for rank-2 shardings the row and column correspond to the partition indices of the first and second
            // dimensions, respectively.
            let row = if rank == 1 { 0 } else { self.partition_index(0, &mesh_coordinates)? };
            let column = self.partition_index(if rank == 1 { 0 } else { 1 }, &mesh_coordinates)?;
            devices_by_cell.entry((row, column)).or_default().push(device_index);
        }

        // Collect the distinct row and column partition indices.
        let row_count = devices_by_cell.keys().map(|(row, _)| *row).max().map_or(0, |max| max + 1);
        let column_count = devices_by_cell.keys().map(|(_, column)| *column).max().map_or(0, |max| max + 1);

        // Build the visualization cell grid.
        let mut cells = vec![vec![String::new(); column_count]; row_count];
        for row_index in 0..row_count {
            for column_index in 0..column_count {
                let label = devices_by_cell
                    .get_mut(&(row_index, column_index))
                    .map(|device_indices| {
                        device_indices.sort_unstable();
                        device_indices.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
                    })
                    .unwrap_or_default();
                cells[row_index][column_index] = label;
            }
        }

        Ok(ShardingVisualization { cells })
    }
}

/// Grid-based visualization of a [`Sharding`] produced by [`Sharding::visualize`]. Each cell in the grid holds a label
/// listing the device indices that share the corresponding partition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardingVisualization {
    /// Row-major grid of device-index labels (e.g., `"0,1"`).
    cells: Vec<Vec<String>>,
}

impl ShardingVisualization {
    /// Renders this [`ShardingVisualization`] to a string. When `colored` is `true` the output contains ANSI 24-bit
    /// color escape sequences. When `false` the output is stable plain text with ASCII borders and no escape sequences.
    pub fn render(&self, colored: bool) -> String {
        let row_count = self.cells.len();
        let column_count = self.cells.first().map_or(0, Vec::len);
        let cell_height = if row_count <= 1 { VISUALIZATION_1D_CELL_HEIGHT } else { VISUALIZATION_2D_CELL_HEIGHT };
        let cell_width = self
            .cells
            .iter()
            .flatten()
            .map(|label| label.chars().count() + 2)
            .max()
            .unwrap_or(0)
            .max(VISUALIZATION_MIN_CELL_WIDTH);
        let label_line = cell_height / 2;
        let mut lines = Vec::new();
        if colored {
            let background_colors = Self::background_colors(row_count, column_count);
            for (row_cells, row_colors) in self.cells.iter().zip(background_colors.iter()) {
                for line_index in 0..cell_height {
                    let mut line = String::new();
                    for (cell, &background_color) in row_cells.iter().zip(row_colors.iter()) {
                        let contents = if line_index == label_line {
                            format!("{cell:^cell_width$}")
                        } else {
                            " ".repeat(cell_width)
                        };
                        line.push_str(
                            Color::colored_text(
                                contents.as_str(),
                                background_color.foreground_color(),
                                background_color,
                            )
                            .as_str(),
                        );
                    }
                    lines.push(line);
                }
            }
        } else {
            let border_segment = "─".repeat(cell_width);
            let border_line = vec![border_segment.as_str(); column_count];
            lines.push(format!("┌{}┐", border_line.join("┬")));
            for (row_index, row_cells) in self.cells.iter().enumerate() {
                for line_index in 0..cell_height {
                    let mut line = String::from("│");
                    for label in row_cells {
                        let contents = if line_index == label_line {
                            format!("{label:^cell_width$}")
                        } else {
                            " ".repeat(cell_width)
                        };
                        line.push_str(contents.as_str());
                        line.push('│');
                    }
                    lines.push(line);
                }
                lines.push(if row_index + 1 == row_count {
                    format!("└{}┘", border_line.join("┴"))
                } else {
                    format!("├{}┤", border_line.join("┼"))
                });
            }
        }
        lines.join("\n")
    }

    /// Returns one background [`Color`] per grid cell that is assigned using a greedy graph-coloring approach which
    /// avoids giving the same color to horizontally or vertically adjacent cells.
    fn background_colors(row_count: usize, column_count: usize) -> Vec<Vec<Color>> {
        let cell_count = row_count * column_count;
        if cell_count == 0 {
            return Vec::new();
        }

        let color_count = VISUALIZATION_COLOR_PALETTE.len();
        let unique_prefix_length = cell_count.min(color_count);
        let mut color_indices = (0..unique_prefix_length).collect::<Vec<_>>();
        let mut next_color_index = 0usize;
        for cell_index in unique_prefix_length..cell_count {
            let column_index = cell_index % column_count;
            let left_neighbor = (column_index > 0).then_some(color_indices[cell_index - 1]);
            let top_neighbor = (cell_index >= column_count).then_some(color_indices[cell_index - column_count]);
            let mut color_index = None;
            for offset in 0..color_count {
                let candidate_color_index = (next_color_index + offset) % color_count;
                if Some(candidate_color_index) != left_neighbor && Some(candidate_color_index) != top_neighbor {
                    color_index = Some(candidate_color_index);
                    next_color_index = (candidate_color_index + 1) % color_count;
                    break;
                }
            }
            color_indices.push(color_index.unwrap());
        }

        color_indices
            .chunks(column_count)
            .map(|row| row.iter().map(|&index| VISUALIZATION_COLOR_PALETTE[index]).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, ShardingDimension};

    use super::*;

    #[test]
    fn test_sharding_visualization() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();

        // Test using a one-dimensional sharding where dimension `0` is sharded along the "x" axis. Devices `0` and
        // `1` share one "x" coordinate and devices `2` and `3` share the other, producing a single-row grid with two
        // grouped cells.
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let visualization = sharding.visualize().unwrap();
        assert_eq!(
            visualization.render(false),
            indoc! {"
                ┌─────┬─────┐
                │ 0,1 │ 2,3 │
                └─────┴─────┘
            "}
            .trim_end()
        );

        // Colored renderings use ANSI 24-bit escape sequences with palette colors as background colors and contrasting
        // foreground colors for the device index labels, without any ASCII box-drawing characters.
        assert_eq!(
            visualization.render(true),
            "\u{1b}[38;2;255;255;255m\u{1b}[48;2;57;59;121m 0,1 \u{1b}[0m\
             \u{1b}[38;2;255;255;255m\u{1b}[48;2;82;84;163m 2,3 \u{1b}[0m"
        );

        // Test using a two-dimensional sharding where each dimension sharded along a different axis.
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])])
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
        );

        // Three-dimensional shardings are not supported and should result in an error.
        let sharding = Sharding::replicated(mesh, 3);
        assert_eq!(sharding.visualize(), Err(ShardingError::UnsupportedVisualizationRank { rank: 3 }));
    }

    #[test]
    fn test_sharding_visualization_background_colors() {
        let row_count = 5;
        let column_count = 5;
        let background_colors = ShardingVisualization::background_colors(row_count, column_count);

        // The first [`VISUALIZATION_COLOR_PALETTE.len()`] colors should all be unique.
        let colors = background_colors.iter().flatten().copied().collect::<Vec<_>>();
        let unique_colors = colors.iter().take(VISUALIZATION_COLOR_PALETTE.len()).copied().collect::<HashSet<_>>();
        assert_eq!(unique_colors.len(), VISUALIZATION_COLOR_PALETTE.len());

        // No two horizontally or vertically adjacent cells should share a color.
        for row_index in 0..row_count {
            for column_index in 0..column_count {
                if column_index + 1 < column_count {
                    assert_ne!(
                        background_colors[row_index][column_index],
                        background_colors[row_index][column_index + 1],
                    );
                }
                if row_index + 1 < row_count {
                    assert_ne!(
                        background_colors[row_index][column_index],
                        background_colors[row_index + 1][column_index],
                    );
                }
            }
        }
    }
}
