use std::collections::HashMap;

use crate::utilities::colors::Color;

use super::{MeshDeviceId, Sharding, ShardingDimension, ShardingError};

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
            let cell = if rank == 1 {
                (0, self.partition_index(0, &mesh_coordinates))
            } else {
                (self.partition_index(0, &mesh_coordinates), self.partition_index(1, &mesh_coordinates))
            };
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
        let row_count = self.cells.len();
        let column_count = self.cells.first().map_or(0, Vec::len);
        let label_line = self.cell_height / 2;
        let mut lines = Vec::new();

        if colored {
            let background_colors = assign_visualization_background_colors(row_count, column_count);
            for (row_index, row_cells) in self.cells.iter().enumerate() {
                for line_index in 0..self.cell_height {
                    let mut line = String::new();
                    for (column_index, label) in row_cells.iter().enumerate() {
                        let contents = if line_index == label_line {
                            center_text(label.as_str(), self.cell_width)
                        } else {
                            " ".repeat(self.cell_width)
                        };
                        let background = background_colors[row_index * column_count + column_index];
                        line.push_str(
                            Color::colored_text(contents.as_str(), background.foreground_color(), background).as_str(),
                        );
                    }
                    lines.push(line);
                }
            }
        } else {
            let top_border = render_horizontal_border('┌', '┬', '┐', column_count, self.cell_width);
            let middle_border = render_horizontal_border('├', '┼', '┤', column_count, self.cell_width);
            let bottom_border = render_horizontal_border('└', '┴', '┘', column_count, self.cell_width);
            lines.push(top_border);

            for (row_index, row_cells) in self.cells.iter().enumerate() {
                for line_index in 0..self.cell_height {
                    let mut line = String::from("│");
                    for label in row_cells {
                        let contents = if line_index == label_line {
                            center_text(label.as_str(), self.cell_width)
                        } else {
                            " ".repeat(self.cell_width)
                        };
                        line.push_str(contents.as_str());
                        line.push('│');
                    }
                    lines.push(line);
                }
                if row_index + 1 == row_count {
                    lines.push(bottom_border.clone());
                } else {
                    lines.push(middle_border.clone());
                }
            }
        }

        lines.join("\n")
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

/// Builds a single horizontal border line for the plain-text visualization grid using box-drawing
/// characters. For example, `render_horizontal_border('┌', '┬', '┐', 3, 5)` produces the string
/// `"┌─────┬─────┬─────┐"`.
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

/// Centers `text` within a field of the given `width` by padding with spaces on both sides. If
/// `text` is already as wide as or wider than `width`, it is truncated to fit.
fn center_text(text: &str, width: usize) -> String {
    let text_width = text.chars().count();
    if text_width >= width {
        return text.chars().take(width).collect();
    }

    let left_padding = (width - text_width) / 2;
    let right_padding = width - text_width - left_padding;
    format!("{}{}{}", " ".repeat(left_padding), text, " ".repeat(right_padding))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, ShardingDimension};

    use super::*;

    fn test_logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
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

    #[test]
    fn test_sharding_visualize_groups_replicated_devices() {
        let mesh = test_logical_mesh_2x2();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

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
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

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
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::sharded(["y"])]).unwrap();

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
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();

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
        let backgrounds = assign_visualization_background_colors(row_count, column_count);
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
}
