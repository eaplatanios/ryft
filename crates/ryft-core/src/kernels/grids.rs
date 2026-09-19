//! Logical kernel grids with checked, bounded extents and deterministic enumeration.
//!
//! Grid dimensions reuse [`Dimension`] identities and bounds. A rank-zero grid contains one program; a grid with
//! any zero extent contains none. Enumeration is row-major and does not imply an execution order for parallel axes.

use std::collections::{HashMap, HashSet};
use std::fmt::Display;

use ryft_macros::Parameter;
use thiserror::Error;

use crate::arrays::{Dimension, DimensionError, DimensionType, DimensionVariable, MAX_DIMENSION_EXTENT};
use crate::axes::{Axis, AxisError};
use crate::parameters::Parameter;
use crate::programs::{Type, TypeError, TypeIdentityRenaming};

/// Errors constructing a grid or binding its runtime extents.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum GridError {
    #[error(transparent)]
    Dimension(#[from] DimensionError),

    #[error(transparent)]
    Axis(#[from] AxisError),

    #[error(transparent)]
    Type(#[from] TypeError),

    #[error("kernel grid dimension {axis} must have a finite upper bound")]
    UnboundedDimension { axis: usize },

    #[error("kernel grid axis name `{name}` is repeated")]
    DuplicateName { name: String },

    #[error("kernel grid axis name cannot be empty")]
    EmptyName,

    #[error("kernel grid expects {expected} extents but received {actual}")]
    RankMismatch { expected: usize, actual: usize },

    #[error("kernel grid dimension {axis} expected extent {expected} but received {actual}")]
    ExtentMismatch { axis: usize, expected: usize, actual: usize },

    #[error("kernel grid program count exceeds the host index range")]
    ProgramCountOverflow,
}

/// Ordering semantics of one grid dimension.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum GridExecution {
    /// Programs at different coordinates may execute concurrently.
    Parallel,

    /// Coordinates execute in increasing order within each fixed tuple of other coordinates.
    Sequential,
}

impl Display for GridExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Parallel => "parallel",
            Self::Sequential => "sequential",
        })
    }
}

/// One logical grid dimension, including its optional name and ordering contract.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct GridDimension {
    /// Canonical extent and symbolic identity.
    extent: Dimension,

    /// Optional name used to select this dimension within the kernel.
    name: Option<String>,

    /// Whether distinct coordinates may execute concurrently.
    execution: GridExecution,
}

impl GridDimension {
    /// Creates an unnamed dimension. [`Grid::new`] validates finite bounds before the dimension is used.
    pub fn new(extent: Dimension, execution: GridExecution) -> Self {
        Self { extent, name: None, execution }
    }

    /// Returns the canonical dimension extent.
    pub fn extent(&self) -> &Dimension {
        &self.extent
    }

    /// Returns this dimension's name, when one was supplied.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the ordering semantics of this dimension.
    pub fn execution(&self) -> GridExecution {
        self.execution
    }

    /// Names this dimension. Names must be nonempty and unique within a [`Grid`].
    pub fn with_name(mut self, name: impl Into<String>) -> Result<Self, GridError> {
        let name = name.into();
        if name.is_empty() {
            return Err(GridError::EmptyName);
        }
        self.name = Some(name);
        Ok(self)
    }
}

impl Display for GridDimension {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = &self.name {
            write!(formatter, "{name:?}=")?;
        }
        match &self.extent {
            Dimension::Static(extent) => write!(formatter, "{extent}:{}", self.execution),
            Dimension::Dynamic(variable) => {
                write!(formatter, "{} ∈ {}:{}", variable, variable.bounds(), self.execution)
            }
        }
    }
}

/// Validated logical grid. Symbolic extents remain unspecialized until execution binds their concrete values.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct Grid {
    /// Dimensions in logical order, independent of a backend's physical launch dimensions.
    dimensions: Vec<GridDimension>,
}

impl Grid {
    /// Creates a grid with finite portable extents and unique dimension names.
    pub fn new(dimensions: Vec<GridDimension>) -> Result<Self, GridError> {
        let mut names = HashSet::new();
        for (axis, dimension) in dimensions.iter().enumerate() {
            let maximum = match dimension.extent() {
                Dimension::Static(extent) => *extent,
                Dimension::Dynamic(variable) => {
                    variable.bounds().upper().ok_or(GridError::UnboundedDimension { axis })? - 1
                }
            };
            if maximum > MAX_DIMENSION_EXTENT {
                return Err(DimensionError::ExtentExceedsBackendWidth {
                    value: maximum,
                    maximum: MAX_DIMENSION_EXTENT,
                }
                .into());
            }
            if let Some(name) = dimension.name()
                && !names.insert(name)
            {
                return Err(GridError::DuplicateName { name: name.to_owned() });
            }
        }
        Ok(Self { dimensions })
    }

    /// Returns the dimensions in logical order.
    pub fn dimensions(&self) -> &[GridDimension] {
        &self.dimensions
    }

    /// Resolves a positional axis, including negative indices, using the canonical [`Axis`] rules.
    pub fn dimension(&self, axis: impl Into<Axis>) -> Result<&GridDimension, GridError> {
        Ok(&self.dimensions[axis.into().normalize(self.dimensions.len())?])
    }

    /// Returns the positional index of a named grid dimension.
    pub fn named_axis(&self, name: &str) -> Option<usize> {
        self.dimensions.iter().position(|dimension| dimension.name() == Some(name))
    }

    /// Returns whether two validated coordinates are ordered by the sequential axes. Parallel coordinates must
    /// agree, and all changing sequential coordinates must move in the same direction. Opposite moves along two
    /// sequential axes are incomparable: serial host replay alone does not establish a semantic ordering.
    pub(crate) fn points_are_ordered(&self, first: &[usize], second: &[usize]) -> bool {
        debug_assert_eq!(first.len(), self.dimensions.len());
        debug_assert_eq!(second.len(), self.dimensions.len());
        let mut increasing = false;
        let mut decreasing = false;
        for ((dimension, first), second) in self.dimensions.iter().zip(first).zip(second) {
            if first != second && dimension.execution() == GridExecution::Parallel {
                return false;
            }
            increasing |= first < second;
            decreasing |= first > second;
        }
        !increasing || !decreasing
    }

    /// Binds concrete extents and enumerates coordinates without changing the grid's specialization identity.
    /// Repeated symbolic dimensions must receive identical extents. Static extents must match exactly, and dynamic
    /// extents must satisfy their declared bounds. The total program count is checked before iteration begins.
    pub fn points(&self, extents: &[usize]) -> Result<GridPoints, GridError> {
        if extents.len() != self.dimensions.len() {
            return Err(GridError::RankMismatch { expected: self.dimensions.len(), actual: extents.len() });
        }
        let mut bindings = HashMap::<&DimensionVariable, usize>::new();
        for (axis, (dimension, actual)) in self.dimensions.iter().zip(extents).enumerate() {
            match dimension.extent() {
                Dimension::Static(expected) if expected != actual => {
                    return Err(GridError::ExtentMismatch { axis, expected: *expected, actual: *actual });
                }
                Dimension::Dynamic(variable) => {
                    if !variable.bounds().contains(*actual) {
                        return Err(DimensionError::BindingOutOfBounds {
                            variable: variable.name().to_owned(),
                            value: *actual,
                            bounds: variable.bounds(),
                        }
                        .into());
                    }
                    if let Some(expected) = bindings.insert(variable, *actual)
                        && expected != *actual
                    {
                        return Err(DimensionError::InputDimensionMismatch {
                            dimension: variable.name().to_owned(),
                            expected,
                            actual: *actual,
                        }
                        .into());
                    }
                }
                _ => {}
            }
        }
        let remaining = if extents.contains(&0) {
            0
        } else {
            extents
                .iter()
                .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                .ok_or(GridError::ProgramCountOverflow)?
        };
        Ok(GridPoints { extents: extents.to_vec(), coordinate: vec![0; extents.len()], remaining })
    }

    /// Specializes bounded symbolic extents to checked concrete values. Names and execution semantics are retained;
    /// repeated symbolic identities must receive the same extent. The original grid remains unchanged. A specialized
    /// grid has its own semantic identity, unlike merely enumerating runtime bindings with [`Self::points`].
    pub fn specialize(&self, extents: &[usize]) -> Result<Self, GridError> {
        self.points(extents)?;
        Ok(Self {
            dimensions: self
                .dimensions
                .iter()
                .zip(extents)
                .map(|(dimension, &extent)| GridDimension { extent: Dimension::Static(extent), ..dimension.clone() })
                .collect(),
        })
    }

    /// Renames symbolic extents with the surrounding program and revalidates their finite bounds. Names and
    /// execution semantics are preserved; no concrete runtime extent is added to the specialization metadata.
    pub fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<Self, GridError> {
        Self::new(
            self.dimensions
                .iter()
                .map(|dimension| {
                    let extent = match &dimension.extent {
                        Dimension::Static(extent) => Dimension::Static(*extent),
                        Dimension::Dynamic(variable) => Dimension::Dynamic(
                            DimensionType::from(variable.clone()).rename_identities(renaming)?.variable().clone(),
                        ),
                    };
                    Ok(GridDimension { extent, name: dimension.name.clone(), execution: dimension.execution })
                })
                .collect::<Result<Vec<_>, TypeError>>()?,
        )
    }
}

impl Display for Grid {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("[")?;
        for (index, dimension) in self.dimensions.iter().enumerate() {
            if index != 0 {
                formatter.write_str(", ")?;
            }
            write!(formatter, "{dimension}")?;
        }
        formatter.write_str("]")
    }
}

/// Checked row-major enumeration returned by [`Grid::points`]. It allocates only one coordinate per yielded program.
#[derive(Clone, Debug)]
pub struct GridPoints {
    /// Runtime extents checked against the originating grid.
    extents: Vec<usize>,

    /// Next logical coordinate.
    coordinate: Vec<usize>,

    /// Number of coordinates remaining.
    remaining: usize,
}

impl Iterator for GridPoints {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let coordinate = self.coordinate.clone();
        self.remaining -= 1;
        if self.remaining != 0 {
            for axis in (0..self.extents.len()).rev() {
                self.coordinate[axis] += 1;
                if self.coordinate[axis] < self.extents[axis] {
                    break;
                }
                self.coordinate[axis] = 0;
            }
        }
        Some(coordinate)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for GridPoints {}
impl std::iter::FusedIterator for GridPoints {}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::arrays::DimensionBounds;

    use super::*;

    #[test]
    fn test_grid_execution() {
        assert_eq!(GridExecution::Parallel.to_string(), "parallel");
        assert_eq!(GridExecution::Sequential.to_string(), "sequential");
        assert_eq!(format!("{:?}", GridExecution::Parallel), "Parallel");
    }

    #[test]
    fn test_grid_dimension_new() {
        let dimension = GridDimension::new(Dimension::Static(3), GridExecution::Parallel);
        assert_eq!(dimension.to_string(), "3:parallel");
        assert_eq!(dimension.clone(), dimension);
        let dimension = GridDimension::new(
            Dimension::Dynamic(DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap())),
            GridExecution::Parallel,
        );
        assert_eq!(dimension.to_string(), "rows ∈ [0, 5):parallel");
    }

    #[test]
    fn test_grid_dimension_extent() {
        assert_eq!(GridDimension::new(Dimension::Static(3), GridExecution::Parallel).extent(), &Dimension::Static(3));
    }

    #[test]
    fn test_grid_dimension_name() {
        assert_eq!(GridDimension::new(Dimension::Static(3), GridExecution::Parallel).name(), None);
    }

    #[test]
    fn test_grid_dimension_execution() {
        assert_eq!(
            GridDimension::new(Dimension::Static(3), GridExecution::Sequential).execution(),
            GridExecution::Sequential
        );
    }

    #[test]
    fn test_grid_dimension_with_name() {
        let dimension = GridDimension::new(Dimension::Static(3), GridExecution::Parallel).with_name("row").unwrap();
        assert_eq!(dimension.name(), Some("row"));
        assert_eq!(dimension.to_string(), "\"row\"=3:parallel");
        assert_eq!(dimension.clone().with_name("row\nnext").unwrap().to_string(), "\"row\\nnext\"=3:parallel");
        assert_eq!(dimension.with_name(""), Err(GridError::EmptyName));
    }

    #[test]
    fn test_grid_new() {
        let dimension = GridDimension::new(Dimension::Static(3), GridExecution::Parallel).with_name("row").unwrap();
        let grid = Grid::new(vec![dimension.clone()]).unwrap();
        assert_eq!(grid.to_string(), "[\"row\"=3:parallel]");
        assert_eq!(HashMap::from([(grid.clone(), 7)])[&grid], 7);
        assert_ne!(grid, Grid::new(vec![]).unwrap());
        assert_eq!(Grid::new(vec![dimension.clone(), dimension]), Err(GridError::DuplicateName { name: "row".into() }));
        let unbounded = DimensionVariable::new("count", DimensionBounds::unbounded());
        assert_eq!(
            Grid::new(vec![GridDimension::new(Dimension::Dynamic(unbounded), GridExecution::Parallel)]),
            Err(GridError::UnboundedDimension { axis: 0 }),
        );
        assert_eq!(
            Grid::new(vec![GridDimension::new(Dimension::Static(MAX_DIMENSION_EXTENT + 1), GridExecution::Parallel)]),
            Err(GridError::Dimension(DimensionError::ExtentExceedsBackendWidth {
                value: MAX_DIMENSION_EXTENT + 1,
                maximum: MAX_DIMENSION_EXTENT,
            })),
        );
    }

    #[test]
    fn test_grid_dimensions() {
        let dimensions = vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)];
        assert_eq!(Grid::new(dimensions.clone()).unwrap().dimensions(), dimensions);
    }

    #[test]
    fn test_grid_dimension() {
        let grid = Grid::new(vec![GridDimension::new(Dimension::Static(2), GridExecution::Parallel)]).unwrap();
        assert_eq!(grid.dimension(-1), Ok(&grid.dimensions()[0]));
        assert_eq!(grid.dimension(1), Err(GridError::Axis(AxisError::OutOfBounds { axis: Axis::from(1), rank: 1 })));
    }

    #[test]
    fn test_grid_named_axis() {
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Static(2), GridExecution::Parallel).with_name("row").unwrap(),
        ])
        .unwrap();
        assert_eq!(grid.named_axis("row"), Some(0));
        assert_eq!(grid.named_axis("column"), None);
    }

    #[test]
    fn test_grid_points_are_ordered() {
        let sequential = Grid::new(vec![
            GridDimension::new(Dimension::Static(2), GridExecution::Sequential),
            GridDimension::new(Dimension::Static(2), GridExecution::Sequential),
        ])
        .unwrap();
        assert!(sequential.points_are_ordered(&[0, 0], &[1, 1]));
        assert!(sequential.points_are_ordered(&[1, 1], &[0, 0]));
        assert!(sequential.points_are_ordered(&[0, 1], &[0, 1]));
        assert!(!sequential.points_are_ordered(&[0, 1], &[1, 0]));
        let mixed = Grid::new(vec![
            GridDimension::new(Dimension::Static(2), GridExecution::Parallel),
            GridDimension::new(Dimension::Static(2), GridExecution::Sequential),
        ])
        .unwrap();
        assert!(mixed.points_are_ordered(&[0, 0], &[0, 1]));
        assert!(!mixed.points_are_ordered(&[0, 0], &[1, 1]));
    }

    #[test]
    fn test_grid_points() {
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Static(2), GridExecution::Parallel),
            GridDimension::new(Dimension::Static(3), GridExecution::Sequential),
        ])
        .unwrap();
        let mut points = grid.points(&[2, 3]).unwrap();
        assert_eq!(points.size_hint(), (6, Some(6)));
        assert_eq!(points.next(), Some(vec![0, 0]));
        assert_eq!(points.len(), 5);
        assert_eq!(points.collect::<Vec<_>>(), vec![vec![0, 1], vec![0, 2], vec![1, 0], vec![1, 1], vec![1, 2]]);
        assert_eq!(Grid::new(vec![]).unwrap().points(&[]).unwrap().collect::<Vec<_>>(), vec![Vec::<usize>::new()]);
        assert_eq!(grid.points(&[3, 3]).unwrap_err(), GridError::ExtentMismatch { axis: 0, expected: 2, actual: 3 });
        assert_eq!(grid.points(&[2]).unwrap_err(), GridError::RankMismatch { expected: 2, actual: 1 });
    }

    #[test]
    fn test_grid_points_dynamic_identity_and_bounds() {
        let variable = DimensionVariable::new("count", DimensionBounds::non_negative(Some(4)).unwrap());
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Dynamic(variable.clone()), GridExecution::Parallel),
            GridDimension::new(Dimension::Dynamic(variable), GridExecution::Parallel),
        ])
        .unwrap();
        assert_eq!(grid.points(&[0, 0]).unwrap().count(), 0);
        assert_eq!(grid.points(&[2, 2]).unwrap().count(), 4);
        assert_eq!(
            grid.points(&[2, 3]).unwrap_err(),
            GridError::Dimension(DimensionError::InputDimensionMismatch {
                dimension: "count".into(),
                expected: 2,
                actual: 3,
            })
        );
        assert_eq!(
            grid.points(&[4, 4]).unwrap_err(),
            GridError::Dimension(DimensionError::BindingOutOfBounds {
                variable: "count".into(),
                value: 4,
                bounds: DimensionBounds::non_negative(Some(4)).unwrap(),
            })
        );
    }

    #[test]
    fn test_grid_points_overflow_and_empty() {
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Static(MAX_DIMENSION_EXTENT), GridExecution::Parallel),
            GridDimension::new(Dimension::Static(3), GridExecution::Parallel),
        ])
        .unwrap();
        assert_eq!(grid.points(&[MAX_DIMENSION_EXTENT, 3]).unwrap_err(), GridError::ProgramCountOverflow);
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Static(0), GridExecution::Parallel),
            GridDimension::new(Dimension::Static(MAX_DIMENSION_EXTENT), GridExecution::Parallel),
        ])
        .unwrap();
        let mut points = grid.points(&[0, MAX_DIMENSION_EXTENT]).unwrap();
        assert_eq!(points.next(), None);
        assert_eq!(points.next(), None);
        assert_eq!(points.len(), 0);
    }

    #[test]
    fn test_grid_specialize() {
        let variable = DimensionVariable::new("count", DimensionBounds::non_negative(Some(4)).unwrap());
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Dynamic(variable.clone()), GridExecution::Sequential)
                .with_name("rows")
                .unwrap(),
            GridDimension::new(Dimension::Dynamic(variable.clone()), GridExecution::Parallel),
        ])
        .unwrap();
        let specialized = grid.specialize(&[2, 2]).unwrap();
        assert_eq!(specialized.dimensions()[0].extent(), &Dimension::Static(2));
        assert_eq!(specialized.dimensions()[0].execution(), GridExecution::Sequential);
        assert_eq!(specialized.dimensions()[0].name(), Some("rows"));
        assert_eq!(specialized.dimensions()[1].extent(), &Dimension::Static(2));
        assert_eq!(specialized.dimensions()[1].execution(), GridExecution::Parallel);
        assert_eq!(grid.dimensions()[0].extent(), &Dimension::Dynamic(variable));
        assert_eq!(grid.specialize(&[0, 0]).unwrap().points(&[0, 0]).unwrap().count(), 0);
        assert_eq!(Grid::new(vec![]).unwrap().specialize(&[]), Ok(Grid::new(vec![]).unwrap()));
    }

    #[test]
    fn test_grid_specialize_checks_bounds_and_identity() {
        let variable = DimensionVariable::new("count", DimensionBounds::non_negative(Some(4)).unwrap());
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Dynamic(variable.clone()), GridExecution::Parallel),
            GridDimension::new(Dimension::Dynamic(variable.clone()), GridExecution::Parallel),
        ])
        .unwrap();
        assert_eq!(grid.specialize(&[2]), Err(GridError::RankMismatch { expected: 2, actual: 1 }));
        assert_eq!(
            grid.specialize(&[2, 3]),
            Err(GridError::Dimension(DimensionError::InputDimensionMismatch {
                dimension: "count".to_owned(),
                expected: 2,
                actual: 3,
            }))
        );
        assert_eq!(
            grid.specialize(&[4, 4]),
            Err(GridError::Dimension(DimensionError::BindingOutOfBounds {
                variable: "count".to_owned(),
                value: 4,
                bounds: variable.bounds(),
            }))
        );
        let static_grid = grid.specialize(&[2, 2]).unwrap();
        assert_eq!(static_grid.specialize(&[3, 3]), Err(GridError::ExtentMismatch { axis: 0, expected: 2, actual: 3 }));
    }

    #[test]
    fn test_grid_rename_type_identities() {
        let source = DimensionVariable::new("source", DimensionBounds::non_negative(Some(4)).unwrap());
        let target = DimensionVariable::new("target", DimensionBounds::non_negative(Some(3)).unwrap());
        let grid = Grid::new(vec![
            GridDimension::new(Dimension::Dynamic(source.clone()), GridExecution::Parallel)
                .with_name("row")
                .unwrap(),
        ])
        .unwrap();
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source.clone(), target.clone()).unwrap();
        let renamed = grid.rename_type_identities(&renaming).unwrap();
        assert_eq!(renamed.dimensions()[0].extent(), &Dimension::Dynamic(target));
        assert_eq!(renamed.dimensions()[0].name(), Some("row"));
        assert_eq!(grid.dimensions()[0].extent(), &Dimension::Dynamic(source.clone()));
        let mut invalid = TypeIdentityRenaming::new();
        invalid
            .insert(source.clone(), DimensionVariable::new("unbounded", DimensionBounds::unbounded()))
            .unwrap();
        assert_eq!(
            grid.rename_type_identities(&invalid),
            Err(GridError::Type(TypeError::invalid(
                "cannot rename dimension variable source with bounds [0, 4) to unbounded with bounds [0, ∞)",
            )))
        );
        let mut wider = TypeIdentityRenaming::new();
        wider
            .insert(source, DimensionVariable::new("wider", DimensionBounds::non_negative(Some(5)).unwrap()))
            .unwrap();
        assert_eq!(
            grid.rename_type_identities(&wider),
            Err(GridError::Type(TypeError::invalid(
                "cannot rename dimension variable source with bounds [0, 4) to wider with bounds [0, 5)",
            )))
        );
    }
}
