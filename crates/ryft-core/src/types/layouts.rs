use std::fmt::Display;

use thiserror::Error;

/// Represents [`Layout`]-related errors.
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum LayoutError {
    #[error("invalid layout: {message}")]
    InvalidLayout { message: String },
}

/// Describes one dimension of a [`Tile`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TileDimension {
    /// Tile dimension with a fixed size (i.e., number of elements).
    Sized(usize),

    /// Tile dimension that is _combined_ with the next more minor logical dimension before tiling is applied,
    /// and thus has no fixed size of its own.
    Combined,
}

impl TileDimension {
    /// Returns the size (i.e., number of elements) of this [`TileDimension`], if it has one. The only case in which a
    /// tile dimension has no fixed size is when it is a [`TileDimension::Combined`] dimension.
    #[inline]
    pub fn size(&self) -> Option<usize> {
        match self {
            Self::Sized(size) => Some(*size),
            Self::Combined => None,
        }
    }

    /// Returns `true` if this [`TileDimension`] is a [`TileDimension::Sized`] dimension.
    #[inline]
    pub fn is_sized(&self) -> bool {
        matches!(self, Self::Sized(_))
    }

    /// Returns `true` if this [`TileDimension`] is a [`TileDimension::Combined`] dimension.
    #[inline]
    pub fn is_combined(&self) -> bool {
        matches!(self, Self::Combined)
    }
}

impl Display for TileDimension {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sized(size) => write!(formatter, "{size}"),
            Self::Combined => write!(formatter, "*"),
        }
    }
}

/// Tile used in a [`TiledLayout`]. Ryft tiled layouts match XLA tiled layouts. Refer to the
/// [official XLA documentation](https://openxla.org/xla/tiled_layout) for more information.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Tile {
    /// Dimensions of this [`ryft_pjrt::Tile`], ordered from the most major dimension to the most minor dimension.
    /// The dimensions of a tile correspond to a suffix of the dimensions of the tiled array.
    pub dimensions: Vec<TileDimension>,
}

impl Tile {
    /// Creates a new [`Tile`] with the provided dimensions.
    #[inline]
    pub fn new(dimensions: Vec<TileDimension>) -> Self {
        Self { dimensions }
    }
}

impl Display for Tile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("(")?;
        let mut dimensions = self.dimensions.iter();
        if let Some(first_dimension) = dimensions.next() {
            write!(formatter, "{first_dimension}")?;
            dimensions.try_for_each(|dimension| write!(formatter, ",{dimension}"))?;
        }
        formatter.write_str(")")
    }
}

impl From<Vec<TileDimension>> for Tile {
    #[inline]
    fn from(dimensions: Vec<TileDimension>) -> Self {
        Self::new(dimensions)
    }
}

/// Tiling-based [`Layout`]. Ryft tiled layouts match XLA tiled layouts. Refer to the
/// [official XLA documentation](https://openxla.org/xla/tiled_layout) for more information.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TiledLayout {
    /// Sequence of logical dimension indices ordered from the most minor physical dimension (i.e., the one with the
    /// fastest varying index) to the most major physical dimension (i.e., the one with the slowest varying index).
    /// This is effectively a map from physical dimension indices to logical dimension indices, and it must have the
    /// same length as the rank of the corresponding array value.
    pub minor_to_major: Vec<usize>,

    /// Sequence of [`Tile`]s that are used in this [`TiledLayout`]. The tiles are nested, with the outermost tiling
    /// appearing first and the innermost tiling appearing last.
    pub tiles: Vec<Tile>,
}

impl TiledLayout {
    /// Creates a new [`TiledLayout`] with the provided minor-to-major dimension ordering and nested [`Tile`]s.
    #[inline]
    pub fn new(minor_to_major: Vec<usize>, tiles: Vec<Tile>) -> Self {
        Self { minor_to_major, tiles }
    }

    /// Returns the number of logical dimensions described by this [`TiledLayout`].
    #[inline]
    pub fn rank(&self) -> usize {
        self.minor_to_major.len()
    }

    /// Returns the `index`-th [`Tile`] of this [`TiledLayout`], or [`None`] if `index` is out-of-bounds.
    #[inline]
    pub fn tile(&self, index: usize) -> Option<&Tile> {
        self.tiles.get(index)
    }
}

impl Display for TiledLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("tiled{")?;
        let mut dimensions = self.minor_to_major.iter();
        if let Some(first_dimension) = dimensions.next() {
            write!(formatter, "{first_dimension}")?;
            dimensions.try_for_each(|dimension| write!(formatter, ",{dimension}"))?;
        }
        if !self.tiles.is_empty() {
            formatter.write_str(":T")?;
            self.tiles.iter().try_for_each(|tile| write!(formatter, "{tile}"))?;
        }
        formatter.write_str("}")
    }
}

/// Strided [`Layout`]. The storage offset of the element at logical index `(i, j, k)` in a 3-dimensional array,
/// for example, is computed as follows when using this layout: `i * strides[0] + j * strides[1] + k * strides[2]`.
/// This offset is relative to the storage location pointed to by the underlying array data.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct StridedLayout {
    /// Sequence of dimension strides (i.e., number of bytes to traverse per dimension). This sequence must have the
    /// same length as the rank of the corresponding array value. Strides are allowed to be negative, in which case the
    /// underlying data pointer may need to refer to the interior of the array storage rather than its beginning.
    pub strides: Vec<isize>,
}

impl StridedLayout {
    /// Creates a new [`StridedLayout`] with the provided byte strides.
    #[inline]
    pub fn new(strides: Vec<isize>) -> Self {
        Self { strides }
    }

    /// Returns the number of logical dimensions described by this [`StridedLayout`].
    #[inline]
    pub fn rank(&self) -> usize {
        self.strides.len()
    }
}

impl Display for StridedLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("strided{")?;
        let mut strides = self.strides.iter();
        if let Some(first_stride) = strides.next() {
            write!(formatter, "{first_stride}")?;
            strides.try_for_each(|stride| write!(formatter, ",{stride}"))?;
        }
        formatter.write_str("}")
    }
}

/// Memory/storage layout of a (potentially) multi-dimensional array that determines the mapping from logical indices
/// to physical offsets in the underlying memory/storage.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Layout {
    /// Tiling-based [`Layout`]. Refer to [`TiledLayout`] for more information.
    Tiled(TiledLayout),

    /// Strided [`Layout`]. Refer to [`StridedLayout`] for more information.
    Strided(StridedLayout),
}

impl Display for Layout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tiled(layout) => write!(formatter, "{layout}"),
            Self::Strided(layout) => write!(formatter, "{layout}"),
        }
    }
}

impl From<TiledLayout> for Layout {
    fn from(value: TiledLayout) -> Self {
        Self::Tiled(value)
    }
}

impl From<StridedLayout> for Layout {
    fn from(value: StridedLayout) -> Self {
        Self::Strided(value)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::{Layout, StridedLayout, Tile, TileDimension, TiledLayout};

    #[test]
    fn test_tile_dimension() {
        let dimension = TileDimension::Sized(16);
        assert_eq!(dimension.size(), Some(16));
        assert!(!dimension.is_combined());
        assert_eq!(format!("{dimension}"), "16");

        let dimension = TileDimension::Combined;
        assert_eq!(dimension.size(), None);
        assert!(dimension.is_combined());
        assert_eq!(format!("{dimension}"), "*");
    }

    #[test]
    fn test_tile() {
        let tile = Tile::new(vec![
            TileDimension::Sized(16),
            TileDimension::Combined,
            TileDimension::Sized(4),
            TileDimension::Sized(2),
        ]);
        assert_eq!(format!("{tile}"), "(16,*,4,2)");
    }

    #[test]
    fn test_tiled_layout() {
        let tiles = vec![
            Tile::new(vec![TileDimension::Sized(8), TileDimension::Combined]),
            Tile::new(vec![TileDimension::Sized(4)]),
        ];
        let layout = TiledLayout::new(vec![2, 1, 0], tiles.clone());
        assert_eq!(layout.rank(), 3);
        assert_eq!(layout.minor_to_major, vec![2, 1, 0]);
        assert_eq!(layout.tiles, tiles);
        assert_eq!(layout.tile(0), Some(&layout.tiles[0]));
        assert_eq!(layout.tile(1), Some(&layout.tiles[1]));
        assert_eq!(layout.tile(2), None);
        assert_eq!(format!("{layout}"), "tiled{2,1,0:T(8,*)(4)}");

        let empty_layout = TiledLayout::new(Vec::new(), Vec::new());
        assert_eq!(empty_layout.rank(), 0);
        assert_eq!(format!("{empty_layout}"), "tiled{}");
    }

    #[test]
    fn test_strided_layout() {
        let layout = StridedLayout::new(vec![24, 8, -4]);
        assert_eq!(layout.rank(), 3);
        assert_eq!(layout.strides, vec![24, 8, -4]);
        assert_eq!(format!("{layout}"), "strided{24,8,-4}");
    }

    #[test]
    fn test_layout() {
        let tiled_layout = Layout::Tiled(TiledLayout::new(
            vec![1, 0],
            vec![Tile::new(vec![TileDimension::Sized(4), TileDimension::Combined])],
        ));
        assert_eq!(format!("{tiled_layout}"), "tiled{1,0:T(4,*)}");

        let strided_layout = Layout::Strided(StridedLayout::new(vec![16, 4]));
        assert_eq!(format!("{strided_layout}"), "strided{16,4}");
    }
}
