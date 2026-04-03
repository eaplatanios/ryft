use std::fmt::Display;

use thiserror::Error;

#[cfg(feature = "xla")]
use ryft_pjrt::{
    Layout as PjrtLayout, StridedLayout as PjrtStridedLayout, Tile as PjrtTile, TileDimension as PjrtTileDimension,
    TiledLayout as PjrtTiledLayout,
};

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

    /// Creates a [`TileDimension`] from the provided [`PjrtTileDimension`].
    #[cfg(feature = "xla")]
    pub fn from_pjrt_tiled_dimension(tile_dimension: PjrtTileDimension) -> Self {
        tile_dimension.into()
    }

    /// Returns the [`PjrtTileDimension`] that corresponds to this [`TileDimension`].
    #[cfg(feature = "xla")]
    pub fn to_pjrt_tiled_dimension(self) -> TileDimension {
        self.into()
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

#[cfg(feature = "xla")]
impl From<PjrtTileDimension> for TileDimension {
    fn from(value: PjrtTileDimension) -> Self {
        match value.size() {
            Some(size) => Self::Sized(size),
            None => Self::Combined,
        }
    }
}

#[cfg(feature = "xla")]
impl From<TileDimension> for PjrtTileDimension {
    fn from(value: TileDimension) -> Self {
        match value {
            TileDimension::Sized(size) => Self::sized(size),
            TileDimension::Combined => Self::combined(),
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

    /// Creates a [`Tile`] from the provided [`PjrtTile`].
    #[cfg(feature = "xla")]
    pub fn from_pjrt_tile(tile: PjrtTile) -> Self {
        tile.into()
    }

    /// Returns the [`PjrtTile`] that corresponds to this [`Tile`].
    #[cfg(feature = "xla")]
    pub fn to_pjrt_tile(self) -> Tile {
        self.into()
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

#[cfg(feature = "xla")]
impl From<PjrtTile> for Tile {
    fn from(value: PjrtTile) -> Self {
        Self::new(value.dimensions.into_iter().map(Into::into).collect())
    }
}

#[cfg(feature = "xla")]
impl From<Tile> for PjrtTile {
    fn from(value: Tile) -> Self {
        Self { dimensions: value.dimensions.into_iter().map(Into::into).collect() }
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

    /// Creates a [`TiledLayout`] from the provided [`PjrtTiledLayout`].
    #[cfg(feature = "xla")]
    pub fn from_pjrt_tiled_layout(layout: PjrtTiledLayout) -> Result<Self, LayoutError> {
        layout.try_into()
    }

    /// Returns the [`PjrtTiledLayout`] that corresponds to this [`TiledLayout`].
    #[cfg(feature = "xla")]
    pub fn to_pjrt_tiled_layout(self) -> Result<PjrtTiledLayout, LayoutError> {
        self.try_into()
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

#[cfg(feature = "xla")]
impl TryFrom<PjrtTiledLayout> for TiledLayout {
    type Error = LayoutError;

    fn try_from(value: PjrtTiledLayout) -> Result<Self, Self::Error> {
        Ok(Self::new(
            value
                .minor_to_major()
                .iter()
                .copied()
                .map(|dimension| {
                    usize::try_from(dimension).map_err(|_| LayoutError::InvalidLayout {
                        message: format!(
                            "invalid minor-to-major dimension index from PJRT: '{dimension}' is out of range",
                        ),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            value.tiles().into_iter().map(Into::into).collect(),
        ))
    }
}

#[cfg(feature = "xla")]
impl TryFrom<TiledLayout> for PjrtTiledLayout {
    type Error = LayoutError;

    fn try_from(value: TiledLayout) -> Result<Self, Self::Error> {
        Ok(Self::new(
            value
                .minor_to_major
                .into_iter()
                .map(|dimension| {
                    u64::try_from(dimension).map_err(|_| LayoutError::InvalidLayout {
                        message: format!(
                            "invalid minor-to-major dimension index for PJRT: '{dimension}' is out of range",
                        ),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            value.tiles.into_iter().map(Into::into).collect(),
        ))
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

    /// Creates a [`StridedLayout`] from the provided [`PjrtStridedLayout`].
    #[cfg(feature = "xla")]
    pub fn from_pjrt_strided_layout(layout: PjrtStridedLayout) -> Result<Self, LayoutError> {
        layout.try_into()
    }

    /// Returns the [`PjrtStridedLayout`] that corresponds to this [`StridedLayout`].
    #[cfg(feature = "xla")]
    pub fn to_pjrt_strided_layout(self) -> Result<PjrtStridedLayout, LayoutError> {
        self.try_into()
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

#[cfg(feature = "xla")]
impl TryFrom<PjrtStridedLayout> for StridedLayout {
    type Error = LayoutError;

    fn try_from(value: PjrtStridedLayout) -> Result<Self, Self::Error> {
        Ok(Self::new(
            value
                .strides()
                .iter()
                .copied()
                .map(|stride| {
                    isize::try_from(stride).map_err(|_| LayoutError::InvalidLayout {
                        message: format!("invalid stride from PJRT: '{stride}' is out of range"),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

#[cfg(feature = "xla")]
impl TryFrom<StridedLayout> for PjrtStridedLayout {
    type Error = LayoutError;

    fn try_from(value: StridedLayout) -> Result<Self, Self::Error> {
        Ok(Self::new(
            value
                .strides
                .into_iter()
                .map(|stride| {
                    i64::try_from(stride).map_err(|_| LayoutError::InvalidLayout {
                        message: format!("invalid stride for PJRT: '{stride}' is out of range"),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
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

impl Layout {
    /// Creates a [`Layout`] from the provided [`PjrtLayout`].
    #[cfg(feature = "xla")]
    pub fn from_pjrt_layout(layout: PjrtLayout) -> Result<Self, LayoutError> {
        layout.try_into()
    }

    /// Returns the [`PjrtLayout`] that corresponds to this [`Layout`].
    #[cfg(feature = "xla")]
    pub fn to_pjrt_layout(self) -> Result<PjrtLayout, LayoutError> {
        self.try_into()
    }
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

#[cfg(feature = "xla")]
impl TryFrom<PjrtLayout> for Layout {
    type Error = LayoutError;

    fn try_from(value: PjrtLayout) -> Result<Self, Self::Error> {
        match value {
            PjrtLayout::Tiled(layout) => Ok(Self::Tiled(layout.try_into()?)),
            PjrtLayout::Strided(layout) => Ok(Self::Strided(layout.try_into()?)),
        }
    }
}

#[cfg(feature = "xla")]
impl TryFrom<Layout> for PjrtLayout {
    type Error = LayoutError;

    fn try_from(value: Layout) -> Result<Self, Self::Error> {
        match value {
            Layout::Tiled(layout) => Ok(Self::Tiled(layout.try_into()?)),
            Layout::Strided(layout) => Ok(Self::Strided(layout.try_into()?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    #[cfg(feature = "xla")]
    use ryft_pjrt::{
        Layout as PjrtLayout, StridedLayout as PjrtStridedLayout, Tile as PjrtTile, TileDimension as PjrtTileDimension,
        TiledLayout as PjrtTiledLayout,
    };

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

    #[cfg(feature = "xla")]
    #[test]
    fn test_layout_from_and_to_pjrt_layout() {
        let tiled_layout = TiledLayout::new(
            vec![1, 0],
            vec![
                Tile::new(vec![TileDimension::Sized(4), TileDimension::Combined]),
                Tile::new(vec![TileDimension::Sized(2)]),
            ],
        );
        let pjrt_tiled_layout = PjrtTiledLayout::new(
            vec![1, 0],
            vec![
                PjrtTile { dimensions: vec![PjrtTileDimension::sized(4), PjrtTileDimension::combined()] },
                PjrtTile { dimensions: vec![PjrtTileDimension::sized(2)] },
            ],
        );
        assert_eq!(TiledLayout::from_pjrt_tiled_layout(pjrt_tiled_layout.clone()), Ok(tiled_layout.clone()));
        assert_eq!(tiled_layout.clone().to_pjrt_tiled_layout(), Ok(pjrt_tiled_layout.clone()));

        let strided_layout = StridedLayout::new(vec![24, 8, -4]);
        let pjrt_strided_layout = PjrtStridedLayout::new(vec![24, 8, -4]);
        assert_eq!(StridedLayout::from_pjrt_strided_layout(pjrt_strided_layout.clone()), Ok(strided_layout.clone()));
        assert_eq!(strided_layout.clone().to_pjrt_strided_layout(), Ok(pjrt_strided_layout.clone()));

        let tiled_layout = Layout::Tiled(tiled_layout);
        let pjrt_tiled_layout = PjrtLayout::Tiled(pjrt_tiled_layout);
        assert_eq!(Layout::from_pjrt_layout(pjrt_tiled_layout.clone()), Ok(tiled_layout.clone()));
        assert_eq!(tiled_layout.to_pjrt_layout(), Ok(pjrt_tiled_layout));

        let strided_layout = Layout::Strided(strided_layout);
        let pjrt_strided_layout = PjrtLayout::Strided(pjrt_strided_layout);
        assert_eq!(Layout::from_pjrt_layout(pjrt_strided_layout.clone()), Ok(strided_layout.clone()));
        assert_eq!(strided_layout.to_pjrt_layout(), Ok(pjrt_strided_layout));
    }
}
