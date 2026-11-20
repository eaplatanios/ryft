//! PJRT representations of `ryft-core` data types and layouts, and dense byte counts for host transfers.
//! This module implements [`ToPjrt`] and [`FromPjrt`] for [`DataType`] (i.e., PJRT [`BufferType`]s) and for
//! [`Layout`], [`TiledLayout`], [`StridedLayout`], [`Tile`], and [`TileDimension`]. Tiled and strided layout
//! conversions are fallible because PJRT stores minor-to-major dimension indices as unsigned 64-bit integers
//! and strides as signed 64-bit integers. [`ArrayTypeExtension`] uses the logical size of a single element to
//! compute the dense byte count of a whole array.

use ryft_core::{
    ArrayType, DataType, DataTypeError, Layout, LayoutError, StridedLayout, Tile, TileDimension, TiledLayout,
};
use ryft_pjrt::{
    BufferType, Layout as PjrtLayout, StridedLayout as PjrtStridedLayout, Tile as PjrtTile,
    TileDimension as PjrtTileDimension, TiledLayout as PjrtTiledLayout,
};

use crate::{ArrayError, Error, FromPjrt, ToPjrt};

impl ToPjrt for TileDimension {
    type Output = PjrtTileDimension;

    #[inline]
    fn to_pjrt(&self) -> PjrtTileDimension {
        match self {
            TileDimension::Sized(size) => PjrtTileDimension::sized(*size),
            TileDimension::Combined => PjrtTileDimension::combined(),
        }
    }
}

impl FromPjrt<PjrtTileDimension> for TileDimension {
    type Output = Self;

    #[inline]
    fn from_pjrt(value: PjrtTileDimension) -> Self {
        match value.size() {
            Some(size) => Self::Sized(size),
            None => Self::Combined,
        }
    }
}

impl ToPjrt for Tile {
    type Output = PjrtTile;

    #[inline]
    fn to_pjrt(&self) -> PjrtTile {
        PjrtTile { dimensions: self.dimensions().iter().map(ToPjrt::to_pjrt).collect() }
    }
}

impl FromPjrt<PjrtTile> for Tile {
    type Output = Self;

    #[inline]
    fn from_pjrt(value: PjrtTile) -> Self {
        Self::new(value.dimensions.into_iter().map(TileDimension::from_pjrt).collect())
    }
}

impl ToPjrt for TiledLayout {
    type Output = Result<PjrtTiledLayout, LayoutError>;

    #[inline]
    fn to_pjrt(&self) -> Result<PjrtTiledLayout, LayoutError> {
        Ok(PjrtTiledLayout::new(
            self.minor_to_major()
                .iter()
                .copied()
                .map(|dimension| {
                    u64::try_from(dimension).map_err(|_| LayoutError::InvalidLayout {
                        message: format!(
                            "invalid minor-to-major dimension index for PJRT: `{dimension}` is out of range",
                        ),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            self.tiles().iter().map(ToPjrt::to_pjrt).collect(),
        ))
    }
}

impl FromPjrt<PjrtTiledLayout> for TiledLayout {
    type Output = Result<Self, LayoutError>;

    #[inline]
    fn from_pjrt(value: PjrtTiledLayout) -> Result<Self, LayoutError> {
        Ok(Self::new(
            value
                .minor_to_major()
                .iter()
                .copied()
                .map(|dimension| {
                    usize::try_from(dimension).map_err(|_| LayoutError::InvalidLayout {
                        message: format!(
                            "invalid minor-to-major dimension index from PJRT: `{dimension}` is out of range",
                        ),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            value.tiles().into_iter().map(Tile::from_pjrt).collect(),
        ))
    }
}

impl ToPjrt for StridedLayout {
    type Output = Result<PjrtStridedLayout, LayoutError>;

    #[inline]
    fn to_pjrt(&self) -> Result<PjrtStridedLayout, LayoutError> {
        Ok(PjrtStridedLayout::new(
            self.strides()
                .iter()
                .copied()
                .map(|stride| {
                    i64::try_from(stride).map_err(|_| LayoutError::InvalidLayout {
                        message: format!("invalid stride for PJRT: `{stride}` is out of range"),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl FromPjrt<PjrtStridedLayout> for StridedLayout {
    type Output = Result<Self, LayoutError>;

    #[inline]
    fn from_pjrt(value: PjrtStridedLayout) -> Result<Self, LayoutError> {
        Ok(Self::new(
            value
                .strides()
                .iter()
                .copied()
                .map(|stride| {
                    isize::try_from(stride).map_err(|_| LayoutError::InvalidLayout {
                        message: format!("invalid stride from PJRT: `{stride}` is out of range"),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl ToPjrt for Layout {
    type Output = Result<PjrtLayout, LayoutError>;

    #[inline]
    fn to_pjrt(&self) -> Result<PjrtLayout, LayoutError> {
        match self {
            Layout::Tiled(layout) => Ok(PjrtLayout::Tiled(layout.to_pjrt()?)),
            Layout::Strided(layout) => Ok(PjrtLayout::Strided(layout.to_pjrt()?)),
        }
    }
}

impl FromPjrt<PjrtLayout> for Layout {
    type Output = Result<Self, LayoutError>;

    #[inline]
    fn from_pjrt(value: PjrtLayout) -> Result<Self, LayoutError> {
        match value {
            PjrtLayout::Tiled(layout) => Ok(Self::Tiled(TiledLayout::from_pjrt(layout)?)),
            PjrtLayout::Strided(layout) => Ok(Self::Strided(StridedLayout::from_pjrt(layout)?)),
        }
    }
}

impl ToPjrt for DataType {
    type Output = BufferType;

    #[inline]
    fn to_pjrt(&self) -> BufferType {
        match self {
            DataType::Token => BufferType::Token,
            // PJRT has no zero-information buffer type. The predicate buffer is a private physical carrier, and logical
            // `Zero` metadata is retained by Ryft and therefore intentionally does not round-trip through `BufferType`.
            DataType::Zero => BufferType::Predicate,
            DataType::Boolean => BufferType::Predicate,
            DataType::I1 => BufferType::I1,
            DataType::I2 => BufferType::I2,
            DataType::I4 => BufferType::I4,
            DataType::I8 => BufferType::I8,
            DataType::I16 => BufferType::I16,
            DataType::I32 => BufferType::I32,
            DataType::I64 => BufferType::I64,
            DataType::U1 => BufferType::U1,
            DataType::U2 => BufferType::U2,
            DataType::U4 => BufferType::U4,
            DataType::U8 => BufferType::U8,
            DataType::U16 => BufferType::U16,
            DataType::U32 => BufferType::U32,
            DataType::U64 => BufferType::U64,
            DataType::F4E2M1FN => BufferType::F4E2M1FN,
            DataType::F6E2M3FN => BufferType::F6E2M3FN,
            DataType::F6E3M2FN => BufferType::F6E3M2FN,
            DataType::F8E3M4 => BufferType::F8E3M4,
            DataType::F8E4M3 => BufferType::F8E4M3,
            DataType::F8E4M3FN => BufferType::F8E4M3FN,
            DataType::F8E4M3FNUZ => BufferType::F8E4M3FNUZ,
            DataType::F8E4M3B11FNUZ => BufferType::F8E4M3B11FNUZ,
            DataType::F8E5M2 => BufferType::F8E5M2,
            DataType::F8E5M2FNUZ => BufferType::F8E5M2FNUZ,
            DataType::F8E8M0FNU => BufferType::F8E8M0FNU,
            DataType::BF16 => BufferType::BF16,
            DataType::F16 => BufferType::F16,
            DataType::F32 => BufferType::F32,
            DataType::F64 => BufferType::F64,
            DataType::C64 => BufferType::C64,
            DataType::C128 => BufferType::C128,
        }
    }
}

impl FromPjrt<BufferType> for DataType {
    type Output = Result<Self, DataTypeError>;

    #[inline]
    fn from_pjrt(value: BufferType) -> Result<Self, DataTypeError> {
        match value {
            BufferType::Invalid => Err(DataTypeError::InvalidDataType {
                message: format!("invalid data type from PJRT: `{value}`"),
                backtrace: std::backtrace::Backtrace::capture().to_string(),
            }),
            BufferType::Token => Ok(Self::Token),
            BufferType::Predicate => Ok(Self::Boolean),
            BufferType::I1 => Ok(Self::I1),
            BufferType::I2 => Ok(Self::I2),
            BufferType::I4 => Ok(Self::I4),
            BufferType::I8 => Ok(Self::I8),
            BufferType::I16 => Ok(Self::I16),
            BufferType::I32 => Ok(Self::I32),
            BufferType::I64 => Ok(Self::I64),
            BufferType::U1 => Ok(Self::U1),
            BufferType::U2 => Ok(Self::U2),
            BufferType::U4 => Ok(Self::U4),
            BufferType::U8 => Ok(Self::U8),
            BufferType::U16 => Ok(Self::U16),
            BufferType::U32 => Ok(Self::U32),
            BufferType::U64 => Ok(Self::U64),
            BufferType::F4E2M1FN => Ok(Self::F4E2M1FN),
            BufferType::F6E2M3FN => Ok(Self::F6E2M3FN),
            BufferType::F6E3M2FN => Ok(Self::F6E3M2FN),
            BufferType::F8E3M4 => Ok(Self::F8E3M4),
            BufferType::F8E4M3 => Ok(Self::F8E4M3),
            BufferType::F8E4M3FN => Ok(Self::F8E4M3FN),
            BufferType::F8E4M3FNUZ => Ok(Self::F8E4M3FNUZ),
            BufferType::F8E4M3B11FNUZ => Ok(Self::F8E4M3B11FNUZ),
            BufferType::F8E5M2 => Ok(Self::F8E5M2),
            BufferType::F8E5M2FNUZ => Ok(Self::F8E5M2FNUZ),
            BufferType::F8E8M0FNU => Ok(Self::F8E8M0FNU),
            BufferType::BF16 => Ok(Self::BF16),
            BufferType::F16 => Ok(Self::F16),
            BufferType::F32 => Ok(Self::F32),
            BufferType::F64 => Ok(Self::F64),
            BufferType::C64 => Ok(Self::C64),
            BufferType::C128 => Ok(Self::C128),
        }
    }
}

/// Extends [`DataType`] with its PJRT executable representation and logical payload size.
pub(crate) trait DataTypeExtension {
    /// Returns the number of bytes one element occupies in a dense array. [`DataType::Zero`] has no logical
    /// payload even when PJRT uses a temporary predicate buffer to represent it.
    fn size_in_bytes(self) -> Result<usize, Error>;

    /// Returns the PJRT buffer type for this data type in a StableHLO executable. Signed and unsigned one-bit
    /// integers both lower to signless `i1` and use predicate buffers. Other types use their [`ToPjrt`] representation.
    fn executable_buffer_type(self) -> BufferType;
}

impl DataTypeExtension for DataType {
    #[inline]
    fn size_in_bytes(self) -> Result<usize, Error> {
        if self.is_zero() {
            return Ok(0);
        }
        Ok(self.to_pjrt().element_size_in_bytes()?)
    }

    #[inline]
    fn executable_buffer_type(self) -> BufferType {
        match self {
            Self::I1 | Self::U1 => BufferType::Predicate,
            _ => self.to_pjrt(),
        }
    }
}

/// Extends [`ArrayType`] with the dense byte counts used by the XLA backend.
pub trait ArrayTypeExtension {
    /// Returns the number of logical payload bytes required by a dense [`Array`](crate::Array) of this type.
    /// [`DataType::Zero`] requires no payload bytes even when the backend uses a temporary buffer for it.
    ///
    /// # Errors
    ///
    /// For types other than [`DataType::Zero`], returns an error if the element count cannot be determined from
    /// the shape or if either the element count or byte count overflows [`usize`].
    fn dense_size_in_bytes(&self) -> Result<usize, Error>;
}

impl ArrayTypeExtension for ArrayType {
    fn dense_size_in_bytes(&self) -> Result<usize, Error> {
        let data_type = self.data_type();
        if data_type.is_zero() {
            return Ok(0);
        }
        let element_size_in_bytes = data_type.size_in_bytes()?;
        let element_count = self
            .element_count()?
            .ok_or_else(|| Error::from(ArrayError::DynamicShape { shape: self.shape().clone() }))?;
        element_count.checked_mul(element_size_in_bytes).ok_or_else(|| Error::SizeLimitExceeded {
            message: format!(
                "dense byte size for array type `{}` exceeds the maximum allowed size of {}",
                self,
                usize::MAX,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Dimension, DimensionBounds, DimensionVariable, Shape};

    use super::*;

    /// Returns the PJRT [`BufferType`] of every [`DataType`] that has one, in [`DataType`] declaration order.
    fn data_type_buffer_types() -> Vec<(DataType, BufferType)> {
        vec![
            (DataType::Token, BufferType::Token),
            (DataType::Boolean, BufferType::Predicate),
            (DataType::I1, BufferType::I1),
            (DataType::I2, BufferType::I2),
            (DataType::I4, BufferType::I4),
            (DataType::I8, BufferType::I8),
            (DataType::I16, BufferType::I16),
            (DataType::I32, BufferType::I32),
            (DataType::I64, BufferType::I64),
            (DataType::U1, BufferType::U1),
            (DataType::U2, BufferType::U2),
            (DataType::U4, BufferType::U4),
            (DataType::U8, BufferType::U8),
            (DataType::U16, BufferType::U16),
            (DataType::U32, BufferType::U32),
            (DataType::U64, BufferType::U64),
            (DataType::F4E2M1FN, BufferType::F4E2M1FN),
            (DataType::F6E2M3FN, BufferType::F6E2M3FN),
            (DataType::F6E3M2FN, BufferType::F6E3M2FN),
            (DataType::F8E3M4, BufferType::F8E3M4),
            (DataType::F8E4M3, BufferType::F8E4M3),
            (DataType::F8E4M3FN, BufferType::F8E4M3FN),
            (DataType::F8E4M3FNUZ, BufferType::F8E4M3FNUZ),
            (DataType::F8E4M3B11FNUZ, BufferType::F8E4M3B11FNUZ),
            (DataType::F8E5M2, BufferType::F8E5M2),
            (DataType::F8E5M2FNUZ, BufferType::F8E5M2FNUZ),
            (DataType::F8E8M0FNU, BufferType::F8E8M0FNU),
            (DataType::BF16, BufferType::BF16),
            (DataType::F16, BufferType::F16),
            (DataType::F32, BufferType::F32),
            (DataType::F64, BufferType::F64),
            (DataType::C64, BufferType::C64),
            (DataType::C128, BufferType::C128),
        ]
    }

    /// Returns a tiled layout with two tiles and its PJRT counterpart.
    fn tiled_layouts() -> (TiledLayout, PjrtTiledLayout) {
        (
            TiledLayout::new(
                vec![1, 0],
                vec![
                    Tile::new(vec![TileDimension::Sized(4), TileDimension::Combined]),
                    Tile::new(vec![TileDimension::Sized(2)]),
                ],
            ),
            PjrtTiledLayout::new(
                vec![1, 0],
                vec![
                    PjrtTile { dimensions: vec![PjrtTileDimension::sized(4), PjrtTileDimension::combined()] },
                    PjrtTile { dimensions: vec![PjrtTileDimension::sized(2)] },
                ],
            ),
        )
    }

    #[test]
    fn test_tile_dimension_to_pjrt() {
        assert_eq!(TileDimension::Sized(4).to_pjrt(), PjrtTileDimension::sized(4));
        assert_eq!(TileDimension::Combined.to_pjrt(), PjrtTileDimension::combined());
    }

    #[test]
    fn test_tile_dimension_from_pjrt() {
        assert_eq!(TileDimension::from_pjrt(PjrtTileDimension::sized(4)), TileDimension::Sized(4));
        assert_eq!(TileDimension::from_pjrt(PjrtTileDimension::combined()), TileDimension::Combined);
    }

    #[test]
    fn test_tile_to_pjrt() {
        assert_eq!(
            Tile::new(vec![TileDimension::Sized(4), TileDimension::Combined]).to_pjrt(),
            PjrtTile { dimensions: vec![PjrtTileDimension::sized(4), PjrtTileDimension::combined()] },
        );
    }

    #[test]
    fn test_tile_from_pjrt() {
        assert_eq!(
            Tile::from_pjrt(PjrtTile { dimensions: vec![PjrtTileDimension::sized(4), PjrtTileDimension::combined()] }),
            Tile::new(vec![TileDimension::Sized(4), TileDimension::Combined]),
        );
    }

    #[test]
    fn test_tiled_layout_to_pjrt() {
        let (tiled_layout, pjrt_tiled_layout) = tiled_layouts();
        assert_eq!(tiled_layout.to_pjrt(), Ok(pjrt_tiled_layout));
    }

    #[test]
    fn test_tiled_layout_from_pjrt() {
        let (tiled_layout, pjrt_tiled_layout) = tiled_layouts();
        assert_eq!(TiledLayout::from_pjrt(pjrt_tiled_layout), Ok(tiled_layout));
    }

    #[test]
    fn test_strided_layout_to_pjrt() {
        assert_eq!(StridedLayout::new(vec![24, 8, -4]).to_pjrt(), Ok(PjrtStridedLayout::new(vec![24, 8, -4])));
    }

    #[test]
    fn test_strided_layout_from_pjrt() {
        assert_eq!(
            StridedLayout::from_pjrt(PjrtStridedLayout::new(vec![24, 8, -4])),
            Ok(StridedLayout::new(vec![24, 8, -4]))
        );
    }

    #[test]
    fn test_layout_to_pjrt() {
        let (tiled_layout, pjrt_tiled_layout) = tiled_layouts();
        assert_eq!(Layout::Tiled(tiled_layout).to_pjrt(), Ok(PjrtLayout::Tiled(pjrt_tiled_layout)));
        assert_eq!(
            Layout::Strided(StridedLayout::new(vec![24, 8, -4])).to_pjrt(),
            Ok(PjrtLayout::Strided(PjrtStridedLayout::new(vec![24, 8, -4]))),
        );
    }

    #[test]
    fn test_layout_from_pjrt() {
        let (tiled_layout, pjrt_tiled_layout) = tiled_layouts();
        assert_eq!(Layout::from_pjrt(PjrtLayout::Tiled(pjrt_tiled_layout)), Ok(Layout::Tiled(tiled_layout)));
        assert_eq!(
            Layout::from_pjrt(PjrtLayout::Strided(PjrtStridedLayout::new(vec![24, 8, -4]))),
            Ok(Layout::Strided(StridedLayout::new(vec![24, 8, -4]))),
        );
    }

    #[test]
    fn test_data_type_to_pjrt() {
        // Zero-space values have no PJRT type and use a predicate carrier. One-bit integer types retain their
        // native PJRT representations independently of the carrier selected by an executable.
        assert_eq!(DataType::Zero.to_pjrt(), BufferType::Predicate);
        for (data_type, buffer_type) in data_type_buffer_types() {
            assert_eq!(data_type.to_pjrt(), buffer_type);
        }
    }

    #[test]
    fn test_data_type_from_pjrt() {
        for (data_type, buffer_type) in data_type_buffer_types() {
            assert_eq!(DataType::from_pjrt(buffer_type), Ok::<_, DataTypeError>(data_type));
        }
        assert!(matches!(
            DataType::from_pjrt(BufferType::Invalid),
            Err(DataTypeError::InvalidDataType { message, .. }) if message == "invalid data type from PJRT: `invalid`",
        ));
    }

    #[test]
    fn test_data_type_executable_buffer_type() {
        assert_eq!(DataType::I1.executable_buffer_type(), BufferType::Predicate);
        assert_eq!(DataType::U1.executable_buffer_type(), BufferType::Predicate);
        assert_eq!(DataType::Boolean.executable_buffer_type(), BufferType::Predicate);
        assert_eq!(DataType::I2.executable_buffer_type(), BufferType::I2);
    }

    #[test]
    fn test_data_type_size_in_bytes() {
        assert_eq!(DataType::Zero.size_in_bytes().unwrap(), 0);
        assert_eq!(DataType::Token.size_in_bytes().unwrap(), 0);
        assert_eq!(DataType::I1.size_in_bytes().unwrap(), 1);
        assert_eq!(DataType::F32.size_in_bytes().unwrap(), size_of::<f32>());
    }

    #[test]
    fn test_array_type_dense_size_in_bytes() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let scalar_type = ArrayType::scalar(DataType::C128);
        let token_type = ArrayType::new(DataType::Token, Shape::new(vec![Dimension::Static(4)]));
        let zero_type = ArrayType::new(DataType::Zero, matrix_type.shape().clone());

        assert_eq!(matrix_type.dense_size_in_bytes().unwrap(), 24);
        assert_eq!(scalar_type.dense_size_in_bytes().unwrap(), 16);
        assert_eq!(token_type.dense_size_in_bytes().unwrap(), 0);
        assert_eq!(zero_type.dense_size_in_bytes().unwrap(), 0);

        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("dynamic", DimensionBounds::unbounded()).into()]),
        );
        let error = dynamic_type.dense_size_in_bytes().unwrap_err();
        assert!(matches!(&error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::DynamicShape { shape } if shape == dynamic_type.shape(),
            ),
        ));
        assert_eq!(error.to_string(), "expected static shape but got `[dynamic]`");

        let oversized_type = ArrayType::new(DataType::U16, Shape::new(vec![Dimension::Static(usize::MAX)]));
        let error = oversized_type.dense_size_in_bytes().unwrap_err();
        let message = format!(
            "dense byte size for array type `{oversized_type}` exceeds the maximum allowed size of {}",
            usize::MAX,
        );
        assert!(matches!(&error, Error::SizeLimitExceeded { message: actual } if actual == &message));
        assert_eq!(error.to_string(), message);
    }
}
