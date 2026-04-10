use ryft_pjrt::{
    BufferType, Layout as PjrtLayout, StridedLayout as PjrtStridedLayout, Tile as PjrtTile,
    TileDimension as PjrtTileDimension, TiledLayout as PjrtTiledLayout,
};

use ryft_core::types::data_types::{DataType, DataTypeError};
use ryft_core::types::layouts::{Layout, LayoutError, StridedLayout, Tile, TileDimension, TiledLayout};

use crate::pjrt::{FromPjrt, ToPjrt};

impl ToPjrt for TileDimension {
    type Output = PjrtTileDimension;

    fn to_pjrt(&self) -> PjrtTileDimension {
        match self {
            TileDimension::Sized(size) => PjrtTileDimension::sized(*size),
            TileDimension::Combined => PjrtTileDimension::combined(),
        }
    }
}

impl FromPjrt<PjrtTileDimension> for TileDimension {
    type Output = Self;

    fn from_pjrt(value: PjrtTileDimension) -> Self {
        match value.size() {
            Some(size) => Self::Sized(size),
            None => Self::Combined,
        }
    }
}

impl ToPjrt for Tile {
    type Output = PjrtTile;

    fn to_pjrt(&self) -> PjrtTile {
        PjrtTile { dimensions: self.dimensions.iter().map(ToPjrt::to_pjrt).collect() }
    }
}

impl FromPjrt<PjrtTile> for Tile {
    type Output = Self;

    fn from_pjrt(value: PjrtTile) -> Self {
        Self::new(value.dimensions.into_iter().map(TileDimension::from_pjrt).collect())
    }
}

impl ToPjrt for TiledLayout {
    type Output = Result<PjrtTiledLayout, LayoutError>;

    fn to_pjrt(&self) -> Result<PjrtTiledLayout, LayoutError> {
        Ok(PjrtTiledLayout::new(
            self.minor_to_major
                .iter()
                .copied()
                .map(|dimension| {
                    u64::try_from(dimension).map_err(|_| LayoutError::InvalidLayout {
                        message: format!(
                            "invalid minor-to-major dimension index for PJRT: '{dimension}' is out of range",
                        ),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            self.tiles.iter().map(ToPjrt::to_pjrt).collect(),
        ))
    }
}

impl FromPjrt<PjrtTiledLayout> for TiledLayout {
    type Output = Result<Self, LayoutError>;

    fn from_pjrt(value: PjrtTiledLayout) -> Result<Self, LayoutError> {
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
            value.tiles().into_iter().map(Tile::from_pjrt).collect(),
        ))
    }
}

impl ToPjrt for StridedLayout {
    type Output = Result<PjrtStridedLayout, LayoutError>;

    fn to_pjrt(&self) -> Result<PjrtStridedLayout, LayoutError> {
        Ok(PjrtStridedLayout::new(
            self.strides
                .iter()
                .copied()
                .map(|stride| {
                    i64::try_from(stride).map_err(|_| LayoutError::InvalidLayout {
                        message: format!("invalid stride for PJRT: '{stride}' is out of range"),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl FromPjrt<PjrtStridedLayout> for StridedLayout {
    type Output = Result<Self, LayoutError>;

    fn from_pjrt(value: PjrtStridedLayout) -> Result<Self, LayoutError> {
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

impl ToPjrt for Layout {
    type Output = Result<PjrtLayout, LayoutError>;

    fn to_pjrt(&self) -> Result<PjrtLayout, LayoutError> {
        match self {
            Layout::Tiled(layout) => Ok(PjrtLayout::Tiled(layout.to_pjrt()?)),
            Layout::Strided(layout) => Ok(PjrtLayout::Strided(layout.to_pjrt()?)),
        }
    }
}

impl FromPjrt<PjrtLayout> for Layout {
    type Output = Result<Self, LayoutError>;

    fn from_pjrt(value: PjrtLayout) -> Result<Self, LayoutError> {
        match value {
            PjrtLayout::Tiled(layout) => Ok(Self::Tiled(TiledLayout::from_pjrt(layout)?)),
            PjrtLayout::Strided(layout) => Ok(Self::Strided(StridedLayout::from_pjrt(layout)?)),
        }
    }
}

impl ToPjrt for DataType {
    type Output = BufferType;

    fn to_pjrt(&self) -> BufferType {
        match self {
            DataType::Token => BufferType::Token,
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

    fn from_pjrt(value: BufferType) -> Result<Self, DataTypeError> {
        match value {
            BufferType::Invalid => Err(DataTypeError::InvalidDataType {
                message: format!("invalid data type from PJRT: '{value}'"),
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_pjrt::{
        BufferType, Layout as PjrtLayout, StridedLayout as PjrtStridedLayout, Tile as PjrtTile,
        TileDimension as PjrtTileDimension, TiledLayout as PjrtTiledLayout,
    };

    use ryft_core::types::data_types::{DataType, DataTypeError};
    use ryft_core::types::layouts::{Layout, StridedLayout, Tile, TileDimension, TiledLayout};

    use super::*;

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
        assert_eq!(TiledLayout::from_pjrt(pjrt_tiled_layout.clone()), Ok(tiled_layout.clone()));
        assert_eq!(tiled_layout.to_pjrt(), Ok(pjrt_tiled_layout.clone()));

        let strided_layout = StridedLayout::new(vec![24, 8, -4]);
        let pjrt_strided_layout = PjrtStridedLayout::new(vec![24, 8, -4]);
        assert_eq!(StridedLayout::from_pjrt(pjrt_strided_layout.clone()), Ok(strided_layout.clone()));
        assert_eq!(strided_layout.to_pjrt(), Ok(pjrt_strided_layout.clone()));

        let tiled_layout = Layout::Tiled(tiled_layout);
        let pjrt_tiled_layout = PjrtLayout::Tiled(pjrt_tiled_layout);
        assert_eq!(Layout::from_pjrt(pjrt_tiled_layout.clone()), Ok(tiled_layout.clone()));
        assert_eq!(tiled_layout.to_pjrt(), Ok(pjrt_tiled_layout));

        let strided_layout = Layout::Strided(strided_layout);
        let pjrt_strided_layout = PjrtLayout::Strided(pjrt_strided_layout);
        assert_eq!(Layout::from_pjrt(pjrt_strided_layout.clone()), Ok(strided_layout.clone()));
        assert_eq!(strided_layout.to_pjrt(), Ok(pjrt_strided_layout));
    }

    #[test]
    fn test_data_type_from_and_to_pjrt_buffer_type() {
        assert!(matches!(
            DataType::from_pjrt(BufferType::Invalid),
            Err(DataTypeError::InvalidDataType { message, .. }) if message == "invalid data type from PJRT: 'invalid'",
        ));
        for &(data_type, buffer_type) in &[
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
        ] {
            assert_eq!(DataType::from_pjrt(buffer_type), Ok::<_, DataTypeError>(data_type));
            assert_eq!(data_type.to_pjrt(), buffer_type);
        }
    }
}
