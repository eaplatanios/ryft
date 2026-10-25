pub mod arrays;
pub mod data;
pub mod dimensions;
pub mod ir;
pub mod layouts;
pub mod memories;

pub use arrays::{ArrayMemberType, ArrayType, ArrayTypeRefinements};
pub use data::{DataType, DataTypeError};
pub use dimensions::{
    Dimension, DimensionBounds, DimensionError, DimensionMemberType, DimensionRefinements, DimensionType,
    DimensionTypeRefinements, DimensionVariable, MAX_DIMENSION_EXTENT, Shape, StaticShape,
};
pub use ir::{ArrayIrType, ArrayIrTypeRefinements};
pub use layouts::{Layout, LayoutError, StridedLayout, Tile, TileDimension, TiledLayout};
pub use memories::Memory;
