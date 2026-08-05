pub mod arrays;
pub mod data;
pub mod dimensions;
pub mod layouts;
pub mod memories;

pub use arrays::{ArrayIrType, ArrayIrTypeRefinements, ArrayType, ArrayTypeRefinements};
pub use data::{DataType, DataTypeError};
pub use dimensions::{
    Dimension, DimensionBounds, DimensionError, DimensionType, DimensionVariable, MAX_DIMENSION_EXTENT, Shape,
    StaticShape,
};
pub use layouts::{Layout, LayoutError, StridedLayout, Tile, TileDimension, TiledLayout};
pub use memories::Memory;
