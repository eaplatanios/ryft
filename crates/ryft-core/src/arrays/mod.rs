pub mod addressing;
pub mod arrays;
pub mod batching;
pub mod broadcasting;
pub mod differentiation;
pub mod dimensions;
pub mod encoding;
pub mod ir;
pub mod macros;
pub mod operations;
pub mod reference_analysis;
pub mod reference_discharge;
pub mod reference_views;
pub mod sharding;
pub mod types;

pub use addressing::{ArrayAddressing, ArrayIndexRange, ArrayIndexRanges, ArraySliceAxis};
pub use arrays::Array;
pub use batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatching, DimensionSource,
    RaggedArrayBatchingPolicy, RaggedAxis, RaggedMaskIdentity, ReplicatedDimensionBatchingPolicy,
    StaticArrayBatchingPolicy,
};
pub use broadcasting::{Broadcastable, BroadcastingError};
pub use differentiation::{ExactShape, ExactShapeDimension, LinearResiduals, materialize_array_tangent};
pub use dimensions::DimensionValue;
pub use encoding::{
    ArrayElement, Complex, bf16, decode_elements, decode_logical_bytes, encode_elements, encode_logical_bytes,
    f4e2m1fn, f6e2m3fn, f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz, f8e8m0fnu,
    f16, i1, i2, i4, u1, u2, u4, validate_storage_bytes,
};
pub use ir::ArrayIrValue;
pub use macros::dispatch_on_array_element_type;
pub use operations::{
    ArrayIrOperation, ArrayIrOperations, ArrayOperation, ArrayOperations, ArrayReferenceViewOperation,
    ArrayTracingContext, DimensionOperation, DimensionOperations, DimensionTracingContext,
    REFERENCE_INDEX_OPERATION_NAME, REFERENCE_SLICE_OPERATION_NAME, ReferenceIndex, ReferenceIndexOperation,
    ReferenceSlice, ReferenceSliceOperation,
};
pub use reference_analysis::{ArrayReferenceAnalysis, ArrayReferenceAnalysisError};
pub use reference_discharge::ArrayReferenceDischarge;
pub use reference_views::{ArrayReference, ArrayReferenceView, ArrayReferenceViewError, ArrayReferenceViewTransform};
pub use sharding::{
    Device, DeviceId, DeviceMesh, LogicalMesh, MeshAxis, MeshAxisType, ProcessIndex, Sharding, ShardingDimension,
    ShardingError, ShardingVisualization,
};
pub use types::{
    ArrayIrType, ArrayIrTypeRefinements, ArrayType, ArrayTypeRefinements, DataType, DataTypeError, Dimension,
    DimensionBounds, DimensionError, DimensionType, DimensionVariable, Layout, LayoutError, MAX_DIMENSION_EXTENT,
    Memory, Shape, StaticShape, StridedLayout, Tile, TileDimension, TiledLayout,
};
