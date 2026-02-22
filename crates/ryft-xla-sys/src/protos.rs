use std::collections::HashMap;

use prost::{Enumeration, Message, Oneof};
use prost_types::{Any as ProtoAny, Duration};

/// Topology of a set of PJRT devices.
///
/// This type corresponds to `PjRtTopologyDescriptionProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
#[prost(reserved = "5 to 8")]
pub struct Topology {
    /// ID that identifies the platform of this topology.
    #[prost(uint64, tag = "1")]
    pub platform_id: u64,

    /// Name that identifies the platform of this topology (e.g., `"cpu"`, `"gpu"`, `"tpu"`, etc.).
    #[prost(string, tag = "2")]
    pub platform_name: String,

    /// String that contains human-readable, platform-specific, version information for this topology
    /// (e.g., the CUDA version for GPU topologies or the `libtpu` version for TPU topologies).
    #[prost(string, tag = "3")]
    pub platform_version: String,

    /// Boolean flag that indicates whether the topology represents a subslice.
    #[prost(bool, tag = "4")]
    pub is_subslice_topology: bool,

    /// Platform-specific Protobuf representation of the topology that may contain additional information.
    #[prost(message, optional, tag = "9")]
    pub platform_specific_topology: Option<ProtoAny>,
}

/// Topology of CPU devices.
///
/// This type corresponds to `CpuTopologyProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct CpuTopology {
    /// CPU devices that belong to this topology.
    #[prost(message, repeated, tag = "1")]
    pub cpu_devices: Vec<CpuTopologyDevice>,

    /// Machine-level attributes associated with this topology.
    #[prost(string, repeated, tag = "4")]
    pub machine_attributes: Vec<String>,
}

/// Description of a CPU device in [`CpuTopology`].
///
/// This type corresponds to `CpuTopologyProto.CpuDevice` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct CpuTopologyDevice {
    /// Process index to which this device belongs.
    #[prost(int32, tag = "2")]
    pub process_index: i32,

    /// Local hardware identifier of this device within its process.
    #[prost(int32, tag = "3")]
    pub local_hardware_id: i32,
}

/// Topology of GPU devices.
///
/// This type corresponds to `GpuTopologyProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
#[prost(reserved = "1, 2, 7")]
pub struct GpuTopology {
    /// GPU platform version (e.g., `"NVIDIA A100-SXM4-40GB"`).
    #[prost(string, tag = "3")]
    pub platform_version: String,

    /// Number of partitions in this topology.
    #[prost(int32, tag = "4")]
    pub num_partitions: i32,

    /// Number of hosts per partition.
    #[prost(int32, tag = "5")]
    pub num_hosts_per_partition: i32,

    /// Number of devices per host.
    #[prost(int32, tag = "6")]
    pub num_devices_per_host: i32,

    /// GPU target configuration.
    #[prost(message, optional, tag = "8")]
    pub gpu_target_config: Option<GpuTargetConfiguration>,
}

/// Represents the type of data that can be stored in PJRT buffers. Specifically, this represents the type
/// of individual elements/values that can be held in rectangular multidimensional arrays.
///
/// This type corresponds to `PrimitiveType` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum BufferType {
    /// Invalid [`BufferType`] that serves as a default.
    Invalid = 0,

    /// [`BufferType`] that represents token values that are threaded between side-effecting operations.
    /// This type is only used for buffers that contain a single value (i.e., that represent scalar values).
    Token = 17,

    /// Predicate [`BufferType`] that represents the `true` and `false` values.
    Predicate = 1,

    /// [`BufferType`] that represents signed 1-bit integer values. Note that this type is not supported in PJRT.
    I1 = 30,

    /// [`BufferType`] that represents signed 2-bit integer values.
    I2 = 26,

    /// [`BufferType`] that represents signed 4-bit integer values.
    I4 = 21,

    /// [`BufferType`] that represents signed 8-bit integer values.
    I8 = 2,

    /// [`BufferType`] that represents signed 16-bit integer values.
    I16 = 3,

    /// [`BufferType`] that represents signed 32-bit integer values.
    I32 = 4,

    /// [`BufferType`] that represents signed 64-bit integer values.
    I64 = 5,

    /// [`BufferType`] that represents unsigned 1-bit integer values. Note that this type is not supported in PJRT.
    U1 = 31,

    /// [`BufferType`] that represents unsigned 2-bit integer values.
    U2 = 27,

    /// [`BufferType`] that represents unsigned 4-bit integer values.
    U4 = 22,

    /// [`BufferType`] that represents unsigned 8-bit integer values.
    U8 = 6,

    /// [`BufferType`] that represents unsigned 16-bit integer values.
    U16 = 7,

    /// [`BufferType`] that represents unsigned 32-bit integer values.
    U32 = 8,

    /// [`BufferType`] that represents unsigned 64-bit integer values.
    U64 = 9,

    /// [`BufferType`] that represents 4-bit floating-point values that are represented using a
    /// [microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
    /// format with 2 exponent bits and 1 mantissa bit. Only finite values are supported (thus the `FN` suffix).
    /// Unlike IEEE floating-point types, infinity and NaN values are not supported.
    F4E2M1FN = 32,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 3 exponent bits and 4 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E3M4 = 29,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E4M3 = 28,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits, and without support for
    /// representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values are
    /// represented with the exponent and mantissa bits all set to `1`. All other bit configurations represent finite
    /// values.
    F8E4M3FN = 20,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2206.02915) with 4 exponent bits and 3 mantissa bits, and without support for
    /// representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values are
    /// represented with the exponent and mantissa bits all set to `0` and the sign bit is set to `1`. All other bit
    /// configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    ///
    /// The difference between this type and [`BufferType::F8E4M3FN`] is that there is an additional exponent value
    /// available. To keep the same dynamic range as an IEEE-like 8-bit floating-point type, the exponent is biased one
    /// more than would be expected given the number of exponent bits (i.e., bias set to `8`).
    F8E4M3FNUZ = 25,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits and a bias of `11`, and
    /// without support for representing infinity values, unlike existing IEEE floating-point types (thus the `FN`
    /// suffix). NaN values are represented with the exponent and mantissa bits all set to `0` and the sign bit is set
    /// to `1`. All other bit configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    F8E4M3B11FNUZ = 23,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 5 exponent bits and 2 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E5M2 = 19,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2206.02915) with 5 exponent bits and 2 mantissa bits, and without support for
    /// representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values are
    /// represented with the exponent and mantissa bits all set to `0` and the sign bit is set to `1`. All other bit
    /// configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    ///
    /// The difference between this type and [`BufferType::F8E5M2`] is that there is an additional exponent value
    /// available. To keep the same dynamic range as an IEEE-like 8-bit floating-point type, the exponent is biased one
    /// more than would be expected given the number of exponent bits (i.e., bias set to `16`).
    F8E5M2FNUZ = 24,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using a
    /// [microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
    /// format with 8 exponent bits and no mantissa or sign bits. Only unsigned finite values are supported
    /// (thus the `FNU` suffix). Unlike IEEE floating-point types, infinity and NaN values are not supported.
    F8E8M0FNU = 33,

    /// [`BufferType`] that represents 16-bit floating-point values with 8 exponent bits, 7 mantissa bits, and 1 sign
    /// bit. This type offers a larger dynamic range than [`BufferType::F16`] at the cost of lower precision.
    BF16 = 16,

    /// [`BufferType`] that represents 16-bit floating-point values with 5 exponent bits, 10 mantissa bits, and 1 sign
    /// bit, using the standard IEEE floating-point representation.
    F16 = 10,

    /// [`BufferType`] that represents 32-bit floating-point values with 8 exponent bits, 24 mantissa bits, and 1 sign
    /// bit, using the standard IEEE floating-point representation.
    F32 = 11,

    /// [`BufferType`] that represents 64-bit floating-point values with 11 exponent bits, 53 mantissa bits, and 1 sign
    /// bit, using the standard IEEE floating-point representation.
    F64 = 12,

    /// [`BufferType`] that represents 64-bit complex-valued floating-point values as pairs of
    /// 32-bit real floating-point values.
    C64 = 15,

    /// [`BufferType`] that represents 128-bit complex-valued floating-point values as pairs of
    /// 64-bit real floating-point values.
    C128 = 18,

    /// [`BufferType`] that represents heterogeneous (i.e., polymorphic) sequences of values. This is used for things
    /// like returning multiple values from a computation and representing them as a single buffer. Note that this type
    /// is not supported in PJRT.
    Tuple = 13,

    /// [`BufferType`] that represents opaque data and which is used for passing context-specific data to custom
    /// operations. Note that this type is not supported in PJRT.
    OpaqueType = 14,

    /// [`BufferType`] that represents a buffer of values. [`Shape`]s that have this as their [`Shape::element_type`]
    /// have the underlying value types specified in [`Shape::tuple_shapes`].
    Buffer = 34,
}

/// Shape of an array that describes the number of dimensions in the array, the size of each dimension,
/// and the type of the elements in the array.
///
/// Tuple shapes are represented as rank zero [`Shape`]s with populated [`Shape::tuple_shapes`].
///
/// Refer to [the official XLA documentation](https://openxla.org/xla/shapes) for more information on shapes.
///
/// This type corresponds to `ShapeProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct Shape {
    /// [`BufferType`] of the array elements.
    #[prost(enumeration = "BufferType", tag = "2")]
    pub element_type: i32,

    /// Size (i.e., number of elements) for each dimension in this shape, or an upper bound on the size if the
    /// corresponding dimension is dynamically-sized. In XLA, dimensions are numbered from `0` to `N - 1` for
    /// `N`-dimensional arrays. The first element of this vector corresponds to the size of dimension `0`, the second
    /// corresponds to the size of dimension `1`, etc. An empty vector indicates that this is the [`Shape`] of a scalar
    /// value. Each element in this list must be non-negative. `0` is considered a valid dimension size.
    #[prost(int64, repeated, tag = "3")]
    pub dimensions: Vec<i64>,

    /// Boolean value indicating whether each dimension of this [`Shape`] is dynamically-sized. This vector has the same
    /// size as [`Shape::dimensions`] and if a value is `true`, that means that the corresponding value in `dimensions`
    /// is an upper-bound on the size of that dimension, rather than the actual size.
    #[prost(bool, repeated, tag = "6")]
    pub is_dynamic_dimension: Vec<bool>,

    /// This is only used for [`Shape`]s of tuples and contains the [`Shape`] of each value in the tuple. If this vector
    /// is non-empty, then [`Shape::dimensions`] and [`Shape::is_dynamic_dimension`] must be empty. Similarly, if those
    /// two vectors are non-empty, then this vector must be empty (i.e., they are _mutually exclusive_).
    #[prost(message, repeated, tag = "4")]
    pub tuple_shapes: Vec<Shape>,

    /// [`Layout`] that is used to back this shape and that represents how the values in the corresponding array
    /// are laid out in memory.
    #[prost(message, optional, boxed, tag = "5")]
    pub layout: Option<Box<Layout>>,
}

/// Describes the encoding type of a "level" in a sparse [`Layout`]. XLA uses "dimension" to refer to the axes of the
/// semantic tensor, and "level" to refer to the axes of the actual storage format (i.e., the operational representation
/// of the sparse tensor in memory). The number of dimensions is usually the same as the number of levels, such as in
/// the compressed sparse row (CSR) storage format. However, the encoding can also map dimensions to higher-order
/// levels, such as in the block-sparse row (BSR) storage format, or to lower-order levels, such as when linearizing
/// dimensions as a single level in the storage. For more information on the different encoding types, refer to
/// [this page](https://developers.google.com/mlir-sparsifier/guides/encode).
///
/// This type corresponds to `DimLevelType` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum DimensionLevel {
    /// The level is dense (i.e., all entries along this level are stored).
    Dense = 0,

    /// The level is compressed (i.e., only non-zero entries along this level are stored). For a compressed level,
    /// each position (i.e., offset into the storage format) interval is represented compactly with a lower bound,
    /// `position(i)`, and an upper bound, `position(i + 1) - 1`, which implies that successive intervals must appear
    /// in order, without any "holes" between them.
    Compressed = 1,

    /// The corresponding dimension is compressed, but with potential trailing zeros. This level type relaxes the
    /// constraints of the [`DimensionLevel::Compressed`] level type by representing each position interval with
    /// a lower bound, `low(i)`, and an upper bound, `high(i)`, which allows intervals to appear in arbitrary order and
    /// with "elbow" room between them.
    LooseCompressed = 3,

    /// The level is a variant of [`DimensionLevel::Compressed`] that contains a single coordinate (i.e., an index
    /// that is stored explicitly), with no sibling elements for each parent.
    Singleton = 2,
}

/// Tile used in a tiling-based [`Layout`]. For more information refer to
/// [this page](https://openxla.org/xla/tiled_layout).
///
/// This type corresponds to `TileProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Eq, Hash, Message)]
pub struct Tile {
    /// Number of elements in each dimension of the [`Tile`], ordered from the most major dimension of the tile to the
    /// most minor dimension of the tile. The dimensions of a tile correspond to a suffix of the dimensions of the
    /// [`Shape`] that is being tiled.
    #[prost(int64, repeated, tag = "1")]
    pub dimensions: Vec<i64>,
}

/// Describes how data is split between different memory spaces. Each [`DimensionSplit`] instance
/// represents a split along one dimension of an array.
///
/// This type corresponds to `SplitConfigProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Eq, Hash, Message)]
pub struct DimensionSplit {
    /// Dimension along which an array is split. This refers to a physical dimension such that `0` is the most major
    /// dimension and `rank - 1` is the most minor dimension.
    #[prost(int64, tag = "1")]
    pub dimension: i64,

    /// Indices which specify the points where the splits occur. For example, if the size of the dimension being split
    /// is `1024`, a `split_indices` value of `[512]` indicates that that dimension is split in two equal parts right
    /// in the middle.
    #[prost(int64, repeated, tag = "2")]
    pub split_indices: Vec<i64>,
}

/// Describes how an array is laid out in memory.
///
/// Refer to [the official XLA documentation](https://openxla.org/xla/shapes) for more information on layouts.
///
/// This type corresponds to `LayoutProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct Layout {
    /// Sequence of dimension numbers ordered from the most minor (i.e., the one with the fastest varying index) to the
    /// most major (i.e., the one with the slowest varying index).
    #[prost(int64, repeated, tag = "1")]
    pub minor_to_major: Vec<i64>,

    /// Sparse dimension encoding levels. Refer to the documentation of [`DimensionLevel`] for more information.
    /// This is only relevant for sparse [`Layout`]s.
    #[prost(enumeration = "DimensionLevel", repeated, tag = "9")]
    pub dimension_levels: Vec<i32>,

    /// Boolean value for each dimension level that indicates if that dimension level is "unique". This vector is
    /// either parallel to [`Layout::dimension_levels`] or empty (in which case all dimension levels are assumed to be
    /// unique). A dimension level is "unique" if there are no duplicate coordinates (i.e., indices that are stored
    /// explicitly) in the storage. This is only relevant for sparse [`Layout`]s.
    #[prost(bool, repeated, tag = "13")]
    pub is_dimension_level_unique: Vec<bool>,

    /// Boolean value for each dimension level that indicates if that dimension level is "ordered". This vector is
    /// either parallel to [`Layout::dimension_levels`] or empty (in which case all dimension levels are assumed to be
    /// ordered). A dimension level is "ordered" if the coordinates (i.e., indices that are stored explicitly) are
    /// sorted in the storage and can thus not appear in an arbitrary order. This is only relevant for sparse
    /// [`Layout`]s.
    #[prost(bool, repeated, tag = "14")]
    pub is_dimension_level_ordered: Vec<bool>,

    /// Sequence of [`Tile`]s that are used in this layout. The tiles are nested with the outermost tiling being the
    /// first tiling in the sequence. For more information on tiling-based [`Layout`]s refer to
    /// [this page](https://openxla.org/xla/tiled_layout).
    #[prost(message, repeated, tag = "6")]
    pub tiles: Vec<Tile>,

    /// [`DimensionSplit`]s specify how the underlying data is split between different memory spaces.
    #[prost(message, repeated, tag = "17")]
    pub splits: Vec<DimensionSplit>,

    /// The array is padded at the end so that its total number of elements is a multiple of this number. Tiling
    /// effectively pads, reshapes, and transposes the shape of an array to another shape. This alignment additionally
    /// pads the resulting array so that its total number of elements is a multiple of the specified number. This is
    /// useful in cases where we want a layout that does not tile the data but still requires it to be padded to a
    /// certain number of elements. If omitted or set to `0`, it will be treated as `1`.
    #[prost(int64, optional, tag = "16")]
    pub alignment: Option<i64>,

    /// Number of bits of each element.
    #[prost(int64, optional, tag = "7")]
    pub element_size_in_bits: Option<i64>,

    /// Memory space in which the array resides. The integer value is interpreted in a backend-specific manner.
    #[prost(int64, optional, tag = "8")]
    pub memory_space: Option<i64>,

    /// [`BufferType`] used for indices into a sparse array. This must be set to one of the unsigned integer types
    /// (i.e., [`BufferType::U1`], [`BufferType::U2`], [`BufferType::U4`], [`BufferType::U8`], [`BufferType::U16`],
    /// [`BufferType::U32`], or [`BufferType::U64`]). If not provided, then the XLA compiler will use the largest
    /// unsigned integer that is naturally supported by the target device. This is only relevant for sparse [`Layout`]s.
    #[prost(enumeration = "BufferType", optional, tag = "11")]
    pub index_type: Option<i32>,

    /// [`BufferType`] used for pointers into a sparse array. This must be set to one of the unsigned integer types
    /// (i.e., [`BufferType::U1`], [`BufferType::U2`], [`BufferType::U4`], [`BufferType::U8`], [`BufferType::U16`],
    /// [`BufferType::U32`], or [`BufferType::U64`]). If not provided, then the XLA compiler will use the largest
    /// unsigned integer that is naturally supported by the target device. This is only relevant for sparse [`Layout`]s.
    #[prost(enumeration = "BufferType", optional, tag = "12")]
    pub pointer_type: Option<i32>,

    /// Optional physical, on-device, dense [`Shape`] used to represent the shape this sparse [`Layout`] belongs to.
    /// This is only relevant for sparse [`Layout`]s.
    #[prost(message, optional, boxed, tag = "10")]
    pub physical_shape: Option<Box<Shape>>,

    /// Size of the dynamic shape metadata that is in front of the shape data in bytes. This field may be non-zero for
    /// a static shape whose associated buffer is for a dynamic shape (e.g., as a result of the `SliceToDynamic`
    /// operation).
    #[prost(int64, optional, tag = "15")]
    pub dynamic_shape_metadata_prefix_size_in_bytes: Option<i64>,
}

/// Type of optimization profile that can be associated with an operation.
///
/// This type corresponds to `ProfileType` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum ProfileType {
    /// Invalid profile type (default).
    Invalid = 0,

    /// Window-based profile.
    Window = 1,

    /// Flag-style profile.
    Flag = 2,

    /// Integer-valued profile.
    Integer = 3,
}

/// Source of the optimization profile associated with an operation.
///
/// This type corresponds to `ProfileSource` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum ProfileSource {
    /// Unknown profile source.
    UnknownSource = 0,

    /// Profile is embedded in the executable.
    Embedded = 1,

    /// Profile is retrieved remotely.
    Remote = 2,
}

/// Compilation event that triggered profile usage.
///
/// This type corresponds to `CompilationEvent` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum CompilationEvent {
    /// Unknown compilation event.
    UnknownEvent = 0,

    /// First compilation of a program.
    FirstCompilation = 1,

    /// Recompilation of a program.
    Recompilation = 2,
}

/// Strategy used to generate an optimization profile.
///
/// This type corresponds to `ProfileGenerationStrategy` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum ProfileGenerationStrategy {
    /// Unknown profile generation strategy.
    Unknown = 0,

    /// Genetic algorithm profile search.
    Ga = 1,

    /// FANTA profile search.
    Fanta = 2,

    /// CFO profile search.
    Cfo = 3,

    /// Exhaustive profile search.
    Exhaustive = 4,

    /// Learned cost-model profile search using a Graph Neural Network.
    LcmGnn = 5,

    /// Learned cost-model profile search using a Mixture-of-Experts model.
    LcmMoe = 6,
}

/// Profile details associated with an [`OpMetadata`] instance.
///
/// This type corresponds to `OpMetadata.ProfileInfo` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct OpMetadataProfileInfo {
    /// Types of optimization profiles that apply to the operation.
    #[prost(enumeration = "ProfileType", repeated, tag = "1")]
    pub profile_type: Vec<i32>,

    /// Relative speedup of the tuned configuration compared to the default configuration.
    #[prost(double, tag = "2")]
    pub relative_speedup: f64,

    /// Source of the profile information.
    #[prost(enumeration = "ProfileSource", tag = "3")]
    pub profile_source: i32,

    /// Compilation event that triggered usage of this profile.
    #[prost(enumeration = "CompilationEvent", tag = "4")]
    pub compilation_event: i32,

    /// Strategy used to produce this profile.
    #[prost(enumeration = "ProfileGenerationStrategy", tag = "5")]
    pub profile_generation_strategy: i32,
}

/// Symbolization metadata attached to an HLO operation.
///
/// This type corresponds to `OpMetadata` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
#[prost(reserved = "6, 7, 11, 13, 14")]
pub struct OpMetadata {
    /// Framework operation type that generated this XLA operation.
    #[prost(string, tag = "1")]
    pub op_type: String,

    /// User-visible operation name.
    #[prost(string, tag = "2")]
    pub op_name: String,

    /// Source file associated with this operation.
    #[prost(string, tag = "3")]
    pub source_file: String,

    /// Source line number associated with this operation.
    #[prost(int32, tag = "4")]
    pub source_line: i32,

    /// Ending source line number associated with this operation.
    #[prost(int32, tag = "17")]
    pub source_end_line: i32,

    /// Source column number associated with this operation.
    #[prost(int32, tag = "18")]
    pub source_column: i32,

    /// Ending source column number associated with this operation.
    #[prost(int32, tag = "19")]
    pub source_end_column: i32,

    /// Deprecated profile type annotations.
    #[deprecated]
    #[prost(enumeration = "ProfileType", repeated, tag = "5")]
    pub profile_type: Vec<i32>,

    /// Size of generated code for this operation (in bytes).
    #[prost(int64, tag = "8")]
    pub size_of_generated_code_in_bytes: i64,

    /// Size of the operation working set in fast device memory (in bytes).
    #[prost(int64, tag = "9")]
    pub size_of_memory_working_set_in_bytes: i64,

    /// Profile information associated with this operation.
    #[prost(message, optional, tag = "10")]
    pub profile_info: Option<OpMetadataProfileInfo>,

    /// Deduplicated operation name used for grouping equivalent operations.
    #[prost(string, tag = "12")]
    pub deduplicated_name: String,

    /// 1-based stack frame index associated with this operation.
    #[prost(int32, tag = "15")]
    pub stack_frame_id: i32,

    /// Scheduled instruction name for this operation.
    #[prost(string, tag = "16")]
    pub scheduling_name: String,
}

/// Axis in a device [`Mesh`].
///
/// This type corresponds to `MeshProto.MeshAxis` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Eq, Hash, Message)]
pub struct MeshAxis {
    /// Axis name.
    #[prost(string, tag = "1")]
    pub name: String,

    /// Axis size.
    #[prost(int64, tag = "2")]
    pub size: i64,
}

/// Logical mesh used by named shardings.
///
/// This type corresponds to `MeshProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct Mesh {
    /// Mesh axes.
    #[prost(message, repeated, tag = "1")]
    pub axes: Vec<MeshAxis>,

    /// Optional explicit device ordering for the mesh.
    #[prost(int64, repeated, tag = "2")]
    pub device_ids: Vec<i64>,
}

/// Sub-axis reference for a split mesh axis.
///
/// This type corresponds to `AxisRefProto.SubAxis` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct AxisReferenceSubAxis {
    /// Product of axis sizes to the left of the sub-axis.
    #[prost(int64, tag = "1")]
    pub pre_size: i64,

    /// Size of the referenced sub-axis.
    #[prost(int64, tag = "2")]
    pub size: i64,
}

/// Reference to a full mesh axis or a split sub-axis.
///
/// This type corresponds to `AxisRefProto` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct AxisReference {
    /// Index into [`Mesh::axes`].
    #[prost(int64, tag = "1")]
    pub mesh_axis_index: i64,

    /// Optional split sub-axis descriptor.
    #[prost(message, optional, tag = "2")]
    pub sub_axis_info: Option<AxisReferenceSubAxis>,
}

/// Sharding of a single logical tensor dimension in a [`NamedSharding`].
///
/// This type corresponds to `NamedShardingProto.DimensionSharding` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct NamedShardingDimension {
    /// Ordered mesh axes used to shard this dimension from major to minor.
    #[prost(message, repeated, tag = "1")]
    pub axes: Vec<AxisReference>,

    /// If `true`, this dimension is closed and cannot be further sharded.
    #[prost(bool, tag = "2")]
    pub is_closed: bool,
}

/// Named sharding representation bound to a specific [`Mesh`].
///
/// This type corresponds to `NamedShardingProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
#[prost(reserved = "1")]
pub struct NamedSharding {
    /// Mesh used by this named sharding.
    #[prost(message, optional, tag = "2")]
    pub mesh: Option<Mesh>,

    /// Per-dimension sharding assignments.
    #[prost(message, repeated, tag = "3")]
    pub dim_shardings: Vec<NamedShardingDimension>,

    /// Explicitly replicated mesh axes.
    #[prost(message, repeated, tag = "4")]
    pub replicated_axes: Vec<AxisReference>,

    /// Mesh axes along which values are unreduced.
    #[prost(message, repeated, tag = "5")]
    pub unreduced_axes: Vec<AxisReference>,

    /// Metadata that records the origin of this sharding.
    #[prost(message, repeated, tag = "6")]
    pub metadata: Vec<OpMetadata>,

    /// Mesh axes that are user-controlled.
    #[prost(message, repeated, tag = "7")]
    pub manual_axes: Vec<AxisReference>,
}

/// Type of an [`OpSharding`].
///
/// This type corresponds to `OpSharding.Type` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum OpShardingType {
    /// Replicated across all devices.
    Replicated = 0,

    /// Maximal sharding (single device executes the operation).
    Maximal = 1,

    /// Tuple sharding where only [`OpSharding::tuple_shardings`] is meaningful.
    Tuple = 2,

    /// Tiled sharding described by tile shape and assignments.
    Other = 3,

    /// Manually sharded operation.
    Manual = 4,

    /// Placeholder sharding with lowest precedence.
    Unknown = 5,

    /// Unreduced sharding where outputs are not all-reduced.
    Unreduced = 6,
}

/// Grouping behavior for shard groups.
///
/// This type corresponds to `OpSharding.ShardGroupType` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum ShardGroupType {
    /// Hard restriction where an operation must match another operation's sharding exactly.
    As = 0,

    /// Soft restriction where an operation prefers matching shardings with the group.
    Like = 1,
}

/// Describes how an operation is partitioned across devices.
///
/// This type corresponds to `OpSharding` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct OpSharding {
    /// Kind of sharding represented by this message.
    #[prost(enumeration = "OpShardingType", tag = "1")]
    pub r#type: i32,

    /// Shape of each sharded tile.
    #[prost(message, optional, tag = "2")]
    pub tile_shape: Option<Shape>,

    /// Shape of the tile-assignment tensor.
    #[prost(int64, repeated, tag = "3")]
    pub tile_assignment_dimensions: Vec<i64>,

    /// Flattened list of assigned device IDs.
    #[prost(int64, repeated, tag = "4")]
    pub tile_assignment_devices: Vec<i64>,

    /// Flattened tuple element shardings for tuple-shaped values.
    #[prost(message, repeated, tag = "5")]
    pub tuple_shardings: Vec<OpSharding>,

    /// If `true`, replicate across the final tile-assignment dimension.
    #[prost(bool, tag = "6")]
    pub replicate_on_last_tile_dim: bool,

    /// Metadata that records the origin of this sharding.
    #[prost(message, repeated, tag = "7")]
    pub metadata: Vec<OpMetadata>,

    /// Sharding type for each trailing tile-assignment subgroup dimension.
    #[prost(enumeration = "OpShardingType", repeated, tag = "8")]
    pub last_tile_dims: Vec<i32>,

    /// Dimensions used to reshape iota-generated device IDs.
    #[prost(int64, repeated, tag = "9")]
    pub iota_reshape_dims: Vec<i64>,

    /// Permutation applied after reshaping iota-generated device IDs.
    #[prost(int32, repeated, tag = "10")]
    pub iota_transpose_perm: Vec<i32>,

    /// If `true`, this operation participates in a shard group.
    #[prost(bool, tag = "11")]
    pub is_shard_group: bool,

    /// Unique identifier for the shard group.
    #[prost(int64, tag = "12")]
    pub shard_group_id: i64,

    /// Grouping behavior for shard-group propagation.
    #[prost(enumeration = "ShardGroupType", tag = "13")]
    pub shard_group_type: i32,

    /// Optional named sharding representation. When populated, legacy fields are ignored.
    #[prost(message, optional, tag = "14")]
    pub named_sharding: Option<NamedSharding>,
}

/// Contains a list of compilation environments. Currently [`DebugOptions`] contains a collection of flags that are
/// the union of flags for multiple different compilation environments. Eventually, those flags will be moved to the
/// specific compilation environments that they apply to.
///
/// This type corresponds to `CompilationEnvironmentsProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct CompilationEnvironments {
    /// List that contains compilation environments.
    #[prost(message, repeated, tag = "1")]
    pub environments: Vec<ProtoAny>,
}

/// Contains flags which affect compilation for GPU environments. This is currently empty but will eventually contain
/// GPU-related flags that are currently in [`DebugOptions`] and will be moved here.
///
/// This type corresponds to `GpuCompilationEnvironment` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct GpuCompilationEnvironment {
    /// Temporary dummy flag that will eventually be replaced with actual compilation flags for the GPU backend.
    #[prost(int64, tag = "1")]
    pub dummy_flag: i64,
}

/// Represent a range of contiguous integer numbers.
///
/// This type corresponds to `IntRangeInclusive` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct Range {
    /// First integer value in the range (inclusive).
    #[prost(int64, tag = "1")]
    pub first: i64,

    /// Last integer value in the range (inclusive).
    #[prost(int64, tag = "2")]
    pub last: i64,
}

/// Set of filters for limiting the thunk buffer debug instrumentation to specific thunks. This is only meaningful in
/// combination with either [`DebugOptions::xla_gpu_experimental_enable_checksum_tracing_on_thunks`].
///
/// This type corresponds to `ThunkBufferDebugFilter` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct ThunkBufferDebugFilter {
    /// Only thunk IDs matching one or more of these ranges will be included.
    #[prost(message, repeated, tag = "1")]
    pub thunk_id_ranges: Vec<Range>,

    /// Only thunks with profile annotations matching one or more of these regular expressions will be included.
    #[prost(string, repeated, tag = "2")]
    pub profile_annotation_regexes: Vec<String>,
}

/// Specifies which backends to enable auto-tuning for during compilation. Auto-tuning is the process of empirically
/// testing different algorithm implementations to find the fastest one for a given operation on specific hardware.
/// This enum controls which library backends are eligible for auto-tuning.
///
/// This type corresponds to `DebugOptions.AutotuneBackend` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum AutoTuneBackend {
    /// Enable auto-tuning for all supported backends.
    All = 0,

    /// Enable auto-tuning only for NVIDIA cuDNN library operations.
    Cudnn = 1,

    /// Enable auto-tuning only for Triton compiler-generated kernels.
    Triton = 2,

    /// Enable auto-tuning only for NVIDIA cuBLAS library operations.
    Cublas = 3,

    /// Enable auto-tuning only for NVIDIA cuBLASLt library operations.
    Cublaslt = 4,
}

/// Controls the behavior of the auto-tuning results cache. Auto-tuning can be expensive and so XLA supports caching the
/// results. This enum controls whether the cache is read-only, updated with new results, or unspecified.
///
/// This type corresponds to `DebugOptions.AutotuneCacheMode` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum AutoTuneCacheMode {
    /// Unspecified cache mode (default behavior).
    Unspecified = 0,

    /// Load cached results if present, otherwise run auto-tuner and save results.
    Update = 1,

    /// Read-only access to cached auto-tuning results without writing new entries.
    Read = 2,
}

/// Type of collective communication operation supported by XLA. Collective operations are communication primitives
/// used in distributed computing where multiple devices or processes participate in a coordinated data exchange.
///
/// This type corresponds to `DebugOptions.CollectiveOpType` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum CollectiveOperationType {
    /// No operation (placeholder).
    NoOp = 0,

    /// Combines values from all participants and distributes the result to all participants.
    AllReduce = 1,

    /// Gathers data from all participants and distributes the concatenated result to all participants.
    AllGather = 2,

    /// Reduces data across participants and scatters different portions to each participant.
    ReduceScatter = 3,

    /// Sends data from one participant to all other participants.
    CollectiveBroadcast = 4,

    /// Each participant sends distinct data to every other participant.
    AllToAll = 5,

    /// Routes data between participants according to a permutation pattern.
    CollectivePermute = 6,

    /// [`Self::AllToAll`] variant where each participant can send/receive different amounts of data.
    RaggedAllToAll = 7,

    /// Matches all collective operation types.
    AllCollectives = 8,
}

/// GPU command types for command buffer recording and execution. Command buffers allow batching multiple GPU operations
/// together for more efficient execution. This enum identifies the type of command being recorded.
///
/// This type corresponds to `DebugOptions.CommandBufferCmdType` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum CommandBufferCommandType {
    /// Invalid or unrecognized command type.
    Invalid = 0,

    /// Fusion kernel (i.e., multiple operations fused into a single kernel).
    Fusion = 1,

    /// NVIDIA cuBLAS library call.
    Cublas = 2,

    /// NVIDIA cuBLASLt library call.
    Cublaslt = 8,

    /// NVIDIA cuDNN library call.
    Cudnn = 3,

    /// Collective communication operations.
    Collectives = 4,

    /// Conditional control flow (i.e., if/else branches) constructs.
    Conditional = 5,

    /// While loop control flow constructs.
    While = 6,

    /// Custom call operations (user-defined or external library calls).
    CustomCall = 7,

    /// Dynamic slice fusion (i.e., fusion with dynamic indexing) operation.
    DynamicSliceFusion = 9,

    /// Dynamic slice fusion operation with copy operations.
    DynamicSliceCopyFusion = 10,
}

/// Controls the execution scheduling strategy within GPU command buffers. This enum determines how operations within a
/// command buffer are scheduled for execution, affecting parallelism and performance.
///
/// This type corresponds to `DebugOptions.CommandBufferSchedulingMode` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum CommandBufferSchedulingMode {
    /// Sequential command execution (i.e., no overlap between operations).
    Serialize = 0,

    /// Identify concurrent operations via data dependency analysis and executes them in parallel.
    Concurrent = 1,

    /// Use the latency-hiding scheduler strategy to overlap independent operations.
    LatencyHidingScheduler = 2,
}

/// Controls how runtime verification checks behave when issues are detected. This enum is used by various detection
/// mechanisms to specify whether to ignore issues, warn about them, or fail compilation/execution.
///
/// This type corresponds to `DebugOptions.DetectionMode` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum DetectionMode {
    /// No checks enabled, meaning that issues are ignored.
    None = 0,

    /// Log warnings when issues are detected, but continue execution.
    Warning = 1,

    /// Halt compilation or execution when issues are detected.
    Fail = 2,
}

/// Controls the usage of the `libnvjitlink` library for PTX linking on NVIDIA GPUs. `libnvjitlink` is NVIDIA's library
/// for just-in-time linking of PTX code. This enum controls whether XLA uses it for linking GPU kernels.
///
/// This type corresponds to `DebugOptions.LibNvJitLinkMode` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum LibNvJitLinkMode {
    /// Automatically use the library if available, avoiding known buggy versions.
    Auto = 0,

    /// Never use the library; always use alternative linking methods.
    Disabled = 1,

    /// Always use the library; fail the compilation if the library is unavailable.
    Enabled = 2,
}

/// Specifies the type of operation pattern for CPU library fusion. The XLA CPU backend can fuse certain operation
/// patterns into optimized library calls. This enum identifies the fusion pattern type.
///
/// This type corresponds to `DebugOptions.LibraryFusionType` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum LibraryFusionType {
    /// Invalid or unrecognized fusion type.
    Invalid = 0,

    /// Matrix multiplication (i.e., dot product) fused with surrounding element-wise operations.
    Dot = 1,

    /// Fused element-wise operations.
    Elementwise = 2,

    /// Fused reduction operations.
    Reduce = 3,

    /// Standalone matrix multiplication (i.e., dot product) operation (i.e., not fused with surrounding operations).
    IndividualDot = 4,

    /// Standalone convolution operation (i.e., not fused with surrounding operations).
    IndividualConvolution = 5,
}

/// Represents the type of algorithm to use for automatic buffer partitioning assignment in
/// _Single Program Multiple Data (SPMD)_ partitioning. These algorithms determine how to assign partitioning strategies
/// to operations when using automatic SPMD partitioning.
///
/// This type corresponds to `DebugOptions.PartitioningAlgorithm` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum PartitioningAlgorithm {
    /// No automatic partitioning (i.e., use manual sharding annotations).
    Noop = 0,

    /// Experimental partitioning algorithm variant 0.
    Exp0 = 1,

    /// Experimental partitioning algorithm variant 1.
    Exp1 = 2,

    /// Experimental partitioning algorithm variant 2.
    Exp2 = 3,
}

/// Controls GPU pipeline parallelism optimizations for _Single Program Multiple Data (SPMD)_-partitioned programs.
/// Pipeline parallelism overlaps computation and communication to improve throughput in distributed training scenarios.
///
/// This type corresponds to `DebugOptions.PipelineParallelismOptLevel` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum PipelineParallelismOptLevel {
    /// Disable pipeline parallelism optimizations.
    Disable = 0,

    /// Enable pipeline parallelism optimizations, including collective-permute decomposition
    /// for overlapping send/receive operations.
    Enable = 1,
}

/// Controls the strictness level for Profile-Guided Latency Estimator (PGLE) validation. PGLE uses Feedback-Directed
/// Optimization (FDO) profiles to estimate operation latencies. This enum controls how strictly the compiler validates
/// that the profile matches the current program.
///
/// This type corresponds to `DebugOptions.PGLEStrictnessLevel` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum PgleStrictnessLevel {
    /// No validation (i.e., mismatches between profile and program are ignored).
    Off = 0,

    /// Log warnings when instructions in the program are missing from the profile.
    Warning = 1,

    /// Halt compilation when the profile does not match the program.
    Error = 2,
}

/// Controls runtime shape validation policies for XLA programs. Shape checking verifies that tensor shapes at runtime
/// match the expected shapes from compilation. This enum controls when and how this verification occurs.
///
/// This type corresponds to `DebugOptions.ShapeChecks` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum ShapeChecks {
    /// No shape verification (i.e., outputs may contain garbage if shapes mismatch at runtime).
    Ignore = 0,

    /// Validate shapes at runtime. This may add some synchronization overhead.
    Runtime = 1,

    /// Reject programs at compile time unless shapes can be proven correct statically.
    Compilation = 2,
}

/// Specifies where training step markers are emitted in XLA programs. Step markers are used for profiling and
/// performance analysis to identify training step boundaries. This enum controls where in the program structure
/// these markers are placed.
///
/// This type corresponds to `DebugOptions.StepMarkerLocation` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum StepMarkerLocation {
    /// Emit step markers at program entry (default).
    StepMarkAtEntry = 0,

    /// Emit step markers at each iteration of the top-level while loop (typically the training loop).
    StepMarkAtTopLevelWhileLoop = 1,

    /// Emit step markers at each iteration of the second-level (nested) while loop.
    StepMarkAtSecondLevelWhileLoop = 3,

    /// Do not emit any step markers.
    StepMarkNone = 2,
}

/// Controls while loop unrolling strategies in XLA. Loop unrolling can improve performance by reducing loop overhead
/// and enabling better optimization opportunities, but increases code size and compilation time.
///
/// This type corresponds to `DebugOptions.WhileLoopUnrolling` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum WhileLoopUnrolling {
    /// Do not unroll while loops.
    NoUnroll = 0,

    /// Use a double-buffering strategy where we unroll by a factor of 2 to overlap computation with communication.
    DoubleBuffer = 1,

    /// Completely unroll the loop using the double-buffering technique.
    FullUnroll = 2,

    /// Automatically decide whether to unroll based on the presence of collective operations (i.e., loop unrolling
    /// is enabled if we have at least one collective operation in the loop body).
    AutoUnroll = 3,
}

/// Controls XNNPACK graph fusion strategies for the XLA CPU backend. [XNNPACK](https://github.com/google/XNNPACK) is
/// an optimized library for neural network inference on CPUs. This enum controls how aggressively XLA fuses operations
/// into XNNPACK subgraphs.
///
/// This type corresponds to `DebugOptions.XnnGraphFusionMode` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum XnnGraphFusionMode {
    /// Disable XNNPACK graph fusion.
    Disabled = 0,

    /// Use greedy fusion strategy (i.e., fuse operations when beneficial according to a cost model).
    Greedy = 1,

    /// Enhanced [`Self::Greedy`] fusion approach with additional heuristics.
    GreedySlinky = 2,

    /// Bypass the cost model and always fuse when possible.
    BypassCostModel = 3,
}

/// Debugging options for XLA compilation and execution. These options control various aspects of the XLA compiler and
/// runtime, including optimizations, debugging features, and backend-specific settings. Many of these options are
/// experimental or internal and may change without notice.
///
/// We use the following naming convention for these options:
///
///   - **Backend-Agnostic Options:** `xla_$flag_name`
///   - **Backend-Specific Options:** `xla_$backend_$flag_name`
///
/// The flags are grouped by backend and are sorted alphabetically by the flag name within each group.
///
/// For more information, refer to the [official XLA documentation](https://openxla.org/xla/flags_guidance).
///
/// This type corresponds to `DebugOptions` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct DebugOptions {
    /// If `true`, host-to-host copies will be allowed even when automatic host compute offloading is disabled.
    #[prost(bool, optional, tag = "439")]
    pub xla_allow_h2h_copy_when_automatic_host_compute_offload_disabled: Option<bool>,

    /// If `true`, then an error will be raised whenever the host offloader would have otherwise automatically
    /// offloaded some computation to the host.
    #[prost(bool, optional, tag = "408")]
    pub xla_disable_automatic_host_compute_offload: Option<bool>,

    /// If `false` some timers will be disabled. This is useful during auto-tuning to avoid timer overhead.
    #[prost(bool, optional, tag = "436")]
    pub xla_enable_scoped_logging_timers: Option<bool>,

    /// If `true`, hash-based cycle detection will be performed in fixed-point loops during compilation.
    #[prost(bool, optional, tag = "370")]
    pub xla_hlo_pass_fix_detect_cycles: Option<bool>,

    /// If `true`, shardings will be kept after _Single Program Multiple Data (SPMD)_ partitioning.
    #[prost(bool, optional, tag = "419")]
    pub xla_keep_shardings_after_spmd: Option<bool>,

    /// If `true`, compilation will fail if the HLO fixing pass cannot converge after a fixed number of iterations.
    #[prost(bool, optional, tag = "363")]
    pub xla_unsupported_crash_on_hlo_pass_fix_max_iterations: Option<bool>,

    /// If `true`, compilation will fail if a pass reports that it changed the HLO, but in fact it did not.
    #[prost(bool, optional, tag = "379")]
    pub xla_unsupported_crash_on_hlo_pass_noop_change: Option<bool>,

    /// If `true`, compilation will fail if a pass reports that it did not change the HLO, but in fact it did.
    #[prost(bool, optional, tag = "380")]
    pub xla_unsupported_crash_on_hlo_pass_silent_hlo_change: Option<bool>,

    /// Number of seconds to wait before terminating a rendezvous call.
    #[prost(int32, optional, tag = "417")]
    pub xla_cpu_collective_call_terminate_timeout_seconds: Option<i32>,

    /// Number of seconds to wait before warning about a stuck rendezvous call.
    #[prost(int32, optional, tag = "418")]
    pub xla_cpu_collective_call_warn_stuck_seconds: Option<i32>,

    /// Number of seconds to wait before terminating a collective operation.
    #[prost(int32, optional, tag = "438")]
    pub xla_cpu_collective_timeout_seconds: Option<i32>,

    /// If `true`, then region analysis will be used in the copy insertion pass.
    #[prost(bool, optional, tag = "337")]
    pub xla_cpu_copy_insertion_use_region_analysis: Option<bool>,

    /// Sets how often the compiler verifies the emitted modules. Higher levels mean more frequent verification.
    /// Currently, only the values `0` and `1` are supported.
    #[prost(int32, optional, tag = "395")]
    pub xla_cpu_emitter_verification_level: Option<i32>,

    /// If `true`, the HLO module scheduler will be optimized for extracting concurrency at extra memory cost.
    /// The live ranges of temporaries will be extended to allow XLA runtime to schedule independent operations
    /// in parallel on separate threads.
    #[prost(bool, optional, tag = "307")]
    pub xla_cpu_enable_concurrency_optimized_scheduler: Option<bool>,

    /// If `true`, unsafe mathematical optimizations will be enabled, including precision reduction
    /// and NaN/infinity-related assumptions.
    #[prost(bool, optional, tag = "99")]
    pub xla_cpu_enable_fast_math: Option<bool>,

    /// If `false`, the compiler will lower `minimum` and `maximum` HLOs to always propagate NaNs.
    #[prost(bool, optional, tag = "140")]
    pub xla_cpu_enable_fast_min_max: Option<bool>,

    /// If `true`, operations that use calculations producing platform-dependent results will be allowed.
    #[prost(bool, optional, tag = "425")]
    pub xla_cpu_enable_platform_dependent_math: Option<bool>,

    /// If `true`, oneDNN custom call thunks will be called in the CPU backend.
    #[prost(bool, optional, tag = "412")]
    pub xla_cpu_experimental_onednn_custom_call: Option<bool>,

    /// Fusion types enabled for oneDNN in the library rewriter pass.
    #[prost(enumeration = "LibraryFusionType", repeated, tag = "399")]
    pub xla_cpu_experimental_onednn_fusion_type: Vec<i32>,

    /// Fusion types enabled for XNNPACK in the library rewriter pass.
    #[prost(enumeration = "LibraryFusionType", repeated, tag = "400")]
    pub xla_cpu_experimental_xnn_fusion_type: Vec<i32>,

    /// Controls the XNN graph fusion HLO pass.
    #[prost(enumeration = "XnnGraphFusionMode", optional, tag = "365")]
    pub xla_cpu_experimental_xnn_graph_fusion_mode: Option<i32>,

    /// Fusion types enabled for YNNPACK in the library rewriter pass, or individual operations.
    #[prost(enumeration = "LibraryFusionType", repeated, tag = "422")]
    pub xla_cpu_experimental_ynn_fusion_type: Vec<i32>,

    /// If `true` and fast math is enabled, reciprocal usage will be forbidden and division will be used instead.
    #[prost(bool, optional, tag = "126")]
    pub xla_cpu_fast_math_honor_division: Option<bool>,

    /// If `true` and fast math is enabled, approximating calculations for functions will be forbidden.
    #[prost(bool, optional, tag = "129")]
    pub xla_cpu_fast_math_honor_functions: Option<bool>,

    /// If `true` and fast math is enabled, operations will be allowed to produce infinite values.
    #[prost(bool, optional, tag = "121")]
    pub xla_cpu_fast_math_honor_infs: Option<bool>,

    /// If `true` and fast math is enabled, operations will be allowed to produce NaN values.
    #[prost(bool, optional, tag = "120")]
    pub xla_cpu_fast_math_honor_nans: Option<bool>,

    /// If `true`, LLVM kernel entry points will be prefixed with the module name and converted to C-style names.
    /// This is useful for AOT compilation to avoid symbol collisions.
    #[prost(bool, optional, tag = "372")]
    pub xla_cpu_generate_unique_c_style_kernel_entry_points: Option<bool>,

    /// When set, the XLA CPU backend will only generate code up to the specified ISA (Instruction Set Architecture),
    /// preventing the use of newer ISA extensions.
    #[prost(string, optional, tag = "333")]
    pub xla_cpu_max_isa: Option<String>,

    /// Number of parts to split the LLVM module into before code generation, enabling parallel compilation.
    #[prost(int32, optional, tag = "323")]
    pub xla_cpu_parallel_codegen_split_count: Option<i32>,

    /// Preferred vector width value passed to the LLVM backend. Defaults to `256`.
    #[prost(int32, optional, tag = "308")]
    pub xla_cpu_prefer_vector_width: Option<i32>,

    /// If `true`, the XLA CPU backend will use fusion emitters for code generation.
    #[prost(bool, optional, tag = "376")]
    pub xla_cpu_use_fusion_emitters: Option<bool>,

    /// If `true`, the XLA CPU backend will use XNNPACK to execute supported operations.
    #[prost(bool, optional, tag = "359")]
    pub xla_cpu_use_xnnpack: Option<bool>,

    /// If `true`, optimizations that ignore the possibility of NaN values will be enabled.
    #[prost(bool, optional, tag = "335")]
    pub xla_enable_fast_math: Option<bool>,

    /// Filter limiting thunk buffer debug instrumentation to specific thunks.
    #[prost(message, optional, tag = "424")]
    pub xla_gpu_experimental_thunk_buffer_debug_filter: Option<ThunkBufferDebugFilter>,

    /// If `true`, an `HloUnoptimizedSnapshot` (serialized unoptimized module plus inputs) will be dumped for each
    /// HLO module run.
    #[prost(bool, optional, tag = "405")]
    pub xla_dump_hlo_unoptimized_snapshots: Option<bool>,

    /// If `true`, communication optimization patterns specified in Enzyme will be enabled.
    #[prost(bool, optional, tag = "429")]
    pub xla_enable_enzyme_comms_opt: Option<bool>,

    /// Path to a denylist file for cuDNN convolutions.
    #[prost(string, optional, tag = "128")]
    pub xla_gpu_algorithm_denylist_path: Option<String>,

    /// Size threshold in bytes for the GPU all-gather combiner.
    #[prost(int64, optional, tag = "212")]
    pub xla_gpu_all_gather_combine_threshold_bytes: Option<i64>,

    /// Number of devices per host for the first stage of the BlueConnect decomposition pass. A value less than `1`
    /// disables this pass.
    #[prost(int32, optional, tag = "159")]
    pub xla_gpu_all_reduce_blueconnect_num_devices_per_host: Option<i32>,

    /// Size threshold in bytes for the GPU all-reduce combiner.
    #[prost(int64, optional, tag = "157")]
    pub xla_gpu_all_reduce_combine_threshold_bytes: Option<i64>,

    /// Platform-specific options to improve analytical latency estimator precision.
    #[prost(map = "string, string", tag = "357")]
    pub xla_gpu_analytical_latency_estimator_options: HashMap<String, String>,

    /// If `true`, dot operations will be wrapped into async computations to enable parallelization of matrix
    /// operations.
    #[prost(bool, optional, tag = "321")]
    pub xla_gpu_async_dot: Option<bool>,

    /// Memory budget in GB per device for _Single Program Multiple Data (SPMD)_ auto-sharding.
    #[prost(int32, optional, tag = "224")]
    pub xla_gpu_auto_spmd_partitioning_memory_budget_gb: Option<i32>,

    /// Relative memory budget ratio for _Single Program Multiple Data (SPMD)_ auto-sharding.
    #[prost(float, optional, tag = "225")]
    pub xla_gpu_auto_spmd_partitioning_memory_budget_ratio: Option<f32>,

    /// Relative precision (tolerance) for comparing different GEMM solutions during auto-tuning.
    #[prost(float, optional, tag = "316")]
    pub xla_gpu_autotune_gemm_rtol: Option<f32>,

    /// Auto-tuning level:
    ///
    ///   - `0` disables auto-tuning,
    ///   - `1` enables auto-tuning without checks,
    ///   - `2` randomizes input data, and
    ///   - `3` and above enable verification with bounds checking.
    #[prost(int32, optional, tag = "123")]
    pub xla_gpu_autotune_level: Option<i32>,

    /// Maximum number of solutions the GEMM auto-tuner will consider. A value of `0` means no limit.
    #[prost(int64, optional, tag = "288")]
    pub xla_gpu_autotune_max_solutions: Option<i64>,

    /// If `true`, each fusion instruction will include a cost model runtime estimate in its backend configuration.
    #[prost(bool, optional, tag = "240")]
    pub xla_gpu_collect_cost_model_stats: Option<bool>,

    /// Factor by which to inflate collective operation costs by running each collective multiple times.
    #[prost(int32, optional, tag = "205")]
    pub xla_gpu_collective_inflation_factor: Option<i32>,

    /// Size threshold in bytes for the GPU collective-permute combiner.
    #[prost(int64, optional, tag = "378")]
    pub xla_gpu_collective_permute_combine_threshold_bytes: Option<i64>,

    /// Minimum data size in bytes to trigger the collective-permute decomposer transformation.
    #[prost(int64, optional, tag = "237")]
    pub xla_gpu_collective_permute_decomposer_threshold: Option<i64>,

    /// If `true`, collective cliques will not be locked for each XLA GPU execution, using permanent cliques instead.
    /// This disables deadlock prevention.
    #[prost(bool, optional, tag = "354")]
    pub xla_gpu_collectives_use_persistent_cliques: Option<bool>,

    /// Command buffer scheduling mode.
    #[prost(enumeration = "CommandBufferSchedulingMode", optional, tag = "404")]
    pub xla_gpu_command_buffer_scheduling_mode: Option<i32>,

    /// If `true`, loop commands with known loop counts and no unsupported nested commands will be unrolled during
    /// command buffer lowering.
    #[prost(bool, optional, tag = "411")]
    pub xla_gpu_command_buffer_unroll_loops: Option<bool>,

    /// If `true`, region analysis will be used in the copy insertion pass.
    #[prost(bool, optional, tag = "236")]
    pub xla_gpu_copy_insertion_use_region_analysis: Option<bool>,

    /// If `true`, the program will crash on verification failures instead of just logging them.
    #[prost(bool, optional, tag = "101")]
    pub xla_gpu_crash_on_verification_failures: Option<bool>,

    /// If `true`, Triton GEMM auto-tuning will be allowed to fall back to cuBLAS when it is faster.
    #[prost(bool, optional, tag = "247")]
    pub xla_gpu_cublas_fallback: Option<bool>,

    /// Path to the directory containing CUDA/PTX tools and libraries.
    #[prost(string, optional, tag = "61")]
    pub xla_gpu_cuda_data_dir: Option<String>,

    /// cuDNN GEMM fusion level: `0` disables, `1` enables for Blackwell and later architectures, `2` enables for
    /// Ampere and later architectures.
    #[prost(int32, optional, tag = "285")]
    pub xla_gpu_cudnn_gemm_fusion_level: Option<i32>,

    /// Maximum number of kernel configurations (plans) to consider during cuDNN GEMM auto-tuning.
    #[prost(int32, optional, tag = "318")]
    pub xla_gpu_cudnn_gemm_max_plans: Option<i32>,

    /// If `true`, the dot precision algorithm `ALG_DOT_BF16_BF16_F32` will be used by default for f32 dot operations.
    #[prost(bool, optional, tag = "441")]
    pub xla_gpu_default_to_alg_dot_bf16_bf16_f32: Option<bool>,

    /// If `true`, run-to-run determinism is guaranteed. This implies excluding nondeterministic operations and
    /// disabling auto-tuning.
    #[prost(bool, optional, tag = "148")]
    pub xla_gpu_deterministic_ops: Option<bool>,

    /// Collective operation types for which async execution should be disabled.
    #[prost(enumeration = "CollectiveOperationType", repeated, tag = "289")]
    pub xla_gpu_disable_async_collectives: Vec<i32>,

    /// If `true`, `ptxas` will be invoked with `-O0` instead of the default `-O3`.
    #[prost(bool, optional, tag = "103")]
    pub xla_gpu_disable_gpuasm_optimizations: Option<bool>,

    /// Threshold size in MB for the dot merger pass, which merges small dot operations to increase occupancy.
    #[prost(int32, optional, tag = "331")]
    pub xla_gpu_dot_merger_threshold_mb: Option<i32>,

    /// File path for writing autotune logs in text format.
    #[prost(string, optional, tag = "292")]
    pub xla_gpu_dump_autotune_logs_to: Option<String>,

    /// File path for writing autotune results. Binary format is used unless the filename
    /// ends with `.txt` or `.textproto`.
    #[prost(string, optional, tag = "222")]
    pub xla_gpu_dump_autotune_results_to: Option<String>,

    /// If `true`, all auto-tuned instructions will be dumped.
    #[prost(bool, optional, tag = "232")]
    pub xla_gpu_dump_autotuned_gemm_fusions: Option<bool>,

    /// If `true`, LLVM IR will be dumped when compiling to PTX.
    #[prost(bool, optional, tag = "155")]
    pub xla_gpu_dump_llvm_ir: Option<bool>,

    /// If `true`, all-gather operations will be combined by dimension; if `false`, they will be combined regardless
    /// of dimension.
    #[prost(bool, optional, tag = "254")]
    pub xla_gpu_enable_all_gather_combine_by_dim: Option<bool>,

    /// If `true`, the analytical latency cost model will be enabled.
    #[prost(bool, optional, tag = "255")]
    pub xla_gpu_enable_analytical_latency_estimator: Option<bool>,

    /// If `true`, the NCCL Speed-of-Light (SoL) analytical cost model will be enabled.
    #[prost(bool, optional, tag = "356")]
    pub xla_gpu_enable_analytical_sol_latency_estimator: Option<bool>,

    /// If `true`, approximation of expensive collective operations will be enabled.
    #[prost(bool, optional, tag = "305")]
    pub xla_gpu_enable_approx_costly_collectives: Option<bool>,

    /// Types of commands that are recorded into command buffers.
    #[prost(enumeration = "CommandBufferCommandType", repeated, tag = "258")]
    pub xla_gpu_enable_command_buffer: Vec<i32>,

    /// If `true`, radix sort using CUB will be enabled.
    #[prost(bool, optional, tag = "259")]
    pub xla_gpu_enable_cub_radix_sort: Option<bool>,

    /// If `true`, cuBLASLt will be used for GEMMs on GPUs.
    #[prost(bool, optional, tag = "166")]
    pub xla_gpu_enable_cublaslt: Option<bool>,

    /// If `true`, cuDNN int8x32 convolution reordering will be enabled.
    #[prost(bool, optional, tag = "189")]
    pub xla_gpu_enable_cudnn_int8x32_convolution_reordering: Option<bool>,

    /// If `true`, layer norm patterns will be rewritten into cuDNN library calls.
    #[prost(bool, optional, tag = "262")]
    pub xla_gpu_enable_cudnn_layer_norm: Option<bool>,

    /// If `true`, address computation fusion will be enabled to optimize dynamic-slice and dynamic-update-slice
    /// operations around library calls.
    #[prost(bool, optional, tag = "105")]
    pub xla_gpu_enable_dynamic_slice_fusion: Option<bool>,

    /// If `true`, the compiler will lower `minimum` and `maximum` HLOs so that `Min(NotNaN, NaN) = NotNaN`,
    /// meaning NaNs will not be propagated.
    #[prost(bool, optional, tag = "100")]
    pub xla_gpu_enable_fast_min_max: Option<bool>,

    /// If `true`, the highest priority async stream will be enabled.
    #[prost(bool, optional, tag = "216")]
    pub xla_gpu_enable_highest_priority_async_stream: Option<bool>,

    /// If `true`, host memory offloading will be enabled on the device.
    #[prost(bool, optional, tag = "296")]
    pub xla_gpu_enable_host_memory_offloading: Option<bool>,

    /// If `true`, the latency hiding scheduler will be enabled.
    #[prost(bool, optional, tag = "186")]
    pub xla_gpu_enable_latency_hiding_scheduler: Option<bool>,

    /// If `true`, the `libnvptxcompiler` library will be used to compile PTX to cuBIN.
    #[prost(bool, optional, tag = "269")]
    pub xla_gpu_enable_libnvptxcompiler: Option<bool>,

    /// If `true`, LLVM module compilation will be parallelized.
    #[prost(bool, optional, tag = "268")]
    pub xla_gpu_enable_llvm_module_compilation_parallelism: Option<bool>,

    /// If `true`, NCCL communicator splitting will be enabled.
    #[prost(bool, optional, tag = "272")]
    pub xla_gpu_enable_nccl_comm_splitting: Option<bool>,

    /// If `true`, NCCL user buffers will be enabled.
    #[prost(bool, optional, tag = "267")]
    pub xla_gpu_enable_nccl_user_buffers: Option<bool>,

    /// If `true`, pipelined all-gather operations will be enabled.
    #[prost(bool, optional, tag = "227")]
    pub xla_gpu_enable_pipelined_all_gather: Option<bool>,

    /// If `true`, pipelined all-reduce operations will be enabled.
    #[prost(bool, optional, tag = "217")]
    pub xla_gpu_enable_pipelined_all_reduce: Option<bool>,

    /// If `true`, pipelined host offloading will be enabled.
    #[prost(bool, optional, tag = "440")]
    pub xla_gpu_enable_pipelined_host_offloading: Option<bool>,

    /// If `true`, pipelined point-to-point communication will be enabled.
    #[prost(bool, optional, tag = "246")]
    pub xla_gpu_enable_pipelined_p2p: Option<bool>,

    /// If `true`, pipelined reduce-scatter operations will be enabled.
    #[prost(bool, optional, tag = "231")]
    pub xla_gpu_enable_pipelined_reduce_scatter: Option<bool>,

    /// If `true`, all-reduce reassociation will be enabled on all-reduce operations converted to a wider type.
    #[prost(bool, optional, tag = "209")]
    pub xla_gpu_enable_reassociation_for_converted_ar: Option<bool>,

    /// If `true`, reduce-scatter operations will be combined by dimension; if `false`, they will be combined
    /// regardless of dimension.
    #[prost(bool, optional, tag = "257")]
    pub xla_gpu_enable_reduce_scatter_combine_by_dim: Option<bool>,

    /// If `true`, reduction epilogue fusion will be enabled in fusion passes.
    #[prost(bool, optional, tag = "243")]
    pub xla_gpu_enable_reduction_epilogue_fusion: Option<bool>,

    /// If `true`, the scatter determinism expander will rewrite scatter operations to be deterministic.
    #[prost(bool, optional, tag = "345")]
    pub xla_gpu_enable_scatter_determinism_expander: Option<bool>,

    /// If `true`, large constants will be shared among multiple GPU executables.
    #[prost(bool, optional, tag = "165")]
    pub xla_gpu_enable_shared_constants: Option<bool>,

    /// If `true`, split-K auto-tuning will be enabled.
    #[prost(bool, optional, tag = "241")]
    pub xla_gpu_enable_split_k_autotuning: Option<bool>,

    /// If `true`, Triton GEMM will be enabled.
    #[prost(bool, optional, tag = "188")]
    pub xla_gpu_enable_triton_gemm: Option<bool>,

    /// If `true`, double buffering for while loops will be enabled.
    #[prost(bool, optional, tag = "248")]
    pub xla_gpu_enable_while_loop_double_buffering: Option<bool>,

    /// If `true`, reduce-scatter operations will be hoisted out of while loops.
    #[prost(bool, optional, tag = "203")]
    pub xla_gpu_enable_while_loop_reduce_scatter_code_motion: Option<bool>,

    /// Determines the while loop unrolling scheme.
    #[prost(enumeration = "WhileLoopUnrolling", optional, tag = "294")]
    pub xla_gpu_enable_while_loop_unrolling: Option<i32>,

    /// If `true`, nondeterministic operations will be excluded from compiled executables
    /// without disabling auto-tuning.
    #[prost(bool, optional, tag = "297")]
    pub xla_gpu_exclude_nondeterministic_ops: Option<bool>,

    /// If `true`, debug information will be embedded in the executable.
    #[prost(bool, optional, tag = "437")]
    pub xla_gpu_executable_embed_debug_info: Option<bool>,

    /// Number of seconds to wait before terminating on a stuck rendezvous.
    #[prost(int32, optional, tag = "328")]
    pub xla_gpu_executable_terminate_timeout_seconds: Option<i32>,

    /// Number of seconds to wait before issuing a warning on a stuck rendezvous.
    #[prost(int32, optional, tag = "327")]
    pub xla_gpu_executable_warn_stuck_timeout_seconds: Option<i32>,

    /// If `true`, an exhaustive tiling search will be performed.
    #[prost(bool, optional, tag = "219")]
    pub xla_gpu_exhaustive_tiling_search: Option<bool>,

    /// If `true`, an unroll factor of 8 will be allowed on Blackwell architectures (guarded by heuristics).
    #[prost(bool, optional, tag = "430")]
    pub xla_gpu_experimental_allow_unroll_factor_eight: Option<bool>,

    /// If `true`, Ahead-of-Time (AOT) compilation flow will be enabled with generated thunks
    /// included in the compiled binary.
    #[prost(bool, optional, tag = "435")]
    pub xla_gpu_experimental_aot_compiled_thunks: Option<bool>,

    /// List of auto-tuner backends to enable. If empty, all backends are enabled.
    #[prost(enumeration = "AutoTuneBackend", repeated, tag = "442")]
    pub xla_gpu_experimental_autotune_backends: Vec<i32>,

    /// Specifies the behavior of the per-kernel auto-tuning cache.
    #[prost(enumeration = "AutoTuneCacheMode", optional, tag = "324")]
    pub xla_gpu_experimental_autotune_cache_mode: Option<i32>,

    /// Directory path for storing the per-kernel auto-tuning cache.
    #[prost(string, optional, tag = "407")]
    pub xla_gpu_experimental_autotuner_cache_dir: Option<String>,

    /// Distance threshold for the `ScheduleAwareCollectiveOpsCSE` pass.
    #[prost(int64, optional, tag = "374")]
    pub xla_gpu_experimental_collective_cse_distance_threshold: Option<i64>,

    /// Path to experimental collective performance tables.
    #[prost(string, optional, tag = "377")]
    pub xla_gpu_experimental_collective_perf_table_path: Option<String>,

    /// If `true`, binary libraries will be disabled in GPU compiler passes.
    #[prost(bool, optional, tag = "329")]
    pub xla_gpu_experimental_disable_binary_libraries: Option<bool>,

    /// If `true`, FDO profiles will be dumped in binary format to a separate file.
    #[prost(bool, optional, tag = "338")]
    pub xla_gpu_experimental_dump_fdo_profiles: Option<bool>,

    /// If `true`, serialized GPU executables will be dumped to files with the `gpu_executable` suffix
    /// in the `xla_dump_to` directory.
    #[prost(bool, optional, tag = "427")]
    pub xla_gpu_experimental_dump_gpu_executable: Option<bool>,

    /// If `true`, windowed `einsum` (collective matmul) rewrite for all-to-all + GEMM will be enabled.
    #[prost(bool, optional, tag = "360")]
    pub xla_gpu_experimental_enable_all_to_all_windowed_einsum: Option<bool>,

    /// If `true`, outputs of selected thunks will be recorded (experimental feature).
    #[prost(bool, optional, tag = "431")]
    pub xla_gpu_experimental_enable_buffer_saver_on_thunks: Option<bool>,

    /// If `true`, checksums of selected thunk inputs/outputs will be recorded (experimental feature).
    #[prost(bool, optional, tag = "414")]
    pub xla_gpu_experimental_enable_checksum_tracing_on_thunks: Option<bool>,

    /// If `true`, auto-tuning between the native and Triton fusion emitters will be enabled.
    #[prost(bool, optional, tag = "409")]
    pub xla_gpu_experimental_enable_fusion_autotuner: Option<bool>,

    /// If `true`, every already-constructed fusion will be redirected to the Triton emitter with automatic tiling.
    #[prost(bool, optional, tag = "334")]
    pub xla_gpu_experimental_enable_fusion_block_level_rewriter: Option<bool>,

    /// If `true`, heuristic-based collective combining will be enabled.
    #[prost(bool, optional, tag = "366")]
    pub xla_gpu_experimental_enable_heuristic_collective_combining: Option<bool>,

    /// If `true`, NCCL symmetric buffers will be enabled.
    #[prost(bool, optional, tag = "406")]
    pub xla_gpu_experimental_enable_nccl_symmetric_buffers: Option<bool>,

    /// If `true`, NVSHMEM will be enabled. This must be set via the `XLA_FLAGS` environment variable before the XLA
    /// client is initialized.
    #[prost(bool, optional, tag = "388")]
    pub xla_gpu_experimental_enable_nvshmem: Option<bool>,

    /// If `true`, GEMMs that underutilize the GPU will be split along the K dimension.
    #[prost(bool, optional, tag = "386")]
    pub xla_gpu_experimental_enable_split_k_rewrite: Option<bool>,

    /// If `true`, fusion for subchannel dequantization sequences will be enabled.
    #[prost(bool, optional, tag = "368")]
    pub xla_gpu_experimental_enable_subchannel_dequantisation_fusion: Option<bool>,

    /// If `true`, the priority fusion pass will prioritize creating Triton fusions.
    #[prost(bool, optional, tag = "340")]
    pub xla_gpu_experimental_enable_triton_heroless_priority_fusion: Option<bool>,

    /// If `true`, Triton's auto warp specialization feature will be used when possible.
    #[prost(bool, optional, tag = "421")]
    pub xla_gpu_experimental_enable_triton_warp_specialization: Option<bool>,

    /// If `true`, sub-byte dot operands will be laid out along the contracting (K) dimension.
    #[prost(bool, optional, tag = "362")]
    pub xla_gpu_experimental_pack_dot_operands_along_k_dimension: Option<bool>,

    /// Maximum number of in-flight collectives the latency hiding scheduler can schedule.
    #[prost(int32, optional, tag = "336")]
    pub xla_gpu_experimental_parallel_collective_overlap_limit: Option<i32>,

    /// Optimization level for _Single Program Multiple Data (SPMD)_-based pipeline parallelism on GPU.
    #[prost(enumeration = "PipelineParallelismOptLevel", optional, tag = "351")]
    pub xla_gpu_experimental_pipeline_parallelism_opt_level: Option<i32>,

    /// If `true`, stream annotation will be enabled.
    #[prost(bool, optional, tag = "342")]
    pub xla_gpu_experimental_stream_annotation: Option<bool>,

    /// If `true`, the auto-tuner pass will be used to autotune fusions instead of the `gemm_fusion_autotuner`.
    #[prost(bool, optional, tag = "396")]
    pub xla_gpu_experimental_use_autotuner_pass: Option<bool>,

    /// If `true`, the ragged dot fusion emitter will be used rather than expanding to a regular dot.
    #[prost(bool, optional, tag = "401")]
    pub xla_gpu_experimental_use_ragged_dot_fusion: Option<bool>,

    /// If `true`, PTX compilation will fail if a kernel spills registers.
    #[prost(bool, optional, tag = "353")]
    pub xla_gpu_fail_ptx_compilation_on_register_spilling: Option<bool>,

    /// If `true`, kernels that spill registers will be filtered out during auto-tuning.
    #[prost(bool, optional, tag = "250")]
    pub xla_gpu_filter_kernels_spilling_registers_on_autotuning: Option<bool>,

    /// Number of seconds to wait before terminating the first collective call rendezvous.
    #[prost(int32, optional, tag = "392")]
    pub xla_gpu_first_collective_call_terminate_timeout_seconds: Option<i32>,

    /// Number of seconds to wait before warning about a stuck first collective call rendezvous.
    #[prost(int32, optional, tag = "391")]
    pub xla_gpu_first_collective_call_warn_stuck_timeout_seconds: Option<i32>,

    /// Overrides the normal multi-threaded compilation setting to use this many threads.
    /// A value of `0` means no override.
    #[prost(int32, optional, tag = "147")]
    pub xla_gpu_force_compilation_parallelism: Option<i32>,

    /// If `true`, convolutions will be forced to use NCHW layout.
    #[prost(bool, optional, tag = "125")]
    pub xla_gpu_force_conv_nchw: Option<bool>,

    /// If `true`, convolutions will be forced to use NHWC layout.
    #[prost(bool, optional, tag = "146")]
    pub xla_gpu_force_conv_nhwc: Option<bool>,

    /// If `true`, flush-to-zero semantics will be enabled in the GPU backend.
    #[prost(bool, optional, tag = "62")]
    pub xla_gpu_ftz: Option<bool>,

    /// If `true`, cuDNN RNG will be used for fused attention.
    #[prost(bool, optional, tag = "235")]
    pub xla_gpu_fused_attention_use_cudnn_rng: Option<bool>,

    /// Path to a `.textproto` file for overriding autotune results.
    #[prost(string, optional, tag = "434")]
    pub xla_gpu_gemm_autotuner_override_file: Option<String>,

    /// Minimum combined number of elements in matmul inputs/outputs for rewriting to cuBLAS or Triton.
    #[prost(int64, optional, tag = "283")]
    pub xla_gpu_gemm_rewrite_size_threshold: Option<i64>,

    /// If `true`, debug info will be generated when compiling PTX.
    #[prost(bool, optional, tag = "348")]
    pub xla_gpu_generate_debug_info: Option<bool>,

    /// If `true`, line info will be generated when compiling PTX.
    #[prost(bool, optional, tag = "349")]
    pub xla_gpu_generate_line_info: Option<bool>,

    /// Minimum number of moved instructions required for a region to be captured as a GPU graph function.
    #[prost(int32, optional, tag = "208")]
    pub xla_gpu_graph_min_graph_size: Option<i32>,

    /// File path for the kernel cache.
    #[prost(string, optional, tag = "306")]
    pub xla_gpu_kernel_cache_file: Option<String>,

    /// Controls the usage of `libnvjitlink`.
    #[prost(enumeration = "LibNvJitLinkMode", optional, tag = "343")]
    pub xla_gpu_libnvjitlink_mode: Option<i32>,

    /// Paths to files containing LLVM code.
    #[prost(string, repeated, tag = "150")]
    pub xla_gpu_llvm_ir_file: Vec<String>,

    /// LLVM backend verification level.
    #[prost(int32, optional, tag = "256")]
    pub xla_gpu_llvm_verification_level: Option<i32>,

    /// File path for loading autotune results. Binary format is used unless the filename ends
    /// with `.txt` or `.textproto`.
    #[prost(string, optional, tag = "223")]
    pub xla_gpu_load_autotune_results_from: Option<String>,

    /// Memory limit slop factor.
    #[prost(int32, optional, tag = "260")]
    pub xla_gpu_memory_limit_slop_factor: Option<i32>,

    /// If `true`, custom calls will be replaced with no-op operations.
    #[prost(bool, optional, tag = "245")]
    pub xla_gpu_mock_custom_calls: Option<bool>,

    /// If `true`, multiple compute streams will be used to run windowed einsum.
    #[prost(bool, optional, tag = "280")]
    pub xla_gpu_multi_streamed_windowed_einsum: Option<bool>,

    /// If `true`, NCCL collectives (e.g., all-reduce) will execute asynchronously.
    #[prost(bool, optional, tag = "393")]
    pub xla_gpu_nccl_async_execution: Option<bool>,

    /// If `true`, blocking NCCL communicators will be used; if `false`, non-blocking communicators will be used.
    #[prost(bool, optional, tag = "390")]
    pub xla_gpu_nccl_blocking_communicators: Option<bool>,

    /// Maximum number of channels (SMs) NCCL will use for collective operations.
    #[prost(int64, optional, tag = "273")]
    pub xla_gpu_nccl_collective_max_nchannels: Option<i64>,

    /// Number of ranks per root rank for NCCL initialization.
    #[prost(int64, optional, tag = "277")]
    pub xla_gpu_nccl_init_max_rank_per_root_ratio: Option<i64>,

    /// Maximum number of channels (SMs) NCCL will use for P2P operations.
    #[prost(int64, optional, tag = "274")]
    pub xla_gpu_nccl_p2p_max_channel_count: Option<i64>,

    /// If `true`, NCCL errors will terminate the process.
    #[prost(bool, optional, tag = "301")]
    pub xla_gpu_nccl_terminate_on_error: Option<bool>,

    /// Number of seconds before terminating jobs stuck in an NCCL rendezvous.
    #[prost(int64, optional, tag = "163")]
    pub xla_gpu_nccl_termination_timeout_seconds: Option<i64>,

    /// Combined operand bytes threshold for enabling windowed einsum. A negative value disables this feature.
    #[prost(int64, optional, tag = "339")]
    pub xla_gpu_operand_bytes_threshold_for_windowed_einsum: Option<i64>,

    /// Override configuration for the GEMM auto-tuner.
    #[prost(string, optional, tag = "295")]
    pub xla_gpu_override_gemm_autotuner: Option<String>,

    /// Directory path for the per-fusion autotune cache.
    #[prost(string, optional, tag = "310")]
    pub xla_gpu_per_fusion_autotune_cache_dir: Option<String>,

    /// PGLE (Profile-Guided Latency Estimator) accuracy checking strictness level.
    #[prost(enumeration = "PgleStrictnessLevel", optional, tag = "341")]
    pub xla_gpu_pgle_accuracy_checker: Option<i32>,

    /// Path to the FDO profile file or directory for PGLE.
    #[prost(string, optional, tag = "210")]
    pub xla_gpu_pgle_profile_file_or_directory_path: Option<String>,

    /// Paths to files containing PTX code.
    #[prost(string, repeated, tag = "127")]
    pub xla_gpu_ptx_file: Vec<String>,

    /// Size threshold in bytes for the GPU reduce-scatter combiner.
    #[prost(int64, optional, tag = "213")]
    pub xla_gpu_reduce_scatter_combine_threshold_bytes: Option<i64>,

    /// Amount of padding in bytes the _redzone_ allocator will put on one side of each buffer. Higher values make it
    /// more likely that we will catch an out-of-bounds read or write. Smaller values consume less memory during
    /// auto-tuning. Note that a fused cuDNN convolution has up to 6 total buffers (4 inputs, 1 output, and 1 scratch),
    /// so this can be multiplied by quite a lot.
    #[prost(int64, optional, tag = "228")]
    pub xla_gpu_redzone_padding_bytes: Option<i64>,

    /// If `true`, compilation will fail if complete AOT auto-tuning results are not available.
    #[prost(bool, optional, tag = "284")]
    pub xla_gpu_require_complete_aot_autotune_results: Option<bool>,

    /// If `true`, the XLA runtime will retain exclusive ownership of the GPU during execution.
    #[prost(bool, optional, tag = "347")]
    pub xla_gpu_require_exclusive_lock: Option<bool>,

    /// Shape checking mode for dynamically shaped operations.
    #[prost(enumeration = "ShapeChecks", optional, tag = "170")]
    pub xla_gpu_shape_checks: Option<i32>,

    /// If `true`, auto-tuning work will be sharded across participating compiler processes.
    #[prost(bool, optional, tag = "304")]
    pub xla_gpu_shard_autotuning: Option<bool>,

    /// If `true`, compilation will abort immediately when the convolution algorithm picker fails.
    #[prost(bool, optional, tag = "156")]
    pub xla_gpu_strict_conv_algorithm_picker: Option<bool>,

    /// Path to a file containing target platform description in `GpuTargetConfigProto` format for deviceless
    /// compilation.
    #[prost(string, optional, tag = "261")]
    pub xla_gpu_target_config_filename: Option<String>,

    /// If `true`, a separate memory space color will be used for temporary buffers.
    #[prost(bool, optional, tag = "312")]
    pub xla_gpu_temp_buffer_use_separate_color: Option<bool>,

    /// Size threshold in MB to enable windowed einsum (collective matmul).
    #[prost(int64, optional, tag = "265")]
    pub xla_gpu_threshold_for_windowed_einsum_mib: Option<i64>,

    /// If `true`, Triton fusion will be created for all supported GEMMs.
    #[prost(bool, optional, tag = "190")]
    pub xla_gpu_triton_gemm_any: Option<bool>,

    /// If `true`, the GPU backend will fall back to using the driver when `ptxas` is not found. It is usually
    /// preferable to not fall back to the driver.
    #[prost(bool, optional, tag = "138")]
    pub xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found: Option<bool>,

    /// If `true`, the embedded device library will be used in code generation.
    #[prost(bool, optional, tag = "420")]
    pub xla_gpu_use_embedded_device_lib: Option<bool>,

    /// If `true`, lld will be used as a library for the linking step.
    #[prost(bool, optional, tag = "389")]
    pub xla_gpu_use_in_process_lld: Option<bool>,

    /// If `true`, `memcpy` will be used for P2P communication within a node (NVLink).
    #[prost(bool, optional, tag = "287")]
    pub xla_gpu_use_memcpy_local_p2p: Option<bool>,

    /// If `true`, cuDNN runtime compiled fusion kernels will be used (requires Ampere or later architecture).
    #[prost(bool, optional, tag = "181")]
    pub xla_gpu_use_runtime_fusion: Option<bool>,

    /// If `true`, numerical results of Triton fusions will be verified against regular emitters.
    #[prost(bool, optional, tag = "291")]
    pub xla_gpu_verify_triton_fusion_numerics: Option<bool>,

    /// If `true`, addresses of HLO operations will be shown in graph dumps.
    #[prost(bool, optional, tag = "2")]
    pub xla_hlo_graph_addresses: Option<bool>,

    /// If `true`, the computation will be instrumented to collect per-HLO cycle counts.
    #[prost(bool, optional, tag = "9")]
    pub xla_hlo_profile: Option<bool>,

    /// List of HLO pass names to disable. Pass names must match exactly.
    #[prost(string, repeated, tag = "30")]
    pub xla_disable_hlo_passes: Vec<String>,

    /// List of HLO pass names to enable exclusively. All other passes will be disabled.
    #[prost(string, repeated, tag = "124")]
    pub xla_enable_hlo_passes_only: Vec<String>,

    /// If `true`, all HLO passes will be disabled. **Warning:** This may break compiler invariants.
    #[prost(bool, optional, tag = "104")]
    pub xla_disable_all_hlo_passes: Option<bool>,

    /// Numerical backend optimization level (similar to `-O` flags in compilers).
    #[prost(int32, optional, tag = "31")]
    pub xla_backend_optimization_level: Option<i32>,

    /// If `true`, the compiler IR string will be embedded in the generated `Executable`.
    #[prost(bool, optional, tag = "33")]
    pub xla_embed_ir_in_executable: Option<bool>,

    /// If `true`, explicit broadcast HLOs will be used instead of implicit broadcasts.
    #[prost(bool, optional, tag = "35")]
    pub xla_eliminate_hlo_implicit_broadcast: Option<bool>,

    /// If `true`, multi-threaded Eigen mode will be used in the CPU backend.
    #[prost(bool, optional, tag = "60")]
    pub xla_cpu_multi_thread_eigen: Option<bool>,

    /// If `true`, `!alias.scope` metadata will be emitted in the generated LLVM IR.
    #[prost(bool, optional, tag = "70")]
    pub xla_llvm_enable_alias_scope_metadata: Option<bool>,

    /// If `true`, `!noalias` metadata will be emitted in the generated LLVM IR.
    #[prost(bool, optional, tag = "71")]
    pub xla_llvm_enable_noalias_metadata: Option<bool>,

    /// If `true`, `!invariant.load` metadata will be emitted in the generated LLVM IR.
    #[prost(bool, optional, tag = "72")]
    pub xla_llvm_enable_invariant_load_metadata: Option<bool>,

    /// If `true`, expensive LLVM optimization passes will be skipped.
    #[prost(bool, optional, tag = "73")]
    pub xla_llvm_disable_expensive_passes: Option<bool>,

    /// If `true`, all permutations of output shape layouts will be tested.
    #[prost(bool, optional, tag = "90")]
    pub xla_test_all_output_layouts: Option<bool>,

    /// If `true`, all permutations of input argument layouts will be tested.
    #[prost(bool, optional, tag = "91")]
    pub xla_test_all_input_layouts: Option<bool>,

    /// If `true`, HLO graph nodes will be colored by sharding information.
    #[prost(bool, optional, tag = "92")]
    pub xla_hlo_graph_sharding_color: Option<bool>,

    /// If `true`, oneDNN thunks will be called for matmul and convolution operations on the CPU backend.
    #[prost(bool, optional, tag = "97")]
    pub xla_cpu_use_onednn: Option<bool>,

    /// If `true`, excess precision in floating-point operations will be allowed to increase output precision.
    #[prost(bool, optional, tag = "122")]
    pub xla_allow_excess_precision: Option<bool>,

    /// Forces the host platform to report this many devices.
    #[prost(int32, optional, tag = "102")]
    pub xla_force_host_platform_device_count: Option<i32>,

    /// If `true`, fast math will be enabled in the HLO evaluator.
    #[prost(bool, optional, tag = "106")]
    pub xla_hlo_evaluator_use_fast_path: Option<bool>,

    /// If `true`, both R1 and scalar index versions of dynamic slice operations will be supported.
    #[prost(bool, optional, tag = "107")]
    pub xla_allow_scalar_index_dynamic_ops: Option<bool>,

    /// Controls where target-specific training step markers are generated.
    #[prost(enumeration = "StepMarkerLocation", optional, tag = "108")]
    pub xla_step_marker_location: Option<i32>,

    /// Directory path to dump HLO modules to.
    #[prost(string, optional, tag = "109")]
    pub xla_dump_to: Option<String>,

    /// If `true`, flags state will be reset before applying command-line flags.
    #[prost(bool, optional, tag = "364")]
    pub xla_flags_reset: Option<bool>,

    /// Regular expression to filter which HLO modules to dump. Only modules with names matching this pattern
    /// will be dumped.
    #[prost(string, optional, tag = "110")]
    pub xla_dump_hlo_module_re: Option<String>,

    /// Regular expression to filter which HLO passes to dump. HLO will be dumped before and after passes matching
    /// this pattern.
    #[prost(string, optional, tag = "111")]
    pub xla_dump_hlo_pass_re: Option<String>,

    /// Regular expression to filter which emitters to dump debug logs for.
    #[prost(string, optional, tag = "433")]
    pub xla_dump_emitter_re: Option<String>,

    /// If `true`, HLO will be dumped in text format.
    #[prost(bool, optional, tag = "112")]
    pub xla_dump_hlo_as_text: Option<bool>,

    /// If `true`, HLO will be dumped as protobuf.
    #[prost(bool, optional, tag = "113")]
    pub xla_dump_hlo_as_proto: Option<bool>,

    /// If `true`, HLO will be dumped in GraphViz DOT format.
    #[prost(bool, optional, tag = "114")]
    pub xla_dump_hlo_as_dot: Option<bool>,

    /// If `true`, HLO will be dumped as a URL (for visualization services).
    #[prost(bool, optional, tag = "115")]
    pub xla_dump_hlo_as_url: Option<bool>,

    /// If `true`, HLO will be dumped as HTML (DOT rendered to SVG and inlined).
    #[prost(bool, optional, tag = "116")]
    pub xla_dump_hlo_as_html: Option<bool>,

    /// If `true`, fusion progress visualization will be dumped.
    #[prost(bool, optional, tag = "149")]
    pub xla_dump_fusion_visualization: Option<bool>,

    /// If `true`, `HloSnapshot` (module plus inputs) will be dumped for each run.
    #[prost(bool, optional, tag = "118")]
    pub xla_dump_hlo_snapshots: Option<bool>,

    /// If `true`, timestamps will be included in dump filenames.
    #[prost(bool, optional, tag = "131")]
    pub xla_dump_include_timestamp: Option<bool>,

    /// Maximum number of HLO modules to dump per directory. A negative value means unbounded.
    #[prost(int32, optional, tag = "132")]
    pub xla_dump_max_hlo_modules: Option<i32>,

    /// If `true`, `HloModuleMetadata` will be dumped as a text proto.
    #[prost(bool, optional, tag = "144")]
    pub xla_dump_module_metadata: Option<bool>,

    /// If `true`, dumped protobuf files will be GZip-compressed.
    #[prost(bool, optional, tag = "151")]
    pub xla_dump_compress_protos: Option<bool>,

    /// If `true`, HLO will be dumped in long text format (more verbose).
    #[prost(bool, optional, tag = "164")]
    pub xla_dump_hlo_as_long_text: Option<bool>,

    /// If `true`, MLIR will be dumped using pretty print form.
    #[prost(bool, optional, tag = "185")]
    pub xla_dump_enable_mlir_pretty_form: Option<bool>,

    /// If `true`, the full HLO configuration will be dumped.
    #[prost(bool, optional, tag = "381")]
    pub xla_dump_full_hlo_config: Option<bool>,

    /// If `true`, NaN detection will be enabled on TPU.
    #[prost(bool, optional, tag = "135")]
    pub xla_tpu_detect_nan: Option<bool>,

    /// If `true`, infinity detection will be enabled on TPU.
    #[prost(bool, optional, tag = "136")]
    pub xla_tpu_detect_inf: Option<bool>,

    /// If `true`, TraceMe annotations will be enabled for the CPU backend.
    #[prost(bool, optional, tag = "137")]
    pub xla_cpu_enable_xprof_trace_me: Option<bool>,

    /// Size constraint per heap in multi-heap allocation.
    #[prost(int32, optional, tag = "142")]
    pub xla_multi_heap_size_constraint_per_heap: Option<i32>,

    /// If `true`, detailed vlog compilation summaries will be enabled.
    #[prost(bool, optional, tag = "252")]
    pub xla_detailed_logging: Option<bool>,

    /// If `true`, HLO module dumping will be enabled.
    #[prost(bool, optional, tag = "253")]
    pub xla_enable_dumping: Option<bool>,

    /// If `true`, LLVM will be forced to inline functions before splitting the module.
    #[prost(bool, optional, tag = "300")]
    pub xla_llvm_force_inline_before_split: Option<bool>,

    /// If `true`, metadata will be excluded from HLO dumps.
    #[prost(bool, optional, tag = "153")]
    pub xla_dump_disable_metadata: Option<bool>,

    /// Regular expression to filter which HLO pipelines to dump.
    #[prost(string, optional, tag = "154")]
    pub xla_dump_hlo_pipeline_re: Option<String>,

    /// If `true`, ARM Compute Library (ACL) calls will be generated for the CPU backend.
    #[prost(bool, optional, tag = "174")]
    pub xla_cpu_use_acl: Option<bool>,

    /// If `true`, fp16 dot and convolution operations will be performed in fp16 instead of being promoted to fp32.
    #[prost(bool, optional, tag = "175")]
    pub xla_cpu_strict_dot_conv_math: Option<bool>,

    /// If `true`, the latency hiding schedule will be dumped.
    #[prost(bool, optional, tag = "182")]
    pub xla_dump_latency_hiding_schedule: Option<bool>,

    /// Partitioning algorithm to use for _Single Program Multiple Data (SPMD)_ in the partition assignment pass.
    #[prost(enumeration = "PartitioningAlgorithm", optional, tag = "187")]
    pub xla_partitioning_algorithm: Option<i32>,

    /// Maximum number of buffer assignments to print when debugging.
    #[prost(int64, optional, tag = "251")]
    pub xla_debug_buffer_assignment_show_max: Option<i64>,

    /// Detection mode for checking unstable reductions before optimizations.
    #[prost(enumeration = "DetectionMode", optional, tag = "403")]
    pub xla_detect_unstable_reductions: Option<i32>,

    /// Detection mode for checking unstable reductions after optimizations.
    #[prost(enumeration = "DetectionMode", optional, tag = "432")]
    pub xla_detect_unstable_reductions_post_optimizations: Option<i32>,

    /// Detection mode for NaN values on GPU.
    #[prost(enumeration = "DetectionMode", optional, tag = "426")]
    pub xla_gpu_detect_nan: Option<i32>,

    /// Detection mode for infinity values on GPU.
    #[prost(enumeration = "DetectionMode", optional, tag = "428")]
    pub xla_gpu_detect_inf: Option<i32>,

    /// If `true`, large constants will be printed in HLO dumps.
    #[prost(bool, optional, tag = "290")]
    pub xla_dump_large_constants: Option<bool>,

    /// Base length for reduce window rewrite.
    #[prost(int64, optional, tag = "293")]
    pub xla_reduce_window_rewrite_base_length: Option<i64>,

    /// Size of the command buffer trace cache.
    #[prost(int64, optional, tag = "311")]
    pub xla_cmd_buffer_trace_cache_size: Option<i64>,

    /// If `true`, syntactic sugar will be used for async operations in HLO dumps and NVTX.
    #[prost(bool, optional, tag = "315")]
    pub xla_syntax_sugar_async_ops: Option<bool>,

    /// If `true`, command buffers will be allowed to launch during profiling.
    #[prost(bool, optional, tag = "317")]
    pub xla_enable_command_buffers_during_profiling: Option<bool>,

    /// If `true`, channel IDs will be ignored in collective operations.
    #[prost(bool, optional, tag = "330")]
    pub xla_ignore_channel_id: Option<bool>,

    /// If `true`, PJRT will be allowed to use automatic layout in HLO.
    #[prost(bool, optional, tag = "344")]
    pub xla_pjrt_allow_auto_layout_in_hlo: Option<bool>,

    /// If `true`, command buffer mode will be tried in tests.
    #[prost(bool, optional, tag = "373")]
    pub xla_test_add_command_buffer_mode: Option<bool>,

    /// Path to experimental matmul performance table for GPU.
    #[prost(string, optional, tag = "383")]
    pub xla_gpu_experimental_matmul_perf_table_path: Option<String>,

    /// If `true`, compilation will exit early after layout assignment and return the layouts.
    #[prost(bool, optional, tag = "397")]
    pub xla_early_exit_with_layouts: Option<bool>,

    /// If `true`, the `IsTritonSupportedInstruction` check for scaled dot operations will be ignored.
    #[prost(bool, optional, tag = "410")]
    pub xla_gpu_experimental_scaled_dot_with_triton: Option<bool>,

    /// If `true`, the RAFT library will be used for TopK operations on GPU.
    #[prost(bool, optional, tag = "413")]
    pub xla_gpu_experimental_use_raft_select_k: Option<bool>,

    /// Extra backend-specific options as key-value pairs.
    #[prost(map = "string, string", tag = "500")]
    pub xla_backend_extra_options: HashMap<String, String>,
}

/// Represents the device IDs assigned to replicas for a single computation. In XLA's device assignment model, each
/// logical computation runs on multiple physical devices (one per replica). This struct holds the mapping from replica
/// index to device ID for a single computation.
///
/// This type corresponds to `DeviceAssignmentProto.ComputationDevice` in
/// [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Eq, Hash, Message)]
pub struct ComputationDeviceAssignment {
    /// Device IDs assigned to each replica of this computation. The length of this vector matches the replica count.
    /// The element at index `i` is the device ID assigned to replica `i` for this computation.
    #[prost(int64, repeated, tag = "1")]
    pub replica_device_ids: Vec<i64>,
}

/// Represents the device assignment for a set of replicated computations. A device assignment maps logical
/// `(replica_index, computation_index)` pairs to physical device IDs. The assignment is organized as a two-dimensional
/// structure where rows are replicas and columns are computations. For `R` replicas and `C` computations, `R * C`
/// devices are required to execute the computation in parallel.
///
/// This type corresponds to `DeviceAssignmentProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct DeviceAssignment {
    /// Number of replicas.
    #[prost(int32, tag = "1")]
    pub replica_count: i32,

    /// Number of computations.
    #[prost(int32, tag = "2")]
    pub computation_count: i32,

    /// [`ComputationDeviceAssignment`] for each computation. The length of this vector must match
    /// [`Self::computation_count`]. Each element contains the device IDs for all replicas of the
    /// corresponding computation.
    #[prost(message, repeated, tag = "3")]
    pub computation_devices: Vec<ComputationDeviceAssignment>,
}

/// Represents the level of effort the compiler should expend on optimization or memory fitting.
///
/// XLA provides options to control the amount of effort the compiler will expend to optimize for runtime performance
/// and to reduce memory requirements. The values in this enum are intended to be comparable across different platforms,
/// such that higher values represent higher amounts of effort.
///
/// Refer to the documentation of [`ExecutableCompilationOptions::optimization_level`] and
/// [`ExecutableCompilationOptions::memory_fitting_level`] for more information on how each level of effort affects
/// program compilation.
///
/// This type corresponds to `ExecutionOptions.EffortLevel` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum EffortLevel {
    /// Unknown effort level meant to be a default placeholder for extensibility.
    EffortUnknown = 0,

    /// Minimal effort.
    EffortO0 = 9,

    /// Reduced effort.
    EffortO1 = 19,

    /// Significant effort (suitable as the default for production workloads).
    EffortO2 = 29,

    /// Maximum effort (enabling expensive and experimental algorithms).
    EffortO3 = 39,
}

/// Configuration options which control how XLA compiles programs into executables, including device placement
/// configuration, partitioning strategies, optimization levels, and sharding propagation options.
///
/// This type corresponds to `ExecutableBuildOptionsProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct ExecutableCompilationOptions {
    /// Ordinal ID of the device for which to compile the program. Valid values are `0` to `<number of devices> - 1`.
    /// These values are identical to the device ordinal IDs used by the XLA stream executor. The built executable will
    /// be executable on any device equivalent to the specified device as determined by the XLA backend. A value of `-1`
    /// indicates that this option has not been set.
    #[prost(int64, tag = "1")]
    pub device_ordinal: i64,

    /// Optional [`Shape`] of the program result/output. If not set, the [`Shape`] will be inferred by the compiler.
    #[prost(message, optional, tag = "2")]
    pub result_shape: Option<Shape>,

    /// [`CompilationEnvironments`] to pass to the compiler that contain environment-specific configuration options.
    #[prost(message, optional, tag = "13")]
    pub compilation_environments: Option<CompilationEnvironments>,

    /// [`DebugOptions`] to pass to the compiler.
    #[prost(message, optional, tag = "3")]
    pub debug_options: Option<DebugOptions>,

    /// Number of replicas of this computation that are to be executed. Defaults to `1`.
    #[prost(int64, tag = "4")]
    pub replica_count: i64,

    /// Number of partitions in this computation. Defaults to `1`.
    #[prost(int64, tag = "5")]
    pub partition_count: i64,

    /// If `true`, use _Single Program Multiple Data (SPMD)_ partitioning when [`Self::partition_count`] is greater than
    /// `1` and XLA is requested to partition the input program. If `false`, use _Multiple Program Multiple Data (MPMD)_
    /// partitioning instead.
    #[prost(bool, tag = "6")]
    pub use_spmd_partitioning: bool,

    /// If `true`, automatically generate shardings for the _Single Program Multiple Data (SPMD)_ partitioner.
    #[prost(bool, tag = "7")]
    pub use_auto_spmd_partitioning: bool,

    /// Amount of effort to spend on optimizing for minimizing program execution time, specified as a value in
    /// `[-1.0, +1.0]`. The baseline is `0.0`, which strongly prioritizes execution time at the cost of longer
    /// compile times and is suitable for production workloads. A value of `-0.5` would be appropriate for
    /// research use cases where faster compilation time is preferred to improve iteration speed. Positive values,
    /// on the other hand, might enable costly optimizations that are disabled by default.
    ///
    /// Refer to [`Self::optimization_level`] for a discrete [`EffortLevel`]-based alternative.
    #[prost(float, tag = "20")]
    pub optimization_effort: f32,

    /// Amount of effort to spend on reducing the memory requirements of the program, specified as a value in
    /// `[-1.0, +1.0]`. The baseline is `0.0`, which expends significant effort on attempting to reduce the memory
    /// requirements of the program. A value of `-1.0` would be appropriate for use cases that wish to spend minimal
    /// effort here and fail as quickly as possible instead. Positive values, on the other hand, might enable costly
    /// algorithms to reduce memory usage that are disabled by default.
    ///
    /// Refer to [`Self::memory_fitting_level`] for a discrete [`EffortLevel`]-based alternative.
    #[prost(float, tag = "21")]
    pub memory_fitting_effort: f32,

    /// Amount of effort to spend on optimizing for minimizing program execution time, specified as an [`EffortLevel`].
    ///
    /// Similar to the `-O` flags in `gcc` or `clang`, this field allows users to influence how much work the compiler
    /// does in optimizing for execution time. Lower optimization levels will cause various HLO passes to behave
    /// differently, typically doing less work, or may disable certain HLO passes entirely. The optimization level may
    /// also influence the compiler backend, such that the exact effect of this field has a dependence on the target
    /// platform.
    ///
    /// In the XLA GPU backend, there are several passes that are disabled by default because they significantly
    /// increase compilation time by increasing the resulting HLO size. Setting `optimization_level` to
    /// [`EffortLevel::EffortO1`] or above will lead to the following behavior:
    ///
    ///   - Collectives commonly used for data-parallel communication will be pipelined. This behavior can also be
    ///     steered in a more granular way by using the following flags:
    ///     - [`DebugOptions::xla_gpu_enable_pipelined_all_gather`]
    ///     - [`DebugOptions::xla_gpu_enable_pipelined_all_reduce`]
    ///     - [`DebugOptions::xla_gpu_enable_pipelined_reduce_scatter`]
    ///   - Unrolling while loops by a factor of two, breaking down the loop-barrier potentially leading to better
    ///     compute-communication overlap and fewer copies (also controllable via
    ///     [`DebugOptions::xla_gpu_enable_while_loop_double_buffering`]).
    ///   - The latency hiding scheduler will do most of the work to hide the communication latency (also controllable
    ///     via [`DebugOptions::xla_gpu_enable_latency_hiding_scheduler`]).
    ///   - To maximize networking bandwidth, the combiner passes will combine pipelined collectives to the maximum
    ///     available memory. The optimization does not kick in if the loop is already unrolled in the input HLO.
    #[prost(enumeration = "EffortLevel", tag = "24")]
    pub optimization_level: i32,

    /// Amount of effort to spend on optimizing the memory usage and memory access patterns of the resulting executable,
    /// specified as an [`EffortLevel`]. This option controls the degree to which the compiler will attempt to reduce
    /// the memory requirements of the resulting executable, though the specific behavior is backend-dependent (e.g., in
    /// the XLA TPU backend this option controls the degree to which the compiler works to keep the TPU's high-bandwidth
    /// memory usage below the HBM capacity).
    #[prost(enumeration = "EffortLevel", tag = "25")]
    pub memory_fitting_level: i32,

    /// If `true`, HLOs should be deduplicated.
    #[prost(bool, tag = "8")]
    pub deduplicate_hlo: bool,

    /// Optional static device assignment for the computation. If set, it specifies the exact device assignment for the
    /// resulting executable. If not set, the computation will be compiled generically and can be run with any device
    /// assignment compatible with [`Self::replica_count`] and [`Self::partition_count`].
    #[prost(message, optional, tag = "9")]
    pub device_assignment: Option<DeviceAssignment>,

    /// If `true`, input and output buffers are aliased if the associated parameter is passed through XLA modules
    /// without being changed.
    #[prost(bool, tag = "10")]
    pub alias_passthrough_params: bool,

    /// If `true`, XLA builds an executable by invoking only `Compiler::RunBackend` and skips invoking
    /// `Compiler::RunHloPasses`. Defaults to `false`, in which case XLA builds an executable by invoking standard
    /// compilation (i.e., running `Compiler::Compile` or both `Compiler::RunHloPasses` and `Compiler::RunBackend`).
    /// Setting this to `true` can be used to speed up the compilation of post-optimization HLO modules.
    #[prost(bool, tag = "11")]
    pub run_backend_only: bool,

    /// Controls whether sharding propagation can propagate to the computation's parameters. Enabling this changes
    /// the input shape of the computation (which is undesirable in general). However, it can be used to run partial
    /// compilation to determine what would be the input sharding of a computation if XLA were allowed to propagate
    /// the sharding. This can be used by higher level frameworks as a way to query intermediate sharding of operations
    /// when multiple computations would be chained and merged together. This is specified as a vector of booleans
    /// because the user can control which parameters can have their sharding substituted. If only one boolean value
    /// is passed in the vector, then it will be interpreted as the value to use for every parameter.
    #[prost(bool, repeated, tag = "18")]
    pub allow_spmd_sharding_propagation_to_parameters: Vec<bool>,

    /// Controls whether sharding propagation can propagate to the computation's outputs. Enabling this changes the
    /// output shape of the computation (which is undesirable in general). However, it can be used to run partial
    /// compilation to determine what would be the output sharding of a computation if XLA were allowed to propagate
    /// the sharding. This can be used by higher level frameworks as a way to query intermediate sharding of operations
    /// when multiple computations would be chained and merged together. This is specified as a vector of booleans
    /// because the user can control (if the output of the computation is a tuple) which elements of the tuple can have
    /// their sharding substituted. If only one boolean value is passed in the vector, then it will be interpreted as
    /// the value to use for every element of the output tuple. One value per element of the tuple means that each value
    /// is attached to one of the output elements.
    #[prost(bool, repeated, tag = "12")]
    pub allow_spmd_sharding_propagation_to_output: Vec<bool>,

    /// Opaque profile data for Feedback-Directed Optimizations (FDO).
    #[prost(bytes = "vec", tag = "14")]
    pub fdo_profile: Vec<u8>,

    /// Available device memory size in bytes for the executable.
    #[prost(int64, optional, tag = "15")]
    pub device_memory_size: Option<i64>,

    /// Mesh shape to use for automatic _Single Program Multiple Data (SPMD)_ partitioning. This is only used when
    /// [`Self::use_auto_spmd_partitioning`] is set to `true`.
    #[prost(int64, repeated, tag = "16")]
    pub auto_spmd_partitioning_mesh_shape: Vec<i64>,

    /// Mesh IDs to use for automatic _Single Program Multiple Data (SPMD)_ partitioning. This is only used when
    /// [`Self::use_auto_spmd_partitioning`] is set to `true`.
    #[prost(int64, repeated, tag = "17")]
    pub auto_spmd_partitioning_mesh_ids: Vec<i64>,

    /// If `true`, use the Shardy partitioner instead of the existing `ShardingPropagation` and `SpmdPartitioner`.
    /// Shardy is a new partitioner that aims to replace the existing partitioning infrastructure.
    #[prost(bool, tag = "19")]
    pub use_shardy_partitioner: bool,

    /// Index of the current process in distributed execution. Used for multi-process compilation and execution.
    #[prost(int64, optional, tag = "22")]
    pub process_index: Option<i64>,

    /// Total number of processes in distributed execution. Used for multi-process compilation and execution.
    #[prost(int64, optional, tag = "23")]
    pub process_count: Option<i64>,

    /// Slice size for distributed execution. Used for multi-process compilation and execution.
    #[prost(int64, optional, tag = "26")]
    pub slice_size: Option<i64>,
}

/// Value of an [`OptionOverride`] in an XLA compilation confirmation.
///
/// This type corresponds to `OptionOverrideProto.value` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Oneof)]
pub enum OptionValue {
    /// String option value.
    #[prost(string, tag = "1")]
    StringField(String),

    /// Boolean option value.
    #[prost(bool, tag = "2")]
    BoolField(bool),

    /// Integer option value.
    #[prost(int64, tag = "3")]
    IntField(i64),

    /// Floating-point option value.
    #[prost(double, tag = "4")]
    DoubleField(f64),
}

/// Override for a compiler option in an XLA compilation confirmation.
///
/// This type corresponds to `OptionOverrideProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct OptionOverride {
    /// Value of this option override.
    #[prost(oneof = "OptionValue", tags = "1, 2, 3, 4")]
    pub value: Option<OptionValue>,
}

/// Precision level for numerical computations. This controls the precision of operations like matrix multiplications
/// and convolutions. The exact meaning is backend-specific, but generally higher precision values trade performance
/// for numerical accuracy.
///
/// This type corresponds to `PrecisionConfig.Precision` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum Precision {
    /// Default precision configuration that is backend-specific.
    Default = 0,

    /// High precision mode, providing better numerical accuracy than [`Self::Default`].
    High = 1,

    /// Highest precision mode, maximizing numerical accuracy at the cost of performance.
    Highest = 2,
}

/// Auto-tuning configuration key for a convolution algorithm.
///
/// This type corresponds to `AutotuneResult.ConvKey` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct AutoTuneConvolutionKey {
    /// Algorithm identifier.
    #[prost(int64, tag = "1")]
    pub algorithm: i64,

    /// If `true`, tensor core operations are enabled for this algorithm.
    #[prost(bool, tag = "2")]
    pub tensor_core_operations_enabled: bool,
}

/// Auto-tuning configuration key for a General Matrix Multiply (GEMM) algorithm.
///
/// This type corresponds to `AutotuneResult.GemmKey` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct AutoTuneGemmKey {
    /// Algorithm identifier.
    #[prost(int64, tag = "1")]
    pub algorithm: i64,

    /// Workspace size (in bytes) used during auto-tuning.
    #[prost(int64, tag = "2")]
    pub workspace_size_in_bytes: i64,
}

/// Auto-tuning configuration key for a CUDA convolution execution plan.
///
/// This type corresponds to `AutotuneResult.CudaConvPlanKey` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Eq, Hash, Message)]
pub struct AutoTuneCudaConvolutionPlanKey {
    /// Execution plan ID string.
    #[prost(string, tag = "1")]
    pub execution_plan_id: String,
}

/// Auto-tuning configuration key for a Triton General Matrix Multiply (GEMM) kernel.
///
/// This type corresponds to `AutotuneResult.TritonGemmKey` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct AutoTuneTritonGemmKey {
    /// Tile size in the `M` dimension (i.e., rows of the output matrix).
    #[prost(int64, tag = "1")]
    pub block_m: i64,

    /// Tile size in the `N` dimension (i.e., columns of the output matrix).
    #[prost(int64, tag = "2")]
    pub block_n: i64,

    /// Tile size in the `K` dimension (i.e., reduction dimension).
    #[prost(int64, tag = "3")]
    pub block_k: i64,

    /// Split-`K` factor for parallel reduction along the `K` dimension.
    #[prost(int64, tag = "4")]
    pub split_k: i64,

    /// Number of pipeline stages for software pipelining.
    #[prost(int64, tag = "5")]
    pub pipeline_stage_count: i64,

    /// Number of warps to use for this kernel.
    #[prost(int64, tag = "6")]
    pub warp_count: i64,

    /// Number of Cooperative Thread Arrays (CTAs) / thread blocks to use for this kernel.
    #[prost(int64, tag = "7")]
    pub cooperative_thread_array_count: i64,

    /// If `true`, Tensor Memory Accelerator (TMA) operations are allowed.
    #[prost(bool, tag = "8")]
    pub is_tma_allowed: bool,

    /// If `true`, warp specialization optimizations are allowed.
    #[prost(bool, tag = "9")]
    pub is_warp_specialization_allowed: bool,
}

/// Auto-tuning configuration key for a custom fused kernel.
///
/// This type corresponds to `AutotuneResult.CustomKernelFusionKey`
/// in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct AutoTuneCustomKernelFusionKey {
    /// Index of the selected kernel implementation.
    #[prost(int64, tag = "1")]
    pub kernel_index: i64,
}

/// Describes the type of math operations that are used in the implementation of an algorithm (e.g., standard math
/// operations or tensor core operations on supported hardware).
///
/// This type corresponds to `stream_executor.dnn.AlgorithmProto.MathType` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum AlgorithmMathType {
    /// Default math operations (e.g., standard floating-point arithmetic).
    DefaultMath = 0,

    /// Tensor core operations which can provide significant speedups for matrix operations but may have reduced
    /// numerical precision compared to [`AlgorithmMathType::DefaultMath`].
    TensorOperationMath = 1,
}

/// Auto-tuning configuration key for a generic algorithm.
///
/// This type corresponds to `stream_executor.dnn.AlgorithmProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct AutoTuneAlgorithmKey {
    /// Algorithm ID that uniquely identifies the algorithm implementation.
    #[prost(int64, tag = "1")]
    pub algorithm_id: i64,

    /// [`AlgorithmMathType`] used in the algorithm implementation.
    #[prost(enumeration = "AlgorithmMathType", tag = "2")]
    pub math_type: i32,

    /// Tuning knob configurations for the algorithm implementation that are specific to that implementation
    /// and allow for fine-grained control over the behavior of the algorithm.
    #[prost(map = "int64, int64", tag = "4")]
    pub tuning_knobs: HashMap<i64, i64>,

    /// Workspace size (in bytes) required by this algorithm. This is only relevant for ROCm since, when using ROCm, it
    /// is impossible to re-query the required workspace size after running the algorithm search, and so we must store
    /// it along with the choice of algorithm. For consistency and convenience, cuDNN uses this field in the same way,
    /// even though it would be possible to re-query for the workspace size from cuDNN at each use.
    #[prost(message, optional, tag = "6")]
    pub workspace_size_in_bytes: Option<u64>,
}

/// Auto-tuning configuration key for a generic backend configuration meant to be used for operations not covered
/// by any of the other more specific auto-tuning configuration key types.
///
/// This type corresponds to `AutotuneResult.BackendConfigKey` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Eq, Hash, Message)]
pub struct AutoTuneBackendConfigurationKey {
    /// Name identifying this configuration type.
    #[prost(string, tag = "1")]
    pub name: String,

    /// Configuration data stored as an [`Any`](ProtoAny) message.
    #[prost(message, optional, tag = "2")]
    pub config: Option<ProtoAny>,
}

/// Key identifying the selected algorithm configuration for an [`AutoTuneResult`].
///
/// This type corresponds to `AutotuneResult.key` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Oneof)]
pub enum AutoTuneResultKey {
    /// Convolution algorithm configuration.
    #[prost(message, tag = "5")]
    Convolution(AutoTuneConvolutionKey),

    /// General Matrix Multiply (GEMM) algorithm configuration.
    #[prost(message, tag = "6")]
    Gemm(AutoTuneGemmKey),

    /// Triton General Matrix Multiply (GEMM) kernel configuration.
    #[prost(message, tag = "17")]
    Triton(AutoTuneTritonGemmKey),

    /// CUDA convolution execution plan configuration.
    #[prost(message, tag = "15")]
    CudaConvolutionPlan(AutoTuneCudaConvolutionPlanKey),

    /// Custom fused kernel configuration.
    #[prost(message, tag = "18")]
    CustomKernelFusion(AutoTuneCustomKernelFusionKey),

    /// Generic algorithm configuration.
    #[prost(message, tag = "16")]
    Algorithm(AutoTuneAlgorithmKey),

    /// Generic backend configuration for other operation types.
    #[prost(message, tag = "19")]
    Other(AutoTuneBackendConfigurationKey),
}

/// Type of failure that occurred during auto-tuning.
///
/// This type corresponds to `AutotuneResult.FailureKind` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum AutoTuneFailureKind {
    /// Unknown or unspecified failure type.
    Unknown = 0,

    /// Failure resulting from writing to memory outside its designated output buffers (i.e., _redzone_ violation).
    RedzoneModified = 1,

    /// Failure resulting from producing a different result than the reference algorithm.
    WrongResult = 2,

    /// Failure resulting from the algorithm being rejected for failing to run or for known bugs.
    Disqualified = 3,
}

/// Reference algorithm configuration used for comparison during auto-tuning failure analysis.
///
/// This type corresponds to `AutotuneResult.FailureResult.key` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Oneof)]
pub enum AutoTuneReferenceKey {
    /// Reference convolution algorithm configuration.
    #[prost(message, tag = "11")]
    ReferenceConvolution(AutoTuneConvolutionKey),

    /// Reference General Matrix Multiply (GEMM) algorithm configuration.
    #[prost(message, tag = "12")]
    ReferenceGemm(AutoTuneGemmKey),

    /// Reference CUDA convolution execution plan configuration.
    #[prost(message, tag = "14")]
    ReferenceCudaConvolutionPlan(AutoTuneCudaConvolutionPlanKey),

    /// Reference generic algorithm configuration.
    #[prost(message, tag = "15")]
    ReferenceAlgorithm(AutoTuneAlgorithmKey),
}

/// Represents an [`AutoTuneResult`] that corresponds to a failure.
///
/// This type corresponds to `AutotuneResult.FailureResult` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct AutoTuneFailureResult {
    /// [`AutoTuneFailureKind`] of the failure that occurred.
    #[prost(enumeration = "AutoTuneFailureKind", tag = "1")]
    pub kind: i32,

    /// Human-readable error message describing the failure.
    #[prost(string, tag = "2")]
    pub message: String,

    /// Memory address of the buffer involved in the failure that is relevant for _redzone_ violations.
    #[prost(int64, tag = "13")]
    pub buffer_address: i64,

    /// Reference algorithm configuration that was compared against. Reference algorithms are typically simpler,
    /// more reliable implementations used to validate the correctness of faster algorithms.
    #[prost(oneof = "AutoTuneReferenceKey", tags = "11, 12, 14, 15")]
    pub key: Option<AutoTuneReferenceKey>,
}

/// Auto-tuning result for a specific operation on a specific device (both of which are contained in the wrapping
/// [`AutoTuneResultEntry`] instance as [`AutoTuneResult`]s contain just the auto-tuning result without any information
/// about the operation or the device).
///
/// This type corresponds to `AutotuneResult` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct AutoTuneResult {
    /// Amount of scratch memory (in bytes) required by the selected algorithm.
    #[prost(int64, tag = "8")]
    pub scratch_bytes: i64,

    /// Measured execution time of the selected algorithm.
    #[prost(message, optional, tag = "9")]
    pub run_time: Option<Duration>,

    /// Failure information if auto-tuning failed in this instance or the algorithm was disqualified.
    #[prost(message, optional, tag = "7")]
    pub failure: Option<AutoTuneFailureResult>,

    /// TKey identifying the selected algorithm configuration.
    #[prost(oneof = "AutoTuneResultKey", tags = "5, 6, 17, 15, 18, 16")]
    pub key: Option<AutoTuneResultKey>,
}

/// Auto-tuning result entry for a specific operation on a specific device.
///
/// This type corresponds to `AutotuneResults.Entry` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct AutoTuneResultEntry {
    /// Device identifier for this auto-tuning result.
    #[prost(string, tag = "1")]
    pub device: String,

    /// HLO representation of the operation that was autotuned.
    #[prost(string, tag = "2")]
    pub hlo: String,

    /// [`AutoTuneResult`] associated with [`Self::device`] and [`Self::hlo`].
    #[prost(message, optional, tag = "3")]
    pub result: Option<AutoTuneResult>,

    /// Auto-tuning cache key version for this [`AutoTuneResultEntry`] that must match the [`AutoTuneResults::version`]
    /// of the wrapping [`AutoTuneResults`] instance.
    #[prost(int32, tag = "4")]
    pub version: i32,
}

/// Collection of auto-tuning results for XLA operations. Auto-tuning results store the best-performing algorithm
/// configurations for specific operations (e.g., convolutions and matrix multiplications) on specific devices.
/// These cached results allow XLA to skip the expensive auto-tuning process on subsequent compilations.
///
/// This type corresponds to `AutotuneResults` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct AutoTuneResults {
    /// Auto-tuning cache key version.
    #[prost(int32, tag = "1")]
    pub version: i32,

    /// Collection of auto-tuning result entries.
    #[prost(message, repeated, tag = "4")]
    pub results: Vec<AutoTuneResultEntry>,
}

/// Performance characteristics for a specific type of GPU execution unit.
///
/// This type corresponds to `stream_executor.ExecutionUnitDescriptionProto.RateInfoProto`
/// in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Message)]
pub struct GpuExecutionRateInformation {
    /// Operating frequency of the execution units in GHz.
    #[prost(float, tag = "1")]
    pub clock_rate_ghz: f32,

    /// Number of execution units of this type available per streaming multiprocessor/compute unit.
    #[prost(int32, tag = "2")]
    pub units_per_core: i32,

    /// Number of operations completed by each execution unit per clock cycle.
    #[prost(int32, tag = "3")]
    pub ops_per_clock: i32,
}

/// Description of the capabilities and performance characteristics of a specific execution unit within a GPU device,
/// such as scalar units (e.g., CUDA Cores) or matrix units (e.g., Tensor Cores).
///
/// This type corresponds to `stream_executor.ExecutionUnitDescriptionProto`
/// in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct GpuExecutionUnitDescription {
    /// Mapping from [`BufferType`]s to performance characteristics for that type.
    #[prost(map = "int32, message", tag = "1")]
    pub rate_information: HashMap<i32, GpuExecutionRateInformation>,
}

/// CUDA feature extension level for NVIDIA GPUs. Starting with the Hopper architecture (i.e., compute capability 9.0),
/// NVIDIA introduced specialized feature sets that provide access to architecture-specific or family-specific
/// capabilities, particularly for Tensor Core operations. These feature extensions affect PTX compatibility
/// and determine which specialized instructions are available.
///
/// Refer to [NVIDIA's Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
/// for more details on feature extensions and their compatibility implications.
///
/// This type corresponds to `stream_executor.CudaComputeCapabilityProto.FeatureExtension`
/// in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum CudaFeatureExtension {
    /// Feature extension level is not specified. For Hopper and newer architectures, this defaults to
    /// [`Self::AcceleratedFeatures`] rather than [`Self::None`].
    Unspecified = 0,

    /// No feature extensions enabled. Code compiled without feature extensions has maximum forward compatibility
    /// across GPU architectures.
    None = 1,

    /// Architecture-specific accelerated features enabled (corresponds to the `sm_*a` PTX suffix). Code using these
    /// features is only compatible with the exact compute capability it was compiled for and has no forward or
    /// backward compatibility with other architectures.
    AcceleratedFeatures = 2,

    /// Family-compatible features enabled (corresponds to the `sm_*f` PTX suffix). Code using these features is
    /// compatible with all GPUs sharing the same major compute capability version, providing broader device support
    /// while still enabling specialized optimizations within a GPU family.
    FamilyCompatibleFeatures = 3,
}

/// CUDA compute capability for NVIDIA GPUs, which identifies the features and performance characteristics
/// supported by NVIDIA GPUs. CUDA compute capabilities correspond to NVIDIA GPU architectures.
///
/// Refer to [NVIDIA's official documentation](https://developer.nvidia.com/cuda/gpus)
/// for more details on specific compute capabilities and their supported features.
///
/// This type corresponds to `stream_executor.CudaComputeCapabilityProto`
/// in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct CudaComputeCapability {
    /// Major version number indicating the GPU architecture generation (e.g., `8` for Ampere, `9` for Hopper, etc.).
    #[prost(int32, tag = "1")]
    pub major: i32,

    /// Minor version number indicating incremental improvements within the architecture generation.
    #[prost(int32, tag = "2")]
    pub minor: i32,

    /// [`CudaFeatureExtension`] specifying which specialized features are enabled for this compute capability.
    #[prost(enumeration = "CudaFeatureExtension", tag = "3")]
    pub feature_extension: i32,
}

/// ROCm compute capability for AMD GPUs, which identifies the features and performance characteristics
/// supported by AMD GPUs. ROCm compute capabilities correspond to AMD GPU architectures.
///
/// Refer to [AMD's official documentation](https://rocm.docs.amd.com/en/latest/)
/// for more details on specific ROCm compute capabilities and their supported features.
///
/// This type corresponds to `stream_executor.RocmComputeCapabilityProto`
/// in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Eq, Hash, Message)]
pub struct RocmComputeCapability {
    /// Graphics Core Next (GCN) architecture name identifying the GPU architecture (e.g., `"gfx908"`, etc.).
    #[prost(string, tag = "1")]
    pub gcn_architecture_name: String,
}

/// GPU compute capability supporting multiple vendor-specific formats.
///
/// This type corresponds to `stream_executor.GpuComputeCapabilityProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Eq, Hash, Oneof)]
pub enum GpuComputeCapability {
    /// Compute capability for NVIDIA CUDA GPUs.
    #[prost(message, tag = "16")]
    CudaComputeCapability(CudaComputeCapability),

    /// Compute capability for AMD ROCm GPUs.
    #[prost(message, tag = "17")]
    RocmComputeCapability(RocmComputeCapability),
}

/// Information about a GPU's hardware characteristics. XLA uses this information to make optimization decisions
/// during compilation, such as choosing tile sizes for kernels or determining memory allocation strategies.
///
/// This type corresponds to `stream_executor.GpuDeviceInfoProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct GpuDeviceInformation {
    /// Maximum number of threads that can be launched in a single block.
    #[prost(int32, tag = "1")]
    pub threads_per_block_limit: i32,

    /// Number of threads in a warp (typically `32` for NVIDIA GPUs and `64` for AMD GPUs).
    #[prost(int32, tag = "2")]
    pub threads_per_warp: i32,

    /// Amount of shared memory available per block in bytes.
    #[prost(int32, tag = "3")]
    pub shared_memory_per_block_in_bytes: i32,

    /// Amount of shared memory available per streaming multiprocessor/compute unit in bytes.
    #[prost(int32, tag = "4")]
    pub shared_memory_per_core_in_bytes: i32,

    /// Maximum number of threads that can execute concurrently on a single streaming multiprocessor/compute unit.
    #[prost(int32, tag = "5")]
    pub threads_per_core_limit: i32,

    /// Number of streaming multiprocessors (NVIDIA) or compute units (AMD) on the GPU.
    #[prost(int32, tag = "6")]
    pub core_count: i32,

    /// Number of floating-point units per streaming multiprocessor/compute unit.
    #[prost(int64, tag = "7")]
    pub fpus_per_core: i64,

    /// Maximum block dimension in the `X` (i.e., first) dimension.
    #[prost(int32, tag = "8")]
    pub block_dimension_limit_x: i32,

    /// Maximum block dimension in the `Y` (i.e., second) dimension.
    #[prost(int32, tag = "9")]
    pub block_dimension_limit_y: i32,

    /// Maximum block dimension in the `Z` (i.e., third) dimension.
    #[prost(int32, tag = "10")]
    pub block_dimension_limit_z: i32,

    /// Memory bandwidth in bytes per second.
    #[prost(int64, tag = "11")]
    pub memory_bandwidth_in_bytes_per_second: i64,

    /// Size of the L2 cache in bytes.
    #[prost(int64, tag = "12")]
    pub l2_cache_size_in_bytes: i64,

    /// GPU core clock rate in GHz.
    #[prost(float, tag = "13")]
    pub clock_rate_ghz: f32,

    /// Total device memory size in bytes.
    #[prost(int64, tag = "14")]
    pub device_memory_size_in_bytes: i64,

    /// Amount of shared memory per block when opting in to higher shared memory configurations in bytes.
    /// Some GPUs support configurable shared memory sizes that trade off L1 cache for additional shared memory.
    #[prost(int32, tag = "15")]
    pub optional_shared_memory_per_block_in_bytes: i32,

    /// Maximum number of registers available per streaming multiprocessor/compute unit.
    #[prost(int64, tag = "18")]
    pub registers_per_core_limit: i64,

    /// Maximum number of registers that can be used by a single block.
    #[prost(int64, tag = "19")]
    pub registers_per_block_limit: i64,

    /// [`GpuExecutionUnitDescription`] for the scalar execution units (e.g., CUDA cores, shader processors, etc.)
    /// that handle conventional arithmetic operations like element-wise additions and multiplications.
    #[prost(message, optional, tag = "20")]
    pub scalar_unit_description: Option<GpuExecutionUnitDescription>,

    /// [`GpuExecutionUnitDescription`] for the matrix/tensor execution units (e.g., NVIDIA Tensor Cores, AMD Matrix
    /// Cores, etc.) of this GPU device that are used to accelerate matrix multiplication and linear algebra operations
    /// commonly used in deep learning workloads.
    #[prost(message, optional, tag = "21")]
    pub matrix_unit_description: Option<GpuExecutionUnitDescription>,

    /// Compute capability of the device specifying the GPU architecture and its supported features.
    #[prost(oneof = "GpuComputeCapability", tags = "16, 17")]
    pub compute_capability: Option<GpuComputeCapability>,
}

/// Version information for a GPU's Deep Neural Network (DNN) library that is used for accelerating neural network
/// operations. For NVIDIA GPUs, this corresponds to the cuDNN version. For AMD GPUs, this corresponds to the MIOpen
/// version.
///
/// This type corresponds to `stream_executor.DnnVersionInfoProto` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct GpuDnnVersionInformation {
    /// Major version number, following [semantic versioning](https://semver.org/) conventions.
    #[prost(int32, tag = "1")]
    pub major: i32,

    /// Minor version number, following [semantic versioning](https://semver.org/) conventions.
    #[prost(int32, tag = "2")]
    pub minor: i32,

    /// Patch version number, following [semantic versioning](https://semver.org/) conventions.
    #[prost(int32, tag = "3")]
    pub patch: i32,
}

/// Version information for a GPU's runtime library. For NVIDIA GPUs, this corresponds to the CUDA runtime version.
/// For AMD GPUs, this corresponds to the ROCm runtime version.
///
/// This type corresponds to `stream_executor.RuntimeVersionProto` in [XLA](https://github.com/openxla/xla).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Message)]
pub struct GpuRuntimeVersion {
    /// Major version number, following [semantic versioning](https://semver.org/) conventions.
    #[prost(int32, tag = "1")]
    pub major: i32,

    /// Minor version number, following [semantic versioning](https://semver.org/) conventions.
    #[prost(int32, tag = "2")]
    pub minor: i32,

    /// Patch version number, following [semantic versioning](https://semver.org/) conventions.
    #[prost(int32, tag = "3")]
    pub patch: i32,
}

/// GPU device target configuration for XLA compilation that aggregates all the information XLA needs
/// to compile and optimize code for a specific GPU device.
///
/// This type corresponds to `stream_executor.GpuTargetConfigProto`
/// in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct GpuTargetConfiguration {
    /// [`GpuDeviceInformation`] about the target GPU device.
    #[prost(message, optional, tag = "1")]
    pub gpu_device_information: Option<GpuDeviceInformation>,

    /// Name of the GPU platform (e.g., `"CUDA"` or `"ROCm"`).
    #[prost(string, tag = "4")]
    pub platform_name: String,

    /// [`GpuDnnVersionInformation`] for the library that is available on the target system.
    #[prost(message, optional, tag = "5")]
    pub dnn_version_information: Option<GpuDnnVersionInformation>,

    /// [`GpuRuntimeVersion`] for the runtime that is available on the target system.
    #[prost(message, optional, tag = "8")]
    pub runtime_version: Option<GpuRuntimeVersion>,

    /// Cached [`AutoTuneResults`] for the target GPU device. These results allow XLA to skip expensive auto-tuning
    /// during compilation by reusing previously determined optimal algorithm configurations.
    #[prost(message, optional, tag = "6")]
    pub autotune_results: Option<AutoTuneResults>,

    /// Human-readable description of the target GPU device, potentially containing additional metadata.
    #[prost(string, tag = "7")]
    pub device_description: String,
}

/// Configuration options for compiling a program into an executable with XLA.
///
/// This type corresponds to `CompileOptionsProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Message)]
pub struct CompilationOptions {
    /// [`Shape`]s of the computation's input arguments. Each [`Shape`] optionally specifies the memory layout for the
    /// corresponding argument. If not specified, the compiler will use potentially backend-specific default layouts.
    #[prost(message, repeated, tag = "1")]
    pub argument_layouts: Vec<Shape>,

    /// If `true`, the computation expects a single tuple argument containing all parameters.
    /// This affects how inputs are passed to the compiled executable.
    #[prost(bool, tag = "2")]
    pub parameter_is_tupled_arguments: bool,

    /// [`ExecutableCompilationOptions`] controlling how the executable is built.
    #[prost(message, optional, tag = "3")]
    pub executable_build_options: Option<ExecutableCompilationOptions>,

    /// If `true`, the compiler will produce a portable program which can later be executed on different devices.
    /// Portable executables are not specialized for specific devices and can be loaded and executed on any compatible
    /// device. This configuration option is useful for _Ahead-Of-Time (AOT)_ compilation.
    #[prost(bool, tag = "4")]
    pub compile_portable_executable: bool,

    /// XLA compilation profile version.
    #[prost(int64, tag = "5")]
    pub profile_version: i64,

    /// Serialized multi-slice configuration for distributed compilation.
    #[prost(bytes = "vec", tag = "6")]
    pub serialized_multi_slice_configuration: Vec<u8>,

    /// Environment option overrides for the compiler.
    #[prost(map = "string, message", tag = "7")]
    pub environment_option_overrides: HashMap<String, OptionOverride>,

    /// [`GpuTargetConfiguration`] for GPU compilation that specifies the target GPU architecture
    /// and capabilities when compiling for GPU backends.
    #[prost(message, optional, tag = "8")]
    pub target_config: Option<GpuTargetConfiguration>,

    /// If `true`, allows in-place modification of the input MLIR module / XLA program during compilation. This is
    /// used to run passes on the MLIR parameter without having to clone it first, thus saving memory. Additionally,
    /// it allows the compiler to deallocate the MLIR module later in the compilation process, when it is not needed
    /// anymore.
    #[prost(bool, tag = "9")]
    pub allow_in_place_mlir_modification: bool,

    /// [`Precision`] used for hardware-accelerated matrix operations.
    #[prost(enumeration = "Precision", tag = "10")]
    pub matrix_unit_operand_precision: i32,
}

/// Statistics about the memory consumption of an executable (i.e., a compiled program).
///
/// This type corresponds to `CompiledMemoryStatsProto` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Eq, Hash, Message)]
pub struct ExecutableMemoryStatistics {
    /// Number of bytes used for storing the generated code in device memory.
    #[prost(int64, tag = "1")]
    pub device_generated_code_size_in_bytes: i64,

    /// Number of bytes used for storing the executable input buffers in device memory.
    #[prost(int64, tag = "2")]
    pub device_input_size_in_bytes: i64,

    /// Number of bytes used for storing the executable output buffers in device memory.
    #[prost(int64, tag = "3")]
    pub device_output_size_in_bytes: i64,

    /// Number of _aliased_ (i.e., re-used) bytes in device memory.
    #[prost(int64, tag = "4")]
    pub device_alias_size_in_bytes: i64,

    /// Number of bytes used for temporary buffers in device memory.
    #[prost(int64, tag = "5")]
    pub device_temporary_size_in_bytes: i64,

    /// Peak number of bytes used in device memory.
    #[prost(int64, tag = "14")]
    pub device_peak_memory_in_bytes: i64,

    /// Number of bytes used for storing the generated code in host memory.
    #[prost(int64, tag = "7")]
    pub host_generated_code_size_in_bytes: i64,

    /// Number of bytes used for storing the executable input buffers in host memory.
    #[prost(int64, tag = "8")]
    pub host_input_size_in_bytes: i64,

    /// Number of bytes used for storing the executable output buffers in host memory.
    #[prost(int64, tag = "9")]
    pub host_output_size_in_bytes: i64,

    /// Number of _aliased_ (i.e., re-used) bytes in host memory.
    #[prost(int64, tag = "10")]
    pub host_alias_size_in_bytes: i64,

    /// Number of bytes used for temporary buffers in host memory.
    #[prost(int64, tag = "11")]
    pub host_temporary_size_in_bytes: i64,

    /// Legacy field that is an undocumented serialized proto message that is not semantically well-defined across HLO
    /// versions and is used only by one Google-internal library as of July 2025.
    #[deprecated]
    #[prost(bytes = "vec", tag = "13")]
    pub serialized_buffer_assignment: Vec<u8>,
}

/// Metadata associated with an executable (i.e., a compiled program).
///
/// This type corresponds to `PjRtExecutableMetadata` in [XLA](https://github.com/openxla/xla).
#[derive(Clone, PartialEq, Eq, Hash, Message)]
pub struct ExecutableMetadata {
    /// [`ExecutableMemoryStatistics`] for the executable.
    #[prost(message, optional, tag = "1")]
    pub memory_statistics: Option<ExecutableMemoryStatistics>,

    /// Additional platform-specific metadata associated with the executable.
    #[prost(message, optional, tag = "2")]
    pub platform_specific_metadata: Option<ProtoAny>,
}

/// Configuration options for a profiling session created through the PJRT profiler extension.
///
/// This type corresponds to `ProfileOptions` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/profiler_options.proto).
#[derive(Clone, PartialEq, Message)]
pub struct ProfileOptions {
    /// Version number used to determine whether default values should be applied to newly added fields.
    /// Clients should set this to the latest version they support so that the profiler can correctly interpret
    /// the remaining fields.
    #[prost(uint32, tag = "5")]
    pub version: u32,

    /// Type of device to profile.
    #[prost(enumeration = "ProfileDeviceType", tag = "6")]
    pub device_type: i32,

    /// Boolean flag indicating whether to include dataset operations in the trace. Defaults to `false` if not provided.
    #[prost(bool, tag = "1")]
    pub include_dataset_operations: bool,

    /// Host (i.e., CPU) tracing level:
    ///  - **Level `0`:** Disables host traces.
    ///  - **Level `1`:** Enables tracing of only user instrumented (or default) _TraceMe_.
    ///    This is the default value, if not provided.
    ///  - **Level `2`:** Enables tracing of all level 1 _TraceMe(s)_ and instrumented high level program execution
    ///    details (e.g., expensive XLA operations, etc.).
    ///  - **Level `3`:** Enables tracing of all level 2 _TraceMe(s)_ and more verbose (i.e., low-level) program
    ///    execution details (e.g., including cheap XLA operations, etc.).
    #[prost(uint32, tag = "2")]
    pub host_tracing_level: u32,

    /// Device (e.g., GPU or TPU) tracing level:
    ///   - **Level `0`:** Disables device traces.
    ///   - **Level `1`:** Enables device traces.
    /// 
    /// More levels might be defined for specific device types (i.e., backends) for controlling the trace verbosity.
    #[prost(uint32, tag = "3")]
    pub device_tracing_level: u32,

    /// Python function call tracing level. Level `0` disables Python tracing, and higher levels increase verbosity.
    #[prost(uint32, tag = "4")]
    pub python_tracing_level: u32,

    /// Boolean flag indicating whether to serialize the HLO proto when XLA is used.
    #[prost(bool, tag = "7")]
    pub enable_hlo_proto: bool,

    /// Local profiler start timestamp in nanoseconds since the UNIX epoch.
    #[prost(uint64, tag = "8")]
    pub start_timestamp_ns: u64,

    /// Profiling duration in milliseconds. A value of `0` (the default) means that profiling will continue until
    /// explicitly interrupted.
    #[prost(uint64, tag = "9")]
    pub duration_ms: u64,

    /// Directory in which to save the profiling data (defaults to not saving the data, if not specified).
    #[prost(string, tag = "10")]
    pub repository_path: String,

    /// Trace options that control host-side trace filtering.
    #[prost(message, optional, tag = "11")]
    pub trace_options: Option<ProfileTraceOptions>,

    /// Advanced configuration key-value pairs for backend-specific or experimental profiler settings.
    #[prost(map = "string, message", tag = "12")]
    pub advanced_configuration: HashMap<String, ProfileAdvancedConfigurationValue>,

    /// Boolean flag indicating whether to raise an error if the profiler fails to start
    /// (as opposed to silently ignoring the failure).
    #[prost(bool, tag = "13")]
    pub raise_error_on_start_failure: bool,

    /// Profiling session identifier. When set, this is used as a subdirectory name under
    /// [`ProfileOptions::repository_path`] for storing the profiling data.
    #[prost(string, tag = "14")]
    pub session_id: String,

    /// Optional hostname override to use for naming the profiling output file in [`ProfileOptions::repository_path`].
    /// If not specified, the actual hostname will be used.
    #[prost(string, tag = "15")]
    pub override_hostname: String,
}

/// Device type to profile.
///
/// This type corresponds to `ProfileOptions.DeviceType` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/profiler_options.proto).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Enumeration)]
#[repr(i32)]
pub enum ProfileDeviceType {
    /// Unknown or unspecified device type.
    Unspecified = 0,

    /// Profile CPU operations.
    Cpu = 1,

    /// Profile CPU and GPU operations.
    Gpu = 2,

    /// Profile CPU and TPU operations.
    Tpu = 3,

    /// Profile CPU and pluggable device (i.e., a custom PJRT plugin backend) operations.
    PluggableDevice = 4,
}

/// Trace options that control host-side trace filtering.
///
/// This type corresponds to `ProfileOptions.TraceOptions` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/profiler_options.proto).
#[derive(Clone, PartialEq, Message)]
pub struct ProfileTraceOptions {
    /// Filter mask for _TraceMe_ events. If this mask is set, a _TraceMe_ event will be recorded if it passes the
    /// filter. Only the lowest 32 bits of the mask are used. The higher 32 bits are reserved and will be ignored.
    #[prost(uint64, tag = "1")]
    pub host_trace_me_filter_mask: u64,
}

/// Value in a [`ProfileOptions`] advanced configuration map entry.
///
/// This type corresponds to `ProfileOptions.AdvancedConfigValue` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/profiler_options.proto).
#[derive(Clone, PartialEq, Message)]
pub struct ProfileAdvancedConfigurationValue {
    /// Configuration value.
    #[prost(oneof = "ProfileAdvancedConfigurationValueKind", tags = "1, 2, 3")]
    pub value: Option<ProfileAdvancedConfigurationValueKind>,
}

/// Kind of value stored in a [`ProfileAdvancedConfigurationValue`].
///
/// This type corresponds to `ProfileOptions.AdvancedConfigValue.value` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/profiler_options.proto).
#[derive(Clone, PartialEq, Oneof)]
pub enum ProfileAdvancedConfigurationValueKind {
    /// String configuration value.
    #[prost(string, tag = "1")]
    StringValue(String),

    /// Boolean configuration value.
    #[prost(bool, tag = "2")]
    BoolValue(bool),

    /// 64-bit integer configuration value.
    #[prost(int64, tag = "3")]
    Int64Value(i64),
}

/// Top-level container for the results of one or more profiling sources. An [`XSpace`] holds one or more [`XPlane`]s,
/// each representing traces from a different profiling source (e.g., a CPU host, a GPU device, or a backend-specific
/// tracer).
///
/// This type corresponds to `XSpace` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/xplane.proto).
#[derive(Clone, PartialEq, Message)]
pub struct XSpace {
    /// Parallel [`XPlane`]s from different profiling sources.
    #[prost(message, repeated, tag = "1")]
    pub planes: Vec<XPlane>,

    /// Errors that were raised during [`XPlane`] generation.
    #[prost(string, repeated, tag = "2")]
    pub errors: Vec<String>,

    /// Warnings that were logged during [`XPlane`] generation.
    #[prost(string, repeated, tag = "3")]
    pub warnings: Vec<String>,

    /// Hostnames from which the [`XPlane`]s were generated.
    #[prost(string, repeated, tag = "4")]
    pub hostnames: Vec<String>,
}

/// Container of multiple parallel timelines (i.e., [`XLine`]s), generated by a profiling source, or by post-processing
/// one or more [`XPlane`]s. A plane represents traces from a specific source (e.g., a CPU host, a GPU device, or a
/// backend-specific tracer), represented as parallel timelines (i.e., [`XLine`]s), [`XEventMetadata`], and
/// [`XStatMetadata`] that are shared across all events within the plane.
///
/// This type corresponds to `XPlane` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/xplane.proto).
#[derive(Clone, PartialEq, Message)]
pub struct XPlane {
    /// Unique identifier for this [`XPlane`] within its parent [`XSpace`].
    #[prost(int64, tag = "1")]
    pub id: i64,

    /// Name of this [`XPlane`] (e.g., `"/host:CPU"`, `"/device:GPU:0"`).
    #[prost(string, tag = "2")]
    pub name: String,

    /// Parallel timelines that grouped in this [`XPlane`]. Note that [`XLine`]s with the same [`XLine::id`] represent
    /// the same timeline across different time ranges.
    #[prost(message, repeated, tag = "3")]
    pub lines: Vec<XLine>,

    /// [`XEvent`] metadata, keyed by [`XEventMetadata::id`]. This map should be used for [`XEvent`]s that share the
    /// same ID over the whole [`XPlane`].
    #[prost(map = "int64, message", tag = "4")]
    pub event_metadata: HashMap<i64, XEventMetadata>,

    /// [`XStat`] metadata, keyed by [`XStatMetadata::id`]. This map should be used for [`XStat`]s that share the
    /// same ID over the whole [`XPlane`].
    #[prost(map = "int64, message", tag = "5")]
    pub stat_metadata: HashMap<i64, XStatMetadata>,

    /// [`XStat`]s associated with this [`XPlane`] as a whole (e.g., device capabilities or configuration).
    #[prost(message, repeated, tag = "6")]
    pub stats: Vec<XStat>,
}

/// Timeline within an [`XPlane`] that contains a sequence of [`XEvent`]s. Each line represents a logical thread,
/// stream, or activity timeline. [`XEvent`] on a line should not partially overlap but may be nested (i.e., one event
/// may be fully contained within another).
///
/// This type corresponds to `XLine` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/xplane.proto).
#[derive(Clone, PartialEq, Message)]
#[prost(reserved = "5, 6, 7, 8")]
pub struct XLine {
    /// Identifier for this [`XLine`] within its parent [`XPlane`]. [`XLine`]s with the same ID across different time
    /// ranges represent the same timeline.
    #[prost(int64, tag = "1")]
    pub id: i64,

    /// Display identifier for this [`XLine`]. [`XLine`]s with the same display ID are grouped in the same row in trace
    /// viewers.
    #[prost(int64, tag = "10")]
    pub display_id: i64,

    /// Name of this [`XLine`].
    #[prost(string, tag = "2")]
    pub name: String,

    /// Display name of this [`XLine`] that is shown in trace viewers. If not specified, [`XLine::name`] will be used.
    #[prost(string, tag = "11")]
    pub display_name: String,

    /// Start time of this [`XLine`] in nanoseconds since the UNIX epoch.
    #[prost(int64, tag = "3")]
    pub timestamp_ns: i64,

    /// Profiling duration of this [`XLine`] in picoseconds.
    #[prost(int64, tag = "9")]
    pub duration_ps: i64,

    /// [`XEvent`]s on this [`XLine`], ordered by start time. Events should not partially overlap but can be nested.
    #[prost(message, repeated, tag = "4")]
    pub events: Vec<XEvent>,
}

/// Discrete timed operation on an [`XLine`] timeline.
///
/// This type corresponds to `XEvent` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/xplane.proto).
#[derive(Clone, PartialEq, Message)]
pub struct XEvent {
    /// ID of the [`XEventMetadata`] associated with this [`XEvent`] in the parent [`XPlane`]'s
    /// [`XPlane::event_metadata`] map.
    #[prost(int64, tag = "1")]
    pub metadata_id: i64,

    /// Timing data for this [`XEvent`].
    #[prost(oneof = "XEventData", tags = "2, 5")]
    pub data: Option<XEventData>,

    /// Duration of this [`XEvent`] in picoseconds. A value of `0` indicates an instant event.
    #[prost(int64, tag = "3")]
    pub duration_ps: i64,

    /// [`XStat`]s associated with this [`XEvent`].
    #[prost(message, repeated, tag = "4")]
    pub stats: Vec<XStat>,
}

/// Timing data for an [`XEvent`].
///
/// This type corresponds to `XEvent.data` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/xplane.proto).
#[derive(Clone, PartialEq, Oneof)]
pub enum XEventData {
    /// Start time of the event in picoseconds, as an offset from the parent [`XLine`]'s [`XLine::timestamp_ns`].
    #[prost(int64, tag = "2")]
    OffsetPs(i64),

    /// Number of occurrences of this event that is used when events are aggregated.
    #[prost(int64, tag = "5")]
    OccurrenceCount(i64),
}

/// Named value associated with an [`XEvent`] or an [`XPlane`] (e.g., a performance counter value, a metric computed
/// by a formula applied over nested [`XEvent`]s and their [`XStat`]s, etc.).
///
/// This type corresponds to `XStat` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/xplane.proto).
#[derive(Clone, PartialEq, Message)]
pub struct XStat {
    /// ID of the [`XStatMetadata`] associated with this [`XStat`] in the parent [`XPlane`]'s
    /// [`XPlane::stat_metadata`] map.
    #[prost(int64, tag = "1")]
    pub metadata_id: i64,

    /// Value of this [`XStat`].
    #[prost(oneof = "XStatValue", tags = "2, 3, 4, 5, 6, 7")]
    pub value: Option<XStatValue>,
}

/// Value stored in an [`XStat`].
///
/// This type corresponds to `XStat.value` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/xplane.proto).
#[derive(Clone, PartialEq, Oneof)]
pub enum XStatValue {
    /// Double-precision floating-point value.
    #[prost(double, tag = "2")]
    DoubleValue(f64),

    /// Unsigned 64-bit integer value.
    #[prost(uint64, tag = "3")]
    Uint64Value(u64),

    /// Signed 64-bit integer value.
    #[prost(int64, tag = "4")]
    Int64Value(i64),

    /// String value.
    #[prost(string, tag = "5")]
    StrValue(String),

    /// Raw bytes value.
    #[prost(bytes, tag = "6")]
    BytesValue(Vec<u8>),

    /// Reference to an [`XStatMetadata`] entry by ID that can be used to look up the metadata in the
    /// parent [`XPlane`]'s [`XPlane::stat_metadata`] map. The string value is stored in the referenced
    /// [`XStatMetadata::name`].
    #[prost(uint64, tag = "7")]
    RefValue(u64),
}

/// Metadata associated with one or more [`XEvent`]s.
///
/// This type corresponds to `XEventMetadata` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/xplane.proto).
#[derive(Clone, PartialEq, Message)]
pub struct XEventMetadata {
    /// Unique identifier for this [`XEvent`] metadata within the parent [`XPlane`]'s [`XPlane::event_metadata`] map.
    #[prost(int64, tag = "1")]
    pub id: i64,

    /// Name of the [`XEvent`].
    #[prost(string, tag = "2")]
    pub name: String,

    /// Display name of this [`XEvent`] that is shown in trace viewers. If not specified, [`XEventMetadata::name`]
    /// will be used.
    #[prost(string, tag = "4")]
    pub display_name: String,

    /// Additional metadata in a serialized format (e.g., a nested Protobuf message).
    #[prost(bytes = "vec", tag = "3")]
    pub metadata: Vec<u8>,

    /// [`XStat`]s that are constant across all [`XEvent`]s with [`XEvent::metadata_id`] that matches this
    /// [`XEventMetadata`]'s [`XEventMetadata::id`].
    #[prost(message, repeated, tag = "5")]
    pub stats: Vec<XStat>,

    /// IDs of child [`XEventMetadata`] instances in the parent [`XPlane`]'s [`XPlane::event_metadata`] map.
    #[prost(int64, repeated, tag = "6")]
    pub child_id: Vec<i64>,
}

/// Metadata associated with one or more [`XStat`]s.
///
/// This type corresponds to `XStatMetadata` in
/// [TSL](https://github.com/google/tsl/blob/main/tsl/profiler/protobuf/xplane.proto).
#[derive(Clone, PartialEq, Message)]
pub struct XStatMetadata {
    /// Unique identifier for this [`XStat`] metadata within the parent [`XPlane`]'s [`XPlane::stat_metadata`] map.
    #[prost(int64, tag = "1")]
    pub id: i64,

    /// Name of the [`XStat`] that should ideally be kept short for efficiency reasons as it may appear multiple times.
    #[prost(string, tag = "2")]
    pub name: String,

    /// Description of the [`XStat`] that may be long.
    #[prost(string, tag = "3")]
    pub description: String,
}
