use std::cell::{Ref, RefCell, RefMut, UnsafeCell};
use std::fmt::{Debug, Display};
use std::iter::Peekable;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::rc::Rc;
use std::str::Chars;

use crate::{Api, Client, Device, Error, Event, HasDefaultMemory, Memory, invoke_pjrt_api_error_fn, slice_from_c_api};

/// Type of the data stored in a [`Buffer`]. Specifically, this represents
/// the type of individual values that are stored in [`Buffer`]s.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufferType {
    /// Invalid [`BufferType`] that serves as a default.
    Invalid,

    /// [`BufferType`] that represents token values that are threaded between side-effecting operations.
    /// This type is only used for buffers that contain a single value (i.e., that represent scalar values).
    Token,

    /// Predicate [`BufferType`] that represents the `true` and `false` values.
    Predicate,

    /// [`BufferType`] that represents signed 2-bit integer values.
    I2,

    /// [`BufferType`] that represents signed 4-bit integer values.
    I4,

    /// [`BufferType`] that represents signed 8-bit integer values.
    I8,

    /// [`BufferType`] that represents signed 16-bit integer values.
    I16,

    /// [`BufferType`] that represents signed 32-bit integer values.
    I32,

    /// [`BufferType`] that represents signed 64-bit integer values.
    I64,

    /// [`BufferType`] that represents unsigned 2-bit integer values.
    U2,

    /// [`BufferType`] that represents unsigned 4-bit integer values.
    U4,

    /// [`BufferType`] that represents unsigned 8-bit integer values.
    U8,

    /// [`BufferType`] that represents unsigned 16-bit integer values.
    U16,

    /// [`BufferType`] that represents unsigned 32-bit integer values.
    U32,

    /// [`BufferType`] that represents unsigned 64-bit integer values.
    U64,

    /// [`BufferType`] that represents 4-bit floating-point values that are represented using a
    /// [microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
    /// format with 2 exponent bits and 1 mantissa bit. Only finite values are supported (thus the `FN` suffix).
    /// Unlike IEEE floating-point types, infinities and NaN values are not supported.
    F4E2M1FN,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 3 exponent bits and 4 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E3M4,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E4M3,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits, and without support for
    /// representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values are
    /// represented with the exponent and mantissa bits all set to `1`. All other bit configurations represent finite
    /// values.
    F8E4M3FN,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2206.02915) with 4 exponent bits and 3 mantissa bits, and without support for
    /// representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values are
    /// represented with the exponent and mantissa bits all set to `0` and the sign bit is set to `1`. All other bit
    /// configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    ///
    /// The difference between this type and [`BufferType::F8E4M3FN`] is that there is an additional exponent value
    /// available. To keep the same dynamic range as an IEEE-like 8-bit floating-point type, the exponent is biased one
    /// more than would be expected given the number of exponent bits (i.e., bias set to `8`).
    F8E4M3FNUZ,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits and a bias of `11`, and
    /// without support for representing infinity values, unlike existing IEEE floating-point types (thus the `FN`
    /// suffix). NaN values are represented with the exponent and mantissa bits all set to `0` and the sign bit is set
    /// to `1`. All other bit configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    F8E4M3B11FNUZ,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 5 exponent bits and 2 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E5M2,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2206.02915) with 5 exponent bits and 2 mantissa bits, and without support for
    /// representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values are
    /// represented with the exponent and mantissa bits all set to `0` and the sign bit is set to `1`. All other bit
    /// configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    ///
    /// The difference between this type and [`BufferType::F8E5M2`] is that there is an additional exponent value
    /// available. To keep the same dynamic range as an IEEE-like 8-bit floating-point type, the exponent is biased one
    /// more than would be expected given the number of exponent bits (i.e., bias set to `16`).
    F8E5M2FNUZ,

    /// [`BufferType`] that represents 8-bit floating-point values that are represented using a
    /// [microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
    /// format with 8 exponent bits and no mantissa or sign bits. Only unsigned finite values are supported
    /// (thus the `FNU` suffix). Unlike IEEE floating-point types, infinity and NaN values are not supported.
    F8E8M0FNU,

    /// [`BufferType`] that represents 16-bit floating-point values with 8 exponent bits, 7 mantissa bits, and 1 sign
    /// bit. This type offers a larger dynamic range than [`BufferType::F16`] at the cost of lower precision.
    BF16,

    /// [`BufferType`] that represents 16-bit floating-point values with 5 exponent bits, 10 mantissa bits, and 1 sign
    /// bit, using the standard IEEE floating-point representation.
    F16,

    /// [`BufferType`] that represents 32-bit floating-point values with 8 exponent bits, 24 mantissa bits, and 1 sign
    /// bit, using the standard IEEE floating-point representation.
    F32,

    /// [`BufferType`] that represents 64-bit floating-point values with 11 exponent bits, 53 mantissa bits, and 1 sign
    /// bit, using the standard IEEE floating-point representation.
    F64,

    /// [`BufferType`] that represents 64-bit complex-valued floating-point values as pairs of
    /// 32-bit real floating-point values.
    C64,

    /// [`BufferType`] that represents 128-bit complex-valued floating-point values as pairs of
    /// 64-bit real floating-point values.
    C128,
}

impl BufferType {
    /// Constructs a new [`BufferType`] from the provided [`PJRT_Buffer_Type`](ffi::PJRT_Buffer_Type)
    /// that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(r#type: ffi::PJRT_Buffer_Type) -> Self {
        match r#type {
            ffi::PJRT_Buffer_Type_TOKEN => Self::Token,
            ffi::PJRT_Buffer_Type_PRED => Self::Predicate,
            ffi::PJRT_Buffer_Type_S2 => Self::I2,
            ffi::PJRT_Buffer_Type_S4 => Self::I4,
            ffi::PJRT_Buffer_Type_S8 => Self::I8,
            ffi::PJRT_Buffer_Type_S16 => Self::I16,
            ffi::PJRT_Buffer_Type_S32 => Self::I32,
            ffi::PJRT_Buffer_Type_S64 => Self::I64,
            ffi::PJRT_Buffer_Type_U2 => Self::U2,
            ffi::PJRT_Buffer_Type_U4 => Self::U4,
            ffi::PJRT_Buffer_Type_U8 => Self::U8,
            ffi::PJRT_Buffer_Type_U16 => Self::U16,
            ffi::PJRT_Buffer_Type_U32 => Self::U32,
            ffi::PJRT_Buffer_Type_U64 => Self::U64,
            ffi::PJRT_Buffer_Type_F4E2M1FN => Self::F4E2M1FN,
            ffi::PJRT_Buffer_Type_F8E3M4 => Self::F8E3M4,
            ffi::PJRT_Buffer_Type_F8E4M3 => Self::F8E4M3,
            ffi::PJRT_Buffer_Type_F8E4M3FN => Self::F8E4M3FN,
            ffi::PJRT_Buffer_Type_F8E4M3FNUZ => Self::F8E4M3FNUZ,
            ffi::PJRT_Buffer_Type_F8E4M3B11FNUZ => Self::F8E4M3B11FNUZ,
            ffi::PJRT_Buffer_Type_F8E5M2 => Self::F8E5M2,
            ffi::PJRT_Buffer_Type_F8E5M2FNUZ => Self::F8E5M2FNUZ,
            ffi::PJRT_Buffer_Type_F8E8M0FNU => Self::F8E8M0FNU,
            ffi::PJRT_Buffer_Type_BF16 => Self::BF16,
            ffi::PJRT_Buffer_Type_F16 => Self::F16,
            ffi::PJRT_Buffer_Type_F32 => Self::F32,
            ffi::PJRT_Buffer_Type_F64 => Self::F64,
            ffi::PJRT_Buffer_Type_C64 => Self::C64,
            ffi::PJRT_Buffer_Type_C128 => Self::C128,
            _ => Self::Invalid,
        }
    }

    /// Returns the [`PJRT_Buffer_Type`](ffi::PJRT_Buffer_Type) that corresponds to this [`BufferType`]
    /// and which can be passed to functions in the PJRT C API.
    #[allow(clippy::wrong_self_convention)]
    pub(crate) unsafe fn to_c_api(&self) -> ffi::PJRT_Buffer_Type {
        match self {
            Self::Invalid => ffi::PJRT_Buffer_Type_INVALID,
            Self::Token => ffi::PJRT_Buffer_Type_TOKEN,
            Self::Predicate => ffi::PJRT_Buffer_Type_PRED,
            Self::I2 => ffi::PJRT_Buffer_Type_S2,
            Self::I4 => ffi::PJRT_Buffer_Type_S4,
            Self::I8 => ffi::PJRT_Buffer_Type_S8,
            Self::I16 => ffi::PJRT_Buffer_Type_S16,
            Self::I32 => ffi::PJRT_Buffer_Type_S32,
            Self::I64 => ffi::PJRT_Buffer_Type_S64,
            Self::U2 => ffi::PJRT_Buffer_Type_U2,
            Self::U4 => ffi::PJRT_Buffer_Type_U4,
            Self::U8 => ffi::PJRT_Buffer_Type_U8,
            Self::U16 => ffi::PJRT_Buffer_Type_U16,
            Self::U32 => ffi::PJRT_Buffer_Type_U32,
            Self::U64 => ffi::PJRT_Buffer_Type_U64,
            Self::F4E2M1FN => ffi::PJRT_Buffer_Type_F4E2M1FN,
            Self::F8E3M4 => ffi::PJRT_Buffer_Type_F8E3M4,
            Self::F8E4M3 => ffi::PJRT_Buffer_Type_F8E4M3,
            Self::F8E4M3FN => ffi::PJRT_Buffer_Type_F8E4M3FN,
            Self::F8E4M3FNUZ => ffi::PJRT_Buffer_Type_F8E4M3FNUZ,
            Self::F8E4M3B11FNUZ => ffi::PJRT_Buffer_Type_F8E4M3B11FNUZ,
            Self::F8E5M2 => ffi::PJRT_Buffer_Type_F8E5M2,
            Self::F8E5M2FNUZ => ffi::PJRT_Buffer_Type_F8E5M2FNUZ,
            Self::F8E8M0FNU => ffi::PJRT_Buffer_Type_F8E8M0FNU,
            Self::BF16 => ffi::PJRT_Buffer_Type_BF16,
            Self::F16 => ffi::PJRT_Buffer_Type_F16,
            Self::F32 => ffi::PJRT_Buffer_Type_F32,
            Self::F64 => ffi::PJRT_Buffer_Type_F64,
            Self::C64 => ffi::PJRT_Buffer_Type_C64,
            Self::C128 => ffi::PJRT_Buffer_Type_C128,
        }
    }

    /// Parses a rendered [`BufferType`] (e.g., an XLA primitive type string) into a [`BufferType`].
    #[allow(clippy::should_implement_trait)]
    pub fn from_str<S: AsRef<str>>(value: S) -> Result<Self, Error> {
        let value = value.as_ref();
        match value.trim().to_ascii_lowercase().as_str() {
            "invalid" => Ok(Self::Invalid),
            "pred" => Ok(Self::Predicate),
            "token" => Ok(Self::Token),
            "s2" | "i2" => Ok(Self::I2),
            "s4" | "i4" => Ok(Self::I4),
            "s8" | "i8" => Ok(Self::I8),
            "s16" | "i16" => Ok(Self::I16),
            "s32" | "i32" => Ok(Self::I32),
            "s64" | "i64" => Ok(Self::I64),
            "u2" => Ok(Self::U2),
            "u4" => Ok(Self::U4),
            "u8" => Ok(Self::U8),
            "u16" => Ok(Self::U16),
            "u32" => Ok(Self::U32),
            "u64" => Ok(Self::U64),
            "f4e2m1fn" => Ok(Self::F4E2M1FN),
            "f8e3m4" => Ok(Self::F8E3M4),
            "f8e4m3" => Ok(Self::F8E4M3),
            "f8e4m3fn" => Ok(Self::F8E4M3FN),
            "f8e4m3fnuz" => Ok(Self::F8E4M3FNUZ),
            "f8e4m3b11fnuz" => Ok(Self::F8E4M3B11FNUZ),
            "f8e5m2" => Ok(Self::F8E5M2),
            "f8e5m2fnuz" => Ok(Self::F8E5M2FNUZ),
            "f8e8m0fnu" => Ok(Self::F8E8M0FNU),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "c64" => Ok(Self::C64),
            "c128" => Ok(Self::C128),
            _ => Err(Error::invalid_argument(format!("invalid buffer type '{value}'"))),
        }
    }

    /// Constructs a [`BufferType`] from the provided [`BufferType`](crate::protos::BufferType) Protobuf.
    pub fn from_proto(r#type: crate::protos::BufferType) -> Self {
        match r#type {
            crate::protos::BufferType::Invalid => Self::Invalid,
            crate::protos::BufferType::Token => Self::Token,
            crate::protos::BufferType::Predicate => Self::Predicate,
            crate::protos::BufferType::I2 => Self::I2,
            crate::protos::BufferType::I4 => Self::I4,
            crate::protos::BufferType::I8 => Self::I8,
            crate::protos::BufferType::I16 => Self::I16,
            crate::protos::BufferType::I32 => Self::I32,
            crate::protos::BufferType::I64 => Self::I64,
            crate::protos::BufferType::U2 => Self::U2,
            crate::protos::BufferType::U4 => Self::U4,
            crate::protos::BufferType::U8 => Self::U8,
            crate::protos::BufferType::U16 => Self::U16,
            crate::protos::BufferType::U32 => Self::U32,
            crate::protos::BufferType::U64 => Self::U64,
            crate::protos::BufferType::F4E2M1FN => Self::F4E2M1FN,
            crate::protos::BufferType::F8E3M4 => Self::F8E3M4,
            crate::protos::BufferType::F8E4M3 => Self::F8E4M3,
            crate::protos::BufferType::F8E4M3FN => Self::F8E4M3FN,
            crate::protos::BufferType::F8E4M3FNUZ => Self::F8E4M3FNUZ,
            crate::protos::BufferType::F8E4M3B11FNUZ => Self::F8E4M3B11FNUZ,
            crate::protos::BufferType::F8E5M2 => Self::F8E5M2,
            crate::protos::BufferType::F8E5M2FNUZ => Self::F8E5M2FNUZ,
            crate::protos::BufferType::F8E8M0FNU => Self::F8E8M0FNU,
            crate::protos::BufferType::BF16 => Self::BF16,
            crate::protos::BufferType::F16 => Self::F16,
            crate::protos::BufferType::F32 => Self::F32,
            crate::protos::BufferType::F64 => Self::F64,
            crate::protos::BufferType::C64 => Self::C64,
            crate::protos::BufferType::C128 => Self::C128,
            _ => Self::Invalid,
        }
    }

    /// Returns the [`BufferType`](crate::protos::BufferType) Protobuf that corresponds to this [`BufferType`].
    pub fn proto(&self) -> crate::protos::BufferType {
        match self {
            Self::Invalid => crate::protos::BufferType::Invalid,
            Self::Token => crate::protos::BufferType::Token,
            Self::Predicate => crate::protos::BufferType::Predicate,
            Self::I2 => crate::protos::BufferType::I2,
            Self::I4 => crate::protos::BufferType::I4,
            Self::I8 => crate::protos::BufferType::I8,
            Self::I16 => crate::protos::BufferType::I16,
            Self::I32 => crate::protos::BufferType::I32,
            Self::I64 => crate::protos::BufferType::I64,
            Self::U2 => crate::protos::BufferType::U2,
            Self::U4 => crate::protos::BufferType::U4,
            Self::U8 => crate::protos::BufferType::U8,
            Self::U16 => crate::protos::BufferType::U16,
            Self::U32 => crate::protos::BufferType::U32,
            Self::U64 => crate::protos::BufferType::U64,
            Self::F4E2M1FN => crate::protos::BufferType::F4E2M1FN,
            Self::F8E3M4 => crate::protos::BufferType::F8E3M4,
            Self::F8E4M3 => crate::protos::BufferType::F8E4M3,
            Self::F8E4M3FN => crate::protos::BufferType::F8E4M3FN,
            Self::F8E4M3FNUZ => crate::protos::BufferType::F8E4M3FNUZ,
            Self::F8E4M3B11FNUZ => crate::protos::BufferType::F8E4M3B11FNUZ,
            Self::F8E5M2 => crate::protos::BufferType::F8E5M2,
            Self::F8E5M2FNUZ => crate::protos::BufferType::F8E5M2FNUZ,
            Self::F8E8M0FNU => crate::protos::BufferType::F8E8M0FNU,
            Self::BF16 => crate::protos::BufferType::BF16,
            Self::F16 => crate::protos::BufferType::F16,
            Self::F32 => crate::protos::BufferType::F32,
            Self::F64 => crate::protos::BufferType::F64,
            Self::C64 => crate::protos::BufferType::C64,
            Self::C128 => crate::protos::BufferType::C128,
        }
    }
}

impl Display for BufferType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Invalid => "invalid",
            Self::Token => "token",
            Self::Predicate => "pred",
            Self::I2 => "i2",
            Self::I4 => "i4",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U2 => "u2",
            Self::U4 => "u4",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F4E2M1FN => "f4e2m1fn",
            Self::F8E3M4 => "f8e3m4",
            Self::F8E4M3 => "f8e4m3",
            Self::F8E4M3FN => "f8e4m3fn",
            Self::F8E4M3FNUZ => "f8e4m3fnuz",
            Self::F8E4M3B11FNUZ => "f8e4m3b11fnuz",
            Self::F8E5M2 => "f8e5m2",
            Self::F8E5M2FNUZ => "f8e5m2fnuz",
            Self::F8E8M0FNU => "f8e8m0fnu",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C64 => "c64",
            Self::C128 => "c128",
        })
    }
}

/// Describes a dimension of a [`Tile`].
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TileDimension(i64);

impl TileDimension {
    /// Creates a new [`TileDimension`] with a fixed size (i.e., number of elements).
    pub fn sized(size: usize) -> Self {
        Self(size as i64)
    }

    /// Creates a new _combined_ [`TileDimension`]. That is a tile dimension that is combined with the next minor
    /// dimension before tiling is applied, and thus has not fixed size of its own.
    pub fn combined() -> Self {
        Self(i64::MIN)
    }

    /// Returns the size (i.e., number of elements) of this [`TileDimension`] if it has one. The only case when a tile
    /// dimension does not have a fixed size is when it is a combined tile dimension. Refer to [`Self::is_combined`]
    /// for more information.
    pub fn size(&self) -> Option<usize> {
        if self.0 == i64::MIN { None } else { Some(self.0 as usize) }
    }

    /// Returns `true` if this [`TileDimension`] is combined with the next minor dimension
    /// before tiling is applied, and thus has no fixed size of its own.
    pub fn is_combined(&self) -> bool {
        self.0 == i64::MIN
    }
}

// Our [`Display`] implementation attempts to match the corresponding XLA rendering.
impl Display for TileDimension {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            i64::MIN => write!(formatter, "*"),
            size => write!(formatter, "{size}"),
        }
    }
}

impl Debug for TileDimension {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "TileDimension[{self}]")
    }
}

/// Tile used in a [`TiledLayout`]. Refer to the [official XLA documentation](https://openxla.org/xla/tiled_layout)
/// for more information.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tile {
    /// Dimensions of this [`Tile`], ordered from the most major dimension to the most minor dimension.
    /// The dimensions of a tile correspond to a suffix of the dimensions of the tiled [`Buffer`].
    pub dimensions: Vec<TileDimension>,
}

// Our [`Display`] implementation attempts to match the corresponding XLA rendering.
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

impl Debug for Tile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Tile[{self}]")
    }
}

/// Tiling-based [`Layout`]. Refer to the [official XLA documentation](https://openxla.org/xla/tiled_layout)
/// for more information.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TiledLayout {
    /// Sequence of dimension indices ordered from the most minor (i.e., the one with the fastest varying index)
    /// to the most major (i.e., the one with the slowest varying index). Refer to [`Self::minor_to_major`]
    /// for more information.
    minor_to_major: Vec<i64>,

    /// Flattened list of [`TileDimension`]s across all [`Tile`]s in this [`TiledLayout`].
    tile_dimensions: Vec<TileDimension>,

    /// Sequence containing the number of [`TileDimension`]s for each [`Tile`] in this [`TiledLayout`].
    tile_dimension_sizes: Vec<usize>,

    /// Number of [`Tile`]s in this [`TiledLayout`].
    tile_count: usize,
}

impl TiledLayout {
    /// Constructs a new [`TiledLayout`] from the provided
    /// [`PJRT_Buffer_MemoryLayout_Tiled`](ffi::PJRT_Buffer_MemoryLayout_Tiled)
    /// handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(layout: ffi::PJRT_Buffer_MemoryLayout_Tiled) -> Self {
        let minor_to_major = unsafe { slice_from_c_api(layout.minor_to_major, layout.minor_to_major_size) }.to_vec();
        let tile_dimension_sizes = unsafe { slice_from_c_api(layout.tile_dim_sizes, layout.num_tiles) }.to_vec();
        let tile_dimensions_count = tile_dimension_sizes.iter().sum();
        let tile_dimensions = unsafe { slice_from_c_api(layout.tile_dims as *const _, tile_dimensions_count) }.to_vec();
        Self { minor_to_major, tile_dimensions, tile_dimension_sizes, tile_count: layout.num_tiles }
    }

    /// Returns the [`PJRT_Buffer_MemoryLayout_Tiled`](ffi::PJRT_Buffer_MemoryLayout_Tiled)
    /// that corresponds to this [`TiledLayout`] and which can be passed to functions in the PJRT C API.
    ///
    /// Note that the resulting [`PJRT_Buffer_MemoryLayout_Tiled`](ffi::PJRT_Buffer_MemoryLayout_Tiled) may become
    /// invalid if this [`TiledLayout`] is dropped. Therefore, you must make sure to keep this [`TiledLayout`] alive
    /// for as long as you intend to use the returned C API data structure. This is among the reasons for which this
    /// function is marked as _unsafe_.
    pub(crate) unsafe fn to_c_api(&self) -> ffi::PJRT_Buffer_MemoryLayout_Tiled {
        ffi::PJRT_Buffer_MemoryLayout_Tiled::new(
            self.minor_to_major.as_ptr(),
            self.minor_to_major.len(),
            self.tile_dimensions.as_ptr() as *const _,
            self.tile_dimension_sizes.as_ptr(),
            self.tile_dimension_sizes.len(),
        )
    }

    /// Creates a new [`TiledLayout`] with the provided minor-to-major dimension ordering and [`Tile`]s.
    /// Refer to [`Self::minor_to_major`] and [`Self::tiles`] for information on the function arguments.
    pub fn new(minor_to_major: Vec<i64>, tiles: Vec<Tile>) -> Self {
        Self {
            minor_to_major,
            tile_dimensions: tiles.iter().flat_map(|tile| tile.dimensions.iter().copied()).collect(),
            tile_dimension_sizes: tiles.iter().map(|tile| tile.dimensions.len()).collect(),
            tile_count: tiles.len(),
        }
    }

    /// Returns the sequence of dimension indices ordered from the most minor (i.e., the one with the fastest
    /// varying index) to the most major (i.e., the one with the slowest varying index). This is effectively a map from
    /// _physical_ dimension indices to _logical_ dimension indices. The first element of this vector is the most minor
    /// physical dimension (i.e., fastest varying index) and the last the most major (i.e., slowest varying index).
    /// The contents of the vector are the indices of the logical dimensions in the shape. Note that this vector
    /// must have the same length as the number of dimensions (i.e., the rank) of the corresponding [`Buffer`].
    pub fn minor_to_major(&self) -> &[i64] {
        self.minor_to_major.as_slice()
    }

    /// Returns the sequence of [`Tile`]s that are used in this [`TiledLayout`]. The tiles are nested with the
    /// outermost tiling being the first tiling in the sequence.
    pub fn tiles(&self) -> Vec<Tile> {
        self.tile_dimension_sizes
            .iter()
            .scan(0, |offset, &dimensions_count| {
                let start_index = *offset;
                let end_index = start_index + dimensions_count;
                *offset = end_index;
                Some(Tile { dimensions: (start_index..end_index).map(|index| self.tile_dimensions[index]).collect() })
            })
            .collect()
    }

    /// Returns the `index`-th [`Tile`] of this [`TiledLayout`] or [`None`] if the provided `index` is out-of-bounds.
    pub fn tile(&self, index: usize) -> Option<Tile> {
        if index >= self.tile_count {
            None
        } else {
            let start_index = self.tile_dimension_sizes.iter().take(index).sum();
            let end_index = start_index + self.tile_dimension_sizes[index];
            Some(Tile { dimensions: self.tile_dimensions[start_index..end_index].to_vec() })
        }
    }
}

// Our [`Display`] implementation attempts to match the corresponding XLA rendering.
impl Display for TiledLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("{")?;
        let mut dimensions = self.minor_to_major.iter();
        if let Some(first_dimension) = dimensions.next() {
            write!(formatter, "{first_dimension}")?;
            dimensions.try_for_each(|dimension| write!(formatter, ",{dimension}"))?;
        }
        let tiles = self.tiles();
        if !tiles.is_empty() {
            formatter.write_str(":T")?;
            tiles.iter().try_for_each(|tile| write!(formatter, "{tile}"))?;
        }
        formatter.write_str("}")
    }
}

impl Debug for TiledLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "TiledLayout[{self}]")
    }
}

/// Strided [`Layout`]. The storage offset of the element at index `(i, j, k)` for a 3-dimensional [`Buffer`],
/// for example, is computed as follows when using this layout: `i * strides[0] + j * strides[1] + k * strides[k]`.
/// This offset is relative to the storage/memory location pointed to by the underlying [`Buffer`] data.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StridedLayout {
    /// Sequence of dimension strides (i.e., number of bytes to traverse per dimension). Refer to the documentation
    /// of [`Self::strides`] for more information.
    strides: Vec<i64>,
}

impl StridedLayout {
    /// Constructs a new [`StridedLayout`] from the provided
    /// [`PJRT_Buffer_MemoryLayout_Strides`](ffi::PJRT_Buffer_MemoryLayout_Strides)
    /// handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(layout: ffi::PJRT_Buffer_MemoryLayout_Strides) -> Self {
        Self { strides: unsafe { slice_from_c_api(layout.byte_strides, layout.num_byte_strides) }.to_vec() }
    }

    /// Returns the [`PJRT_Buffer_MemoryLayout_Strides`](ffi::PJRT_Buffer_MemoryLayout_Strides)
    /// that corresponds to this [`StridedLayout`] and which can be passed to functions in the PJRT C API.
    ///
    /// Note that the resulting [`PJRT_Buffer_MemoryLayout_Strides`](ffi::PJRT_Buffer_MemoryLayout_Strides) may become
    /// invalid if this [`StridedLayout`] is dropped. Therefore, you must make sure to keep this [`StridedLayout`] alive
    /// for as long as you intend to use the returned C API data structure. This is among the reasons for which this
    /// function is marked as _unsafe_.
    pub(crate) unsafe fn to_c_api(&self) -> ffi::PJRT_Buffer_MemoryLayout_Strides {
        ffi::PJRT_Buffer_MemoryLayout_Strides::new(self.strides.as_ptr(), self.strides.len())
    }

    /// Creates a new [`StridedLayout`] with the provided strides.
    /// Refer to [`Self::strides`] for information on how strides are defined.
    pub fn new(strides: Vec<i64>) -> Self {
        Self { strides }
    }

    /// Returns the sequence of dimension strides (i.e., number of bytes to traverse per dimension). This [`Vec`] must
    /// have the same length as the number of dimensions of the corresponding [`Buffer`]. Note that strides are allowed
    /// to be negative, in which case the data may need to point to the interior of the buffer and not necessarily
    /// its start (i.e., there must be an appropriate corresponding offset).
    pub fn strides(&self) -> &[i64] {
        self.strides.as_slice()
    }
}

// Our [`Display`] implementation attempts to match the corresponding XLA rendering.
impl Display for StridedLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("strides(")?;
        let mut strides = self.strides.iter();
        if let Some(first_stride) = strides.next() {
            write!(formatter, "{first_stride}")?;
            strides.try_for_each(|stride| write!(formatter, ",{stride}"))?;
        }
        formatter.write_str(")")
    }
}

impl Debug for StridedLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "StridedLayout[{self}]")
    }
}

/// Memory/storage layout of a [`Buffer`] that determines the mapping from logical indices into the [`Buffer`]
/// to physical offsets in the underlying memory/storage.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layout {
    /// Tiling-based [`Layout`]. Refer to [`TiledLayout`] for more information.
    Tiled(TiledLayout),

    /// Strided [`Layout`]. Refer to [`StridedLayout`] for more information.
    Strided(StridedLayout),
}

impl Layout {
    /// Constructs a new [`Layout`] from the provided [`PJRT_Buffer_MemoryLayout`](ffi::PJRT_Buffer_MemoryLayout)
    /// handle that came from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(layout: *const ffi::PJRT_Buffer_MemoryLayout) -> Result<Self, Error> {
        if layout.is_null() {
            Err(Error::invalid_argument("the provided PJRT buffer memory layout handle is a null pointer"))
        } else {
            unsafe {
                match (*layout).memory_layout_type {
                    ffi::PJRT_Buffer_MemoryLayout_Type_Tiled => {
                        Ok(Layout::Tiled(TiledLayout::from_c_api((*layout).memory_layout.tiled)))
                    }
                    ffi::PJRT_Buffer_MemoryLayout_Type_Strides => {
                        Ok(Layout::Strided(StridedLayout::from_c_api((*layout).memory_layout.strides)))
                    }
                    _ => Err(Error::invalid_argument("unknown PJRT buffer memory layout type")),
                }
            }
        }
    }

    /// Returns the [`PJRT_Buffer_MemoryLayout`](ffi::PJRT_Buffer_MemoryLayout) that corresponds to this [`Layout`]
    /// and which can be passed to functions in the PJRT C API.
    ///
    /// Note that the resulting [`PJRT_Buffer_MemoryLayout`](ffi::PJRT_Buffer_MemoryLayout) may become invalid if this
    /// [`Layout`] is dropped. Therefore, you must make sure to keep this [`Layout`] alive for as long as you intend to
    /// use the returned C API data structure. This is among the reasons for which this function is marked as _unsafe_.
    pub(crate) unsafe fn to_c_api(&self) -> ffi::PJRT_Buffer_MemoryLayout {
        match self {
            Self::Tiled(layout) => ffi::PJRT_Buffer_MemoryLayout::new(
                ffi::PJRT_Buffer_MemoryLayout_Value { tiled: unsafe { layout.to_c_api() } },
                ffi::PJRT_Buffer_MemoryLayout_Type_Tiled,
            ),
            Self::Strided(layout) => ffi::PJRT_Buffer_MemoryLayout::new(
                ffi::PJRT_Buffer_MemoryLayout_Value { strides: unsafe { layout.to_c_api() } },
                ffi::PJRT_Buffer_MemoryLayout_Type_Strides,
            ),
        }
    }

    /// Parses a rendered [`Layout`] (e.g., an XLA layout string) into a [`Layout`].
    #[allow(clippy::should_implement_trait)]
    pub fn from_str<S: AsRef<str>>(value: S) -> Result<Self, Error> {
        /// Parses and consumes one balanced parenthesized group from `characters`. The provided [`Peekable`] must
        /// be positioned at `'('`. The returned string contains only the inner content, with the outer parentheses
        /// removed. Nested parentheses are supported and preserved in the returned content. Returns an
        /// [`Error::InvalidArgument`] if the next character is not `'('` or if the input ends before a
        /// matching closing `')'` is found.
        fn parse_parenthesized(characters: &mut Peekable<Chars>, context: &str) -> Result<String, Error> {
            if characters.next() != Some('(') {
                return Err(Error::invalid_argument(format!("expected '(' while parsing {context}")));
            }

            let mut content = String::new();
            let mut depth = 1;
            while depth > 0 {
                match characters.next() {
                    Some('(') => {
                        depth += 1;
                        content.push('(');
                    }
                    Some(')') => {
                        depth -= 1;
                        if depth > 0 {
                            content.push(')');
                        }
                    }
                    Some(c) => content.push(c),
                    None => {
                        return Err(Error::invalid_argument(format!(
                            "unexpected end of string while parsing {context}"
                        )));
                    }
                }
            }

            Ok(content)
        }

        let value = value.as_ref().trim();
        if let Some(strides) = value.strip_prefix("strides(").and_then(|layout| layout.strip_suffix(')')) {
            let strides = if strides.trim().is_empty() {
                Vec::new()
            } else {
                strides
                    .split(',')
                    .map(|stride| {
                        stride
                            .trim()
                            .parse::<i64>()
                            .map_err(|error| Error::invalid_argument(format!("invalid stride '{stride}': {error}")))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            Ok(Layout::Strided(StridedLayout::new(strides)))
        } else {
            let rendered_layout = value
                .strip_prefix('{')
                .and_then(|value| value.strip_suffix('}'))
                .ok_or_else(|| Error::invalid_argument(format!("layout string must be enclosed in braces: {value}")))?;
            let (minor_to_major, properties) = rendered_layout.split_once(':').unwrap_or((rendered_layout, ""));

            let minor_to_major = if minor_to_major.trim().is_empty() {
                Vec::new()
            } else {
                minor_to_major
                    .split(',')
                    .map(|dimension| {
                        dimension.trim().parse::<i64>().map_err(|error| {
                            Error::invalid_argument(format!("invalid dimension index '{dimension}': {error}"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };

            let mut tiles = Vec::new();
            let mut characters = properties.chars().peekable();
            while let Some(property) = characters.next() {
                match property {
                    'T' => {
                        while characters.peek() == Some(&'(') {
                            let dimensions = parse_parenthesized(&mut characters, "tile")?;
                            let dimensions = if dimensions.trim().is_empty() {
                                Vec::new()
                            } else {
                                dimensions
                                    .split(',')
                                    .map(|dimension| {
                                        let dimension = dimension.trim();
                                        if dimension == "*" {
                                            Ok(TileDimension::combined())
                                        } else {
                                            let dimension = dimension.parse::<i64>().map_err(|error| {
                                                Error::invalid_argument(format!(
                                                    "invalid tile dimension '{dimension}': {error}",
                                                ))
                                            })?;
                                            if dimension < 0 {
                                                Err(Error::invalid_argument(format!(
                                                    "invalid tile dimension '{dimension}': \
                                                      expected non-negative value or '*'",
                                                )))
                                            } else {
                                                Ok(TileDimension::sized(dimension as usize))
                                            }
                                        }
                                    })
                                    .collect::<Result<Vec<_>, _>>()?
                            };
                            tiles.push(Tile { dimensions });
                        }
                    }
                    'L' | 'E' | 'M' => {
                        // XLA layout properties are ignored since they are not supported by [`Layout`].
                        parse_parenthesized(&mut characters, "layout property")?;
                    }
                    'S' => {
                        if characters.peek() == Some(&'C') {
                            characters.next();
                            let mut has_split_group = false;
                            while characters.peek() == Some(&'(') {
                                has_split_group = true;
                                parse_parenthesized(&mut characters, "split config")?;
                            }
                            if !has_split_group {
                                return Err(Error::invalid_argument("expected '(' while parsing split config"));
                            }
                        } else {
                            parse_parenthesized(&mut characters, "layout property")?;
                        }
                    }
                    'D' | '#' | '*' | 'P' => {
                        return Err(Error::invalid_argument("sparse layouts are not supported"));
                    }
                    property if property.is_whitespace() => {
                        // We simply skip whitespace characters to match XLA's permissive parsing of layouts.
                    }
                    _ => {
                        return Err(Error::invalid_argument(format!("unsupported layout property '{property}'")));
                    }
                }
            }

            Ok(Layout::Tiled(TiledLayout::new(minor_to_major, tiles)))
        }
    }

    /// Constructs a [`Layout`] from the provided [`Layout`](crate::protos::Layout) Protobuf.
    pub fn from_proto(layout: crate::protos::Layout) -> Result<Self, Error> {
        Ok(Layout::Tiled(TiledLayout::new(
            layout.minor_to_major,
            layout
                .tiles
                .into_iter()
                .map(|tile| {
                    let dimensions = tile
                        .dimensions
                        .into_iter()
                        .map(|dimension| match dimension {
                            i64::MIN => Ok(TileDimension::combined()),
                            value if value >= 0 => Ok(TileDimension::sized(value as usize)),
                            value => Err(Error::invalid_argument(format!(
                                "invalid tile dimension '{value}': expected non-negative value or '*'",
                            ))),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Tile { dimensions })
                })
                .collect::<Result<Vec<_>, _>>()?,
        )))
    }

    /// Returns the [`Layout`](crate::protos::Layout) Protobuf that corresponds to this [`Layout`].
    pub fn proto(&self) -> Result<crate::protos::Layout, Error> {
        match self {
            Layout::Tiled(layout) => Ok(crate::protos::Layout {
                minor_to_major: layout.minor_to_major().to_vec(),
                tiles: layout
                    .tiles()
                    .into_iter()
                    .map(|tile| crate::protos::Tile {
                        dimensions: tile
                            .dimensions
                            .iter()
                            .map(|dimension| {
                                if dimension.is_combined() {
                                    i64::MIN
                                } else {
                                    dimension.size().expect("tile dimension should have a fixed size") as i64
                                }
                            })
                            .collect(),
                    })
                    .collect(),
                ..Default::default()
            }),
            Layout::Strided(_) => {
                Err(Error::invalid_argument("strided layouts cannot be represented in XLA layout Protobuf messages"))
            }
        }
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

impl Debug for Layout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Layout[{self}]")
    }
}

/// Buffer that resides on a specific [`Memory`] and can transitively figure out which [`Device`]s are able to access
/// it. Buffers act as _lazy_ or _asynchronous_ references to multidimensional arrays, often also referred to as
/// tensors, meaning that while [`Buffer`]s are returned from operations immediately, the underlying data may not be
/// ready immediately.
///
/// [`Buffer`]s also represent the primary means of transferring data between the host (i.e., the CPU on which the
/// current Rust program is executing) and PJRT [`Device`]s.
///
/// # Asynchronous Nature
///
/// By describing [`Buffer`]s as _lazy_ or _asynchronous_ we mean that they may not be already allocated and
/// populated by the time they are created or used in computations; they can represent buffers that are the result
/// of a computation and/or a data transfer that has not been completed yet. [`Buffer::ready`] can be used to get an
/// [`Event`] that represents their current state (and which implements [`Future`] and be awaited on to block until the
/// buffer is ready). Furthermore, even if a buffer is not ready yet, it can be passed as input to a program execution,
/// enqueuing computations that depend on it. Once the value of a buffer is needed, execution will block until the
/// buffer declares itself ready via the [`Event`] returned by the [`Buffer::ready`] function.
///
/// # Shape Semantics
///
/// Given that [`Buffer`]s act as references to multidimensional arrays, they also have a _shape_ associated with them,
/// or rather a [_bounded dynamic shape_](https://openxla.org/stablehlo/dynamism). Due to these bounded dynamism
/// semantics, users must distinguish between:
///
///   - **On-Device Shape:** Each [`Buffer`] occupies a fixed, statically allocated region of device memory determined
///     by its on-device _bounded_ shape. This corresponds to the full, padded array shape and memory layout, including
///     any padding values that are required for alignment or to reach the static upper bound for each dimension.
///     Refer to [`Buffer::dimensions`] and [`Buffer::on_device_size_in_bytes`] for more information.
///
///   - **Logical (Unpadded) Shape:** The _valid_ data within a [`Buffer`] may be smaller than its corresponding
///     physical allocation. For example, a buffer allocated for `[128]` elements may contain only `[15]` _valid_
///     elements at runtime that correspond to a 15-element vector. We refer to the _valid_ data shape as the _logical_
///     shape of an array. There may be instances where you need to check this shape at runtime if the producing program
///     involves dynamic operations (e.g., boolean masking, variable sequence lengths, etc.). Refer to
///     [`Buffer::unpadded_dimensions`] for more information.
///
/// # Usage
///
/// [`Buffer`]s can be created in multiple ways:
///
///   - **[`Client::buffer`]:** Copies data from a host buffer.
///   - **[`Client::borrowed_buffer`] and [`Client::borrowed_mut_buffer`]:** Borrows data from a host buffer.
///   - **[`Client::uninitialized_buffer`]:** Allocates memory on a [`Memory`] without initializing it.
///   - **[`Client::borrowed_on_device_buffer`]:** Borrows data from an on-device buffer.
///   - **[`Client::error_buffer`]:** Creates a _poisoned_ [`Buffer`] that signals an error.
///   - **[`Client::alias_buffer`], [`Client::fulfill_alias_buffer`], and [`Client::fulfill_alias_buffer_with_error`]:**
///     Creates an uninitialized [`Buffer`] handle that aliases the result of some computation that has not been
///     completed yet (using [`Client::alias_buffer`]), and then _fulfills_ it with some data later on (using
///     [`Client::fulfill_alias_buffer`]) or _poisons_ it with an error (using
///     [`Client::fulfill_alias_buffer_with_error`]).
///   - **[`Client::host_to_device_transfer_manager`]:** Creates a
///     [`HostToDeviceTransferManager`](crate::HostToDeviceTransferManager) that can be used to transfer
///     data from the host to a PJRT [`Device`] with support for **pipelining**. Refer to the documentation
///     of [`HostToDeviceTransferManager`](crate::HostToDeviceTransferManager) for more information.
///   - **[`LoadedExecutable::execute`](crate::LoadedExecutable::executable):** Executes a program and returns the
///     results of that execution as [`Buffer`]s.
///
/// Similarly, data can be transferred from [`Buffer`]s to the host using multiple ways: [`Buffer::copy_to_host`],
/// [`Buffer::copy_to_host_buffer`], [`Buffer::copy_raw_to_host`], [`Buffer::copy_raw_to_host_buffer`], and
/// [`Buffer::copy_raw_to_host_buffer_future`].
///
/// Finally, [`Buffer::copy_to_memory`] and [`Buffer::copy_to_device`] can be used to transfer data between
/// [`Memory`]s/[`Device`]s that are attached to the same host machine.
///
/// # Cross-Host Transfers
///
/// Cross-host transfers are also supported by some [`Plugin`](crate::Plugin)s for when operating
/// in a distributed environment. For such transfers you can use [`Client::cross_host_send_buffers`] and
/// [`Client::cross_host_receive_buffers`] to send and receive [`Buffer`]s to and from other hosts, respectively.
///
/// The lifetime parameter `'o` captures the lifetime of the owner of this [`Buffer`] (e.g., a [`Client`]),
/// ensuring that the owner outlives the buffer.
pub struct Buffer<'o> {
    /// Handle that represents this [`Buffer`] in the PJRT C API.
    handle: *mut ffi::PJRT_Buffer,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// Handle of the [`Client`] that owns this [`Buffer`]. Note that it is safe to hold a raw pointer here because
    /// the corresponding [`Client`] is guaranteed to outlive this [`Buffer`] by design. The reason we do not hold
    /// a reference to the [`Client`] itself is to avoid having to carry around an additional lifetime for the
    /// [`KeyValueStore`](crate::KeyValueStore) that is associated with that [`Client`].
    client: *mut crate::clients::ffi::PJRT_Client,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`Buffer`].
    owner: PhantomData<&'o ()>,
}

impl Buffer<'_> {
    /// Constructs a new [`Buffer`] from the provided [`PJRT_Buffer`](ffi::PJRT_Buffer) handle that came
    /// from a function in the PJRT C API.
    pub(crate) unsafe fn from_c_api(
        handle: *mut ffi::PJRT_Buffer,
        api: Api,
        client: *mut crate::clients::ffi::PJRT_Client,
    ) -> Result<Self, Error> {
        if handle.is_null() {
            Err(Error::invalid_argument("the provided PJRT buffer handle is a null pointer"))
        } else if client.is_null() {
            Err(Error::invalid_argument("the provided PJRT client handle is a null pointer"))
        } else {
            Ok(Self { handle, api, client, owner: PhantomData })
        }
    }

    /// Returns the [`PJRT_Buffer`](ffi::PJRT_Buffer) that corresponds to this [`Buffer`] and which can
    /// be passed to functions in the PJRT C API.
    pub(crate) unsafe fn to_c_api(&self) -> *mut ffi::PJRT_Buffer {
        self.handle
    }

    /// Returns the underlying PJRT [`Api`].
    pub(crate) fn api(&self) -> Api {
        self.api
    }

    /// Returns the [`BufferType`] of the elements stored in this [`Buffer`].
    pub fn element_type(&self) -> Result<BufferType, Error> {
        use ffi::PJRT_Buffer_ElementType_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_ElementType, { buffer = self.to_c_api() }, { element_type })
            .map(|r#type| unsafe { BufferType::from_c_api(r#type) })
    }

    /// Returns the rank (i.e., the number of dimensions) of this [`Buffer`].
    pub fn rank(&self) -> Result<usize, Error> {
        Ok(self.dimensions()?.len())
    }

    /// Returns the dimension sizes of this [`Buffer`].
    pub fn dimensions(&self) -> Result<&[u64], Error> {
        use ffi::PJRT_Buffer_Dimensions_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_Dimensions, { buffer = self.to_c_api() }, { dims, num_dims })
            .map(|(dimensions, dimensions_count)| unsafe {
                slice_from_c_api(dimensions as *const u64, dimensions_count)
            })
    }

    /// Returns the unpadded dimension sizes of this [`Buffer`]. These dimension sizes are usually the same as
    /// [`Buffer::dimensions`], but for implementations that support dynamically-sized dimensions via padding to
    /// a fixed size, any dynamic dimensions may have a smaller unpadded size than the padded size reported by
    /// [`Buffer::dimensions`]. Note that, _dynamic_ dimensions are those whose length is only known at runtime,
    /// as opposed to _static_ dimensions whose size is fixed and known at compile time).
    pub fn unpadded_dimensions(&self) -> Result<&[u64], Error> {
        use ffi::PJRT_Buffer_UnpaddedDimensions_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Buffer_UnpaddedDimensions,
            { buffer = self.to_c_api() },
            { unpadded_dims, num_dims },
        )
        .map(|(dimensions, dimensions_count)| unsafe { slice_from_c_api(dimensions as *const u64, dimensions_count) })
    }

    /// Returns the indices of the dynamically-sized dimensions of this [`Buffer`], or an empty slice if all dimensions
    /// are static. _Dynamic_ dimensions are those whose length is only known at runtime, as opposed to _static_
    /// dimensions whose size is fixed and known at compile time.
    pub fn dynamic_dimensions(&self) -> Result<&[u64], Error> {
        use ffi::PJRT_Buffer_DynamicDimensionIndices_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Buffer_DynamicDimensionIndices,
            { buffer = self.to_c_api() },
            { dynamic_dim_indices, num_dynamic_dims },
        )
        .map(|(dimensions, dimensions_count)| unsafe { slice_from_c_api(dimensions as *const u64, dimensions_count) })
    }

    /// Returns the storage [`Device`] of this [`Buffer`] (i.e., the device on which it is placed).
    pub fn device(&'_ self) -> Result<Device<'_>, Error> {
        use ffi::PJRT_Buffer_Device_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_Device, { buffer = self.to_c_api() }, { device })
            .and_then(|handle| unsafe { Device::from_c_api(handle, self.api()) })
    }

    /// Returns `true` if this [`Buffer`] is placed on a CPU, thus allowing for certain optimizations.
    pub fn is_on_cpu(&self) -> Result<bool, Error> {
        use ffi::PJRT_Buffer_IsOnCpu_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_IsOnCpu, { buffer = self.to_c_api() }, { is_on_cpu })
    }

    /// Returns the storage [`Memory`] of this [`Buffer`] (i.e., the memory on which it is allocated).
    pub fn memory(&'_ self) -> Result<Memory<'_>, Error> {
        use ffi::PJRT_Buffer_Memory_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_Memory, { buffer = self.to_c_api() }, { memory })
            .and_then(|handle| unsafe { Memory::from_c_api(handle, self.api()) })
    }

    /// Returns the storage [`Layout`] of this [`Buffer`].
    #[deprecated(note = "use [`Client::layouts_extension`] instead")]
    pub fn layout(&self) -> Result<Layout, Error> {
        // We cannot use our `invoke_pjrt_api_error_fn!` macro here because we need to construct some uninitialized
        // memory that will be initialized by a C API function call. We use [`MaybeUninit`] for this purpose.
        unsafe {
            let api_handle = self.api().to_c_api();
            let api_fn_offset = std::mem::offset_of!(crate::ffi::PJRT_Api, PJRT_Buffer_GetMemoryLayout);
            let api_struct_size = (*api_handle).struct_size;
            if api_struct_size <= api_fn_offset {
                return Err(Error::unimplemented(
                    "`PJRT_Buffer_GetMemoryLayout` is not available in the loaded PJRT plugin".to_string(),
                ));
            }
            let api_fn = (*api_handle).PJRT_Buffer_GetMemoryLayout.ok_or_else(|| {
                Error::unimplemented("`PJRT_Buffer_GetMemoryLayout` is not implemented in the loaded PJRT plugin")
            })?;
            let mut uninit_args = MaybeUninit::<ffi::PJRT_Buffer_GetMemoryLayout_Args>::uninit();
            let args = uninit_args.as_mut_ptr();
            std::ptr::addr_of_mut!((*args).struct_size).write(size_of::<ffi::PJRT_Buffer_GetMemoryLayout_Args>());
            std::ptr::addr_of_mut!((*args).extension_start).write(std::ptr::null_mut());
            std::ptr::addr_of_mut!((*args).buffer).write(self.to_c_api());
            let error = api_fn(args as *mut _);
            if error.is_null() {
                let args = uninit_args.assume_init();
                Layout::from_c_api(&args.layout)
            } else {
                match Error::from_c_api(error, self.api()) {
                    Ok(None) => {
                        let args = uninit_args.assume_init();
                        Layout::from_c_api(&args.layout)
                    }
                    Ok(Some(error)) => Err(error),
                    Err(error) => Err(error),
                }
            }
        }
    }

    /// Returns the [`BufferSpecification`] of this [`Buffer`].
    pub fn specification(&self) -> Result<BufferSpecification<&[u64]>, Error> {
        Ok(BufferSpecification {
            element_type: self.element_type()?,
            dimensions: self.dimensions()?,
            #[allow(deprecated)]
            layout: Some(self.layout()?),
        })
    }

    /// Returns the number of bytes that this [`Buffer`] occupies in the underlying [`Memory`].
    pub fn on_device_size_in_bytes(&self) -> Result<usize, Error> {
        use ffi::PJRT_Buffer_OnDeviceSizeInBytes_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_OnDeviceSizeInBytes, { buffer = self.to_c_api() }, {
            on_device_size_in_bytes
        })
    }

    /// Copies the underlying data of this [`Buffer`] into a [`Vec`] that is allocated on host memory, with the
    /// provided optional [`Layout`]. This is similar to [`Self::copy_to_host_buffer`], except that it allocates
    /// a buffer for the result instead of taking in a reference to a pre-allocated buffer. Refer to the documentation
    /// of that function for more information.
    pub fn copy_to_host(&self, layout: Option<Layout>) -> Result<Event<Vec<u8>>, Error> {
        use ffi::PJRT_Buffer_ToHostBuffer_Args;
        let mut layout_handle = layout.as_ref().map(|layout| unsafe { layout.to_c_api() });
        let layout_handle = layout_handle.as_mut().map(|layout| layout as *mut _).unwrap_or(std::ptr::null_mut());

        // Invoke `PJRT_Buffer_ToHostBuffer` with `dst` set to a null pointer to get the required `dst_size`.
        let size = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Buffer_ToHostBuffer,
            {
                src = self.to_c_api(),
                host_layout = layout_handle,
                dst = std::ptr::null_mut(),
                dst_size = 0,
            },
            { dst_size },
        )?;

        // Allocate a buffer with the appropriate size and invoke `PJRT_Buffer_ToHostBuffer` again, passing that buffer.
        let mut buffer = Vec::new();
        buffer.reserve_exact(size);
        unsafe { buffer.set_len(size) };
        let buffer_slice = &mut buffer.as_mut_slice();
        let event = self.copy_to_host_buffer(layout, buffer_slice)?;
        let event_handle = unsafe { event.to_c_api() };
        std::mem::forget(event);

        // Return an `Event` with `buffer` as its output.
        unsafe { Event::from_c_api(event_handle, self.api(), buffer) }
    }

    /// Copies the underlying data of this [`Buffer`] into a buffer that is allocated on host memory, with the
    /// provided optional [`Layout`]. If no layout is provided, then the resulting data will have the same layout
    /// as this [`Buffer`].
    ///
    /// Note that this copy is an asynchronous (i.e., non-blocking) operation and this [`Buffer`] will be kept alive
    /// for the duration of this operation. If the buffer is dropped while the copy is still taking place, the
    /// underlying memory will not be freed by PJRT until the copy is completed.
    pub fn copy_to_host_buffer<'b, B: AsMut<[u8]>>(
        &self,
        layout: Option<Layout>,
        buffer: &'b mut B,
    ) -> Result<Event<&'b mut B>, Error> {
        use ffi::PJRT_Buffer_ToHostBuffer_Args;
        let mut layout_handle = layout.as_ref().map(|layout| unsafe { layout.to_c_api() });
        let layout_handle = layout_handle.as_mut().map(|layout| layout as *mut _).unwrap_or(std::ptr::null_mut());
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Buffer_ToHostBuffer,
            {
                src = self.to_c_api(),
                host_layout = layout_handle,
                dst = buffer.as_mut() as *mut _ as *mut _,
                dst_size = buffer.as_mut().len(),
            },
            { event },
        )
        .and_then(|handle| unsafe { Event::from_c_api(handle, self.api(), buffer) })
    }

    /// Copies a slice of the raw underlying data of this [`Buffer`] into a [`Vec`] that is allocated on host memory.
    /// This is similar to [`Self::copy_raw_to_host_buffer`], except that it allocates a buffer for the result with
    /// the provided `size` instead of taking in a reference to a pre-allocated buffer. Refer to the documentation of
    /// that function for more information.
    pub fn copy_raw_to_host(&self, offset: usize, size: usize) -> Result<Event<Vec<u8>>, Error> {
        unsafe {
            let mut buffer = Vec::new();
            buffer.reserve_exact(size);
            buffer.set_len(size);
            let buffer_slice = &mut buffer.as_mut_slice();
            let event = self.copy_raw_to_host_buffer(buffer_slice, offset)?;
            let event_handle = event.to_c_api();
            std::mem::forget(event);
            Event::from_c_api(event_handle, self.api(), buffer)
        }
    }

    /// Copies a slice of the raw underlying data of this [`Buffer`] into a buffer that is allocated on host memory.
    /// The slice is determined by the provided `offset` and the length of the provided `buffer`. Specifically, assuming
    /// the underling buffer data is represented as a contiguous byte array, the slice that will be copied consists of
    /// the `buffer.as_ref().len()` bytes in that array starting at the `offset` position.
    ///
    /// Note that this copy is an asynchronous (i.e., non-blocking) operation and this [`Buffer`] will be kept alive
    /// for the duration of this operation. If the buffer is dropped while the copy is still taking place, the
    /// underlying memory will not be freed by PJRT until the copy is completed.
    pub fn copy_raw_to_host_buffer<'b, B: AsMut<[u8]>>(
        &self,
        buffer: &'b mut B,
        offset: usize,
    ) -> Result<Event<&'b mut B>, Error> {
        use ffi::PJRT_Buffer_CopyRawToHost_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Buffer_CopyRawToHost,
            {
                buffer = self.to_c_api(),
                dst = buffer.as_mut() as *mut _ as *mut _,
                offset = offset as i64,
                transfer_size = buffer.as_mut().len() as i64,
            },
            { event },
        )
        .and_then(|handle| unsafe { Event::from_c_api(handle, self.api(), buffer) })
    }

    /// Performs the same operation as [`Buffer::copy_raw_to_host_buffer`] except that instead of requiring the host
    /// buffer as one of its arguments, it returns a closure (i.e., an [`FnOnce`]) that takes a host buffer and an
    /// optional [`Error`] as input. The host buffer is populated with the data from this [`Buffer`] if and only if
    /// the provided error is [`None`].
    ///
    /// The returned closure is a required completion signal for this operation and must be invoked *exactly* once,
    /// including when the destination host buffer is not ready or when the operation must fail. In other words, if
    /// an [`Error`] occurs while preparing the destination host buffer, you must still call the closure with
    /// `Some(error)` (which forwards the error to PJRT) instead of skipping the call. Otherwise, the associated
    /// future/event may never be resolved by PJRT, thus never rendering this [`Buffer`] ready.
    #[allow(clippy::type_complexity)]
    pub fn copy_raw_to_host_buffer_future<'b, B: AsMut<[u8]>>(
        &self,
        offset: usize,
    ) -> Result<(impl FnOnce(&'b mut B, Option<Error>), Event<()>), Error> {
        use ffi::PJRT_Buffer_CopyRawToHostFuture_Args;
        let (event_handle, callback_data, callback_fn) = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Buffer_CopyRawToHostFuture,
            {
                buffer = self.to_c_api(),
                offset = offset as i64,
                transfer_size = self.on_device_size_in_bytes()? as i64,
            },
            { event, callback_data, future_ready_callback },
        )?;

        let callback = move |buffer: &'b mut B, error: Option<Error>| {
            if let Some(callback_fn) = callback_fn {
                use ffi::PJRT_Buffer_CopyRawToHostFuture_Callback_Args;
                let (error_code, error_message) = match error {
                    Some(error) => (error.code(), error.message()),
                    None => (crate::errors::ffi::PJRT_Error_Code_OK, std::ffi::CString::new("").unwrap()),
                };
                unsafe {
                    callback_fn(&mut PJRT_Buffer_CopyRawToHostFuture_Callback_Args::new(
                        callback_data,
                        error_code,
                        error_message.as_ptr(),
                        error_message.count_bytes(),
                        buffer.as_mut() as *mut _ as *mut _,
                    ) as *mut _)
                };
            }
        };

        let event = unsafe { Event::from_c_api(event_handle, self.api(), ()) }?;

        Ok((callback, event))
    }

    /// Copies this [`Buffer`] to the provided [`Memory`] within the same [`Client`].
    /// If this buffer is already on the provided memory, then this function will return an [`Error`].
    ///
    /// Note that this is an asynchronous (i.e., non-blocking) copy operation and that the resulting [`Buffer`] may
    /// not be ready for use by downstream operations immediately (i.e., following the standard conventions for the
    /// semantics of PJRT [`Buffer`]s).
    pub fn copy_to_memory(&self, memory: Memory) -> Result<Self, Error> {
        use ffi::PJRT_Buffer_CopyToMemory_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Buffer_CopyToMemory,
            {
                buffer = self.to_c_api(),
                dst_memory = memory.to_c_api(),
            },
            { dst_buffer },
        )
        .and_then(|handle| unsafe { Self::from_c_api(handle, self.api(), self.client) })
    }

    /// Copies this [`Buffer`] to the provided [`Device`] within the same [`Client`].
    /// If this buffer is already on the provided device, then this function will return an [`Error`].
    ///
    /// Note that this is an asynchronous (i.e., non-blocking) copy operation and that the resulting [`Buffer`] may
    /// not be ready for use by downstream operations immediately (i.e., following the standard conventions for the
    /// semantics of PJRT [`Buffer`]s).
    pub fn copy_to_device(&self, device: Device) -> Result<Self, Error> {
        use ffi::PJRT_Buffer_CopyToDevice_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Buffer_CopyToDevice,
            {
                buffer = self.to_c_api(),
                dst_device = device.to_c_api(),
            },
            { dst_buffer },
        )
        .and_then(|handle| unsafe { Self::from_c_api(handle, self.api(), self.client) })
    }

    /// Returns an [`Event`] (which is also a [`Future`]) that is triggered when either the data in this [`Buffer`]
    /// becomes ready (i.e., after an asynchronous operation like a copy completes), or an error occurs. Note that
    /// if this [`Buffer`] has been deleted or donated, then the returned [`Event`] will immediately indicate an error.
    /// However, if this function is called on a [`Buffer`] before [`Buffer::delete`] is called, the returned [`Event`]
    /// will not transition to an error state after [`Buffer::delete`] is called.
    pub fn ready(&self) -> Result<Event<()>, Error> {
        use ffi::PJRT_Buffer_ReadyEvent_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_ReadyEvent, { buffer = self.to_c_api() }, { event })
            .and_then(|handle| unsafe { Event::from_c_api(handle, self.api(), ()) })
    }

    /// Increments the external reference count for this [`Buffer`]. The reference count of the buffer indicates the
    /// number of borrows of the underlying buffer data by external entities (e.g., other frameworks like PyTorch,
    /// DLPack, etc.). While this count is greater than `0`, this buffer should not be deleted or moved by the PJRT
    /// implementation (e.g., for memory compaction).
    ///
    /// This function is marked as unsafe because it can result in memory leaks if it is called without calling
    /// [`Buffer::decrease_external_reference_count`] later on on the same [`Buffer`].
    pub unsafe fn increase_external_reference_count(&self) -> Result<(), Error> {
        use ffi::PJRT_Buffer_IncreaseExternalReferenceCount_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_IncreaseExternalReferenceCount, { buffer = self.to_c_api() })
    }

    /// Decrements the external reference count for this [`Buffer`]. Note that this function will return an [`Error`]
    /// if the reference count is `0` and cannot be decremented further.
    ///
    /// This function is marked as unsafe because it can result in panics if it is called without having called
    /// [`Buffer::increase_external_reference_count`] earlier on on the same [`Buffer`].
    pub unsafe fn decrease_external_reference_count(&self) -> Result<(), Error> {
        use ffi::PJRT_Buffer_DecreaseExternalReferenceCount_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_DecreaseExternalReferenceCount, { buffer = self.to_c_api() })
    }

    /// Returns the _opaque_ device memory data pointer of this [`Buffer`], meaning that it is a handle that the
    /// specific [`Device`] backend understands (e.g., for an NVIDIA GPU, this is could be a `CUdeviceptr` and for other
    /// accelerators, it might be a handle to a mapped memory region). Generally, you are not supposed to dereference
    /// this pointer directly in Rust; instead you are typically expected to pass it to another library that knows how
    /// to use it. It is primarily meant to be used for interoperability with other frameworks (e.g., for converting
    /// a buffer to pass it to PyTorch using DLPack).
    ///
    /// This function is effectively the inverse of [`Client::borrowed_on_device_buffer`].
    ///
    /// Note that the returned pointer may become invalid at any point unless the external reference count for this
    /// [`Buffer`] is greater than `0`. Refer to [`Buffer::increase_external_reference_count`] and
    /// [`Buffer::decrease_external_reference_count`] for more information on buffer reference counts.
    pub unsafe fn as_ptr(&self) -> Result<*mut std::ffi::c_void, Error> {
        use ffi::PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Buffer_OpaqueDeviceMemoryDataPointer,
            { buffer = self.to_c_api() },
            { device_memory_ptr },
        )
    }

    /// Returns the (platform-dependent) address of this [`Buffer`] that often matches the physical address of the
    /// [`Buffer`] on the underlying [`Device`] (though it is not always guaranteed to match).
    ///
    /// This function is _unsafe_ because it bypasses the standard synchronization mechanisms of PJRT. For example,
    /// accessing the returned pointer does not guarantee that the underlying [`Device`] has finished writing to the
    /// buffer. You could read garbage data or cause a device fault if a kernel is still executing. Furthermore, the
    /// returned pointer might only be valid within a specific device context or stream, which this API does not
    /// guarantee is active.
    ///
    /// This function is meant to be used primarily for low-level debugging, profiling tools, or very specific internal
    /// hacks where the developer guarantees external synchronization (e.g., manually waiting for the event to complete
    /// before reading the pointer).
    pub unsafe fn unsafe_pointer(&self) -> Result<usize, Error> {
        use ffi::PJRT_Buffer_UnsafePointer_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_UnsafePointer, { buffer = self.to_c_api() }, {
            buffer_pointer
        })
    }

    /// _Donates_ this [`Buffer`] and returns a new [`Buffer`] that will be ready only when both this [`Buffer`] is
    /// ready and its new _dependency_ that is returned in the form of a callback function. That callback function must
    /// be called to indicate that the dependency is ready. Furthermore, an [`Error`] can be optionally provided as an
    /// argument to that function in order to indicate that something went wrong and _poison_ the resulting [`Buffer`].
    pub fn donate_with_control_dependency(self) -> Result<(Self, impl FnOnce(Option<Error>)), Error> {
        use ffi::PJRT_Buffer_DonateWithControlDependency_Args;
        let (callback_data, callback_fn, buffer_handle) = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Buffer_DonateWithControlDependency,
            { buffer = self.to_c_api() },
            { callback_data, dependency_ready_callback, out_buffer },
        )?;

        let callback = move |error: Option<Error>| {
            if let Some(callback_fn) = callback_fn {
                use ffi::PJRT_Buffer_DonateWithControlDependency_Callback_Args;
                let (error_code, error_message) = match error {
                    Some(error) => (error.code(), error.message()),
                    None => (crate::errors::ffi::PJRT_Error_Code_OK, std::ffi::CString::new("").unwrap()),
                };
                let mut args = PJRT_Buffer_DonateWithControlDependency_Callback_Args::new(
                    callback_data,
                    error_code,
                    error_message.as_ptr(),
                    error_message.count_bytes(),
                );
                unsafe { callback_fn(&mut args as *mut _) };
            }
        };

        let buffer = unsafe { Self::from_c_api(buffer_handle, self.api(), self.client)? };

        Ok((buffer, callback))
    }

    /// Returns `true` if and only if this [`Buffer`] has been deleted using [`Buffer::delete`].
    pub fn is_deleted(&self) -> Result<bool, Error> {
        use ffi::PJRT_Buffer_IsDeleted_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_IsDeleted, { buffer = self.to_c_api() }, { is_deleted })
    }

    /// Drops this [`Buffer`]'s reference to its associated device memory without dropping this [`Buffer`] instance
    /// itself. After this function is called, this buffer should only be used as a placeholder. The underlying device
    /// memory will be freed when all async operations using the buffer have completed according to the allocation
    /// semantics of the underlying platform.
    ///
    /// # Safety
    ///
    /// This function is marked as unsafe because it results in eagerly deallocating the underlying memory
    /// before the [`Buffer`] instance is dropped, making it unsafe to use. Only [`Buffer::is_deleted`] is considered
    /// safe to call on this [`Buffer`] after this function has been called.
    pub unsafe fn delete(&self) -> Result<(), Error> {
        use ffi::PJRT_Buffer_Delete_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_Delete, { buffer = self.to_c_api() })
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        use ffi::PJRT_Buffer_Destroy_Args;
        invoke_pjrt_api_error_fn!(self.api(), PJRT_Buffer_Destroy, { buffer = self.to_c_api() })
            .expect("failed to destroy PJRT buffer");
    }
}

// This [`PartialEq`] implementation is expensive and should generally be avoided unless absolutely necessary.
impl PartialEq for Buffer<'_> {
    fn eq(&self, other: &Self) -> bool {
        let self_specification = self.specification();
        let other_specification = other.specification();
        self_specification.is_ok()
            && other_specification.is_ok()
            && self_specification.unwrap() == other_specification.unwrap()
            && {
                let self_data = self.copy_to_host(None).and_then(|event| event.r#await());
                let other_data = other.copy_to_host(None).and_then(|event| event.r#await());
                self_data.is_ok() && other_data.is_ok() && self_data == other_data
            }
    }
}

impl Eq for Buffer<'_> {}

/// Specification that can be used to allocate a [`Buffer`] (e.g., using [`Client::uninitialized_buffer`]).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BufferSpecification<D: AsRef<[u64]>> {
    /// [`BufferType`] of the elements stored in the buffer.
    pub element_type: BufferType,

    /// Dimensions (i.e., shape) of the buffer.
    pub dimensions: D,

    /// Optional memory [`Layout`] of the buffer. If [`None`], then it is assumed to be a dense layout
    /// with dimensions in major-to-minor order.
    pub layout: Option<Layout>,
}

impl BufferSpecification<Vec<u64>> {
    /// Parses a rendered [`BufferSpecification`] (e.g., an XLA shape string) into a [`BufferSpecification`].
    #[allow(clippy::should_implement_trait)]
    pub fn from_str<S: AsRef<str>>(value: S) -> Result<BufferSpecification<Vec<u64>>, Error> {
        let value = value.as_ref().trim();
        let opening_bracket_index = value.find('[').ok_or_else(|| {
            Error::invalid_argument(format!("buffer specification is missing dimensions list: {value}"))
        })?;
        let closing_bracket_index = value[(opening_bracket_index + 1)..]
            .find(']')
            .map(|index| index + opening_bracket_index + 1)
            .ok_or_else(|| {
                Error::invalid_argument(format!("buffer specification dimensions list is missing closing ']': {value}"))
            })?;
        let element_type = BufferType::from_str(&value[..opening_bracket_index]).map_err(|_| {
            let element_type = value[..opening_bracket_index].trim();
            Error::invalid_argument(format!("invalid buffer specification element type '{element_type}'"))
        })?;

        let dimensions = &value[(opening_bracket_index + 1)..closing_bracket_index];
        let dimensions = if dimensions.trim().is_empty() {
            Vec::new()
        } else {
            dimensions
                .split(',')
                .map(|dimension| {
                    let dimension = dimension.trim();
                    if dimension == "?" {
                        Ok(0_u64)
                    } else if let Some(dimension_upper_bound) = dimension.strip_prefix("<=") {
                        dimension_upper_bound.trim().parse::<u64>().map_err(|error| {
                            Error::invalid_argument(format!(
                                "invalid dynamic dimension upper bound '{dimension_upper_bound}': {error}",
                            ))
                        })
                    } else {
                        dimension.trim().parse::<u64>().map_err(|error| {
                            Error::invalid_argument(format!("invalid dimension size '{dimension}': {error}"))
                        })
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let layout = value[(closing_bracket_index + 1)..].trim();
        let layout = if layout.is_empty() { None } else { Some(Layout::from_str(layout)?) };

        Ok(BufferSpecification { element_type, dimensions, layout })
    }

    /// Constructs a [`BufferSpecification`] from the provided [`Shape`](crate::protos::Shape) Protobuf.
    pub fn from_proto(shape: crate::protos::Shape) -> Result<Self, Error> {
        if !shape.tuple_shapes.is_empty() {
            return Err(Error::invalid_argument("tuple shapes cannot be represented as a buffer specification"));
        }

        Ok(Self {
            element_type: crate::protos::BufferType::try_from(shape.element_type)
                .map_err(|_| Error::invalid_argument(format!("invalid shape element type: {}", shape.element_type)))
                .map(BufferType::from_proto)?,
            dimensions: shape
                .dimensions
                .iter()
                .map(|dimension| {
                    if *dimension == i64::MIN {
                        Ok(0_u64)
                    } else if *dimension < 0 {
                        Err(Error::invalid_argument(format!(
                            "invalid shape dimension '{dimension}': expected non-negative value",
                        )))
                    } else {
                        Ok(*dimension as u64)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
            layout: shape.layout.map(|layout| Layout::from_proto(*layout)).transpose()?,
        })
    }
}

impl<D: AsRef<[u64]>> BufferSpecification<D> {
    /// Returns the [`Shape`](crate::protos::Shape) Protobuf that corresponds to this [`BufferSpecification`].
    pub fn proto(&self) -> Result<crate::protos::Shape, Error> {
        Ok(crate::protos::Shape {
            element_type: self.element_type.proto() as i32,
            dimensions: self.dimensions.as_ref().iter().map(|dimension| *dimension as i64).collect::<Vec<_>>(),
            is_dynamic_dimension: vec![false; self.dimensions.as_ref().len()],
            tuple_shapes: Vec::new(),
            layout: self.layout.as_ref().map(Layout::proto).transpose()?.map(Box::new),
        })
    }
}

// Our [`Display`] implementation attempts to match the corresponding XLA rendering.
impl<D: AsRef<[u64]>> Display for BufferSpecification<D> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}[", self.element_type)?;
        let mut dimensions = self.dimensions.as_ref().iter();
        if let Some(first_dimension) = dimensions.next() {
            write!(formatter, "{first_dimension}")?;
            dimensions.try_for_each(|dimension| write!(formatter, ",{dimension}"))?;
        }
        write!(formatter, "]")?;
        if let Some(layout) = &self.layout {
            write!(formatter, "{layout}")?;
        }
        Ok(())
    }
}

impl<D: AsRef<[u64]>> Debug for BufferSpecification<D> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "BufferSpecification[{self}]")
    }
}

/// Represents the assumptions that PJRT can make about a host buffer that is provided to it via the [`Client::buffer`]
/// function (i.e., how it is allowed to treat the memory that is handed to it). Specifically, it dictates whether PJRT
/// needs to make an immediate copy of the provided data or whether it can borrow it without copying.
pub enum HostBufferSemantics {
    /// PJRT may not hold references to the provided data after the call to [`Client::buffer`] returns. The caller
    /// promises to not mutate or free the provided data **only** for the duration of the [`Client::buffer`] invocation.
    ImmutableOnlyDuringCall,

    /// PJRT may hold reference to the provided data after the call to [`Client::buffer`] returns while it completes
    /// a transfer of the data to the target [`Device`]. The caller promises to not mutate or free the provided data
    /// until the transfer completes.
    ImmutableUntilTransferCompletes,

    /// The [`Buffer`] returned by [`Client::buffer`] may alias the provided data internally, and PJRT may use that data
    /// as long as the [`Buffer`] is alive. PJRT promises to not mutate the contents of the buffer (i.e., it will not
    /// use it for aliased output buffers). The caller promises to keep the data alive and to not mutate it as long as
    /// the [`Buffer`] is alive.
    ImmutableZeroCopy,

    /// The [`Buffer`] returned by [`Client::buffer`] may alias the provided data internally, and PJRT may use that data
    /// as long as the [`Buffer`] is alive. PJRT is also allowed to mutate that data (i.e., use it for aliased output
    /// buffers). The caller promises to keep the data alive and not mutate it for as long as the [`Buffer`] is alive
    /// (otherwise we could end up in a data race with PJRT). Note that on non-CPU platforms this mode is equivalent to
    /// [`HostBufferSemantics::ImmutableUntilTransferCompletes`] because the resulting [`Buffer`] will contain a copy
    /// of the data anyway since it needs to be allocated in a different [`Memory`].
    MutableZeroCopy,
}

impl HostBufferSemantics {
    /// Returns the [`PJRT_HostBufferSemantics`](ffi::PJRT_HostBufferSemantics) that corresponds to this
    /// [`HostBufferSemantics`] instance and which can be passed to functions in the PJRT C API.
    unsafe fn to_c_api(&self) -> ffi::PJRT_HostBufferSemantics {
        match self {
            Self::ImmutableOnlyDuringCall => ffi::PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
            Self::ImmutableUntilTransferCompletes => ffi::PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
            Self::ImmutableZeroCopy => ffi::PJRT_HostBufferSemantics_kImmutableZeroCopy,
            Self::MutableZeroCopy => ffi::PJRT_HostBufferSemantics_kMutableZeroCopy,
        }
    }
}

/// Contains a pointer to the data contained in a host buffer (which can be passed to [`Client::buffer`]
/// to construct a PJRT buffer by copying that data), along with an optional function to free that data once
/// [`Client::buffer`] is done using it.
pub struct HostBufferData {
    /// Pointer to the underlying host buffer data. This pointer will be passed to the PJRT C API
    /// and must remain valid for the duration specified by the buffer's [`HostBufferSemantics`].
    pub(crate) ptr: *const std::ffi::c_void,

    /// Optional callback that will be called when PJRT is done with the host buffer to drop it
    /// (or reduce its reference count if it wrapped in an [`Rc`]).
    pub(crate) drop_fn: Option<Box<dyn FnOnce()>>,
}

impl HostBufferData {
    /// Constructs a new [`HostBufferData`] instance for the provided buffer.
    pub(crate) fn from_host_buffer<B: AsRef<[u8]>>(buffer: B) -> Self {
        let buffer = buffer.as_ref();
        Self {
            ptr: unsafe { slice_from_c_api(buffer.as_ptr() as *const std::ffi::c_void, buffer.len()) }.as_ptr(),
            drop_fn: None,
        }
    }

    /// Constructs a new [`HostBufferData`] instance for the provided buffer reference. If `mutable` is `true`, then the
    /// resulting [`HostBufferData`] will hold a mutable reference to the underlying host buffer data until its
    /// `drop_fn` is invoked by PJRT. If `mutable` is `false`, then it will hold an immutable reference to the
    /// underlying host buffer data.
    pub(crate) fn from_host_buffer_rc_refcell<B: AsRef<[u8]>>(buffer: &Rc<RefCell<B>>, mutable: bool) -> Self {
        let buffer_clone_raw = Rc::into_raw(buffer.clone()) as *const RefCell<()>;
        let ptr = {
            let buffer = buffer.borrow();
            let buffer = buffer.as_ref();
            let slice = unsafe { slice_from_c_api(buffer.as_ptr() as *const std::ffi::c_void, buffer.len()) };
            slice.as_ptr()
        };

        // Construct the data that will be captured by the `drop_fn` closure that PJRT will invoke once it is done using
        // the host buffer that. The data is a [`Box`]ed [`HostBufferReference`] that holds a borrow guard for the host
        // buffer data reference (which is transmuted to a type with a `'static` lifetime so that we can [`Box`] it;
        // this is safe in this case because the backing storage is guaranteed to be kept alive for the duration of this
        // guard via `rc_clone_raw`) and a raw pointer representing the [`Rc`] that owns the host buffer data and which
        // will be used to decrease its reference count once PJRT is done using the host buffer data.
        let data = unsafe {
            Box::into_raw(Box::new(HostBufferReference {
                ptr: buffer_clone_raw,
                guard: if mutable {
                    HostBufferReferenceGuard::Mutable(std::mem::transmute::<RefMut<'_, ()>, RefMut<'_, ()>>(
                        (*buffer_clone_raw).borrow_mut(),
                    ))
                } else {
                    HostBufferReferenceGuard::Immutable(std::mem::transmute::<Ref<'_, ()>, Ref<'_, ()>>(
                        (*buffer_clone_raw).borrow(),
                    ))
                },
            })) as *const std::ffi::c_void
        };

        Self {
            ptr,
            drop_fn: Some(Box::new(move || unsafe {
                // First, `drop` the reference guard to make sure that runtime borrow checking rules are followed
                // appropriately (and that the subsequent drop of the [`Rc`] does not fail). Then, drop the [`Rc`]
                // that owns the host buffer data, therefore decreasing its reference count. Note that we need to
                // use `std::hint::black_box` here to prevent Dead Code Elimination (DCE) in the Rust compiler from
                // removing these calls.
                let data = Box::from_raw(data as *mut HostBufferReference);
                drop(std::hint::black_box(data.guard));
                drop(std::hint::black_box(Rc::from_raw(data.ptr)))
            })),
        }
    }
}

/// Internal helper for holding a reference to a host buffer that needs to be kept alive until PJRT is done with it
/// along with a [`HostBufferReferenceGuard`] for it. This is captured by a [`HostBufferData::drop_fn`] closure such
/// that it can be dropped once PJRT is done using the host buffer.
struct HostBufferReference {
    /// Raw pointer to the [`Rc`] that owns the host buffer data and that can be used to decrease its reference count
    /// once PJRT is done using the host buffer data.
    ptr: *const RefCell<()>,

    /// Reference guard for the host buffer data that makes sure that Rust borrow checking rules are followed
    /// at runtime using a [`RefCell`] for the host buffer data.
    guard: HostBufferReferenceGuard,
}

/// Internal helper guard for references to host buffer data that need to be held until PJRT is done with them.
/// This is an enum because we need to handle immutable and mutable host buffer borrows differently
/// (i.e., with different guards).
enum HostBufferReferenceGuard {
    /// Immutable reference guard. Note that the `'static` lifetime is fake but needed so that we can [`Box`] it.
    /// We never actually use this guard other than dropping it in a [`HostBufferData::drop_fn`] implementation,
    /// and so we need the `#[allow(dead_code)]` to disable a warning.
    #[allow(dead_code)]
    Immutable(Ref<'static, ()>),

    /// Mutable reference guard. Note that the `'static` lifetime is fake but needed so that we can [`Box`] it.
    /// We never actually use this guard other than dropping it in a [`HostBufferData::drop_fn`] implementation,
    /// and so we need the `#[allow(dead_code)]` to disable a warning.
    #[allow(dead_code)]
    Mutable(RefMut<'static, ()>),
}

/// Represents a host buffer that can be copied to a [`Device`] via [`Client::buffer`]
/// to construct a PJRT [`Buffer`] with the same underlying data.
pub trait HostBuffer {
    /// [`HostBufferSemantics`] that PJRT should use when handling this host buffer.
    fn host_buffer_semantics() -> HostBufferSemantics;

    /// [`HostBufferData`] that corresponds to this host buffer. The returned data structure may capture `self`,
    /// allowing cleanup callbacks to hold owned data for the duration of the data transfer.
    unsafe fn data(&self) -> HostBufferData;
}

impl HostBuffer for &[u8] {
    fn host_buffer_semantics() -> HostBufferSemantics {
        HostBufferSemantics::ImmutableOnlyDuringCall
    }

    unsafe fn data(&self) -> HostBufferData {
        HostBufferData::from_host_buffer(self)
    }
}

impl HostBuffer for Rc<RefCell<&[u8]>> {
    fn host_buffer_semantics() -> HostBufferSemantics {
        HostBufferSemantics::ImmutableUntilTransferCompletes
    }

    unsafe fn data(&self) -> HostBufferData {
        HostBufferData::from_host_buffer_rc_refcell(self, false)
    }
}

impl<const N: usize> HostBuffer for &[u8; N] {
    fn host_buffer_semantics() -> HostBufferSemantics {
        HostBufferSemantics::ImmutableOnlyDuringCall
    }

    unsafe fn data(&self) -> HostBufferData {
        HostBufferData::from_host_buffer(self)
    }
}

impl<const N: usize> HostBuffer for Rc<RefCell<&[u8; N]>> {
    fn host_buffer_semantics() -> HostBufferSemantics {
        HostBufferSemantics::ImmutableUntilTransferCompletes
    }

    unsafe fn data(&self) -> HostBufferData {
        HostBufferData::from_host_buffer_rc_refcell(self, false)
    }
}

/// Represents a region of host memory that has been _pinned_ or _page-locked_ for _Direct Memory Access (DMA)_.
/// Pinning memory prevents the operating system from swapping those memory pages to disk. Most high-speed DMA
/// controllers (like those on GPUs or TPUs) cannot perform transfers to _pageable_ memory because physical addresses
/// could change mid-transfer. This makes future transfers from the pinned memory region to PJRT [`Device`]s much
/// faster by avoiding an additional pinning step (and copy) for them.
///
/// Note that, typically, you do not need to use this function and can instead rely on [`Client::buffer`],
/// [`Client::borrowed_buffer`], [`Client::borrowed_mut_buffer`], and [`Client::host_to_device_transfer_manager`]
/// for efficient and asynchronous data transfers between the host and specific [`Device`]s.
///
/// # Safety
///
/// While this type is safe to use, it requires the caller to uphold certain invariants:
///
///   - The memory referenced by this [`DmaMappedBuffer`] must remain valid (allocated and not freed)
///     for the entire lifetime of the [`DmaMappedBuffer`] (or a [`Buffer`] resulting from calling
///     [`DmaMappedBuffer::into_buffer`]).
///   - All data transfers involving this memory must complete before this [`DmaMappedBuffer`] (or a [`Buffer`]
///     resulting from calling [`DmaMappedBuffer::into_buffer`]) is dropped.
///
/// # Platform Support
///
/// DMA mapping may not be implemented by all PJRT [`Plugin`](crate::Plugin)s. Furthermore, when implemented, backends
/// may have specific alignment requirements (e.g., most require at least page alignment).
pub struct DmaMappedBuffer<'c> {
    /// Pointer to the pinned host memory.
    ptr: *mut std::ffi::c_void,

    /// Size in bytes of the pinned memory region.
    len: usize,

    /// Underlying PJRT [`Api`].
    api: Api,

    /// Handle of the [`Client`] that owns this [`DmaMappedBuffer`]. Note that it is safe to hold a raw pointer here
    /// because the corresponding [`Client`] is guaranteed to outlive this [`DmaMappedBuffer`] by design. The reason we
    /// do not hold a reference to the [`Client`] itself is to avoid having to carry around an additional lifetime for
    /// the [`KeyValueStore`](crate::KeyValueStore) that is associated with that [`Client`].
    client: *mut crate::clients::ffi::PJRT_Client,

    /// [`PhantomData`] used to track the lifetime of the [`Client`] that owns this [`DmaMappedBuffer`].
    owner: PhantomData<&'c ()>,
}

impl<'c> DmaMappedBuffer<'c> {
    /// Converts this [`DmaMappedBuffer`] to a [`Buffer`] by transferring the underlying data to the target
    /// [`Memory`] via [`Client::buffer`]. Internally, uses [`HostBufferSemantics::ImmutableZeroCopy`] which enables
    /// true zero-copy on CPU and automatically falls back to a DMA-accelerated copy on non-CPU backends. The underlying
    /// pinned memory region will be unpinned when the returned [`Buffer`] is done using the underlying data.
    ///
    /// # Safety
    ///
    /// This function is marked as unsafe because the caller must ensure that the underlying memory region is not
    /// freed/deallocated while the resulting [`Buffer`] is still alive.
    pub unsafe fn into_buffer<D: AsRef<[u64]>, M: HasDefaultMemory>(
        self,
        specification: BufferSpecification<D>,
        memory: M,
    ) -> Result<Buffer<'c>, Error> {
        /// Internal [`HostBuffer`] wrapper for DMA-mapped memory that uses [`HostBufferSemantics::ImmutableZeroCopy`].
        /// On CPU, this enables true zero-copy and on non-CPU backends, PJRT falls back to a DMA-accelerated copy.
        struct DmaHostBuffer {
            /// Pointer to the underlying pinned memory region, passed to PJRT as the host buffer data.
            ptr: *const std::ffi::c_void,

            /// [`UnsafeCell`] that holds the original [`DmaMappedBuffer`] so that [`HostBuffer::data`] can move it into
            /// the returned [`HostBufferData::drop_fn`] closure. When PJRT is done with the host data, the closure runs
            /// (or is dropped on error), dropping the [`DmaMappedBuffer`] and triggering a DMA _unmap_ via its [`Drop`]
            /// implementation. If [`HostBuffer::data`] is never called (i.e., if [`Client::buffer`] fails before
            /// reaching it), the [`DmaMappedBuffer`] will be cleaned up when this struct is dropped.
            dma: UnsafeCell<Option<DmaMappedBuffer<'static>>>,
        }

        impl HostBuffer for DmaHostBuffer {
            fn host_buffer_semantics() -> HostBufferSemantics {
                HostBufferSemantics::ImmutableZeroCopy
            }

            unsafe fn data(&self) -> HostBufferData {
                let dma = unsafe { &mut *self.dma.get() }
                    .take()
                    .expect("`DmaHostBuffer::data()` must only be called exactly once");
                HostBufferData { ptr: self.ptr, drop_fn: Some(Box::new(move || drop(dma))) }
            }
        }

        // Create a temporary, non-owning [`Client`] wrapper to call [`Client::buffer`].
        // [`ManuallyDrop`] prevents the [`Client::drop`] from destroying the underlying PJRT client.
        let client = std::mem::ManuallyDrop::new(unsafe { Client::from_c_api(self.client, self.api, None) }?);

        // Extract the underlying data pointer before calling [`std::mem::transmute`] on `self`.
        let ptr = self.ptr as *const std::ffi::c_void;

        // Erase the phantom lifetime so that the [`DmaMappedBuffer`] can be moved into a [`Box<dyn FnOnce()>`] which
        // requires a `'static` lifetime. This is not generally safe but the requirements are outlined in the
        // documentation of [`DmaMappedBuffer::into_buffer`] which is also marked as unsafe.
        let dma: DmaMappedBuffer<'static> = unsafe { std::mem::transmute(self) };

        // [`Client::buffer`] takes a [`DmaHostBuffer`] by value. On success, that [`DmaHostBuffer`]'s drop function
        // will be invoked when PJRT fires the "done" event. On failure, it will be invoked when that [`DmaHostBuffer`]
        // itself is dropped. Note that the [`std::mem::transmute`] that follows is safe as we are guaranteed that
        // the client will outlive the buffer due to the `'c` lifetime of [`DmaMappedBuffer`].
        client
            .buffer(
                DmaHostBuffer { ptr, dma: UnsafeCell::new(Some(dma)) },
                specification.element_type,
                specification.dimensions,
                None,
                memory,
                specification.layout,
            )
            .map(|buffer| unsafe { std::mem::transmute::<_, Buffer<'c>>(buffer) })
    }

    /// Returns the underlying data as a slice of bytes.
    pub fn data(&self) -> &[u8] {
        unsafe { slice_from_c_api(self.ptr as *const u8, self.len) }
    }

    /// Returns the size in bytes of the pinned memory region.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the pinned memory region has zero length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

unsafe impl Send for DmaMappedBuffer<'_> {}
unsafe impl Sync for DmaMappedBuffer<'_> {}

impl Drop for DmaMappedBuffer<'_> {
    fn drop(&mut self) {
        use ffi::PJRT_Client_DmaUnmap_Args;
        invoke_pjrt_api_error_fn!(
            self.api,
            PJRT_Client_DmaUnmap,
            {
                client = self.client,
                data = self.ptr,
            },
        )
        .expect("failed to unmap DMA memory");
    }
}

/// Token that can be used to _fulfill_ an alias [`Buffer`]. This token is an opaque struct that contains a callback
/// function and any necessary context required for fulfilling a specific alias [`Buffer`]. This object effectively
/// represents the "write permission" for that alias [`Buffer`].
///
/// Refer to [`Client::alias_buffer`] for more information about alias [`Buffer`]s.
#[repr(transparent)]
pub struct AliasBufferFulfillmentToken {
    /// Handle that represents this [`AliasBufferFulfillmentToken`] in the PJRT C API.
    handle: *mut ffi::PJRT_FulfillAliasBufferCallback,
}

impl<'s> Client<'s> {
    /// Creates a new [`Buffer`] by asynchronously transferring data from a host buffer to a [`Device`] or [`Memory`].
    ///
    /// The behavior of this function depends on the [`HostBufferSemantics`] specified by the `data` type. Those
    /// semantics determine how long `data` needs to stay alive. This function will only ever result in using
    /// [`HostBufferSemantics::ImmutableOnlyDuringCall`] and [`HostBufferSemantics::ImmutableUntilTransferCompletes`].
    /// Refer to the documentation of [`Client::borrowed_buffer`] and [`Client::borrowed_mut_buffer`] if you want to use
    /// [`HostBufferSemantics::ImmutableZeroCopy`] or [`HostBufferSemantics::MutableZeroCopy`], respectively.
    ///
    /// Note that the resulting [`Buffer`] may not be ready when this function returns (as it performs an asynchronous
    /// data transfer under the hood). To get a [`Future`] for when the resulting [`Buffer`] becomes ready, you must the
    /// [`Buffer::ready`] function.
    ///
    /// # Parameters
    ///
    ///   - `data`: Host buffer containing the data to transfer. `element_type` and `dimensions` determine the size
    ///     that this buffer should have.
    ///   - `element_type`: [`BufferType`] of the elements in the new [`Buffer`].
    ///   - `dimensions`: Dimensions (i.e., shape) of the new [`Buffer`].
    ///   - `byte_strides`: Optional byte strides for each dimension of `buffer`. If [`None`], the array is assumed
    ///     to have a dense layout with dimensions in major-to-minor order. Note that strides can be negative, in which
    ///     case the data pointer may need to point to the interior of the buffer.
    ///   - `memory`: [`Memory`] on which to place the new [`Buffer`].
    ///   - `device_layout`: Optional memory [`Layout`] for the resulting [`Buffer`]. If [`None`], a dense layout
    ///     with dimensions in major-to-minor order is assumed.
    pub fn buffer<'c, B: HostBuffer, D: AsRef<[u64]>, M: HasDefaultMemory>(
        &'c self,
        data: B,
        element_type: BufferType,
        dimensions: D,
        byte_strides: Option<&'_ [i64]>,
        memory: M,
        device_layout: Option<Layout>,
    ) -> Result<Buffer<'c>, Error> {
        use ffi::PJRT_Client_BufferFromHostBuffer_Args;

        // Call the appropriate PJRT C API function to create the new buffer.
        let data = unsafe { data.data() };
        let dimensions = dimensions.as_ref().iter().map(|&dimension| dimension as i64).collect::<Vec<_>>();
        let (buffer_handle, done_event_handle) = invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_BufferFromHostBuffer,
            {
                client = self.to_c_api(),
                data = data.ptr,
                data_type = element_type.to_c_api(),
                dims = dimensions.as_ptr(),
                num_dims = dimensions.len(),
                byte_strides = byte_strides.map(|strides| strides.as_ptr()).unwrap_or(std::ptr::null()),
                num_byte_strides = byte_strides.map(|strides| strides.len()).unwrap_or(0),
                host_buffer_semantics = B::host_buffer_semantics().to_c_api(),
                device = std::ptr::null_mut(),
                memory = memory.default_memory().to_c_api(),
                device_layout = device_layout
                    .as_ref()
                    .map(|layout| &layout.to_c_api() as *const _ as *mut _)
                    .unwrap_or(std::ptr::null_mut()),
            },
            { buffer, done_with_host_buffer },
        )?;
        let buffer = unsafe { Buffer::from_c_api(buffer_handle, self.api(), self.to_c_api())? };
        let done_event = unsafe { Event::from_c_api(done_event_handle, self.api(), ()) };

        // Register a callback to drop the host buffer data after the copy is completed.
        if let Ok(done_event) = done_event
            && let Some(drop_fn) = data.drop_fn
        {
            // Register the callback that will be invoked once the host buffer data has been copied.
            done_event.on_ready(|_| {
                // We ignore the error because there is nothing we can do with it here,
                // and if something goes wrong, it should be reflected in [`Buffer::ready`].
                drop_fn();
            })?;
        }

        Ok(buffer)
    }

    /// Constructs an immutable [`Buffer`] whose underlying data is shared with the provided `data`. Refer to the
    /// documentation of [`Client::buffer`] for the meaning of the arguments of this function.
    ///
    /// Note that PJRT will hold an immutable reference to the underlying data until the resulting [`Buffer`] is
    /// If the provided `device` is not the host CPU, then PJRT will really only hold an immutable reference to the
    /// underlying data until that data is copied to the target device, creating an entirely new [`Buffer`] with no
    /// shared data. That is because it is not possible to represent shared data between the CPU and other devices
    /// in PJRT. In those other cases, this function behaves equivalently to [`Client::buffer`].
    pub fn borrowed_buffer<'c, B: AsRef<[u8]>, D: AsRef<[u64]>, M: HasDefaultMemory>(
        &'c self,
        data: Rc<RefCell<B>>,
        element_type: BufferType,
        dimensions: D,
        byte_strides: Option<&'_ [i64]>,
        memory: M,
        device_layout: Option<Layout>,
    ) -> Result<Buffer<'c>, Error> {
        /// Internal helper that wraps an `Rc<RefCell<B>>` and provides a custom [`HostBuffer`]
        /// implementation for it that uses different [`HostBufferSemantics`] than its default implementation.
        struct BorrowedHostBuffer<B: AsRef<[u8]>> {
            data: Rc<RefCell<B>>,
        }

        impl<B: AsRef<[u8]>> HostBuffer for BorrowedHostBuffer<B> {
            fn host_buffer_semantics() -> HostBufferSemantics {
                HostBufferSemantics::ImmutableZeroCopy
            }

            unsafe fn data(&self) -> HostBufferData {
                HostBufferData::from_host_buffer_rc_refcell(&self.data, false)
            }
        }

        self.buffer(BorrowedHostBuffer { data }, element_type, dimensions, byte_strides, memory, device_layout)
    }

    /// Constructs a mutable [`Buffer`] whose underlying data is shared with the provided `data`. Refer to the
    /// documentation of [`Client::buffer`] for the meaning of the arguments of this function.
    ///
    /// Note that PJRT will hold a mutable reference to the underlying data until the resulting [`Buffer`] is dropped.
    /// If the provided `device` is not the host CPU, then PJRT will really only hold a mutable reference to the
    /// underlying data until that data is copied to the target device, creating an entirely new [`Buffer`] with no
    /// shared data. That is because it is not possible to represent shared data between the CPU and other devices
    /// in PJRT. In those other cases, this function behaves equivalently to [`Client::buffer`].
    pub fn borrowed_mut_buffer<'c, B: AsRef<[u8]>, D: AsRef<[u64]>, M: HasDefaultMemory>(
        &'c self,
        data: Rc<RefCell<B>>,
        element_type: BufferType,
        dimensions: D,
        byte_strides: Option<&'_ [i64]>,
        memory: M,
        device_layout: Option<Layout>,
    ) -> Result<Buffer<'c>, Error> {
        /// Internal helper that wraps an `Rc<RefCell<B>>` and provides a custom [`HostBuffer`]
        /// implementation for it that uses different [`HostBufferSemantics`] than its default implementation.
        struct BorrowedMutHostBuffer<B: AsRef<[u8]>> {
            data: Rc<RefCell<B>>,
        }

        impl<B: AsRef<[u8]>> HostBuffer for BorrowedMutHostBuffer<B> {
            fn host_buffer_semantics() -> HostBufferSemantics {
                HostBufferSemantics::MutableZeroCopy
            }

            unsafe fn data(&self) -> HostBufferData {
                HostBufferData::from_host_buffer_rc_refcell(&self.data, true)
            }
        }

        self.buffer(BorrowedMutHostBuffer { data }, element_type, dimensions, byte_strides, memory, device_layout)
    }

    /// Creates a new uninitialized [`Buffer`].
    ///
    /// # Parameters
    ///
    ///   - `specification`: [`BufferSpecification`] for the new buffer.
    ///   - `memory`: [`Memory`] on which to place the new buffer.
    pub fn uninitialized_buffer<D: AsRef<[u64]>, M: HasDefaultMemory>(
        &'_ self,
        specification: BufferSpecification<D>,
        memory: M,
    ) -> Result<Buffer<'_>, Error> {
        use ffi::PJRT_Client_CreateUninitializedBuffer_Args;
        let layout = specification.layout.map(|layout| unsafe { layout.to_c_api() });
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_CreateUninitializedBuffer,
            {
                client = self.to_c_api(),
                shape_dims = specification.dimensions.as_ref().as_ptr() as *const i64,
                shape_num_dims = specification.dimensions.as_ref().len(),
                shape_element_type = specification.element_type.to_c_api(),
                shape_layout = layout.map(|layout| &layout as *const _ as *mut _).unwrap_or(std::ptr::null_mut()),
                device = std::ptr::null_mut(),
                memory = memory.default_memory().to_c_api(),
            },
            { buffer },
        )
        .and_then(|handle| unsafe { Buffer::from_c_api(handle, self.api(), self.to_c_api()) })
    }

    /// Creates a new [`Buffer`] that wraps a borrowed (i.e., non-owned) on-device buffer that is already allocated
    /// (typically by another library) on a specific [`Memory`]. The buffer may be mutated, for example, if the buffer
    /// is donated when executing a PJRT program. Note that this function may not be implemented on all hardware
    /// platforms, in which case it would return an [`Error::Unimplemented`].
    ///
    /// This function can be used for wrapping things like [DLPack](https://github.com/dmlc/dlpack) tensors
    /// and [Direct Memory Access (DMA)](https://en.wikipedia.org/wiki/Direct_memory_access)-mapped buffers
    /// to reduce the number of copies that need to be made when moving data between frameworks or between the host
    /// and a specific device, for example.
    ///
    /// This function is effectively the inverse of [`Buffer::as_ptr`].
    ///
    /// # Safety
    ///
    /// This function is marked as **unsafe** because we cannot use the compiler to enforce that it is used correctly.
    /// Specifically, the buffer pointed to by `device_buffer` needs to have the appropriate size that matches the
    /// provided `specification` and it must also not be freed until the returned [`Buffer`] and aliases of it that may
    /// be created are dropped. Furthermore, `on_drop_callback` needs to be implemented correctly to avoid memory leaks.
    ///
    /// # Parameters
    ///
    ///   - `ptr`: Pointer to the non-owned device buffer that is to be wrapped into a [`Buffer`].
    ///   - `drop_fn`: Callback that will be invoked when the returned [`Buffer`] is done using the data pointed to by
    ///     `ptr`. This can be used to free the associated memory, for example.
    ///   - `specification`: [`BufferSpecification`] for the new buffer.
    ///   - `memory`: [`Memory`] on which `ptr` is allocated.
    ///   - `stream`: Optional platform-specific stream handle that should contain the work or events needed to
    ///     materialize the on-device buffer that `ptr` points to. This function will append an event to this stream
    ///     that indicates when the returned [`Buffer`] is ready to use. This is intended to support
    ///     [DLPack](https://github.com/dmlc/dlpack) on GPUs and is not expected to be supported on
    ///     all hardware platforms.
    pub unsafe fn borrowed_on_device_buffer<D: AsRef<[u64]>, M: HasDefaultMemory, F: FnOnce()>(
        &'_ self,
        ptr: *mut std::ffi::c_void,
        drop_fn: F,
        specification: BufferSpecification<D>,
        memory: M,
        stream: Option<*mut std::ffi::c_void>,
    ) -> Result<Buffer<'_>, Error> {
        use ffi::PJRT_Client_CreateViewOfDeviceBuffer_Args;
        let layout = specification.layout.map(|layout| unsafe { layout.to_c_api() });

        extern "C" fn callback<F: FnOnce()>(_ptr: *mut std::ffi::c_void, arg: *mut std::ffi::c_void) {
            unsafe { Box::from_raw(arg as *mut F)() };
        }

        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_CreateViewOfDeviceBuffer,
            {
                client = self.to_c_api(),
                device_buffer_ptr = ptr,
                dims = specification.dimensions.as_ref().as_ptr() as *const i64,
                num_dims = specification.dimensions.as_ref().len(),
                element_type = specification.element_type.to_c_api(),
                layout = layout.map(|layout| &layout as *const _ as *mut _).unwrap_or(std::ptr::null_mut()),
                device = std::ptr::null_mut(),
                memory = memory.default_memory().to_c_api(),
                stream = stream.unwrap_or(std::ptr::null_mut()) as isize,
                on_delete_callback = callback::<F>,
                on_delete_callback_arg = Box::into_raw(Box::new(drop_fn)) as *mut std::ffi::c_void,
            },
            { buffer },
        )
        .and_then(|handle| unsafe { Buffer::from_c_api(handle, self.api(), self.to_c_api()) })
    }

    /// Creates a new _poisoned_ [`Buffer`] that represents an error state. Instead of a standard buffer containing
    /// valid data, this function creates a [`Buffer`] that carries a specific [`Error`]. This is particularly useful
    /// for asynchronous execution and error propagation across the PJRT interface.
    ///
    /// The resulting buffer:
    ///   - **Does not Contain Actual Data:** Any attempt to read from it or use it as an input for a subsequent
    ///     operation will typically result in an error.
    ///   - **Encapsulates an [`Error`]:** It acts as a container for an [`Error`].
    ///   - **Propagates Failures:** In complex data-flow graphs, if a step fails, you can return this error buffer.
    ///     Subsequent operations that receive this buffer as an input will "see" the error and fail immediately with
    ///     the same error, rather than trying to process invalid memory.
    ///
    /// # Parameters
    ///
    ///   - `error`: [`Error`] to store in the new buffer.
    ///   - `specification`: [`BufferSpecification`] for the new buffer.
    ///   - `memory`: [`Memory`] on which to place the new buffer.
    pub fn error_buffer<D: AsRef<[u64]>, M: HasDefaultMemory>(
        &'_ self,
        error: Error,
        specification: BufferSpecification<D>,
        memory: M,
    ) -> Result<Buffer<'_>, Error> {
        use ffi::PJRT_Client_CreateErrorBuffer_Args;
        let layout = specification.layout.map(|layout| unsafe { layout.to_c_api() });
        let error_message = error.message();
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_CreateErrorBuffer,
            {
                client = self.to_c_api(),
                error_code = error.code(),
                error_message = error_message.as_ptr(),
                error_message_size = error_message.count_bytes(),
                shape_dims = specification.dimensions.as_ref().as_ptr() as *const i64,
                shape_num_dims = specification.dimensions.as_ref().len(),
                shape_element_type = specification.element_type.to_c_api(),
                shape_layout = layout.map(|layout| &layout as *const _ as *mut _).unwrap_or(std::ptr::null_mut()),
                memory = memory.default_memory().to_c_api(),
            },
            { buffer },
        )
        .and_then(|handle| unsafe { Buffer::from_c_api(handle, self.api(), self.to_c_api()) })
    }

    /// Creates a [`Buffer`] that is an alias of another [`Buffer`]. The returned alias buffer is going to be
    /// uninitialized and must be _fulfilled_ later by passing the returned [`AliasBufferFulfillmentToken`] to
    /// [`Client::fulfill_alias_buffer`] or [`Client::fulfill_alias_buffer_with_error`].
    ///
    /// This function provides a mechanism for effectively creating a _future_ in the PJRT runtime. It creates a valid
    /// [`Buffer`] that represents a value that will be computed or transferred at a later point in time. This function
    /// allocates the metadata and identity for a buffer but does not necessarily allocate the backing physical storage
    /// or populate it with data immediately. It returns two objects:
    ///
    ///   - **Alias Buffer:** The resulting alias [`Buffer`] which consumers can use immediately to build downstream
    ///     computations (e.g., by executing PJRT programs that use this buffer as one of their inputs, returning
    ///     output buffers that are also _lazy_ and whose values will not be available until the alias [`Buffer`] is
    ///     _fulfilled_ and the program is executed using that input).
    ///   - **Fulfillment Token:** A capability token that can later on be used via the [`Client::fulfill_alias_buffer`]
    ///     or [`Client::fulfill_alias_buffer_with_error`] functions to bind the alias [`Buffer`] to actual data, thus
    ///     _fulfilling_ it and enabling the execution of downstream computations that depend on it.
    ///
    /// The power of this function lies in its ability to separate the definition of a buffer (e.g., its memory location
    /// and [`BufferSpecification`]) from its instantiation, and it is made possible by the _lazy_/_asynchronous_ nature
    /// of PJRT [`Buffer`]s. This enables things like:
    ///
    ///   - **Latency Hiding:** In a distributed system, a worker node might know that it will receive a 1024 x 1024
    ///     [`BufferType::F32`] matrix from a peer node. By using [`Client::alias_buffer`] it can create a [`Buffer`]
    ///     representing that (yet unavailable) matrix and use it to kick off a downstream computation. The parts of
    ///     that computation that depend on the value of that buffer will eventually get executed once that buffer is
    ///     _fulfilled_, but other parts of the computation may execute earlier, thus hiding some potential latency.
    ///   - **Graph Construction:** Passing the resulting alias [`Buffer`] to a program execution as one of its inputs
    ///     will result in the compiler/runtime constructing an execution graph with a dependency on the data stored
    ///     in that buffer. The execution of that graph will block on the [`Device`] (or in the _stream_) until that
    ///     alias [`Buffer`] is _fulfilled_, but the host thread will remain free to queue more work.
    ///
    /// # Parameters
    ///
    ///   - `specification`: [`BufferSpecification`] for the new alias buffer.
    ///   - `memory`: [`Memory`] on which the underlying buffer is expected to reside. This is critical for scheduling.
    ///     Even though the data is not there yet, the compiler needs to know the destination memory space to insert any
    ///     necessary DMA transfers, etc., for downstream operations.
    pub fn alias_buffer<D: AsRef<[u64]>, M: HasDefaultMemory>(
        &'_ self,
        specification: BufferSpecification<D>,
        memory: M,
    ) -> Result<(Buffer<'_>, AliasBufferFulfillmentToken), Error> {
        use ffi::PJRT_Client_CreateAliasBuffer_Args;
        let layout = specification.layout.map(|layout| unsafe { layout.to_c_api() });
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_CreateAliasBuffer,
            {
                client = self.to_c_api(),
                memory = memory.default_memory().to_c_api(),
                shape_dims = specification.dimensions.as_ref().as_ptr() as *const i64,
                shape_num_dims = specification.dimensions.as_ref().len(),
                shape_element_type = specification.element_type.to_c_api(),
                shape_layout = layout.map(|layout| &layout as *const _ as *mut _).unwrap_or(std::ptr::null_mut()),
            },
            { alias_buffer, fulfill_alias_buffer_cb },
        )
        .and_then(|(buffer_handle, token_handle)| unsafe {
            Buffer::from_c_api(buffer_handle, self.api(), self.to_c_api())
                .map(|buffer| (buffer, AliasBufferFulfillmentToken { handle: token_handle }))
        })
    }

    /// _Fulfills_ an alias [`Buffer`] that corresponds to the provided [`AliasBufferFulfillmentToken`] by binding it
    /// to the data that the provided [`Buffer`] contains. Refer to the documentation of the [`Client::alias_buffer`]
    /// function for information on what an alias buffer is. That function is also the means by which one can obtain
    /// an alias buffer as well as the [`AliasBufferFulfillmentToken`] that corresponds to it and which can be passed
    /// to this function to _fulfill_ it by binding data to it.
    ///
    /// Calling this function will allow any operations that are blocked because they depend on the value of the alias
    /// [`Buffer`] to resume execution.
    ///
    /// If instead of providing a value for _fulfilling_ the alias buffer you want to _poison_ it with an [`Error`],
    /// you must use the [`Client::fulfill_alias_buffer_with_error`] function instead.
    ///
    /// # Parameters
    ///
    ///   - `token`: [`AliasBufferFulfillmentToken`] that corresponds to the alias buffer that will be _fulfilled_.
    ///   - `buffer`: [`Buffer`] with which to _fulfill_ the alias buffer. After calling this function, this and the
    ///     alias buffer that corresponds to the provided `token` will share the same underlying memory (i.e., after
    ///     calling this function the alias buffer will become effectively a reference to this buffer). That memory
    ///     will not be freed while either of these two buffers is alive. Note that this buffer must have the same
    ///     [`BufferSpecification`] as the alias buffer and must reside in the same [`Memory`].
    pub fn fulfill_alias_buffer(&'_ self, token: AliasBufferFulfillmentToken, buffer: Buffer<'_>) -> Result<(), Error> {
        use ffi::PJRT_Client_FulfillAliasBuffer_Args;
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_FulfillAliasBuffer,
            {
                client = self.to_c_api(),
                buffer = buffer.to_c_api(),
                status_code = crate::errors::ffi::PJRT_Error_Code_OK,
                error_message = std::ptr::null(),
                error_message_size = 0,
                fulfill_alias_buffer_cb = token.handle,
            },
        )
    }

    /// _Fulfills_ an alias [`Buffer`] that corresponds to the provided [`AliasBufferFulfillmentToken`] by _poisoning_
    /// it with the provided [`Error`]. Refer to the documentation of the [`Client::alias_buffer`] function for
    /// information on what an alias buffer is. That function is also the means by which one can obtain an alias
    /// buffer as well as the [`AliasBufferFulfillmentToken`] that corresponds to it and which can be passed to
    /// this function to _fulfill_ it by binding data to it.
    ///
    /// Calling this function will cause any operations that are blocked because they depend on the value of the
    /// alias [`Buffer`] to fail execution by propagating the provided [`Error`] as the execution error. Refer to
    /// the documentation of [`Client::error_buffer`] for more information on buffer _poisoning_. This enables
    /// robust fault tolerance in asynchronous computation graphs that avoids crashing the host process.
    ///
    /// # Parameters
    ///
    ///   - `token`: [`AliasBufferFulfillmentToken`] that corresponds to the alias buffer that will be _fulfilled_.
    ///   - `error`: [`Error`] with which to poison the alias buffer.
    pub fn fulfill_alias_buffer_with_error(
        &self,
        token: AliasBufferFulfillmentToken,
        error: Error,
    ) -> Result<(), Error> {
        use ffi::PJRT_Client_FulfillAliasBuffer_Args;
        let error_message = error.message();
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_FulfillAliasBuffer,
            {
                client = self.to_c_api(),
                buffer = std::ptr::null_mut(),
                status_code = error.code(),
                error_message = error_message.as_ptr(),
                error_message_size = error_message.count_bytes(),
                fulfill_alias_buffer_cb = token.handle,
            },
        )
    }

    /// Creates a new [`DmaMappedBuffer`] by _pinning_ the memory referenced by `ptr` (of size `len`). Refer to the
    /// documentation of [`DmaMappedBuffer`] for more information on _pinning_ and _Direct Memory Access (DMA)_, as
    /// well as for any caveats or limitations of this functionality in PJRT.
    pub unsafe fn dma_map(&'_ self, ptr: *mut std::ffi::c_void, len: usize) -> Result<DmaMappedBuffer<'_>, Error> {
        use ffi::PJRT_Client_DmaMap_Args;
        let client = unsafe { self.to_c_api() };
        invoke_pjrt_api_error_fn!(
            self.api(),
            PJRT_Client_DmaMap,
            {
                client = client,
                data = ptr,
                size = len,
            },
        )
        .map(|_| DmaMappedBuffer { ptr, len, api: self.api(), client, owner: PhantomData })
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    use std::marker::{PhantomData, PhantomPinned};

    use crate::clients::ffi::PJRT_Client;
    use crate::devices::ffi::PJRT_Device;
    use crate::errors::ffi::{PJRT_Error, PJRT_Error_Code};
    use crate::events::ffi::PJRT_Event;
    use crate::ffi::PJRT_Extension_Base;
    use crate::memories::ffi::PJRT_Memory;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_Buffer {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    pub type PJRT_Buffer_Type = std::ffi::c_uint;
    pub const PJRT_Buffer_Type_INVALID: PJRT_Buffer_Type = 0;
    pub const PJRT_Buffer_Type_PRED: PJRT_Buffer_Type = 1;
    pub const PJRT_Buffer_Type_S8: PJRT_Buffer_Type = 2;
    pub const PJRT_Buffer_Type_S16: PJRT_Buffer_Type = 3;
    pub const PJRT_Buffer_Type_S32: PJRT_Buffer_Type = 4;
    pub const PJRT_Buffer_Type_S64: PJRT_Buffer_Type = 5;
    pub const PJRT_Buffer_Type_U8: PJRT_Buffer_Type = 6;
    pub const PJRT_Buffer_Type_U16: PJRT_Buffer_Type = 7;
    pub const PJRT_Buffer_Type_U32: PJRT_Buffer_Type = 8;
    pub const PJRT_Buffer_Type_U64: PJRT_Buffer_Type = 9;
    pub const PJRT_Buffer_Type_F16: PJRT_Buffer_Type = 10;
    pub const PJRT_Buffer_Type_F32: PJRT_Buffer_Type = 11;
    pub const PJRT_Buffer_Type_F64: PJRT_Buffer_Type = 12;
    pub const PJRT_Buffer_Type_BF16: PJRT_Buffer_Type = 13;
    pub const PJRT_Buffer_Type_C64: PJRT_Buffer_Type = 14;
    pub const PJRT_Buffer_Type_C128: PJRT_Buffer_Type = 15;
    pub const PJRT_Buffer_Type_F8E5M2: PJRT_Buffer_Type = 16;
    pub const PJRT_Buffer_Type_F8E4M3FN: PJRT_Buffer_Type = 17;
    pub const PJRT_Buffer_Type_F8E4M3B11FNUZ: PJRT_Buffer_Type = 18;
    pub const PJRT_Buffer_Type_F8E5M2FNUZ: PJRT_Buffer_Type = 19;
    pub const PJRT_Buffer_Type_F8E4M3FNUZ: PJRT_Buffer_Type = 20;
    pub const PJRT_Buffer_Type_S4: PJRT_Buffer_Type = 21;
    pub const PJRT_Buffer_Type_U4: PJRT_Buffer_Type = 22;
    pub const PJRT_Buffer_Type_TOKEN: PJRT_Buffer_Type = 23;
    pub const PJRT_Buffer_Type_S2: PJRT_Buffer_Type = 24;
    pub const PJRT_Buffer_Type_U2: PJRT_Buffer_Type = 25;
    pub const PJRT_Buffer_Type_F8E4M3: PJRT_Buffer_Type = 26;
    pub const PJRT_Buffer_Type_F8E3M4: PJRT_Buffer_Type = 27;
    pub const PJRT_Buffer_Type_F8E8M0FNU: PJRT_Buffer_Type = 28;
    pub const PJRT_Buffer_Type_F4E2M1FN: PJRT_Buffer_Type = 29;

    pub type PJRT_Buffer_MemoryLayout_Type = std::ffi::c_uint;
    pub const PJRT_Buffer_MemoryLayout_Type_Tiled: PJRT_Buffer_MemoryLayout_Type = 0;
    pub const PJRT_Buffer_MemoryLayout_Type_Strides: PJRT_Buffer_MemoryLayout_Type = 1;

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct PJRT_Buffer_MemoryLayout_Tiled {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub minor_to_major: *const i64,
        pub minor_to_major_size: usize,
        pub tile_dims: *const i64,
        pub tile_dim_sizes: *const usize,
        pub num_tiles: usize,
    }

    impl PJRT_Buffer_MemoryLayout_Tiled {
        pub fn new(
            minor_to_major: *const i64,
            minor_to_major_size: usize,
            tile_dims: *const i64,
            tile_dim_sizes: *const usize,
            num_tiles: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                minor_to_major,
                minor_to_major_size,
                tile_dims,
                tile_dim_sizes,
                num_tiles,
            }
        }
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct PJRT_Buffer_MemoryLayout_Strides {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub byte_strides: *const i64,
        pub num_byte_strides: usize,
    }

    impl PJRT_Buffer_MemoryLayout_Strides {
        pub fn new(byte_strides: *const i64, num_byte_strides: usize) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                byte_strides,
                num_byte_strides,
            }
        }
    }

    #[repr(C)]
    pub union PJRT_Buffer_MemoryLayout_Value {
        pub tiled: PJRT_Buffer_MemoryLayout_Tiled,
        pub strides: PJRT_Buffer_MemoryLayout_Strides,
    }

    #[repr(C)]
    pub struct PJRT_Buffer_MemoryLayout {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub memory_layout: PJRT_Buffer_MemoryLayout_Value,
        pub memory_layout_type: PJRT_Buffer_MemoryLayout_Type,
    }

    impl PJRT_Buffer_MemoryLayout {
        pub fn new(
            memory_layout: PJRT_Buffer_MemoryLayout_Value,
            memory_layout_type: PJRT_Buffer_MemoryLayout_Type,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                memory_layout,
                memory_layout_type,
            }
        }
    }

    #[repr(C)]
    pub struct PJRT_Buffer_ElementType_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub element_type: PJRT_Buffer_Type,
    }

    impl PJRT_Buffer_ElementType_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), buffer, element_type: 0 }
        }
    }

    pub type PJRT_Buffer_ElementType = unsafe extern "C" fn(args: *mut PJRT_Buffer_ElementType_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_Dimensions_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub dims: *const i64,
        pub num_dims: usize,
    }

    impl PJRT_Buffer_Dimensions_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                dims: std::ptr::null(),
                num_dims: 0,
            }
        }
    }

    pub type PJRT_Buffer_Dimensions = unsafe extern "C" fn(args: *mut PJRT_Buffer_Dimensions_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_UnpaddedDimensions_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub unpadded_dims: *const i64,
        pub num_dims: usize,
    }

    impl PJRT_Buffer_UnpaddedDimensions_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                unpadded_dims: std::ptr::null(),
                num_dims: 0,
            }
        }
    }

    pub type PJRT_Buffer_UnpaddedDimensions =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_UnpaddedDimensions_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_DynamicDimensionIndices_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub dynamic_dim_indices: *const usize,
        pub num_dynamic_dims: usize,
    }

    impl PJRT_Buffer_DynamicDimensionIndices_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                dynamic_dim_indices: std::ptr::null(),
                num_dynamic_dims: 0,
            }
        }
    }

    pub type PJRT_Buffer_DynamicDimensionIndices =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_DynamicDimensionIndices_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_Device_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub device: *mut PJRT_Device,
    }

    impl PJRT_Buffer_Device_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                device: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Buffer_Device = unsafe extern "C" fn(args: *mut PJRT_Buffer_Device_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_IsOnCpu_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub is_on_cpu: bool,
    }

    impl PJRT_Buffer_IsOnCpu_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), buffer, is_on_cpu: false }
        }
    }

    pub type PJRT_Buffer_IsOnCpu = unsafe extern "C" fn(args: *mut PJRT_Buffer_IsOnCpu_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_Memory_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub memory: *mut PJRT_Memory,
    }

    impl PJRT_Buffer_Memory_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                memory: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Buffer_Memory = unsafe extern "C" fn(args: *mut PJRT_Buffer_Memory_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_GetMemoryLayout_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub layout: PJRT_Buffer_MemoryLayout,
    }

    pub type PJRT_Buffer_GetMemoryLayout =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_GetMemoryLayout_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_OnDeviceSizeInBytes_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub on_device_size_in_bytes: usize,
    }

    impl PJRT_Buffer_OnDeviceSizeInBytes_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                on_device_size_in_bytes: 0,
            }
        }
    }

    pub type PJRT_Buffer_OnDeviceSizeInBytes =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_OnDeviceSizeInBytes_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_ToHostBuffer_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub src: *mut PJRT_Buffer,
        pub host_layout: *mut PJRT_Buffer_MemoryLayout,
        pub dst: *mut std::ffi::c_void,
        pub dst_size: usize,
        pub event: *mut PJRT_Event,
    }

    impl PJRT_Buffer_ToHostBuffer_Args {
        pub fn new(
            src: *mut PJRT_Buffer,
            host_layout: *mut PJRT_Buffer_MemoryLayout,
            dst: *mut std::ffi::c_void,
            dst_size: usize,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                src,
                host_layout,
                dst,
                dst_size,
                event: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Buffer_ToHostBuffer =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_ToHostBuffer_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_CopyRawToHost_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub dst: *mut std::ffi::c_void,
        pub offset: i64,
        pub transfer_size: i64,
        pub event: *mut PJRT_Event,
    }

    impl PJRT_Buffer_CopyRawToHost_Args {
        pub fn new(buffer: *mut PJRT_Buffer, dst: *mut std::ffi::c_void, offset: i64, transfer_size: i64) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                dst,
                offset,
                transfer_size,
                event: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Buffer_CopyRawToHost =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_CopyRawToHost_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_CopyRawToHostFuture_Callback_Args {
        pub struct_size: usize,
        pub callback_data: *mut std::ffi::c_void,
        pub error_code: PJRT_Error_Code,
        pub error_message: *const std::ffi::c_char,
        pub error_message_size: usize,
        pub dst: *mut std::ffi::c_void,
    }

    impl PJRT_Buffer_CopyRawToHostFuture_Callback_Args {
        pub fn new(
            callback_data: *mut std::ffi::c_void,
            error_code: PJRT_Error_Code,
            error_message: *const std::ffi::c_char,
            error_message_size: usize,
            dst: *mut std::ffi::c_void,
        ) -> Self {
            Self { struct_size: size_of::<Self>(), callback_data, error_code, error_message, error_message_size, dst }
        }
    }

    pub type PJRT_Buffer_CopyRawToHostFuture_Callback =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_CopyRawToHostFuture_Callback_Args);

    #[repr(C)]
    pub struct PJRT_Buffer_CopyRawToHostFuture_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub offset: i64,
        pub transfer_size: i64,
        pub event: *mut PJRT_Event,
        pub callback_data: *mut std::ffi::c_void,
        pub future_ready_callback: Option<PJRT_Buffer_CopyRawToHostFuture_Callback>,
    }

    impl PJRT_Buffer_CopyRawToHostFuture_Args {
        pub fn new(buffer: *mut PJRT_Buffer, offset: i64, transfer_size: i64) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                offset,
                transfer_size,
                event: std::ptr::null_mut(),
                callback_data: std::ptr::null_mut(),
                future_ready_callback: None,
            }
        }
    }

    pub type PJRT_Buffer_CopyRawToHostFuture =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_CopyRawToHostFuture_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_CopyToMemory_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub dst_memory: *mut PJRT_Memory,
        pub dst_buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Buffer_CopyToMemory_Args {
        pub fn new(buffer: *mut PJRT_Buffer, dst_memory: *mut PJRT_Memory) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                dst_memory,
                dst_buffer: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Buffer_CopyToMemory =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_CopyToMemory_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_CopyToDevice_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub dst_device: *mut PJRT_Device,
        pub dst_buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Buffer_CopyToDevice_Args {
        pub fn new(buffer: *mut PJRT_Buffer, dst_device: *mut PJRT_Device) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                dst_device,
                dst_buffer: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Buffer_CopyToDevice =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_CopyToDevice_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_ReadyEvent_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub event: *mut PJRT_Event,
    }

    impl PJRT_Buffer_ReadyEvent_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                event: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Buffer_ReadyEvent = unsafe extern "C" fn(args: *mut PJRT_Buffer_ReadyEvent_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_IncreaseExternalReferenceCount_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Buffer_IncreaseExternalReferenceCount_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), buffer }
        }
    }

    pub type PJRT_Buffer_IncreaseExternalReferenceCount =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_IncreaseExternalReferenceCount_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_DecreaseExternalReferenceCount_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Buffer_DecreaseExternalReferenceCount_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), buffer }
        }
    }

    pub type PJRT_Buffer_DecreaseExternalReferenceCount =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_DecreaseExternalReferenceCount_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub device_memory_ptr: *mut std::ffi::c_void,
    }

    impl PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                device_memory_ptr: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Buffer_OpaqueDeviceMemoryDataPointer =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_UnsafePointer_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub buffer_pointer: usize,
    }

    impl PJRT_Buffer_UnsafePointer_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), buffer, buffer_pointer: 0 }
        }
    }

    pub type PJRT_Buffer_UnsafePointer =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_UnsafePointer_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_DonateWithControlDependency_Callback_Args {
        pub struct_size: usize,
        pub callback_data: *mut std::ffi::c_void,
        pub error_code: PJRT_Error_Code,
        pub error_message: *const std::ffi::c_char,
        pub error_message_size: usize,
    }

    impl PJRT_Buffer_DonateWithControlDependency_Callback_Args {
        pub fn new(
            callback_data: *mut std::ffi::c_void,
            error_code: PJRT_Error_Code,
            error_message: *const std::ffi::c_char,
            error_message_size: usize,
        ) -> Self {
            Self { struct_size: size_of::<Self>(), callback_data, error_code, error_message, error_message_size }
        }
    }

    pub type PJRT_Buffer_DonateWithControlDependency_Callback =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_DonateWithControlDependency_Callback_Args);

    #[repr(C)]
    pub struct PJRT_Buffer_DonateWithControlDependency_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub callback_data: *mut std::ffi::c_void,
        pub dependency_ready_callback: Option<PJRT_Buffer_DonateWithControlDependency_Callback>,
        pub out_buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Buffer_DonateWithControlDependency_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                buffer,
                callback_data: std::ptr::null_mut(),
                dependency_ready_callback: None,
                out_buffer: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Buffer_DonateWithControlDependency =
        unsafe extern "C" fn(args: *mut PJRT_Buffer_DonateWithControlDependency_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_IsDeleted_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
        pub is_deleted: bool,
    }

    impl PJRT_Buffer_IsDeleted_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), buffer, is_deleted: false }
        }
    }

    pub type PJRT_Buffer_IsDeleted = unsafe extern "C" fn(args: *mut PJRT_Buffer_IsDeleted_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_Delete_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Buffer_Delete_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), buffer }
        }
    }

    pub type PJRT_Buffer_Delete = unsafe extern "C" fn(args: *mut PJRT_Buffer_Delete_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Buffer_Destroy_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Buffer_Destroy_Args {
        pub fn new(buffer: *mut PJRT_Buffer) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), buffer }
        }
    }

    pub type PJRT_Buffer_Destroy = unsafe extern "C" fn(args: *mut PJRT_Buffer_Destroy_Args) -> *mut PJRT_Error;

    pub type PJRT_HostBufferSemantics = std::ffi::c_uint;
    pub const PJRT_HostBufferSemantics_kImmutableOnlyDuringCall: PJRT_HostBufferSemantics = 0;
    pub const PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes: PJRT_HostBufferSemantics = 1;
    pub const PJRT_HostBufferSemantics_kImmutableZeroCopy: PJRT_HostBufferSemantics = 2;
    pub const PJRT_HostBufferSemantics_kMutableZeroCopy: PJRT_HostBufferSemantics = 3;

    #[repr(C)]
    pub struct PJRT_Client_BufferFromHostBuffer_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub data: *const std::ffi::c_void,
        pub data_type: PJRT_Buffer_Type,
        pub dims: *const i64,
        pub num_dims: usize,
        pub byte_strides: *const i64,
        pub num_byte_strides: usize,
        pub host_buffer_semantics: PJRT_HostBufferSemantics,
        pub device: *mut PJRT_Device,
        pub memory: *mut PJRT_Memory,
        pub device_layout: *mut PJRT_Buffer_MemoryLayout,
        pub done_with_host_buffer: *mut PJRT_Event,
        pub buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Client_BufferFromHostBuffer_Args {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            client: *mut PJRT_Client,
            data: *const std::ffi::c_void,
            data_type: PJRT_Buffer_Type,
            dims: *const i64,
            num_dims: usize,
            byte_strides: *const i64,
            num_byte_strides: usize,
            host_buffer_semantics: PJRT_HostBufferSemantics,
            device: *mut PJRT_Device,
            memory: *mut PJRT_Memory,
            device_layout: *mut PJRT_Buffer_MemoryLayout,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                data,
                data_type,
                dims,
                num_dims,
                byte_strides,
                num_byte_strides,
                host_buffer_semantics,
                device,
                memory,
                device_layout,
                done_with_host_buffer: std::ptr::null_mut(),
                buffer: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Client_BufferFromHostBuffer =
        unsafe extern "C" fn(args: *mut PJRT_Client_BufferFromHostBuffer_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_CreateUninitializedBuffer_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub shape_dims: *const i64,
        pub shape_num_dims: usize,
        pub shape_element_type: PJRT_Buffer_Type,
        pub shape_layout: *mut PJRT_Buffer_MemoryLayout,
        pub device: *mut PJRT_Device,
        pub memory: *mut PJRT_Memory,
        pub buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Client_CreateUninitializedBuffer_Args {
        pub fn new(
            client: *mut PJRT_Client,
            shape_dims: *const i64,
            shape_num_dims: usize,
            shape_element_type: PJRT_Buffer_Type,
            shape_layout: *mut PJRT_Buffer_MemoryLayout,
            device: *mut PJRT_Device,
            memory: *mut PJRT_Memory,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                shape_dims,
                shape_num_dims,
                shape_element_type,
                shape_layout,
                device,
                memory,
                buffer: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Client_CreateUninitializedBuffer =
        unsafe extern "C" fn(args: *mut PJRT_Client_CreateUninitializedBuffer_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_CreateViewOfDeviceBuffer_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub device_buffer_ptr: *mut std::ffi::c_void,
        pub dims: *const i64,
        pub num_dims: usize,
        pub element_type: PJRT_Buffer_Type,
        pub layout: *mut PJRT_Buffer_MemoryLayout,
        pub device: *mut PJRT_Device,
        pub on_delete_callback:
            unsafe extern "C" fn(device_buffer_ptr: *mut std::ffi::c_void, user_arg: *mut std::ffi::c_void),
        pub on_delete_callback_arg: *mut std::ffi::c_void,
        pub stream: isize,
        pub buffer: *mut PJRT_Buffer,
        pub memory: *mut PJRT_Memory,
    }

    impl PJRT_Client_CreateViewOfDeviceBuffer_Args {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            client: *mut PJRT_Client,
            device_buffer_ptr: *mut std::ffi::c_void,
            dims: *const i64,
            num_dims: usize,
            element_type: PJRT_Buffer_Type,
            layout: *mut PJRT_Buffer_MemoryLayout,
            device: *mut PJRT_Device,
            memory: *mut PJRT_Memory,
            stream: isize,
            on_delete_callback: unsafe extern "C" fn(
                device_buffer_ptr: *mut std::ffi::c_void,
                user_arg: *mut std::ffi::c_void,
            ),
            on_delete_callback_arg: *mut std::ffi::c_void,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                device_buffer_ptr,
                dims,
                num_dims,
                element_type,
                layout,
                device,
                on_delete_callback,
                on_delete_callback_arg,
                stream,
                buffer: std::ptr::null_mut(),
                memory,
            }
        }
    }

    pub type PJRT_Client_CreateViewOfDeviceBuffer =
        unsafe extern "C" fn(args: *mut PJRT_Client_CreateViewOfDeviceBuffer_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_CreateErrorBuffer_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub error_code: PJRT_Error_Code,
        pub error_message: *const std::ffi::c_char,
        pub error_message_size: usize,
        pub shape_dims: *const i64,
        pub shape_num_dims: usize,
        pub shape_element_type: PJRT_Buffer_Type,
        pub shape_layout: *mut PJRT_Buffer_MemoryLayout,
        pub memory: *mut PJRT_Memory,
        pub buffer: *mut PJRT_Buffer,
    }

    impl PJRT_Client_CreateErrorBuffer_Args {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            client: *mut PJRT_Client,
            error_code: PJRT_Error_Code,
            error_message: *const std::ffi::c_char,
            error_message_size: usize,
            shape_dims: *const i64,
            shape_num_dims: usize,
            shape_element_type: PJRT_Buffer_Type,
            shape_layout: *mut PJRT_Buffer_MemoryLayout,
            memory: *mut PJRT_Memory,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                error_code,
                error_message,
                error_message_size,
                shape_dims,
                shape_num_dims,
                shape_element_type,
                shape_layout,
                memory,
                buffer: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Client_CreateErrorBuffer =
        unsafe extern "C" fn(args: *mut PJRT_Client_CreateErrorBuffer_Args) -> *mut PJRT_Error;

    // We represent opaque C types as structs with a particular structure that is following the convention
    // suggested in [the Rustonomicon](https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs).
    #[repr(C)]
    pub struct PJRT_FulfillAliasBufferCallback {
        _data: [u8; 0],
        _marker: PhantomData<(*mut u8, PhantomPinned)>,
    }

    #[repr(C)]
    pub struct PJRT_Client_CreateAliasBuffer_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub memory: *mut PJRT_Memory,
        pub shape_dims: *const i64,
        pub shape_num_dims: usize,
        pub shape_element_type: PJRT_Buffer_Type,
        pub shape_layout: *mut PJRT_Buffer_MemoryLayout,
        pub alias_buffer: *mut PJRT_Buffer,
        pub fulfill_alias_buffer_cb: *mut PJRT_FulfillAliasBufferCallback,
    }

    impl PJRT_Client_CreateAliasBuffer_Args {
        pub fn new(
            client: *mut PJRT_Client,
            memory: *mut PJRT_Memory,
            shape_dims: *const i64,
            shape_num_dims: usize,
            shape_element_type: PJRT_Buffer_Type,
            shape_layout: *mut PJRT_Buffer_MemoryLayout,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                memory,
                shape_dims,
                shape_num_dims,
                shape_element_type,
                shape_layout,
                alias_buffer: std::ptr::null_mut(),
                fulfill_alias_buffer_cb: std::ptr::null_mut(),
            }
        }
    }

    pub type PJRT_Client_CreateAliasBuffer =
        unsafe extern "C" fn(args: *mut PJRT_Client_CreateAliasBuffer_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_FulfillAliasBuffer_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub buffer: *mut PJRT_Buffer,
        pub status_code: PJRT_Error_Code,
        pub error_message: *const std::ffi::c_char,
        pub error_message_size: usize,
        pub fulfill_alias_buffer_cb: *mut PJRT_FulfillAliasBufferCallback,
    }

    impl PJRT_Client_FulfillAliasBuffer_Args {
        pub fn new(
            client: *mut PJRT_Client,
            buffer: *mut PJRT_Buffer,
            status_code: PJRT_Error_Code,
            error_message: *const std::ffi::c_char,
            error_message_size: usize,
            fulfill_alias_buffer_cb: *mut PJRT_FulfillAliasBufferCallback,
        ) -> Self {
            Self {
                struct_size: size_of::<Self>(),
                extension_start: std::ptr::null_mut(),
                client,
                buffer,
                status_code,
                error_message,
                error_message_size,
                fulfill_alias_buffer_cb,
            }
        }
    }

    pub type PJRT_Client_FulfillAliasBuffer =
        unsafe extern "C" fn(args: *mut PJRT_Client_FulfillAliasBuffer_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_DmaMap_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub data: *mut std::ffi::c_void,
        pub size: usize,
    }

    impl PJRT_Client_DmaMap_Args {
        pub fn new(client: *mut PJRT_Client, data: *mut std::ffi::c_void, size: usize) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), client, data, size }
        }
    }

    pub type PJRT_Client_DmaMap = unsafe extern "C" fn(args: *mut PJRT_Client_DmaMap_Args) -> *mut PJRT_Error;

    #[repr(C)]
    pub struct PJRT_Client_DmaUnmap_Args {
        pub struct_size: usize,
        pub extension_start: *mut PJRT_Extension_Base,
        pub client: *mut PJRT_Client,
        pub data: *mut std::ffi::c_void,
    }

    impl PJRT_Client_DmaUnmap_Args {
        pub fn new(client: *mut PJRT_Client, data: *mut std::ffi::c_void) -> Self {
            Self { struct_size: size_of::<Self>(), extension_start: std::ptr::null_mut(), client, data }
        }
    }

    pub type PJRT_Client_DmaUnmap = unsafe extern "C" fn(args: *mut PJRT_Client_DmaUnmap_Args) -> *mut PJRT_Error;
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::tests::{TestPlatform, test_cpu_client, test_for_each_platform};
    use crate::{
        Buffer, BufferSpecification, BufferType, Error, HostBuffer, HostBufferData, HostBufferSemantics, Layout,
        StridedLayout, Tile, TileDimension, TiledLayout,
    };

    use super::ffi;

    #[test]
    fn test_buffer_type() {
        let types = [
            BufferType::Invalid,
            BufferType::Token,
            BufferType::Predicate,
            BufferType::I2,
            BufferType::I4,
            BufferType::I8,
            BufferType::I16,
            BufferType::I32,
            BufferType::I64,
            BufferType::U2,
            BufferType::U4,
            BufferType::U8,
            BufferType::U16,
            BufferType::U32,
            BufferType::U64,
            BufferType::F4E2M1FN,
            BufferType::F8E3M4,
            BufferType::F8E4M3,
            BufferType::F8E4M3FN,
            BufferType::F8E4M3FNUZ,
            BufferType::F8E4M3B11FNUZ,
            BufferType::F8E5M2,
            BufferType::F8E5M2FNUZ,
            BufferType::F8E8M0FNU,
            BufferType::BF16,
            BufferType::F16,
            BufferType::F32,
            BufferType::F64,
            BufferType::C64,
            BufferType::C128,
        ];

        types.iter().copied().for_each(|r#type| unsafe {
            assert_eq!(BufferType::from_c_api(r#type.to_c_api()), r#type);
            assert_eq!(BufferType::from_proto(r#type.proto()), r#type);
            assert_eq!(BufferType::from_str(r#type.to_string()), Ok(r#type));
        });
        assert_eq!(unsafe { BufferType::from_c_api(u32::MAX) }, BufferType::Invalid);

        assert_eq!(format!("{}", BufferType::F64), "f64".to_string());
        assert_eq!(format!("{:?}", BufferType::F64), "F64".to_string());
    }

    #[test]
    fn test_tile_dimension() {
        let dimension = TileDimension::sized(16);
        assert_eq!(dimension.size(), Some(16));
        assert!(!dimension.is_combined());
        assert_eq!(format!("{dimension}"), "16");
        assert_eq!(format!("{dimension:?}"), "TileDimension[16]");

        let dimension = TileDimension::combined();
        assert_eq!(dimension.size(), None);
        assert!(dimension.is_combined());
        assert_eq!(format!("{dimension}"), "*");
        assert_eq!(format!("{dimension:?}"), "TileDimension[*]");
    }

    #[test]
    fn test_tile() {
        let tile = Tile {
            dimensions: vec![
                TileDimension::sized(16),
                TileDimension::combined(),
                TileDimension::sized(4),
                TileDimension::sized(2),
            ],
        };
        assert_eq!(format!("{tile}"), "(16,*,4,2)");
        assert_eq!(format!("{tile:?}"), "Tile[(16,*,4,2)]");
    }

    #[test]
    fn test_tiled_layout() {
        let tiles = vec![
            Tile { dimensions: vec![TileDimension::sized(8), TileDimension::combined()] },
            Tile { dimensions: vec![TileDimension::sized(4)] },
        ];
        let layout = TiledLayout::new(vec![2, 1, 0], tiles.clone());
        assert_eq!(unsafe { TiledLayout::from_c_api(layout.to_c_api()) }, layout);
        assert_eq!(layout.minor_to_major(), &[2, 1, 0]);
        assert_eq!(layout.tiles(), tiles);
        assert_eq!(layout.tile(0), Some(tiles[0].clone()));
        assert_eq!(layout.tile(1), Some(tiles[1].clone()));
        assert_eq!(layout.tile(2), None);

        let empty_layout = TiledLayout::new(Vec::new(), Vec::new());
        assert_eq!(empty_layout.minor_to_major(), &[]);
        assert_eq!(empty_layout.tiles(), Vec::<Tile>::new());
        assert_eq!(empty_layout.tile(0), None);
        assert_eq!(format!("{layout}"), "{2,1,0:T(8,*)(4)}");
        assert_eq!(format!("{layout:?}"), "TiledLayout[{2,1,0:T(8,*)(4)}]");
        assert_eq!(format!("{empty_layout}"), "{}");
        assert_eq!(format!("{empty_layout:?}"), "TiledLayout[{}]");
    }

    #[test]
    fn test_strided_layout() {
        let layout = StridedLayout::new(vec![24, 8, -4]);
        assert_eq!(layout.strides(), &[24, 8, -4]);
        assert_eq!(unsafe { StridedLayout::from_c_api(layout.to_c_api()) }, layout);
        assert_eq!(format!("{layout}"), "strides(24,8,-4)");
        assert_eq!(format!("{layout:?}"), "StridedLayout[strides(24,8,-4)]");
    }

    #[test]
    fn test_layout() {
        // Test round-tripping a [`TiledLayout`] through the C API.
        let layout = Layout::Tiled(TiledLayout::new(
            vec![1, 0],
            vec![Tile { dimensions: vec![TileDimension::sized(4), TileDimension::combined()] }],
        ));
        assert_eq!(unsafe { Layout::from_c_api(&layout.to_c_api() as *const _) }, Ok(layout.clone()));
        assert_eq!(Layout::from_str(layout.clone().to_string()), Ok(layout.clone()));
        assert_eq!(Layout::from_proto(layout.clone().proto().unwrap()), Ok(layout.clone()));
        assert_eq!(format!("{layout}"), "{1,0:T(4,*)}");
        assert_eq!(format!("{layout:?}"), "Layout[{1,0:T(4,*)}]");

        // Test round-tripping a [`StridedLayout`] through the C API.
        let layout = Layout::Strided(StridedLayout::new(vec![16, 4]));
        assert_eq!(unsafe { Layout::from_c_api(&layout.to_c_api() as *const _) }, Ok(layout.clone()));
        assert_eq!(Layout::from_str(layout.clone().to_string()), Ok(layout.clone()));
        assert!(matches!(
            layout.clone().proto(),
            Err(Error::InvalidArgument { message, .. })
              if message == "strided layouts cannot be represented in XLA layout Protobuf messages",
        ));
        assert_eq!(format!("{layout}"), "strides(16,4)");
        assert_eq!(format!("{layout:?}"), "Layout[strides(16,4)]");

        assert!(matches!(
            Layout::from_str("{:D(S~,C,H+)#(u4)*(u8)P(s64[4,2])}"),
            Err(Error::InvalidArgument { message, .. }) if message == "sparse layouts are not supported",
        ));

        assert!(matches!(
            Layout::from_str("{1,0:Q(123)}"),
            Err(Error::InvalidArgument { message, .. }) if message == "unsupported layout property 'Q'",
        ));

        // Test creating an invalid [`Layout`].
        let invalid_layout = ffi::PJRT_Buffer_MemoryLayout::new(
            ffi::PJRT_Buffer_MemoryLayout_Value {
                strides: ffi::PJRT_Buffer_MemoryLayout_Strides::new(std::ptr::null(), 0),
            },
            u32::MAX,
        );
        assert!(matches!(
            unsafe { Layout::from_c_api(&invalid_layout as *const _) },
            Err(Error::InvalidArgument { message, .. }) if message == "unknown PJRT buffer memory layout type",
        ));

        // Test creating a [`Layout`] from a null pointer.
        assert!(matches!(
            unsafe { Layout::from_c_api(std::ptr::null()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT buffer memory layout handle is a null pointer",
        ));
    }

    #[allow(deprecated)]
    #[test]
    fn test_buffer() {
        let client = test_cpu_client();
        let device = client.addressable_devices().unwrap()[0].clone();

        // Test constructing an invalid [`Buffer`].
        assert!(matches!(
            unsafe { Buffer::from_c_api(std::ptr::null_mut(), client.api(), client.to_c_api()) },
            Err(Error::InvalidArgument { message, .. })
                if message == "the provided PJRT buffer handle is a null pointer",
        ));

        // Test constructing a valid [`Buffer`].
        let buffer = client.buffer(&[1u8, 2u8, 3u8, 4u8], BufferType::U8, [4u64], None, device.clone(), None).unwrap();
        assert!(!unsafe { buffer.to_c_api() }.is_null());
        assert_eq!(buffer.element_type(), Ok(BufferType::U8));
        assert_eq!(buffer.rank(), Ok(1));
        assert_eq!(buffer.dimensions(), Ok([4u64].as_slice()));
        assert_eq!(buffer.unpadded_dimensions(), Ok([4u64].as_slice()));
        assert_eq!(buffer.dynamic_dimensions(), Ok([].as_slice()));
        assert_eq!(buffer.device().unwrap().id(), device.id());
        assert_eq!(buffer.is_on_cpu(), Ok(true));
        assert_eq!(buffer.memory(), device.default_memory());
        assert!(matches!(
            buffer.layout(),
            Ok(Layout::Tiled(TiledLayout {
                minor_to_major,
                tile_dimensions,
                tile_dimension_sizes,
                tile_count: 0,
            })) if minor_to_major == &[0] && tile_dimensions.is_empty() && tile_dimension_sizes.is_empty(),
        ));
        assert!(matches!(
            buffer.specification(),
            Ok(BufferSpecification {
                element_type: BufferType::U8,
                dimensions: [4u64],
                layout: Some(Layout::Tiled(TiledLayout {
                    minor_to_major,
                    tile_dimensions,
                    tile_dimension_sizes,
                    tile_count: 0,
                })),
            }) if minor_to_major == &[0] && tile_dimensions.is_empty() && tile_dimension_sizes.is_empty(),
        ));
        assert_eq!(buffer.on_device_size_in_bytes(), Ok(4));
        assert_eq!(buffer.ready().unwrap().r#await(), Ok(()));

        // Test external reference counting.
        assert!(unsafe { buffer.decrease_external_reference_count() }.is_err());
        assert!(unsafe { buffer.increase_external_reference_count() }.is_ok());
        assert!(unsafe { buffer.increase_external_reference_count() }.is_ok());
        assert!(unsafe { buffer.decrease_external_reference_count() }.is_ok());
        assert!(unsafe { buffer.decrease_external_reference_count() }.is_ok());
        assert!(unsafe { buffer.decrease_external_reference_count() }.is_err());

        assert!(unsafe { buffer.as_ptr() }.is_ok());
        assert!(unsafe { buffer.unsafe_pointer() }.is_ok());

        // Test equality comparisons.
        assert!(buffer == buffer);
        assert!(
            buffer == client.buffer(&[1u8, 2u8, 3u8, 4u8], BufferType::U8, [4u64], None, device.clone(), None).unwrap()
        );

        // Same buffer specification but different underlying data.
        assert!(
            buffer != client.buffer(&[1u8, 2u8, 3u8, 5u8], BufferType::U8, [4u64], None, device.clone(), None).unwrap()
        );

        // Same underlying data but different buffer specification.
        assert!(
            buffer
                != client.buffer(&[1u8, 2u8, 3u8, 4u8], BufferType::U16, [2u64], None, device.clone(), None).unwrap()
        );

        // Test buffer deletion.
        assert_eq!(buffer.is_deleted(), Ok(false));
        assert!(unsafe { buffer.delete() }.is_ok());
        assert_eq!(buffer.is_deleted(), Ok(true));
    }

    #[test]
    fn test_buffer_copy_to_host() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let data = [41u8, 42u8, 43u8, 44u8];
            let buffer = client
                .buffer(data.as_slice(), BufferType::U8, [data.len() as u64], None, device.clone(), None)
                .unwrap();
            assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(data.to_vec()));
        });
    }

    #[test]
    fn test_buffer_copy_to_host_buffer() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let data = [51u8, 52u8, 53u8, 54u8];
            let buffer = client
                .buffer(data.as_slice(), BufferType::U8, [data.len() as u64], None, device.clone(), None)
                .unwrap();
            let mut destination = vec![0u8; data.len()];
            let _ = buffer.copy_to_host_buffer(None, &mut destination).unwrap().r#await().unwrap();
            assert_eq!(destination.as_slice(), data.as_slice());
        });
    }

    #[test]
    fn test_buffer_copy_raw_to_host() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let data = [61u8, 62u8, 63u8, 64u8, 65u8, 66u8];
            let buffer = client
                .buffer(data.as_slice(), BufferType::U8, [data.len() as u64], None, device.clone(), None)
                .unwrap();
            assert_eq!(buffer.copy_raw_to_host(2, 3).unwrap().r#await(), Ok(vec![63u8, 64u8, 65u8]));
            assert!(buffer.copy_raw_to_host(2, 10).unwrap().r#await().is_err());
        });
    }

    #[test]
    fn test_buffer_copy_raw_to_host_buffer() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let data = [71u8, 72u8, 73u8, 74u8, 75u8];
            let buffer = client
                .buffer(data.as_slice(), BufferType::U8, [data.len() as u64], None, device.clone(), None)
                .unwrap();
            let mut destination = vec![0u8; 3];
            let _ = buffer.copy_raw_to_host_buffer(&mut destination, 1).unwrap().r#await().unwrap();
            assert_eq!(destination.as_slice(), vec![72u8, 73u8, 74u8]);
        });
    }

    #[test]
    fn test_buffer_copy_raw_to_host_buffer_future() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let data = [81u8, 82u8, 83u8, 84u8];

            // Test for a successful copy.
            let buffer = client
                .buffer(data.as_slice(), BufferType::U8, [data.len() as u64], None, device.clone(), None)
                .unwrap();
            let (callback, event) = buffer.copy_raw_to_host_buffer_future::<Vec<u8>>(0).unwrap();
            let mut destination = vec![0u8; data.len()];
            callback(&mut destination, None);
            assert_eq!(event.r#await(), Ok(()));
            assert_eq!(destination.as_slice(), data.as_slice());

            // Test for a failed copy.
            let buffer = client
                .buffer(data.as_slice(), BufferType::U8, [data.len() as u64], None, device.clone(), None)
                .unwrap();
            let (callback, event) = buffer.copy_raw_to_host_buffer_future::<Vec<u8>>(0).unwrap();
            let mut destination = vec![0u8; data.len()];
            callback(&mut destination, Some(Error::aborted("test error")));
            assert!(matches!(
                event.r#await(),
                Err(Error::Aborted { message, .. }) if message == "test error",
            ));
            assert_eq!(destination.as_slice(), &[0u8; 4]);
        });
    }

    #[test]
    fn test_buffer_copy_to_memory() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let memory_0 = client.addressable_memories().unwrap()[0];
            let memory_1 = client.addressable_memories().unwrap()[1];
            let data = [101u8, 102u8, 103u8, 104u8];
            let buffer = client
                .buffer(data.as_slice(), BufferType::U8, [data.len() as u64], None, memory_0.clone(), None)
                .unwrap();
            assert_eq!(buffer.memory().unwrap(), memory_0);
            let buffer = buffer.copy_to_memory(memory_0).unwrap();
            assert_eq!(buffer.memory().unwrap(), memory_0);
            let buffer = buffer.copy_to_memory(memory_1).unwrap();
            assert_eq!(buffer.memory().unwrap(), memory_1);
            assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(data.to_vec()));
        });
    }

    #[test]
    fn test_buffer_copy_to_device() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let devices = client.addressable_devices().unwrap();
            let device_0 = &devices[0];
            let device_1 = if devices.len() > 1 { &devices[1] } else { device_0 };
            let data = [101u8, 102u8, 103u8, 104u8];
            let buffer = client
                .buffer(data.as_slice(), BufferType::U8, [data.len() as u64], None, device_0.clone(), None)
                .unwrap();
            assert_eq!(buffer.device().unwrap(), device_0.clone());
            let buffer = buffer.copy_to_device(device_0.clone()).unwrap();
            assert_eq!(buffer.device().unwrap(), device_0.clone());
            let buffer = buffer.copy_to_device(device_1.clone()).unwrap();
            assert_eq!(buffer.device().unwrap(), device_1.clone());
            assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(data.to_vec()));
        });
    }

    #[test]
    fn test_buffer_donate_with_control_dependency() {
        test_for_each_platform!(|_plugin, client, platform| {
            let device = client.addressable_devices().unwrap()[0].clone();
            let buffer = client
                .buffer(&[1u8, 2u8, 3u8, 4u8], BufferType::U16, [2u64, 1u64], None, device.clone(), None)
                .unwrap();
            let buffer_and_callback = buffer.donate_with_control_dependency();
            match platform {
                TestPlatform::Cuda12 | TestPlatform::Cuda13 | TestPlatform::Rocm7 | TestPlatform::Tpu => {
                    // Test invoking the callback with no error.
                    let (buffer, callback) = buffer_and_callback.unwrap();
                    assert!(!buffer.ready().unwrap().ready().unwrap());
                    callback(None);
                    buffer.ready().unwrap().r#await().unwrap();

                    // Test invoking the callback with an error.
                    let (buffer, callback) = client
                        .buffer(&[1u8, 2u8, 3u8, 4u8], BufferType::U8, [4u64], None, device, None)
                        .unwrap()
                        .donate_with_control_dependency()
                        .unwrap();
                    let error = Error::aborted("test error");
                    callback(Some(error.clone()));
                    assert_eq!(buffer.ready().unwrap().r#await(), Err(error));
                }
                _ => assert!(matches!(buffer_and_callback, Err(Error::Unimplemented { .. }))),
            }
        });
    }

    #[test]
    fn test_buffer_specification() {
        let specification = BufferSpecification {
            element_type: BufferType::F32,
            dimensions: vec![2, 3],
            layout: Some(Layout::Tiled(TiledLayout::new(
                vec![1, 0],
                vec![
                    Tile { dimensions: vec![TileDimension::sized(4), TileDimension::sized(2)] },
                    Tile { dimensions: vec![TileDimension::combined()] },
                ],
            ))),
        };
        assert_eq!(BufferSpecification::from_str(specification.to_string()), Ok(specification.clone()));
        assert_eq!(BufferSpecification::from_proto(specification.proto().unwrap()), Ok(specification.clone()));
        assert_eq!(format!("{specification}"), "f32[2,3]{1,0:T(4,2)(*)}");
        assert_eq!(format!("{specification:?}"), "BufferSpecification[f32[2,3]{1,0:T(4,2)(*)}]");

        let specification = BufferSpecification {
            element_type: BufferType::F32,
            dimensions: vec![2, 3],
            layout: Some(Layout::Strided(StridedLayout::new(vec![12, 4]))),
        };
        assert_eq!(BufferSpecification::from_str(specification.to_string()), Ok(specification.clone()));
        assert!(matches!(
            specification.clone().proto(),
            Err(Error::InvalidArgument { message, .. })
              if message == "strided layouts cannot be represented in XLA layout Protobuf messages",
        ));
        assert_eq!(format!("{specification}"), "f32[2,3]strides(12,4)");
        assert_eq!(format!("{specification:?}"), "BufferSpecification[f32[2,3]strides(12,4)]");
    }

    #[test]
    fn test_host_buffer_data() {
        // Test [`HostBufferData::from_host_buffer`].
        let data = [1u8, 2u8, 3u8, 4u8];
        let host_buffer_data = HostBufferData::from_host_buffer(data.as_slice());
        assert_eq!(host_buffer_data.ptr, data.as_ptr() as *const std::ffi::c_void);
        assert!(host_buffer_data.drop_fn.is_none());

        // Test [`HostBufferData::from_host_buffer_rc_refcell`] with `mutable = false`.
        let data = &[7u8, 8u8, 9u8, 10u8];
        let rc = Rc::new(RefCell::new(data));
        assert_eq!(Rc::strong_count(&rc), 1);
        let mut host_buffer_data = HostBufferData::from_host_buffer_rc_refcell(&rc, false);
        assert_eq!(host_buffer_data.ptr, data.as_ptr() as *const std::ffi::c_void);
        assert_eq!(Rc::strong_count(&rc), 2);
        host_buffer_data.drop_fn.take().unwrap()();
        assert_eq!(Rc::strong_count(&rc), 1);

        // Test [`HostBufferData::from_host_buffer_rc_refcell`] with `mutable = true`.
        let data = &[11u8, 12u8, 13u8, 14u8];
        let rc = Rc::new(RefCell::new(data));
        assert_eq!(Rc::strong_count(&rc), 1);
        let mut host_buffer_data = HostBufferData::from_host_buffer_rc_refcell(&rc, true);
        assert_eq!(host_buffer_data.ptr, data.as_ptr() as *const std::ffi::c_void);
        assert_eq!(Rc::strong_count(&rc), 2);
        host_buffer_data.drop_fn.take().unwrap()();
        assert_eq!(Rc::strong_count(&rc), 1);
    }

    #[test]
    fn test_host_buffer() {
        // Test using a `&[u8]`.
        let value: &[u8] = &[1u8, 2u8, 3u8, 4u8];
        assert!(matches!(<&[u8] as HostBuffer>::host_buffer_semantics(), HostBufferSemantics::ImmutableOnlyDuringCall));
        let data = unsafe { <&[u8] as HostBuffer>::data(&value) };
        assert_eq!(data.ptr, value.as_ptr() as *const std::ffi::c_void);
        assert!(data.drop_fn.is_none());

        // Test using an `Rc<RefCell<&[u8]>>`.
        let value: &[u8] = &[1u8, 2u8, 3u8, 4u8];
        let rc = Rc::new(RefCell::new(value));
        assert!(matches!(
            <Rc<RefCell<&[u8]>> as HostBuffer>::host_buffer_semantics(),
            HostBufferSemantics::ImmutableUntilTransferCompletes,
        ));
        let mut data = unsafe { <Rc<RefCell<&[u8]>> as HostBuffer>::data(&rc) };
        assert_eq!(data.ptr, value.as_ptr() as *const std::ffi::c_void);
        data.drop_fn.take().unwrap()();

        // Test using a `&[u8; 4]`.
        let value = &[1u8, 2u8, 3u8, 4u8];
        assert!(matches!(
            <&[u8; 4] as HostBuffer>::host_buffer_semantics(),
            HostBufferSemantics::ImmutableOnlyDuringCall,
        ));
        let data = unsafe { <&[u8; 4] as HostBuffer>::data(&value) };
        assert_eq!(data.ptr, value.as_ptr() as *const std::ffi::c_void);
        assert!(data.drop_fn.is_none());

        // Test using an `Rc<RefCell<&[u8; 4]>>`.
        let value = &[1u8, 2u8, 3u8, 4u8];
        let rc = Rc::new(RefCell::new(value));
        assert!(matches!(
            <Rc<RefCell<&[u8; 4]>> as HostBuffer>::host_buffer_semantics(),
            HostBufferSemantics::ImmutableUntilTransferCompletes,
        ));
        let mut data = unsafe { <Rc<RefCell<&[u8; 4]>> as HostBuffer>::data(&rc) };
        assert_eq!(data.ptr, value.as_ptr() as *const std::ffi::c_void);
        data.drop_fn.take().unwrap()();
    }

    #[test]
    fn test_dma_mapped_buffer() {
        test_for_each_platform!(|_plugin, client, platform| {
            match platform {
                TestPlatform::Cpu => assert!(true),
                _ => {
                    let device = client.addressable_devices().unwrap().remove(0);
                    let mut data = [111u8, 112u8, 113u8, 114u8];
                    let dma_mapped_buffer = unsafe { client.dma_map(&mut data as *mut u8 as *mut _, 4) }.unwrap();
                    assert_eq!(dma_mapped_buffer.data(), data);
                    assert_eq!(dma_mapped_buffer.len(), data.len());
                    assert!(!dma_mapped_buffer.is_empty());
                    let buffer = unsafe {
                        dma_mapped_buffer.into_buffer(
                            BufferSpecification {
                                element_type: BufferType::U8,
                                dimensions: [data.len() as u64],
                                layout: None,
                            },
                            device.default_memory().unwrap(),
                        )
                    }
                    .unwrap();
                    assert_eq!(buffer.copy_to_host(None).unwrap().r#await(), Ok(data.to_vec()));
                }
            }
        });
    }

    #[test]
    fn test_client_buffer() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let data = [1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8];
            let buffer = client
                .buffer(data.as_slice(), BufferType::U8, [data.len() as u64], None, device.clone(), None)
                .unwrap();
            assert_eq!(buffer.element_type(), Ok(BufferType::U8));
            assert_eq!(buffer.dimensions(), Ok([data.len() as u64].as_slice()));
            assert_eq!(buffer.device().unwrap().id(), device.id());
        });
    }

    #[test]
    fn test_client_borrowed_buffer() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let data = vec![21u8, 22u8, 23u8, 24u8, 25u8, 26u8];
            let rc = Rc::new(RefCell::new(data.as_slice()));
            let buffer =
                client.borrowed_buffer(rc.clone(), BufferType::U8, [6u64], None, device.clone(), None).unwrap();
            assert_eq!(buffer.element_type(), Ok(BufferType::U8));
            assert_eq!(buffer.dimensions(), Ok([data.len() as u64].as_slice()));
            assert_eq!(buffer.device().unwrap().id(), device.id());
        });
    }

    #[test]
    fn test_client_borrowed_mut_buffer() {
        test_for_each_platform!(|_plugin, client, _platform| {
            let device = client.addressable_devices().unwrap().remove(0);
            let data = vec![31u8, 32u8, 33u8, 34u8];
            let rc = Rc::new(RefCell::new(data.as_slice()));
            let buffer =
                client.borrowed_mut_buffer(rc.clone(), BufferType::U8, [4u64], None, device.clone(), None).unwrap();
            assert_eq!(buffer.element_type(), Ok(BufferType::U8));
            assert_eq!(buffer.dimensions(), Ok([data.len() as u64].as_slice()));
            assert_eq!(buffer.device().unwrap().id(), device.id());
        });
    }

    #[test]
    fn test_client_uninitialized_buffer() {
        let client = test_cpu_client();
        let device = client.addressable_devices().unwrap()[0].clone();
        let specification = BufferSpecification { element_type: BufferType::U8, dimensions: [4u64], layout: None };
        let buffer = client.uninitialized_buffer(specification, device.clone()).unwrap();
        assert_eq!(buffer.element_type(), Ok(BufferType::U8));
        assert_eq!(buffer.dimensions(), Ok([4u64].as_slice()));
        assert_eq!(buffer.dynamic_dimensions(), Ok([].as_slice()));
        assert_eq!(buffer.device().unwrap().id(), device.id());
    }

    #[test]
    fn test_client_borrowed_on_device_buffer() {
        let client = test_cpu_client();
        let device = client.addressable_devices().unwrap()[0].clone();
        let buffer = client.buffer(&[1u8, 2u8, 3u8, 4u8], BufferType::U8, [4u64], None, device.clone(), None).unwrap();
        let specification = BufferSpecification {
            element_type: buffer.element_type().unwrap(),
            dimensions: buffer.dimensions().unwrap().to_vec(),
            layout: None,
        };
        let borrowed_buffer = unsafe {
            client
                .borrowed_on_device_buffer(buffer.as_ptr().unwrap(), || {}, specification, device, None)
                .unwrap()
        };
        assert!(borrowed_buffer.ready().unwrap().r#await().is_ok());
        assert_eq!(borrowed_buffer.element_type(), Ok(BufferType::U8));
        assert_eq!(borrowed_buffer.dimensions(), Ok([4u64].as_slice()));
        assert_eq!(borrowed_buffer.copy_to_host(None).unwrap().r#await().unwrap(), vec![1u8, 2u8, 3u8, 4u8]);
    }

    #[test]
    fn test_client_error_buffer() {
        let client = test_cpu_client();
        let device = client.addressable_devices().unwrap()[0].clone();
        let error = Error::aborted("test error");
        let specification = BufferSpecification { element_type: BufferType::U8, dimensions: [4u64], layout: None };
        let buffer = client.error_buffer(error.clone(), specification, device).unwrap();
        assert!(matches!(
            buffer.ready().unwrap().r#await(),
            Err(Error::Aborted { message, .. }) if message.contains("test error"),
        ));
    }

    #[test]
    fn test_client_alias_buffer_and_fulfillment() {
        let client = test_cpu_client();
        let device = client.addressable_devices().unwrap()[0].clone();
        let specification = BufferSpecification { element_type: BufferType::U8, dimensions: [4u64], layout: None };

        // Create a new alias buffer and fulfill it with some other buffer.
        let (alias_buffer, token) = client.alias_buffer(specification.clone(), device.clone()).unwrap();
        let buffer = client.buffer(&[1u8, 2u8, 3u8, 4u8], BufferType::U8, [4u64], None, device.clone(), None).unwrap();
        assert!(client.fulfill_alias_buffer(token, buffer).is_ok());
        assert_eq!(alias_buffer.ready().unwrap().r#await(), Ok(()));
        assert_eq!(alias_buffer.dimensions(), Ok([4u64].as_slice()));
        assert_eq!(alias_buffer.copy_to_host(None).unwrap().r#await().unwrap(), vec![1u8, 2u8, 3u8, 4u8]);

        // Create a new alias buffer and fulfill it with an error.
        let (alias_buffer, token) = client.alias_buffer(specification, device.clone()).unwrap();
        let error = Error::aborted("test error");
        match client.fulfill_alias_buffer_with_error(token, error.clone()) {
            Ok(_) => {
                assert!(matches!(
                    alias_buffer.ready().unwrap().r#await(),
                    Err(Error::Aborted { message, .. }) if message.contains("test error"),
                ));
            }
            Err(Error::Aborted { message, .. }) if message.contains("test error") => assert!(true),
            _ => assert!(false),
        }
    }
}
