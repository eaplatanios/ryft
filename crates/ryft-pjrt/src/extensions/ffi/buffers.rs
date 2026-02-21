use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use crate::extensions::ffi::errors::FfiError;
use crate::slice_from_c_api;

/// Type of the data stored in an [`FfiBuffer`]. Specifically, this represents
/// the type of individual values that are stored in [`FfiBuffer`]s.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FfiBufferType {
    /// Invalid [`FfiBufferType`] that serves as a default.
    Invalid,

    /// [`FfiBufferType`] that represents token values that are threaded between side-effecting operations.
    /// This type is only used for buffers that contain a single value (i.e., that represent scalar values).
    Token,

    /// Predicate [`FfiBufferType`] that represents the `true` and `false` values.
    Predicate,

    /// [`FfiBufferType`] that represents signed 1-bit integer values. In XLA this corresponds to `S1`/`int1`,
    /// with representable values `0` and `-1`. [`FfiBuffer`] storage for this [`FfiBufferType`] is unpacked and
    /// byte-backed (one logical element per byte).
    I1,

    /// [`FfiBufferType`] that represents signed 2-bit integer values.
    I2,

    /// [`FfiBufferType`] that represents signed 4-bit integer values.
    I4,

    /// [`FfiBufferType`] that represents signed 8-bit integer values.
    I8,

    /// [`FfiBufferType`] that represents signed 16-bit integer values.
    I16,

    /// [`FfiBufferType`] that represents signed 32-bit integer values.
    I32,

    /// [`FfiBufferType`] that represents signed 64-bit integer values.
    I64,

    /// [`FfiBufferType`] that represents unsigned 1-bit integer values.
    U1,

    /// [`FfiBufferType`] that represents unsigned 2-bit integer values.
    U2,

    /// [`FfiBufferType`] that represents unsigned 4-bit integer values.
    U4,

    /// [`FfiBufferType`] that represents unsigned 8-bit integer values.
    U8,

    /// [`FfiBufferType`] that represents unsigned 16-bit integer values.
    U16,

    /// [`FfiBufferType`] that represents unsigned 32-bit integer values.
    U32,

    /// [`FfiBufferType`] that represents unsigned 64-bit integer values.
    U64,

    /// [`FfiBufferType`] that represents 4-bit floating-point values that are represented using a
    /// [microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
    /// format with 2 exponent bits and 1 mantissa bit. Only finite values are supported (thus the `FN` suffix).
    /// Unlike IEEE floating-point types, infinities and NaN values are not supported.
    F4E2M1FN,

    /// [`FfiBufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 3 exponent bits and 4 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E3M4,

    /// [`FfiBufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E4M3,

    /// [`FfiBufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits, and without support for
    /// representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values are
    /// represented with the exponent and mantissa bits all set to `1`. All other bit configurations represent finite
    /// values.
    F8E4M3FN,

    /// [`FfiBufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2206.02915) with 4 exponent bits and 3 mantissa bits, and without support for
    /// representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values are
    /// represented with the exponent and mantissa bits all set to `0` and the sign bit is set to `1`. All other bit
    /// configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    ///
    /// The difference between this type and [`FfiBufferType::F8E4M3FN`] is that there is an additional exponent value
    /// available. To keep the same dynamic range as an IEEE-like 8-bit floating-point type, the exponent is biased one
    /// more than would be expected given the number of exponent bits (i.e., bias set to `8`).
    F8E4M3FNUZ,

    /// [`FfiBufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits and a bias of `11`, and
    /// without support for representing infinity values, unlike existing IEEE floating-point types (thus the `FN`
    /// suffix). NaN values are represented with the exponent and mantissa bits all set to `0` and the sign bit is set
    /// to `1`. All other bit configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    F8E4M3B11FNUZ,

    /// [`FfiBufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 5 exponent bits and 2 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E5M2,

    /// [`FfiBufferType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2206.02915) with 5 exponent bits and 2 mantissa bits, and without support for
    /// representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values are
    /// represented with the exponent and mantissa bits all set to `0` and the sign bit is set to `1`. All other bit
    /// configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    ///
    /// The difference between this type and [`FfiBufferType::F8E5M2`] is that there is an additional exponent value
    /// available. To keep the same dynamic range as an IEEE-like 8-bit floating-point type, the exponent is biased one
    /// more than would be expected given the number of exponent bits (i.e., bias set to `16`).
    F8E5M2FNUZ,

    /// [`FfiBufferType`] that represents 8-bit floating-point values that are represented using a
    /// [microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
    /// format with 8 exponent bits and no mantissa or sign bits. Only unsigned finite values are supported
    /// (thus the `FNU` suffix). Unlike IEEE floating-point types, infinity and NaN values are not supported.
    F8E8M0FNU,

    /// [`FfiBufferType`] that represents 16-bit floating-point values with 8 exponent bits, 7 mantissa bits, and 1 sign
    /// bit. This type offers a larger dynamic range than [`FfiBufferType::F16`] at the cost of lower precision.
    BF16,

    /// [`FfiBufferType`] that represents 16-bit floating-point values with 5 exponent bits, 10 mantissa bits, and 1 sign
    /// bit, using the standard IEEE floating-point representation.
    F16,

    /// [`FfiBufferType`] that represents 32-bit floating-point values with 8 exponent bits, 24 mantissa bits, and 1 sign
    /// bit, using the standard IEEE floating-point representation.
    F32,

    /// [`FfiBufferType`] that represents 64-bit floating-point values with 11 exponent bits, 53 mantissa bits, and 1 sign
    /// bit, using the standard IEEE floating-point representation.
    F64,

    /// [`FfiBufferType`] that represents 64-bit complex-valued floating-point values as pairs of
    /// 32-bit real floating-point values.
    C64,

    /// [`FfiBufferType`] that represents 128-bit complex-valued floating-point values as pairs of
    /// 64-bit real floating-point values.
    C128,
}

impl FfiBufferType {
    /// Constructs a new [`FfiBufferType`] from the provided [`XLA_FFI_DataType`](ffi::XLA_FFI_DataType)
    /// that came from a function in the XLA FFI API.
    pub unsafe fn from_c_api(r#type: ffi::XLA_FFI_DataType) -> Self {
        match r#type {
            ffi::XLA_FFI_DataType_TOKEN => Self::Token,
            ffi::XLA_FFI_DataType_PRED => Self::Predicate,
            ffi::XLA_FFI_DataType_S1 => Self::I1,
            ffi::XLA_FFI_DataType_S2 => Self::I2,
            ffi::XLA_FFI_DataType_S4 => Self::I4,
            ffi::XLA_FFI_DataType_S8 => Self::I8,
            ffi::XLA_FFI_DataType_S16 => Self::I16,
            ffi::XLA_FFI_DataType_S32 => Self::I32,
            ffi::XLA_FFI_DataType_S64 => Self::I64,
            ffi::XLA_FFI_DataType_U1 => Self::U1,
            ffi::XLA_FFI_DataType_U2 => Self::U2,
            ffi::XLA_FFI_DataType_U4 => Self::U4,
            ffi::XLA_FFI_DataType_U8 => Self::U8,
            ffi::XLA_FFI_DataType_U16 => Self::U16,
            ffi::XLA_FFI_DataType_U32 => Self::U32,
            ffi::XLA_FFI_DataType_U64 => Self::U64,
            ffi::XLA_FFI_DataType_F4E2M1FN => Self::F4E2M1FN,
            ffi::XLA_FFI_DataType_F8E3M4 => Self::F8E3M4,
            ffi::XLA_FFI_DataType_F8E4M3 => Self::F8E4M3,
            ffi::XLA_FFI_DataType_F8E4M3FN => Self::F8E4M3FN,
            ffi::XLA_FFI_DataType_F8E4M3FNUZ => Self::F8E4M3FNUZ,
            ffi::XLA_FFI_DataType_F8E4M3B11FNUZ => Self::F8E4M3B11FNUZ,
            ffi::XLA_FFI_DataType_F8E5M2 => Self::F8E5M2,
            ffi::XLA_FFI_DataType_F8E5M2FNUZ => Self::F8E5M2FNUZ,
            ffi::XLA_FFI_DataType_F8E8M0FNU => Self::F8E8M0FNU,
            ffi::XLA_FFI_DataType_BF16 => Self::BF16,
            ffi::XLA_FFI_DataType_F16 => Self::F16,
            ffi::XLA_FFI_DataType_F32 => Self::F32,
            ffi::XLA_FFI_DataType_F64 => Self::F64,
            ffi::XLA_FFI_DataType_C64 => Self::C64,
            ffi::XLA_FFI_DataType_C128 => Self::C128,
            _ => Self::Invalid,
        }
    }

    /// Returns the [`XLA_FFI_DataType`](ffi::XLA_FFI_DataType) that corresponds to this [`FfiBufferType`]
    /// and which can be passed to functions in the XLA FFI API.
    pub unsafe fn to_c_api(&self) -> ffi::XLA_FFI_DataType {
        match self {
            Self::Invalid => ffi::XLA_FFI_DataType_INVALID,
            Self::Token => ffi::XLA_FFI_DataType_TOKEN,
            Self::Predicate => ffi::XLA_FFI_DataType_PRED,
            Self::I1 => ffi::XLA_FFI_DataType_S1,
            Self::I2 => ffi::XLA_FFI_DataType_S2,
            Self::I4 => ffi::XLA_FFI_DataType_S4,
            Self::I8 => ffi::XLA_FFI_DataType_S8,
            Self::I16 => ffi::XLA_FFI_DataType_S16,
            Self::I32 => ffi::XLA_FFI_DataType_S32,
            Self::I64 => ffi::XLA_FFI_DataType_S64,
            Self::U1 => ffi::XLA_FFI_DataType_U1,
            Self::U2 => ffi::XLA_FFI_DataType_U2,
            Self::U4 => ffi::XLA_FFI_DataType_U4,
            Self::U8 => ffi::XLA_FFI_DataType_U8,
            Self::U16 => ffi::XLA_FFI_DataType_U16,
            Self::U32 => ffi::XLA_FFI_DataType_U32,
            Self::U64 => ffi::XLA_FFI_DataType_U64,
            Self::F4E2M1FN => ffi::XLA_FFI_DataType_F4E2M1FN,
            Self::F8E3M4 => ffi::XLA_FFI_DataType_F8E3M4,
            Self::F8E4M3 => ffi::XLA_FFI_DataType_F8E4M3,
            Self::F8E4M3FN => ffi::XLA_FFI_DataType_F8E4M3FN,
            Self::F8E4M3FNUZ => ffi::XLA_FFI_DataType_F8E4M3FNUZ,
            Self::F8E4M3B11FNUZ => ffi::XLA_FFI_DataType_F8E4M3B11FNUZ,
            Self::F8E5M2 => ffi::XLA_FFI_DataType_F8E5M2,
            Self::F8E5M2FNUZ => ffi::XLA_FFI_DataType_F8E5M2FNUZ,
            Self::F8E8M0FNU => ffi::XLA_FFI_DataType_F8E8M0FNU,
            Self::BF16 => ffi::XLA_FFI_DataType_BF16,
            Self::F16 => ffi::XLA_FFI_DataType_F16,
            Self::F32 => ffi::XLA_FFI_DataType_F32,
            Self::F64 => ffi::XLA_FFI_DataType_F64,
            Self::C64 => ffi::XLA_FFI_DataType_C64,
            Self::C128 => ffi::XLA_FFI_DataType_C128,
        }
    }

    /// Parses a rendered [`FfiBufferType`] (e.g., an XLA primitive type string) into a [`FfiBufferType`].
    pub fn from_str<S: AsRef<str>>(value: S) -> Result<Self, FfiError> {
        let value = value.as_ref();
        match value.trim().to_ascii_lowercase().as_str() {
            "invalid" => Ok(Self::Invalid),
            "pred" => Ok(Self::Predicate),
            "token" => Ok(Self::Token),
            "s1" | "i1" => Ok(Self::I1),
            "s2" | "i2" => Ok(Self::I2),
            "s4" | "i4" => Ok(Self::I4),
            "s8" | "i8" => Ok(Self::I8),
            "s16" | "i16" => Ok(Self::I16),
            "s32" | "i32" => Ok(Self::I32),
            "s64" | "i64" => Ok(Self::I64),
            "u1" => Ok(Self::U1),
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
            _ => Err(FfiError::invalid_argument(format!("invalid XLA FFI data type '{value}'"))),
        }
    }

    /// Constructs a [`FfiBufferType`] from the provided [`BufferType`](crate::protos::BufferType) Protobuf.
    pub fn from_proto(r#type: crate::protos::BufferType) -> Self {
        match r#type {
            crate::protos::BufferType::Invalid => Self::Invalid,
            crate::protos::BufferType::Token => Self::Token,
            crate::protos::BufferType::Predicate => Self::Predicate,
            crate::protos::BufferType::I1 => Self::I1,
            crate::protos::BufferType::I2 => Self::I2,
            crate::protos::BufferType::I4 => Self::I4,
            crate::protos::BufferType::I8 => Self::I8,
            crate::protos::BufferType::I16 => Self::I16,
            crate::protos::BufferType::I32 => Self::I32,
            crate::protos::BufferType::I64 => Self::I64,
            crate::protos::BufferType::U1 => Self::U1,
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

    /// Returns the [`BufferType`](crate::protos::BufferType) Protobuf that corresponds to this [`FfiBufferType`].
    pub fn proto(&self) -> crate::protos::BufferType {
        match self {
            Self::Invalid => crate::protos::BufferType::Invalid,
            Self::Token => crate::protos::BufferType::Token,
            Self::Predicate => crate::protos::BufferType::Predicate,
            Self::I1 => crate::protos::BufferType::I1,
            Self::I2 => crate::protos::BufferType::I2,
            Self::I4 => crate::protos::BufferType::I4,
            Self::I8 => crate::protos::BufferType::I8,
            Self::I16 => crate::protos::BufferType::I16,
            Self::I32 => crate::protos::BufferType::I32,
            Self::I64 => crate::protos::BufferType::I64,
            Self::U1 => crate::protos::BufferType::U1,
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

impl Display for FfiBufferType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Invalid => "invalid",
            Self::Token => "token",
            Self::Predicate => "pred",
            Self::I1 => "i1",
            Self::I2 => "i2",
            Self::I4 => "i4",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U1 => "u1",
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

/// Represents an input/argument or output/result buffer in an [`FfiCallFrame`](crate::extensions::ffi::FfiCallFrame).
/// XLA FFI buffers expose only a subset of the higher-level PJRT [`Buffer`](crate::Buffer) functionality.
/// In particular, they provide a borrowed view over raw input/argument and output/result buffers for the duration
/// of a single [`FfiHandler`](crate::extensions::ffi::FfiHandler) invocation.
pub struct FfiBuffer<'o> {
    /// Handle that represents this [`FfiBuffer`] in the XLA FFI API.
    handle: *const ffi::XLA_FFI_Buffer,

    /// [`PhantomData`] used to track the lifetime of the owner of this [`FfiBuffer`].
    owner: PhantomData<&'o ()>,
}

impl<'o> FfiBuffer<'o> {
    /// Constructs a new [`FfiBuffer`] from the provided [`XLA_FFI_Buffer`](ffi::XLA_FFI_Buffer) handle that came
    /// from a function in the XLA FFI API.
    pub unsafe fn from_c_api(handle: *const ffi::XLA_FFI_Buffer) -> Result<Self, FfiError> {
        if handle.is_null() {
            Err(FfiError::invalid_argument("the provided XLA FFI buffer handle is a null pointer"))
        } else {
            Ok(Self { handle, owner: PhantomData })
        }
    }

    /// Returns the [`XLA_FFI_Buffer`](ffi::XLA_FFI_Buffer) that corresponds to this [`FfiBuffer`] and which can
    /// be passed to functions in the XLA FFI API.
    pub unsafe fn to_c_api(&self) -> *const ffi::XLA_FFI_Buffer {
        self.handle
    }

    /// Returns the [`FfiBufferType`] of the elements stored in this [`FfiBuffer`].
    pub fn element_type(&self) -> FfiBufferType {
        unsafe { FfiBufferType::from_c_api((*self.handle).data_type) }
    }

    /// Returns the rank (i.e., the number of dimensions) of this [`FfiBuffer`].
    pub fn rank(&self) -> usize {
        unsafe { (*self.handle).rank as usize }
    }

    /// Returns the dimension sizes of this [`FfiBuffer`].
    pub fn dimensions(&self) -> &'o [i64] {
        unsafe { slice_from_c_api((*self.handle).dimensions, self.rank()) }
    }

    /// Returns a pointer to the underlying data of this [`FfiBuffer`].
    pub unsafe fn data(&self) -> *mut std::ffi::c_void {
        unsafe { (*self.handle).data }
    }
}

#[allow(dead_code, non_camel_case_types, non_snake_case, non_upper_case_globals)]
pub(crate) mod ffi {
    pub type XLA_FFI_DataType = std::ffi::c_uint;
    pub const XLA_FFI_DataType_INVALID: XLA_FFI_DataType = 0;
    pub const XLA_FFI_DataType_PRED: XLA_FFI_DataType = 1;
    pub const XLA_FFI_DataType_S1: XLA_FFI_DataType = 30;
    pub const XLA_FFI_DataType_S2: XLA_FFI_DataType = 26;
    pub const XLA_FFI_DataType_S4: XLA_FFI_DataType = 21;
    pub const XLA_FFI_DataType_S8: XLA_FFI_DataType = 2;
    pub const XLA_FFI_DataType_S16: XLA_FFI_DataType = 3;
    pub const XLA_FFI_DataType_S32: XLA_FFI_DataType = 4;
    pub const XLA_FFI_DataType_S64: XLA_FFI_DataType = 5;
    pub const XLA_FFI_DataType_U1: XLA_FFI_DataType = 31;
    pub const XLA_FFI_DataType_U2: XLA_FFI_DataType = 27;
    pub const XLA_FFI_DataType_U4: XLA_FFI_DataType = 22;
    pub const XLA_FFI_DataType_U8: XLA_FFI_DataType = 6;
    pub const XLA_FFI_DataType_U16: XLA_FFI_DataType = 7;
    pub const XLA_FFI_DataType_U32: XLA_FFI_DataType = 8;
    pub const XLA_FFI_DataType_U64: XLA_FFI_DataType = 9;
    pub const XLA_FFI_DataType_F16: XLA_FFI_DataType = 10;
    pub const XLA_FFI_DataType_F32: XLA_FFI_DataType = 11;
    pub const XLA_FFI_DataType_F64: XLA_FFI_DataType = 12;
    pub const XLA_FFI_DataType_BF16: XLA_FFI_DataType = 16;
    pub const XLA_FFI_DataType_C64: XLA_FFI_DataType = 15;
    pub const XLA_FFI_DataType_C128: XLA_FFI_DataType = 18;
    pub const XLA_FFI_DataType_TOKEN: XLA_FFI_DataType = 17;
    pub const XLA_FFI_DataType_F8E5M2: XLA_FFI_DataType = 19;
    pub const XLA_FFI_DataType_F8E3M4: XLA_FFI_DataType = 29;
    pub const XLA_FFI_DataType_F8E4M3: XLA_FFI_DataType = 28;
    pub const XLA_FFI_DataType_F8E4M3FN: XLA_FFI_DataType = 20;
    pub const XLA_FFI_DataType_F8E4M3B11FNUZ: XLA_FFI_DataType = 23;
    pub const XLA_FFI_DataType_F8E5M2FNUZ: XLA_FFI_DataType = 24;
    pub const XLA_FFI_DataType_F8E4M3FNUZ: XLA_FFI_DataType = 25;
    pub const XLA_FFI_DataType_F4E2M1FN: XLA_FFI_DataType = 32;
    pub const XLA_FFI_DataType_F8E8M0FNU: XLA_FFI_DataType = 33;

    #[repr(C)]
    pub struct XLA_FFI_Buffer {
        pub struct_size: usize,
        pub extension_start: *mut crate::extensions::ffi::ffi::XLA_FFI_Extension_Base,
        pub data_type: XLA_FFI_DataType,
        pub data: *mut std::ffi::c_void,
        pub rank: i64,
        pub dimensions: *const i64,
    }
}

#[cfg(test)]
mod tests {
    use super::{FfiBuffer, FfiBufferType, ffi};

    #[test]
    fn test_ffi_buffer_type() {
        let types = [
            FfiBufferType::Invalid,
            FfiBufferType::Token,
            FfiBufferType::Predicate,
            FfiBufferType::I1,
            FfiBufferType::I2,
            FfiBufferType::I4,
            FfiBufferType::I8,
            FfiBufferType::I16,
            FfiBufferType::I32,
            FfiBufferType::I64,
            FfiBufferType::U1,
            FfiBufferType::U2,
            FfiBufferType::U4,
            FfiBufferType::U8,
            FfiBufferType::U16,
            FfiBufferType::U32,
            FfiBufferType::U64,
            FfiBufferType::F4E2M1FN,
            FfiBufferType::F8E3M4,
            FfiBufferType::F8E4M3,
            FfiBufferType::F8E4M3FN,
            FfiBufferType::F8E4M3FNUZ,
            FfiBufferType::F8E4M3B11FNUZ,
            FfiBufferType::F8E5M2,
            FfiBufferType::F8E5M2FNUZ,
            FfiBufferType::F8E8M0FNU,
            FfiBufferType::BF16,
            FfiBufferType::F16,
            FfiBufferType::F32,
            FfiBufferType::F64,
            FfiBufferType::C64,
            FfiBufferType::C128,
        ];

        types.iter().copied().for_each(|r#type| unsafe {
            assert_eq!(FfiBufferType::from_c_api(r#type.to_c_api()), r#type);
            assert_eq!(FfiBufferType::from_proto(r#type.proto()), r#type);
            assert_eq!(FfiBufferType::from_str(r#type.to_string()), Ok(r#type));
        });
        assert_eq!(unsafe { FfiBufferType::from_c_api(u32::MAX) }, FfiBufferType::Invalid);

        assert_eq!(format!("{}", FfiBufferType::F64), "f64".to_string());
        assert_eq!(format!("{:?}", FfiBufferType::F64), "F64".to_string());
    }

    #[test]
    fn test_ffi_buffer() {
        // Test using a three-dimensional buffer.
        let buffer = ffi::XLA_FFI_Buffer {
            struct_size: size_of::<ffi::XLA_FFI_Buffer>(),
            extension_start: std::ptr::null_mut(),
            data_type: ffi::XLA_FFI_DataType_F32,
            data: &[0u8; 24] as *const u8 as *mut _,
            rank: 3,
            dimensions: &[2i64, 3i64, 4i64] as *const _,
        };
        let buffer = unsafe { FfiBuffer::from_c_api(&buffer as *const _) }.unwrap();
        assert_eq!(buffer.element_type(), FfiBufferType::F32);
        assert_eq!(buffer.rank(), 3);
        assert_eq!(buffer.dimensions(), &[2, 3, 4]);
        assert!(unsafe { !buffer.data().is_null() });

        // Test using a scalar.
        let buffer = ffi::XLA_FFI_Buffer {
            struct_size: size_of::<ffi::XLA_FFI_Buffer>(),
            extension_start: std::ptr::null_mut(),
            data_type: ffi::XLA_FFI_DataType_F64,
            data: &[0u8; 24] as *const u8 as *mut _,
            rank: 0,
            dimensions: &[] as *const _,
        };
        let buffer = unsafe { FfiBuffer::from_c_api(&buffer as *const _) }.unwrap();
        assert_eq!(buffer.element_type(), FfiBufferType::F64);
        assert_eq!(buffer.rank(), 0);
        assert_eq!(buffer.dimensions(), &[] as &[i64]);
    }
}
