use std::fmt::Display;

pub use crate::types_v0::*;

/// Type of the data stored in an array or scalar value. Specifically, this represents
/// the type of individual values that are stored in that value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElementType {
    /// Invalid [`ElementType`] that serves as a default.
    Invalid,

    /// [`ElementType`] that represents token values that are threaded between side-effecting operations.
    /// This type is only used for values that contain a single token (i.e., that represent scalar values).
    Token,

    /// Predicate [`ElementType`] that represents the `true` and `false` values.
    Predicate,

    /// [`ElementType`] that represents signed 1-bit integer values, with the only representable values being
    /// `0` and `-1`.
    I1,

    /// [`ElementType`] that represents signed 2-bit integer values.
    I2,

    /// [`ElementType`] that represents signed 4-bit integer values.
    I4,

    /// [`ElementType`] that represents signed 8-bit integer values.
    I8,

    /// [`ElementType`] that represents signed 16-bit integer values.
    I16,

    /// [`ElementType`] that represents signed 32-bit integer values.
    I32,

    /// [`ElementType`] that represents signed 64-bit integer values.
    I64,

    /// [`ElementType`] that represents unsigned 1-bit integer values.
    U1,

    /// [`ElementType`] that represents unsigned 2-bit integer values.
    U2,

    /// [`ElementType`] that represents unsigned 4-bit integer values.
    U4,

    /// [`ElementType`] that represents unsigned 8-bit integer values.
    U8,

    /// [`ElementType`] that represents unsigned 16-bit integer values.
    U16,

    /// [`ElementType`] that represents unsigned 32-bit integer values.
    U32,

    /// [`ElementType`] that represents unsigned 64-bit integer values.
    U64,

    /// [`ElementType`] that represents 4-bit floating-point values that are represented using a
    /// [microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
    /// format with 2 exponent bits and 1 mantissa bit. Only finite values are supported (thus the `FN` suffix).
    /// Unlike IEEE floating-point types, infinities and NaN values are not supported.
    F4E2M1FN,

    /// [`ElementType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 3 exponent bits and 4 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E3M4,

    /// [`ElementType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E4M3,

    /// [`ElementType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits, and without support
    /// for representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values
    /// are represented with the exponent and mantissa bits all set to `1`. All other bit configurations represent
    /// finite values.
    F8E4M3FN,

    /// [`ElementType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2206.02915) with 4 exponent bits and 3 mantissa bits, and without support
    /// for representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values
    /// are represented with the exponent and mantissa bits all set to `0` and the sign bit is set to `1`. All other
    /// bit configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    ///
    /// The difference between this type and [`ElementType::F8E4M3FN`] is that there is an additional exponent value
    /// available. To keep the same dynamic range as an IEEE-like 8-bit floating-point type, the exponent is biased one
    /// more than would be expected given the number of exponent bits (i.e., bias set to `8`).
    F8E4M3FNUZ,

    /// [`ElementType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 4 exponent bits and 3 mantissa bits and a bias of `11`,
    /// and without support for representing infinity values, unlike existing IEEE floating-point types (thus the `FN`
    /// suffix). NaN values are represented with the exponent and mantissa bits all set to `0` and the sign bit is set
    /// to `1`. All other bit configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    F8E4M3B11FNUZ,

    /// [`ElementType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2209.05433) with 5 exponent bits and 2 mantissa bits, and with support for
    /// representing infinity and NaN values similar to existing IEEE floating-point types.
    F8E5M2,

    /// [`ElementType`] that represents 8-bit floating-point values that are represented using the format described in
    /// [this paper](https://arxiv.org/abs/2206.02915) with 5 exponent bits and 2 mantissa bits, and without support
    /// for representing infinity values, unlike existing IEEE floating-point types (thus the `FN` suffix). NaN values
    /// are represented with the exponent and mantissa bits all set to `0` and the sign bit is set to `1`. All other
    /// bit configurations represent finite values. Zero values are unsigned (thus the `UZ` suffix).
    ///
    /// The difference between this type and [`ElementType::F8E5M2`] is that there is an additional exponent value
    /// available. To keep the same dynamic range as an IEEE-like 8-bit floating-point type, the exponent is biased one
    /// more than would be expected given the number of exponent bits (i.e., bias set to `16`).
    F8E5M2FNUZ,

    /// [`ElementType`] that represents 8-bit floating-point values that are represented using a
    /// [microscaling](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
    /// format with 8 exponent bits and no mantissa or sign bits. Only unsigned finite values are supported
    /// (thus the `FNU` suffix). Unlike IEEE floating-point types, infinity and NaN values are not supported.
    F8E8M0FNU,

    /// [`ElementType`] that represents 16-bit floating-point values with 8 exponent bits, 7 mantissa bits, and 1 sign
    /// bit. This type offers a larger dynamic range than [`ElementType::F16`] at the cost of lower precision.
    BF16,

    /// [`ElementType`] that represents 16-bit floating-point values with 5 exponent bits, 10 mantissa bits, and 1
    /// sign bit, using the standard IEEE floating-point representation.
    F16,

    /// [`ElementType`] that represents 32-bit floating-point values with 8 exponent bits, 24 mantissa bits, and 1
    /// sign bit, using the standard IEEE floating-point representation.
    F32,

    /// [`ElementType`] that represents 64-bit floating-point values with 11 exponent bits, 53 mantissa bits, and 1
    /// sign bit, using the standard IEEE floating-point representation.
    F64,

    /// [`ElementType`] that represents 64-bit complex-valued floating-point values as pairs of
    /// 32-bit real floating-point values.
    C64,

    /// [`ElementType`] that represents 128-bit complex-valued floating-point values as pairs of
    /// 64-bit real floating-point values.
    C128,
}

impl Display for ElementType {
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

#[cfg(feature = "xla")]
impl ElementType {
    /// Creates an [`ElementType`] from the corresponding PJRT buffer type.
    pub fn from_buffer_type(element_type: ryft_pjrt::BufferType) -> Self {
        element_type.into()
    }

    /// Returns the corresponding PJRT buffer type.
    pub fn to_buffer_type(self) -> ryft_pjrt::BufferType {
        self.into()
    }
}

#[cfg(feature = "xla")]
impl From<ryft_pjrt::BufferType> for ElementType {
    fn from(element_type: ryft_pjrt::BufferType) -> Self {
        match element_type {
            ryft_pjrt::BufferType::Invalid => Self::Invalid,
            ryft_pjrt::BufferType::Token => Self::Token,
            ryft_pjrt::BufferType::Predicate => Self::Predicate,
            ryft_pjrt::BufferType::I1 => Self::I1,
            ryft_pjrt::BufferType::I2 => Self::I2,
            ryft_pjrt::BufferType::I4 => Self::I4,
            ryft_pjrt::BufferType::I8 => Self::I8,
            ryft_pjrt::BufferType::I16 => Self::I16,
            ryft_pjrt::BufferType::I32 => Self::I32,
            ryft_pjrt::BufferType::I64 => Self::I64,
            ryft_pjrt::BufferType::U1 => Self::U1,
            ryft_pjrt::BufferType::U2 => Self::U2,
            ryft_pjrt::BufferType::U4 => Self::U4,
            ryft_pjrt::BufferType::U8 => Self::U8,
            ryft_pjrt::BufferType::U16 => Self::U16,
            ryft_pjrt::BufferType::U32 => Self::U32,
            ryft_pjrt::BufferType::U64 => Self::U64,
            ryft_pjrt::BufferType::F4E2M1FN => Self::F4E2M1FN,
            ryft_pjrt::BufferType::F8E3M4 => Self::F8E3M4,
            ryft_pjrt::BufferType::F8E4M3 => Self::F8E4M3,
            ryft_pjrt::BufferType::F8E4M3FN => Self::F8E4M3FN,
            ryft_pjrt::BufferType::F8E4M3FNUZ => Self::F8E4M3FNUZ,
            ryft_pjrt::BufferType::F8E4M3B11FNUZ => Self::F8E4M3B11FNUZ,
            ryft_pjrt::BufferType::F8E5M2 => Self::F8E5M2,
            ryft_pjrt::BufferType::F8E5M2FNUZ => Self::F8E5M2FNUZ,
            ryft_pjrt::BufferType::F8E8M0FNU => Self::F8E8M0FNU,
            ryft_pjrt::BufferType::BF16 => Self::BF16,
            ryft_pjrt::BufferType::F16 => Self::F16,
            ryft_pjrt::BufferType::F32 => Self::F32,
            ryft_pjrt::BufferType::F64 => Self::F64,
            ryft_pjrt::BufferType::C64 => Self::C64,
            ryft_pjrt::BufferType::C128 => Self::C128,
        }
    }
}

#[cfg(feature = "xla")]
impl From<ElementType> for ryft_pjrt::BufferType {
    fn from(element_type: ElementType) -> Self {
        match element_type {
            ElementType::Invalid => Self::Invalid,
            ElementType::Token => Self::Token,
            ElementType::Predicate => Self::Predicate,
            ElementType::I1 => Self::I1,
            ElementType::I2 => Self::I2,
            ElementType::I4 => Self::I4,
            ElementType::I8 => Self::I8,
            ElementType::I16 => Self::I16,
            ElementType::I32 => Self::I32,
            ElementType::I64 => Self::I64,
            ElementType::U1 => Self::U1,
            ElementType::U2 => Self::U2,
            ElementType::U4 => Self::U4,
            ElementType::U8 => Self::U8,
            ElementType::U16 => Self::U16,
            ElementType::U32 => Self::U32,
            ElementType::U64 => Self::U64,
            ElementType::F4E2M1FN => Self::F4E2M1FN,
            ElementType::F8E3M4 => Self::F8E3M4,
            ElementType::F8E4M3 => Self::F8E4M3,
            ElementType::F8E4M3FN => Self::F8E4M3FN,
            ElementType::F8E4M3FNUZ => Self::F8E4M3FNUZ,
            ElementType::F8E4M3B11FNUZ => Self::F8E4M3B11FNUZ,
            ElementType::F8E5M2 => Self::F8E5M2,
            ElementType::F8E5M2FNUZ => Self::F8E5M2FNUZ,
            ElementType::F8E8M0FNU => Self::F8E8M0FNU,
            ElementType::BF16 => Self::BF16,
            ElementType::F16 => Self::F16,
            ElementType::F32 => Self::F32,
            ElementType::F64 => Self::F64,
            ElementType::C64 => Self::C64,
            ElementType::C128 => Self::C128,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ElementType;

    const DISPLAY_CASES: &[(ElementType, &str)] = &[
        (ElementType::Invalid, "invalid"),
        (ElementType::Token, "token"),
        (ElementType::Predicate, "pred"),
        (ElementType::I1, "i1"),
        (ElementType::I2, "i2"),
        (ElementType::I4, "i4"),
        (ElementType::I8, "i8"),
        (ElementType::I16, "i16"),
        (ElementType::I32, "i32"),
        (ElementType::I64, "i64"),
        (ElementType::U1, "u1"),
        (ElementType::U2, "u2"),
        (ElementType::U4, "u4"),
        (ElementType::U8, "u8"),
        (ElementType::U16, "u16"),
        (ElementType::U32, "u32"),
        (ElementType::U64, "u64"),
        (ElementType::F4E2M1FN, "f4e2m1fn"),
        (ElementType::F8E3M4, "f8e3m4"),
        (ElementType::F8E4M3, "f8e4m3"),
        (ElementType::F8E4M3FN, "f8e4m3fn"),
        (ElementType::F8E4M3FNUZ, "f8e4m3fnuz"),
        (ElementType::F8E4M3B11FNUZ, "f8e4m3b11fnuz"),
        (ElementType::F8E5M2, "f8e5m2"),
        (ElementType::F8E5M2FNUZ, "f8e5m2fnuz"),
        (ElementType::F8E8M0FNU, "f8e8m0fnu"),
        (ElementType::BF16, "bf16"),
        (ElementType::F16, "f16"),
        (ElementType::F32, "f32"),
        (ElementType::F64, "f64"),
        (ElementType::C64, "c64"),
        (ElementType::C128, "c128"),
    ];

    #[test]
    fn test_element_type_display_matches_xla_names() {
        for &(element_type, expected_name) in DISPLAY_CASES {
            assert_eq!(element_type.to_string(), expected_name);
        }
    }

    #[cfg(feature = "xla")]
    const BUFFER_TYPE_CASES: &[(ElementType, ryft_pjrt::BufferType)] = &[
        (ElementType::Invalid, ryft_pjrt::BufferType::Invalid),
        (ElementType::Token, ryft_pjrt::BufferType::Token),
        (ElementType::Predicate, ryft_pjrt::BufferType::Predicate),
        (ElementType::I1, ryft_pjrt::BufferType::I1),
        (ElementType::I2, ryft_pjrt::BufferType::I2),
        (ElementType::I4, ryft_pjrt::BufferType::I4),
        (ElementType::I8, ryft_pjrt::BufferType::I8),
        (ElementType::I16, ryft_pjrt::BufferType::I16),
        (ElementType::I32, ryft_pjrt::BufferType::I32),
        (ElementType::I64, ryft_pjrt::BufferType::I64),
        (ElementType::U1, ryft_pjrt::BufferType::U1),
        (ElementType::U2, ryft_pjrt::BufferType::U2),
        (ElementType::U4, ryft_pjrt::BufferType::U4),
        (ElementType::U8, ryft_pjrt::BufferType::U8),
        (ElementType::U16, ryft_pjrt::BufferType::U16),
        (ElementType::U32, ryft_pjrt::BufferType::U32),
        (ElementType::U64, ryft_pjrt::BufferType::U64),
        (ElementType::F4E2M1FN, ryft_pjrt::BufferType::F4E2M1FN),
        (ElementType::F8E3M4, ryft_pjrt::BufferType::F8E3M4),
        (ElementType::F8E4M3, ryft_pjrt::BufferType::F8E4M3),
        (ElementType::F8E4M3FN, ryft_pjrt::BufferType::F8E4M3FN),
        (ElementType::F8E4M3FNUZ, ryft_pjrt::BufferType::F8E4M3FNUZ),
        (ElementType::F8E4M3B11FNUZ, ryft_pjrt::BufferType::F8E4M3B11FNUZ),
        (ElementType::F8E5M2, ryft_pjrt::BufferType::F8E5M2),
        (ElementType::F8E5M2FNUZ, ryft_pjrt::BufferType::F8E5M2FNUZ),
        (ElementType::F8E8M0FNU, ryft_pjrt::BufferType::F8E8M0FNU),
        (ElementType::BF16, ryft_pjrt::BufferType::BF16),
        (ElementType::F16, ryft_pjrt::BufferType::F16),
        (ElementType::F32, ryft_pjrt::BufferType::F32),
        (ElementType::F64, ryft_pjrt::BufferType::F64),
        (ElementType::C64, ryft_pjrt::BufferType::C64),
        (ElementType::C128, ryft_pjrt::BufferType::C128),
    ];

    #[cfg(feature = "xla")]
    #[test]
    fn test_element_type_round_trips_with_buffer_type() {
        for &(element_type, buffer_type) in BUFFER_TYPE_CASES {
            assert_eq!(ElementType::from_buffer_type(buffer_type), element_type);
            assert_eq!(element_type.to_buffer_type(), buffer_type);
        }
    }
}
