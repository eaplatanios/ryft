use std::fmt::Display;

#[cfg(feature = "xla")]
use ryft_pjrt::BufferType;

#[cfg(feature = "xla")]
use crate::errors::Error;

/// Type of the data stored in an array, tensor, matrix, vector, scalar, etc. Specifically, this represents the type of
/// individual values that are stored in that array, etc.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ElementType {
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
    /// Creates an [`ElementType`] from the provided [`BufferType`]. Returns an [`Error::InvalidElementType`]
    /// when the provided [`BufferType`] is [`BufferType::Invalid`], which is a PJRT-only sentinel value.
    pub fn from_buffer_type(buffer_type: BufferType) -> Result<Self, Error> {
        buffer_type.try_into()
    }

    /// Returns the [`BufferType`] that corresponds to this [`ElementType`].
    pub fn to_buffer_type(self) -> BufferType {
        self.into()
    }
}

#[cfg(feature = "xla")]
impl TryFrom<BufferType> for ElementType {
    type Error = Error;

    fn try_from(buffer_type: BufferType) -> Result<Self, Self::Error> {
        match buffer_type {
            BufferType::Invalid => {
                Err(Error::InvalidElementType { message: format!("invalid element type from PJRT: '{buffer_type}'") })
            }
            BufferType::Token => Ok(Self::Token),
            BufferType::Predicate => Ok(Self::Predicate),
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

#[cfg(feature = "xla")]
impl From<ElementType> for BufferType {
    fn from(element_type: ElementType) -> Self {
        match element_type {
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
    #[cfg(feature = "xla")]
    use crate::errors::Error;

    #[cfg(feature = "xla")]
    use super::BufferType;
    use super::ElementType;

    #[test]
    fn test_element_type() {
        for &(element_type, expected_name) in &[
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
        ] {
            assert_eq!(element_type.to_string(), expected_name);
        }
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_element_type_from_buffer_type() {
        assert!(matches!(
            ElementType::from_buffer_type(BufferType::Invalid),
            Err(Error::InvalidElementType { message }) if message == "invalid element type: 'invalid'",
        ));

        for &(element_type, buffer_type) in &[
            (ElementType::Token, BufferType::Token),
            (ElementType::Predicate, BufferType::Predicate),
            (ElementType::I1, BufferType::I1),
            (ElementType::I2, BufferType::I2),
            (ElementType::I4, BufferType::I4),
            (ElementType::I8, BufferType::I8),
            (ElementType::I16, BufferType::I16),
            (ElementType::I32, BufferType::I32),
            (ElementType::I64, BufferType::I64),
            (ElementType::U1, BufferType::U1),
            (ElementType::U2, BufferType::U2),
            (ElementType::U4, BufferType::U4),
            (ElementType::U8, BufferType::U8),
            (ElementType::U16, BufferType::U16),
            (ElementType::U32, BufferType::U32),
            (ElementType::U64, BufferType::U64),
            (ElementType::F4E2M1FN, BufferType::F4E2M1FN),
            (ElementType::F8E3M4, BufferType::F8E3M4),
            (ElementType::F8E4M3, BufferType::F8E4M3),
            (ElementType::F8E4M3FN, BufferType::F8E4M3FN),
            (ElementType::F8E4M3FNUZ, BufferType::F8E4M3FNUZ),
            (ElementType::F8E4M3B11FNUZ, BufferType::F8E4M3B11FNUZ),
            (ElementType::F8E5M2, BufferType::F8E5M2),
            (ElementType::F8E5M2FNUZ, BufferType::F8E5M2FNUZ),
            (ElementType::F8E8M0FNU, BufferType::F8E8M0FNU),
            (ElementType::BF16, BufferType::BF16),
            (ElementType::F16, BufferType::F16),
            (ElementType::F32, BufferType::F32),
            (ElementType::F64, BufferType::F64),
            (ElementType::C64, BufferType::C64),
            (ElementType::C128, BufferType::C128),
        ] {
            assert_eq!(ElementType::from_buffer_type(buffer_type), Ok(element_type));
            assert_eq!(element_type.to_buffer_type(), buffer_type);
        }
    }
}
