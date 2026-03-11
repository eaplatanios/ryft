use std::fmt::Display;

use ryft_macros::Parameter;

#[cfg(feature = "xla")]
use ryft_pjrt::BufferType;

use crate::errors::Error;
use crate::parameters::Parameter;
use crate::types::Type;

/// Represents a primitive data type that can be stored in arrays, tensors, matrices, vectors, scalars, etc., which
/// range from standard numeric types including booleans, integers, floating-point numbers, and complex numbers of
/// various precisions to advanced data types that mirror [LLVM/MLIR types](https://mlir.llvm.org/docs/Dialects/Builtin)
/// like [8-bit floating-point variants](https://arxiv.org/abs/2209.05433).
///
/// # Type Promotion
///
/// The data types form a hierarchy for type promotion when data of multiple types are mixed together in operations
/// that follows the [type promotion semantics of JAX](https://docs.jax.dev/en/latest/type_promotion.html). At a high
/// level, type promotion is governed by the following rules:
///
///   - [`DataType::Boolean`] can be promoted to any numeric type.
///   - Integer types can be promoted to wider integer types and floating-point types.
///   - Floating-point types can be promoted to wider floating-point types.
///   - Real types can be promoted to complex types with at least twice the number of bits (i.e., so that the real
///     and imaginary parts can be promoted without loss of precision).
///
/// The type promotion logic is implemented in [`DataType::promotable_to`]. Note that these type promotion rules only
/// apply for automatic promotions. If you want to convert between [`DataType`]s violating these rules, you can still
/// do so explicitly using casting. `ryft` requires you to be explicit in such cases due to the risks around loss of
/// precision in arbitrary data type conversions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum DataType {
    /// [`DataType`] that represents token values that are threaded between side-effecting operations.
    /// This type is only used for values that contain a single token (i.e., that represent scalar values).
    Token,

    /// Boolean [`DataType`] that represents `true`/`false` values but can be promoted to any other [`DataType`].
    Boolean,

    /// [`DataType`] that represents 1-bit signed integer values, with the only representable values being `0` and `-1`.
    I1,

    /// [`DataType`] that represents 2-bit signed integer values, where the first bit corresponds to the sign.
    I2,

    /// [`DataType`] that represents 4-bit signed integer values, where the first bit corresponds to the sign.
    I4,

    /// [`DataType`] that represents 8-bit signed integer values, where the first bit corresponds to the sign.
    I8,

    /// [`DataType`] that represents 16-bit signed integer values, where the first bit corresponds to the sign.
    I16,

    /// [`DataType`] that represents 32-bit signed integer values, where the first bit corresponds to the sign.
    I32,

    /// [`DataType`] that represents 64-bit signed integer values, where the first bit corresponds to the sign.
    I64,

    /// [`DataType`] that represents 1-bit unsigned integer values.
    U1,

    /// [`DataType`] that represents 2-bit unsigned integer values.
    U2,

    /// [`DataType`] that represents 4-bit unsigned integer values.
    U4,

    /// [`DataType`] that represents 8-bit unsigned integer values.
    U8,

    /// [`DataType`] that represents 16-bit unsigned integer values.
    U16,

    /// [`DataType`] that represents 32-bit unsigned integer values.
    U32,

    /// [`DataType`] that represents 64-bit unsigned integer values.
    U64,

    /// [`DataType`] that represents 4-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E2M1 (1 bit for the sign, 2 bits for the exponent, and 1 bits for the mantissa).
    ///   - Exponent Bias: 1.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Not supported.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 1) * (1 * [E == 0] + M_0 * 2^-1)`.
    ///
    /// The `FN` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type can
    /// only represent finite values.
    F4E2M1FN,

    /// [`DataType`] that represents 6-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E2M3 (1 bit for the sign, 2 bits for the exponent, and 3 bits for the mantissa).
    ///   - Exponent Bias: 1.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Not supported.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 1) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    ///
    /// The `FN` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type can
    /// only represent finite values.
    F6E2M3FN,

    /// [`DataType`] that represents 6-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E3M2 (1 bit for the sign, 3 bits for the exponent, and 2 bits for the mantissa).
    ///   - Exponent Bias: 3.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Not supported.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 3) * (1 * [E == 0] + M_1 * 2^-1 + M_0 * 2^-2)`.
    ///
    /// The `FN` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type can
    /// only represent finite values.
    F6E3M2FN,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E3M4 (1 bit for the sign, 3 bits for the exponent, and 4 bits for the mantissa).
    ///   - Exponent Bias: 3.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 3) * (1 * [E == 0] + M_3 * 2^-1 + M_2 * 2^-2 + M_1 * 2^-3 + M_0 * 2^-4)`.
    F8E3M4,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E4M3 (1 bit for the sign, 4 bits for the exponent, and 3 bits for the mantissa).
    ///   - Exponent Bias: 7.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 7) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    F8E4M3,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E4M3 (1 bit for the sign, 4 bits for the exponent, and 3 bits for the mantissa).
    ///   - Exponent Bias: 7.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Supported with the sign bit set to `0` and all other bits set to `1`.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 7) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    ///
    /// The `FN` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type can
    /// only represent finite values.
    F8E4M3FN,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E4M3 (1 bit for the sign, 4 bits for the exponent, and 3 bits for the mantissa).
    ///   - Exponent Bias: 8.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Supported with the sign bit set to `1` and all other bits set to `0`.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 8) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    ///
    /// The `FNUZ` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type is
    /// not consistent with the IEEE 754 standard. The `FN` indicates that it can only represent finite values, and the
    /// `UZ` stands for "unsigned zero".
    F8E4M3FNUZ,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E4M3 (1 bit for the sign, 4 bits for the exponent, and 3 bits for the mantissa).
    ///   - Exponent Bias: 11.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Supported with the sign bit set to `1` and all other bits set to `0`.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 11) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    ///
    /// The `FNUZ` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type is
    /// not consistent with the IEEE 754 standard. The `FN` indicates that it can only represent finite values, and the
    /// `UZ` stands for "unsigned zero".
    F8E4M3B11FNUZ,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E5M2 (1 bit for the sign, 5 bits for the exponent, and 2 bits for the mantissa).
    ///   - Exponent Bias: 15.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and the mantissa bits
    ///     set to `01`, `10`, or `11`.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 15) * (1 * [E == 0] + M_1 * 2^-1 + M_0 * 2^-2)`.
    F8E5M2,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S1E5M2 (1 bit for the sign, 5 bits for the exponent, and 2 bits for the mantissa).
    ///   - Exponent Bias: 16.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Supported with the sign bit set to `1` and all other bits set to `0`.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 16) * (1 * [E == 0] + M_2 * 2^-1 + M_1 * 2^-2 + M_0 * 2^-3)`.
    ///
    /// The `FNUZ` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type is
    /// not consistent with the IEEE 754 standard. The `FN` indicates that it can only represent finite values, and the
    /// `UZ` stands for "unsigned zero".
    F8E5M2FNUZ,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S0E8M0 (0 bits for the sign, 8 bits for the exponent, and 0 bits for the mantissa).
    ///   - Exponent Bias: 127.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Not supported.
    ///   - Denormalized: Not supported.
    ///
    /// The value of a number in this representation can be computed as:
    /// `2^(E - 127)`.
    ///
    /// The `FNU` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type can
    /// only represent finite values and that it has no sign bit.
    F8E8M0FNU,

    /// [`DataType`] that represents 16-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard. Instead, it follows the [bfloat16 format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format),
    /// with the following characteristics:
    ///
    ///   - Bit Encoding: S1E8M7 (1 bit for the sign, 8 bits for the exponent, and 7 bits for the mantissa).
    ///   - Exponent Bias: 127.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 127) * (1 * [E == 0] + M_6 * 2^-1 + ... + M_0 * 2^-7)`.
    ///
    /// [`DataType::BF16`] has lower precision but higher dynamic range compared to [`DataType::F16`].
    ///
    /// Refer to [this page](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) for more information
    /// on this [DataType].
    BF16,

    /// [`DataType`] that represents 16-bit floating-point values following the
    /// [IEEE 754 standard](https://en.wikipedia.org/wiki/IEEE_754), with the following characteristics:
    ///
    ///   - Bit Encoding: S1E5M10 (1 bit for the sign, 5 bits for the exponent, and 10 bits for the mantissa).
    ///   - Exponent Bias: 15.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 15) * (1 * [E == 0] + M_9 * 2^-1 + ... + M_0 * 2^-10)`.
    ///
    /// Refer to [this page](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) for more information
    /// on this [DataType].
    F16,

    /// [`DataType`] that represents 32-bit floating-point values following the
    /// [IEEE 754 standard](https://en.wikipedia.org/wiki/IEEE_754), with the following characteristics:
    ///
    ///   - Bit Encoding: S1E8M23 (1 bit for the sign, 8 bits for the exponent, and 23 bits for the mantissa).
    ///   - Exponent Bias: 127.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 127) * (1 * [E == 0] + M_22 * 2^-1 + ... + M_0 * 2^-23)`.
    ///
    /// Refer to [this page](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) for more information
    /// on this [DataType].
    F32,

    /// [`DataType`] that represents 64-bit floating-point values following the
    /// [IEEE 754 standard](https://en.wikipedia.org/wiki/IEEE_754), with the following characteristics:
    ///
    ///   - Bit Encoding: S1E11M52 (1 bit for the sign, 11 bits for the exponent, and 52 bits for the mantissa).
    ///   - Exponent Bias: 1023.
    ///   - Infinity Values: Supported with the exponent set to all `1`s and the mantissa to all `0`s.
    ///   - Not-a-Number (NaN) Values: Supported with the exponent bits set to all `1`s and arbitrary mantissa bit
    ///     values except for all `0`s.
    ///   - Denormalized: When the exponent is all zeros, the number is interpreted as denormalized, meaning that the
    ///     implied leading bit of the mantissa is considered to be a `0` instead of a `1`.
    ///
    /// The value of a number in this representation can be computed as:
    /// `(-1)^S * 2^(E - 127) * (1 * [E == 0] + M_22 * 2^-1 + ... + M_0 * 2^-23)`.
    ///
    /// Refer to [this page](https://en.wikipedia.org/wiki/Double-precision_floating-point_format) for more information
    /// on this [DataType].
    F64,

    /// [`DataType`] that represents 64-bit complex numbers where the real and imaginary parts
    /// are [`DataType::F32`].
    C64,

    /// [`DataType`] that represents 128-bit complex numbers where the real and imaginary parts
    /// are [`DataType::F64`].
    C128,
}

impl DataType {
    /// Constructs a new [`DataType`] that is the widest data type from the provided data types and which all of the
    /// provided data types can be promoted to.
    ///
    /// Note that this operation is *order-invariant* meaning that it will return the same [`DataType`] irrespective
    /// of the order in which the input [`DataType`]s are provided.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::DataType;
    /// let x = DataType::Boolean;
    /// let y = DataType::U16;
    /// let z = DataType::F32;
    ///
    /// assert_eq!(DataType::promoted(&[&x]), Ok(x));
    /// assert_eq!(DataType::promoted(&[&x, &y]), Ok(y));
    /// assert_eq!(DataType::promoted(&[&x, &z]), Ok(z));
    /// assert_eq!(DataType::promoted(&[&z, &y]), Ok(z));
    /// assert_eq!(DataType::promoted(&[&x, &y, &z]), Ok(z));
    /// ```
    pub fn promoted(data_types: &[&Self]) -> Result<Self, Error> {
        if data_types.is_empty() {
            return Err(Error::invalid_argument(
                "cannot construct a promoted data type from an empty collection of data types",
            ));
        }

        let maybe_widest = data_types.iter().copied().reduce(|lhs, rhs| if lhs.promotable_to(rhs) { rhs } else { lhs });
        data_types.iter().fold(Ok(*maybe_widest.unwrap()), |rhs, lhs| match rhs {
            Ok(rhs) if lhs.promotable_to(&rhs) => Ok(rhs),
            Ok(rhs) => Err(Self::incompatible_promotion_error(**lhs, rhs)),
            Err(error) => Err(error),
        })
    }

    /// Promotes this [`DataType`] to the provided [`DataType`]. Refer to the documentation of [`DataType`] for more
    /// information on type promotions and the rules that govern them.
    #[inline]
    pub fn promote_to(&self, other: &DataType) -> Result<DataType, Error> {
        if self.promotable_to(other) { Ok(*other) } else { Err(Self::incompatible_promotion_error(*self, *other)) }
    }

    /// Returns `true` if this [`DataType`] can be promoted to the provided [`DataType`]. Note that this function will
    /// always return `true` when `self == other`. Refer to the documentation of [`DataType`] for more information on
    /// type promotions and the rules that govern them.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::DataType;
    /// assert!(DataType::I32.promotable_to(&DataType::F64));
    /// assert!(DataType::F32.promotable_to(&DataType::C64));
    /// assert!(!DataType::F64.promotable_to(&DataType::I32));
    /// ```
    #[inline]
    pub fn promotable_to(&self, other: &Self) -> bool {
        match (self, other) {
            (DataType::Token, DataType::Token) => true,
            (DataType::Token, _) => false,
            (DataType::Boolean, _) => true,
            (_, DataType::Boolean) => false,
            (
                DataType::I1,
                DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64,
            ) => false,
            (DataType::I1, _) => true,
            (
                DataType::I2,
                DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64,
            ) => false,
            (DataType::I2, _) => true,
            (
                DataType::I4,
                DataType::I1
                | DataType::I2
                | DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64,
            ) => false,
            (DataType::I4, _) => true,
            (
                DataType::I8,
                DataType::I1
                | DataType::I2
                | DataType::I4
                | DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64,
            ) => false,
            (DataType::I8, _) => true,
            (
                DataType::I16,
                DataType::I1
                | DataType::I2
                | DataType::I4
                | DataType::I8
                | DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64,
            ) => false,
            (DataType::I16, _) => true,
            (
                DataType::I32,
                DataType::I1
                | DataType::I2
                | DataType::I4
                | DataType::I8
                | DataType::I16
                | DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64,
            ) => false,
            (DataType::I32, _) => true,
            (
                DataType::I64,
                DataType::I1
                | DataType::I2
                | DataType::I4
                | DataType::I8
                | DataType::I16
                | DataType::I32
                | DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8
                | DataType::U16
                | DataType::U32
                | DataType::U64,
            ) => false,
            (DataType::I64, _) => true,
            (DataType::U1, DataType::I1) => false,
            (DataType::U1, _) => true,
            (DataType::U2, DataType::I2) => false,
            (DataType::U2, _) => true,
            (DataType::U4, DataType::I1 | DataType::I2 | DataType::I4 | DataType::U1 | DataType::U2) => false,
            (DataType::U4, _) => true,
            (
                DataType::U8,
                DataType::I1 | DataType::I2 | DataType::I4 | DataType::I8 | DataType::U1 | DataType::U2 | DataType::U4,
            ) => false,
            (DataType::U8, _) => true,
            (
                DataType::U16,
                DataType::I1
                | DataType::I2
                | DataType::I4
                | DataType::I8
                | DataType::I16
                | DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8,
            ) => false,
            (DataType::U16, _) => true,
            (
                DataType::U32,
                DataType::I1
                | DataType::I2
                | DataType::I4
                | DataType::I8
                | DataType::I16
                | DataType::I32
                | DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8
                | DataType::U16,
            ) => false,
            (DataType::U32, _) => true,
            (
                DataType::U64,
                DataType::I1
                | DataType::I2
                | DataType::I4
                | DataType::I8
                | DataType::I16
                | DataType::I32
                | DataType::I64
                | DataType::U1
                | DataType::U2
                | DataType::U4
                | DataType::U8
                | DataType::U16
                | DataType::U32,
            ) => false,
            (DataType::U64, _) => true,
            (
                DataType::F4E2M1FN,
                DataType::F4E2M1FN
                | DataType::F6E2M3FN
                | DataType::F6E3M2FN
                | DataType::F8E3M4
                | DataType::F8E4M3
                | DataType::F8E5M2
                | DataType::F8E4M3FN
                | DataType::F8E4M3B11FNUZ
                | DataType::F8E5M2FNUZ
                | DataType::F8E4M3FNUZ
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F4E2M1FN, _) => false,
            (
                DataType::F6E2M3FN,
                DataType::F6E2M3FN
                | DataType::F8E3M4
                | DataType::F8E4M3
                | DataType::F8E5M2
                | DataType::F8E4M3FN
                | DataType::F8E4M3B11FNUZ
                | DataType::F8E5M2FNUZ
                | DataType::F8E4M3FNUZ
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F6E2M3FN, _) => false,
            (
                DataType::F6E3M2FN,
                DataType::F6E3M2FN
                | DataType::F8E3M4
                | DataType::F8E4M3
                | DataType::F8E5M2
                | DataType::F8E4M3FN
                | DataType::F8E4M3B11FNUZ
                | DataType::F8E5M2FNUZ
                | DataType::F8E4M3FNUZ
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F6E3M2FN, _) => false,
            (
                DataType::F8E3M4,
                DataType::F8E3M4
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F8E3M4, _) => false,
            (
                DataType::F8E4M3,
                DataType::F8E4M3
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F8E4M3, _) => false,
            (
                DataType::F8E5M2,
                DataType::F8E5M2
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F8E5M2, _) => false,
            (
                DataType::F8E4M3FN,
                DataType::F8E4M3FN
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F8E4M3FN, _) => false,
            (
                DataType::F8E4M3B11FNUZ,
                DataType::F8E4M3B11FNUZ
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F8E4M3B11FNUZ, _) => false,
            (
                DataType::F8E5M2FNUZ,
                DataType::F8E5M2FNUZ
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F8E5M2FNUZ, _) => false,
            (
                DataType::F8E4M3FNUZ,
                DataType::F8E4M3FNUZ
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F8E4M3FNUZ, _) => false,
            (
                DataType::F8E8M0FNU,
                DataType::F8E8M0FNU
                | DataType::BF16
                | DataType::F16
                | DataType::F32
                | DataType::F64
                | DataType::C64
                | DataType::C128,
            ) => true,
            (DataType::F8E8M0FNU, _) => false,
            (DataType::BF16, DataType::BF16 | DataType::F32 | DataType::F64 | DataType::C64 | DataType::C128) => true,
            (DataType::BF16, _) => false,
            (DataType::F16, DataType::F16 | DataType::F32 | DataType::F64 | DataType::C64 | DataType::C128) => true,
            (DataType::F16, _) => false,
            (DataType::F32, DataType::F32 | DataType::F64 | DataType::C64 | DataType::C128) => true,
            (DataType::F32, _) => false,
            (DataType::F64, DataType::F64 | DataType::C128) => true,
            (DataType::F64, _) => false,
            (DataType::C64, DataType::C64 | DataType::C128) => true,
            (DataType::C64, _) => false,
            (DataType::C128, DataType::C128) => true,
            (DataType::C128, _) => false,
        }
    }

    fn incompatible_promotion_error(lhs: DataType, rhs: DataType) -> Error {
        Error::invalid_data_type_promotion(format!("cannot promote type `{lhs}` to type `{rhs}`"))
    }
}

impl Display for DataType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            DataType::Token => "token",
            DataType::Boolean => "bool",
            DataType::I1 => "i1",
            DataType::I2 => "i2",
            DataType::I4 => "i4",
            DataType::I8 => "i8",
            DataType::I16 => "i16",
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::U1 => "u1",
            DataType::U2 => "u2",
            DataType::U4 => "u4",
            DataType::U8 => "u8",
            DataType::U16 => "u16",
            DataType::U32 => "u32",
            DataType::U64 => "u64",
            DataType::F4E2M1FN => "f4e2m1fn",
            DataType::F6E2M3FN => "f6e2m3fn",
            DataType::F6E3M2FN => "f6e3m2fn",
            DataType::F8E3M4 => "f8e3m4",
            DataType::F8E4M3 => "f8e4m3",
            DataType::F8E4M3FN => "f8e4m3fn",
            DataType::F8E4M3FNUZ => "f8e4m3fnuz",
            DataType::F8E4M3B11FNUZ => "f8e4m3b11fnuz",
            DataType::F8E5M2 => "f8e5m2",
            DataType::F8E5M2FNUZ => "f8e5m2fnuz",
            DataType::F8E8M0FNU => "f8e8m0fnu",
            DataType::BF16 => "bf16",
            DataType::F16 => "f16",
            DataType::F32 => "f32",
            DataType::F64 => "f64",
            DataType::C64 => "c64",
            DataType::C128 => "c128",
        })
    }
}

impl Type for DataType {
    /// Returns `true` if this [`DataType`] is a subtype of the provided [`DataType`] (i.e., that this data type can
    /// be promoted to the provided data type).
    #[inline]
    fn is_subtype_of(&self, other: &Self) -> bool {
        self.promotable_to(other)
    }
}

#[cfg(feature = "xla")]
impl DataType {
    /// Creates a [`DataType`] from the provided PJRT [`BufferType`]. Returns
    /// [`crate::errors::Error::InvalidDataType`] when the provided [`BufferType`] is [`BufferType::Invalid`], which is
    /// a PJRT-only sentinel value.
    pub fn from_pjrt_buffer_type(buffer_type: BufferType) -> Result<Self, Error> {
        buffer_type.try_into()
    }

    /// Returns the PJRT [`BufferType`] that corresponds to this [`DataType`].
    ///
    /// Returns [`crate::errors::Error::InvalidDataType`] when this [`DataType`] has no PJRT representation.
    pub fn to_pjrt_buffer_type(self) -> Result<BufferType, Error> {
        self.try_into()
    }
}

#[cfg(feature = "xla")]
impl TryFrom<BufferType> for DataType {
    type Error = Error;

    fn try_from(buffer_type: BufferType) -> Result<Self, Self::Error> {
        match buffer_type {
            BufferType::Invalid => {
                Err(Error::invalid_data_type(format!("invalid data type from PJRT: '{buffer_type}'")))
            }
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

#[cfg(feature = "xla")]
impl TryFrom<DataType> for BufferType {
    type Error = Error;

    fn try_from(data_type: DataType) -> Result<Self, Self::Error> {
        match data_type {
            DataType::Token => Ok(Self::Token),
            DataType::Boolean => Ok(Self::Predicate),
            DataType::I1 => Ok(Self::I1),
            DataType::I2 => Ok(Self::I2),
            DataType::I4 => Ok(Self::I4),
            DataType::I8 => Ok(Self::I8),
            DataType::I16 => Ok(Self::I16),
            DataType::I32 => Ok(Self::I32),
            DataType::I64 => Ok(Self::I64),
            DataType::U1 => Ok(Self::U1),
            DataType::U2 => Ok(Self::U2),
            DataType::U4 => Ok(Self::U4),
            DataType::U8 => Ok(Self::U8),
            DataType::U16 => Ok(Self::U16),
            DataType::U32 => Ok(Self::U32),
            DataType::U64 => Ok(Self::U64),
            DataType::F4E2M1FN => Ok(Self::F4E2M1FN),
            DataType::F6E2M3FN | DataType::F6E3M2FN => {
                Err(Error::invalid_data_type(format!("data type '{data_type}' has no corresponding PJRT buffer type")))
            }
            DataType::F8E3M4 => Ok(Self::F8E3M4),
            DataType::F8E4M3 => Ok(Self::F8E4M3),
            DataType::F8E4M3FN => Ok(Self::F8E4M3FN),
            DataType::F8E4M3FNUZ => Ok(Self::F8E4M3FNUZ),
            DataType::F8E4M3B11FNUZ => Ok(Self::F8E4M3B11FNUZ),
            DataType::F8E5M2 => Ok(Self::F8E5M2),
            DataType::F8E5M2FNUZ => Ok(Self::F8E5M2FNUZ),
            DataType::F8E8M0FNU => Ok(Self::F8E8M0FNU),
            DataType::BF16 => Ok(Self::BF16),
            DataType::F16 => Ok(Self::F16),
            DataType::F32 => Ok(Self::F32),
            DataType::F64 => Ok(Self::F64),
            DataType::C64 => Ok(Self::C64),
            DataType::C128 => Ok(Self::C128),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::errors::Error;

    #[cfg(feature = "xla")]
    use super::BufferType;
    use super::DataType;

    #[test]
    fn test_data_type_promoted() {
        assert_eq!(DataType::promoted(&[&DataType::Boolean]), Ok(DataType::Boolean));
        assert_eq!(DataType::promoted(&[&DataType::Boolean, &DataType::C64]), Ok(DataType::C64));
        assert_eq!(DataType::promoted(&[&DataType::Boolean, &DataType::I2, &DataType::C64]), Ok(DataType::C64));
        assert_eq!(DataType::promoted(&[&DataType::F32, &DataType::I2, &DataType::BF16]), Ok(DataType::F32));
        assert_eq!(DataType::promoted(&[&DataType::F16, &DataType::BF16, &DataType::F64]), Ok(DataType::F64));
        assert_eq!(DataType::promoted(&[&DataType::F4E2M1FN, &DataType::F6E2M3FN]), Ok(DataType::F6E2M3FN));

        assert!(matches!(
            DataType::promoted(&[]),
            Err(Error::InvalidArgument { message, .. })
                if message == "cannot construct a promoted data type from an empty collection of data types",
        ));
        assert!(matches!(
            DataType::promoted(&[&DataType::F16, &DataType::BF16]),
            Err(Error::InvalidDataTypePromotion { .. }),
        ));
        assert!(matches!(
            DataType::promoted(&[&DataType::F8E3M4, &DataType::F8E8M0FNU]),
            Err(Error::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `f8e8m0fnu` to type `f8e3m4`",
        ));
    }

    #[test]
    fn test_data_type_promote_to() {
        assert_eq!(DataType::Boolean.promote_to(&DataType::Boolean), Ok(DataType::Boolean));
        assert_eq!(DataType::Boolean.promote_to(&DataType::C128), Ok(DataType::C128));
        assert_eq!(DataType::U4.promote_to(&DataType::I8), Ok(DataType::I8));
        assert_eq!(DataType::F8E5M2FNUZ.promote_to(&DataType::BF16), Ok(DataType::BF16));
        assert_eq!(DataType::BF16.promote_to(&DataType::F32), Ok(DataType::F32));
        assert_eq!(DataType::F64.promote_to(&DataType::C128), Ok(DataType::C128));

        assert!(matches!(
            DataType::U4.promote_to(&DataType::I4),
            Err(Error::InvalidDataTypePromotion { message, .. }) if message == "cannot promote type `u4` to type `i4`",
        ));
        assert!(matches!(
            DataType::I16.promote_to(&DataType::Boolean),
            Err(Error::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `i16` to type `bool`",
        ));
        assert!(matches!(
            DataType::F6E2M3FN.promote_to(&DataType::F4E2M1FN),
            Err(Error::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `f6e2m3fn` to type `f4e2m1fn`",
        ));
        assert!(matches!(
            DataType::F6E2M3FN.promote_to(&DataType::F6E3M2FN),
            Err(Error::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `f6e2m3fn` to type `f6e3m2fn`",
        ));
        assert!(matches!(
            DataType::F64.promote_to(&DataType::C64),
            Err(Error::InvalidDataTypePromotion { message, .. })
                if message == "cannot promote type `f64` to type `c64`",
        ));
    }

    #[test]
    fn test_data_type_promotable_to() {
        assert!(DataType::Boolean.promotable_to(&DataType::F4E2M1FN));
        assert!(DataType::Boolean.promotable_to(&DataType::BF16));
        assert!(DataType::Boolean.promotable_to(&DataType::C128));
        assert!(DataType::U4.promotable_to(&DataType::U4));
        assert!(DataType::U4.promotable_to(&DataType::I64));
        assert!(DataType::F8E4M3B11FNUZ.promotable_to(&DataType::BF16));
        assert!(DataType::F8E8M0FNU.promotable_to(&DataType::F16));
        assert!(DataType::F16.promotable_to(&DataType::F32));
        assert!(DataType::F32.promotable_to(&DataType::C64));

        assert!(!DataType::U8.promotable_to(&DataType::I8));
        assert!(!DataType::I2.promotable_to(&DataType::Boolean));
        assert!(!DataType::F6E2M3FN.promotable_to(&DataType::U2));
        assert!(!DataType::F6E3M2FN.promotable_to(&DataType::F6E2M3FN));
        assert!(!DataType::F8E4M3B11FNUZ.promotable_to(&DataType::F4E2M1FN));
    }

    #[test]
    fn test_data_type_to_string() {
        assert_eq!(DataType::Token.to_string(), "token");
        assert_eq!(DataType::Boolean.to_string(), "bool");
        assert_eq!(DataType::U4.to_string(), "u4");
        assert_eq!(DataType::I64.to_string(), "i64");
        assert_eq!(DataType::F6E2M3FN.to_string(), "f6e2m3fn");
        assert_eq!(DataType::F8E4M3FNUZ.to_string(), "f8e4m3fnuz");
        assert_eq!(DataType::BF16.to_string(), "bf16");
        assert_eq!(DataType::F64.to_string(), "f64");
        assert_eq!(DataType::C128.to_string(), "c128");
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_data_type_from_pjrt_buffer_type() {
        assert!(matches!(
            DataType::from_pjrt_buffer_type(BufferType::Invalid),
            Err(Error::InvalidDataType { message, .. }) if message == "invalid data type from PJRT: 'invalid'",
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
            assert_eq!(DataType::from_pjrt_buffer_type(buffer_type), Ok(data_type));
            assert_eq!(data_type.to_pjrt_buffer_type(), Ok(buffer_type));
        }
    }

    #[cfg(feature = "xla")]
    #[test]
    fn test_data_type_to_pjrt_buffer_type_rejects_unsupported_types() {
        assert!(matches!(DataType::F6E2M3FN.to_pjrt_buffer_type(), Err(Error::InvalidDataType { .. }),));
        assert!(matches!(DataType::F6E3M2FN.to_pjrt_buffer_type(), Err(Error::InvalidDataType { .. }),));
    }
}
