//! [`Type`]s for multi-dimensional arrays with broadcasting semantics.
//!
//! This module provides [`Type`]s for representing multi-dimensional arrays with various data types and dimensions.
//! It implements [NumPy-like broadcasting semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html)
//! and [JAX-like data type promotion rules](https://docs.jax.dev/en/latest/type_promotion.html).

use std::fmt::Display;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::parameters::Parameter;
use crate::types::r#type::Type;

/// Represents the primitive data types that can be stored in arrays which range from standard numeric types including
/// booleans, integers, floating-point numbers, and complex numbers of various precisions to advanced data types that
/// mirror [LLVM/MLIR types](https://mlir.llvm.org/docs/Dialects/Builtin) like
/// [8-bit floating-point variants](https://arxiv.org/abs/2209.05433).
///
/// # Type Promotion
///
/// The data types form a hierarchy for type promotion when data of multiple types is mixed together in an operation,
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
    /// Boolean [`DataType`] that represents `true`/`false` values but can be promoted to any other [`DataType`].
    Boolean,

    /// [`DataType`] that represents 2-bit signed integer values, where the first bit corresponds to the sign.
    Int2,

    /// [`DataType`] that represents 4-bit signed integer values, where the first bit corresponds to the sign.
    Int4,

    /// [`DataType`] that represents 8-bit signed integer values, where the first bit corresponds to the sign.
    Int8,

    /// [`DataType`] that represents 16-bit signed integer values, where the first bit corresponds to the sign.
    Int16,

    /// [`DataType`] that represents 32-bit signed integer values, where the first bit corresponds to the sign.
    Int32,

    /// [`DataType`] that represents 64-bit signed integer values, where the first bit corresponds to the sign.
    Int64,

    /// [`DataType`] that represents 2-bit unsigned integer values.
    UnsignedInt2,

    /// [`DataType`] that represents 4-bit unsigned integer values.
    UnsignedInt4,

    /// [`DataType`] that represents 8-bit unsigned integer values.
    UnsignedInt8,

    /// [`DataType`] that represents 16-bit unsigned integer values.
    UnsignedInt16,

    /// [`DataType`] that represents 32-bit unsigned integer values.
    UnsignedInt32,

    /// [`DataType`] that represents 64-bit unsigned integer values.
    UnsignedInt64,

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
    Float4E2M1FN,

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
    Float6E2M3FN,

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
    Float6E3M2FN,

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
    Float8E3M4,

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
    Float8E4M3,

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
    Float8E4M3FN,

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
    Float8E4M3FNUZ,

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
    Float8E4M3B11FNUZ,

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
    Float8E5M2,

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
    Float8E5M2FNUZ,

    /// [`DataType`] that represents 8-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
    ///
    ///   - Bit Encoding: S0E8M0 (0 bits for the sign, 8 bits for the exponent, and 0 bits for the mantissa).
    ///   - Exponent Bias: 127.
    ///   - Infinity Values: Not supported.
    ///   - Not-a-Number (NaN) Values: Supported with all bits set to `1`.
    ///
    /// The value of a number in this representation can be computed as: `2^(E - 127)`.
    ///
    /// This type is intended to be used for representing scaling factors and so it cannot represent zeros and negative
    /// numbers. The values it can represent are powers of two in the range `[-127, 127]` and NaN.
    ///
    /// The `FNU` name suffix is for consistency with the corresponding LLVM/MLIR type, signaling that this type is
    /// not consistent with the IEEE 754 standard. The `FN` indicates that it can only represent finite values, and the
    /// `U` stands for "unsigned".
    Float8E8M0FNU,

    /// [`DataType`] that represents 16-bit floating-point values. This is not a standard type as defined by the
    /// IEEE 754 standard, but it follows similar conventions with the following characteristics:
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
    /// [`DataType::BFloat16`] has lower precision but higher dynamic range compared to [`DataType::Float16`].
    ///
    /// Refer to [this page](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) for more information
    /// on this [DataType].
    BFloat16,

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
    Float16,

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
    Float32,

    /// [`DataType`] that represents 32-bit floating-point values following the
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
    Float64,

    /// [`DataType`] that represents 64-bit complex numbers where the real and imaginary parts
    /// are [`DataType::Float32`].
    Complex64,

    /// [`DataType`] that represents 128-bit complex numbers where the real and imaginary parts
    /// are [`DataType::Float64`].
    Complex128,
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
    /// # use ryft_core::types::array_type::DataType;
    /// let x = DataType::Boolean;
    /// let y = DataType::UnsignedInt16;
    /// let z = DataType::Float32;
    ///
    /// assert_eq!(DataType::promoted(&[&x]), Ok(x.clone()));
    /// assert_eq!(DataType::promoted(&[&x, &y]), Ok(y.clone()));
    /// assert_eq!(DataType::promoted(&[&x, &z]), Ok(z.clone()));
    /// assert_eq!(DataType::promoted(&[&z, &y]), Ok(z.clone()));
    /// assert_eq!(DataType::promoted(&[&x, &y, &z]), Ok(z.clone()));
    /// ```
    pub fn promoted(data_types: &[&Self]) -> Result<Self, DataTypePromotionError> {
        if data_types.is_empty() {
            return Err(DataTypePromotionError::Empty);
        }

        let maybe_widest = data_types.into_iter().reduce(|lhs, rhs| if lhs.promotable_to(rhs) { rhs } else { lhs });
        data_types.iter().fold(Ok(**maybe_widest.unwrap()), |rhs, lhs| match rhs {
            Ok(rhs) if lhs.promotable_to(&rhs) => Ok(rhs),
            Ok(rhs) => Err(DataTypePromotionError::Incompatible { lhs: **lhs, rhs }),
            Err(error) => Err(error),
        })
    }

    /// Promotes this [`DataType`] to the provided [`DataType`]. Refer to the documentation of [`DataType`] for more
    /// information on type promotions and the rules that govern them.
    #[inline]
    pub fn promote_to(&self, other: &DataType) -> Result<DataType, DataTypePromotionError> {
        if self.promotable_to(other) {
            Ok(other.clone())
        } else {
            Err(DataTypePromotionError::Incompatible { lhs: self.clone(), rhs: other.clone() })
        }
    }

    /// Returns `true` if this [`DataType`] can be promoted to the provided [`DataType`]. Note that this function will
    /// always return `true` when `self == other`. Refer to the documentation of [`DataType`] for more information on
    /// type promotions and the rules that govern them.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::array_type::DataType;
    /// # use ryft_core::types::Type;
    /// assert!(DataType::Int32.promotable_to(&DataType::Float64));
    /// assert!(DataType::Float32.promotable_to(&DataType::Complex64));
    /// assert!(!DataType::Float64.promotable_to(&DataType::Int32));
    /// ```
    #[inline]
    pub fn promotable_to(&self, other: &Self) -> bool {
        match (&self, &other) {
            (DataType::Boolean, _) => true,
            (_, DataType::Boolean) => false,
            (
                DataType::Int2,
                DataType::UnsignedInt2
                | DataType::UnsignedInt4
                | DataType::UnsignedInt8
                | DataType::UnsignedInt16
                | DataType::UnsignedInt32
                | DataType::UnsignedInt64,
            ) => false,
            (DataType::Int2, _) => true,
            (
                DataType::Int4,
                DataType::Int2
                | DataType::UnsignedInt2
                | DataType::UnsignedInt4
                | DataType::UnsignedInt8
                | DataType::UnsignedInt16
                | DataType::UnsignedInt32
                | DataType::UnsignedInt64,
            ) => false,
            (DataType::Int4, _) => true,
            (
                DataType::Int8,
                DataType::Int2
                | DataType::Int4
                | DataType::UnsignedInt2
                | DataType::UnsignedInt4
                | DataType::UnsignedInt8
                | DataType::UnsignedInt16
                | DataType::UnsignedInt32
                | DataType::UnsignedInt64,
            ) => false,
            (DataType::Int8, _) => true,
            (
                DataType::Int16,
                DataType::Int2
                | DataType::Int4
                | DataType::Int8
                | DataType::UnsignedInt2
                | DataType::UnsignedInt4
                | DataType::UnsignedInt8
                | DataType::UnsignedInt16
                | DataType::UnsignedInt32
                | DataType::UnsignedInt64,
            ) => false,
            (DataType::Int16, _) => true,
            (
                DataType::Int32,
                DataType::Int2
                | DataType::Int4
                | DataType::Int8
                | DataType::Int16
                | DataType::UnsignedInt2
                | DataType::UnsignedInt4
                | DataType::UnsignedInt8
                | DataType::UnsignedInt16
                | DataType::UnsignedInt32
                | DataType::UnsignedInt64,
            ) => false,
            (DataType::Int32, _) => true,
            (
                DataType::Int64,
                DataType::Int2
                | DataType::Int4
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::UnsignedInt2
                | DataType::UnsignedInt4
                | DataType::UnsignedInt8
                | DataType::UnsignedInt16
                | DataType::UnsignedInt32
                | DataType::UnsignedInt64,
            ) => false,
            (DataType::Int64, _) => true,
            (DataType::UnsignedInt2, DataType::Int2) => false,
            (DataType::UnsignedInt2, _) => true,
            (DataType::UnsignedInt4, DataType::Int2 | DataType::Int4 | DataType::UnsignedInt2) => false,
            (DataType::UnsignedInt4, _) => true,
            (
                DataType::UnsignedInt8,
                DataType::Int2 | DataType::Int4 | DataType::Int8 | DataType::UnsignedInt2 | DataType::UnsignedInt4,
            ) => false,
            (DataType::UnsignedInt8, _) => true,
            (
                DataType::UnsignedInt16,
                DataType::Int2
                | DataType::Int4
                | DataType::Int8
                | DataType::Int16
                | DataType::UnsignedInt2
                | DataType::UnsignedInt4
                | DataType::UnsignedInt8,
            ) => false,
            (DataType::UnsignedInt16, _) => true,
            (
                DataType::UnsignedInt32,
                DataType::Int2
                | DataType::Int4
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::UnsignedInt2
                | DataType::UnsignedInt4
                | DataType::UnsignedInt8
                | DataType::UnsignedInt16,
            ) => false,
            (DataType::UnsignedInt32, _) => true,
            (
                DataType::UnsignedInt64,
                DataType::Int2
                | DataType::Int4
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UnsignedInt2
                | DataType::UnsignedInt4
                | DataType::UnsignedInt8
                | DataType::UnsignedInt16
                | DataType::UnsignedInt32,
            ) => false,
            (DataType::UnsignedInt64, _) => true,
            (
                DataType::Float4E2M1FN,
                DataType::Float4E2M1FN
                | DataType::Float6E2M3FN
                | DataType::Float6E3M2FN
                | DataType::Float8E3M4
                | DataType::Float8E4M3
                | DataType::Float8E5M2
                | DataType::Float8E4M3FN
                | DataType::Float8E4M3B11FNUZ
                | DataType::Float8E5M2FNUZ
                | DataType::Float8E4M3FNUZ
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float4E2M1FN, _) => false,
            (
                DataType::Float6E2M3FN,
                DataType::Float6E2M3FN
                | DataType::Float8E3M4
                | DataType::Float8E4M3
                | DataType::Float8E5M2
                | DataType::Float8E4M3FN
                | DataType::Float8E4M3B11FNUZ
                | DataType::Float8E5M2FNUZ
                | DataType::Float8E4M3FNUZ
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float6E2M3FN, _) => false,
            (
                DataType::Float6E3M2FN,
                DataType::Float6E3M2FN
                | DataType::Float8E3M4
                | DataType::Float8E4M3
                | DataType::Float8E5M2
                | DataType::Float8E4M3FN
                | DataType::Float8E4M3B11FNUZ
                | DataType::Float8E5M2FNUZ
                | DataType::Float8E4M3FNUZ
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float6E3M2FN, _) => false,
            (
                DataType::Float8E3M4,
                DataType::Float8E3M4
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float8E3M4, _) => false,
            (
                DataType::Float8E4M3,
                DataType::Float8E4M3
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float8E4M3, _) => false,
            (
                DataType::Float8E5M2,
                DataType::Float8E5M2
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float8E5M2, _) => false,
            (
                DataType::Float8E4M3FN,
                DataType::Float8E4M3FN
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float8E4M3FN, _) => false,
            (
                DataType::Float8E4M3B11FNUZ,
                DataType::Float8E4M3B11FNUZ
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float8E4M3B11FNUZ, _) => false,
            (
                DataType::Float8E5M2FNUZ,
                DataType::Float8E5M2FNUZ
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float8E5M2FNUZ, _) => false,
            (
                DataType::Float8E4M3FNUZ,
                DataType::Float8E4M3FNUZ
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float8E4M3FNUZ, _) => false,
            (
                DataType::Float8E8M0FNU,
                DataType::Float8E8M0FNU
                | DataType::BFloat16
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Complex64
                | DataType::Complex128,
            ) => true,
            (DataType::Float8E8M0FNU, _) => false,
            (
                DataType::BFloat16,
                DataType::BFloat16 | DataType::Float32 | DataType::Float64 | DataType::Complex64 | DataType::Complex128,
            ) => true,
            (DataType::BFloat16, _) => false,
            (
                DataType::Float16,
                DataType::Float16 | DataType::Float32 | DataType::Float64 | DataType::Complex64 | DataType::Complex128,
            ) => true,
            (DataType::Float16, _) => false,
            (DataType::Float32, DataType::Float32 | DataType::Float64 | DataType::Complex64 | DataType::Complex128) => {
                true
            }
            (DataType::Float32, _) => false,
            (DataType::Float64, DataType::Float64 | DataType::Complex128) => true,
            (DataType::Float64, _) => false,
            (DataType::Complex64, DataType::Complex64 | DataType::Complex128) => true,
            (DataType::Complex64, _) => false,
            (DataType::Complex128, DataType::Complex128) => true,
            (DataType::Complex128, _) => false,
        }
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

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            DataType::Boolean => write!(f, "bool"),
            DataType::Int2 => write!(f, "i2"),
            DataType::Int4 => write!(f, "i4"),
            DataType::Int8 => write!(f, "i8"),
            DataType::Int16 => write!(f, "i16"),
            DataType::Int32 => write!(f, "i32"),
            DataType::Int64 => write!(f, "i64"),
            DataType::UnsignedInt2 => write!(f, "u2"),
            DataType::UnsignedInt4 => write!(f, "u4"),
            DataType::UnsignedInt8 => write!(f, "u8"),
            DataType::UnsignedInt16 => write!(f, "u16"),
            DataType::UnsignedInt32 => write!(f, "u32"),
            DataType::UnsignedInt64 => write!(f, "u64"),
            DataType::Float4E2M1FN => write!(f, "f4e2m1fn"),
            DataType::Float6E2M3FN => write!(f, "f6e2m3fn"),
            DataType::Float6E3M2FN => write!(f, "f6e3m2fn"),
            DataType::Float8E3M4 => write!(f, "f8e3m4"),
            DataType::Float8E4M3 => write!(f, "f8e4m3"),
            DataType::Float8E5M2 => write!(f, "f8e5m2"),
            DataType::Float8E4M3FN => write!(f, "f8e4m3fn"),
            DataType::Float8E4M3B11FNUZ => write!(f, "f8e4m3b11fnuz"),
            DataType::Float8E5M2FNUZ => write!(f, "f8e5m2fnuz"),
            DataType::Float8E4M3FNUZ => write!(f, "f8e4m3fnuz"),
            DataType::Float8E8M0FNU => write!(f, "f8e8m0fnu"),
            DataType::BFloat16 => write!(f, "bf16"),
            DataType::Float16 => write!(f, "f16"),
            DataType::Float32 => write!(f, "f32"),
            DataType::Float64 => write!(f, "f64"),
            DataType::Complex64 => write!(f, "c64"),
            DataType::Complex128 => write!(f, "c128"),
        }
    }
}

/// Error returned when a [`DataType`] cannot be promoted to another [`DataType`].
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum DataTypePromotionError {
    /// Error returned when attempting to compute a promoted [`DataType`] for an empty collection
    /// of [`DataType`]s (i.e., using [`DataType::promoted`]).
    #[error("Cannot construct a promoted data type from an empty collection of data types.")]
    Empty,

    /// Error returned when a [`DataType`] promotion fails due to incompatible data types.
    #[error("Cannot promote type `{lhs}` to type `{rhs}`.")]
    Incompatible { lhs: DataType, rhs: DataType },
}

/// Represents the size of an array dimension. Array dimensions can be either statically known at compilation time
/// or dynamic in which case their sizes will only be known at runtime at runtime. Dynamic dimensions may optionally
/// have an upper bound for their size that may be used for optimizations by the compiler. Note that by compilation
/// here we do not refer to the compilation of the Rust program but rather to the compilation of an array program
/// within our Rust library.
///
/// Note that the [`Display`] implementation of [`Size`] renders static sizes as just a number, dynamic sizes
/// with an upper bound as `<` followed by the upper bound, and dynamic sizes with no upper bound as `*`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Parameter)]
pub enum Size {
    /// Static size that is known at compilation time.
    Static(usize),

    /// Dynamic size that is not known until runtime and which has an optional upper bound. The upper bound,
    /// if present, represents an exclusive upper bound on the value that this size can have (i.e., the maximum
    /// possible value plus one). This can enable certain optimizations and static checks (though it is, of course,
    /// not as powerful as a static size).
    Dynamic(Option<usize>),
}

impl Size {
    /// Returns the value of this [`Size`] if it is a [`Size::Static`] and `None` otherwise.
    #[inline]
    pub fn value(&self) -> Option<usize> {
        match &self {
            Self::Static(size) => Some(*size),
            Self::Dynamic(_) => None,
        }
    }

    /// Returns an (exclusive) upper bound for the value of this [`Size`] if such a bound is known.
    /// For [`Size::Static`] sizes, this function will return the underlying value plus one as the upper bound.
    /// For [`Size::Dynamic`] sizes, this function will return the upper bound for that size, if one exists, and
    /// `None` otherwise.
    #[inline]
    pub fn upper_bound(&self) -> Option<usize> {
        match &self {
            Self::Static(size) => Some(*size + 1),
            Self::Dynamic(upper_bound) => *upper_bound,
        }
    }
}

impl Display for Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Static(size) => write!(f, "{size}"),
            Self::Dynamic(Some(upper_bound)) => write!(f, "<{upper_bound}"),
            Self::Dynamic(None) => write!(f, "*"),
        }
    }
}

impl From<usize> for Size {
    fn from(value: usize) -> Self {
        Self::Static(value)
    }
}

/// Represents the shape of an array (i.e., the number of dimensions in the array and the [`Size`] of each dimension).
///
/// Note that the [`Display`] implementation of [`Shape`] renders shapes as the rendered dimension sizes
/// in a comma-separated list surrounded by square brackets.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Parameter)]
pub struct Shape {
    /// [`Size`]s of the array dimensions ordered from outermost to innermost.
    pub dimensions: Vec<Size>,
}

impl Shape {
    /// Constructs a new [`Shape`] with the provided dimension [`Size`]s.
    #[inline]
    pub fn new(dimensions: Vec<Size>) -> Self {
        Self { dimensions }
    }

    /// Constructs a new scalar [`Shape`]. The resulting [`Shape::dimensions`] will be empty.
    #[inline]
    pub fn scalar() -> Self {
        Self::new(Vec::new())
    }

    /// Returns the rank (i.e., the number of dimensions) of this [`Shape`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::array_type::{Shape, Size};
    ///
    /// // Scalar.
    /// assert_eq!(Shape::scalar().rank(), 0);
    ///
    /// // Vector with 42 elements.
    /// assert_eq!(Shape::new(vec![Size::Static(42)]).rank(), 1);
    ///
    /// // Matrix with 42 rows and up to 10 columns.
    /// assert_eq!(Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]).rank(), 2);
    ///
    /// // Matrix with an unknown number of rows and 42 columns.
    /// assert_eq!(Shape::new(vec![Size::Dynamic(None), Size::Static(42)]).rank(), 2);
    /// ```
    #[inline]
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Returns the [`Size`] of the `index`-th dimension of this [`Shape`]. A negative `index` can be used to obtain
    /// dimension sizes using the end of the dimensions vector as the reference point. For example, an index value of
    /// `-1` will result in the last dimension (i.e., innermost) `Size` being returned.
    #[inline]
    pub fn dimension(&self, index: i32) -> Size {
        if index >= 0 {
            self.dimensions[index as usize]
        } else {
            self.dimensions[(self.dimensions.len() as i32 + index) as usize]
        }
    }

    /// Constructs a new [`Shape`] that is the "smallest" shape that all of the provided shapes can be broadcast to
    /// using [NumPy-like broadcasting semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html). Since
    /// this shape is the "smallest" such shape, we also know that its number of dimensions must match that of one of
    /// the provided [`Shape`]s).
    ///
    /// Note that this operation is *order-invariant* meaning that it will return the same [`Shape`] irrespective
    /// of the order in which the input [`Shape`]s are provided.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::array_type::{Shape, Size};
    /// let w = Shape::scalar();
    /// let x = Shape::new(vec![Size::Static(42), Size::Static(42)]);
    /// let y = Shape::new(vec![Size::Dynamic(Some(10))]);
    /// let z = Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]);
    ///
    /// assert_eq!(Shape::broadcast(&[&w]), Ok(w.clone()));
    /// assert_eq!(Shape::broadcast(&[&w, &y]), Ok(y.clone()));
    /// assert_eq!(Shape::broadcast(&[&w, &z]), Ok(z.clone()));
    /// assert_eq!(Shape::broadcast(&[&z, &y]), Ok(z.clone()));
    /// assert_eq!(Shape::broadcast(&[&w, &y, &z]), Ok(z.clone()));
    /// assert!(Shape::broadcast(&[&x, &w, &y, &z]).is_err());
    /// ```
    pub fn broadcast(shapes: &[&Self]) -> Result<Self, ShapeBroadcastingError> {
        if shapes.is_empty() {
            return Err(ShapeBroadcastingError::Empty);
        }

        shapes.iter().fold(Ok(Shape::scalar()), |lhs, rhs| match lhs {
            Ok(lhs) => {
                // Handle differing array ranks by (conceptually) padding the shorter shape with ones
                // on the left (i.e., as a prefix), up to the rank of the longer shape.
                let broadcast_rank = lhs.rank().max(rhs.rank());
                let lhs_offset = broadcast_rank - lhs.rank();
                let rhs_offset = broadcast_rank - rhs.rank();
                let mut broadcast_dimensions = Vec::with_capacity(broadcast_rank);
                for i in 0..broadcast_rank {
                    let lhs_size = if i < lhs_offset { Size::Static(1) } else { lhs.dimensions[i - lhs_offset] };
                    let rhs_size = if i < rhs_offset { Size::Static(1) } else { rhs.dimensions[i - rhs_offset] };
                    let broadcast_size = match (&lhs_size, &rhs_size) {
                        (_, Size::Static(1)) => lhs_size,
                        (Size::Static(1), _) => rhs_size,
                        (Size::Static(a), Size::Static(b)) if a == b => lhs_size,
                        (Size::Dynamic(a), Size::Dynamic(b)) if a == b => lhs_size,
                        _ => {
                            return Err(ShapeBroadcastingError::Incompatible { lhs: lhs.clone(), rhs: (*rhs).clone() });
                        }
                    };
                    broadcast_dimensions.push(broadcast_size);
                }

                Ok(Shape { dimensions: broadcast_dimensions })
            }
            Err(error) => Err(error),
        })
    }

    /// Broadcasts this [`Shape`] to the provided [`Shape`] using
    /// [NumPy-like semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html).
    ///
    /// Note that this operation is *not necessarily symmetric* meaning that `x.broadcast_to(y)` is not necessarily
    /// going to be equal `y.broadcast_to(x)` for all `x` and `y`. In fact, it is only going to be equal if `x == y`,
    /// in which case we will also have `x.broadcast_to(y) == y.broadcast_to(x) == x == y`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::array_type::{Shape, Size};
    /// let w = Shape::new(vec![Size::Static(42), Size::Static(42)]);
    /// let x = Shape::new(vec![]);
    /// let y = Shape::new(vec![Size::Dynamic(Some(10))]);
    /// let z = Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]);
    ///
    /// assert_eq!(x.broadcast_to(&x), Ok(x.clone()));
    /// assert_eq!(x.broadcast_to(&y), Ok(y.clone()));
    /// assert_eq!(x.broadcast_to(&z), Ok(z.clone()));
    /// assert!(z.broadcast_to(&y).is_err());
    /// assert_eq!(x.broadcast_to(&y).unwrap().broadcast_to(&z), Ok(z.clone()));
    /// assert_eq!(x.broadcast_to(&w), Ok(w.clone()));
    /// assert!(w.broadcast_to(&x).is_err());
    /// ```
    pub fn broadcast_to(&self, other: &Self) -> Result<Self, ShapeBroadcastingError> {
        if self.rank() > other.rank() {
            return Err(ShapeBroadcastingError::Incompatible { lhs: self.clone(), rhs: other.clone() });
        }

        // Handle differing array ranks by (conceptually) padding the dimension sizes of this shape
        // with ones on the left (i.e., as a prefix), up to the rank of the other shape.
        let broadcast_rank = other.rank();
        let offset = broadcast_rank - self.rank();
        let mut broadcast_shape = Vec::with_capacity(broadcast_rank);
        for i in 0..broadcast_rank {
            let lhs_size = if i < offset { Size::Static(1) } else { self.dimensions[i - offset] };
            let rhs_size = other.dimensions[i];
            let broadcast_size = match (&lhs_size, &rhs_size) {
                (Size::Static(1), _) => rhs_size,
                (Size::Static(a), Size::Static(b)) if a == b => rhs_size,
                (Size::Dynamic(a), Size::Dynamic(b)) if a == b => rhs_size,
                _ => return Err(ShapeBroadcastingError::Incompatible { lhs: self.clone(), rhs: other.clone() }),
            };
            broadcast_shape.push(broadcast_size);
        }

        Ok(Self { dimensions: broadcast_shape })
    }

    /// Returns `true` if this [`Shape`] can be broadcast to the provided [`Shape`], and `false` otherwise.
    /// Refer to the documentation of [`Shape::broadcast_to`] for more information on [`Shape`] broadcasting.
    pub fn broadcastable_to(&self, other: &Self) -> bool {
        if self.rank() > other.rank() {
            return false;
        }

        let broadcast_rank = other.rank();
        let offset = broadcast_rank - self.rank();
        for i in 0..broadcast_rank {
            let lhs_size = if i < offset { Size::Static(1) } else { self.dimensions[i - offset] };
            let rhs_size = other.dimensions[i];
            match (&lhs_size, &rhs_size) {
                (Size::Static(1), _) => continue,
                (Size::Static(a), Size::Static(b)) if a == b => continue,
                (Size::Dynamic(a), Size::Dynamic(b)) if a == b => continue,
                _ => return false,
            };
        }
        true
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.dimensions.iter().map(|dimension| dimension.to_string()).collect::<Vec<_>>().join(", "))
    }
}

/// Error returned when a [`Shape`] cannot be broadcast to another [`Shape`].
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ShapeBroadcastingError {
    /// Error returned when attempting to compute a broadcast [`Shape`] for an empty collection
    /// of [`Shape`]s (i.e., using [`Shape::broadcast`]).
    #[error("Cannot construct a broadcast shape from an empty collection of shapes.")]
    Empty,

    /// Error returned when a [`Shape`] broadcasting fails due to incompatible shapes.
    #[error("Cannot promote shape `{lhs}` to shape `{rhs}`.")]
    Incompatible { lhs: Shape, rhs: Shape },
}

/// Represents the type of a potentially multi-dimensional array.
///
/// Note that the [`Display`] implementation of [`ArrayType`] renders array types simply as their
/// [`DataType`]s followed by their [`Shape`]s.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::types::array_type::{ArrayType, DataType, Shape, Size};
///
/// // Boolean scalar.
/// assert_eq!(
///   ArrayType::new(DataType::Boolean, Shape::scalar()).to_string(),
///   "bool[]",
/// );
///
/// // 64-bit unsigned integer vector with 42 elements.
/// assert_eq!(
///   ArrayType::new(DataType::UnsignedInt64, Shape::new(vec![Size::Static(42)])).to_string(),
///   "u64[42]",
/// );
///
/// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
/// assert_eq!(
///   ArrayType::new(DataType::Float32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))])).to_string(),
///   "f32[42, <10]",
/// );
///
/// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
/// assert_eq!(
///   ArrayType::new(DataType::Complex64, Shape::new(vec![Size::Dynamic(None), Size::Static(42)])).to_string(),
///   "c64[*, 42]",
/// );
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Hash, Parameter)]
pub struct ArrayType {
    /// [`DataType`] of the elements stored in the array.
    pub data_type: DataType,

    /// [`Shape`] of the array.
    pub shape: Shape,
}

impl ArrayType {
    /// Constructs a new [`ArrayType`] with the provided [`DataType`] and [`Shape`].
    #[inline]
    pub fn new(data_type: DataType, shape: Shape) -> Self {
        Self { data_type, shape }
    }

    /// Constructs a new "scalar" [`ArrayType`] with the provided [`DataType`]. The resulting [`ArrayType::shape`]
    /// will be a scalar (i.e., have rank 0).
    #[inline]
    pub fn scalar(data_type: DataType) -> Self {
        Self::new(data_type, Shape::scalar())
    }

    /// Returns the rank (i.e., the number of dimensions) of this [`ArrayType`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::array_type::{ArrayType, DataType, Shape, Size};
    ///
    /// // Boolean scalar.
    /// assert_eq!(ArrayType::new(DataType::Boolean, Shape::scalar()).rank(), 0);
    ///
    /// // 64-bit unsigned integer vector with 42 elements.
    /// assert_eq!(ArrayType::new(DataType::UnsignedInt64, Shape::new(vec![Size::Static(42)])).rank(), 1);
    ///
    /// // 32-bit floating-point number matrix with 42 rows and up to 10 columns.
    /// assert_eq!(ArrayType::new(DataType::Float32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))])).rank(), 2);
    ///
    /// // 64-bit complex number matrix with an unknown number of rows and 42 columns.
    /// assert_eq!(ArrayType::new(DataType::Complex64, Shape::new(vec![Size::Dynamic(None), Size::Static(42)])).rank(), 2);
    /// ```
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the [`Size`] of the `index`-th dimension of this array type's [`Shape`]. A negative `index` can be used
    /// to obtain dimension sizes using the end of the dimensions vector as the reference point. For example, an index
    /// value of `-1` will result in the last dimension (i.e., innermost) `Size` being returned.
    #[inline]
    pub fn dimension(&self, index: i32) -> Size {
        self.shape.dimension(index)
    }

    /// Constructs a new [`ArrayType`] that is the narrowest array type that all of the provided array types can be
    /// broadcast to. That new array type has the following properties:
    ///
    ///   1. All of the provided array type [`DataType`]s can be promoted to its [`DataType`]. Refer to
    ///      [`DataType::promoted`] for more information on data type promotion.
    ///   2. All of the provided array type [`Shape`]s can be broadcast to its [`Shape`]. Refer to [`Shape::broadcast`]
    ///      for more information on shape broadcasting.
    ///
    /// Note that this operation is *order-invariant* meaning that it will return the same [`ArrayType`] irrespective
    /// of the order in which the input [`ArrayType`]s are provided.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::array_type::{ArrayType, DataType, Shape, Size};
    /// let w = ArrayType::new(DataType::Float32, Shape::new(vec![Size::Static(42), Size::Static(42)]));
    /// let x = ArrayType::new(DataType::Boolean, Shape::scalar());
    /// let y = ArrayType::new(DataType::UnsignedInt16, Shape::new(vec![Size::Dynamic(Some(10))]));
    /// let z = ArrayType::new(DataType::Float32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]));
    ///
    /// assert_eq!(ArrayType::broadcast(&[&x]), Ok(x.clone()));
    /// assert_eq!(ArrayType::broadcast(&[&x, &y]), Ok(y.clone()));
    /// assert_eq!(ArrayType::broadcast(&[&x, &z]), Ok(z.clone()));
    /// assert_eq!(ArrayType::broadcast(&[&z, &y]), Ok(z.clone()));
    /// assert_eq!(ArrayType::broadcast(&[&x, &y, &z]), Ok(z.clone()));
    /// assert!(ArrayType::broadcast(&[&w, &x, &y, &z]).is_err());
    /// ```
    pub fn broadcast(types: &[&Self]) -> Result<Self, ArrayTypeBroadcastingError> {
        if types.is_empty() {
            return Err(ArrayTypeBroadcastingError::Empty);
        }

        types.iter().fold(Ok(Self::scalar(DataType::Boolean)), |lhs, rhs| match lhs {
            Ok(lhs) => {
                let broadcast_data_type = DataType::promoted(&[&lhs.data_type, &rhs.data_type])?;
                let broadcast_shape = Shape::broadcast(&[&lhs.shape, &rhs.shape])?;
                Ok(Self { data_type: broadcast_data_type, shape: broadcast_shape })
            }
            Err(error) => Err(error),
        })
    }

    /// Broadcasts this [`ArrayType`] to the provided [`ArrayType`]. This consists of:
    ///
    ///   1. Promoting the underlying [`DataType`] using
    ///      [JAX-like promotion rules](https://docs.jax.dev/en/latest/type_promotion.html) to the provided
    ///      array type's [`DataType`].
    ///   2. Broadcasting the underlying dimension [`Size`]s using
    ///      [NumPy-like semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html) to the provided
    ///      array type's dimension [`Size`]s.
    ///
    /// Note that this operation is *not necessarily symmetric* meaning that `x.broadcast_to(y)` is not necessarily
    /// going to be equal `y.broadcast_to(x)` for all `x` and `y`. In fact, it is only going to be equal if `x == y`,
    /// in which case we will also have `x.broadcast_to(y) == y.broadcast_to(x) == x == y`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::types::array_type::{ArrayType, DataType, Shape, Size};
    /// let w = ArrayType::new(DataType::Float32, Shape::new(vec![Size::Static(42), Size::Static(42)]));
    /// let x = ArrayType::new(DataType::Boolean, Shape::scalar());
    /// let y = ArrayType::new(DataType::UnsignedInt16, Shape::new(vec![Size::Dynamic(Some(10))]));
    /// let z = ArrayType::new(DataType::Float32, Shape::new(vec![Size::Static(42), Size::Dynamic(Some(10))]));
    ///
    /// assert_eq!(x.broadcast_to(&x), Ok(x.clone()));
    /// assert_eq!(x.broadcast_to(&y), Ok(y.clone()));
    /// assert_eq!(x.broadcast_to(&z), Ok(z.clone()));
    /// assert!(z.broadcast_to(&y).is_err());
    /// assert_eq!(x.broadcast_to(&y).unwrap().broadcast_to(&z), Ok(z.clone()));
    /// assert_eq!(x.broadcast_to(&w), Ok(w.clone()));
    /// assert!(w.broadcast_to(&x).is_err());
    /// ```
    #[inline]
    pub fn broadcast_to(&self, other: &Self) -> Result<Self, ArrayTypeBroadcastingError> {
        let broadcast_data_type = self.data_type.promote_to(&other.data_type)?;
        let broadcast_shape = self.shape.broadcast_to(&other.shape)?;
        Ok(ArrayType { data_type: broadcast_data_type, shape: broadcast_shape })
    }

    /// Returns `true` if this [`ArrayType`] can be broadcast to the provided [`ArrayType`], and `false` otherwise.
    /// Refer to the documentation of [`ArrayType::broadcast_to`] for more information on [`ArrayType`] broadcasting.
    #[inline]
    pub fn broadcastable_to(&self, other: &Self) -> bool {
        self.data_type.promotable_to(&other.data_type) && self.shape.broadcastable_to(&other.shape)
    }
}

impl Type for ArrayType {
    /// Returns `true` if this [`ArrayType`] is a subtype of the provided [`ArrayType`], and `false` otherwise.
    ///
    /// An array type is considered a subtype of another if it is broadcastable to the other array type. For more
    /// information on broadcasting semantics for array types, refer to [`ArrayType::broadcastable_to`] and
    /// [`ArrayType::broadcast_to`].
    #[inline]
    fn is_subtype_of(&self, other: &Self) -> bool {
        self.broadcastable_to(&other)
    }
}

impl Display for ArrayType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.data_type, self.shape)
    }
}

/// Represents an [`Error`] related to [`ArrayType`] broadcasting. For more information on broadcasting,
/// refer to [`ArrayType::broadcast`], [`ArrayType::broadcast_to`], and [`ArrayType::broadcastable_to`].
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ArrayTypeBroadcastingError {
    /// Error returned when attempting to compute a broadcast [`ArrayType`] for an empty collection of
    /// [`ArrayType`]s (i.e., using [`ArrayType::broadcast`]).
    #[error("Cannot construct a broadcast array type from an empty collection of array types.")]
    Empty,

    /// Error returned when failing to broadcast two [`ArrayType`]s due to their [`DataType`]s not being
    /// compatible (i.e., the [`DataType`] of one cannot be promoted to the [`DataType`] of the other).
    #[error("{0}")]
    IncompatibleDataTypes(#[from] DataTypePromotionError),

    /// Error returned when failing to broadcast two [`ArrayType`]s due to their [`Shape`]s not being
    /// compatible (i.e., the [`Shape`] of one cannot be broadcast to the [`Shape`] of the other).
    #[error("{0}")]
    IncompatibleShapes(#[from] ShapeBroadcastingError),
}

#[cfg(test)]
mod tests {
    use super::*;

    use DataType::*;

    #[test]
    fn test_data_type_promoted() {
        assert_eq!(DataType::promoted(&[&Boolean]), Ok(Boolean));
        assert_eq!(DataType::promoted(&[&Boolean, &Complex64]), Ok(Complex64));
        assert_eq!(DataType::promoted(&[&Boolean, &Int2, &Complex64]), Ok(Complex64));
        assert_eq!(DataType::promoted(&[&Float32, &Int2, &BFloat16]), Ok(Float32));
        assert_eq!(DataType::promoted(&[&Float16, &BFloat16, &Float64]), Ok(Float64));
        assert_eq!(DataType::promoted(&[&Float4E2M1FN, &Float6E2M3FN]), Ok(Float6E2M3FN));

        assert!(DataType::promoted(&[]).is_err());
        assert!(DataType::promoted(&[&Float16, &BFloat16]).is_err());
        assert!(DataType::promoted(&[&Float8E3M4, &Float8E8M0FNU]).is_err());
    }

    #[test]
    fn test_data_type_promote_to() {
        assert_eq!(Boolean.promote_to(&Boolean), Ok(Boolean));
        assert_eq!(Boolean.promote_to(&Complex128), Ok(Complex128));
        assert_eq!(UnsignedInt4.promote_to(&Int8), Ok(Int8));
        assert_eq!(Float8E5M2FNUZ.promote_to(&BFloat16), Ok(BFloat16));
        assert_eq!(BFloat16.promote_to(&Float32), Ok(Float32));
        assert_eq!(Float64.promote_to(&Complex128), Ok(Complex128));

        assert!(UnsignedInt4.promote_to(&Int4).is_err());
        assert!(Int16.promote_to(&Boolean).is_err());
        assert!(Float6E2M3FN.promote_to(&Float4E2M1FN).is_err());
        assert!(Float6E2M3FN.promote_to(&Float6E3M2FN).is_err());
        assert!(Float64.promote_to(&Complex64).is_err());
    }

    #[test]
    fn test_data_type_promotable_to() {
        assert!(Boolean.promotable_to(&Float4E2M1FN));
        assert!(Boolean.promotable_to(&BFloat16));
        assert!(Boolean.promotable_to(&Complex128));
        assert!(UnsignedInt4.promotable_to(&UnsignedInt4));
        assert!(UnsignedInt4.promotable_to(&Int64));
        assert!(Float8E4M3B11FNUZ.promotable_to(&BFloat16));
        assert!(Float8E8M0FNU.promotable_to(&Float16));
        assert!(Float16.promotable_to(&Float32));
        assert!(Float32.promotable_to(&Complex64));

        assert!(!UnsignedInt8.promotable_to(&Int8));
        assert!(!Int2.promotable_to(&Boolean));
        assert!(!Float6E2M3FN.promotable_to(&UnsignedInt2));
        assert!(!Float6E3M2FN.promotable_to(&Float6E2M3FN));
        assert!(!Float8E4M3B11FNUZ.promotable_to(&Float4E2M1FN));
    }

    #[test]
    fn test_data_type_to_string() {
        assert_eq!(Boolean.to_string(), "bool");
        assert_eq!(UnsignedInt4.to_string(), "u4");
        assert_eq!(Int64.to_string(), "i64");
        assert_eq!(Float8E4M3FNUZ.to_string(), "f8e4m3fnuz");
        assert_eq!(BFloat16.to_string(), "bf16");
        assert_eq!(Float64.to_string(), "f64");
        assert_eq!(Complex128.to_string(), "c128");
    }

    #[test]
    fn test_size_value() {
        assert_eq!(Size::Static(1).value(), Some(1));
        assert_eq!(Size::Static(42).value(), Some(42));
        assert_eq!(Size::Dynamic(None).value(), None);
        assert_eq!(Size::Dynamic(Some(42)).value(), None);
    }

    #[test]
    fn test_size_upper_bound() {
        assert_eq!(Size::Static(1).upper_bound(), Some(2));
        assert_eq!(Size::Static(42).upper_bound(), Some(43));
        assert_eq!(Size::Dynamic(None).upper_bound(), None);
        assert_eq!(Size::Dynamic(Some(42)).upper_bound(), Some(42));
    }

    #[test]
    fn test_size_to_string() {
        assert_eq!(Size::Static(1).to_string(), "1");
        assert_eq!(Size::Static(42).to_string(), "42");
        assert_eq!(Size::Dynamic(None).to_string(), "*");
        assert_eq!(Size::Dynamic(Some(42)).to_string(), "<42");
    }

    #[test]
    fn test_shape_rank() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Dynamic(None)]);

        assert_eq!(s0.rank(), 0);
        assert_eq!(s1.rank(), 1);
        assert_eq!(s2.rank(), 2);
    }

    #[test]
    fn test_shape_dimension() {
        let s0 = Shape::new(vec![Size::Static(42)]);
        let s1 = Shape::new(vec![Size::Static(4), Size::Dynamic(None)]);

        assert_eq!(s0.dimension(0), Size::Static(42));
        assert_eq!(s1.dimension(1), Size::Dynamic(None));
        assert_eq!(s1.dimension(-2), Size::Static(4));
    }

    #[test]
    fn test_shape_broadcast() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        assert_eq!(Shape::broadcast(&[&s0]), Ok(s0.clone()));
        assert_eq!(Shape::broadcast(&[&s0, &s0]), Ok(s0.clone()));
        assert_eq!(Shape::broadcast(&[&s0, &s1]), Ok(s1.clone()));
        assert_eq!(Shape::broadcast(&[&s0, &s2]), Ok(s2.clone()));
        assert_eq!(Shape::broadcast(&[&s0, &s3]), Ok(s3.clone()));
        assert_eq!(Shape::broadcast(&[&s1, &s0]), Ok(s1.clone()));
        assert_eq!(Shape::broadcast(&[&s1, &s2]), Ok(s1.clone()));
        assert_eq!(Shape::broadcast(&[&s2, &s0]), Ok(s2.clone()));
        assert_eq!(Shape::broadcast(&[&s2, &s3]), Ok(s3.clone()));
        assert_eq!(Shape::broadcast(&[&s3, &s0]), Ok(s3.clone()));
        assert_eq!(Shape::broadcast(&[&s3, &s2]), Ok(s3.clone()));
        assert_eq!(Shape::broadcast(&[&s4, &s5]), Ok(s4.clone()));

        assert!(Shape::broadcast(&[]).is_err());
        assert!(Shape::broadcast(&[&s1, &s3]).is_err());
        assert!(Shape::broadcast(&[&s1, &s4]).is_err());
        assert!(Shape::broadcast(&[&s1, &s5]).is_err());
    }

    #[test]
    fn test_shape_broadcast_to() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        assert_eq!(s0.broadcast_to(&s0), Ok(s0.clone()));
        assert_eq!(s0.broadcast_to(&s1), Ok(s1.clone()));
        assert_eq!(s0.broadcast_to(&s2), Ok(s2.clone()));
        assert_eq!(s0.broadcast_to(&s3), Ok(s3.clone()));
        assert_eq!(s2.broadcast_to(&s1), Ok(s1.clone()));
        assert_eq!(s5.broadcast_to(&s4), Ok(s4.clone()));

        assert!(s1.broadcast_to(&s0).is_err());
        assert!(s1.broadcast_to(&s2).is_err());
        assert!(s1.broadcast_to(&s3).is_err());
        assert!(s1.broadcast_to(&s4).is_err());
        assert!(s1.broadcast_to(&s5).is_err());
        assert!(s2.broadcast_to(&s0).is_err());
        assert!(s3.broadcast_to(&s0).is_err());
        assert!(s4.broadcast_to(&s5).is_err());
    }

    #[test]
    fn test_shape_broadcastable_to() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        assert!(s0.broadcastable_to(&s0));
        assert!(s0.broadcastable_to(&s1));
        assert!(s0.broadcastable_to(&s2));
        assert!(s0.broadcastable_to(&s3));
        assert!(s2.broadcastable_to(&s1));
        assert!(s5.broadcastable_to(&s4));

        assert!(!s1.broadcastable_to(&s0));
        assert!(!s1.broadcastable_to(&s2));
        assert!(!s1.broadcastable_to(&s3));
        assert!(!s1.broadcastable_to(&s4));
        assert!(!s1.broadcastable_to(&s5));
        assert!(!s2.broadcastable_to(&s0));
        assert!(!s3.broadcastable_to(&s0));
        assert!(!s4.broadcastable_to(&s5));
    }

    #[test]
    fn test_shape_to_string() {
        let s0 = Shape::scalar();
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        assert_eq!(s0.to_string(), "[]");
        assert_eq!(s1.to_string(), "[42, 4, 2]");
        assert_eq!(s2.to_string(), "[4, 1]");
        assert_eq!(s3.to_string(), "[4, <1]");
        assert_eq!(s4.to_string(), "[*, 42, *]");
        assert_eq!(s5.to_string(), "[42, *]");
    }

    #[test]
    fn test_array_type_rank() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(Float32, s1);
        let t2 = ArrayType::new(Float8E3M4, s2);

        assert_eq!(t0.rank(), 0);
        assert_eq!(t1.rank(), 3);
        assert_eq!(t2.rank(), 2);
    }

    #[test]
    fn test_array_type_dimension() {
        let s0 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s1 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::new(Float32, s0);
        let t1 = ArrayType::new(Float8E3M4, s1);

        assert_eq!(t0.dimension(0), Size::Static(42));
        assert_eq!(t0.dimension(2), Size::Static(2));
        assert_eq!(t0.dimension(-2), Size::Static(4));
        assert_eq!(t1.dimension(0), Size::Static(42));
        assert_eq!(t1.dimension(1), Size::Dynamic(None));
        assert_eq!(t1.dimension(-1), Size::Dynamic(None));
    }

    #[test]
    fn test_array_type_broadcast() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(Float32, s1);
        let t2 = ArrayType::new(BFloat16, s2);
        let t3 = ArrayType::new(Float16, s3);
        let t4 = ArrayType::new(Complex64, s4);
        let t5 = ArrayType::new(Float8E4M3FN, s5);

        assert_eq!(ArrayType::broadcast(&[&t0]), Ok(t0.clone()));
        assert_eq!(ArrayType::broadcast(&[&t0, &t0]), Ok(t0.clone()));
        assert_eq!(ArrayType::broadcast(&[&t0, &t1]), Ok(t1.clone()));
        assert_eq!(ArrayType::broadcast(&[&t0, &t2]), Ok(t2.clone()));
        assert_eq!(ArrayType::broadcast(&[&t0, &t3]), Ok(t3.clone()));
        assert_eq!(ArrayType::broadcast(&[&t1, &t0]), Ok(t1.clone()));
        assert_eq!(ArrayType::broadcast(&[&t1, &t2]), Ok(t1.clone()));
        assert_eq!(ArrayType::broadcast(&[&t2, &t0]), Ok(t2.clone()));
        assert_eq!(ArrayType::broadcast(&[&t4, &t5]), Ok(t4.clone()));

        assert!(ArrayType::broadcast(&[]).is_err());
        assert!(ArrayType::broadcast(&[&t2, &t3]).is_err());
        assert!(ArrayType::broadcast(&[&t3, &t2]).is_err());
        assert!(ArrayType::broadcast(&[&t1, &t3]).is_err());
        assert!(ArrayType::broadcast(&[&t1, &t4]).is_err());
        assert!(ArrayType::broadcast(&[&t1, &t5]).is_err());
    }

    #[test]
    fn test_array_type_broadcast_to() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(Float32, s1);
        let t2 = ArrayType::new(BFloat16, s2);
        let t3 = ArrayType::new(Float16, s3);
        let t4 = ArrayType::new(Complex64, s4);
        let t5 = ArrayType::new(Float8E4M3FN, s5);

        assert_eq!(t0.broadcast_to(&t0), Ok(t0.clone()));
        assert_eq!(t0.broadcast_to(&t1), Ok(t1.clone()));
        assert_eq!(t0.broadcast_to(&t2), Ok(t2.clone()));
        assert_eq!(t0.broadcast_to(&t3), Ok(t3.clone()));
        assert_eq!(t2.broadcast_to(&t1), Ok(t1.clone()));
        assert_eq!(t5.broadcast_to(&t4), Ok(t4.clone()));

        assert!(t1.broadcast_to(&t0).is_err());
        assert!(t1.broadcast_to(&t2).is_err());
        assert!(t1.broadcast_to(&t3).is_err());
        assert!(t1.broadcast_to(&t4).is_err());
        assert!(t1.broadcast_to(&t5).is_err());
        assert!(t2.broadcast_to(&t3).is_err());
        assert!(t3.broadcast_to(&t0).is_err());
        assert!(t3.broadcast_to(&t2).is_err());
        assert!(t4.broadcast_to(&t5).is_err());
    }

    #[test]
    fn test_array_type_broadcastable_to() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(Float32, s1);
        let t2 = ArrayType::new(BFloat16, s2);
        let t3 = ArrayType::new(Float16, s3);
        let t4 = ArrayType::new(Complex64, s4);
        let t5 = ArrayType::new(Float8E4M3FN, s5);

        assert!(t0.broadcastable_to(&t0));
        assert!(t0.broadcastable_to(&t1));
        assert!(t0.broadcastable_to(&t2));
        assert!(t0.broadcastable_to(&t3));
        assert!(t2.broadcastable_to(&t1));
        assert!(t5.broadcastable_to(&t4));

        assert!(!t1.broadcastable_to(&t0));
        assert!(!t1.broadcastable_to(&t2));
        assert!(!t1.broadcastable_to(&t3));
        assert!(!t1.broadcastable_to(&t4));
        assert!(!t1.broadcastable_to(&t5));
        assert!(!t2.broadcastable_to(&t3));
        assert!(!t3.broadcastable_to(&t0));
        assert!(!t3.broadcastable_to(&t2));
        assert!(!t4.broadcastable_to(&t5));
    }

    #[test]
    fn test_array_type_to_string() {
        let s1 = Shape::new(vec![Size::Static(42), Size::Static(4), Size::Static(2)]);
        let s2 = Shape::new(vec![Size::Static(4), Size::Static(1)]);
        let s3 = Shape::new(vec![Size::Static(4), Size::Dynamic(Some(1))]);
        let s4 = Shape::new(vec![Size::Dynamic(None), Size::Static(42), Size::Dynamic(None)]);
        let s5 = Shape::new(vec![Size::Static(42), Size::Dynamic(None)]);

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(Float32, s1);
        let t2 = ArrayType::new(BFloat16, s2);
        let t3 = ArrayType::new(Float16, s3);
        let t4 = ArrayType::new(Complex64, s4);
        let t5 = ArrayType::new(Float8E4M3FN, s5);

        assert_eq!(t0.to_string(), "bool[]");
        assert_eq!(t1.to_string(), "f32[42, 4, 2]");
        assert_eq!(t2.to_string(), "bf16[4, 1]");
        assert_eq!(t3.to_string(), "f16[4, <1]");
        assert_eq!(t4.to_string(), "c64[*, 42, *]");
        assert_eq!(t5.to_string(), "f8e4m3fn[42, *]");
    }
}
