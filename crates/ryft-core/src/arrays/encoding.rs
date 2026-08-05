use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::num::FpCategory;

use half::{bf16, f16};
use num_complex::Complex;

use crate::arrays::addressing::ArrayAddressing;
use crate::programs::ProgramError;
use crate::programs::types::TypeError;
use crate::types::{ArrayType, DataType};

/// Value type with exactly one portable array-element byte encoding. [`ArrayElement`] pairs each supported Rust type
/// with the one [`DataType`] it represents (e.g., [`f32`] with [`DataType::F32`]) and pins the exact bytes that one
/// value of that type contributes to physical array storage, so that typed Rust slices convert to and from stored array
/// bytes deterministically on every platform:
///
///   - Booleans encode as one byte holding `0` or `1`.
///   - Sub-byte integers (i.e., [`i1`], [`i2`], [`i4`], [`u1`], [`u2`], and [`u4`]) encode as two's complement in the
///     low bits of one byte, with all higher bits set to zero.
///   - Integers encode as their little-endian two's-complement bytes.
///   - Low-precision floating-point values (e.g., [`f8e4m3fn`] and [`f4e2m1fn`]) encode as their one-byte sign,
///     exponent, and mantissa bit patterns, which occupy only the low bits of that byte for the four- and six-bit
///     formats, with all higher bits set to zero.
///   - Floating-point values (i.e., [`bf16`], [`f16`](struct@f16), [`f32`], and [`f64`]) encode as their little-endian
///     IEEE bit patterns, preserving signed zeros and NaN payload bits exactly.
///   - [`Complex`] values encode their real component immediately followed by their imaginary component.
///
/// For example, `-2i32` always encodes as `[254, 255, 255, 255]` and `Complex::new(1.5f32, -2.0)` always encodes as
/// `[0, 0, 192, 63, 0, 0, 0, 192]`, regardless of the host's native endianness. The compile-time [`DataType`] pairing
/// makes mismatched conversions fail before any bytes move: decoding an `f32[2]` array as [`i32`] values is an error
/// even though both encodings are four bytes wide.
///
/// The trait is sealed because its encodings define the array storage format itself and a foreign implementation could
/// write bytes that the checked storage boundary rejects, or worse, reinterprets with a different meaning. Sub-byte
/// integers, which have no corresponding Rust type of their own, are represented by the checked [`i1`], [`i2`], [`i4`],
/// [`u1`], [`u2`], and [`u4`] newtypes, which own the two's-complement-in-the-low-bits encoding described above (the
/// padded sub-byte layout of [DLPack v1.x](https://dmlc.github.io/dlpack/latest/index.html), which NumPy and JAX also
/// use). Low-precision floating-point formats are likewise represented by the conversion-only [`f4e2m1fn`],
/// [`f6e2m3fn`], [`f6e3m2fn`], [`f8e3m4`], [`f8e4m3`], [`f8e4m3fn`], [`f8e4m3fnuz`], [`f8e4m3b11fnuz`], [`f8e5m2`],
/// [`f8e5m2fnuz`], and [`f8e8m0fnu`] newtypes, whose exact bit layouts follow the
/// [DLPack v1.x](https://dmlc.github.io/dlpack/latest/index.html) data-type standard and whose
/// rounding conversions to and from [`f32`] and [`f64`] follow [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes),
/// the reference implementation of these formats.
pub trait ArrayElement: private::Codec {}

impl<T: private::Codec> ArrayElement for T {}

mod private {
    use crate::types::DataType;

    /// Private implementation contract that seals and implements [`super::ArrayElement`].
    pub trait Codec: Copy {
        /// [`DataType`] represented by this Rust value.
        const DATA_TYPE: DataType;

        /// Number of bytes in this value's portable representation.
        const BYTE_COUNT: usize;

        /// Writes this value's portable representation to an exactly sized byte slice.
        fn encode(self, bytes: &mut [u8]);

        /// Decodes one value from a byte slice whose length matches its representation.
        fn decode(bytes: &[u8]) -> Self;
    }
}

impl private::Codec for bool {
    const DATA_TYPE: DataType = DataType::Boolean;
    const BYTE_COUNT: usize = 1;

    #[inline]
    fn encode(self, bytes: &mut [u8]) {
        bytes[0] = u8::from(self);
    }

    #[inline]
    fn decode(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
}

/// Checked signed 1-bit integer array element representing [`DataType::I1`] values in the range `-1..=0`. The value is
/// stored as two's complement in the low bit of one storage byte, with all higher bits set to zero, so `-1` is stored
/// as the byte `0x01`. The wrapped native value stays sign-extended, so [`i1::value`] returns `-1` rather than `1` for
/// that byte. This unpacked one-element-per-byte representation is the padded sub-byte layout of
/// [DLPack v1.x](https://dmlc.github.io/dlpack/latest/index.html), which NumPy and JAX also use.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct i1(i8);

/// Checked signed 2-bit integer array element representing [`DataType::I2`] values in the range `-2..=1`. The value is
/// stored as two's complement in the low two bits of one storage byte, with all higher bits set to zero, so `-1` is
/// stored as the byte `0x03`. The wrapped native value stays sign-extended, so [`i2::value`] returns `-1` rather than
/// `3` for that byte. This unpacked one-element-per-byte representation is the padded sub-byte layout of
/// [DLPack v1.x](https://dmlc.github.io/dlpack/latest/index.html), which NumPy and JAX also use.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct i2(i8);

/// Checked signed 4-bit integer array element representing [`DataType::I4`] values in the range `-8..=7`. The value is
/// stored as two's complement in the low four bits of one storage byte, with all higher bits set to zero, so `-1` is
/// stored as the byte `0x0f`. The wrapped native value stays sign-extended, so [`i4::value`] returns `-1` rather than
/// `15` for that byte. This unpacked one-element-per-byte representation is the padded sub-byte layout of
/// [DLPack v1.x](https://dmlc.github.io/dlpack/latest/index.html), which NumPy and JAX also use.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct i4(i8);

// Implements the checked element API and the sealed codec for a signed sub-byte integer newtype that wraps its
// sign-extended native `i8` value and stores it as two's complement in the low `$bit_count` bits of one storage byte.
macro_rules! impl_codec_for_signed_sub_byte_integer_type {
    ($type:ident, $data_type:ident, $bit_count:literal) => {
        impl $type {
            /// Smallest representable value.
            pub const MIN: Self = Self(-(1i8 << ($bit_count - 1)));

            /// Largest representable value.
            pub const MAX: Self = Self((1i8 << ($bit_count - 1)) - 1);

            /// Bit mask selecting the storage bits that hold the two's complement value.
            const BIT_MASK: u8 = (1u8 << $bit_count) - 1;

            /// Creates a new element holding `value`, which must lie within the representable range.
            pub fn new(value: i8) -> Result<Self, TypeError> {
                if !(Self::MIN.0..=Self::MAX.0).contains(&value) {
                    return Err(TypeError::invalid(format!(
                        "value {} is out of range for {} array elements",
                        value,
                        DataType::$data_type,
                    )));
                }
                Ok(Self(value))
            }

            /// Returns the sign-extended native value of this element.
            #[inline]
            pub fn value(self) -> i8 {
                self.0
            }

            /// Creates a new element from one storage byte whose low bits hold a two's complement value.
            /// All higher bits must be zero.
            pub fn from_bits(bits: u8) -> Result<Self, TypeError> {
                if bits & !Self::BIT_MASK != 0 {
                    return Err(TypeError::invalid(format!(
                        "byte {:#04x} is not a valid {} array-element encoding",
                        bits,
                        DataType::$data_type,
                    )));
                }

                // Shift the value up to the sign bit and back down again to sign-extend it arithmetically.
                Ok(Self(((bits << (8 - $bit_count)) as i8) >> (8 - $bit_count)))
            }

            /// Returns this element's storage byte, holding its two's complement value in the low bits with all higher
            /// bits set to zero.
            #[inline]
            pub fn to_bits(self) -> u8 {
                self.0 as u8 & Self::BIT_MASK
            }
        }

        impl Display for $type {
            #[inline]
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}", self.0)
            }
        }

        impl private::Codec for $type {
            const DATA_TYPE: DataType = DataType::$data_type;
            const BYTE_COUNT: usize = 1;

            #[inline]
            fn encode(self, bytes: &mut [u8]) {
                bytes[0] = self.to_bits();
            }

            #[inline]
            fn decode(bytes: &[u8]) -> Self {
                // Storage bytes are validated before they are decoded, so their high bits are known to be zero.
                Self::from_bits(bytes[0]).unwrap()
            }
        }
    };
}

impl_codec_for_signed_sub_byte_integer_type!(i1, I1, 1);
impl_codec_for_signed_sub_byte_integer_type!(i2, I2, 2);
impl_codec_for_signed_sub_byte_integer_type!(i4, I4, 4);

// Implements the sealed codec for integer primitives using their exact little-endian two's-complement bit patterns.
macro_rules! impl_codec_for_integer_type {
    ($type:ty, $data_type:ident) => {
        impl private::Codec for $type {
            const DATA_TYPE: DataType = DataType::$data_type;
            const BYTE_COUNT: usize = size_of::<Self>();

            #[inline]
            fn encode(self, bytes: &mut [u8]) {
                bytes.copy_from_slice(&self.to_le_bytes());
            }

            #[inline]
            fn decode(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().unwrap())
            }
        }
    };
}

impl_codec_for_integer_type!(i8, I8);
impl_codec_for_integer_type!(i16, I16);
impl_codec_for_integer_type!(i32, I32);
impl_codec_for_integer_type!(i64, I64);

/// Checked unsigned 1-bit integer array element representing [`DataType::U1`] values in the range `0..=1`. The value
/// is stored in the low bit of one storage byte, with all higher bits set to zero. This unpacked one-element-per-byte
/// representation is the padded sub-byte layout of [DLPack v1.x](https://dmlc.github.io/dlpack/latest/index.html),
/// which NumPy and JAX also use.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct u1(u8);

/// Checked unsigned 2-bit integer array element representing [`DataType::U2`] values in the range `0..=3`. The value is
/// stored in the low two bits of one storage byte, with all higher bits set to zero. This unpacked one-element-per-byte
/// representation is the padded sub-byte layout of [DLPack v1.x](https://dmlc.github.io/dlpack/latest/index.html),
/// which NumPy and JAX also use.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct u2(u8);

/// Checked unsigned 4-bit integer array element representing [`DataType::U4`] values in the range `0..=15`.
/// The value is stored in the low four bits of one storage byte, with all higher bits set to zero. This unpacked
/// one-element-per-byte representation is the padded sub-byte layout of
/// [DLPack v1.x](https://dmlc.github.io/dlpack/latest/index.html), which NumPy and JAX also use.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct u4(u8);

// Implements the checked element API and the sealed codec for an unsigned sub-byte integer newtype that wraps its
// native `u8` value and stores it in the low `$bit_count` bits of one storage byte.
macro_rules! impl_unsigned_sub_byte_array_element {
    ($type:ident, $data_type:ident, $bit_count:literal) => {
        impl $type {
            /// Smallest representable value.
            pub const MIN: Self = Self(0);

            /// Largest representable value.
            pub const MAX: Self = Self(Self::BIT_MASK);

            /// Bit mask selecting the storage bits that hold the value.
            const BIT_MASK: u8 = (1u8 << $bit_count) - 1;

            /// Creates a new element holding `value`, which must lie within the representable range.
            pub fn new(value: u8) -> Result<Self, TypeError> {
                if value > Self::MAX.0 {
                    return Err(TypeError::invalid(format!(
                        "value {} is out of range for {} array elements",
                        value,
                        DataType::$data_type,
                    )));
                }
                Ok(Self(value))
            }

            /// Returns the native value of this element.
            #[inline]
            pub fn value(self) -> u8 {
                self.0
            }

            /// Creates a new element from one storage byte whose low bits hold the value. All higher bits must be zero.
            pub fn from_bits(bits: u8) -> Result<Self, TypeError> {
                if bits & !Self::BIT_MASK != 0 {
                    return Err(TypeError::invalid(format!(
                        "byte {:#04x} is not a valid {} array-element encoding",
                        bits,
                        DataType::$data_type,
                    )));
                }
                Ok(Self(bits))
            }

            /// Returns this element's storage byte, holding its value in the low bits with all higher bits set to zero.
            #[inline]
            pub fn to_bits(self) -> u8 {
                self.0
            }
        }

        impl Display for $type {
            #[inline]
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}", self.0)
            }
        }

        impl private::Codec for $type {
            const DATA_TYPE: DataType = DataType::$data_type;
            const BYTE_COUNT: usize = 1;

            #[inline]
            fn encode(self, bytes: &mut [u8]) {
                bytes[0] = self.to_bits();
            }

            #[inline]
            fn decode(bytes: &[u8]) -> Self {
                // Storage bytes are validated before they are decoded, so their high bits are known to be zero.
                Self::from_bits(bytes[0]).unwrap()
            }
        }
    };
}

impl_unsigned_sub_byte_array_element!(u1, U1, 1);
impl_unsigned_sub_byte_array_element!(u2, U2, 2);
impl_unsigned_sub_byte_array_element!(u4, U4, 4);

impl_codec_for_integer_type!(u8, U8);
impl_codec_for_integer_type!(u16, U16);
impl_codec_for_integer_type!(u32, U32);
impl_codec_for_integer_type!(u64, U64);

/// Special-value policy of a low-precision floating-point format. The class fixes which bit patterns a format reserves
/// for infinities and NaNs, whether it has a negative zero, and how a conversion whose rounded result falls beyond the
/// format's finite range is resolved.
#[derive(Copy, Clone, Debug)]
enum LowPrecisionFloatingPointFormatClass {
    /// IEEE-style format that reserves its all-ones exponent field. It represents a signed infinity with a zero
    /// mantissa, and a signed NaN with a nonzero mantissa. Overflowing conversions produce a signed infinity.
    Ieee,

    /// Finite format whose only reserved pattern is the all-ones exponent and mantissa, denoting a signed NaN, so it
    /// has no infinities. Overflowing conversions produce that NaN, carrying the input's sign.
    SignedNan,

    /// Finite format whose only reserved pattern is the sign-bit-only encoding that would otherwise be negative zero,
    /// denoting one unsigned NaN, so it has neither infinities nor a negative zero. Overflowing conversions produce
    /// that NaN.
    UnsignedNan,

    /// Finite format in which every bit pattern denotes a finite value, leaving room for neither infinities nor NaNs.
    /// Overflowing conversions saturate to the signed largest finite value and NaN inputs are rejected.
    NoNan,

    /// Sign-less format whose whole storage byte is an exponent field, so every encoding but the reserved `0xff` NaN
    /// denotes a positive power of two and the format has no zero, no negative value, and no infinity. Negative and
    /// overflowing conversions produce that NaN and zero inputs are rejected.
    ExponentOnly,
}

/// Descriptor of a low-precision floating-point format, driving the shared decoding and rounding engine behind every
/// low-precision floating-point element type in this module. A [`LowPrecisionFloatingPointFormat`] lays out one storage
/// byte as an optional sign bit above `exponent_bits` exponent bits above `mantissa_bits` mantissa bits. A zero
/// exponent field denotes a subnormal value `2^(1 - bias) * mantissa / 2^m` and any other exponent field `e` denotes
/// `2^(e - bias) * (1 + mantissa / 2^m)`, where `m` is `mantissa_bits`. The [`LowPrecisionFloatingPointFormatClass`]
/// then reinterprets the format's reserved patterns as infinities or NaNs.
#[derive(Copy, Clone, Debug)]
struct LowPrecisionFloatingPointFormat {
    /// [`DataType`] whose element encoding this [`LowPrecisionFloatingPointFormat`] describes.
    data_type: DataType,

    /// Number of exponent bits of this [`LowPrecisionFloatingPointFormat`].
    exponent_bits: u32,

    /// Number of mantissa bits of this [`LowPrecisionFloatingPointFormat`].
    mantissa_bits: u32,

    /// Amount subtracted from a stored exponent field to obtain the exponent it represents.
    bias: i32,

    /// [`LowPrecisionFloatingPointFormatClass`] of this [`LowPrecisionFloatingPointFormat`].
    class: LowPrecisionFloatingPointFormatClass,
}

impl LowPrecisionFloatingPointFormat {
    /// Bit mask selecting the sign bit, which is always zero for the sign-less
    /// [`LowPrecisionFloatingPointFormatClass::ExponentOnly`] format.
    const fn sign_mask(self) -> u8 {
        match self.class {
            LowPrecisionFloatingPointFormatClass::ExponentOnly => 0,
            _ => 1 << (self.exponent_bits + self.mantissa_bits),
        }
    }

    /// Bit mask selecting the magnitude bits, namely the exponent and mantissa fields.
    const fn magnitude_mask(self) -> u8 {
        match self.class {
            LowPrecisionFloatingPointFormatClass::ExponentOnly => u8::MAX,
            _ => self.sign_mask() - 1,
        }
    }

    /// Bit mask selecting every storage-byte bit that belongs to this format. Bits outside the mask must be zero,
    /// which only constrains the four- and six-bit formats.
    const fn bit_mask(self) -> u8 {
        self.sign_mask() | self.magnitude_mask()
    }

    /// Largest magnitude encoding that denotes a finite value. The encoding one past it is either reserved for a
    /// special value or outside the format, and evaluating the format's formula there yields the virtual overflow
    /// candidate that anchors rounding at the top of the range.
    const fn max_finite_magnitude(self) -> u8 {
        match self.class {
            LowPrecisionFloatingPointFormatClass::Ieee => {
                // The whole all-ones exponent field is reserved for infinities and NaNs.
                self.magnitude_mask() - (1 << self.mantissa_bits)
            }
            LowPrecisionFloatingPointFormatClass::SignedNan => {
                // Only the all-ones exponent and mantissa pattern is reserved, for the canonical NaN.
                self.magnitude_mask() - 1
            }
            LowPrecisionFloatingPointFormatClass::UnsignedNan | LowPrecisionFloatingPointFormatClass::NoNan => {
                // The reserved NaN pattern carries the sign bit, so every magnitude encoding is finite.
                self.magnitude_mask()
            }
            LowPrecisionFloatingPointFormatClass::ExponentOnly => {
                // Only `0xff` is reserved, for the single NaN.
                u8::MAX - 1
            }
        }
    }

    /// Encoding of the smallest representable value, which is the most negative finite value for every signed format
    /// and the smallest positive power of two for the sign-less [`LowPrecisionFloatingPointFormatClass::ExponentOnly`]
    /// format.
    const fn min_bits(self) -> u8 {
        match self.class {
            LowPrecisionFloatingPointFormatClass::ExponentOnly => 0,
            _ => self.max_finite_magnitude() | self.sign_mask(),
        }
    }

    /// Encoding whose exponent field is all ones and whose mantissa is zero, which denotes an infinity in the
    /// [`LowPrecisionFloatingPointFormatClass::Ieee`] formats.
    const fn infinity_bits(self, negative: bool) -> u8 {
        let exponent = (((1u16 << self.exponent_bits) - 1) << self.mantissa_bits) as u8;
        exponent | if negative { self.sign_mask() } else { 0 }
    }

    /// Canonical quiet-NaN encoding of this format, or `None` for the microscaling formats, which have no NaN
    /// encoding. The sign is honored only by the classes whose NaN encodings carry a sign bit.
    const fn nan_bits(self, negative: bool) -> Option<u8> {
        match self.class {
            LowPrecisionFloatingPointFormatClass::Ieee => {
                // A canonical quiet NaN sets the most significant mantissa bit under an all-ones exponent field.
                Some(self.infinity_bits(negative) | 1 << (self.mantissa_bits - 1))
            }
            LowPrecisionFloatingPointFormatClass::SignedNan => {
                Some(self.magnitude_mask() | if negative { self.sign_mask() } else { 0 })
            }
            LowPrecisionFloatingPointFormatClass::UnsignedNan => Some(self.sign_mask()),
            LowPrecisionFloatingPointFormatClass::NoNan => None,
            LowPrecisionFloatingPointFormatClass::ExponentOnly => Some(u8::MAX),
        }
    }

    /// Applies a sign to one finite magnitude encoding. The `fnuz` formats have no negative zero, so a negative zero
    /// collapses onto positive zero there instead of onto their NaN, which is the sign-bit-only encoding.
    fn signed_bits(self, magnitude: u8, negative: bool) -> u8 {
        let unsigned_zero = magnitude == 0 && matches!(self.class, LowPrecisionFloatingPointFormatClass::UnsignedNan);
        magnitude | if negative && !unsigned_zero { self.sign_mask() } else { 0 }
    }

    /// Encoding produced by a conversion whose rounded result falls beyond this format's finite range, following the
    /// overflow policy of its [`LowPrecisionFloatingPointFormatClass`].
    fn overflow_bits(self, negative: bool) -> u8 {
        match self.class {
            LowPrecisionFloatingPointFormatClass::Ieee => self.infinity_bits(negative),
            LowPrecisionFloatingPointFormatClass::NoNan => self.signed_bits(self.max_finite_magnitude(), negative),
            _ => {
                // The remaining classes have no infinity encoding, so an overflow lands on their canonical NaN,
                // which every one of them has.
                self.nan_bits(negative).unwrap()
            }
        }
    }

    /// Encoding of the value nearest to one finite nonzero `magnitude`, with `negative` applied to the result.
    /// Rounding scans every finite magnitude encoding of this format plus the virtual overflow candidate one step
    /// past its largest finite magnitude, which reproduces round-to-nearest-even exactly, including at the overflow
    /// boundary. A result beyond the finite range follows the overflow policy of this format's
    /// [`LowPrecisionFloatingPointFormatClass`].
    fn nearest_bits(self, magnitude: f64, negative: bool) -> u8 {
        let max_finite_magnitude = u16::from(self.max_finite_magnitude());
        // An input above the virtual overflow candidate is unambiguously an overflow. Resolving it before the scan
        // also keeps the distance comparisons below meaningful, because the distances from every candidate round to
        // the same value once the input is astronomically larger than this format's whole range.
        if magnitude > self.decode_magnitude(max_finite_magnitude + 1) {
            return self.overflow_bits(negative);
        }

        let mut nearest = 0;
        let mut nearest_distance = f64::INFINITY;
        for candidate in 0..=max_finite_magnitude + 1 {
            let distance = (self.decode_magnitude(candidate) - magnitude).abs();

            // Candidate values increase strictly with their encoding, so at most two candidates can be equidistant
            // from the input and exactly one of those two has an even encoding, which is the one to round to.
            if distance < nearest_distance || (distance == nearest_distance && candidate % 2 == 0) {
                nearest = candidate;
                nearest_distance = distance;
            }
        }

        if nearest > max_finite_magnitude {
            return self.overflow_bits(negative);
        }

        // A nonzero input can still round down onto the zero encoding, which the `fnuz` formats leave unsigned.
        self.signed_bits(nearest as u8, negative)
    }

    /// Exact value denoted by one magnitude encoding, evaluated purely from this format's exponent and
    /// mantissa formula. Reserved special-value patterns are not recognized, so passing one magnitude past
    /// [`LowPrecisionFloatingPointFormat::max_finite_magnitude`] yields the virtual overflow candidate
    /// used while rounding.
    fn decode_magnitude(self, magnitude: u16) -> f64 {
        let mantissa_scale = f64::from(1u32 << self.mantissa_bits);
        let mantissa = f64::from(u32::from(magnitude) & ((1u32 << self.mantissa_bits) - 1));
        let exponent = i32::from(magnitude >> self.mantissa_bits);
        if exponent == 0 && !matches!(self.class, LowPrecisionFloatingPointFormatClass::ExponentOnly) {
            // A zero exponent field denotes a subnormal value, whose implicit leading mantissa bit is zero.
            return 2f64.powi(1 - self.bias) * mantissa / mantissa_scale;
        }
        2f64.powi(exponent - self.bias) * (1.0 + mantissa / mantissa_scale)
    }

    /// Encodes the value nearest to `value` in this [`LowPrecisionFloatingPointFormat`], rounding to nearest with
    /// exact ties broken toward the even encoding. NaN, infinite, and zero inputs, along with finite inputs whose
    /// rounded result falls beyond the format's finite range, follow the policies of this format's
    /// [`LowPrecisionFloatingPointFormatClass`].
    fn encode(self, value: f64) -> Result<u8, TypeError> {
        let negative = value.is_sign_negative();
        let exponent_only = matches!(self.class, LowPrecisionFloatingPointFormatClass::ExponentOnly);
        match value.classify() {
            FpCategory::Nan => self
                .nan_bits(negative)
                .ok_or_else(|| TypeError::invalid(format!("data type {} cannot represent NaN", self.data_type))),
            FpCategory::Zero if exponent_only => {
                Err(TypeError::invalid(format!("data type {} cannot represent zero", self.data_type)))
            }
            FpCategory::Zero => Ok(self.signed_bits(0, negative)),
            FpCategory::Infinite => Ok(self.overflow_bits(negative)),
            FpCategory::Normal | FpCategory::Subnormal if negative && exponent_only => {
                // A sign-less format has no negative value to round to, so negative inputs collapse onto its NaN.
                Ok(self.nan_bits(negative).unwrap())
            }
            FpCategory::Normal | FpCategory::Subnormal => Ok(self.nearest_bits(value.abs(), negative)),
        }
    }

    /// Decodes the exact value denoted by one storage byte in this [`LowPrecisionFloatingPointFormat`].
    /// Bits outside this format's encoding are ignored.
    fn decode(self, bits: u8) -> f64 {
        let bits = bits & self.bit_mask();
        let negative = bits & self.sign_mask() != 0;
        let magnitude = bits & self.magnitude_mask();
        let mantissa = magnitude & ((1u8 << self.mantissa_bits) - 1);
        let exponent = u16::from(magnitude >> self.mantissa_bits);
        let is_special = match self.class {
            LowPrecisionFloatingPointFormatClass::Ieee => exponent == (1u16 << self.exponent_bits) - 1,
            LowPrecisionFloatingPointFormatClass::SignedNan => magnitude == self.magnitude_mask(),
            LowPrecisionFloatingPointFormatClass::UnsignedNan => negative && magnitude == 0,
            LowPrecisionFloatingPointFormatClass::NoNan => false,
            LowPrecisionFloatingPointFormatClass::ExponentOnly => bits == u8::MAX,
        };
        if is_special {
            return match self.class {
                LowPrecisionFloatingPointFormatClass::Ieee if mantissa == 0 && negative => f64::NEG_INFINITY,
                LowPrecisionFloatingPointFormatClass::Ieee if mantissa == 0 && !negative => f64::INFINITY,
                LowPrecisionFloatingPointFormatClass::Ieee | LowPrecisionFloatingPointFormatClass::SignedNan
                    if negative =>
                {
                    // Only the classes whose NaN encodings carry a sign bit have a negative NaN to decode.
                    -f64::NAN
                }
                _ => f64::NAN,
            };
        }
        let value = self.decode_magnitude(u16::from(magnitude));
        if negative { -value } else { value }
    }
}

/// Conversion-only 4-bit floating-point array element representing [`DataType::F4E2M1FN`] values. Values occupy the low
/// four bits of one storage byte, laid out as one sign bit, two exponent bits, and one mantissa bit with exponent bias
/// 1, and all higher bits of the byte are zero. Every encoding denotes a finite value. The largest finite value is `6`,
/// the smallest positive normal value is `1`, the smallest positive subnormal value is `0.5`, and negative zero is
/// `0x08`. This microscaling format has no infinity and no NaN encoding, so infinities and values that round beyond
/// `6` saturate to `±6`, while NaN inputs are rejected with a [`TypeError`]. Conversions round to nearest with ties
/// to even, following [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float4_e2m1fn).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f4e2m1fn(u8);

/// Conversion-only 6-bit floating-point array element representing [`DataType::F6E2M3FN`] values. Values occupy the low
/// six bits of one storage byte, laid out as one sign bit, two exponent bits, and three mantissa bits with exponent
/// bias 1, and all higher bits of the byte are zero. Every encoding denotes a finite value. The largest finite value
/// is `7.5`, the smallest positive normal value is `1`, the smallest positive subnormal value is `0.125`, and negative
/// zero is `0x20`. This microscaling format has no infinity and no NaN encoding, so infinities and values that round
/// beyond `7.5` saturate to `±7.5`, while NaN inputs are rejected with a [`TypeError`]. Conversions round to nearest
/// with ties to even, following [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float6_e2m3fn).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f6e2m3fn(u8);

/// Conversion-only 6-bit floating-point array element representing [`DataType::F6E3M2FN`] values. Values occupy the low
/// six bits of one storage byte, laid out as one sign bit, three exponent bits, and two mantissa bits with exponent
/// bias 3, and all higher bits of the byte are zero. Every encoding denotes a finite value. The largest finite value
/// is `28`, the smallest positive normal value is `0.25`, the smallest positive subnormal value is `0.0625`, and
/// negative zero is `0x20`. This microscaling format has no infinity and no NaN encoding, so infinities and values
/// that round beyond `28` saturate to `±28`, while NaN inputs are rejected with a [`TypeError`]. Conversions round
/// to nearest with ties to even, following [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float6_e3m2fn).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f6e3m2fn(u8);

/// Conversion-only 8-bit floating-point array element representing [`DataType::F8E3M4`] values. Values fill one storage
/// byte laid out as one sign bit, three exponent bits, and four mantissa bits with exponent bias 3. The largest finite
/// value is `15.5`, the smallest positive normal value is `2^-2`, the smallest positive subnormal value is `2^-6`, and
/// negative zero is `0x80`. This IEEE-style format reserves its all-ones exponent field. `0x70` and `0xf0` are the
/// infinities, and any nonzero mantissa under that exponent is a NaN, whose canonical quiet encoding is `0x78` (`0xf8`
/// when negative). Infinities and values that round beyond `15.5` therefore convert to `±∞`. Conversions round to
/// nearest with ties to even, following [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float8_e3m4).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f8e3m4(u8);

/// Conversion-only 8-bit floating-point array element representing [`DataType::F8E4M3`] values. Values fill one storage
/// byte laid out as one sign bit, four exponent bits, and three mantissa bits with exponent bias 7. The largest finite
/// value is `240`, the smallest positive normal value is `2^-6`, the smallest positive subnormal value is `2^-9`, and
/// negative zero is `0x80`. This IEEE-style format reserves its all-ones exponent field. `0x78` and `0xf8` are the
/// infinities, and any nonzero mantissa under that exponent is a NaN, whose canonical quiet encoding is `0x7c` (`0xfc`
/// when negative). Infinities and values that round beyond `240` therefore convert to `±∞`. Conversions round to
/// nearest with ties to even, following [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float8_e4m3).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f8e4m3(u8);

/// Conversion-only 8-bit floating-point array element representing [`DataType::F8E4M3FN`] values. Values fill one
/// storage byte laid out as one sign bit, four exponent bits, and three mantissa bits with exponent bias 7. The largest
/// finite value is `448`, the smallest positive normal value is `2^-6`, the smallest positive subnormal value is
/// `2^-9`, and negative zero is `0x80`. The format is finite. It has no infinity encoding, and its single NaN is the
/// all-ones exponent and mantissa pattern `0x7f` (`0xff` when negative). Infinities and values that round beyond `448`,
/// meaning past the midpoint `464` between `448` and the next formula step `480`, therefore convert to that NaN,
/// carrying the input's sign. Conversions round to nearest with ties to even, following
/// [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float8_e4m3fn).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f8e4m3fn(u8);

/// Conversion-only 8-bit floating-point array element representing [`DataType::F8E4M3FNUZ`] values. Values fill one
/// storage byte laid out as one sign bit, four exponent bits, and three mantissa bits with exponent bias 8. The largest
/// finite value is `240`, the smallest positive normal value is `2^-7`, and the smallest positive subnormal value is
/// `2^-10`. The format is finite with unsigned zero. It has no infinity encoding and no negative zero, because `0x80`,
/// the pattern that would be negative zero, is its single NaN. Negative zero inputs therefore encode as positive zero,
/// while infinities and values that round beyond `240` convert to that NaN. Conversions round to nearest with ties to
/// even, following [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float8_e4m3fnuz).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f8e4m3fnuz(u8);

/// Conversion-only 8-bit floating-point array element representing [`DataType::F8E4M3B11FNUZ`] values. Values fill one
/// storage byte laid out as one sign bit, four exponent bits, and three mantissa bits with exponent bias 11, which
/// trades range for precision near zero relative to [`f8e4m3fnuz`]. The largest finite value is `30`, the smallest
/// positive normal value is `2^-10`, and the smallest positive subnormal value is `2^-13`. The format is finite with
/// unsigned zero. It has no infinity encoding and no negative zero, because `0x80`, the pattern that would be negative
/// zero, is its single NaN. Negative zero inputs therefore encode as positive zero, while infinities and values that
/// round beyond `30` convert to that NaN. Conversions round to nearest with ties to even, following
/// [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float8_e4m3b11fnuz).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f8e4m3b11fnuz(u8);

/// Conversion-only 8-bit floating-point array element representing [`DataType::F8E5M2`] values. Values fill one storage
/// byte laid out as one sign bit, five exponent bits, and two mantissa bits with exponent bias 15, which is the
/// truncation of [`f16`](struct@f16) to eight bits. The largest finite value is `57344`, the smallest positive normal
/// value is `2^-14`, the smallest positive subnormal value is `2^-16`, and negative zero is `0x80`. This IEEE-style
/// format reserves its all-ones exponent field. `0x7c` and `0xfc` are the infinities, and any nonzero mantissa under
/// that exponent is a NaN, whose canonical quiet encoding is `0x7e` (`0xfe` when negative). Infinities and values that
/// round beyond `57344` therefore convert to `±∞`. Conversions round to nearest with ties to even, following
/// [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float8_e5m2).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f8e5m2(u8);

/// Conversion-only 8-bit floating-point array element representing [`DataType::F8E5M2FNUZ`] values. Values fill one
/// storage byte laid out as one sign bit, five exponent bits, and two mantissa bits with exponent bias 16. The largest
/// finite value is `57344`, the smallest positive normal value is `2^-15`, and the smallest positive subnormal value is
/// `2^-17`. The format is finite with unsigned zero: it has no infinity encoding and no negative zero, because `0x80`,
/// the pattern that would be negative zero, is its single NaN. Negative zero inputs therefore encode as positive zero,
/// while infinities and values that round beyond `57344` convert to that NaN. Conversions round to nearest with ties to
/// even, following [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float8_e5m2fnuz).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f8e5m2fnuz(u8);

/// Conversion-only 8-bit floating-point array element representing [`DataType::F8E8M0FNU`] values, the block scale
/// factors of the microscaling formats. The whole storage byte is an eight-bit exponent field with bias 127 and there
/// are no sign or mantissa bits, so every encoding `e` in `0x00..=0xfe` denotes exactly `2^(e - 127)`, spanning
/// `2^-127` through `2^127`, and `0xff` is the single NaN. The format has no zero, no negative value, and no infinity,
/// so [`f8e8m0fnu::MIN`] is the smallest positive value `2^-127` rather than a negative one, zero inputs are rejected
/// with a [`TypeError`], and negative, infinite, and overflowing inputs convert to NaN. Conversions round to nearest
/// with ties to even, so `1.5` and `3` both round to `2` while `6` rounds to `8`, following
/// [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes#float8_e8m0fnu).
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Debug)]
pub struct f8e8m0fnu(u8);

// Implements the shared conversion API and the numeric comparison semantics of a low-precision floating-point newtype
// that wraps its storage byte, driving every conversion through the shared `FloatFormat` engine.
macro_rules! impl_low_precision_floating_point_type {
    ($type:ident, $data_type:ident, $class:ident, $exponent_bits:literal, $mantissa_bits:literal, $bias:literal) => {
        impl $type {
            /// Format descriptor driving every conversion of this element type.
            const FORMAT: LowPrecisionFloatingPointFormat = LowPrecisionFloatingPointFormat {
                data_type: DataType::$data_type,
                exponent_bits: $exponent_bits,
                mantissa_bits: $mantissa_bits,
                bias: $bias,
                class: LowPrecisionFloatingPointFormatClass::$class,
            };

            /// Smallest representable value.
            pub const MIN: Self = Self(Self::FORMAT.min_bits());

            /// Largest representable finite value.
            pub const MAX: Self = Self(Self::FORMAT.max_finite_magnitude());

            /// Returns this element's storage byte.
            #[inline]
            pub fn to_bits(self) -> u8 {
                self.0
            }

            /// Rounds `value` to the nearest representable value, breaking exact ties toward the even encoding.
            /// Returns an error only when this format has no encoding for the input at all, namely a NaN input
            /// to a microscaling format or a zero input to [`f8e8m0fnu`].
            #[inline]
            pub fn from_f32(value: f32) -> Result<Self, TypeError> {
                // Widening to `f64` is exact, so the `f64` conversion performs the only rounding step.
                Self::from_f64(f64::from(value))
            }

            /// Rounds `value` to the nearest representable value, breaking exact ties toward the even encoding.
            /// Returns an error only when this format has no encoding for the input at all, namely a NaN input to a
            /// microscaling format or a zero input to [`f8e8m0fnu`].
            #[inline]
            pub fn from_f64(value: f64) -> Result<Self, TypeError> {
                Self::FORMAT.encode(value).map(Self)
            }

            /// Returns this element's exact value as an [`f32`] value.
            #[inline]
            pub fn to_f32(self) -> f32 {
                self.to_f64() as f32
            }

            /// Returns this element's exact value as an [`f64`] value.
            #[inline]
            pub fn to_f64(self) -> f64 {
                Self::FORMAT.decode(self.0)
            }
        }

        impl Display for $type {
            #[inline]
            fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}", self.to_f64())
            }
        }

        impl PartialEq for $type {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.to_f64() == other.to_f64()
            }
        }

        impl PartialOrd for $type {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.to_f64().partial_cmp(&other.to_f64())
            }
        }
    };
}

// Implements the NaN vocabulary of a low-precision floating-point newtype whose format reserves a NaN encoding.
macro_rules! impl_nan_for_low_precision_floating_point_type {
    ($type:ident) => {
        impl $type {
            /// Canonical NaN value of this format.
            pub const NAN: Self = Self(match Self::FORMAT.nan_bits(false) {
                Some(bits) => bits,
                // This constant is only generated for formats that reserve a NaN encoding.
                None => unreachable!(),
            });

            /// Returns whether this element is NaN.
            #[inline]
            pub fn is_nan(self) -> bool {
                self.to_f64().is_nan()
            }
        }
    };
}

// Implements the infinity vocabulary of an IEEE-style low-precision floating-point newtype.
macro_rules! impl_infinities_for_low_precision_floating_point_type {
    ($type:ident) => {
        impl $type {
            /// Positive infinity value of this format.
            pub const INFINITY: Self = Self(Self::FORMAT.infinity_bits(false));

            /// Negative infinity value of this format.
            pub const NEG_INFINITY: Self = Self(Self::FORMAT.infinity_bits(true));
        }
    };
}

// Implements the storage-byte constructor and the sealed codec for a low-precision floating-point newtype whose
// encoding fills a whole storage byte, so that every byte is a valid encoding.
macro_rules! impl_codec_for_byte_low_precision_floating_point_type {
    ($type:ident) => {
        impl $type {
            /// Creates a new element from its storage byte. Every byte encodes a value of this format.
            #[inline]
            pub fn from_bits(bits: u8) -> Self {
                Self(bits)
            }
        }

        impl private::Codec for $type {
            const DATA_TYPE: DataType = Self::FORMAT.data_type;
            const BYTE_COUNT: usize = 1;

            #[inline]
            fn encode(self, bytes: &mut [u8]) {
                bytes[0] = self.to_bits();
            }

            #[inline]
            fn decode(bytes: &[u8]) -> Self {
                Self::from_bits(bytes[0])
            }
        }
    };
}

// Implements the checked storage-byte constructor and the sealed codec for a low-precision floating-point newtype
// whose encoding occupies only the low bits of a storage byte, requiring all higher bits to be zero.
macro_rules! impl_codec_for_sub_byte_low_precision_floating_point_type {
    ($type:ident) => {
        impl $type {
            /// Creates a new element from one storage byte whose low bits hold its encoding.
            /// All higher bits must be zero.
            pub fn from_bits(bits: u8) -> Result<Self, TypeError> {
                if bits & !Self::FORMAT.bit_mask() != 0 {
                    return Err(TypeError::invalid(format!(
                        "byte {bits:#04x} is not a valid {} array-element encoding",
                        Self::FORMAT.data_type,
                    )));
                }
                Ok(Self(bits))
            }
        }

        impl private::Codec for $type {
            const DATA_TYPE: DataType = Self::FORMAT.data_type;
            const BYTE_COUNT: usize = 1;

            #[inline]
            fn encode(self, bytes: &mut [u8]) {
                bytes[0] = self.to_bits();
            }

            #[inline]
            fn decode(bytes: &[u8]) -> Self {
                // Storage bytes are validated before they are decoded, so their high bits are known to be zero.
                Self::from_bits(bytes[0]).unwrap()
            }
        }
    };
}

impl_low_precision_floating_point_type!(f4e2m1fn, F4E2M1FN, NoNan, 2, 1, 1);
impl_codec_for_sub_byte_low_precision_floating_point_type!(f4e2m1fn);

impl_low_precision_floating_point_type!(f6e2m3fn, F6E2M3FN, NoNan, 2, 3, 1);
impl_codec_for_sub_byte_low_precision_floating_point_type!(f6e2m3fn);

impl_low_precision_floating_point_type!(f6e3m2fn, F6E3M2FN, NoNan, 3, 2, 3);
impl_codec_for_sub_byte_low_precision_floating_point_type!(f6e3m2fn);

impl_low_precision_floating_point_type!(f8e3m4, F8E3M4, Ieee, 3, 4, 3);
impl_codec_for_byte_low_precision_floating_point_type!(f8e3m4);
impl_nan_for_low_precision_floating_point_type!(f8e3m4);
impl_infinities_for_low_precision_floating_point_type!(f8e3m4);

impl_low_precision_floating_point_type!(f8e4m3, F8E4M3, Ieee, 4, 3, 7);
impl_codec_for_byte_low_precision_floating_point_type!(f8e4m3);
impl_nan_for_low_precision_floating_point_type!(f8e4m3);
impl_infinities_for_low_precision_floating_point_type!(f8e4m3);

impl_low_precision_floating_point_type!(f8e4m3fn, F8E4M3FN, SignedNan, 4, 3, 7);
impl_codec_for_byte_low_precision_floating_point_type!(f8e4m3fn);
impl_nan_for_low_precision_floating_point_type!(f8e4m3fn);

impl_low_precision_floating_point_type!(f8e4m3fnuz, F8E4M3FNUZ, UnsignedNan, 4, 3, 8);
impl_codec_for_byte_low_precision_floating_point_type!(f8e4m3fnuz);
impl_nan_for_low_precision_floating_point_type!(f8e4m3fnuz);

impl_low_precision_floating_point_type!(f8e4m3b11fnuz, F8E4M3B11FNUZ, UnsignedNan, 4, 3, 11);
impl_codec_for_byte_low_precision_floating_point_type!(f8e4m3b11fnuz);
impl_nan_for_low_precision_floating_point_type!(f8e4m3b11fnuz);

impl_low_precision_floating_point_type!(f8e5m2, F8E5M2, Ieee, 5, 2, 15);
impl_codec_for_byte_low_precision_floating_point_type!(f8e5m2);
impl_nan_for_low_precision_floating_point_type!(f8e5m2);
impl_infinities_for_low_precision_floating_point_type!(f8e5m2);

impl_low_precision_floating_point_type!(f8e5m2fnuz, F8E5M2FNUZ, UnsignedNan, 5, 2, 16);
impl_codec_for_byte_low_precision_floating_point_type!(f8e5m2fnuz);
impl_nan_for_low_precision_floating_point_type!(f8e5m2fnuz);

impl_low_precision_floating_point_type!(f8e8m0fnu, F8E8M0FNU, ExponentOnly, 8, 0, 127);
impl_codec_for_byte_low_precision_floating_point_type!(f8e8m0fnu);
impl_nan_for_low_precision_floating_point_type!(f8e8m0fnu);

// Implements the sealed codec for real floating-point primitives while preserving their exact payload bits.
macro_rules! impl_codec_for_floating_point_type {
    ($type:ty, $bits:ty, $data_type:ident) => {
        impl private::Codec for $type {
            const DATA_TYPE: DataType = DataType::$data_type;
            const BYTE_COUNT: usize = size_of::<$bits>();

            #[inline]
            fn encode(self, bytes: &mut [u8]) {
                bytes.copy_from_slice(&self.to_bits().to_le_bytes());
            }

            #[inline]
            fn decode(bytes: &[u8]) -> Self {
                Self::from_bits(<$bits>::from_le_bytes(bytes.try_into().unwrap()))
            }
        }
    };
}

impl_codec_for_floating_point_type!(bf16, u16, BF16);
impl_codec_for_floating_point_type!(f16, u16, F16);
impl_codec_for_floating_point_type!(f32, u32, F32);
impl_codec_for_floating_point_type!(f64, u64, F64);

impl private::Codec for Complex<f32> {
    const DATA_TYPE: DataType = DataType::C64;
    const BYTE_COUNT: usize = 8;

    #[inline]
    fn encode(self, bytes: &mut [u8]) {
        self.re.encode(&mut bytes[..4]);
        self.im.encode(&mut bytes[4..]);
    }

    #[inline]
    fn decode(bytes: &[u8]) -> Self {
        Self::new(<f32 as private::Codec>::decode(&bytes[..4]), <f32 as private::Codec>::decode(&bytes[4..]))
    }
}

impl private::Codec for Complex<f64> {
    const DATA_TYPE: DataType = DataType::C128;
    const BYTE_COUNT: usize = 16;

    #[inline]
    fn encode(self, bytes: &mut [u8]) {
        self.re.encode(&mut bytes[..8]);
        self.im.encode(&mut bytes[8..]);
    }

    #[inline]
    fn decode(bytes: &[u8]) -> Self {
        Self::new(<f64 as private::Codec>::decode(&bytes[..8]), <f64 as private::Codec>::decode(&bytes[8..]))
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Encodes the provided typed elements in logical row-major order into a new physical storage buffer described by
/// `r#type`.
/// Explicit strided and tiled layouts determine where each element is written. Bytes not occupied by logical elements,
/// including layout holes and tile padding, are initialized to zero.
///
/// # Parameters
///
///   - `r#type`: static array type that determines the element data type, shape, and physical layout.
///   - `elements`: logical row-major elements. Their type must represent `r#type`'s [`DataType`], and their count must
///     equal the array's logical element count.
///
/// # Errors
///
/// Returns an error if `r#type` cannot describe materialized storage, if `T` represents a different [`DataType`], or
/// if `elements` has the wrong length.
pub fn encode_elements<T: ArrayElement>(r#type: &ArrayType, elements: &[T]) -> Result<Vec<u8>, ProgramError> {
    if r#type.data_type() != T::DATA_TYPE {
        return Err(TypeError::invalid(format!(
            "cannot encode {} values as array elements of data type {}",
            T::DATA_TYPE,
            r#type.data_type(),
        ))
        .into());
    }
    let addressing = ArrayAddressing::new(r#type.clone())?;
    if elements.len() != addressing.element_count() {
        return Err(TypeError::invalid(format!(
            "array type {} requires {} logical elements but got {}",
            r#type,
            addressing.element_count(),
            elements.len(),
        ))
        .into());
    }
    debug_assert_eq!(T::BYTE_COUNT, addressing.element_byte_width());

    // Starting from zero establishes the required contents of layout holes and tile padding without tracking them
    // separately from the element ranges that are written below.
    let mut storage = vec![0; addressing.storage_byte_len()];
    if addressing.is_dense_row_major() {
        // Dense row-major storage has the same ordering as `elements`, so it needs no per-element address lookup.
        for (element, bytes) in elements.iter().zip(storage.chunks_exact_mut(T::BYTE_COUNT)) {
            element.encode(bytes);
        }
    } else {
        // Explicit layouts scatter each logical row-major element into its physical storage range.
        for (index, element) in elements.iter().enumerate() {
            element.encode(&mut storage[addressing.byte_range_for_flat_index(index)]);
        }
    }
    Ok(storage)
}

/// Converts validated logical element bytes in row-major order into a new physical storage buffer described by
/// `r#type`. The input contains only encoded logical elements; it does not include layout holes or tile padding.
/// Explicit layouts determine where each element is written, and all unoccupied storage bytes are initialized to zero.
///
/// # Parameters
///
///   - `r#type`: static array type that determines the element encoding, logical shape, and physical layout.
///   - `bytes`: concatenated logical element encodings in row-major order.
///
/// # Errors
///
/// Returns an error if `r#type` cannot describe materialized storage, if `bytes` has the wrong length, or if any
/// element is not a valid encoding of `r#type`'s [`DataType`].
pub fn encode_logical_bytes(r#type: &ArrayType, bytes: &[u8]) -> Result<Vec<u8>, ProgramError> {
    let addressing = ArrayAddressing::new(r#type.clone())?;
    if bytes.len() != addressing.logical_byte_len() {
        return Err(TypeError::invalid(format!(
            "array type {} requires {} logical element bytes but got {}",
            r#type,
            addressing.logical_byte_len(),
            bytes.len(),
        ))
        .into());
    }

    // Validate the portable element encodings before their logical boundaries are obscured by physical placement.
    let element_byte_width = addressing.element_byte_width();
    validate_logical_bytes(r#type.data_type(), element_byte_width, bytes)?;
    if addressing.is_dense_row_major() {
        // Logical and physical order coincide, and dense storage contains no holes or padding to initialize.
        return Ok(bytes.to_vec());
    }

    // Zero initialization establishes all unoccupied storage bytes before logical elements are scattered into place.
    let mut storage = vec![0; addressing.storage_byte_len()];
    for element in 0..addressing.element_count() {
        let logical_start = element * element_byte_width;
        let logical_range = logical_start..logical_start + element_byte_width;
        storage[addressing.byte_range_for_flat_index(element)].copy_from_slice(&bytes[logical_range]);
    }
    Ok(storage)
}

/// Validates physical array storage and decodes its elements as `T` in logical row-major order. Physical layout order,
/// holes, and tile padding are not exposed in the returned vector.
///
/// # Parameters
///
///   - `r#type`: static array type that describes the supplied physical storage.
///   - `bytes`: physical storage bytes, including any holes or tile padding required by `r#type`.
///
/// # Errors
///
/// Returns an error if `r#type` cannot describe materialized storage, if `T` represents a different [`DataType`], or
/// if `bytes` is not valid physical storage for `r#type`.
pub fn decode_elements<T: ArrayElement>(r#type: &ArrayType, bytes: &[u8]) -> Result<Vec<T>, ProgramError> {
    if r#type.data_type() != T::DATA_TYPE {
        return Err(TypeError::invalid(format!(
            "cannot decode array elements of data type {} as {} values",
            r#type.data_type(),
            T::DATA_TYPE,
        ))
        .into());
    }
    let addressing = ArrayAddressing::new(r#type.clone())?;

    // Validation makes every subsequent codec call infallible and rejects nonzero holes or tile padding.
    validate_storage_bytes_with_addressing(&addressing, bytes)?;
    debug_assert_eq!(T::BYTE_COUNT, addressing.element_byte_width());
    if addressing.is_dense_row_major() {
        // Dense storage can be decoded sequentially without consulting the addressing descriptor per element.
        return Ok(bytes.chunks_exact(T::BYTE_COUNT).map(T::decode).collect());
    }

    // Gather explicit physical layouts back into logical row-major order.
    Ok((0..addressing.element_count())
        .map(|element| T::decode(&bytes[addressing.byte_range_for_flat_index(element)]))
        .collect())
}

/// Validates physical array storage and returns its encoded elements as contiguous logical row-major bytes. Layout
/// holes and tile padding must contain zero and are omitted from the returned representation.
pub fn decode_logical_bytes(r#type: &ArrayType, bytes: &[u8]) -> Result<Vec<u8>, ProgramError> {
    let addressing = ArrayAddressing::new(r#type.clone())?;

    // Validate before dropping physical-layout information so malformed element encodings and nonzero padding cannot
    // be hidden by the logical projection.
    validate_storage_bytes_with_addressing(&addressing, bytes)?;
    if addressing.is_dense_row_major() {
        // Dense physical storage is already the requested contiguous logical representation.
        return Ok(bytes.to_vec());
    }

    // Gather only logical element ranges, deliberately omitting holes and tile padding.
    let mut logical_bytes = Vec::with_capacity(addressing.logical_byte_len());
    for element in 0..addressing.element_count() {
        logical_bytes.extend_from_slice(&bytes[addressing.byte_range_for_flat_index(element)]);
    }

    Ok(logical_bytes)
}

/// Validates that `bytes` is a complete physical storage buffer for `r#type`. Validation covers the layout-derived
/// storage length, every logical element encoding, and the requirement that layout holes and tile padding contain zero.
/// Returns a [`ProgramError`] if validation fails.
#[inline]
pub fn validate_storage_bytes(r#type: &ArrayType, bytes: &[u8]) -> Result<(), ProgramError> {
    let addressing = ArrayAddressing::new(r#type.clone())?;
    validate_storage_bytes_with_addressing(&addressing, bytes)
}

/// Validates the storage length, every logical element encoding, and that layout holes and tile padding contain only
/// zero bytes, using the provided [`ArrayAddressing`].
fn validate_storage_bytes_with_addressing(addressing: &ArrayAddressing, bytes: &[u8]) -> Result<(), ProgramError> {
    if bytes.len() != addressing.storage_byte_len() {
        return Err(TypeError::invalid(format!(
            "array type {} requires {} physical storage bytes but got {}",
            addressing.r#type(),
            addressing.storage_byte_len(),
            bytes.len(),
        ))
        .into());
    }

    if addressing.is_dense_row_major() {
        return validate_logical_bytes(addressing.r#type().data_type(), addressing.element_byte_width(), bytes);
    }

    let mut logical_nonzero_byte_count = 0usize;
    for element in 0..addressing.element_count() {
        let element_bytes = &bytes[addressing.byte_range_for_flat_index(element)];
        validate_element_bytes(addressing.r#type().data_type(), element, element_bytes)?;
        logical_nonzero_byte_count += element_bytes.iter().filter(|byte| **byte != 0).count();
    }

    // Validated layouts have disjoint element ranges. Any nonzero byte not counted inside those ranges must therefore
    // belong to a layout hole or tile padding.
    if bytes.iter().filter(|byte| **byte != 0).count() != logical_nonzero_byte_count {
        return Err(TypeError::invalid("array layout holes and tile padding must contain zero bytes").into());
    }

    Ok(())
}

/// Validates contiguous logical element encodings.
fn validate_logical_bytes(data_type: DataType, element_byte_width: usize, bytes: &[u8]) -> Result<(), ProgramError> {
    if element_byte_width == 0 {
        return Ok(());
    }
    for (element, bytes) in bytes.chunks_exact(element_byte_width).enumerate() {
        validate_element_bytes(data_type, element, bytes)?;
    }
    Ok(())
}

/// Validates one element's data-type-specific bit representation. Sub-byte integer encodings are two's complement in
/// the low bits of one storage byte, matching the [`i1`], [`i2`], [`i4`], [`u1`], [`u2`], and [`u4`] element types in
/// this module, so all of their higher bits must be zero.
fn validate_element_bytes(data_type: DataType, element: usize, bytes: &[u8]) -> Result<(), ProgramError> {
    let valid = match data_type {
        DataType::Boolean => matches!(bytes, [0 | 1]),
        DataType::I1 | DataType::U1 => bytes[0] & !0b1 == 0,
        DataType::I2 | DataType::U2 => bytes[0] & !0b11 == 0,
        DataType::I4 | DataType::U4 | DataType::F4E2M1FN => bytes[0] & !0b1111 == 0,
        DataType::F6E2M3FN | DataType::F6E3M2FN => bytes[0] & !0b11_1111 == 0,
        _ => true,
    };
    if !valid {
        return Err(TypeError::invalid(format!(
            "array element {element} has invalid {data_type} byte encoding {bytes:?}",
        ))
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::types::{Dimension, Layout, Shape, StridedLayout, Tile, TileDimension, TiledLayout};

    use super::*;

    // Asserts that every valid encoding of a low-precision floating-point format round-trips through `to_f64` and
    // `from_f64`, that its NaN encodings collapse onto the canonical positive and negative NaN encodings passed as
    // `$nan_bits`, and that decoded values increase strictly across the format's finite magnitude encodings, which run
    // from zero through `$max_finite_magnitude`.
    macro_rules! assert_low_precision_float_encodings {
        ($type:ident, $from_bits:expr, $bit_count:literal, $max_finite_magnitude:literal, $nan_bits:expr $(,)?) => {{
            let from_bits: fn(u8) -> $type = $from_bits;
            let nan_bits: Option<(u8, u8)> = $nan_bits;
            for bits in 0..(1u16 << $bit_count) {
                let bits = bits as u8;
                let value = from_bits(bits).to_f64();
                let expected = match nan_bits {
                    Some((positive, negative)) if value.is_nan() => {
                        if value.is_sign_negative() {
                            negative
                        } else {
                            positive
                        }
                    }
                    _ => bits,
                };
                assert_eq!($type::from_f64(value).map($type::to_bits), Ok(expected), "encoding {bits:#04x}");
            }
            let mut previous = f64::NEG_INFINITY;
            for magnitude in 0..=$max_finite_magnitude {
                let value = from_bits(magnitude).to_f64();
                assert!(value > previous, "encoding {magnitude:#04x} decodes to {value}");
                previous = value;
            }
        }};
    }

    #[test]
    fn test_sub_byte_integer_construction() {
        assert_eq!(i1::new(-1).map(i1::value), Ok(-1));
        assert_eq!(i1::new(0).map(i1::value), Ok(0));
        assert_eq!(i2::new(-2).map(i2::value), Ok(-2));
        assert_eq!(i2::new(1).map(i2::value), Ok(1));
        assert_eq!(i4::new(-8).map(i4::value), Ok(-8));
        assert_eq!(i4::new(7).map(i4::value), Ok(7));
        assert_eq!(u1::new(0).map(u1::value), Ok(0));
        assert_eq!(u1::new(1).map(u1::value), Ok(1));
        assert_eq!(u2::new(0).map(u2::value), Ok(0));
        assert_eq!(u2::new(3).map(u2::value), Ok(3));
        assert_eq!(u4::new(0).map(u4::value), Ok(0));
        assert_eq!(u4::new(15).map(u4::value), Ok(15));

        assert!(matches!(
            i1::new(1),
            Err(TypeError::Invalid { message }) if message == "value 1 is out of range for i1 array elements",
        ));
        assert!(matches!(
            i2::new(-3),
            Err(TypeError::Invalid { message }) if message == "value -3 is out of range for i2 array elements",
        ));
        assert!(matches!(
            i4::new(8),
            Err(TypeError::Invalid { message }) if message == "value 8 is out of range for i4 array elements",
        ));
        assert!(matches!(
            i4::new(-9),
            Err(TypeError::Invalid { message }) if message == "value -9 is out of range for i4 array elements",
        ));
        assert!(matches!(
            u1::new(2),
            Err(TypeError::Invalid { message }) if message == "value 2 is out of range for u1 array elements",
        ));
        assert!(matches!(
            u2::new(4),
            Err(TypeError::Invalid { message }) if message == "value 4 is out of range for u2 array elements",
        ));
        assert!(matches!(
            u4::new(16),
            Err(TypeError::Invalid { message }) if message == "value 16 is out of range for u4 array elements",
        ));
    }

    #[test]
    fn test_sub_byte_integer_bit_encodings() {
        assert_eq!(i1::new(-1).unwrap().to_bits(), 0x01);
        assert_eq!(i1::new(0).unwrap().to_bits(), 0x00);
        assert_eq!(i2::new(-2).unwrap().to_bits(), 0x02);
        assert_eq!(i2::new(-1).unwrap().to_bits(), 0x03);
        assert_eq!(i2::new(1).unwrap().to_bits(), 0x01);
        assert_eq!(i4::new(-8).unwrap().to_bits(), 0x08);
        assert_eq!(i4::new(-1).unwrap().to_bits(), 0x0f);
        assert_eq!(i4::new(7).unwrap().to_bits(), 0x07);
        assert_eq!(u1::new(1).unwrap().to_bits(), 0x01);
        assert_eq!(u2::new(3).unwrap().to_bits(), 0x03);
        assert_eq!(u4::new(15).unwrap().to_bits(), 0x0f);

        assert_eq!(i1::from_bits(0x01).map(i1::value), Ok(-1));
        assert_eq!(i2::from_bits(0x03).map(i2::value), Ok(-1));
        assert_eq!(i4::from_bits(0x0f).map(i4::value), Ok(-1));
        assert_eq!(i4::from_bits(0x08).map(i4::value), Ok(-8));
        assert_eq!(u4::from_bits(0x0f).map(u4::value), Ok(15));

        // Every valid bit pattern of every format round-trips through the storage-byte representation.
        for bits in 0..=0x01u8 {
            assert_eq!(i1::from_bits(bits).map(i1::to_bits), Ok(bits));
            assert_eq!(u1::from_bits(bits).map(u1::to_bits), Ok(bits));
        }
        for bits in 0..=0x03u8 {
            assert_eq!(i2::from_bits(bits).map(i2::to_bits), Ok(bits));
            assert_eq!(u2::from_bits(bits).map(u2::to_bits), Ok(bits));
        }
        for bits in 0..=0x0fu8 {
            assert_eq!(i4::from_bits(bits).map(i4::to_bits), Ok(bits));
            assert_eq!(u4::from_bits(bits).map(u4::to_bits), Ok(bits));
        }

        assert!(matches!(
            i1::from_bits(0x02),
            Err(TypeError::Invalid { message }) if message == "byte 0x02 is not a valid i1 array-element encoding",
        ));
        assert!(matches!(
            u1::from_bits(0x02),
            Err(TypeError::Invalid { message }) if message == "byte 0x02 is not a valid u1 array-element encoding",
        ));
        assert!(matches!(
            i2::from_bits(0x04),
            Err(TypeError::Invalid { message }) if message == "byte 0x04 is not a valid i2 array-element encoding",
        ));
        assert!(matches!(
            u2::from_bits(0xff),
            Err(TypeError::Invalid { message }) if message == "byte 0xff is not a valid u2 array-element encoding",
        ));
        assert!(matches!(
            i4::from_bits(0x10),
            Err(TypeError::Invalid { message }) if message == "byte 0x10 is not a valid i4 array-element encoding",
        ));
        assert!(matches!(
            u4::from_bits(0x10),
            Err(TypeError::Invalid { message }) if message == "byte 0x10 is not a valid u4 array-element encoding",
        ));
    }

    #[test]
    fn test_sub_byte_integer_limits_display_ordering_and_hashing() {
        assert_eq!((i1::MIN.value(), i1::MAX.value()), (-1, 0));
        assert_eq!((i2::MIN.value(), i2::MAX.value()), (-2, 1));
        assert_eq!((i4::MIN.value(), i4::MAX.value()), (-8, 7));
        assert_eq!((u1::MIN.value(), u1::MAX.value()), (0, 1));
        assert_eq!((u2::MIN.value(), u2::MAX.value()), (0, 3));
        assert_eq!((u4::MIN.value(), u4::MAX.value()), (0, 15));

        assert_eq!(i1::MIN.to_string(), "-1");
        assert_eq!(i2::new(-2).unwrap().to_string(), "-2");
        assert_eq!(i4::new(-8).unwrap().to_string(), "-8");
        assert_eq!(i4::MAX.to_string(), "7");
        assert_eq!(u1::MAX.to_string(), "1");
        assert_eq!(u2::new(2).unwrap().to_string(), "2");
        assert_eq!(u4::MAX.to_string(), "15");

        assert!(i4::new(-8).unwrap() < i4::new(7).unwrap());
        assert!(i4::new(-1).unwrap() < i4::new(0).unwrap());
        assert!(i1::MIN < i1::MAX);
        assert!(u4::new(1).unwrap() < u4::MAX);

        // Derived hashing is consistent with equality, so elements work as map keys.
        let names = HashMap::from([(i4::MIN, "min"), (i4::MAX, "max")]);
        assert_eq!(names.get(&i4::new(-8).unwrap()), Some(&"min"));
        assert_eq!(names.get(&i4::new(7).unwrap()), Some(&"max"));
        assert_eq!(names.get(&i4::new(0).unwrap()), None);
    }

    #[test]
    fn test_low_precision_float_decoding() {
        // Golden decodes pin each format's extremes (largest finite, smallest normal, smallest subnormal) and its
        // reserved special-value encodings.
        assert_eq!(f4e2m1fn::from_bits(0x07).map(f4e2m1fn::to_f64), Ok(6.0));
        assert_eq!(f4e2m1fn::from_bits(0x0f).map(f4e2m1fn::to_f64), Ok(-6.0));
        assert_eq!(f4e2m1fn::from_bits(0x02).map(f4e2m1fn::to_f64), Ok(1.0));
        assert_eq!(f4e2m1fn::from_bits(0x01).map(f4e2m1fn::to_f64), Ok(0.5));
        assert_eq!((f4e2m1fn::MAX.to_bits(), f4e2m1fn::MIN.to_bits()), (0x07, 0x0f));

        assert_eq!(f6e2m3fn::from_bits(0x1f).map(f6e2m3fn::to_f64), Ok(7.5));
        assert_eq!(f6e2m3fn::from_bits(0x3f).map(f6e2m3fn::to_f64), Ok(-7.5));
        assert_eq!(f6e2m3fn::from_bits(0x08).map(f6e2m3fn::to_f64), Ok(1.0));
        assert_eq!(f6e2m3fn::from_bits(0x01).map(f6e2m3fn::to_f64), Ok(0.125));
        assert_eq!((f6e2m3fn::MAX.to_bits(), f6e2m3fn::MIN.to_bits()), (0x1f, 0x3f));

        assert_eq!(f6e3m2fn::from_bits(0x1f).map(f6e3m2fn::to_f64), Ok(28.0));
        assert_eq!(f6e3m2fn::from_bits(0x3f).map(f6e3m2fn::to_f64), Ok(-28.0));
        assert_eq!(f6e3m2fn::from_bits(0x04).map(f6e3m2fn::to_f64), Ok(0.25));
        assert_eq!(f6e3m2fn::from_bits(0x01).map(f6e3m2fn::to_f64), Ok(0.0625));
        assert_eq!((f6e3m2fn::MAX.to_bits(), f6e3m2fn::MIN.to_bits()), (0x1f, 0x3f));

        assert_eq!(f8e3m4::from_bits(0x6f).to_f64(), 15.5);
        assert_eq!(f8e3m4::from_bits(0xef).to_f64(), -15.5);
        assert_eq!(f8e3m4::from_bits(0x10).to_f64(), 2f64.powi(-2));
        assert_eq!(f8e3m4::from_bits(0x01).to_f64(), 2f64.powi(-6));
        assert_eq!(f8e3m4::from_bits(0x70).to_f64(), f64::INFINITY);
        assert_eq!(f8e3m4::from_bits(0xf0).to_f64(), f64::NEG_INFINITY);
        assert!(f8e3m4::from_bits(0x71).to_f64().is_nan());
        assert!(f8e3m4::from_bits(0x78).to_f64().is_nan());
        assert_eq!((f8e3m4::MAX.to_bits(), f8e3m4::MIN.to_bits()), (0x6f, 0xef));

        assert_eq!(f8e4m3::from_bits(0x77).to_f64(), 240.0);
        assert_eq!(f8e4m3::from_bits(0xf7).to_f64(), -240.0);
        assert_eq!(f8e4m3::from_bits(0x08).to_f64(), 2f64.powi(-6));
        assert_eq!(f8e4m3::from_bits(0x01).to_f64(), 2f64.powi(-9));
        assert_eq!(f8e4m3::from_bits(0x78).to_f64(), f64::INFINITY);
        assert_eq!(f8e4m3::from_bits(0xf8).to_f64(), f64::NEG_INFINITY);
        assert!(f8e4m3::from_bits(0x79).to_f64().is_nan());
        assert!(f8e4m3::from_bits(0x7c).to_f64().is_nan());
        assert_eq!((f8e4m3::MAX.to_bits(), f8e4m3::MIN.to_bits()), (0x77, 0xf7));

        assert_eq!(f8e4m3fn::from_bits(0x7e).to_f64(), 448.0);
        assert_eq!(f8e4m3fn::from_bits(0xfe).to_f64(), -448.0);
        assert_eq!(f8e4m3fn::from_bits(0x08).to_f64(), 2f64.powi(-6));
        assert_eq!(f8e4m3fn::from_bits(0x01).to_f64(), 2f64.powi(-9));
        assert!(f8e4m3fn::from_bits(0x7f).to_f64().is_nan());
        assert!(f8e4m3fn::from_bits(0xff).to_f64().is_nan());
        assert_eq!((f8e4m3fn::MAX.to_bits(), f8e4m3fn::MIN.to_bits()), (0x7e, 0xfe));

        assert_eq!(f8e4m3fnuz::from_bits(0x7f).to_f64(), 240.0);
        assert_eq!(f8e4m3fnuz::from_bits(0xff).to_f64(), -240.0);
        assert_eq!(f8e4m3fnuz::from_bits(0x08).to_f64(), 2f64.powi(-7));
        assert_eq!(f8e4m3fnuz::from_bits(0x01).to_f64(), 2f64.powi(-10));
        assert!(f8e4m3fnuz::from_bits(0x80).to_f64().is_nan());
        assert_eq!((f8e4m3fnuz::MAX.to_bits(), f8e4m3fnuz::MIN.to_bits()), (0x7f, 0xff));

        assert_eq!(f8e4m3b11fnuz::from_bits(0x7f).to_f64(), 30.0);
        assert_eq!(f8e4m3b11fnuz::from_bits(0xff).to_f64(), -30.0);
        assert_eq!(f8e4m3b11fnuz::from_bits(0x08).to_f64(), 2f64.powi(-10));
        assert_eq!(f8e4m3b11fnuz::from_bits(0x01).to_f64(), 2f64.powi(-13));
        assert!(f8e4m3b11fnuz::from_bits(0x80).to_f64().is_nan());
        assert_eq!((f8e4m3b11fnuz::MAX.to_bits(), f8e4m3b11fnuz::MIN.to_bits()), (0x7f, 0xff));

        assert_eq!(f8e5m2::from_bits(0x7b).to_f64(), 57344.0);
        assert_eq!(f8e5m2::from_bits(0xfb).to_f64(), -57344.0);
        assert_eq!(f8e5m2::from_bits(0x04).to_f64(), 2f64.powi(-14));
        assert_eq!(f8e5m2::from_bits(0x01).to_f64(), 2f64.powi(-16));
        assert_eq!(f8e5m2::from_bits(0x01).to_f32(), 2f32.powi(-16));
        assert_eq!(f8e5m2::from_bits(0x7c).to_f64(), f64::INFINITY);
        assert_eq!(f8e5m2::from_bits(0xfc).to_f64(), f64::NEG_INFINITY);
        assert!(f8e5m2::from_bits(0x7d).to_f64().is_nan());
        assert!(f8e5m2::from_bits(0x7e).to_f64().is_nan());
        assert_eq!((f8e5m2::MAX.to_bits(), f8e5m2::MIN.to_bits()), (0x7b, 0xfb));

        assert_eq!(f8e5m2fnuz::from_bits(0x7f).to_f64(), 57344.0);
        assert_eq!(f8e5m2fnuz::from_bits(0xff).to_f64(), -57344.0);
        assert_eq!(f8e5m2fnuz::from_bits(0x04).to_f64(), 2f64.powi(-15));
        assert_eq!(f8e5m2fnuz::from_bits(0x01).to_f64(), 2f64.powi(-17));
        assert!(f8e5m2fnuz::from_bits(0x80).to_f64().is_nan());
        assert_eq!((f8e5m2fnuz::MAX.to_bits(), f8e5m2fnuz::MIN.to_bits()), (0x7f, 0xff));

        assert_eq!(f8e8m0fnu::from_bits(0x7f).to_f64(), 1.0);
        assert_eq!(f8e8m0fnu::from_bits(0x80).to_f64(), 2.0);
        assert_eq!(f8e8m0fnu::from_bits(0xfe).to_f64(), 2f64.powi(127));
        assert_eq!(f8e8m0fnu::from_bits(0x00).to_f64(), 2f64.powi(-127));
        assert_eq!(f8e8m0fnu::from_bits(0x00).to_f32(), 2f32.powi(-127));
        assert!(f8e8m0fnu::from_bits(0xff).to_f64().is_nan());
        assert_eq!((f8e8m0fnu::MAX.to_bits(), f8e8m0fnu::MIN.to_bits()), (0xfe, 0x00));

        // Every format with a sign bit decodes its two zero encodings to the two signed zeros.
        for negative_zero in [f4e2m1fn::from_bits(0x08).unwrap().to_f64(), f6e2m3fn::from_bits(0x20).unwrap().to_f64()]
        {
            assert_eq!(negative_zero, 0.0);
            assert!(negative_zero.is_sign_negative());
        }
        assert_eq!(f8e4m3fn::from_bits(0x00).to_f64(), 0.0);
        assert!(f8e4m3fn::from_bits(0x00).to_f64().is_sign_positive());
        assert!(f8e4m3fn::from_bits(0x80).to_f64().is_sign_negative());
    }

    #[test]
    fn test_low_precision_float_bit_round_trips() {
        assert_low_precision_float_encodings!(f4e2m1fn, |bits| f4e2m1fn::from_bits(bits).unwrap(), 4, 0x07, None);
        assert_low_precision_float_encodings!(f6e2m3fn, |bits| f6e2m3fn::from_bits(bits).unwrap(), 6, 0x1f, None);
        assert_low_precision_float_encodings!(f6e3m2fn, |bits| f6e3m2fn::from_bits(bits).unwrap(), 6, 0x1f, None);
        assert_low_precision_float_encodings!(f8e3m4, f8e3m4::from_bits, 8, 0x6f, Some((0x78, 0xf8)));
        assert_low_precision_float_encodings!(f8e4m3, f8e4m3::from_bits, 8, 0x77, Some((0x7c, 0xfc)));
        assert_low_precision_float_encodings!(f8e4m3fn, f8e4m3fn::from_bits, 8, 0x7e, Some((0x7f, 0xff)));
        assert_low_precision_float_encodings!(f8e4m3fnuz, f8e4m3fnuz::from_bits, 8, 0x7f, Some((0x80, 0x80)));
        assert_low_precision_float_encodings!(f8e4m3b11fnuz, f8e4m3b11fnuz::from_bits, 8, 0x7f, Some((0x80, 0x80)));
        assert_low_precision_float_encodings!(f8e5m2, f8e5m2::from_bits, 8, 0x7b, Some((0x7e, 0xfe)));
        assert_low_precision_float_encodings!(f8e5m2fnuz, f8e5m2fnuz::from_bits, 8, 0x7f, Some((0x80, 0x80)));
        assert_low_precision_float_encodings!(f8e8m0fnu, f8e8m0fnu::from_bits, 8, 0xfe, Some((0xff, 0xff)));
    }

    #[test]
    fn test_low_precision_float_checked_bit_encodings() {
        assert_eq!(f4e2m1fn::from_bits(0x0f).map(f4e2m1fn::to_bits), Ok(0x0f));
        assert_eq!(f6e2m3fn::from_bits(0x3f).map(f6e2m3fn::to_bits), Ok(0x3f));
        assert_eq!(f6e3m2fn::from_bits(0x3f).map(f6e3m2fn::to_bits), Ok(0x3f));

        assert!(matches!(
            f4e2m1fn::from_bits(0x10),
            Err(TypeError::Invalid { message })
                if message == "byte 0x10 is not a valid f4e2m1fn array-element encoding",
        ));
        assert!(matches!(
            f6e2m3fn::from_bits(0x40),
            Err(TypeError::Invalid { message })
                if message == "byte 0x40 is not a valid f6e2m3fn array-element encoding",
        ));
        assert!(matches!(
            f6e3m2fn::from_bits(0xff),
            Err(TypeError::Invalid { message })
                if message == "byte 0xff is not a valid f6e3m2fn array-element encoding",
        ));
    }

    #[test]
    fn test_microscaling_float_rounding() {
        // The microscaling formats have no infinity and no NaN, so overflowing inputs saturate and NaN is rejected.
        assert_eq!(f4e2m1fn::from_f64(7.0).map(f4e2m1fn::to_f64), Ok(6.0));
        assert_eq!(f4e2m1fn::from_f64(1e10).map(f4e2m1fn::to_f64), Ok(6.0));
        assert_eq!(f4e2m1fn::from_f64(f64::MAX).map(f4e2m1fn::to_bits), Ok(0x07));
        assert_eq!(f4e2m1fn::from_f64(f64::MIN).map(f4e2m1fn::to_bits), Ok(0x0f));
        assert_eq!(f4e2m1fn::from_f64(f64::INFINITY).map(f4e2m1fn::to_bits), Ok(0x07));
        assert_eq!(f4e2m1fn::from_f64(f64::NEG_INFINITY).map(f4e2m1fn::to_f64), Ok(-6.0));
        assert_eq!(f4e2m1fn::from_f64(f64::NEG_INFINITY).map(f4e2m1fn::to_bits), Ok(0x0f));
        // 0.25 lies halfway between 0 and 0.5, and 0.75 halfway between 0.5 and 1, so both round to an even encoding.
        assert_eq!(f4e2m1fn::from_f64(0.25).map(f4e2m1fn::to_bits), Ok(0x00));
        assert_eq!(f4e2m1fn::from_f64(0.75).map(f4e2m1fn::to_bits), Ok(0x02));
        assert_eq!(f4e2m1fn::from_f64(-0.75).map(f4e2m1fn::to_f64), Ok(-1.0));
        assert_eq!(f4e2m1fn::from_f64(-1e-30).map(f4e2m1fn::to_bits), Ok(0x08));
        assert_eq!(f6e3m2fn::from_f64(-1e-30).map(f6e3m2fn::to_bits), Ok(0x20));
        assert!(matches!(
            f4e2m1fn::from_f64(f64::NAN),
            Err(TypeError::Invalid { message }) if message == "data type f4e2m1fn cannot represent NaN",
        ));

        assert_eq!(f6e2m3fn::from_f64(100.0).map(f6e2m3fn::to_f64), Ok(7.5));
        assert_eq!(f6e2m3fn::from_f64(7.75).map(f6e2m3fn::to_bits), Ok(0x1f));
        assert_eq!(f6e2m3fn::from_f64(-7.75).map(f6e2m3fn::to_bits), Ok(0x3f));
        // 1.0625 lies halfway between the neighboring values 1 and 1.125, so the even encoding wins.
        assert_eq!(f6e2m3fn::from_f64(1.0625).map(f6e2m3fn::to_bits), Ok(0x08));
        assert!(matches!(
            f6e2m3fn::from_f32(f32::NAN),
            Err(TypeError::Invalid { message }) if message == "data type f6e2m3fn cannot represent NaN",
        ));

        assert_eq!(f6e3m2fn::from_f64(100.0).map(f6e3m2fn::to_f64), Ok(28.0));
        assert_eq!(f6e3m2fn::from_f64(30.0).map(f6e3m2fn::to_bits), Ok(0x1f));
        assert_eq!(f6e3m2fn::from_f64(-30.0).map(f6e3m2fn::to_bits), Ok(0x3f));
        // 1.125 lies halfway between the neighboring values 1 and 1.25, so the even encoding wins.
        assert_eq!(f6e3m2fn::from_f64(1.125).map(f6e3m2fn::to_bits), Ok(0x0c));
        assert!(matches!(
            f6e3m2fn::from_f64(f64::NAN),
            Err(TypeError::Invalid { message }) if message == "data type f6e3m2fn cannot represent NaN",
        ));
    }

    #[test]
    fn test_low_precision_float_rounding() {
        // Rounding ties choose the even encoding, which at the overflow boundary can be either the largest finite
        // value or the virtual candidate one step past it.
        assert_eq!(f8e3m4::from_f64(15.7).map(f8e3m4::to_f64), Ok(15.5));
        assert_eq!(f8e3m4::from_f64(15.75).map(f8e3m4::to_bits), Ok(0x70));

        assert_eq!(f8e4m3::from_f64(247.0).map(f8e4m3::to_bits), Ok(0x77));
        assert_eq!(f8e4m3::from_f64(248.0).map(f8e4m3::to_bits), Ok(0x78));

        assert_eq!(f8e4m3fn::from_f64(464.0).map(f8e4m3fn::to_bits), Ok(0x7e));
        assert_eq!(f8e4m3fn::from_f64(465.0).map(f8e4m3fn::to_bits), Ok(0x7f));
        assert_eq!(f8e4m3fn::from_f64(-465.0).map(f8e4m3fn::to_bits), Ok(0xff));
        // 21 lies halfway between the neighboring values 20 and 22, so the even mantissa wins.
        assert_eq!(f8e4m3fn::from_f64(20.0).map(f8e4m3fn::to_bits), Ok(0x5a));
        assert_eq!(f8e4m3fn::from_f64(22.0).map(f8e4m3fn::to_bits), Ok(0x5b));
        assert_eq!(f8e4m3fn::from_f64(21.0).map(f8e4m3fn::to_bits), Ok(0x5a));
        assert_eq!(f8e4m3fn::from_f64(20.9).map(f8e4m3fn::to_f64), Ok(20.0));
        assert_eq!(f8e4m3fn::from_f64(21.1).map(f8e4m3fn::to_f64), Ok(22.0));

        assert_eq!(f8e4m3fnuz::from_f64(247.0).map(f8e4m3fnuz::to_bits), Ok(0x7f));
        assert_eq!(f8e4m3fnuz::from_f64(248.0).map(f8e4m3fnuz::to_bits), Ok(0x80));
        assert_eq!(f8e4m3fnuz::from_f64(-0.0).map(f8e4m3fnuz::to_bits), Ok(0x00));
        assert_eq!(f8e4m3fnuz::from_f64(f64::NEG_INFINITY).map(f8e4m3fnuz::to_bits), Ok(0x80));

        assert_eq!(f8e4m3b11fnuz::from_f64(30.9).map(f8e4m3b11fnuz::to_bits), Ok(0x7f));
        assert_eq!(f8e4m3b11fnuz::from_f64(31.0).map(f8e4m3b11fnuz::to_bits), Ok(0x80));

        assert_eq!(f8e5m2::from_f64(61439.0).map(f8e5m2::to_bits), Ok(0x7b));
        assert_eq!(f8e5m2::from_f64(61440.0).map(f8e5m2::to_bits), Ok(0x7c));
        assert_eq!(f8e5m2::from_f64(f64::MAX).map(f8e5m2::to_bits), Ok(0x7c));
        assert_eq!(f8e5m2::from_f64(-61440.0).map(f8e5m2::to_bits), Ok(0xfc));
        assert_eq!(f8e5m2::from_f64(f64::INFINITY).map(f8e5m2::to_bits), Ok(0x7c));

        assert_eq!(f8e5m2fnuz::from_f64(61439.0).map(f8e5m2fnuz::to_bits), Ok(0x7f));
        assert_eq!(f8e5m2fnuz::from_f64(61440.0).map(f8e5m2fnuz::to_bits), Ok(0x80));
        assert_eq!(f8e5m2fnuz::from_f64(-0.0).map(f8e5m2fnuz::to_bits), Ok(0x00));

        // An input that underflows keeps its sign, except in the `fnuz` formats, whose sign-bit-only encoding is NaN
        // rather than a negative zero.
        assert_eq!(f8e4m3fn::from_f64(-1e-30).map(f8e4m3fn::to_bits), Ok(0x80));
        assert_eq!(f8e4m3fn::from_f64(1e-30).map(f8e4m3fn::to_bits), Ok(0x00));
        assert_eq!(f8e5m2::from_f64(-1e-30).map(f8e5m2::to_bits), Ok(0x80));
        assert_eq!(f8e4m3fnuz::from_f64(-1e-30).map(f8e4m3fnuz::to_bits), Ok(0x00));
        assert_eq!(f8e4m3b11fnuz::from_f64(-1e-30).map(f8e4m3b11fnuz::to_bits), Ok(0x00));
        assert_eq!(f8e5m2fnuz::from_f64(-1e-30).map(f8e5m2fnuz::to_bits), Ok(0x00));

        // Inputs far above a format's range overflow just like inputs just past its rounding boundary.
        assert_eq!(f8e4m3fn::from_f64(f64::MAX).map(f8e4m3fn::to_bits), Ok(0x7f));
        assert_eq!(f8e4m3fn::from_f64(f64::MIN).map(f8e4m3fn::to_bits), Ok(0xff));
        assert_eq!(f8e4m3::from_f64(f64::MAX).map(f8e4m3::to_bits), Ok(0x78));
        assert_eq!(f8e3m4::from_f64(f64::MAX).map(f8e3m4::to_bits), Ok(0x70));
        assert_eq!(f8e4m3fnuz::from_f64(f64::MAX).map(f8e4m3fnuz::to_bits), Ok(0x80));
        assert_eq!(f8e4m3b11fnuz::from_f64(f64::MAX).map(f8e4m3b11fnuz::to_bits), Ok(0x80));
        assert_eq!(f8e5m2fnuz::from_f64(f64::MAX).map(f8e5m2fnuz::to_bits), Ok(0x80));

        // Widening an `f32` input to `f64` is exact, so both conversions round identically.
        assert_eq!(f8e4m3fn::from_f32(21.0).map(f8e4m3fn::to_bits), f8e4m3fn::from_f64(21.0).map(f8e4m3fn::to_bits));
        assert_eq!(f8e5m2::from_f32(61439.0).map(f8e5m2::to_bits), f8e5m2::from_f64(61439.0).map(f8e5m2::to_bits));
        assert_eq!(f8e5m2::from_f32(61439.0).map(f8e5m2::to_bits), Ok(0x7b));
        assert_eq!(f8e3m4::from_f32(15.75).map(f8e3m4::to_bits), Ok(0x70));
    }

    #[test]
    fn test_exponent_only_float_rounding() {
        assert_eq!(f8e8m0fnu::from_f64(1.0).map(f8e8m0fnu::to_bits), Ok(0x7f));
        assert_eq!(f8e8m0fnu::from_f64(2.0).map(f8e8m0fnu::to_bits), Ok(0x80));
        // Consecutive powers of two are the only representable values, so ties round to the even exponent encoding.
        assert_eq!(f8e8m0fnu::from_f64(1.5).map(f8e8m0fnu::to_bits), Ok(0x80));
        assert_eq!(f8e8m0fnu::from_f64(3.0).map(f8e8m0fnu::to_f64), Ok(2.0));
        assert_eq!(f8e8m0fnu::from_f64(6.0).map(f8e8m0fnu::to_bits), Ok(0x82));
        assert_eq!(f8e8m0fnu::from_f64(6.0).map(f8e8m0fnu::to_f64), Ok(8.0));
        assert_eq!(f8e8m0fnu::from_f64(2f64.powi(-127)).map(f8e8m0fnu::to_bits), Ok(0x00));
        assert_eq!(f8e8m0fnu::from_f32(1.5).map(f8e8m0fnu::to_bits), Ok(0x80));

        // Negative, infinite, and overflowing inputs have no encoding other than the single NaN.
        assert_eq!(f8e8m0fnu::from_f64(-1.0).map(f8e8m0fnu::to_bits), Ok(0xff));
        assert_eq!(f8e8m0fnu::from_f64(f64::MAX).map(f8e8m0fnu::to_bits), Ok(0xff));
        assert_eq!(f8e8m0fnu::from_f64(f64::INFINITY).map(f8e8m0fnu::to_bits), Ok(0xff));
        assert_eq!(f8e8m0fnu::from_f64(f64::NEG_INFINITY).map(f8e8m0fnu::to_bits), Ok(0xff));
        assert_eq!(f8e8m0fnu::from_f64(f64::NAN).map(f8e8m0fnu::to_bits), Ok(0xff));

        assert!(matches!(
            f8e8m0fnu::from_f64(0.0),
            Err(TypeError::Invalid { message }) if message == "data type f8e8m0fnu cannot represent zero",
        ));
        assert!(matches!(
            f8e8m0fnu::from_f64(-0.0),
            Err(TypeError::Invalid { message }) if message == "data type f8e8m0fnu cannot represent zero",
        ));
    }

    #[test]
    fn test_low_precision_float_special_values() {
        assert_eq!(f8e3m4::NAN.to_bits(), 0x78);
        assert_eq!(f8e4m3::NAN.to_bits(), 0x7c);
        assert_eq!(f8e4m3fn::NAN.to_bits(), 0x7f);
        assert_eq!(f8e4m3fnuz::NAN.to_bits(), 0x80);
        assert_eq!(f8e4m3b11fnuz::NAN.to_bits(), 0x80);
        assert_eq!(f8e5m2::NAN.to_bits(), 0x7e);
        assert_eq!(f8e5m2fnuz::NAN.to_bits(), 0x80);
        assert_eq!(f8e8m0fnu::NAN.to_bits(), 0xff);
        assert!(f8e3m4::NAN.is_nan());
        assert!(f8e4m3::NAN.is_nan());
        assert!(f8e4m3fn::NAN.is_nan());
        assert!(f8e4m3fnuz::NAN.is_nan());
        assert!(f8e4m3b11fnuz::NAN.is_nan());
        assert!(f8e5m2::NAN.is_nan());
        assert!(f8e5m2fnuz::NAN.is_nan());
        assert!(f8e8m0fnu::NAN.is_nan());
        assert!(!f8e4m3fn::MAX.is_nan());
        assert!(!f8e5m2::INFINITY.is_nan());

        assert_eq!((f8e3m4::INFINITY.to_bits(), f8e3m4::NEG_INFINITY.to_bits()), (0x70, 0xf0));
        assert_eq!((f8e4m3::INFINITY.to_bits(), f8e4m3::NEG_INFINITY.to_bits()), (0x78, 0xf8));
        assert_eq!((f8e5m2::INFINITY.to_bits(), f8e5m2::NEG_INFINITY.to_bits()), (0x7c, 0xfc));
        assert_eq!(f8e5m2::INFINITY.to_f64(), f64::INFINITY);
        assert_eq!(f8e5m2::NEG_INFINITY.to_f64(), f64::NEG_INFINITY);

        // Comparisons are numeric, so the two signed zeros compare equal even though their encodings differ, and NaN
        // is unequal to itself and unordered against everything.
        assert_eq!(f8e4m3fn::from_f64(0.0), f8e4m3fn::from_f64(-0.0));
        assert_eq!(
            (f8e4m3fn::from_f64(0.0).map(f8e4m3fn::to_bits), f8e4m3fn::from_f64(-0.0).map(f8e4m3fn::to_bits)),
            (Ok(0x00), Ok(0x80)),
        );
        assert!(f8e4m3fn::NAN != f8e4m3fn::NAN);
        assert_eq!(f8e4m3fn::NAN.partial_cmp(&f8e4m3fn::MAX), None);
        assert_eq!(f8e4m3fn::MIN.partial_cmp(&f8e4m3fn::MAX), Some(Ordering::Less));
        assert!(f8e4m3fn::MIN < f8e4m3fn::MAX);
        assert!(f8e5m2::NEG_INFINITY < f8e5m2::MIN);
        assert!(f8e5m2::MAX < f8e5m2::INFINITY);
        assert!(f8e8m0fnu::MIN < f8e8m0fnu::MAX);
        assert!(f4e2m1fn::MIN < f4e2m1fn::MAX);

        assert_eq!(f8e4m3fn::MAX.to_string(), "448");
        assert_eq!(f8e4m3fn::MIN.to_string(), "-448");
        assert_eq!(f8e5m2::INFINITY.to_string(), "inf");
        assert_eq!(f8e4m3::NAN.to_string(), "NaN");
        assert_eq!(f8e8m0fnu::from_bits(0x7f).to_string(), "1");
        assert_eq!(f4e2m1fn::from_bits(0x03).map(|element| element.to_string()), Ok("1.5".to_string()));
    }

    #[test]
    fn test_array_element_encoding() {
        let booleans = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)]));
        let boolean_bytes = encode_elements(&booleans, &[false, true]).unwrap();
        assert_eq!(boolean_bytes, [0, 1]);
        assert_eq!(decode_elements::<bool>(&booleans, &boolean_bytes), Ok(vec![false, true]));

        let integers = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(2)]));
        let integer_bytes = encode_elements(&integers, &[1i32, -2]).unwrap();
        assert_eq!(integer_bytes, [1, 0, 0, 0, 254, 255, 255, 255]);
        assert_eq!(decode_elements::<i32>(&integers, &integer_bytes), Ok(vec![1, -2]));

        // Floating-point payloads round-trip bit-exactly, including signed zeros and NaN payload bits.
        let floats = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let float_values = [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234)];
        let float_bytes = encode_elements(&floats, &float_values).unwrap();
        assert_eq!(float_bytes, [0, 0, 0, 128, 52, 18, 192, 127]);
        let decoded = decode_elements::<f32>(&floats, &float_bytes).unwrap();
        assert_eq!(decoded.into_iter().map(f32::to_bits).collect::<Vec<_>>(), [0x8000_0000, 0x7fc0_1234]);

        let complex = ArrayType::scalar(DataType::C64);
        let complex_bytes = encode_elements(&complex, &[Complex::new(1.5f32, -2.0)]).unwrap();
        assert_eq!(complex_bytes, [0, 0, 192, 63, 0, 0, 0, 192]);
        assert_eq!(decode_elements::<Complex<f32>>(&complex, &complex_bytes), Ok(vec![Complex::new(1.5, -2.0)]));
    }

    #[test]
    fn test_sub_byte_integer_array_encoding() {
        let r#type = ArrayType::new(DataType::I4, Shape::new(vec![Dimension::Static(4)]));
        let elements = [-8, -1, 0, 7].map(|value| i4::new(value).unwrap());
        let bytes = encode_elements(&r#type, &elements).unwrap();
        assert_eq!(bytes, [0x08, 0x0f, 0x00, 0x07]);
        assert_eq!(validate_storage_bytes(&r#type, &bytes), Ok(()));
        let decoded = decode_elements::<i4>(&r#type, &bytes).unwrap();
        assert_eq!(decoded, elements.to_vec());
        assert_eq!(decoded.into_iter().map(i4::value).collect::<Vec<_>>(), [-8, -1, 0, 7]);
    }

    #[test]
    fn test_low_precision_float_array_encoding() {
        let r#type = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(4)]));
        let elements = [-448.0, -0.5, 0.0, 448.0].map(|value| f8e4m3fn::from_f64(value).unwrap());
        let bytes = encode_elements(&r#type, &elements).unwrap();
        assert_eq!(bytes, [0xfe, 0xb0, 0x00, 0x7e]);
        assert_eq!(validate_storage_bytes(&r#type, &bytes), Ok(()));
        let decoded = decode_elements::<f8e4m3fn>(&r#type, &bytes).unwrap();
        assert_eq!(decoded, elements.to_vec());
        assert_eq!(decoded.into_iter().map(f8e4m3fn::to_f64).collect::<Vec<_>>(), [-448.0, -0.5, 0.0, 448.0]);

        // The four- and six-bit formats occupy only the low bits of their storage bytes.
        let narrow = ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2)]));
        let narrow_elements = [-6.0, 1.5].map(|value| f4e2m1fn::from_f64(value).unwrap());
        let narrow_bytes = encode_elements(&narrow, &narrow_elements).unwrap();
        assert_eq!(narrow_bytes, [0x0f, 0x03]);
        assert_eq!(validate_storage_bytes(&narrow, &narrow_bytes), Ok(()));
        assert_eq!(decode_elements::<f4e2m1fn>(&narrow, &narrow_bytes), Ok(narrow_elements.to_vec()));
    }

    #[test]
    fn test_layout_aware_array_encoding() {
        let shape = Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]);
        let values = [1u8, 2, 3, 4, 5, 6];
        let cases = [
            (ArrayType::new(DataType::U8, shape.clone()), vec![1, 2, 3, 4, 5, 6]),
            // Explicit layouts that reproduce dense row-major storage follow the bulk dense encoding paths.
            (
                ArrayType::new(DataType::U8, shape.clone())
                    .with_layout(Layout::Strided(StridedLayout::new(vec![3, 1]))),
                vec![1, 2, 3, 4, 5, 6],
            ),
            (
                ArrayType::new(DataType::U8, shape.clone())
                    .with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], Vec::new()))),
                vec![1, 2, 3, 4, 5, 6],
            ),
            (
                ArrayType::new(DataType::U8, shape.clone())
                    .with_layout(Layout::Strided(StridedLayout::new(vec![4, 1]))),
                vec![1, 2, 3, 0, 4, 5, 6],
            ),
            (
                ArrayType::new(DataType::U8, shape.clone())
                    .with_layout(Layout::Strided(StridedLayout::new(vec![-4, 1]))),
                vec![4, 5, 6, 0, 1, 2, 3],
            ),
            (
                ArrayType::new(DataType::U8, shape.clone())
                    .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], Vec::new()))),
                vec![1, 4, 2, 5, 3, 6],
            ),
            (
                ArrayType::new(DataType::U8, shape).with_layout(Layout::Tiled(TiledLayout::new(
                    vec![1, 0],
                    vec![Tile::new(vec![TileDimension::Sized(2), TileDimension::Sized(2)])],
                ))),
                vec![1, 2, 4, 5, 3, 0, 6, 0],
            ),
        ];

        for (r#type, expected_bytes) in cases {
            let bytes = encode_elements(&r#type, &values).unwrap();
            assert_eq!(bytes, expected_bytes);
            assert_eq!(bytes.len(), ArrayAddressing::new(r#type.clone()).unwrap().storage_byte_len());
            assert_eq!(decode_elements::<u8>(&r#type, &bytes), Ok(values.to_vec()));
            assert_eq!(decode_logical_bytes(&r#type, &bytes), Ok(values.to_vec()));
        }
    }

    #[test]
    fn test_array_byte_validation() {
        let booleans = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(2)]));
        assert_eq!(encode_logical_bytes(&booleans, &[0, 1]), Ok(vec![0, 1]));
        assert!(matches!(
            encode_logical_bytes(&booleans, &[0, 2]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array element 1 has invalid bool byte encoding [2]",
        ));
        assert!(matches!(
            validate_storage_bytes(&booleans, &[0]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array type bool[2] requires 2 physical storage bytes but got 1",
        ));

        let narrow = ArrayType::scalar(DataType::I2);
        assert_eq!(encode_logical_bytes(&narrow, &[0b11]), Ok(vec![0b11]));
        assert!(matches!(
            encode_logical_bytes(&narrow, &[0b100]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array element 0 has invalid i2 byte encoding [4]",
        ));
        let low_precision = ArrayType::scalar(DataType::F4E2M1FN);
        assert_eq!(encode_logical_bytes(&low_precision, &[0b1111]), Ok(vec![0b1111]));
        assert!(matches!(
            encode_logical_bytes(&low_precision, &[0b1_0000]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array element 0 has invalid f4e2m1fn byte encoding [16]",
        ));

        assert!(matches!(
            encode_elements(&booleans, &[0u8, 1]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot encode u8 values as array elements of data type bool",
        ));
        assert!(matches!(
            encode_elements(&booleans, &[true]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array type bool[2] requires 2 logical elements but got 1",
        ));

        let padded = ArrayType::new(DataType::U8, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]))
            .with_layout(Layout::Tiled(TiledLayout::new(
                vec![1, 0],
                vec![Tile::new(vec![TileDimension::Sized(2), TileDimension::Sized(2)])],
            )));
        let mut bytes = encode_elements(&padded, &[1u8, 2, 3, 4, 5, 6]).unwrap();
        bytes[5] = 255;
        bytes[7] = 254;
        assert!(matches!(
            validate_storage_bytes(&padded, &bytes),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "array layout holes and tile padding must contain zero bytes",
        ));

        let payload_free = ArrayType::new(DataType::Token, Shape::new(vec![Dimension::Static(usize::MAX)]));
        assert_eq!(encode_logical_bytes(&payload_free, &[]), Ok(Vec::new()));
        assert_eq!(validate_storage_bytes(&payload_free, &[]), Ok(()));
    }
}
