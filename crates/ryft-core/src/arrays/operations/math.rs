//! Reference [`Array`] kernels for the mathematics operation family contracts.
//!
//! This module owns the element-level arithmetic contracts of the reference backend — the per-element analogues of
//! the value-level capabilities, together with their per-data-type instantiations — and the kernels built on them.
//! Each kernel decodes its operands through their physical addressing and materializes one owned result. Element data
//! type promotion and NumPy-style broadcasting follow the corresponding operations' type-inference rules, so the
//! eager results match what a staged program computes.

use std::cmp::Ordering;
use std::sync::Arc;

use half::{bf16, f16};
use num_complex::Complex;

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::arrays::Array;
use crate::arrays::broadcasting::Broadcastable;
use crate::arrays::encoding::{
    ArrayElement, f4e2m1fn, f6e2m3fn, f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2,
    f8e5m2fnuz, f8e8m0fnu, i1, i2, i4, u1, u2, u4,
};
use crate::arrays::macros::dispatch_on_array_element_type;
use crate::arrays::operations::manipulation::ElementConversionTarget;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::StaticShape;
use crate::macros::check_types;
use crate::operations::math::erf::erf_f64;
use crate::operations::math::reduce::reduce_abstract;
use crate::operations::{
    Abs, Add, Atan2, Ceil, ConvertElementType, Cos, Div, Dot, DotDimensionNumbers, DotOperation, Erf, Exp, Floor, Log,
    Logistic, Max, Min, Mul, Neg, Pow, Reduce, ReductionKind, Rem, Round, Rsqrt, ScaledDot, Sign, Sin, Sqrt, Sub, Tanh,
    scaled_dot_composition,
};
use crate::programs::{Operation, ProgramError, TypeError, Typed};

// TODO(eaplatanios): Review this.

// The value-level capability traits such as `Add` and `Mul` deliberately cannot serve the element layer: they are
// blanket-implemented for every `Value` by dispatching the corresponding operation, so coherence rejects concrete
// implementations for element types, and their contract (including element type promotion) is the program-level
// operation semantics that the kernels below implement rather than consume. The element-level analogues below
// therefore mirror that vocabulary in narrow capability groups. Operations with the same supported element class and
// conversion strategy share one trait, while integer wrapping, low-precision re-encoding, and count conversion stay
// defined in one place per element family.

/// Element-level analogue of the [`Zero`](crate::operations::Zero) capability: the additive identity of one array
/// element type. The extraction is fallible because `f8e8m0fnu` has no zero.
trait ElementZero: ArrayElement {
    /// Returns this element type's additive identity.
    fn zero() -> Result<Self, ProgramError>;
}

/// Element-level analogue of the [`Add`](crate::operations::Add) capability, using the element type's ordinary
/// arithmetic semantics (deterministic two's-complement wrapping for integers and round-to-nearest-even re-encoding
/// for the low-precision floating-point formats).
pub(crate) trait ElementAdd: ArrayElement {
    /// Adds two elements.
    fn add(self, right: Self) -> Result<Self, ProgramError>;
}

/// Element-level analogue of the [`Sub`](crate::operations::Sub) capability, using the element type's ordinary
/// arithmetic semantics.
trait ElementSub: ArrayElement {
    /// Subtracts `right` from this element.
    fn sub(self, right: Self) -> Result<Self, ProgramError>;
}

/// Element-level analogue of the [`Mul`](crate::operations::Mul) capability, using the element type's ordinary
/// arithmetic semantics.
pub(crate) trait ElementMul: ArrayElement {
    /// Multiplies two elements.
    fn mul(self, right: Self) -> Result<Self, ProgramError>;
}

/// Element-level analogue of the [`Div`](crate::operations::Div) capability, including checked integer division
/// failures.
trait ElementDiv: ArrayElement {
    /// Divides this element by `right`.
    fn div(self, right: Self) -> Result<Self, ProgramError>;
}

/// Element-level analogue of the [`Rem`](crate::operations::Rem) capability, including checked integer zero-divisor
/// failures.
trait ElementRem: ArrayElement {
    /// Computes the truncating remainder of this element divided by `right`.
    fn rem(self, right: Self) -> Result<Self, ProgramError>;
}

/// Element-level analogue of the [`Neg`](crate::operations::Neg) capability.
trait ElementNeg: ArrayElement {
    /// Negates this element.
    fn neg(self) -> Result<Self, ProgramError>;
}

/// Element-level analogue of the [`Abs`](crate::operations::Abs) capability. Complex magnitudes use a real output
/// element type.
trait ElementAbs: ArrayElement {
    /// Element type produced by absolute value.
    type Output: ArrayElement;

    /// Computes this element's absolute value or complex magnitude.
    fn abs(self) -> Result<Self::Output, ProgramError>;
}

/// Floating-point math operations shared by real floating-point and complex array elements.
trait ElementFloatMath: ArrayElement {
    /// Computes the sine of this element.
    fn sin(self) -> Result<Self, ProgramError>;

    /// Computes the cosine of this element.
    fn cos(self) -> Result<Self, ProgramError>;

    /// Computes `atan2(self, x)`.
    fn atan2(self, x: Self) -> Result<Self, ProgramError>;

    /// Computes the natural exponential of this element.
    fn exp(self) -> Result<Self, ProgramError>;

    /// Computes the natural logarithm of this element.
    fn log(self) -> Result<Self, ProgramError>;

    /// Computes the principal square root of this element.
    fn sqrt(self) -> Result<Self, ProgramError>;

    /// Computes the reciprocal of the principal square root of this element.
    fn rsqrt(self) -> Result<Self, ProgramError>;

    /// Computes the hyperbolic tangent of this element.
    fn tanh(self) -> Result<Self, ProgramError>;

    /// Computes `1 / (1 + exp(-self))`.
    fn logistic(self) -> Result<Self, ProgramError>;

    /// Raises this element to `exponent`.
    fn pow(self, exponent: Self) -> Result<Self, ProgramError>;
}

/// Operations supported only by real floating-point array elements.
trait ElementRealFloatMath: ArrayElement {
    /// Computes the Gauss error function of this element.
    fn erf(self) -> Result<Self, ProgramError>;

    /// Rounds this element toward negative infinity.
    fn floor(self) -> Result<Self, ProgramError>;

    /// Rounds this element toward positive infinity.
    fn ceil(self) -> Result<Self, ProgramError>;

    /// Rounds this element to the nearest integer, resolving ties toward the nearest even integer.
    fn round(self) -> Result<Self, ProgramError>;
}

/// Sign extraction for signed-integer, floating-point, and complex array elements.
trait ElementSign: ArrayElement {
    /// Computes this element's sign according to [`Sign`](crate::operations::Sign) semantics.
    fn sign(self) -> Result<Self, ProgramError>;
}

/// Element-level mean divisor, serving mean reductions, which have no capability analogue of their own because a
/// mean lowers to a sum followed by a division by the reduced element count.
trait ElementDivideByCount: ArrayElement {
    /// Divides this element by `count` after converting `count` to the element type.
    fn divide_by_count(self, count: usize) -> Result<Self, ProgramError>;
}

// Implements typed arithmetic for signed primitive integers with deterministic two's-complement wrapping.
macro_rules! impl_array_arithmetic_for_signed_integer {
    ($type:ty) => {
        impl ElementZero for $type {
            #[inline]
            fn zero() -> Result<Self, ProgramError> {
                Ok(0)
            }
        }

        impl ElementAdd for $type {
            #[inline]
            fn add(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self.wrapping_add(right))
            }
        }

        impl ElementSub for $type {
            #[inline]
            fn sub(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self.wrapping_sub(right))
            }
        }

        impl ElementMul for $type {
            #[inline]
            fn mul(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self.wrapping_mul(right))
            }
        }

        impl ElementDiv for $type {
            fn div(self, right: Self) -> Result<Self, ProgramError> {
                if right == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer scalar of data type {} by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                if self == Self::MIN && right == -1 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide the minimum integer scalar of data type {} by -1",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(self / right)
            }
        }

        impl ElementRem for $type {
            fn rem(self, right: Self) -> Result<Self, ProgramError> {
                if right == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot compute the remainder of an integer scalar of data type {} with a zero divisor",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(self.wrapping_rem(right))
            }
        }

        impl ElementNeg for $type {
            #[inline]
            fn neg(self) -> Result<Self, ProgramError> {
                Ok(self.wrapping_neg())
            }
        }

        impl ElementAbs for $type {
            type Output = Self;

            #[inline]
            fn abs(self) -> Result<Self, ProgramError> {
                Ok(self.wrapping_abs())
            }
        }

        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = count as Self;
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type {} by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                if self == Self::MIN && divisor == -1 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide the minimum integer array element of data type {} by -1",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(self / divisor)
            }
        }
    };
}

// Implements typed arithmetic for unsigned primitive integers with deterministic modular wrapping.
macro_rules! impl_array_arithmetic_for_unsigned_integer {
    ($type:ty) => {
        impl ElementZero for $type {
            #[inline]
            fn zero() -> Result<Self, ProgramError> {
                Ok(0)
            }
        }

        impl ElementAdd for $type {
            #[inline]
            fn add(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self.wrapping_add(right))
            }
        }

        impl ElementSub for $type {
            #[inline]
            fn sub(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self.wrapping_sub(right))
            }
        }

        impl ElementMul for $type {
            #[inline]
            fn mul(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self.wrapping_mul(right))
            }
        }

        impl ElementDiv for $type {
            fn div(self, right: Self) -> Result<Self, ProgramError> {
                if right == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer scalar of data type {} by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(self / right)
            }
        }

        impl ElementRem for $type {
            fn rem(self, right: Self) -> Result<Self, ProgramError> {
                if right == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot compute the remainder of an integer scalar of data type {} with a zero divisor",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(self % right)
            }
        }

        impl ElementNeg for $type {
            #[inline]
            fn neg(self) -> Result<Self, ProgramError> {
                Ok(self.wrapping_neg())
            }
        }

        impl ElementAbs for $type {
            type Output = Self;

            fn abs(self) -> Result<Self, ProgramError> {
                Err(TypeError::invalid(format!(
                    "cannot compute the absolute value of a scalar of data type {}",
                    Self::data_type(),
                ))
                .into())
            }
        }

        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = count as Self;
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type {} by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(self / divisor)
            }
        }
    };
}

impl_array_arithmetic_for_signed_integer!(i8);
impl_array_arithmetic_for_signed_integer!(i16);
impl_array_arithmetic_for_signed_integer!(i32);
impl_array_arithmetic_for_signed_integer!(i64);
impl_array_arithmetic_for_unsigned_integer!(u8);
impl_array_arithmetic_for_unsigned_integer!(u16);
impl_array_arithmetic_for_unsigned_integer!(u32);
impl_array_arithmetic_for_unsigned_integer!(u64);

// Implements modular arithmetic for a signed sub-byte integer's checked low-bit encoding.
macro_rules! impl_array_arithmetic_for_signed_sub_byte_integer {
    ($type:ty) => {
        impl ElementZero for $type {
            #[inline]
            fn zero() -> Result<Self, ProgramError> {
                Ok(Self::from_bits(0).unwrap())
            }
        }

        impl ElementAdd for $type {
            #[inline]
            fn add(self, right: Self) -> Result<Self, ProgramError> {
                let bit_mask = Self::MIN.to_bits() | Self::MAX.to_bits();
                Ok(Self::from_bits(self.to_bits().wrapping_add(right.to_bits()) & bit_mask).unwrap())
            }
        }

        impl ElementSub for $type {
            #[inline]
            fn sub(self, right: Self) -> Result<Self, ProgramError> {
                let bit_mask = Self::MIN.to_bits() | Self::MAX.to_bits();
                Ok(Self::from_bits(self.to_bits().wrapping_sub(right.to_bits()) & bit_mask).unwrap())
            }
        }

        impl ElementMul for $type {
            #[inline]
            fn mul(self, right: Self) -> Result<Self, ProgramError> {
                let bit_mask = Self::MIN.to_bits() | Self::MAX.to_bits();
                Ok(Self::from_bits(self.to_bits().wrapping_mul(right.to_bits()) & bit_mask).unwrap())
            }
        }

        impl ElementDiv for $type {
            fn div(self, right: Self) -> Result<Self, ProgramError> {
                if right.value() == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer scalar of data type {} by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                if self == Self::MIN && right.value() == -1 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide the minimum integer scalar of data type {} by -1",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(Self::new(self.value() / right.value()).unwrap())
            }
        }

        impl ElementRem for $type {
            fn rem(self, right: Self) -> Result<Self, ProgramError> {
                if right.value() == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot compute the remainder of an integer scalar of data type {} with a zero divisor",
                        Self::data_type(),
                    ))
                    .into());
                }
                let bit_mask = Self::MIN.to_bits() | Self::MAX.to_bits();
                Ok(Self::from_bits(self.value().wrapping_rem(right.value()) as u8 & bit_mask).unwrap())
            }
        }

        impl ElementNeg for $type {
            #[inline]
            fn neg(self) -> Result<Self, ProgramError> {
                let bit_mask = Self::MIN.to_bits() | Self::MAX.to_bits();
                Ok(Self::from_bits(self.to_bits().wrapping_neg() & bit_mask).unwrap())
            }
        }

        impl ElementAbs for $type {
            type Output = Self;

            fn abs(self) -> Result<Self, ProgramError> {
                if Self::data_type() == DataType::I1 {
                    return Err(TypeError::invalid(
                        "cannot compute the absolute value of a scalar of data type i1".to_string(),
                    )
                    .into());
                }
                let bit_mask = Self::MIN.to_bits() | Self::MAX.to_bits();
                Ok(Self::from_bits(self.value().wrapping_abs() as u8 & bit_mask).unwrap())
            }
        }

        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let bit_mask = Self::MIN.to_bits() | Self::MAX.to_bits();
                let divisor = Self::from_bits(count as u8 & bit_mask).unwrap().value();
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type {} by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                if self == Self::MIN && divisor == -1 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide the minimum integer array element of data type {} by -1",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(Self::new(self.value() / divisor).unwrap())
            }
        }
    };
}

// Implements modular arithmetic for an unsigned sub-byte integer's checked low-bit encoding.
macro_rules! impl_array_arithmetic_for_unsigned_sub_byte_integer {
    ($type:ty) => {
        impl ElementZero for $type {
            #[inline]
            fn zero() -> Result<Self, ProgramError> {
                Ok(Self::from_bits(0).unwrap())
            }
        }

        impl ElementAdd for $type {
            #[inline]
            fn add(self, right: Self) -> Result<Self, ProgramError> {
                Ok(Self::from_bits(self.to_bits().wrapping_add(right.to_bits()) & Self::MAX.to_bits()).unwrap())
            }
        }

        impl ElementSub for $type {
            #[inline]
            fn sub(self, right: Self) -> Result<Self, ProgramError> {
                Ok(Self::from_bits(self.to_bits().wrapping_sub(right.to_bits()) & Self::MAX.to_bits()).unwrap())
            }
        }

        impl ElementMul for $type {
            #[inline]
            fn mul(self, right: Self) -> Result<Self, ProgramError> {
                Ok(Self::from_bits(self.to_bits().wrapping_mul(right.to_bits()) & Self::MAX.to_bits()).unwrap())
            }
        }

        impl ElementDiv for $type {
            fn div(self, right: Self) -> Result<Self, ProgramError> {
                if right.value() == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer scalar of data type {} by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(Self::new(self.value() / right.value()).unwrap())
            }
        }

        impl ElementRem for $type {
            fn rem(self, right: Self) -> Result<Self, ProgramError> {
                if right.value() == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot compute the remainder of an integer scalar of data type {} with a zero divisor",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(Self::new(self.value() % right.value()).unwrap())
            }
        }

        impl ElementNeg for $type {
            #[inline]
            fn neg(self) -> Result<Self, ProgramError> {
                Ok(Self::from_bits(self.to_bits().wrapping_neg() & Self::MAX.to_bits()).unwrap())
            }
        }

        impl ElementAbs for $type {
            type Output = Self;

            fn abs(self) -> Result<Self, ProgramError> {
                Err(TypeError::invalid(format!(
                    "cannot compute the absolute value of a scalar of data type {}",
                    Self::data_type(),
                ))
                .into())
            }
        }

        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = Self::from_bits(count as u8 & Self::MAX.to_bits()).unwrap().value();
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type {} by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                Ok(Self::new(self.value() / divisor).unwrap())
            }
        }
    };
}

impl_array_arithmetic_for_signed_sub_byte_integer!(i1);
impl_array_arithmetic_for_signed_sub_byte_integer!(i2);
impl_array_arithmetic_for_signed_sub_byte_integer!(i4);
impl_array_arithmetic_for_unsigned_sub_byte_integer!(u1);
impl_array_arithmetic_for_unsigned_sub_byte_integer!(u2);
impl_array_arithmetic_for_unsigned_sub_byte_integer!(u4);

// Implements arithmetic for a low-precision floating-point format through its exact f64 conversion contract.
macro_rules! impl_array_arithmetic_for_low_precision_float {
    ($type:ty) => {
        impl ElementZero for $type {
            #[inline]
            fn zero() -> Result<Self, ProgramError> {
                Ok(Self::from_f64(0.0)?)
            }
        }

        impl ElementAdd for $type {
            #[inline]
            fn add(self, right: Self) -> Result<Self, ProgramError> {
                Ok(Self::from_f64(self.to_f64() + right.to_f64())?)
            }
        }

        impl ElementSub for $type {
            #[inline]
            fn sub(self, right: Self) -> Result<Self, ProgramError> {
                Ok(Self::from_f64(self.to_f64() - right.to_f64())?)
            }
        }

        impl ElementMul for $type {
            #[inline]
            fn mul(self, right: Self) -> Result<Self, ProgramError> {
                Ok(Self::from_f64(self.to_f64() * right.to_f64())?)
            }
        }

        impl ElementDiv for $type {
            #[inline]
            fn div(self, right: Self) -> Result<Self, ProgramError> {
                Ok(Self::from_f64(self.to_f64() / right.to_f64())?)
            }
        }

        impl ElementRem for $type {
            #[inline]
            fn rem(self, right: Self) -> Result<Self, ProgramError> {
                Ok(Self::from_f64(self.to_f64() % right.to_f64())?)
            }
        }

        impl ElementNeg for $type {
            fn neg(self) -> Result<Self, ProgramError> {
                if Self::data_type() == DataType::F8E8M0FNU {
                    return Err(TypeError::invalid("cannot negate a scalar of data type f8e8m0fnu".to_string()).into());
                }
                Ok(Self::from_f64(-self.to_f64())?)
            }
        }

        impl ElementAbs for $type {
            type Output = Self;

            #[inline]
            fn abs(self) -> Result<Self, ProgramError> {
                Ok(Self::from_f64(self.to_f64().abs())?)
            }
        }

        impl ElementDivideByCount for $type {
            #[inline]
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = Self::from_f64(count as f64)?;
                Ok(Self::from_f64(self.to_f64() / divisor.to_f64())?)
            }
        }
    };
}

impl_array_arithmetic_for_low_precision_float!(f4e2m1fn);
impl_array_arithmetic_for_low_precision_float!(f6e2m3fn);
impl_array_arithmetic_for_low_precision_float!(f6e3m2fn);
impl_array_arithmetic_for_low_precision_float!(f8e3m4);
impl_array_arithmetic_for_low_precision_float!(f8e4m3);
impl_array_arithmetic_for_low_precision_float!(f8e4m3fn);
impl_array_arithmetic_for_low_precision_float!(f8e4m3fnuz);
impl_array_arithmetic_for_low_precision_float!(f8e4m3b11fnuz);
impl_array_arithmetic_for_low_precision_float!(f8e5m2);
impl_array_arithmetic_for_low_precision_float!(f8e5m2fnuz);
impl_array_arithmetic_for_low_precision_float!(f8e8m0fnu);

// Implements ordinary arithmetic for a native or half-precision real floating-point type.
macro_rules! impl_array_arithmetic_for_float {
    ($type:ty, $from_count:expr, $abs:expr) => {
        impl ElementZero for $type {
            #[inline]
            fn zero() -> Result<Self, ProgramError> {
                Ok($from_count(0))
            }
        }

        impl ElementAdd for $type {
            #[inline]
            fn add(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self + right)
            }
        }

        impl ElementSub for $type {
            #[inline]
            fn sub(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self - right)
            }
        }

        impl ElementMul for $type {
            #[inline]
            fn mul(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self * right)
            }
        }

        impl ElementDiv for $type {
            #[inline]
            fn div(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self / right)
            }
        }

        impl ElementRem for $type {
            #[inline]
            fn rem(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self % right)
            }
        }

        impl ElementNeg for $type {
            #[inline]
            fn neg(self) -> Result<Self, ProgramError> {
                Ok(-self)
            }
        }

        impl ElementAbs for $type {
            type Output = Self;

            #[inline]
            fn abs(self) -> Result<Self, ProgramError> {
                Ok($abs(self))
            }
        }

        impl ElementDivideByCount for $type {
            #[inline]
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                Ok(self / $from_count(count))
            }
        }
    };
}

impl_array_arithmetic_for_float!(bf16, |count: usize| bf16::from_f64(count as f64), |value: bf16| {
    bf16::from_f32(value.to_f32().abs())
});
impl_array_arithmetic_for_float!(f16, |count: usize| f16::from_f64(count as f64), |value: f16| {
    f16::from_f32(value.to_f32().abs())
});
impl_array_arithmetic_for_float!(f32, |count: usize| count as f32, f32::abs);
impl_array_arithmetic_for_float!(f64, |count: usize| count as f64, f64::abs);

// Divides finite complex elements after normalizing by the denominator's largest component. The quotient is unchanged,
// while the normalized formula avoids overflow in the direct norm-squared implementation.
macro_rules! divide_complex_array_element {
    ($left:expr, $right:expr) => {{
        let left = $left;
        let right = $right;
        let direct = left / right;
        if direct.re.is_finite() && direct.im.is_finite()
            || !left.re.is_finite()
            || !left.im.is_finite()
            || !right.re.is_finite()
            || !right.im.is_finite()
            || right.re == 0.0 && right.im == 0.0
        {
            direct
        } else if right.im == 0.0 {
            Complex::new(left.re / right.re, left.im / right.re)
        } else if right.re == 0.0 {
            Complex::new(left.im / right.im, -left.re / right.im)
        } else {
            let scale = right.re.abs().max(right.im.abs());
            let left = Complex::new(left.re / scale, left.im / scale);
            let right = Complex::new(right.re / scale, right.im / scale);
            if right.re.abs() >= right.im.abs() {
                let ratio = right.im / right.re;
                let denominator = right.re + right.im * ratio;
                Complex::new((left.re + left.im * ratio) / denominator, (left.im - left.re * ratio) / denominator)
            } else {
                let ratio = right.re / right.im;
                let denominator = right.im + right.re * ratio;
                Complex::new((left.re * ratio + left.im) / denominator, (left.im * ratio - left.re) / denominator)
            }
        }
    }};
}

// Implements complex arithmetic; division by a real count acts componentwise to avoid an unnecessary complex norm.
macro_rules! impl_array_arithmetic_for_complex {
    ($component:ty) => {
        impl ElementZero for Complex<$component> {
            #[inline]
            fn zero() -> Result<Self, ProgramError> {
                Ok(Complex::new(0.0, 0.0))
            }
        }

        impl ElementAdd for Complex<$component> {
            #[inline]
            fn add(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self + right)
            }
        }

        impl ElementSub for Complex<$component> {
            #[inline]
            fn sub(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self - right)
            }
        }

        impl ElementMul for Complex<$component> {
            #[inline]
            fn mul(self, right: Self) -> Result<Self, ProgramError> {
                Ok(self * right)
            }
        }

        impl ElementDiv for Complex<$component> {
            #[inline]
            fn div(self, right: Self) -> Result<Self, ProgramError> {
                Ok(divide_complex_array_element!(self, right))
            }
        }

        impl ElementNeg for Complex<$component> {
            #[inline]
            fn neg(self) -> Result<Self, ProgramError> {
                Ok(-self)
            }
        }

        impl ElementAbs for Complex<$component> {
            type Output = $component;

            #[inline]
            fn abs(self) -> Result<Self::Output, ProgramError> {
                Ok(self.norm())
            }
        }

        impl ElementDivideByCount for Complex<$component> {
            #[inline]
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                // Dividing by a real count is componentwise by definition, which also sidesteps the generic complex
                // division's norm computation, whose intermediate values can overflow for large counts.
                let divisor = count as $component;
                Ok(Complex::new(self.re / divisor, self.im / divisor))
            }
        }
    };
}

impl_array_arithmetic_for_complex!(f32);
impl_array_arithmetic_for_complex!(f64);

// Implements the real floating-point math families through the working precision and exact re-encoding contract of
// each element family. Half precision uses `f32`, native primitive types use themselves, and low-precision formats
// use `f64`, matching the scalar reference semantics.
macro_rules! impl_array_math_for_real_float {
    // Implements a low-precision format through its checked `f64` conversion contract.
    (@low $type:ty) => {
        impl_array_math_for_real_float!(@impl
            $type,
            f64,
            |value: $type| value.to_f64(),
            |value| Ok(<$type>::from_f64(value)?),
            |value: $type| value.to_f64(),
            |value| Ok(<$type>::from_f64(value)?),
        );
    };

    // Implements a half-precision format through its native `f32` arithmetic and `f64` error-function path.
    (@half $type:ty) => {
        impl_array_math_for_real_float!(@impl
            $type,
            f32,
            <$type>::to_f32,
            |value| Ok(<$type>::from_f32(value)),
            <$type>::to_f64,
            |value| Ok(<$type>::from_f64(value)),
        );
    };

    // Implements a native floating-point type without changing working precision.
    (@native $type:ty) => {
        impl_array_math_for_real_float!(@impl
            $type,
            $type,
            |value| value,
            Ok,
            |value: $type| value as f64,
            |value| Ok(value as $type),
        );
    };

    // Generates the implementations after the element family's conversion functions have been selected.
    (@impl $type:ty, $work:ty, $decode:expr, $encode:expr, $to_f64:expr, $from_f64:expr $(,)?) => {
        impl ElementFloatMath for $type {
            #[inline]
            fn sin(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::sin(($decode)(self)))
            }

            #[inline]
            fn cos(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::cos(($decode)(self)))
            }

            #[inline]
            fn atan2(self, x: Self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::atan2(($decode)(self), ($decode)(x)))
            }

            #[inline]
            fn exp(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::exp(($decode)(self)))
            }

            #[inline]
            fn log(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::ln(($decode)(self)))
            }

            #[inline]
            fn sqrt(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::sqrt(($decode)(self)))
            }

            #[inline]
            fn rsqrt(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::recip(<$work>::sqrt(($decode)(self))))
            }

            #[inline]
            fn tanh(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::tanh(($decode)(self)))
            }

            #[inline]
            fn logistic(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::recip(<$work>::exp(-($decode)(self)) + 1.0))
            }

            #[inline]
            fn pow(self, exponent: Self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::powf(($decode)(self), ($decode)(exponent)))
            }
        }

        impl ElementRealFloatMath for $type {
            #[inline]
            fn erf(self) -> Result<Self, ProgramError> {
                ($from_f64)(erf_f64(($to_f64)(self)))
            }

            #[inline]
            fn floor(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::floor(($decode)(self)))
            }

            #[inline]
            fn ceil(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::ceil(($decode)(self)))
            }

            #[inline]
            fn round(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::round_ties_even(($decode)(self)))
            }
        }

        impl ElementSign for $type {
            fn sign(self) -> Result<Self, ProgramError> {
                let value = ($to_f64)(self);
                if value.is_nan() || value == 0.0 { Ok(self) } else { ($from_f64)(value.signum()) }
            }
        }
    };
}

// Instantiates real math for low-precision formats through their checked f64 conversion contracts.
macro_rules! impl_array_math_for_low_precision_float {
    ($($type:ty),+ $(,)?) => {$(
        impl_array_math_for_real_float!(@low $type);
    )+};
}

impl_array_math_for_low_precision_float!(
    f4e2m1fn,
    f6e2m3fn,
    f6e3m2fn,
    f8e3m4,
    f8e4m3,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e4m3b11fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f8e8m0fnu,
);
impl_array_math_for_real_float!(@half bf16);
impl_array_math_for_real_float!(@half f16);
impl_array_math_for_real_float!(@native f32);
impl_array_math_for_real_float!(@native f64);

// Implements the analytic continuations shared by complex element types. Sine and cosine use `expm1`-based
// hyperbolic components so purely imaginary extreme inputs preserve their non-NaN real/imaginary zero component.
macro_rules! impl_array_math_for_complex {
    ($component:ty) => {
        impl ElementFloatMath for Complex<$component> {
            fn sin(self) -> Result<Self, ProgramError> {
                let expm1_imaginary = self.im.exp_m1();
                let expm1_negative_imaginary = (-self.im).exp_m1();
                let sinh_imaginary = (expm1_imaginary - expm1_negative_imaginary) / 2.0;
                let cosh_imaginary = (expm1_imaginary + expm1_negative_imaginary + 2.0) / 2.0;
                let imaginary = self.re.cos() * sinh_imaginary;
                Ok(Complex::new(if self.re == 0.0 { 0.0 } else { self.re.sin() * cosh_imaginary }, imaginary))
            }

            fn cos(self) -> Result<Self, ProgramError> {
                let expm1_imaginary = self.im.exp_m1();
                let expm1_negative_imaginary = (-self.im).exp_m1();
                let sinh_imaginary = (expm1_imaginary - expm1_negative_imaginary) / 2.0;
                let cosh_imaginary = (expm1_imaginary + expm1_negative_imaginary + 2.0) / 2.0;
                Ok(Complex::new(
                    self.re.cos() * cosh_imaginary,
                    if self.re == 0.0 { 0.0 } else { -self.re.sin() * sinh_imaginary },
                ))
            }

            fn atan2(self, x: Self) -> Result<Self, ProgramError> {
                let imaginary_unit = Complex::new(0.0, 1.0);
                let radius = (x * x + self * self).sqrt();
                Ok(-imaginary_unit * divide_complex_array_element!(x + imaginary_unit * self, radius).ln())
            }

            #[inline]
            fn exp(self) -> Result<Self, ProgramError> {
                Ok(Complex::exp(self))
            }

            #[inline]
            fn log(self) -> Result<Self, ProgramError> {
                Ok(Complex::ln(self))
            }

            #[inline]
            fn sqrt(self) -> Result<Self, ProgramError> {
                Ok(Complex::sqrt(self))
            }

            #[inline]
            fn rsqrt(self) -> Result<Self, ProgramError> {
                Ok(Complex::inv(&Complex::sqrt(self)))
            }

            #[inline]
            fn tanh(self) -> Result<Self, ProgramError> {
                Ok(Complex::tanh(self))
            }

            #[inline]
            fn logistic(self) -> Result<Self, ProgramError> {
                Ok(Complex::inv(&(Complex::exp(-self) + 1.0)))
            }

            #[inline]
            fn pow(self, exponent: Self) -> Result<Self, ProgramError> {
                Ok(Complex::powc(self, exponent))
            }
        }

        impl ElementSign for Complex<$component> {
            fn sign(self) -> Result<Self, ProgramError> {
                let norm = self.norm();
                Ok(if norm == 0.0 { self } else { self / norm })
            }
        }
    };
}

impl_array_math_for_complex!(f32);
impl_array_math_for_complex!(f64);

// Implements sign extraction for primitive signed integers.
macro_rules! impl_array_sign_for_signed_integer {
    ($($type:ty),+ $(,)?) => {$(
        impl ElementSign for $type {
            #[inline]
            fn sign(self) -> Result<Self, ProgramError> {
                Ok(self.signum())
            }
        }
    )+};
}

impl_array_sign_for_signed_integer!(i8, i16, i32, i64);

// Implements sign extraction for checked signed sub-byte integers through their sign-extended values.
macro_rules! impl_array_sign_for_signed_sub_byte_integer {
    ($($type:ty),+ $(,)?) => {$(
        impl ElementSign for $type {
            #[inline]
            fn sign(self) -> Result<Self, ProgramError> {
                Ok(Self::new(self.value().signum()).unwrap())
            }
        }
    )+};
}

impl_array_sign_for_signed_sub_byte_integer!(i1, i2, i4);

/// Element-level minimum and maximum contract shared by reduction and scatter. The identities and selection rules
/// follow JAX's `lax` extrema: Booleans order `false < true`, floating-point extrema propagate NaNs and order `-0`
/// below `+0`, and complex values compare lexicographically by `(real, imaginary)`.
pub(crate) trait ElementExtremum: ArrayElement {
    /// Returns the identity for a maximum reduction.
    fn maximum_identity() -> Self;

    /// Returns the identity for a minimum reduction.
    fn minimum_identity() -> Self;

    /// Returns the maximum of `self` and `right`.
    fn maximum(self, right: Self) -> Self;

    /// Returns the minimum of `self` and `right`.
    fn minimum(self, right: Self) -> Self;
}

// Implements extrema for totally ordered element types whose Rust bounds are also the JAX reduction identities.
macro_rules! impl_array_extrema_for_ordered_element {
    ($type:ty, $minimum:expr, $maximum:expr) => {
        impl ElementExtremum for $type {
            #[inline]
            fn maximum_identity() -> Self {
                $minimum
            }

            #[inline]
            fn minimum_identity() -> Self {
                $maximum
            }

            #[inline]
            fn maximum(self, right: Self) -> Self {
                self.max(right)
            }

            #[inline]
            fn minimum(self, right: Self) -> Self {
                self.min(right)
            }
        }
    };
}

impl_array_extrema_for_ordered_element!(bool, false, true);
impl_array_extrema_for_ordered_element!(i1, i1::MIN, i1::MAX);
impl_array_extrema_for_ordered_element!(i2, i2::MIN, i2::MAX);
impl_array_extrema_for_ordered_element!(i4, i4::MIN, i4::MAX);
impl_array_extrema_for_ordered_element!(i8, i8::MIN, i8::MAX);
impl_array_extrema_for_ordered_element!(i16, i16::MIN, i16::MAX);
impl_array_extrema_for_ordered_element!(i32, i32::MIN, i32::MAX);
impl_array_extrema_for_ordered_element!(i64, i64::MIN, i64::MAX);
impl_array_extrema_for_ordered_element!(u1, u1::MIN, u1::MAX);
impl_array_extrema_for_ordered_element!(u2, u2::MIN, u2::MAX);
impl_array_extrema_for_ordered_element!(u4, u4::MIN, u4::MAX);
impl_array_extrema_for_ordered_element!(u8, u8::MIN, u8::MAX);
impl_array_extrema_for_ordered_element!(u16, u16::MIN, u16::MAX);
impl_array_extrema_for_ordered_element!(u32, u32::MIN, u32::MAX);
impl_array_extrema_for_ordered_element!(u64, u64::MIN, u64::MAX);

/// Selects the larger floating-point element with NaN propagation and `-0` ordered below `+0`. Exact ties preserve
/// `left`, including its original encoding.
fn maximum_float_element<T: Copy>(left: T, right: T, to_f64: impl Fn(T) -> f64) -> T {
    let left_value = to_f64(left);
    let right_value = to_f64(right);
    if left_value.is_nan() {
        left
    } else if right_value.is_nan() || left_value.total_cmp(&right_value) == Ordering::Less {
        right
    } else {
        left
    }
}

/// Selects the smaller floating-point element with NaN propagation and `-0` ordered below `+0`. Exact ties preserve
/// `left`, including its original encoding.
fn minimum_float_element<T: Copy>(left: T, right: T, to_f64: impl Fn(T) -> f64) -> T {
    let left_value = to_f64(left);
    let right_value = to_f64(right);
    if left_value.is_nan() {
        left
    } else if right_value.is_nan() || left_value.total_cmp(&right_value) == Ordering::Greater {
        right
    } else {
        left
    }
}

// Implements floating-point extrema while retaining the selected operand's exact encoding.
macro_rules! impl_array_extrema_for_float {
    ($type:ty, $minimum:expr, $maximum:expr, $to_f64:expr $(,)?) => {
        impl ElementExtremum for $type {
            #[inline]
            fn maximum_identity() -> Self {
                $minimum
            }

            #[inline]
            fn minimum_identity() -> Self {
                $maximum
            }

            #[inline]
            fn maximum(self, right: Self) -> Self {
                maximum_float_element(self, right, $to_f64)
            }

            #[inline]
            fn minimum(self, right: Self) -> Self {
                minimum_float_element(self, right, $to_f64)
            }
        }
    };
}

impl_array_extrema_for_float!(f4e2m1fn, f4e2m1fn::MIN, f4e2m1fn::MAX, f4e2m1fn::to_f64);
impl_array_extrema_for_float!(f6e2m3fn, f6e2m3fn::MIN, f6e2m3fn::MAX, f6e2m3fn::to_f64);
impl_array_extrema_for_float!(f6e3m2fn, f6e3m2fn::MIN, f6e3m2fn::MAX, f6e3m2fn::to_f64);
impl_array_extrema_for_float!(f8e3m4, f8e3m4::NEG_INFINITY, f8e3m4::INFINITY, f8e3m4::to_f64);
impl_array_extrema_for_float!(f8e4m3, f8e4m3::NEG_INFINITY, f8e4m3::INFINITY, f8e4m3::to_f64);
impl_array_extrema_for_float!(f8e4m3fn, f8e4m3fn::MIN, f8e4m3fn::MAX, f8e4m3fn::to_f64);
impl_array_extrema_for_float!(f8e4m3fnuz, f8e4m3fnuz::MIN, f8e4m3fnuz::MAX, f8e4m3fnuz::to_f64);
impl_array_extrema_for_float!(f8e4m3b11fnuz, f8e4m3b11fnuz::MIN, f8e4m3b11fnuz::MAX, f8e4m3b11fnuz::to_f64,);
impl_array_extrema_for_float!(f8e5m2, f8e5m2::NEG_INFINITY, f8e5m2::INFINITY, f8e5m2::to_f64);
impl_array_extrema_for_float!(f8e5m2fnuz, f8e5m2fnuz::MIN, f8e5m2fnuz::MAX, f8e5m2fnuz::to_f64);
impl_array_extrema_for_float!(f8e8m0fnu, f8e8m0fnu::MIN, f8e8m0fnu::MAX, f8e8m0fnu::to_f64);
impl_array_extrema_for_float!(bf16, bf16::NEG_INFINITY, bf16::INFINITY, bf16::to_f64);
impl_array_extrema_for_float!(f16, f16::NEG_INFINITY, f16::INFINITY, f16::to_f64);
impl_array_extrema_for_float!(f32, f32::NEG_INFINITY, f32::INFINITY, f64::from);
impl_array_extrema_for_float!(f64, f64::NEG_INFINITY, f64::INFINITY, |value| value);

// Implements JAX's lexicographic complex extrema. Ordinary floating-point comparisons intentionally make every NaN
// comparison false, so an unordered real or imaginary component selects `right`, exactly like JAX's compare/select
// lowering.
macro_rules! impl_array_extrema_for_complex {
    ($component:ty) => {
        impl ElementExtremum for Complex<$component> {
            #[inline]
            fn maximum_identity() -> Self {
                Complex::new(<$component>::NEG_INFINITY, 0.0)
            }

            #[inline]
            fn minimum_identity() -> Self {
                Complex::new(<$component>::INFINITY, 0.0)
            }

            #[inline]
            fn maximum(self, right: Self) -> Self {
                let select_left = if self.re == right.re { self.im > right.im } else { self.re > right.re };
                if select_left { self } else { right }
            }

            #[inline]
            fn minimum(self, right: Self) -> Self {
                let select_left = if self.re == right.re { self.im < right.im } else { self.re < right.re };
                if select_left { self } else { right }
            }
        }
    };
}

impl_array_extrema_for_complex!(f32);
impl_array_extrema_for_complex!(f64);

impl Array {
    /// Replaces every element of this array in place through one typed function. The physical layout is preserved,
    /// and uniquely owned output buffers are mutated without another payload allocation.
    fn map_elements_in_place<T: ArrayElement>(
        &mut self,
        function: impl Fn(T) -> Result<T, ProgramError>,
    ) -> Result<(), ProgramError> {
        debug_assert_eq!(self.r#type().data_type(), T::data_type());
        let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let bytes = self.storage_bytes_mut();
        for element in 0..addressing.element_count() {
            let range = addressing.byte_range_for_flat_index(element);
            let value = T::decode(&bytes[range.clone()]);
            function(value)?.encode(&mut bytes[range]);
        }
        Ok(())
    }

    /// Reduces typed elements directly from addressed input storage into one addressed output buffer. `identity`
    /// initializes every output cell, including those whose reduced axes are empty.
    fn reduce_elements<T: ArrayElement>(
        &self,
        output_type: ArrayType,
        axes: &[usize],
        identity: T,
        combine: impl Fn(T, T) -> Result<T, ProgramError>,
    ) -> Result<Self, ProgramError> {
        debug_assert_eq!(self.r#type().data_type(), T::data_type());
        debug_assert_eq!(output_type.data_type(), T::data_type());
        let input_shape = self.r#type().static_shape().unwrap();
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut reduce_mask = vec![false; input_shape.rank()];
        axes.iter().for_each(|axis| reduce_mask[*axis] = true);

        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        for output in 0..output_addressing.element_count() {
            identity.encode(&mut bytes[output_addressing.byte_range_for_flat_index(output)]);
        }

        let mut input_index = vec![0usize; input_shape.rank()];
        let mut output_index = vec![0usize; output_type.rank()];
        for _ in 0..input_addressing.element_count() {
            let mut output_axis = 0usize;
            for axis in 0..input_shape.rank() {
                if !reduce_mask[axis] {
                    output_index[output_axis] = input_index[axis];
                    output_axis += 1;
                }
            }
            let input_value = T::decode(&self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]);
            let output_range = output_addressing.byte_range_unchecked(&output_index);
            let value = combine(T::decode(&bytes[output_range.clone()]), input_value)?;
            value.encode(&mut bytes[output_range]);
            input_addressing.advance_index(&mut input_index);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }

    /// Executes a typed sum or mean reduction, sharing the same wrapping addition and applying mean division in
    /// place after accumulation.
    fn reduce_sum_or_mean_elements<T: ElementZero + ElementAdd + ElementDivideByCount>(
        &self,
        output_type: ArrayType,
        axes: &[usize],
        mean: bool,
    ) -> Result<Self, ProgramError> {
        let mut output = self.reduce_elements::<T>(output_type, axes, T::zero()?, T::add)?;
        if mean {
            let shape = self.r#type().static_shape().unwrap();
            let count = axes.iter().map(|axis| shape[*axis]).product::<usize>().max(1);
            output.map_elements_in_place::<T>(|value| value.divide_by_count(count))?;
        }
        Ok(output)
    }

    /// Computes one generalized dot directly over typed elements in each operand's physical storage. Logical index
    /// construction follows the StableHLO output order `[batching..., lhs_result..., rhs_result...]`. No operand-sized
    /// payload is materialized: the only payload allocation is the result, and temporary index state is bounded by
    /// operand rank.
    fn dot_elements<T: ElementZero + ElementAdd + ElementMul>(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        debug_assert_eq!(self.r#type().data_type(), T::data_type());
        debug_assert_eq!(rhs.r#type().data_type(), T::data_type());
        let mut output_types = DotOperation::new(dimensions.clone())
            .infer_output_types(&[self.r#type().into_owned(), rhs.r#type().into_owned()], &[])?;
        let output_type = output_types.remove(0);
        let lhs_shape = self.r#type().static_shape().unwrap();
        let rhs_shape = rhs.r#type().static_shape().unwrap();
        let output_shape = output_type.static_shape().unwrap();
        let output_strides = output_shape.row_major_strides();
        let lhs_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let rhs_addressing = ArrayAddressing::new(rhs.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;

        let lhs_batching = dimensions.lhs_batching_dimensions();
        let rhs_batching = dimensions.rhs_batching_dimensions();
        let lhs_contracting = dimensions.lhs_contracting_dimensions();
        let rhs_contracting = dimensions.rhs_contracting_dimensions();
        let lhs_result = (0..lhs_shape.rank())
            .filter(|axis| !lhs_batching.contains(axis) && !lhs_contracting.contains(axis))
            .collect::<Vec<_>>();
        let rhs_result = (0..rhs_shape.rank())
            .filter(|axis| !rhs_batching.contains(axis) && !rhs_contracting.contains(axis))
            .collect::<Vec<_>>();
        let contracting_shape =
            StaticShape::new(lhs_contracting.iter().map(|axis| lhs_shape[*axis]).collect::<Vec<_>>());
        let contracting_strides = contracting_shape.row_major_strides();
        let contracting_count = contracting_shape.dimensions().iter().product();

        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let mut lhs_index = vec![0usize; lhs_shape.rank()];
        let mut rhs_index = vec![0usize; rhs_shape.rank()];
        for output_flat in 0..output_addressing.element_count() {
            // Decode the result coordinate directly into the corresponding batch and non-contracting operand axes.
            let mut output_axis = 0usize;
            for (&lhs_axis, &rhs_axis) in lhs_batching.iter().zip(rhs_batching) {
                let coordinate = (output_flat / output_strides[output_axis]) % output_shape[output_axis];
                lhs_index[lhs_axis] = coordinate;
                rhs_index[rhs_axis] = coordinate;
                output_axis += 1;
            }
            for &lhs_axis in &lhs_result {
                lhs_index[lhs_axis] = (output_flat / output_strides[output_axis]) % output_shape[output_axis];
                output_axis += 1;
            }
            for &rhs_axis in &rhs_result {
                rhs_index[rhs_axis] = (output_flat / output_strides[output_axis]) % output_shape[output_axis];
                output_axis += 1;
            }

            let mut accumulator = T::zero()?;
            for contracting_flat in 0..contracting_count {
                for (contracting_axis, (&lhs_axis, &rhs_axis)) in
                    lhs_contracting.iter().zip(rhs_contracting).enumerate()
                {
                    let coordinate = (contracting_flat / contracting_strides[contracting_axis])
                        % contracting_shape[contracting_axis];
                    lhs_index[lhs_axis] = coordinate;
                    rhs_index[rhs_axis] = coordinate;
                }
                let lhs_value = T::decode(&self.storage_bytes()[lhs_addressing.byte_range_unchecked(&lhs_index)]);
                let rhs_value = T::decode(&rhs.storage_bytes()[rhs_addressing.byte_range_unchecked(&rhs_index)]);
                accumulator = accumulator.add(lhs_value.mul(rhs_value)?)?;
            }
            accumulator.encode(&mut bytes[output_addressing.byte_range_for_flat_index(output_flat)]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl Abs for Array {
    fn abs(&self) -> Result<Self, ProgramError> {
        // The absolute value of a complex array is its elementwise magnitude, so the element data type maps to its
        // real part data type, mirroring the `AbsOperation` type-inference contract.
        let data_type = match self.r#type().data_type() {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            other => other,
        };
        let output_type = self.r#type().into_owned().with_data_type(data_type);
        if Self::element_count(&output_type) == 0 {
            let addressing = ArrayAddressing::new(output_type.clone())?;
            return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
        }
        let input_type = self.r#type().data_type();
        if !((input_type.is_signed() && input_type != DataType::I1)
            || input_type.is_floating_point()
            || input_type.is_complex())
        {
            return Err(TypeError::invalid(format!(
                "cannot compute the absolute value of a scalar of data type {input_type}",
            ))
            .into());
        }
        dispatch_on_array_element_type!(@numeric input_type, |Element| {
            self.map_elements::<Element, <Element as ElementAbs>::Output>(output_type, |value| {
                <Element as ElementAbs>::abs(value)
            })
        })
    }
}

impl Neg for Array {
    fn neg(&self) -> Result<Self, ProgramError> {
        if Self::element_count(self.r#type().as_ref()) == 0 {
            let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
            return Ok(Self::new_unchecked(
                self.r#type().into_owned(),
                Arc::new(vec![0; addressing.storage_byte_len()]),
            ));
        }
        let data_type = self.r#type().data_type();
        if !data_type.is_numeric() {
            return Err(TypeError::invalid(format!("cannot negate a scalar of data type {data_type}")).into());
        }
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            self.map_elements::<Element, Element>(self.r#type().into_owned(), <Element as ElementNeg>::neg)
        })
    }
}

impl std::ops::Neg for Array {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Neg::neg(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

macro_rules! impl_array_binary_arithmetic {
    // Generates one numeric binary capability over the shared promotion, broadcasting, and typed-codec path.
    (@numeric $trait:ident, $method:ident, $element_trait:ident) => {
        impl $trait for Array {
            fn $method(&self, right: &Self) -> Result<Self, ProgramError> {
                let output_type = Broadcastable::broadcast(self.r#type().as_ref(), right.r#type().as_ref())
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                if Self::element_count(&output_type) == 0 {
                    let addressing = ArrayAddressing::new(output_type.clone())?;
                    return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
                }
                if !self.r#type().data_type().is_numeric() || !right.r#type().data_type().is_numeric() {
                    return Err(TypeError::invalid(format!(
                        "cannot apply `{}` to scalars of data types {} and {}",
                        stringify!($method),
                        self.r#type().data_type(),
                        right.r#type().data_type(),
                    ))
                    .into());
                }
                let data_type = output_type.data_type();
                let left = self.promoted_to(data_type)?;
                let right = right.promoted_to(data_type)?;
                dispatch_on_array_element_type!(@numeric data_type, |Element| {
                    left.binary_elements::<Element, Element>(&right, output_type, |left, right| {
                        <Element as $element_trait>::$method(left, right)
                    })
                })
            }
        }
    };

    // Generates a real-only binary capability whose invalid-type diagnostic uses a descriptive operation noun.
    (@real $trait:ident, $method:ident, $element_trait:ident, $noun:literal) => {
        impl $trait for Array {
            fn $method(&self, right: &Self) -> Result<Self, ProgramError> {
                let output_type = Broadcastable::broadcast(self.r#type().as_ref(), right.r#type().as_ref())
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                if Self::element_count(&output_type) == 0 {
                    let addressing = ArrayAddressing::new(output_type.clone())?;
                    return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
                }
                if !self.r#type().data_type().is_real() || !right.r#type().data_type().is_real() {
                    return Err(TypeError::invalid(format!(
                        concat!("cannot compute the ", $noun, " of scalars of data types {} and {}"),
                        self.r#type().data_type(),
                        right.r#type().data_type(),
                    ))
                    .into());
                }
                let data_type = output_type.data_type();
                let left = self.promoted_to(data_type)?;
                let right = right.promoted_to(data_type)?;
                dispatch_on_array_element_type!(@real data_type, |Element| {
                    left.binary_elements::<Element, Element>(&right, output_type, |left, right| {
                        <Element as $element_trait>::$method(left, right)
                    })
                })
            }
        }
    };

    // Generates a real-only extremum capability that selects and preserves one operand's exact element encoding.
    (@extremum $trait:ident, $method:ident, $element_method:ident, $noun:literal) => {
        impl $trait for Array {
            fn $method(&self, right: &Self) -> Result<Self, ProgramError> {
                let output_type = Broadcastable::broadcast(self.r#type().as_ref(), right.r#type().as_ref())
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                if Self::element_count(&output_type) == 0 {
                    let addressing = ArrayAddressing::new(output_type.clone())?;
                    return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
                }
                if !self.r#type().data_type().is_real() || !right.r#type().data_type().is_real() {
                    return Err(TypeError::invalid(format!(
                        concat!("cannot compute the ", $noun, " of scalars of data types {} and {}"),
                        self.r#type().data_type(),
                        right.r#type().data_type(),
                    ))
                    .into());
                }
                let data_type = output_type.data_type();
                let left = self.promoted_to(data_type)?;
                let right = right.promoted_to(data_type)?;
                dispatch_on_array_element_type!(@real data_type, |Element| {
                    left.binary_elements::<Element, Element>(&right, output_type, |left, right| {
                        Ok(<Element as ElementExtremum>::$element_method(left, right))
                    })
                })
            }
        }
    };
}

impl_array_binary_arithmetic!(@numeric Add, add, ElementAdd);
impl_array_binary_arithmetic!(@numeric Sub, sub, ElementSub);
impl_array_binary_arithmetic!(@numeric Mul, mul, ElementMul);
impl_array_binary_arithmetic!(@numeric Div, div, ElementDiv);

impl std::ops::Add for Array {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Add::add(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Sub for Array {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Sub::sub(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Mul for Array {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Mul::mul(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Mul<f64> for Array {
    type Output = Self;

    /// Scales every element by `rhs`, converting `rhs` into this array's element data type first so that scaling
    /// preserves the array's type (e.g., scaling an `f32` array does not promote it to `f64`).
    fn mul(self, rhs: f64) -> Self::Output {
        let data_type = self.r#type().data_type();
        let factor = dispatch_on_array_element_type!(data_type, |Element| {
            Self::scalar(<Element as ElementConversionTarget>::from_real(rhs).unwrap_or_else(|error| panic!("{error}")))
        });
        Mul::mul(&self, &factor).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl std::ops::Div for Array {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Div::div(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
    }
}

macro_rules! impl_array_unary_math {
    // Generates a unary operation supported by both real floating-point and complex elements.
    (@float_math $trait:ident, $method:ident, $noun:literal) => {
        impl $trait for Array {
            fn $method(&self) -> Result<Self, ProgramError> {
                if Self::element_count(self.r#type().as_ref()) == 0 {
                    let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
                    return Ok(Self::new_unchecked(self.r#type().into_owned(), Arc::new(vec![0; addressing.storage_byte_len()])));
                }
                let data_type = self.r#type().data_type();
                if !data_type.is_floating_point() && !data_type.is_complex() {
                    return Err(TypeError::invalid(format!(
                        concat!("cannot compute the ", $noun, " of a scalar of data type {}"),
                        data_type,
                    ))
                    .into());
                }
                if data_type.is_complex() {
                    dispatch_on_array_element_type!(@complex data_type, |Element| {
                        self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                            <Element as ElementFloatMath>::$method(value)
                        })
                    })
                } else {
                    dispatch_on_array_element_type!(@float data_type, |Element| {
                        self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                            <Element as ElementFloatMath>::$method(value)
                        })
                    })
                }
            }
        }
    };

    // Generates a unary operation supported only by real floating-point elements.
    (@real_float $trait:ident, $method:ident, $error:literal) => {
        impl $trait for Array {
            fn $method(&self) -> Result<Self, ProgramError> {
                if Self::element_count(self.r#type().as_ref()) == 0 {
                    let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
                    return Ok(Self::new_unchecked(self.r#type().into_owned(), Arc::new(vec![0; addressing.storage_byte_len()])));
                }
                let data_type = self.r#type().data_type();
                if !data_type.is_floating_point() {
                    return Err(TypeError::invalid(format!($error, data_type)).into());
                }
                dispatch_on_array_element_type!(@float data_type, |Element| {
                    self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                        <Element as ElementRealFloatMath>::$method(value)
                    })
                })
            }
        }
    };

    // Generates sign extraction over its disjoint signed-integer, floating-point, and complex element classes.
    (@sign) => {
        impl Sign for Array {
            fn sign(&self) -> Result<Self, ProgramError> {
                if Self::element_count(self.r#type().as_ref()) == 0 {
                    let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
                    return Ok(Self::new_unchecked(self.r#type().into_owned(), Arc::new(vec![0; addressing.storage_byte_len()])));
                }
                let data_type = self.r#type().data_type();
                if !data_type.is_signed() && !data_type.is_floating_point() && !data_type.is_complex() {
                    return Err(TypeError::invalid(format!(
                        "cannot compute the sign of a scalar of data type {}",
                        data_type,
                    ))
                    .into());
                }
                if data_type.is_signed() {
                    dispatch_on_array_element_type!(@signed data_type, |Element| {
                        self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                            <Element as ElementSign>::sign(value)
                        })
                    })
                } else if data_type.is_complex() {
                    dispatch_on_array_element_type!(@complex data_type, |Element| {
                        self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                            <Element as ElementSign>::sign(value)
                        })
                    })
                } else {
                    dispatch_on_array_element_type!(@float data_type, |Element| {
                        self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                            <Element as ElementSign>::sign(value)
                        })
                    })
                }
            }
        }
    };
}

macro_rules! impl_array_binary_float_math {
    ($trait:ident, $method:ident, $argument:ident) => {
        impl $trait for Array {
            fn $method(&self, $argument: &Self) -> Result<Self, ProgramError> {
                let output_type = Broadcastable::broadcast(self.r#type().as_ref(), $argument.r#type().as_ref())
                    .map_err(|error| TypeError::invalid(error.to_string()))?;
                if Self::element_count(&output_type) == 0 {
                    let addressing = ArrayAddressing::new(output_type.clone())?;
                    return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
                }
                let left_type = self.r#type().data_type();
                let right_type = $argument.r#type().data_type();
                check_types!(@float, stringify!($method), [left_type, right_type]);
                let data_type = output_type.data_type();
                let left = self.promoted_to(data_type)?;
                let right = $argument.promoted_to(data_type)?;
                if data_type.is_complex() {
                    dispatch_on_array_element_type!(@complex data_type, |Element| {
                        left.binary_elements::<Element, Element>(&right, output_type, |left, right| {
                            <Element as ElementFloatMath>::$method(left, right)
                        })
                    })
                } else {
                    dispatch_on_array_element_type!(@float data_type, |Element| {
                        left.binary_elements::<Element, Element>(&right, output_type, |left, right| {
                            <Element as ElementFloatMath>::$method(left, right)
                        })
                    })
                }
            }
        }
    };
}

impl_array_unary_math!(@float_math Sin, sin, "sine");
impl_array_unary_math!(@float_math Cos, cos, "cosine");
impl_array_binary_float_math!(Atan2, atan2, x);
impl_array_unary_math!(@float_math Exp, exp, "exponential");
impl_array_unary_math!(@float_math Log, log, "logarithm");
impl_array_unary_math!(@float_math Sqrt, sqrt, "square root");
impl_array_unary_math!(@float_math Rsqrt, rsqrt, "reciprocal square root");
impl_array_unary_math!(@float_math Tanh, tanh, "hyperbolic tangent");
impl_array_unary_math!(@float_math Logistic, logistic, "logistic");
impl_array_unary_math!(@real_float Erf, erf, "cannot compute the error function of a scalar of data type {}");
impl_array_binary_float_math!(Pow, pow, exponent);
impl_array_unary_math!(@sign);
impl_array_unary_math!(@real_float Floor, floor, "cannot compute the floor of a scalar of data type {}");
impl_array_unary_math!(@real_float Ceil, ceil, "cannot compute the ceiling of a scalar of data type {}");
impl_array_unary_math!(@real_float Round, round, "cannot round a scalar of data type {}");

impl_array_binary_arithmetic!(@extremum Max, max, maximum, "maximum");
impl_array_binary_arithmetic!(@extremum Min, min, minimum, "minimum");
impl_array_binary_arithmetic!(@real Rem, rem, ElementRem, "remainder");

impl ScaledDot for Array {
    fn scaled_dot(
        &self,
        lhs_scales: &Self,
        rhs: &Self,
        rhs_scales: &Self,
        block_size: usize,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError> {
        scaled_dot_composition(self, lhs_scales, rhs, rhs_scales, None, block_size, accumulation_type)
    }

    fn scaled_dot_with_global_scale(
        &self,
        lhs_scales: &Self,
        rhs: &Self,
        rhs_scales: &Self,
        global_scale: &Self,
        block_size: usize,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError> {
        scaled_dot_composition(self, lhs_scales, rhs, rhs_scales, Some(global_scale), block_size, accumulation_type)
    }
}

impl Dot for Array {
    /// Computes an accumulation-typed dot by upcasting both operands to `accumulation_type` and delegating to the
    /// ordinary evaluator, which is exactly the upcast-then-accumulate contract of
    /// [`DotOperation::with_accumulation_type`].
    fn dot_with_accumulation_type(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Self {
        let lhs = self.convert_element_type(accumulation_type).unwrap_or_else(|error| panic!("{error}"));
        let rhs = rhs.convert_element_type(accumulation_type).unwrap_or_else(|error| panic!("{error}"));
        lhs.dot(&rhs, dimensions)
    }

    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Self {
        // TODO(eaplatanios): What about the accumulation type?
        let data_type = self.r#type().data_type();
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            self.dot_elements::<Element>(rhs, dimensions)
        })
        .unwrap_or_else(|error| panic!("{error}"))
    }
}

impl Reduce for Array {
    fn reduce(&self, axes: &[usize], kind: ReductionKind) -> Self {
        if axes.is_empty() {
            return self.clone();
        }
        let data_type = self.r#type().data_type();
        // Reuse the abstract rule for validation and for the complete result metadata. The concrete kernel below then
        // decodes directly from the input's physical layout into the result's addressed storage.
        let output_type =
            reduce_abstract(self.r#type().as_ref(), axes, kind, "reduce").unwrap_or_else(|error| panic!("{error}"));
        if data_type == DataType::Zero {
            return Self::new(output_type, Vec::new()).unwrap();
        }
        let output = match kind {
            ReductionKind::Sum | ReductionKind::Mean => {
                dispatch_on_array_element_type!(@numeric data_type, |Element| {
                    self.reduce_sum_or_mean_elements::<Element>(
                        output_type,
                        axes,
                        kind == ReductionKind::Mean,
                    )
                })
            }
            ReductionKind::Max | ReductionKind::Min => {
                dispatch_on_array_element_type!(data_type, |Element| {
                    let identity = if kind == ReductionKind::Max {
                        <Element as ElementExtremum>::maximum_identity()
                    } else {
                        <Element as ElementExtremum>::minimum_identity()
                    };
                    self.reduce_elements::<Element>(output_type, axes, identity, |left, right| {
                        Ok(if kind == ReductionKind::Max {
                            <Element as ElementExtremum>::maximum(left, right)
                        } else {
                            <Element as ElementExtremum>::minimum(left, right)
                        })
                    })
                })
            }
            ReductionKind::Any => {
                self.reduce_elements::<bool>(output_type, axes, false, |left, right| Ok(left | right))
            }
            ReductionKind::All => self.reduce_elements::<bool>(output_type, axes, true, |left, right| Ok(left & right)),
        };
        output.unwrap_or_else(|error| panic!("{error}"))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::f16;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::array_type;
    use crate::arrays::encoding::{f8e4m3fn, f8e8m0fnu, i2, i4};
    use crate::arrays::types::layouts::{Layout, StridedLayout};
    use crate::operations::complex::Complex;
    use crate::operations::math::erf::erf_f64;
    use crate::programs::Typed;

    use super::*;

    #[test]
    fn test_array_arithmetic() {
        // Elementwise arithmetic with scalar broadcasting.
        let vector = Array::vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(vector.add(&Array::scalar(1.0)).unwrap(), Array::vector(vec![2.0, 3.0, 4.0]));
        assert_eq!(vector.sub(&Array::vector(vec![0.5, 1.0, 1.5])).unwrap(), Array::vector(vec![0.5, 1.0, 1.5]));
        assert_eq!(vector.mul(&vector).unwrap(), Array::vector(vec![1.0, 4.0, 9.0]));
        assert_eq!(vector.div(&Array::scalar(2.0)).unwrap(), Array::vector(vec![0.5, 1.0, 1.5]));
        assert_eq!(vector.neg().unwrap(), Array::vector(vec![-1.0, -2.0, -3.0]));
        // Mixed-precision operands promote to the common element data type.
        let promoted = Array::vector(vec![1.0f32, 2.0]).add(&Array::vector(vec![0.5f64, 0.5])).unwrap();
        assert_eq!(promoted, Array::vector(vec![1.5f64, 2.5]));
        // General broadcasting traverses arbitrary input layouts while mixed element types normalize through the
        // canonical conversion kernel.
        let left_type =
            array_type(DataType::F32, &[2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 4])));
        let left = Array::from_elements(left_type, &[1.0f32, 2.0]).unwrap();
        let right_type =
            array_type(DataType::F64, &[1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![24, -8])));
        let right = Array::from_elements(right_type, &[0.5f64, 1.0, 1.5]).unwrap();
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().into_owned(), array_type(DataType::F64, &[2, 3]));
        assert_eq!(sum.elements::<f64>(), Ok(vec![1.5, 2.0, 2.5, 2.5, 3.0, 3.5]));
        // The `std::ops` sugar delegates to the fallible capabilities.
        assert_eq!(vector.clone() + Array::scalar(1.0), Array::vector(vec![2.0, 3.0, 4.0]));
        assert_eq!(-vector.clone(), Array::vector(vec![-1.0, -2.0, -3.0]));
        // Scaling by an `f64` preserves the array's element data type.
        let scaled = Array::vector(vec![1.0f32, 2.0]) * 2.0;
        assert_eq!(scaled, Array::vector(vec![2.0f32, 4.0]));
        // Integer arithmetic wraps deterministically, matching the scalar reference backend.
        let wrapped = Array::vector(vec![255u8]).add(&Array::vector(vec![1u8])).unwrap();
        assert_eq!(wrapped.elements::<u8>(), Ok(vec![0]));
    }

    #[test]
    fn test_array_low_precision_float_arithmetic() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_f64s(array_type(DataType::F8E4M3FN, &[2]), vec![1.0, 2.0]);
        let right = Array::from_f64s(array_type(DataType::F8E4M3FN, &[2]), vec![0.5, 0.25]);
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().into_owned(), array_type(DataType::F8E4M3FN, &[2]));
        assert_eq!(sum.to_f64s(), vec![1.5, 2.25]);
        assert_eq!(left.sub(&right).unwrap().to_f64s(), vec![0.5, 1.75]);
        assert_eq!(left.mul(&right).unwrap().to_f64s(), vec![0.5, 0.5]);
        assert_eq!(left.div(&right).unwrap().to_f64s(), vec![2.0, 8.0]);
        assert_eq!(left.rem(&right).unwrap().to_f64s(), vec![0.0, 0.0]);
        assert_eq!(left.neg().unwrap().to_f64s(), vec![-1.0, -2.0]);
        assert_eq!(left.neg().unwrap().abs().unwrap(), left);
    }

    #[test]
    fn test_array_math() {
        let vector = Array::vector(vec![0.0, 1.0]);
        assert_abs_diff_eq!(vector.sin().unwrap(), Array::vector(vec![0.0, 1.0f64.sin()]), epsilon = 1e-12);
        assert_abs_diff_eq!(vector.cos().unwrap(), Array::vector(vec![1.0, 1.0f64.cos()]), epsilon = 1e-12);
        assert_abs_diff_eq!(vector.exp().unwrap(), Array::vector(vec![1.0, 1.0f64.exp()]), epsilon = 1e-12);
        assert_abs_diff_eq!(
            Array::vector(vec![1.0, 4.0]).sqrt().unwrap(),
            Array::vector(vec![1.0, 2.0]),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(
            Array::vector(vec![1.0, std::f64::consts::E]).log().unwrap(),
            Array::vector(vec![0.0, 1.0]),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(
            Array::vector(vec![1.0]).atan2(&Array::vector(vec![1.0])).unwrap(),
            Array::vector(vec![std::f64::consts::FRAC_PI_4]),
            epsilon = 1e-12,
        );
        assert_eq!(Array::vector(vec![-1.5, 2.5]).abs().unwrap(), Array::vector(vec![1.5, 2.5]));
        // The absolute value of a complex array is its elementwise magnitude with a real element data type.
        let complex = Array::vector(vec![3.0]).complex(&Array::vector(vec![4.0])).unwrap();
        let magnitude = complex.abs().unwrap();
        assert_eq!(magnitude.r#type().into_owned(), array_type(DataType::F64, &[1]));
        assert_abs_diff_eq!(magnitude, Array::vector(vec![5.0]), epsilon = 1e-12);
        // Elementwise extrema retain the selected operand's NaN payload and IEEE signed-zero encoding.
        let nan = f32::from_bits(0x7fc0_1234);
        assert_eq!(
            Array::scalar(nan).max(&Array::scalar(1.0f32)).unwrap().elements::<f32>().unwrap()[0].to_bits(),
            nan.to_bits(),
        );
        assert_eq!(
            Array::scalar(-0.0f32).max(&Array::scalar(0.0f32)).unwrap().elements::<f32>().unwrap()[0].to_bits(),
            0.0f32.to_bits(),
        );
        assert_eq!(
            Array::scalar(-0.0f32).min(&Array::scalar(0.0f32)).unwrap().elements::<f32>().unwrap()[0].to_bits(),
            (-0.0f32).to_bits(),
        );
    }

    #[test]
    fn test_array_transcendental_math_uses_typed_storage() {
        // Unary kernels preserve arbitrary physical layouts while traversing elements in logical order.
        let input_type = array_type(DataType::F64, &[2]).with_layout(Layout::Strided(StridedLayout::new(vec![-16])));
        let input = Array::from_elements(input_type.clone(), &[0.0f64, 1.0]).unwrap();
        let exponential = input.exp().unwrap();
        assert_eq!(exponential.r#type().as_ref(), &input_type);
        assert_eq!(exponential.elements::<f64>(), Ok(vec![1.0, 1.0f64.exp()]));

        // Binary kernels perform complete broadcasting after promoting both physical inputs to their common type.
        let left_type =
            array_type(DataType::F32, &[2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 4])));
        let left = Array::from_elements(left_type, &[0.0f32, 1.0]).unwrap();
        let right_type =
            array_type(DataType::F64, &[1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![24, -8])));
        let right = Array::from_elements(right_type, &[1.0f64, 1.0, -1.0]).unwrap();
        let angles = left.atan2(&right).unwrap();
        assert_eq!(angles.r#type().into_owned(), array_type(DataType::F64, &[2, 3]));
        assert_abs_diff_eq!(
            angles,
            Array::matrix(
                2,
                3,
                vec![
                    0.0,
                    0.0,
                    std::f64::consts::PI,
                    std::f64::consts::FRAC_PI_4,
                    std::f64::consts::FRAC_PI_4,
                    3.0 * std::f64::consts::FRAC_PI_4
                ],
            ),
            epsilon = 1e-12,
        );
        let bases = Array::matrix(2, 1, vec![2.0f32, 3.0]);
        let exponents = Array::matrix(1, 3, vec![1.0f64, 2.0, 3.0]);
        assert_eq!(bases.pow(&exponents).unwrap(), Array::matrix(2, 3, vec![2.0f64, 4.0, 8.0, 3.0, 9.0, 27.0]),);
        assert!(matches!(
            Array::scalar(1i32).atan2(&Array::scalar(1.0f64)),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "'atan2' does not support input data type i32",
        ));
        assert!(matches!(
            Array::scalar(2.0f64).pow(&Array::scalar(3i32)),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "'pow' does not support input data type i32",
        ));

        // Low-precision formats decode, compute, and re-encode without constructing intermediary scalar values.
        let low_precision = Array::from_elements(
            array_type(DataType::F8E4M3FN, &[2]),
            &[f8e4m3fn::from_f64(0.0).unwrap(), f8e4m3fn::from_f64(1.0).unwrap()],
        )
        .unwrap();
        assert_eq!(
            low_precision.exp().unwrap().to_f64s(),
            vec![1.0, f8e4m3fn::from_f64(1.0f64.exp()).unwrap().to_f64()]
        );
    }

    #[test]
    fn test_array_real_float_math_and_sign_use_typed_storage() {
        let input = Array::vector(vec![-1.5f64, -0.0, 2.5, 3.5]);
        assert_eq!(input.floor().unwrap(), Array::vector(vec![-2.0, -0.0, 2.0, 3.0]));
        assert_eq!(input.ceil().unwrap(), Array::vector(vec![-1.0, -0.0, 3.0, 4.0]));
        assert_eq!(input.round().unwrap(), Array::vector(vec![-2.0, -0.0, 2.0, 4.0]));
        assert_eq!(Array::vector(vec![1.0f64, 4.0]).rsqrt().unwrap(), Array::vector(vec![1.0, 0.5]));
        assert_abs_diff_eq!(
            Array::vector(vec![-1.0f64, 0.0, 1.0]).erf().unwrap(),
            Array::vector(vec![erf_f64(-1.0), 0.0, erf_f64(1.0)]),
            epsilon = 1e-12,
        );

        // Sign preserves IEEE signed zero and NaN behavior and also covers signed sub-byte integers.
        let signs = Array::from_elements(
            array_type(DataType::F64, &[4]),
            &[-2.0f64, -0.0, 0.0, f64::from_bits(0x7ff8_0000_0000_1234)],
        )
        .unwrap()
        .sign()
        .unwrap()
        .elements::<f64>()
        .unwrap();
        assert_eq!(signs[0], -1.0);
        assert_eq!(signs[1].to_bits(), (-0.0f64).to_bits());
        assert_eq!(signs[2].to_bits(), 0.0f64.to_bits());
        assert_eq!(signs[3].to_bits(), 0x7ff8_0000_0000_1234);
        let narrow =
            Array::from_elements(array_type(DataType::I2, &[3]), &[i2::MIN, i2::new(0).unwrap(), i2::MAX]).unwrap();
        assert_eq!(
            narrow.sign().unwrap().elements::<i2>(),
            Ok(vec![i2::new(-1).unwrap(), i2::new(0).unwrap(), i2::new(1).unwrap()]),
        );
        assert!(matches!(
            Array::scalar(1u8).sign(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot compute the sign of a scalar of data type u8",
        ));
    }

    #[test]
    fn test_array_reduce() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(matrix.reduce(&[1], ReductionKind::Sum), Array::vector(vec![6.0, 15.0]));
        assert_eq!(matrix.reduce(&[1], ReductionKind::Mean), Array::vector(vec![2.0, 5.0]));
        assert_eq!(
            matrix.reduce(&[0, 1], ReductionKind::Sum),
            Array::from_f64s(array_type(DataType::F64, &[]), vec![21.0])
        );
        assert_eq!(matrix.reduce(&[], ReductionKind::Sum), matrix);
        // Max and min use the data type's reduction identities and ordinary ordering.
        let integers = Array::vector(vec![3i32, -1, 2]);
        assert_eq!(integers.reduce(&[0], ReductionKind::Max).elements::<i32>(), Ok(vec![3]));
        assert_eq!(integers.reduce(&[0], ReductionKind::Min).elements::<i32>(), Ok(vec![-1]));
        // Boolean reductions.
        let booleans = Array::vector(vec![true, false, true]);
        assert_eq!(booleans.reduce(&[0], ReductionKind::Any).elements::<bool>(), Ok(vec![true]));
        assert_eq!(booleans.reduce(&[0], ReductionKind::All).elements::<bool>(), Ok(vec![false]));
        assert_eq!(booleans.reduce(&[0], ReductionKind::Max).elements::<bool>(), Ok(vec![true]));
        assert_eq!(booleans.reduce(&[0], ReductionKind::Min).elements::<bool>(), Ok(vec![false]));

        // Numeric and Boolean reductions traverse arbitrary layouts and produce the abstract rule's dense result.
        let r#type = array_type(DataType::U16, &[2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 2])));
        let matrix = Array::from_elements(r#type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(matrix.reduce(&[1], ReductionKind::Sum).elements::<u16>(), Ok(vec![6, 15]));
        let r#type = array_type(DataType::Boolean, &[3]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
        let booleans = Array::from_elements(r#type, &[true, false, true]).unwrap();
        assert_eq!(booleans.reduce(&[0], ReductionKind::Any).elements::<bool>(), Ok(vec![true]));

        // Sub-byte accumulation wraps in the declared width, and low-precision accumulation re-encodes each step.
        let narrow = Array::matrix(
            2,
            2,
            vec![i4::new(7).unwrap(), i4::new(2).unwrap(), i4::new(-8).unwrap(), i4::new(-3).unwrap()],
        );
        assert_eq!(
            narrow.reduce(&[1], ReductionKind::Sum).elements::<i4>(),
            Ok(vec![i4::new(-7).unwrap(), i4::new(5).unwrap()]),
        );
        let low_precision = Array::vector(vec![f8e4m3fn::from_f64(1.0).unwrap(), f8e4m3fn::from_f64(0.5).unwrap()]);
        assert_eq!(
            low_precision.reduce(&[0], ReductionKind::Sum).elements::<f8e4m3fn>(),
            Ok(vec![f8e4m3fn::from_f64(1.5).unwrap()]),
        );

        // Complex sums and means preserve both components, while empty sums materialize the numeric identity.
        let complex = Array::vector(vec![ComplexNumber::new(2.0f32, 4.0), ComplexNumber::new(4.0, 8.0)]);
        assert_eq!(
            complex.reduce(&[0], ReductionKind::Sum).elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(6.0, 12.0)]),
        );
        assert_eq!(
            complex.reduce(&[0], ReductionKind::Mean).elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(3.0, 6.0)]),
        );
        let empty = Array::from_elements::<i32>(array_type(DataType::I32, &[2, 0]), &[]).unwrap();
        assert_eq!(empty.reduce(&[1], ReductionKind::Sum).elements::<i32>(), Ok(vec![0, 0]));

        // Floating-point extrema propagate NaNs and order negative zero below positive zero.
        let nan = Array::vector(vec![1.0f32, f32::NAN]);
        assert!(nan.reduce(&[0], ReductionKind::Max).elements::<f32>().unwrap()[0].is_nan());
        let zeros = Array::vector(vec![-0.0f32, 0.0]);
        assert_eq!(zeros.reduce(&[0], ReductionKind::Max).elements::<f32>().unwrap()[0].to_bits(), 0.0f32.to_bits(),);
        assert_eq!(zeros.reduce(&[0], ReductionKind::Min).elements::<f32>().unwrap()[0].to_bits(), (-0.0f32).to_bits(),);

        // Complex extrema compare `(real, imaginary)` lexicographically, including their JAX-compatible identities.
        let complex = Array::vector(vec![
            ComplexNumber::new(1.0f32, 5.0),
            ComplexNumber::new(2.0, -3.0),
            ComplexNumber::new(2.0, 4.0),
        ]);
        assert_eq!(
            complex.reduce(&[0], ReductionKind::Max).elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(2.0, 4.0)]),
        );
        assert_eq!(
            complex.reduce(&[0], ReductionKind::Min).elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(1.0, 5.0)]),
        );
        let empty = Array::from_elements::<ComplexNumber<f32>>(array_type(DataType::C64, &[2, 0]), &[]).unwrap();
        assert_eq!(
            empty.reduce(&[1], ReductionKind::Max).elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(f32::NEG_INFINITY, 0.0), ComplexNumber::new(f32::NEG_INFINITY, 0.0),]),
        );
        let empty = Array::from_elements::<f8e8m0fnu>(array_type(DataType::F8E8M0FNU, &[2, 0]), &[]).unwrap();
        assert_eq!(
            empty.reduce(&[1], ReductionKind::Max).elements::<f8e8m0fnu>(),
            Ok(vec![f8e8m0fnu::MIN, f8e8m0fnu::MIN]),
        );
        assert_eq!(
            empty.reduce(&[1], ReductionKind::Min).elements::<f8e8m0fnu>(),
            Ok(vec![f8e8m0fnu::MAX, f8e8m0fnu::MAX]),
        );
    }

    #[test]
    fn test_array_dot() {
        // Ordinary matrix multiplication uses the generalized contraction order.
        let lhs = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let dimensions = DotDimensionNumbers::new(vec![1], vec![0], vec![], vec![]);
        let product = lhs.dot(&rhs, &dimensions);
        assert_eq!(product.r#type().into_owned(), array_type(DataType::F64, &[2, 2]));
        assert_eq!(product.to_f64s(), vec![58.0, 64.0, 139.0, 154.0]);

        // Both operands are decoded through their physical layouts rather than through dense logical payload copies.
        let lhs_type = array_type(DataType::U16, &[2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![-6, 2])));
        let rhs_type = array_type(DataType::U16, &[3, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![4, -2])));
        let lhs = Array::from_elements(lhs_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let rhs = Array::from_elements(rhs_type, &[7u16, 8, 9, 10, 11, 12]).unwrap();
        assert_eq!(lhs.dot(&rhs, &dimensions).elements::<u16>(), Ok(vec![58, 64, 139, 154]));

        // Batched generalized contraction places batch axes before both operands' non-contracting axes.
        let lhs = Array::from_elements(array_type(DataType::I32, &[2, 2, 2]), &[1i32, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let rhs = Array::from_elements(array_type(DataType::I32, &[2, 2, 1]), &[2i32, 3, 4, 5]).unwrap();
        let batched = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
        let product = lhs.dot(&rhs, &batched);
        assert_eq!(product.r#type().into_owned(), array_type(DataType::I32, &[2, 2, 1]));
        assert_eq!(product.elements::<i32>(), Ok(vec![8, 18, 50, 68]));

        // Narrow integer products and sums wrap at the declared element width, and complex accumulation retains both
        // components.
        let lhs = Array::from_elements(array_type(DataType::I4, &[1, 2]), &[i4::new(7).unwrap(), i4::new(7).unwrap()])
            .unwrap();
        let rhs = Array::from_elements(array_type(DataType::I4, &[2, 1]), &[i4::new(2).unwrap(), i4::new(2).unwrap()])
            .unwrap();
        assert_eq!(lhs.dot(&rhs, &dimensions).elements::<i4>(), Ok(vec![i4::new(-4).unwrap()]));
        let lhs = Array::matrix(1, 2, vec![ComplexNumber::new(1.0f32, 2.0), ComplexNumber::new(3.0, -1.0)]);
        let rhs = Array::matrix(2, 1, vec![ComplexNumber::new(2.0f32, -1.0), ComplexNumber::new(0.5, 4.0)]);
        assert_eq!(
            lhs.dot(&rhs, &dimensions).elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(9.5, 14.5)]),
        );

        // Preferred accumulation first promotes both inputs and then runs the same typed contraction at the wider
        // element data type.
        let lhs = Array::matrix(1, 2, vec![f16::from_f32(1.5), f16::from_f32(2.0)]);
        let rhs = Array::matrix(2, 1, vec![f16::from_f32(2.0), f16::from_f32(3.0)]);
        let product = lhs.dot_with_accumulation_type(&rhs, &dimensions, DataType::F32);
        assert_eq!(product.r#type().data_type(), DataType::F32);
        assert_eq!(product.elements::<f32>(), Ok(vec![9.0]));

        // An empty contracting dimension materializes one additive identity for every result coordinate.
        let lhs = Array::from_elements::<f32>(array_type(DataType::F32, &[2, 0]), &[]).unwrap();
        let rhs = Array::from_elements::<f32>(array_type(DataType::F32, &[0, 3]), &[]).unwrap();
        assert_eq!(lhs.dot(&rhs, &dimensions).elements::<f32>(), Ok(vec![0.0; 6]));
    }

    #[test]
    fn test_array_complex_math() {
        // Elementwise complex math decodes and encodes the complex element types directly.
        let left = Array::vector(vec![ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)]);
        let right = Array::vector(vec![ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)]);
        let left_values = [ComplexNumber::new(1.0f64, 2.0), ComplexNumber::new(0.5f64, -1.0)];
        let right_values = [ComplexNumber::new(0.5f64, -1.0), ComplexNumber::new(2.0f64, 0.5)];
        let expect = |values: [ComplexNumber<f64>; 2]| Array::vector(values.to_vec());
        assert_eq!(
            left.add(&right).unwrap(),
            expect([left_values[0] + right_values[0], left_values[1] + right_values[1]]),
        );
        assert_eq!(
            left.sub(&right).unwrap(),
            expect([left_values[0] - right_values[0], left_values[1] - right_values[1]]),
        );
        assert_eq!(
            left.mul(&right).unwrap(),
            expect([left_values[0] * right_values[0], left_values[1] * right_values[1]]),
        );
        assert_abs_diff_eq!(
            left.div(&right).unwrap(),
            expect([left_values[0] / right_values[0], left_values[1] / right_values[1]]),
            epsilon = 1e-12,
        );
        assert_eq!(left.neg().unwrap(), expect([-left_values[0], -left_values[1]]));
        assert_abs_diff_eq!(left.exp().unwrap(), expect([left_values[0].exp(), left_values[1].exp()]), epsilon = 1e-12);
        assert_abs_diff_eq!(left.log().unwrap(), expect([left_values[0].ln(), left_values[1].ln()]), epsilon = 1e-12);
        assert_abs_diff_eq!(
            left.sqrt().unwrap(),
            expect([left_values[0].sqrt(), left_values[1].sqrt()]),
            epsilon = 1e-12,
        );
        assert_abs_diff_eq!(left.sin().unwrap(), expect([left_values[0].sin(), left_values[1].sin()]), epsilon = 1e-12);
        assert_abs_diff_eq!(left.cos().unwrap(), expect([left_values[0].cos(), left_values[1].cos()]), epsilon = 1e-12);
        // The absolute value is the elementwise magnitude with a real element data type.
        let magnitude = left.abs().unwrap();
        assert_eq!(magnitude.r#type().into_owned(), array_type(DataType::F64, &[2]));
        assert_abs_diff_eq!(
            magnitude,
            Array::vector(vec![left_values[0].norm(), left_values[1].norm()]),
            epsilon = 1e-12,
        );
        // Division normalizes large finite denominators when the direct norm-squared formula would overflow.
        let large = Array::scalar(ComplexNumber::new(1e308f64, 1e308));
        assert_abs_diff_eq!(large.div(&large).unwrap(), Array::scalar(ComplexNumber::new(1.0, 0.0)), epsilon = 1e-12,);
    }

    #[test]
    fn test_array_integer_semantics() {
        // Negation wraps deterministically for unsigned and two's-complement signed elements, matching the scalar
        // reference backend (and StableHLO's integer semantics), rather than panicking or saturating.
        let unsigned = Array::vector(vec![0u8, 1, 255]);
        assert_eq!(unsigned.neg().unwrap().elements::<u8>(), Ok(vec![0, 255, 1]));
        let minimum = Array::vector(vec![i8::MIN, -5]);
        assert_eq!(minimum.neg().unwrap().elements::<i8>(), Ok(vec![i8::MIN, 5]));
        // Sub-byte arithmetic uses the declared bit width for every wrapping operation.
        let narrow = Array::vector(vec![i4::new(7).unwrap(), i4::new(-8).unwrap()]);
        assert_eq!(
            narrow.add(&Array::scalar(i4::new(1).unwrap())).unwrap().elements::<i4>(),
            Ok(vec![i4::MIN, i4::new(-7).unwrap()]),
        );
        assert_eq!(
            narrow.sub(&Array::scalar(i4::new(1).unwrap())).unwrap().elements::<i4>(),
            Ok(vec![i4::new(6).unwrap(), i4::new(7).unwrap()]),
        );
        assert_eq!(narrow.neg().unwrap().elements::<i4>(), Ok(vec![i4::new(-7).unwrap(), i4::MIN]));
        assert_eq!(narrow.abs().unwrap().elements::<i4>(), Ok(vec![i4::new(7).unwrap(), i4::MIN]));
        // Exceptional integer division and remainder inputs return the same structured errors as native-width array
        // arithmetic rather than panicking.
        assert!(matches!(
            Array::vector(vec![1i32]).div(&Array::vector(vec![0i32])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot divide an integer scalar of data type i32 by zero",
        ));
        assert!(matches!(
            Array::vector(vec![i8::MIN]).div(&Array::vector(vec![-1i8])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot divide the minimum integer scalar of data type i8 by -1",
        ));
        assert!(matches!(
            Array::vector(vec![1u8]).rem(&Array::vector(vec![0u8])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message
                    == "cannot compute the remainder of an integer scalar of data type u8 with a zero divisor",
        ));
    }
}
