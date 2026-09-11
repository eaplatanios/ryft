//! Reference [`Array`] kernels for the mathematics operation family contracts.
//!
//! This module owns the element-level arithmetic contracts of the reference backend — the per-element analogues of
//! the value-level capabilities, together with their per-data-type instantiations — and the kernels built on them.
//! Each kernel decodes its operands through their physical addressing and materializes one owned result. Element data
//! type promotion and NumPy-style broadcasting follow the corresponding operations' type-inference rules, so the
//! eager results match what a staged program computes.

use std::sync::Arc;

use half::{bf16, f16};
use num_complex::Complex;

use crate::arrays::addressing::ArrayAddressing;
use crate::arrays::arrays::Array;
use crate::arrays::encoding::{
    ArrayElement, NumericArrayElement, RealArrayElement, f4e2m1fn, f6e2m3fn, f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz,
    f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz, f8e8m0fnu, i1, i2, i4, u1, u2, u4,
};
use crate::arrays::macros::dispatch_on_array_element_type;
use crate::arrays::operations::collectives::decode_nonnegative_integer_metadata;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::data::DataType;
use crate::arrays::types::dimensions::{Dimension, Shape, StaticShape};
use crate::macros::impl_array_elementwise_operation;
use crate::operations::math::erf::erf_f64;
use crate::operations::math::log_sum_exp::{log_sum_exp_abstract, validate_log_sum_exp_data_type};
use crate::operations::math::reduce::reduce_abstract;
use crate::operations::{
    Abs, Add, Atan2, Ceil, ConvertElementType, Cos, Div, Dot, DotDimensionNumbers, DotOperation, Erf, Exp, Floor, Log,
    Log1p, LogAddExp, LogSumExp, Logistic, Mul, MulOperation, Neg, Pow, RAGGED_DOT_OPERATION_NAME, RaggedDot,
    RaggedDotDimensionNumbers, RaggedDotMode, RaggedDotOperation, Reduce, ReductionKind, Rem, Reshape, Round, Rsqrt,
    Sign, Sin, Slice, Sqrt, Sub, Tanh,
};
use crate::programs::{Operation, ProgramError, TypeError, Typed};

// TODO(eaplatanios): Review this.

// These contracts operate on decoded storage elements, keeping integer wrapping and low-precision re-encoding in
// one place per element family. They complement the value-level capabilities with the scalar arithmetic needed by
// elementwise kernels and reductions; basic arithmetic is provided by NumericArrayElement and RealArrayElement,
// while extrema and identities are provided by ArrayElement.

/// Floating-point math operations shared by real floating-point and complex array elements.
trait ElementFloatMath: NumericArrayElement {
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
pub(crate) trait ElementRealFloatMath: RealArrayElement {
    /// Computes the Gauss error function of this element.
    fn erf(self) -> Result<Self, ProgramError>;

    /// Computes `log(1 + self)` without forming `1 + self`.
    fn log1p(self) -> Result<Self, ProgramError>;

    /// Computes `log(exp(self) + exp(other))` without forming either exponential.
    fn log_add_exp(self, other: Self) -> Result<Self, ProgramError>;

    /// Rounds this element toward negative infinity.
    fn floor(self) -> Result<Self, ProgramError>;

    /// Rounds this element toward positive infinity.
    fn ceil(self) -> Result<Self, ProgramError>;

    /// Rounds this element to the nearest integer, resolving ties toward the nearest even integer.
    fn round(self) -> Result<Self, ProgramError>;
}

/// Element-level mean divisor, serving mean reductions, which have no capability analogue of their own because a
/// mean lowers to a sum followed by a division by the reduced element count.
trait ElementDivideByCount: NumericArrayElement {
    /// Divides this element by `count` after converting `count` to the element type.
    fn divide_by_count(self, count: usize) -> Result<Self, ProgramError>;
}

// Implements typed arithmetic for signed primitive integers with deterministic two's-complement wrapping.
macro_rules! impl_array_arithmetic_for_signed_integer {
    ($type:ty) => {
        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = count as Self;
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type `{}` by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                if self == Self::MIN && divisor == -1 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide the minimum integer array element of data type `{}` by -1",
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
        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = count as Self;
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type `{}` by zero",
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
        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let bit_mask = Self::MIN.to_bits() | Self::MAX.to_bits();
                let divisor = Self::from_bits(count as u8 & bit_mask).unwrap().value();
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type `{}` by zero",
                        Self::data_type(),
                    ))
                    .into());
                }
                if self == Self::MIN && divisor == -1 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide the minimum integer array element of data type `{}` by -1",
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
        impl ElementDivideByCount for $type {
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                let divisor = Self::from_bits(count as u8 & Self::MAX.to_bits()).unwrap().value();
                if divisor == 0 {
                    return Err(TypeError::invalid(format!(
                        "cannot divide an integer array element of data type `{}` by zero",
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
    ($type:ty, $from_count:expr) => {
        impl ElementDivideByCount for $type {
            #[inline]
            fn divide_by_count(self, count: usize) -> Result<Self, ProgramError> {
                Ok(self / $from_count(count))
            }
        }
    };
}

impl_array_arithmetic_for_float!(bf16, |count: usize| bf16::from_f64(count as f64));
impl_array_arithmetic_for_float!(f16, |count: usize| f16::from_f64(count as f64));
impl_array_arithmetic_for_float!(f32, |count: usize| count as f32);
impl_array_arithmetic_for_float!(f64, |count: usize| count as f64);

// Implements complex arithmetic; division by a real count acts componentwise to avoid an unnecessary complex norm.
macro_rules! impl_array_arithmetic_for_complex {
    ($component:ty) => {
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
            fn log1p(self) -> Result<Self, ProgramError> {
                ($encode)(<$work>::ln_1p(($decode)(self)))
            }

            fn log_add_exp(self, other: Self) -> Result<Self, ProgramError> {
                // The pinned `select(isnan(a - b), a + b, max(a, b) + log1p(exp(-|a - b|)))` construction. The
                // difference is NaN exactly when it is undefined (same-sign infinities) or when an operand is NaN,
                // which is what routes those cases through the saturating sum.
                let left = ($decode)(self);
                let right = ($decode)(other);
                let delta = left - right;
                ($encode)(if <$work>::is_nan(delta) {
                    left + right
                } else {
                    <$work>::max(left, right) + <$work>::ln_1p(<$work>::exp(-<$work>::abs(delta)))
                })
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
                Ok(-imaginary_unit * NumericArrayElement::div(x + imaginary_unit * self, radius)?.ln())
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
    };
}

impl_array_math_for_complex!(f32);
impl_array_math_for_complex!(f64);

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

    /// Computes one numerically stable `log(sum(exp(x)))` directly over typed elements, following the guarded
    /// construction that [`LogSumExpOperation`](crate::operations::math::LogSumExpOperation) documents: a maximum
    /// reduction whose identity is the element type's own lowest value, that maximum replaced by zero wherever it is
    /// not finite, and then `log(sum(exp(x - safe_maximum))) + safe_maximum`. Every intermediate is held in the
    /// element's own encoding, so the result matches what the equivalent staged program computes.
    fn log_sum_exp_elements<T: ElementFloatMath>(
        &self,
        output_type: ArrayType,
        axes: &[usize],
    ) -> Result<Self, ProgramError> {
        debug_assert_eq!(self.r#type().data_type(), T::data_type());
        debug_assert_eq!(output_type.data_type(), T::data_type());
        let zero = T::zero()?;
        let mut maximums = self.reduce_elements::<T>(output_type.clone(), axes, T::max_identity(), |left, right| {
            Ok(ArrayElement::max(&left, &right))
        })?;
        maximums.map_elements_in_place::<T>(|value| {
            Ok(if value.convert_to::<f64>()?.is_finite() { value } else { zero })
        })?;

        let input_shape = self.r#type().static_shape().unwrap();
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut reduce_mask = vec![false; input_shape.rank()];
        axes.iter().for_each(|axis| reduce_mask[*axis] = true);

        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        for output in 0..output_addressing.element_count() {
            zero.encode(&mut bytes[output_addressing.byte_range_for_flat_index(output)]);
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
            let maximum = T::decode(&maximums.storage_bytes()[output_range.clone()]);
            let shifted = input_value.sub(maximum)?.exp()?;
            let sum = T::decode(&bytes[output_range.clone()]).add(shifted)?;
            sum.encode(&mut bytes[output_range]);
            input_addressing.advance_index(&mut input_index);
        }

        for output in 0..output_addressing.element_count() {
            let range = output_addressing.byte_range_for_flat_index(output);
            let maximum = T::decode(&maximums.storage_bytes()[range.clone()]);
            let value = T::decode(&bytes[range.clone()]).log()?.add(maximum)?;
            value.encode(&mut bytes[range]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }

    /// Executes a typed sum or mean reduction, sharing the same wrapping addition and applying mean division in
    /// place after accumulation.
    fn reduce_sum_or_mean_elements<T: ElementDivideByCount>(
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

    /// Allocates an array whose logical elements are initialized to the additive identity.
    fn zeroed<T: ArrayElement>(output_type: ArrayType) -> Result<Self, ProgramError> {
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];
        let zero = T::zero()?;
        for element in 0..output_addressing.element_count() {
            zero.encode(&mut bytes[output_addressing.byte_range_for_flat_index(element)]);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }

    /// Evaluates grouped generalized dot extent-exactly. Each concrete group's raw cumulative interval is clipped to
    /// the physical ragged extent, the resulting pair of operand slices is contracted by the ordinary generalized-dot
    /// kernel, and the result is written into its output window. This keeps temporary storage proportional to one
    /// group rather than the whole operand times the group count.
    fn ragged_dot_elements<T: NumericArrayElement>(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        let mut output_types = RaggedDotOperation::new(dimensions.clone()).infer_output_types(
            &[self.r#type().into_owned(), rhs.r#type().into_owned(), group_sizes.r#type().into_owned()],
            &[],
        )?;
        let output_type = output_types.remove(0);
        let dot_dimensions = dimensions.dot_dimensions();
        let ragged_axis = dimensions.lhs_ragged_dimensions()[0];
        let mode = dimensions.mode(self.r#type().rank())?;
        if mode == RaggedDotMode::Batch {
            return self.dot_elements::<T>(rhs, dot_dimensions);
        }
        let prefix_axes = dimensions.group_sizes_prefix_dimensions(self.r#type().rank())?;
        let prefix_shape = prefix_axes
            .iter()
            .map(|axis| self.r#type().shape().dimensions()[*axis].value().unwrap())
            .collect::<Vec<_>>();
        let prefix_count = prefix_shape.iter().product::<usize>();
        let group_count = group_sizes.r#type().shape().dimensions().last().unwrap().value().ok_or_else(|| {
            ProgramError::InvalidArgument {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` requires a static group count for eager evaluation"),
            }
        })?;
        let sizes = decode_nonnegative_integer_metadata(group_sizes, RAGGED_DOT_OPERATION_NAME, "group_sizes")?;
        let expected_size_count = if group_sizes.r#type().rank() == 1 {
            group_count
        } else {
            prefix_count.checked_mul(group_count).ok_or_else(|| ProgramError::InvalidArgument {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes element count does not fit in `usize`"),
            })?
        };
        if sizes.len() != expected_size_count {
            return Err(ProgramError::InvalidArgument {
                message: format!("`{RAGGED_DOT_OPERATION_NAME}` group sizes storage does not match its shape"),
            });
        }
        let ragged_extent = self.r#type().shape().dimensions()[ragged_axis].value().unwrap();
        let lhs_shape = self.r#type().static_shape().unwrap();
        let rhs_shape = rhs.r#type().static_shape().unwrap();
        let lhs_strides = vec![1; lhs_shape.rank()];
        let rhs_strides = vec![1; rhs_shape.rank()];
        let output_strides = vec![1; output_type.rank()];
        let lhs_result = crate::operations::dot::lhs_result_axes(dot_dimensions, self.r#type().rank());
        let non_contracting_metadata = (mode == RaggedDotMode::NonContracting).then(|| {
            let rhs_group_axis = dimensions.rhs_group_dimensions()[0];
            let rhs_slice_shape = Shape::new(
                rhs_shape
                    .dimensions()
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, dimension)| {
                        (axis != rhs_group_axis).then(|| {
                            let is_prefix_axis = dot_dimensions
                                .lhs_batching_dimensions()
                                .iter()
                                .zip(dot_dimensions.rhs_batching_dimensions())
                                .any(|(lhs_axis, rhs_axis)| *rhs_axis == axis && prefix_axes.contains(lhs_axis));
                            Dimension::Static(if is_prefix_axis { 1 } else { *dimension })
                        })
                    })
                    .collect(),
            );
            let remap_rhs_axis = |axis: usize| if axis < rhs_group_axis { axis } else { axis - 1 };
            let dense_dimensions = DotDimensionNumbers::new(
                dot_dimensions.lhs_contracting_dimensions().to_vec(),
                dot_dimensions.rhs_contracting_dimensions().iter().map(|axis| remap_rhs_axis(*axis)).collect(),
                dot_dimensions.lhs_batching_dimensions().to_vec(),
                dot_dimensions.rhs_batching_dimensions().iter().map(|axis| remap_rhs_axis(*axis)).collect(),
            );
            let ragged_position = lhs_result.iter().position(|axis| *axis == ragged_axis).unwrap();
            let ragged_output_axis = dot_dimensions.lhs_batching_dimensions().len() + ragged_position;
            (rhs_group_axis, rhs_slice_shape, dense_dimensions, ragged_output_axis)
        });
        let contracting_rhs_ragged_axis = (mode == RaggedDotMode::Contracting).then(|| {
            let contracting_position =
                dot_dimensions.lhs_contracting_dimensions().iter().position(|axis| *axis == ragged_axis).unwrap();
            dot_dimensions.rhs_contracting_dimensions()[contracting_position]
        });
        let mut output = Self::zeroed::<T>(output_type)?;
        for prefix in 0..prefix_count {
            let mut remainder = prefix;
            let mut prefix_coordinates = vec![0; prefix_axes.len()];
            for (coordinate, extent) in prefix_coordinates.iter_mut().zip(prefix_shape.iter()).rev() {
                *coordinate = remainder % extent;
                remainder /= extent;
            }
            let metadata_prefix = if group_sizes.r#type().rank() == 1 { 0 } else { prefix };
            let group_range = metadata_prefix * group_count..(metadata_prefix + 1) * group_count;
            let mut lhs_starts = vec![0; lhs_shape.rank()];
            let mut lhs_limits = lhs_shape.dimensions().to_vec();
            for (&axis, &coordinate) in prefix_axes.iter().zip(prefix_coordinates.iter()) {
                lhs_starts[axis] = coordinate;
                lhs_limits[axis] = coordinate + 1;
            }
            let mut rhs_starts = vec![0; rhs_shape.rank()];
            let mut rhs_limits = rhs_shape.dimensions().to_vec();
            for (&lhs_axis, &rhs_axis) in
                dot_dimensions.lhs_batching_dimensions().iter().zip(dot_dimensions.rhs_batching_dimensions())
            {
                if let Some(prefix_position) = prefix_axes.iter().position(|axis| *axis == lhs_axis) {
                    let coordinate = prefix_coordinates[prefix_position];
                    rhs_starts[rhs_axis] = coordinate;
                    rhs_limits[rhs_axis] = coordinate + 1;
                }
            }
            if mode == RaggedDotMode::Contracting {
                for (&lhs_axis, &rhs_axis) in
                    dot_dimensions.lhs_contracting_dimensions().iter().zip(dot_dimensions.rhs_contracting_dimensions())
                {
                    if let Some(prefix_position) = prefix_axes.iter().position(|axis| *axis == lhs_axis) {
                        let coordinate = prefix_coordinates[prefix_position];
                        rhs_starts[rhs_axis] = coordinate;
                        rhs_limits[rhs_axis] = coordinate + 1;
                    }
                }
            }
            let mut output_starts = vec![0; output.r#type().rank()];
            let mut output_limits = output.r#type().static_shape().unwrap().dimensions().to_vec();
            match mode {
                RaggedDotMode::NonContracting => {
                    for (&axis, &coordinate) in prefix_axes.iter().zip(prefix_coordinates.iter()) {
                        if let Some(position) =
                            dot_dimensions.lhs_batching_dimensions().iter().position(|candidate| *candidate == axis)
                        {
                            output_starts[position] = coordinate;
                        } else {
                            let position = lhs_result.iter().position(|candidate| *candidate == axis).unwrap();
                            let position = dot_dimensions.lhs_batching_dimensions().len() + position;
                            output_starts[position] = coordinate;
                        }
                    }
                }
                RaggedDotMode::Contracting => {
                    for (position, lhs_axis) in dot_dimensions.lhs_batching_dimensions().iter().enumerate() {
                        if let Some(prefix_position) = prefix_axes.iter().position(|axis| axis == lhs_axis) {
                            output_starts[position + 1] = prefix_coordinates[prefix_position];
                            output_limits[position + 1] = prefix_coordinates[prefix_position] + 1;
                        }
                    }
                }
                RaggedDotMode::Batch => unreachable!(),
            }
            let mut raw_ragged_start = 0usize;
            for (group, &group_size) in sizes[group_range].iter().enumerate() {
                if raw_ragged_start >= ragged_extent {
                    break;
                }
                let ragged_start = raw_ragged_start;
                let ragged_limit = raw_ragged_start.saturating_add(group_size).min(ragged_extent);
                raw_ragged_start = ragged_limit;
                if ragged_start == ragged_limit {
                    continue;
                }
                lhs_starts[ragged_axis] = ragged_start;
                lhs_limits[ragged_axis] = ragged_limit;
                let lhs_slice = self.slice(&lhs_starts, &lhs_limits, &lhs_strides)?;
                let dot = match mode {
                    RaggedDotMode::NonContracting => {
                        let (rhs_group_axis, rhs_slice_shape, dense_dimensions, ragged_output_axis) =
                            non_contracting_metadata.as_ref().unwrap();
                        rhs_starts[*rhs_group_axis] = group;
                        rhs_limits[*rhs_group_axis] = group + 1;
                        let rhs_slice = rhs.slice(&rhs_starts, &rhs_limits, &rhs_strides)?;
                        let rhs_slice = rhs_slice.reshape(rhs_slice_shape.clone())?;
                        output_starts[*ragged_output_axis] = ragged_start;
                        lhs_slice.dot_elements::<T>(&rhs_slice, dense_dimensions)?
                    }
                    RaggedDotMode::Contracting => {
                        let rhs_ragged_axis = contracting_rhs_ragged_axis.unwrap();
                        rhs_starts[rhs_ragged_axis] = ragged_start;
                        rhs_limits[rhs_ragged_axis] = ragged_limit;
                        let rhs_slice = rhs.slice(&rhs_starts, &rhs_limits, &rhs_strides)?;
                        output_starts[0] = group;
                        output_limits[0] = group + 1;
                        let dot = lhs_slice.dot_elements::<T>(&rhs_slice, dot_dimensions)?;
                        let mut dimensions = vec![Dimension::Static(1)];
                        dimensions.extend_from_slice(dot.r#type().shape().dimensions());
                        let dot = dot.reshape(Shape::new(dimensions))?;
                        let current = output.slice(&output_starts, &output_limits, &output_strides)?;
                        Add::add(&current, &dot)?
                    }
                    RaggedDotMode::Batch => unreachable!(),
                };
                output = output.replace_block(&dot, &output_starts);
            }
        }
        Ok(output)
    }

    fn dot_elements<T: NumericArrayElement>(
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

            let mut accumulator = if T::data_type() == DataType::F8E8M0FNU { None } else { Some(T::zero()?) };
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
                let product = lhs_value.mul(rhs_value)?;
                accumulator = Some(match accumulator {
                    Some(accumulator) => accumulator.add(product)?,
                    None => product,
                });
            }
            let accumulator = match accumulator {
                Some(accumulator) => accumulator,
                None => T::zero()?,
            };
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
                "cannot compute the absolute value of a scalar of data type `{input_type}`",
            ))
            .into());
        }
        dispatch_on_array_element_type!(@numeric input_type, |Element| {
            self.map_elements::<Element, <Element as NumericArrayElement>::Magnitude>(output_type, |value| {
                <Element as NumericArrayElement>::abs(value)
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
            return Err(TypeError::invalid(format!("cannot negate a scalar of data type `{data_type}`")).into());
        }
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            self.map_elements::<Element, Element>(self.r#type().into_owned(), <Element as ArrayElement>::neg)
        })
    }
}

impl std::ops::Neg for Array {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Neg::neg(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

impl_array_elementwise_operation!(
    @binary
    Add, add,
    operation = "add",
    inputs = @numeric,
    checks = [@same_unreduced_axes, @same_reduced_axes],
    |lhs, rhs| NumericArrayElement::add(lhs, rhs),
);

impl_array_elementwise_operation!(
    @binary
    Sub, sub,
    operation = "sub",
    inputs = @numeric,
    checks = [@same_unreduced_axes, @same_reduced_axes],
    |lhs, rhs| NumericArrayElement::sub(lhs, rhs),
);

impl Mul for Array {
    fn mul(&self, rhs: &Self) -> Result<Self, ProgramError> {
        // Multiplication combines reduction states bilinearly rather than requiring congruent operand metadata.
        // Use the operation's inference before evaluating elements so empty inputs obey the same contract.
        let mut output_types = MulOperation::<ArrayType>::new()
            .infer_output_types(&[self.r#type().into_owned(), rhs.r#type().into_owned()], &[])?;
        let output_type = output_types.remove(0);
        if Self::element_count(&output_type) == 0 {
            let addressing = ArrayAddressing::new(output_type.clone())?;
            return Ok(Self::new_unchecked(output_type, Arc::new(vec![0; addressing.storage_byte_len()])));
        }
        let data_type = output_type.data_type();
        let lhs = self.promoted_to(data_type)?;
        let rhs = rhs.promoted_to(data_type)?;
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            lhs.map_element_pairs::<Element, Element>(&rhs, output_type, NumericArrayElement::mul)
        })
    }
}

impl_array_elementwise_operation!(
    @binary
    Div, div,
    operation = "div",
    inputs = @numeric,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| NumericArrayElement::div(lhs, rhs),
);

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
            Self::scalar(Element::from_real(rhs).unwrap_or_else(|error| panic!("{error}")))
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

impl_array_elementwise_operation!(
    @unary
    Sin, sin,
    operation = "sin",
    inputs = @float,
    checks = [@no_unreduced],
    |input| ElementFloatMath::sin(input),
);

impl_array_elementwise_operation!(
    @unary
    Cos, cos,
    operation = "cos",
    inputs = @float,
    checks = [@no_unreduced],
    |input| ElementFloatMath::cos(input),
);

impl_array_elementwise_operation!(
    @binary
    Atan2, atan2,
    operation = "atan2",
    inputs = @float,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| ElementFloatMath::atan2(lhs, rhs),
);

impl_array_elementwise_operation!(
    @unary
    Exp, exp,
    operation = "exp",
    inputs = @float,
    checks = [@no_unreduced],
    |input| ElementFloatMath::exp(input),
);

impl_array_elementwise_operation!(
    @unary
    Log, log,
    operation = "log",
    inputs = @float,
    checks = [@no_unreduced],
    |input| ElementFloatMath::log(input),
);

impl_array_elementwise_operation!(
    @unary
    Log1p, log1p,
    operation = "log1p",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| ElementRealFloatMath::log1p(input),
);

impl_array_elementwise_operation!(
    @binary
    LogAddExp, log_add_exp,
    operation = "log_add_exp",
    inputs = @float @real,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| ElementRealFloatMath::log_add_exp(lhs, rhs),
);

impl_array_elementwise_operation!(
    @unary
    Sqrt, sqrt,
    operation = "sqrt",
    inputs = @float,
    checks = [@no_unreduced],
    |input| ElementFloatMath::sqrt(input),
);

impl_array_elementwise_operation!(
    @unary
    Rsqrt, rsqrt,
    operation = "rsqrt",
    inputs = @float,
    checks = [@no_unreduced],
    |input| ElementFloatMath::rsqrt(input),
);

impl_array_elementwise_operation!(
    @unary
    Tanh, tanh,
    operation = "tanh",
    inputs = @float,
    checks = [@no_unreduced],
    |input| ElementFloatMath::tanh(input),
);

impl_array_elementwise_operation!(
    @unary
    Logistic, logistic,
    operation = "logistic",
    inputs = @float,
    checks = [@no_unreduced],
    |input| ElementFloatMath::logistic(input),
);

impl_array_elementwise_operation!(
    @unary
    Erf, erf,
    operation = "erf",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| ElementRealFloatMath::erf(input),
);

impl_array_elementwise_operation!(
    @binary
    Pow, pow,
    operation = "pow",
    inputs = @float,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| ElementFloatMath::pow(lhs, rhs),
);

impl Sign for Array {
    fn sign(&self) -> Result<Self, ProgramError> {
        if Self::element_count(self.r#type().as_ref()) == 0 {
            let addressing = ArrayAddressing::new(self.r#type().into_owned())?;
            return Ok(Self::new_unchecked(
                self.r#type().into_owned(),
                Arc::new(vec![0; addressing.storage_byte_len()]),
            ));
        }
        let data_type = self.r#type().data_type();
        if !data_type.is_signed() && !data_type.is_floating_point() && !data_type.is_complex() {
            return Err(TypeError::invalid(format!(
                "cannot compute the sign of a scalar of data type `{}`",
                data_type,
            ))
            .into());
        }
        if data_type.is_signed() {
            dispatch_on_array_element_type!(@signed data_type, |Element| {
                self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                    <Element as NumericArrayElement>::sign(value)
                })
            })
        } else if data_type.is_complex() {
            dispatch_on_array_element_type!(@complex data_type, |Element| {
                self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                    <Element as NumericArrayElement>::sign(value)
                })
            })
        } else {
            dispatch_on_array_element_type!(@float data_type, |Element| {
                self.map_elements::<Element, Element>(self.r#type().into_owned(), |value| {
                    <Element as NumericArrayElement>::sign(value)
                })
            })
        }
    }
}

impl_array_elementwise_operation!(
    @unary
    Floor, floor,
    operation = "floor",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| ElementRealFloatMath::floor(input),
);

impl_array_elementwise_operation!(
    @unary
    Ceil, ceil,
    operation = "ceil",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| ElementRealFloatMath::ceil(input),
);

impl_array_elementwise_operation!(
    @unary
    Round, round,
    operation = "round",
    inputs = @float @real,
    checks = [@no_unreduced],
    |input| ElementRealFloatMath::round(input),
);

impl_array_elementwise_operation!(
    @binary
    Rem, rem,
    operation = "rem",
    inputs = @numeric @real,
    checks = [@no_unreduced, @same_reduced_axes],
    |lhs, rhs| RealArrayElement::rem(lhs, rhs),
);

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

impl RaggedDot for Array {
    fn ragged_dot_general(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        let data_type = self.r#type().data_type();
        dispatch_on_array_element_type!(@numeric data_type, |Element| {
            self.ragged_dot_elements::<Element>(rhs, group_sizes, dimensions)
        })
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
                        <Element as ArrayElement>::max_identity()
                    } else {
                        <Element as ArrayElement>::min_identity()
                    };
                    self.reduce_elements::<Element>(output_type, axes, identity, |left, right| {
                        Ok(if kind == ReductionKind::Max {
                            ArrayElement::max(&left, &right)
                        } else {
                            ArrayElement::min(&left, &right)
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

impl LogSumExp for Array {
    fn log_sum_exp(&self, axes: &[usize]) -> Result<Self, ProgramError> {
        // Reducing along no axes is the identity, but only for the operands this primitive accepts at all, so the
        // element data type is validated before the shortcut is taken.
        if axes.is_empty() {
            validate_log_sum_exp_data_type(self.r#type().data_type())?;
            return Ok(self.clone());
        }
        // Reuse the abstract rule for validation and for the complete result metadata. The concrete kernel below then
        // decodes directly from the input's physical layout into the result's addressed storage.
        let output_type = log_sum_exp_abstract(self.r#type().as_ref(), axes)?;
        let data_type = output_type.data_type();
        dispatch_on_array_element_type!(@float data_type, |Element| {
            self.log_sum_exp_elements::<Element>(output_type, axes)
        })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::f16;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::encoding::{f8e4m3fn, f8e8m0fnu, i2, i4};
    use crate::arrays::sharding::meshes::{LogicalMesh, MeshAxis, MeshAxisType};
    use crate::arrays::sharding::shardings::{Sharding, ShardingDimension};
    use crate::arrays::types::arrays::ArrayType;
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
            ArrayType::new_static(DataType::F32, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 4])));
        let left = Array::from_elements(left_type, &[1.0f32, 2.0]).unwrap();
        let right_type =
            ArrayType::new_static(DataType::F64, [1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![24, -8])));
        let right = Array::from_elements(right_type, &[0.5f64, 1.0, 1.5]).unwrap();
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 3]));
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
    fn test_array_mul_reduction_state() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::replicated()]).unwrap();
        let partial_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(sharding.clone().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let reduced_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(sharding.with_reduced_axes(["x"]).unwrap())
            .unwrap();
        let lhs = Array::from_elements(partial_type.clone(), &[2.0f32, 3.0]).unwrap();
        let rhs = Array::from_elements(reduced_type, &[4.0f32, 5.0]).unwrap();
        let expected = Array::from_elements(partial_type, &[8.0f32, 15.0]).unwrap();

        // A partial sum times an operand reduced over the same mesh axes remains a partial sum in either order.
        assert_eq!(lhs.mul(&rhs), Ok(expected.clone()));
        assert_eq!(rhs.mul(&lhs), Ok(expected));

        // Multiplying two partial sums is invalid, including when no scalar evaluation would occur.
        assert!(matches!(
            lhs.mul(&lhs),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`mul` cannot multiply two operands that are both unreduced",
        ));
        let empty_type = lhs.r#type().into_owned().with_shape(Shape::new(vec![Dimension::Static(0)]));
        let empty = Array::from_elements(empty_type, &[] as &[f32]).unwrap();
        assert!(matches!(
            empty.mul(&empty),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`mul` cannot multiply two operands that are both unreduced",
        ));
    }

    #[test]
    fn test_array_low_precision_float_arithmetic() {
        // Low-precision arithmetic computes through decoded values and re-encodes the nearest representable result.
        let left = Array::from_f64s(ArrayType::new_static(DataType::F8E4M3FN, [2]), vec![1.0, 2.0]);
        let right = Array::from_f64s(ArrayType::new_static(DataType::F8E4M3FN, [2]), vec![0.5, 0.25]);
        let sum = left.add(&right).unwrap();
        assert_eq!(sum.r#type().into_owned(), ArrayType::new_static(DataType::F8E4M3FN, [2]));
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
        assert_eq!(magnitude.r#type().into_owned(), ArrayType::new_static(DataType::F64, [1]));
        assert_abs_diff_eq!(magnitude, Array::vector(vec![5.0]), epsilon = 1e-12);
    }

    #[test]
    fn test_array_transcendental_math_uses_typed_storage() {
        // Unary kernels preserve arbitrary physical layouts while traversing elements in logical order.
        let input_type =
            ArrayType::new_static(DataType::F64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-16])));
        let input = Array::from_elements(input_type.clone(), &[0.0f64, 1.0]).unwrap();
        let exponential = input.exp().unwrap();
        assert_eq!(exponential.r#type().as_ref(), &input_type);
        assert_eq!(exponential.elements::<f64>(), Ok(vec![1.0, 1.0f64.exp()]));

        // Binary kernels perform complete broadcasting after promoting both physical inputs to their common type.
        let left_type =
            ArrayType::new_static(DataType::F32, [2, 1]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 4])));
        let left = Array::from_elements(left_type, &[0.0f32, 1.0]).unwrap();
        let right_type =
            ArrayType::new_static(DataType::F64, [1, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![24, -8])));
        let right = Array::from_elements(right_type, &[1.0f64, 1.0, -1.0]).unwrap();
        let angles = left.atan2(&right).unwrap();
        assert_eq!(angles.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 3]));
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
                if message == "`atan2` does not support input data type `i32`",
        ));
        assert!(matches!(
            Array::scalar(2.0f64).pow(&Array::scalar(3i32)),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`pow` does not support input data type `i32`",
        ));

        // Low-precision formats decode, compute, and re-encode without constructing intermediary scalar values.
        let low_precision = Array::from_elements(
            ArrayType::new_static(DataType::F8E4M3FN, [2]),
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
            ArrayType::new_static(DataType::F64, [4]),
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
            Array::from_elements(ArrayType::new_static(DataType::I2, [3]), &[i2::MIN, i2::new(0).unwrap(), i2::MAX])
                .unwrap();
        assert_eq!(
            narrow.sign().unwrap().elements::<i2>(),
            Ok(vec![i2::new(-1).unwrap(), i2::new(0).unwrap(), i2::new(1).unwrap()]),
        );
        assert!(matches!(
            Array::scalar(1u8).sign(),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot compute the sign of a scalar of data type `u8`",
        ));
    }

    #[test]
    fn test_array_reduce() {
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(matrix.reduce(&[1], ReductionKind::Sum), Array::vector(vec![6.0, 15.0]));
        assert_eq!(matrix.reduce(&[1], ReductionKind::Mean), Array::vector(vec![2.0, 5.0]));
        assert_eq!(
            matrix.reduce(&[0, 1], ReductionKind::Sum),
            Array::from_f64s(ArrayType::new_static(DataType::F64, []), vec![21.0])
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
        let r#type =
            ArrayType::new_static(DataType::U16, [2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 2])));
        let matrix = Array::from_elements(r#type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(matrix.reduce(&[1], ReductionKind::Sum).elements::<u16>(), Ok(vec![6, 15]));
        let r#type =
            ArrayType::new_static(DataType::Boolean, [3]).with_layout(Layout::Strided(StridedLayout::new(vec![-1])));
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
        let empty = Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [2, 0]), &[]).unwrap();
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
        let empty =
            Array::from_elements::<ComplexNumber<f32>>(ArrayType::new_static(DataType::C64, [2, 0]), &[]).unwrap();
        assert_eq!(
            empty.reduce(&[1], ReductionKind::Max).elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(f32::NEG_INFINITY, 0.0), ComplexNumber::new(f32::NEG_INFINITY, 0.0),]),
        );
        let empty = Array::from_elements::<f8e8m0fnu>(ArrayType::new_static(DataType::F8E8M0FNU, [2, 0]), &[]).unwrap();
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
        assert_eq!(product.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2, 2]));
        assert_eq!(product.to_f64s(), vec![58.0, 64.0, 139.0, 154.0]);

        // Both operands are decoded through their physical layouts rather than through dense logical payload copies.
        let lhs_type =
            ArrayType::new_static(DataType::U16, [2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![-6, 2])));
        let rhs_type =
            ArrayType::new_static(DataType::U16, [3, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![4, -2])));
        let lhs = Array::from_elements(lhs_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let rhs = Array::from_elements(rhs_type, &[7u16, 8, 9, 10, 11, 12]).unwrap();
        assert_eq!(lhs.dot(&rhs, &dimensions).elements::<u16>(), Ok(vec![58, 64, 139, 154]));

        // Batched generalized contraction places batch axes before both operands' non-contracting axes.
        let lhs = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 2]), &[1i32, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        let rhs = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 1]), &[2i32, 3, 4, 5]).unwrap();
        let batched = DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]);
        let product = lhs.dot(&rhs, &batched);
        assert_eq!(product.r#type().into_owned(), ArrayType::new_static(DataType::I32, [2, 2, 1]));
        assert_eq!(product.elements::<i32>(), Ok(vec![8, 18, 50, 68]));

        // Narrow integer products and sums wrap at the declared element width, and complex accumulation retains both
        // components.
        let lhs = Array::from_elements(
            ArrayType::new_static(DataType::I4, [1, 2]),
            &[i4::new(7).unwrap(), i4::new(7).unwrap()],
        )
        .unwrap();
        let rhs = Array::from_elements(
            ArrayType::new_static(DataType::I4, [2, 1]),
            &[i4::new(2).unwrap(), i4::new(2).unwrap()],
        )
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
        let lhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [2, 0]), &[]).unwrap();
        let rhs = Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [0, 3]), &[]).unwrap();
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
        assert_eq!(magnitude.r#type().into_owned(), ArrayType::new_static(DataType::F64, [2]));
        assert_abs_diff_eq!(
            magnitude,
            Array::vector(vec![left_values[0].norm(), left_values[1].norm()]),
            epsilon = 1e-12,
        );
        // Ratio-based division can still overflow when both denominator components are near the largest value.
        let large = Array::scalar(ComplexNumber::new(1e308f64, 1e308));
        let quotient = large.div(&large).unwrap().elements::<ComplexNumber<f64>>().unwrap()[0];
        assert!(quotient.re.is_nan());
        assert_eq!(quotient.im.to_bits(), 0.0f64.to_bits());
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
                if message == "cannot divide an integer scalar of data type `i32` by zero",
        ));
        assert!(matches!(
            Array::vector(vec![i8::MIN]).div(&Array::vector(vec![-1i8])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "cannot divide the minimum integer scalar of data type `i8` by -1",
        ));
        assert!(matches!(
            Array::vector(vec![1u8]).rem(&Array::vector(vec![0u8])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message
                    == "cannot compute the remainder of an integer scalar of data type `u8` with a zero divisor",
        ));
    }
}
