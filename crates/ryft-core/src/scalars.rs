use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::Display;

use half::{bf16, f16};
use num_complex::Complex;
use ryft_macros::Parameter;

#[cfg(test)]
use crate::contexts::Context;
use crate::contexts::EagerContext;
use crate::operations::arithmetic::{Abs, Add, Div, Mul, Neg, Sub};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::complex::{Conjugate, Imaginary, Real};
use crate::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::operations::control_flow::{Select, SelectCondition, WhilePredicate};
use crate::operations::differentiation::StopGradient;
use crate::operations::exponential::{Exponential, Logarithm, SquareRoot};
use crate::operations::scalars::ScalarOperation;
use crate::operations::tag::Tag;
use crate::operations::trigonometric::{Atan2, Cos, Sin};
use crate::operations::{BooleanLike, Operation};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Value};
use crate::tracing::TracingContext;
use crate::types::{DataType, TypeError, Typed};

/// [`TracingContext`] over the scalar universe, pairing [`DataType`] types and [`Scalar`] staged constants with the
/// [`ScalarOperation`] family.
pub type ScalarTracingContext = TracingContext<Scalar, ScalarOperation<Scalar>>;

/// Scalar [`Value`] whose [`Type`](crate::Type) is a [`DataType`] and which is meant to be used primarily for testing
/// the Ryft infrastructure and machinery with programs that do not involve multidimensional arrays.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::scalars::Scalar;
/// # use ryft_core::types::{DataType, Typed};
/// let scalar = Scalar::from(1.5f64);
/// assert_eq!(scalar.r#type().into_owned(), DataType::F64);
/// assert_eq!(scalar, 1.5f64);
/// assert_eq!(scalar + Scalar::from(0.5f64), Scalar::from(2.0f64));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Parameter)]
pub enum Scalar {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    BF16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
    C64(Complex<f32>),
    C128(Complex<f64>),
    // TODO(eaplatanios): Support token values and the remaining quantized floating-point types. Once more are
    //  supported, we will also need to update certain implementations later on in this module that return a
    //  `TypeError` when they encounter any of those types.
}

// `PartialOrd` is implemented manually rather than derived because the complex variants are unordered: same-variant
// comparisons delegate to the payload's own partial order (which does not exist for complex payloads), and
// cross-variant comparisons are `None` (the derived implementation would have ordered them by variant declaration
// order, which is meaningless for scalars of different data types).
impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Scalar::Bool(left), Scalar::Bool(right)) => left.partial_cmp(right),
            (Scalar::I8(left), Scalar::I8(right)) => left.partial_cmp(right),
            (Scalar::I16(left), Scalar::I16(right)) => left.partial_cmp(right),
            (Scalar::I32(left), Scalar::I32(right)) => left.partial_cmp(right),
            (Scalar::I64(left), Scalar::I64(right)) => left.partial_cmp(right),
            (Scalar::U8(left), Scalar::U8(right)) => left.partial_cmp(right),
            (Scalar::U16(left), Scalar::U16(right)) => left.partial_cmp(right),
            (Scalar::U32(left), Scalar::U32(right)) => left.partial_cmp(right),
            (Scalar::U64(left), Scalar::U64(right)) => left.partial_cmp(right),
            (Scalar::BF16(left), Scalar::BF16(right)) => left.partial_cmp(right),
            (Scalar::F16(left), Scalar::F16(right)) => left.partial_cmp(right),
            (Scalar::F32(left), Scalar::F32(right)) => left.partial_cmp(right),
            (Scalar::F64(left), Scalar::F64(right)) => left.partial_cmp(right),
            _ => None,
        }
    }
}

impl Display for Scalar {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar::Bool(value) => Display::fmt(value, formatter),
            Scalar::I8(value) => Display::fmt(value, formatter),
            Scalar::I16(value) => Display::fmt(value, formatter),
            Scalar::I32(value) => Display::fmt(value, formatter),
            Scalar::I64(value) => Display::fmt(value, formatter),
            Scalar::U8(value) => Display::fmt(value, formatter),
            Scalar::U16(value) => Display::fmt(value, formatter),
            Scalar::U32(value) => Display::fmt(value, formatter),
            Scalar::U64(value) => Display::fmt(value, formatter),
            Scalar::BF16(value) => Display::fmt(value, formatter),
            Scalar::F16(value) => Display::fmt(value, formatter),
            Scalar::F32(value) => Display::fmt(value, formatter),
            Scalar::F64(value) => Display::fmt(value, formatter),
            // Complex scalars render as `<real>+<imaginary>i` (e.g., `1.5+2i`), folding a negative imaginary part's
            // sign into the separator (e.g., `1.5-2i`) so that the rendering reads as ordinary complex notation.
            Scalar::C64(value) if value.im.is_sign_negative() => write!(formatter, "{}-{}i", value.re, -value.im),
            Scalar::C64(value) => write!(formatter, "{}+{}i", value.re, value.im),
            Scalar::C128(value) if value.im.is_sign_negative() => write!(formatter, "{}-{}i", value.re, -value.im),
            Scalar::C128(value) => write!(formatter, "{}+{}i", value.re, value.im),
        }
    }
}

impl Typed for Scalar {
    type Type = DataType;

    fn r#type(&self) -> Cow<'_, DataType> {
        Cow::Owned(match self {
            Scalar::Bool(_) => DataType::Boolean,
            Scalar::I8(_) => DataType::I8,
            Scalar::I16(_) => DataType::I16,
            Scalar::I32(_) => DataType::I32,
            Scalar::I64(_) => DataType::I64,
            Scalar::U8(_) => DataType::U8,
            Scalar::U16(_) => DataType::U16,
            Scalar::U32(_) => DataType::U32,
            Scalar::U64(_) => DataType::U64,
            Scalar::BF16(_) => DataType::BF16,
            Scalar::F16(_) => DataType::F16,
            Scalar::F32(_) => DataType::F32,
            Scalar::F64(_) => DataType::F64,
            Scalar::C64(_) => DataType::C64,
            Scalar::C128(_) => DataType::C128,
        })
    }
}

impl Value for Scalar {
    type DispatchDomain = EagerContext<Scalar>;
    type ExecutionDomain = EagerContext<Scalar, ScalarOperation<Scalar>>;

    #[inline]
    fn dispatch_domain(&self) -> EagerContext<Scalar> {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> EagerContext<Scalar, ScalarOperation<Scalar>> {
        EagerContext::new()
    }
}

// Conversions from each supported Rust primitive into the corresponding [`Scalar`] variant. These let later stages
// and numeric-literal tests write `Scalar::from(0.0)` without naming the variant explicitly.
macro_rules! impl_from_primitive_for_scalar {
    ($ty:ty, $variant:ident) => {
        impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self {
                Scalar::$variant(value)
            }
        }
    };
}

impl_from_primitive_for_scalar!(bool, Bool);
impl_from_primitive_for_scalar!(i8, I8);
impl_from_primitive_for_scalar!(i16, I16);
impl_from_primitive_for_scalar!(i32, I32);
impl_from_primitive_for_scalar!(i64, I64);
impl_from_primitive_for_scalar!(u8, U8);
impl_from_primitive_for_scalar!(u16, U16);
impl_from_primitive_for_scalar!(u32, U32);
impl_from_primitive_for_scalar!(u64, U64);
impl_from_primitive_for_scalar!(bf16, BF16);
impl_from_primitive_for_scalar!(f16, F16);
impl_from_primitive_for_scalar!(f32, F32);
impl_from_primitive_for_scalar!(f64, F64);
impl_from_primitive_for_scalar!(Complex<f32>, C64);
impl_from_primitive_for_scalar!(Complex<f64>, C128);

// Equality against each supported Rust primitive, comparing only within the matching variant so that a [`Scalar`] of a
// different [`DataType`] never compares equal to a primitive (e.g., a `Scalar::F32` is never equal to an `f64`). These
// let later stages and numeric-literal tests write `scalar == 0.0` directly.
macro_rules! impl_partial_eq_primitive_for_scalar {
    ($ty:ty, $variant:ident) => {
        impl PartialEq<$ty> for Scalar {
            fn eq(&self, other: &$ty) -> bool {
                matches!(self, Scalar::$variant(value) if value == other)
            }
        }

        impl PartialEq<Scalar> for $ty {
            fn eq(&self, other: &Scalar) -> bool {
                matches!(other, Scalar::$variant(value) if value == self)
            }
        }
    };
}

impl_partial_eq_primitive_for_scalar!(bool, Bool);
impl_partial_eq_primitive_for_scalar!(i8, I8);
impl_partial_eq_primitive_for_scalar!(i16, I16);
impl_partial_eq_primitive_for_scalar!(i32, I32);
impl_partial_eq_primitive_for_scalar!(i64, I64);
impl_partial_eq_primitive_for_scalar!(u8, U8);
impl_partial_eq_primitive_for_scalar!(u16, U16);
impl_partial_eq_primitive_for_scalar!(u32, U32);
impl_partial_eq_primitive_for_scalar!(u64, U64);
impl_partial_eq_primitive_for_scalar!(bf16, BF16);
impl_partial_eq_primitive_for_scalar!(f16, F16);
impl_partial_eq_primitive_for_scalar!(f32, F32);
impl_partial_eq_primitive_for_scalar!(f64, F64);
impl_partial_eq_primitive_for_scalar!(Complex<f32>, C64);
impl_partial_eq_primitive_for_scalar!(Complex<f64>, C128);

impl BooleanLike for Scalar {
    #[inline]
    fn as_boolean(&self) -> Self {
        // `Self::boolean` decodes every `Scalar` variant without failing, so the `unwrap` here is safe.
        Scalar::Bool(self.boolean().unwrap())
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        Ok(match self {
            Scalar::Bool(value) => *value,
            Scalar::I8(value) => *value != 0,
            Scalar::I16(value) => *value != 0,
            Scalar::I32(value) => *value != 0,
            Scalar::I64(value) => *value != 0,
            Scalar::U8(value) => *value != 0,
            Scalar::U16(value) => *value != 0,
            Scalar::U32(value) => *value != 0,
            Scalar::U64(value) => *value != 0,
            Scalar::BF16(value) => *value != bf16::ZERO,
            Scalar::F16(value) => *value != f16::ZERO,
            Scalar::F32(value) => *value != 0.0,
            Scalar::F64(value) => *value != 0.0,
            Scalar::C64(value) => value.re != 0.0 || value.im != 0.0,
            Scalar::C128(value) => value.re != 0.0 || value.im != 0.0,
        })
    }
}

// A `Scalar` predicate is always rank-0 and so the scalar `WhilePredicate` defaults (its own truth value decides
// continuation, and a true predicate takes the candidate wholesale) are exactly its semantics.
impl WhilePredicate for Scalar {}

impl<O: Operation<DataType>> Zero<Scalar> for EagerContext<Scalar, O> {
    #[inline]
    fn zero(&self, r#type: &DataType) -> Result<Scalar, ProgramError> {
        Ok(match r#type {
            DataType::Boolean => Scalar::Bool(false),
            DataType::I8 => Scalar::I8(0),
            DataType::I16 => Scalar::I16(0),
            DataType::I32 => Scalar::I32(0),
            DataType::I64 => Scalar::I64(0),
            DataType::U8 => Scalar::U8(0),
            DataType::U16 => Scalar::U16(0),
            DataType::U32 => Scalar::U32(0),
            DataType::U64 => Scalar::U64(0),
            DataType::BF16 => Scalar::BF16(bf16::ZERO),
            DataType::F16 => Scalar::F16(f16::ZERO),
            DataType::F32 => Scalar::F32(0.0),
            DataType::F64 => Scalar::F64(0.0),
            DataType::C64 => Scalar::C64(Complex::new(0.0, 0.0)),
            DataType::C128 => Scalar::C128(Complex::new(0.0, 0.0)),
            other => {
                return Err(
                    TypeError { message: format!("data type {other} is not supported in the scalar domain") }.into()
                );
            }
        })
    }
}

impl ZeroLike for Scalar {
    #[inline]
    fn zero_like(&self) -> Self {
        match self {
            Scalar::Bool(_) => Scalar::Bool(false),
            Scalar::I8(_) => Scalar::I8(0),
            Scalar::I16(_) => Scalar::I16(0),
            Scalar::I32(_) => Scalar::I32(0),
            Scalar::I64(_) => Scalar::I64(0),
            Scalar::U8(_) => Scalar::U8(0),
            Scalar::U16(_) => Scalar::U16(0),
            Scalar::U32(_) => Scalar::U32(0),
            Scalar::U64(_) => Scalar::U64(0),
            Scalar::BF16(_) => Scalar::BF16(bf16::ZERO),
            Scalar::F16(_) => Scalar::F16(f16::ZERO),
            Scalar::F32(_) => Scalar::F32(0.0),
            Scalar::F64(_) => Scalar::F64(0.0),
            Scalar::C64(_) => Scalar::C64(Complex::new(0.0, 0.0)),
            Scalar::C128(_) => Scalar::C128(Complex::new(0.0, 0.0)),
        }
    }
}

impl<O: Operation<DataType>> One<Scalar> for EagerContext<Scalar, O> {
    #[inline]
    fn one(&self, r#type: &DataType) -> Result<Scalar, ProgramError> {
        Ok(match r#type {
            DataType::Boolean => Scalar::Bool(true),
            DataType::I8 => Scalar::I8(1),
            DataType::I16 => Scalar::I16(1),
            DataType::I32 => Scalar::I32(1),
            DataType::I64 => Scalar::I64(1),
            DataType::U8 => Scalar::U8(1),
            DataType::U16 => Scalar::U16(1),
            DataType::U32 => Scalar::U32(1),
            DataType::U64 => Scalar::U64(1),
            DataType::BF16 => Scalar::BF16(bf16::ONE),
            DataType::F16 => Scalar::F16(f16::ONE),
            DataType::F32 => Scalar::F32(1.0),
            DataType::F64 => Scalar::F64(1.0),
            DataType::C64 => Scalar::C64(Complex::new(1.0, 0.0)),
            DataType::C128 => Scalar::C128(Complex::new(1.0, 0.0)),
            other => {
                return Err(
                    TypeError { message: format!("data type {other} is not supported in the scalar domain") }.into()
                );
            }
        })
    }
}

impl OneLike for Scalar {
    #[inline]
    fn one_like(&self) -> Self {
        match self {
            Scalar::Bool(_) => Scalar::Bool(true),
            Scalar::I8(_) => Scalar::I8(1),
            Scalar::I16(_) => Scalar::I16(1),
            Scalar::I32(_) => Scalar::I32(1),
            Scalar::I64(_) => Scalar::I64(1),
            Scalar::U8(_) => Scalar::U8(1),
            Scalar::U16(_) => Scalar::U16(1),
            Scalar::U32(_) => Scalar::U32(1),
            Scalar::U64(_) => Scalar::U64(1),
            Scalar::BF16(_) => Scalar::BF16(bf16::ONE),
            Scalar::F16(_) => Scalar::F16(f16::ONE),
            Scalar::F32(_) => Scalar::F32(1.0),
            Scalar::F64(_) => Scalar::F64(1.0),
            Scalar::C64(_) => Scalar::C64(Complex::new(1.0, 0.0)),
            Scalar::C128(_) => Scalar::C128(Complex::new(1.0, 0.0)),
        }
    }
}

impl Neg for Scalar {
    #[inline]
    fn neg(&self) -> Result<Scalar, ProgramError> {
        Ok(match *self {
            Scalar::I8(value) => Scalar::I8(-value),
            Scalar::I16(value) => Scalar::I16(-value),
            Scalar::I32(value) => Scalar::I32(-value),
            Scalar::I64(value) => Scalar::I64(-value),
            Scalar::BF16(value) => Scalar::BF16(-value),
            Scalar::F16(value) => Scalar::F16(-value),
            Scalar::F32(value) => Scalar::F32(-value),
            Scalar::F64(value) => Scalar::F64(-value),
            Scalar::C64(value) => Scalar::C64(-value),
            Scalar::C128(value) => Scalar::C128(-value),
            other => {
                return Err(
                    TypeError { message: format!("cannot negate a scalar of data type {}", other.r#type()) }.into()
                );
            }
        })
    }
}

impl std::ops::Neg for Scalar {
    type Output = Scalar;

    #[inline]
    fn neg(self) -> Scalar {
        Neg::neg(&self).unwrap_or_else(|error| panic!("{error}"))
    }
}

// TODO(eaplatanios): Support data type promotion / broadcasting.
macro_rules! impl_binary_arithmetic_for_scalar {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl $trait for Scalar {
            #[inline]
            fn $method(&self, rhs: &Scalar) -> Result<Scalar, ProgramError> {
                Ok(match (*self, *rhs) {
                    (Scalar::I8(left), Scalar::I8(right)) => Scalar::I8(left $operator right),
                    (Scalar::I16(left), Scalar::I16(right)) => Scalar::I16(left $operator right),
                    (Scalar::I32(left), Scalar::I32(right)) => Scalar::I32(left $operator right),
                    (Scalar::I64(left), Scalar::I64(right)) => Scalar::I64(left $operator right),
                    (Scalar::U8(left), Scalar::U8(right)) => Scalar::U8(left $operator right),
                    (Scalar::U16(left), Scalar::U16(right)) => Scalar::U16(left $operator right),
                    (Scalar::U32(left), Scalar::U32(right)) => Scalar::U32(left $operator right),
                    (Scalar::U64(left), Scalar::U64(right)) => Scalar::U64(left $operator right),
                    (Scalar::BF16(left), Scalar::BF16(right)) => Scalar::BF16(left $operator right),
                    (Scalar::F16(left), Scalar::F16(right)) => Scalar::F16(left $operator right),
                    (Scalar::F32(left), Scalar::F32(right)) => Scalar::F32(left $operator right),
                    (Scalar::F64(left), Scalar::F64(right)) => Scalar::F64(left $operator right),
                    (Scalar::C64(left), Scalar::C64(right)) => Scalar::C64(left $operator right),
                    (Scalar::C128(left), Scalar::C128(right)) => Scalar::C128(left $operator right),
                    (left, right) => {
                        return Err(TypeError {
                            message: format!(
                                "cannot apply `{}` to scalars of data types {} and {}",
                                stringify!($method),
                                left.r#type(),
                                right.r#type(),
                            ),
                        }
                        .into());
                    }
                })
            }
        }

        impl std::ops::$trait for Scalar {
            type Output = Scalar;

            #[inline]
            fn $method(self, rhs: Scalar) -> Scalar {
                $trait::$method(&self, &rhs).unwrap_or_else(|error| panic!("{error}"))
            }
        }
    };
}

impl_binary_arithmetic_for_scalar!(Add, add, +);
impl_binary_arithmetic_for_scalar!(Sub, sub, -);
impl_binary_arithmetic_for_scalar!(Mul, mul, *);
impl_binary_arithmetic_for_scalar!(Div, div, /);

// TODO(eaplatanios): Review from here onwards.

impl Sin for Scalar {
    /// Computes the elementwise sine of this [`Scalar`]. Only the floating-point and complex variants support sine
    /// (the complex sine being the analytic continuation `sin(z)`); any other variant returns a [`TypeError`].
    fn sin(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().sin())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().sin())),
            Scalar::F32(value) => Scalar::F32(value.sin()),
            Scalar::F64(value) => Scalar::F64(value.sin()),
            Scalar::C64(value) => Scalar::C64(value.sin()),
            Scalar::C128(value) => Scalar::C128(value.sin()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the sine of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Cos for Scalar {
    /// Computes the elementwise cosine of this [`Scalar`]. Only the floating-point and complex variants support cosine
    /// (the complex cosine being the analytic continuation `cos(z)`); any other variant returns a [`TypeError`].
    fn cos(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().cos())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().cos())),
            Scalar::F32(value) => Scalar::F32(value.cos()),
            Scalar::F64(value) => Scalar::F64(value.cos()),
            Scalar::C64(value) => Scalar::C64(value.cos()),
            Scalar::C128(value) => Scalar::C128(value.cos()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the cosine of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

// The `Complex` construction capability is implemented with a path-qualified trait name because this module already
// uses `num_complex::Complex` pervasively as the complex payload type.
impl crate::operations::complex::Complex for Scalar {
    /// Constructs a complex [`Scalar`] from this value as the real part and `imaginary` as the imaginary part. Only
    /// same-precision `f32` and `f64` part pairs are supported; any other combination returns a [`TypeError`].
    fn complex(&self, imaginary: &Self) -> Result<Self, ProgramError> {
        Ok(match (*self, *imaginary) {
            (Scalar::F32(real), Scalar::F32(imaginary)) => Scalar::C64(Complex::new(real, imaginary)),
            (Scalar::F64(real), Scalar::F64(imaginary)) => Scalar::C128(Complex::new(real, imaginary)),
            (real, imaginary) => {
                return Err(TypeError {
                    message: format!(
                        "cannot construct a complex scalar from parts of data types {} and {}",
                        real.r#type(),
                        imaginary.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl Conjugate for Scalar {
    /// Computes the complex conjugate of this [`Scalar`]. Only the complex variants support conjugation; any other
    /// variant returns a [`TypeError`].
    fn conjugate(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::C64(value) => Scalar::C64(value.conj()),
            Scalar::C128(value) => Scalar::C128(value.conj()),
            other => {
                return Err(TypeError {
                    message: format!("cannot conjugate a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Real for Scalar {
    /// Extracts the real part of this complex [`Scalar`]. Only the complex variants support the extraction; any other
    /// variant returns a [`TypeError`].
    fn real(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::C64(value) => Scalar::F32(value.re),
            Scalar::C128(value) => Scalar::F64(value.re),
            other => {
                return Err(TypeError {
                    message: format!("cannot extract the real part of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Imaginary for Scalar {
    /// Extracts the imaginary part of this complex [`Scalar`]. Only the complex variants support the extraction; any
    /// other variant returns a [`TypeError`].
    fn imaginary(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::C64(value) => Scalar::F32(value.im),
            Scalar::C128(value) => Scalar::F64(value.im),
            other => {
                return Err(TypeError {
                    message: format!("cannot extract the imaginary part of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Exponential for Scalar {
    /// Computes the elementwise natural exponential of this [`Scalar`]. Only the floating-point and complex variants
    /// support the exponential (the complex exponential being the analytic continuation `e^z`); any other variant
    /// returns a [`TypeError`].
    fn exponential(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().exp())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().exp())),
            Scalar::F32(value) => Scalar::F32(value.exp()),
            Scalar::F64(value) => Scalar::F64(value.exp()),
            Scalar::C64(value) => Scalar::C64(value.exp()),
            Scalar::C128(value) => Scalar::C128(value.exp()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the exponential of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Logarithm for Scalar {
    /// Computes the elementwise natural logarithm of this [`Scalar`]. Only the floating-point and complex variants
    /// support the logarithm (the complex logarithm being the principal branch `ln(z)`); any other variant returns a
    /// [`TypeError`].
    fn logarithm(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().ln())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().ln())),
            Scalar::F32(value) => Scalar::F32(value.ln()),
            Scalar::F64(value) => Scalar::F64(value.ln()),
            Scalar::C64(value) => Scalar::C64(value.ln()),
            Scalar::C128(value) => Scalar::C128(value.ln()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the logarithm of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl SquareRoot for Scalar {
    /// Computes the elementwise square root of this [`Scalar`]. Only the floating-point and complex variants support
    /// the square root (the complex square root being the principal branch `√z`); any other variant returns a
    /// [`TypeError`].
    fn square_root(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().sqrt())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().sqrt())),
            Scalar::F32(value) => Scalar::F32(value.sqrt()),
            Scalar::F64(value) => Scalar::F64(value.sqrt()),
            Scalar::C64(value) => Scalar::C64(value.sqrt()),
            Scalar::C128(value) => Scalar::C128(value.sqrt()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the square root of a scalar of data type {}", other.r#type()),
                }
                .into());
            }
        })
    }
}

impl Atan2 for Scalar {
    /// Computes the elementwise two-argument arc tangent `atan2(self, x)` for this [`Scalar`]. Only same-variant
    /// floating-point operand pairs are supported; any other combination returns a [`TypeError`].
    fn atan2(&self, x: &Self) -> Result<Self, ProgramError> {
        Ok(match (*self, *x) {
            (Scalar::BF16(y), Scalar::BF16(x)) => Scalar::BF16(bf16::from_f32(y.to_f32().atan2(x.to_f32()))),
            (Scalar::F16(y), Scalar::F16(x)) => Scalar::F16(f16::from_f32(y.to_f32().atan2(x.to_f32()))),
            (Scalar::F32(y), Scalar::F32(x)) => Scalar::F32(y.atan2(x)),
            (Scalar::F64(y), Scalar::F64(x)) => Scalar::F64(y.atan2(x)),
            (y, x) => {
                return Err(TypeError {
                    message: format!(
                        "cannot compute the arc tangent of scalars of data types {} and {}",
                        y.r#type(),
                        x.r#type(),
                    ),
                }
                .into());
            }
        })
    }
}

impl Abs for Scalar {
    /// Computes the elementwise absolute value of this [`Scalar`]: the magnitude `|z|` (with a real result) for the
    /// complex variants, and `|x|` for the signed-integer and floating-point variants. Any other variant returns a
    /// [`TypeError`].
    fn abs(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::I8(value) => Scalar::I8(value.abs()),
            Scalar::I16(value) => Scalar::I16(value.abs()),
            Scalar::I32(value) => Scalar::I32(value.abs()),
            Scalar::I64(value) => Scalar::I64(value.abs()),
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().abs())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().abs())),
            Scalar::F32(value) => Scalar::F32(value.abs()),
            Scalar::F64(value) => Scalar::F64(value.abs()),
            Scalar::C64(value) => Scalar::F32(value.norm()),
            Scalar::C128(value) => Scalar::F64(value.norm()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the absolute value of a scalar of data type {}", other.r#type(),),
                }
                .into());
            }
        })
    }
}

impl StopGradient for Scalar {
    /// Returns this [`Scalar`] unchanged while marking it as a constant for differentiation purposes.
    #[inline]
    fn stop_gradient(&self) -> Self {
        *self
    }
}

impl Tag for Scalar {
    /// Returns this [`Scalar`] unchanged. Tagging is the identity on concrete values; the tag only matters when staging
    /// through a [`Tracer`](crate::tracing::Tracer).
    #[inline]
    fn tag(self, _key: &str) -> Self {
        self
    }
}

impl Compare for Scalar {
    type Output = Scalar;

    /// Compares two equal-[`DataType`] [`Scalar`]s and returns the Boolean result as an honestly Boolean-typed
    /// [`Scalar::Bool`], never a numeric variant. Mismatched variants return a [`TypeError`]. Complex scalars are
    /// unordered, so they support only the equality directions and return a [`TypeError`] for ordered ones.
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self::Output, ProgramError> {
        if let (Scalar::C64(_), Scalar::C64(_)) | (Scalar::C128(_), Scalar::C128(_)) = (self, rhs) {
            return match direction {
                ComparisonDirection::Equal => Ok(Scalar::Bool(self == rhs)),
                ComparisonDirection::NotEqual => Ok(Scalar::Bool(self != rhs)),
                _ => Err(TypeError {
                    message: format!(
                        "cannot apply an ordered comparison to unordered complex scalars of data type {}",
                        self.r#type(),
                    ),
                }
                .into()),
            };
        }
        let ordering = match (self, rhs) {
            (Scalar::Bool(left), Scalar::Bool(right)) => left.partial_cmp(right),
            (Scalar::I8(left), Scalar::I8(right)) => left.partial_cmp(right),
            (Scalar::I16(left), Scalar::I16(right)) => left.partial_cmp(right),
            (Scalar::I32(left), Scalar::I32(right)) => left.partial_cmp(right),
            (Scalar::I64(left), Scalar::I64(right)) => left.partial_cmp(right),
            (Scalar::U8(left), Scalar::U8(right)) => left.partial_cmp(right),
            (Scalar::U16(left), Scalar::U16(right)) => left.partial_cmp(right),
            (Scalar::U32(left), Scalar::U32(right)) => left.partial_cmp(right),
            (Scalar::U64(left), Scalar::U64(right)) => left.partial_cmp(right),
            (Scalar::BF16(left), Scalar::BF16(right)) => left.partial_cmp(right),
            (Scalar::F16(left), Scalar::F16(right)) => left.partial_cmp(right),
            (Scalar::F32(left), Scalar::F32(right)) => left.partial_cmp(right),
            (Scalar::F64(left), Scalar::F64(right)) => left.partial_cmp(right),
            (left, right) => {
                return Err(TypeError {
                    message: format!("cannot compare scalars of data types {} and {}", left.r#type(), right.r#type()),
                }
                .into());
            }
        };
        let result = match direction {
            ComparisonDirection::Equal => ordering == Some(Ordering::Equal),
            ComparisonDirection::NotEqual => ordering != Some(Ordering::Equal),
            ComparisonDirection::LessThan => ordering == Some(Ordering::Less),
            ComparisonDirection::LessThanOrEqual => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
            ComparisonDirection::GreaterThan => ordering == Some(Ordering::Greater),
            ComparisonDirection::GreaterThanOrEqual => matches!(ordering, Some(Ordering::Greater | Ordering::Equal)),
        };
        Ok(Scalar::Bool(result))
    }
}

// TODO(eaplatanios): Introduce a `Cast` trait if we do not have one already and also support it for arrays.
impl Scalar {
    /// Casts this [`Scalar`] to `target`, converting the carried numeric value. Only value-level type *promotion*
    /// (widening) casts are supported: `self`'s [`DataType`] must equal or be promotable to `target`, which is
    /// exactly what the eager value semantics need in order to match an operation's promoting type inference (for
    /// example, promoting a `select` branch to the promotion of the two branch data types). A cast to the same data
    /// type is the identity, and a non-promotable `target` is a [`TypeError`].
    ///
    /// Every such widening promotion is exact through an `f64` intermediate: an integer promotion target only ever
    /// has sources that fit exactly in an `f64` (the only integers that do not, large `I64`/`U64` values, promote to
    /// `F64` rather than to an integer target), and a floating-point target adopts the intended, possibly rounding,
    /// promotion semantics. Complex promotions widen per component: a complex source widens to a wider complex
    /// target, and a real source promotes to a complex target with a zero imaginary part.
    pub fn cast(&self, target: DataType) -> Result<Scalar, ProgramError> {
        let source = self.r#type().into_owned();
        if source == target {
            return Ok(*self);
        }
        if !source.is_promotable_to(target) {
            return Err(
                TypeError { message: format!("cannot promote scalar of data type {source} to {target}") }.into()
            );
        }
        let value = match self {
            Scalar::Bool(value) => f64::from(*value),
            Scalar::I8(value) => *value as f64,
            Scalar::I16(value) => *value as f64,
            Scalar::I32(value) => *value as f64,
            Scalar::I64(value) => *value as f64,
            Scalar::U8(value) => *value as f64,
            Scalar::U16(value) => *value as f64,
            Scalar::U32(value) => *value as f64,
            Scalar::U64(value) => *value as f64,
            Scalar::BF16(value) => value.to_f64(),
            Scalar::F16(value) => value.to_f64(),
            Scalar::F32(value) => *value as f64,
            Scalar::F64(value) => *value,
            Scalar::C64(value) => {
                // The promotion lattice admits only the wider complex type as a widening target for a `C64` source
                // (the same-type case returned above), so this widens per component.
                return Ok(Scalar::C128(Complex::new(value.re as f64, value.im as f64)));
            }
            Scalar::C128(_) => {
                // A `C128` source has no widening target other than itself, which the same-type case above already
                // handled, so `is_promotable_to` has rejected the cast before this point.
                return Err(
                    TypeError { message: format!("cannot promote scalar of data type {source} to {target}") }.into()
                );
            }
        };
        Ok(match target {
            DataType::I8 => Scalar::I8(value as i8),
            DataType::I16 => Scalar::I16(value as i16),
            DataType::I32 => Scalar::I32(value as i32),
            DataType::I64 => Scalar::I64(value as i64),
            DataType::U8 => Scalar::U8(value as u8),
            DataType::U16 => Scalar::U16(value as u16),
            DataType::U32 => Scalar::U32(value as u32),
            DataType::U64 => Scalar::U64(value as u64),
            DataType::BF16 => Scalar::BF16(bf16::from_f64(value)),
            DataType::F16 => Scalar::F16(f16::from_f64(value)),
            DataType::F32 => Scalar::F32(value as f32),
            DataType::F64 => Scalar::F64(value),
            DataType::C64 => Scalar::C64(Complex::new(value as f32, 0.0)),
            DataType::C128 => Scalar::C128(Complex::new(value, 0.0)),
            other => {
                return Err(
                    TypeError { message: format!("cannot cast scalar to unsupported data type {other}") }.into()
                );
            }
        })
    }
}

impl Select for Scalar {
    type Condition = bool;

    /// Selects between `on_true` and `on_false` based on a plain `condition`, mirroring the broadcasting
    /// [`SelectOperation`](crate::operations::control_flow::SelectOperation) type-inference contract: the selected
    /// branch is promoted to the promotion of the two branch data types, so `select(condition, f32, f64)` yields an
    /// `f64` like `jnp.where`. The condition is decoded from a [`Scalar::Bool`] through [`BooleanLike`] before
    /// reaching here, so this only needs the resolved `bool`.
    #[inline]
    fn select(condition: &bool, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        let target = DataType::promoted(&[on_true.r#type().into_owned(), on_false.r#type().into_owned()])
            .map_err(|error| TypeError { message: error.to_string() })?;
        let selected = if *condition { on_true } else { on_false };
        selected.cast(target)
    }
}

impl SelectCondition for Scalar {
    type Condition = bool;

    /// Extracts the selection condition carried by this [`Scalar`], decoding its in-band Boolean payload through
    /// [`BooleanLike::boolean`].
    #[inline]
    fn select_condition(&self) -> Result<bool, ProgramError> {
        self.boolean()
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::constants::{OneOperation, ZeroOperation};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;

    use super::*;

    #[test]
    fn test_scalar_complex_display() {
        // Complex scalars render as `<real>+<imaginary>i`, folding a negative imaginary part's sign into the
        // separator.
        assert_eq!(Scalar::from(Complex::new(1.5f32, 2.0f32)).to_string(), "1.5+2i");
        assert_eq!(Scalar::from(Complex::new(1.5f32, -2.0f32)).to_string(), "1.5-2i");
        assert_eq!(Scalar::from(Complex::new(-0.5f64, 0.25f64)).to_string(), "-0.5+0.25i");
        assert_eq!(Scalar::from(Complex::new(0.0f64, -1.0f64)).to_string(), "0-1i");
    }

    #[test]
    fn test_scalar_complex_arithmetic() {
        let left = Scalar::from(Complex::new(1.0f64, 2.0f64));
        let right = Scalar::from(Complex::new(3.0f64, -1.0f64));

        // The complex variants support the full fallible arithmetic surface, computing in the complex field.
        assert_eq!(left + right, Scalar::from(Complex::new(4.0f64, 1.0f64)));
        assert_eq!(left - right, Scalar::from(Complex::new(-2.0f64, 3.0f64)));
        assert_eq!(left * right, Scalar::from(Complex::new(5.0f64, 5.0f64)));
        assert_eq!(-left, Scalar::from(Complex::new(-1.0f64, -2.0f64)));
        assert_eq!(
            (left / right) * right,
            Scalar::from(Complex::new(1.0f64, 2.0f64) / Complex::new(3.0f64, -1.0f64) * Complex::new(3.0f64, -1.0f64)),
        );

        // The complex sine and cosine are the analytic continuations computed by `num_complex`.
        assert_eq!(left.sin(), Ok(Scalar::from(Complex::new(1.0f64, 2.0f64).sin())));
        assert_eq!(left.cos(), Ok(Scalar::from(Complex::new(1.0f64, 2.0f64).cos())));

        // Constants and Boolean-ness: zero/one carry a zero imaginary part, and a complex scalar is truthy exactly
        // when it is not the complex zero.
        assert_eq!(left.zero_like(), Scalar::from(Complex::new(0.0f64, 0.0f64)));
        assert_eq!(left.one_like(), Scalar::from(Complex::new(1.0f64, 0.0f64)));
        assert_eq!(
            EagerContext::<Scalar, ScalarOperation<Scalar>>::new().bind(ZeroOperation::new(DataType::C64), &[]),
            Ok(vec![Scalar::from(Complex::new(0.0f32, 0.0f32))]),
        );
        assert_eq!(left.boolean(), Ok(true));
        assert_eq!(left.zero_like().boolean(), Ok(false));
        assert_eq!(Scalar::from(Complex::new(0.0f64, 0.5f64)).boolean(), Ok(true));

        // Complex scalars are unordered.
        assert_eq!(left.partial_cmp(&right), None);

        // Mixed-variant arithmetic is rejected like every other unequal-variant pair.
        assert!(Add::add(&left, &Scalar::from(1.0f64)).is_err());
    }

    #[test]
    fn test_scalar_complex_compare() {
        let left = Scalar::from(Complex::new(1.0f64, 2.0f64));
        let right = Scalar::from(Complex::new(3.0f64, -1.0f64));

        // Only the equality directions are defined for the unordered complex scalars.
        assert_eq!(left.compare(&left, ComparisonDirection::Equal), Ok(Scalar::Bool(true)));
        assert_eq!(left.compare(&right, ComparisonDirection::Equal), Ok(Scalar::Bool(false)));
        assert_eq!(left.compare(&right, ComparisonDirection::NotEqual), Ok(Scalar::Bool(true)));
        assert_eq!(
            left.compare(&right, ComparisonDirection::LessThan),
            Err(TypeError {
                message: "cannot apply an ordered comparison to unordered complex scalars of data type c128"
                    .to_string(),
            }
            .into()),
        );
    }

    #[test]
    fn test_scalar_complex_cast() {
        // Real sources promote to complex targets with a zero imaginary part, and complex sources widen per
        // component.
        assert_eq!(Scalar::from(1.5f32).cast(DataType::C64), Ok(Scalar::from(Complex::new(1.5f32, 0.0f32))));
        assert_eq!(Scalar::from(3i16).cast(DataType::C128), Ok(Scalar::from(Complex::new(3.0f64, 0.0f64))));
        assert_eq!(
            Scalar::from(Complex::new(1.5f32, -2.0f32)).cast(DataType::C128),
            Ok(Scalar::from(Complex::new(1.5f64, -2.0f64))),
        );

        // Narrowing complex casts and complex-to-real casts are rejected.
        assert_eq!(
            Scalar::from(Complex::new(1.5f64, 0.0f64)).cast(DataType::C64),
            Err(TypeError { message: "cannot promote scalar of data type c128 to c64".to_string() }.into()),
        );
        assert_eq!(
            Scalar::from(Complex::new(1.5f32, 0.0f32)).cast(DataType::F64),
            Err(TypeError { message: "cannot promote scalar of data type c64 to f64".to_string() }.into()),
        );
    }

    #[test]
    fn test_scalar_complex_program_constant_rendering() {
        // A complex constant is staged and rendered like any other constant (a `const` binding typed `c64`; the
        // value-literal syntax itself is covered by the `Display` test above), and interpretation recovers the
        // carried complex value.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::C64);
        let constant = builder.add_constant(Scalar::from(Complex::new(1.5f32, -2.0f32)));
        let output =
            builder.add_instruction(crate::operations::arithmetic::MulOperation, vec![input, constant]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:c64 .
                let %1:c64 = const
                    %2:c64 = mul %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Interpreting the program multiplies in the complex field.
        let outputs = program.interpret(vec![Scalar::from(Complex::new(2.0f32, 1.0f32))]).unwrap();
        assert_eq!(outputs, vec![Scalar::from(Complex::new(2.0f32, 1.0f32) * Complex::new(1.5f32, -2.0f32))]);
    }

    #[test]
    fn test_scalar_domain() {
        // [`EagerContext<Scalar, ScalarOperation<Scalar>>`] is a zero-sized token.
        assert_eq!(size_of::<EagerContext<Scalar, ScalarOperation<Scalar>>>(), 0);

        // It is an eager `Context`. Binding a nullary zero/one operation interprets it directly over concrete
        // [`Scalar`] values, yielding the corresponding scalar identity for the requested [`DataType`].
        assert_eq!(
            EagerContext::<Scalar, ScalarOperation<Scalar>>::new().bind(ZeroOperation::new(DataType::F64), &[]),
            Ok(vec![Scalar::from(0.0)]),
        );
        assert_eq!(
            EagerContext::<Scalar, ScalarOperation<Scalar>>::default().bind(OneOperation::new(DataType::F64), &[]),
            Ok(vec![Scalar::from(1.0)]),
        );
    }

    #[test]
    fn test_scalar_cast_promotes_widening_and_rejects_narrowing() {
        // A cast to the same data type is the identity.
        assert_eq!(Scalar::from(2.5f32).cast(DataType::F32), Ok(Scalar::from(2.5f32)));

        // Widening promotions convert the carried value exactly: float widening, integer-to-float, integer widening,
        // and Boolean-to-numeric.
        assert_eq!(Scalar::from(2.5f32).cast(DataType::F64), Ok(Scalar::from(2.5f64)));
        assert_eq!(Scalar::from(3i32).cast(DataType::F64), Ok(Scalar::from(3.0f64)));
        assert_eq!(Scalar::from(3i16).cast(DataType::I32), Ok(Scalar::from(3i32)));
        assert_eq!(Scalar::from(true).cast(DataType::U16), Ok(Scalar::from(1u16)));
        assert_eq!(Scalar::from(f16::from_f32(1.5)).cast(DataType::F32), Ok(Scalar::from(1.5f32)));

        // Narrowing (non-promotable) casts are rejected rather than silently truncating.
        assert_eq!(
            Scalar::from(2.5f64).cast(DataType::I32),
            Err(TypeError { message: "cannot promote scalar of data type f64 to i32".to_string() }.into()),
        );
    }
}
