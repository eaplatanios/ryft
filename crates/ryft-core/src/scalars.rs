use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::Display;

use half::{bf16, f16};
use ryft_macros::Parameter;

use crate::contexts::{Context, EagerContext};
use crate::domains::Domain;
use crate::operations::arithmetic::{Add, Div, Mul, Neg, Sub};
use crate::operations::compare::{Compare, ComparisonDirection};
use crate::operations::constants::{One, OneLike, Zero, ZeroLike};
use crate::operations::control_flow::{Select, SelectCondition};
use crate::operations::differentiation::StopGradient;
use crate::operations::scalars::ScalarOperation;
use crate::operations::tag::Tag;
use crate::operations::trigonometric::{Cos, Sin};
use crate::operations::{BooleanLike, InterpretableOperation, Operation};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Value};
use crate::types::{DataType, TypeError, Typed};

/// Stateless [`Domain`] that uses [`DataType`] to represent [`Type`](crate::Type)s and [`Scalar`] to represent runtime
/// [`Value`]s. [`ScalarDomain`] is the minimal scalar-only backend used throughout tests and examples in this crate.
/// It demonstrates the intended role of an eager [`Context`] in the smallest possible form. There are no device
/// handles, no mesh states, and no backend registries. There are just the built-in [`ScalarOperation`] variants plus
/// [`DataType`]-driven construction of scalar values. Note that because [`Scalar`] reports the [`DataType`] of
/// whichever variant it holds, this single non-generic domain interprets scalar [`Program`](crate::Program)s over
/// every supported scalar [`DataType`] without monomorphizing over one Rust primitive at a time.
#[derive(Copy, Clone, Debug, Default)]
pub struct ScalarDomain;

impl ScalarDomain {
    /// Creates a new [`ScalarDomain`].
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Domain for ScalarDomain {
    type Type = DataType;
    type Value = Scalar;
    type Constant = Scalar;
    type Operation = ScalarOperation<Scalar>;
}

impl Context for ScalarDomain {
    #[inline]
    fn lift(&self, constant: Scalar) -> Result<Scalar, ProgramError> {
        Ok(constant)
    }

    #[inline]
    fn bind<P: Into<Self::Operation>>(
        &self,
        operation: P,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        // `ScalarDomain` is an eager `Context` whose `bind` interprets the operation directly over `Scalar` values.
        operation.into().interpret(&EagerContext::new(), inputs)
    }
}

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
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Parameter)]
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
    // TODO(eaplatanios): Support token values, complex values, and the remaining quantized floating-point types.
    //  Once more are supported, we will also need to update certain implementations later on in this module that
    //  return a `TypeError` when they encounter any of those types.
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
        }
    }
}

impl Typed<DataType> for Scalar {
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
        })
    }
}

impl Value<DataType> for Scalar {
    type InterpretationContext = EagerContext<DataType, Self>;

    #[inline]
    fn interpretation_context(&self) -> Option<Self::InterpretationContext> {
        Some(EagerContext::new())
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
        })
    }
}

impl<O: Operation<DataType>> Zero<DataType, Scalar> for EagerContext<DataType, Scalar, O> {
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
        }
    }
}

impl<O: Operation<DataType>> One<DataType, Scalar> for EagerContext<DataType, Scalar, O> {
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
    /// Computes the elementwise sine of this [`Scalar`]. Only the floating-point variants support sine; any other
    /// variant returns a [`TypeError`].
    fn sin(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().sin())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().sin())),
            Scalar::F32(value) => Scalar::F32(value.sin()),
            Scalar::F64(value) => Scalar::F64(value.sin()),
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
    /// Computes the elementwise cosine of this [`Scalar`]. Only the floating-point variants support cosine; any other
    /// variant returns a [`TypeError`].
    fn cos(&self) -> Result<Self, ProgramError> {
        Ok(match self {
            Scalar::BF16(value) => Scalar::BF16(bf16::from_f32(value.to_f32().cos())),
            Scalar::F16(value) => Scalar::F16(f16::from_f32(value.to_f32().cos())),
            Scalar::F32(value) => Scalar::F32(value.cos()),
            Scalar::F64(value) => Scalar::F64(value.cos()),
            other => {
                return Err(TypeError {
                    message: format!("cannot compute the cosine of a scalar of data type {}", other.r#type()),
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
    /// [`Scalar::Bool`], never a numeric variant. Mismatched variants return a [`TypeError`].
    fn compare(&self, rhs: &Self, direction: ComparisonDirection) -> Result<Self::Output, ProgramError> {
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

impl Select for Scalar {
    type Condition = bool;

    /// Selects between `on_true` and `on_false` based on a plain `condition`, mirroring the per-primitive scalar
    /// selection semantics. The condition is decoded from a [`Scalar::Bool`] through [`BooleanLike`] before reaching
    /// here, so this only needs the resolved `bool`.
    #[inline]
    fn select(condition: &bool, on_true: &Self, on_false: &Self) -> Result<Self, ProgramError> {
        Ok(if *condition { *on_true } else { *on_false })
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
    use crate::operations::constants::{OneOperation, ZeroOperation};

    use super::*;

    #[test]
    fn test_scalar_domain() {
        // [`ScalarDomain`] is a zero-sized token.
        assert_eq!(size_of::<ScalarDomain>(), 0);

        // It is an eager `Context`. Binding a nullary zero/one operation interprets it directly over concrete
        // [`Scalar`] values, yielding the corresponding scalar identity for the requested [`DataType`].
        assert_eq!(ScalarDomain::new().bind(ZeroOperation::new(DataType::F64), &[]), Ok(vec![Scalar::from(0.0)]));
        assert_eq!(ScalarDomain::default().bind(OneOperation::new(DataType::F64), &[]), Ok(vec![Scalar::from(1.0)]));
    }
}
