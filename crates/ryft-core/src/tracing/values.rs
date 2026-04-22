//! Leaf-level identity helpers and default scalar leaf impls for the tracing pipeline.
//!
//! The core leaf contracts [`Traceable`](crate::tracing::Traceable) and
//! [`Value`](crate::tracing::Value) live with the staged IR in [`programs`](crate::tracing::programs).
//! This module keeps the remaining value-level helpers
//! that build on top of those contracts:
//!
//! - [`ZeroLike`] and [`OneLike`] for synthesizing identity values from an existing exemplar.
//! - The built-in scalar implementations of [`Traceable`](crate::tracing::Traceable),
//!   [`Value`](crate::tracing::Value), [`ZeroLike`], and [`OneLike`].

use half::{bf16, f16};

use crate::types::ArrayType;

use super::programs::{Traceable, Value};

/// Returns a zero value with the same structure as an existing value.
///
/// [`ZeroLike`] is the local, value-level counterpart to
/// [`Engine::zero`](crate::tracing_v2::Engine::zero). When a transform already has an exemplar in
/// hand, it uses this trait instead of going back through abstract metadata. That is especially
/// important for wrappers like [`Tracer`](crate::tracing_v2::Tracer) and
/// [`JvpTracer`](crate::tracing_v2::JvpTracer), which can stage or derive a zero from their
/// existing state even when abstract synthesis alone would be insufficient.
pub trait ZeroLike {
    /// Returns a zero value with the same shape as `self`.
    fn zero_like(&self) -> Self;
}

/// Returns a one value with the same structure as an existing value.
///
/// This mirrors [`ZeroLike`] for the multiplicative identity. It is used in the same places where
/// transforms need a unit seed from an exemplar, such as reverse-mode pullbacks for scalar-output
/// functions.
pub trait OneLike {
    /// Returns a one value with the same shape as `self`.
    fn one_like(&self) -> Self;
}

macro_rules! impl_scalar_value_traits {
    ($ty:ty, $zero:expr, $one:expr) => {
        impl Value<ArrayType> for $ty {}

        impl Traceable<ArrayType> for $ty {
            #[inline]
            fn is_zero(&self) -> bool {
                *self == self.zero_like()
            }

            #[inline]
            fn is_one(&self) -> bool {
                *self == self.one_like()
            }
        }

        impl ZeroLike for $ty {
            #[inline]
            fn zero_like(&self) -> Self {
                $zero
            }
        }

        impl OneLike for $ty {
            #[inline]
            fn one_like(&self) -> Self {
                $one
            }
        }
    };
}

impl_scalar_value_traits!(bool, false, true);
impl_scalar_value_traits!(i8, 0i8, 1i8);
impl_scalar_value_traits!(i16, 0i16, 1i16);
impl_scalar_value_traits!(i32, 0i32, 1i32);
impl_scalar_value_traits!(i64, 0i64, 1i64);
impl_scalar_value_traits!(u8, 0u8, 1u8);
impl_scalar_value_traits!(u16, 0u16, 1u16);
impl_scalar_value_traits!(u32, 0u32, 1u32);
impl_scalar_value_traits!(u64, 0u64, 1u64);
impl_scalar_value_traits!(bf16, bf16::ZERO, bf16::ONE);
impl_scalar_value_traits!(f16, f16::ZERO, f16::ONE);
impl_scalar_value_traits!(f32, 0.0f32, 1.0f32);
impl_scalar_value_traits!(f64, 0.0f64, 1.0f64);

#[cfg(test)]
mod tests {
    use half::{bf16, f16};

    use crate::{
        tracing::{Traceable, Value},
        tracing_v2::{Cos, Sin, test_support},
        types::ArrayType,
        types::{DataType, Typed},
    };

    use super::{OneLike, ZeroLike};

    fn assert_scalar_value_type<V: Value<ArrayType>>(value: V, expected_type: DataType) {
        assert_eq!(value.r#type().into_owned(), ArrayType::scalar(expected_type));
    }

    fn assert_scalar_identities<V>(value: V, zero: V, one: V)
    where
        V: Value<ArrayType> + ZeroLike + OneLike + std::fmt::Debug + PartialEq,
    {
        assert_eq!(value.zero_like(), zero);
        assert_eq!(value.one_like(), one);
        assert!(Traceable::is_zero(&zero));
        assert!(Traceable::is_one(&one));
    }

    #[test]
    fn test_scalar_leaf_traits_report_expected_values() {
        assert_scalar_value_type(false, DataType::Boolean);
        assert_scalar_value_type(1i8, DataType::I8);
        assert_scalar_value_type(1i16, DataType::I16);
        assert_scalar_value_type(1i32, DataType::I32);
        assert_scalar_value_type(1i64, DataType::I64);
        assert_scalar_value_type(1u8, DataType::U8);
        assert_scalar_value_type(1u16, DataType::U16);
        assert_scalar_value_type(1u32, DataType::U32);
        assert_scalar_value_type(1u64, DataType::U64);
        assert_scalar_value_type(bf16::from_f32(1.25), DataType::BF16);
        assert_scalar_value_type(f16::from_f32(1.25), DataType::F16);
        assert_eq!(<f32 as Typed<ArrayType>>::r#type(&1.25f32).into_owned(), ArrayType::scalar(DataType::F32));
        assert_eq!(<f64 as Typed<ArrayType>>::r#type(&2.5f64).into_owned(), ArrayType::scalar(DataType::F64));
        assert_scalar_identities(false, false, true);
        assert_scalar_identities(5i32, 0i32, 1i32);
        assert_scalar_identities(5u32, 0u32, 1u32);
        assert_scalar_identities(bf16::from_f32(5.0), bf16::from_f32(0.0), bf16::from_f32(1.0));
        assert_scalar_identities(f16::from_f32(5.0), f16::from_f32(0.0), f16::from_f32(1.0));
        assert_scalar_identities(3.0f32, 0.0f32, 1.0f32);
        assert_scalar_identities(7.0f64, 0.0f64, 1.0f64);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(Sin::sin(angle), angle.sin());
        assert_eq!(Cos::cos(angle), angle.cos());
        test_support::assert_reference_scalar_sine_jit_rendering();
    }
}
