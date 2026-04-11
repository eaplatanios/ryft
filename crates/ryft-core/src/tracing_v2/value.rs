//! Foundational leaf-value traits used by `tracing_v2`.
//!
//! The tracing prototype distinguishes between two related concepts:
//!
//! - a concrete leaf value such as `f64` or an `ndarray::Array2<f64>`;
//! - the [`ArrayType`](crate::types::ArrayType) metadata that tracing needs in order to stage programs without
//!   inspecting full values.
//!
//! The traits in this module capture that boundary. They are intentionally small so that future tensor-like leaf
//! types, including PJRT-backed arrays, can adopt the tracing machinery by implementing a compact set of behaviors.

use std::ops::{Add, Mul, Neg};

use crate::{
    parameters::Parameter,
    types::{ArrayType, Typed},
};

/// Minimal floating-point surface used by the scalar tracing primitives.
///
/// Later backends can extend tracing to richer value types by implementing these operations on their leaf type.
pub trait FloatExt: Clone + Add<Output = Self> + Mul<Output = Self> + Neg<Output = Self> {
    /// Computes the elementwise sine.
    fn sin(self) -> Self;

    /// Computes the elementwise cosine.
    fn cos(self) -> Self;
}

/// Creates a zero value with the same logical shape as `self`.
pub trait ZeroLike: Clone {
    /// Returns a zero value with the same shape as `self`.
    fn zero_like(&self) -> Self;
}

/// Creates a one value with the same logical shape as `self`.
pub trait OneLike: Clone {
    /// Returns a one value with the same shape as `self`.
    fn one_like(&self) -> Self;
}

/// Convenience trait for stageable leaves used by `tracing_v2`.
///
/// [`TraceValue`] identifies leaf values that can appear in staged graphs and participate in abstract evaluation. It
/// ties each runtime leaf to the shared [`ArrayType`](crate::types::ArrayType) descriptor used by `tracing_v2`
/// via [`Typed`], while deliberately not implying eager numeric operations such as [`FloatExt`] or
/// differentiation-specific capabilities such as [`ZeroLike`]. Those requirements live on the primitive operations
/// and transforms that actually use them.
pub trait TraceValue: Clone + Parameter + Typed<ArrayType> + 'static {}

impl FloatExt for f32 {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl TraceValue for f32 {}

impl ZeroLike for f32 {
    #[inline]
    fn zero_like(&self) -> Self {
        0.0
    }
}

impl OneLike for f32 {
    #[inline]
    fn one_like(&self) -> Self {
        1.0
    }
}

impl FloatExt for f64 {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl TraceValue for f64 {}

impl ZeroLike for f64 {
    #[inline]
    fn zero_like(&self) -> Self {
        0.0
    }
}

impl OneLike for f64 {
    #[inline]
    fn one_like(&self) -> Self {
        1.0
    }
}

/// Value types that support compile-time identity detection for algebraic simplification.
///
/// This trait enables the graph builder to eliminate trivial operations such as multiplying by one,
/// adding zero, or scaling by an identity factor at graph construction time.
pub trait IdentityValue: TraceValue {
    /// Returns `true` if every element of this value is exactly zero.
    fn is_zero(&self) -> bool;

    /// Returns `true` if every element of this value is exactly one.
    fn is_one(&self) -> bool;
}

impl IdentityValue for f32 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

impl IdentityValue for f64 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

#[cfg(any(feature = "ndarray", test))]
impl IdentityValue for ndarray::Array2<f32> {
    fn is_zero(&self) -> bool {
        self.iter().all(|&x| x == 0.0)
    }

    fn is_one(&self) -> bool {
        self.iter().all(|&x| x == 1.0)
    }
}

#[cfg(any(feature = "ndarray", test))]
impl IdentityValue for ndarray::Array2<f64> {
    fn is_zero(&self) -> bool {
        self.iter().all(|&x| x == 0.0)
    }

    fn is_one(&self) -> bool {
        self.iter().all(|&x| x == 1.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tracing_v2::test_support,
        types::ArrayType,
        types::{DataType, Typed},
    };

    use super::*;

    #[test]
    fn scalar_leaf_traits_report_expected_values() {
        assert_eq!(<f32 as Typed<ArrayType>>::tpe(&1.25f32), ArrayType::scalar(DataType::F32));
        assert_eq!(<f64 as Typed<ArrayType>>::tpe(&2.5f64), ArrayType::scalar(DataType::F64));
        assert_eq!(3.0f32.zero_like(), 0.0);
        assert_eq!(3.0f64.one_like(), 1.0);
        test_support::assert_reference_scalar_sine_jit_rendering();
    }

    #[test]
    fn float_ext_matches_scalar_intrinsics() {
        let angle = 0.75f64;
        assert_eq!(FloatExt::sin(angle), angle.sin());
        assert_eq!(FloatExt::cos(angle), angle.cos());
        test_support::assert_reference_scalar_sine_jit_rendering();
    }
}
