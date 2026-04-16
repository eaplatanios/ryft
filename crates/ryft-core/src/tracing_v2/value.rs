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
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    types::{ArrayType, Type, Typed},
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

/// Returns a zero value with the same structure as an existing value.
///
/// This is the universal "zero-like" capability: every type that participates in differentiation or batching can
/// produce a zero from an exemplar, including tracer wrappers that cannot be synthesized from abstract metadata alone.
pub trait ZeroLike: Clone {
    /// Returns a zero value with the same shape as `self`.
    fn zero_like(&self) -> Self;
}

/// Synthesizes a zero value from abstract type metadata alone.
///
/// Unlike [`ZeroLike`], which produces a zero from an existing exemplar value, this trait is implemented on the
/// _type-metadata_ side: `Self` is a type parameterized by some [`Type`] `T` (e.g., [`ArrayType`] for backend array
/// leaves, or `Pair<ArrayType>` for compound structures), and [`Zero::zero`] produces the corresponding concrete value
/// by reparameterizing from `T` to `P`.
///
/// The target value type `P` must be capable of representing any shape described by the metadata. This makes the
/// conversion infallible — the trait should only be implemented when every metadata value maps to a valid zero of type
/// `P`. For value types that cannot represent arbitrary shapes (e.g., `f32` can only represent scalar metadata), use
/// [`ZeroLike`] instead, which produces a zero from an existing exemplar value.
pub trait Zero<T: Type + Parameter, P: Parameter + Typed<T>>: Parameterized<T> {
    /// Constructs one zero value from the abstract metadata in `self`.
    fn zero(&self) -> Self::To<P>
    where
        Self::Family: ParameterizedFamily<P>;
}

/// Returns a one value with the same structure as an existing value.
///
/// This mirrors [`ZeroLike`] for the multiplicative identity.
pub trait OneLike: Clone {
    /// Returns a one value with the same shape as `self`.
    fn one_like(&self) -> Self;
}

/// Synthesizes a one value from abstract type metadata alone.
///
/// This mirrors [`Zero`] for the multiplicative identity. `Self` is a type parameterized by some [`Type`] `T`, and
/// [`One::one`] produces the corresponding concrete value by reparameterizing from `T` to `P`. Like [`Zero`], this
/// trait should only be implemented when every metadata value maps to a valid one of type `P`.
pub trait One<T: Type + Parameter, P: Parameter + Typed<T>>: Parameterized<T> {
    /// Constructs one one-valued value from the abstract metadata in `self`.
    fn one(&self) -> Self::To<P>
    where
        Self::Family: ParameterizedFamily<P>;
}

/// Marker trait for concrete tracing leaves that participate in eager transform dispatch.
///
/// Exact identity detection itself lives on [`TraceValue`]. This marker exists only to partition the blanket impls
/// for concrete leaf regimes from the corresponding traced-leaf impls for
/// [`JitTracer`](crate::tracing_v2::JitTracer).
pub trait ConcreteTraceValue: TraceValue {}

/// Convenience trait for stageable leaves used by `tracing_v2`.
///
/// [`TraceValue`] identifies leaf values that can appear in staged graphs and participate in abstract evaluation. It
/// ties each runtime leaf to the shared [`ArrayType`](crate::types::ArrayType) descriptor used by `tracing_v2`
/// via [`Typed`], while deliberately not implying eager numeric operations such as [`FloatExt`] or
/// differentiation-specific capabilities such as [`ZeroLike`]. Those requirements live on the primitive operations
/// and transforms that actually use them.
///
/// Leaf types that support exact algebraic identity detection should override [`TraceValue::is_zero`] and
/// [`TraceValue::is_one`]. The default implementations return `false`, which keeps purely abstract leaves valid while
/// opting them out of constant-identity simplification.
pub trait TraceValue: Clone + Parameter + Typed<ArrayType> + 'static {
    /// Returns `true` if every element of this value is exactly zero.
    #[inline]
    fn is_zero(&self) -> bool {
        false
    }

    /// Returns `true` if every element of this value is exactly one.
    #[inline]
    fn is_one(&self) -> bool {
        false
    }
}

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

impl TraceValue for f32 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

impl ConcreteTraceValue for f32 {}

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

impl TraceValue for f64 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

impl ConcreteTraceValue for f64 {}

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

#[cfg(test)]
mod tests {
    use crate::{
        tracing_v2::test_support,
        types::ArrayType,
        types::{DataType, Typed},
    };

    use super::*;

    #[test]
    fn test_scalar_leaf_traits_report_expected_values() {
        assert_eq!(<f32 as Typed<ArrayType>>::tpe(&1.25f32), ArrayType::scalar(DataType::F32));
        assert_eq!(<f64 as Typed<ArrayType>>::tpe(&2.5f64), ArrayType::scalar(DataType::F64));
        assert_eq!(ZeroLike::zero_like(&3.0f32), 0.0);
        assert_eq!(ZeroLike::zero_like(&7.0f64), 0.0);
        assert_eq!(OneLike::one_like(&3.0f32), 1.0);
        assert_eq!(OneLike::one_like(&3.0f64), 1.0);
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
