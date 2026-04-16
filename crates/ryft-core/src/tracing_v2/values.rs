//! Foundational leaf-value traits used by `tracing_v2`.
//!
//! This module defines two closely related but distinct traits that together govern which types can participate in
//! the tracing system:
//!
//! - [`Traceable`] — the base trait for **any** type that can appear as a leaf in a traced computation, whether it
//!   holds concrete data (like `f64`) or is itself a tracing wrapper (like [`JitTracer<V>`](crate::tracing_v2::JitTracer)).
//!   Every leaf in a staged graph implements this trait.
//!
//! - [`Value`] — a marker subtrait of [`Traceable`] that identifies **concrete, non-tracer** leaves. Types like `f32`,
//!   `f64`, or backend-backed arrays implement [`Value`]; tracing wrappers like
//!   [`JitTracer`](crate::tracing_v2::JitTracer) deliberately do **not**. This distinction exists to resolve Rust
//!   trait-coherence conflicts: transforms such as `jvp`, `grad`, and `vmap` each provide two blanket implementations
//!   of their dispatch trait — one for concrete leaves (`V: Value`) that performs eager evaluation, and one for
//!   `JitTracer<V>` that stages the operation symbolically. Without the [`Value`] marker, both impls would match
//!   `JitTracer<V>` (since it implements [`Traceable`]), and the compiler would reject the overlap.
//!
//! The traits are intentionally small so that future tensor-like leaf types, including PJRT-backed arrays, can adopt
//! the tracing machinery by implementing a compact set of behaviors.

use std::ops::{Add, Mul, Neg};

use half::{bf16, f16};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    types::{ArrayType, Type, Typed},
};

// TODO(eaplatanios): This should not require `Clone`.
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
pub trait Zero<T: Parameter + Type, P: Parameter + Typed<T>>: Parameterized<T> {
    /// Constructs one zero value from the abstract metadata in `self`.
    fn zero(&self) -> Self::To<P>
    where
        Self::Family: ParameterizedFamily<P>;
}

// TODO(eaplatanios): This should not require `Clone`.
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
pub trait One<T: Parameter + Type, P: Parameter + Typed<T>>: Parameterized<T> {
    /// Constructs one one-valued value from the abstract metadata in `self`.
    fn one(&self) -> Self::To<P>
    where
        Self::Family: ParameterizedFamily<P>;
}

/// Implements [`Type`], [`Typed<Self>`](Typed), [`Zero<Self, Self>`](Zero), and [`One<Self, Self>`](One) for a scalar
/// type that serves as its own type metadata. Since a scalar type describes exactly one shape (a single value), every
/// instance of the type is type-compatible and the zero/one synthesis is infallible.
macro_rules! impl_scalar_self_typed_zero_one {
    ($ty:ty, $zero:expr, $one:expr) => {
        impl Type for $ty {
            #[inline]
            fn is_compatible_with(&self, _other: &Self) -> bool {
                true
            }
        }

        impl Typed<$ty> for $ty {
            #[inline]
            fn tpe(&self) -> $ty {
                *self
            }
        }

        impl Zero<$ty, $ty> for $ty {
            #[inline]
            fn zero(&self) -> $ty {
                $zero
            }
        }

        impl One<$ty, $ty> for $ty {
            #[inline]
            fn one(&self) -> $ty {
                $one
            }
        }
    };
}

impl_scalar_self_typed_zero_one!(bool, false, true);
impl_scalar_self_typed_zero_one!(i8, 0, 1);
impl_scalar_self_typed_zero_one!(i16, 0, 1);
impl_scalar_self_typed_zero_one!(i32, 0, 1);
impl_scalar_self_typed_zero_one!(i64, 0, 1);
impl_scalar_self_typed_zero_one!(u8, 0, 1);
impl_scalar_self_typed_zero_one!(u16, 0, 1);
impl_scalar_self_typed_zero_one!(u32, 0, 1);
impl_scalar_self_typed_zero_one!(u64, 0, 1);
impl_scalar_self_typed_zero_one!(bf16, bf16::ZERO, bf16::ONE);
impl_scalar_self_typed_zero_one!(f16, f16::ZERO, f16::ONE);
impl_scalar_self_typed_zero_one!(f32, 0.0, 1.0);
impl_scalar_self_typed_zero_one!(f64, 0.0, 1.0);

/// Marker trait that identifies concrete, non-tracer leaves.
///
/// [`Value`] is a subtrait of [`Traceable`] implemented by types that carry real data — scalars like `f32`, dense
/// arrays, backend-backed tensors, etc. Tracing wrappers such as [`JitTracer`](crate::tracing_v2::JitTracer) must
/// **not** implement this trait.
///
/// The sole purpose of this marker is to give Rust's coherence checker a way to tell two blanket impls apart.
/// Each composable transform (e.g., `jvp`, `grad`, `vmap`) provides:
///
/// 1. an impl for `V: Value` — eager dispatch that evaluates the transform on concrete data, and
/// 2. an impl for `JitTracer<V>` — symbolic dispatch that stages the transform into the enclosing traced graph.
///
/// Because `JitTracer<V>` implements [`Traceable`] but not [`Value`], the two impls never overlap.
pub trait Value: Traceable {}

/// Base trait for any leaf type that can participate in traced computations.
///
/// [`Traceable`] is implemented by **every** type that can appear as a leaf in a staged graph — both concrete data
/// types (e.g., `f32`, `f64`, backend arrays) and tracing wrappers (e.g.,
/// [`JitTracer`](crate::tracing_v2::JitTracer)). It ties each leaf to the shared
/// [`ArrayType`](crate::types::ArrayType) descriptor used by the tracing infrastructure via [`Typed`], while
/// deliberately not implying eager numeric operations such as [`FloatExt`] or differentiation-specific capabilities
/// such as [`ZeroLike`]. Those requirements live on the primitive operations and transforms that actually need them.
///
/// Concrete leaves that support exact algebraic identity detection should override [`Traceable::is_zero`] and
/// [`Traceable::is_one`]. The default implementations return `false`, which keeps purely abstract or traced leaves
/// valid while opting them out of constant-identity simplification.
///
/// See also [`Value`], the marker subtrait that distinguishes concrete leaves from tracing wrappers.
pub trait Traceable: Clone + Parameter + Typed<ArrayType> + 'static {
    /// Returns `true` if every element of this value is exactly zero.
    ///
    /// The graph builder calls this on constant atoms during [`Op::try_simplify`](crate::tracing_v2::Op::try_simplify)
    /// to detect and eliminate algebraic identities at staging time — for example, folding `x + 0` into `x` or `x * 0`
    /// into `0` without emitting the operation into the staged graph.
    ///
    /// The default returns `false`, which is always safe: it simply opts the value out of identity-based
    /// simplification. Concrete leaf types that can inspect their contents (e.g., `f32`, dense arrays) should override
    /// this to return an accurate answer. Tracing wrappers like [`JitTracer`](crate::tracing_v2::JitTracer) cannot
    /// meaningfully inspect their contents at staging time and therefore keep the default.
    #[inline]
    fn is_zero(&self) -> bool {
        false
    }

    /// Returns `true` if every element of this value is exactly one.
    ///
    /// This is the multiplicative-identity counterpart of [`Traceable::is_zero`]. The graph builder uses it during
    /// [`Op::try_simplify`](crate::tracing_v2::Op::try_simplify) to fold operations like `x * 1` into `x` or
    /// `scale(x, 1)` into `x`.
    ///
    /// The same defaulting rationale applies: `false` is always safe, and only concrete leaf types that can inspect
    /// their contents should override this.
    #[inline]
    fn is_one(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// f32/f64 tracing support
// ---------------------------------------------------------------------------

/// Minimal floating-point surface used by the scalar tracing primitives.
///
/// Later backends can extend tracing to richer value types by implementing these operations on their leaf type.
pub trait FloatExt: Clone + Add<Output = Self> + Mul<Output = Self> + Neg<Output = Self> {
    /// Computes the elementwise sine.
    fn sin(self) -> Self;

    /// Computes the elementwise cosine.
    fn cos(self) -> Self;
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

impl Traceable for f32 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

impl Value for f32 {}

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

impl Traceable for f64 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

impl Value for f64 {}

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
