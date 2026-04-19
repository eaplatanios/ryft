//! Foundational leaf-value traits used by `tracing_v2`.
//!
//! This module defines two closely related but distinct traits that together govern which types can participate in
//! the tracing system:
//!
//! - [`Traceable`] â€” the base trait for **any** type that can appear as a leaf in a traced computation, whether it
//!   holds concrete data (like `f64`) or is itself a tracing wrapper (like [`Tracer<V>`](crate::tracing_v2::Tracer)).
//!   Every leaf in a staged program implements this trait.
//!
//! - [`Value`] â€” a marker subtrait of [`Traceable`] that identifies **concrete, non-tracer** leaves. Types like `f32`,
//!   `f64`, or backend-backed arrays implement [`Value`]; tracing wrappers like
//!   [`Tracer`](crate::tracing_v2::Tracer) deliberately do **not**. This distinction exists to resolve Rust
//!   trait-coherence conflicts: transforms such as `jvp`, `grad`, and `vmap` each provide two blanket implementations
//!   of their dispatch trait â€” one for concrete leaves (`V: Value`) that performs eager evaluation, and one for
//!   `Tracer<V>` that stages the operation symbolically. Without the [`Value`] marker, both impls would match
//!   `Tracer<V>` (since it implements [`Traceable`]), and the compiler would reject the overlap.
//!
//! The traits are intentionally small so that future tensor-like leaf types, including PJRT-backed arrays, can adopt
//! the tracing machinery by implementing a compact set of behaviors.

use half::{bf16, f16};

use crate::{
    parameters::Parameter,
    types::{ArrayType, Type, Typed},
};

/// Marker trait that identifies concrete, non-tracer leaves.
///
/// [`Value`] is a subtrait of [`Traceable`] implemented by types that carry real data â€” scalars like `f32`, dense
/// arrays, backend-backed tensors, etc. Tracing wrappers such as [`Tracer`](crate::tracing_v2::Tracer) must
/// **not** implement this trait.
///
/// The sole purpose of this marker is to give Rust's coherence checker a way to tell two blanket impls apart.
/// Each composable transform (e.g., `jvp`, `grad`, `vmap`) provides:
///
/// 1. an impl for `V: Value<T>` â€” eager dispatch that evaluates the transform on concrete data, and
/// 2. an impl for `Tracer<V>` â€” symbolic dispatch that stages the transform into the enclosing traced program.
///
/// Because `Tracer<V>` implements [`Traceable`] but not [`Value`], the two impls never overlap.
pub trait Value<T: Type>: Traceable<T> {}

/// Base trait for any leaf type that can participate in traced computations.
///
/// [`Traceable`] is implemented by **every** type that can appear as a leaf in a staged program â€” both concrete data
/// types (e.g., `f32`, `f64`, backend arrays) and tracing wrappers (e.g.,
/// [`Tracer`](crate::tracing_v2::Tracer)). It ties each leaf to a type descriptor `T` via [`Typed`], while
/// deliberately not implying eager numeric operations such as [`Sin`](crate::tracing_v2::Sin) or
/// differentiation-specific capabilities
/// such as [`ZeroLike`]. Those requirements live on the primitive operations and transforms that actually need them.
///
/// The type parameter `T` determines the abstract metadata used to describe leaf shapes and element types. The
/// primary instantiation is [`ArrayType`](crate::types::ArrayType), used throughout the core tracing infrastructure.
///
/// Concrete leaves that support exact algebraic identity detection should override [`Traceable::is_zero`] and
/// [`Traceable::is_one`]. The default implementations return `false`, which keeps purely abstract or traced leaves
/// valid while opting them out of constant-identity simplification.
///
/// # Why the `'static` bound?
///
/// Tracing a closure produces a [`Program`](crate::tracing_v2::Program) whose lifetime is intentionally decoupled
/// from the trace scope: the whole point of staging is to return the traced artifact, store it, and replay it later
/// on fresh inputs. Constants of type `V` captured during tracing get baked directly into that staged output, so `V`
/// cannot borrow from the tracing closure's local state without dragging its lifetime along. The `'static` bound
/// enforces exactly this invariant structurally. As a side benefit, it also enables the
/// [`TypeId`](std::any::TypeId)-keyed extension registry that
/// [`CustomPrimitive`](crate::tracing_v2::CustomPrimitive) uses to dispatch optional transform rules (`Any`-based
/// downcasting fundamentally requires `'static`).
///
/// # Implementing [`Traceable`] for new leaf types
///
/// The bound is rarely a real constraint â€” in practice, the rule is simply "own your data":
///
/// - Small [`Copy`] scalars (`f32`, `i32`, `half::bf16`, ...) satisfy `'static` trivially and can implement
///   [`Traceable`] directly, as the built-in scalar impls below illustrate.
/// - Heavier payloads (array buffers, tensors, device allocations) should wrap the underlying handle in
///   [`Arc`](std::sync::Arc) (or [`Rc`](std::rc::Rc) for single-threaded cases). This keeps the leaf cheaply
///   cloneable â€” which the [`Clone`] supertrait also demands â€” and severs any tie to a caller's scope. PJRT-backed
///   arrays and similar backend values all take this shape.
///
/// Avoid leaf types that borrow from external state: a reference-carrying wrapper cannot be baked into a staged
/// program that outlives the trace, and it will not satisfy `'static` either.
///
/// See also [`Value`], the marker subtrait that distinguishes concrete leaves from tracing wrappers.
pub trait Traceable<T: Type>: Clone + Parameter + Typed<T> + 'static {
    /// Returns `true` if every element of this value is exactly zero.
    ///
    /// The program builder calls this on constant atoms during [`Op::try_simplify`](crate::tracing_v2::Op::try_simplify)
    /// to detect and eliminate algebraic identities at staging time â€” for example, folding `x + 0` into `x` or `x * 0`
    /// into `0` without emitting the operation into the staged program.
    ///
    /// The default returns `false`, which is always safe: it simply opts the value out of identity-based
    /// simplification. Concrete leaf types that can inspect their contents (e.g., `f32`, dense arrays) should override
    /// this to return an accurate answer. Tracing wrappers like [`Tracer`](crate::tracing_v2::Tracer) cannot
    /// meaningfully inspect their contents at staging time and therefore keep the default.
    #[inline]
    fn is_zero(&self) -> bool {
        false
    }

    /// Returns `true` if every element of this value is exactly one.
    ///
    /// This is the multiplicative-identity counterpart of [`Traceable::is_zero`]. The program builder uses it during
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

/// Returns a zero value with the same structure as an existing value.
///
/// This is the universal "zero-like" capability: every type that participates in differentiation or batching can
/// produce a zero from an exemplar, including tracer wrappers that cannot be synthesized from abstract metadata alone.
pub trait ZeroLike {
    /// Returns a zero value with the same shape as `self`.
    fn zero_like(&self) -> Self;
}

/// Returns a one value with the same structure as an existing value.
///
/// This mirrors [`ZeroLike`] for the multiplicative identity.
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
        tracing_v2::{Cos, Sin, test_support},
        types::ArrayType,
        types::{DataType, Typed},
    };

    use super::*;

    fn assert_scalar_value_type<V: Value<ArrayType>>(value: V, expected_type: DataType) {
        assert_eq!(value.tpe().into_owned(), ArrayType::scalar(expected_type));
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
        assert_eq!(<f32 as Typed<ArrayType>>::tpe(&1.25f32).into_owned(), ArrayType::scalar(DataType::F32));
        assert_eq!(<f64 as Typed<ArrayType>>::tpe(&2.5f64).into_owned(), ArrayType::scalar(DataType::F64));
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
