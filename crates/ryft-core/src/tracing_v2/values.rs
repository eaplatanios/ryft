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

use crate::{
    parameters::Parameter,
    types::{ArrayType, Type, Typed},
};

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

/// Marker trait that identifies concrete, non-tracer leaves.
///
/// [`Value`] is a subtrait of [`Traceable`] implemented by types that carry real data — scalars like `f32`, dense
/// arrays, backend-backed tensors, etc. Tracing wrappers such as [`JitTracer`](crate::tracing_v2::JitTracer) must
/// **not** implement this trait.
///
/// The sole purpose of this marker is to give Rust's coherence checker a way to tell two blanket impls apart.
/// Each composable transform (e.g., `jvp`, `grad`, `vmap`) provides:
///
/// 1. an impl for `V: Value<T>` — eager dispatch that evaluates the transform on concrete data, and
/// 2. an impl for `JitTracer<V>` — symbolic dispatch that stages the transform into the enclosing traced graph.
///
/// Because `JitTracer<V>` implements [`Traceable`] but not [`Value`], the two impls never overlap.
pub trait Value<T: Type>: Traceable<T> {}

/// Base trait for any leaf type that can participate in traced computations.
///
/// [`Traceable`] is implemented by **every** type that can appear as a leaf in a staged graph — both concrete data
/// types (e.g., `f32`, `f64`, backend arrays) and tracing wrappers (e.g.,
/// [`JitTracer`](crate::tracing_v2::JitTracer)). It ties each leaf to a type descriptor `T` via [`Typed`], while
/// deliberately not implying eager numeric operations such as [`FloatExt`] or differentiation-specific capabilities
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
/// Tracing a closure produces a [`Program`](crate::tracing_v2::Program) (and, downstream, a
/// [`CompiledFunction`](crate::tracing_v2::CompiledFunction)) whose lifetime is intentionally decoupled from the
/// trace scope: the whole point of staging is to return the traced artifact, store it, and replay it later on fresh
/// inputs. Constants of type `V` captured during tracing get baked directly into that staged output, so `V` cannot
/// borrow from the tracing closure's local state without dragging its lifetime along. The `'static` bound enforces
/// exactly this invariant structurally. As a side benefit, it also enables the [`TypeId`](std::any::TypeId)-keyed
/// extension registry that [`CustomPrimitive`](crate::tracing_v2::CustomPrimitive) uses to dispatch optional
/// transform rules (`Any`-based downcasting fundamentally requires `'static`).
///
/// # Implementing [`Traceable`] for new leaf types
///
/// The bound is rarely a real constraint — in practice, the rule is simply "own your data":
///
/// - Small [`Copy`] scalars (`f32`, `i32`, `half::bf16`, ...) satisfy `'static` trivially and can implement
///   [`Traceable`] directly, as the built-in `f32`/`f64` impls below illustrate.
/// - Heavier payloads (array buffers, tensors, device allocations) should wrap the underlying handle in
///   [`Arc`](std::sync::Arc) (or [`Rc`](std::rc::Rc) for single-threaded cases). This keeps the leaf cheaply
///   cloneable — which the [`Clone`] supertrait also demands — and severs any tie to a caller's scope. PJRT-backed
///   arrays, shard-map tensors, and similar backend values all take this shape.
///
/// Avoid leaf types that borrow from external state: a reference-carrying wrapper cannot be baked into a staged
/// graph that outlives the trace, and it will not satisfy `'static` either.
///
/// See also [`Value`], the marker subtrait that distinguishes concrete leaves from tracing wrappers.
pub trait Traceable<T: Type>: Clone + Parameter + Typed<T> + 'static {
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

impl Traceable<ArrayType> for f32 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

impl Value<ArrayType> for f32 {}

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

impl Traceable<ArrayType> for f64 {
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn is_one(&self) -> bool {
        *self == 1.0
    }
}

impl Value<ArrayType> for f64 {}

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
        assert_eq!(<f32 as Typed<ArrayType>>::tpe(&1.25f32).into_owned(), ArrayType::scalar(DataType::F32));
        assert_eq!(<f64 as Typed<ArrayType>>::tpe(&2.5f64).into_owned(), ArrayType::scalar(DataType::F64));
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
