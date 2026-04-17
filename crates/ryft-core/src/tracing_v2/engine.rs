//! Context-carrying value construction for staged graph transforms.
//!
//! The [`Engine`] trait is the abstraction used by differentiation, rematerialization, and
//! vectorization transforms whenever they need to *synthesize* a concrete value from abstract
//! type metadata alone — most commonly a representative zero used as a seed when replaying a
//! staged graph.
//!
//! The trait exists because different leaf value types require different amounts of context to
//! materialize a value from a [`Type`] descriptor:
//!
//! - Scalar-valued leaves over [`ArrayType`] metadata carry no shape beyond the type itself, so a
//!   stateless [`ArrayScalarEngine<V>`] is sufficient.
//! - Sharded and device-backed leaves such as [`ShardMapTensor`](crate) or PJRT-backed arrays
//!   need additional context (a mesh handle, a device, a PJRT client) to construct a valid
//!   concrete value. These leaves use stateful engines that carry the required handles.
//!
//! Engines are intentionally kept small: they expose metadata-only synthesis (zero and one) and
//! also select the staged [`OpSet`](crate::tracing_v2::OpSet) used by user-facing tracing
//! transforms. Per-equation evaluation paths (`InterpretableOp::interpret`, `abstract_eval`, and
//! similar) remain engine-free so that the common fast path is never forced through a dispatch
//! layer.
//!
//! ## Performance
//!
//! Engines are always passed by shared reference (`&E`) to user-facing transforms. Stateless
//! engines are zero-sized types, so engine arguments cost nothing at runtime. Every concrete
//! engine method is marked `#[inline]` so the compiler can fully elide the call at monomorphized
//! call sites.

use std::{fmt::Display, marker::PhantomData};

use half::{bf16, f16};

use crate::{
    tracing_v2::CoreOpSet,
    types::{ArrayType, Type},
};

/// Synthesizes concrete leaf values from abstract type metadata.
///
/// An [`Engine`] carries whatever context is required to construct a value of
/// [`Value`](Engine::Value) from a [`Type`](Engine::Type) descriptor. The sole responsibility
/// of the trait is metadata-driven zero/one construction plus staged-op-set selection for the
/// user-facing tracing transforms. The hot evaluation paths do not use it, so engine dispatch is
/// restricted to the few call sites that genuinely need representative synthesis or an explicit
/// backend token.
///
/// Implementations should be cheap to clone (the common case is a [`Copy`] zero-sized type) and
/// must return values whose [`Typed::tpe`] matches the input type metadata.
pub trait Engine {
    /// Abstract type metadata interpreted by this engine.
    type Type: Type + Display;

    /// Concrete leaf value produced by this engine.
    type Value;

    /// Closed staged-op universe selected by this engine for public tracing transforms.
    ///
    /// Engines that are used only for metadata synthesis may still set this to a tracing op-set
    /// type that is never exercised. Public tracing APIs add the stronger bound
    /// `Self::OpSet: OpSet<Self::Type, Self::Value>` only when they actually stage a program.
    type OpSet;

    /// Returns the additive-identity value corresponding to the provided type metadata.
    fn zero(&self, r#type: &Self::Type) -> Self::Value;

    /// Returns the multiplicative-identity value corresponding to the provided type metadata.
    fn one(&self, r#type: &Self::Type) -> Self::Value;
}

/// Stateless engine that synthesizes scalar-compatible values from [`ArrayType`] metadata.
///
/// [`ArrayScalarEngine<V>`] is a zero-sized type used whenever a test or scalar-only pipeline needs
/// an engine whose [`Type`](Engine::Type) is [`ArrayType`] but whose [`Value`](Engine::Value) is a
/// scalar such as `f32` or `f64`. The engine ignores the supplied [`ArrayType`] metadata and returns
/// the canonical scalar zero or one, which is well-defined because scalar leaves represent exactly
/// one shape.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArrayScalarEngine<V> {
    marker: PhantomData<fn() -> V>,
}

impl<V> ArrayScalarEngine<V> {
    /// Returns a new [`ArrayScalarEngine<V>`]. This is a no-op at runtime since the engine is
    /// zero-sized.
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

macro_rules! impl_engine_for_array_scalar_engine {
    ($ty:ty, $zero:expr, $one:expr) => {
        impl Engine for ArrayScalarEngine<$ty> {
            type Type = ArrayType;
            type Value = $ty;
            type OpSet = CoreOpSet;

            #[inline]
            fn zero(&self, _type: &ArrayType) -> $ty {
                $zero
            }

            #[inline]
            fn one(&self, _type: &ArrayType) -> $ty {
                $one
            }
        }
    };
}

impl_engine_for_array_scalar_engine!(bool, false, true);
impl_engine_for_array_scalar_engine!(i8, 0, 1);
impl_engine_for_array_scalar_engine!(i16, 0, 1);
impl_engine_for_array_scalar_engine!(i32, 0, 1);
impl_engine_for_array_scalar_engine!(i64, 0, 1);
impl_engine_for_array_scalar_engine!(u8, 0, 1);
impl_engine_for_array_scalar_engine!(u16, 0, 1);
impl_engine_for_array_scalar_engine!(u32, 0, 1);
impl_engine_for_array_scalar_engine!(u64, 0, 1);
impl_engine_for_array_scalar_engine!(bf16, bf16::ZERO, bf16::ONE);
impl_engine_for_array_scalar_engine!(f16, f16::ZERO, f16::ONE);
impl_engine_for_array_scalar_engine!(f32, 0.0, 1.0);
impl_engine_for_array_scalar_engine!(f64, 0.0, 1.0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DataType;

    #[test]
    fn test_array_scalar_engine_is_zero_sized() {
        assert_eq!(size_of::<ArrayScalarEngine<f64>>(), 0);
        assert_eq!(size_of::<ArrayScalarEngine<f32>>(), 0);
    }

    #[test]
    fn test_array_scalar_engine_produces_canonical_zero_and_one() {
        let engine = ArrayScalarEngine::<f64>::new();
        let r#type = ArrayType::scalar(DataType::F64);
        assert_eq!(Engine::zero(&engine, &r#type), 0.0);
        assert_eq!(Engine::one(&engine, &r#type), 1.0);
    }
}
