//! Context-carrying value construction for staged program transforms.
//!
//! The [`Engine`] trait is the abstraction used by differentiation, rematerialization, and
//! vectorization transforms whenever they need to *synthesize* a concrete value from abstract
//! type metadata alone — most commonly a representative zero used as a seed when replaying a
//! staged program.
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
//! also choose the staged ordinary and linear operation carriers used by user-facing tracing
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

use crate::{
    tracing_v2::{LinearPrimitiveOp, PrimitiveOp},
    types::{ArrayType, Type},
};

/// Synthesizes concrete leaf values from abstract type metadata.
///
/// [`Engine`] is the backend token threaded through the public `tracing_v2` transforms. It has
/// two closely related jobs:
///
/// 1. choose the closed operation carriers that ordinary and linear staged programs should store,
///    and
/// 2. synthesize representative zero and one values from abstract metadata when a transform needs
///    an exemplar but only knows a leaf's type.
///
/// That second responsibility is what lets higher-order transforms stay generic. Linearization,
/// reverse-mode transposition, rematerialization, and traced `vmap` all occasionally need to
/// rebuild a value from shape/type information alone; [`Engine`] is the narrow seam where
/// backend-specific knowledge enters the otherwise backend-agnostic transform code.
///
/// Implementations should be cheap to clone (the common case is a [`Copy`] zero-sized type) and
/// must return values whose type metadata agrees with the input descriptor.
pub trait Engine {
    /// Abstract type metadata interpreted by this engine.
    ///
    /// This is the descriptor carried by staged atoms and used during abstract evaluation. For the
    /// default core pipeline it is usually [`ArrayType`](crate::types::ArrayType), but the trait is
    /// generic so backends can substitute a richer metadata type if needed.
    type Type: Type + Display;

    /// Concrete leaf value produced by this engine.
    ///
    /// The value is what program replay and eager transforms actually operate on. In other words,
    /// [`Engine::Type`] is the abstract description used while staging, while [`Engine::Value`] is
    /// the runtime leaf that inhabits traced programs once they are executed.
    type Value;

    /// Ordinary staged operation type selected by this engine for public tracing transforms.
    ///
    /// Programs produced by [`trace`](crate::tracing_v2::trace) and
    /// [`interpret_and_trace`](crate::tracing_v2::interpret_and_trace) store this carrier.
    type TracingOperation: Clone + 'static;

    /// Linear staged operation type selected by this engine for tangent and cotangent programs.
    ///
    /// Linear programs produced by [`jvp_program`](crate::tracing_v2::jvp_program),
    /// [`vjp`](crate::tracing_v2::vjp), and related transforms store this carrier.
    type LinearOperation: Clone + 'static;

    /// Returns the additive-identity value corresponding to the provided type metadata.
    ///
    /// Transforms use this when they need a representative value for a leaf without having a
    /// concrete witness available, for example when replaying a staged program from retained input
    /// types or constructing zero cotangents in a transposed linear program.
    fn zero(&self, r#type: &Self::Type) -> Self::Value;

    /// Returns the multiplicative-identity value corresponding to the provided type metadata.
    ///
    /// This is used less frequently than [`Engine::zero`] but plays the same architectural role:
    /// it lets traced code materialize identity seeds without depending on an existing exemplar.
    fn one(&self, r#type: &Self::Type) -> Self::Value;
}

/// Stateless engine that synthesizes scalar-compatible values from [`ArrayType`] metadata.
///
/// [`ArrayScalarEngine<V>`] is the "minimal backend" used throughout tests and scalar-only
/// examples. It demonstrates the intended role of an [`Engine`] in the smallest possible form:
/// there is no device handle, no mesh state, and no backend registry, just the choice of the
/// built-in primitive carriers plus metadata-driven construction of scalar zeros and ones.
///
/// The engine ignores most of the supplied [`ArrayType`] metadata because scalar leaves have a
/// single canonical runtime representation. That makes it a good teaching example for the rest of
/// the tracing stack: if a transform works against [`ArrayScalarEngine`], the same code path can be
/// reused by richer engines that need sharding, device, or runtime context.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArrayScalarEngine<V> {
    marker: PhantomData<fn() -> V>,
}

impl<V> ArrayScalarEngine<V> {
    /// Returns a new [`ArrayScalarEngine<V>`].
    ///
    /// This is a no-op at runtime because the engine is zero-sized; the method mainly exists to
    /// give examples and tests an explicit, readable backend token.
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
            type TracingOperation = PrimitiveOp<ArrayType, $ty>;
            type LinearOperation = LinearPrimitiveOp<ArrayType, $ty>;

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
