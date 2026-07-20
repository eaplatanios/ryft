use std::hash::Hash;
use std::sync::Arc;

use dyn_eq::DynEq;
use dyn_hash::DynHash;
use thiserror::Error;

use crate::axes::AxisError;
use crate::broadcasting::BroadcastingError;
use crate::parameters::ParameterError;
use crate::programs::types::TypeError;
use crate::sharding::ShardingError;
use crate::types::{DataTypeError, LayoutError};

/// Represents errors that can occur in `ryft-core`.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum Error {
    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    DataType(#[from] DataTypeError),

    #[error(transparent)]
    Layout(#[from] LayoutError),

    #[error(transparent)]
    Type(#[from] TypeError),

    #[error(transparent)]
    Broadcasting(#[from] BroadcastingError),

    #[error(transparent)]
    Sharding(#[from] ShardingError),

    #[error(transparent)]
    Axis(#[from] AxisError),

    #[error("{0}")]
    Custom(Arc<dyn CustomError>),
}

impl Error {
    /// Wraps an operation- or transform-specific error in a [`Custom`](Error::Custom) variant. The concrete error
    /// can later be recovered using [`Error::downcast_custom`].
    #[inline]
    pub fn custom(error: impl CustomError) -> Self {
        Error::Custom(Arc::new(error))
    }

    /// Returns the wrapped custom error downcast to `T` when this is a [`Custom`](Error::Custom) variant holding
    /// a `T`, and [`None`] otherwise.
    #[inline]
    pub fn downcast_custom<T: CustomError>(&self) -> Option<&T> {
        match self {
            // Deref through the `Arc` to the `dyn CustomError`, upcast to `&dyn std::error::Error`, and then use the
            // standard error downcast. Going through the `Arc` directly would downcast the `Arc` instead of the error.
            Error::Custom(custom) => (&**custom as &dyn std::error::Error).downcast_ref::<T>(),
            _ => None,
        }
    }
}

/// Object-safe error that operations and transforms can surface through the `Custom` variants of [`Error`](enum@Error)
/// and other error types in `ryft-core` without those enums enumerating every extension error. This keeps the core
/// error types decoupled from operation and transform extensibility: a new operation or transform carries its own typed
/// error, boxes it behind this trait to travel through the core APIs, and the concrete type is recovered later with
/// [`Error::downcast_custom`] or a similarly named function on another error enum.
///
/// A blanket implementation covers every type that is `'static`, [`Error`](std::error::Error), [`Send`], [`Sync`],
/// [`Eq`], and [`Hash`], so that most errors satisfy [`CustomError`] automatically and do not require a handwritten
/// implementation. The [`DynEq`] and [`DynHash`] bounds enable enclosing error enums to derive [`PartialEq`], [`Eq`],
/// and [`Hash`] over a boxed `dyn CustomError` automatically using the `#[derive]` macro.
pub trait CustomError: 'static + std::error::Error + Send + Sync + DynEq + DynHash {}

impl<T: 'static + std::error::Error + Send + Sync + Eq + Hash> CustomError for T {}

dyn_eq::eq_trait_object!(CustomError);
dyn_hash::hash_trait_object!(CustomError);

/// Adapter that lets closure-taking entry points accept both plain and fallible closures. An entry point that expects
/// a closure producing `T` and reports errors as `E` can instead accept any closure output implementing
/// `MaybeFallible<T, E>`. Returning `T` directly requires no wrapping, while returning `Result<T, SourceError>`
/// for any `SourceError` that converts into `E` enables using `?` inside the closure.
///
/// This dual-mode contract is only sound where the expected closure output `T` has a concrete outer type
/// constructor at the entry point (e.g., the reverse-mode gradient entry points, whose closures produce a
/// [`LinearizationTracer`](crate::LinearizationTracer) or a tracer/auxiliary tuple). A plain output can then never
/// unify with [`Result`], which is what lets type inference select between the two implementations unambiguously.
/// When `T` is itself a fully generic parameter inferred from the closure (e.g., the traced output structures of
/// [`trace`](crate::trace), [`batch`](crate::batch), [`vjp`](crate::vjp), etc.), a [`Result`]-returning closure
/// would make both implementations applicable and inference ambiguous, and so those entry points accept fallible
/// closures only.
pub trait MaybeFallible<T, E> {
    /// Converts this closure output into a [`Result`], wrapping plain outputs in [`Ok`] and converting the error type
    /// of already fallible outputs into `E`.
    fn into_result(self) -> Result<T, E>;
}

impl<T, E> MaybeFallible<T, E> for T {
    #[inline]
    fn into_result(self) -> Result<T, E> {
        Ok(self)
    }
}

impl<T, E, SourceError: Into<E>> MaybeFallible<T, E> for Result<T, SourceError> {
    #[inline]
    fn into_result(self) -> Result<T, E> {
        self.map_err(Into::into)
    }
}
