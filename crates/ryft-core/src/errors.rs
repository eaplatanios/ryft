use std::hash::Hash;
use std::sync::Arc;

use dyn_eq::DynEq;
use dyn_hash::DynHash;
use thiserror::Error;

use crate::broadcasting::BroadcastingError;
use crate::parameters::ParameterError;
use crate::sharding::ShardingError;
use crate::types::{DataTypeError, LayoutError, TypeError};

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

/// Object-safe error that operations and transforms can surface through the `Custom` variants of [`Error`] and other
/// error types in `ryft-core` without those enums enumerating every extension error. This keeps the core error types
/// decoupled from operation and transform extensibility: a new operation or transform carries its own typed error,
/// boxes it behind this trait to travel through the core APIs, and the concrete type is recovered later with
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
