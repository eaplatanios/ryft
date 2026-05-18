//! Errors surfaced by the backend-agnostic compilation pipeline.

use thiserror::Error;

use crate::tracing::TracingError;

/// Errors that the backend-agnostic compilation pipeline can surface.
///
/// `E` is the backend-specific [`CompilationDomain::Error`](super::CompilationDomain::Error) type;
/// most call sites instantiate it as the engine's own error so that backend failures bubble up
/// through [`CompilationError::Backend`] without an additional translation step.
#[derive(Debug, Error)]
pub enum CompilationError<E> {
    /// The caller supplied jit options that are inconsistent with the function's signature —
    /// for example a captured-state hash that doesn't match the expected shape, or an
    /// engine-level options-validation failure surfaced by the core pipeline.
    #[error("invalid jit options: {reason}")]
    InvalidJitOptions {
        /// Human-readable explanation of which constraint was violated.
        reason: String,
    },

    /// An error surfaced while tracing the user function into a [`Program`](crate::tracing::Program).
    #[error("{0}")]
    Tracing(#[from] TracingError),

    /// An error surfaced by the backend (lowering, compilation, execution, serialization, etc.).
    #[error("{0}")]
    Backend(E),
}

impl<E> CompilationError<E> {
    /// Constructs a [`CompilationError::InvalidJitOptions`] with the supplied human-readable
    /// explanation.
    #[inline]
    pub fn invalid_jit_options(reason: impl Into<String>) -> Self {
        Self::InvalidJitOptions { reason: reason.into() }
    }
}
