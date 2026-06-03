use thiserror::Error;

use crate::tracing::TracingError;

/// Errors that the backend-agnostic compilation pipeline can surface.
///
/// `BackendError` is the backend-specific [`CompilationDomain::Error`](super::CompilationDomain::Error) type; most
/// call sites instantiate it as the domain's own error so that backend failures bubble up through
/// [`CompilationError::Backend`] without an additional translation step.
#[derive(Debug, Error)]
pub enum CompilationError<BackendError> {
    /// An error surfaced while tracing the user function into a [`Program`](crate::tracing::Program).
    #[error("{0}")]
    Tracing(#[from] TracingError),

    /// An error surfaced by the backend (lowering, compilation, execution, serialization, etc.).
    #[error("{0}")]
    Backend(BackendError),
}
