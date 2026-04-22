use thiserror::Error;

use crate::{
    batching::BatchingError,
    parameters::ParameterError,
    tracing_v2::{AtomId, CustomOperationError, DifferentiationError},
    types::TypeError,
};

/// Error type shared by the staging and transform pipeline.
///
/// [`TracingError`](crate::tracing::TracingError) intentionally spans the tracing subsystem:
/// primitive abstract evaluation, staged program construction, higher-order transform synthesis,
/// and program replay. The batching-specific failures live in [`BatchingError`] and are wrapped
/// here when batching participates inside a tracing flow.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TracingError {
    #[error("tracing values that are used in the same operation must share the same tracing engine")]
    MismatchedEngines,

    #[error("tracing values that are used in the same operation must share the same program builder")]
    MismatchedProgramBuilders,

    #[error("invalid number of inputs; expected {expected} but got {got}")]
    InvalidInputCount { expected: usize, got: usize },

    #[error("invalid number of outputs; expected {expected} but got {got}")]
    InvalidOutputCount { expected: usize, got: usize },

    #[error("unbound atom ID: {id}")]
    UnboundAtomId { id: AtomId },

    #[error("encountered malformed program: {0}")]
    MalformedProgram(&'static str),

    #[error("encountered program builder that escaped its tracing scope")]
    EscapedProgramBuilder,

    #[error("encountered poisoned tracer where a live tracer was required")]
    PoisonedTracer,

    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    Type(#[from] TypeError),

    #[error(transparent)]
    Differentiation(#[from] DifferentiationError),

    #[error(transparent)]
    Batching(#[from] BatchingError),

    #[error(transparent)]
    CustomOperation(#[from] CustomOperationError),
}
