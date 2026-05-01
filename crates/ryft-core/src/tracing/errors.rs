use thiserror::Error;

use super::programs::AtomId;

use crate::parameters::ParameterError;
use crate::tracing_v2::{BatchingError, ControlFlowError, CustomOperationError, DifferentiationError};
use crate::types::TypeError;

/// Represents errors related to tracing in `ryft-core`.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TracingError {
    #[error("tracing values that are used in the same operation must share the same program builder")]
    MismatchedProgramBuilders,

    #[error("invalid number of inputs; expected {expected} but got {got}")]
    InvalidInputCount { expected: usize, got: usize },

    #[error("invalid number of outputs; expected {expected} but got {got}")]
    InvalidOutputCount { expected: usize, got: usize },

    #[error("unbound atom ID: {id}")]
    UnboundAtomId { id: AtomId },

    #[error("encountered malformed program: {0}")]
    MalformedProgram(String),

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
    ControlFlow(#[from] ControlFlowError),

    #[error(transparent)]
    CustomOperation(#[from] CustomOperationError),
}
