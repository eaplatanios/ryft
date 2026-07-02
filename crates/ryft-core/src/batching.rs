use thiserror::Error;

use crate::parameters::ParameterError;
use crate::programs::ProgramError;
use crate::types::ArrayType;

/// Represents batching-related errors.
///
/// [`BatchingError`] and [`ProgramError`] deliberately form a conversion cycle in which each type can carry the
/// other. Batching rules get executed by binding operations (i.e., via [`Context::bind`](crate::Context::bind) and
/// [`StagingContext::stage_operation`](crate::StagingContext::stage_operation)), which can result in [`ProgramError`]s.
/// So, [`BatchingError`]s travel up a trace, type-erased, inside [`ProgramError::Custom`] payloads. In the other
/// direction, the public [`Batch::batch`](crate::Batch::batch) entry point is typed to [`BatchingError`], and a
/// batching trace can also fail for reasons that are not batching-related. Those program errors surface through the
/// [`BatchingError::Program`] variant. The paired [`From`] implementations keep this cycle normalized instead of
/// letting the two types nest: converting to [`ProgramError`] unwraps a [`BatchingError::Program`] back into the
/// program error that it carries and wraps every other variant in [`ProgramError::Custom`], while converting to
/// [`BatchingError`] unwraps a [`ProgramError::Custom`] payload holding a [`BatchingError`] and wraps every other
/// program error in [`BatchingError::Program`]. Round trips therefore never nest one error type inside the other,
/// and `?` re-types errors correctly at both boundaries. Outside of these conversions, a [`BatchingError`] carried by
/// a [`ProgramError`] can be recovered using [`ProgramError::downcast_custom`].
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BatchingError {
    #[error("encountered an empty batch")]
    EmptyBatch,

    #[error("mismatched batch sizes across batched leaves; expected size {expected} but got {actual}")]
    MismatchedBatchSizes { expected: usize, actual: usize },

    #[error("{message}")]
    MisalignedBatchAxes { message: String },

    #[error("batch axis {axis} of array type {type} has dynamic size")]
    DynamicBatchAxis { r#type: ArrayType, axis: usize },

    #[error("batch axis {axis} is out of bounds for array type {type}")]
    BatchAxisOutOfBounds { r#type: ArrayType, axis: usize },

    #[error("{message}")]
    UnsupportedOperation { message: String },

    #[error("mismatched batch output axes; expected {expected:?} but got {actual:?}")]
    MismatchedOutputAxes { expected: Option<usize>, actual: Option<usize> },

    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    Program(ProgramError),
}

impl From<ProgramError> for BatchingError {
    #[inline]
    fn from(error: ProgramError) -> Self {
        if let Some(batching) = error.downcast_custom::<BatchingError>() {
            batching.clone()
        } else {
            BatchingError::Program(error)
        }
    }
}

impl From<BatchingError> for ProgramError {
    #[inline]
    fn from(error: BatchingError) -> Self {
        match error {
            BatchingError::Program(error) => error,
            error => ProgramError::custom(error),
        }
    }
}
