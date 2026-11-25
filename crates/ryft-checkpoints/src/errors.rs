//! Error type of `ryft-checkpoints`.

use std::path::Path;

use ryft_core::{ParameterError, ProgramError};
use thiserror::Error;

use crate::codecs::CodecError;

// TODO(eaplatanios): Review this module.

/// Error returned when reading, validating, or importing a checkpoint fails.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum CheckpointError {
    #[error("failed to access checkpoint file `{path}`: {message}")]
    Io { path: String, message: String },

    #[error("invalid checkpoint header: {message}")]
    InvalidHeader { message: String },

    #[error("invalid checkpoint array `{name}`: {message}")]
    InvalidArray { name: String, message: String },

    #[error("unknown checkpoint array `{name}`")]
    UnknownArray { name: String },

    #[error("checkpoint does not match its manifest: {message}")]
    ManifestMismatch { message: String },

    #[error("invalid checkpoint manifest: {message}")]
    InvalidManifest { message: String },

    #[error("failed to import checkpoint: {message}")]
    Import { message: String },

    #[error("invalid checkpoint file `{path}`: {message}")]
    InvalidFile { path: String, message: String },

    #[error("invalid checkpoint argument: {message}")]
    InvalidArgument { message: String },

    #[error("checkpoint `{path}` already exists")]
    ExistingCheckpoint { path: String },

    #[error("`{path}` is not a finalized checkpoint")]
    UnfinalizedCheckpoint { path: String },

    #[error("missing checkpoint: {message}")]
    MissingCheckpoint { message: String },

    #[error("background checkpoint save panicked: {message}")]
    Panic { message: String },

    #[error("checkpoint operation timed out: {message}")]
    Timeout { message: String },

    #[error("checkpoint barrier `{key}` failed: {message}")]
    Barrier { key: String, message: String },

    #[error("failed to convert checkpoint leaf `{path}`: {message}")]
    Leaf { path: String, message: String },

    #[error(transparent)]
    Codec(#[from] CodecError),

    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    Program(#[from] ProgramError),
}

impl CheckpointError {
    /// Creates a [`CheckpointError::Io`] for a failed access of `path`.
    pub(crate) fn from_io_error(path: &Path, error: std::io::Error) -> Self {
        Self::Io { path: path.display().to_string(), message: error.to_string() }
    }
}
