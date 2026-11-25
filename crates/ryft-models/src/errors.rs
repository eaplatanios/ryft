//! Crate-level error type of `ryft-models`.

use ryft_checkpoints::CheckpointError;
use ryft_core::ProgramError;
use thiserror::Error;

use crate::inkling2::Inkling2ConfigurationError;
use crate::json::JsonError;
use crate::renderers::RendererError;
use crate::tokenizers::TokenizerError;

// TODO(eaplatanios): Review this module.

/// Error returned by `ryft-models`, which aggregates the errors that its modules own. Every module returns its own
/// error type, and this type is for callers that combine several of them (e.g., a program that reads a
/// configuration, imports a checkpoint, renders a conversation, and evaluates a model).
#[derive(Clone, Debug, PartialEq, Error)]
pub enum Error {
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),

    #[error(transparent)]
    Configuration(#[from] Inkling2ConfigurationError),

    #[error(transparent)]
    Json(#[from] JsonError),

    #[error(transparent)]
    Program(#[from] ProgramError),

    #[error(transparent)]
    Renderer(#[from] RendererError),

    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_error() {
        // Module errors convert into the crate error and keep their messages.
        let error = Error::from(JsonError::Invalid { message: "empty".to_string() });
        assert_eq!(error.to_string(), "invalid JSON: empty");
        assert_eq!(error, Error::Json(JsonError::Invalid { message: "empty".to_string() }));
        let error = Error::from(CheckpointError::UnknownArray { name: "weights".to_string() });
        assert_eq!(error.to_string(), "unknown checkpoint array `weights`");
    }
}
