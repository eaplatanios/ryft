use std::backtrace::Backtrace;

use thiserror::Error;

/// Represents errors that can occur in `ryft-mlir`. Each variant includes a `backtrace` field that captures the call
/// stack at the point where the error was created, which is useful for debugging. Note that it is represented as a
/// [`String`] and not as a [`Backtrace`] because using the latter is only currently supported in unstable Rust.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Error {
    /// Error that represents when the caller of a function provides an invalid input.
    #[error("{message}")]
    InvalidArgument { message: String, backtrace: String },

    /// Error that represents when MLIR fails to parse a rendered intermediate representation (IR) string.
    #[error("{message}")]
    ParsingError { message: String, backtrace: String },

    /// Error that represents errors raised internally by the MLIR native library.
    #[error("{message}")]
    Internal { message: String, backtrace: String },
}

impl Error {
    /// Creates a new [`Error::InvalidArgument`].
    pub fn invalid_argument<M: Into<String>>(message: M) -> Self {
        Self::InvalidArgument { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::ParsingError`].
    pub fn parsing_error<M: Into<String>>(message: M) -> Self {
        Self::ParsingError { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::Internal`].
    pub fn internal<M: Into<String>>(message: M) -> Self {
        Self::Internal { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }
}
