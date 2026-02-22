// TODO(eaplatanios): Do we even need to take a dependency on [thiserror]?
use thiserror::Error;

// TODO(eaplatanios): Add a ryft `Result` type.
// TODO(eaplatanios): Error messages should be concise lowercase sentences without trailing punctuation.
// TODO(eaplatanios): Localize error types and leverage the `source` feature of [thiserror].

#[derive(Error, Clone, Debug, Eq, PartialEq)]
pub enum Error {
    #[error("Runtime error: {message}.\n{backtrace}")]
    RuntimeError { message: String, backtrace: String },

    #[error("Got more parameters than expected.")]
    UnusedParams,

    #[error("Expected at least {expected_count} parameters but got fewer.")]
    InsufficientParams { expected_count: usize },
}
