use thiserror::Error;

// TODO(eaplatanios): Add a ryft `Result` type.
// TODO(eaplatanios): Error messages should be concise lowercase sentences without trailing punctuation.

#[derive(Error, Clone, Debug, Eq, PartialEq)]
pub enum Error {
    #[error("Runtime error: {message}.\n{backtrace}")]
    RuntimeError { message: String, backtrace: String },

    #[error("Got more parameters than expected.")]
    UnusedParams,

    // TODO(eaplatanios): `Error::InsufficientParams` should also include a [`ParameterPath`] for the missing parameter.
    //  and the same should be used in `from_params_with_remainder` whenever relevant.
    #[error("Expected at least {expected_count} parameters but got fewer.")]
    InsufficientParams { expected_count: usize },

    #[error("Named parameter path mismatch. Expected '{expected_path}' but got '{actual_path}'.")]
    NamedParamPathMismatch { expected_path: String, actual_path: String },
}
