use thiserror::Error;

#[derive(Error, Clone, Debug, Eq, PartialEq)]
pub enum Error {
    #[error("runtime error: {message}")]
    RuntimeError { message: String, backtrace: String },

    // TODO(eaplatanios): Can we unify some of the parameter-related errors?
    #[error("got more parameters than expected")]
    UnusedParameters,

    // TODO(eaplatanios): `Error::InsufficientParameters` should also include a [`ParameterPath`] for the missing
    //  parameter and the same should be used in `from_parameters_with_remainder` whenever relevant.
    #[error("expected at least {expected_count} parameters but got fewer")]
    InsufficientParameters { expected_count: usize },

    #[error("missing prefix for parameter at '{path}' path")]
    MissingPrefixForParameterPath { path: String },

    #[error("unused parameter at '{path}' path")]
    UnusedParameter { path: String },

    #[error("missing parameter at '{path}' path")]
    MissingParameterPath { path: String },

    #[error("unknown parameter path '{path}'")]
    UnknownParameterPath { path: String },

    #[error("expected exactly {expected_count} replacement parameter values but got {actual_count}")]
    ParameterReplacementCountMismatch { expected_count: usize, actual_count: usize },
}
