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

    #[error("named parameter path mismatch; expected '{expected_path}' but got '{actual_path}'")]
    NamedParameterPathMismatch { expected_path: String, actual_path: String },

    #[error("missing prefix for parameter path '{path}'")]
    MissingPrefixForPath { path: String },

    #[error("unused prefix path '{path}'")]
    UnusedPrefixPath { path: String },

    #[error("missing named parameter path '{path}'")]
    MissingNamedParameterPath { path: String },

    #[error("unknown named parameter path '{path}'")]
    UnknownNamedParameterPath { path: String },

    #[error("expected exactly {expected_count} replacement values but got {actual_count}")]
    ReplacementCountMismatch { expected_count: usize, actual_count: usize },
}
