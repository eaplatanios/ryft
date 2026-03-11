use std::backtrace::Backtrace;

use thiserror::Error;

/// Represents errors that can occur in `ryft-core`.
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Error {
    /// Error returned when an argument value is invalid.
    #[error("{message}")]
    InvalidArgument { message: String, backtrace: String },

    /// Error returned when a data type is invalid.
    #[error("{message}")]
    InvalidDataType { message: String, backtrace: String },

    /// Error returned when a data type promotion is invalid.
    #[error("{message}")]
    InvalidDataTypePromotion { message: String, backtrace: String },

    /// Error returned when extra parameter values remain unused.
    #[error(
        "{}",
        match paths.as_deref() {
            None => "got more parameters than expected".to_string(),
            Some(paths) => format!(
                "got more parameters than expected; unused parameter paths: {}",
                paths.iter().map(|path| format!("'{path}'")).collect::<Vec<_>>().join(", "),
            ),
        }
    )]
    UnusedParameters { paths: Option<Vec<String>> },

    /// Error returned when parameter values are missing.
    #[error(
        "{}",
        match paths.as_deref() {
            None => format!("got fewer parameters than expected; expected at least {expected_count}"),
            Some(paths) => format!(
                "got fewer parameters than expected; expected at least {expected_count}; missing parameter paths: {}",
                paths.iter().map(|path| format!("'{path}'")).collect::<Vec<_>>().join(", "),
            ),
        }
    )]
    MissingParameters { expected_count: usize, paths: Option<Vec<String>> },

    /// Error returned when parameter combinations are ambiguous.
    #[error(
        "got ambiguous parameter values while combining parameterized values; conflicting values: {}",
        values.iter().map(|value| format!("'{value}'")).collect::<Vec<_>>().join(", "),
    )]
    AmbiguousParameterCombination { values: Vec<String> },
}

impl Error {
    /// Creates a new [`Error::InvalidArgument`].
    pub fn invalid_argument<M: Into<String>>(message: M) -> Self {
        Self::InvalidArgument { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::InvalidDataType`].
    pub fn invalid_data_type<M: Into<String>>(message: M) -> Self {
        Self::InvalidDataType { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    /// Creates a new [`Error::InvalidDataTypePromotion`].
    pub fn invalid_data_type_promotion<M: Into<String>>(message: M) -> Self {
        Self::InvalidDataTypePromotion { message: message.into(), backtrace: Backtrace::capture().to_string() }
    }

    // TODO(eaplatanios): Add constructors for the parameter-related errors.
}
