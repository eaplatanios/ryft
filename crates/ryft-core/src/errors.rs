use thiserror::Error;

#[derive(Error, Clone, Debug, Eq, PartialEq)]
pub enum Error {
    #[error("runtime error: {message}")]
    RuntimeError { message: String, backtrace: String },

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

    #[error(
        "got ambiguous parameter values while combining parameterized values; conflicting values: {}",
        values.iter().map(|value| format!("'{value}'")).collect::<Vec<_>>().join(", "),
    )]
    AmbiguousParameterCombination { values: Vec<String> },

    #[error("unknown parameter path '{path}'")]
    UnknownParameterPath { path: String },

    #[error("expected exactly {expected_count} replacement parameter values but got {actual_count}")]
    ParameterReplacementCountMismatch { expected_count: usize, actual_count: usize },
}
