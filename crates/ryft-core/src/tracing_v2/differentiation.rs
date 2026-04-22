use thiserror::Error;

/// Errors emitted by the differentiation helpers in [`crate::tracing_v2`].
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
    /// Traced forward-mode differentiation was invoked without any staged input leaves.
    #[error("traced jvp requires at least one input leaf to recover the staging context")]
    MissingTracedJvpInputLeaves,

    /// Traced reverse-mode differentiation was invoked without any staged input leaves.
    #[error("traced reverse-mode requires at least one input leaf to recover the staging context")]
    MissingTracedReverseModeInputLeaves,

    /// Traced rematerialization was invoked without any staged input leaves.
    #[error("traced rematerialize requires at least one input leaf to recover the staging context")]
    MissingTracedRematerializeInputLeaves,

    /// Linear rematerialization replay was invoked without any tangent leaves.
    #[error("linear rematerialize replay requires at least one tangent leaf to recover the staging context")]
    MissingLinearRematerializeReplayTangentLeaves,

    /// Linear rematerialization transpose was invoked without any output cotangent leaves.
    #[error(
        "linear rematerialize transpose requires at least one output cotangent leaf to recover the staging context"
    )]
    MissingLinearRematerializeTransposeCotangentLeaves,

    /// Dense Jacobian materialization produced an unexpected number of rows.
    #[error("invalid Jacobian row count; expected {expected} but got {got}")]
    InvalidJacobianRowCount { expected: usize, got: usize },

    /// Dense Jacobian materialization produced a row with an unexpected width.
    #[error("invalid Jacobian row width; expected {expected} but got {got}")]
    InvalidJacobianRowWidth { expected: usize, got: usize },

    /// Dense Jacobian materialization produced an unexpected number of columns.
    #[error("invalid Jacobian column count; expected {expected} but got {got}")]
    InvalidJacobianColumnCount { expected: usize, got: usize },

    /// Dense Jacobian materialization produced a column with an unexpected height.
    #[error("invalid Jacobian column height; expected {expected} but got {got}")]
    InvalidJacobianColumnHeight { expected: usize, got: usize },
}
