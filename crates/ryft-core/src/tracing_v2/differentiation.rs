//! Error types owned by the differentiation helpers in [`crate::tracing_v2`].
//!
//! These errors describe shape and materialization mismatches that arise while building dense
//! Jacobian- or Hessian-like views from traced differentiation results.

use thiserror::Error;

/// Errors emitted by the differentiation helpers in [`crate::tracing_v2`].
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DifferentiationError {
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
