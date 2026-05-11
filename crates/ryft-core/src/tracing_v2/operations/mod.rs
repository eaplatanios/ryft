/// Elementwise addition linearization and differentiation rules.
pub mod add;

/// Elementwise cosine differentiation rules.
pub mod cos;

/// Value-level identity helpers and built-in scalar constant traits.
pub mod constants;

/// Higher-order condition and while-loop operations.
pub mod control_flow;

/// Elementwise division differentiation rules.
pub mod div;

/// Linear left matrix multiplication.
pub mod left_matmul;

/// Matrix capability layer shared by matrix staged operations.
pub mod matrix;

/// Matrix multiplication.
pub mod matmul;

/// Matrix transposition.
pub mod matrix_transpose;

/// Elementwise multiplication differentiation rules.
pub mod mul;

/// Elementwise negation.
pub mod neg;

/// Reusable operation carriers for the built-in operation set and static backend extensions.
pub mod primitive;

/// Reshaping primitive.
pub mod reshape;

/// Linear right matrix multiplication.
pub mod right_matmul;

/// Scalar and tensor scaling.
pub mod scale;

/// Elementwise sine differentiation rules.
pub mod sin;

/// Elementwise subtraction linearization and differentiation rules.
pub mod sub;

pub use control_flow::{
    ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, FlatProgram, WhileOperation,
    flat_program_input_types, flat_program_output_types,
};
pub use left_matmul::{LeftMatMul, LeftMatMulOperation, SupportsLeftMatMul};
pub use matmul::{MatMul, MatMulOperation, SupportsMatMul};
pub use matrix::{MatrixOps, MatrixValue};
pub use matrix_transpose::{MatrixTranspose, MatrixTransposeOperation, SupportsMatrixTranspose};
pub use primitive::{ArrayOperation, LinearArrayOperation, NoOperationExtension, TracerReplayValue};
pub use reshape::{Reshape, ReshapeOperation, ReshapeOps, ReshapeValue, SupportsReshape};
pub use right_matmul::{RightMatMul, RightMatMulOperation, SupportsRightMatMul};
