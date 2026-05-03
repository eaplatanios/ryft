/// Elementwise addition linearization and differentiation rules.
pub mod add;

/// Elementwise cosine.
pub mod cos;

/// Value-level identity helpers and built-in scalar constant traits.
pub mod constants;

/// Higher-order condition and while-loop operations.
pub mod control_flow;

/// Custom-primitive escape hatch.
pub mod custom;

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

/// Closed default carriers for the built-in operation set.
pub mod primitive;

/// Traced rematerialization boundary.
pub mod rematerialize;

/// Reshaping primitive.
pub mod reshape;

/// Linear right matrix multiplication.
pub mod right_matmul;

/// Scalar and tensor scaling.
pub mod scale;

/// Elementwise sine.
pub mod sin;

/// Elementwise subtraction linearization and differentiation rules.
pub mod sub;

pub use control_flow::{
    ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, FlatProgram, WhileOperation,
    flat_program_input_types, flat_program_output_types,
};
pub use cos::{Cos, CosOperation, SupportsCos};
pub use custom::{
    CustomOperationError, CustomPrimitive, CustomPrimitiveExtensions, LinearCustomPrimitive, SupportsCustom,
    SupportsLinearCustom,
};
pub use left_matmul::{LeftMatMulOperation, SupportsLeftMatMul};
pub use matmul::{MatMulOperation, SupportsMatMul};
pub use matrix_transpose::{MatrixTransposeOperation, SupportsMatrixTranspose};
pub use neg::{NegOperation, SupportsNeg};
pub use primitive::{ArrayOperation, LinearArrayOperation, LinearScalarOperation, ScalarOperation};
pub use rematerialize::{
    FlatTracedRematerialize, LinearRematerializeOperation, RematerializeOperation, SupportsLinearRematerialize,
    SupportsRematerialize,
};
pub use reshape::{ReshapeOperation, SupportsReshape};
pub use right_matmul::{RightMatMulOperation, SupportsRightMatMul};
pub use scale::{ScaleOperation, SupportsScale};
pub use sin::{Sin, SinOperation, SupportsSin};
