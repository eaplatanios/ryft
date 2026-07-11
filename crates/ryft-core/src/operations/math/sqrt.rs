use crate::macros::define_elementwise_operation;

/// Canonical operation name for [`SquareRootOperation`].
pub const SQUARE_ROOT_OPERATION_NAME: &'static str = "square_root";

// TODO(eaplatanios): Review this macro invocation.
define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise square root of one value (i.e., `x ↦ √x`, the
    /// principal branch `√z` on complex operands) while preserving its type metadata. Only floating-point and complex
    /// operands are supported.
    SquareRootOperation, SQUARE_ROOT_OPERATION_NAME, SquareRoot, square_root,
    /// Value-level elementwise square-root capability. [`SquareRoot`] fills the same role for
    /// [`SquareRootOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
);

// TODO(eaplatanios): Add tests.
