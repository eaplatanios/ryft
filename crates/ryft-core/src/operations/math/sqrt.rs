use crate::macros::define_elementwise_operation;

/// Canonical operation name for [`SqrtOperation`].
pub const SQRT_OPERATION_NAME: &str = "sqrt";

// TODO(eaplatanios): Review this macro invocation.
define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise square root of one value (i.e., `x ↦ √x`, the
    /// principal branch `√z` on complex operands) while preserving its type metadata. Only floating-point and complex
    /// operands are supported.
    SqrtOperation, SQRT_OPERATION_NAME, Sqrt, sqrt,
    /// Value-level elementwise square-root capability. [`Sqrt`] fills the same role for
    /// [`SqrtOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
);

// TODO(eaplatanios): Add tests.
