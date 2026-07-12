use crate::macros::define_elementwise_operation;

/// Canonical operation name for [`LogOperation`].
pub const LOG_OPERATION_NAME: &'static str = "log";

// TODO(eaplatanios): Review this macro invocation.
define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise natural logarithm of one value (i.e.,
    /// `x ↦ ln(x)`, the principal branch `ln(z)` on complex operands) while preserving its type metadata. Only
    /// floating-point and complex operands are supported.
    LogOperation, LOG_OPERATION_NAME, Log, log,
    /// Value-level elementwise natural-logarithm capability. [`Log`] fills the same role for
    /// [`LogOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
);

// TODO(eaplatanios): Add tests.
