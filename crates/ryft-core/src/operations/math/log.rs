use crate::macros::define_elementwise_operation;

/// Canonical operation name for [`LogarithmOperation`].
pub const LOGARITHM_OPERATION_NAME: &'static str = "logarithm";

// TODO(eaplatanios): Review this macro invocation.
define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise natural logarithm of one value (i.e.,
    /// `x ↦ ln(x)`, the principal branch `ln(z)` on complex operands) while preserving its type metadata. Only
    /// floating-point and complex operands are supported.
    LogarithmOperation, LOGARITHM_OPERATION_NAME, Logarithm, logarithm,
    /// Value-level elementwise natural-logarithm capability. [`Logarithm`] fills the same role for
    /// [`LogarithmOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
);

// TODO(eaplatanios): Add tests.
