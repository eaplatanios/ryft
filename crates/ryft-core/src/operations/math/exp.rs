use crate::macros::define_elementwise_operation;

/// Canonical operation name for [`ExpOperation`].
pub const EXP_OPERATION_NAME: &str = "exp";

// TODO(eaplatanios): Review this macro invocation.
define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise natural exponential of one value (i.e.,
    /// `x ↦ eˣ`, the analytic continuation `e^z` on complex operands) while preserving its type metadata. Only
    /// floating-point and complex operands are supported.
    ExpOperation, EXP_OPERATION_NAME, Exp, exp,
    /// Value-level elementwise natural-exponential capability. [`Exp`] fills the same role for
    /// [`ExpOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
);

// TODO(eaplatanios): Add tests.
