use crate::macros::define_elementwise_operation;

/// Canonical operation name for [`ExponentialOperation`].
pub const EXPONENTIAL_OPERATION_NAME: &'static str = "exponential";

// TODO(eaplatanios): Review this macro invocation.
define_elementwise_operation!(
    @unary
    /// [`Operation`](crate::Operation) that computes the elementwise natural exponential of one value (i.e.,
    /// `x ↦ eˣ`, the analytic continuation `e^z` on complex operands) while preserving its type metadata. Only
    /// floating-point and complex operands are supported.
    ExponentialOperation, EXPONENTIAL_OPERATION_NAME, Exponential, exponential,
    /// Value-level elementwise natural-exponential capability. [`Exponential`] fills the same role for
    /// [`ExponentialOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
);

// TODO(eaplatanios): Add tests.
