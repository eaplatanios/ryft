//! Contains machinery for forward and reverse mode automatic _differentiation_.
//!
//! Differentiation is expressed as [`Context`](crate::Context) composition and [`Program`](crate::Program)
//! transformation rather than as a backend-specific facility. Values flowing through a [`DifferentiationContext`]
//! carry a primal and a tangent, operation-owned Jacobian-Vector Product (JVP) rules propagate those duals, partial
//! evaluation splits primal work from reusable linear work, and transposition turns the resulting linear program into
//! a pullback for reverse mode differentiation.
//!
//! # Entry Points
//!
//! The free functions operate on ordinary structured values and recover the execution context from their inputs:
//!
//! - [`jvp`] computes a Jacobian-vector product for one set of primals and tangents. It is the direct forward mode
//!   transform and is usually best when there are few input directions or when both primal and tangent outputs are
//!   needed immediately.
//! - [`linearize`] runs the primal computation once and returns its value together with a reusable [`Pushforward`]
//!   that can be applied to many tangent inputs without repeating nonlinear primal work.
//! - [`vjp`] runs the primal computation and returns a reusable [`Pullback`] mapping output cotangents to input
//!   cotangents. Conceptually, `vjp = linearize + transpose`.
//! - [`value_and_gradient`] and [`gradient`] are scalar-output conveniences that seed the pullback with one. The
//!   `*_with_aux` variants preserve nondifferentiated auxiliary output, while the `*_holomorphic` variants explicitly
//!   opt into the complex holomorphic contract.
//!
//! Context-generic code can use [`ForwardModeDifferentiate`] and [`ReverseModeDifferentiate`] directly, and already
//! traced programs expose the corresponding program-level JVP, linearization, and transposition methods documented on
//! [`Program`](crate::programs::Program).
//!
//! # Forward Mode Differentiation
//!
//! [`DifferentiationDual`] pairs each primal with a [`MaybeZero`](crate::programs::MaybeZero) tangent, and a
//! [`DifferentiationTracer`] carries that dual through a [`DifferentiationContext`]. When an operation is bound, its
//! [`DifferentiableOperation`] rule receives primal and tangent inputs, stages or evaluates the primal operation, and
//! produces tangent outputs. Symbolic zero tangents avoid materializing unnecessary zero arrays.
//!
//! [`linearize`] composes differentiation with [`PartialEvaluationContext`](crate::PartialEvaluationContext).
//! Primals are known and tangents are unknown. Nonlinear primal work is evaluated once, values needed by the tangent
//! computation become residuals, and the residual program becomes a reusable [`Pushforward`]. Host control flow may
//! branch on concrete primals during this process, so only the taken path is linearized.
//!
//! The following is an illustration of forward mode differentiation:
//!
//! ```text
//! ┌─────────────────────┐   apply operation JVP rules   ┌───────────────────────────┐
//! │ Primals + Tangents  │ ────────────────────────────▶ │ Outputs + Output Tangents │
//! └─────────────────────┘                               └───────────────────────────┘
//! ```
//!
//! # Reverse Mode Differentiation
//!
//! Reverse mode differentiation reuses forward linearization instead of maintaining an independent nonlinear trace.
//! The linearized tangent program is transposed by applying [`TransposableOperation`] rules in reverse dataflow order,
//! and the result is a [`Pullback`] that accepts output cotangents, consumes saved residuals, and accumulates input
//! cotangents. This architecture keeps primal execution, residualization, and linear algebra as separate, composable
//! concerns.
//!
//! The following is an illustration of reverse mode differentiation:
//!
//! ```text
//!        ┌─────────────────────────────┐
//!        │ Primals + Unknown Tangents  │
//!        └──────────────┬──────────────┘
//!                       │ linearize (JVP with unknown tangents under partial evaluation)
//!                       ▼
//! ┌───────────────────────────────────────────┐
//! │ Primal Outputs + Residuals + Pushforward  │
//! └─────────────────────┬─────────────────────┘
//!                       │ transpose the linear pushforward
//!                       ▼
//!             ┌───────────────────┐
//!             │ Reusable Pullback │
//!             └───────────────────┘
//! ```
//!
//! # Extending differentiation
//!
//! Implement [`DifferentiableType`] for type descriptors that possess tangent and cotangent spaces. Implement
//! [`DifferentiableOperation`] for primitive JVP rules. Rules should express tangent behavior through the provided
//! context and preserve symbolic zeros where possible. Implement [`TransposableOperation`] for linear primitives that
//! may occur in a pushforward. Finally, implement [`DifferentiableProgramOperation`] and
//! [`TransposableProgramOperation`] for operation families that recursively contain flat programs.
//! [`LinearizableProgramOperation`] is the operation-family fixed point used by program linearization. Higher-order
//! operation logic belongs with the operation that owns the nested program. Wrapper operation enums should provide
//! family dispatch and forward to those payload rules.

pub mod forward;
pub mod reverse;
pub mod types;

use thiserror::Error;

use crate::parameters::ParameterError;
use crate::programs::ProgramError;
use crate::types::TypeError;

pub use forward::{
    DifferentiableOperation, DifferentiableProgramOperation, DifferentiationContext, DifferentiationDual,
    DifferentiationTracer, ForwardModeDifferentiate, LinearizableProgramOperation, Linearization, LinearizationTracer,
    Pushforward, jvp, linearize,
};
pub use reverse::{
    Pullback, ReverseModeDifferentiate, TransposableOperation, TransposableProgramOperation, gradient,
    gradient_holomorphic, gradient_holomorphic_with_aux, gradient_with_aux, value_and_gradient,
    value_and_gradient_holomorphic, value_and_gradient_holomorphic_with_aux, value_and_gradient_with_aux, vjp,
};
pub use types::DifferentiableType;

/// Represents differentiation-related errors.
///
/// [`DifferentiationError`] and [`ProgramError`] form the same normalized conversion cycle as
/// [`BatchingError`](crate::BatchingError) and [`ProgramError`] do. Every differentiation surface including
/// the value-level entry points (i.e., `jvp`, `linearize`, `vjp`, etc.), the program-level transforms (i.e.,
/// `Program::jvp`, `Program::linearize`, `Program::transpose`, etc.), and the per-operation rule traits (i.e.,
/// [`DifferentiableOperation`] and [`TransposableOperation`]), returns [`DifferentiationError`], while the errors those
/// rules produce *through* the kernel (i.e., binding and staging operations) are [`ProgramError`]s. The paired [`From`]
/// implementations keep this cycle normalized instead of letting the two types nest: converting to [`ProgramError`]
/// unwraps a [`DifferentiationError::Program`] back into the program error that it carries and wraps every other
/// variant in [`ProgramError::Custom`], while converting to [`DifferentiationError`] unwraps a [`ProgramError::Custom`]
/// payload holding a [`DifferentiationError`] and wraps every other program error in [`DifferentiationError::Program`].
/// Roundtrips therefore never nest one error type inside the other, and `?` re-types errors correctly at both
/// boundaries. Outside of these conversions, a [`DifferentiationError`] carried by a [`ProgramError`] can be recovered
/// using [`ProgramError::downcast_custom`].
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
    /// Error returned when a differentiation entry point is invoked on an input with no leaf values/parameters.
    /// Differentiating a function of no inputs is degenerate (there is no direction to perturb, so the Jacobian has no
    /// columns), and the free entry points additionally have no leaf value to recover a context from, so every entry
    /// point rejects an empty input up front instead of silently returning vacuous tangents or gradients.
    #[error("differentiation requires an input with at least one leaf value")]
    EmptyInput,

    /// Error returned when reverse mode differentiation is requested for a function whose output is not a single
    /// scalar. Reverse mode differentiation seeds the output cotangent with the multiplicative identity (i.e., a scalar
    /// value of `1`) and pulls it back to the inputs, which yields a gradient only when the output is a rank-0 scalar.
    /// A non-scalar output describes a vector-valued function whose full derivative is a Jacobian. Because program
    /// interpretation binds inputs positionally without checking their types, seeding such an output with a ones
    /// cotangent would not fail but would instead silently compute the gradient of the sum of the outputs. So, the
    /// gradient entry points reject it up front using this error variant. Use a Jacobian transform for non-scalar
    /// outputs.
    #[error("gradient output must be a rank-0 scalar but got {output_type}")]
    NonScalarGradientOutput { output_type: String },

    /// Error returned when reverse mode differentiation is requested for a function whose output type carries no
    /// cotangent space (i.e., a non-differentiable type such as a Boolean or an integer scalar). Reverse mode
    /// differentiation seeds the output cotangent with the multiplicative identity (i.e., a scalar value of `1`), but
    /// a non-differentiable output has no such value to seed. So, in this case, the gradient is degenerate and the
    /// gradient entry points reject it up front rather than fabricating a seed.
    #[error("gradient output type {output_type} is non-differentiable and carries no cotangent space")]
    NonDifferentiableGradientOutput { output_type: String },

    /// Error returned when reverse mode differentiation is requested through a plain (i.e., non-holomorphic) gradient
    /// entry point for a function whose scalar output is complex. A single reverse mode seed recovers the derivative
    /// of a complex-output function only when the function is holomorphic (i.e., complex-differentiable), a promise
    /// the plain entry points do not ask for, so they reject complex outputs up front instead of silently computing a
    /// value that is not a derivative. Use the `*_holomorphic` gradient entry points when the function is holomorphic,
    /// or split the function into its real and imaginary parts and differentiate those otherwise.
    #[error(
        "gradient output type {output_type} is complex; use a holomorphic gradient entry point \
        if the function is holomorphic"
    )]
    ComplexGradientOutput { output_type: String },

    #[error(transparent)]
    Program(ProgramError),
}

impl From<TypeError> for DifferentiationError {
    #[inline]
    fn from(error: TypeError) -> Self {
        DifferentiationError::Program(error.into())
    }
}

impl From<ParameterError> for DifferentiationError {
    #[inline]
    fn from(error: ParameterError) -> Self {
        DifferentiationError::Program(error.into())
    }
}

impl From<ProgramError> for DifferentiationError {
    #[inline]
    fn from(error: ProgramError) -> Self {
        if let Some(differentiation) = error.downcast_custom::<DifferentiationError>() {
            differentiation.clone()
        } else {
            DifferentiationError::Program(error)
        }
    }
}

impl From<DifferentiationError> for ProgramError {
    #[inline]
    fn from(error: DifferentiationError) -> Self {
        match error {
            DifferentiationError::Program(error) => error,
            error => ProgramError::custom(error),
        }
    }
}

/// Adapter that lets the reverse-mode gradient entry points accept both plain and fallible closures. The gradient
/// family (i.e., [`value_and_gradient`], [`gradient`], and their holomorphic and auxiliary-output variants) accepts
/// any closure output implementing `MaybeFallible<T>`, where `T` is the expected traced output shape. Returning `T`
/// directly requires no wrapping, while returning `Result<T, E>` for any `E` that converts into
/// [`DifferentiationError`] (e.g., [`ProgramError`]) enables using `?` inside the closure.
///
/// This dual-mode contract is only offered where the expected traced output shape has a concrete outer constructor
/// (i.e., a [`LinearizationTracer`] or a tracer/auxiliary tuple), which is what lets type inference select between
/// the two implementations unambiguously. Entry points with fully generic traced outputs (i.e., [`jvp`], [`linearize`],
/// and [`vjp`]) accept fallible closures only.
pub trait MaybeFallible<T> {
    /// Converts this closure output into a [`Result`], wrapping plain outputs in [`Ok`] and converting the error type
    /// of already fallible outputs into [`DifferentiationError`].
    fn into_result(self) -> Result<T, DifferentiationError>;
}

impl<T> MaybeFallible<T> for T {
    #[inline]
    fn into_result(self) -> Result<T, DifferentiationError> {
        Ok(self)
    }
}

impl<T, E: Into<DifferentiationError>> MaybeFallible<T> for Result<T, E> {
    #[inline]
    fn into_result(self) -> Result<T, DifferentiationError> {
        self.map_err(Into::into)
    }
}
