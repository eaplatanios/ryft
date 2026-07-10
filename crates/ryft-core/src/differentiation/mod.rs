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
/// [`BatchingError`](crate::BatchingError) and [`ProgramError`] do. The differentiation entry points that are typed at
/// [`ProgramError`] for composability (i.e., `jvp`, `linearize`, and `vjp`, whose errors must flow through enclosing
/// traces) carry their differentiation-specific failures type-erased inside [`ProgramError::Custom`] payloads, while
/// the gradient entry points are typed at [`DifferentiationError`] directly. The paired [`From`] implementations keep
/// this cycle normalized instead of letting the two types nest: converting to [`ProgramError`] unwraps a
/// [`DifferentiationError::Program`] back into the program error that it carries and wraps every other variant in
/// [`ProgramError::Custom`], while converting to [`DifferentiationError`] unwraps a [`ProgramError::Custom`] payload
/// holding a [`DifferentiationError`] and wraps every other program error in [`DifferentiationError::Program`]. Round
/// trips therefore never nest one error type inside the other, and `?` re-types errors correctly at both boundaries.
/// Outside of these conversions, a [`DifferentiationError`] carried by a [`ProgramError`] can be recovered using
/// [`ProgramError::downcast_custom`].
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
    /// Error returned when a differentiation entry point is invoked on an input with no leaf values/parameters.
    /// Differentiating a function of no inputs is degenerate (there is no direction to perturb, so the Jacobian has no
    /// columns), and the free entry points additionally have no leaf value to recover a context from, so every entry
    /// point rejects an empty input up front instead of silently returning vacuous tangents or gradients.
    #[error("differentiation requires an input with at least one leaf value")]
    EmptyInput,

    /// Error returned when reverse-mode differentiation is requested for a function whose output is not a single
    /// scalar. Reverse mode differentiation seeds the output cotangent with the multiplicative identity (i.e., a scalar
    /// value of `1`) and pulls it back to the inputs, which yields a gradient only when the output is a rank-0 scalar.
    /// A non-scalar output describes a vector-valued function whose full derivative is a Jacobian. Because program
    /// interpretation binds inputs positionally without checking their types, seeding such an output with a ones
    /// cotangent would not fail but would instead silently compute the gradient of the sum of the outputs. So, the
    /// gradient entry points reject it up front using this error variant. Use a Jacobian transform for non-scalar
    /// outputs.
    #[error("gradient output must be a rank-0 scalar but got {output_type}")]
    NonScalarGradientOutput { output_type: String },

    /// Error returned when reverse-mode differentiation is requested for a function whose output type carries no
    /// cotangent space (i.e., a non-differentiable type such as a Boolean or an integer scalar). Reverse mode
    /// differentiation seeds the output cotangent with the multiplicative identity (i.e., a scalar value of `1`), but
    /// a non-differentiable output has no such value to seed. So, in this case, the gradient is degenerate and the
    /// gradient entry points reject it up front rather than fabricating a seed.
    #[error("gradient output type {output_type} is non-differentiable and carries no cotangent space")]
    NonDifferentiableGradientOutput { output_type: String },

    /// Error returned when reverse-mode differentiation is requested through a plain (i.e., non-holomorphic) gradient
    /// entry point for a function whose scalar output is complex. A single reverse-mode seed recovers the derivative
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
