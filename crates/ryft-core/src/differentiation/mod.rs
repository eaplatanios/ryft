pub mod forward;
pub mod reverse;
pub mod types;

use thiserror::Error;

use crate::programs::ProgramError;

pub use forward::{
    DifferentiableOperation, DifferentiableProgramOperation, DifferentiationContext, DifferentiationDual,
    DifferentiationTracer, LinearizableProgramOperation, Linearization, LinearizationTracer, Pushforward,
};
pub use reverse::{Pullback, TransposableOperation, TransposableProgramOperation};
pub use types::DifferentiableType;

/// Represents differentiation-related errors.
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
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

    #[error(transparent)]
    Program(#[from] ProgramError),
}
