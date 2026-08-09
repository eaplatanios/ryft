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
//!   `*_with_aux` variants preserve non-differentiated auxiliary output, while the `*_holomorphic` variants explicitly
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
//! Implement [`DifferentiableType`] for types that possess tangent and cotangent spaces. Implement
//! [`DifferentiableOperation`] for primitive JVP rules. Rules should express tangent behavior through the provided
//! context and preserve symbolic zeros where possible. Implement [`MemberDifferentiableOperation`] when a homogeneous
//! member operation needs values from its projection's parent context while constructing its derivative. Ordinary
//! homogeneous projected rules should use [`jvp_projected_operation`]. Implement [`TransposableOperation`] for linear
//! primitives that may occur in a homogeneous pushforward. A member payload whose parent instruction has a mixed
//! signature needs no separate rule, because [`transpose_mixed_operation`] delegates that instruction's member-typed
//! data operands, wherever they sit in its operand list, to that same homogeneous rule. Higher-order
//! [`Operation`](crate::Operation) logic belongs with the operation whose instruction attaches to the nested
//! [`Region`](crate::Region). Wrapper operation enums should provide family dispatch and forward to those
//! payload rules.

pub mod batching;
pub mod elementwise;
pub mod forward;
pub mod hessian;
pub mod jacobian;
pub mod linear;
pub mod reverse;
pub mod types;

use std::fmt::{Display, Formatter};

use thiserror::Error;

use crate::parameters::ParameterError;
use crate::programs::{ProgramError, TypeError};

pub use batching::CotangentBatchingPolicy;
pub use elementwise::{
    BinaryElementwiseJvpOperands, BroadcastDerivativeAlignment, ElementwiseDerivativeAlignment,
    UnaryElementwiseJvpOperands, binary_elementwise_jvp, unary_elementwise_jvp,
};
pub use forward::{
    DifferentiableOperation, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationTracer,
    ForwardModeDifferentiate, Linearization, LinearizationTracer, MemberDifferentiableOperation, Pushforward, jvp,
    jvp_projected_operation, linearize,
};
pub use hessian::{
    Hessian, HessianBlock, HessianDifferentiate, hessian, hessian_holomorphic, hessian_holomorphic_with_aux,
    hessian_with_aux,
};
pub use jacobian::{
    Jacobian, JacobianBlock, JacobianDifferentiate, jacobian_forward, jacobian_forward_holomorphic,
    jacobian_forward_holomorphic_with_aux, jacobian_forward_with_aux, jacobian_reverse, jacobian_reverse_holomorphic,
    jacobian_reverse_holomorphic_with_aux, jacobian_reverse_with_aux,
};
pub use linear::{LinearCallOperation, ResidualZeroProvider};
pub use reverse::{
    Pullback, ReverseModeDifferentiate, TransposableOperation, TranspositionDriver, gradient, gradient_holomorphic,
    gradient_holomorphic_with_aux, gradient_with_aux, transpose_mixed_operation, transpose_projected_operation,
    value_and_gradient, value_and_gradient_holomorphic, value_and_gradient_holomorphic_with_aux,
    value_and_gradient_with_aux, vjp,
};
pub use types::{DenseDifferentiableType, DifferentiableType};

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
    /// A non-scalar output describes a vector-valued function whose full derivative is a Jacobian. Seeding that output
    /// with an all-ones cotangent would instead compute the gradient of the sum of its elements, which is a different
    /// operation and not a canonical gradient of the vector-valued function. So, the gradient entry points reject it
    /// up front using this error variant. Use a Jacobian transform for non-scalar outputs.
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

    /// Error returned when a Jacobian or Hessian transform encounters a structured input or output parameter
    /// with no tangent or cotangent space.
    #[error("{transform} {role} parameter at {path} has non-differentiable type {type}")]
    NonDifferentiableParameter {
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: String,
        r#type: String,
    },

    /// Error returned when a non-holomorphic Jacobian or Hessian transform receives a complex parameter whose
    /// derivative requires an explicit holomorphy promise or a real-coordinate representation that Ryft does not
    /// yet expose.
    #[error("{transform} {role} parameter at {path} has complex type {type}; use holomorphic {transform} instead")]
    ComplexParameter {
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: String,
        r#type: String,
    },

    /// Error returned when a holomorphic Jacobian or Hessian transform receives a non-complex parameter.
    #[error("holomorphic {transform} {role} parameter at {path} must be complex but has type {type}")]
    NonComplexParameter {
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: String,
        r#type: String,
    },

    /// Error returned when a Jacobian or Hessian transform cannot enumerate a finite coordinate space for one of its
    /// structured input or output parameters.
    #[error("{transform} {role} parameter at {path} does not have a finite static coordinate space: {type}")]
    NonFiniteCoordinateSpace {
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: String,
        r#type: String,
    },

    /// Error returned when the finite coordinate count of a Jacobian or Hessian parameter structure exceeds `usize`.
    #[error("{transform} {role} coordinate count overflows usize at parameter {path}: {type}")]
    CoordinateCountOverflow {
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: String,
        r#type: String,
    },

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

/// Derivative transform family that was active when an error occurred. Holomorphic entry points use the same forward-
/// or reverse-Jacobian family as their non-holomorphic counterparts. Holomorphy changes the admissible element types,
/// not the mathematical derivative object being materialized.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DerivativeTransform {
    /// Forward-mode Jacobian materialization, including its holomorphic entry point.
    JacobianForward,

    /// Reverse-mode Jacobian materialization, including its holomorphic entry point.
    JacobianReverse,

    /// Hessian materialization.
    Hessian,
}

impl Display for DerivativeTransform {
    #[inline]
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JacobianForward => write!(formatter, "forward Jacobian"),
            Self::JacobianReverse => write!(formatter, "reverse Jacobian"),
            Self::Hessian => write!(formatter, "hessian"),
        }
    }
}

/// Role of a structured parameter in a derivative transform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DifferentiationParameterRole {
    /// Parameter belongs to the differentiated input structure.
    Input,

    /// Parameter belongs to the differentiated output structure.
    Output,

    /// Parameter describes a materialized derivative block.
    Derivative,
}

impl Display for DifferentiationParameterRole {
    #[inline]
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input => write!(formatter, "input"),
            Self::Output => write!(formatter, "output"),
            Self::Derivative => write!(formatter, "derivative"),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_derivative_transform_display() {
        assert_eq!(DerivativeTransform::JacobianForward.to_string(), "forward Jacobian");
        assert_eq!(DerivativeTransform::JacobianReverse.to_string(), "reverse Jacobian");
        assert_eq!(DerivativeTransform::Hessian.to_string(), "hessian");
    }

    #[test]
    fn test_differentiation_parameter_role_display() {
        assert_eq!(DifferentiationParameterRole::Input.to_string(), "input");
        assert_eq!(DifferentiationParameterRole::Output.to_string(), "output");
        assert_eq!(DifferentiationParameterRole::Derivative.to_string(), "derivative");
    }

    #[test]
    fn test_differentiation_parameter_error_display() {
        assert_eq!(
            DifferentiationError::ComplexParameter {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$.value".to_string(),
                r#type: "c64[]".to_string(),
            }
            .to_string(),
            "forward Jacobian input parameter at $.value has complex type c64[]; use holomorphic forward Jacobian \
             instead",
        );
        assert_eq!(
            DifferentiationError::NonComplexParameter {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Output,
                path: "$".to_string(),
                r#type: "f64[]".to_string(),
            }
            .to_string(),
            "holomorphic reverse Jacobian output parameter at $ must be complex but has type f64[]",
        );
    }
}
