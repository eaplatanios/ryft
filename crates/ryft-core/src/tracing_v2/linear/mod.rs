/// Structured differential materialization helpers (forward- and reverse-mode Jacobians, Hessian).
mod differential;
/// Public reverse-mode APIs built from traced programs and staged pullbacks.
mod reverse;

pub use differential::{
    CoordinateValue, DifferentiableDomainExtension, Differential, DifferentialBlock, DifferentialRow, Hessian,
    Jacobian, jacrev,
};
pub use reverse::{gradient, gradient_with_aux, value_and_gradient, value_and_gradient_with_aux};
