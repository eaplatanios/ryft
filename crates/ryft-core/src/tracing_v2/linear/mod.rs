/// Structured differential materialization helpers (forward- and reverse-mode Jacobians, Hessian).
mod differential;
/// Public reverse-mode APIs built from traced programs and staged pullbacks.
mod reverse;

pub use differential::{
    CoordinateValue, DifferentiableDomainExtension, Differential, DifferentialBlock, DifferentialRow, Hessian,
    Jacobian, jacrev,
};
pub use reverse::{grad, grad_with_aux, value_and_grad, value_and_grad_with_aux};
