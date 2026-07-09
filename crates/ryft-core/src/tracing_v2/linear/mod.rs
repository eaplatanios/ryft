/// Structured differential materialization helpers (forward- and reverse-mode Jacobians, Hessian).
mod differential;

pub use differential::{
    CoordinateBasis, DifferentiableDomainExtension, Differential, DifferentialBlock, DifferentialRow, Hessian,
    Jacobian, jacrev,
};
