/// Higher-order custom-derivative operations (`custom_jvp` / `custom_vjp`).
pub mod custom_derivatives;

/// Recomputed primal operation payload used inside linear programs.
pub mod recompute;

pub use custom_derivatives::{
    CustomJvp, CustomJvpOperation, CustomVjp, CustomVjpOperation, CustomVjpResidual, custom_jvp, custom_vjp,
    transpose_primal_custom_vjp,
};
pub use recompute::RecomputeOperation;
