/// Named-axis collective primitives (`psum`, `pmean`, `pmax`).
pub mod collective;

/// Higher-order custom-derivative operations (`custom_jvp` / `custom_vjp`).
pub mod custom_derivatives;

/// Generalized dot product (tensor contraction) primitive.
pub mod dot;

/// Memory-space transfer primitive.
pub mod memory;

/// Axis-collapsing reduction primitive.
pub mod reduce;

/// Recomputed primal operation payload used inside linear programs.
pub mod recompute;

/// Per-element select / `where` primitive.
pub mod select;

pub use collective::{Collective, CollectiveKind, CollectiveOperation, forward_collective_to_parent};
pub use custom_derivatives::{
    CustomJvp, CustomJvpOperation, CustomVjp, CustomVjpOperation, CustomVjpResidual, custom_jvp, custom_vjp,
    transpose_primal_custom_vjp,
};
pub use dot::{
    Dot, DotDimensionNumbers, DotOperation, DotOps, adjoint_dimensions_for_left_dot, adjoint_dimensions_for_right_dot,
    dot_general_evaluate, lhs_result_axes, lift_dot_dimensions, rhs_result_axes,
};
pub use memory::{TRANSFER_TO_MEMORY_OPERATION_NAME, TransferToMemory, TransferToMemoryOperation};
pub use recompute::RecomputeOperation;
pub use reduce::{Reduce, ReduceOperation, ReductionKind, lift_reduce_axes, reduce_abstract, reduce_evaluate};
