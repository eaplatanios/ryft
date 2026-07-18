/// Named-axis collective primitives (`psum`, `pmean`, `pmax`).
pub mod collective;

/// Axis-joining concatenation differentiation and batching rules.
pub mod concatenation;

/// Value-level identity helpers and built-in scalar constant traits.
pub mod constants;

/// Higher-order condition and while-loop operations.
pub mod control_flow;

/// Higher-order custom-derivative operations (`custom_jvp` / `custom_vjp`).
pub mod custom_derivatives;

/// Generalized dot product (tensor contraction) primitive.
pub mod dot;

/// Indexed gather differentiation rules.
pub mod gather;

/// Memory-space transfer primitive.
pub mod memory;

/// Edge and interior padding differentiation and batching rules.
pub mod padding;

/// Reusable operation types for the built-in operation set and static backend extensions.

/// Axis-collapsing reduction primitive.
pub mod reduce;

/// Recomputed primal operation payload used inside linear programs.
pub mod recompute;

/// Reshaping primitive.
pub mod reshape;

/// Statically shaped scan loop differentiation and batching rules.
pub mod scan;

/// Indexed scatter differentiation rules.
pub mod scatter;

/// Per-element select / `where` primitive.
pub mod select;

/// Resharding and sharding-constraint differentiation and batching rules.
pub mod sharding;

/// Static and dynamic slicing differentiation and batching rules.
pub mod slicing;

/// N-dimensional axis-permutation differentiation and batching rules.
pub mod transpose;

pub use collective::{Collective, CollectiveKind, CollectiveOperation, forward_collective_to_parent};
pub use control_flow::transpose_primal_condition;
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
pub use reshape::{ReshapeOps, ReshapeValue, lift_reshape_shapes};
pub use scan::transpose_primal_scan;
pub use transpose::lift_permutation;
