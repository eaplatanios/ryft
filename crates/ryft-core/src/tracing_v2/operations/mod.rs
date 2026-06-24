/// Elementwise addition linearization and differentiation rules.
pub mod add;

/// Value-capability bundles aggregating the repeated trait-bound clusters used by the primitive operation enums.
pub mod bounds;

/// N-dimensional broadcast differentiation and batching rules.
pub mod broadcasting;

/// Named-axis collective primitives (`psum`, `pmean`, `pmax`).
pub mod collective;

/// Axis-joining concatenation differentiation and batching rules.
pub mod concatenation;

/// Elementwise pairwise comparison primitive.
pub mod compare;

/// Elementwise cosine differentiation rules.
pub mod cos;

/// Value-level identity helpers and built-in scalar constant traits.
pub mod constants;

/// Higher-order condition and while-loop operations.
pub mod control_flow;

/// Higher-order custom-derivative operations (`custom_jvp` / `custom_vjp`).
pub mod custom_derivatives;

/// Elementwise division differentiation rules.
pub mod div;

/// Generalized dot product (tensor contraction) primitive.
pub mod dot;

/// Indexed gather differentiation rules.
pub mod gather;

/// Elementwise logical operation batching and differentiation rules.
pub mod logical;

/// Memory-space transfer primitive.
pub mod memory;

/// Elementwise multiplication differentiation rules.
pub mod mul;

/// Elementwise negation.
pub mod neg;

/// Edge and interior padding differentiation and batching rules.
pub mod padding;

/// Reusable operation types for the built-in operation set and static backend extensions.
pub mod primitive;

/// Axis-collapsing reduction primitive.
pub mod reduce;

/// Recomputed primal operation payload used inside linear programs.
pub mod recompute;

/// Reshaping primitive.
pub mod reshape;

/// Captured-factor payloads and materialization.
pub mod captures;

/// Scalar and tensor scaling.
pub mod scale;

/// Statically shaped scan loop differentiation and batching rules.
pub mod scan;

/// Indexed scatter differentiation rules.
pub mod scatter;

/// Per-element select / `where` primitive.
pub mod select;

/// Resharding and sharding-constraint differentiation and batching rules.
pub mod sharding;

/// Elementwise sine differentiation rules.
pub mod sin;

/// Static and dynamic slicing differentiation and batching rules.
pub mod slicing;

/// Gradient-severing `stop_gradient` differentiation rule.
pub mod stop_gradient;

/// Elementwise subtraction linearization and differentiation rules.
pub mod sub;

/// N-dimensional axis-permutation differentiation and batching rules.
pub mod transpose;

pub use bounds::{
    SupportsArithmeticOperations, SupportsComparisonOperations, SupportsConstantOperations,
    SupportsLinearAlgebraOperations, SupportsLinearArithmeticOperations, SupportsLinearArrayOperation,
    SupportsLinearScalarOperation, SupportsManipulationOperations, SupportsTrigonometricOperations,
};
pub use broadcasting::lift_broadcast;
pub use captures::{MaterializeCaptureOperation, ValueOrCapture};
pub use collective::{Collective, CollectiveKind, CollectiveOperation, forward_collective_to_parent};
pub use control_flow::{DefactorizableOperation, DefactorizedOperation};
pub use custom_derivatives::{
    CustomJvp, CustomJvpOperation, CustomVjp, CustomVjpCallOperation, CustomVjpOperation, CustomVjpResidual,
    custom_jvp, custom_vjp,
};
pub use dot::{
    Dot, DotDimensionNumbers, DotOperation, DotOps, LeftDot, LeftDotOperation, MaybeDot, RightDot, RightDotOperation,
    adjoint_dimensions_for_left_dot, adjoint_dimensions_for_right_dot, dot_general_evaluate, lhs_result_axes,
    lift_dot_dimensions, lift_left_dot_dimensions, lift_right_dot_dimensions, rhs_result_axes,
};
pub use memory::{TRANSFER_TO_MEMORY_OPERATION_NAME, TransferToMemory, TransferToMemoryOperation};
pub use primitive::{ArrayOperation, LinearArrayOperation};
pub use recompute::RecomputeOperation;
pub use reduce::{Reduce, ReduceOperation, ReductionKind, lift_reduce_axes, reduce_abstract, reduce_evaluate};
pub use reshape::{ReshapeOps, ReshapeValue, lift_reshape_shapes};
pub use select::LinearSelectOperation;
pub use transpose::lift_permutation;
