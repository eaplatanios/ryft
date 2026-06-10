/// Elementwise addition linearization and differentiation rules.
pub mod add;

/// Value-capability bundles aggregating the repeated trait-bound clusters used by the primitive operation enums.
pub mod bounds;

/// N-dimensional broadcast primitive.
pub mod broadcast;

/// Named-axis collective primitives (`psum`, `pmean`, `pmax`).
pub mod collective;

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

/// Elementwise logical operations on Boolean arrays.
pub mod logical;

/// Matrix capability layer shared by matrix staged operations.
pub mod matrix;

/// Elementwise multiplication differentiation rules.
pub mod mul;

/// Elementwise negation.
pub mod neg;

/// Reusable operation types for the built-in operation set and static backend extensions.
pub mod primitive;

/// Axis-collapsing reduction primitive.
pub mod reduce;

/// Reshaping primitive.
pub mod reshape;

/// Factor-payload mapping for the scalar linear operation.
pub mod scalars;

/// Scalar and tensor scaling.
pub mod scale;

/// Per-element select / `where` primitive.
pub mod select;

/// Elementwise sine differentiation rules.
pub mod sin;

/// Gradient-severing `stop_gradient` differentiation rule.
pub mod stop_gradient;

/// Elementwise subtraction linearization and differentiation rules.
pub mod sub;

/// N-dimensional axis-permutation primitive.
pub mod transpose;

pub use bounds::{
    SupportsArithmeticOperations, SupportsComparisonOperations, SupportsConstantOperations,
    SupportsLinearAlgebraOperations, SupportsLinearArithmeticOperations, SupportsLinearArrayOperation,
    SupportsLinearScalarOperation, SupportsManipulationOperations, SupportsTrigonometricOperations,
};
pub use broadcast::{
    Broadcast, BroadcastInDim, BroadcastInDimOperation, BroadcastLike, BroadcastTo, SupportsBroadcastInDim,
    broadcast_in_dim_abstract, broadcast_in_dim_evaluate, lift_broadcast_in_dim,
};
pub use collective::{
    Collective, CollectiveKind, CollectiveOperation, SupportsCollective, forward_collective_to_parent,
};
pub use compare::{Compare, CompareKind, CompareOperation, SupportsCompare};
pub use control_flow::{
    ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, FlatProgram, WhileOperation,
    flat_program_input_types, flat_program_output_types,
};
pub use custom_derivatives::{
    CustomJvp, CustomJvpOperation, CustomVjp, CustomVjpCallOperation, CustomVjpOperation, CustomVjpResidual,
    SupportsCustomJvp, SupportsCustomVjp, SupportsCustomVjpCall, custom_jvp, custom_vjp,
};
pub use dot::{
    Dot, DotDimensionNumbers, DotOperation, LeftDot, LeftDotOperation, MaybeDot, RightDot, RightDotOperation,
    SupportsDot, SupportsLeftDot, SupportsRightDot, adjoint_dimensions_for_left_dot, adjoint_dimensions_for_right_dot,
    dot_general_evaluate, lhs_result_axes, lift_dot_dimensions, lift_left_dot_dimensions, lift_right_dot_dimensions,
    rhs_result_axes,
};
pub use logical::{LogicalBinary, LogicalKind, LogicalNot, LogicalOperation, SupportsLogical};
pub use matrix::DotOps;
pub use primitive::{ArrayOperation, LinearArrayOperation};
pub use reduce::{
    Reduce, ReduceOperation, ReductionKind, SupportsReduce, lift_reduce_axes, reduce_abstract, reduce_evaluate,
};
pub use reshape::{Reshape, ReshapeOperation, ReshapeOps, ReshapeValue, SupportsReshape, lift_reshape_shapes};
pub use select::{Select, SelectOperation, SupportsSelect};
pub use transpose::{
    SupportsTranspose, Transpose, TransposeOperation, inverse_permutation, lift_permutation, transpose_abstract_nd,
    transpose_evaluate, transpose_is_identity,
};
