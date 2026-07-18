#[cfg(feature = "benchmarking")]
/// Internal benchmark-case definitions that stay within the plain `tracing_v2` staged IR.
pub(crate) mod benchmark_support;
#[cfg(feature = "benchmarking")]
/// IR benchmarking utilities that emit raw artifacts and normalized summaries for comparison.
pub mod benchmarking;
/// Linearization, transposition, dense Jacobians, and reverse-mode APIs over staged linear programs.
pub mod linear;
/// Semantic operation traits and built-in operation enums.
///
/// Per-op staging stays on small operation-local capability traits rather than on catch-all
/// `Supports*` bundles.
pub mod operations;
pub mod rematerialization;
#[cfg(test)]
pub(crate) mod unroll;

#[cfg(test)]
pub(crate) mod test_util;

pub use crate::differentiation::{
    ForwardModeDifferentiate, ReverseModeDifferentiate, gradient, gradient_holomorphic, gradient_holomorphic_with_aux,
    gradient_with_aux, jvp, linearize, value_and_gradient, value_and_gradient_holomorphic,
    value_and_gradient_holomorphic_with_aux, value_and_gradient_with_aux, vjp,
};
pub use crate::operations::math::{Cos, Sin};
pub use crate::operations::tag::{TAG_OPERATION_NAME, Tag, TagOperation};
pub use crate::tracing::NestedTracer;
pub use linear::{
    DenseDifferentiableType, DenseDifferentiate, Hessian, HessianBlock, Jacobian, JacobianBlock, hessian,
    hessian_holomorphic, hessian_holomorphic_with_aux, hessian_with_aux, jacfwd, jacfwd_holomorphic,
    jacfwd_holomorphic_with_aux, jacfwd_with_aux, jacrev, jacrev_holomorphic, jacrev_holomorphic_with_aux,
    jacrev_with_aux,
};
pub use operations::RecomputeOperation;
pub use operations::collective::{
    AXIS_INDEX_OPERATION_NAME, AxisIndexOperation, Collective, CollectiveKind, CollectiveOperation,
    forward_collective_to_parent,
};
pub use operations::custom_derivatives::transpose_primal_custom_vjp;
pub use operations::dot::{Dot, DotDimensionNumbers, DotOperation, DotOps};
pub use rematerialization::{
    DotsSaveable, DotsWithNoBatchDimsSaveable, EitherStorage, EverythingSaveable, MemoryTransferStorage, NoStorage,
    NothingSaveable, OffloadDotsWithNoBatchDims, PolicyFn, RematerializationCandidate, RematerializationDecision,
    RematerializationError, RematerializationPolicy, RematerializationProducer, RematerializationRejection,
    RematerializationRejectionKind, Rematerialize, RematerializeOperation, ResidualStorage,
    SaveAndOffloadOnlyTheseNames, SaveAnyNamesButThese, SaveAnythingExceptTheseNames, SaveFromBothPolicies,
    SaveOnlyTheseNames, rematerialize,
};
