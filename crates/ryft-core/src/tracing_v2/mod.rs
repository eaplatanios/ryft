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
    ForwardModeDifferentiate, Jacobian, JacobianBlock, JacobianDifferentiate, ReverseModeDifferentiate, gradient,
    gradient_holomorphic, gradient_holomorphic_with_aux, gradient_with_aux, jacobian_forward,
    jacobian_forward_holomorphic, jacobian_forward_holomorphic_with_aux, jacobian_forward_with_aux, jacobian_reverse,
    jacobian_reverse_holomorphic, jacobian_reverse_holomorphic_with_aux, jacobian_reverse_with_aux, jvp, linearize,
    value_and_gradient, value_and_gradient_holomorphic, value_and_gradient_holomorphic_with_aux,
    value_and_gradient_with_aux, vjp,
};
pub use crate::operations::math::{Cos, Sin};
pub use crate::operations::tag::{TAG_OPERATION_NAME, Tag, TagOperation};
pub use crate::tracing::NestedTracer;
pub use linear::{
    Hessian, HessianBlock, HessianDifferentiate, hessian, hessian_holomorphic, hessian_holomorphic_with_aux,
    hessian_with_aux,
};
pub use operations::RecomputeOperation;
pub use operations::custom_derivatives::transpose_primal_custom_vjp;
pub use rematerialization::{
    DotsSaveable, DotsWithNoBatchDimsSaveable, EitherStorage, EverythingSaveable, MemoryTransferStorage, NoStorage,
    NothingSaveable, OffloadDotsWithNoBatchDims, PolicyFn, RematerializationCandidate, RematerializationDecision,
    RematerializationError, RematerializationPolicy, RematerializationProducer, RematerializationRejection,
    RematerializationRejectionKind, Rematerialize, RematerializeOperation, ResidualStorage,
    SaveAndOffloadOnlyTheseNames, SaveAnyNamesButThese, SaveAnythingExceptTheseNames, SaveFromBothPolicies,
    SaveOnlyTheseNames, rematerialize,
};
