/// Explicit batching and `batch` support for staged programs.
pub mod batching;
#[cfg(feature = "benchmarking")]
/// Internal benchmark-case definitions that stay within the plain `tracing_v2` staged IR.
pub(crate) mod benchmark_support;
#[cfg(feature = "benchmarking")]
/// IR benchmarking utilities that emit raw artifacts and normalized summaries for comparison.
pub mod benchmarking;
/// Capture-free linearization core and the automatic-differentiation transforms built on it.
///
/// The heart of this module is the JAX-style linearization pipeline: it traces a user closure into a staged primal
/// [`Program`](crate::Program) and builds a single combined JVP program over the ordinary primal operation family —
/// with tangent coefficients carried as plain operand edges rather than symbolic captures — which is then partially
/// evaluated into a known primal sub-program and an unknown linear tangent sub-program
/// ([`Program::linearize`](crate::Program::linearize)), whose tangent half transposes directly in the primal
/// operation family for reverse mode. The value-level entry points on [`ForwardModeDifferentiate`] (`jvp` and
/// `linearize`) and [`ReverseModeDifferentiate`] (`vjp`, `value_and_gradient`, and `gradient`) run the closure
/// directly on differentiation duals, so whether a transform runs eagerly or stages a program is decided by the
/// context's value type rather than by a mode flag.
pub mod differentiation;
#[cfg(test)]
mod forward;
/// Linearization, transposition, dense Jacobians, and reverse-mode APIs over staged linear programs.
pub mod linear;
/// Semantic operation traits and built-in operation enums.
///
/// Per-op staging stays on small operation-local capability traits rather than on catch-all
/// `Supports*` bundles.
pub mod operations;
pub mod rematerialization;
pub(crate) mod unroll;

#[cfg(test)]
pub(crate) mod test_util;

pub use crate::operations::tag::{MaybeTag, TAG_OPERATION_NAME, Tag, TagOperation};
pub use crate::operations::trigonometric::{Cos, Sin};
pub use crate::tracing::NestedTracer;
pub use differentiation::{
    ForwardModeDifferentiate, ReverseModeDifferentiate, gradient, gradient_holomorphic, gradient_holomorphic_with_aux,
    gradient_with_aux, jvp, linearize, value_and_gradient, value_and_gradient_holomorphic,
    value_and_gradient_holomorphic_with_aux, value_and_gradient_with_aux, vjp,
};
pub use linear::{
    CoordinateValue, DifferentiableDomainExtension, Differential, DifferentialBlock, DifferentialRow, Hessian,
    Jacobian, jacrev,
};
pub use operations::collective::{
    AXIS_INDEX_OPERATION_NAME, AxisIndexOperation, Collective, CollectiveKind, CollectiveOperation,
    forward_collective_to_parent,
};
pub use operations::control_flow::transpose_primal_condition;
pub use operations::custom_derivatives::transpose_primal_custom_vjp;
pub use operations::dot::{Dot, DotDimensionNumbers, DotOperation, DotOps};
pub use operations::reshape::{ReshapeOps, ReshapeValue};
pub use operations::scan::transpose_primal_scan;
pub use operations::{ArrayOperation, RecomputeOperation};
pub use rematerialization::{
    OffloadingRematerializationPolicy, RematerializationPolicy, RematerializationVerdict, Rematerialize,
    RematerializeCallOperation, RematerializeOperation, ResidualHandling, rematerialize,
};
