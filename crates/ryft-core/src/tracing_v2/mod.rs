/// Explicit batching and `batch` support for staged programs.
pub mod batching;
#[cfg(feature = "benchmarking")]
/// Internal benchmark-case definitions that stay within the plain `tracing_v2` staged IR.
pub(crate) mod benchmark_support;
#[cfg(feature = "benchmarking")]
/// IR benchmarking utilities that emit raw artifacts and normalized summaries for comparison.
pub mod benchmarking;
/// Symbolic linearization core and the forward-mode automatic-differentiation transforms built on it.
///
/// The heart of this module is [`linearize_program`](differentiation::linearize_program), which turns a staged
/// primal [`Program`](crate::programs::Program) into a [`Linearization`](differentiation::Linearization): a
/// residual-extended primal program paired with a residualized pushforward program, both kept symbolic so the
/// artifact can be interpreted eagerly, spliced into an enclosing trace, or embedded as program data inside
/// higher-order operations. The value-level entry points on
/// [`DifferentiationContext`](differentiation::DifferentiationContext) — `linearize`, `jvp`, `vjp`,
/// `value_and_gradient` — are sugar that traces the user closure into a primal program and then linearizes or
/// replays it, so whether a transform runs eagerly or stages a program is decided by the context's value type
/// rather than by a mode flag.
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

#[cfg(test)]
pub(crate) mod test_util;

pub use crate::operations::trigonometric::{Cos, Sin};
pub use batching::{
    ArrayBatch, Batch, BatchAxes, BatchAxis, BatchContext, BatchableOperation, BatchableProgramOperation,
    BatchingContext, BatchingTracer, ProgramBatchingContext, ProgramBatchingOutputAxes, batch, batch_program,
};
pub use differentiation::{
    CaptureParameterizedOperation, DifferentiableOperation, DifferentiationContext, DifferentiationError,
    DirectLinearOperationOf, JvpTracer, LinearOperationOf, LinearizableProgramOperation, Linearization,
    LinearizationTracer, NestedLinearization, PrimalTracingContext, Pushforward, ResidualizedOperation, TangentContext,
    ZeroTangentOperation, linearize_program,
};
pub use linear::{
    CoordinateValue, DifferentiableDomainExtension, Differential, DifferentialBlock, DifferentialRow, Hessian,
    Jacobian, grad, grad_with_aux, jacrev, value_and_grad, value_and_grad_with_aux,
};
pub use operations::captures::{MaterializeCaptureOperation, ValueOrCapture};
pub use operations::collective::{Collective, CollectiveKind, CollectiveOperation, forward_collective_to_parent};
pub use operations::control_flow::{DefactorizableProgramOperation, DefactorizedOperation};
pub use operations::dot::{
    Dot, DotDimensionNumbers, DotOperation, DotOps, LeftDot, LeftDotOperation, RightDot, RightDotOperation,
};
pub use operations::reshape::{ReshapeOps, ReshapeValue};
pub use operations::select::LinearSelectOperation;
pub use operations::{ArrayOperation, LinearArrayOperation, RecomputeOperation};
pub use rematerialization::{
    MaybeRematerializationName, OffloadingRematerializationPolicy, REMATERIALIZATION_NAME_OPERATION_NAME,
    RematerializationName, RematerializationNameOperation, RematerializationPolicy, RematerializationVerdict,
    Rematerialize, ResidualHandling, rematerialize,
};
