/// Explicit batching and `vmap` support for staged programs.
pub mod batching;
#[cfg(feature = "benchmarking")]
/// Internal benchmark-case definitions that stay within the plain `tracing_v2` staged IR.
pub(crate) mod benchmark_support;
#[cfg(feature = "benchmarking")]
/// IR benchmarking utilities that emit raw artifacts and normalized summaries for comparison.
pub mod benchmarking;
/// Errors raised while materializing dense Jacobian- and Hessian-style differentiation results.
pub mod differentiation;
/// Forward-mode automatic differentiation over paired primal/tangent leaves.
pub mod forward;
/// Linearization, transposition, dense Jacobians, and reverse-mode APIs over staged linear programs.
pub mod linear;
/// Semantic operation traits and built-in carriers.
///
/// Per-op staging stays on small operation-local capability traits rather than on catch-all
/// `Supports*` bundles.
pub mod operations;
#[cfg(test)]
pub(crate) mod test_util;
pub use crate::operations::trigonometric::{Cos, Sin};
pub use batching::{
    ArrayBatch, Batchable, BatchableOperation, BatchingDomain, BatchingError, Vmap, VmapDispatch,
    VmapDispatchTracerMarker, VmapDispatchValueMarker, align_batch_axis, apply_elementwise_batch, apply_with_axes,
    broadcast_to_batched, lift_elementwise, vmap,
};
pub use differentiation::{
    DifferentiableDomain, DifferentiableOperation, DifferentiableTracingDomain, DifferentiationError, JvpContext,
    JvpTracer, LinearOperationCarrier, LinearOperationCarrierFamily, LinearOperationExtensionFamily, LinearValue,
    LinearizableDomain, LinearizationContext, LinearizationTracer,
};
pub use linear::{
    CoordinateValue, DifferentiableDomainExtension, Differential, DifferentialBlock, DifferentialRow, Hessian,
    Jacobian, grad_with_aux, jacrev, value_and_grad, value_and_grad_with_aux,
};
pub use operations::collective::{
    Collective, CollectiveKind, CollectiveOperation, MaybeCollective, SupportsCollective,
};
pub use operations::dot::{
    Dot, DotDimensionNumbers, DotOperation, LeftDot, LeftDotOperation, RightDot, RightDotOperation, SupportsDot,
    SupportsLeftDot, SupportsRightDot,
};
pub use operations::matrix::DotOps;
pub use operations::reshape::{Reshape, ReshapeOps, ReshapeValue};
pub use operations::select::{Select, SelectOperation, SupportsSelect};
pub use operations::transpose::{SupportsTranspose, Transpose, TransposeOperation};
pub use operations::{
    ArrayOperation, ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, FlatProgram,
    LinearArrayOperation, NoOperationExtension, TracerReplayValue, WhileOperation,
};
