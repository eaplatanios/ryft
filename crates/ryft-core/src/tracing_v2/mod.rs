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
/// Backend token interfaces for metadata-driven value synthesis and active carrier selection.
pub mod engines;
/// Forward-mode automatic differentiation over paired primal/tangent leaves.
pub mod forward;
/// Symbolic tracing entry points that capture staged programs from Rust closures.
pub mod jit;
/// Linearization, transposition, dense Jacobians, and reverse-mode APIs over staged linear programs.
pub mod linear;
/// Semantic operation traits, built-in carriers, and custom-primitive extension points.
///
/// Per-op staging stays on small operation-local capability traits rather than on catch-all
/// `Supports*` bundles.
pub mod operations;
#[cfg(test)]
/// Test-only helpers shared by `tracing_v2` unit tests.
pub(crate) mod test_support;

pub use batching::{ArrayBatch, BatchableOperation, BatchingError, interpret_batched_program, vmap};
pub use differentiation::DifferentiationError;
pub use engines::{
    DifferentiableEngine, DifferentiableStagingEngine, DifferentiationStagingEngine, Engine, StagingEngine,
};
pub use forward::{Differentiable, JvpContext, JvpTracer, jvp};
pub use jit::{Tracer, TracerState, TracingEngine, interpret_and_trace, trace};
pub use linear::{
    CoordinateValue, DenseJacobian, RematerializationPolicy, compile_grad, compile_grad_with_policy, grad,
    grad_with_aux, hessian, jacfwd, jacrev, jvp_program, value_and_grad, value_and_grad_with_aux, vjp,
};
pub use operations::matrix::{MatrixOps, MatrixValue};
pub use operations::rematerialize::rematerialize;
pub use operations::reshape::{ReshapeOps, ReshapeValue};
pub use operations::{
    ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, Cos, CustomOperationError,
    CustomPrimitive, CustomPrimitiveExtensions, DifferentiableOperation, FlatProgram, LinearCustomPrimitive,
    LinearOperation, LinearPrimitiveOperation, PrimitiveOperation, Sin, WhileOperation,
};
