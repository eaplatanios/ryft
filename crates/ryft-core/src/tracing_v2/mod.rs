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
/// Semantic operation traits, built-in carriers, and custom-primitive extension points.
///
/// Per-op staging stays on small operation-local capability traits rather than on catch-all
/// `Supports*` bundles.
pub mod operations;
#[cfg(test)]
pub(crate) mod test_util;
pub use batching::{ArrayBatch, BatchableOperation, BatchingError, interpret_batched_program, vmap};
pub use differentiation::{
    Differentiable, DifferentiableEngine, DifferentiableOperation, DifferentiableOperationTracingEngine,
    DifferentiableTracingEngine, DifferentiationError, JvpContext, JvpTracer, Tangent,
};
pub use linear::{
    ConcreteValueAndGrad, CoordinateValue, DenseJacobian, TracedValueAndGrad, ValueAndGradDispatch, grad_with_aux,
    jacrev, linearize, value_and_grad, value_and_grad_with_aux, vjp,
};
pub use operations::matrix::{MatrixOps, MatrixValue};
pub use operations::reshape::{ReshapeOps, ReshapeValue};
pub use operations::{
    ArrayOperation, ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, Cos,
    CustomOperationError, CustomPrimitive, CustomPrimitiveExtensions, FlatProgram, LinearArrayOperation,
    LinearCustomPrimitive, Sin, WhileOperation,
};
