#[cfg(feature = "benchmarking")]
/// Internal benchmark-case definitions that stay within the plain `tracing_v2` staged IR.
pub(crate) mod benchmark_support;
#[cfg(feature = "benchmarking")]
/// IR benchmarking utilities that emit raw artifacts and normalized summaries for comparison.
pub mod benchmarking;
/// Errors raised while materializing dense Jacobian- and Hessian-style differentiation results.
pub mod differentiation;
/// Backend token interface for metadata-driven value synthesis and carrier selection.
pub mod engine;
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

pub use differentiation::DifferentiationError;
pub use engine::Engine;
pub use forward::{Dual, JvpTracer, TangentSpace, jvp};
pub use jit::{Tracer, TracerState, interpret_and_trace, trace};
pub use linear::{
    CoordinateValue, DenseJacobian, RematerializationPolicy, compile_grad, compile_grad_with_policy, grad, hessian,
    jacfwd, jacrev, jvp_program, value_and_grad, vjp,
};
pub use linear::{LinearTerm, Linearized};
pub use operations::matrix::{MatrixOps, MatrixTangentSpace, MatrixValue};
pub use operations::rematerialize::rematerialize;
pub use operations::reshape::{ReshapeOps, ReshapeTangentSpace, ReshapeValue};
pub use operations::{
    Cos, CustomOperationError, CustomPrimitive, CustomPrimitiveExtensions, DifferentiableOperation,
    LinearCustomPrimitive, LinearOperation, LinearPrimitiveOperation, PrimitiveOperation, Sin,
};
