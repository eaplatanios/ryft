//! Prototype tracing design for `ryft-core`.
//!
//! The key idea in this version is that staged computation is represented using a shared graph over
//! an open set of operation types:
//!
//! - `Parameterized<P>` lifts and lowers structured inputs and outputs.
//! - `Graph<O, V, In, Out>` is the common staging form.
//! - Each equation stores an operation object, not a tag enum.
//! - Primitive ops carry their own `eval`, `jvp`, `batch`, and transpose rules.
//! - Tracer values carry the transform-local state needed to stage nested traces.

use thiserror::Error;

use crate::parameters::ParameterError;

pub(crate) mod batch;
#[cfg(feature = "benchmarking")]
pub(crate) mod benchmark_support;
#[cfg(feature = "benchmarking")]
pub mod benchmarking;
pub mod forward;
pub mod graph;
pub mod jit;
pub mod linear;
pub mod operations;
pub mod ops;
pub mod program;
#[cfg(test)]
pub(crate) mod test_support;
mod value;

pub use batch::{Batch, stack, unstack, vmap};
pub use forward::{Dual, JvpTracer, TangentSpace, jvp};
pub use graph::{AtomId, AtomSource, Equation, Graph, GraphBuilder};
pub use jit::try_jit;
pub use jit::{CompiledFunction, JitTracer, jit};
pub use linear::{
    CoordinateValue, DenseJacobian, LinearProgram, RematerializationPolicy, compile_grad, compile_grad_with_policy,
    grad, hessian, jacfwd, jacrev, jvp_program, linearize, value_and_grad, vjp,
};
pub use linear::{LinearTerm, Linearized};
pub use operations::matrix::{MatrixOps, MatrixTangentSpace, MatrixValue};
pub use operations::rematerialize::rematerialize;
pub use operations::reshape::{ReshapeOps, ReshapeTangentSpace, ReshapeValue};
pub use ops::{
    CustomPrimitive, CustomPrimitiveExtensions, DifferentiableOp, InterpretableOp, LinearOp, Op, PrimitiveOp,
    VectorizableOp,
};
pub use program::Program;
pub use program::{ProgramBuilder, ProgramOpRef};
pub use value::{ConcreteTraceValue, FloatExt, OneLike, TraceValue, ZeroLike};

/// Error type shared by the prototype tracing transforms.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TraceError {
    /// Structured inputs or outputs did not have the same `Parameterized` shape.
    #[error("mismatched parameter structures")]
    MismatchedParameterStructure,

    /// A batching transform encountered zero lanes and therefore could not infer a batch size.
    #[error("encountered an empty batch")]
    EmptyBatch,

    /// A transform needed a seed value but the parameterized value contained no leaves.
    #[error("encountered an empty parameterized value while a seed value was required")]
    EmptyParameterizedValue,

    /// Different batched leaves disagreed on the number of lanes they carried.
    #[error("mismatched batch sizes across batched leaves")]
    MismatchedBatchSize,

    /// A primitive or staged graph received the wrong number of inputs.
    #[error("invalid number of inputs; expected {expected} but got {got}")]
    InvalidInputCount { expected: usize, got: usize },

    /// A primitive or staged graph produced the wrong number of outputs.
    #[error("invalid number of outputs; expected {expected} but got {got}")]
    InvalidOutputCount { expected: usize, got: usize },

    /// A staged graph referenced an atom that was never defined.
    #[error("unbound atom ID: {id}")]
    UnboundAtomId { id: usize },

    /// Abstract evaluation detected incompatible operand metadata for a primitive application.
    #[error("incompatible abstract values while tracing operation '{op}'")]
    IncompatibleAbstractValues { op: &'static str },

    /// A custom primitive was used by a transform without registering the required rule.
    #[error("custom primitive '{op}' does not provide a '{transform}' rule")]
    MissingCustomRule { op: &'static str, transform: &'static str },

    /// An internal tracing invariant was violated while constructing or replaying a program.
    #[error("{0}")]
    InternalInvariantViolation(&'static str),

    /// A higher-order traced operation failed while deriving or replaying its internal program.
    #[error("higher-order op '{op}' failed: {message}")]
    HigherOrderOpFailure { op: &'static str, message: String },

    /// Wrapper around parameter-lifting failures from the `Parameterized` infrastructure.
    #[error(transparent)]
    Parameter(#[from] ParameterError),
}
