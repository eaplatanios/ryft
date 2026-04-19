//! Composable tracing and staged-program infrastructure for `ryft-core`.
//!
//! Staged computation is represented as a shared [`Graph`] over an open set of operation types.
//! Each equation stores an operation object rather than a tag enum, so backends extend the tracing
//! surface by contributing their own op carrier instead of editing a central dispatch table.
//!
//! # Module layout
//!
//! - `ops::core` — foundational operation traits: [`Op`], [`InterpretableOp`],
//!   [`LinearOperation`], [`DifferentiableOp`], [`VectorizableOp`]. Every op carrier implements
//!   these; transforms consume them.
//! - `ops::staging` — small hidden capability traits (`AddTracingOperation`,
//!   `MatMulLinearOperation`, etc.) that the transforms bound themselves on. Backends implement
//!   one per op they stage.
//! - `ops::primitive` — the built-in [`PrimitiveOp`] / [`LinearPrimitiveOp`] carriers that
//!   provide a ready-to-use op-set for the default tracing pipeline.
//! - `ops::custom` — the [`CustomPrimitive`] / [`LinearCustomPrimitive`] subsystem for
//!   layering user-supplied ops onto any backend carrier without modifying the built-in enums.
//! - [`graph`] and [`program`] — the shared staging container plus the surface [`Program`] /
//!   [`LinearProgram`] types. [`Traceable`] and [`Value`] live alongside them as the leaf-value
//!   traits.
//! - [`engine`] — the [`Engine`] trait backends implement. It pins the concrete op-set via
//!   [`Engine::TracingOperation`] and [`Engine::LinearOperation`] associated types, which is how
//!   op selection is surfaced to transforms rather than via umbrella capability bundles.
//!
//! # Transforms
//!
//! - [`forward`] — forward-mode AD via [`jvp`], producing [`Dual`] tangents and the underlying
//!   [`JvpTracer`].
//! - [`linear`] — linearization, transposition, and reverse-mode AD: [`jvp_program`], [`vjp`],
//!   [`grad`], [`value_and_grad`], plus [`jacrev`] / [`jacfwd`] / [`hessian`] helpers.
//! - `batch` — vectorization via [`vmap`], [`stack`], [`unstack`].
//! - [`jit`](mod@self::jit) — staged-program capture via [`trace_program`] and compilation via
//!   [`jit`](fn@self::jit::jit).

use thiserror::Error;

use crate::parameters::ParameterError;

pub(crate) mod batch;
#[cfg(feature = "benchmarking")]
pub(crate) mod benchmark_support;
#[cfg(feature = "benchmarking")]
pub mod benchmarking;
pub mod engine;
pub mod forward;
pub mod graph;
pub mod jit;
pub mod linear;
pub mod operations;
pub mod program;
#[cfg(test)]
pub(crate) mod test_support;
mod values;

pub use batch::{Batch, stack, unstack, vmap};
pub use engine::Engine;
pub use forward::{Dual, JvpTracer, TangentSpace, jvp};
pub use graph::{Atom, AtomId, Equation, Graph, GraphBuilder};
pub use jit::{
    CompiledFunction, JitTracer, TraceInput, TraceOutput, TypeTracing, jit, jit_from_types, trace_program,
    trace_program_from_types,
};
pub use linear::{
    CoordinateValue, DenseJacobian, LinearProgram, RematerializationPolicy, compile_grad, compile_grad_with_policy,
    grad, hessian, jacfwd, jacrev, jvp_program, value_and_grad, vjp,
};
pub use linear::{LinearTerm, Linearized};
pub use operations::matrix::{MatrixOps, MatrixTangentSpace, MatrixValue};
pub use operations::rematerialize::rematerialize;
pub use operations::reshape::{ReshapeOps, ReshapeTangentSpace, ReshapeValue};
pub use operations::{
    Cos, CustomPrimitive, CustomPrimitiveExtensions, DifferentiableOp, InterpretableOp, LinearCustomPrimitive,
    LinearOperation, LinearPrimitiveOp, Op, PrimitiveOp, Sin, VectorizableOp,
};
pub use program::Program;
pub use program::{LinearProgramBuilder, LinearProgramOpRef, ProgramBuilder, ProgramOpRef};
pub use values::{OneLike, Traceable, Value, ZeroLike};

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
