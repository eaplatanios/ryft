//! Composable tracing, staging, and autodiff infrastructure for `ryft-core`.
//!
//! `tracing_v2` is the shared execution model behind the library's higher-order transforms. User
//! code starts as ordinary Rust over structured values, and the modules here decide whether that
//! code should run eagerly on concrete leaves, replay symbolically on [`Tracer`] leaves to capture
//! a staged [`Program`], or replay on richer traced values such as [`JvpTracer`] and
//! [`LinearTerm`] to derive tangent and cotangent programs.
//!
//! The central design choice is that staged instructions store operation objects rather than a single
//! global opcode enum. That keeps the operation universe open: the default core pipeline uses
//! [`PrimitiveOperation`] and [`LinearPrimitiveOperation`], while backend crates can contribute their own closed
//! carriers through [`Engine::TracingOperation`] and [`Engine::LinearOperation`] without rewriting
//! the tracing and differentiation logic.
//!
//! # Big Picture
//!
//! A typical `tracing_v2` flow looks like this:
//!
//! 1. A transform such as [`interpret_and_trace`], [`jvp`], [`vjp`], or
//!    [`crate::batching::vmap`] receives a Rust
//!    closure.
//! 2. The transform chooses a leaf regime: concrete values, [`Tracer`] values,
//!    [`crate::batching::Batch`] values, [`JvpTracer`] values, or staged linear terms.
//! 3. Primitive trait impls in [`operations`] either execute eagerly or record instructions into a
//!    [`ProgramBuilder`].
//! 4. The resulting [`Program`] is simplified, replayed, transposed, or
//!    handed off to a backend-specific lowering layer.
//!
//! This is what makes JIT tracing, batching, and autodiff feel like variations on one staged IR
//! instead of separate subsystems.
//!
//! # Module Layout
//!
//! - [`operations`] defines the semantic primitive traits and the built-in operation carriers.
//! - The internal programs module owns the staged IR itself and the core leaf contracts
//!   ([`Traceable`], [`Value`]): atoms, instructions, builders, executable programs, and the traits
//!   that tie leaf values to them.
//! - The internal values module defines the remaining value-level identity helpers
//!   ([`ZeroLike`], [`OneLike`]) and the built-in scalar leaf impls.
//! - [`engine`] defines the backend token that selects op carriers and synthesizes representative
//!   values from abstract metadata.
//! - [`jit`](self::jit) captures ordinary staged programs from traced execution.
//! - [`forward`] layers forward-mode differentiation on top of the same staging model.
//! - [`linear`] turns staged primal programs into linear maps, pullbacks, dense Jacobians, and
//!   compiled gradients.
//! - [`crate::batching`] owns the explicit batching surface and traced `vmap` operation types.
//!
//! # Role In The Library
//!
//! `tracing_v2` is the bridge between user-facing math code and backend-specific execution. The
//! core crate owns tracing semantics, staged-program manipulation, and transform construction;
//! backend crates reuse that machinery to decide how captured programs are represented and lowered.

use thiserror::Error;

use crate::batching::BatchingError;
use crate::parameters::ParameterError;
use crate::types::TypeError;

#[cfg(feature = "benchmarking")]
pub(crate) mod benchmark_support;
#[cfg(feature = "benchmarking")]
pub mod benchmarking;
pub mod engine;
pub mod forward;
pub mod jit;
pub mod linear;
pub mod operations;
pub(crate) mod programs;
#[cfg(test)]
pub(crate) mod test_support;
mod values;

pub use engine::Engine;
pub use forward::{Dual, JvpTracer, TangentSpace, jvp};
pub use jit::{Tracer, interpret_and_trace, trace};
pub use linear::{
    CoordinateValue, DenseJacobian, RematerializationPolicy, compile_grad, compile_grad_with_policy, grad, hessian,
    jacfwd, jacrev, jvp_program, value_and_grad, vjp,
};
pub use linear::{LinearTerm, Linearized};
pub use operations::matrix::{MatrixOps, MatrixTangentSpace, MatrixValue};
pub use operations::rematerialize::rematerialize;
pub use operations::reshape::{ReshapeOps, ReshapeTangentSpace, ReshapeValue};
pub use operations::{
    Cos, CustomPrimitive, CustomPrimitiveExtensions, DifferentiableOperation, LinearCustomPrimitive, LinearOperation,
    LinearPrimitiveOperation, PrimitiveOperation, Sin, VectorizableOperation,
};
pub use programs::{
    Atom, AtomId, Instruction, InterpretableOperation, Operation, Program, ProgramBuilder, Traceable, Value,
};
pub use values::{OneLike, ZeroLike};

/// Error type shared by the `tracing_v2` staging and transform pipeline.
///
/// [`TracingError`] intentionally spans the tracing subsystem: primitive abstract evaluation,
/// staged program construction, higher-order transform synthesis, and program replay. The
/// batching-specific failures now live in [`BatchingError`] and are wrapped here when batching
/// participates inside a tracing flow.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TracingError {
    /// Structured inputs or outputs did not have the same `Parameterized` shape.
    #[error("mismatched parameter structures")]
    MismatchedParameterStructure,

    /// A primitive or staged program received the wrong number of inputs.
    #[error("invalid number of inputs; expected {expected} but got {got}")]
    InvalidInputCount { expected: usize, got: usize },

    /// A primitive or staged program produced the wrong number of outputs.
    #[error("invalid number of outputs; expected {expected} but got {got}")]
    InvalidOutputCount { expected: usize, got: usize },

    /// A staged program referenced an atom that was never defined.
    #[error("unbound atom ID: {id}")]
    UnboundAtomId { id: AtomId },

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

    /// Wrapper around abstract type-level reasoning failures.
    #[error(transparent)]
    Type(#[from] TypeError),

    /// Wrapper around batching- and vmapping-specific failures.
    #[error(transparent)]
    Batching(#[from] BatchingError),
}
