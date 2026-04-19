//! Composable tracing, staging, and autodiff infrastructure for `ryft-core`.
//!
//! `tracing_v2` is the shared execution model behind the library's higher-order transforms. User
//! code starts as ordinary Rust over structured values, and the modules here decide whether that
//! code should run eagerly on concrete leaves, replay symbolically on [`Tracer`] leaves to capture
//! a staged [`Program`], or replay on richer traced values such as [`JvpTracer`] and
//! [`LinearTerm`] to derive tangent and cotangent programs.
//!
//! The central design choice is that staged equations store operation objects rather than a single
//! global opcode enum. That keeps the operation universe open: the default core pipeline uses
//! [`PrimitiveOp`] and [`LinearPrimitiveOp`], while backend crates can contribute their own closed
//! carriers through [`Engine::TracingOperation`] and [`Engine::LinearOperation`] without rewriting
//! the tracing and differentiation logic.
//!
//! # Big Picture
//!
//! A typical `tracing_v2` flow looks like this:
//!
//! 1. A transform such as [`interpret_and_trace`], [`jvp`], [`vjp`], or [`vmap`] receives a Rust
//!    closure.
//! 2. The transform chooses a leaf regime: concrete values, [`Tracer`] values, [`Batch`] values,
//!    [`JvpTracer`] values, or staged linear terms.
//! 3. Primitive trait impls in [`operations`] either execute eagerly or record equations into a
//!    [`ProgramBuilder`].
//! 4. The resulting [`Program`] or [`LinearProgram`] is simplified, replayed, transposed, or
//!    handed off to a backend-specific lowering layer.
//!
//! This is what makes JIT tracing, batching, and autodiff feel like variations on one staged IR
//! instead of separate subsystems.
//!
//! # Module Layout
//!
//! - [`operations`] defines the semantic primitive traits and the built-in operation carriers.
//! - The internal programs module owns the staged IR itself: atoms, equations, builders, and
//!   executable programs.
//! - The internal values module defines the leaf-level contracts ([`Traceable`], [`Value`],
//!   [`ZeroLike`], [`OneLike`]) that let the same transform code work over concrete values and
//!   tracer wrappers.
//! - [`engine`] defines the backend token that selects op carriers and synthesizes representative
//!   values from abstract metadata.
//! - [`jit`](self::jit) captures ordinary staged programs from traced execution.
//! - [`forward`] layers forward-mode differentiation on top of the same staging model.
//! - [`linear`] turns staged primal programs into linear maps, pullbacks, dense Jacobians, and
//!   compiled gradients.
//! - [`Batch`] together with [`stack`], [`unstack`], and [`vmap`] provides the explicit batching
//!   surface.
//!
//! # Role In The Library
//!
//! `tracing_v2` is the bridge between user-facing math code and backend-specific execution. The
//! core crate owns tracing semantics, staged-program manipulation, and transform construction;
//! backend crates reuse that machinery to decide how captured programs are represented and lowered.

use thiserror::Error;

use crate::parameters::ParameterError;

pub(crate) mod batch;
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

pub use batch::{Batch, stack, unstack, vmap};
pub use engine::Engine;
pub use forward::{Dual, JvpTracer, TangentSpace, jvp};
pub use jit::{Tracer, interpret_and_trace, trace};
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
pub(crate) use programs::is_identity_one;
pub use programs::{
    Atom, AtomId, Equation, LinearProgramBuilder, LinearProgramOpRef, Program, ProgramBuilder, ProgramOpRef,
};
pub use values::{OneLike, Traceable, Value, ZeroLike};

/// Error type shared by the `tracing_v2` staging and transform pipeline.
///
/// [`TracingError`] intentionally spans the whole subsystem: primitive abstract evaluation, staged
/// program construction, batching, higher-order transform synthesis, and program replay. Keeping
/// the error vocabulary in one place lets the public transform APIs stay small while still
/// preserving the failure modes that matter when debugging tracing behavior.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TracingError {
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

    /// A primitive or staged program received the wrong number of inputs.
    #[error("invalid number of inputs; expected {expected} but got {got}")]
    InvalidInputCount { expected: usize, got: usize },

    /// A primitive or staged program produced the wrong number of outputs.
    #[error("invalid number of outputs; expected {expected} but got {got}")]
    InvalidOutputCount { expected: usize, got: usize },

    /// A staged program referenced an atom that was never defined.
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
