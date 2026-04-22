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

#[cfg(feature = "benchmarking")]
pub(crate) mod benchmark_support;
#[cfg(feature = "benchmarking")]
pub mod benchmarking;
pub mod differentiation;
pub mod engine;
pub mod forward;
pub mod jit;
pub mod linear;
pub mod operations;
pub(crate) mod programs;
#[cfg(test)]
pub(crate) mod test_support;
mod values;

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
    LinearCustomPrimitive, LinearOperation, LinearPrimitiveOperation, PrimitiveOperation, Sin, VectorizableOperation,
};
pub use programs::{
    Atom, AtomId, Instruction, InterpretableOperation, Operation, Program, ProgramBuilder, Traceable, Value,
};
pub use values::{OneLike, ZeroLike};
