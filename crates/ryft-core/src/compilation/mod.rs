//! Backend-neutral staged compilation.
//!
//! Compilation follows an explicit lifecycle:
//!
//! 1. [`stage`] traces a Rust closure into a typed [`StagedFunction`] and retains runtime captures separately from the
//!    source program.
//! 2. [`StagedFunction::lower`] asks the [`CompilationDomain`] to produce a backend-owned [`LoweredFunction`].
//! 3. [`LoweredFunction::compile`] reuses the domain's [`CompilationContext`] and returns a [`CompiledFunction`].
//! 4. [`CompiledFunction::call`] validates runtime signatures and executes the shared backend artifact.
//!
//! The convenience [`compile`] and [`compile_with_options`] entry points perform the first three stages in sequence.
//! Retaining an intermediate handle makes staging and lowering reusable and inspectable without introducing compiler-
//! specific representations into `ryft-core`.
//!
//! Backends own lowering, executable compilation, target options, cache identity, serialization, and execution through
//! [`CompilationDomain`]. Runtime values captured by a closure live in a [`ClosedProgram`] side table and appear in the
//! source IR only as typed [`CaptureReference`]s. Nested compiled calls remain transformable through the operation-owned
//! [`CompiledProgramOperation`] rule.

pub mod captures;
pub mod context;
pub mod disk_cache;
pub mod domain;
pub mod function;
pub mod options;

pub use captures::{CaptureReference, CapturingContext, ClosedProgram};
pub use context::{CompilationCacheStatistics, CompilationContext};
pub use disk_cache::DiskCache;
pub use domain::CompilationDomain;
pub use function::{
    CompilationTracer, CompiledFunction, CompiledProgramOperation, FlatCompilationProgram, LoweredFunction,
    StagedFunction, compile, compile_with_options, stage, stage_with_capture_references, stage_with_captures,
    try_stage, try_stage_with_capture_references, try_stage_with_captures,
};
pub use options::CompilationOptions;
