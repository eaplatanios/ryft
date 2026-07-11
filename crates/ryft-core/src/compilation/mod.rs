//! Backend-neutral staged compilation.
//!
//! Compilation follows an explicit lifecycle:
//!
//! 1. [`stage`] traces a Rust closure into a typed [`StagedFunction`] and retains runtime captures separately from the
//!    source program.
//! 2. [`StagedFunction::lower`] asks the [`CompilationDomain`] to produce a backend-owned [`LoweredFunction`].
//! 3. [`LoweredFunction::compile`] reuses the domain's [`CompilationContext`] and returns a [`CompiledFunction`].
//! 4. [`CompiledFunction::call`] validates runtime signatures and executes the shared backend artifact.
//! 5. [`CompiledFunction::into_executable`] can discard transform metadata and retain only an [`ExecutableFunction`]
//!    suitable for runtime deployment and conditional cross-thread sharing.
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
pub mod exchange;
pub mod function;
pub mod options;

pub use captures::{CaptureReference, CapturingContext, ClosedProgram};
pub use context::{
    CompilationCacheLevel, CompilationCacheOutcome, CompilationCacheStatistics, CompilationContext, CompilationEvent,
    CompilationMissReason,
};
pub use disk_cache::DiskCache;
pub use domain::{AnalyzableCompilationDomain, CompilationDomain};
pub use exchange::{CompilationArtifactExchange, CompilationArtifactExchangePolicy, CompilationExchangeError};
pub use function::{
    CompilationTracer, CompiledFunction, CompiledProgramOperation, ExecutableFunction, FlatCompilationProgram,
    JitCacheCapacities, JitCacheStatistics, JittedFunction, LoweredFunction, Specialization, StagedFunction, compile,
    compile_with_options, jit, jit_with_options, stage, stage_with_capture_references, stage_with_captures, try_jit,
    try_jit_with_options, try_jit_with_options_and_capacities, try_jit_with_options_and_capacity, try_stage,
    try_stage_with_capture_references, try_stage_with_captures,
};
pub use options::CompilationOptions;
