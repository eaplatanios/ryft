// TODO(eaplatanios): Review this module.

//! Contains backend-neutral machinery for _Just-In-Time (JIT) compilation_.
//!
//! This module supplies the common JIT lifecycle shared by compiler backends. Core traces typed Ryft programs, manages
//! runtime captures and structured signatures, coordinates specialization and executable caches, and validates calls.
//! A backend supplies lowering, executable compilation, target-specific options, exact cache identity, serialization,
//! and execution through [`CompilationDomain`]. Backend representations therefore remain outside `ryft-core` while
//! every backend follows the same lifecycle. The following diagram illustrates that lifecycle:
//!
//! ```text
//! ┌───────────────────┐
//! │    Rust Closure   │
//! └─────────┬─────────┘
//!           │ trace
//!           ▼
//! ┌───────────────────┐   embed in an outer trace   ┌───────────────────────┐
//! │  Staged Function  │ ──────────────────────────▶ │ Nested Call Operation │
//! └─────────┬─────────┘                             └───────────────────────┘
//!           │ lower
//!           ▼
//! ┌───────────────────┐
//! │  Lowered Function │
//! └─────────┬─────────┘
//!           │ compile
//!           ▼
//! ┌───────────────────┐   discard transform metadata   ┌─────────────────────┐
//! │ Compiled Function │ ─────────────────────────────▶ │ Executable Function │
//! └─────────┬─────────┘                                └─────────────────────┘
//!           │ call
//!           ▼
//! ┌───────────────────┐
//! │   Runtime Values  │
//! └───────────────────┘
//! ```
//!
//! # Entry Points
//!
//! - Use [`jit`] or [`try_jit`] for a retained JIT dispatcher. A [`JittedFunction`] specializes on explicit static
//!   host parameters, the dynamic parameter structure, and prepared abstract input types. Its first call for a
//!   specialization traces, lowers, and requests compilation; warm calls dispatch directly to the executable.
//! - Use [`compile`] or [`compile_with_options`] when the abstract input signature is already known and one compiled
//!   specialization is sufficient. These functions perform staging, lowering, and compilation in sequence.
//! - Use [`stage`] or [`try_stage`] when source IR must be inspected, transformed, embedded in an outer trace, or
//!   lowered later, and continue explicitly with [`StagedFunction::lower`] and [`LoweredFunction::compile`].
//!
//! Backend-facing crates normally wrap these generic entry points with value- and option-specific APIs. New
//! backend-neutral code can call them directly.
//!
//! # Lifecycle Handles
//!
//! 1. [`StagedFunction`] owns a typed [`ClosedProgram`], concrete runtime captures, public input and output
//!    signatures, output structure, and the domain. It contains no backend lowering or executable, and its
//!    [`StagedFunction::call`] method stages a nested call into an active context rather than executing at runtime.
//! 2. [`LoweredFunction`] owns the backend's [`CompilationDomain::LoweredProgram`], the source handle, and the
//!    options used to lower it. Compilation computes the domain's exact key and consults its [`CompilationContext`].
//! 3. [`CompiledFunction`] combines the executable with the staged and lowered metadata required for inspection and
//!    transformations, and [`CompiledFunction::call`] performs runtime execution.
//! 4. [`ExecutableFunction`] retains only the executable, domain, captures, signatures, and output structure. Use
//!    [`CompiledFunction::into_executable`] when transform metadata is no longer needed; this runtime-only handle
//!    gains `Send` and `Sync` structurally whenever its backend fields do.
//!
//! # Captures and Nested Calls
//!
//! Runtime values closed over by a traced closure are not embedded as literal data. A [`ClosedProgram`] keeps them in
//! a side table while the source IR stores typed [`CaptureReference`] indices. Before lowering, captures are opened
//! as leading flat inputs, and execution supplies arguments in the same `[captures..., public inputs...]` order. This
//! keeps IR compact, preserves device-resident buffers, and lets compilation depend on capture types rather than
//! data.
//!
//! [`CapturingContext`] lets ordinary and transform contexts register captures through their parent. To embed a
//! staged function in a larger trace, an operation family implements [`CompiledProgramOperation`]; that operation
//! then owns how the nested call behaves under lowering, batching, differentiation, partial evaluation, and
//! interpretation.
//!
//! # Specialization and Caching
//!
//! [`JittedFunction`] has separate bounded LRU caches for traced, lowered, and compiled specializations. This
//! frontend cache avoids repeating lifecycle stages for one closure. [`CompilationContext`] is the backend-artifact
//! cache below it and is shared by every compilation using the same domain handle:
//!
//! ```text
//! Compilation Request
//!          │
//!          ├── in-memory LRU hit ───────────────────────────────────────────────────────────────▶ executable
//!          ├── same key in flight ──────────────▶ wait for producer ────────────────────────────▶ executable
//!          ├── persistent executable hit ───────▶ deserialize ──────────────────────────────────▶ executable
//!          ├── distributed artifact available ──▶ deserialize ──────────────────────────────────▶ executable
//!          └── backend compile ─────────────────▶ persist / publish, then insert into the LRU ──▶ executable
//! ```
//!
//! Same-key misses are single-flight while different keys may compile concurrently. [`DiskCache`] provides optional
//! checksummed persistent storage, and [`CompilationArtifactExchange`] optionally shares serialized artifacts among
//! processes according to [`CompilationArtifactExchangePolicy`]. Cache statistics and structured
//! [`CompilationEvent`]s report activity across tracing, lowering, dispatch, persistence, exchange, compilation, and
//! execution.
//!
//! # Extending Compilation
//!
//! Implement [`CompilationDomain`] after defining the backend's [`Domain`](crate::contexts::Domain). The required
//! contract supplies backend-owned lowered and compiled program types, options, an error, a complete compilation key,
//! lowering, compilation, output signatures, and flat execution. Key equality must mean that compiled artifacts are
//! interchangeable and must include every compile-relevant program, option, compiler, target, and topology property.
//!
//! Optional hooks support signature-affecting options, persistent keying and executable codecs, a shared
//! [`CompilationContext`], broader runtime type compatibility, and safe executable replacement. Implement
//! [`AnalyzableCompilationDomain`] to expose cost or memory analysis without recompilation, and implement
//! [`CompiledProgramOperation`] on the operation family when staged functions must compose inside other traces.
//!
//! # Reading order
//!
//! 1. Start with [`CompilationDomain`] for the core/backend ownership boundary.
//! 2. Read [`JittedFunction`], [`StagedFunction`], [`LoweredFunction`], [`CompiledFunction`], and
//!    [`ExecutableFunction`] for the public lifecycle.
//! 3. Read [`ClosedProgram`] and [`CaptureReference`] for capture handling.
//! 4. Read [`CompilationContext`] for single-flight compilation and cache tiers.
//! 5. Read [`DiskCache`] and [`CompilationArtifactExchange`] only when adding persistence or distributed sharing.
//! 6. Continue with a backend crate such as `ryft-xla` for a concrete [`CompilationDomain`] implementation.

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
