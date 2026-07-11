//! Contains backend-neutral machinery for _Just-In-Time (JIT) compilation_.
//!
//! This module supplies the common JIT lifecycle shared by compiler backends. Core traces typed Ryft programs, manages
//! runtime captures and structured signatures, coordinates specialization and executable caches, and validates calls.
//! A backend supplies lowering, executable compilation, target-specific options, exact cache identity, serialization,
//! and execution through [`CompilationDomain`]. Backend representations therefore remain outside `ryft-core` while
//! every backend follows the same lifecycle. The following diagram illustrates that lifecycle:
//!
//! ```text
//!                                                  Per-Jitted-Function Caches
//!                        specialization key = static parameters + input paths + runtime-derived abstract types
//!
//! ┌──────────────────────────────┐
//! │         Rust Closure         │
//! └──────────────┬───────────────┘
//!                │ trace
//!                ▼
//! ┌──────────────────────────────┐                           ┌──────────────────────────────┐
//! │ Program                      │ ◀────── store / reuse ───▶ │ Trace Cache                  │
//! │ Staged Function handle       │                           │ retains Programs             │
//! └──────────────┬───────────────┘                           └──────────────────────────────┘
//!                │ lower
//!                ▼
//! ┌──────────────────────────────┐                           ┌──────────────────────────────┐
//! │ Lowered Program              │ ◀────── store / reuse ───▶ │ Lowering Cache               │
//! │ Lowered Function handle      │                           │ retains Lowered Programs     │
//! └──────────────┬───────────────┘                           └──────────────────────────────┘
//!                │ derive the domain cache key
//!                ▼
//! ┌──────────────────────────────┐
//! │ Compilation Context          │
//! │ resolves shared cache tiers  │
//! └──────────────┬───────────────┘
//!                │ restore or compile, then attach metadata
//!                ▼
//! ┌──────────────────────────────┐                           ┌──────────────────────────────┐
//! │ Compiled Program             │ ◀────── store / reuse ───▶ │ Dispatch Cache               │
//! │ Compiled Function handle     │                           │ retains Compiled Programs    │
//! │ shared through an Arc        │                           │                              │
//! └──────────────┬───────────────┘                           └──────────────────────────────┘
//!                │ call
//!                ▼
//! ┌──────────────────────────────┐
//! │ Runtime Values               │
//! └──────────────────────────────┘
//!
//! ┌──────────────────────────────┐                           ┌──────────────────────────────┐
//! │ Program                      │ ─── embed in outer trace ▶ │ Nested Call Operation        │
//! │ Staged Function handle       │                           │                              │
//! └──────────────────────────────┘                           └──────────────────────────────┘
//!
//! ┌──────────────────────────────┐                           ┌──────────────────────────────┐
//! │ Compiled Program             │ ── discard metadata ────▶ │ Executable Program           │
//! │ Compiled Function handle     │                           │                              │
//! └──────────────────────────────┘                           └──────────────────────────────┘
//! ```
//!
//! # Entry Points
//!
//! - Use [`jit`] or [`try_jit`] for a retained JIT dispatcher. A [`JittedFunction`] specializes on explicit static
//!   host parameters, the dynamic parameter structure, and runtime-derived abstract input types. Its first call for a
//!   specialization traces, lowers, and requests compilation; warm calls dispatch directly to the executable.
//! - Use [`stage_function`] for ordinary capture-free staging. Construct a [`CompilationStagingRequest`] and pass it to
//!   [`CompilationDomain::stage`] when fallibility, runtime captures, or symbolic capture references are needed. Then
//!   continue with [`CompilationDomain::lower`] and [`CompilationDomain::compile`].
//!
//! Backend-facing crates normally wrap these generic entry points with value- and option-specific APIs. New
//! backend-neutral code can call them directly.
//!
//! # Lifecycle Handles
//!
//! 1. [`StagedFunction`] owns a typed [`ClosedProgram`], concrete runtime captures, public input and output
//!    signatures, output structure, and compilation options. It contains no backend lowering or executable, and its
//!    [`StagedFunction::call`] method stages a nested call into an active context rather than executing at runtime.
//! 2. [`LoweredFunction`] owns the backend's [`CompilationDomain::LoweredProgram`], the source handle, and the
//!    options used to lower it. Compilation computes the domain's exact key and consults its [`CompilationContext`].
//! 3. [`CompiledFunction`] combines the executable with the staged and lowered metadata required for inspection and
//!    transformations. Runtime execution uses [`CompilationDomain::call`].
//! 4. [`ExecutableProgram`] retains only the executable, captures, signatures, and output structure. Use
//!    [`CompiledFunction::into_executable_program`] when transform metadata is no longer needed; this runtime-only handle
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
//! cache below it and is shared by every compilation using the same domain handle. A JIT call checks the dispatch
//! cache first, then the lowering cache, and finally the trace cache; a hit resumes the lifecycle from the retained
//! artifact shown above, while a miss produces and inserts that artifact. The shared context then resolves the
//! domain's [`CompilationCacheDomain::CacheKey`] through these tiers:
//!
//! ```text
//! Compilation Context lookup (keyed by CompilationCacheDomain::CacheKey)
//!          │
//!          ├── 1. Memory LRU hit ───────────────────────────────▶ shared Compiled Program
//!          │      stores Arc<CompiledProgram>
//!          │
//!          ├── 2. Same key in flight ──▶ wait for producer ─────▶ shared Compiled Program
//!          │      coordinates one producer; different keys remain concurrent
//!          │
//!          ├── 3. Persistent Disk Cache hit ──▶ deserialize ─────▶ shared Compiled Program
//!          │      stores a serialized Compiled Program
//!          │
//!          ├── 4. Distributed Artifact Exchange ─▶ deserialize ─▶ shared Compiled Program
//!          │      shares the serialized Compiled Program between processes
//!          │
//!          └── 5. Backend compilation ──────────────────────────▶ shared Compiled Program
//!                 │
//!                 ├── always insert into the Memory LRU
//!                 ├── optionally serialize into the Persistent Disk Cache
//!                 └── optionally publish through the Distributed Artifact Exchange
//! ```
//!
//! Same-key misses are single-flight while different keys may compile concurrently. [`DiskCache`] provides optional
//! checksummed persistent storage, and [`CompilationArtifactExchange`] optionally shares serialized artifacts among
//! processes according to [`CompilationArtifactExchangePolicy`]. [`JitCacheStatistics`] reports frontend dispatch
//! activity, while [`CompilationCacheStatistics`] and structured [`CompilationEvent`]s report shared-context
//! activity across the memory, persistent, exchange, and backend compilation tiers.
//!
//! # Extending Compilation
//!
//! Implement [`CompilationDomain`] after defining the backend's [`Domain`](crate::contexts::Domain). The required
//! contract supplies backend-owned lowered and compiled program types, options, an error, lowering, compilation,
//! output signatures, and flat execution. Lowering must fold every compile-relevant option, compiler, target, and
//! topology property into the lowered artifact, which keeps that artifact self-describing for caching and
//! persistence.
//!
//! Implement [`CompilationCacheDomain`] to opt into the shared [`CompilationContext`]: it derives cache identity
//! from the lowered program alone, and key equality must mean that compiled artifacts are interchangeable. Optional
//! hooks add persistent keying and executable codecs. Implement [`AnalyzableCompilationDomain`] to expose cost or
//! memory analysis without recompilation, and implement [`CompiledProgramOperation`] on the operation family when
//! staged functions must compose inside other traces.
//!
//! # Reading order
//!
//! 1. Start with [`CompilationDomain`] for the core/backend ownership boundary.
//! 2. Read [`JittedFunction`], [`StagedFunction`], [`LoweredFunction`], [`CompiledFunction`], and
//!    [`ExecutableProgram`] for the public lifecycle.
//! 3. Read [`ClosedProgram`] and [`CaptureReference`] for capture handling.
//! 4. Read [`CompilationContext`] for single-flight compilation and cache tiers.
//! 5. Read [`DiskCache`] and [`CompilationArtifactExchange`] only when adding persistence or distributed sharing.
//! 6. Continue with a backend crate such as `ryft-xla` for a concrete [`CompilationDomain`] implementation.

pub mod captures;
pub mod contexts;
pub mod disk_cache;
pub mod exchange;
pub mod function;

pub use captures::{CaptureReference, CapturingContext, ClosedProgram};
pub use contexts::{
    AnalyzableCompilationDomain, CompilationCacheDomain, CompilationCacheLevel, CompilationCacheOutcome,
    CompilationCacheStatistics, CompilationContext, CompilationDomain, CompilationEvent, CompilationMissReason,
};
pub use disk_cache::DiskCache;
pub use exchange::{CompilationArtifactExchange, CompilationArtifactExchangePolicy, CompilationExchangeError};
pub use function::{
    CompilationCall, CompilationStagingRequest, CompilationTracer, CompiledFunction, CompiledProgramOperation,
    ExecutableProgram, FlatCompilationProgram, JitCacheCapacities, JitCacheStatistics, JittedFunction, LoweredFunction,
    Specialization, StagedFunction, call_function, jit, jit_with_options, stage_function, try_jit,
    try_jit_with_options, try_jit_with_options_and_capacities, try_jit_with_options_and_capacity,
};
