//! Contains backend-neutral machinery for _Just-In-Time (JIT) compilation_.
//!
//! This module supplies the common JIT lifecycle shared by compiler backends. Core traces typed Ryft programs, manages
//! runtime captures and structured signatures, coordinates specialization and executable caches, and validates calls.
//! A backend supplies target-specific options, lowering, executable compilation, and execution through
//! [`CompilationDomain`]. Backends opt into exact cache identity and serialization through
//! [`CompilationCacheDomain`]. Backend representations therefore remain outside `ryft-core` while every backend
//! follows the same lifecycle.
//!
//! # Retained JIT Lifecycle
//!
//! A warm call follows the cache-hit edge directly to execution. A cold call traces and lowers once in the producing
//! thread, delegates backend artifact reuse to the shared compilation context, publishes the executable, and then
//! executes it. Refer to [`CompiledFunctionDispatcher`] for a rendered diagram of this dispatch loop.
//!
//! Two transitions exist outside this dispatch loop. A staged function can be embedded as a nested call boundary in
//! an enclosing trace instead of being lowered directly, and a compiled function can shed its staged and lowered
//! transform metadata into the runtime-only executable handle that the cache retains. The secondary transitions are
//! included in the lifecycle diagram on [`CompiledFunctionDispatcher`].
//!
//! # Terminology
//!
//! A _program_ is the transformable IR value: atoms, instructions, regions, and boundary types, carrying neither
//! captures nor a structured signature. A _function_ is the callable package that pairs a program with everything a
//! call needs, namely the capture table, the structured input and output signatures, the compilation options, and the
//! backend artifacts. Every lifecycle handle below is therefore a function rather than a program, and
//! [`CompiledFunctionDispatcher`] is not a program at all: it is a retained dispatcher over a Rust closure that
//! produces one program per specialization. The `*Program` names inside the domain contracts
//! ([`CompilationDomain::LoweredProgram`], [`CompilationDomain::CompiledProgram`], and [`FlatCompilationProgram`])
//! name the backend and IR artifacts that those functions carry.
//!
//! # Entry Points
//!
//! - Use [`jit`] or [`try_jit`] for a retained JIT dispatcher. A [`CompiledFunctionDispatcher`] specializes on explicit
//!   static host parameters, the dynamic input's parameter structure, and the domain's runtime-derived dispatch key.
//!   Its first call for a specialization traces, lowers, and requests compilation; warm calls dispatch directly to
//!   the retained executable.
//! - Use [`stage_function`] for ordinary capture-free staging. Construct a [`CompilationStagingRequest`] and pass it to
//!   [`CompilationDomain::stage`] when fallibility, runtime captures, or symbolic capture references are needed. Then
//!   continue with [`CompilationDomain::lower`] and [`CompilationDomain::compile`].
//! - Domains with transactional external state implement [`StatefulCompilationDomain`]. Invoke their executable
//!   handles through [`call_function_statefully`] or [`call_function_statefully_async`]; retained dispatchers expose
//!   the same capability through [`CompiledFunctionDispatcher::call_statefully`] and
//!   [`CompiledFunctionDispatcher::call_statefully_async`].
//!
//! Backend-facing crates normally wrap these generic entry points with value- and option-specific APIs. New
//! backend-neutral code can call them directly.
//!
//! # Transactional External State
//!
//! [`StatefulCompilationDomain`] is an opt-in execution capability, not a relaxation of the ordinary pure call
//! contract. A stateful backend snapshots each external holder, chains that holder's cumulative predecessor dependency
//! into the new invocation's joined completion instead of blocking on it, publishes read leases and mutation
//! reservations atomically after successful submission, and installs hidden final state without exposing it as a
//! public function result. Calls involving the same mutated holder therefore serialize by generation without any host
//! wait, read-only calls may overlap, and calls involving independent holders do not acquire a global lock.
//!
//! [`ReferenceExecution`] owns whole-invocation completion. Its blocking `r#await` observes execution failure before
//! returning the reconstructed public output, while dropping it never cancels submitted work. A failure before
//! submission leaves every holder unchanged. After submission, a failure that makes the final state ambiguous
//! poisons the complete affected mutation group; later holder access reports that stored cause. Read-only holders are
//! leased rather than donated and remain usable after a failed reader. Concrete backends may support only a subset of
//! shapes, memory placements, shardings, and process topologies and must reject the rest before launch.
//!
//! # Lifecycle Handles
//!
//! 1. [`StagedFunction`] owns a typed [`ClosedProgram`](crate::captures::ClosedProgram), concrete runtime captures,
//!    public input and output signatures, output structure, and compilation options. It contains no backend lowering
//!    or executable, and its [`StagedFunction::call`] method stages a nested call into an active context rather than
//!    executing at runtime.
//! 2. [`LoweredFunction`] owns the backend's [`CompilationDomain::LoweredProgram`], the source handle, and the
//!    options used to lower it. Compilation computes the domain's exact key and consults its [`CompilationContext`].
//! 3. [`CompiledFunction`] combines the executable with the staged and lowered metadata required for inspection and
//!    transformations. Pure runtime execution uses [`CompilationDomain::call`]; transactional external state uses
//!    [`StatefulCompilationDomain::call_statefully_async`].
//! 4. [`ExecutableFunction`] retains only the executable, captures, signatures, and output structure. Use
//!    [`CompiledFunction::into_executable_function`] when transform metadata is no longer needed; this runtime-only
//!    handle gains `Send` and `Sync` structurally whenever its backend fields do.
//!
//! # Captures and Nested Calls
//!
//! Runtime values closed over by a traced closure are not embedded as literal data. A
//! [`ClosedProgram`](crate::captures::ClosedProgram) keeps them in a side table while the source IR stores typed
//! [`CaptureReference`](crate::captures::CaptureReference) indices. Before lowering, captures are lifted
//! as leading flat inputs, and execution supplies arguments in the same `[captures..., public inputs...]` order. This
//! keeps IR compact, preserves device-resident buffers, and lets compilation depend on capture types rather than
//! data.
//!
//! [`CapturingContext`](crate::captures::CapturingContext) lets ordinary and transform contexts register captures
//! through their parent. To embed a staged function in a larger trace, an operation family implements
//! [`CompiledCallOperation`]; that operation then owns how the nested call behaves under lowering, batching,
//! differentiation, partial evaluation, and interpretation.
//!
//! # Specialization and Caching
//!
//! [`CompiledFunctionDispatcher`] keeps one bounded LRU cache of compiled specializations, built on the general
//! [`SpecializationCache`](crate::specialization::SpecializationCache) primitive and keyed by the static parameters,
//! the input parameter structure, and the domain's dispatch key. A hit dispatches directly to the retained
//! executable; a miss traces the Rust closure, lowers the staged source, requests compilation, and inserts the
//! resulting executable. Tracing and lowering are cheap relative to backend compilation, so there are no intermediate
//! trace/lowering caches: after an eviction or failure the miss path simply reruns those stages, and the shared
//! [`CompilationContext`] below still deduplicates the expensive backend compile.
//!
//! Cloned [`CompiledFunctionDispatcher`] handles share one cache, and the dispatcher is usable from multiple threads
//! whenever its domain, closure, static parameters, and artifact types are thread-safe: same-thread recursive dispatch
//! of the specialization currently being produced is rejected with an error, while concurrent cold misses on
//! different threads deliberately duplicate the cheap frontend work with idempotent inserts and leave backend-compile
//! coordination to the shared context.
//!
//! [`CompilationContext`] is the backend-artifact cache shared by every compilation using the same domain handle,
//! and it resolves the domain's [`CompilationCacheDomain::CacheKey`] through memory, same-key single-flight
//! coordination, optional persistent storage, optional distributed exchange, and finally backend compilation. Refer
//! to [`CompilationContext`] for the rendered tier diagram.
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
//! memory analysis without recompilation, and implement [`CompiledCallOperation`] on the operation family when
//! staged functions must compose inside other traces.
//!
//! The generic [`CompilationDomain`] methods use [`StageRequest`], [`LoweringRequest`], [`CompileRequest`], and
//! [`CallRequest`] as typed backend-extension witnesses. These traits keep the structured input/output bounds in one
//! place and expose only the transition-specific artifact operations a backend needs. Ordinary callers construct a
//! [`CompilationStagingRequest`] or use [`stage_function`]; backend implementations name the witness traits in their
//! method bounds and complete each transition through their `trace`, `into_lowered`, `into_compiled`, or `reconstruct`
//! methods.
//!
//! # Reading Order
//!
//! 1. Start with [`CompilationDomain`] for the core/backend ownership boundary.
//! 2. Read [`StageRequest`], [`LoweringRequest`], [`CompileRequest`], and [`CallRequest`] alongside
//!    [`StagedFunction`], [`LoweredFunction`], [`CompiledFunction`], and [`ExecutableFunction`] for the typed
//!    lifecycle.
//! 3. Read [`ClosedProgram`](crate::captures::ClosedProgram) and
//!    [`CaptureReference`](crate::captures::CaptureReference) in [`crate::captures`] for capture handling.
//! 4. Read [`CompilationContext`] for single-flight compilation and cache tiers.
//! 5. Read [`DiskCache`] and [`CompilationArtifactExchange`] only when adding persistence or distributed sharing.
//! 6. Continue with a backend crate such as `ryft-xla` for a concrete [`CompilationDomain`] implementation.

pub mod contexts;
pub mod disk_cache;
pub mod exchange;
pub mod function;

pub use contexts::{
    AnalyzableCompilationDomain, CompilationCacheDomain, CompilationCacheLevel, CompilationCacheOutcome,
    CompilationCacheStatistics, CompilationContext, CompilationDomain, CompilationEvent, CompilationMissReason,
    StatefulCompilationDomain,
};
pub use disk_cache::DiskCache;
pub use exchange::{CompilationArtifactExchange, CompilationArtifactExchangePolicy, CompilationExchangeError};
pub use function::{
    CallRequest, CompilationCall, CompilationStagingRequest, CompilationTracer, CompileRequest, CompiledCallOperation,
    CompiledFunction, CompiledFunctionDispatcher, ExecutableFunction, FlatCompilationProgram,
    FunctionSpecializationKey, JitCacheStatistics, LoweredFunction, LoweringRequest, ReferenceExecution, StageRequest,
    StagedFunction, call_function, call_function_statefully, call_function_statefully_async, jit, jit_with_options,
    stage_function, try_jit, try_jit_with_options, try_jit_with_options_and_capacity,
};
