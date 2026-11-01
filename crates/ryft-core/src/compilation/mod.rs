//! Contains backend-neutral machinery for _Just-In-Time (JIT) compilation_.
//!
//! This module supplies the common JIT lifecycle shared by compiler backends. Core traces typed Ryft programs, manages
//! runtime captures and structured signatures, coordinates specialization and executable caches, and validates calls.
//! A backend supplies target-specific options, lowering, executable compilation, and execution through
//! [`CompilationDomain`]s. Backends opt into exact cache identity and serialization through
//! [`CompilationCacheDomain`]s. Backend representations therefore remain outside `ryft-core`
//! while every backend follows the same lifecycle.
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
//! A _program_ is the transformable IR value (i.e., atoms, instructions, regions, and boundary types), which carries
//! neither captures nor a structured signature. A _function_ is the callable package that pairs a program with
//! everything a call needs, namely the capture table, the structured input and output signatures, the compilation
//! options, and the backend artifacts. Every lifecycle handle below is therefore a function rather than a program, and
//! [`CompiledFunctionDispatcher`] is not a program at all but a retained dispatcher over a Rust closure that produces
//! one program per specialization. The `*Program` names inside the domain contracts
//! ([`CompilationDomain::LoweredProgram`], [`CompilationDomain::CompiledProgram`], and [`FlatCompilationProgram`])
//! name the backend and IR artifacts that those functions carry.
//!
//! # Entry Points
//!
//!   - Use [`jit`], [`jit_with_options`], or [`jit_with_options_and_capacity`] for a retained JIT dispatcher over a
//!     fallible closure, which returns its outputs wrapped in [`Ok`] when it cannot fail. A
//!     [`CompiledFunctionDispatcher`] specializes on explicit static host parameters, the dynamic input's parameter
//!     structure, and the domain's runtime-derived dispatch key. Its first call for a specialization traces, lowers,
//!     and requests compilation; warm calls dispatch directly to the retained executable.
//!   - Use [`stage`] for ordinary capture-free staging. Call [`CompilationDomain::stage`] directly when fallibility,
//!     runtime captures, or symbolic capture references are needed. Then continue with [`CompilationDomain::lower`]
//!     and [`CompilationDomain::compile`].
//!   - Execute an [`ExecutableFunction`] through [`ExecutableFunction::call`]. Domains with transactional
//!     external state implement [`StatefulCompilationDomain`], and their executable functions are invoked through
//!     [`ExecutableFunction::call_statefully`] or [`ExecutableFunction::call_statefully_async`] instead. Retained
//!     dispatchers expose the same functions (e.g., [`CompiledFunctionDispatcher::call_statefully`]).
//!
//! Backend-facing crates normally wrap these generic entry points with value- and option-specific APIs.
//! New backend-neutral code can call them directly.
//!
//! # Transactional External State
//!
//! [`StatefulCompilationDomain`] is an opt-in execution capability, not a relaxation of the ordinary pure call
//! contract. A stateful backend snapshots each external reference, chains that reference's cumulative predecessor
//! dependency into the new invocation's joined completion instead of blocking on it, publishes read leases and
//! submitted mutations atomically after successful submission, and installs hidden final state without exposing it as
//! a public function result. Calls involving the same mutated reference allocation therefore serialize by generation
//! without any host wait, read-only calls may overlap, and calls involving independent allocations do not acquire a
//! global lock.
//!
//! [`ReferenceExecution`] owns whole-invocation completion. Its blocking `r#await` observes execution failure before
//! returning the reconstructed public output, while dropping it never cancels submitted work. A failure before
//! submission leaves every reference unchanged. After submission, a failure that makes the final state ambiguous
//! poisons the complete affected mutation group; later reference access reports that stored cause. Read-only
//! references are leased rather than donated and remain usable after a failed reader. Concrete backends may support
//! only a subset of shapes, memory placements, shardings, and process topologies and must reject the rest before
//! launch.
//!
//! # Lifecycle Handles
//!
//!   1. [`StagedFunction`] owns a typed [`ClosedProgram`](crate::ClosedProgram), concrete runtime captures, public
//!      input and output signatures, output structure, and compilation options. It contains no backend lowering or
//!      executable, and its [`StagedFunction::call`] method stages a nested call into an active context rather than
//!      executing at runtime.
//!   2. [`LoweredFunction`] owns the backend's [`CompilationDomain::LoweredProgram`], the source handle, and the
//!      options used to lower it. Compilation computes the domain's exact key and consults its [`CompilationContext`].
//!   3. [`CompiledFunction`] combines the executable with the staged and lowered metadata required for inspection and
//!      transformations. Pure runtime execution uses [`CompilationDomain::call`]; transactional external state uses
//!      [`StatefulCompilationDomain::call_statefully_async`].
//!   4. [`ExecutableFunction`] retains only the executable, captures, signatures, and output structure. Use
//!      [`CompiledFunction::into_executable_function`] when transform metadata is no longer needed. This runtime-only
//!      handle gains [`Send`] and [`Sync`] structurally whenever its backend fields do.
//!
//! # Captures and Nested Calls
//!
//! Runtime values closed over by a traced closure are not embedded as literal data.
//! A [`ClosedProgram`](crate::ClosedProgram) keeps them in a side table while the source IR stores typed
//! [`CaptureReference`](crate::CaptureReference) indices. Before lowering, captures are lifted as leading flat inputs,
//! and execution supplies arguments in the same `[captures..., public inputs...]` order. This keeps IR compact,
//! preserves device-resident buffers, and lets compilation depend on capture types rather than data.
//!
//! [`CapturingContext`](crate::CapturingContext) lets ordinary and transform contexts register captures through their
//! parent. To embed a staged function in a larger trace, an operation family implements [`CompiledCallOperation`], and
//! that operation then owns how the nested call behaves under lowering, batching, differentiation, partial evaluation,
//! and interpretation.
//!
//! # Specialization and Caching
//!
//! [`CompiledFunctionDispatcher`] keeps one bounded LRU cache of compiled specializations, built on the general
//! [`SpecializationCache`](crate::SpecializationCache) primitive and keyed by the static parameters, the input
//! parameter structure, and the domain's dispatch key. A hit dispatches directly to the retained executable, while a
//! miss traces the Rust closure, lowers the staged source, requests compilation, and inserts the resulting executable.
//! Tracing and lowering are cheap relative to backend compilation, so there are no intermediate tracing or lowering
//! caches. After an eviction or failure, the miss path simply reruns those stages, and the shared
//! [`CompilationContext`] below still deduplicates the expensive backend compilation.
//!
//! That argument does not carry over to eager (i.e., operation-by-operation) dispatch, where a compiled program is
//! reused for every application of the same operation and the relevant comparison is with _executing_ one small
//! operation rather than with compiling it. Backends that execute eager operations through compiled programs therefore
//! key applications of operations directly to compiled programs in front of tracing and lowering, using the operation's
//! identity (refer to the `#[ryft(dispatch(identity))]` attribute of `#[derive(Operation)]`) together with the input
//! types, like JAX's `(primitive, params)` dispatch cache. For example, the XLA backend keeps such a cache in each of
//! its sessions.
//!
//! Cloned [`CompiledFunctionDispatcher`] handles share one cache, and the dispatcher is usable from multiple threads
//! whenever its domain, closure, static parameters, and artifact types are thread-safe. Recursive dispatch on one
//! thread of the specialization currently being produced is rejected with an error, while concurrent cold misses on
//! different threads deliberately duplicate the cheap frontend work with idempotent inserts and leave the coordination
//! of backend compilation to the shared context.
//!
//! [`CompilationContext`] is the backend-artifact cache shared by every compilation using the same domain handle, and
//! it resolves the domain's [`CompilationCacheDomain::CacheKey`] through memory, same-key single-flight coordination,
//! optional persistent storage, optional distributed exchange, and finally backend compilation. Refer to
//! [`CompilationContext`] for the rendered tier diagram.
//!
//! Misses for the same key are single flight, while different keys may compile concurrently. [`DiskCache`] provides
//! optional checksummed persistent storage, and [`CompilationArtifactExchange`] optionally shares serialized artifacts
//! among processes according to [`CompilationArtifactExchangePolicy`]. [`JitCacheStatistics`] reports frontend dispatch
//! activity, while [`CompilationCacheStatistics`] and structured [`CompilationEvent`]s report the activity of the
//! shared context across the memory, persistent, exchange, and backend compilation tiers.
//!
//! # Extending Compilation
//!
//! Implement [`CompilationDomain`] after defining the backend's [`Domain`](crate::Domain). The required contract
//! supplies backend-owned lowered and compiled program types, options, an error, lowering, compilation, output
//! signatures, and flat execution. Lowering must fold every compile-relevant option, compiler, target, and topology
//! property into the lowered artifact, which keeps that artifact self-describing for caching and persistence.
//!
//! Implement [`CompilationCacheDomain`] to opt into the shared [`CompilationContext`]: it derives cache identity from
//! the lowered program alone, and key equality must mean that compiled artifacts are interchangeable. Optional hooks
//! add persistent keying and executable codecs. Implement [`AnalyzableCompilationDomain`] to expose cost or memory
//! analysis without recompilation, and implement [`CompiledCallOperation`] on the operation family when staged
//! functions must compose inside other traces.
//!
//! Core implements [`CompilationDomain::stage`], which backends must not override. Backends customize staging
//! through [`CompilationDomain::staged_input_types`] and [`CompilationDomain::staged_output_types`]. The remaining
//! [`CompilationDomain`] functions accept [`LoweringRequest`], [`CompilationRequest`], and [`CallRequest`]
//! implementations. These traits keep the structured input and output bounds of each stage in one place and expose
//! only the operations that a backend needs to complete it. They are also required by a limitation of the Rust
//! compiler, which [`CompilationDomain`] documents. Ordinary callers use [`CompilationDomain::stage`] or [`stage`],
//! while backend implementations name the request traits in their bounds and complete each stage
//! through their `into_lowered`, `into_compiled`, or `reconstruct` functions.

pub mod caching;
pub mod contexts;
pub mod exchange;
pub mod functions;

pub use caching::{
    CompilationCacheLevel, CompilationCacheOutcome, CompilationCacheStatistics, CompilationContext, CompilationEvent,
    CompilationMissReason, DiskCache,
};
pub use contexts::{
    AnalyzableCompilationDomain, CallRequest, CompilationCacheDomain, CompilationDomain, CompilationRequest,
    LoweringRequest, StatefulCompilationDomain,
};
pub use exchange::{CompilationArtifactExchange, CompilationArtifactExchangePolicy, CompilationExchangeError};
pub use functions::{
    CompilationCall, CompilationTracer, CompiledCallOperation, CompiledFunction, CompiledFunctionDispatcher,
    ExecutableFunction, FlatCompilationProgram, FunctionSpecializationKey, JitCacheStatistics, LoweredFunction,
    ReferenceExecution, StagedFunction, jit, jit_with_options, jit_with_options_and_capacity, stage,
};
