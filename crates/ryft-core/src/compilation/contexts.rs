use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use lru::LruCache;

use crate::captures::CaptureConstant;
use crate::compilation::disk_cache::{CacheDigest, DiskCache};
use crate::compilation::exchange::{
    CompilationArtifactExchange, CompilationArtifactExchangePolicy, CompilationExchangeError,
};
use crate::contexts::Domain;
use crate::parameters::Parameterized;
use crate::programs::{ProgramError, Type};

use super::function::{
    CallRequest, CompileRequest, CompiledFunction, ExecutableFunction, LoweredFunction, LoweringRequest,
    ReferenceExecution, StageRequest, StagedFunction,
};

/// [`CompilationDomain`]s are [`Domain`]s that support lowering, compiling, and executing staged programs.
/// Compilation is deliberately split into four semantic stages:
///
///   1. [`Self::stage`] traces a closure using options fixed before tracing.
///   2. [`Self::lower`] transforms the staged artifact into [`Self::LoweredProgram`].
///   3. [`Self::compile`] transforms the lowered artifact into [`Self::CompiledProgram`].
///   4. [`Self::call`] executes an [`ExecutableFunction`] against structured runtime inputs.
///
/// The domain is the active service for every transition; staged, lowered, compiled, and executable artifacts remain
/// passive values. Each method owns its complete backend-specific transition, including any validation, caching, or
/// runtime conversion policy the backend chooses to apply.
pub trait CompilationDomain: Domain<Constant: CaptureConstant> + Clone {
    /// Backend-defined key for one retained-JIT input specialization.
    type DispatchKey: Clone + Eq + Hash;

    /// Backend-specific lowered program representation produced by [`Self::lower`].
    type LoweredProgram;

    /// Backend-specific compiled program representation produced by [`Self::compile`].
    type CompiledProgram;

    /// Backend-specific compilation options type. Meshes, sharding and layout overrides, donation declarations,
    /// compiler flags, etc., are all represented as part of this type.
    type Options;

    /// Backend-specific error type. Staging and call-boundary errors flow through it as [`ProgramError`]s and
    /// that is why it requires [`From<ProgramError>`].
    type Error: std::error::Error + From<ProgramError>;

    /// Derives the effective staged input types and their retained-JIT cache key from one runtime input signature.
    ///
    /// Most domains return the input types unchanged and use them as the key. Backends may instead apply a
    /// semantics-preserving dispatch policy such as bounded-shape bucketing. The returned key must compare equal
    /// exactly when the returned effective types are interchangeable for tracing, lowering, compilation, and calls.
    ///
    /// The effective types are returned as a shared slice so that key construction on the retained-dispatch hot path
    /// never deep-clones the type vector: domains whose key embeds the effective types (e.g., exact dispatch) share
    /// one allocation between the key and the returned signature, and the caller clones individual types only on a
    /// specialization miss, when it prepares the staging request.
    fn dispatch_signature(
        &self,
        input_types: Vec<Self::Type>,
        options: &Self::Options,
    ) -> Result<(Self::DispatchKey, Arc<[Self::Type]>), Self::Error>;

    /// Traces a fallible function with explicit runtime captures and their symbolic references.
    ///
    /// This is the only staging primitive. Implementations apply all tracing-sensitive options to the effective input
    /// and output signatures and retain those same options in the returned artifact.
    fn stage<Request>(
        &self,
        request: Request,
    ) -> Result<StagedFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Self: Sized,
        Request: StageRequest<Self>;

    /// Performs the complete backend-specific lowering transition.
    ///
    /// The returned lowered artifact must be self-describing: every compile-relevant option, compiler, target, and
    /// topology property must be folded into [`Self::LoweredProgram`] during this transition, because caching
    /// backends derive compiled-program identity from the lowered program alone.
    fn lower<Request>(
        &self,
        staged: Request,
    ) -> Result<LoweredFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Self: Sized,
        Request: LoweringRequest<Self>;

    /// Performs the complete backend-specific compilation transition.
    fn compile<Request>(
        &self,
        lowered: Request,
    ) -> Result<CompiledFunction<Self, Request::Input, Request::Output>, Self::Error>
    where
        Self: Sized,
        Request: CompileRequest<Self>;

    /// Performs the complete backend-specific structured execution transition.
    fn call<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, Self::Error>
    where
        Self: Sized,
        Request: CallRequest<Self>;
}

/// Optional execution capability for compiled functions that bind external mutable reference holders.
///
/// Implementations must keep hidden state results out of [`CallRequest::RuntimeOutput`] and preserve their holder
/// transaction guarantees even when the public output has no leaves or fails reconstruction.
pub trait StatefulCompilationDomain: CompilationDomain {
    /// Performs a completion-bearing stateful structured execution transition.
    fn call_statefully_async<Request>(
        &self,
        request: Request,
    ) -> ReferenceExecution<Request::RuntimeOutput, Self::Error>
    where
        Self: Sized,
        Request: CallRequest<Self>;

    /// Executes a stateful call and waits for whole-invocation completion before returning.
    #[inline]
    fn call_statefully<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, Self::Error>
    where
        Self: Sized,
        Request: CallRequest<Self>,
    {
        self.call_statefully_async(request).r#await()
    }
}

/// Optional capability for caching backend-compiled programs.
///
/// Cache identity is derived from the lowered program alone: [`CompilationDomain::lower`] folds every
/// compile-relevant option, compiler, target, and topology property into
/// [`CompilationDomain::LoweredProgram`], so key equality always means that compiled programs are interchangeable.
pub trait CompilationCacheDomain: CompilationDomain {
    /// Exact structural cache key for interchangeable compiled programs.
    type CacheKey: Clone + Eq + Hash + Send + Sync + 'static;

    /// Constructs the exact cache key for `program`.
    fn compilation_key(&self, program: &Self::LoweredProgram) -> Result<Self::CacheKey, Self::Error>;

    /// Returns stable canonical bytes for persistent cache identity, or `None` when this domain does not support
    /// persistent executable caching.
    ///
    /// These bytes must remain stable across processes that may share the cache and must cover the complete
    /// computation, options, compiler/backend versions, and target topology. Core hashes them only for filename-safe
    /// addressing; it does not add missing semantic state.
    #[inline]
    fn persistent_cache_key(&self, _key: &Self::CacheKey) -> Option<Vec<u8>> {
        None
    }

    /// Serializes a compiled program for persistent caching. `Ok(None)` means unsupported.
    #[inline]
    fn serialize_program(&self, _program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(None)
    }

    /// Deserializes a persistent-cache payload. `Ok(None)` means unsupported or incompatible.
    #[inline]
    fn deserialize_program(&self, _bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
        Ok(None)
    }
}

/// Optional capability for inspecting a compiled program without recompiling it.
///
/// Analysis is deliberately separate from [`CompilationDomain`] because not every backend or runtime plugin can
/// expose compiler cost and memory information. Implementations should cache immutable backend results when querying
/// them is non-trivial. Calling this method must never trigger compilation.
pub trait AnalyzableCompilationDomain: CompilationDomain {
    /// Backend-owned, typed analysis report.
    type Analysis;

    /// Analyzes the compiled program carried by an [`ExecutableFunction`] without recompiling it.
    fn analyze<Input: Parameterized<Self::Type>, Output: Parameterized<Self::Type>>(
        &self,
        executable_function: &ExecutableFunction<Self, Input, Output>,
    ) -> Result<Self::Analysis, Self::Error>
    where
        Self: Sized;
}

/// Default in-memory compile-cache capacity. Use [`CompilationContext::with_capacity`] when a
/// workload needs a different bound.
const DEFAULT_CACHE_CAPACITY: usize = 8192;

/// Default number of structured compilation events retained for diagnostics.
const DEFAULT_EVENT_CAPACITY: usize = 0;

/// Cache tier or lifecycle stage that produced a [`CompilationEvent`].
///
/// Events are emitted only by [`CompilationContext`], so the variants cover exactly the tiers that the shared
/// compiled-program cache coordinates.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompilationCacheLevel {
    /// Process-local compiled-program memory cache.
    Memory,

    /// Persistent serialized-program cache.
    Persistent,

    /// Distributed serialized-program exchange.
    Exchange,

    /// Backend executable compilation.
    Backend,
}

/// Outcome of one compilation lifecycle or cache operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompilationCacheOutcome {
    /// A reusable entry was found.
    Hit,
    /// No reusable entry was found.
    Miss,
    /// The caller waited for another producer.
    Wait,
    /// The operation completed successfully.
    Succeeded,
    /// The operation failed.
    Failed,
    /// The operation was unsupported or disabled.
    Skipped,
}

/// Stable explanation for a compilation-cache miss or failure.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CompilationMissReason {
    /// No entry existed for the requested key.
    NotFound,

    /// The domain or configured service does not support the requested cache tier.
    Unsupported,

    /// A stored artifact was incompatible with the current domain or runtime.
    Incompatible,

    /// The configured deadline elapsed.
    TimedOut,

    /// Reading or receiving an artifact failed.
    ReadFailed,

    /// Backend deserialization failed.
    DeserializationFailed,

    /// Serializing or writing an artifact failed.
    WriteFailed,

    /// The backend producer failed.
    ProducerFailed,
}

/// Structured diagnostic event for one compilation lifecycle or cache operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CompilationEvent {
    /// Lifecycle or cache tier that emitted this event.
    pub level: CompilationCacheLevel,
    /// Result of the operation.
    pub outcome: CompilationCacheOutcome,
    /// Host duration spent in this operation.
    pub duration: Duration,
    /// Stable miss or failure explanation, when applicable.
    pub miss_reason: Option<CompilationMissReason>,
}

/// Snapshot of process-local compilation-cache activity.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CompilationCacheStatistics {
    /// Lookups served directly by the in-memory LRU.
    pub memory_hits: u64,

    /// Lookups restored from a validated persistent-cache entry.
    pub persistent_hits: u64,

    /// Lookups that became the single producer for a missing key.
    pub misses: u64,

    /// Backend producer closures invoked after both cache tiers missed.
    pub compilations: u64,

    /// Same-key lookups that waited for an in-flight producer.
    pub waits: u64,

    /// Persistent read, write, serialization, or deserialization failures degraded to misses.
    pub persistent_errors: u64,

    /// Lookups restored from a distributed artifact exchange.
    pub exchange_hits: u64,

    /// Follower lookups that waited for a distributed artifact.
    pub exchange_waits: u64,

    /// Compiled artifacts successfully published by producer processes.
    pub exchange_publishes: u64,

    /// Distributed exchange, serialization, or deserialization failures.
    pub exchange_errors: u64,

    /// Distributed waits that completed without an artifact before their deadline.
    pub exchange_timeouts: u64,

    /// Distributed misses or failures that fell back to local compilation.
    pub exchange_fallbacks: u64,

    /// Total host nanoseconds spent in persistent lookup and deserialization.
    pub persistent_lookup_duration_ns: u64,

    /// Total host nanoseconds spent in process-local lookup and same-key waiting.
    pub memory_lookup_duration_ns: u64,

    /// Total host nanoseconds spent waiting, receiving, and deserializing exchanged artifacts.
    pub exchange_duration_ns: u64,

    /// Total host nanoseconds spent in backend producer closures.
    pub compilation_duration_ns: u64,
}

#[derive(Default)]
struct AtomicCompilationCacheStatistics {
    memory_hits: AtomicU64,
    persistent_hits: AtomicU64,
    misses: AtomicU64,
    compilations: AtomicU64,
    waits: AtomicU64,
    persistent_errors: AtomicU64,
    exchange_hits: AtomicU64,
    exchange_waits: AtomicU64,
    exchange_publishes: AtomicU64,
    exchange_errors: AtomicU64,
    exchange_timeouts: AtomicU64,
    exchange_fallbacks: AtomicU64,
    persistent_lookup_duration_ns: AtomicU64,
    memory_lookup_duration_ns: AtomicU64,
    exchange_duration_ns: AtomicU64,
    compilation_duration_ns: AtomicU64,
}

impl AtomicCompilationCacheStatistics {
    fn snapshot(&self) -> CompilationCacheStatistics {
        CompilationCacheStatistics {
            memory_hits: self.memory_hits.load(Ordering::Relaxed),
            persistent_hits: self.persistent_hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            compilations: self.compilations.load(Ordering::Relaxed),
            waits: self.waits.load(Ordering::Relaxed),
            persistent_errors: self.persistent_errors.load(Ordering::Relaxed),
            exchange_hits: self.exchange_hits.load(Ordering::Relaxed),
            exchange_waits: self.exchange_waits.load(Ordering::Relaxed),
            exchange_publishes: self.exchange_publishes.load(Ordering::Relaxed),
            exchange_errors: self.exchange_errors.load(Ordering::Relaxed),
            exchange_timeouts: self.exchange_timeouts.load(Ordering::Relaxed),
            exchange_fallbacks: self.exchange_fallbacks.load(Ordering::Relaxed),
            persistent_lookup_duration_ns: self.persistent_lookup_duration_ns.load(Ordering::Relaxed),
            memory_lookup_duration_ns: self.memory_lookup_duration_ns.load(Ordering::Relaxed),
            exchange_duration_ns: self.exchange_duration_ns.load(Ordering::Relaxed),
            compilation_duration_ns: self.compilation_duration_ns.load(Ordering::Relaxed),
        }
    }

    fn clear(&self) {
        self.memory_hits.store(0, Ordering::Relaxed);
        self.persistent_hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.compilations.store(0, Ordering::Relaxed);
        self.waits.store(0, Ordering::Relaxed);
        self.persistent_errors.store(0, Ordering::Relaxed);
        self.exchange_hits.store(0, Ordering::Relaxed);
        self.exchange_waits.store(0, Ordering::Relaxed);
        self.exchange_publishes.store(0, Ordering::Relaxed);
        self.exchange_errors.store(0, Ordering::Relaxed);
        self.exchange_timeouts.store(0, Ordering::Relaxed);
        self.exchange_fallbacks.store(0, Ordering::Relaxed);
        self.persistent_lookup_duration_ns.store(0, Ordering::Relaxed);
        self.memory_lookup_duration_ns.store(0, Ordering::Relaxed);
        self.exchange_duration_ns.store(0, Ordering::Relaxed);
        self.compilation_duration_ns.store(0, Ordering::Relaxed);
    }
}

enum InFlightState<P> {
    Pending,
    Ready(Arc<P>),
    Failed,
}

struct InFlightCompilation<P> {
    state: Mutex<InFlightState<P>>,
    ready: Condvar,
}

impl<P> InFlightCompilation<P> {
    fn new() -> Self {
        Self { state: Mutex::new(InFlightState::Pending), ready: Condvar::new() }
    }

    fn wait(&self) -> Option<Arc<P>> {
        let mut state = self.state.lock().expect("in-flight compilation mutex should not be poisoned");
        loop {
            match &*state {
                InFlightState::Pending => {
                    state = self.ready.wait(state).expect("in-flight compilation mutex should not be poisoned");
                }
                InFlightState::Ready(program) => return Some(Arc::clone(program)),
                InFlightState::Failed => return None,
            }
        }
    }

    fn finish(&self, state: InFlightState<P>) {
        *self.state.lock().expect("in-flight compilation mutex should not be poisoned") = state;
        self.ready.notify_all();
    }
}

/// Process-local cache of compiled programs, generic over the
/// [`CompilationDomain`] backend.
///
/// Construct one [`CompilationContext`] per backend handle at program start and reuse it across
/// compilation requests and any backend-specific helpers that look up entries in the cache.
///
/// The cache is keyed by the domain's structurally-typed [`CompilationCacheDomain::CacheKey`] — `Eq` on the key
/// guarantees no silent collisions, in contrast to a hash-only cache. On cache hit the cached program is returned
/// without invoking the producer closure. On miss the producer runs and the result is inserted.
///
/// The in-memory tier uses LRU eviction with a default capacity of `8192` entries.
/// Use [`CompilationContext::with_capacity`] to override.
///
/// An optional [`DiskCache`] second-tier is configured via [`CompilationContext::with_disk_cache`]. When present, the
/// disk tier is consulted between the in-memory tier and the producer closure. The cache uses
/// [`CompilationCacheDomain::serialize_program`] and [`CompilationCacheDomain::deserialize_program`] to round-trip
/// programs; any error from either method is treated as a cache miss for that entry. An optional
/// [`CompilationArtifactExchange`] may provide the next serialized tier according to the configured
/// [`CompilationArtifactExchangePolicy`].
///
/// # Cache Resolution
///
/// ```mermaid
/// %%{init: {"themeCSS": ".nodeLabel code { white-space: nowrap !important; }"}}%%
/// flowchart TD
///   request["Compile Request and Domain Cache Key"] --> memory["1. In-Memory LRU"]
///   memory -->|"hit"| shared["Shared Compiled Program in &lt;code&gt;Arc&lt;/code&gt;"]
///   memory -->|"miss"| flight["2. Same-Key Single-Flight Election"]
///   flight -->|"another producer"| wait["Wait and Retry Memory Lookup"]
///   wait --> memory
///   publish["Publish Successful Artifact"]
///   flight -->|"elected producer"| disk["3. Optional Persistent Disk Cache"]
///   disk -->|"hit and deserialize"| publish
///   disk -->|"miss or disabled"| exchange["4. Optional Distributed Artifact Exchange"]
///   exchange -->|"hit and deserialize"| publish
///   exchange -->|"miss or unavailable"| backend["5. Backend Compilation"]
///   backend --> publish
///   publish --> memory_insert["Insert into In-Memory LRU"]
///   memory_insert --> shared
///   publish -.->|"best-effort serialization"| disk_store["Store in Persistent Disk Cache"]
///   publish -.->|"policy-controlled publication"| exchange_store["Publish through Artifact Exchange"]
/// ```
///
/// Same-key misses coordinate one backend producer, while requests for different keys remain concurrent. Every
/// successful source publishes to memory; persistent and exchange writes remain best-effort or policy-controlled.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct CompilationContext<D: CompilationCacheDomain> {
    /// In-memory LRU keyed by the domain's structural [`CompilationCacheDomain::CacheKey`].
    programs: Mutex<LruCache<D::CacheKey, Arc<D::CompiledProgram>>>,

    /// Per-key producer coordination. Entries exist only while a cache miss is being restored or
    /// compiled, so unrelated keys never wait on one another's backend work.
    in_flight: Mutex<HashMap<D::CacheKey, Arc<InFlightCompilation<D::CompiledProgram>>>>,

    /// Optional disk-backed second-tier cache.
    disk_cache: Option<DiskCache>,

    /// Optional byte-oriented exchange for sharing serialized programs across processes.
    artifact_exchange: Option<Arc<dyn CompilationArtifactExchange>>,

    /// Distributed exchange behavior. Disabled by default to preserve local compilation semantics.
    artifact_exchange_policy: CompilationArtifactExchangePolicy,

    /// Lock-free counters used for verification and operational observability.
    statistics: AtomicCompilationCacheStatistics,

    /// Bounded structured event history used for diagnostics without a logging dependency.
    recent_events: Mutex<VecDeque<CompilationEvent>>,

    /// Maximum number of entries retained in `recent_events`. Zero disables retention.
    event_capacity: usize,

    /// Optional non-blocking reporting hook invoked after an event is retained.
    event_reporter: Option<Arc<dyn Fn(&CompilationEvent) + Send + Sync>>,
}

struct InFlightProducer<'a, D: CompilationCacheDomain> {
    context: &'a CompilationContext<D>,
    cache_key: Option<D::CacheKey>,
    in_flight: Option<Arc<InFlightCompilation<D::CompiledProgram>>>,
}

impl<'a, D: CompilationCacheDomain> InFlightProducer<'a, D> {
    fn new(
        context: &'a CompilationContext<D>,
        cache_key: D::CacheKey,
        in_flight: Arc<InFlightCompilation<D::CompiledProgram>>,
    ) -> Self {
        Self { context, cache_key: Some(cache_key), in_flight: Some(in_flight) }
    }

    fn cache_key(&self) -> &D::CacheKey {
        self.cache_key.as_ref().expect("active producer owns its cache key")
    }

    fn finish(mut self, program: D::CompiledProgram) -> Arc<D::CompiledProgram> {
        let cache_key = self.cache_key.take().expect("active producer owns its cache key");
        let in_flight = self.in_flight.take().expect("active producer owns its in-flight state");
        self.context.finish_success(cache_key, in_flight, program)
    }
}

impl<D: CompilationCacheDomain> Drop for InFlightProducer<'_, D> {
    fn drop(&mut self) {
        if let (Some(cache_key), Some(in_flight)) = (self.cache_key.take(), self.in_flight.take()) {
            self.context.finish_failure(&cache_key, in_flight);
        }
    }
}

impl<D: CompilationCacheDomain> CompilationContext<D> {
    /// Creates a [`CompilationContext`] with the default cache capacity and no disk-cache tier.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CACHE_CAPACITY)
    }

    /// Creates a [`CompilationContext`] with an explicit cache capacity. `capacity` must be
    /// greater than zero; values of zero are silently clamped to one entry.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(capacity.max(1)).expect("cache capacity is clamped to at least one");
        Self {
            programs: Mutex::new(LruCache::new(capacity)),
            in_flight: Mutex::new(HashMap::new()),
            disk_cache: None,
            artifact_exchange: None,
            artifact_exchange_policy: CompilationArtifactExchangePolicy::Disabled,
            statistics: AtomicCompilationCacheStatistics::default(),
            recent_events: Mutex::new(VecDeque::with_capacity(DEFAULT_EVENT_CAPACITY)),
            event_capacity: DEFAULT_EVENT_CAPACITY,
            event_reporter: None,
        }
    }

    /// Attaches a [`DiskCache`] rooted at `directory` as a second-tier persistent compile cache
    /// behind this context's in-memory LRU. On cache miss, [`Self::get_or_compile`] consults
    /// disk before invoking the supplied producer. Domains that explicitly support persistence
    /// can serialize sufficiently expensive compilations so future processes restart warm.
    ///
    /// Returns an [`std::io::Error`] only when the directory itself can't be opened or created;
    /// subsequent persistent read, write, or codec errors are recorded in [`Self::statistics`]
    /// and degrade to the in-memory path.
    pub fn with_disk_cache(mut self, directory: impl Into<PathBuf>) -> std::io::Result<Self> {
        self.disk_cache = Some(DiskCache::open(directory)?);
        Ok(self)
    }

    /// Attaches an already configured [`DiskCache`], including any custom capacity or write
    /// thresholds.
    #[inline]
    pub fn with_configured_disk_cache(mut self, disk_cache: DiskCache) -> Self {
        self.disk_cache = Some(disk_cache);
        self
    }

    /// Attaches a [`DiskCache`] populated from the [`DiskCache::ENV_VAR`] environment variable,
    /// if it is set. Returns a context without a disk cache when the variable is absent, and
    /// returns invalid configuration or filesystem failures explicitly.
    pub fn with_disk_cache_from_env(mut self) -> std::io::Result<Self> {
        self.disk_cache = DiskCache::from_env()?;
        Ok(self)
    }

    /// Attaches a distributed compiled-artifact exchange using `policy`.
    ///
    /// The exchange is used only when the domain supplies stable bytes through
    /// [`CompilationCacheDomain::persistent_cache_key`] and can serialize and deserialize compiled programs. Process
    /// zero compiles and publishes; follower processes receive and restore. A single-process exchange is ignored.
    pub fn with_artifact_exchange(
        mut self,
        exchange: Arc<dyn CompilationArtifactExchange>,
        policy: CompilationArtifactExchangePolicy,
    ) -> Self {
        self.artifact_exchange = Some(exchange);
        self.artifact_exchange_policy = policy;
        self
    }

    /// Configures the number of structured events retained by [`Self::recent_events`].
    ///
    /// A zero capacity disables retention while leaving an installed reporter active.
    pub fn with_event_capacity(mut self, capacity: usize) -> Self {
        self.event_capacity = capacity;
        self.recent_events = Mutex::new(VecDeque::with_capacity(capacity));
        self
    }

    /// Installs a reporting hook invoked for every structured compilation event.
    ///
    /// The hook runs without holding internal cache or event-buffer locks. It should return promptly so it does not add
    /// latency to compilation paths.
    pub fn with_event_reporter(mut self, reporter: Arc<dyn Fn(&CompilationEvent) + Send + Sync>) -> Self {
        self.event_reporter = Some(reporter);
        self
    }

    /// Returns the attached [`DiskCache`], if any.
    #[inline]
    pub fn disk_cache(&self) -> Option<&DiskCache> {
        self.disk_cache.as_ref()
    }

    /// Returns the attached distributed artifact exchange, if any.
    #[inline]
    pub fn artifact_exchange(&self) -> Option<&dyn CompilationArtifactExchange> {
        self.artifact_exchange.as_deref()
    }

    /// Returns the configured distributed artifact exchange policy.
    #[inline]
    pub fn artifact_exchange_policy(&self) -> CompilationArtifactExchangePolicy {
        self.artifact_exchange_policy
    }

    /// Returns the number of compiled programs currently cached in the in-memory tier.
    ///
    /// Mostly useful for telemetry and tests that need to confirm that repeated compilations of
    /// the same structural signature reuse the cached program instead of recompiling.
    #[inline]
    pub fn cache_size(&self) -> usize {
        self.programs.lock().expect("compile cache mutex should not be poisoned").len()
    }

    /// Removes every entry from the in-memory cache.
    #[inline]
    pub fn clear_cache(&self) {
        self.programs.lock().expect("compile cache mutex should not be poisoned").clear();
    }

    /// Returns a lock-free snapshot of cache activity counters.
    #[inline]
    pub fn statistics(&self) -> CompilationCacheStatistics {
        self.statistics.snapshot()
    }

    /// Resets every cache activity counter without changing cached programs.
    #[inline]
    pub fn clear_statistics(&self) {
        self.statistics.clear();
    }

    /// Returns the retained structured events in emission order.
    pub fn recent_events(&self) -> Vec<CompilationEvent> {
        self.recent_events
            .lock()
            .expect("compilation event mutex should not be poisoned")
            .iter()
            .copied()
            .collect()
    }

    /// Removes every retained structured event without changing cache entries or statistics.
    #[inline]
    pub fn clear_events(&self) {
        self.recent_events.lock().expect("compilation event mutex should not be poisoned").clear();
    }

    /// Performs the complete cached compile transition for `request`.
    ///
    /// This owns the orchestration shared by caching backends: it derives the exact structural key through
    /// [`CompilationCacheDomain::compilation_key`], resolves the shared cache tiers through
    /// [`Self::get_or_compile`] with `produce` as the backend producer, validates that the compiled output types
    /// refine the lowered declaration, and assembles the compiled artifact. Backends that need a different policy
    /// can keep calling [`Self::get_or_compile`] directly.
    ///
    /// # Parameters
    ///
    ///   - `domain`: Compilation domain performing the transition.
    ///   - `request`: Lowered artifact consumed by the transition.
    ///   - `produce`: Backend compilation of the lowered program, invoked only when every cache tier misses.
    ///   - `output_types`: Effective flat output types of a compiled program.
    pub fn compile_request<Request, Produce, OutputTypes>(
        &self,
        domain: &D,
        request: Request,
        produce: Produce,
        output_types: OutputTypes,
    ) -> Result<CompiledFunction<D, Request::Input, Request::Output>, D::Error>
    where
        Request: CompileRequest<D>,
        Produce: FnOnce(&D::LoweredProgram) -> Result<D::CompiledProgram, D::Error>,
        OutputTypes: FnOnce(&D::CompiledProgram) -> Vec<D::Type>,
    {
        let cache_key = domain.compilation_key(request.lowered().lowered_program())?;
        let program = self.get_or_compile(domain, cache_key, || produce(request.lowered().lowered_program()))?;
        let compiled_output_types = output_types(program.as_ref());
        let declared_output_types = request.lowered().output_types();
        if compiled_output_types.len() != declared_output_types.len() {
            return Err(D::Error::from(ProgramError::InvalidOutputCount {
                expected: declared_output_types.len(),
                actual: compiled_output_types.len(),
            }));
        }
        for (declared, actual) in declared_output_types.iter().zip(&compiled_output_types) {
            if !declared.is_refined_by(actual) {
                return Err(D::Error::from(ProgramError::InvalidArgument {
                    message: format!("output type {actual} does not refine declared type {declared}"),
                }));
            }
        }
        Ok(request.into_compiled(program, compiled_output_types))
    }

    /// Returns a shared cached program for `cache_key`, restoring or producing it when absent.
    ///
    /// On cache hit, the entry is moved to the most-recently-used position. On miss, the new
    /// entry is inserted; if the cache is at capacity, the least-recently-used entry is evicted.
    ///
    /// Same-key misses are single-flight: exactly one caller restores or compiles while the rest
    /// wait for its result. Different keys may compile concurrently. When a producer fails, its
    /// error is returned only to that caller; waiters retry through a new producer election.
    ///
    /// Persistent caching is consulted only when both a [`DiskCache`] is attached and
    /// [`CompilationCacheDomain::persistent_cache_key`] returns stable key bytes. Persistent I/O or
    /// codec failures are counted and degraded to misses.
    pub fn get_or_compile<F: FnOnce() -> Result<D::CompiledProgram, D::Error>>(
        &self,
        domain: &D,
        cache_key: D::CacheKey,
        produce: F,
    ) -> Result<Arc<D::CompiledProgram>, D::Error> {
        let mut produce = Some(produce);
        loop {
            let lookup_start = Instant::now();
            if let Some(program) = self
                .programs
                .lock()
                .expect("compile cache mutex should not be poisoned")
                .get(&cache_key)
                .map(Arc::clone)
            {
                self.statistics.memory_hits.fetch_add(1, Ordering::Relaxed);
                self.record_tier(
                    CompilationCacheLevel::Memory,
                    CompilationCacheOutcome::Hit,
                    None,
                    lookup_start.elapsed(),
                    &self.statistics.memory_lookup_duration_ns,
                );
                return Ok(program);
            }

            let (in_flight, is_producer) = {
                let mut in_flight = self.in_flight.lock().expect("in-flight cache mutex should not be poisoned");
                match in_flight.entry(cache_key.clone()) {
                    Entry::Occupied(entry) => (Arc::clone(entry.get()), false),
                    Entry::Vacant(entry) => {
                        let compilation = Arc::new(InFlightCompilation::new());
                        entry.insert(Arc::clone(&compilation));
                        (compilation, true)
                    }
                }
            };

            if !is_producer {
                self.statistics.waits.fetch_add(1, Ordering::Relaxed);
                let wait_start = Instant::now();
                if let Some(program) = in_flight.wait() {
                    self.record_tier(
                        CompilationCacheLevel::Memory,
                        CompilationCacheOutcome::Wait,
                        None,
                        wait_start.elapsed(),
                        &self.statistics.memory_lookup_duration_ns,
                    );
                    return Ok(program);
                }
                continue;
            }

            // A prior producer can insert the program after our first LRU check and remove its
            // in-flight marker before we elect ourselves. Recheck after election so that narrow
            // handoff window cannot trigger a duplicate compilation.
            if let Some(program) = self
                .programs
                .lock()
                .expect("compile cache mutex should not be poisoned")
                .get(&cache_key)
                .map(Arc::clone)
            {
                self.in_flight.lock().expect("in-flight cache mutex should not be poisoned").remove(&cache_key);
                in_flight.finish(InFlightState::Ready(Arc::clone(&program)));
                self.statistics.memory_hits.fetch_add(1, Ordering::Relaxed);
                self.record_tier(
                    CompilationCacheLevel::Memory,
                    CompilationCacheOutcome::Hit,
                    None,
                    lookup_start.elapsed(),
                    &self.statistics.memory_lookup_duration_ns,
                );
                return Ok(program);
            }

            self.statistics.misses.fetch_add(1, Ordering::Relaxed);
            self.record_tier(
                CompilationCacheLevel::Memory,
                CompilationCacheOutcome::Miss,
                Some(CompilationMissReason::NotFound),
                lookup_start.elapsed(),
                &self.statistics.memory_lookup_duration_ns,
            );
            let producer = produce.take().expect("each cache lookup becomes a producer at most once");
            return self.restore_or_compile(domain, cache_key, in_flight, producer);
        }
    }

    fn restore_or_compile<F: FnOnce() -> Result<D::CompiledProgram, D::Error>>(
        &self,
        domain: &D,
        cache_key: D::CacheKey,
        in_flight: Arc<InFlightCompilation<D::CompiledProgram>>,
        produce: F,
    ) -> Result<Arc<D::CompiledProgram>, D::Error> {
        let in_flight_producer = InFlightProducer::new(self, cache_key, in_flight);
        let configured_exchange = self
            .artifact_exchange
            .as_ref()
            .filter(|exchange| self.artifact_exchange_policy.timeout().is_some() && exchange.process_count() != 1);
        let persistent_key = (self.disk_cache.is_some() || configured_exchange.is_some())
            .then(|| domain.persistent_cache_key(in_flight_producer.cache_key()))
            .flatten();
        let persistent = self
            .disk_cache
            .as_ref()
            .zip(persistent_key.as_ref())
            .map(|(cache, key)| (cache, CacheDigest::from_bytes(key.as_slice())));

        let exchange = match (configured_exchange, persistent_key.as_deref()) {
            (Some(_), None) => {
                self.record_event(CompilationEvent {
                    level: CompilationCacheLevel::Exchange,
                    outcome: CompilationCacheOutcome::Skipped,
                    duration: Duration::ZERO,
                    miss_reason: Some(CompilationMissReason::Unsupported),
                });
                self.require_exchange_or_record_fallback("domain does not provide a persistent compilation key")?;
                None
            }
            (Some(exchange), Some(key)) => self.exchange_preflight(exchange, key)?.then_some(exchange),
            (None, _) => None,
        };

        if let Some((disk_cache, digest)) = persistent.as_ref()
            && let Some((program, bytes)) = self.restore_from_disk(domain, disk_cache, digest)
        {
            if let Some(exchange) = exchange
                && exchange.process_index() == 0
            {
                let key =
                    persistent_key.as_ref().expect("an active artifact exchange has a persistent compilation key");
                self.publish_to_exchange(exchange, key.as_slice(), bytes.as_slice())?;
            }
            return Ok(in_flight_producer.finish(program));
        }

        if let Some(exchange) = exchange
            && exchange.process_index() != 0
        {
            let key = persistent_key.as_ref().expect("active exchange has a persistent compilation key");
            if let Some(program) = self.receive_from_exchange(domain, exchange, key.as_slice())? {
                return Ok(in_flight_producer.finish(program));
            }
        }

        self.compile_and_publish(domain, in_flight_producer, persistent, persistent_key.as_deref(), exchange, produce)
    }

    /// Preflights an active exchange for `key`, returning whether the exchange participates in this compilation.
    ///
    /// A preflight failure is fatal when the policy requires sharing; otherwise it is recorded and the exchange is
    /// deactivated for this compilation.
    fn exchange_preflight(
        &self,
        exchange: &Arc<dyn CompilationArtifactExchange>,
        key: &[u8],
    ) -> Result<bool, D::Error> {
        let timeout =
            self.artifact_exchange_policy.timeout().expect("an active artifact exchange policy has a timeout");
        let preflight_start = Instant::now();
        match exchange.preflight(key, timeout) {
            Ok(()) => Ok(true),
            Err(error) => {
                match &error {
                    CompilationExchangeError::TimedOut => {
                        self.statistics.exchange_timeouts.fetch_add(1, Ordering::Relaxed);
                    }
                    CompilationExchangeError::Incompatible { .. } | CompilationExchangeError::Failed { .. } => {
                        self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
                self.record_tier(
                    CompilationCacheLevel::Exchange,
                    CompilationCacheOutcome::Failed,
                    Some(match &error {
                        CompilationExchangeError::TimedOut => CompilationMissReason::TimedOut,
                        CompilationExchangeError::Incompatible { .. } => CompilationMissReason::Incompatible,
                        CompilationExchangeError::Failed { .. } => CompilationMissReason::ReadFailed,
                    }),
                    preflight_start.elapsed(),
                    &self.statistics.exchange_duration_ns,
                );
                self.require_exchange_or_record_fallback(error.to_string().as_str())?;
                Ok(false)
            }
        }
    }

    /// Restores a compiled program from the persistent disk tier, returning its serialized payload alongside it so
    /// an exchange leader can republish without reserializing. Every outcome short of a hit is recorded and
    /// degraded to `None`.
    fn restore_from_disk(
        &self,
        domain: &D,
        disk_cache: &DiskCache,
        digest: &CacheDigest,
    ) -> Option<(D::CompiledProgram, Vec<u8>)> {
        let lookup_start = Instant::now();
        let (outcome, miss_reason, restored) = match disk_cache.get(digest) {
            Ok(Some(bytes)) => match domain.deserialize_program(bytes.as_slice()) {
                Ok(Some(program)) => (CompilationCacheOutcome::Hit, None, Some((program, bytes))),
                Ok(None) => (CompilationCacheOutcome::Miss, Some(CompilationMissReason::Incompatible), None),
                Err(_error) => {
                    (CompilationCacheOutcome::Failed, Some(CompilationMissReason::DeserializationFailed), None)
                }
            },
            Ok(None) => (CompilationCacheOutcome::Miss, Some(CompilationMissReason::NotFound), None),
            Err(_error) => (CompilationCacheOutcome::Failed, Some(CompilationMissReason::ReadFailed), None),
        };
        match outcome {
            CompilationCacheOutcome::Hit => {
                self.statistics.persistent_hits.fetch_add(1, Ordering::Relaxed);
            }
            CompilationCacheOutcome::Failed => {
                self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        self.record_tier(
            CompilationCacheLevel::Persistent,
            outcome,
            miss_reason,
            lookup_start.elapsed(),
            &self.statistics.persistent_lookup_duration_ns,
        );
        restored
    }

    /// Waits for the artifact published by the exchange leader on a follower process.
    ///
    /// Returns the restored program on success and `None` when the policy permits falling back to local
    /// compilation after a miss, timeout, or failure.
    fn receive_from_exchange(
        &self,
        domain: &D,
        exchange: &Arc<dyn CompilationArtifactExchange>,
        key: &[u8],
    ) -> Result<Option<D::CompiledProgram>, D::Error> {
        let timeout =
            self.artifact_exchange_policy.timeout().expect("an active artifact exchange policy has a timeout");
        self.statistics.exchange_waits.fetch_add(1, Ordering::Relaxed);
        let exchange_start = Instant::now();
        match exchange.receive(key, timeout) {
            Ok(Some(bytes)) => match domain.deserialize_program(bytes.as_slice()) {
                Ok(Some(program)) => {
                    self.statistics.exchange_hits.fetch_add(1, Ordering::Relaxed);
                    self.record_tier(
                        CompilationCacheLevel::Exchange,
                        CompilationCacheOutcome::Hit,
                        None,
                        exchange_start.elapsed(),
                        &self.statistics.exchange_duration_ns,
                    );
                    Ok(Some(program))
                }
                Ok(None) => {
                    self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                    self.record_tier(
                        CompilationCacheLevel::Exchange,
                        CompilationCacheOutcome::Miss,
                        Some(CompilationMissReason::Incompatible),
                        exchange_start.elapsed(),
                        &self.statistics.exchange_duration_ns,
                    );
                    self.require_exchange_or_record_fallback("received compilation artifact is incompatible")?;
                    Ok(None)
                }
                Err(error) => {
                    self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                    self.record_tier(
                        CompilationCacheLevel::Exchange,
                        CompilationCacheOutcome::Failed,
                        Some(CompilationMissReason::DeserializationFailed),
                        exchange_start.elapsed(),
                        &self.statistics.exchange_duration_ns,
                    );
                    if !self.artifact_exchange_policy.permits_local_fallback() {
                        return Err(error);
                    }
                    self.statistics.exchange_fallbacks.fetch_add(1, Ordering::Relaxed);
                    Ok(None)
                }
            },
            Ok(None) | Err(CompilationExchangeError::TimedOut) => {
                self.statistics.exchange_timeouts.fetch_add(1, Ordering::Relaxed);
                self.record_tier(
                    CompilationCacheLevel::Exchange,
                    CompilationCacheOutcome::Miss,
                    Some(CompilationMissReason::TimedOut),
                    exchange_start.elapsed(),
                    &self.statistics.exchange_duration_ns,
                );
                self.require_exchange_or_record_fallback("timed out waiting for a compilation artifact")?;
                Ok(None)
            }
            Err(CompilationExchangeError::Failed { message }) => {
                self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                self.record_tier(
                    CompilationCacheLevel::Exchange,
                    CompilationCacheOutcome::Failed,
                    Some(CompilationMissReason::ReadFailed),
                    exchange_start.elapsed(),
                    &self.statistics.exchange_duration_ns,
                );
                self.require_exchange_or_record_fallback(message.as_str())?;
                Ok(None)
            }
            Err(CompilationExchangeError::Incompatible { message }) => {
                self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                self.record_tier(
                    CompilationCacheLevel::Exchange,
                    CompilationCacheOutcome::Failed,
                    Some(CompilationMissReason::Incompatible),
                    exchange_start.elapsed(),
                    &self.statistics.exchange_duration_ns,
                );
                self.require_exchange_or_record_fallback(message.as_str())?;
                Ok(None)
            }
        }
    }

    /// Publishes serialized `artifact` bytes for `key` through the exchange leader.
    ///
    /// Publish failures are fatal only when the policy requires sharing. Publish durations deliberately stay out of
    /// the exchange duration counter, which tracks time spent restoring artifacts.
    fn publish_to_exchange(
        &self,
        exchange: &Arc<dyn CompilationArtifactExchange>,
        key: &[u8],
        artifact: &[u8],
    ) -> Result<(), D::Error> {
        let publish_start = Instant::now();
        match exchange.publish(key, artifact) {
            Ok(()) => {
                self.statistics.exchange_publishes.fetch_add(1, Ordering::Relaxed);
                self.record_event(CompilationEvent {
                    level: CompilationCacheLevel::Exchange,
                    outcome: CompilationCacheOutcome::Succeeded,
                    duration: publish_start.elapsed(),
                    miss_reason: None,
                });
                Ok(())
            }
            Err(error) => {
                self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                self.record_event(CompilationEvent {
                    level: CompilationCacheLevel::Exchange,
                    outcome: CompilationCacheOutcome::Failed,
                    duration: publish_start.elapsed(),
                    miss_reason: Some(CompilationMissReason::WriteFailed),
                });
                self.require_exchange_or_record_fallback(error.to_string().as_str())
            }
        }
    }

    fn compile_and_publish<F: FnOnce() -> Result<D::CompiledProgram, D::Error>>(
        &self,
        domain: &D,
        in_flight_producer: InFlightProducer<'_, D>,
        persistent: Option<(&DiskCache, CacheDigest)>,
        persistent_key: Option<&[u8]>,
        exchange: Option<&Arc<dyn CompilationArtifactExchange>>,
        produce: F,
    ) -> Result<Arc<D::CompiledProgram>, D::Error> {
        self.statistics.compilations.fetch_add(1, Ordering::Relaxed);
        let compile_start = Instant::now();
        let program = match produce() {
            Ok(program) => program,
            Err(error) => {
                self.record_tier(
                    CompilationCacheLevel::Backend,
                    CompilationCacheOutcome::Failed,
                    Some(CompilationMissReason::ProducerFailed),
                    compile_start.elapsed(),
                    &self.statistics.compilation_duration_ns,
                );
                if let Some(exchange) = exchange
                    && exchange.process_index() == 0
                    && let Some(key) = persistent_key
                {
                    match exchange.publish_failure(key, error.to_string().as_str()) {
                        Ok(()) => {
                            self.statistics.exchange_publishes.fetch_add(1, Ordering::Relaxed);
                            self.record_event(CompilationEvent {
                                level: CompilationCacheLevel::Exchange,
                                outcome: CompilationCacheOutcome::Succeeded,
                                duration: compile_start.elapsed(),
                                miss_reason: Some(CompilationMissReason::ProducerFailed),
                            });
                        }
                        Err(_publish_error) => {
                            self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                            self.record_event(CompilationEvent {
                                level: CompilationCacheLevel::Exchange,
                                outcome: CompilationCacheOutcome::Failed,
                                duration: compile_start.elapsed(),
                                miss_reason: Some(CompilationMissReason::WriteFailed),
                            });
                        }
                    }
                }
                return Err(error);
            }
        };
        let compile_duration = compile_start.elapsed();
        self.record_tier(
            CompilationCacheLevel::Backend,
            CompilationCacheOutcome::Succeeded,
            None,
            compile_duration,
            &self.statistics.compilation_duration_ns,
        );

        let is_exchange_leader = exchange.is_none_or(|exchange| exchange.process_index() == 0);
        let should_serialize_for_disk = is_exchange_leader
            && persistent.as_ref().is_some_and(|(disk_cache, _)| disk_cache.should_serialize(compile_duration));
        let should_serialize_for_exchange = exchange.is_some_and(|exchange| exchange.process_index() == 0);
        if should_serialize_for_disk || should_serialize_for_exchange {
            let serialization_start = Instant::now();
            match domain.serialize_program(&program) {
                Ok(Some(bytes)) => {
                    if let Some((disk_cache, digest)) = persistent.as_ref()
                        && disk_cache.should_persist(compile_duration, bytes.len())
                        && let Err(_error) = disk_cache.put(digest, bytes.as_slice())
                    {
                        self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
                        self.record_event(CompilationEvent {
                            level: CompilationCacheLevel::Persistent,
                            outcome: CompilationCacheOutcome::Failed,
                            duration: serialization_start.elapsed(),
                            miss_reason: Some(CompilationMissReason::WriteFailed),
                        });
                    }
                    if let Some(exchange) = exchange
                        && exchange.process_index() == 0
                        && let Some(key) = persistent_key
                    {
                        self.publish_to_exchange(exchange, key, bytes.as_slice())?;
                    }
                }
                Ok(None) => {
                    if should_serialize_for_exchange {
                        self.record_event(CompilationEvent {
                            level: CompilationCacheLevel::Exchange,
                            outcome: CompilationCacheOutcome::Skipped,
                            duration: serialization_start.elapsed(),
                            miss_reason: Some(CompilationMissReason::Unsupported),
                        });
                        self.require_exchange_or_record_fallback(
                            "domain does not support compiled-program serialization",
                        )?;
                    }
                }
                Err(error) => {
                    if should_serialize_for_disk {
                        self.statistics.persistent_errors.fetch_add(1, Ordering::Relaxed);
                    }
                    if should_serialize_for_exchange {
                        self.statistics.exchange_errors.fetch_add(1, Ordering::Relaxed);
                        self.record_event(CompilationEvent {
                            level: CompilationCacheLevel::Exchange,
                            outcome: CompilationCacheOutcome::Failed,
                            duration: serialization_start.elapsed(),
                            miss_reason: Some(CompilationMissReason::WriteFailed),
                        });
                        if !self.artifact_exchange_policy.permits_local_fallback() {
                            return Err(error);
                        }
                        self.statistics.exchange_fallbacks.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        Ok(in_flight_producer.finish(program))
    }

    fn require_exchange_or_record_fallback(&self, message: &str) -> Result<(), D::Error> {
        if self.artifact_exchange_policy.permits_local_fallback() {
            self.statistics.exchange_fallbacks.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            Err(ProgramError::InvalidArgument {
                message: format!("compilation artifact exchange requirement was not satisfied: {message}"),
            }
            .into())
        }
    }

    fn add_duration(counter: &AtomicU64, duration: Duration) {
        let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        let _previous =
            counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| Some(value.saturating_add(nanoseconds)));
    }

    /// Accumulates `duration` into `duration_counter` and records one structured event for the operation.
    fn record_tier(
        &self,
        level: CompilationCacheLevel,
        outcome: CompilationCacheOutcome,
        miss_reason: Option<CompilationMissReason>,
        duration: Duration,
        duration_counter: &AtomicU64,
    ) {
        Self::add_duration(duration_counter, duration);
        self.record_event(CompilationEvent { level, outcome, duration, miss_reason });
    }

    pub(crate) fn record_event(&self, event: CompilationEvent) {
        if self.event_capacity > 0 {
            let mut events = self.recent_events.lock().expect("compilation event mutex should not be poisoned");
            if events.len() == self.event_capacity {
                events.pop_front();
            }
            events.push_back(event);
        }
        if let Some(reporter) = self.event_reporter.as_ref() {
            reporter(&event);
        }
    }

    fn finish_success(
        &self,
        cache_key: D::CacheKey,
        in_flight: Arc<InFlightCompilation<D::CompiledProgram>>,
        program: D::CompiledProgram,
    ) -> Arc<D::CompiledProgram> {
        let program = Arc::new(program);
        self.programs
            .lock()
            .expect("compile cache mutex should not be poisoned")
            .put(cache_key.clone(), Arc::clone(&program));
        self.in_flight.lock().expect("in-flight cache mutex should not be poisoned").remove(&cache_key);
        in_flight.finish(InFlightState::Ready(Arc::clone(&program)));
        program
    }

    fn finish_failure(&self, cache_key: &D::CacheKey, in_flight: Arc<InFlightCompilation<D::CompiledProgram>>) {
        self.in_flight.lock().expect("in-flight cache mutex should not be poisoned").remove(cache_key);
        in_flight.finish(InFlightState::Failed);
    }
}

impl<D: CompilationCacheDomain> Default for CompilationContext<D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Barrier, Condvar, Mutex, mpsc};
    use std::thread;
    use std::time::Duration;

    use crate::arrays::{Array, ArrayOperation, ArrayType};
    use crate::captures::CaptureReference;
    use crate::contexts::Domain;
    use crate::programs::ProgramError;

    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct TestCompiledProgram(usize);

    #[derive(Clone, Default)]
    struct TestDomain {
        persistent: bool,
    }

    impl TestDomain {
        fn persistent() -> Self {
            Self { persistent: true }
        }
    }

    impl Domain for TestDomain {
        type Type = ArrayType;
        type Value = Array;
        type Constant = CaptureReference<ArrayType>;
        type Operation = ArrayOperation<Array>;
    }

    impl CompilationDomain for TestDomain {
        type DispatchKey = Arc<[ArrayType]>;
        type LoweredProgram = Vec<ArrayType>;
        type CompiledProgram = TestCompiledProgram;
        type Options = ();
        type Error = ProgramError;

        fn dispatch_signature(
            &self,
            input_types: Vec<ArrayType>,
            _options: &Self::Options,
        ) -> Result<(Self::DispatchKey, Arc<[ArrayType]>), Self::Error> {
            let input_types: Arc<[ArrayType]> = input_types.into();
            Ok((input_types.clone(), input_types))
        }

        fn stage<Request>(
            &self,
            request: Request,
        ) -> Result<StagedFunction<Self, Request::Input, Request::Output>, Self::Error>
        where
            Request: StageRequest<Self>,
        {
            request.trace(|_, output_types| Ok(output_types))
        }

        fn lower<Request>(
            &self,
            staged: Request,
        ) -> Result<LoweredFunction<Self, Request::Input, Request::Output>, Self::Error>
        where
            Request: LoweringRequest<Self>,
        {
            let output_types = staged.staged().output_types().to_vec();
            Ok(staged.into_lowered(output_types.clone(), output_types))
        }

        fn compile<Request>(
            &self,
            lowered: Request,
        ) -> Result<CompiledFunction<Self, Request::Input, Request::Output>, Self::Error>
        where
            Request: CompileRequest<Self>,
        {
            Ok(lowered.into_compiled(Arc::new(TestCompiledProgram(0)), Vec::new()))
        }

        fn call<Request>(&self, request: Request) -> Result<Request::RuntimeOutput, Self::Error>
        where
            Request: CallRequest<Self>,
        {
            let executable = request.executable().clone();
            Request::reconstruct(&executable, Vec::new())
        }
    }

    impl CompilationCacheDomain for TestDomain {
        type CacheKey = u8;

        fn compilation_key(&self, _program: &Vec<ArrayType>) -> Result<u8, ProgramError> {
            Ok(0)
        }

        fn persistent_cache_key(&self, key: &u8) -> Option<Vec<u8>> {
            self.persistent.then(|| vec![*key])
        }

        fn serialize_program(&self, program: &Self::CompiledProgram) -> Result<Option<Vec<u8>>, Self::Error> {
            Ok(self.persistent.then(|| program.0.to_le_bytes().to_vec()))
        }

        fn deserialize_program(&self, bytes: &[u8]) -> Result<Option<Self::CompiledProgram>, Self::Error> {
            if !self.persistent {
                return Ok(None);
            }
            let bytes: [u8; size_of::<usize>()] = bytes.try_into().map_err(|_| {
                ProgramError::MalformedProgram("test persistent payload has the wrong length".to_string())
            })?;
            Ok(Some(TestCompiledProgram(usize::from_le_bytes(bytes))))
        }
    }

    #[derive(Default)]
    struct TestExchangeState {
        artifacts: Mutex<HashMap<Vec<u8>, Vec<u8>>>,
        ready: Condvar,
        preflight_fails: AtomicBool,
        receive_fails: AtomicBool,
        publish_fails: AtomicBool,
    }

    struct TestExchange {
        process_index: usize,
        state: Arc<TestExchangeState>,
    }

    impl TestExchange {
        fn new(process_index: usize, state: Arc<TestExchangeState>) -> Self {
            Self { process_index, state }
        }
    }

    impl CompilationArtifactExchange for TestExchange {
        fn process_index(&self) -> usize {
            self.process_index
        }

        fn process_count(&self) -> usize {
            2
        }

        fn preflight(&self, _key: &[u8], _timeout: Duration) -> Result<(), CompilationExchangeError> {
            if self.state.preflight_fails.load(Ordering::Relaxed) {
                Err(CompilationExchangeError::Incompatible {
                    message: "expected preflight incompatibility".to_string(),
                })
            } else {
                Ok(())
            }
        }

        fn publish(&self, key: &[u8], artifact: &[u8]) -> Result<(), CompilationExchangeError> {
            if self.state.publish_fails.load(Ordering::Relaxed) {
                return Err(CompilationExchangeError::Failed { message: "expected publish failure".to_string() });
            }
            self.state
                .artifacts
                .lock()
                .expect("test exchange mutex should not be poisoned")
                .insert(key.to_vec(), artifact.to_vec());
            self.state.ready.notify_all();
            Ok(())
        }

        fn receive(&self, key: &[u8], timeout: Duration) -> Result<Option<Vec<u8>>, CompilationExchangeError> {
            if self.state.receive_fails.load(Ordering::Relaxed) {
                return Err(CompilationExchangeError::Failed { message: "expected receive failure".to_string() });
            }
            let artifacts = self.state.artifacts.lock().expect("test exchange mutex should not be poisoned");
            let (artifacts, _timeout) = self
                .state
                .ready
                .wait_timeout_while(artifacts, timeout, |artifacts| !artifacts.contains_key(key))
                .expect("test exchange mutex should not be poisoned");
            Ok(artifacts.get(key).cloned())
        }
    }

    #[test]
    fn test_compilation_context_returns_shared_memory_hit() {
        let context = CompilationContext::<TestDomain>::new();
        let domain = TestDomain::default();
        let producer_calls = AtomicUsize::new(0);

        let first = context
            .get_or_compile(&domain, 1, || {
                producer_calls.fetch_add(1, Ordering::Relaxed);
                Ok(TestCompiledProgram(7))
            })
            .unwrap();
        let second = context
            .get_or_compile(&domain, 1, || {
                producer_calls.fetch_add(1, Ordering::Relaxed);
                Ok(TestCompiledProgram(8))
            })
            .unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.0, 7);
        assert_eq!(producer_calls.load(Ordering::Relaxed), 1);
        let statistics = context.statistics();
        assert_eq!(statistics.memory_hits, 1);
        assert_eq!(statistics.misses, 1);
        assert_eq!(statistics.compilations, 1);
    }

    #[test]
    fn test_compilation_context_evicts_least_recently_used_program() {
        let context = CompilationContext::<TestDomain>::with_capacity(1);
        let domain = TestDomain::default();
        let producer_calls = AtomicUsize::new(0);
        for key in [1, 2, 1] {
            context
                .get_or_compile(&domain, key, || {
                    producer_calls.fetch_add(1, Ordering::Relaxed);
                    Ok(TestCompiledProgram(key as usize))
                })
                .unwrap();
        }

        assert_eq!(producer_calls.load(Ordering::Relaxed), 3);
        assert_eq!(context.cache_size(), 1);
    }

    #[test]
    fn test_compilation_context_does_not_cache_producer_errors() {
        let context = CompilationContext::<TestDomain>::new();
        let domain = TestDomain::default();

        let failed = context
            .get_or_compile(&domain, 1, || Err(ProgramError::MalformedProgram("expected test failure".to_string())));
        assert!(failed.is_err());
        let program = context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(9))).unwrap();

        assert_eq!(program.0, 9);
        assert_eq!(context.statistics().compilations, 2);
    }

    #[test]
    fn test_compilation_context_clears_in_flight_state_after_producer_panic() {
        let context = CompilationContext::<TestDomain>::new();
        let domain = TestDomain::default();

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            context
                .get_or_compile(&domain, 1, || -> Result<TestCompiledProgram, ProgramError> {
                    panic!("expected producer panic")
                })
                .unwrap()
        }));
        assert!(panic.is_err());

        let program = context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(13))).unwrap();
        assert_eq!(program.0, 13);
    }

    #[test]
    fn test_compilation_context_single_flights_same_key() {
        const THREAD_COUNT: usize = 8;

        let context = Arc::new(CompilationContext::<TestDomain>::new());
        let start = Arc::new(Barrier::new(THREAD_COUNT + 1));
        let release = Arc::new(AtomicBool::new(false));
        let producer_calls = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..THREAD_COUNT {
            let context = Arc::clone(&context);
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            let producer_calls = Arc::clone(&producer_calls);
            threads.push(thread::spawn(move || {
                start.wait();
                context
                    .get_or_compile(&TestDomain::default(), 1, || {
                        producer_calls.fetch_add(1, Ordering::Relaxed);
                        while !release.load(Ordering::Acquire) {
                            thread::yield_now();
                        }
                        Ok(TestCompiledProgram(11))
                    })
                    .unwrap()
            }));
        }
        start.wait();

        let deadline = Instant::now() + Duration::from_secs(5);
        while (producer_calls.load(Ordering::Relaxed) == 0 || context.statistics().waits < (THREAD_COUNT - 1) as u64)
            && Instant::now() < deadline
        {
            thread::yield_now();
        }
        release.store(true, Ordering::Release);
        let programs = threads.into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>();

        assert_eq!(producer_calls.load(Ordering::Relaxed), 1);
        assert!(programs.iter().all(|program| Arc::ptr_eq(program, &programs[0])));
        assert_eq!(context.statistics().waits, (THREAD_COUNT - 1) as u64);
    }

    #[test]
    fn test_compilation_context_compiles_unrelated_keys_concurrently() {
        let context = Arc::new(CompilationContext::<TestDomain>::new());
        let start = Arc::new(Barrier::new(3));
        let release = Arc::new(AtomicBool::new(false));
        let (entered_sender, entered_receiver) = mpsc::channel();
        let mut threads = Vec::new();
        for key in [1, 2] {
            let context = Arc::clone(&context);
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            let entered_sender = entered_sender.clone();
            threads.push(thread::spawn(move || {
                start.wait();
                context
                    .get_or_compile(&TestDomain::default(), key, || {
                        entered_sender.send(key).unwrap();
                        while !release.load(Ordering::Acquire) {
                            thread::yield_now();
                        }
                        Ok(TestCompiledProgram(key as usize))
                    })
                    .unwrap()
            }));
        }
        start.wait();

        let first = entered_receiver.recv_timeout(Duration::from_secs(5)).unwrap();
        let second = entered_receiver.recv_timeout(Duration::from_secs(5));
        release.store(true, Ordering::Release);
        for thread in threads {
            thread.join().unwrap();
        }

        assert_ne!(first, second.expect("unrelated producer was serialized behind the first key"));
        assert_eq!(context.statistics().compilations, 2);
    }

    #[test]
    fn test_compilation_context_restores_valid_persistent_entry() {
        let directory = tempfile::tempdir().unwrap();
        let domain = TestDomain::persistent();
        let disk_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let first_context = CompilationContext::<TestDomain>::new().with_configured_disk_cache(disk_cache);
        first_context.get_or_compile(&domain, 3, || Ok(TestCompiledProgram(17))).unwrap();

        let disk_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let second_context = CompilationContext::<TestDomain>::new().with_configured_disk_cache(disk_cache);
        let restored = second_context
            .get_or_compile(&domain, 3, || -> Result<TestCompiledProgram, ProgramError> {
                panic!("persistent hit must not invoke producer")
            })
            .unwrap();

        assert_eq!(restored.0, 17);
        assert_eq!(second_context.statistics().persistent_hits, 1);
        assert_eq!(second_context.statistics().compilations, 0);
    }

    #[test]
    fn test_compilation_context_skips_disk_for_unsupported_domain() {
        let directory = tempfile::tempdir().unwrap();
        let context = CompilationContext::<TestDomain>::new()
            .with_configured_disk_cache(DiskCache::open(directory.path()).unwrap());
        context.get_or_compile(&TestDomain::default(), 1, || Ok(TestCompiledProgram(1))).unwrap();

        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        assert_eq!(context.statistics().persistent_errors, 0);
    }

    #[test]
    fn test_compilation_context_default_threshold_skips_fast_compilation() {
        let directory = tempfile::tempdir().unwrap();
        let context = CompilationContext::<TestDomain>::new()
            .with_configured_disk_cache(DiskCache::open(directory.path()).unwrap());
        context.get_or_compile(&TestDomain::persistent(), 1, || Ok(TestCompiledProgram(1))).unwrap();

        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn test_compilation_context_restores_artifact_published_by_leader() {
        let state = Arc::new(TestExchangeState::default());
        let policy = CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(5) };
        let follower_context = CompilationContext::<TestDomain>::new()
            .with_artifact_exchange(Arc::new(TestExchange::new(1, Arc::clone(&state))), policy);
        let follower = thread::spawn(move || {
            let program = follower_context
                .get_or_compile(&TestDomain::persistent(), 7, || -> Result<TestCompiledProgram, ProgramError> {
                    panic!("follower must restore the leader artifact")
                })
                .unwrap();
            (program.0, follower_context.statistics())
        });

        let leader_context = CompilationContext::<TestDomain>::new()
            .with_artifact_exchange(Arc::new(TestExchange::new(0, Arc::clone(&state))), policy);
        let leader =
            leader_context.get_or_compile(&TestDomain::persistent(), 7, || Ok(TestCompiledProgram(29))).unwrap();
        let (follower_value, follower_statistics) = follower.join().unwrap();

        assert_eq!(leader.0, 29);
        assert_eq!(follower_value, 29);
        assert_eq!(leader_context.statistics().compilations, 1);
        assert_eq!(leader_context.statistics().exchange_publishes, 1);
        assert_eq!(follower_statistics.compilations, 0);
        assert_eq!(follower_statistics.exchange_hits, 1);
        assert_eq!(follower_statistics.exchange_waits, 1);
    }

    #[test]
    fn test_compilation_context_publishes_leader_persistent_hit_to_cold_follower() {
        let directory = tempfile::tempdir().unwrap();
        let domain = TestDomain::persistent();
        let seed_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        CompilationContext::<TestDomain>::new()
            .with_configured_disk_cache(seed_cache)
            .get_or_compile(&domain, 8, || Ok(TestCompiledProgram(53)))
            .unwrap();

        let state = Arc::new(TestExchangeState::default());
        let policy = CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(5) };
        let follower_context = CompilationContext::<TestDomain>::new()
            .with_artifact_exchange(Arc::new(TestExchange::new(1, Arc::clone(&state))), policy);
        let follower = thread::spawn(move || {
            follower_context
                .get_or_compile(&TestDomain::persistent(), 8, || -> Result<TestCompiledProgram, ProgramError> {
                    panic!("follower must restore the leader's persistent artifact")
                })
                .unwrap()
                .0
        });

        let leader_cache = DiskCache::open(directory.path()).unwrap().with_write_thresholds(Duration::ZERO, 0);
        let leader_context = CompilationContext::<TestDomain>::new()
            .with_configured_disk_cache(leader_cache)
            .with_artifact_exchange(Arc::new(TestExchange::new(0, state)), policy);
        let leader = leader_context
            .get_or_compile(&domain, 8, || -> Result<TestCompiledProgram, ProgramError> {
                panic!("leader must restore its persistent artifact")
            })
            .unwrap();

        assert_eq!(leader.0, 53);
        assert_eq!(follower.join().unwrap(), 53);
        assert_eq!(leader_context.statistics().persistent_hits, 1);
        assert_eq!(leader_context.statistics().exchange_publishes, 1);
        assert_eq!(leader_context.statistics().compilations, 0);
    }

    #[test]
    fn test_compilation_context_falls_back_after_exchange_failure_when_permitted() {
        let state = Arc::new(TestExchangeState::default());
        state.receive_fails.store(true, Ordering::Relaxed);
        let context = CompilationContext::<TestDomain>::new().with_artifact_exchange(
            Arc::new(TestExchange::new(1, state)),
            CompilationArtifactExchangePolicy::PreferSharing {
                timeout: Duration::from_millis(10),
                fallback_to_local_compile: true,
            },
        );

        let program = context.get_or_compile(&TestDomain::persistent(), 2, || Ok(TestCompiledProgram(31))).unwrap();

        assert_eq!(program.0, 31);
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 1);
        assert_eq!(context.statistics().compilations, 1);
    }

    #[test]
    fn test_compilation_context_rejects_preflight_disagreement_before_compilation() {
        let state = Arc::new(TestExchangeState::default());
        state.preflight_fails.store(true, Ordering::Relaxed);
        let context = CompilationContext::<TestDomain>::new().with_artifact_exchange(
            Arc::new(TestExchange::new(0, state)),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(1) },
        );
        let producer_calls = AtomicUsize::new(0);

        let result = context.get_or_compile(&TestDomain::persistent(), 3, || {
            producer_calls.fetch_add(1, Ordering::Relaxed);
            Ok(TestCompiledProgram(33))
        });

        assert!(result.is_err());
        assert_eq!(producer_calls.load(Ordering::Relaxed), 0);
        assert_eq!(context.statistics().exchange_errors, 1);
        assert_eq!(context.statistics().compilations, 0);
    }

    #[test]
    fn test_compilation_context_fails_unsupported_required_exchange_without_waiting() {
        let context = CompilationContext::<TestDomain>::new().with_artifact_exchange(
            Arc::new(TestExchange::new(1, Arc::new(TestExchangeState::default()))),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_secs(30) },
        );
        let start = Instant::now();

        let result = context.get_or_compile(&TestDomain::default(), 4, || Ok(TestCompiledProgram(35)));

        assert!(result.is_err());
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(context.statistics().compilations, 0);
    }

    #[test]
    fn test_compilation_context_fails_after_required_exchange_timeout() {
        let context = CompilationContext::<TestDomain>::new().with_artifact_exchange(
            Arc::new(TestExchange::new(1, Arc::new(TestExchangeState::default()))),
            CompilationArtifactExchangePolicy::RequireSharing { timeout: Duration::from_millis(1) },
        );
        let producer_calls = AtomicUsize::new(0);

        let result = context.get_or_compile(&TestDomain::persistent(), 4, || {
            producer_calls.fetch_add(1, Ordering::Relaxed);
            Ok(TestCompiledProgram(37))
        });

        assert!(result.is_err());
        assert_eq!(producer_calls.load(Ordering::Relaxed), 0);
        assert_eq!(context.statistics().exchange_timeouts, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 0);
    }

    #[test]
    fn test_compilation_context_falls_back_after_exchange_timeout_when_permitted() {
        let context = CompilationContext::<TestDomain>::new().with_artifact_exchange(
            Arc::new(TestExchange::new(1, Arc::new(TestExchangeState::default()))),
            CompilationArtifactExchangePolicy::PreferSharing {
                timeout: Duration::from_millis(1),
                fallback_to_local_compile: true,
            },
        );

        let program = context.get_or_compile(&TestDomain::persistent(), 5, || Ok(TestCompiledProgram(41))).unwrap();

        assert_eq!(program.0, 41);
        assert_eq!(context.statistics().exchange_timeouts, 1);
        assert_eq!(context.statistics().exchange_fallbacks, 1);
        assert_eq!(context.statistics().compilations, 1);
    }

    #[test]
    fn test_compilation_context_retains_bounded_structured_events() {
        let context = CompilationContext::<TestDomain>::new().with_event_capacity(2);
        let domain = TestDomain::default();
        context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(43))).unwrap();
        context.get_or_compile(&domain, 1, || Ok(TestCompiledProgram(47))).unwrap();

        let events = context.recent_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].level, CompilationCacheLevel::Backend);
        assert_eq!(events[0].outcome, CompilationCacheOutcome::Succeeded);
        assert_eq!(events[1].level, CompilationCacheLevel::Memory);
        assert_eq!(events[1].outcome, CompilationCacheOutcome::Hit);
    }
}
